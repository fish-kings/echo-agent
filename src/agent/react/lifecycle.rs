use crate::agent::CancellationToken;
use crate::error::{AgentError, ReactError, Result};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard};
use tokio::sync::Notify;

/// Agent 关闭流程的内部状态。
struct CloseState {
    /// 是否仍接受新的 turn。
    accepting: bool,
    /// 下一个待分配的 turn id。
    next_turn_id: u64,
    /// 当前处于活跃状态的 turn 及其取消令牌。
    turns: HashMap<u64, CancellationToken>,
    /// 未按规范完成终结（settlement）的 turn 数量。
    unsettled_turns: usize,
}

impl Default for CloseState {
    /// 创建初始关闭状态：接受新轮次且尚未分配轮次 ID。
    fn default() -> Self {
        Self {
            accepting: true,
            next_turn_id: 0,
            turns: HashMap::new(),
            unsettled_turns: 0,
        }
    }
}

/**
单个 ReactAgent 所接受 turn 的准入与取消权威（authority）。

turn 的执行结果仍由既有的执行路径持有。该权威只负责：
限制新工作的接入、发起取消请求，并让 `Agent::close`
等待到每个已接受的 turn 都释放其租约（lease）为止。
*/
pub(crate) struct ReactAgentCloseAuthority {
    /// 关闭流程的内部可变状态。
    state: Mutex<CloseState>,
    /// 用于在所有 turn 都空闲（idle）时通知等待者的信号。
    idle: Notify,
}

impl Default for ReactAgentCloseAuthority {
    /// 创建处于可接受状态的关闭权威对象。
    fn default() -> Self {
        Self {
            state: Mutex::new(CloseState::default()),
            idle: Notify::new(),
        }
    }
}

impl ReactAgentCloseAuthority {
    /// 获取内部状态的锁；即使锁被污染（poisoned）也能恢复取到内部数据。
    fn lock_state(&self) -> MutexGuard<'_, CloseState> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// 准入一个新的 turn，返回其租约与取消令牌；若 Agent 已关闭则报错。
    pub(crate) fn admit(
        self: &Arc<Self>,
        cancel: Option<CancellationToken>,
    ) -> Result<(ReactAgentTurnLease, CancellationToken)> {
        let mut state = self.lock_state();
        if !state.accepting {
            return Err(ReactError::Agent(Box::new(AgentError::Cancelled(
                "Agent is closed to new turns".to_string(),
            ))));
        }
        let next_turn_id = state.next_turn_id.checked_add(1).ok_or_else(|| {
            ReactError::Other("Agent turn admission identity exhausted".to_string())
        })?;
        let turn_id = state.next_turn_id;
        state.next_turn_id = next_turn_id;
        /*
        关闭单个 Agent 时绝不能取消调用方的作用域。子令牌仍会
        观察到调用方的取消，而 Agent 的关闭则保持在局部范围内。
        */
        let cancel = cancel
            .map(|parent| parent.child_token())
            .unwrap_or_default();
        state.turns.insert(turn_id, cancel.clone());
        Ok((
            ReactAgentTurnLease {
                authority: Arc::clone(self),
                turn_id,
                active: true,
                started: false,
            },
            cancel,
        ))
    }

    /// 开始关闭流程：停止接受新 turn，并取消所有仍处于活跃状态的 turn。
    pub(crate) fn begin_close(&self) {
        let turns = {
            let mut state = self.lock_state();
            state.accepting = false;
            state.turns.values().cloned().collect::<Vec<_>>()
        };
        for cancel in turns {
            cancel.cancel();
        }
    }

    /// 等待所有 turn 都归于空闲；若存在未规范终结的 turn 则返回错误。
    pub(crate) async fn wait_until_idle(&self) -> Result<()> {
        loop {
            let notified = self.idle.notified();
            let unsettled_turns = {
                let state = self.lock_state();
                state.turns.is_empty().then_some(state.unsettled_turns)
            };
            if let Some(unsettled_turns) = unsettled_turns {
                if unsettled_turns > 0 {
                    return Err(ReactError::Other(format!(
                        "Agent close blocked by {unsettled_turns} turn(s) that ended without canonical settlement"
                    )));
                }
                return Ok(());
            }
            notified.await;
        }
    }
}

/// 一个已被准入 turn 所持有的关闭租约（close lease）。
#[must_use]
pub(crate) struct ReactAgentTurnLease {
    /// 所属的关闭权威。
    authority: Arc<ReactAgentCloseAuthority>,
    /// 该租约对应的 turn id。
    turn_id: u64,
    /// 该租约当前是否仍处于活跃状态。
    active: bool,
    /// 该 turn 是否已经开始规范化执行（决定 drop 时是否记为欠账）。
    started: bool,
}

impl ReactAgentTurnLease {
    /// 标记规范 turn 权威现在欠一次终态终结（terminal settlement）。
    pub(crate) fn mark_started(&mut self) {
        self.started = true;
    }

    /// 仅在完成规范的终态终结之后，才释放该关闭租约。
    pub(crate) fn settle(mut self) {
        self.release(false);
    }

    /// 释放该租约；`record_debt` 为 true 时会记为一次未规范终结的欠账。
    fn release(&mut self, record_debt: bool) {
        if !self.active {
            return;
        }
        self.active = false;
        let idle = {
            let mut state = self.authority.lock_state();
            state.turns.remove(&self.turn_id);
            if record_debt {
                state.unsettled_turns = state.unsettled_turns.saturating_add(1);
            }
            state.turns.is_empty()
        };
        if idle {
            self.authority.idle.notify_waiters();
        }
    }
}

impl Drop for ReactAgentTurnLease {
    /// 租约被丢弃时释放轮次，并按是否已开始记录关闭债务。
    fn drop(&mut self) {
        self.release(self.started);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 验证关闭 Agent 只取消其子作用域，不影响调用方令牌。
    #[tokio::test]
    async fn close_cancels_only_the_agent_child_scope() -> Result<()> {
        let authority = Arc::new(ReactAgentCloseAuthority::default());
        let parent = CancellationToken::new();
        let (lease, child) = authority.admit(Some(parent.clone()))?;

        authority.begin_close();

        assert!(child.is_cancelled());
        assert!(!parent.is_cancelled());
        lease.settle();
        authority.wait_until_idle().await
    }

    /// 验证已开始的轮次被直接丢弃时会被记为关闭债务。
    #[tokio::test]
    async fn started_turn_drop_is_reported_as_close_debt() -> Result<()> {
        let authority = Arc::new(ReactAgentCloseAuthority::default());
        let (mut lease, _) = authority.admit(None)?;
        lease.mark_started();
        drop(lease);

        authority.begin_close();
        let error = authority.wait_until_idle().await.err().ok_or_else(|| {
            ReactError::Other("unsettled turn did not block Agent close".to_string())
        })?;
        assert!(error.to_string().contains("without canonical settlement"));
        Ok(())
    }
}
