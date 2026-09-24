//! 针对活跃 Agent 运行的同一轮次（same-turn）用户引导（steering）。

use crate::llm::types::Message;
use echo_core::agent::{AgentSteerReceipt, AgentSteerState, AgentSteerTurnOutcome};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

/// 引导（steer）操作失败时返回的错误类型。
pub type TurnSteerError = echo_core::agent::AgentSteerError;

/// 当前正在进行的轮次的状态：轮次身份、是否可引导、
/// 待处理的引导消息队列，以及所有已接受消息的生命周期句柄。
#[derive(Default)]
struct ActiveTurn {
    /// 轮次 ID
    id: String,
    /// 该轮次的不透明世代（incarnation）句柄，用于区分同名但不同代的轮次
    incarnation: Arc<()>,
    /// 该轮次当前是否接受引导输入
    steerable: bool,
    /// 已接受但尚未被消费（drain）的引导消息队列
    pending: VecDeque<TrackedSteer>,
    /// 本轮次所有已接受引导消息的生命周期句柄
    lifecycles: Vec<Arc<SteerLifecycle>>,
}

/// 一条被跟踪的引导消息：消息内容及其生命周期句柄。
struct TrackedSteer {
    /// 引导消息内容
    message: Message,
    /// 与该消息关联的生命周期句柄
    lifecycle: Arc<SteerLifecycle>,
}

/// 一条引导消息的生命周期状态广播器。
struct SteerLifecycle {
    /// 状态变化的广播发送端
    state: tokio::sync::watch::Sender<AgentSteerState>,
}

impl SteerLifecycle {
    /// 将状态从 `Accepted` 推进为 `Drained`（若当前确实处于 `Accepted`）。
    fn mark_drained(&self) {
        self.state.send_if_modified(|state| {
            if matches!(state, AgentSteerState::Accepted) {
                *state = AgentSteerState::Drained;
                true
            } else {
                false
            }
        });
    }

    /// 将状态结算为终止态；若已经结算过则不做任何改动。
    fn settle(&self, outcome: AgentSteerTurnOutcome) {
        self.state.send_if_modified(|state| {
            if matches!(state, AgentSteerState::TurnSettled { .. }) {
                false
            } else {
                let drained = state.was_drained();
                *state = AgentSteerState::TurnSettled { outcome, drained };
                true
            }
        });
    }
}

/// 从邮箱中取出的一批待处理引导消息。
pub(crate) struct SteerDrainBatch {
    /// 本批次的引导消息
    pending: Vec<TrackedSteer>,
}

impl SteerDrainBatch {
    /// 该批次是否为空。
    pub(crate) fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    /// 该批次包含的消息数量。
    pub(crate) fn len(&self) -> usize {
        self.pending.len()
    }

    /// 按 FIFO 顺序遍历本批次的消息内容。
    pub(crate) fn messages(&self) -> impl Iterator<Item = &Message> {
        self.pending.iter().map(|steer| &steer.message)
    }

    /**
     * 只有在调用方已同步地将所有消息插入活跃轮次上下文之后，
     * 才能发布 `Drained` 状态。
     */
    pub(crate) fn mark_drained(self) {
        for steer in self.pending {
            steer.lifecycle.mark_drained();
        }
    }
}

/// 当前活跃轮次的引导消息邮箱，跨轮次复用同一个实例。
#[derive(Default)]
pub(crate) struct TurnSteerMailbox {
    /// 当前活跃的轮次（若存在）
    active: Mutex<Option<ActiveTurn>>,
}

impl TurnSteerMailbox {
    /// 开始一个新轮次，并结算（以 `Dropped` 结果）上一个未结算的轮次（若有）。
    pub(crate) fn begin(self: &Arc<Self>, turn_id: String) -> ActiveTurnLease {
        let incarnation = Arc::new(());
        let replaced = if let Ok(mut active) = self.active.lock() {
            active.replace(ActiveTurn {
                id: turn_id.clone(),
                incarnation: incarnation.clone(),
                steerable: false,
                pending: VecDeque::new(),
                lifecycles: Vec::new(),
            })
        } else {
            None
        };
        settle_active_turn(replaced, AgentSteerTurnOutcome::Dropped);
        ActiveTurnLease {
            mailbox: Arc::clone(self),
            turn_id,
            incarnation,
            settled: false,
        }
    }

    /// 设置指定轮次（按 ID 与世代匹配）当前是否可被引导。
    pub(crate) fn set_steerable(&self, turn_id: &str, incarnation: &Arc<()>, steerable: bool) {
        if let Ok(mut active) = self.active.lock()
            && let Some(turn) = active.as_mut()
            && turn.id == turn_id
            && Arc::ptr_eq(&turn.incarnation, incarnation)
        {
            turn.steerable = steerable;
        }
    }

    /// 向当前活跃轮次注入一条引导消息，仅返回该轮次的 ID。
    pub(crate) fn steer(
        &self,
        expected_turn_id: Option<&str>,
        message: Message,
    ) -> Result<String, TurnSteerError> {
        self.steer_tracked(expected_turn_id, message)
            .map(|receipt| receipt.turn_id().to_string())
    }

    /**
     * 向当前活跃轮次注入一条引导消息，并返回一份可用于观察该消息
     * 生命周期（Accepted -> Drained -> TurnSettled）的回执。
     *
     * 若消息内容为空、当前没有活跃轮次、轮次 ID 不匹配，或该轮次
     * 当前不可引导，则会返回相应的错误。
     */
    pub(crate) fn steer_tracked(
        &self,
        expected_turn_id: Option<&str>,
        message: Message,
    ) -> Result<AgentSteerReceipt, TurnSteerError> {
        let has_content = message
            .content
            .as_text()
            .is_some_and(|text| !text.trim().is_empty())
            || message
                .content
                .parts()
                .is_some_and(|parts| !parts.is_empty());
        if !has_content {
            return Err(TurnSteerError::EmptyInput);
        }
        let mut active = self
            .active
            .lock()
            .map_err(|_| TurnSteerError::StateUnavailable)?;
        let turn = active.as_mut().ok_or(TurnSteerError::NoActiveTurn)?;
        if let Some(expected) = expected_turn_id
            && expected != turn.id
        {
            return Err(TurnSteerError::TurnMismatch {
                expected: expected.to_string(),
                actual: turn.id.clone(),
            });
        }
        if !turn.steerable {
            return Err(TurnSteerError::NotSteerable {
                turn_id: turn.id.clone(),
            });
        }
        let steer_id = uuid::Uuid::new_v4().to_string();
        let turn_id = turn.id.clone();
        let (state, receiver) = tokio::sync::watch::channel(AgentSteerState::Accepted);
        let lifecycle = Arc::new(SteerLifecycle { state });
        turn.pending.push_back(TrackedSteer {
            message,
            lifecycle: lifecycle.clone(),
        });
        turn.lifecycles.push(lifecycle);
        Ok(AgentSteerReceipt::new(steer_id, turn_id, receiver))
    }

    /**
     * 取出指定轮次（按 ID 与世代匹配）当前所有待处理的引导消息。
     * 若轮次不匹配或已不存在，则返回一个空批次。
     */
    pub(crate) fn take_pending(&self, turn_id: &str, incarnation: &Arc<()>) -> SteerDrainBatch {
        let Ok(mut active) = self.active.lock() else {
            return SteerDrainBatch {
                pending: Vec::new(),
            };
        };
        let Some(turn) = active.as_mut() else {
            return SteerDrainBatch {
                pending: Vec::new(),
            };
        };
        if turn.id != turn_id || !Arc::ptr_eq(&turn.incarnation, incarnation) {
            return SteerDrainBatch {
                pending: Vec::new(),
            };
        }
        SteerDrainBatch {
            pending: turn.pending.drain(..).collect(),
        }
    }

    /// 结束指定轮次（按 ID 与世代匹配），并以给定结果结算其所有引导消息。
    fn finish(&self, turn_id: &str, incarnation: &Arc<()>, outcome: AgentSteerTurnOutcome) {
        let finished = if let Ok(mut active) = self.active.lock()
            && active.as_ref().is_some_and(|turn| {
                turn.id == turn_id && Arc::ptr_eq(&turn.incarnation, incarnation)
            }) {
            active.take()
        } else {
            None
        };
        settle_active_turn(finished, outcome);
    }
}

/// 以给定结果结算某个（可能不存在的）活跃轮次的所有引导消息。
fn settle_active_turn(active: Option<ActiveTurn>, outcome: AgentSteerTurnOutcome) {
    let Some(active) = active else {
        return;
    };
    for lifecycle in active.lifecycles {
        lifecycle.settle(outcome);
    }
}

/**
 * 一个活跃轮次的作用域租约（lease）。持有者负责在轮次结束时
 * 调用 [`ActiveTurnLease::settle`]；若在结算前被丢弃，则会自动
 * 以 `Dropped` 结果结算（失败关闭语义）。
 */
pub(crate) struct ActiveTurnLease {
    /// 所属的引导消息邮箱
    mailbox: Arc<TurnSteerMailbox>,
    /// 该租约对应的轮次 ID
    turn_id: String,
    /// 该轮次的世代句柄
    incarnation: Arc<()>,
    /// 是否已经显式结算过
    settled: bool,
}

impl ActiveTurnLease {
    /// 该租约对应轮次的世代句柄。
    pub(crate) fn incarnation(&self) -> Arc<()> {
        self.incarnation.clone()
    }

    /// 设置该轮次当前是否可被引导。
    pub(crate) fn set_steerable(&self, steerable: bool) {
        self.mailbox
            .set_steerable(&self.turn_id, &self.incarnation, steerable);
    }

    /// 以给定结果显式结算该轮次。
    pub(crate) fn settle(mut self, outcome: AgentSteerTurnOutcome) {
        self.mailbox
            .finish(&self.turn_id, &self.incarnation, outcome);
        self.settled = true;
    }
}

impl Drop for ActiveTurnLease {
    /// 失败关闭：若租约在显式结算之前被丢弃，则以 `Dropped` 结果结算。
    fn drop(&mut self) {
        if !self.settled {
            self.mailbox.finish(
                &self.turn_id,
                &self.incarnation,
                AgentSteerTurnOutcome::Dropped,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn lease_scopes_active_turn_and_preserves_fifo() {
        let mailbox = Arc::new(TurnSteerMailbox::default());
        let lease = mailbox.begin("turn-1".to_string());
        lease.set_steerable(true);
        assert!(
            mailbox
                .steer(Some("turn-1"), Message::user("first".to_string()))
                .is_ok()
        );
        assert!(
            mailbox
                .steer(Some("turn-1"), Message::user("second".to_string()))
                .is_ok()
        );
        let drained = mailbox.take_pending("turn-1", &lease.incarnation());
        assert_eq!(drained.len(), 2);
        assert_eq!(
            drained
                .messages()
                .next()
                .and_then(|message| message.content.as_text()),
            Some("first".to_string())
        );
        assert_eq!(
            drained
                .messages()
                .nth(1)
                .and_then(|message| message.content.as_text()),
            Some("second".to_string())
        );
        drained.mark_drained();
        drop(lease);
        assert_eq!(
            mailbox.steer(Some("turn-1"), Message::user("late".to_string())),
            Err(TurnSteerError::NoActiveTurn)
        );
    }

    #[test]
    fn rejects_mismatch_and_non_steerable_turns() {
        let mailbox = Arc::new(TurnSteerMailbox::default());
        let lease = mailbox.begin("turn-2".to_string());
        assert!(matches!(
            mailbox.steer(Some("turn-1"), Message::user("x".to_string())),
            Err(TurnSteerError::TurnMismatch { .. })
        ));
        assert!(matches!(
            mailbox.steer(Some("turn-2"), Message::user("x".to_string())),
            Err(TurnSteerError::NotSteerable { .. })
        ));
        drop(lease);
    }

    #[tokio::test]
    async fn tracked_receipt_is_accepted_before_real_drain_then_settles() -> Result<(), String> {
        let mailbox = Arc::new(TurnSteerMailbox::default());
        let lease = mailbox.begin("tracked-turn".to_string());
        lease.set_steerable(true);
        let mut receipt = mailbox
            .steer_tracked(
                Some("tracked-turn"),
                Message::user("tracked input".to_string()),
            )
            .map_err(|error| error.to_string())?;

        assert_eq!(receipt.state(), AgentSteerState::Accepted);
        assert!(!receipt.state().was_drained());
        let drained = mailbox.take_pending("tracked-turn", &lease.incarnation());
        assert_eq!(drained.len(), 1);
        drained.mark_drained();
        assert_eq!(receipt.wait_for_drained().await, AgentSteerState::Drained);

        lease.settle(AgentSteerTurnOutcome::Completed);
        assert_eq!(
            receipt.wait_for_turn_settled().await,
            AgentSteerState::TurnSettled {
                outcome: AgentSteerTurnOutcome::Completed,
                drained: true,
            }
        );
        Ok(())
    }

    #[tokio::test]
    async fn tracked_terminal_before_drain_preserves_non_consumption() -> Result<(), String> {
        let mailbox = Arc::new(TurnSteerMailbox::default());
        let lease = mailbox.begin("cancelled-turn".to_string());
        lease.set_steerable(true);
        let mut receipt = mailbox
            .steer_tracked(
                Some("cancelled-turn"),
                Message::user("not drained".to_string()),
            )
            .map_err(|error| error.to_string())?;

        lease.settle(AgentSteerTurnOutcome::Cancelled);
        let terminal = receipt.wait_for_drained().await;
        assert_eq!(
            terminal,
            AgentSteerState::TurnSettled {
                outcome: AgentSteerTurnOutcome::Cancelled,
                drained: false,
            }
        );
        assert!(!terminal.was_drained());
        Ok(())
    }

    #[tokio::test]
    async fn dropped_owner_terminalizes_every_tracked_input() -> Result<(), String> {
        let mailbox = Arc::new(TurnSteerMailbox::default());
        let lease = mailbox.begin("dropped-turn".to_string());
        lease.set_steerable(true);
        let mut receipt = mailbox
            .steer_tracked(Some("dropped-turn"), Message::user("input".to_string()))
            .map_err(|error| error.to_string())?;
        drop(lease);

        assert_eq!(
            receipt.wait_for_turn_settled().await,
            AgentSteerState::TurnSettled {
                outcome: AgentSteerTurnOutcome::Dropped,
                drained: false,
            }
        );
        Ok(())
    }

    #[tokio::test]
    async fn tracked_concurrent_acceptance_has_unique_receipts_and_one_fifo_drain()
    -> Result<(), String> {
        let mailbox = Arc::new(TurnSteerMailbox::default());
        let lease = mailbox.begin("concurrent-turn".to_string());
        lease.set_steerable(true);
        let mut tasks = Vec::new();
        for index in 0..16_u8 {
            let mailbox = mailbox.clone();
            tasks.push(tokio::spawn(async move {
                mailbox.steer_tracked(
                    Some("concurrent-turn"),
                    Message::user(format!("input-{index}")),
                )
            }));
        }

        let mut receipts = Vec::new();
        for task in tasks {
            receipts.push(
                task.await
                    .map_err(|error| format!("steer task failed: {error}"))?
                    .map_err(|error| error.to_string())?,
            );
        }
        let ids = receipts
            .iter()
            .map(|receipt| receipt.steer_id().to_string())
            .collect::<HashSet<_>>();
        assert_eq!(ids.len(), receipts.len());
        let drained = mailbox.take_pending("concurrent-turn", &lease.incarnation());
        assert_eq!(drained.len(), receipts.len());
        drained.mark_drained();
        assert!(
            receipts
                .iter()
                .all(|receipt| receipt.state() == AgentSteerState::Drained)
        );
        lease.settle(AgentSteerTurnOutcome::Completed);
        assert!(receipts.iter().all(|receipt| {
            receipt.state()
                == AgentSteerState::TurnSettled {
                    outcome: AgentSteerTurnOutcome::Completed,
                    drained: true,
                }
        }));
        Ok(())
    }

    #[tokio::test]
    async fn tracked_same_id_stale_lease_cannot_mutate_new_incarnation() -> Result<(), String> {
        let mailbox = Arc::new(TurnSteerMailbox::default());
        let stale_settle = mailbox.begin("same-turn".to_string());
        stale_settle.set_steerable(true);
        let mut first_receipt = mailbox
            .steer_tracked(
                Some("same-turn"),
                Message::user("first generation".to_string()),
            )
            .map_err(|error| error.to_string())?;

        let stale_drop = mailbox.begin("same-turn".to_string());
        stale_drop.set_steerable(true);
        assert_eq!(
            first_receipt.wait_for_turn_settled().await,
            AgentSteerState::TurnSettled {
                outcome: AgentSteerTurnOutcome::Dropped,
                drained: false,
            }
        );
        stale_settle.set_steerable(false);
        stale_settle.settle(AgentSteerTurnOutcome::Completed);
        let mut second_receipt = mailbox
            .steer_tracked(
                Some("same-turn"),
                Message::user("second generation".to_string()),
            )
            .map_err(|error| error.to_string())?;
        assert_eq!(second_receipt.state(), AgentSteerState::Accepted);

        let current = mailbox.begin("same-turn".to_string());
        current.set_steerable(true);
        assert_eq!(
            second_receipt.wait_for_turn_settled().await,
            AgentSteerState::TurnSettled {
                outcome: AgentSteerTurnOutcome::Dropped,
                drained: false,
            }
        );
        drop(stale_drop);
        let mut current_receipt = mailbox
            .steer_tracked(
                Some("same-turn"),
                Message::user("current generation".to_string()),
            )
            .map_err(|error| error.to_string())?;
        assert_eq!(current_receipt.state(), AgentSteerState::Accepted);
        current.settle(AgentSteerTurnOutcome::Cancelled);
        assert_eq!(
            current_receipt.wait_for_turn_settled().await,
            AgentSteerState::TurnSettled {
                outcome: AgentSteerTurnOutcome::Cancelled,
                drained: false,
            }
        );
        Ok(())
    }
}
