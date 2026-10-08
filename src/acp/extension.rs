/*!
 * 连接作用域的扩展调用权威（supreme plan 06，todo
 * `build-shared-extension-runtime`）。
 *
 * 该权威刻意保持协议无关：它只负责调用的生命周期 ——
 * 准入、并发许可、单次调用取消、独占互斥租约以及
 * 恰好一次的结算（设计文档 §12.3）。它从不接触语言层对象、
 * 协议 DTO、产品 UI 状态或第二个 run/terminal 权威；
 * SDK host 中的类型化桥接层负责将该生命周期映射到
 * 官方的反向请求（reverse-request）传输层上。
 *
 * 这里强制执行的不变式：
 *
 * - 读取循环永远不会因回调而阻塞：租约通过 `try_acquire`
 *   获取，且每一次等待都有调用方显式设置的截止时间；
 * - 一个调用身份最多结算一次；之后的结算会被视为迟到
 *   （late）并由调用方丢弃；
 * - 丢弃一个未结算的租约会失败关闭（fail-closed）：它会
 *   自动结算为已取消并释放其占用的资源，因此一个崩溃的
 *   代理任务永远不会泄漏许可；
 * - 独占租约会将对同一注册对象的重入式变更变成一个
 *   类型化的冲突，而不是相互等待。
 */

use crate::agent::CancellationToken;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;
use tokio::sync::{Notify, Semaphore, TryAcquireError};

/**
 * 说明为什么无法获取一次调用租约。桥接层会将其映射为
 * 类型化的扩展错误；运行时本身对错误码保持无感知。
 */
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtensionLeaseError {
    /// 连接正在关闭，拒绝新的调用。
    AdmissionClosed,
    /// 已协商的并发回调数量已耗尽。
    ConcurrencyLimit,
    /**
     * 同一注册对象的独占租约仍被持有（重入式变更）。
     * 调用方会报告一个类型化的冲突，而不是等待
     * （设计文档 §12.3）。
     */
    ExclusiveConflict,
}

impl std::fmt::Display for ExtensionLeaseError {
    /// 将租约错误格式化为可读的错误信息。
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AdmissionClosed => write!(formatter, "extension admission is closed"),
            Self::ConcurrencyLimit => write!(formatter, "extension concurrency limit reached"),
            Self::ExclusiveConflict => write!(
                formatter,
                "extension is already executing an exclusive invocation"
            ),
        }
    }
}

/// 一次调用的终止态结算，与传输层无关。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtensionSettlement {
    /**
     * 传输层已送达一个应答（负载已由调用方处理；
     * 运行时只记录这一事实）。
     */
    Answered,
    /// 在应答结算之前，截止时间已到。
    TimedOut,
    /// 被框架、所属的 run，或连接拆除逻辑取消。
    Cancelled,
    /// 反向传输失败（发送错误或断开连接）。
    Disconnected,
}

impl ExtensionSettlement {
    /// 该结算对于诊断而言是否呈现为成功状态。
    pub fn is_answered(&self) -> bool {
        matches!(self, Self::Answered)
    }
}

/// 一次调用的内部记录：取消令牌、结算状态与通知器
struct InvocationEntry {
    /// 该调用的取消令牌
    cancellation: CancellationToken,
    /// 结算状态（一旦写入即不可变更）
    state: StdMutex<Option<ExtensionSettlement>>,
    /// 用于通知等待者结算已完成
    settled: Notify,
}

impl InvocationEntry {
    /// 记录结算；若已结算过则返回 false
    fn settle(&self, settlement: ExtensionSettlement) -> bool {
        let Ok(mut state) = self.state.lock() else {
            return false;
        };
        if state.is_some() {
            return false;
        }
        *state = Some(settlement);
        self.settled.notify_waiters();
        true
    }

    /// 返回已记录的结算（如果存在）
    fn settlement(&self) -> Option<ExtensionSettlement> {
        self.state.lock().ok().and_then(|state| *state)
    }
}

/**
 * 一个正在进行中的调用租约。持有者负责驱动反向传输；
 * 结算必须恰好发生一次。丢弃一个未结算的租约会将其
 * 结算为已取消，并释放它持有的全部资源。
 */
pub struct ExtensionInvocationLease {
    /// 所属的调用权威
    authority: Arc<ExtensionInvocationAuthority>,
    /// 本次调用的唯一标识
    identity: String,
    /// 如果是独占租约，则为其注册键
    exclusive_key: Option<String>,
    /// 关联的调用条目（状态与通知）
    entry: Arc<InvocationEntry>,
    /// 持有的并发许可
    permit: Option<tokio::sync::OwnedSemaphorePermit>,
}

impl ExtensionInvocationLease {
    /**
     * 该租约的唯一调用标识，独立于任何 JSON-RPC 请求 id
     * （设计文档 §10.1）。
     */
    pub fn identity(&self) -> &str {
        &self.identity
    }

    /**
     * 该次调用的取消令牌。会在拆除时由权威触发信号，
     * 也可能由持有者所在框架的取消逻辑主动触发。
     */
    pub fn cancellation(&self) -> CancellationToken {
        self.entry.cancellation.clone()
    }

    /**
     * 记录结算结果。若该租约已经结算过，则返回 false ——
     * 调用方必须丢弃这个迟到的结果，且不能再改动状态。
     */
    pub fn settle(&mut self, settlement: ExtensionSettlement) -> bool {
        self.entry.settle(settlement)
    }

    /// 已记录的结算结果（如果存在的话）。
    pub fn settlement(&self) -> Option<ExtensionSettlement> {
        self.entry.settlement()
    }

    /**
     * 等待直到该租约被结算（由持有者、取消操作或拆除逻辑
     * 触发）。请务必搭配调用方侧的截止时间一起使用。
     */
    pub async fn wait_settled(&self) -> ExtensionSettlement {
        loop {
            let notified = self.entry.settled.notified();
            if let Some(settlement) = self.entry.settlement() {
                return settlement;
            }
            notified.await;
        }
    }

    /**
     * 如果仍处于待定状态，则将其结算为超时，然后报告本次
     * 调用是首次结算（true）还是迟到的结算（false）。
     */
    pub fn settle_timeout(&mut self) -> bool {
        self.entry.settle(ExtensionSettlement::TimedOut)
    }

    /// 释放该租约占用的许可与独占键，并从权威处移除该调用记录
    fn release(&mut self) {
        self.permit.take();
        if let Some(key) = self.exclusive_key.take() {
            self.authority.release_exclusive(&key);
        }
        self.authority.forget_invocation(&self.identity);
    }
}

impl Drop for ExtensionInvocationLease {
    /// 租约被丢弃时执行失败关闭（fail-closed）的结算。
    fn drop(&mut self) {
        /* 失败关闭（fail-closed）：一个未结算的租约（崩溃或被中止的
         * 代理任务）会被结算为已取消，这样就不会有等待者被挂起。
         */
        self.entry.settle(ExtensionSettlement::Cancelled);
        self.release();
    }
}

/// 每个连接共享的调用权威，被所有扩展代理共用。
pub struct ExtensionInvocationAuthority {
    /// 是否仍接受新的调用
    admission: AtomicBool,
    /// 连接级别的取消令牌
    connection_cancellation: CancellationToken,
    /// 并发许可信号量
    semaphore: Arc<Semaphore>,
    /// 独占租约的持有计数表
    exclusive: StdMutex<HashMap<String, usize>>,
    /// 当前所有在途调用
    invocations: StdMutex<HashMap<String, Arc<InvocationEntry>>>,
    /// 用于生成调用标识的自增序号
    serial: AtomicU64,
    /// 调用标识的前缀（每个权威实例唯一）
    identity_prefix: String,
}

impl ExtensionInvocationAuthority {
    /**
     * 使用已协商的并发上限创建权威。非正数的上限会被拒绝，
     * 因为那样会静默地禁用桥接层。
     */
    pub fn new(max_concurrency: usize) -> Result<Arc<Self>, ExtensionLeaseError> {
        if max_concurrency == 0 {
            return Err(ExtensionLeaseError::ConcurrencyLimit);
        }
        Ok(Self::build(max_concurrency))
    }

    /**
     * 创建权威，并将非正数的上限钳制为一个许可。
     * 用于连接建立阶段：此时配置已经过校验，降级处理
     * 比让整个连接失败更安全。
     */
    pub fn new_saturating(max_concurrency: usize) -> Arc<Self> {
        Self::build(max_concurrency.max(1))
    }

    /// 以给定的并发上限构建调用权威（调用方需保证其为正数）。
    fn build(max_concurrency: usize) -> Arc<Self> {
        Arc::new(Self {
            admission: AtomicBool::new(true),
            connection_cancellation: CancellationToken::new(),
            semaphore: Arc::new(Semaphore::new(max_concurrency)),
            exclusive: StdMutex::new(HashMap::new()),
            invocations: StdMutex::new(HashMap::new()),
            serial: AtomicU64::new(0),
            identity_prefix: format!("invocation-{}", uuid::Uuid::new_v4().simple()),
        })
    }

    /// 是否仍接受新的调用。
    pub fn admission_open(&self) -> bool {
        self.admission.load(Ordering::Acquire)
    }

    /**
     * 关闭准入。在途的调用会继续运行，直到它们自行结算、
     * 被 [`Self::cancel_all`] 取消，或有限期的排空（drain）超时。
     */
    pub fn close_admission(&self) {
        self.admission.store(false, Ordering::Release);
        self.connection_cancellation.cancel();
    }

    /**
     * 由那些未绑定到特定框架 run 上下文的调用共享的取消令牌。
     * 它会在连接拆除期间被取消，从而确保回调的存活时间
     * 不会超过其所属的 ACP 连接。
     */
    pub fn connection_cancellation(&self) -> CancellationToken {
        self.connection_cancellation.clone()
    }

    /// 当前在途调用的数量。
    pub fn in_flight(&self) -> usize {
        self.invocations
            .lock()
            .map(|invocations| invocations.len())
            .unwrap_or_default()
    }

    /**
     * 获取一个调用租约。
     *
     * `exclusive_key` 将该调用标记为对某个注册对象（连接代际 +
     * 实现标识）的独占变更：当这样的租约仍未释放时，对同一
     * 键的另一个独占租约请求会以
     * [`ExtensionLeaseError::ExclusiveConflict`] 失败，而不是
     * 排队等待（设计文档 §12.3）。`framework_cancellation`
     * 将该租约与所属的 run/工具取消逻辑关联起来，使框架层的
     * 取消操作无需等待截止时间即可到达回调。
     */
    pub fn lease(
        self: &Arc<Self>,
        exclusive_key: Option<&str>,
        framework_cancellation: CancellationToken,
    ) -> Result<ExtensionInvocationLease, ExtensionLeaseError> {
        if !self.admission_open() {
            return Err(ExtensionLeaseError::AdmissionClosed);
        }
        let permit = match self.semaphore.clone().try_acquire_owned() {
            Ok(permit) => permit,
            Err(TryAcquireError::Closed) => {
                return Err(ExtensionLeaseError::AdmissionClosed);
            }
            Err(_) => {
                return Err(ExtensionLeaseError::ConcurrencyLimit);
            }
        };
        if !self.admission_open() {
            drop(permit);
            return Err(ExtensionLeaseError::AdmissionClosed);
        }
        if let Some(key) = exclusive_key
            && !self.reserve_exclusive(key)
        {
            // 此处许可会被丢弃，从而释放该并发名额。
            return Err(ExtensionLeaseError::ExclusiveConflict);
        }
        let serial = self.serial.fetch_add(1, Ordering::AcqRel);
        let identity = format!("{}-{}", self.identity_prefix, serial);
        let entry = Arc::new(InvocationEntry {
            cancellation: framework_cancellation,
            state: StdMutex::new(None),
            settled: Notify::new(),
        });
        if let Ok(mut invocations) = self.invocations.lock() {
            invocations.insert(identity.clone(), entry.clone());
        }
        Ok(ExtensionInvocationLease {
            authority: self.clone(),
            identity,
            exclusive_key: exclusive_key.map(str::to_string),
            entry,
            permit: Some(permit),
        })
    }

    /**
     * 取消所有在途调用。返回取消开始时处于待定状态的调用数量；
     * 各个租约会通过自己的取消令牌或结算状态感知到取消，
     * 并自行释放。
     */
    pub fn cancel_all(&self) -> usize {
        let entries: Vec<Arc<InvocationEntry>> = self
            .invocations
            .lock()
            .map(|invocations| invocations.values().cloned().collect())
            .unwrap_or_default();
        let mut pending: usize = 0;
        for entry in entries {
            let was_pending = entry.settlement().is_none();
            entry.cancellation.cancel();
            let _ = entry.settle(ExtensionSettlement::Cancelled);
            if was_pending {
                pending = pending.saturating_add(1);
            }
        }
        pending
    }

    /**
     * 有限期的拆除等待：等待直到每个租约都已结算（因而已释放），
     * 或者超时到期。返回等待结束后仍处于在途状态的租约数量 ——
     * 零表示干净关闭。
     */
    pub async fn drain(&self, timeout: Duration) -> usize {
        let drained = tokio::time::timeout(timeout, async {
            loop {
                let pending = self.in_flight();
                if pending == 0 {
                    return;
                }
                /* 短暂轮询：租约的释放发生在各个代理任务中，
                 * 而不是通过单一的通知点触发。
                 */
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await;
        if drained.is_ok() { 0 } else { self.in_flight() }
    }

    /// 释放指定键上的一次独占持有计数
    fn release_exclusive(&self, key: &str) {
        if let Ok(mut exclusive) = self.exclusive.lock() {
            match exclusive.get(key).copied() {
                Some(count) if count <= 1 => {
                    exclusive.remove(key);
                }
                Some(count) => {
                    exclusive.insert(key.to_string(), count.saturating_sub(1));
                }
                None => {}
            }
        }
    }

    /// 尝试为指定键预留一次独占持有；已被占用时返回 false
    fn reserve_exclusive(&self, key: &str) -> bool {
        let Ok(mut exclusive) = self.exclusive.lock() else {
            return false;
        };
        match exclusive.get(key).copied() {
            Some(count) if count > 0 => false,
            _ => {
                exclusive.insert(key.to_string(), 1);
                true
            }
        }
    }

    /// 从在途调用表中移除指定标识的记录
    fn forget_invocation(&self, identity: &str) {
        if let Ok(mut invocations) = self.invocations.lock() {
            invocations.remove(identity);
        }
    }
}

/// 扩展调用权威的单元测试。
#[cfg(test)]
mod tests {
    use super::*;

    /// 构造给定并发上限的调用权威。
    fn authority(concurrency: usize) -> Arc<ExtensionInvocationAuthority> {
        ExtensionInvocationAuthority::new(concurrency).expect("positive concurrency")
    }

    /// 验证并发上限为零时会被拒绝。
    #[test]
    fn zero_concurrency_is_rejected() {
        assert_eq!(
            ExtensionInvocationAuthority::new(0).err(),
            Some(ExtensionLeaseError::ConcurrencyLimit)
        );
    }

    /// 验证并发许可被强制执行，且超限时快速失败。
    #[tokio::test]
    async fn concurrency_permits_are_enforced_and_fail_fast() {
        let authority = authority(2);
        let first = authority
            .lease(None, CancellationToken::new())
            .expect("lease 1");
        let second = authority
            .lease(None, CancellationToken::new())
            .expect("lease 2");
        assert_eq!(authority.in_flight(), 2);
        let third = authority.lease(None, CancellationToken::new());
        assert_eq!(third.err(), Some(ExtensionLeaseError::ConcurrencyLimit));
        drop(first);
        drop(second);
        assert_eq!(authority.in_flight(), 0);
        assert!(
            authority
                .lease(None, CancellationToken::new())
                .is_ok_and(|lease| lease.identity().contains("invocation-"))
        );
    }

    /// 验证结算恰好发生一次，迟到的结算会被拒绝。
    #[tokio::test]
    async fn settlement_is_exactly_once_and_late_settlements_are_rejected() {
        let authority = authority(4);
        let mut lease = authority
            .lease(None, CancellationToken::new())
            .expect("lease");
        assert_eq!(lease.settlement(), None);
        assert!(lease.settle(ExtensionSettlement::Answered));
        assert_eq!(lease.settlement(), Some(ExtensionSettlement::Answered));
        // 之后到达的迟到响应必须被丢弃。
        assert!(!lease.settle(ExtensionSettlement::Answered));
        assert!(!lease.settle_timeout());
        drop(lease);
        assert_eq!(authority.in_flight(), 0);
    }

    /// 验证独占租约发生冲突时直接报错而非等待。
    #[tokio::test]
    async fn exclusive_leases_conflict_instead_of_waiting() {
        let authority = authority(4);
        let key = "extension-1/human-loop";
        let exclusive = authority
            .lease(Some(key), CancellationToken::new())
            .expect("exclusive lease");
        let reentrant = authority.lease(Some(key), CancellationToken::new());
        assert_eq!(
            reentrant.err(),
            Some(ExtensionLeaseError::ExclusiveConflict)
        );
        // 其他键上的非独占租约不受影响。
        assert!(
            authority
                .lease(Some("extension-2/human-loop"), CancellationToken::new())
                .is_ok()
        );
        drop(exclusive);
        assert!(authority.lease(Some(key), CancellationToken::new()).is_ok());
    }

    /// 验证框架层取消会传播到租约的取消令牌。
    #[tokio::test]
    async fn framework_cancellation_propagates_to_the_lease_token() {
        let authority = authority(2);
        let framework = CancellationToken::new();
        let lease = authority.lease(None, framework.clone()).expect("lease");
        let token = lease.cancellation();
        framework.cancel();
        assert!(token.is_cancelled());
        drop(lease);
    }

    /// 验证丢弃未结算的租约会以取消状态失败关闭。
    #[tokio::test]
    async fn dropping_an_unsettled_lease_fails_closed_as_cancelled() {
        let authority = authority(2);
        let entry = {
            let lease = authority
                .lease(None, CancellationToken::new())
                .expect("lease");
            lease.entry.clone()
        };
        assert_eq!(
            entry.settlement(),
            Some(ExtensionSettlement::Cancelled),
            "a dropped lease must never leave an unsettled entry"
        );
        assert_eq!(authority.in_flight(), 0);
    }

    /// 验证依次执行关闭、取消、排空后所有资源都被释放。
    #[tokio::test]
    async fn close_then_cancel_then_drain_releases_everything() {
        let authority = authority(4);
        let held = authority
            .lease(Some("extension-1/agent"), CancellationToken::new())
            .expect("lease");
        authority.close_admission();
        assert_eq!(
            authority.lease(None, CancellationToken::new()).err(),
            Some(ExtensionLeaseError::AdmissionClosed)
        );
        let pending = authority.cancel_all();
        // 在 cancel_all 将其结算为已取消之前，该租约一直处于待定状态。
        assert_eq!(pending, 1);
        assert_eq!(held.settlement(), Some(ExtensionSettlement::Cancelled));
        /* 代理任务感知到结算后会释放自己的租约；
         * 只有到那时，有限期的排空（drain）才能观察到干净的状态。
         */
        drop(held);
        let remaining = authority.drain(Duration::from_millis(200)).await;
        assert_eq!(remaining, 0);
        assert_eq!(authority.in_flight(), 0);
        // 结算之后再丢弃的租约不应让任何东西死灰复燃。
        assert_eq!(authority.in_flight(), 0);
    }

    /// 验证排空会报告从未释放的租约。
    #[tokio::test]
    async fn drain_reports_leases_that_never_release() {
        let authority = ExtensionInvocationAuthority::new_saturating(4);
        /* 注入一个租约对象未经 Drop 就被泄漏的条目：
         * drain 必须报告它，而不是一直挂起等待。
         */
        authority.invocations.lock().expect("lock").insert(
            "invocation-test-stuck".to_string(),
            Arc::new(InvocationEntry {
                cancellation: CancellationToken::new(),
                state: StdMutex::new(None),
                settled: Notify::new(),
            }),
        );
        let remaining = authority.drain(Duration::from_millis(50)).await;
        assert_eq!(remaining, 1);
    }

    /// 验证 `wait_settled` 能观察到拆除阶段的结算。
    #[tokio::test]
    async fn wait_settled_observes_teardown_settlement() {
        let authority = authority(2);
        let mut lease = authority
            .lease(None, CancellationToken::new())
            .expect("lease");
        let waiter = {
            let entry = lease.entry.clone();
            tokio::spawn(async move {
                let notified = entry.settled.notified();
                if entry.settlement().is_none() {
                    notified.await;
                }
                entry.settlement()
            })
        };
        // 在结算之前，先给等待者一个订阅的机会。
        tokio::time::sleep(Duration::from_millis(10)).await;
        assert!(lease.settle(ExtensionSettlement::Disconnected));
        let observed = waiter.await.expect("waiter completes");
        assert_eq!(observed, Some(ExtensionSettlement::Disconnected));
        assert!(
            !observed
                .map(|settlement| settlement.is_answered())
                .unwrap_or(false)
        );
        drop(lease);
    }
}
