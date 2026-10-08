/*!
共享扩展调用运行时回归测试（supreme plan 06，todo
`build-shared-extension-runtime`）。

这些测试使用一个可控的伪桥接传输来驱动与协议无关的
[`echo_agent::acp::ExtensionInvocationAuthority`]：伪桥接获取租约的方式与
SDK 宿主桥接完全一致（租约 → 发送 → 带截止时间等待 → 恰好结算一次），
因此并发许可、截止时间结算、取消传播、迟到响应丢弃、重入冲突以及关闭顺序
都是针对真实的生命周期实现观察得到的。
*/

#![cfg(feature = "acp")]

use echo_agent::acp::{
    AcpAdapterConfig, AcpConnectionServices, ExtensionInvocationAuthority, ExtensionLeaseError,
    ExtensionSettlement,
};
use echo_agent::acp::{AcpSessionContext, AcpSessionFactory, SessionRegistry};
use echo_agent::agent::EventIdentity;
use echo_agent::agent::{CancellationToken, EventEnvelope};
use echo_agent::error::Result;
use echo_agent::runtime::{TurnMode, TurnRequest};
use futures::StreamExt as _;
use futures::future::BoxFuture;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

/**
一次伪造的反向调用，驱动方式与真实桥接一致：获取租约、带截止时间等待传输、
恰好结算一次，然后观察迟到的结果。`TransportBehavior` 控制伪 SDK 端的行为。
*/
struct FakeBridge {
    /// 被测的扩展调用权威，所有租约都从这里获取。
    authority: Arc<ExtensionInvocationAuthority>,
}

/// 伪 SDK 端对一次反向调用的响应方式。
enum TransportBehavior {
    /// 在给定延迟后应答。
    AnswerAfter(Duration),
    /// 永不应答；由截止时间或取消来结算租约。
    Silent,
}

impl FakeBridge {
    /// 基于给定的调用权威创建一个共享的伪桥接。
    fn new(authority: Arc<ExtensionInvocationAuthority>) -> Arc<Self> {
        Arc::new(Self { authority })
    }

    /**
    驱动一次非流式调用直到完成，并返回调用方观察到的结算结果，
    与桥接中的 select! 逻辑保持一致。
    该方法会消耗桥接句柄，使返回的 future 为 `'static`，
    从而可以像真实的代理任务一样被 spawn。
    */
    async fn invoke(
        self: Arc<Self>,
        exclusive_key: Option<&'static str>,
        framework_cancellation: CancellationToken,
        behavior: TransportBehavior,
        deadline: Duration,
    ) -> ExtensionSettlement {
        let mut lease = match self.authority.lease(exclusive_key, framework_cancellation) {
            Ok(lease) => lease,
            Err(error) => panic!("lease acquisition failed: {error}"),
        };
        let cancellation = lease.cancellation();
        tokio::select! {
            _answered = async {
                match behavior {
                    TransportBehavior::AnswerAfter(delay) => {
                        tokio::time::sleep(delay).await;
                    }
                    TransportBehavior::Silent => {
                        std::future::pending::<()>().await;
                    }
                }
            } => {
                lease.settle(ExtensionSettlement::Answered);
                ExtensionSettlement::Answered
            }
            () = cancellation.cancelled() => {
                // 桥接先发送取消通知，然后在本地结算。
                lease.settle(ExtensionSettlement::Cancelled);
                ExtensionSettlement::Cancelled
            }
            _ = tokio::time::sleep(deadline) => {
                lease.settle_timeout();
                ExtensionSettlement::TimedOut
            }
        }
    }
}

/// 以给定并发上限构建调用权威；并发数必须为正。
fn authority(concurrency: usize) -> Arc<ExtensionInvocationAuthority> {
    match ExtensionInvocationAuthority::new(concurrency) {
        Ok(authority) => authority,
        Err(error) => panic!("positive concurrency must build: {error}"),
    }
}

/// 并发调用共享许可，许可耗尽时新的租约立即失败而不是等待。
#[tokio::test]
async fn concurrent_invocations_share_permits_and_fail_fast_when_exhausted() {
    let bridge = FakeBridge::new(authority(2));
    let first = tokio::spawn(bridge.clone().invoke(
        None,
        CancellationToken::new(),
        TransportBehavior::Silent,
        Duration::from_secs(30),
    ));
    let second = tokio::spawn(bridge.clone().invoke(
        None,
        CancellationToken::new(),
        TransportBehavior::Silent,
        Duration::from_secs(30),
    ));
    // 等待两个调用都持有各自的许可。
    while bridge.authority.in_flight() < 2 {
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
    let third = bridge.authority.lease(None, CancellationToken::new());
    assert_eq!(third.err(), Some(ExtensionLeaseError::ConcurrencyLimit));
    assert_eq!(bridge.authority.in_flight(), 2);
    // 通过取消释放两个静默租约，从而归还许可。
    let cancel_all = bridge.authority.cancel_all();
    assert_eq!(cancel_all, 2);
    let (first, second) = tokio::join!(first, second);
    assert_eq!(first.ok(), Some(ExtensionSettlement::Cancelled));
    assert_eq!(second.ok(), Some(ExtensionSettlement::Cancelled));
    let remaining = bridge.authority.drain(Duration::from_millis(200)).await;
    assert_eq!(remaining, 0);
}

/// 截止时间到达时结算为超时，之后迟到的应答会被丢弃。
#[tokio::test]
async fn deadline_settles_timeout_and_late_answers_are_discarded() {
    let bridge = FakeBridge::new(authority(4));
    // 伪 SDK 保持静默；25ms 的截止时间必须结算为超时。
    let observed = bridge
        .clone()
        .invoke(
            None,
            CancellationToken::new(),
            TransportBehavior::Silent,
            Duration::from_millis(25),
        )
        .await;
    assert_eq!(observed, ExtensionSettlement::TimedOut);
    // 结算之后到达的迟到应答不得覆盖已有结算。
    let mut lease = match bridge.authority.lease(None, CancellationToken::new()) {
        Ok(lease) => lease,
        Err(error) => panic!("permit freed after timeout: {error}"),
    };
    assert!(lease.settle_timeout());
    assert!(!lease.settle(ExtensionSettlement::Answered));
    assert_eq!(lease.settlement(), Some(ExtensionSettlement::TimedOut));
    drop(lease);
}

/// 框架侧取消会在截止时间之前把调用结算为已取消。
#[tokio::test]
async fn framework_cancellation_settles_cancelled_before_the_deadline() {
    let bridge = FakeBridge::new(authority(2));
    let framework = CancellationToken::new();
    let invocation = bridge.clone().invoke(
        None,
        framework.clone(),
        TransportBehavior::Silent,
        Duration::from_secs(30),
    );
    // 所属的运行在截止时间之前很早就取消了。
    tokio::time::sleep(Duration::from_millis(10)).await;
    framework.cancel();
    let observed = invocation.await;
    assert_eq!(observed, ExtensionSettlement::Cancelled);
    let remaining = bridge.authority.drain(Duration::from_millis(200)).await;
    assert_eq!(remaining, 0);
}

/// 对同一独占键的重入立即返回冲突错误，而不是等待。
#[tokio::test]
async fn exclusive_reentry_returns_conflict_without_waiting() {
    let authority = authority(4);
    let key = "extension-1/human_loop_request";
    let exclusive = match authority.lease(Some(key), CancellationToken::new()) {
        Ok(lease) => lease,
        Err(error) => panic!("exclusive lease: {error}"),
    };
    let reentrant = authority.lease(Some(key), CancellationToken::new());
    assert_eq!(
        reentrant.err(),
        Some(ExtensionLeaseError::ExclusiveConflict),
        "re-entrant exclusive mutation must be a typed conflict, not a wait"
    );
    // 某个注册项持有租约期间，其它独立注册项仍可正常工作。
    assert!(
        authority
            .lease(
                Some("extension-2/human_loop_request"),
                CancellationToken::new()
            )
            .is_ok()
    );
    drop(exclusive);
    assert!(authority.lease(Some(key), CancellationToken::new()).is_ok());
}

/// 拆除流程按顺序执行：先关闭准入，再取消在途调用，最后排空。
#[tokio::test]
async fn teardown_closes_admission_then_cancels_then_drains_in_order() {
    let bridge = FakeBridge::new(authority(4));
    let in_flight = tokio::spawn(bridge.clone().invoke(
        Some("extension-1/custom_agent"),
        CancellationToken::new(),
        TransportBehavior::Silent,
        Duration::from_secs(30),
    ));
    while bridge.authority.in_flight() < 1 {
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
    // 与适配器关闭链一致的关闭顺序：准入、取消、排空。
    bridge.authority.close_admission();
    assert_eq!(
        bridge.authority.lease(None, CancellationToken::new()).err(),
        Some(ExtensionLeaseError::AdmissionClosed)
    );
    bridge.authority.cancel_all();
    let observed = in_flight.await.ok();
    assert_eq!(observed, Some(ExtensionSettlement::Cancelled));
    let leaked = bridge.authority.drain(Duration::from_millis(200)).await;
    assert_eq!(
        leaked, 0,
        "a cancelled callback that returned must release its lease"
    );
}

// ── 连接级接线 ─────────────────────────────────────────────────────────

/// 测试用的空 Agent：立即返回最终答案，并统计流式轮次数。
struct NullAgent {
    /// 已执行的流式轮次计数。
    turns: Arc<AtomicUsize>,
}

impl echo_agent::agent::Agent for NullAgent {
    fn name(&self) -> &str {
        "extension-runtime-fixture"
    }

    fn model_name(&self) -> &str {
        "fixture-model"
    }

    fn system_prompt(&self) -> &str {
        "fixture"
    }

    fn execute<'a>(&'a self, _task: &'a str) -> BoxFuture<'a, Result<String>> {
        Box::pin(async { Ok("done".to_string()) })
    }

    fn execute_stream<'a>(
        &'a self,
        _task: &'a str,
    ) -> BoxFuture<'a, Result<futures::stream::BoxStream<'a, Result<echo_agent::agent::AgentEvent>>>>
    {
        Box::pin(async {
            let turns = self.turns.clone();
            Ok(futures::stream::once(async move {
                turns.fetch_add(1, Ordering::AcqRel);
                Ok(echo_agent::agent::AgentEvent::FinalAnswer(
                    "done".to_string(),
                ))
            })
            .boxed())
        })
    }
}

/// 构建一组使用 [`NullAgent`] 会话工厂的连接服务。
fn test_services() -> Arc<AcpConnectionServices> {
    let turns = Arc::new(AtomicUsize::new(0));
    let factory = move |_context: AcpSessionContext| {
        let turns = turns.clone();
        async move { Ok(Box::new(NullAgent { turns }) as Box<dyn echo_agent::agent::Agent>) }
    };
    let factory: Arc<dyn AcpSessionFactory> = Arc::new(factory);
    let registry = Arc::new(SessionRegistry::new(factory, 8));
    let config = Arc::new(AcpAdapterConfig::default());
    Arc::new(AcpConnectionServices::new(registry, config))
}

/**
共享连接服务只暴露唯一一个扩展调用权威，
并由统一关闭链将其与其它所有连接状态一起排空。
*/
#[tokio::test]
async fn connection_services_own_one_extension_authority_and_close_drains_it() -> Result<()> {
    let services = test_services();
    let authority = services.extensions();
    assert!(authority.admission_open());

    let held = match authority.lease(Some("extension-1/tool_execute"), CancellationToken::new()) {
        Ok(lease) => lease,
        Err(error) => panic!("lease: {error}"),
    };
    assert_eq!(authority.in_flight(), 1);

    /* 统一关闭链：先关闭准入，再取消在途调用，
    然后在关闭 Session 之前进行有界排空。 */
    services.close_admission();
    authority.close_admission();
    authority.cancel_all();
    assert_eq!(
        held.settlement(),
        Some(ExtensionSettlement::Cancelled),
        "connection close must settle in-flight extension invocations"
    );
    drop(held);
    let leaked = authority.drain(Duration::from_millis(200)).await;
    assert_eq!(leaked, 0);
    services.close_sessions().await?;
    Ok(())
}

/// 连接关闭超时时保留未结算的扩展租约，以便后续重试关闭。
#[tokio::test]
async fn connection_close_retains_unsettled_extension_for_retry() -> Result<()> {
    let services = test_services();
    let authority = services.extensions();
    let held = authority
        .lease(Some("extension-close-retry"), CancellationToken::new())
        .map_err(|error| echo_agent::error::ReactError::Other(error.to_string()))?;

    assert!(services.close(Duration::from_millis(20)).await.is_err());
    assert_eq!(authority.in_flight(), 1);
    assert_eq!(held.settlement(), Some(ExtensionSettlement::Cancelled));
    drop(held);
    services.close(Duration::from_secs(1)).await?;
    Ok(())
}

/**
由 prompt 驱动的运行在旁边有扩展租约执行时，仍使用同一个 Session/Run 权威；
运行通过框架回执结算，租约通过自身生命周期结算，二者互不代为结算
（设计 §12.1：一个 Run 权威，一个调用权威）。
*/
#[tokio::test]
async fn extension_leases_coexist_with_framework_runs_without_second_terminal() -> Result<()> {
    let services = test_services();
    services
        .sessions()
        .initialize(agent_client_protocol::schema::v1::ClientCapabilities::default())
        .await;
    let session = services
        .sessions()
        .create(agent_client_protocol::schema::v1::NewSessionRequest::new(
            std::path::PathBuf::from("/tmp/extension-runtime"),
        ))
        .await?;
    let acp_session = services
        .sessions()
        .get(&session)
        .await
        .ok_or_else(|| echo_agent::error::ReactError::Other("session missing".to_string()))?;
    let active = acp_session.begin_turn()?;
    let run_id = active.turn.id().to_string();
    let identity = EventIdentity::new(format!("stream-{run_id}"), run_id.clone())?;
    let turn = TurnRequest::new(identity, "run next to an extension lease")
        .mode(TurnMode::Chat)
        .cancel(active.turn.cancellation());
    let spec = echo_agent::acp::RunStartSpec {
        session: acp_session.clone(),
        active,
        run_id: run_id.clone(),
        stream_id: format!("stream-{run_id}"),
        turn,
        projector: None,
        journal: None,
        observers: Vec::new(),
    };
    let (entry, task) = services.prepare_run(spec).await?;
    tokio::spawn(task);

    let bridge = FakeBridge::new(services.extensions().clone());
    let invocation = bridge.invoke(
        None,
        CancellationToken::new(),
        TransportBehavior::AnswerAfter(Duration::from_millis(5)),
        Duration::from_secs(5),
    );
    let receipt = entry.wait_receipt().await;
    assert_eq!(
        receipt.outcome,
        echo_agent::runtime::TurnOutcome::Completed,
        "the framework receipt stays the only run terminal"
    );
    assert_eq!(invocation.await, ExtensionSettlement::Answered);
    let _ = services.remove_run(&run_id).await;
    Ok(())
}

/* 在某些 feature 组合下 EventEnvelope 只在文档位置被引用，
此辅助函数用于消除未使用警告。 */
/// 仅用于引用 `EventEnvelope` 类型的见证函数。
#[allow(dead_code)]
fn _envelope_type_witness() -> Option<EventEnvelope> {
    None
}
