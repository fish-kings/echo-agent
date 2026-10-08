pub mod http;
pub mod sse;
pub mod stdio;

use futures::future::BoxFuture;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use tokio::sync::{Notify, oneshot};
use tokio::time::Instant;

use super::types::{JsonRpcNotification, JsonRpcRequest, JsonRpcResponse};
use crate::redaction::{json_with_secrets, text_with_secrets};
use echo_core::error::{McpError, ReactError, Result};

/**
MCP 传输层抽象

负责在 Client 和 Server 之间传递 JSON-RPC 消息，
屏蔽底层通信细节（进程 stdin/stdout 或 HTTP）。
*/
pub trait McpTransport: Send + Sync {
    /// 发送请求并等待响应（传输层自动管理请求 ID）
    fn send(&self, request: JsonRpcRequest) -> BoxFuture<'_, Result<JsonRpcResponse>>;

    /// 发送通知（无需等待响应）
    fn notify(&self, notification: JsonRpcNotification) -> BoxFuture<'_, Result<()>>;

    /**
    等待传输层拥有的清理工作全部完成后关闭连接。

    成功返回意味着到达了一个安全的生命周期检查点：传输层的挂起请求
    注册表中不再有任何请求残留，且其拥有的 I/O 任务或子进程都已经
    到达各自受限的关闭终态。
    */
    fn close(&self) -> BoxFuture<'_, Result<()>>;

    /**
    获取通知接收通道（用于接收服务端推送的通知）
    返回 None 表示该传输层不支持通知接收
    */
    fn notification_rx(&self) -> Option<Arc<dyn super::types::JsonRpcNotificationReceiver>>;
}

type PendingSender = oneshot::Sender<Result<JsonRpcResponse>>;

/**
请求/响应类传输层共用的、支持取消安全的挂起请求注册表。

这里刻意使用同步 Mutex：注册操作从不跨越 await 点持有锁，
这样注册守卫（guard）就能在 `Drop` 中直接移除被放弃的请求，
而无需再启动一个额外的清理任务。
*/
pub(super) struct PendingRequests {
    /// 标记该注册表是否已关闭；关闭后拒绝新的注册。
    closed: AtomicBool,
    /// 请求 ID 到对应响应发送端的映射。
    entries: Mutex<HashMap<u64, PendingSender>>,
}

/// 一次挂起请求注册的守卫；`Drop` 时自动从注册表中移除对应条目。
pub(super) struct PendingRegistration {
    /// 该注册对应的请求 ID。
    id: u64,
    /// 所属的挂起请求注册表。
    pending: Arc<PendingRequests>,
}

/// 关闭流程的状态机。
#[derive(Clone)]
enum CloseStatus {
    /// 尚未开始关闭。
    Open,
    /// 关闭流程正在进行中。
    Closing,
    /// 关闭流程已成功完成。
    Settled,
    /// 关闭流程失败，携带失败原因。
    Failed(String),
}

/// 单飞（single-flight）关闭回执：其清理任务的生产者在调用方 drop 后依然存活。
pub(super) struct CloseCoordinator {
    /// 当前生效的关闭回执。
    current: Mutex<Arc<CloseReceipt>>,
}

/// 一次关闭流程的状态与等待通知。
pub(super) struct CloseReceipt {
    /// 关闭流程的当前状态。
    status: Mutex<CloseStatus>,
    /// 用于唤醒等待该回执状态变化的任务。
    changed: Notify,
}

impl CloseCoordinator {
    /// 创建一个初始状态为 `Open` 的关闭协调器。
    pub(super) fn new() -> Arc<Self> {
        Arc::new(Self {
            current: Mutex::new(Arc::new(CloseReceipt {
                status: Mutex::new(CloseStatus::Open),
                changed: Notify::new(),
            })),
        })
    }

    /**
    尝试开始一次关闭流程：若已有关闭在进行或已完成，则复用其回执并返回
    `false`（表示调用方无需再发起清理）；否则将状态置为 `Closing` 并返回
    `true`（表示调用方应负责执行清理并调用 `complete`）。
    */
    pub(super) fn begin(&self) -> (bool, Arc<CloseReceipt>) {
        let mut current = self
            .current
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let status = current.status().clone();
        let start = match status {
            CloseStatus::Open => {
                *current.status() = CloseStatus::Closing;
                true
            }
            CloseStatus::Failed(_) => {
                *current = Arc::new(CloseReceipt {
                    status: Mutex::new(CloseStatus::Closing),
                    changed: Notify::new(),
                });
                true
            }
            CloseStatus::Closing | CloseStatus::Settled => false,
        };
        (start, Arc::clone(&current))
    }
}

impl CloseReceipt {
    /// 加锁获取当前关闭状态。
    fn status(&self) -> MutexGuard<'_, CloseStatus> {
        self.status
            .lock()
            .unwrap_or_else(|error| error.into_inner())
    }

    /// 将关闭流程标记为完成（成功或失败），并唤醒所有等待者。
    pub(super) fn complete(&self, result: &Result<()>) {
        *self.status() = match result {
            Ok(()) => CloseStatus::Settled,
            Err(error) => CloseStatus::Failed(error.to_string()),
        };
        self.changed.notify_waiters();
    }

    /// 等待该关闭回执结算（成功或失败），超过 `deadline` 仍未结算则返回超时错误。
    pub(super) async fn wait(&self, deadline: Instant, label: &str) -> Result<()> {
        loop {
            let changed = self.changed.notified();
            let status = self.status().clone();
            match status {
                CloseStatus::Open => {
                    return Err(close_error(format!(
                        "{label} close coordinator was not started"
                    )));
                }
                CloseStatus::Settled => return Ok(()),
                CloseStatus::Failed(error) => return Err(close_error(error)),
                CloseStatus::Closing => {
                    tokio::time::timeout_at(deadline, changed)
                        .await
                        .map_err(|_| {
                            close_error(format!(
                                "{label} close receipt did not settle before deadline"
                            ))
                        })?;
                }
            }
        }
    }
}

impl PendingRequests {
    /// 创建一个空的、未关闭的挂起请求注册表。
    pub(super) fn new() -> Arc<Self> {
        Arc::new(Self {
            closed: AtomicBool::new(false),
            entries: Mutex::new(HashMap::new()),
        })
    }

    /// 加锁获取内部的请求条目表。
    fn entries(&self) -> MutexGuard<'_, HashMap<u64, PendingSender>> {
        self.entries
            .lock()
            .unwrap_or_else(|error| error.into_inner())
    }

    /**
    注册一个新的挂起请求，返回用于接收响应的 receiver 以及注册守卫；
    若注册表已关闭则返回错误。
    */
    pub(super) fn register(
        self: &Arc<Self>,
        id: u64,
    ) -> Result<(
        oneshot::Receiver<Result<JsonRpcResponse>>,
        PendingRegistration,
    )> {
        let (sender, receiver) = oneshot::channel();
        let mut entries = self.entries();
        if self.closed.load(Ordering::Acquire) {
            return Err(transport_closed_error());
        }
        entries.insert(id, sender);
        drop(entries);
        Ok((
            receiver,
            PendingRegistration {
                id,
                pending: Arc::clone(self),
            },
        ))
    }

    /// 用给定响应结算指定 ID 对应的挂起请求（若存在）。
    pub(super) fn resolve(&self, id: u64, response: JsonRpcResponse) {
        if let Some(sender) = self.entries().remove(&id) {
            let _ = sender.send(Ok(response));
        }
    }

    /// 阻止新的请求注册，并让所有已接受的挂起请求以失败结算。返回被结算的请求数。
    pub(super) fn close(&self) -> usize {
        self.closed.store(true, Ordering::Release);
        let entries = std::mem::take(&mut *self.entries());
        let count = entries.len();
        for sender in entries.into_values() {
            let _ = sender.send(Err(transport_closed_error()));
        }
        count
    }

    /// 若注册表已关闭则返回错误，否则返回 `Ok(())`。
    pub(super) fn ensure_open(&self) -> Result<()> {
        if self.closed.load(Ordering::Acquire) {
            Err(transport_closed_error())
        } else {
            Ok(())
        }
    }

    /// 返回当前挂起请求的数量（仅测试使用）。
    #[cfg(test)]
    pub(super) fn len(&self) -> usize {
        self.entries().len()
    }
}

impl Drop for PendingRegistration {
    /// 从注册表中移除该请求条目，用于清理被放弃（未收到响应）的请求。
    fn drop(&mut self) {
        self.pending.entries().remove(&self.id);
    }
}

/// 构造一个表示“传输层已关闭”的错误。
pub(super) fn transport_closed_error() -> ReactError {
    ReactError::Mcp(Box::new(McpError::TransportClosed))
}

/// 构造一个带有指定消息的连接失败错误。
pub(super) fn close_error(message: impl Into<String>) -> ReactError {
    ReactError::Mcp(Box::new(McpError::ConnectionFailed(message.into())))
}

/// 对 JSON-RPC 响应中的错误信息与错误数据做脱敏处理，移除其中的敏感值。
pub(super) fn redact_response_error(response: &mut JsonRpcResponse, secrets: &[String]) {
    let Some(error) = response.error.as_mut() else {
        return;
    };
    error.message = text_with_secrets(&error.message, secrets.iter());
    if let Some(data) = error.data.as_mut() {
        *data = json_with_secrets(data, secrets);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcp::types::JsonRpcError;

    #[tokio::test]
    async fn retry_cannot_overwrite_original_waiters_failure() -> Result<()> {
        let coordinator = CloseCoordinator::new();
        let (start, first) = coordinator.begin();
        assert!(start);
        let (start, waiter) = coordinator.begin();
        assert!(!start);
        let deadline = Instant::now() + std::time::Duration::from_secs(1);
        let waiting = waiter.wait(deadline, "original");
        tokio::pin!(waiting);
        assert!(futures::poll!(&mut waiting).is_pending());
        first.complete(&Err(close_error("original failure")));
        let (start, retry) = coordinator.begin();
        assert!(start);
        retry.complete(&Ok(()));
        let error = waiting
            .await
            .err()
            .ok_or_else(|| close_error("lost original failure"))?;
        assert!(error.to_string().contains("original failure"));
        retry.wait(deadline, "retry").await?;
        assert!(first.wait(deadline, "first").await.is_err());
        Ok(())
    }

    #[test]
    fn transport_error_redaction_removes_configured_values() {
        let mut response = JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id: Some(serde_json::json!(1)),
            result: None,
            error: Some(JsonRpcError {
                code: -32000,
                message: "server echoed opaque-secret".to_string(),
                data: Some(serde_json::json!({"detail": "opaque-secret"})),
            }),
        };
        redact_response_error(&mut response, &["opaque-secret".to_string()]);
        let debug = format!("{response:?}");
        assert!(!debug.contains("opaque-secret"));
    }

    #[tokio::test]
    async fn abandoned_registration_is_removed_without_async_cleanup() -> Result<()> {
        let pending = PendingRequests::new();
        let (_receiver, registration) = pending.register(7)?;
        assert_eq!(pending.len(), 1);

        drop(registration);

        assert_eq!(pending.len(), 0);
        Ok(())
    }

    #[tokio::test]
    async fn close_fails_every_pending_request_and_fences_new_registration() -> Result<()> {
        let pending = PendingRequests::new();
        let (first, _first_registration) = pending.register(1)?;
        let (second, _second_registration) = pending.register(2)?;

        assert_eq!(pending.close(), 2);
        assert!(matches!(first.await, Ok(Err(ReactError::Mcp(_)))));
        assert!(matches!(second.await, Ok(Err(ReactError::Mcp(_)))));
        assert!(pending.register(3).is_err());
        Ok(())
    }
}
