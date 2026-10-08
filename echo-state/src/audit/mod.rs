/*!
审计日志

完整记录 tool 调用链、护栏阻断、权限拒绝等事件，支持合规审查。

# 核心类型

- [`AuditEvent`][]: 审计事件
- [`AuditLogger`]: 日志记录器 trait
- [`AuditCallback`]: 基于 `AgentCallback` 的自动审计
*/

/// 文件审计日志后端。
pub mod file;
/// 内存审计日志后端。
pub mod memory;

pub use echo_core::audit::*;

use echo_core::utils::retention::ContentRetentionPolicy;
use futures::future::BoxFuture;
use serde_json::Value;
use std::cell::Cell;
use std::collections::HashMap;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{SyncSender, TrySendError};
use std::sync::{Arc, Mutex};

/// 诊断失败通知队列的容量上限。
const DIAGNOSTIC_DELIVERY_CAPACITY: usize = 1_024;
/// 进程级计数器：未能投递的诊断失败通知数量。
static DIAGNOSTIC_DELIVERY_DROPPED: AtomicU64 = AtomicU64::new(0);

thread_local! {
    /// 标记当前线程是否正处于诊断分发过程中，用于检测重入上报。
    static IN_DIAGNOSTIC_DISPATCH: Cell<bool> = const { Cell::new(false) };
}

/// 持久化投递失败的诊断记录类别。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DiagnosticRecordKind {
    /// 通过 `RunStore` 存储的结构化执行追踪。
    Trace,
    /// 通过 [`AuditLogger`] 存储的审计事件。
    Audit,
}

impl DiagnosticRecordKind {
    /// 返回供结构化观察者使用的稳定小写标识符。
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Trace => "trace",
            Self::Audit => "audit",
        }
    }
}

/// 产生诊断投递失败的持久化操作。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DiagnosticDeliveryOperation {
    /// 持久化初始诊断记录。
    Start,
    /// 向已有诊断记录追加一个事件。
    Append,
    /// 加载终态诊断收尾所需的记录。
    Load,
    /// 持久化终态诊断投影。
    Finalize,
    /// 持久化一条独立的审计记录。
    Record,
}

impl DiagnosticDeliveryOperation {
    /// 返回供结构化观察者使用的稳定小写标识符。
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Start => "start",
            Self::Append => "append",
            Self::Load => "load",
            Self::Finalize => "finalize",
            Self::Record => "record",
        }
    }
}

/**
诊断持久化失败时发出的结构化事实。

该事实刻意与 Agent 执行结果分离。观察者可以保留、计数或据此告警，
但无法改写 Agent 运行时已经确定的生产者终态。
*/
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct DiagnosticDeliveryFailure {
    /// 后端失败发生的时间（早于异步上报）。
    pub occurred_at: chrono::DateTime<chrono::Utc>,
    /// 追踪或审计记录类别。
    pub record_kind: DiagnosticRecordKind,
    /// 失败的持久化操作。
    pub operation: DiagnosticDeliveryOperation,
    /// 可用时的 run、trace 或 session 标识。
    pub record_id: Option<String>,
    /// 不含诊断载荷的后端错误信息。
    pub error: String,
}

impl DiagnosticDeliveryFailure {
    /// 构造一条诊断投递失败事实（会对错误信息应用默认保留策略）。
    pub fn new(
        record_kind: DiagnosticRecordKind,
        operation: DiagnosticDeliveryOperation,
        record_id: Option<String>,
        error: impl Into<String>,
    ) -> Self {
        let mut failure = Self {
            occurred_at: chrono::Utc::now(),
            record_kind,
            operation,
            record_id,
            error: error.into(),
        };
        failure.apply_retention();
        failure
    }

    /// 使用默认内容保留策略清洗错误文本（脱敏与截断）。
    fn apply_retention(&mut self) {
        self.error = ContentRetentionPolicy::default().sanitize_text(&self.error);
    }
}

/**
诊断持久化失败的消费者。

框架在一个有界的诊断分发器上调用此回调，而不是在 Agent 生产者上调用。
该回调仅用于观察，没有控制性返回值。实现仍应尽快返回，以免延迟后续的诊断通知。
若应用需要把审计写入作为业务提交的一部分，必须直接调用 [`AuditLogger::log`]
并在该提交边界上绑定其 `Result`，而不是依赖 Agent 回调。

# 示例

```rust
use echo_state::audit::{DiagnosticDeliveryFailure, DiagnosticDeliveryObserver};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Default)]
struct FailureCounter(AtomicUsize);

impl DiagnosticDeliveryObserver for FailureCounter {
    fn on_failure(&self, _failure: DiagnosticDeliveryFailure) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }
}

let observer: Arc<dyn DiagnosticDeliveryObserver> = Arc::new(FailureCounter::default());
let _ = observer;
```
*/
pub trait DiagnosticDeliveryObserver: Send + Sync {
    /// 观察一次失败的诊断持久化操作。
    fn on_failure(&self, failure: DiagnosticDeliveryFailure);
}

/// 默认观察者：发出一个稳定的结构化 `tracing` 事件。
#[derive(Debug, Default)]
struct TracingDiagnosticDeliveryObserver;

impl DiagnosticDeliveryObserver for TracingDiagnosticDeliveryObserver {
    /// 对失败信息应用保留策略后，以结构化 tracing 错误事件输出。
    fn on_failure(&self, mut failure: DiagnosticDeliveryFailure) {
        failure.apply_retention();
        tracing::error!(
            target: "echo_agent::diagnostic_delivery",
            record_kind = failure.record_kind.as_str(),
            operation = failure.operation.as_str(),
            record_id_present = failure.record_id.is_some(),
            occurred_at = %failure.occurred_at,
            error = %failure.error,
            "diagnostic persistence delivery failed"
        );
    }
}

/// 发送给诊断分发线程的一条通知。
struct DiagnosticDeliveryNotification {
    /// 可选的自定义观察者；为空时仅触发默认 tracing 观察者。
    observer: Option<Arc<dyn DiagnosticDeliveryObserver>>,
    /// 待投递的失败事实。
    failure: DiagnosticDeliveryFailure,
}

/// 以饱和方式递增诊断通知丢弃计数。
fn increment_diagnostic_delivery_dropped() {
    let _ =
        DIAGNOSTIC_DELIVERY_DROPPED.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
            Some(value.saturating_add(1))
        });
}

/// 惰性初始化并返回诊断分发线程的发送端；线程创建失败时返回 `None`。
fn diagnostic_delivery_sender() -> Option<&'static SyncSender<DiagnosticDeliveryNotification>> {
    /// 诊断分发线程发送端的全局单例。
    static SENDER: OnceLock<Option<SyncSender<DiagnosticDeliveryNotification>>> = OnceLock::new();
    SENDER
        .get_or_init(|| {
            let (sender, receiver) =
                std::sync::mpsc::sync_channel::<DiagnosticDeliveryNotification>(
                    DIAGNOSTIC_DELIVERY_CAPACITY,
                );
            let spawn = std::thread::Builder::new()
                .name("echo-diagnostic-delivery".to_string())
                .spawn(move || {
                    while let Ok(notification) = receiver.recv() {
                        IN_DIAGNOSTIC_DISPATCH.with(|in_dispatch| {
                            in_dispatch.set(true);
                            let delivered =
                                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                    TracingDiagnosticDeliveryObserver
                                        .on_failure(notification.failure.clone());
                                    if let Some(observer) = notification.observer {
                                        observer.on_failure(notification.failure);
                                    }
                                }));
                            in_dispatch.set(false);
                            if delivered.is_err() {
                                increment_diagnostic_delivery_dropped();
                            }
                        });
                    }
                });
            match spawn {
                Ok(_) => Some(sender),
                Err(_) => None,
            }
        })
        .as_ref()
}

/**
未能投递的诊断失败通知数量。

这是一个饱和的进程级计数器，涵盖分发器初始化失败、队列满/断开、
重入上报以及观察者 unwind 等情况。它只是尽力而为的自诊断信息，不是执行结果。
*/
pub fn diagnostic_delivery_dropped_count() -> u64 {
    DIAGNOSTIC_DELIVERY_DROPPED.load(Ordering::Relaxed)
}

/// 在生产者工作开始前初始化进程级诊断分发器，返回是否初始化成功。
#[doc(hidden)]
pub fn initialize_diagnostic_delivery() -> bool {
    diagnostic_delivery_sender().is_some()
}

/**
将默认的结构化失败事件以及可选的自定义观察者通知放入队列，
不会阻塞 Agent 生产者。
*/
#[doc(hidden)]
pub fn report_diagnostic_delivery_failure(
    observer: Option<Arc<dyn DiagnosticDeliveryObserver>>,
    mut failure: DiagnosticDeliveryFailure,
) {
    failure.apply_retention();
    if IN_DIAGNOSTIC_DISPATCH.with(Cell::get) {
        increment_diagnostic_delivery_dropped();
        return;
    }
    let Some(sender) = diagnostic_delivery_sender() else {
        increment_diagnostic_delivery_dropped();
        return;
    };
    let notification = DiagnosticDeliveryNotification { observer, failure };
    if matches!(
        sender.try_send(notification),
        Err(TrySendError::Full(_)) | Err(TrySendError::Disconnected(_))
    ) {
        increment_diagnostic_delivery_dropped();
    }
}

/// 存储工具调用开始时的信息
struct ToolCallInfo {
    /// 已按保留策略清洗过的调用参数。
    args: Value,
    /// 调用开始时刻，用于计算耗时。
    started_at: std::time::Instant,
}

/**
基于 `AgentCallback` 的审计日志自动记录器

实现 `AgentCallback`，将所有回调事件自动写入 `AuditLogger`。

# 示例

```rust
use echo_state::audit::{memory::InMemoryAuditLogger, AuditCallback};
use std::sync::Arc;

let logger = Arc::new(InMemoryAuditLogger::new());
let audit_cb = Arc::new(AuditCallback::new(logger, "my-agent", None));
// 将 `audit_cb` 接入你自己的 agent/runtime 层，或通过 `echo_agent` façade 使用。
let _ = audit_cb;
```
*/
pub struct AuditCallback {
    /// 审计事件的写入后端。
    logger: Arc<dyn AuditLogger>,
    /// 写入前应用的内容保留（脱敏/截断）策略。
    retention: ContentRetentionPolicy,
    /// 审计后端写入失败时通知的自定义观察者。
    diagnostic_delivery_observer: Option<Arc<dyn DiagnosticDeliveryObserver>>,
    /// 事件归属的 Agent 名称。
    agent_name: String,
    /// 事件归属的会话 ID。
    session_id: Option<String>,
    /// tool_call_id → ToolCallInfo（参数 + 开始时间）
    tool_calls: Mutex<HashMap<String, ToolCallInfo>>,
    /**
    用于生成唯一 tool_call_id 的单调递增计数器。
    使用全局计数器而非按工具名分别计数，可避免并发调用共享同一工具名时查找出现前缀冲突。
    */
    next_call_id: AtomicU64,
}

impl AuditCallback {
    /// 使用给定的审计后端、Agent 名称与可选会话 ID 创建审计回调，并预先初始化诊断分发器。
    pub fn new(
        logger: Arc<dyn AuditLogger>,
        agent_name: impl Into<String>,
        session_id: Option<String>,
    ) -> Self {
        let _ = initialize_diagnostic_delivery();
        Self {
            logger,
            retention: ContentRetentionPolicy::default(),
            diagnostic_delivery_observer: None,
            agent_name: agent_name.into(),
            session_id,
            tool_calls: Mutex::new(HashMap::new()),
            next_call_id: AtomicU64::new(1),
        }
    }

    /**
    替换审计后端拒绝回调记录时所使用的观察者。
    该回调仅用于观察，永远不会改变执行流程。
    */
    pub fn with_diagnostic_delivery_observer(
        mut self,
        observer: Arc<dyn DiagnosticDeliveryObserver>,
    ) -> Self {
        self.diagnostic_delivery_observer = Some(observer);
        self
    }

    /// 设置内容保留策略，并用新策略重新清洗已记录的在途调用参数。
    pub fn with_retention_policy(mut self, retention: ContentRetentionPolicy) -> Self {
        for info in self
            .tool_calls
            .get_mut()
            .unwrap_or_else(|e| e.into_inner())
            .values_mut()
        {
            retention.sanitize_json(&mut info.args);
        }
        self.retention = retention;
        self
    }

    /// 对事件应用保留策略后写入后端；写入失败时异步上报诊断失败，不影响调用方。
    async fn record_event(&self, mut event: AuditEvent) {
        event.apply_retention(&self.retention);
        let record_id = event.trace_id.clone().or_else(|| event.session_id.clone());
        if let Err(error) = self.logger.log(event).await {
            report_diagnostic_delivery_failure(
                self.diagnostic_delivery_observer.clone(),
                DiagnosticDeliveryFailure::new(
                    DiagnosticRecordKind::Audit,
                    DiagnosticDeliveryOperation::Record,
                    record_id,
                    error.to_string(),
                ),
            );
        }
    }

    /**
    为工具调用生成唯一的 map key。

    Key 格式为 `"{tool}#{n}"`：以工具名为前缀 + 全局单调递增序号。前缀让
    [`pop_tool_call`] 能按工具名做前缀匹配（一个工具名下可能有多个并发
    in-flight 调用），全局序号保证 key 唯一、且可比较出最旧的调用。

    之前实现返回 `call_{n}`，而 [`pop_tool_call`] 按 `"{tool}#"` 前缀查 ——
    两者格式不匹配，pop 永远命中不了，导致 on_tool_end/on_tool_error 的
    duration_ms 恒为 0、input 恒为 Null（P1-13）。统一为 `tool#N` 修复。
    */
    fn make_tool_call_id(&self, tool: &str) -> String {
        let n = self.next_call_id.fetch_add(1, Ordering::Relaxed);
        format!("{}#{}", tool, n)
    }

    /// 以当前会话与 Agent 名称构造一个带当前时间戳的审计事件。
    fn make_event(&self, event_type: AuditEventType) -> AuditEvent {
        AuditEvent::now(self.session_id.clone(), self.agent_name.clone(), event_type)
    }

    /**
    按工具名查找并移除最旧的在途工具调用。

    对 key 中的工具名部分做精确前缀匹配（不依赖迭代顺序），然后移除并返回
    内嵌序号最小的条目。即使并发调用共享同一工具名，结果也是确定的。
    */
    fn pop_tool_call(&self, tool: &str) -> Option<ToolCallInfo> {
        let mut map = self
            .tool_calls
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        // 收集所有匹配该工具追踪前缀（"{tool}#N"）的 key。
        let candidates: Vec<String> = map
            .keys()
            .filter(|k| k.starts_with(&format!("{}#", tool)))
            .cloned()
            .collect();
        if candidates.is_empty() {
            return None;
        }
        // 移除序号最小（最旧）的条目。
        let key = candidates.into_iter().min_by_key(|k| {
            k.split('#')
                .nth(1)
                .and_then(|s| s.parse::<u64>().ok())
                .unwrap_or(u64::MAX)
        })?;
        map.remove(&key)
    }

    /// 记录一次工具调用的开始：清洗参数后连同开始时间存入在途表。
    fn remember_tool_call(&self, call_id: String, args: &Value) {
        let mut args = args.clone();
        self.retention.sanitize_json(&mut args);
        self.tool_calls
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .insert(
                call_id,
                ToolCallInfo {
                    args,
                    started_at: std::time::Instant::now(),
                },
            );
    }

    /// 按精确 call_id 取出并移除在途工具调用记录。
    fn take_tool_call(&self, call_id: &str) -> Option<ToolCallInfo> {
        self.tool_calls
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .remove(call_id)
    }
}

impl echo_core::agent::AgentCallback for AuditCallback {
    /// 工具开始（无 call_id）：生成 `tool#N` 形式的内部 key 并记录参数。
    fn on_tool_start<'a>(
        &'a self,
        _agent: &'a str,
        tool: &'a str,
        args: &'a Value,
    ) -> BoxFuture<'a, ()> {
        Box::pin(async move {
            let tool_call_id = self.make_tool_call_id(tool);
            self.remember_tool_call(tool_call_id, args);
        })
    }

    /// 工具开始（带 call_id）：以 call_id 为 key 记录参数。
    fn on_tool_start_with_id<'a>(
        &'a self,
        _agent: &'a str,
        call_id: &'a str,
        _tool: &'a str,
        args: &'a Value,
    ) -> BoxFuture<'a, ()> {
        Box::pin(async move {
            self.remember_tool_call(call_id.to_string(), args);
        })
    }

    /// 工具成功结束（无 call_id）：按工具名匹配最旧调用并写入成功的 ToolCall 事件。
    fn on_tool_end<'a>(
        &'a self,
        _agent: &'a str,
        tool: &'a str,
        result: &'a str,
    ) -> BoxFuture<'a, ()> {
        Box::pin(async move {
            let (duration_ms, input) = self
                .pop_tool_call(tool)
                .map(|info| (info.started_at.elapsed().as_millis() as u64, info.args))
                .unwrap_or((0, Value::Null));

            let event = self.make_event(AuditEventType::ToolCall {
                call_id: None,
                tool: tool.to_string(),
                input,
                output: result.to_string(),
                success: true,
                duration_ms,
            });
            self.record_event(event).await;
        })
    }

    /// 工具成功结束（带 call_id）：按 call_id 关联并写入成功的 ToolCall 事件。
    fn on_tool_end_with_id<'a>(
        &'a self,
        _agent: &'a str,
        call_id: &'a str,
        tool: &'a str,
        result: &'a str,
    ) -> BoxFuture<'a, ()> {
        Box::pin(async move {
            let (duration_ms, input) = self
                .take_tool_call(call_id)
                .map(|info| (info.started_at.elapsed().as_millis() as u64, info.args))
                .unwrap_or((0, Value::Null));
            let event = self.make_event(AuditEventType::ToolCall {
                call_id: Some(call_id.to_string()),
                tool: tool.to_string(),
                input,
                output: result.to_string(),
                success: true,
                duration_ms,
            });
            self.record_event(event).await;
        })
    }

    /// 工具执行出错（无 call_id）：按工具名匹配最旧调用并写入失败的 ToolCall 事件。
    fn on_tool_error<'a>(
        &'a self,
        _agent: &'a str,
        tool: &'a str,
        err: &'a echo_core::error::ReactError,
    ) -> BoxFuture<'a, ()> {
        Box::pin(async move {
            let (duration_ms, input) = self
                .pop_tool_call(tool)
                .map(|info| (info.started_at.elapsed().as_millis() as u64, info.args))
                .unwrap_or((0, Value::Null));

            let event = self.make_event(AuditEventType::ToolCall {
                call_id: None,
                tool: tool.to_string(),
                input,
                output: err.to_string(),
                success: false,
                duration_ms,
            });
            self.record_event(event).await;
        })
    }

    /// 工具执行出错（带 call_id）：按 call_id 关联并写入失败的 ToolCall 事件。
    fn on_tool_error_with_id<'a>(
        &'a self,
        _agent: &'a str,
        call_id: &'a str,
        tool: &'a str,
        error: &'a echo_core::error::ReactError,
    ) -> BoxFuture<'a, ()> {
        Box::pin(async move {
            let (duration_ms, input) = self
                .take_tool_call(call_id)
                .map(|info| (info.started_at.elapsed().as_millis() as u64, info.args))
                .unwrap_or((0, Value::Null));
            let event = self.make_event(AuditEventType::ToolCall {
                call_id: Some(call_id.to_string()),
                tool: tool.to_string(),
                input,
                output: error.to_string(),
                success: false,
                duration_ms,
            });
            self.record_event(event).await;
        })
    }

    /**
    工具被中断（带 call_id）：优先使用开始时记录的参数；若从未开始，
    则使用准入时的输入（经保留策略清洗）并将耗时记为 0。
    */
    fn on_tool_interrupted_with_id<'a>(
        &'a self,
        _agent: &'a str,
        call_id: &'a str,
        tool: &'a str,
        input: &'a Value,
        error: &'a echo_core::error::ReactError,
    ) -> BoxFuture<'a, ()> {
        Box::pin(async move {
            let (duration_ms, input) = self
                .take_tool_call(call_id)
                .map(|info| (info.started_at.elapsed().as_millis() as u64, info.args))
                .unwrap_or_else(|| {
                    let mut input = input.clone();
                    self.retention.sanitize_json(&mut input);
                    (0, input)
                });
            let event = self.make_event(AuditEventType::ToolCall {
                call_id: Some(call_id.to_string()),
                tool: tool.to_string(),
                input,
                output: error.to_string(),
                success: false,
                duration_ms,
            });
            self.record_event(event).await;
        })
    }

    /// 最终回答：写入 FinalAnswer 审计事件。
    fn on_final_answer<'a>(&'a self, _agent: &'a str, answer: &'a str) -> BoxFuture<'a, ()> {
        Box::pin(async move {
            let event = self.make_event(AuditEventType::FinalAnswer {
                content: answer.to_string(),
            });
            self.record_event(event).await;
        })
    }
}

/// 审计模块的单元测试。
#[cfg(test)]
mod tests {
    use super::*;
    use echo_core::agent::AgentCallback;
    use echo_core::error::{ReactError, Result};

    /// 在临时 tracing 订阅者下执行 `action`，捕获期间产生的日志字段文本。
    pub(super) fn capture_audit_logs<T>(action: impl FnOnce() -> T) -> (T, String) {
        /// 将 tracing 事件字段写入共享字符串缓冲区的捕获订阅者。
        struct Capture(Arc<Mutex<String>>);

        impl tracing::field::Visit for Capture {
            /// 以 `name=value;` 形式记录字符串字段。
            fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
                use std::fmt::Write;
                let _ = write!(
                    self.0.lock().unwrap_or_else(|e| e.into_inner()),
                    "{}={value:?};",
                    field.name()
                );
            }

            /// 以 `name=value;` 形式记录 Debug 字段。
            fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
                use std::fmt::Write;
                let _ = write!(
                    self.0.lock().unwrap_or_else(|e| e.into_inner()),
                    "{}={value:?};",
                    field.name()
                );
            }
        }

        impl tracing::Subscriber for Capture {
            /// 对所有 callsite 始终声明感兴趣。
            fn register_callsite(
                &self,
                _: &'static tracing::Metadata<'static>,
            ) -> tracing::subscriber::Interest {
                tracing::subscriber::Interest::always()
            }

            /// 启用所有元数据。
            fn enabled(&self, _: &tracing::Metadata<'_>) -> bool {
                true
            }
            /// 为新 span 分配固定 ID。
            fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
                tracing::span::Id::from_u64(1)
            }
            /// 忽略 span 记录。
            fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
            /// 忽略 span 之间的 follows-from 关系。
            fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
            /// 将事件字段写入捕获缓冲区。
            fn event(&self, event: &tracing::Event<'_>) {
                event.record(&mut Capture(self.0.clone()));
            }
            /// 忽略进入 span。
            fn enter(&self, _: &tracing::span::Id) {}
            /// 忽略退出 span。
            fn exit(&self, _: &tracing::span::Id) {}
        }

        let output = Arc::new(Mutex::new(String::new()));
        let capture = tracing::Dispatch::new(Capture(output.clone()));
        /*
        当只存在一个 Dispatch 时，若某 callsite 首次在另一线程上被观察到，
        tracing-core 0.1.36 可能缓存 `Interest::never`。
        这里额外保持注册第二个 Dispatch，使缓存重建走多 dispatch 路径。
        待 tokio-rs/tracing#3611 合入后移除此处理。
        */
        let _callsite_cache_guard =
            tracing::Dispatch::new(tracing::subscriber::NoSubscriber::default());
        let result = tracing::dispatcher::with_default(&capture, action);
        let text = output.lock().unwrap_or_else(|e| e.into_inner()).clone();
        (result, text)
    }

    /// 验证默认诊断日志会对被修改过的错误信息和记录 ID 做脱敏。
    #[test]
    fn audit_default_diagnostic_logging_sanitizes_mutated_error() {
        let mut failure = sample_failure();
        failure.record_id = Some("token=raw-record-identity-secret".into());
        failure.error = "Bearer abcdefghijklmnopqrstuvwxyz".into();
        let (_, logs) = capture_audit_logs(|| {
            /*
            确定性地复现上游竞态：Capture 注册后，另一个没有订阅者的线程
            先于 Capture 线程观察到这个静态 callsite。
            */
            let uncaptured = sample_failure();
            std::thread::spawn(move || {
                TracingDiagnosticDeliveryObserver.on_failure(uncaptured);
            })
            .join()
            .unwrap_or(());
            TracingDiagnosticDeliveryObserver.on_failure(failure);
        });
        assert!(logs.contains("diagnostic persistence delivery failed"));
        assert!(logs.contains("[REDACTED]"));
        assert!(!logs.contains("abcdefghijklmnopqrstuvwxyz"));
        assert!(!logs.contains("raw-record-identity-secret"));
    }

    /// 验证内存审计日志默认保留策略及策略变更后都会净化存储副本。
    #[tokio::test]
    async fn audit_memory_default_retention_and_policy_change_sanitize_stored_copies() -> Result<()>
    {
        let logger = memory::InMemoryAuditLogger::new();
        let event = AuditEvent::now(
            None,
            "agent".into(),
            AuditEventType::ToolCall {
                call_id: Some("retention-call".into()),
                tool: "shell".into(),
                input: serde_json::json!({"password":"tiny", "count":9}),
                output: format!("Bearer abcdefghijklmnopqrstuvwxyz {}", "中".repeat(17_000)),
                success: true,
                duration_ms: 100,
            },
        );
        logger.log(event.clone()).await?;
        let before = serde_json::to_string(&logger.snapshot())?;
        assert!(!before.contains("tiny"));
        assert!(!before.contains("abcdefghijklmnopqrstuvwxyz"));
        assert!(before.contains("[TRUNCATED]"));
        assert!(serde_json::to_string(&event)?.contains("tiny"));
        let logger = logger.with_retention_policy(ContentRetentionPolicy {
            max_string_chars: 0,
            max_array_items: 0,
        });
        let stored = logger.query(AuditFilter::default()).await?;
        assert!(
            matches!(stored.first().map(|e| &e.event_type), Some(AuditEventType::ToolCall { input, output, duration_ms: 100, .. }) if input.as_object().is_some_and(|fields| fields.len() == 1 && fields.values().any(|value| value.as_str() == Some("[TRUNCATED OBJECT]"))) && output == "...[TRUNCATED]")
        );
        Ok(())
    }

    /// 将审计事件原样记录到内存列表中的测试日志器。
    #[derive(Default)]
    struct RecordingAuditLogger(Mutex<Vec<AuditEvent>>);

    impl AuditLogger for RecordingAuditLogger {
        /// 追加一条审计事件。
        fn log<'a>(&'a self, event: AuditEvent) -> BoxFuture<'a, Result<()>> {
            Box::pin(async move {
                self.0.lock().unwrap_or_else(|e| e.into_inner()).push(event);
                Ok(())
            })
        }

        /// 返回全部已记录事件，忽略过滤条件。
        fn query<'a>(&'a self, _filter: AuditFilter) -> BoxFuture<'a, Result<Vec<AuditEvent>>> {
            Box::pin(async move { Ok(self.0.lock().unwrap_or_else(|e| e.into_inner()).clone()) })
        }
    }

    /// 验证自定义后端只会收到经过保留策略处理的副本。
    #[tokio::test]
    async fn audit_callback_custom_backend_receives_only_retained_copies() -> Result<()> {
        let logger = Arc::new(RecordingAuditLogger::default());
        let callback = AuditCallback::new(logger.clone(), "agent", Some("session".into()));
        let args = serde_json::json!({"password": "tiny", "count": 9});
        callback.on_tool_start("agent", "shell", &args).await;
        {
            let calls = callback
                .tool_calls
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            assert!(
                calls
                    .values()
                    .all(|info| !info.args.to_string().contains("tiny"))
            );
        }
        assert_eq!(args.get("password"), Some(&Value::String("tiny".into())));
        callback
            .on_tool_end("agent", "shell", "Bearer abcdefghijklmnopqrstuvwxyz")
            .await;
        callback.on_tool_start("agent", "shell", &args).await;
        callback
            .on_tool_error(
                "agent",
                "shell",
                &ReactError::Other("password=abcdefgh".into()),
            )
            .await;
        callback
            .on_final_answer("agent", r#"{"secret":"short"}"#)
            .await;
        let events = logger.query(AuditFilter::default()).await?;
        assert_eq!(events.len(), 3);
        let encoded = serde_json::to_string(&events)?;
        for secret in ["tiny", "abcdefghijklmnopqrstuvwxyz", "abcdefgh", "short"] {
            assert!(!encoded.contains(secret));
        }
        assert!(matches!(
            events.first().map(|e| &e.event_type),
            Some(AuditEventType::ToolCall { success: true, .. })
        ));
        assert!(matches!(
            events.get(1).map(|e| &e.event_type),
            Some(AuditEventType::ToolCall { success: false, .. })
        ));
        Ok(())
    }

    /// 验证锁中毒后，乱序完成的工具调用仍能按 call_id 正确关联。
    #[tokio::test]
    async fn audit_callback_correlates_reverse_completion_by_call_id_after_poison() -> Result<()> {
        let logger = Arc::new(RecordingAuditLogger::default());
        let callback = AuditCallback::new(logger.clone(), "agent", Some("session".into()));
        let poisoned = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = callback
                .tool_calls
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            std::panic::resume_unwind(Box::new("poison callback correlation".to_string()));
        }));
        assert!(poisoned.is_err());

        callback
            .on_tool_start_with_id(
                "agent",
                "call-a",
                "shell",
                &serde_json::json!({"label": "a"}),
            )
            .await;
        callback
            .on_tool_start_with_id(
                "agent",
                "call-b",
                "shell",
                &serde_json::json!({"label": "b"}),
            )
            .await;
        callback
            .on_tool_end_with_id("agent", "call-b", "shell", "output-b")
            .await;
        callback
            .on_tool_error_with_id(
                "agent",
                "call-a",
                "shell",
                &ReactError::Other("error-a".to_string()),
            )
            .await;

        let events = logger.query(AuditFilter::default()).await?;
        assert!(matches!(
            events.first().map(|event| &event.event_type),
            Some(AuditEventType::ToolCall { call_id: Some(call_id), input, output, success: true, .. })
            if call_id == "call-b" && input.get("label") == Some(&serde_json::json!("b")) && output == "output-b"
        ));
        assert!(matches!(
            events.get(1).map(|event| &event.event_type),
            Some(AuditEventType::ToolCall { call_id: Some(call_id), input, output, success: false, .. })
            if call_id == "call-a" && input.get("label") == Some(&serde_json::json!("a")) && output == "error-a"
        ));
        Ok(())
    }

    /// 验证中断事件在有无 start 回调时都使用已准入的输入。
    #[tokio::test]
    async fn audit_callback_interruption_uses_admitted_input_with_or_without_start() -> Result<()> {
        let logger = Arc::new(RecordingAuditLogger::default());
        let callback = AuditCallback::new(logger.clone(), "agent", Some("session".into()));
        let error = ReactError::Other("interrupted".to_string());

        callback
            .on_tool_interrupted_with_id(
                "agent",
                "before-start",
                "shell",
                &serde_json::json!({"label": "admitted", "password": "tiny-secret"}),
                &error,
            )
            .await;
        callback
            .on_tool_start_with_id(
                "agent",
                "after-start",
                "shell",
                &serde_json::json!({"label": "started"}),
            )
            .await;
        callback
            .on_tool_interrupted_with_id(
                "agent",
                "after-start",
                "shell",
                &serde_json::json!({"label": "must-not-replace-start"}),
                &error,
            )
            .await;

        let events = logger.query(AuditFilter::default()).await?;
        assert_eq!(events.len(), 2);
        assert!(matches!(
            events.first().map(|event| &event.event_type),
            Some(AuditEventType::ToolCall { call_id: Some(call_id), input, success: false, duration_ms: 0, .. })
                if call_id == "before-start"
                    && input.get("label") == Some(&serde_json::json!("admitted"))
                    && input.get("password") == Some(&serde_json::json!("[REDACTED]"))
        ));
        assert!(matches!(
            events.get(1).map(|event| &event.event_type),
            Some(AuditEventType::ToolCall { call_id: Some(call_id), input, success: false, .. })
                if call_id == "after-start"
                    && input.get("label") == Some(&serde_json::json!("started"))
        ));
        assert!(callback.take_tool_call("before-start").is_none());
        assert!(callback.take_tool_call("after-start").is_none());
        Ok(())
    }

    /// 验证内存与文件审计在零限制下仍保持类型化契约。
    #[tokio::test]
    async fn audit_memory_and_file_retention_preserve_typed_contract_at_zero_limits() -> Result<()>
    {
        let policy = ContentRetentionPolicy {
            max_string_chars: 0,
            max_array_items: 0,
        };
        let memory = memory::InMemoryAuditLogger::new().with_retention_policy(policy);
        let temp = std::env::temp_dir().join(format!("echo-audit-typed-{}", uuid::Uuid::new_v4()));
        let path = temp.join("audit.jsonl");
        let file = file::FileAuditLogger::new(&path)?.with_retention_policy(policy);
        let kinds = vec![
            AuditEventType::UserInput {
                content: "Bearer abcdefghijklmnopqrstuvwxyz".into(),
            },
            AuditEventType::FinalAnswer {
                content: "中文字符".repeat(30),
            },
            AuditEventType::LlmCall {
                model: "model-long-name".into(),
                prompt_tokens: Some(123),
                completion_tokens: Some(456),
            },
            AuditEventType::ToolCall {
                call_id: Some("zero-limit-call".into()),
                tool: "shell".into(),
                input: serde_json::json!({"password":"short", "values": [1, 2], "count": 42}),
                output: "password=abcdefgh".into(),
                success: false,
                duration_ms: 987,
            },
            AuditEventType::GuardBlock {
                guard: "guard".into(),
                direction: echo_core::guard::GuardDirection::Input,
                reason: "token=abcdefgh".into(),
            },
            AuditEventType::PermissionDenied {
                tool: "shell".into(),
                required: vec![
                    echo_core::tools::permission::ToolPermission::Read,
                    echo_core::tools::permission::ToolPermission::Write,
                ],
                reason: "secret=abcdefgh".into(),
            },
            AuditEventType::ApprovalRequested {
                tool: "shell".into(),
                args_hash: "hash-long-name".into(),
                risk_level: "high".into(),
            },
            AuditEventType::ApprovalCompleted {
                tool: "shell".into(),
                decision: "denied".into(),
                scope: "session".into(),
                reason: Some("password=abcdefgh".into()),
                duration_ms: 654,
            },
        ];
        for kind in kinds {
            let mut event = AuditEvent::now(
                Some("session-long-name".into()),
                "agent-long-name".into(),
                kind,
            );
            event.trace_id = Some("trace-long-name".into());
            memory.log(event.clone()).await?;
            file.log(event).await?;
        }
        let filter = AuditFilter {
            session_id: Some("session-long-name".into()),
            agent_name: Some("agent-long-name".into()),
            ..Default::default()
        };
        let retained = memory.query(filter.clone()).await?;
        assert_eq!(retained.len(), 8);
        assert_eq!(
            serde_json::to_value(&retained)?,
            serde_json::to_value(file.query(filter).await?)?
        );
        for event in &retained {
            assert_eq!(event.trace_id.as_deref(), Some("trace-long-name"));
            let encoded = serde_json::to_string(event)?;
            let _: AuditEvent = serde_json::from_str(&encoded)?;
            for secret in ["abcdefghijklmnopqrstuvwxyz", "abcdefgh", "short"] {
                assert!(!encoded.contains(secret));
            }
        }
        assert!(matches!(
            retained.get(2).map(|e| &e.event_type),
            Some(AuditEventType::LlmCall {
                prompt_tokens: Some(123),
                completion_tokens: Some(456),
                ..
            })
        ));
        assert!(matches!(
            retained.get(3).map(|e| &e.event_type),
            Some(AuditEventType::ToolCall {
                success: false,
                duration_ms: 987,
                ..
            })
        ));
        assert!(
            matches!(retained.get(5).map(|e| &e.event_type), Some(AuditEventType::PermissionDenied { required, .. }) if required.len() == 2)
        );
        drop(file);
        let reopened = file::FileAuditLogger::new(path)?.with_retention_policy(policy);
        assert_eq!(reopened.query(AuditFilter::default()).await?.len(), 8);
        drop(reopened);
        std::fs::remove_dir_all(temp)?;
        Ok(())
    }

    /// 验证内存与文件审计都会完整脱敏私钥正文。
    #[tokio::test]
    async fn memory_and_file_audit_redact_complete_private_key_body() -> Result<()> {
        let directory =
            std::env::temp_dir().join(format!("echo-audit-private-key-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&directory)?;
        let path = directory.join("audit.jsonl");
        let memory = memory::InMemoryAuditLogger::new();
        let file = file::FileAuditLogger::new(&path)?;
        let private_key_body = "MIIEvQIBADANBgkqhkiG9w0B";
        let event = AuditEvent::now(
            Some("session-pem".to_string()),
            "agent".to_string(),
            AuditEventType::UserInput {
                content: format!(
                    "-----BEGIN PRIVATE KEY-----\n{private_key_body}\n-----END PRIVATE KEY-----"
                ),
            },
        );
        memory.log(event.clone()).await?;
        file.log(event).await?;
        let filter = AuditFilter {
            session_id: Some("session-pem".to_string()),
            ..Default::default()
        };
        let in_memory = memory.query(filter.clone()).await?;
        let on_disk = file.query(filter).await?;
        assert_eq!(
            serde_json::to_value(&in_memory)?,
            serde_json::to_value(&on_disk)?
        );
        let serialized = std::fs::read_to_string(&path)?;
        assert!(!serialized.contains(private_key_body));
        assert!(!serialized.contains("BEGIN PRIVATE KEY"));
        assert!(serialized.contains("[REDACTED]"));
        drop(file);
        std::fs::remove_dir_all(directory)?;
        Ok(())
    }

    /// 验证诊断错误在分发给自定义观察者前已被保留策略处理。
    #[test]
    fn audit_diagnostic_errors_are_retained_before_custom_dispatch() -> Result<()> {
        let observer = Arc::new(RecordingObserver::default());
        let mut failure = DiagnosticDeliveryFailure::new(
            DiagnosticRecordKind::Audit,
            DiagnosticDeliveryOperation::Record,
            Some("session".into()),
            "Bearer abcdefghijklmnopqrstuvwxyz",
        );
        assert!(!format!("{failure:?}").contains("abcdefghijklmnopqrstuvwxyz"));
        failure.error = format!("password=abcdefgh {}", "中".repeat(20_000));
        report_diagnostic_delivery_failure(Some(observer.clone()), failure);
        let failures = observer.wait_for_count(1)?;
        let failure = failures
            .first()
            .ok_or_else(|| ReactError::Other("missing failure".into()))?;
        assert!(!failure.error.contains("abcdefgh"));
        assert!(failure.error.contains("[TRUNCATED]"));
        assert_eq!(failure.record_id.as_deref(), Some("session"));
        assert_eq!(failure.operation, DiagnosticDeliveryOperation::Record);
        Ok(())
    }

    /// 每次写入都返回注入错误的审计日志器。
    struct FailingAuditLogger;

    impl AuditLogger for FailingAuditLogger {
        /// 始终返回注入的持久化失败。
        fn log<'a>(&'a self, _event: AuditEvent) -> BoxFuture<'a, Result<()>> {
            Box::pin(async {
                Err(ReactError::Other(
                    "injected audit persistence failure".to_string(),
                ))
            })
        }

        /// 始终返回空结果。
        fn query<'a>(&'a self, _filter: AuditFilter) -> BoxFuture<'a, Result<Vec<AuditEvent>>> {
            Box::pin(async { Ok(Vec::new()) })
        }
    }

    /// 记录诊断投递失败并支持等待的观察者。
    #[derive(Default)]
    struct RecordingObserver {
        /// 已收到的失败记录。
        failures: Mutex<Vec<DiagnosticDeliveryFailure>>,
        /// 失败列表变化时通知等待者。
        changed: std::sync::Condvar,
    }

    impl DiagnosticDeliveryObserver for RecordingObserver {
        /// 记录失败并唤醒等待者。
        fn on_failure(&self, failure: DiagnosticDeliveryFailure) {
            self.failures
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .push(failure);
            self.changed.notify_all();
        }
    }

    impl RecordingObserver {
        /// 等待直至至少收到 `count` 条失败记录，超时返回错误。
        fn wait_for_count(&self, count: usize) -> Result<Vec<DiagnosticDeliveryFailure>> {
            let failures = self
                .failures
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            let (failures, wait) = self
                .changed
                .wait_timeout_while(failures, std::time::Duration::from_secs(5), |failures| {
                    failures.len() < count
                })
                .map_err(|error| ReactError::Other(format!("observer wait failed: {error}")))?;
            if wait.timed_out() && failures.len() < count {
                return Err(ReactError::Other(format!(
                    "timed out waiting for {count} diagnostic failures"
                )));
            }
            Ok(failures.clone())
        }
    }

    /// 在回调中重入上报失败的观察者。
    struct ReentrantObserver {
        /// 回调完成后发送通知。
        completed: std::sync::mpsc::Sender<()>,
    }

    impl DiagnosticDeliveryObserver for ReentrantObserver {
        /// 重入上报失败后发送完成通知。
        fn on_failure(&self, failure: DiagnosticDeliveryFailure) {
            report_diagnostic_delivery_failure(None, failure);
            let _ = self.completed.send(());
        }
    }

    /// 在回调中触发 unwind 的观察者。
    struct UnwindingObserver {
        /// 回调被调用时发送通知。
        attempted: std::sync::mpsc::Sender<()>,
    }

    impl DiagnosticDeliveryObserver for UnwindingObserver {
        /// 发送通知后主动触发 unwind。
        fn on_failure(&self, _failure: DiagnosticDeliveryFailure) {
            let _ = self.attempted.send(());
            /*
            有意的故障注入：诊断分发器会捕获此 unwind，递增丢弃计数，并保持存活。
            */
            std::panic::resume_unwind(Box::new("injected observer unwind".to_string()));
        }
    }

    /// 构造一条示例诊断投递失败记录。
    fn sample_failure() -> DiagnosticDeliveryFailure {
        DiagnosticDeliveryFailure::new(
            DiagnosticRecordKind::Audit,
            DiagnosticDeliveryOperation::Record,
            Some("session-46".to_string()),
            "injected audit persistence failure",
        )
    }

    /// 验证后端写入失败会上报诊断，但不改变回调结果。
    #[tokio::test]
    async fn audit_callback_reports_backend_failure_without_changing_callback_result() -> Result<()>
    {
        let observer = Arc::new(RecordingObserver::default());
        let callback = AuditCallback::new(
            Arc::new(FailingAuditLogger),
            "agent",
            Some("session-46".to_string()),
        )
        .with_diagnostic_delivery_observer(observer.clone());

        callback
            .on_final_answer("agent", "producer answer remains successful")
            .await;

        let failures = observer.wait_for_count(1)?;
        let failure = failures
            .first()
            .ok_or_else(|| ReactError::Other("missing diagnostic delivery failure".to_string()))?;
        assert_eq!(failure.record_kind, DiagnosticRecordKind::Audit);
        assert_eq!(failure.operation, DiagnosticDeliveryOperation::Record);
        assert_eq!(failure.record_id.as_deref(), Some("session-46"));
        assert!(failure.error.contains("injected audit persistence failure"));
        Ok(())
    }

    /// 验证重入观察者被限制并计入丢弃计数。
    #[test]
    fn reentrant_observer_is_bounded_and_counted() -> Result<()> {
        let dropped_before = diagnostic_delivery_dropped_count();
        let (completed, completion) = std::sync::mpsc::channel();
        report_diagnostic_delivery_failure(
            Some(Arc::new(ReentrantObserver { completed })),
            sample_failure(),
        );

        completion
            .recv_timeout(std::time::Duration::from_secs(5))
            .map_err(|error| {
                ReactError::Other(format!("reentrant observer did not run: {error}"))
            })?;
        assert!(diagnostic_delivery_dropped_count() > dropped_before);
        Ok(())
    }

    /// 验证观察者 unwind 不会阻断后续诊断投递。
    #[test]
    fn unwinding_observer_does_not_stop_later_diagnostic_delivery() -> Result<()> {
        let dropped_before = diagnostic_delivery_dropped_count();
        let (attempted, attempt) = std::sync::mpsc::channel();
        report_diagnostic_delivery_failure(
            Some(Arc::new(UnwindingObserver { attempted })),
            sample_failure(),
        );
        attempt
            .recv_timeout(std::time::Duration::from_secs(5))
            .map_err(|error| {
                ReactError::Other(format!("unwinding observer did not run: {error}"))
            })?;

        let observer = Arc::new(RecordingObserver::default());
        report_diagnostic_delivery_failure(Some(observer.clone()), sample_failure());
        let _ = observer.wait_for_count(1)?;
        assert!(diagnostic_delivery_dropped_count() > dropped_before);
        Ok(())
    }
}
