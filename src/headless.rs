/*!
Headless（无头）模式——运行单条 prompt、打印输出、然后退出。

专为 CI/CD 流水线、脚本化调用与非交互式自动化场景设计。
Agent 只运行一条 prompt，收集输出，并返回一个结构化结果，
调用方可以打印它并据此决定退出码。

# 示例

```rust,no_run
use echo_agent::headless::{HeadlessConfig, run_headless};

# #[tokio::main]
# async fn main() {
let config = HeadlessConfig {
    prompt: "List all Rust files in the project".into(),
    exit_on_error: true,
    output_format: "text".into(),
    max_iterations: Some(10),
    cancel_token: None,
};

let result = run_headless(config, |builder| builder).await;
println!("{}", result.output);
std::process::exit(result.exit_code());
# }
```
*/

use crate::agent::Agent;
use crate::agent::react::builder::ReactAgentBuilder;
use crate::runtime::{
    AgentTurnDriver, EventSink, SinkControl, TurnDeliveryOutcome, TurnMode, TurnOutcome,
    TurnReceipt, TurnRequest,
};
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

struct HeadlessEventSink;

#[async_trait::async_trait]
impl EventSink for HeadlessEventSink {
    async fn on_event(
        &self,
        _envelope: crate::agent::EventEnvelope,
    ) -> crate::error::Result<SinkControl> {
        Ok(SinkControl::Continue)
    }
}

/// Headless（非交互式）Agent 执行的配置。
pub struct HeadlessConfig {
    /// 要执行的 prompt。
    pub prompt: String,

    /// 若 Agent 报告失败，则以错误退出。
    pub exit_on_error: bool,

    /// 输出格式：`"text"`（默认）或 `"json"`。
    pub output_format: String,

    /// 强制停止前允许的最大迭代次数（安全限制）。
    pub max_iterations: Option<usize>,

    /**
    调用方持有的可选取消令牌。

    Headless 会为本次运行派生一个子令牌，因此取消本次运行不会
    取消调用方的父级作用域或其兄弟任务。
    */
    pub cancel_token: Option<CancellationToken>,
}

impl Default for HeadlessConfig {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            exit_on_error: true,
            output_format: "text".into(),
            max_iterations: None,
            cancel_token: None,
        }
    }
}

/// 一次 headless 执行的结果。
#[derive(Debug, Clone)]
pub struct HeadlessResult {
    /// Agent 的最终输出文本。
    pub output: String,

    /// 本次执行是否成功。
    pub success: bool,

    /// 使用的模型名称。
    pub model: String,

    /// 请求的输出格式。
    pub format: String,

    /// 失败的运行是否应产生非零的进程退出码。
    pub exit_on_error: bool,
}

/// 一次 headless 运行的共享状态：结果槽位、取消令牌与关闭所有权。
struct HeadlessRunState {
    /// 结果一旦发布即固定在此处。
    result: std::sync::Mutex<Option<HeadlessResult>>,
    /// 用于通知等待者“结果已就绪”。
    result_ready: tokio::sync::Notify,
    /// 本次运行的取消令牌。
    cancel: CancellationToken,
    /// 保证 Agent 关闭只被真正执行一次的门闩。
    close_gate: tokio::sync::Mutex<()>,
    /// 当前持有的、待关闭的 Agent（关闭成功后被清空）。
    close_owner: std::sync::Mutex<Option<Arc<dyn Agent>>>,
}

impl HeadlessRunState {
    /// 创建一个新的共享运行状态。
    fn new(cancel: CancellationToken, agent: Option<Arc<dyn Agent>>) -> Self {
        Self {
            result: std::sync::Mutex::new(None),
            result_ready: tokio::sync::Notify::new(),
            cancel,
            close_gate: tokio::sync::Mutex::new(()),
            close_owner: std::sync::Mutex::new(agent),
        }
    }

    /// 发布结果（只有第一次调用生效），并唤醒所有等待者。
    fn publish(&self, result: HeadlessResult) {
        let mut slot = self
            .result
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if slot.is_none() {
            *slot = Some(result);
            drop(slot);
            self.result_ready.notify_waiters();
        }
    }

    /// 若结果已发布，则返回其克隆。
    fn result(&self) -> Option<HeadlessResult> {
        self.result
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    /// 关闭持有的 Agent（若尚未关闭）；重复调用是安全的。
    async fn close_once(&self) -> crate::error::Result<()> {
        let _gate = self.close_gate.lock().await;
        let agent = self
            .close_owner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();
        let Some(agent) = agent else {
            return Ok(());
        };
        agent.close().await?;
        let mut owner = self
            .close_owner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if owner
            .as_ref()
            .is_some_and(|current| Arc::ptr_eq(current, &agent))
        {
            owner.take();
        }
        Ok(())
    }
}

/**
一次 Headless 执行所保留的持有者与结果回执。

`wait` 可以被取消，而不会取消其所拥有的清理任务。使用
[`Self::cancel`] 请求取消 Turn；若首次资源关闭尝试被报告为失败，
则使用 [`Self::retry_close`] 重试。
*/
#[derive(Clone)]
pub struct HeadlessRunHandle {
    /// 共享的运行状态。
    state: Arc<HeadlessRunState>,
}

impl HeadlessRunHandle {
    /// 构造一个已经携带结果的、立即完成的句柄。
    fn completed(result: HeadlessResult) -> Self {
        let state = Arc::new(HeadlessRunState::new(CancellationToken::new(), None));
        state.publish(result);
        Self { state }
    }

    /// 请求取消其拥有的 Turn。
    pub fn cancel(&self) {
        self.state.cancel.cancel();
    }

    /// 等待执行完成、且首次 Agent 关闭尝试结算完毕。
    pub async fn wait(&self) -> HeadlessResult {
        loop {
            let notified = self.state.result_ready.notified();
            if let Some(result) = self.state.result() {
                return result;
            }
            notified.await;
        }
    }

    /**
    在一次关闭失败被报告之后，重试关闭同一个 Agent 持有者。

    在 [`Self::wait`] 观察到运行回执之前，本方法都会返回错误。
    关闭成功之后，重复调用是幂等的。
    */
    pub async fn retry_close(&self) -> crate::error::Result<()> {
        if self.state.result().is_none() {
            return Err(crate::error::ReactError::Other(
                "Agent close cannot be retried before Headless execution settles".to_string(),
            ));
        }
        self.state.close_once().await
    }
}

/**
驱动任务对共享状态的持有者；若任务在发布结果前被丢弃，
会通过 `fallback` 兜底发布一个失败结果。
*/
struct HeadlessTaskOwner {
    /// 共享的运行状态。
    state: Arc<HeadlessRunState>,
    /// 尚未真正结算时使用的兜底失败结果。
    fallback: Option<HeadlessResult>,
}

impl HeadlessTaskOwner {
    /// 发布最终结果，并清除兜底结果（不再需要）。
    fn publish(&mut self, result: HeadlessResult) {
        self.state.publish(result);
        self.fallback.take();
    }
}

impl Drop for HeadlessTaskOwner {
    fn drop(&mut self) {
        if let Some(result) = self.fallback.take() {
            self.state.publish(result);
        }
    }
}

/**
在 `wait` 提前返回（被取消）时，兜底触发取消信号的守卫；
正常完成时应调用 [`Self::disarm`] 解除该行为。
*/
struct HeadlessWaitCancellationGuard {
    /// 需要在守卫触发时取消的令牌。
    cancel: CancellationToken,
    /// 该守卫当前是否仍处于“激活”状态。
    active: bool,
}

impl HeadlessWaitCancellationGuard {
    /// 解除该守卫，使其在丢弃时不再触发取消。
    fn disarm(&mut self) {
        self.active = false;
    }
}

impl Drop for HeadlessWaitCancellationGuard {
    fn drop(&mut self) {
        if self.active {
            self.cancel.cancel();
        }
    }
}

impl HeadlessResult {
    /// 计算进程退出码：成功为 0，失败为 1。
    pub fn exit_code(&self) -> i32 {
        if self.success || !self.exit_on_error {
            0
        } else {
            1
        }
    }

    /// 按请求的输出格式，将结果格式化为待打印到 stdout 的文本。
    pub fn format_output(&self) -> String {
        match self.format.as_str() {
            "json" => {
                let json = serde_json::json!({
                    "success": self.success,
                    "model": self.model,
                    "output": self.output,
                });
                serde_json::to_string_pretty(&json).unwrap_or_else(|_| self.output.clone())
            }
            _ => self.output.clone(),
        }
    }
}

/**
以 headless 模式运行 Agent。

`configure` 闭包会收到一个 [`ReactAgentBuilder`]，调用方可以借此在
Agent 被构建并运行之前设置模型、系统 prompt、工具等。

# 参数

* `config` — headless 执行参数（prompt、格式、限制）
* `configure` — 用于定制 agent builder 的闭包

# 返回值

一个携带 Agent 输出、成功标志与元数据的 [`HeadlessResult`]。
*/
pub async fn run_headless<F>(config: HeadlessConfig, configure: F) -> HeadlessResult
where
    F: FnOnce(ReactAgentBuilder) -> ReactAgentBuilder,
{
    let handle = start_headless(config, configure);
    wait_headless_handle(handle).await
}

/// 等待一个 headless 句柄的结果；若本次等待自身被取消，则取消该运行。
async fn wait_headless_handle(handle: HeadlessRunHandle) -> HeadlessResult {
    let mut cancellation_guard = HeadlessWaitCancellationGuard {
        cancel: handle.state.cancel.clone(),
        active: true,
    };
    let result = handle.wait().await;
    cancellation_guard.disarm();
    result
}

/**
启动 Headless 执行，并同步返回它所保留的持有者。

这是一个对取消安全的入口：调用方可以先停止等待，之后再观察同一个
结果，或者在 Agent 关闭失败时重试。
*/
pub fn start_headless<F>(config: HeadlessConfig, configure: F) -> HeadlessRunHandle
where
    F: FnOnce(ReactAgentBuilder) -> ReactAgentBuilder,
{
    let exit_on_error = config.exit_on_error;
    if config.prompt.is_empty() {
        return HeadlessRunHandle::completed(HeadlessResult {
            output: "Error: empty prompt".into(),
            success: false,
            model: String::new(),
            format: config.output_format,
            exit_on_error,
        });
    }

    // 构建 agent
    let builder = ReactAgentBuilder::new();
    let builder = configure(builder);

    // 若设置了 max_iterations，则应用该配置
    let builder = if let Some(max) = config.max_iterations {
        builder.max_iterations(max)
    } else {
        builder
    };

    let agent = match builder.build() {
        Ok(a) => a,
        Err(e) => {
            return HeadlessRunHandle::completed(HeadlessResult {
                output: format!("Error building agent: {}", e),
                success: false,
                model: String::new(),
                format: config.output_format,
                exit_on_error,
            });
        }
    };

    start_headless_agent(config, Arc::new(agent))
}

/// 使用给定配置与已构建好的 Agent 启动 headless 执行任务。
fn start_headless_agent(config: HeadlessConfig, agent: Arc<dyn Agent>) -> HeadlessRunHandle {
    /* 调用方拥有所配置的令牌。Headless 的取消必须局限在本次运行之内，
    绝不能取消调用方作用域中的兄弟任务。 */
    let cancel = config
        .cancel_token
        .as_ref()
        .map(CancellationToken::child_token)
        .unwrap_or_default();
    let state = Arc::new(HeadlessRunState::new(cancel.clone(), Some(agent.clone())));
    let handle = HeadlessRunHandle {
        state: Arc::clone(&state),
    };
    let runtime = match tokio::runtime::Handle::try_current() {
        Ok(runtime) => runtime,
        Err(error) => {
            state.publish(HeadlessResult {
                output: format!("Error starting headless run: {error}"),
                success: false,
                model: agent.model_name().to_string(),
                format: config.output_format,
                exit_on_error: config.exit_on_error,
            });
            return handle;
        }
    };
    /* 在 spawn 之前构造持有者，这样即使任务从未被轮询就被丢弃，
    也仍然会发布一个失败回执，并保留该 Agent 以便重试关闭。 */
    let task_owner = HeadlessTaskOwner {
        state: Arc::clone(&state),
        fallback: Some(HeadlessResult {
            output: "Error: headless execution owner ended before settlement".to_string(),
            success: false,
            model: agent.model_name().to_string(),
            format: config.output_format.clone(),
            exit_on_error: config.exit_on_error,
        }),
    };
    let task = runtime.spawn(async move {
        let mut task_owner = task_owner;
        let mut result = execute_headless_agent(config, agent.as_ref(), cancel).await;
        if let Err(error) = state.close_once().await {
            if !result.success {
                result.output.push_str("; ");
            } else {
                result.output.clear();
            }
            result
                .output
                .push_str(&format!("Error closing agent: {error}"));
            result.success = false;
        }
        task_owner.publish(result);
    });
    /* 该任务会持有共享状态和 Agent，直到第一次 close 尝试为止，
    即使等待者被取消也会发布结果回执。
    仅仅丢弃 JoinHandle 并不会放弃这个逻辑上的持有者。 */
    std::mem::drop(task);
    handle
}

async fn execute_headless_agent(
    config: HeadlessConfig,
    agent: &dyn Agent,
    cancel: CancellationToken,
) -> HeadlessResult {
    let model = agent.model_name().to_string();
    let exit_on_error = config.exit_on_error;
    let identity_value = format!("headless-{}", uuid::Uuid::new_v4());
    let execution = match crate::agent::EventIdentity::new(&identity_value, &identity_value) {
        Ok(identity) => {
            let request = TurnRequest::new(identity, config.prompt)
                .mode(TurnMode::Execute)
                .cancel(cancel);
            let receipt = AgentTurnDriver
                .drive(agent, request, &HeadlessEventSink)
                .await;
            headless_result_from_receipt(receipt)
        }
        Err(error) => (format!("Error: {error}"), false),
    };
    let (output, success) = execution;
    HeadlessResult {
        output,
        success,
        model,
        format: config.output_format,
        exit_on_error,
    }
}

fn headless_result_from_receipt(receipt: TurnReceipt) -> (String, bool) {
    let delivery_failure = match &receipt.delivery {
        TurnDeliveryOutcome::Failed(failure) => Some(failure.message.clone()),
        _ => None,
    };
    let delivery_ok = matches!(receipt.delivery, TurnDeliveryOutcome::Delivered);
    let (output, success) = match (receipt.outcome, receipt.final_answer, delivery_failure) {
        (TurnOutcome::Completed, Some(output), None) if delivery_ok => (output, true),
        (TurnOutcome::Completed, Some(_), Some(error)) => {
            (format!("Error: turn delivery failed: {error}"), false)
        }
        (TurnOutcome::Completed, Some(_), None) => (
            "Error: completed turn was not fully delivered".to_string(),
            false,
        ),
        (TurnOutcome::Completed, None, _) => (
            "Error: completed turn did not include a final answer".to_string(),
            false,
        ),
        (TurnOutcome::Cancelled, _, _) => ("Cancelled".to_string(), false),
        (TurnOutcome::Failed(failure), _, _) => (
            format!("Error ({}): {}", failure.code, failure.message),
            false,
        ),
    };
    (output, success)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::AgentEvent;
    use crate::error::{AgentFailure, ReactError};
    use crate::testing::MockLlmClient;
    use echo_core::agent::TurnId;
    use futures::StreamExt;
    use futures::future::BoxFuture;
    use futures::stream::BoxStream;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::time::Duration;

    struct ClosingAgent {
        closes: Arc<AtomicUsize>,
        fail_first_close: Arc<AtomicBool>,
    }

    impl Agent for ClosingAgent {
        fn name(&self) -> &str {
            "closing-agent"
        }

        fn model_name(&self) -> &str {
            "closing-model"
        }

        fn system_prompt(&self) -> &str {
            "test"
        }

        fn execute<'a>(&'a self, _task: &'a str) -> BoxFuture<'a, crate::error::Result<String>> {
            Box::pin(async { Ok("done".to_string()) })
        }

        fn execute_stream<'a>(
            &'a self,
            _task: &'a str,
        ) -> BoxFuture<'a, crate::error::Result<BoxStream<'a, crate::error::Result<AgentEvent>>>>
        {
            Box::pin(async {
                Ok(
                    futures::stream::iter(vec![Ok(AgentEvent::FinalAnswer("done".to_string()))])
                        .boxed(),
                )
            })
        }

        fn close(&self) -> BoxFuture<'_, crate::error::Result<()>> {
            Box::pin(async move {
                self.closes.fetch_add(1, Ordering::AcqRel);
                if self.fail_first_close.swap(false, Ordering::AcqRel) {
                    Err(ReactError::Other("close failed".to_string()))
                } else {
                    Ok(())
                }
            })
        }
    }

    struct CancellableHeadlessAgent {
        started: Arc<tokio::sync::Notify>,
        closes: Arc<AtomicUsize>,
        closed: Arc<tokio::sync::Notify>,
    }

    impl Agent for CancellableHeadlessAgent {
        fn name(&self) -> &str {
            "cancellable-headless-agent"
        }

        fn model_name(&self) -> &str {
            "cancellable-model"
        }

        fn system_prompt(&self) -> &str {
            "test"
        }

        fn execute<'a>(&'a self, _task: &'a str) -> BoxFuture<'a, crate::error::Result<String>> {
            Box::pin(async { Ok("done".to_string()) })
        }

        fn execute_stream<'a>(
            &'a self,
            _task: &'a str,
        ) -> BoxFuture<'a, crate::error::Result<BoxStream<'a, crate::error::Result<AgentEvent>>>>
        {
            let started = Arc::clone(&self.started);
            Box::pin(async move {
                Ok(futures::stream::once(async move {
                    started.notify_one();
                    futures::future::pending::<crate::error::Result<AgentEvent>>().await
                })
                .boxed())
            })
        }

        fn close(&self) -> BoxFuture<'_, crate::error::Result<()>> {
            Box::pin(async move {
                self.closes.fetch_add(1, Ordering::AcqRel);
                self.closed.notify_waiters();
                Ok(())
            })
        }
    }

    #[test]
    fn test_headless_config_default() {
        let config = HeadlessConfig::default();
        assert!(config.prompt.is_empty());
        assert!(config.exit_on_error);
        assert_eq!(config.output_format, "text");
        assert!(config.max_iterations.is_none());
        assert!(config.cancel_token.is_none());
    }

    #[test]
    fn test_headless_result_exit_code() {
        let ok = HeadlessResult {
            output: "done".into(),
            success: true,
            model: "test".into(),
            format: "text".into(),
            exit_on_error: true,
        };
        assert_eq!(ok.exit_code(), 0);

        let fail = HeadlessResult {
            output: "error".into(),
            success: false,
            model: "test".into(),
            format: "text".into(),
            exit_on_error: true,
        };
        assert_eq!(fail.exit_code(), 1);

        let tolerated_failure = HeadlessResult {
            output: "error".into(),
            success: false,
            model: "test".into(),
            format: "text".into(),
            exit_on_error: false,
        };
        assert_eq!(tolerated_failure.exit_code(), 0);
    }

    #[test]
    fn test_headless_result_format_json() {
        let result = HeadlessResult {
            output: "hello world".into(),
            success: true,
            model: "test-model".into(),
            format: "json".into(),
            exit_on_error: true,
        };
        let formatted = result.format_output();
        assert!(formatted.contains("\"success\": true"));
        assert!(formatted.contains("\"model\": \"test-model\""));
        assert!(formatted.contains("hello world"));
    }

    #[test]
    fn completed_execution_with_failed_delivery_is_not_headless_success() -> crate::error::Result<()>
    {
        let receipt = TurnReceipt {
            turn_id: TurnId::new("headless-delivery-failed")?,
            outcome: TurnOutcome::Completed,
            delivery: TurnDeliveryOutcome::Failed(AgentFailure::from(&ReactError::Other(
                "stdout unavailable".to_string(),
            ))),
            final_answer: Some("done".to_string()),
            final_message_id: None,
            prompt_tokens: 0,
            completion_tokens: 0,
            llm_calls: 0,
            compaction_count: 0,
            last_event_sequence: 1,
            elapsed: Duration::ZERO,
        };
        let (output, success) = headless_result_from_receipt(receipt);
        assert!(!success);
        assert!(output.contains("turn delivery failed"));
        Ok(())
    }

    #[test]
    fn test_headless_result_format_text() {
        let result = HeadlessResult {
            output: "hello world".into(),
            success: true,
            model: "test-model".into(),
            format: "text".into(),
            exit_on_error: true,
        };
        assert_eq!(result.format_output(), "hello world");
    }

    #[tokio::test]
    async fn run_headless_uses_shared_turn_driver_terminal_contract() {
        let llm = Arc::new(MockLlmClient::new().with_response("driver result"));
        let result = run_headless(
            HeadlessConfig {
                prompt: "run through the shared driver".to_string(),
                ..HeadlessConfig::default()
            },
            |builder| builder.llm_client(llm).system_prompt("test"),
        )
        .await;

        assert!(result.success);
        assert_eq!(result.output, "driver result");
        assert_eq!(result.exit_code(), 0);
    }

    #[tokio::test]
    async fn headless_awaits_agent_close_and_surfaces_cleanup_failure() -> crate::error::Result<()>
    {
        let closes = Arc::new(AtomicUsize::new(0));
        let fail_first_close = Arc::new(AtomicBool::new(true));
        let handle = start_headless_agent(
            HeadlessConfig {
                prompt: "close after settlement".to_string(),
                ..HeadlessConfig::default()
            },
            Arc::new(ClosingAgent {
                closes: Arc::clone(&closes),
                fail_first_close,
            }),
        );
        let result = handle.wait().await;

        assert_eq!(closes.load(Ordering::Acquire), 1);
        assert!(!result.success);
        assert!(result.output.contains("close failed"));
        handle.retry_close().await?;
        assert_eq!(closes.load(Ordering::Acquire), 2);
        Ok(())
    }

    #[tokio::test]
    async fn cancelled_headless_waiter_keeps_owned_cleanup_running() -> crate::error::Result<()> {
        let started = Arc::new(tokio::sync::Notify::new());
        let closes = Arc::new(AtomicUsize::new(0));
        let closed = Arc::new(tokio::sync::Notify::new());
        let parent_cancel = CancellationToken::new();
        let sibling_cancel = parent_cancel.child_token();
        let handle = start_headless_agent(
            HeadlessConfig {
                prompt: "wait until cancelled".to_string(),
                cancel_token: Some(parent_cancel.clone()),
                ..HeadlessConfig::default()
            },
            Arc::new(CancellableHeadlessAgent {
                started: Arc::clone(&started),
                closes: Arc::clone(&closes),
                closed: Arc::clone(&closed),
            }),
        );
        let waiter_handle = handle.clone();
        let waiter = tokio::spawn(async move { wait_headless_handle(waiter_handle).await });
        tokio::time::timeout(Duration::from_secs(2), started.notified())
            .await
            .map_err(|_| ReactError::Other("Headless turn did not start".to_string()))?;
        let retry_error = handle.retry_close().await.err().ok_or_else(|| {
            ReactError::Other("close retry was accepted before execution settled".to_string())
        })?;
        assert!(
            retry_error
                .to_string()
                .contains("before Headless execution settles")
        );
        assert_eq!(closes.load(Ordering::Acquire), 0);
        let closed_wait = closed.notified();
        tokio::pin!(closed_wait);
        closed_wait.as_mut().enable();
        waiter.abort();
        let _ = waiter.await;
        tokio::time::timeout(Duration::from_secs(2), closed_wait)
            .await
            .map_err(|_| ReactError::Other("cancelled waiter abandoned Agent close".to_string()))?;
        assert_eq!(closes.load(Ordering::Acquire), 1);
        assert!(!parent_cancel.is_cancelled());
        assert!(!sibling_cancel.is_cancelled());
        let result = handle.wait().await;
        assert!(!result.success);
        assert!(result.output.contains("Cancelled"));
        Ok(())
    }

    #[test]
    fn missing_runtime_returns_failure_without_dropping_close_owner() -> crate::error::Result<()> {
        let closes = Arc::new(AtomicUsize::new(0));
        let handle = start_headless_agent(
            HeadlessConfig {
                prompt: "requires a runtime".to_string(),
                ..HeadlessConfig::default()
            },
            Arc::new(ClosingAgent {
                closes: Arc::clone(&closes),
                fail_first_close: Arc::new(AtomicBool::new(false)),
            }),
        );
        let result = handle.state.result().ok_or_else(|| {
            ReactError::Other("missing-runtime failure was not published".to_string())
        })?;
        assert!(!result.success);
        assert!(result.output.contains("Error starting headless run"));
        assert_eq!(closes.load(Ordering::Acquire), 0);

        let runtime = tokio::runtime::Runtime::new()
            .map_err(|error| ReactError::Other(format!("test runtime failed: {error}")))?;
        runtime.block_on(handle.retry_close())?;
        assert_eq!(closes.load(Ordering::Acquire), 1);
        Ok(())
    }

    #[test]
    fn runtime_drop_before_first_poll_publishes_failure_and_retains_owner()
    -> crate::error::Result<()> {
        let closes = Arc::new(AtomicUsize::new(0));
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|error| ReactError::Other(format!("test runtime failed: {error}")))?;
        let enter_guard = runtime.enter();
        let handle = start_headless_agent(
            HeadlessConfig {
                prompt: "runtime drops before polling".to_string(),
                ..HeadlessConfig::default()
            },
            Arc::new(ClosingAgent {
                closes: Arc::clone(&closes),
                fail_first_close: Arc::new(AtomicBool::new(false)),
            }),
        );
        drop(enter_guard);
        drop(runtime);

        let result = handle.state.result().ok_or_else(|| {
            ReactError::Other("unpolled task did not publish a failure receipt".to_string())
        })?;
        assert!(!result.success);
        assert!(result.output.contains("owner ended before settlement"));
        assert_eq!(closes.load(Ordering::Acquire), 0);

        let retry_runtime = tokio::runtime::Runtime::new()
            .map_err(|error| ReactError::Other(format!("retry runtime failed: {error}")))?;
        retry_runtime.block_on(handle.retry_close())?;
        assert_eq!(closes.load(Ordering::Acquire), 1);
        Ok(())
    }
}
