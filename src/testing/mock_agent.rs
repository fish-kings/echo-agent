/*!
 * Mock Agent，实现了 [`Agent`] trait，用于在测试多 Agent 编排逻辑时替代真实的子 Agent。
 *
 * 在测试编排逻辑时，我们通常希望：
 * - 避免发起真实的 LLM 调用
 * - 控制每个子 Agent 的返回内容
 * - 验证某个子 Agent 被调用了多少次、每次收到的任务是什么
 *
 * # 示例
 *
 * ```rust
 * use echo_agent::testing::MockAgent;
 * use echo_agent::agent::Agent;
 *
 * # #[tokio::main]
 * # async fn main() -> echo_agent::error::Result<()> {
 * let agent = MockAgent::new("math_agent")
 *     .with_response("The result is 42")
 *     .with_response("The result is 100");
 *
 * let r1 = agent.execute("compute 6 * 7").await?;
 * let r2 = agent.execute("compute 10 * 10").await?;
 * assert_eq!(r1, "The result is 42");
 * assert_eq!(r2, "The result is 100");
 * assert_eq!(agent.call_count(), 2);
 * assert_eq!(agent.calls().first().map(String::as_str), Some("compute 6 * 7"));
 * # Ok(())
 * # }
 * ```
 */

use crate::agent::{Agent, AgentEvent, CancellationToken};
use crate::error::{AgentError, ReactError, Result};
use futures::future::BoxFuture;
use futures::stream::BoxStream;
use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

// ── MockAgent ─────────────────────────────────────────────────────────────────

/// 可感知取消的 Agent 事件脚本中的一个步骤。
pub enum MockAgentStep {
    /// 依次产出的一个 Agent 事件。
    Event(Box<AgentEvent>),
    /// 以该错误消息结束当前流。
    Error(String),
    /// 在产出下一步之前等待指定时长（可被取消打断）。
    Delay(std::time::Duration),
}

/// [`FailingMockAgent`] 发出的失败类别。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MockAgentFailure {
    /// 初始化失败。
    Initialization,
    /// 子 Agent 执行失败。
    Subagent,
    /// 超时失败。
    Timeout,
    /// 被取消。
    Cancelled,
    /// 权限被拒绝。
    PermissionDenied,
}

impl MockAgentFailure {
    /// 将失败类别与消息组合成对应的 `ReactError`。
    fn error(self, message: String) -> ReactError {
        let error = match self {
            Self::Initialization => AgentError::InitializationFailed(message),
            Self::Subagent => AgentError::SubagentError(message),
            Self::Timeout => AgentError::Timeout(message),
            Self::Cancelled => AgentError::Cancelled(message),
            Self::PermissionDenied => AgentError::PermissionDenied(message),
        };
        ReactError::Agent(Box::new(error))
    }
}

/** 可编排脚本的 Mock Agent。

按顺序返回预设的响应，脚本耗尽后以失败告终（fail closed）。
`execute()` 和 `chat()` 收到的消息都会被记录，可通过
[`calls()`](MockAgent::calls) 查看。
`reset()` 会清空调用历史，模拟真实 Agent 的对话重置语义。
*/
pub struct MockAgent {
    /// 该 Agent 的名称。
    name: String,
    /// 该 Agent 报告的模型名称。
    model_name: String,
    /// 该 Agent 的系统提示词。
    system_prompt: String,
    /// 待消费的预设文本响应队列。
    responses: Arc<Mutex<VecDeque<String>>>,
    /// 待消费的预设事件脚本队列（每次流式调用消费一个）。
    event_scripts: Arc<Mutex<VecDeque<Vec<MockAgentStep>>>>,
    /// 已收到的任务/消息字符串，按调用顺序记录。
    calls: Arc<Mutex<Vec<String>>>,
    /**
     * 通过 `execute_stream_message_with_cancel` 收到的多模态消息
     * （用于记录分发逻辑是否把附件转发给了子 Agent）。
     */
    messages: Arc<Mutex<Vec<echo_core::llm::types::Message>>>,
    /// 流式调用中收到的值域调用上下文（value-scoped invocation metadata）。
    invocation_contexts: Arc<Mutex<Vec<echo_core::agent::AgentInvocationContext>>>,
    /**
     * 按顺序记录的 `set_working_dir` 调用（Sprint 8 隔离性测试）。
     * 每一项是要求该 Agent 绑定的路径（`None` 表示清除）。
     */
    working_dirs: Arc<Mutex<Vec<Option<std::path::PathBuf>>>>,
    /// 返回结果前的人为延迟（用于后台分发相关的测试）。
    delay_ms: u64,
    /// 脚本/响应队列耗尽后使用的默认成功响应。
    default_success: Option<String>,
}

/* 所有可观察状态都位于 Arc<Mutex> 之后，因此克隆会共享调用/消息历史 ——
测试可以克隆一个 MockAgent、注册这个克隆体，再对原对象做断言。
*/
impl Clone for MockAgent {
    /**
     * 克隆该 Mock Agent；由于所有可观察状态都在 `Arc<Mutex>` 之后，
     * 克隆体与原对象共享调用/消息历史。
     */
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            model_name: self.model_name.clone(),
            system_prompt: self.system_prompt.clone(),
            responses: self.responses.clone(),
            event_scripts: self.event_scripts.clone(),
            calls: self.calls.clone(),
            messages: self.messages.clone(),
            invocation_contexts: self.invocation_contexts.clone(),
            working_dirs: self.working_dirs.clone(),
            delay_ms: self.delay_ms,
            default_success: self.default_success.clone(),
        }
    }
}

impl MockAgent {
    /// 创建一个具名的 Mock Agent。
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            model_name: "mock-model".to_string(),
            system_prompt: "You are a mock agent".to_string(),
            responses: Arc::new(Mutex::new(VecDeque::new())),
            event_scripts: Arc::new(Mutex::new(VecDeque::new())),
            calls: Arc::new(Mutex::new(Vec::new())),
            messages: Arc::new(Mutex::new(Vec::new())),
            invocation_contexts: Arc::new(Mutex::new(Vec::new())),
            working_dirs: Arc::new(Mutex::new(Vec::new())),
            delay_ms: 0,
            default_success: None,
        }
    }

    /// 为每次响应设置一个人为延迟（用于后台分发相关的测试）。
    pub fn with_delay_ms(mut self, delay_ms: u64) -> Self {
        self.delay_ms = delay_ms;
        self
    }

    /// 设置模型名称（用于需要校验 model_name 的测试）。
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model_name = model.into();
        self
    }

    /// 设置系统提示词。
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = prompt.into();
        self
    }

    /// 追加一个预设响应。
    pub fn with_response(self, text: impl Into<String>) -> Self {
        self.responses
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push_back(text.into());
        self
    }

    /// 批量追加多个预设响应。
    pub fn with_responses(self, texts: impl IntoIterator<Item = impl Into<String>>) -> Self {
        {
            let mut q = self.responses.lock().unwrap_or_else(|e| e.into_inner());
            for t in texts {
                q.push_back(t.into());
            }
        }
        self
    }

    /// 为下一次流式调用追加一段完整的中性事件脚本。
    pub fn with_event_script(self, steps: Vec<MockAgentStep>) -> Self {
        self.event_scripts
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .push_back(steps);
        self
    }

    /// 显式允许在脚本耗尽后继续调用，返回给定的默认成功响应。
    pub fn with_default_success(mut self, text: impl Into<String>) -> Self {
        self.default_success = Some(text.into());
        self
    }

    /// 返回尚未被消费的预设响应数量。
    pub fn remaining(&self) -> usize {
        self.responses
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .len()
    }

    /// 被调用的总次数。
    pub fn call_count(&self) -> usize {
        self.calls.lock().unwrap_or_else(|e| e.into_inner()).len()
    }

    /// 所有历史调用的任务字符串（按时间顺序）。
    pub fn calls(&self) -> Vec<String> {
        self.calls.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// 最近一次调用的任务字符串（从未调用过时返回 `None`）。
    pub fn last_task(&self) -> Option<String> {
        self.calls
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .last()
            .cloned()
    }

    /// 通过消息分发路径收到的多模态消息数量。
    pub fn message_count(&self) -> usize {
        self.messages
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .len()
    }

    /**
     * 最近收到的多模态消息（如果分发从未转发过 `Message` 则返回 `None`）。
     * 用于验证子 Agent 是否看到了用户的附件。
     */
    pub fn last_message(&self) -> Option<echo_core::llm::types::Message> {
        self.messages
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .last()
            .cloned()
    }

    /** 清空调用历史（不影响响应队列）。

    仅用于测试断言的重置，不等价于 `Agent::reset()`。
    */
    pub fn reset_calls(&self) {
        self.calls.lock().unwrap_or_else(|e| e.into_inner()).clear();
    }

    /**
     * 按顺序记录的所有 `set_working_dir` 调用（Sprint 8 隔离性测试）。
     * 每一项是要求该 Agent 绑定的路径；`None` 表示清除。
     */
    pub fn working_dir_calls(&self) -> Vec<Option<std::path::PathBuf>> {
        self.working_dirs
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// 值域调用方法（value-scoped streaming methods）收到的调用上下文。
    pub fn invocation_contexts(&self) -> Vec<echo_core::agent::AgentInvocationContext> {
        self.invocation_contexts
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clone()
    }

    /// 取出（并清空）值域流式方法记录的调用上下文。
    pub fn take_invocation_contexts(&self) -> Vec<echo_core::agent::AgentInvocationContext> {
        let mut contexts = self
            .invocation_contexts
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        std::mem::take(&mut *contexts)
    }

    /**
     * 弹出下一个预设文本响应；队列为空时回退到默认成功响应，
     * 两者都没有则返回错误。
     */
    fn next_response(&self) -> Result<String> {
        self.responses
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .pop_front()
            .or_else(|| self.default_success.clone())
            .ok_or_else(|| {
                ReactError::Agent(Box::new(AgentError::SubagentError(
                    "MockAgent response script exhausted".to_string(),
                )))
            })
    }

    /**
     * 构建下一次流式调用的事件流：优先使用预设的事件脚本，
     * 否则退化为发出一个 `FinalAnswer` 事件（取自 `next_response`）。
     */
    fn next_event_stream<'a>(
        &'a self,
        cancel: CancellationToken,
    ) -> Result<BoxStream<'a, Result<AgentEvent>>> {
        let scripted = self
            .event_scripts
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .pop_front();
        let steps = match scripted {
            Some(steps) => steps,
            None => vec![MockAgentStep::Event(Box::new(AgentEvent::FinalAnswer(
                self.next_response()?,
            )))],
        };
        let initial_delay = self.delay_ms;
        let events = async_stream::stream! {
            if initial_delay > 0 {
                tokio::select! {
                    _ = cancel.cancelled() => {
                        yield Ok(AgentEvent::Cancelled);
                        return;
                    }
                    _ = tokio::time::sleep(std::time::Duration::from_millis(initial_delay)) => {}
                }
            }
            for step in steps {
                match step {
                    MockAgentStep::Event(event) => yield Ok(*event),
                    MockAgentStep::Error(message) => {
                        yield Err(ReactError::Agent(Box::new(AgentError::SubagentError(message))));
                        return;
                    }
                    MockAgentStep::Delay(delay) => {
                        tokio::select! {
                            _ = cancel.cancelled() => {
                                yield Ok(AgentEvent::Cancelled);
                                return;
                            }
                            _ = tokio::time::sleep(delay) => {}
                        }
                    }
                }
            }
        };
        Ok(Box::pin(events))
    }
}

impl Agent for MockAgent {
    /// 返回该 Agent 的名称。
    fn name(&self) -> &str {
        &self.name
    }

    /// 返回该 Agent 报告的模型名称。
    fn model_name(&self) -> &str {
        &self.model_name
    }

    /// 返回该 Agent 的系统提示词。
    fn system_prompt(&self) -> &str {
        &self.system_prompt
    }

    /// 记录任务并返回下一个预设文本响应。
    fn execute<'a>(&'a self, task: &'a str) -> BoxFuture<'a, Result<String>> {
        Box::pin(async move {
            if self.delay_ms > 0 {
                tokio::time::sleep(std::time::Duration::from_millis(self.delay_ms)).await;
            }
            self.calls
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(task.to_string());
            self.next_response()
        })
    }

    /// 记录任务并返回下一个预设事件流（使用一个新的取消令牌）。
    fn execute_stream<'a>(
        &'a self,
        task: &'a str,
    ) -> BoxFuture<'a, Result<BoxStream<'a, Result<AgentEvent>>>> {
        Box::pin(async move {
            self.calls
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .push(task.to_string());
            self.next_event_stream(CancellationToken::new())
        })
    }

    /// 记录任务并返回下一个预设事件流，使用调用方提供的取消令牌。
    fn execute_stream_with_cancel<'a>(
        &'a self,
        task: &'a str,
        cancel: CancellationToken,
    ) -> BoxFuture<'a, Result<BoxStream<'a, Result<AgentEvent>>>> {
        Box::pin(async move {
            self.calls
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .push(task.to_string());
            self.next_event_stream(cancel)
        })
    }

    /**
     * 多模态分发的重载：记录收到的消息（便于测试断言子 Agent 是否看到了
     * 用户附件），并像文本路径一样消费一个预设响应。若不重载此方法，
     * trait 的默认实现会拒绝多模态分发，导致无法测试消息转发逻辑。
     */
    fn execute_stream_message_with_cancel<'a>(
        &'a self,
        message: echo_core::llm::types::Message,
        cancel: CancellationToken,
    ) -> BoxFuture<'a, Result<BoxStream<'a, Result<AgentEvent>>>> {
        Box::pin(async move {
            self.messages
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(message.clone());
            // 同时以常规调用（文本部分）记录，保持与 last_task 的一致性。
            let text = message.content.as_text().unwrap_or_default();
            self.calls
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(text);
            self.next_event_stream(cancel)
        })
    }

    /// 记录调用上下文，然后委托给 `execute_stream_with_cancel`。
    fn execute_stream_with_invocation_context<'a>(
        &'a self,
        task: &'a str,
        cancel: CancellationToken,
        invocation: echo_core::agent::AgentInvocationContext,
    ) -> BoxFuture<'a, Result<BoxStream<'a, Result<AgentEvent>>>> {
        Box::pin(async move {
            self.invocation_contexts
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .push(invocation);
            self.execute_stream_with_cancel(task, cancel).await
        })
    }

    /// 记录调用上下文，然后委托给 `execute_stream_message_with_cancel`。
    fn execute_stream_message_with_invocation_context<'a>(
        &'a self,
        message: echo_core::llm::types::Message,
        cancel: CancellationToken,
        invocation: echo_core::agent::AgentInvocationContext,
    ) -> BoxFuture<'a, Result<BoxStream<'a, Result<AgentEvent>>>> {
        Box::pin(async move {
            self.invocation_contexts
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .push(invocation);
            self.execute_stream_message_with_cancel(message, cancel)
                .await
        })
    }

    /**
     * `chat()` 同样记录调用并消费预设响应队列。
     * 注意：MockAgent 不维护真实的对话上下文，这里只是满足调用契约。
     */
    fn chat<'a>(&'a self, message: &'a str) -> BoxFuture<'a, Result<String>> {
        Box::pin(async move {
            self.calls
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(message.to_string());
            self.next_response()
        })
    }

    /// 记录消息并返回下一个预设事件流（使用一个新的取消令牌）。
    fn chat_stream<'a>(
        &'a self,
        message: &'a str,
    ) -> BoxFuture<'a, Result<BoxStream<'a, Result<AgentEvent>>>> {
        Box::pin(async move {
            self.calls
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .push(message.to_string());
            self.next_event_stream(CancellationToken::new())
        })
    }

    /// 记录消息并返回下一个预设事件流，使用调用方提供的取消令牌。
    fn chat_stream_with_cancel<'a>(
        &'a self,
        message: &'a str,
        cancel: CancellationToken,
    ) -> BoxFuture<'a, Result<BoxStream<'a, Result<AgentEvent>>>> {
        Box::pin(async move {
            self.calls
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .push(message.to_string());
            self.next_event_stream(cancel)
        })
    }

    /// 清空调用历史，模拟真实 Agent 的重置语义。
    fn reset(&self) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
        Box::pin(async move {
            self.calls.lock().unwrap_or_else(|e| e.into_inner()).clear();
        })
    }

    /**
     * 记录 `set_working_dir` 调用，便于 Sprint 8 隔离性测试验证子 Agent
     * 是否被 chroot 进了对应的 worktree（以及之后是否被清除）。
     */
    fn set_working_dir(&self, path: Option<std::path::PathBuf>) {
        self.working_dirs
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(path);
    }

    /// 记录一次“清除工作目录”的调用（以 `None` 形式追加）。
    fn clear_working_dir(&self) {
        self.working_dirs
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(None);
    }
}

/// 总是返回错误的 Mock Agent（用于测试编排逻辑的容错行为）。
pub struct FailingMockAgent {
    /// 该 Agent 的名称。
    name: String,
    /// 默认的错误消息（脚本耗尽时使用）。
    error_message: String,
    /// 每次调用返回的类型化失败类别。
    failure: MockAgentFailure,
    /// 待消费的预设事件脚本队列。
    event_scripts: Arc<Mutex<VecDeque<Vec<MockAgentStep>>>>,
    /// 已收到的任务字符串，按调用顺序记录。
    calls: Arc<Mutex<Vec<String>>>,
}

impl FailingMockAgent {
    /// 创建一个总是失败的 Mock Agent。
    pub fn new(name: impl Into<String>, error_message: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            error_message: error_message.into(),
            failure: MockAgentFailure::Initialization,
            event_scripts: Arc::new(Mutex::new(VecDeque::new())),
            calls: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// 选择每次调用返回的类型化 Agent 失败类别。
    pub fn with_failure(mut self, failure: MockAgentFailure) -> Self {
        self.failure = failure;
        self
    }

    /** 为一次流式调用追加部分事件、延迟以及最终的错误。

    `MockAgentStep::Error` 会被映射为所选的类型化失败类别。
    当脚本耗尽时，会发出配置的默认错误。
    */
    pub fn with_event_script(self, steps: Vec<MockAgentStep>) -> Self {
        self.event_scripts
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .push_back(steps);
        self
    }

    /// 获取该 Mock Agent 被调用的次数。
    pub fn call_count(&self) -> usize {
        self.calls.lock().unwrap_or_else(|e| e.into_inner()).len()
    }

    /// 所有历史调用的任务字符串（按时间顺序）。
    pub fn calls(&self) -> Vec<String> {
        self.calls.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// 记录一次调用的任务字符串。
    fn record_call(&self, task: &str) {
        self.calls
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .push(task.to_string());
    }

    /**
     * 构建下一次流式调用的事件流：使用预设脚本，若脚本耗尽则
     * 直接产出配置的默认错误。
     */
    fn next_event_stream<'a>(
        &'a self,
        cancel: CancellationToken,
    ) -> BoxStream<'a, Result<AgentEvent>> {
        let steps = self
            .event_scripts
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .pop_front()
            .unwrap_or_else(|| vec![MockAgentStep::Error(self.error_message.clone())]);
        let events = async_stream::stream! {
            for step in steps {
                match step {
                    MockAgentStep::Event(event) => yield Ok(*event),
                    MockAgentStep::Error(message) => {
                        yield Err(self.failure.error(message));
                        return;
                    }
                    MockAgentStep::Delay(delay) => {
                        tokio::select! {
                            _ = cancel.cancelled() => {
                                yield Ok(AgentEvent::Cancelled);
                                return;
                            }
                            _ = tokio::time::sleep(delay) => {}
                        }
                    }
                }
            }
        };
        Box::pin(events)
    }
}

impl Agent for FailingMockAgent {
    /// 返回该 Agent 的名称。
    fn name(&self) -> &str {
        &self.name
    }

    /// 固定返回 `"mock-model"`。
    fn model_name(&self) -> &str {
        "mock-model"
    }

    /// 固定返回 `"failing mock agent"`。
    fn system_prompt(&self) -> &str {
        "failing mock agent"
    }

    /// 记录任务，并返回配置的类型化失败错误。
    fn execute<'a>(&'a self, task: &'a str) -> BoxFuture<'a, Result<String>> {
        Box::pin(async move {
            self.record_call(task);
            Err(self.failure.error(self.error_message.clone()))
        })
    }

    /// 记录任务并返回失败事件流（使用一个新的取消令牌）。
    fn execute_stream<'a>(
        &'a self,
        task: &'a str,
    ) -> BoxFuture<'a, Result<BoxStream<'a, Result<AgentEvent>>>> {
        Box::pin(async move {
            self.record_call(task);
            Ok(self.next_event_stream(CancellationToken::new()))
        })
    }

    /// 记录任务并返回失败事件流，使用调用方提供的取消令牌。
    fn execute_stream_with_cancel<'a>(
        &'a self,
        task: &'a str,
        cancel: CancellationToken,
    ) -> BoxFuture<'a, Result<BoxStream<'a, Result<AgentEvent>>>> {
        Box::pin(async move {
            self.record_call(task);
            Ok(self.next_event_stream(cancel))
        })
    }

    /// 委托给 `execute`。
    fn chat<'a>(&'a self, message: &'a str) -> BoxFuture<'a, Result<String>> {
        Box::pin(async move { self.execute(message).await })
    }

    /// 清空调用历史。
    fn reset(&self) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
        Box::pin(async move {
            self.calls.lock().unwrap_or_else(|e| e.into_inner()).clear();
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;

    /// 验证 MockAgent 能按脚本先发出部分事件，再以错误结束流。
    #[tokio::test]
    async fn mock_agent_scripts_partial_event_then_error() -> Result<()> {
        let agent = MockAgent::new("partial").with_event_script(vec![
            MockAgentStep::Event(Box::new(AgentEvent::Token("part".to_string()))),
            MockAgentStep::Error("stream failed".to_string()),
        ]);
        let mut events = agent.execute_stream("work").await?;

        assert!(matches!(
            events.next().await,
            Some(Ok(AgentEvent::Token(token))) if token == "part"
        ));
        assert!(matches!(
            events.next().await,
            Some(Err(ReactError::Agent(error)))
                if matches!(*error, AgentError::SubagentError(ref message) if message == "stream failed")
        ));
        assert!(events.next().await.is_none());
        Ok(())
    }

    /**
     * 验证 FailingMockAgent 能按脚本先发出部分事件，再以指定类型化错误结束流，
     * 且调用历史被正确记录。
     */
    #[tokio::test]
    async fn failing_mock_agent_scripts_partial_event_then_typed_error() -> Result<()> {
        let agent = FailingMockAgent::new("partial-failure", "default failure")
            .with_failure(MockAgentFailure::Timeout)
            .with_event_script(vec![
                MockAgentStep::Event(Box::new(AgentEvent::Token("part".to_string()))),
                MockAgentStep::Error("deadline".to_string()),
            ]);
        let mut events = agent
            .execute_stream_with_cancel("work", CancellationToken::new())
            .await?;

        assert!(matches!(
            events.next().await,
            Some(Ok(AgentEvent::Token(token))) if token == "part"
        ));
        assert!(matches!(
            events.next().await,
            Some(Err(ReactError::Agent(error)))
                if matches!(*error, AgentError::Timeout(ref message) if message == "deadline")
        ));
        assert!(events.next().await.is_none());
        assert_eq!(agent.calls(), vec!["work".to_string()]);
        Ok(())
    }
}
