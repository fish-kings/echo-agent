/*!
 * AgentHandle —— 围绕共享 ReactAgent 的作用域访问包装器。
 *
 * 取代原始的 `Arc<RwLock<ReactAgent>>`，使调用方无需直接加锁
 * 或引入 `ReactAgent`。该 handle 提供：
 *
 * | 方法 | 锁 | 闭包 |
 * |--------|------|---------|
 * | [`read`](AgentHandle::read) | 读锁（异步） | 同步 |
 * | [`read_async`](AgentHandle::read_async) | 读锁（异步） | 异步（装箱） |
 * | [`write`](AgentHandle::write) | 写锁（异步） | 同步 |
 * | [`write_async`](AgentHandle::write_async) | 写锁（异步） | 异步（装箱） |
 * | [`try_write`](AgentHandle::try_write) | 写锁（尝试） | 同步 |
 * | [`inner`](AgentHandle::inner) | 无 | 逃生舱口 |
 *
 * `read_async` 和 `write_async` 会在 `.await` 期间持有锁；不要将它们
 * 用于长时间运行的流式操作或请求/响应循环。
 *
 * 执行串行化由 `ReactAgent.execution_mutex` 内部处理
 * —— 调用方在调用 `execute()`、`chat()` 或流式方法之前
 * 不需要额外加锁。
 */

use crate::agent::{Agent, AgentEvent, AgentInvocationContext, CancellationToken};
use crate::error::Result;
use crate::llm::types::Message;
use crate::workflow::SharedAgent;
use futures::future::BoxFuture;
use futures::stream::BoxStream;
use futures::stream::StreamExt;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::RwLock;

use super::ReactAgent;

// ── RwLockAgentWrapper ─────────────────────────────────────────────────────────

/**
 * 通过委托给 `RwLock<ReactAgent>` 来实现 [`Agent`] 的包装器。
 *
 * 这使得 `AgentHandle -> SharedAgent` 的转换得以用于
 * Graph 工作流方法。每个 `Agent` trait 方法都会获取读锁，
 * 调用对应的 `ReactAgent` 方法，并在持锁范围内等待结果。
 *
 * 不可变字段（`name`、`model_name`、`system_prompt`）会在构造时
 * 缓存下来，以避免为返回 `&str` 的同步 trait 方法持有锁。
 */
struct RwLockAgentWrapper {
    /// 底层共享的 Agent 实例
    inner: Arc<RwLock<ReactAgent>>,
    /// 缓存的 Agent 名称
    name: String,
    /// 缓存的模型名称
    model_name: String,
    /// 缓存的系统提示词
    system_prompt: String,
}

/// 一次需要在持锁状态下转发的流式调用
enum LockedStreamCall {
    Execute(String),
    ExecuteWithCancel(String, CancellationToken),
    ExecuteWithInvocation(String, CancellationToken, AgentInvocationContext),
    ExecuteMessageWithCancel(Message, CancellationToken),
    ExecuteMessageWithInvocation(Message, CancellationToken, AgentInvocationContext),
    Chat(String),
    ChatWithCancel(String, CancellationToken),
    ChatMessageWithCancel(Message, CancellationToken),
    ChatMessageWithInvocation(Message, CancellationToken, AgentInvocationContext),
}

/// 将 Agent 的流式结果转发到 mpsc 通道
async fn relay_agent_stream(
    result: Result<BoxStream<'_, Result<AgentEvent>>>,
    tx: &tokio::sync::mpsc::Sender<Result<AgentEvent>>,
) {
    match result {
        Ok(mut stream) => {
            while let Some(event) = stream.next().await {
                if tx.send(event).await.is_err() {
                    break;
                }
            }
        }
        Err(error) => {
            let _ = tx.send(Err(error)).await;
        }
    }
}

impl RwLockAgentWrapper {
    async fn from_handle(handle: &AgentHandle) -> Self {
        let guard = handle.agent.read().await;
        Self {
            inner: handle.agent.clone(),
            name: guard.name().to_string(),
            model_name: guard.model_name().to_string(),
            system_prompt: guard.system_prompt().to_string(),
        }
    }

    fn forward_stream<'a>(
        &'a self,
        call: LockedStreamCall,
    ) -> BoxFuture<'a, Result<BoxStream<'a, Result<AgentEvent>>>> {
        let inner = self.inner.clone();
        Box::pin(async move {
            let (tx, rx) = tokio::sync::mpsc::channel::<Result<AgentEvent>>(64);

            tokio::spawn(async move {
                let guard = inner.read().await;
                match call {
                    LockedStreamCall::Execute(task) => {
                        relay_agent_stream(guard.execute_stream(&task).await, &tx).await;
                    }
                    LockedStreamCall::ExecuteWithCancel(task, cancel) => {
                        relay_agent_stream(
                            guard.execute_stream_with_cancel(&task, cancel).await,
                            &tx,
                        )
                        .await;
                    }
                    LockedStreamCall::ExecuteWithInvocation(task, cancel, invocation) => {
                        relay_agent_stream(
                            guard
                                .execute_stream_with_invocation_context(&task, cancel, invocation)
                                .await,
                            &tx,
                        )
                        .await;
                    }
                    LockedStreamCall::ExecuteMessageWithCancel(message, cancel) => {
                        relay_agent_stream(
                            guard
                                .execute_stream_message_with_cancel(message, cancel)
                                .await,
                            &tx,
                        )
                        .await;
                    }
                    LockedStreamCall::ExecuteMessageWithInvocation(message, cancel, invocation) => {
                        relay_agent_stream(
                            guard
                                .execute_stream_message_with_invocation_context(
                                    message, cancel, invocation,
                                )
                                .await,
                            &tx,
                        )
                        .await;
                    }
                    LockedStreamCall::Chat(message) => {
                        relay_agent_stream(guard.chat_stream(&message).await, &tx).await;
                    }
                    LockedStreamCall::ChatWithCancel(message, cancel) => {
                        relay_agent_stream(
                            guard.chat_stream_with_cancel(&message, cancel).await,
                            &tx,
                        )
                        .await;
                    }
                    LockedStreamCall::ChatMessageWithCancel(message, cancel) => {
                        relay_agent_stream(
                            guard.chat_stream_message_with_cancel(message, cancel).await,
                            &tx,
                        )
                        .await;
                    }
                    LockedStreamCall::ChatMessageWithInvocation(message, cancel, invocation) => {
                        relay_agent_stream(
                            guard
                                .chat_stream_message_with_invocation_context(
                                    message, cancel, invocation,
                                )
                                .await,
                            &tx,
                        )
                        .await;
                    }
                }
            });

            let stream = futures::stream::unfold(rx, |mut rx| async move {
                rx.recv().await.map(|event| (event, rx))
            });
            Ok(stream.boxed())
        })
    }
}

impl Agent for RwLockAgentWrapper {
    fn name(&self) -> &str {
        &self.name
    }
    fn model_name(&self) -> &str {
        &self.model_name
    }
    fn system_prompt(&self) -> &str {
        &self.system_prompt
    }

    fn tool_names(&self) -> Vec<String> {
        self.inner
            .try_read()
            .ok()
            .map(|agent| agent.tool_names())
            .unwrap_or_default()
    }

    fn tool_definitions(&self) -> Vec<crate::llm::ToolDefinition> {
        self.inner
            .try_read()
            .ok()
            .map(|agent| agent.tool_definitions())
            .unwrap_or_default()
    }

    fn disabled_tool_names(&self) -> std::collections::HashSet<String> {
        self.inner
            .try_read()
            .ok()
            .map(|agent| agent.disabled_tool_names())
            .unwrap_or_default()
    }

    fn tool_visibility_policy(&self) -> crate::agent::ToolVisibilityPolicy {
        self.inner
            .try_read()
            .ok()
            .map(|agent| agent.tool_visibility_policy())
            .unwrap_or_default()
    }

    fn working_dir(&self) -> Option<std::path::PathBuf> {
        self.inner
            .try_read()
            .ok()
            .and_then(|agent| agent.working_dir())
    }

    fn skill_names(&self) -> Vec<String> {
        self.inner
            .try_read()
            .ok()
            .map(|agent| agent.skill_names())
            .unwrap_or_default()
    }

    fn mcp_server_names(&self) -> Vec<String> {
        self.inner
            .try_read()
            .ok()
            .map(|agent| agent.mcp_server_names())
            .unwrap_or_default()
    }

    fn close<'a>(&'a self) -> BoxFuture<'a, Result<()>> {
        let inner = self.inner.clone();
        Box::pin(async move {
            let guard = inner.read().await;
            guard.close().await
        })
    }

    fn steer_input(
        &self,
        expected_turn_id: Option<&str>,
        message: crate::llm::types::Message,
    ) -> std::result::Result<String, echo_core::agent::AgentSteerError> {
        let guard = self
            .inner
            .try_read()
            .map_err(|_| echo_core::agent::AgentSteerError::StateUnavailable)?;
        guard.steer_input(expected_turn_id, message)
    }

    fn steer_input_tracked(
        &self,
        expected_turn_id: Option<&str>,
        message: crate::llm::types::Message,
    ) -> std::result::Result<echo_core::agent::AgentSteerReceipt, echo_core::agent::AgentSteerError>
    {
        let guard = self
            .inner
            .try_read()
            .map_err(|_| echo_core::agent::AgentSteerError::StateUnavailable)?;
        guard.steer_input_tracked(expected_turn_id, message)
    }

    fn execute<'a>(&'a self, task: &'a str) -> BoxFuture<'a, Result<String>> {
        let inner = self.inner.clone();
        let task = task.to_string();
        Box::pin(async move {
            let guard = inner.read().await;
            guard.execute(&task).await
        })
    }

    fn chat<'a>(&'a self, message: &'a str) -> BoxFuture<'a, Result<String>> {
        let inner = self.inner.clone();
        let message = message.to_string();
        Box::pin(async move {
            let guard = inner.read().await;
            guard.chat(&message).await
        })
    }

    fn execute_stream<'a>(
        &'a self,
        task: &'a str,
    ) -> BoxFuture<'a, Result<BoxStream<'a, Result<AgentEvent>>>> {
        self.forward_stream(LockedStreamCall::Execute(task.to_string()))
    }

    fn execute_stream_with_cancel<'a>(
        &'a self,
        task: &'a str,
        cancel: CancellationToken,
    ) -> BoxFuture<'a, Result<BoxStream<'a, Result<AgentEvent>>>> {
        self.forward_stream(LockedStreamCall::ExecuteWithCancel(
            task.to_string(),
            cancel,
        ))
    }

    fn execute_stream_with_invocation_context<'a>(
        &'a self,
        task: &'a str,
        cancel: CancellationToken,
        invocation: AgentInvocationContext,
    ) -> BoxFuture<'a, Result<BoxStream<'a, Result<AgentEvent>>>> {
        self.forward_stream(LockedStreamCall::ExecuteWithInvocation(
            task.to_string(),
            cancel,
            invocation,
        ))
    }

    fn execute_stream_message_with_cancel<'a>(
        &'a self,
        message: Message,
        cancel: CancellationToken,
    ) -> BoxFuture<'a, Result<BoxStream<'a, Result<AgentEvent>>>> {
        self.forward_stream(LockedStreamCall::ExecuteMessageWithCancel(message, cancel))
    }

    fn execute_stream_message_with_invocation_context<'a>(
        &'a self,
        message: Message,
        cancel: CancellationToken,
        invocation: AgentInvocationContext,
    ) -> BoxFuture<'a, Result<BoxStream<'a, Result<AgentEvent>>>> {
        self.forward_stream(LockedStreamCall::ExecuteMessageWithInvocation(
            message, cancel, invocation,
        ))
    }

    fn chat_stream<'a>(
        &'a self,
        message: &'a str,
    ) -> BoxFuture<'a, Result<BoxStream<'a, Result<AgentEvent>>>> {
        self.forward_stream(LockedStreamCall::Chat(message.to_string()))
    }

    fn chat_stream_with_cancel<'a>(
        &'a self,
        message: &'a str,
        cancel: CancellationToken,
    ) -> BoxFuture<'a, Result<BoxStream<'a, Result<AgentEvent>>>> {
        self.forward_stream(LockedStreamCall::ChatWithCancel(
            message.to_string(),
            cancel,
        ))
    }

    fn chat_stream_message_with_cancel<'a>(
        &'a self,
        message: Message,
        cancel: CancellationToken,
    ) -> BoxFuture<'a, Result<BoxStream<'a, Result<AgentEvent>>>> {
        self.forward_stream(LockedStreamCall::ChatMessageWithCancel(message, cancel))
    }

    fn chat_stream_message_with_invocation_context<'a>(
        &'a self,
        message: Message,
        cancel: CancellationToken,
        invocation: AgentInvocationContext,
    ) -> BoxFuture<'a, Result<BoxStream<'a, Result<AgentEvent>>>> {
        self.forward_stream(LockedStreamCall::ChatMessageWithInvocation(
            message, cancel, invocation,
        ))
    }

    fn reset(&self) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
        let inner = self.inner.clone();
        Box::pin(async move {
            let guard = inner.read().await;
            guard.reset().await;
        })
    }

    fn current_run_id(&self) -> Option<String> {
        self.inner
            .try_read()
            .ok()
            .and_then(|agent| agent.current_run_id())
    }

    fn token_usage_summary(&self) -> crate::tokenizer::UsageSummary {
        self.inner
            .try_read()
            .ok()
            .map(|agent| agent.token_usage_summary())
            .unwrap_or_default()
    }

    fn set_external_context(&self, context: &crate::tools::ExternalRunContext) {
        if let Ok(agent) = self.inner.try_read() {
            agent.set_external_context(context);
        }
    }

    fn clear_external_context(&self) {
        if let Ok(agent) = self.inner.try_read() {
            agent.clear_external_context();
        }
    }

    fn set_working_dir(&self, path: Option<std::path::PathBuf>) {
        if let Ok(agent) = self.inner.try_read() {
            agent.set_working_dir(path);
        }
    }

    fn clear_working_dir(&self) {
        if let Ok(agent) = self.inner.try_read() {
            agent.clear_working_dir();
        }
    }

    fn register_tool(&self, tool: Box<dyn crate::tools::Tool>) {
        if let Ok(agent) = self.inner.try_read() {
            agent.register_tool(tool);
        }
    }

    fn remove_tool(&self, name: &str) -> bool {
        self.inner
            .try_read()
            .ok()
            .map(|agent| agent.remove_tool(name))
            .unwrap_or(false)
    }

    fn messages(&self) -> Vec<Message> {
        self.inner
            .try_read()
            .ok()
            .map(|agent| agent.messages())
            .unwrap_or_default()
    }

    fn set_system_prompt(&self, prompt: &str) {
        if let Ok(agent) = self.inner.try_read() {
            agent.set_system_prompt(prompt);
        }
    }

    fn delegate_to<'a>(&'a self, target: &'a str, task: &'a str) -> BoxFuture<'a, Result<String>> {
        let inner = self.inner.clone();
        let target = target.to_string();
        let task = task.to_string();
        Box::pin(async move {
            let guard = inner.read().await;
            guard.delegate_to(&target, &task).await
        })
    }
}

// ── AgentHandle ────────────────────────────────────────────────────────────────

/**
 * 面向共享 [`ReactAgent`] 的作用域访问句柄。
 *
 * 提供对 Agent 的安全、符合人体工程学的访问方式，调用方无需直接管理锁。
 * 执行串行化（防止并发的 `execute`/`chat`/流式调用）由 `ReactAgent`
 * 的 `execution_mutex` 内部处理。
 *
 * Clone 的开销很低（仅是 Arc 克隆）。
 */
#[derive(Clone)]
pub struct AgentHandle {
    /// 底层共享的 ReactAgent 实例
    agent: Arc<RwLock<ReactAgent>>,
}

impl AgentHandle {
    /// 向当前活跃轮次注入用户输入，而不等待其流式结果。
    pub async fn steer_input(
        &self,
        expected_turn_id: Option<&str>,
        message: crate::llm::types::Message,
    ) -> std::result::Result<String, crate::agent::TurnSteerError> {
        let guard = self.agent.read().await;
        guard.steer_input(expected_turn_id, message)
    }

    /**
     * 注入输入并保留一份回执（receipt），该回执由当前活跃轮次真实的
     * 邮箱清空（mailbox-drain）和终止边界驱动。
     */
    pub async fn steer_input_tracked(
        &self,
        expected_turn_id: Option<&str>,
        message: crate::llm::types::Message,
    ) -> std::result::Result<crate::agent::AgentSteerReceipt, crate::agent::TurnSteerError> {
        let guard = self.agent.read().await;
        guard.steer_input_tracked(expected_turn_id, message)
    }

    /// 包装一个已存在的 `Arc<RwLock<ReactAgent>>`。
    pub fn from_arc(agent: Arc<RwLock<ReactAgent>>) -> Self {
        Self { agent }
    }

    /// 从一个拥有所有权的 `ReactAgent` 构建。
    pub fn new(agent: ReactAgent) -> Self {
        Self {
            agent: Arc::new(RwLock::new(agent)),
        }
    }

    /// 逃生舱口 —— 返回内部的 `Arc<RwLock<ReactAgent>>`。
    pub fn inner(&self) -> &Arc<RwLock<ReactAgent>> {
        &self.agent
    }

    /**
     * 生成同一个底层 `ReactAgent` 的 [`SharedAgent`] 视图。
     *
     * 返回的 `SharedAgent` 会将所有 [`Agent`] trait 调用通过 `RwLock`
     * 委托出去。不可变字段会在转换时被缓存。
     */
    pub async fn as_shared_agent(&self) -> SharedAgent {
        Arc::new(RwLockAgentWrapper::from_handle(self).await) as SharedAgent
    }

    /// 获取读锁，执行一个**同步**闭包，并返回其结果。
    pub async fn read<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&ReactAgent) -> R,
    {
        let guard = self.agent.read().await;
        f(&guard)
    }

    /**
     * 获取读锁，执行一个返回装箱 future 的闭包，
     * 并在持锁状态下 `.await` 该 future。
     *
     * 不要将此方法用于流式操作或其他长时间运行的操作。
     */
    pub async fn read_async<F, R>(&self, f: F) -> R
    where
        F: for<'a> FnOnce(&'a ReactAgent) -> Pin<Box<dyn Future<Output = R> + Send + 'a>>,
    {
        let guard = self.agent.read().await;
        f(&guard).await
    }

    /// 获取写锁，执行一个**同步**闭包，并返回其结果。
    pub async fn write<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut ReactAgent) -> R,
    {
        let mut guard = self.agent.write().await;
        f(&mut guard)
    }

    /**
     * 获取写锁，执行一个返回装箱 future 的闭包，
     * 并在持锁状态下 `.await` 该 future。
     *
     * 不要将此方法用于流式操作或其他长时间运行的操作；在被 await 的
     * future 完成之前，它会阻塞所有读者和写者。
     */
    pub async fn write_async<F, R>(&self, f: F) -> R
    where
        F: for<'a> FnOnce(&'a mut ReactAgent) -> Pin<Box<dyn Future<Output = R> + Send + 'a>>,
    {
        let mut guard = self.agent.write().await;
        f(&mut guard).await
    }

    /**
     * 非阻塞的写入尝试 —— 如果锁已被其他调用方持有，
     * 则返回 `None`。
     */
    pub fn try_write<F, R>(&self, f: F) -> Option<R>
    where
        F: FnOnce(&mut ReactAgent) -> R,
    {
        let mut guard = self.agent.try_write().ok()?;
        Some(f(&mut guard))
    }

    /**
     * 将该 handle 转换为一个在运行时通过 `RwLock` 进行委托的
     * `Box<dyn Agent>`。不可变字段（name、model_name、system_prompt）
     * 会在转换时被缓存；其余所有 [`Agent`] trait 方法都会在每次调用时
     * 获取一次读锁。
     *
     * 当一个 Agent 必须同时满足以下两点时，这个方法会很有用：
     * - 可以通过 `AgentHandle` 访问（用于工具注册/生命周期管理），
     * - 可以作为 `Box<dyn Agent>` 传递（例如传给子 Agent 注册表）。
     */
    pub async fn to_boxed_agent(&self) -> Box<dyn Agent> {
        Box::new(RwLockAgentWrapper::from_handle(self).await)
    }
}
