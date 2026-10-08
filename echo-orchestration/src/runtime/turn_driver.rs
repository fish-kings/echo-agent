/*!
面向单次有限 Agent 调用的通用轮次驱动器。

# 目的

这是从框架侧抽取出的驱动器模式，EKO 目前在各个入口分别实现了它
（交互式轮次使用 `drive_chat`，无头任务运行使用 `drive_run_async`）。
它只驱动一次有限的 Agent 调用：用带版本的 [`EventEnvelope`] 传输层包装原始事件流，
把每个信封转发给唯一的 [`EventSink`]，强制执行"恰好一个终态"契约，
并返回一个类型化的 [`TurnReceipt`]，其中执行结果与投递结果相互独立。
聊天、执行/无头以及任务驱动的调用方只在请求字段和 sink 上有所不同；
循环、终态映射与用量统计都是共享的。

# 业界依据

- OpenAI Codex 的 `Thread -> Turn -> Item`：一个轮次是具有唯一类型化终态的有限单元；
  进度以稳定身份的事件形式投递，最终的 `turn/completed` 携带收敛后的状态。
- Claude Code 子代理：调用方收到结构化结果，而不是从"流已返回"推断成功。

# 分层决策

驱动 Agent 流、信封排序、终态映射和用量统计属于任何消费者都需要的框架关注点；
产品策略（聊天日志、TaskRuntime 运行、前台准入、续跑）留在应用层并组合本驱动器。
应用自身的 `TurnOutcome`/驱动循环将在后续计划的工作中迁移到这一原语上，
届时会删除重复的循环。

# 示例

```
use echo_core::agent::{Agent, EventIdentity};
use echo_orchestration::runtime::turn_driver::{
    AgentTurnDriver, EventSink, SinkControl, TurnDeliveryOutcome, TurnMode, TurnRequest,
};

# struct MyAgent;
# impl Agent for MyAgent {
#     fn name(&self) -> &str { "my-agent" }
#     fn model_name(&self) -> &str { "test-model" }
#     fn system_prompt(&self) -> &str { "" }
#     fn execute<'a>(&'a self, _task: &'a str)
#         -> futures::future::BoxFuture<'a, echo_core::error::Result<String>> {
#         Box::pin(async { Ok("done".to_string()) })
#     }
#     fn execute_stream<'a>(&'a self, _task: &'a str) -> futures::future::BoxFuture<
#         'a,
#         echo_core::error::Result<
#             futures::stream::BoxStream<'a, echo_core::error::Result<echo_core::agent::AgentEvent>>,
#         >,
#     > {
#         Box::pin(async {
#             use futures::StreamExt;
#             Ok(futures::stream::iter([Ok(echo_core::agent::AgentEvent::FinalAnswer(
#                 "done".to_string(),
#             ))])
#             .boxed())
#         })
#     }
# }
# struct PrintSink;
# #[async_trait::async_trait]
# impl EventSink for PrintSink {
#     async fn on_event(&self, _envelope: echo_core::agent::EventEnvelope)
#         -> echo_core::error::Result<SinkControl> {
#         Ok(SinkControl::Continue)
#     }
# }
# async fn run() -> echo_core::error::Result<()> {
let agent = MyAgent;
let identity = EventIdentity::new("stream-1", "turn-1")?;
let request = TurnRequest::new(identity, "hello").mode(TurnMode::Execute);
let receipt = AgentTurnDriver.drive(&agent, request, &PrintSink).await;
assert_eq!(receipt.outcome.status(), "completed");
assert_eq!(receipt.delivery, TurnDeliveryOutcome::Delivered);
assert_eq!(receipt.final_answer.as_deref(), Some("done"));
# Ok(())
# }
*/

use async_trait::async_trait;
use echo_core::agent::{
    Agent, AgentEvent, AgentInvocationContext, AgentSteerTurnOutcome, CancellationToken,
    EventEnvelope, EventIdentity, ExecutionUsage, MessageId, TurnId, envelope_event_stream_after,
};
use echo_core::error::{AgentFailure, ReactError};
use echo_core::llm::Message;
use futures::StreamExt;
use std::time::{Duration, Instant};

/// 本轮驱动的 Agent 流类型。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TurnMode {
    /// 多轮对话流（`chat_stream_with_cancel`）。
    #[default]
    Chat,
    /// 全新上下文的执行流（`execute_stream_with_cancel`）。
    Execute,
}

/// 待驱动的一次有限 Agent 调用。
#[derive(Debug, Clone)]
pub struct TurnRequest {
    /// 文本或结构化的多模态输入。
    pub input: TurnInput,
    /// 打在每个发出信封上的传输身份。
    pub identity: EventIdentity,
    /// 聊天流或执行流类型。
    pub mode: TurnMode,
    /** 本次调用的协作式取消令牌。`None` 时会启动一个独立作用域，
    sink 仍可通过返回 `false` 关闭它。 */
    pub cancel: Option<CancellationToken>,
    /** 最后一个已持久化的信封序号；包装后的流从
    `last_persisted_sequence + 1` 开始继续编号。 */
    pub last_persisted_sequence: u64,
    /** 按值作用域的运行、工具可见性、工作目录与历史元数据。
    结构化消息调用方使用感知调用上下文的 Agent API。 */
    pub invocation: Option<AgentInvocationContext>,
    /// 初始输入生命周期的发布端（通过 `with_input_receipt` 附加）。
    input_publisher: Option<TurnInputLifecycle>,
}

/**
[`AgentTurnDriver`] 接受的一条初始输入的生命周期状态。

`Accepted` 由驱动器在请求/身份校验通过后、调用 Agent 流 API 之前立即发布。
具体的 Agent 在其输入进入模型上下文后发布 `Drained`。
它绝不会从输出信封、EOF 或终态事件中推断得出。
终态记录所属轮次的结果，以及输入是否到达了排空边界。
*/
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TurnInputState {
    /// 已创建，尚未被驱动器接受。
    Pending,
    /// 驱动器已完成校验并即将调用 Agent 流 API。
    Accepted,
    /// 输入已进入模型上下文。
    Drained,
    /// 所属轮次已结算。
    TurnSettled {
        /// 所属轮次的终态结果。
        outcome: AgentSteerTurnOutcome,
        /// 输入在结算前是否已到达排空边界。
        drained: bool,
    },
}

/// 输入生命周期的共享内部状态。
struct TurnInputLifecycleInner {
    /// 生命周期状态的广播发送端。
    state: tokio::sync::watch::Sender<TurnInputState>,
    /// 是否已结算，保证终态只发布一次。
    settled: std::sync::atomic::AtomicBool,
}

impl std::fmt::Debug for TurnInputLifecycleInner {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("TurnInputLifecycleInner")
            .field(
                "settled",
                &self.settled.load(std::sync::atomic::Ordering::Acquire),
            )
            .finish()
    }
}

/// 初始输入生命周期的发布端句柄，可克隆共享。
#[derive(Clone)]
struct TurnInputLifecycle {
    /// 共享内部状态。
    inner: std::sync::Arc<TurnInputLifecycleInner>,
}

impl std::fmt::Debug for TurnInputLifecycle {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("TurnInputLifecycle")
            .field("inner", &self.inner)
            .finish()
    }
}

/// 一条初始轮次输入的无损回执。
#[derive(Debug, Clone)]
pub struct TurnInputReceipt {
    /// 所属轮次的标识。
    turn_id: String,
    /// 生命周期状态的接收端。
    state: tokio::sync::watch::Receiver<TurnInputState>,
}

impl TurnInputReceipt {
    /// 创建处于 `Pending` 状态的回执及其配对的生命周期发布端。
    fn new(turn_id: String) -> (Self, TurnInputLifecycle) {
        let (state, receiver) = tokio::sync::watch::channel(TurnInputState::Pending);
        let inner = std::sync::Arc::new(TurnInputLifecycleInner {
            state,
            settled: std::sync::atomic::AtomicBool::new(false),
        });
        (
            Self {
                turn_id,
                state: receiver,
            },
            TurnInputLifecycle { inner },
        )
    }

    /// 拥有该输入的轮次的稳定标识。
    pub fn turn_id(&self) -> &str {
        &self.turn_id
    }

    /// 不等待，直接获取最新生命周期状态的快照。
    pub fn state(&self) -> TurnInputState {
        if self.state.has_changed().is_err() {
            return self.synthesize_closed_terminal();
        }
        self.state.borrow().clone()
    }

    /// 发布端已关闭时合成终态：保留已有终态，否则视为 `Dropped`。
    fn synthesize_closed_terminal(&self) -> TurnInputState {
        let current = self.state.borrow().clone();
        match current {
            state @ TurnInputState::TurnSettled { .. } => state,
            state => TurnInputState::TurnSettled {
                outcome: AgentSteerTurnOutcome::Dropped,
                drained: matches!(state, TurnInputState::Drained),
            },
        }
    }

    /// 等待 Agent 接受输入，或轮次先行结算。
    pub async fn wait_for_accepted(&mut self) -> TurnInputState {
        loop {
            let state = self.state();
            if !matches!(state, TurnInputState::Pending) {
                return state;
            }
            if self.state.changed().await.is_err() {
                return self.synthesize_closed_terminal();
            }
        }
    }

    /// 等待输入到达驱动器真实的排空边界。
    pub async fn wait_for_drained(&mut self) -> TurnInputState {
        loop {
            let state = self.state();
            if !matches!(state, TurnInputState::Pending | TurnInputState::Accepted) {
                return state;
            }
            if self.state.changed().await.is_err() {
                return self.synthesize_closed_terminal();
            }
        }
    }

    /// 等待所属轮次的类型化终态结果。
    pub async fn wait_for_turn_settled(&mut self) -> TurnInputState {
        loop {
            let state = self.state();
            if matches!(state, TurnInputState::TurnSettled { .. }) {
                return state;
            }
            if self.state.changed().await.is_err() {
                return self.synthesize_closed_terminal();
            }
        }
    }
}

impl TurnInputLifecycle {
    /// 将状态从 `Pending` 推进到 `Accepted`。
    fn mark_accepted(&self) {
        self.inner.state.send_if_modified(|state| {
            if matches!(state, TurnInputState::Pending) {
                *state = TurnInputState::Accepted;
                true
            } else {
                false
            }
        });
    }

    /// 将状态从 `Pending`/`Accepted` 推进到 `Drained`。
    fn mark_drained(&self) {
        self.inner.state.send_if_modified(|state| {
            if matches!(state, TurnInputState::Pending | TurnInputState::Accepted) {
                *state = TurnInputState::Drained;
                true
            } else {
                false
            }
        });
    }

    /// 以给定结果结算轮次；只有第一次调用生效。
    fn settle(&self, outcome: AgentSteerTurnOutcome) {
        if self
            .inner
            .settled
            .swap(true, std::sync::atomic::Ordering::AcqRel)
        {
            return;
        }
        let drained = matches!(*self.inner.state.borrow(), TurnInputState::Drained);
        let terminal = TurnInputState::TurnSettled { outcome, drained };
        self.inner.state.send_replace(terminal);
    }
}

impl Drop for TurnInputLifecycle {
    /// 最后一个句柄被丢弃时，若尚未结算则以 `Dropped` 结算。
    fn drop(&mut self) {
        if std::sync::Arc::strong_count(&self.inner) == 1 {
            self.settle(AgentSteerTurnOutcome::Dropped);
        }
    }
}

impl echo_core::agent::AgentInputLifecycle for TurnInputLifecycle {
    /// 委托给内部的 `mark_drained`。
    fn mark_drained(&self) {
        TurnInputLifecycle::mark_drained(self);
    }
}

/// 一次驱动轮次所接受的输入。
#[derive(Debug, Clone)]
pub enum TurnInput {
    /// 纯文本输入。
    Text(String),
    /// 结构化（可含多模态内容）的消息输入。
    Message(Message),
}

impl TurnRequest {
    /// 以文本输入创建请求，默认聊天模式。
    pub fn new(identity: EventIdentity, message: impl Into<String>) -> Self {
        Self {
            input: TurnInput::Text(message.into()),
            identity,
            mode: TurnMode::Chat,
            cancel: None,
            last_persisted_sequence: 0,
            invocation: None,
            input_publisher: None,
        }
    }

    /// 以结构化消息创建请求，默认聊天模式。
    pub fn from_message(identity: EventIdentity, message: Message) -> Self {
        Self {
            input: TurnInput::Message(message),
            identity,
            mode: TurnMode::Chat,
            cancel: None,
            last_persisted_sequence: 0,
            invocation: None,
            input_publisher: None,
        }
    }

    /// 设置流类型（聊天或执行）。
    pub fn mode(mut self, mode: TurnMode) -> Self {
        self.mode = mode;
        self
    }

    /// 设置协作式取消令牌。
    pub fn cancel(mut self, cancel: CancellationToken) -> Self {
        self.cancel = Some(cancel);
        self
    }

    /// 设置最后一个已持久化的信封序号，用于续接编号。
    pub fn last_persisted_sequence(mut self, sequence: u64) -> Self {
        self.last_persisted_sequence = sequence;
        self
    }

    /// 设置调用上下文元数据。
    pub fn invocation(mut self, invocation: AgentInvocationContext) -> Self {
        self.invocation = Some(invocation);
        self
    }

    /**
    为这条初始输入附加一个无损生命周期回执。

    返回的回执初始为 `Pending`；同一个驱动器会在真实执行路径上
    依次发布 `Accepted`、`Drained` 和 `TurnSettled`。
    */
    pub fn with_input_receipt(mut self) -> (Self, TurnInputReceipt) {
        let (receipt, lifecycle) = TurnInputReceipt::new(self.identity.turn_id.to_string());
        self.input_publisher = Some(lifecycle);
        (self, receipt)
    }
}

/**
一次驱动轮次的类型化终态。

信封传输层保证恰好有一个 Agent 终态事件；该值把这一事实带回给调用方，
从而绝不会从"流已返回"推断成功。
*/
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TurnOutcome {
    /// 轮次以最终答案正常完成。
    Completed,
    /// 轮次被取消。
    Cancelled,
    /// 轮次以类型化失败结束。
    Failed(AgentFailure),
}

/**
一次驱动轮次所发出信封的投递结果。

它有意与 [`TurnOutcome`] 正交。Agent 生产方负责执行结算，
而 sink 负责判定观察到的信封是否被日志、投影或观察者接受。
*/
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TurnDeliveryOutcome {
    /// 没有任何信封到达 sink。
    NotAttempted,
    /// 迄今观察到的每个信封都已被 sink 接受。
    Delivered,
    /// sink 显式停止接受信封。
    Closed,
    /// sink 在投递信封时拒绝或失败。
    Failed(AgentFailure),
}

impl TurnOutcome {
    /// 用于投影的稳定状态字符串（`completed`/`cancelled`/`failed`）。
    pub fn status(&self) -> &'static str {
        match self {
            Self::Completed => "completed",
            Self::Cancelled => "cancelled",
            Self::Failed(_) => "failed",
        }
    }

    /**
    若 [`AgentEvent`] 负载是终态，则将其映射为类型化结果。

    非终态事件返回 `None`。终态类型为取消的 `Error` 映射为 [`Self::Cancelled`]，
    与框架的类型化失败契约保持一致。自定义 sink 可用它以与驱动器相同的方式投影终态信封。
    */
    pub fn classify(event: &AgentEvent) -> Option<Self> {
        match event {
            AgentEvent::FinalAnswer(_) => Some(Self::Completed),
            AgentEvent::Cancelled => Some(Self::Cancelled),
            AgentEvent::Error { failure, .. } => {
                if failure.terminal_kind == echo_core::error::AgentTerminalKind::Cancelled {
                    Some(Self::Cancelled)
                } else {
                    Some(Self::Failed(failure.clone()))
                }
            }
            _ => None,
        }
    }
}

/// 一次驱动轮次的有界控制面回执。
#[derive(Debug, Clone)]
pub struct TurnReceipt {
    /// 来自请求身份的轮次标识。
    pub turn_id: TurnId,
    /// 类型化终态结果；总是由驱动器设置。
    pub outcome: TurnOutcome,
    /// 观察到的信封是否被事件 sink 接受。
    pub delivery: TurnDeliveryOutcome,
    /// 轮次以最终答案完成时的答案文本。
    pub final_answer: Option<String>,
    /// 驱动器观察到的最终答案信封所携带的消息标识。
    pub final_message_id: Option<MessageId>,
    /// 本轮累计的、由提供方上报的提示词 token 数。
    pub prompt_tokens: u64,
    /// 本轮累计的、由提供方上报的补全 token 数。
    pub completion_tokens: u64,
    /// 上报了用量的提供方调用次数。
    pub llm_calls: u64,
    /// Agent 发出的显式上下文压缩边界次数。
    pub compaction_count: u64,
    /** 驱动器观察到的最后一个信封序号。
    这并不意味着失败的 sink 也提交了相同的序号。 */
    pub last_event_sequence: u64,
    /// 轮次的挂钟耗时。
    pub elapsed: Duration,
}

impl TurnReceipt {
    /// 返回该已完成轮次所携带的稳定用量事实。
    pub fn usage(&self) -> ExecutionUsage {
        ExecutionUsage {
            duration_ms: Some(u64::try_from(self.elapsed.as_millis()).unwrap_or(u64::MAX)),
            tokens_used: (self.llm_calls > 0)
                .then(|| self.prompt_tokens.saturating_add(self.completion_tokens)),
            iterations: None,
        }
    }

    /// 用于投影的稳定状态字符串（`completed`/`cancelled`/`failed`）。
    pub fn status(&self) -> &'static str {
        self.outcome.status()
    }

    /** 当执行无法启动，或调用方需要在流存在之前报告失败时，
    构造一个类型化的失败回执。 */
    pub fn failed(
        turn_id: impl Into<String>,
        failure: AgentFailure,
    ) -> echo_core::error::Result<Self> {
        Ok(Self {
            turn_id: TurnId::new(turn_id)?,
            outcome: TurnOutcome::Failed(failure),
            delivery: TurnDeliveryOutcome::NotAttempted,
            final_answer: None,
            final_message_id: None,
            prompt_tokens: 0,
            completion_tokens: 0,
            llm_calls: 0,
            compaction_count: 0,
            last_event_sequence: 0,
            elapsed: Duration::ZERO,
        })
    }

    /** 当执行在流产生正常终态事件之前被停止时，
    构造一个类型化的取消回执。 */
    pub fn cancelled(turn_id: impl Into<String>) -> echo_core::error::Result<Self> {
        Ok(Self {
            turn_id: TurnId::new(turn_id)?,
            outcome: TurnOutcome::Cancelled,
            delivery: TurnDeliveryOutcome::NotAttempted,
            final_answer: None,
            final_message_id: None,
            prompt_tokens: 0,
            completion_tokens: 0,
            llm_calls: 0,
            compaction_count: 0,
            last_event_sequence: 0,
            elapsed: Duration::ZERO,
        })
    }

    /// 构造不含最终答案与用量的失败回执。
    fn failure_receipt(
        turn_id: TurnId,
        failure: AgentFailure,
        delivery: TurnDeliveryOutcome,
        last_event_sequence: u64,
        started: Instant,
    ) -> Self {
        Self::terminal_receipt(
            turn_id,
            TurnOutcome::Failed(failure),
            delivery,
            last_event_sequence,
            started,
        )
    }

    /// 以给定终态和投递结果构造不含最终答案与用量的回执。
    fn terminal_receipt(
        turn_id: TurnId,
        outcome: TurnOutcome,
        delivery: TurnDeliveryOutcome,
        last_event_sequence: u64,
        started: Instant,
    ) -> Self {
        Self {
            turn_id,
            outcome,
            delivery,
            final_answer: None,
            final_message_id: None,
            prompt_tokens: 0,
            completion_tokens: 0,
            llm_calls: 0,
            compaction_count: 0,
            last_event_sequence,
            elapsed: started.elapsed(),
        }
    }
}

/**
sink 处理一个信封后返回的控制信号。

`on_event` 会按顺序收到每个信封，包括终态信封。
`Closed` 表示消费者有意断开；`Err` 表示投递或持久化失败，
它与驱动器已观察到的生产方终态分开记录。
*/
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SinkControl {
    /// 继续接收后续信封。
    Continue,
    /// 消费者有意断开，驱动器将取消本轮。
    Closed,
}

/// 单个驱动轮次的逐消费者事件处理器。
#[async_trait]
pub trait EventSink: Send + Sync {
    /** 在驱动器完成统计之后取得一个信封的所有权，
    使产品适配器能够无损地保留或转发它。 */
    async fn on_event(&self, envelope: EventEnvelope) -> echo_core::error::Result<SinkControl>;
}

/// 取调用上下文（缺省则用默认值），并写入给定的取消令牌。
fn invocation_with_cancel(
    invocation: Option<AgentInvocationContext>,
    token: &CancellationToken,
) -> AgentInvocationContext {
    let mut invocation = invocation.unwrap_or_default();
    invocation.cancel = Some(token.clone());
    invocation
}

/**
通过一个 sink 驱动一次有限的 Agent 调用。

契约见模块文档；驱动器是无状态的，可复用于任意多个轮次。
*/
#[derive(Debug, Default)]
pub struct AgentTurnDriver;

impl AgentTurnDriver {
    /// 在 `agent` 上驱动 `request`，并把信封转发给 `sink`。
    pub async fn drive(
        &self,
        agent: &dyn Agent,
        mut request: TurnRequest,
        sink: &dyn EventSink,
    ) -> TurnReceipt {
        let started = Instant::now();
        let turn_id = request.identity.turn_id.clone();
        let input_lifecycle = request.input_publisher.take();
        if let Err(error) = request.identity.validate() {
            if let Some(lifecycle) = input_lifecycle.as_ref() {
                lifecycle.settle(AgentSteerTurnOutcome::Failed);
            }
            return TurnReceipt::failure_receipt(
                turn_id,
                AgentFailure::from(&error),
                TurnDeliveryOutcome::NotAttempted,
                request.last_persisted_sequence,
                started,
            );
        }
        if let Some(lifecycle) = input_lifecycle.as_ref() {
            lifecycle.mark_accepted();
        }
        let token = request
            .cancel
            .clone()
            .or_else(|| {
                request
                    .invocation
                    .as_ref()
                    .and_then(|invocation| invocation.cancel.clone())
            })
            .unwrap_or_default();
        let input_lifecycle_for_agent = input_lifecycle.as_ref().map(|lifecycle| {
            std::sync::Arc::new(lifecycle.clone())
                as std::sync::Arc<dyn echo_core::agent::AgentInputLifecycle>
        });
        let invocation = request
            .invocation
            .map(|mut invocation| {
                invocation.cancel = Some(token.clone());
                invocation.input_lifecycle = input_lifecycle_for_agent.clone();
                invocation
            })
            .or_else(|| {
                input_lifecycle_for_agent.map(|input_lifecycle| {
                    echo_core::agent::AgentInvocationContext {
                        cancel: Some(token.clone()),
                        input_lifecycle: Some(input_lifecycle),
                        ..echo_core::agent::AgentInvocationContext::default()
                    }
                })
            });

        let input = request.input;
        let raw = match (&input, request.mode, invocation) {
            (TurnInput::Text(message), TurnMode::Chat, None) => {
                agent.chat_stream_with_cancel(message, token.clone()).await
            }
            (TurnInput::Text(message), TurnMode::Execute, None) => {
                agent
                    .execute_stream_with_cancel(message, token.clone())
                    .await
            }
            (TurnInput::Text(message), TurnMode::Execute, Some(invocation)) => {
                agent
                    .execute_stream_with_invocation_context(message, token.clone(), invocation)
                    .await
            }
            (TurnInput::Text(message), TurnMode::Chat, Some(invocation)) => {
                agent
                    .chat_stream_message_with_invocation_context(
                        Message::user(message.clone()),
                        token.clone(),
                        invocation,
                    )
                    .await
            }
            (TurnInput::Message(message), TurnMode::Chat, invocation) => {
                agent
                    .chat_stream_message_with_invocation_context(
                        message.clone(),
                        token.clone(),
                        invocation_with_cancel(invocation, &token),
                    )
                    .await
            }
            (TurnInput::Message(message), TurnMode::Execute, invocation) => {
                agent
                    .execute_stream_message_with_invocation_context(
                        message.clone(),
                        token.clone(),
                        invocation_with_cancel(invocation, &token),
                    )
                    .await
            }
        };
        let raw = match raw {
            Ok(stream) => stream,
            Err(error) => {
                let failure = AgentFailure::from(&error);
                let cancelled =
                    failure.terminal_kind == echo_core::error::AgentTerminalKind::Cancelled;
                if let Some(lifecycle) = input_lifecycle.as_ref() {
                    lifecycle.settle(if cancelled {
                        AgentSteerTurnOutcome::Cancelled
                    } else {
                        AgentSteerTurnOutcome::Failed
                    });
                }
                let mut delivery = TurnDeliveryOutcome::NotAttempted;
                let next_sequence = request.last_persisted_sequence.checked_add(1);
                let mut last_event_sequence = request.last_persisted_sequence;
                if let Some(sequence) = next_sequence
                    && let Ok(envelope) = EventEnvelope::new(
                        &request.identity,
                        sequence,
                        request.identity.parent_event_id.clone(),
                        AgentEvent::from_error("turn_driver", &error),
                    )
                {
                    last_event_sequence = envelope.sequence;
                    match sink.on_event(envelope).await {
                        Ok(SinkControl::Continue) => {
                            delivery = TurnDeliveryOutcome::Delivered;
                        }
                        Ok(SinkControl::Closed) => {
                            delivery = TurnDeliveryOutcome::Closed;
                        }
                        Err(sink_error) => {
                            delivery = TurnDeliveryOutcome::Failed(AgentFailure::from(&sink_error));
                        }
                    }
                }
                return TurnReceipt::terminal_receipt(
                    turn_id,
                    if cancelled {
                        TurnOutcome::Cancelled
                    } else {
                        TurnOutcome::Failed(failure)
                    },
                    delivery,
                    last_event_sequence,
                    started,
                );
            }
        };
        let mut outcome: Option<TurnOutcome> = None;
        let mut delivery = TurnDeliveryOutcome::NotAttempted;
        let mut final_answer: Option<String> = None;
        let mut final_message_id: Option<MessageId> = None;
        let mut prompt_tokens: u64 = 0;
        let mut completion_tokens: u64 = 0;
        let mut llm_calls: u64 = 0;
        let mut compaction_count: u64 = 0;
        let mut last_event_sequence = request.last_persisted_sequence;
        let mut stream =
            envelope_event_stream_after(raw, request.identity, request.last_persisted_sequence);
        while let Some(item) = stream.next().await {
            let envelope = match item {
                Ok(envelope) => envelope,
                Err(error) => {
                    /* 传输层只会在第一个事件之前出错（身份无效）；
                    将其映射为类型化失败。 */
                    outcome = Some(TurnOutcome::Failed(AgentFailure::from(&error)));
                    break;
                }
            };
            last_event_sequence = envelope.sequence;
            match &envelope.payload {
                AgentEvent::LlmUsage {
                    prompt_tokens: prompt,
                    completion_tokens: completion,
                    usage_reported,
                    ..
                } if *usage_reported => {
                    llm_calls = llm_calls.saturating_add(1);
                    prompt_tokens =
                        prompt_tokens.saturating_add(u64::try_from(*prompt).unwrap_or(u64::MAX));
                    completion_tokens = completion_tokens
                        .saturating_add(u64::try_from(*completion).unwrap_or(u64::MAX));
                }
                AgentEvent::FinalAnswer(answer) => {
                    if outcome.is_none() {
                        outcome = Some(TurnOutcome::Completed);
                        final_answer = Some(answer.clone());
                        final_message_id = envelope.message_id.clone();
                    }
                }
                AgentEvent::ContextCompressed { .. } => {
                    compaction_count = compaction_count.saturating_add(1);
                }
                AgentEvent::Cancelled => {
                    outcome.get_or_insert(TurnOutcome::Cancelled);
                }
                AgentEvent::Error { failure, .. } => {
                    outcome.get_or_insert(
                        if failure.terminal_kind == echo_core::error::AgentTerminalKind::Cancelled {
                            TurnOutcome::Cancelled
                        } else {
                            TurnOutcome::Failed(failure.clone())
                        },
                    );
                }
                _ => {}
            }
            match sink.on_event(envelope).await {
                Ok(SinkControl::Continue) => {
                    delivery = TurnDeliveryOutcome::Delivered;
                }
                Ok(SinkControl::Closed) => {
                    delivery = TurnDeliveryOutcome::Closed;
                    token.cancel();
                    if outcome.is_none() {
                        outcome = Some(TurnOutcome::Cancelled);
                    }
                    break;
                }
                Err(error) => {
                    delivery = TurnDeliveryOutcome::Failed(AgentFailure::from(&error));
                    token.cancel();
                    if outcome.is_none() {
                        outcome = Some(TurnOutcome::Failed(AgentFailure::from(&error)));
                        final_answer = None;
                        final_message_id = None;
                    }
                    break;
                }
            }
        }

        let outcome = outcome.unwrap_or_else(|| {
            TurnOutcome::Failed(AgentFailure::from(&ReactError::Other(
                "turn stream ended without a terminal event".to_string(),
            )))
        });
        if let Some(lifecycle) = input_lifecycle.as_ref() {
            lifecycle.settle(match &outcome {
                TurnOutcome::Completed => AgentSteerTurnOutcome::Completed,
                TurnOutcome::Cancelled => AgentSteerTurnOutcome::Cancelled,
                TurnOutcome::Failed(_) => AgentSteerTurnOutcome::Failed,
            });
        }
        TurnReceipt {
            turn_id,
            outcome,
            delivery,
            final_answer,
            final_message_id,
            prompt_tokens,
            completion_tokens,
            llm_calls,
            compaction_count,
            last_event_sequence,
            elapsed: started.elapsed(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use echo_core::agent::AgentEvent;
    use futures::future::BoxFuture;
    use futures::stream::BoxStream;
    use std::sync::Arc;
    use std::sync::Mutex;
    use std::sync::atomic::Ordering;
    use std::time::Duration as StdDuration;

    /** 重放脚本化事件的 Agent，可选地加入延迟，以便在流中途观察取消。
    由于 `AgentEvent` 不是 `Clone`，每个流都会重新构建事件。 */
    struct ScriptedAgent {
        /// Agent 名称。
        name: &'static str,
        /// 每次建流时调用以生成事件序列。
        script: fn() -> Vec<AgentEvent>,
        /// 每个事件之前的可选延迟。
        delay: Option<StdDuration>,
    }

    impl ScriptedAgent {
        /// 以给定脚本创建无延迟的 Agent。
        fn new(script: fn() -> Vec<AgentEvent>) -> Self {
            Self {
                name: "scripted",
                script,
                delay: None,
            }
        }

        /// 设置每个事件之前的延迟。
        fn with_delay(mut self, delay: StdDuration) -> Self {
            self.delay = Some(delay);
            self
        }
    }

    impl Agent for ScriptedAgent {
        fn name(&self) -> &str {
            self.name
        }

        fn model_name(&self) -> &str {
            "test-model"
        }

        fn system_prompt(&self) -> &str {
            ""
        }

        fn execute<'a>(
            &'a self,
            _task: &'a str,
        ) -> BoxFuture<'a, echo_core::error::Result<String>> {
            Box::pin(async { Ok(String::new()) })
        }

        fn execute_stream<'a>(
            &'a self,
            _task: &'a str,
        ) -> BoxFuture<
            'a,
            echo_core::error::Result<BoxStream<'a, echo_core::error::Result<AgentEvent>>>,
        > {
            let script = self.script;
            let delay = self.delay;
            Box::pin(async move {
                let stream = futures::stream::iter((script)().into_iter().map(Ok));
                match delay {
                    None => Ok(stream.boxed()),
                    Some(delay) => Ok(stream
                        .then(move |item| async move {
                            tokio::time::sleep(delay).await;
                            item
                        })
                        .boxed()),
                }
            })
        }
    }

    /// 在调用上下文中立即标记输入已排空的测试 Agent。
    struct ImmediateDrainAgent;

    impl Agent for ImmediateDrainAgent {
        fn name(&self) -> &str {
            "immediate-drain"
        }

        fn model_name(&self) -> &str {
            "test-model"
        }

        fn system_prompt(&self) -> &str {
            ""
        }

        fn execute<'a>(
            &'a self,
            _task: &'a str,
        ) -> BoxFuture<'a, echo_core::error::Result<String>> {
            Box::pin(async { Ok(String::new()) })
        }

        fn execute_stream<'a>(
            &'a self,
            _task: &'a str,
        ) -> BoxFuture<
            'a,
            echo_core::error::Result<BoxStream<'a, echo_core::error::Result<AgentEvent>>>,
        > {
            Box::pin(async {
                Ok(
                    futures::stream::iter([Ok(AgentEvent::FinalAnswer("done".to_string()))])
                        .boxed(),
                )
            })
        }

        fn chat_stream_message_with_invocation_context<'a>(
            &'a self,
            _message: Message,
            _cancel: CancellationToken,
            invocation: AgentInvocationContext,
        ) -> BoxFuture<
            'a,
            echo_core::error::Result<BoxStream<'a, echo_core::error::Result<AgentEvent>>>,
        > {
            if let Some(lifecycle) = invocation.input_lifecycle {
                lifecycle.mark_drained();
            }
            Box::pin(async {
                Ok(
                    futures::stream::iter([Ok(AgentEvent::FinalAnswer("done".to_string()))])
                        .boxed(),
                )
            })
        }
    }

    /// 流永不就绪的测试 Agent。
    struct PendingAgent;

    impl Agent for PendingAgent {
        fn name(&self) -> &str {
            "pending"
        }

        fn model_name(&self) -> &str {
            "test-model"
        }

        fn system_prompt(&self) -> &str {
            ""
        }

        fn execute<'a>(
            &'a self,
            _task: &'a str,
        ) -> BoxFuture<'a, echo_core::error::Result<String>> {
            Box::pin(std::future::pending())
        }

        fn execute_stream<'a>(
            &'a self,
            _task: &'a str,
        ) -> BoxFuture<
            'a,
            echo_core::error::Result<BoxStream<'a, echo_core::error::Result<AgentEvent>>>,
        > {
            Box::pin(std::future::pending())
        }

        fn chat_stream_message_with_invocation_context<'a>(
            &'a self,
            _message: Message,
            _cancel: CancellationToken,
            _invocation: AgentInvocationContext,
        ) -> BoxFuture<
            'a,
            echo_core::error::Result<BoxStream<'a, echo_core::error::Result<AgentEvent>>>,
        > {
            Box::pin(std::future::pending())
        }
    }

    /// 记录信封序号、可在指定数量后关闭的测试 sink。
    #[derive(Default)]
    struct RecordingSink {
        /// 已收到信封的序号。
        events: Mutex<Vec<u64>>,
        /// 收到该数量的信封后返回 `Closed`。
        close_after: Option<usize>,
    }

    #[async_trait]
    impl EventSink for RecordingSink {
        async fn on_event(&self, envelope: EventEnvelope) -> echo_core::error::Result<SinkControl> {
            let mut events = self
                .events
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            events.push(envelope.sequence);
            Ok(match self.close_after {
                Some(limit) if events.len() >= limit => SinkControl::Closed,
                _ => SinkControl::Continue,
            })
        }
    }

    /// 按标签构造测试用事件身份。
    fn identity(label: &str) -> EventIdentity {
        EventIdentity::new(format!("stream-{label}"), format!("turn-{label}"))
            .expect("valid identity")
    }

    /// 构造一个已上报的用量事件。
    fn usage_event(prompt: usize, completion: usize) -> AgentEvent {
        AgentEvent::LlmUsage {
            model: "test-model".to_string(),
            prompt_tokens: prompt,
            completion_tokens: completion,
            total_tokens: prompt + completion,
            cached_prompt_tokens: 0,
            cache_creation_prompt_tokens: 0,
            usage_reported: true,
        }
    }

    /// Agent 在调用上下文中立即排空时，初始输入以 drained=true 结算。
    #[tokio::test]
    async fn tracked_initial_input_uses_immediate_agent_context_drain() {
        let (request, mut input) =
            TurnRequest::new(identity("initial-immediate"), "hello").with_input_receipt();
        assert_eq!(input.state(), TurnInputState::Pending);
        let receipt = AgentTurnDriver
            .drive(&ImmediateDrainAgent, request, &RecordingSink::default())
            .await;
        assert_eq!(receipt.outcome, TurnOutcome::Completed);
        assert_eq!(
            input.wait_for_turn_settled().await,
            TurnInputState::TurnSettled {
                outcome: AgentSteerTurnOutcome::Completed,
                drained: true,
            }
        );
    }

    /// Agent 不发布排空时，初始输入以 drained=false 结算。
    #[tokio::test]
    async fn tracked_initial_input_without_agent_publisher_keeps_no_drain() {
        let (request, mut input) = TurnRequest::new(identity("initial-generic"), "hello")
            .mode(TurnMode::Execute)
            .with_input_receipt();
        let receipt = AgentTurnDriver
            .drive(
                &ScriptedAgent::new(|| vec![AgentEvent::FinalAnswer("done".to_string())]),
                request,
                &RecordingSink::default(),
            )
            .await;
        assert_eq!(receipt.outcome, TurnOutcome::Completed);
        assert_eq!(
            input.wait_for_turn_settled().await,
            TurnInputState::TurnSettled {
                outcome: AgentSteerTurnOutcome::Completed,
                drained: false,
            }
        );
    }

    /// 调用 Agent 流之前即发布 Accepted；丢弃驱动 future 后以 Dropped 结算。
    #[tokio::test]
    async fn tracked_initial_input_publishes_accepted_before_stream_call() {
        let (request, mut input) =
            TurnRequest::new(identity("initial-accepted"), "hello").with_input_receipt();
        let sink = RecordingSink::default();
        let mut drive = Box::pin(AgentTurnDriver.drive(&PendingAgent, request, &sink));
        assert!(matches!(
            futures::poll!(drive.as_mut()),
            std::task::Poll::Pending
        ));
        assert_eq!(input.state(), TurnInputState::Accepted);
        drop(drive);
        assert_eq!(
            input.wait_for_turn_settled().await,
            TurnInputState::TurnSettled {
                outcome: AgentSteerTurnOutcome::Dropped,
                drained: false,
            }
        );
    }

    /// 取消终态以类型化的 Cancelled 结算且未排空。
    #[tokio::test]
    async fn tracked_initial_input_cancelled_terminal_is_typed_without_drain() {
        let (request, mut input) = TurnRequest::new(identity("initial-cancelled"), "hello")
            .mode(TurnMode::Execute)
            .with_input_receipt();
        let receipt = AgentTurnDriver
            .drive(
                &ScriptedAgent::new(|| vec![AgentEvent::Cancelled]),
                request,
                &RecordingSink::default(),
            )
            .await;
        assert_eq!(receipt.outcome, TurnOutcome::Cancelled);
        assert_eq!(
            input.wait_for_turn_settled().await,
            TurnInputState::TurnSettled {
                outcome: AgentSteerTurnOutcome::Cancelled,
                drained: false,
            }
        );
    }

    /// 流启动失败以类型化的 Failed 结算且未排空。
    #[tokio::test]
    async fn tracked_initial_input_start_failure_is_typed_without_drain() {
        let (request, mut input) = TurnRequest::new(identity("initial-failed"), "hello")
            .mode(TurnMode::Execute)
            .with_input_receipt();
        let receipt = AgentTurnDriver
            .drive(&FailingAgent, request, &RecordingSink::default())
            .await;
        assert!(matches!(receipt.outcome, TurnOutcome::Failed(_)));
        assert_eq!(
            input.wait_for_turn_settled().await,
            TurnInputState::TurnSettled {
                outcome: AgentSteerTurnOutcome::Failed,
                drained: false,
            }
        );
    }

    /// 终态信封投递失败时，输入仍按执行终态 Completed 结算。
    #[tokio::test]
    async fn tracked_initial_input_terminal_sink_failure_keeps_execution_terminal() {
        let (request, mut input) = TurnRequest::new(identity("initial-sink-failed"), "hello")
            .mode(TurnMode::Execute)
            .with_input_receipt();
        let receipt = AgentTurnDriver
            .drive(
                &ScriptedAgent::new(|| vec![AgentEvent::FinalAnswer("done".to_string())]),
                request,
                &FailingSink,
            )
            .await;
        assert_eq!(receipt.outcome, TurnOutcome::Completed);
        assert!(matches!(receipt.delivery, TurnDeliveryOutcome::Failed(_)));
        assert_eq!(
            input.wait_for_turn_settled().await,
            TurnInputState::TurnSettled {
                outcome: AgentSteerTurnOutcome::Completed,
                drained: false,
            }
        );
    }

    /// 正常完成的轮次映射最终答案，序号从 1 开始。
    #[tokio::test]
    async fn completed_turn_maps_final_answer_and_sequences_from_one() {
        let agent: Arc<dyn Agent> = Arc::new(ScriptedAgent::new(|| {
            vec![
                usage_event(10, 5),
                AgentEvent::FinalAnswer("done".to_string()),
            ]
        }));
        let sink = RecordingSink::default();
        let request = TurnRequest::new(identity("ok"), "hello");
        let receipt = AgentTurnDriver.drive(agent.as_ref(), request, &sink).await;
        assert_eq!(receipt.outcome, TurnOutcome::Completed);
        assert_eq!(receipt.delivery, TurnDeliveryOutcome::Delivered);
        assert_eq!(receipt.final_answer.as_deref(), Some("done"));
        assert_eq!(receipt.prompt_tokens, 10);
        assert_eq!(receipt.completion_tokens, 5);
        assert_eq!(receipt.llm_calls, 1);
        assert_eq!(receipt.last_event_sequence, 2);
        let events = sink
            .events
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        assert_eq!(*events, vec![1, 2]);
    }

    /// 执行模式下取消终态映射为 Cancelled。
    #[tokio::test]
    async fn execute_mode_maps_cancelled_terminal() {
        let agent: Arc<dyn Agent> = Arc::new(ScriptedAgent::new(|| vec![AgentEvent::Cancelled]));
        let sink = RecordingSink::default();
        let request = TurnRequest::new(identity("cancel"), "task").mode(TurnMode::Execute);
        let receipt = AgentTurnDriver.drive(agent.as_ref(), request, &sink).await;
        assert_eq!(receipt.outcome, TurnOutcome::Cancelled);
        assert_eq!(receipt.delivery, TurnDeliveryOutcome::Delivered);
        assert_eq!(receipt.status(), "cancelled");
        assert!(receipt.final_answer.is_none());
    }

    /// 错误终态映射为类型化失败。
    #[tokio::test]
    async fn error_terminal_maps_typed_failure() {
        let failure = AgentFailure::from(&ReactError::Other("boom".to_string()));
        let agent: Arc<dyn Agent> = Arc::new(ScriptedAgent::new(|| {
            vec![AgentEvent::Error {
                source: "test".to_string(),
                message: "boom".to_string(),
                failure: AgentFailure::from(&ReactError::Other("boom".to_string())),
            }]
        }));
        let request = TurnRequest::new(identity("fail"), "hello");
        let receipt = AgentTurnDriver
            .drive(agent.as_ref(), request, &RecordingSink::default())
            .await;
        assert_eq!(receipt.outcome, TurnOutcome::Failed(failure));
        assert_eq!(receipt.delivery, TurnDeliveryOutcome::Delivered);
        assert_eq!(receipt.status(), "failed");
    }

    /// 用量在多次提供方调用间累加。
    #[tokio::test]
    async fn usage_accumulates_across_provider_calls() {
        let agent: Arc<dyn Agent> = Arc::new(ScriptedAgent::new(|| {
            vec![
                usage_event(100, 40),
                usage_event(7, 3),
                AgentEvent::FinalAnswer("summed".to_string()),
            ]
        }));
        let request = TurnRequest::new(identity("usage"), "hello");
        let receipt = AgentTurnDriver
            .drive(agent.as_ref(), request, &RecordingSink::default())
            .await;
        assert_eq!(receipt.prompt_tokens, 107);
        assert_eq!(receipt.completion_tokens, 43);
        assert_eq!(receipt.llm_calls, 2);
    }

    /// 回执携带最终消息标识与压缩次数。
    #[tokio::test]
    async fn receipt_owns_final_message_and_compaction_facts() -> Result<(), String> {
        let agent: Arc<dyn Agent> = Arc::new(ScriptedAgent::new(|| {
            vec![
                AgentEvent::ContextCompressed {
                    before_count: 12,
                    after_count: 5,
                    before_tokens: 1_200,
                    after_tokens: 500,
                },
                AgentEvent::ContextCompressed {
                    before_count: 8,
                    after_count: 4,
                    before_tokens: 800,
                    after_tokens: 400,
                },
                AgentEvent::FinalAnswer("finished".to_string()),
            ]
        }));
        let identity = EventIdentity::for_chat(
            Some("conversation-receipt".to_string()),
            "turn-receipt",
            "message-receipt",
            None,
        )
        .map_err(|error| error.to_string())?;
        let receipt = AgentTurnDriver
            .drive(
                agent.as_ref(),
                TurnRequest::new(identity, "hello"),
                &RecordingSink::default(),
            )
            .await;

        assert_eq!(receipt.outcome, TurnOutcome::Completed);
        assert_eq!(receipt.compaction_count, 2);
        assert_eq!(
            receipt
                .final_message_id
                .as_ref()
                .map(echo_core::agent::MessageId::as_str),
            Some("message-receipt")
        );
        Ok(())
    }

    /// 终态投递失败时保留完成事实与最终答案。
    #[tokio::test]
    async fn failed_terminal_delivery_keeps_final_completion_facts() -> Result<(), String> {
        let agent: Arc<dyn Agent> = Arc::new(ScriptedAgent::new(|| {
            vec![AgentEvent::FinalAnswer("undelivered".to_string())]
        }));
        let identity = EventIdentity::for_chat(
            Some("conversation-delivery-failure".to_string()),
            "turn-delivery-failure",
            "message-delivery-failure",
            None,
        )
        .map_err(|error| error.to_string())?;
        let receipt = AgentTurnDriver
            .drive(
                agent.as_ref(),
                TurnRequest::new(identity, "hello"),
                &FailingSink,
            )
            .await;

        assert_eq!(receipt.outcome, TurnOutcome::Completed);
        assert!(matches!(receipt.delivery, TurnDeliveryOutcome::Failed(_)));
        assert_eq!(receipt.final_answer.as_deref(), Some("undelivered"));
        assert_eq!(
            receipt
                .final_message_id
                .as_ref()
                .map(echo_core::agent::MessageId::as_str),
            Some("message-delivery-failure")
        );
        Ok(())
    }

    /// 取消终态投递失败时保留执行终态。
    #[tokio::test]
    async fn cancelled_terminal_delivery_keeps_execution_terminal() {
        let agent: Arc<dyn Agent> = Arc::new(ScriptedAgent::new(|| vec![AgentEvent::Cancelled]));
        let receipt = AgentTurnDriver
            .drive(
                agent.as_ref(),
                TurnRequest::new(identity("cancel-delivery-failure"), "hello"),
                &FailingSink,
            )
            .await;
        assert_eq!(receipt.outcome, TurnOutcome::Cancelled);
        assert!(matches!(receipt.delivery, TurnDeliveryOutcome::Failed(_)));
    }

    /// 生产方错误的投递失败时保留原始失败。
    #[tokio::test]
    async fn producer_error_delivery_failure_keeps_original_failure() {
        let producer_failure = AgentFailure::from(&ReactError::Other("producer boom".to_string()));
        let agent: Arc<dyn Agent> = Arc::new(ScriptedAgent::new(|| {
            vec![AgentEvent::Error {
                source: "producer".to_string(),
                message: "producer boom".to_string(),
                failure: AgentFailure::from(&ReactError::Other("producer boom".to_string())),
            }]
        }));
        let receipt = AgentTurnDriver
            .drive(
                agent.as_ref(),
                TurnRequest::new(identity("error-delivery-failure"), "hello"),
                &FailingSink,
            )
            .await;
        assert_eq!(receipt.outcome, TurnOutcome::Failed(producer_failure));
        assert!(matches!(receipt.delivery, TurnDeliveryOutcome::Failed(_)));
    }

    /// 续接编号从已持久化序号之后继续。
    #[tokio::test]
    async fn resumed_sequencing_continues_after_persisted_sequence() {
        let agent: Arc<dyn Agent> = Arc::new(ScriptedAgent::new(|| vec![AgentEvent::Cancelled]));
        let sink = RecordingSink::default();
        let request = TurnRequest::new(identity("resume"), "hello").last_persisted_sequence(41);
        let receipt = AgentTurnDriver.drive(agent.as_ref(), request, &sink).await;
        assert_eq!(receipt.last_event_sequence, 42);
        let events = sink
            .events
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        assert_eq!(*events, vec![42]);
    }

    /// sink 关闭会取消轮次并产出 Cancelled 终态。
    #[tokio::test]
    async fn closed_sink_cancels_and_yields_cancelled_terminal() {
        let agent: Arc<dyn Agent> = Arc::new(
            ScriptedAgent::new(|| {
                (0..5)
                    .map(|index| AgentEvent::Token(format!("t{index}")))
                    .chain(std::iter::once(AgentEvent::FinalAnswer("late".to_string())))
                    .collect()
            })
            .with_delay(StdDuration::from_millis(5)),
        );
        let sink = RecordingSink {
            close_after: Some(1),
            ..RecordingSink::default()
        };
        let request = TurnRequest::new(identity("closed"), "hello");
        let receipt = AgentTurnDriver.drive(agent.as_ref(), request, &sink).await;
        /* 令牌触发后，默认的取消包装器会产出 Cancelled，
        且流不再投递后续事件。 */
        assert_eq!(receipt.outcome, TurnOutcome::Cancelled);
        assert_eq!(receipt.delivery, TurnDeliveryOutcome::Closed);
    }

    /// sink 在终态时关闭不影响已完成的执行结果。
    #[tokio::test]
    async fn terminal_closed_sink_keeps_completed_execution() {
        let agent: Arc<dyn Agent> = Arc::new(ScriptedAgent::new(|| {
            vec![AgentEvent::FinalAnswer("done".to_string())]
        }));
        let sink = RecordingSink {
            close_after: Some(1),
            ..RecordingSink::default()
        };
        let receipt = AgentTurnDriver
            .drive(
                agent.as_ref(),
                TurnRequest::new(identity("terminal-closed"), "hello"),
                &sink,
            )
            .await;
        assert_eq!(receipt.outcome, TurnOutcome::Completed);
        assert_eq!(receipt.delivery, TurnDeliveryOutcome::Closed);
        assert_eq!(receipt.final_answer.as_deref(), Some("done"));
    }

    /// 流启动即失败的测试 Agent。
    struct FailingAgent;

    impl Agent for FailingAgent {
        fn name(&self) -> &str {
            "failing"
        }

        fn model_name(&self) -> &str {
            "test-model"
        }

        fn system_prompt(&self) -> &str {
            ""
        }

        fn execute<'a>(
            &'a self,
            _task: &'a str,
        ) -> BoxFuture<'a, echo_core::error::Result<String>> {
            Box::pin(async { Ok(String::new()) })
        }

        fn execute_stream<'a>(
            &'a self,
            _task: &'a str,
        ) -> BoxFuture<
            'a,
            echo_core::error::Result<BoxStream<'a, echo_core::error::Result<AgentEvent>>>,
        > {
            Box::pin(async {
                Err(echo_core::error::ReactError::Other(
                    "stream could not start".to_string(),
                ))
            })
        }
    }

    /// 流启动时即返回取消错误的测试 Agent。
    struct CancelledStartAgent;

    impl Agent for CancelledStartAgent {
        fn name(&self) -> &str {
            "cancelled-start"
        }

        fn model_name(&self) -> &str {
            "test-model"
        }

        fn system_prompt(&self) -> &str {
            ""
        }

        fn execute<'a>(
            &'a self,
            _task: &'a str,
        ) -> BoxFuture<'a, echo_core::error::Result<String>> {
            Box::pin(async { Ok(String::new()) })
        }

        fn execute_stream<'a>(
            &'a self,
            _task: &'a str,
        ) -> BoxFuture<
            'a,
            echo_core::error::Result<BoxStream<'a, echo_core::error::Result<AgentEvent>>>,
        > {
            Box::pin(async {
                Err(echo_core::error::AgentError::Cancelled(
                    "cancelled before stream start".to_string(),
                )
                .into())
            })
        }
    }

    /// 流启动失败映射为失败回执并投递一个错误信封。
    #[tokio::test]
    async fn stream_start_failure_maps_to_failed_receipt() {
        let agent: Arc<dyn Agent> = Arc::new(FailingAgent);
        let request = TurnRequest::new(identity("startfail"), "hello");
        let sink = RecordingSink::default();
        let receipt = AgentTurnDriver.drive(agent.as_ref(), request, &sink).await;
        assert_eq!(receipt.outcome.status(), "failed");
        assert_eq!(receipt.delivery, TurnDeliveryOutcome::Delivered);
        assert_eq!(receipt.last_event_sequence, 1);
        assert_eq!(
            *sink
                .events
                .lock()
                .unwrap_or_else(|error| error.into_inner()),
            vec![1]
        );
    }

    /// 流启动时取消映射为取消回执。
    #[tokio::test]
    async fn stream_start_cancellation_maps_to_cancelled_receipt() {
        let (request, mut input) = TurnRequest::new(identity("start-cancelled"), "hello")
            .mode(TurnMode::Execute)
            .with_input_receipt();
        let sink = RecordingSink::default();
        let receipt = AgentTurnDriver
            .drive(&CancelledStartAgent, request, &sink)
            .await;

        assert_eq!(receipt.outcome, TurnOutcome::Cancelled);
        assert_eq!(receipt.delivery, TurnDeliveryOutcome::Delivered);
        assert_eq!(receipt.last_event_sequence, 1);
        assert_eq!(
            input.wait_for_turn_settled().await,
            TurnInputState::TurnSettled {
                outcome: AgentSteerTurnOutcome::Cancelled,
                drained: false,
            }
        );
    }

    /// 流启动失败信封投递失败时保留原始失败。
    #[tokio::test]
    async fn stream_start_delivery_failure_keeps_original_failure() {
        let receipt = AgentTurnDriver
            .drive(
                &FailingAgent,
                TurnRequest::new(identity("start-delivery-failure"), "hello"),
                &FailingSink,
            )
            .await;
        assert!(matches!(receipt.outcome, TurnOutcome::Failed(_)));
        assert!(matches!(receipt.delivery, TurnDeliveryOutcome::Failed(_)));
        assert_eq!(receipt.last_event_sequence, 1);
    }

    /// 每次投递都失败的测试 sink。
    struct FailingSink;

    #[async_trait]
    impl EventSink for FailingSink {
        async fn on_event(
            &self,
            _envelope: EventEnvelope,
        ) -> echo_core::error::Result<SinkControl> {
            Err(ReactError::Other(
                "injected durable sink failure".to_string(),
            ))
        }
    }

    /// sink 失败被记为 Failed 而非 Cancelled。
    #[tokio::test]
    async fn sink_failure_is_failed_not_cancelled() {
        let agent: Arc<dyn Agent> = Arc::new(ScriptedAgent::new(|| {
            vec![
                AgentEvent::Token("partial".to_string()),
                AgentEvent::FinalAnswer("must not win".to_string()),
            ]
        }));
        let receipt = AgentTurnDriver
            .drive(
                agent.as_ref(),
                TurnRequest::new(identity("sink-failure"), "hello"),
                &FailingSink,
            )
            .await;
        assert!(matches!(receipt.outcome, TurnOutcome::Failed(_)));
        assert!(matches!(receipt.delivery, TurnDeliveryOutcome::Failed(_)));
        assert!(receipt.final_answer.is_none());
        assert_eq!(receipt.last_event_sequence, 1);
    }

    /// 记录调用上下文、消息文本和取消状态的测试 Agent。
    #[derive(Default)]
    struct InvocationRecordingAgent {
        working_dir: Mutex<Option<std::path::PathBuf>>,
        message_text: Mutex<Option<String>>,
        cancel_states: Mutex<Vec<(bool, bool)>>,
    }

    impl Agent for InvocationRecordingAgent {
        fn name(&self) -> &str {
            "invocation-recording"
        }

        fn model_name(&self) -> &str {
            "test-model"
        }

        fn system_prompt(&self) -> &str {
            ""
        }

        fn execute<'a>(
            &'a self,
            _task: &'a str,
        ) -> BoxFuture<'a, echo_core::error::Result<String>> {
            Box::pin(async { Ok(String::new()) })
        }

        fn execute_stream<'a>(
            &'a self,
            _task: &'a str,
        ) -> BoxFuture<
            'a,
            echo_core::error::Result<BoxStream<'a, echo_core::error::Result<AgentEvent>>>,
        > {
            Box::pin(async {
                Err(ReactError::Other(
                    "text path should not be selected".to_string(),
                ))
            })
        }

        fn chat_stream_message_with_invocation_context<'a>(
            &'a self,
            message: Message,
            cancel: CancellationToken,
            invocation: AgentInvocationContext,
        ) -> BoxFuture<
            'a,
            echo_core::error::Result<BoxStream<'a, echo_core::error::Result<AgentEvent>>>,
        > {
            Box::pin(async move {
                *self
                    .message_text
                    .lock()
                    .unwrap_or_else(|error| error.into_inner()) = message.content.as_text();
                let invocation_cancelled = invocation
                    .cancel
                    .as_ref()
                    .is_some_and(CancellationToken::is_cancelled);
                self.cancel_states
                    .lock()
                    .unwrap_or_else(|error| error.into_inner())
                    .push((cancel.is_cancelled(), invocation_cancelled));
                *self
                    .working_dir
                    .lock()
                    .unwrap_or_else(|error| error.into_inner()) = invocation.working_dir;
                Ok(
                    futures::stream::iter([Ok(AgentEvent::FinalAnswer("message-ok".to_string()))])
                        .boxed(),
                )
            })
        }
    }

    /// 结构化聊天消息保留调用上下文。
    #[tokio::test]
    async fn structured_chat_message_preserves_invocation_context() {
        let agent = Arc::new(InvocationRecordingAgent::default());
        let expected = std::path::PathBuf::from("/tmp/eko-turn-driver-test");
        let request = TurnRequest::from_message(
            identity("message-invocation"),
            Message::user("hello".to_string()),
        )
        .invocation(AgentInvocationContext {
            working_dir: Some(expected.clone()),
            ..AgentInvocationContext::default()
        });
        let receipt = AgentTurnDriver
            .drive(agent.as_ref(), request, &RecordingSink::default())
            .await;
        assert_eq!(receipt.outcome, TurnOutcome::Completed);
        assert_eq!(receipt.final_answer.as_deref(), Some("message-ok"));
        assert_eq!(
            *agent
                .working_dir
                .lock()
                .unwrap_or_else(|error| error.into_inner()),
            Some(expected)
        );
    }

    /// 带调用上下文的文本聊天走结构化聊天契约。
    #[tokio::test]
    async fn text_chat_with_invocation_uses_structured_chat_contract() {
        let agent = Arc::new(InvocationRecordingAgent::default());
        let expected = std::path::PathBuf::from("/tmp/eko-text-chat-test");
        let request = TurnRequest::new(identity("text-invocation"), "text hello").invocation(
            AgentInvocationContext {
                working_dir: Some(expected.clone()),
                ..AgentInvocationContext::default()
            },
        );

        let receipt = AgentTurnDriver
            .drive(agent.as_ref(), request, &RecordingSink::default())
            .await;
        assert_eq!(receipt.outcome, TurnOutcome::Completed);
        assert_eq!(
            agent
                .message_text
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .as_deref(),
            Some("text hello")
        );
        assert_eq!(
            *agent
                .working_dir
                .lock()
                .unwrap_or_else(|error| error.into_inner()),
            Some(expected)
        );
    }

    /// 请求级取消令牌优先于调用上下文中的令牌；仅有调用上下文令牌时保留它。
    #[tokio::test]
    async fn request_cancel_precedes_invocation_cancel_and_invocation_only_is_retained() {
        let agent = Arc::new(InvocationRecordingAgent::default());
        let invocation_only = CancellationToken::new();
        invocation_only.cancel();
        let first = TurnRequest::new(identity("invocation-cancel"), "first").invocation(
            AgentInvocationContext {
                cancel: Some(invocation_only),
                ..AgentInvocationContext::default()
            },
        );
        AgentTurnDriver
            .drive(agent.as_ref(), first, &RecordingSink::default())
            .await;

        let request_cancel = CancellationToken::new();
        let shadowed_invocation_cancel = CancellationToken::new();
        shadowed_invocation_cancel.cancel();
        let second = TurnRequest::new(identity("request-cancel"), "second")
            .invocation(AgentInvocationContext {
                cancel: Some(shadowed_invocation_cancel),
                ..AgentInvocationContext::default()
            })
            .cancel(request_cancel);
        AgentTurnDriver
            .drive(agent.as_ref(), second, &RecordingSink::default())
            .await;

        assert_eq!(
            *agent
                .cancel_states
                .lock()
                .unwrap_or_else(|error| error.into_inner()),
            vec![(true, true), (false, false)]
        );
    }

    /// 每次投递前异步延迟的测试 sink。
    struct DelayedSink {
        events: tokio::sync::Mutex<Vec<(u64, bool)>>,
    }

    #[async_trait]
    impl EventSink for DelayedSink {
        async fn on_event(&self, envelope: EventEnvelope) -> echo_core::error::Result<SinkControl> {
            tokio::time::sleep(StdDuration::from_millis(5)).await;
            self.events.lock().await.push((
                envelope.sequence,
                TurnOutcome::classify(&envelope.payload).is_some(),
            ));
            Ok(SinkControl::Continue)
        }
    }

    /// 异步 sink 保持顺序且不阻塞运行时推进。
    #[tokio::test(flavor = "current_thread")]
    async fn async_sink_preserves_order_and_allows_runtime_progress() {
        let agent: Arc<dyn Agent> = Arc::new(ScriptedAgent::new(|| {
            vec![
                AgentEvent::Token("one".to_string()),
                AgentEvent::Token("two".to_string()),
                AgentEvent::FinalAnswer("done".to_string()),
            ]
        }));
        let sink = DelayedSink {
            events: tokio::sync::Mutex::new(Vec::new()),
        };
        let runtime_progressed = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let progressed = Arc::clone(&runtime_progressed);
        let heartbeat = tokio::spawn(async move {
            tokio::time::sleep(StdDuration::from_millis(1)).await;
            progressed.store(true, Ordering::SeqCst);
        });

        let receipt = AgentTurnDriver
            .drive(
                agent.as_ref(),
                TurnRequest::new(identity("async-sink"), "hello"),
                &sink,
            )
            .await;
        assert!(heartbeat.await.is_ok());
        assert!(runtime_progressed.load(Ordering::SeqCst));
        assert_eq!(receipt.outcome, TurnOutcome::Completed);
        assert_eq!(
            *sink.events.lock().await,
            vec![(1, false), (2, false), (3, true)]
        );
    }

    /// 缺少终态事件时得到失败回执。
    #[tokio::test]
    async fn missing_terminal_is_a_failed_receipt() {
        let agent: Arc<dyn Agent> = Arc::new(ScriptedAgent::new(|| {
            vec![AgentEvent::Token("partial".to_string())]
        }));
        let request = TurnRequest::new(identity("noterminal"), "hello");
        let receipt = AgentTurnDriver
            .drive(agent.as_ref(), request, &RecordingSink::default())
            .await;
        assert_eq!(receipt.outcome.status(), "failed");
        assert_eq!(receipt.delivery, TurnDeliveryOutcome::Delivered);
    }

    /// sink 在回执统计完成后取得终态信封，且恰好一个终态。
    #[tokio::test]
    async fn sink_takes_terminal_envelope_after_receipt_accounting() {
        /// 取得并保存信封所有权的测试 sink。
        #[derive(Default)]
        struct OwningSink {
            envelopes: Mutex<Vec<EventEnvelope>>,
        }
        #[async_trait]
        impl EventSink for OwningSink {
            async fn on_event(
                &self,
                envelope: EventEnvelope,
            ) -> echo_core::error::Result<SinkControl> {
                self.envelopes
                    .lock()
                    .unwrap_or_else(|error| error.into_inner())
                    .push(envelope);
                Ok(SinkControl::Continue)
            }
        }
        let agent: Arc<dyn Agent> = Arc::new(ScriptedAgent::new(|| {
            vec![
                usage_event(7, 3),
                AgentEvent::Token("t".to_string()),
                AgentEvent::FinalAnswer("fin".to_string()),
            ]
        }));
        let sink = OwningSink::default();
        let request = TurnRequest::new(identity("exactly"), "hello");
        let receipt = AgentTurnDriver.drive(agent.as_ref(), request, &sink).await;
        assert_eq!(receipt.outcome, TurnOutcome::Completed);
        assert_eq!(receipt.final_answer.as_deref(), Some("fin"));
        assert_eq!(receipt.prompt_tokens, 7);
        assert_eq!(receipt.completion_tokens, 3);
        assert_eq!(receipt.llm_calls, 1);

        let envelopes = sink
            .envelopes
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let terminals: Vec<&EventEnvelope> = envelopes
            .iter()
            .filter(|envelope| TurnOutcome::classify(&envelope.payload).is_some())
            .collect();
        assert_eq!(terminals.len(), 1);
        assert!(matches!(
            terminals.first().map(|envelope| &envelope.payload),
            Some(AgentEvent::FinalAnswer(answer)) if answer == "fin"
        ));
    }
}
