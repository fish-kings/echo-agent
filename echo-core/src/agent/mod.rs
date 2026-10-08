//! Agent 核心 trait、事件与回调接口

/// 与具体产品无关的、基于键的执行准入控制。
pub mod admission;
/// 统一的 `AgentBuilder` trait。
pub mod builder;
/// Critic trait 及其内置实现。
mod critic;
/// Agent 事件信封（envelope）与带身份的事件流。
mod event_envelope;
/// Agent 工厂：基于配置创建 agent。
pub mod factory;
/// 用于影响 agent 行为的干预回调接口。
pub mod intervention;
/// 提示词模板管理系统。
pub mod prompt_template;
/// 共享的 agent 类型定义。
mod types;

pub use admission::{
    ExecutionAdmission, KeyedExecutionAdmission, KeyedExecutionAdmissionError, KeyedExecutionLease,
    KeyedExecutionRetirement,
};
pub use factory::{AgentFactory, AgentFactoryConfig};
pub use intervention::{CallbackBridge, InterventionCallback, InterventionResult};
pub use prompt_template::PromptTemplateManager;

pub use critic::{CompositeCritic, CompositeStrategy, Critic, StaticCritic, ThresholdCritic};
pub use event_envelope::{
    AGENT_EVENT_SCHEMA_VERSION, ConversationId, EventEnvelope, EventId, EventIdentity, ExecutionId,
    MessageId, RunId, StreamId, TurnId, envelope_event_stream, envelope_event_stream_after,
    validate_envelope_trajectory, validate_event_trajectory,
};
pub use types::{Critique, CritiqueOutput, critique_output_schema};

use crate::error::{ReactError, Result};
use crate::llm::ToolDefinition;
use crate::llm::types::Message;
use futures::future::BoxFuture;
use futures::stream::BoxStream;
use futures::stream::StreamExt as _;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::future::Future;
use std::pin::Pin;
pub use tokio_util::sync::CancellationToken;

/**
一次有限 Agent 执行的、与具体产品无关的用量事实。

特定 provider 的计数器仍保留在其原生响应上。这个值是主 Agent turn 与
被委派的 Subagent 执行共用的稳定结果面。
*/
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionUsage {
    /// 总的墙钟耗时（毫秒）。
    pub duration_ms: Option<u64>,
    /// provider 上报用量时的输入加输出 token 总数。
    pub tokens_used: Option<u64>,
    /// 当执行路径上报时的 ReAct 迭代次数。
    pub iterations: Option<u64>,
}

impl ExecutionUsage {
    /// 返回耗时（毫秒），缺失时默认为 0。
    pub fn duration_millis(&self) -> u64 {
        self.duration_ms.unwrap_or(0)
    }
}

/// 当调用方尝试引导一个活跃 agent turn 时返回的带类型失败。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgentSteerError {
    /// 该 agent 实现未暴露实时引导能力。
    Unsupported,
    /// 当前没有活跃的 turn。
    NoActiveTurn,
    /// 活跃 turn 与调用方给出的确切预期身份不匹配。
    TurnMismatch {
        /// 调用方预期的 turn 标识。
        expected: String,
        /// 实际活跃的 turn 标识。
        actual: String,
    },
    /// 该 turn 存在，但尚未到达一个安全的注入点。
    NotSteerable {
        /// 该 turn 的标识。
        turn_id: String,
    },
    /// 空指令永远不会被接受。
    EmptyInput,
    /// 无法访问引导状态。
    StateUnavailable,
}

impl std::fmt::Display for AgentSteerError {
    /// 输出引导错误的英文描述。
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unsupported => f.write_str("live steering is not supported by this agent"),
            Self::NoActiveTurn => f.write_str("no active turn to steer"),
            Self::TurnMismatch { expected, actual } => {
                write!(
                    f,
                    "active turn mismatch: expected {expected}, actual {actual}"
                )
            }
            Self::NotSteerable { turn_id } => write!(f, "turn {turn_id} is not steerable"),
            Self::EmptyInput => f.write_str("steer input is empty"),
            Self::StateUnavailable => f.write_str("turn steer state is unavailable"),
        }
    }
}

impl std::error::Error for AgentSteerError {}

/**
一次已被接受的引导输入的可观察生命周期阶段。

「已接受」只意味着活跃 turn 的邮箱拥有该输入。[`AgentSteerPhase::Drained`]
表示 ReAct 循环已经把它纳入模型上下文，而 [`AgentSteerPhase::TurnSettled`]
表示所属的根 turn 已到达其真正的终止边界。
*/
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentSteerPhase {
    /// 活跃 turn 的邮箱拥有该输入，但模型上下文尚未拥有。
    Accepted,
    /// 该输入已被插入活跃 turn 的模型上下文中。
    Drained,
    /// 根 turn 已到达一个终止边界。
    TurnSettled,
}

/// 拥有某条引导输入的根 turn 的终止结果。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentSteerTurnOutcome {
    /// 根 turn 正常完成。
    Completed,
    /// 取消信号到达了根 turn。
    Cancelled,
    /// provider、工具、校验或运行时处理失败。
    Failed,
    /**
    turn 的持有者在发布带类型的终态之前被丢弃或中止。这是一种终止状态，
    但不得被理解为成功。
    */
    Dropped,
}

impl AgentSteerTurnOutcome {
    /// 该终止结果的稳定小写线上标识符。
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Completed => "completed",
            Self::Cancelled => "cancelled",
            Self::Failed => "failed",
            Self::Dropped => "dropped",
        }
    }

    /// 解析一个稳定的小写线上标识符。
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "completed" => Some(Self::Completed),
            "cancelled" => Some(Self::Cancelled),
            "failed" => Some(Self::Failed),
            "dropped" => Some(Self::Dropped),
            _ => None,
        }
    }
}

/**
通用初始输入生命周期的生产者钩子。

turn driver 拥有接受与终止结算的所有权。若某个 Agent 实现确切知道其输入
进入模型上下文的时刻，可以通过此钩子发布 drain 边界。不掌握该信息的实现
则让回执保持 `drained = false`。
*/
pub trait AgentInputLifecycle: Send + Sync {
    /// 标记该输入已被纳入模型上下文（drained）。
    fn mark_drained(&self);
}

/// 针对一条引导输入观察到的当前持久消费边界。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentSteerState {
    /// 邮箱拥有该输入。
    Accepted,
    /// 活跃模型上下文拥有该输入。
    Drained,
    /// 根 turn 已终止。
    TurnSettled {
        /// 根 turn 的终止结果。
        outcome: AgentSteerTurnOutcome,
        /// 该输入是否在 turn 结算之前就已进入模型上下文。
        drained: bool,
    },
}

impl AgentSteerState {
    /// 返回粗粒度的生命周期阶段。
    pub fn phase(&self) -> AgentSteerPhase {
        match self {
            Self::Accepted => AgentSteerPhase::Accepted,
            Self::Drained => AgentSteerPhase::Drained,
            Self::TurnSettled { .. } => AgentSteerPhase::TurnSettled,
        }
    }

    /// 该输入是否在当前状态之前就已进入模型上下文。
    pub fn was_drained(&self) -> bool {
        matches!(
            self,
            Self::Drained | Self::TurnSettled { drained: true, .. }
        )
    }
}

/**
被某个活跃 Agent turn 接受的一条引导输入的可追踪回执。

该回执由 Agent 实现驱动。产品层适配器应保留其持久化输入直到
`state().was_drained()` 为 true；当需要根 turn 的终止语义时使用
`wait_for_turn_settled`。
*/
#[derive(Debug, Clone)]
pub struct AgentSteerReceipt {
    /// 该引导输入的稳定标识。
    steer_id: String,
    /// 接受该输入的根 turn 标识。
    turn_id: String,
    /// 观察生命周期状态变化的 watch 通道接收端。
    state: tokio::sync::watch::Receiver<AgentSteerState>,
    /// 发送端关闭后合成的终止状态缓存。
    closed_terminal: std::sync::Arc<std::sync::Mutex<Option<AgentSteerState>>>,
}

impl AgentSteerReceipt {
    /// 构造一个新的回执（框架内部使用）。
    #[doc(hidden)]
    pub fn new(
        steer_id: String,
        turn_id: String,
        state: tokio::sync::watch::Receiver<AgentSteerState>,
    ) -> Self {
        Self {
            steer_id,
            turn_id,
            state,
            closed_terminal: std::sync::Arc::new(std::sync::Mutex::new(None)),
        }
    }

    /// 该引导输入的稳定标识。
    pub fn steer_id(&self) -> &str {
        &self.steer_id
    }

    /// 接受该输入的根 turn 标识。
    pub fn turn_id(&self) -> &str {
        &self.turn_id
    }

    /// 在不等待的情况下，快照当前观察到的最新生命周期状态。
    pub fn state(&self) -> AgentSteerState {
        if let Some(state) = self.cached_closed_terminal() {
            return state;
        }
        if self.state.has_changed().is_err() {
            return self.synthesize_closed_terminal();
        }
        self.state.borrow().clone()
    }

    /// 读取已缓存的、发送端关闭后合成的终止状态（若已有）。
    fn cached_closed_terminal(&self) -> Option<AgentSteerState> {
        self.closed_terminal
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clone()
    }

    /// 当发送端已关闭时，基于最后一次观察到的状态合成一个终止状态并缓存。
    fn synthesize_closed_terminal(&self) -> AgentSteerState {
        let mut cached = self
            .closed_terminal
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if let Some(state) = cached.as_ref() {
            return state.clone();
        }
        let current = self.state.borrow().clone();
        let terminal = match current {
            state @ AgentSteerState::TurnSettled { .. } => state,
            state => AgentSteerState::TurnSettled {
                outcome: AgentSteerTurnOutcome::Dropped,
                drained: state.was_drained(),
            },
        };
        *cached = Some(terminal.clone());
        terminal
    }

    /// 等待直到该输入被 drain，或其 turn 先行结算。
    pub async fn wait_for_drained(&mut self) -> AgentSteerState {
        loop {
            let state = self.state();
            if !matches!(state, AgentSteerState::Accepted) {
                return state;
            }
            if self.state.changed().await.is_err() {
                return self.synthesize_closed_terminal();
            }
        }
    }

    /// 等待所属根 turn 真正的终止边界。
    pub async fn wait_for_turn_settled(&mut self) -> AgentSteerState {
        loop {
            let state = self.state();
            if matches!(state, AgentSteerState::TurnSettled { .. }) {
                return state;
            }
            if self.state.changed().await.is_err() {
                return self.synthesize_closed_terminal();
            }
        }
    }
}

#[cfg(test)]
mod steer_receipt_tests {
    use super::*;

    #[test]
    fn turn_outcome_wire_names_round_trip() {
        for outcome in [
            AgentSteerTurnOutcome::Completed,
            AgentSteerTurnOutcome::Cancelled,
            AgentSteerTurnOutcome::Failed,
            AgentSteerTurnOutcome::Dropped,
        ] {
            assert_eq!(
                AgentSteerTurnOutcome::parse(outcome.as_str()),
                Some(outcome)
            );
        }
        assert_eq!(AgentSteerTurnOutcome::parse("unknown"), None);
    }

    #[tokio::test]
    async fn closed_sender_terminal_is_shared_across_receipt_clones() {
        let (sender, receiver) = tokio::sync::watch::channel(AgentSteerState::Accepted);
        let mut first = AgentSteerReceipt::new(
            "steer-closed".to_string(),
            "turn-closed".to_string(),
            receiver,
        );
        let mut second = first.clone();
        drop(sender);

        let (first_state, second_state) = tokio::join!(
            first.wait_for_turn_settled(),
            second.wait_for_turn_settled()
        );
        let expected = AgentSteerState::TurnSettled {
            outcome: AgentSteerTurnOutcome::Dropped,
            drained: false,
        };
        assert_eq!(first_state, expected);
        assert_eq!(second_state, expected);
        assert_eq!(first.state(), expected);
        assert_eq!(second.state(), expected);
    }

    #[tokio::test]
    async fn closed_sender_preserves_prior_drain() {
        let (sender, receiver) = tokio::sync::watch::channel(AgentSteerState::Drained);
        let mut receipt = AgentSteerReceipt::new(
            "steer-drained".to_string(),
            "turn-drained".to_string(),
            receiver,
        );
        drop(sender);

        assert_eq!(
            receipt.wait_for_turn_settled().await,
            AgentSteerState::TurnSettled {
                outcome: AgentSteerTurnOutcome::Dropped,
                drained: true,
            }
        );
    }
}

/**
应用于一次 ReAct 调用的可选软预算。

`None` 字段保留既有行为。硬性的迭代次数上限仍由 `AgentConfig::max_iterations`
决定；本策略只额外增加一个显式的收尾（wind-down）时机，以及一个由
provider 上报的模型 token 上限。
*/
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RunBudgetPolicy {
    /// 当剩余迭代次数达到这个值时，注入一次性的收尾指令。
    pub iteration_wind_down_remaining: Option<usize>,
    /// 当 provider 上报的模型 token 数达到这个值后，进入仅输出最终答案的模式。
    pub max_model_tokens: Option<usize>,
}

/// run 预算控制器做出的可观察决策。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BudgetDecision {
    /// 正常的 ReAct 执行可以继续。
    Continue,
    /// run 应当收敛，但工具仍然可用。
    WindDown,
    /// 下一次模型调用必须只产出文本，不带工具。
    FinalOnly,
    /// 硬性预算已耗尽，run 必须失败。
    HardStop,
}

/// 修改了模型请求的工具调用的策略阶段。
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ToolInvocationRewrite {
    /// 由干预机制重定向到了另一个工具。
    InterventionRedirect,
    /// 由干预机制修改了参数。
    InterventionArguments,
    /// 由 PreToolUse hook 修改。
    PreToolUseHook,
    /// 由审批流程修改。
    Approval,
}

/// 一次工具执行的请求身份与生效身份。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolInvocation {
    /// 模型最初请求的工具名。
    pub requested_name: String,
    /// 模型最初请求的参数。
    pub requested_args: Value,
    /// 经过策略改写后，最终生效的工具名。
    pub name: String,
    /// 经过策略改写后，最终生效的参数。
    pub args: Value,
    /// 依次应用的改写来源列表。
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub rewrites: Vec<ToolInvocationRewrite>,
}

/// 一次流式 agent 调用的、限定值作用域的元数据。
#[derive(Clone, Default)]
pub struct AgentInvocationContext {
    /// 传播进本次调用的工具上下文中的 run 元数据。
    pub runtime: Option<crate::tools::ExternalRunContext>,
    /**
    与 `runtime` 携带的产品对话不同的可选运行时 checkpoint 身份。

    `None` 保留既有行为：运行时对话覆盖值同时也用于 checkpoint。
    持久化运行时可以把它设置为一个临时化身，同时让产品事件和 transcript
    保持在一个稳定的会话 ID 上。当同一个 Agent 实例收到不同的运行时状态
    身份时，框架会在准备模型输入之前重置或恢复该身份；只有相同身份时
    才会复用热上下文。
    */
    pub runtime_state_id: Option<String>,
    /**
    用于仅追加式 transcript 投影的可选模型上下文代次（generation）。

    设置后，框架会在 Agent 上跟踪已经投影过的前缀，只追加本代次中的
    新消息。这可以防止一个全新的模型上下文，被错误地按内容去重到
    某个恰好以相同文本结尾的旧产品 transcript 上。该值必须等于从显式
    `runtime_state_id`、运行时对话或已配置对话解析出的生效运行时状态
    身份。不匹配时，会在产生任何准入副作用或 checkpoint 持久化之前，
    以运行时状态错误的形式安全失败（fail closed）。
    */
    pub transcript_generation_id: Option<String>,
    /// 本次调用的工作目录。`None` 表示使用 agent 配置的默认值。
    pub working_dir: Option<std::path::PathBuf>,
    /// 在入队之前随本次调用一并捕获的取消令牌。
    pub cancel: Option<CancellationToken>,
    /**
    仅对本次调用隐藏的工具名集合。

    这些排除项会在创建 run 快照时与 agent 级别的默认值合并。
    它们绝不会修改共享的工具注册表。
    */
    pub disabled_tools: Option<std::collections::HashSet<String>>,
    /**
    对模型可见其 schema 的初始工具名集合。
    `None` 表示保持完整的合规工具集合全部可见。
    */
    pub visible_tools: Option<std::collections::HashSet<String>>,
    /// 本次调用的预算策略。`None` 表示使用 agent 的默认值。
    pub run_budget: Option<RunBudgetPolicy>,
    /**
    插入在 system prompt 之后、当前输入之前的结构化对话 turn。
    这是限定值作用域的，绝不会修改 agent 已配置的 system prompt。
    */
    pub history: Option<Vec<Message>>,
    /// 在本次调用所派生的 Agent、subagent 与工具工作期间保留的不透明资源。
    pub resource_guards: Vec<crate::tools::InvocationResourceGuard>,
    /**
    初始输入 drain 边界的可选通用生产者。

    这是仅供框架使用的生命周期管道。产品策略与持久回执的所有权
    仍归 turn driver 的调用方所有。
    */
    pub input_lifecycle: Option<std::sync::Arc<dyn AgentInputLifecycle>>,
}

impl std::fmt::Debug for AgentInvocationContext {
    /// 输出调用上下文的调试摘要，敏感或较大的字段仅显示数量或占位符。
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentInvocationContext")
            .field("runtime_state_id", &self.runtime_state_id)
            .field("transcript_generation_id", &self.transcript_generation_id)
            .field(
                "run_id",
                &self
                    .runtime
                    .as_ref()
                    .and_then(|runtime| runtime.run_id.as_deref()),
            )
            .field(
                "turn_id",
                &self
                    .runtime
                    .as_ref()
                    .and_then(|runtime| runtime.turn_id.as_deref()),
            )
            .field(
                "execution_id",
                &self
                    .runtime
                    .as_ref()
                    .and_then(|runtime| runtime.execution_id.as_deref()),
            )
            .field("working_dir", &self.working_dir)
            .field(
                "cancel",
                &self.cancel.as_ref().map(|_| "<CancellationToken>"),
            )
            .field(
                "disabled_tools",
                &self
                    .disabled_tools
                    .as_ref()
                    .map(std::collections::HashSet::len),
            )
            .field(
                "visible_tools",
                &self
                    .visible_tools
                    .as_ref()
                    .map(std::collections::HashSet::len),
            )
            .field("run_budget", &self.run_budget)
            .field("history_messages", &self.history.as_ref().map(Vec::len))
            .field("resource_guard_count", &self.resource_guards.len())
            .field("resource_guards", &self.resource_guards)
            .field("has_input_lifecycle", &self.input_lifecycle.is_some())
            .finish()
    }
}

/**
Agent 执行期间产生的事件

覆盖 Agent 生命周期的每个阶段，供进度条、日志、UI 更新等场景使用。
*/
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
#[non_exhaustive]
// 把 ToolStream 装箱会破坏所有消费者共用的公开事件契约。
#[allow(clippy::large_enum_variant)]
pub enum AgentEvent {
    // ── LLM 交互 ──────────────────────────────────────────────────────────
    /// LLM 正在生成一个 token（流式）
    Token(String),
    /// LLM 推理开始
    ThinkStart,
    /// LLM 推理结束
    ThinkEnd {
        /// 消耗的 prompt token 数
        prompt_tokens: usize,
        /// 消耗的 completion token 数
        completion_tokens: usize,
    },
    /**
    provider 针对单次请求上报的 LLM 用量。

    `ThinkEnd` 作为一个精简、向后兼容的 UI 事件保留下来。这个信息更丰富的
    事件携带缓存可观测性数据。当 `usage_reported` 为 false 时，说明
    provider 或网关没有返回用量元数据；调用方必须把缓存相关字段视为
    未知，而不是零命中率。
    */
    LlmUsage {
        /// 该请求使用的模型。
        model: String,
        /// provider 上报的 prompt/输入 token 数。
        prompt_tokens: usize,
        /// provider 上报的 completion/输出 token 数。
        completion_tokens: usize,
        /// provider 上报的总 token 数；缺失时为 prompt + completion。
        total_tokens: usize,
        /// 由 provider 侧缓存命中提供的 prompt/输入 token 数。
        cached_prompt_tokens: usize,
        /// 写入 provider 侧缓存的 prompt/输入 token 数。
        cache_creation_prompt_tokens: usize,
        /// provider 响应中是否包含用量元数据。
        usage_reported: bool,
    },
    /// 一次 run 预算越过了某个行为边界。
    BudgetDecision {
        /// harness 做出的决策。
        decision: BudgetDecision,
        /// 稳定的、机器可读的原因。
        reason: String,
        /// 当前迭代计数（从 1 开始）。
        iteration: usize,
        /// 本次调用中累计的、由 provider 上报的模型 token 数。
        reported_model_tokens: usize,
        /// 当至少有一次 provider 响应缺失用量元数据时为 false。
        usage_complete: bool,
    },

    // ── 工具调用 ──────────────────────────────────────────────────────────
    /// 经过策略改写之后、执行之前发出的规范化调用。
    ToolCall {
        /// 稳定的工具调用身份（模型的 tool_call_id，或生成的 UUID）。
        call_id: String,
        /// 请求/生效的调用信息及改写来源。
        invocation: ToolInvocation,
    },
    /// 工具执行已完成，无论成功还是失败。
    ToolResult {
        /// 与前面 [`Self::ToolCall`] 相匹配的稳定工具调用身份。
        call_id: String,
        /// 生效的工具名。
        name: String,
        /// 完整结果，包括带类型的失败信息、artifact 元数据与截断标记。
        result: crate::tools::ToolResult,
    },
    /// 流式工具进度/输出事件（不是终止生命周期事件）。
    ToolStream {
        /// 与前面 [`Self::ToolCall`] 相匹配的稳定工具调用身份。
        call_id: String,
        /// 工具名
        name: String,
        /**
        stream 事件负载（`Progress` / `Output`；`Complete` 会被 ReAct
        runner 映射为 [`Self::ToolResult`]）。
        */
        event: crate::tools::ToolStreamEvent,
    },
    /**
    在一批工具开始执行之前发出。
    ToolBatchStart 与 ToolBatchEnd 之间的所有工具都是并发执行的。
    */
    ToolBatchStart {
        /// 本批次中的工具数量
        tool_count: usize,
    },
    /// 在该批次中所有工具都执行完成后发出。
    ToolBatchEnd,

    // ── Guard 与安全 ──────────────────────────────────────────────────────
    /// 某个 guard 被触发
    GuardTriggered {
        /// guard 名称
        guard: String,
        /// 该动作是否被阻断
        blocked: bool,
    },

    // ── 记忆与编排 ──────────────────────────────────────────────────────────
    /// 已回忆出长期记忆
    MemoryRecalled {
        /// 被回忆出的记忆条目数量
        count: usize,
    },
    /// 上下文因超出 token 限制而被自动压缩
    ContextCompressed {
        /// 压缩前的消息数
        before_count: usize,
        /// 压缩后的消息数
        after_count: usize,
        /// 压缩前的估算 token 数
        before_tokens: usize,
        /// 压缩后的估算 token 数
        after_tokens: usize,
    },
    /// transcript 投影在终止发布之前，达到了一个带类型的持久化结算点。
    TranscriptProjectionSettlement(crate::memory::TranscriptProjectionSettlement),

    // ── 可视化 ────────────────────────────────────────────────────────────
    /// 图表生成（vega-lite JSON spec）
    Chart {
        /// vega-lite 图表 spec。
        spec: Value,
    },

    // ── 错误 ────────────────────────────────────────────────────────────
    /// 通用 Agent 错误（非工具错误，例如 LLM 调用失败、guard 拒绝等）
    Error {
        /// 错误来源（例如 "llm"、"guard"、"config"）
        source: String,
        /// 错误信息
        message: String,
        /// 供重试、生命周期与 UI 适配器使用的稳定分类。
        failure: crate::error::AgentFailure,
    },

    /**
    安全提示：agent 即将执行一个需要用户关注的动作。
    包含具体动作、原因、风险等级和所需权限。
    */
    SafetyNotice {
        /// agent 即将执行的动作。
        action: String,
        /// 为何需要该动作。
        reason: String,
        /// 风险描述。
        risk: String,
        /// 所需的权限等级。
        permission: String,
    },
    /// 发生了一次工具参数校验错误。
    ParameterError {
        /// 工具名。
        tool: String,
        /// 参数名。
        parameter: String,
        /// 期望的类型。
        expected: String,
        /// 实际收到的类型。
        got: String,
    },

    // ── 终止状态 ──────────────────────────────────────────────────────────────
    /// 最终答案
    FinalAnswer(String),
    /// 已取消
    Cancelled,
}

/**
一个 Agent 事件所属的生命周期阶段

把 `AgentEvent` 的每个变体映射进一个统一的阶段模型，用于：
- **状态持久化**：checkpoint 只在阶段边界处创建
- **前端渲染**：按阶段路由到对应的 UI 组件，而无需匹配每一个具体变体
- **开发者理解**：新人先理解阶段，再理解具体事件
*/
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum AgentPhase {
    /// LLM 推理进行中（Token → ThinkStart → ThinkEnd）
    Thinking,
    /// 工具执行进行中（ToolCall -> 带类型的 ToolResult）
    Acting,
    /// 已产出最终结果或已取消
    Terminal,
}

impl AgentEvent {
    /// 构造一个终止事件，同时保留带类型的框架失败信息。
    pub fn from_error(source: impl Into<String>, error: &ReactError) -> Self {
        let failure = crate::error::AgentFailure::from(error);
        Self::Error {
            source: source.into(),
            message: failure.message.clone(),
            failure,
        }
    }

    /// 为没有带类型错误的边界，构造一个未分类的终止失败。
    pub fn error_message(source: impl Into<String>, message: impl Into<String>) -> Self {
        let source = source.into();
        let failure = crate::error::AgentFailure::message(&source, message);
        Self::Error {
            source,
            message: failure.message.clone(),
            failure,
        }
    }

    /// 该事件是否结束本次调用的事件流。
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            AgentEvent::FinalAnswer(_) | AgentEvent::Cancelled | AgentEvent::Error { .. }
        )
    }

    /// 返回 `ThinkEnd` 的 prompt token 数。
    pub fn prompt_tokens(&self) -> Option<usize> {
        match self {
            AgentEvent::ThinkEnd { prompt_tokens, .. } => Some(*prompt_tokens),
            AgentEvent::LlmUsage { prompt_tokens, .. } => Some(*prompt_tokens),
            _ => None,
        }
    }

    /// 返回 `ThinkEnd` 的 completion token 数。
    pub fn completion_tokens(&self) -> Option<usize> {
        match self {
            AgentEvent::ThinkEnd {
                completion_tokens, ..
            } => Some(*completion_tokens),
            AgentEvent::LlmUsage {
                completion_tokens, ..
            } => Some(*completion_tokens),
            _ => None,
        }
    }

    /// 返回 `ThinkEnd` 的总 token 用量。
    pub fn total_tokens(&self) -> Option<usize> {
        match self {
            AgentEvent::ThinkEnd {
                prompt_tokens,
                completion_tokens,
            } => Some(prompt_tokens.saturating_add(*completion_tokens)),
            AgentEvent::LlmUsage { total_tokens, .. } => Some(*total_tokens),
            _ => None,
        }
    }

    /// 供只追踪单一 token 计数的旧调用点使用的兼容辅助方法。
    pub fn tokens_used(&self) -> Option<usize> {
        self.total_tokens()
    }

    /**
    返回该事件的生命周期阶段

    用于前端按阶段路由渲染、状态机推导等类似场景。

    # 示例

    ```
    use echo_core::agent::{AgentEvent, AgentPhase};

    let event = AgentEvent::ThinkStart;
    assert_eq!(event.phase(), AgentPhase::Thinking);

    let event = AgentEvent::FinalAnswer("done".into());
    assert_eq!(event.phase(), AgentPhase::Terminal);
    ```
    */
    pub fn phase(&self) -> AgentPhase {
        match self {
            AgentEvent::Token(_)
            | AgentEvent::ThinkStart
            | AgentEvent::ThinkEnd { .. }
            | AgentEvent::LlmUsage { .. }
            | AgentEvent::BudgetDecision { .. }
            | AgentEvent::MemoryRecalled { .. }
            | AgentEvent::ContextCompressed { .. }
            | AgentEvent::Chart { .. } => AgentPhase::Thinking,

            AgentEvent::ToolCall { .. }
            | AgentEvent::ToolResult { .. }
            | AgentEvent::ToolStream { .. }
            | AgentEvent::ToolBatchStart { .. }
            | AgentEvent::ToolBatchEnd
            | AgentEvent::GuardTriggered { .. }
            | AgentEvent::TranscriptProjectionSettlement(_)
            | AgentEvent::SafetyNotice { .. }
            | AgentEvent::ParameterError { .. } => AgentPhase::Acting,

            AgentEvent::FinalAnswer(_) | AgentEvent::Cancelled | AgentEvent::Error { .. } => {
                AgentPhase::Terminal
            }
        }
    }

    /**
    该事件是否是一个可持久化的快照点（阶段边界事件）

    当这些事件发生时，Agent 状态处于一个"稳定点"——没有正在进行的 LLM
    调用或工具执行，适合用来保存 checkpoint、从 checkpoint 恢复，
    或进行时间旅行式调试。

    # 示例

    ```
    use echo_core::agent::AgentEvent;

    assert!(AgentEvent::ThinkEnd { prompt_tokens: 100, completion_tokens: 50 }.is_checkpoint());
    assert!(AgentEvent::FinalAnswer("done".into()).is_checkpoint());
    assert!(!AgentEvent::Token("hello".into()).is_checkpoint());
    ```
    */
    pub fn is_checkpoint(&self) -> bool {
        matches!(
            self,
            AgentEvent::ThinkEnd { .. }
                | AgentEvent::ToolResult { .. }
                | AgentEvent::ParameterError { .. }
                | AgentEvent::ContextCompressed { .. }
                | AgentEvent::TranscriptProjectionSettlement(_)
                | AgentEvent::FinalAnswer(_)
                | AgentEvent::Cancelled
                | AgentEvent::Error { .. }
        )
    }
}

#[cfg(test)]
mod accounting_tests {
    use super::AgentEvent;

    #[test]
    fn total_tokens_saturates_without_panicking() {
        let event = AgentEvent::ThinkEnd {
            prompt_tokens: usize::MAX,
            completion_tokens: 1,
        };
        assert_eq!(event.total_tokens(), Some(usize::MAX));
        assert_eq!(event.tokens_used(), Some(usize::MAX));
    }

    #[test]
    fn transcript_projection_settlement_is_observable_checkpoint_not_terminal() {
        let event = AgentEvent::TranscriptProjectionSettlement(
            crate::memory::TranscriptProjectionSettlement {
                status: crate::memory::TranscriptProjectionSettlementStatus::Settled,
                operation_id: Some("operation".to_string()),
                conversation_id: Some("conversation".to_string()),
                generation_id: Some("generation".to_string()),
                attempt: 1,
                error_class: None,
                detail: None,
            },
        );

        assert!(!event.is_terminal());
        assert!(event.is_checkpoint());
        assert_eq!(event.phase(), super::AgentPhase::Acting);
    }
}

/// 从 LLM 响应中解析出的步骤类型
#[derive(Debug)]
pub enum StepType {
    /// 思考步骤（内部推理）
    Thought(String),
    /// 工具调用步骤
    Call {
        /// 工具调用 ID（唯一标识符）
        tool_call_id: String,
        /// 函数名
        function_name: String,
        /// 函数参数（JSON 格式）
        arguments: Value,
    },
}

/**
统一的 Agent 执行接口

所有方法都接受 `&self`，以便一个 `Agent` 可以通过 `Arc` 共享。
**但是**，其底层的可变状态（对话历史、上下文、工具缓存）在同一实例上
并发调用 `execute` / `chat_stream` 时*并不*安全。调用方**必须**序列化访问——
通常通过 `Arc<RwLock<Agent>>` 或 `Arc<tokio::sync::Mutex<Agent>>`。

workflow 层在通过 plan-execute 和多 agent 拓扑驱动 agent 时，已经强制执行了
这种序列化。
*/
/**
可共享的、agent 全局范围的工具可见性策略。

该策略会在创建一次调用快照时被读取。与惰性构建的 Subagent 共享它，
可以在不让框架耦合任何特定产品的工具控制存储的前提下，保持它们注册的
能力面一致。
*/
#[derive(Debug, Clone, Default)]
pub struct ToolVisibilityPolicy {
    /// 当前被禁用的工具名集合；`None` 表示没有禁用项。
    disabled: std::sync::Arc<std::sync::RwLock<Option<std::collections::HashSet<String>>>>,
}

impl ToolVisibilityPolicy {
    /// 设置被禁用的工具名集合（空集合会被归一化为 `None`）。
    pub fn set_disabled(&self, names: Option<std::collections::HashSet<String>>) {
        if let Ok(mut guard) = self.disabled.write() {
            *guard = names.filter(|names| !names.is_empty());
        }
    }

    /// 返回当前被禁用的工具名集合（未设置时为空集合）。
    pub fn disabled_names(&self) -> std::collections::HashSet<String> {
        self.disabled
            .read()
            .map(|guard| guard.clone().unwrap_or_default())
            .unwrap_or_default()
    }
}

/// Agent 核心 trait：定义执行、对话、流式输出、工具管理与生命周期控制等统一接口。
pub trait Agent: Send + Sync {
    /// 用于日志、事件与编排中的、人类可读的 agent 名称。
    fn name(&self) -> &str;
    /// 该 agent 当前绑定的模型标识符。
    fn model_name(&self) -> &str;
    /// 为该 agent 行为奠基的 system prompt。
    fn system_prompt(&self) -> &str;

    /// 当前暴露给模型的工具名列表。
    fn tool_names(&self) -> Vec<String> {
        vec![]
    }

    /// 序列化进 LLM 请求中的工具定义。
    fn tool_definitions(&self) -> Vec<ToolDefinition> {
        vec![]
    }

    /**
    被 agent 全局能力策略对模型隐藏的已注册工具。仅限本次调用的排除项
    保留在 [`AgentInvocationContext::disabled_tools`] 中。
    */
    fn disabled_tool_names(&self) -> std::collections::HashSet<String> {
        self.tool_visibility_policy().disabled_names()
    }

    /// 供未来调用快照使用的、agent 全局范围的可见性策略。
    fn tool_visibility_policy(&self) -> ToolVisibilityPolicy {
        ToolVisibilityPolicy::default()
    }

    /// 为后续调用配置的、生效中的工作目录。
    fn working_dir(&self) -> Option<std::path::PathBuf> {
        None
    }

    /// 该 agent 可用的、人类可读的 skill 标识符。
    fn skill_names(&self) -> Vec<String> {
        vec![]
    }

    /// 该 agent 可用的、已配置的 MCP server 标识符。
    fn mcp_server_names(&self) -> Vec<String> {
        vec![]
    }

    /**
    在丢弃该 agent 之前释放外部资源。

    若清理失败（例如 MCP server 断开连接出错、flush 失败）返回 `Err`，
    使调用方可以记录日志或处理资源泄漏。
    */
    fn close<'a>(&'a self) -> BoxFuture<'a, Result<()>> {
        Box::pin(async { Ok(()) })
    }

    /**
    执行一个任务并返回最终答案。

    **面向任务**：重置/恢复上下文，可能先运行一个规划阶段，再进入 ReAct 循环。
    用于独立的单轮任务，此时 agent 要么全新开始，要么从 checkpoint 恢复。

    # ⚠️ 警告：不要用于多轮对话 UI

    `execute()` 在每次调用时都会**清空对话历史**（只有 system prompt 会保留）。
    在 REPL / 终端 UI / 聊天机器人循环中调用它，会使 agent"忘记"所有先前的对话轮次。

    对于任何 agent 必须跨调用记住先前消息的场景，**请改用
    [`chat()`](Agent::chat)**。

    | 场景 | 正确的方法 |
    |---|---|
    | CLI 一次性命令 | `execute()` ✓ |
    | Workflow / 流水线节点 | `execute()` ✓ |
    | 批处理（相互独立的任务） | `execute()` ✓ |
    | REPL / 终端 UI / 聊天机器人 | `chat()` ✓ |
    | 多轮对话 | `chat()` ✓ |
    */
    fn execute<'a>(&'a self, task: &'a str) -> BoxFuture<'a, Result<String>>;

    /**
    执行一个任务并流式产出生命周期事件。

    与 [`Self::execute`] 语义相同，但返回一个 [`AgentEvent`] 流，
    用于实时可观测性。

    # ⚠️ 警告

    每次调用都会清空对话历史——详情见 [`execute()`](Agent::execute)。
    多轮对话 UI 请使用 [`chat_stream()`](Agent::chat_stream)。
    */
    fn execute_stream<'a>(
        &'a self,
        task: &'a str,
    ) -> BoxFuture<'a, Result<BoxStream<'a, Result<AgentEvent>>>>;

    /**
    执行一个任务，支持协作式取消。

    默认实现用一个感知取消的包装器包裹 [`Self::execute_stream`]。当 `cancel`
    被触发时，该 stream 会产出 [`AgentEvent::Cancelled`] 并终止。

    # ⚠️ 警告

    每次调用都会清空对话历史——详情见 [`execute()`](Agent::execute)。
    多轮对话 UI 请使用 [`chat_stream_with_cancel()`](Agent::chat_stream_with_cancel)。
    */
    fn execute_stream_with_cancel<'a>(
        &'a self,
        task: &'a str,
        cancel: CancellationToken,
    ) -> BoxFuture<'a, Result<BoxStream<'a, Result<AgentEvent>>>> {
        Box::pin(async move {
            let stream = self.execute_stream(task).await?;
            Ok(cancel_aware_stream(stream, cancel))
        })
    }

    /**
    以一个不可变值携带调用元数据来执行。

    默认实现保持对不消费调用元数据的 agent 的兼容性。
    */
    fn execute_stream_with_invocation_context<'a>(
        &'a self,
        task: &'a str,
        cancel: CancellationToken,
        invocation: AgentInvocationContext,
    ) -> BoxFuture<'a, Result<BoxStream<'a, Result<AgentEvent>>>> {
        Box::pin(async move {
            let stream = self.execute_stream_with_cancel(task, cancel).await?;
            Ok(invocation_retaining_stream(stream, invocation))
        })
    }

    /**
    在一次多轮对话中与该 agent 聊天。

    **面向对话**：保留既有上下文（不重置），追加用户消息，运行 ReAct 循环。
    用于 agent 需要跨调用累积状态的交互式多轮对话场景。

    对于任何按顺序发送多条消息的 UI（REPL、终端 UI、聊天机器人、网页聊天），
    **优先使用它而不是 [`execute()`](Agent::execute)**。

    默认实现委托给 [`Self::execute`]；具体实现（如 `ReactAgent`）会覆盖它
    以避免重置上下文。
    */
    fn chat<'a>(&'a self, message: &'a str) -> BoxFuture<'a, Result<String>> {
        self.execute(message)
    }

    /**
    与该 agent 聊天并流式产出生命周期事件。

    与 [`Self::chat`] 语义相同，但返回一个 [`AgentEvent`] 流，
    用于实时可观测性。

    对于任何按顺序发送多条消息的 UI，**优先使用它而不是
    [`execute_stream()`](Agent::execute_stream)**。
    */
    fn chat_stream<'a>(
        &'a self,
        message: &'a str,
    ) -> BoxFuture<'a, Result<BoxStream<'a, Result<AgentEvent>>>> {
        self.execute_stream(message)
    }

    /**
    支持协作式取消的聊天流式变体。

    默认实现用一个感知取消的包装器包裹 [`Self::chat_stream`]。当 `cancel`
    被触发时，该 stream 会产出 [`AgentEvent::Cancelled`] 并终止。
    */
    fn chat_stream_with_cancel<'a>(
        &'a self,
        message: &'a str,
        cancel: CancellationToken,
    ) -> BoxFuture<'a, Result<BoxStream<'a, Result<AgentEvent>>>> {
        Box::pin(async move {
            let stream = self.chat_stream(message).await?;
            Ok(cancel_aware_stream(stream, cancel))
        })
    }

    /**
    使用结构化多模态消息的多轮聊天流式方法。

    实现必须保持与 [`Self::chat_stream_with_cancel`] 相同的对话语义：
    追加到既有上下文，而不是重置它。默认实现不支持，因为从一个拥有所有权
    的消息中提取出借用的文本回退值，在生命周期上是不安全的。
    */
    fn chat_stream_message_with_cancel<'a>(
        &'a self,
        _message: Message,
        _cancel: CancellationToken,
    ) -> BoxFuture<'a, Result<BoxStream<'a, Result<AgentEvent>>>> {
        Box::pin(async move {
            Err(crate::error::ReactError::Other(
                "this agent does not implement multimodal streaming (chat_stream_message_with_cancel)"
                    .to_string(),
            ))
        })
    }

    /**
    带限定值作用域调用元数据的结构化多轮聊天。

    默认实现委托给既有的结构化聊天方法，使第三方 agent 无需改变语义
    即可选择接入元数据。
    */
    fn chat_stream_message_with_invocation_context<'a>(
        &'a self,
        message: Message,
        cancel: CancellationToken,
        invocation: AgentInvocationContext,
    ) -> BoxFuture<'a, Result<BoxStream<'a, Result<AgentEvent>>>> {
        Box::pin(async move {
            let stream = self
                .chat_stream_message_with_cancel(message, cancel)
                .await?;
            Ok(invocation_retaining_stream(stream, invocation))
        })
    }

    /**
    带取消支持的流式任务执行（多模态版本）。

    接受一个预先构建好的 [`Message`]，使经由 subagent 委派派发的 subagent
    能够看到用户附带的图片/文件。`ReactAgent` 会覆盖此方法，
    使其经由真正的多模态流水线路由。

    默认实现**不支持**——未实现多模态流式处理的 agent，在被派发多模态任务时
    会返回一个错误。这样可以保持 trait 签名在生命周期上的安全性（提取出的
    文本回退值会借用一个局部变量）。需要文本回退值的调用方应自行提取文本，
    并使用 [`execute_stream_with_cancel`](Self::execute_stream_with_cancel)。
    */
    fn execute_stream_message_with_cancel<'a>(
        &'a self,
        _message: Message,
        _cancel: CancellationToken,
    ) -> BoxFuture<'a, Result<BoxStream<'a, Result<AgentEvent>>>> {
        Box::pin(async move {
            Err(crate::error::ReactError::Other(
                "this agent does not implement multimodal streaming (execute_stream_message_with_cancel)".to_string(),
            ))
        })
    }

    /**
    带限定值作用域调用元数据的多模态流式处理。

    默认实现委托给既有的多模态方法，使第三方 agent 保持源码兼容。
    */
    fn execute_stream_message_with_invocation_context<'a>(
        &'a self,
        message: Message,
        cancel: CancellationToken,
        invocation: AgentInvocationContext,
    ) -> BoxFuture<'a, Result<BoxStream<'a, Result<AgentEvent>>>> {
        Box::pin(async move {
            let stream = self
                .execute_stream_message_with_cancel(message, cancel)
                .await?;
            Ok(invocation_retaining_stream(stream, invocation))
        })
    }

    /**
    重置内存中的对话状态。

    实现应清空上下文，并触发 `SessionStart("clear")` hook，
    使已注册的 hook 能够响应该重置事件。
    */
    fn reset(&self) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
        Box::pin(async {})
    }

    /**
    若该 agent 正在跟踪某个 run ID，返回它。
    默认值：None。ReactAgent 会覆盖此方法。
    */
    fn current_run_id(&self) -> Option<String> {
        None
    }

    /**
    返回该 agent 实例的累计 token 用量快照。

    轻量级和第三方 agent 可以保持空的默认实现。团队编排会在执行前后
    比较快照，从而在不依赖具体 agent 实现的情况下收集用量信息。
    */
    fn token_usage_summary(&self) -> crate::tokenizer::UsageSummary {
        crate::tokenizer::UsageSummary::default()
    }

    /**
    在实现现有的安全点，向当前 turn 注入一条消息。
    不支持实时引导的实现保持这个带类型的默认返回值。
    */
    fn steer_input(
        &self,
        _expected_turn_id: Option<&str>,
        _message: Message,
    ) -> std::result::Result<String, AgentSteerError> {
        Err(AgentSteerError::Unsupported)
    }

    /**
    注入一条消息，并跟踪它经历接受、模型上下文 drain、根 turn 结算的全过程。

    只支持旧版 [`Self::steer_input`] 的实现保持源码兼容，
    在能够提供真实的生命周期信号之前，在这里返回
    [`AgentSteerError::Unsupported`]。
    */
    fn steer_input_tracked(
        &self,
        _expected_turn_id: Option<&str>,
        _message: Message,
    ) -> std::result::Result<AgentSteerReceipt, AgentSteerError> {
        Err(AgentSteerError::Unsupported)
    }

    // ── External run context (跨 spawn 安全的值传递, 见 ExternalRunContext) ──

    /**
    旧版的、agent 全局范围的 run 上下文 setter。

    **背景**：`tokio::task_local!` 不会跨 `tokio::spawn` 继承。subagent instance
    在框架层的 `tokio::spawn`（subagent_executor.rs 的 dispatch_fork）里执行，
    应用层经 task_local 注入的 run_id / cancel / trace_sink 全部丢失。
    新代码应使用 `AgentInvocationContext` 的 value-scoped streaming methods；
    此 setter 仅为外部兼容保留。
    */
    fn set_external_context(&self, _ctx: &crate::tools::ExternalRunContext) {}

    /// 清除通过旧版 setter 安装的上下文。
    fn clear_external_context(&self) {}

    /**
    为该 agent 的工具调用绑定一个工作目录（Sprint 8 worktree
    隔离）。设置后，每一次 shell/文件/git 工具调用都会在 `path` 内部运行
    （经由 `ToolContext.working_dir`）。`None` 会清除它（恢复默认 cwd）。

    默认值：noop。新的调用路径应使用 `AgentInvocationContext::working_dir`；
    此 setter 仅为需要刻意配置 agent 全局默认值的调用方保留兼容性。
    */
    fn set_working_dir(&self, _path: Option<std::path::PathBuf>) {}

    /**
    清除已绑定的工作目录（等价于 `set_working_dir(None)`）。
    默认值：noop。ReactAgent 会覆盖它。
    */
    fn clear_working_dir(&self) {}

    // ── 动态能力方法（默认 noop）────────────────────

    /**
    在运行时动态注册一个工具。

    默认值：noop（不做任何操作直接返回）。ReactAgent 会覆盖此方法，
    把工具添加进它的 ToolManager。
    */
    fn register_tool(&self, _tool: Box<dyn crate::tools::Tool>) {}

    /**
    在运行时按名称动态移除一个工具。

    若找到并移除该工具返回 `true`，否则返回 `false`。
    默认值：返回 `false`（不做任何操作）。ReactAgent 会覆盖此方法。
    */
    fn remove_tool(&self, _name: &str) -> bool {
        false
    }

    /**
    获取当前对话历史。

    默认值：空列表。ReactAgent 会覆盖此方法，
    返回其 ContextManager 中的消息。
    */
    fn messages(&self) -> Vec<Message> {
        vec![]
    }

    /**
    在运行时更新 system prompt。

    默认值：noop。ReactAgent 会覆盖此方法，
    更新其配置并把该 prompt 重新注入上下文。
    */
    fn set_system_prompt(&self, _prompt: &str) {}

    /**
    把一个任务委派给某个具名的 subagent 或团队成员。

    默认值：返回错误（"delegation not supported"）。
    当启用了 Subagent 特性时，ReactAgent 会覆盖此方法。
    */
    fn delegate_to<'a>(
        &'a self,
        _target: &'a str,
        _task: &'a str,
    ) -> BoxFuture<'a, Result<String>> {
        Box::pin(async {
            Err(ReactError::Other(
                "delegation not supported by this agent".into(),
            ))
        })
    }
}

// ── 为 Box<dyn Agent> 提供的统一实现 ──────────────────────────────────────

/**
允许 `Box<dyn Agent>` 直接作为一个 `Agent` 使用。

这使得 `Arc<Box<dyn Agent>>` 可以强制转换为 `Arc<dyn Agent>`，
这对于从 `SharedAgent` 中移除外层 `Mutex` 是必需的。
*/
impl Agent for Box<dyn Agent> {
    /// 委托给内部 agent 的 [`Agent::name`]。
    fn name(&self) -> &str {
        self.as_ref().name()
    }
    /// 委托给内部 agent 的 [`Agent::model_name`]。
    fn model_name(&self) -> &str {
        self.as_ref().model_name()
    }
    /// 委托给内部 agent 的 [`Agent::system_prompt`]。
    fn system_prompt(&self) -> &str {
        self.as_ref().system_prompt()
    }
    /// 委托给内部 agent 的 [`Agent::tool_names`]。
    fn tool_names(&self) -> Vec<String> {
        self.as_ref().tool_names()
    }
    /// 委托给内部 agent 的 [`Agent::tool_definitions`]。
    fn tool_definitions(&self) -> Vec<crate::llm::ToolDefinition> {
        self.as_ref().tool_definitions()
    }
    /// 委托给内部 agent 的 [`Agent::skill_names`]。
    fn skill_names(&self) -> Vec<String> {
        self.as_ref().skill_names()
    }
    /// 委托给内部 agent 的 [`Agent::mcp_server_names`]。
    fn mcp_server_names(&self) -> Vec<String> {
        self.as_ref().mcp_server_names()
    }
    /// 委托给内部 agent 的 [`Agent::token_usage_summary`]。
    fn token_usage_summary(&self) -> crate::tokenizer::UsageSummary {
        self.as_ref().token_usage_summary()
    }
    /// 委托给内部 agent 的 [`Agent::steer_input`]。
    fn steer_input(
        &self,
        expected_turn_id: Option<&str>,
        message: Message,
    ) -> std::result::Result<String, AgentSteerError> {
        self.as_ref().steer_input(expected_turn_id, message)
    }
    /// 委托给内部 agent 的 [`Agent::steer_input_tracked`]。
    fn steer_input_tracked(
        &self,
        expected_turn_id: Option<&str>,
        message: Message,
    ) -> std::result::Result<AgentSteerReceipt, AgentSteerError> {
        self.as_ref().steer_input_tracked(expected_turn_id, message)
    }
    /// 委托给内部 agent 的 [`Agent::close`]。
    fn close<'a>(&'a self) -> BoxFuture<'a, Result<()>> {
        self.as_ref().close()
    }
    /// 委托给内部 agent 的 [`Agent::execute`]。
    fn execute<'a>(&'a self, task: &'a str) -> BoxFuture<'a, Result<String>> {
        self.as_ref().execute(task)
    }
    /// 委托给内部 agent 的 [`Agent::execute_stream`]。
    fn execute_stream<'a>(
        &'a self,
        task: &'a str,
    ) -> BoxFuture<'a, Result<BoxStream<'a, Result<AgentEvent>>>> {
        self.as_ref().execute_stream(task)
    }
    /// 委托给内部 agent 的 [`Agent::execute_stream_with_cancel`]。
    fn execute_stream_with_cancel<'a>(
        &'a self,
        task: &'a str,
        cancel: CancellationToken,
    ) -> BoxFuture<'a, Result<BoxStream<'a, Result<AgentEvent>>>> {
        self.as_ref().execute_stream_with_cancel(task, cancel)
    }
    /// 委托给内部 agent 的 [`Agent::execute_stream_with_invocation_context`]。
    fn execute_stream_with_invocation_context<'a>(
        &'a self,
        task: &'a str,
        cancel: CancellationToken,
        invocation: AgentInvocationContext,
    ) -> BoxFuture<'a, Result<BoxStream<'a, Result<AgentEvent>>>> {
        self.as_ref()
            .execute_stream_with_invocation_context(task, cancel, invocation)
    }
    /// 委托给内部 agent 的 [`Agent::chat`]。
    fn chat<'a>(&'a self, message: &'a str) -> BoxFuture<'a, Result<String>> {
        self.as_ref().chat(message)
    }
    /// 委托给内部 agent 的 [`Agent::chat_stream`]。
    fn chat_stream<'a>(
        &'a self,
        message: &'a str,
    ) -> BoxFuture<'a, Result<BoxStream<'a, Result<AgentEvent>>>> {
        self.as_ref().chat_stream(message)
    }
    /// 委托给内部 agent 的 [`Agent::chat_stream_with_cancel`]。
    fn chat_stream_with_cancel<'a>(
        &'a self,
        message: &'a str,
        cancel: CancellationToken,
    ) -> BoxFuture<'a, Result<BoxStream<'a, Result<AgentEvent>>>> {
        self.as_ref().chat_stream_with_cancel(message, cancel)
    }
    /// 委托给内部 agent 的 [`Agent::chat_stream_message_with_cancel`]。
    fn chat_stream_message_with_cancel<'a>(
        &'a self,
        message: Message,
        cancel: CancellationToken,
    ) -> BoxFuture<'a, Result<BoxStream<'a, Result<AgentEvent>>>> {
        self.as_ref()
            .chat_stream_message_with_cancel(message, cancel)
    }
    /// 委托给内部 agent 的 [`Agent::chat_stream_message_with_invocation_context`]。
    fn chat_stream_message_with_invocation_context<'a>(
        &'a self,
        message: Message,
        cancel: CancellationToken,
        invocation: AgentInvocationContext,
    ) -> BoxFuture<'a, Result<BoxStream<'a, Result<AgentEvent>>>> {
        self.as_ref()
            .chat_stream_message_with_invocation_context(message, cancel, invocation)
    }
    /// 委托给内部 agent 的 [`Agent::execute_stream_message_with_cancel`]。
    fn execute_stream_message_with_cancel<'a>(
        &'a self,
        message: Message,
        cancel: CancellationToken,
    ) -> BoxFuture<'a, Result<BoxStream<'a, Result<AgentEvent>>>> {
        self.as_ref()
            .execute_stream_message_with_cancel(message, cancel)
    }
    /// 委托给内部 agent 的 [`Agent::execute_stream_message_with_invocation_context`]。
    fn execute_stream_message_with_invocation_context<'a>(
        &'a self,
        message: Message,
        cancel: CancellationToken,
        invocation: AgentInvocationContext,
    ) -> BoxFuture<'a, Result<BoxStream<'a, Result<AgentEvent>>>> {
        self.as_ref()
            .execute_stream_message_with_invocation_context(message, cancel, invocation)
    }
    /// 委托给内部 agent 的 [`Agent::current_run_id`]。
    fn current_run_id(&self) -> Option<String> {
        self.as_ref().current_run_id()
    }
    /// 委托给内部 agent 的 [`Agent::reset`]。
    fn reset(&self) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
        self.as_ref().reset()
    }
    /// 委托给内部 agent 的 [`Agent::set_external_context`]。
    fn set_external_context(&self, ctx: &crate::tools::ExternalRunContext) {
        self.as_ref().set_external_context(ctx);
    }
    /// 委托给内部 agent 的 [`Agent::clear_external_context`]。
    fn clear_external_context(&self) {
        self.as_ref().clear_external_context();
    }
    /// 委托给内部 agent 的 [`Agent::set_working_dir`]。
    fn set_working_dir(&self, path: Option<std::path::PathBuf>) {
        self.as_ref().set_working_dir(path);
    }
    /// 委托给内部 agent 的 [`Agent::clear_working_dir`]。
    fn clear_working_dir(&self) {
        self.as_ref().clear_working_dir();
    }
    /// 委托给内部 agent 的 [`Agent::register_tool`]。
    fn register_tool(&self, tool: Box<dyn crate::tools::Tool>) {
        self.as_ref().register_tool(tool)
    }
    /// 委托给内部 agent 的 [`Agent::remove_tool`]。
    fn remove_tool(&self, name: &str) -> bool {
        self.as_ref().remove_tool(name)
    }
    /// 委托给内部 agent 的 [`Agent::messages`]。
    fn messages(&self) -> Vec<Message> {
        self.as_ref().messages()
    }
    /// 委托给内部 agent 的 [`Agent::set_system_prompt`]。
    fn set_system_prompt(&self, prompt: &str) {
        self.as_ref().set_system_prompt(prompt)
    }
    /// 委托给内部 agent 的 [`Agent::delegate_to`]。
    fn delegate_to<'a>(&'a self, target: &'a str, task: &'a str) -> BoxFuture<'a, Result<String>> {
        self.as_ref().delegate_to(target, task)
    }
}

/**
用一个支持协作式取消的包装器包裹一个 agent 事件 stream。

当 `cancel` 被触发时，该 stream 会产出 [`AgentEvent::Cancelled`] 并终止。
`execute_stream_with_cancel` 和 `chat_stream_with_cancel` 的默认实现共用它。
*/
fn cancel_aware_stream<'a>(
    mut stream: BoxStream<'a, Result<AgentEvent>>,
    cancel: CancellationToken,
) -> BoxStream<'a, Result<AgentEvent>> {
    let wrapped = async_stream::try_stream! {
        loop {
            tokio::select! {
                _ = cancel.cancelled() => {
                    yield AgentEvent::Cancelled;
                    break;
                }
                next = stream.next() => {
                    match next {
                        Some(event) => yield event?,
                        None => break,
                    }
                }
            }
        }
    };
    Box::pin(wrapped)
}

/// 包装事件流，使调用上下文（及其资源守卫）一直存活到流被丢弃为止。
fn invocation_retaining_stream<'a>(
    mut stream: BoxStream<'a, Result<AgentEvent>>,
    invocation: AgentInvocationContext,
) -> BoxStream<'a, Result<AgentEvent>> {
    let wrapped = async_stream::try_stream! {
        let _invocation = invocation;
        while let Some(event) = stream.next().await {
            yield event?;
        }
    };
    Box::pin(wrapped)
}

#[cfg(test)]
mod invocation_retaining_stream_tests {
    use super::*;
    use crate::tools::InvocationResourceGuard;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// 被丢弃时递增计数器的测试守卫。
    struct DropCounter(Arc<AtomicUsize>);

    impl Drop for DropCounter {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    /// 仅实现必需方法、其余使用默认实现的测试 agent。
    struct DefaultInvocationAgent;

    impl Agent for DefaultInvocationAgent {
        fn name(&self) -> &str {
            "default-invocation"
        }

        fn model_name(&self) -> &str {
            "test"
        }

        fn system_prompt(&self) -> &str {
            "test"
        }

        fn execute<'a>(&'a self, _task: &'a str) -> BoxFuture<'a, Result<String>> {
            Box::pin(async { Ok("done".to_string()) })
        }

        fn execute_stream<'a>(
            &'a self,
            _task: &'a str,
        ) -> BoxFuture<'a, Result<BoxStream<'a, Result<AgentEvent>>>> {
            Box::pin(async {
                Ok(Box::pin(futures::stream::pending()) as BoxStream<'a, Result<AgentEvent>>)
            })
        }

        fn chat_stream_message_with_cancel<'a>(
            &'a self,
            _message: Message,
            _cancel: CancellationToken,
        ) -> BoxFuture<'a, Result<BoxStream<'a, Result<AgentEvent>>>> {
            Box::pin(async {
                Ok(Box::pin(futures::stream::pending()) as BoxStream<'a, Result<AgentEvent>>)
            })
        }

        fn execute_stream_message_with_cancel<'a>(
            &'a self,
            _message: Message,
            _cancel: CancellationToken,
        ) -> BoxFuture<'a, Result<BoxStream<'a, Result<AgentEvent>>>> {
            Box::pin(async {
                Ok(Box::pin(futures::stream::pending()) as BoxStream<'a, Result<AgentEvent>>)
            })
        }
    }

    /// 构造携带一个 `DropCounter` 资源守卫的调用上下文。
    fn invocation(drops: &Arc<AtomicUsize>) -> AgentInvocationContext {
        AgentInvocationContext {
            resource_guards: vec![InvocationResourceGuard::new(DropCounter(Arc::clone(drops)))],
            ..AgentInvocationContext::default()
        }
    }

    #[tokio::test]
    async fn default_text_invocation_retains_guards_until_stream_drop() -> Result<()> {
        let drops = Arc::new(AtomicUsize::new(0));
        let stream = DefaultInvocationAgent
            .execute_stream_with_invocation_context(
                "task",
                CancellationToken::new(),
                invocation(&drops),
            )
            .await?;
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        drop(stream);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        Ok(())
    }

    #[tokio::test]
    async fn default_multimodal_chat_retains_guards_until_stream_drop() -> Result<()> {
        let drops = Arc::new(AtomicUsize::new(0));
        let stream = DefaultInvocationAgent
            .chat_stream_message_with_invocation_context(
                Message::user("chat".to_string()),
                CancellationToken::new(),
                invocation(&drops),
            )
            .await?;
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        drop(stream);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        Ok(())
    }

    #[tokio::test]
    async fn default_multimodal_execute_retains_guards_until_stream_drop() -> Result<()> {
        let drops = Arc::new(AtomicUsize::new(0));
        let stream = DefaultInvocationAgent
            .execute_stream_message_with_invocation_context(
                Message::user("execute".to_string()),
                CancellationToken::new(),
                invocation(&drops),
            )
            .await?;
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        drop(stream);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        Ok(())
    }
}

/// Agent 生命周期回调接口
pub trait AgentCallback: Send + Sync {
    /// 供定向移除使用的可选稳定回调种类。
    fn callback_kind(&self) -> Option<&'static str> {
        None
    }

    /// 供移除某个具体回调实例使用的可选稳定标识符。
    fn callback_id(&self) -> Option<&str> {
        None
    }

    /// 在模型开始一次推理步骤之前调用。
    fn on_think_start<'a>(
        &'a self,
        _agent: &'a str,
        _messages: &'a [Message],
    ) -> BoxFuture<'a, ()> {
        Box::pin(async {})
    }

    /// 在模型推理步骤结束后调用，携带 token 用量信息。
    fn on_think_end<'a>(
        &'a self,
        _agent: &'a str,
        _steps: &'a [StepType],
        _prompt_tokens: usize,
        _completion_tokens: usize,
    ) -> BoxFuture<'a, ()> {
        Box::pin(async {})
    }

    /// 在一次工具调用开始之前调用。
    fn on_tool_start<'a>(
        &'a self,
        _agent: &'a str,
        _tool: &'a str,
        _args: &'a Value,
    ) -> BoxFuture<'a, ()> {
        Box::pin(async {})
    }

    /**
    在一次工具调用开始之前调用，携带其规范调用身份。

    默认实现为只需要工具级生命周期通知的回调保留源码兼容性。
    */
    fn on_tool_start_with_id<'a>(
        &'a self,
        agent: &'a str,
        _call_id: &'a str,
        tool: &'a str,
        args: &'a Value,
    ) -> BoxFuture<'a, ()> {
        self.on_tool_start(agent, tool, args)
    }

    /// 在一次工具调用成功后调用。
    fn on_tool_end<'a>(
        &'a self,
        _agent: &'a str,
        _tool: &'a str,
        _result: &'a str,
    ) -> BoxFuture<'a, ()> {
        Box::pin(async {})
    }

    /// 在一次工具调用成功后调用，携带其规范调用身份。
    fn on_tool_end_with_id<'a>(
        &'a self,
        agent: &'a str,
        _call_id: &'a str,
        tool: &'a str,
        result: &'a str,
    ) -> BoxFuture<'a, ()> {
        self.on_tool_end(agent, tool, result)
    }

    /// 在一次工具调用失败时调用。
    fn on_tool_error<'a>(
        &'a self,
        _agent: &'a str,
        _tool: &'a str,
        _err: &'a ReactError,
    ) -> BoxFuture<'a, ()> {
        Box::pin(async {})
    }

    /// 在一次工具调用失败后调用，携带其规范调用身份。
    fn on_tool_error_with_id<'a>(
        &'a self,
        agent: &'a str,
        _call_id: &'a str,
        tool: &'a str,
        error: &'a ReactError,
    ) -> BoxFuture<'a, ()> {
        self.on_tool_error(agent, tool, error)
    }

    /**
    结算一次被运行时中断、且未捕获到结果的调用。

    完整的输入使持久化类回调即使在中断发生于其正常的开始回调之前，
    也能够保留这次已被受理的调用。默认实现会发出一个错误终态，
    并刻意不伪造一次重复的开始通知。
    */
    fn on_tool_interrupted_with_id<'a>(
        &'a self,
        agent: &'a str,
        call_id: &'a str,
        tool: &'a str,
        _input: &'a Value,
        error: &'a ReactError,
    ) -> BoxFuture<'a, ()> {
        self.on_tool_error_with_id(agent, call_id, tool, error)
    }

    /// 在 agent 产出最终答案时调用。
    fn on_final_answer<'a>(&'a self, _agent: &'a str, _answer: &'a str) -> BoxFuture<'a, ()> {
        Box::pin(async {})
    }

    /// 在每一次外层控制循环迭代结束时调用。
    fn on_iteration<'a>(&'a self, _agent: &'a str, _iteration: usize) -> BoxFuture<'a, ()> {
        Box::pin(async {})
    }
}
