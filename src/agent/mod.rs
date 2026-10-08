/*!
Agent 模块

定义核心的 [`Agent`] trait、事件枚举 [`AgentEvent`]，以及回调接口 [`AgentCallback`]。

## 架构

框架提供了单一、健壮的 [`ReactAgent`]，实现了 ReAct
（Think-Act-Observe，思考-行动-观察）模式。不同的执行范式通过
可组合的工具与配置来表达，而非拆分为多种 Agent 类型：

| 能力 | 实现机制 |
|------------|-----------|
| ReAct 推理 | [`react`] —— 默认循环 |
| 任务规划 | 工具（plan、create_task、update_task） |
| 自我反思 | [`critic`] —— 评估与反馈工具 |
| 子 Agent 协作 | `subagent` —— 多 Agent 编排 |

# 快速开始

```rust,no_run
use echo_agent::prelude::*;

# fn main() -> echo_agent::error::Result<()> {
let agent = ReactAgentBuilder::new()
    .model("qwen3-max")
    .system_prompt("You are a helpful assistant")
    .enable_tools()
    .build()?;

println!("Agent name: {}", agent.name());
println!("Model: {}", agent.model_name());
# Ok(())
# }
```
*/

pub use echo_core::agent::builder::AgentBuilder as AgentBuilderTrait;
pub use echo_core::agent::{
    AGENT_EVENT_SCHEMA_VERSION, Agent, AgentCallback, AgentEvent, AgentInputLifecycle,
    AgentInvocationContext, AgentSteerError, AgentSteerPhase, AgentSteerReceipt, AgentSteerState,
    AgentSteerTurnOutcome, BudgetDecision, CancellationToken, EventEnvelope, EventId,
    EventIdentity, ExecutionUsage, InterventionCallback, InterventionResult, StepType, StreamId,
    ToolInvocation, ToolInvocationRewrite, ToolVisibilityPolicy, envelope_event_stream,
    envelope_event_stream_after, validate_envelope_trajectory, validate_event_trajectory,
};

/// Agent 取消后允许其安顿（settle）收尾的最长等待时长。
pub(crate) const AGENT_CANCELLATION_SETTLE_PERIOD: std::time::Duration =
    std::time::Duration::from_secs(12);

// ── 核心子模块 ───────────────────────────────────────────────────────

/// Agent 执行准入控制。
pub mod admission;
/// Agent 执行回调接口与实现。
pub mod callbacks;
/// Agent 配置类型。
pub mod config;
/// 输出质量评审（Critic）能力。
pub mod critic;
/// 默认的 Agent 工厂实现。
pub mod default_factory;
/// Agent 运行句柄。
pub mod handle;
/// ReAct Agent 核心实现。
pub mod react;
/// Agent 状态快照。
pub mod snapshot;
/// 运行中引导（steer）消息处理。
mod steer;

/// 子 Agent 派发与团队协作。
#[cfg(feature = "subagent")]
pub mod subagent;

// ── 重新导出 ──────────────────────────────────────────────────────────────

pub use crate::agent::handle::AgentHandle;
pub use crate::agent::react::builder::ReactAgentBuilder;
pub use crate::agent::react::run::pipeline::ToolExecutionPipeline;
pub use crate::agent::react::structured::StructuredAgent;
pub use crate::agent::react::{
    PreparedAgentModelDeactivation, PreparedAgentModelGeneration, PreparedCriticUpdate,
    PreparedTokenLimit, ReactAgent,
};
pub use crate::agent::snapshot::AgentRunSnapshot;
pub use crate::agent::steer::TurnSteerError;
pub use config::AgentConfig;
pub use echo_core::agent::{
    ExecutionAdmission, KeyedExecutionAdmission, KeyedExecutionAdmissionError, KeyedExecutionLease,
    KeyedExecutionRetirement,
};

/**
Agent 工厂相关类型 —— 从 echo-core 重新导出，并带有 facade 层的覆盖实现。

本模块提供 [`factory::AgentFactory`]、[`factory::AgentFactoryConfig`]，
以及 [`factory::DefaultAgentFactory`]（使用 `ReactAgentBuilder` 的
具体 facade 层实现）。
*/
pub mod factory {
    pub use crate::agent::default_factory::DefaultAgentFactory;
    pub use echo_core::agent::factory::{AgentFactory, AgentFactoryConfig};
}

/// 为兼容宏与精简 API 而提供的别名。
pub type AgentBuilder = ReactAgentBuilder;
