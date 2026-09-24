//! 子 Agent（Subagent）系统 —— 多层级 Agent 委派。
//!
//! 提供三种执行模式：
//! - **Sync（同步）**：父 Agent 阻塞等待子 Agent 返回（即当前 `AgentDispatchTool` 的行为）
//! - **Fork（分叉）**：独立运行；除非显式请求过滤后的历史，否则上下文是全新的
//! - **Teammate（队友）**：并行独立运行的 Agent，通过 join/cancel 句柄进行控制

pub mod builder;
pub mod context;
pub mod control;
pub mod events;
pub mod executor;
pub mod hooks;
pub mod isolation;
pub mod prompt;
pub mod registry;
pub mod team;
pub mod types;
pub mod usage;

// 重新导出最常用的类型
pub use builder::SubagentBuilder;
pub use context::{ContextInheritance, SubagentContext};
pub use control::{
    ActiveAttemptSummary, SubagentAttemptIdentity, SubagentCommandIdentity, SubagentCommandPhase,
    SubagentControlError, SubagentControlPhase, SubagentGuidanceQueueReceipt,
    SubagentInterruptOutcome, SubagentInterruptRequestDisposition, SubagentInterruptRequestReceipt,
    SubagentMessageReceipt,
};
pub use events::{
    SubagentEvent, SubagentEventBus, SubagentEventEnvelope, SubagentEventGap, SubagentEventPayload,
    SubagentEventPublisher, SubagentEventReplay, SubagentInvocationIdentity,
};
pub use executor::{
    BackgroundSubagentHandle, DispatchRequest, SubagentAttemptControlHandle, SubagentExecutor,
    SubagentExecutorConfig, TeammateHandle, default_uplink_sink, merge_observed_evidence,
    subagent_status_from_error,
};
pub use hooks::{SubagentHookContext, SubagentHookRegistry, SubagentHooks, SubagentRetryDecision};
pub use isolation::{
    IsolationError, IsolationHandle, IsolationOutcome, IsolationProvider, IsolationRequest,
    SharedIsolationProvider,
};
pub use prompt::{
    CompiledSubagentInvocation, CompiledSubagentSystemPrompt, ContextTransferPolicy,
    DefaultSubagentPromptCompiler, PromptActor, PromptDiagnostics, PromptSectionDiagnostic,
    SubagentExecutionBoundary, SubagentInvocation, SubagentPromptCompiler,
    SubagentSystemPromptInput, SubagentTaskContext, ToolCapability, ToolCapabilitySnapshot,
    compiled_current_message, filter_history, remove_duplicate_current_message, with_compiled_task,
};
pub use registry::{AgentFactory, FnAgentFactory, SubagentRegistry};
pub use team::{
    Team, TeamAgent, TeamAgentBuilder, TeamConfig, TeamDispatchController, TeamDispatchRequest,
    TeamExecutionResult, TeamMember, TeamRole, TeamRuntime, TeamRuntimeHandle,
    TeamRuntimeServiceHandle, TeamSpec, TeamStrategy, execute_team, execute_team_on_runtime,
    execute_team_on_runtime_service,
};
pub use types::{
    ExecutionMode, ObservedIsolation, RegisteredSubagent, SubagentAccessMode, SubagentArtifact,
    SubagentDefinition, SubagentEvidence, SubagentEvidenceSource, SubagentKind, SubagentOutcome,
    SubagentResult, SubagentStatus, SubagentTouchedFiles, SubagentVerification,
    SubagentVerificationStatus, parse_json_objects, parse_subagent_outcome, render_result_contract,
    split_subagent_output,
};
