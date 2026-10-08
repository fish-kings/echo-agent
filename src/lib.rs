/*!
# 可观测性与演化管线

框架为可观测性与自我改进提供了一套分层体系：

```text
trace (执行追踪)
  ↓ 提供 Run, RunEvent, RunStore 等执行追踪数据
eval (评测框架) [feature = "eval"]
  ↓ 基于 trace 数据运行 EvalCase，生成 EvalReport
improve (离线辅助) [feature = "improve"]
  ├─ 显式导出微调轨迹
  └─ 与 eval 同时启用时，分析评测结果并生成改进建议
evolution (结构化演化)
  └─ 管理 typed memory、证据候选、change audit、security、skill 生命周期
```

- [`trace`]: 执行追踪基础设施 — 完整记录单次执行的 Run/RunEvent/RunStore
- `eval`: 评测框架 — 定义 EvalCase/SuccessCriteria，基于 trace 运行评测
- `improve`: 离线辅助 — 显式轨迹导出；与 `eval` 同时启用时提供离线评测优化
- [`evolution`]: 结构化演化 — typed memory、证据候选、change audit、security、skill lifecycle
*/

#![doc = include_str!("../README.md")]
#![cfg_attr(docsrs, feature(doc_cfg))]

extern crate self as echo_agent;

#[doc(hidden)]
pub use echo_core::__macro_support;

// ── 核心模块（始终编译） ──────────────────────────────────────────

/// Agent 核心：ReAct Agent、事件、回调与执行管线。
pub mod agent;
/// 审计日志：记录 agent 行为的审计事件。
pub mod audit;
/// 与 Provider 无关的 token 预算契约。
pub mod budget {
    pub use echo_core::budget::*;
}
/// 上下文压缩策略与上下文管理器。
pub mod compression;
/// 框架与 Agent 的配置加载。
pub mod config;
/// 上下文组装：具备预算感知能力的消息列表构建。
pub mod context;
/// 统一的错误类型与 Result 别名。
pub mod error;
/// 评测框架：定义并运行 eval 用例，生成评测报告。
#[cfg(feature = "eval")]
#[cfg_attr(docsrs, doc(cfg(feature = "eval")))]
pub mod eval;
/// 结构化演化：typed memory、证据候选、变更审计与 skill 生命周期。
pub mod evolution;
/// 内容守卫：对输入/输出进行安全与合规检查。
pub mod guard;
/// 无头（非交互）运行 agent 的入口。
pub mod headless;
/// 框架 hook 契约。Skill hook 的加载仍位于 [`skills::hooks`]。
pub mod hooks {
    pub use echo_core::hooks::*;
}
/// 离线改进辅助：轨迹导出与基于评测的改进建议。
#[cfg(feature = "improve")]
#[cfg_attr(docsrs, doc(cfg(feature = "improve")))]
pub mod improve;
/// 意图路由：在进入 ReAct 循环前进行轻量级意图分类。
pub mod intent;
/// LLM 客户端、请求/响应类型与模型档案。
pub mod llm;
/// 记忆存储：键值存储、向量嵌入与快照。
pub mod memory;
/// 从对话中提取关键事实并提升为长期记忆。
pub mod memory_promoter;
/// 显式指定的框架数据根路径。
pub mod paths;
/// 插件门面与不可变的准备（preparation）API。
pub mod plugin;
/**
持久化的、按路由/负载类型化的投递原语。

这是持久化投递的标准 SDK 入口。应用保留各自的路由与负载类型，
无需任何面向应用的转换层。
*/
pub mod delivery {
    pub use crate::state::delivery::*;
}
/// Agent 能力档案与用户偏好档案契约。
pub mod profiles {
    pub use echo_state::profiles::*;
}
/// 通用重试策略与重试辅助函数。
pub mod retry;
/// Agent 轮次运行时门面（重导出 `echo_orchestration::runtime`）。
pub mod runtime;
/// 命令执行沙箱（本地、Docker、K8s）。
pub mod sandbox;
/// 基于 Cron 的任务调度。
pub mod scheduler;
/// 密钥扫描与脱敏，防止凭据泄露。
pub mod security;
/// 技能（Skill）系统：内置技能、外部技能加载与 hook。
pub mod skills;
/// 状态持久化与持久化投递。
pub mod state;
/// Skill 执行遥测契约及其存储。
pub mod skill_telemetry {
    pub use echo_state::skill_telemetry::*;
}
/// 测试辅助：Mock Agent、Mock LLM 客户端与 Mock 工具。
#[cfg(any(test, feature = "testing"))]
pub mod testing;
/// 分词器与 token 计数。
pub mod tokenizer;
/// 工具系统：工具定义、权限、执行配置与内置工具。
pub mod tools;
/// 执行追踪：Run、RunEvent 与 RunStore。
pub mod trace;
/// 通用工具模块（重导出自 echo_core）。
pub mod utils;
/// 工作流编排：顺序、并发、DAG 与图工作流。
pub mod workflow;

/**
子 Agent 的执行、生命周期与类型化结果契约。

该门面让使用方停留在公开 SDK 接口上，而不必深入内部的 `agent` 模块树。
*/
#[cfg(feature = "subagent")]
#[cfg_attr(docsrs, doc(cfg(feature = "subagent")))]
pub mod subagent {
    pub use crate::agent::subagent::*;
}

// ── 可选模块（由 feature 控制） ────────────────────────────────────────

/// ACP（Agent Client Protocol）协议支持。
#[cfg(feature = "acp")]
#[cfg_attr(docsrs, doc(cfg(feature = "acp")))]
pub mod acp;

/// A2A（Agent-to-Agent）协议的客户端与服务端。
#[cfg(feature = "a2a")]
#[cfg_attr(docsrs, doc(cfg(feature = "a2a")))]
pub mod a2a;

/// 外部消息渠道接入。
#[cfg(feature = "channels")]
#[cfg_attr(docsrs, doc(cfg(feature = "channels")))]
pub mod channels;

/// 人在回路（human-in-the-loop）审批与输入。
#[cfg(feature = "human-loop")]
#[cfg_attr(docsrs, doc(cfg(feature = "human-loop")))]
pub mod human_loop;

/// 统一 hook 桥接 —— 将任务/子 Agent 的 hook 接入中央 HookRegistry。
pub mod hooks_bridge;

/// MCP（Model Context Protocol）服务器接入与工具。
#[cfg(feature = "mcp")]
#[cfg_attr(docsrs, doc(cfg(feature = "mcp")))]
pub mod mcp;

/// LSP（Language Server Protocol）集成。
#[cfg(feature = "lsp")]
#[cfg_attr(docsrs, doc(cfg(feature = "lsp")))]
pub mod lsp;

/// 任务模型及任务管理工具。
pub mod tasks;

/// 遥测：指标与 OpenTelemetry 初始化。
#[cfg(feature = "telemetry")]
#[cfg_attr(docsrs, doc(cfg(feature = "telemetry")))]
pub mod telemetry;

/// Agent 拓扑追踪与可视化数据。
#[cfg(feature = "topology")]
#[cfg_attr(docsrs, doc(cfg(feature = "topology")))]
pub mod topology;

#[cfg(feature = "project-rules")]
pub use echo_core::project_rules;

// ── 声明式宏 ──────────────────────────────────────────────────────

/// 声明式宏定义。
mod macros;

// ── 过程宏重导出 ─────────────────────────────────────────────

pub use echo_macros::{
    Tool, audit_logger, callback, compressor, guard, handler, permission_policy, tool,
};

// ── Prelude（预导入） ─────────────────────────────────────────────────────────────────

/**
常用类型重导出。

通过 `use echo_agent::prelude::*` 一次性导入全部内容。
*/
pub mod prelude {
    // Agent
    pub use crate::agent::{
        AGENT_EVENT_SCHEMA_VERSION, Agent, AgentCallback, AgentConfig, AgentEvent, AgentHandle,
        AgentRunSnapshot, AgentSteerPhase, AgentSteerReceipt, AgentSteerState,
        AgentSteerTurnOutcome, CancellationToken, EventEnvelope, EventId, EventIdentity,
        ExecutionAdmission, ExecutionUsage, InterventionCallback, InterventionResult,
        KeyedExecutionAdmission, KeyedExecutionAdmissionError, KeyedExecutionLease,
        KeyedExecutionRetirement, PreparedAgentModelDeactivation, PreparedAgentModelGeneration,
        PreparedCriticUpdate, PreparedTokenLimit, ReactAgent, ReactAgentBuilder, StepType,
        StreamId, StructuredAgent, ToolExecutionPipeline, ToolInvocation, ToolInvocationRewrite,
        envelope_event_stream, envelope_event_stream_after, validate_envelope_trajectory,
        validate_event_trajectory,
    };
    // 持久化投递
    pub use crate::state::delivery::{
        DeliveryClaim, DeliveryClaimDraft, DeliveryEnvelope, DeliveryEvent, DeliveryInFlight,
        DeliveryLedger, DeliveryLedgerConfig, DeliveryLedgerError, DeliveryLedgerProjection,
        DeliveryOutcome, DeliveryPayload, DeliveryPhase, DeliveryRecord, DeliveryRoute,
        DeliverySettlement, DeliveryTransition,
    };
    // 子 Agent 结果与通用证据视图
    #[cfg(feature = "subagent")]
    pub use crate::agent::subagent::{
        BackgroundSubagentHandle, DispatchRequest, SubagentArtifact, SubagentAttemptIdentity,
        SubagentBuilder, SubagentCommandIdentity, SubagentCommandPhase, SubagentControlError,
        SubagentControlPhase, SubagentEvent, SubagentEventBus, SubagentEventEnvelope,
        SubagentEventGap, SubagentEventPayload, SubagentEventPublisher, SubagentEventReplay,
        SubagentEvidence, SubagentEvidenceSource, SubagentExecutor, SubagentExecutorConfig,
        SubagentGuidanceQueueReceipt, SubagentInterruptOutcome, SubagentInvocationIdentity,
        SubagentMessageReceipt, SubagentOutcome, SubagentResult, SubagentStatus,
        SubagentTouchedFiles, SubagentVerification, SubagentVerificationStatus, TeamAgent,
        TeamAgentBuilder, TeamConfig, TeamExecutionResult, TeamMember, TeamRole, TeamRuntime,
        TeamSpec, TeamStrategy,
    };
    // 提示词模板
    pub use crate::budget::TokenBudgetConfig;
    pub use echo_core::agent::{PromptTemplateManager, RunBudgetPolicy};
    // 配置
    pub use crate::config::{AgentSettings, FrameworkConfig, ModelConfig};

    /// [`ReactAgentBuilder`]（标准构建器类型）的便捷别名。
    pub type AgentBuilder = ReactAgentBuilder;

    // LLM
    pub use crate::llm::types::{
        ContentPart, ImageUrl, LinkedResource, Message, MessageContent, Role, ToolCall,
    };
    pub use crate::llm::{
        AnthropicClient, ChatChunk, ChatRequest, ChatResponse, JsonSchemaSpec, LlmApiProtocol,
        LlmClient, LlmConfig, LlmTimeouts, ModelInputModality, ModelProfile, ModelProfileOverride,
        ModelProfileResolver, OpenAiClient, ProviderCapabilities, ResponseFormat, ResponsesClient,
        SimpleChatOptions, ThinkingProfile, ToolDefinition, infer_context_window,
        resolve_protocol_endpoint, resolve_thinking_profile,
    };

    // 工具
    pub use crate::tools::builtin::think::ThinkTool;
    pub use crate::tools::control::{
        ToolControlError, ToolControlMutation, ToolControlService, ToolControlSnapshot,
    };
    pub use crate::tools::permission::{
        DefaultPermissionPolicy, PermissionDecision, PermissionMode, PermissionPolicy,
        PermissionRule, RuleBehavior, RuleMatcher, RuleRegistry, RuleSource, ToolPermission,
    };
    pub use crate::tools::{
        CommandPolicy, CommandPolicyDecision, StandardToolPack, Tool, ToolAccess, ToolCapabilities,
        ToolExecutionConfig, ToolFailure, ToolFailureCategory, ToolPack, ToolPackEntry,
        ToolParameters, ToolRecoveryAction, ToolResult, ToolRiskLevel, ToolSideEffect,
        ToolStreamEvent,
    };

    // Web 工具
    #[cfg(feature = "web")]
    #[cfg_attr(docsrs, doc(cfg(feature = "web")))]
    pub use crate::tools::web::{WebFetchTool, WebSearchTool};

    // 媒体工具
    #[cfg(feature = "media")]
    #[cfg_attr(docsrs, doc(cfg(feature = "media")))]
    pub use crate::tools::media::ImageFetchTool;

    // 上下文压缩
    pub use crate::compression::compressor::{
        HybridCompressor, HybridCompressorBuilder, IncrementalSummaryCompressor,
        SlidingWindowCompressor, SummaryCompressor, default_summary_prompt,
    };
    pub use crate::compression::horizon::{VisibilityHorizonCompressor, VisibilityHorizonConfig};
    pub use crate::compression::{
        CompressionInput, CompressionOutput, ContextCompressor, ContextManager, ForceCompressStats,
        PrepareResult,
    };

    // 分词器
    pub use crate::tokenizer::{HeuristicTokenizer, SimpleTokenizer, Tokenizer};

    // 记忆
    #[cfg(feature = "sqlite")]
    #[cfg_attr(docsrs, doc(cfg(feature = "sqlite")))]
    pub use crate::memory::SqliteStore;
    pub use crate::memory::{
        Embedder, EmbeddingStore, FileStore, HttpEmbedder, InMemoryStore, SnapshotManager,
        SnapshotPolicy, StateSnapshot, Store, StoreItem,
    };
    pub use crate::profiles::{AgentProfile, ProfileStore, UserProfile};
    pub use crate::skill_telemetry::{SkillExecutionRecord, SkillTelemetry, SkillTelemetryStore};
    /* 类型化记忆的元数据类型。它们定义在 echo_core::memory 中，但在此处
    重导出，使下游产品（例如 echo-agent-app-core 的 TaskRuntime 记忆桥接）
    可以通过 echo_agent 门面访问，而无需直接依赖 echo_core —— 保持门面
    作为唯一的集成入口。 */
    pub use echo_core::memory::{
        MemoryMeta, MemoryRisk, MemoryScope, MemorySource, MemoryStatus, MemoryType,
    };

    // 技能
    #[cfg(feature = "files")]
    pub use crate::skills::builtin::FileSystemSkill;
    #[cfg(feature = "shell")]
    pub use crate::skills::builtin::ShellSkill;
    pub use crate::skills::{
        Skill, SkillInfo, SkillRegistry,
        external::{
            ActivateSkillTool, DiscoveryScope, PromptContext, ReadSkillResourceTool,
            RunSkillScriptTool, SkillContent, SkillDescriptor, SkillDocument, SkillLoadPolicy,
            SkillLoader, SkillResourceEntry, SkillResourceKind, SkillSource,
        },
        hooks::{
            CompressHookStats, HookAction, HookContext, HookEvent, HookEventCategory, HookRegistry,
            HookResult, HookRule, HookSource, HooksDefinition, McpExecutorFn,
            UnifiedHookExecutorFn,
        },
    };

    // 内容守卫
    #[cfg(feature = "content-guard")]
    pub use crate::guard::llm::LlmGuard;
    #[cfg(feature = "content-guard")]
    pub use crate::guard::rule::{RuleGuard, RuleGuardBuilder};
    pub use crate::guard::{Guard, GuardDirection, GuardManager, GuardResult};

    // 审计
    pub use crate::audit::{
        AuditCallback, AuditEvent, AuditEventType, AuditFilter, AuditLogger, FileAuditLogger,
        InMemoryAuditLogger,
    };

    // 工作流
    pub use crate::workflow::{
        ConcurrentWorkflow, DagWorkflow, DataPipelineConfig, DataPipelineLanguage, Graph,
        GraphBuilder, GraphResult, SequentialWorkflow, SharedAgent, SharedState, StepOutput,
        Workflow, WorkflowDefinition, WorkflowEvent, WorkflowOutput, WritingPipelineConfig,
        run_data_pipeline, run_writing_pipeline, shared_agent,
    };

    // 沙箱
    pub use crate::sandbox::{
        DockerSandbox, ExecutionResult as SandboxResult, IsolationLevel, K8sSandbox, LocalSandbox,
        ResourceLimits, SandboxCommand, SandboxExecutor, SandboxManager, SandboxOutputChannel,
        SandboxPolicy, SandboxStreamEvent, SandboxStreamFailure, SecurityLevel,
    };

    // 熔断器
    pub use echo_core::circuit_breaker::{CircuitBreaker, CircuitBreakerConfig};

    // 重试
    pub use crate::retry::{RetryPolicy, with_retry, with_retry_if};

    // 错误
    pub use crate::error::Result;

    // 无头运行
    pub use crate::headless::{
        HeadlessConfig, HeadlessResult, HeadlessRunHandle, run_headless, start_headless,
    };

    // 执行追踪
    pub use crate::trace::{
        ErrorPattern, InMemoryRunStore, JsonlRunStore, Run, RunEvent, RunStatus, RunStore,
        RunSummary, SessionSummary, TokenBreakdown, ToolUsageStats, TraceAnalyzer,
    };

    // 测试辅助
    #[cfg(any(test, feature = "testing"))]
    pub use crate::testing::{FailingMockAgent, MockAgent, MockEmbedder, MockLlmClient, MockTool};
}

/// 可选模块的高级类型重导出（需启用对应 feature）。
pub mod advanced {
    #[cfg(feature = "human-loop")]
    #[cfg_attr(docsrs, doc(cfg(feature = "human-loop")))]
    pub use crate::human_loop::{
        ApprovalDecision, ApprovalResponder, ConsoleHumanLoopProvider, HumanLoopEvent,
        HumanLoopHandler, HumanLoopManager, HumanLoopProvider, HumanLoopRequest, HumanLoopResponse,
        InputResponder, WebSocketHumanLoopProvider, WebhookHumanLoopProvider, dispatch_event,
    };

    #[cfg(feature = "mcp")]
    #[cfg_attr(docsrs, doc(cfg(feature = "mcp")))]
    pub use crate::mcp::{
        McpManager, McpServerConfig, McpServerId, McpServerOwner, McpTargetChange,
        McpTargetReceipt, McpTool, TransportConfig,
    };

    #[cfg(feature = "channels")]
    #[cfg_attr(docsrs, doc(cfg(feature = "channels")))]
    pub use crate::channels::AgentChannelHandler;

    #[cfg(feature = "telemetry")]
    #[cfg_attr(docsrs, doc(cfg(feature = "telemetry")))]
    pub use crate::telemetry::{Metrics, TelemetryConfig, init_telemetry, shutdown_telemetry};

    #[cfg(feature = "a2a")]
    #[cfg_attr(docsrs, doc(cfg(feature = "a2a")))]
    pub use crate::a2a::{
        A2AClient, A2AServer, A2AStreamEvent, AgentCapabilities, AgentCard, AgentProvider,
        AgentSkill, JwtClaims, JwtConfig, JwtConfigError, TaskState, get_claims, serve,
        serve_with_auth,
    };

    #[cfg(feature = "topology")]
    #[cfg_attr(docsrs, doc(cfg(feature = "topology")))]
    pub use crate::topology::{
        NodeType, TopologyCallback, TopologyData, TopologyEdge, TopologyNode, TopologyStats,
        TopologyTracker,
    };

    pub use crate::tasks::{
        Task, TaskCreateTool, TaskExecution, TaskListTool, TaskRevisionService, TaskSpec,
        TaskStatus, TaskUpdateTool,
    };

    // Critic 模块 —— 针对 agent 输出的评估与反馈工具
    pub use crate::agent::critic::{
        CompositeCritic, CompositeStrategy, Critic, Critique, CritiqueOutput, LlmCritic,
        ReviewTool, StaticCritic, ThresholdCritic, critique_output_schema,
    };
}
