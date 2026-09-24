//! Hook 类型定义 —— 整个工作区 crate 共享。
//!
//! 本模块包含统一 hook 系统的核心类型，之所以定义在 `echo-core` 中，
//! 是为了让 `echo-execution` 和 `echo-orchestration` 都能引用这些类型，
//! 而不产生循环依赖。
//!
//! 执行引擎（`HookRegistry`、`parse_hook_output` 等）仍位于
//! `echo-execution/src/skills/hooks.rs`。

use crate::tools::permission::{PermissionDecision, PermissionMode};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

// ── Hook 事件类别 ────────────────────────────────────────────────

/// Hook 事件的类别，用于确定 matcher（匹配器）的语义。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HookEventCategory {
    /// 工具执行事件 —— matcher 按工具名过滤。
    Tool,
    /// 会话生命周期事件 —— matcher 按生命周期提示过滤。
    Lifecycle,
    /// Subagent 派发事件 —— matcher 按 subagent 类型/名称过滤。
    Subagent,
    /// 任务执行事件 —— matcher 按任务主题/名称过滤。
    Task,
    /// 错误/失败事件 —— 通常不支持 matcher。
    Error,
    /// 演化/记忆生命周期事件 —— matcher 按记忆来源或层级过滤。
    Evolution,
}

// ── Subagent 终止状态 ────────────────────────────────────────────────

/** Subagent 运行的终止状态，随 `SubagentStop` 事件一起携带。

业界共识（Claude Code / Codex / OpenAI Agents SDK / AGTP）：都收敛到
「两个边界事件（Start/Stop）+ 终止事件上的一个状态枚举」，而不是为每个终止态
（Stop/Cancelled/Failed/TimedOut）单独设一个事件。每一次具体的派发尝试都会发出
一对 Start/Stop；重试则是一次拥有自己 Start/Stop 对的新尝试。

按此模型，`SubagentCancelled`（原先独立的事件）已被移除 —— cancelled 现在是
`SubagentStop` 的一个状态值，而不是单独的事件。
*/
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SubagentStopStatus {
    /// Subagent 正常结束并产出结果。
    Completed,
    /// Subagent 以错误退出 / 报告了失败。
    Failed,
    /// Subagent 被取消（用户中止、父 run 被取消等）。
    Cancelled,
    /// Subagent 触发了截止时间 / 超时。
    TimedOut,
}

impl SubagentStopStatus {
    /// 稳定的线上名称（用于序列化的 hook payload 和日志）。
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
            Self::TimedOut => "timed_out",
        }
    }
}

// ── 任务终止状态 ────────────────────────────────────────────────

/** PlanTask 节点的终止状态，随 `TaskCompleted` 事件一起携带。

业界共识：timeout/cancelled 是终止事件上的状态值，而不是独立的事件类型
（Codex `CommandExecutionStatus`；这避免了「没有承载对象」的问题 ——
`TodoStatus` 没有 Timeout/Cancelled 变体，因此在 PlanTask 层触发它们本身就是
一种类型错误）。这些值与宿主应用 `TodoStatus` 的终止态一一对应，
另外为因截止时间而结束的运行增加了一个 TimedOut 原因。
*/
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskTerminalStatus {
    /// PlanTask 成功完成（执行/评审/集成各关卡均已通过）。
    Completed,
    /// PlanTask 终止性失败（重试预算耗尽或硬失败）。
    Failed,
    /// PlanTask 被取消（父 run 在其运行前就被取消）。
    Cancelled,
    /// PlanTask 触发了截止时间 / 超时。
    TimedOut,
    /// 因上游依赖无法完成而跳过该 PlanTask。
    Skipped,
}

impl TaskTerminalStatus {
    /// 稳定的线上名称（序列化 payload + 日志）。
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
            Self::TimedOut => "timed_out",
            Self::Skipped => "skipped",
        }
    }
}

// ── Hook 事件 ─────────────────────────────────────────────────────────

/** Hook 在 agent 生命周期中触发的时机。

## 事件类别

| 类别 | 事件 | Matcher 语义 |
|----------|--------|-------------------|
| Tool | `PreToolUse`, `PostToolUse`, `PostToolUseFailure`, `PermissionRequest`, `PermissionDenied` | 工具名（精确匹配、glob、用 `|` 分隔） |
| Lifecycle | `SessionStart`, `SessionEnd`, `Stop`, `Notification`, `UserPromptSubmit`, `PreCompact`, `PostCompact`, `ConfigChange`, `InstructionsLoaded`, `PostToolBatch` | 生命周期提示（如 "startup"、"permission_prompt"） |
| Subagent | `SubagentStart`, `SubagentStop` | Subagent 类型/名称。`SubagentStop` 携带一个 `subagent_stop_status`（completed/failed/cancelled/timed_out）；不存在单独的 `SubagentCancelled` 事件。 |
| Task | `TaskCreated`, `TaskStarted`, `TaskCompleted` | 任务主题/名称。`TaskCompleted` 携带终止状态。 |
| Error | `StopFailure` | 不支持 |
| Evolution | `PostMemoryWrite`, `MemoryLayerChange` | 记忆来源或层级名称 |
*/
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum HookEvent {
    // ── 工具执行事件 ──
    /// 工具执行前。可以阻断或修改输入。
    PreToolUse,
    /// 工具执行成功后。
    PostToolUse,
    /// 工具执行失败后。
    PostToolUseFailure,
    /// 权限对话框出现时。可以自动批准/拒绝。
    PermissionRequest,
    /// 工具被拒绝权限后。可以指示重试。
    PermissionDenied,

    // ── 会话生命周期事件 ──
    /// 会话开始或恢复。
    SessionStart,
    /// 会话终止。
    SessionEnd,
    /// Agent 完成响应。可以请求继续。
    Stop,
    /// Agent 需要用户关注。
    Notification,
    /// 用户提交了一个 prompt。可以注入上下文或阻断。
    UserPromptSubmit,
    /// 上下文压缩之前。
    PreCompact,
    /// 上下文压缩之后。
    PostCompact,
    /// 配置文件发生变化。
    ConfigChange,
    /// Skills/instructions 已加载。可用于加载后校验。
    InstructionsLoaded,
    /// 一批并行工具调用全部结束后。用于聚合处理。
    PostToolBatch,

    // ── Subagent 事件 ──
    /// Subagent 派发之前。
    SubagentStart,
    /// Subagent 完成后（无论成功或失败）。
    SubagentStop,

    // ── Task 事件 ──
    // 与业界对齐的三阶段任务生命周期（Codex item.started/
    // item.completed；Cursor plan-submit vs build-start；Claude Code
    // TaskCreated/TaskCompleted）。TaskCreated = 节点进入可执行图
    // （计划修订已提交）；TaskStarted = 调度器选中该节点，即将派发；
    // TaskCompleted = 节点达到终止状态，携带在 `task_terminal_status` 上。
    // Timeout/Cancelled 不是独立事件 —— 它们是 `TaskCompleted` 的状态值
    // （对齐 Codex CommandExecutionStatus 模型）。
    /// 任务节点进入可执行图（计划修订已提交）。
    TaskCreated,
    /// 调度器已选中该任务，即将派发执行。
    TaskStarted,
    /// 任务达到终止状态（见 `task_terminal_status`）。
    TaskCompleted,

    // ── 错误事件 ──
    /// Agent 遇到不可恢复的错误。
    StopFailure,

    // ── 插件生命周期事件 ──
    /// 插件已加载，其组件已注册。
    PluginLoaded,
    /// 插件已被禁用或卸载。
    PluginDisabled,

    // ── 扩展的任务事件 ──
    // TaskTimeout / TaskCancelled 已移除：终止原因现在是
    // `TaskCompleted` 的状态值（见 TaskTerminalStatus），而不是独立
    // 事件。与业界对齐（Codex CommandExecutionStatus；Claude Code
    // 没有 TaskStarted，但它的 TaskCompleted 对终止原因是无关的）。
    // 这避免了此前「没有承载对象」的问题 —— TodoStatus 没有
    // Timeout/Cancelled 变体，因此在 PlanTask 层触发它们本身就是一种
    // 类型错误；它们应属于终止事件的状态字段。

    // ── 扩展的 Subagent 事件 ──
    // SubagentCancelled 已移除：终止模型现在统一为
    // SubagentStop(status = completed|failed|cancelled|timed_out)，
    // 恰好发出一次。Cancelled 是 SubagentStop 的一个状态，而不是独立
    // 事件。（Claude Code / Codex / OpenAI Agents SDK / AGTP 均收敛到
    // 这种「两事件 + 状态枚举」模型。）

    // ── 演化事件 ──
    /// 任意一条记忆被持久化到 Store 之后。
    PostMemoryWrite,
    /// 记忆在层级之间被提升或降级之后。
    MemoryLayerChange,
    /// 从记忆模式中检测到 skill 候选之后。
    SkillCandidateDetected,
    /// skill 在生命周期状态之间转换之后。
    SkillLifecycleTransition,
    /// skill 健康检查完成之后。
    SkillHealthCheck,
    /// skill 补丁应用之后。
    SkillPatchApplied,
    /// 两个或多个 skill 合并之后。
    SkillMergeApplied,
    /// 一条记忆被提升为 AGENTS.md 中的规则之后。
    RulePromoted,
}

impl HookEvent {
    /// 按稳定展示顺序列出所有受支持的 hook 事件。
    pub const ALL: &'static [Self] = &[
        Self::PreToolUse,
        Self::PostToolUse,
        Self::PostToolUseFailure,
        Self::PermissionRequest,
        Self::PermissionDenied,
        Self::SessionStart,
        Self::SessionEnd,
        Self::Stop,
        Self::Notification,
        Self::UserPromptSubmit,
        Self::PreCompact,
        Self::PostCompact,
        Self::ConfigChange,
        Self::InstructionsLoaded,
        Self::PostToolBatch,
        Self::SubagentStart,
        Self::SubagentStop,
        Self::TaskCreated,
        Self::TaskStarted,
        Self::TaskCompleted,
        Self::StopFailure,
        Self::PluginLoaded,
        Self::PluginDisabled,
        Self::PostMemoryWrite,
        Self::MemoryLayerChange,
        Self::SkillCandidateDetected,
        Self::SkillLifecycleTransition,
        Self::SkillHealthCheck,
        Self::SkillPatchApplied,
        Self::SkillMergeApplied,
        Self::RulePromoted,
    ];

    /// 事件类别 —— 决定 matcher 语义与路由方式。
    pub fn category(self) -> HookEventCategory {
        match self {
            HookEvent::PreToolUse
            | HookEvent::PostToolUse
            | HookEvent::PostToolUseFailure
            | HookEvent::PermissionRequest
            | HookEvent::PermissionDenied => HookEventCategory::Tool,

            HookEvent::SubagentStart | HookEvent::SubagentStop => HookEventCategory::Subagent,

            HookEvent::TaskCreated | HookEvent::TaskStarted | HookEvent::TaskCompleted => {
                HookEventCategory::Task
            }

            HookEvent::StopFailure => HookEventCategory::Error,

            HookEvent::SessionStart
            | HookEvent::SessionEnd
            | HookEvent::Stop
            | HookEvent::Notification
            | HookEvent::UserPromptSubmit
            | HookEvent::PreCompact
            | HookEvent::PostCompact
            | HookEvent::ConfigChange
            | HookEvent::InstructionsLoaded
            | HookEvent::PostToolBatch
            | HookEvent::PluginLoaded
            | HookEvent::PluginDisabled => HookEventCategory::Lifecycle,

            HookEvent::PostMemoryWrite
            | HookEvent::MemoryLayerChange
            | HookEvent::SkillCandidateDetected
            | HookEvent::SkillLifecycleTransition
            | HookEvent::SkillHealthCheck
            | HookEvent::SkillPatchApplied
            | HookEvent::SkillMergeApplied
            | HookEvent::RulePromoted => HookEventCategory::Evolution,
        }
    }

    /// 该事件是否为工具执行事件（使用 tool_name 作为 matcher）。
    pub fn is_tool_event(self) -> bool {
        self.category() == HookEventCategory::Tool
    }

    /// 该事件是否支持 matcher 字段。
    pub fn supports_matcher(self) -> bool {
        matches!(
            self.category(),
            HookEventCategory::Tool
                | HookEventCategory::Lifecycle
                | HookEventCategory::Subagent
                | HookEventCategory::Task
                | HookEventCategory::Evolution
        )
    }

    /// 以静态字符串返回 PascalCase 形式的事件名（不分配内存）。
    ///
    /// 在热路径上优先使用它，而不是 `format!("{:?}", event)`。
    pub const fn as_str(self) -> &'static str {
        match self {
            HookEvent::PreToolUse => "PreToolUse",
            HookEvent::PostToolUse => "PostToolUse",
            HookEvent::PostToolUseFailure => "PostToolUseFailure",
            HookEvent::PermissionRequest => "PermissionRequest",
            HookEvent::PermissionDenied => "PermissionDenied",
            HookEvent::SessionStart => "SessionStart",
            HookEvent::SessionEnd => "SessionEnd",
            HookEvent::Stop => "Stop",
            HookEvent::Notification => "Notification",
            HookEvent::UserPromptSubmit => "UserPromptSubmit",
            HookEvent::PreCompact => "PreCompact",
            HookEvent::PostCompact => "PostCompact",
            HookEvent::ConfigChange => "ConfigChange",
            HookEvent::InstructionsLoaded => "InstructionsLoaded",
            HookEvent::PostToolBatch => "PostToolBatch",
            HookEvent::SubagentStart => "SubagentStart",
            HookEvent::SubagentStop => "SubagentStop",
            HookEvent::TaskCreated => "TaskCreated",
            HookEvent::TaskStarted => "TaskStarted",
            HookEvent::TaskCompleted => "TaskCompleted",
            HookEvent::StopFailure => "StopFailure",
            HookEvent::PluginLoaded => "PluginLoaded",
            HookEvent::PluginDisabled => "PluginDisabled",
            HookEvent::PostMemoryWrite => "PostMemoryWrite",
            HookEvent::MemoryLayerChange => "MemoryLayerChange",
            HookEvent::SkillCandidateDetected => "SkillCandidateDetected",
            HookEvent::SkillLifecycleTransition => "SkillLifecycleTransition",
            HookEvent::SkillHealthCheck => "SkillHealthCheck",
            HookEvent::SkillPatchApplied => "SkillPatchApplied",
            HookEvent::SkillMergeApplied => "SkillMergeApplied",
            HookEvent::RulePromoted => "RulePromoted",
        }
    }

    /// 解析 hook 配置和用户可见的测试命令所使用的规范 PascalCase 事件名。
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "PreToolUse" => Some(Self::PreToolUse),
            "PostToolUse" => Some(Self::PostToolUse),
            "PostToolUseFailure" => Some(Self::PostToolUseFailure),
            "PermissionRequest" => Some(Self::PermissionRequest),
            "PermissionDenied" => Some(Self::PermissionDenied),
            "SessionStart" => Some(Self::SessionStart),
            "SessionEnd" => Some(Self::SessionEnd),
            "Stop" => Some(Self::Stop),
            "Notification" => Some(Self::Notification),
            "UserPromptSubmit" => Some(Self::UserPromptSubmit),
            "PreCompact" => Some(Self::PreCompact),
            "PostCompact" => Some(Self::PostCompact),
            "ConfigChange" => Some(Self::ConfigChange),
            "InstructionsLoaded" => Some(Self::InstructionsLoaded),
            "PostToolBatch" => Some(Self::PostToolBatch),
            "SubagentStart" => Some(Self::SubagentStart),
            "SubagentStop" => Some(Self::SubagentStop),
            "TaskCreated" => Some(Self::TaskCreated),
            "TaskStarted" => Some(Self::TaskStarted),
            "TaskCompleted" => Some(Self::TaskCompleted),
            "StopFailure" => Some(Self::StopFailure),
            "PluginLoaded" => Some(Self::PluginLoaded),
            "PluginDisabled" => Some(Self::PluginDisabled),
            "PostMemoryWrite" => Some(Self::PostMemoryWrite),
            "MemoryLayerChange" => Some(Self::MemoryLayerChange),
            "SkillCandidateDetected" => Some(Self::SkillCandidateDetected),
            "SkillLifecycleTransition" => Some(Self::SkillLifecycleTransition),
            "SkillHealthCheck" => Some(Self::SkillHealthCheck),
            "SkillPatchApplied" => Some(Self::SkillPatchApplied),
            "SkillMergeApplied" => Some(Self::SkillMergeApplied),
            "RulePromoted" => Some(Self::RulePromoted),
            _ => None,
        }
    }
}

// ── 压缩统计 ──────────────────────────────────────────────────

/// 传递给 PreCompact/PostCompact hook 的压缩统计信息。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CompressHookStats {
    /// 压缩前的消息数。
    pub before_count: usize,
    /// 压缩后的消息数。
    pub after_count: usize,
    /// 压缩前的估算 token 数。
    pub before_tokens: usize,
    /// 压缩后的估算 token 数。
    pub after_tokens: usize,
}

// ── Hook 上下文 ────────────────────────────────────────────────────────

/// 传递给 hook 执行的上下文数据，随事件类型不同而不同。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HookContext {
    /// 触发本次 hook 的事件。
    pub event: HookEvent,
    /// 会话 ID。
    pub session_id: String,
    /// Agent 名称。
    pub agent_name: String,
    /// 事件发生时的工作目录。
    pub cwd: String,

    // ── 工具事件字段 ──
    /// 工具名（PreToolUse、PostToolUse、PostToolUseFailure、PermissionRequest、PermissionDenied）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,
    /// 工具输入（PreToolUse、PostToolUse、PostToolUseFailure、PermissionRequest、PermissionDenied）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_input: Option<Value>,
    /// 工具输出（仅 PostToolUse）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_output: Option<String>,
    /// 工具错误信息（仅 PostToolUseFailure）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_error: Option<String>,

    // ── PermissionDenied 字段 ──
    /// 权限被拒绝的原因（仅 PermissionDenied）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub denial_reason: Option<String>,
    /// 模型是否可以用不同参数重试（仅 PermissionDenied）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retry_allowed: Option<bool>,

    // ── 生命周期事件字段 ──
    /// 生命周期事件的 matcher 提示。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub matcher: Option<String>,
    /// 用户 prompt 文本（仅 UserPromptSubmit）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    /// 压缩统计信息（仅 PreCompact、PostCompact）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compress_stats: Option<CompressHookStats>,
    /// 变化的配置文件路径（仅 ConfigChange）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub config_path: Option<String>,
    /// 已加载的 skill 名称（仅 InstructionsLoaded）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub skill_names: Option<Vec<String>>,

    // ── PostToolBatch 字段 ──
    /// 本批次中的工具名（仅 PostToolBatch）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub batch_tool_names: Option<Vec<String>>,
    /// 批次中成功的工具数量（仅 PostToolBatch）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub batch_success_count: Option<usize>,
    /// 批次中失败的工具数量（仅 PostToolBatch）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub batch_failure_count: Option<usize>,

    // ── Subagent 字段 ──
    /// Subagent 名称/类型（仅 SubagentStart、SubagentStop）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subagent_name: Option<String>,
    /// Subagent 执行模式（仅 SubagentStart、SubagentStop）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subagent_mode: Option<String>,
    /// Subagent 任务描述（仅 SubagentStart）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subagent_task: Option<String>,
    /// Subagent 结果摘要（仅 SubagentStop）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subagent_result: Option<String>,
    /// Subagent 运行的终止状态（仅 SubagentStop）。
    ///
    /// 在 SubagentStop 上下文中始终存在。使消费者无需解析自由文本结果，
    /// 即可区分 completed/failed/cancelled/timed_out。与业界对齐
    /// （见 [`SubagentStopStatus`]）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subagent_stop_status: Option<SubagentStopStatus>,

    // ── 任务字段 ──
    /// 任务 ID（TaskCreated、TaskStarted、TaskCompleted）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_id: Option<String>,
    /// 任务主题/名称（TaskCreated、TaskStarted、TaskCompleted）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_subject: Option<String>,
    /// 任务结果摘要（仅 TaskCompleted）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_result: Option<String>,
    /// PlanTask 节点的终止状态（仅 TaskCompleted）。
    ///
    /// 在 TaskCompleted 上下文中始终存在。使消费者无需解析自由文本，
    /// 即可区分 completed/failed/cancelled/timed_out/skipped。
    /// 取代了此前独立的 TaskTimeout/TaskCancelled 事件。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_terminal_status: Option<TaskTerminalStatus>,

    // ── Run / 计划关联字段 ──
    // 这些字段让 hook 能把一次 Task/Subagent 事件关联回所属的
    // TaskRun、计划修订版本，以及具体的尝试次数。它们是可选的，
    // 因为定义 HookContext 的框架层并不总能知道这些值 ——
    // 由应用运行时（例如宿主应用的 task_runtime）在触发时负责填充。
    /// 所属 TaskRun ID（已知时用于 Task/Subagent 事件）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub run_id: Option<String>,
    /// 事件所属的计划修订版本（已知时用于 Task/Subagent 事件）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub plan_revision: Option<String>,
    /// SubagentRun ID（仅 Subagent 事件）：`{run_id}:{task_id}:{rev}:{attempt}`。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subagent_run_id: Option<String>,
    /// 该任务对应 SubagentRun 的尝试序号，从 1 开始（Subagent 事件）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attempt: Option<u32>,

    // ── StopFailure 字段 ──
    /// 错误信息（仅 StopFailure）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub failure_error: Option<String>,
    /// 错误类别："max_iterations"、"api_error"、"tool_batch_timeout" 等。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub failure_category: Option<String>,

    // ── Stop hook 防循环 ──
    /// 该 Stop hook 是否为被重新触发的调用（用于防止无限循环）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_hook_active: Option<bool>,

    // ── 演化事件字段 ──
    /// 被写入的记忆键（仅 PostMemoryWrite）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_key: Option<String>,
    /// 触发写入的记忆来源（仅 PostMemoryWrite）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_source: Option<String>,
    /// 记忆被提升/降级的来源层级（仅 MemoryLayerChange）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_from_layer: Option<String>,
    /// 记忆被提升/降级的目标层级（仅 MemoryLayerChange）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_to_layer: Option<String>,
}

impl Default for HookContext {
    fn default() -> Self {
        Self {
            event: HookEvent::PreToolUse, // 任意占位值，总会被工厂方法覆盖
            session_id: String::new(),
            agent_name: String::new(),
            cwd: HookContext::cwd_default(),
            tool_name: None,
            tool_input: None,
            tool_output: None,
            tool_error: None,
            denial_reason: None,
            retry_allowed: None,
            matcher: None,
            prompt: None,
            compress_stats: None,
            config_path: None,
            skill_names: None,
            batch_tool_names: None,
            batch_success_count: None,
            batch_failure_count: None,
            subagent_name: None,
            subagent_mode: None,
            subagent_task: None,
            subagent_result: None,
            subagent_stop_status: None,
            task_id: None,
            task_subject: None,
            task_result: None,
            task_terminal_status: None,
            run_id: None,
            plan_revision: None,
            subagent_run_id: None,
            attempt: None,
            failure_error: None,
            failure_category: None,
            stop_hook_active: None,
            memory_key: None,
            memory_source: None,
            memory_from_layer: None,
            memory_to_layer: None,
        }
    }
}

impl HookContext {
    /// 返回默认的当前工作目录字符串（失败时回退为空字符串）。
    fn cwd_default() -> String {
        std::env::current_dir()
            .map(|p| p.display().to_string())
            .unwrap_or_default()
    }

    /// 构建一个仅含 matcher、无副作用的上下文，供 hook 干跑（dry run）使用。
    pub fn for_dry_run(event: HookEvent, matcher: &str) -> Self {
        if event.is_tool_event() {
            Self {
                event,
                tool_name: Some(matcher.to_string()),
                tool_input: Some(serde_json::json!({})),
                ..Self::default()
            }
        } else {
            Self::for_lifecycle(event, matcher, "dry-run", "dry-run")
        }
    }

    // ── 工具事件的工厂方法 ──

    /// 构建 PreToolUse 事件的上下文。
    pub fn for_pre_tool_use(
        tool_name: &str,
        tool_input: &Value,
        session_id: &str,
        agent_name: &str,
    ) -> Self {
        Self {
            event: HookEvent::PreToolUse,
            session_id: session_id.to_string(),
            agent_name: agent_name.to_string(),
            tool_name: Some(tool_name.to_string()),
            tool_input: Some(tool_input.clone()),
            ..Self::default()
        }
    }

    /// 构建 PostToolUse 事件的上下文。
    pub fn for_post_tool_use(
        tool_name: &str,
        tool_input: &Value,
        tool_output: &str,
        session_id: &str,
        agent_name: &str,
    ) -> Self {
        Self {
            event: HookEvent::PostToolUse,
            session_id: session_id.to_string(),
            agent_name: agent_name.to_string(),
            tool_name: Some(tool_name.to_string()),
            tool_input: Some(tool_input.clone()),
            tool_output: Some(tool_output.to_string()),
            ..Self::default()
        }
    }

    /// 构建 PostToolUseFailure 事件的上下文。
    pub fn for_post_tool_use_failure(
        tool_name: &str,
        tool_input: &Value,
        tool_error: &str,
        session_id: &str,
        agent_name: &str,
    ) -> Self {
        Self {
            event: HookEvent::PostToolUseFailure,
            session_id: session_id.to_string(),
            agent_name: agent_name.to_string(),
            tool_name: Some(tool_name.to_string()),
            tool_input: Some(tool_input.clone()),
            tool_error: Some(tool_error.to_string()),
            ..Self::default()
        }
    }

    /// 构建 PermissionRequest 事件的上下文。
    pub fn for_permission_request(
        tool_name: &str,
        tool_input: &Value,
        session_id: &str,
        agent_name: &str,
    ) -> Self {
        Self {
            event: HookEvent::PermissionRequest,
            session_id: session_id.to_string(),
            agent_name: agent_name.to_string(),
            tool_name: Some(tool_name.to_string()),
            tool_input: Some(tool_input.clone()),
            ..Self::default()
        }
    }

    /// 构建 PermissionDenied 事件的上下文。
    pub fn for_permission_denied(
        tool_name: &str,
        tool_input: &Value,
        denial_reason: &str,
        retry_allowed: bool,
        session_id: &str,
        agent_name: &str,
    ) -> Self {
        Self {
            event: HookEvent::PermissionDenied,
            session_id: session_id.to_string(),
            agent_name: agent_name.to_string(),
            tool_name: Some(tool_name.to_string()),
            tool_input: Some(tool_input.clone()),
            denial_reason: Some(denial_reason.to_string()),
            retry_allowed: Some(retry_allowed),
            ..Self::default()
        }
    }

    // ── 生命周期事件的工厂方法 ──

    /// 构建 SessionStart 事件的上下文。
    pub fn for_session_start(matcher: &str, session_id: &str, agent_name: &str) -> Self {
        Self {
            event: HookEvent::SessionStart,
            session_id: session_id.to_string(),
            agent_name: agent_name.to_string(),
            matcher: Some(matcher.to_string()),
            ..Self::default()
        }
    }

    /// 构建 SessionEnd 事件的上下文。
    pub fn for_session_end(matcher: &str, session_id: &str, agent_name: &str) -> Self {
        Self {
            event: HookEvent::SessionEnd,
            session_id: session_id.to_string(),
            agent_name: agent_name.to_string(),
            matcher: Some(matcher.to_string()),
            ..Self::default()
        }
    }

    /// 没有专用工厂方法的生命周期事件的通用构造函数。
    pub fn for_lifecycle(
        event: HookEvent,
        matcher: &str,
        session_id: &str,
        agent_name: &str,
    ) -> Self {
        Self {
            event,
            session_id: session_id.to_string(),
            agent_name: agent_name.to_string(),
            matcher: Some(matcher.to_string()),
            ..Self::default()
        }
    }

    /// 构建 Stop 事件的上下文。
    pub fn for_stop(
        matcher: Option<&str>,
        session_id: &str,
        agent_name: &str,
        stop_hook_active: bool,
    ) -> Self {
        Self {
            event: HookEvent::Stop,
            session_id: session_id.to_string(),
            agent_name: agent_name.to_string(),
            matcher: matcher.map(|s| s.to_string()),
            stop_hook_active: Some(stop_hook_active),
            ..Self::default()
        }
    }

    /// 构建 Notification 事件的上下文。
    pub fn for_notification(matcher: &str, session_id: &str, agent_name: &str) -> Self {
        Self {
            event: HookEvent::Notification,
            session_id: session_id.to_string(),
            agent_name: agent_name.to_string(),
            matcher: Some(matcher.to_string()),
            ..Self::default()
        }
    }

    /// 构建 UserPromptSubmit 事件的上下文。
    pub fn for_user_prompt_submit(
        prompt: &str,
        matcher: Option<&str>,
        session_id: &str,
        agent_name: &str,
    ) -> Self {
        Self {
            event: HookEvent::UserPromptSubmit,
            session_id: session_id.to_string(),
            agent_name: agent_name.to_string(),
            matcher: matcher.map(|s| s.to_string()),
            prompt: Some(prompt.to_string()),
            ..Self::default()
        }
    }

    /// 构建 PreCompact 事件的上下文。
    pub fn for_pre_compact(
        stats: &CompressHookStats,
        matcher: &str,
        session_id: &str,
        agent_name: &str,
    ) -> Self {
        Self {
            event: HookEvent::PreCompact,
            session_id: session_id.to_string(),
            agent_name: agent_name.to_string(),
            matcher: Some(matcher.to_string()),
            compress_stats: Some(stats.clone()),
            ..Self::default()
        }
    }

    /// 构建 PostCompact 事件的上下文。
    pub fn for_post_compact(
        stats: &CompressHookStats,
        matcher: &str,
        session_id: &str,
        agent_name: &str,
    ) -> Self {
        Self {
            event: HookEvent::PostCompact,
            session_id: session_id.to_string(),
            agent_name: agent_name.to_string(),
            matcher: Some(matcher.to_string()),
            compress_stats: Some(stats.clone()),
            ..Self::default()
        }
    }

    /// 构建 ConfigChange 事件的上下文。
    pub fn for_config_change(
        config_path: &str,
        matcher: Option<&str>,
        session_id: &str,
        agent_name: &str,
    ) -> Self {
        Self {
            event: HookEvent::ConfigChange,
            session_id: session_id.to_string(),
            agent_name: agent_name.to_string(),
            matcher: matcher.map(|s| s.to_string()),
            config_path: Some(config_path.to_string()),
            ..Self::default()
        }
    }

    /// 构建 InstructionsLoaded 事件的上下文。
    pub fn for_instructions_loaded(
        matcher: &str,
        skill_names: &[String],
        session_id: &str,
        agent_name: &str,
    ) -> Self {
        Self {
            event: HookEvent::InstructionsLoaded,
            session_id: session_id.to_string(),
            agent_name: agent_name.to_string(),
            matcher: Some(matcher.to_string()),
            skill_names: Some(skill_names.to_vec()),
            ..Self::default()
        }
    }

    // ── 聚合事件的工厂方法 ──

    /// 构建 PostToolBatch 事件的上下文。
    pub fn for_post_tool_batch(
        batch_tool_names: &[String],
        success_count: usize,
        failure_count: usize,
        session_id: &str,
        agent_name: &str,
    ) -> Self {
        Self {
            event: HookEvent::PostToolBatch,
            session_id: session_id.to_string(),
            agent_name: agent_name.to_string(),
            matcher: Some(batch_tool_names.join("|")),
            batch_tool_names: Some(batch_tool_names.to_vec()),
            batch_success_count: Some(success_count),
            batch_failure_count: Some(failure_count),
            ..Self::default()
        }
    }

    // ── Subagent 事件的工厂方法 ──

    /// 构建 SubagentStart 事件的上下文。
    pub fn for_subagent_start(
        subagent_name: &str,
        subagent_mode: &str,
        task: &str,
        session_id: &str,
        agent_name: &str,
    ) -> Self {
        Self {
            event: HookEvent::SubagentStart,
            session_id: session_id.to_string(),
            agent_name: agent_name.to_string(),
            matcher: Some(subagent_name.to_string()),
            subagent_name: Some(subagent_name.to_string()),
            subagent_mode: Some(subagent_mode.to_string()),
            subagent_task: Some(task.to_string()),
            ..Self::default()
        }
    }

    /// 构建 SubagentStop 事件的上下文。
    pub fn for_subagent_stop(
        subagent_name: &str,
        subagent_mode: &str,
        result: &str,
        status: SubagentStopStatus,
        session_id: &str,
        agent_name: &str,
    ) -> Self {
        Self {
            event: HookEvent::SubagentStop,
            session_id: session_id.to_string(),
            agent_name: agent_name.to_string(),
            matcher: Some(subagent_name.to_string()),
            subagent_name: Some(subagent_name.to_string()),
            subagent_mode: Some(subagent_mode.to_string()),
            subagent_result: Some(result.to_string()),
            subagent_stop_status: Some(status),
            ..Self::default()
        }
    }

    /// 为该上下文附加可选的 TaskRun/SubagentRun 标识信息。
    ///
    /// 纯框架层的派发可能无法获知每一个标识符。应用运行时应尽量填充所有可
    /// 得字段，而不是把它们编码进 matcher 或 result 字符串里。
    pub fn with_run_correlation(
        mut self,
        run_id: Option<&str>,
        plan_revision: Option<&str>,
        subagent_run_id: Option<&str>,
        attempt: Option<u32>,
    ) -> Self {
        self.run_id = run_id.map(str::to_string);
        self.plan_revision = plan_revision.map(str::to_string);
        self.subagent_run_id = subagent_run_id.map(str::to_string);
        self.attempt = attempt;
        self
    }

    // ── 任务事件的工厂方法 ──

    /// 构建 TaskCreated 事件的上下文。
    pub fn for_task_created(
        task_id: &str,
        task_subject: &str,
        session_id: &str,
        agent_name: &str,
    ) -> Self {
        Self {
            event: HookEvent::TaskCreated,
            session_id: session_id.to_string(),
            agent_name: agent_name.to_string(),
            matcher: Some(task_subject.to_string()),
            task_id: Some(task_id.to_string()),
            task_subject: Some(task_subject.to_string()),
            ..Self::default()
        }
    }

    /// 构建 TaskStarted 事件的上下文（调度器已选中该任务，即将派发）。
    pub fn for_task_started(
        task_id: &str,
        task_subject: &str,
        session_id: &str,
        agent_name: &str,
    ) -> Self {
        Self {
            event: HookEvent::TaskStarted,
            session_id: session_id.to_string(),
            agent_name: agent_name.to_string(),
            matcher: Some(task_subject.to_string()),
            task_id: Some(task_id.to_string()),
            task_subject: Some(task_subject.to_string()),
            ..Self::default()
        }
    }

    /// 构建 TaskCompleted 事件的上下文（终止状态）。
    ///
    /// `status` 是结构化的终止原因（completed/failed/cancelled/
    /// timed_out/skipped），取代了此前独立的 TaskTimeout/Cancelled 事件。
    pub fn for_task_completed(
        task_id: &str,
        task_subject: &str,
        result: &str,
        status: TaskTerminalStatus,
        session_id: &str,
        agent_name: &str,
    ) -> Self {
        Self {
            event: HookEvent::TaskCompleted,
            session_id: session_id.to_string(),
            agent_name: agent_name.to_string(),
            matcher: Some(task_subject.to_string()),
            task_id: Some(task_id.to_string()),
            task_subject: Some(task_subject.to_string()),
            task_result: Some(result.to_string()),
            task_terminal_status: Some(status),
            ..Self::default()
        }
    }

    // ── 错误事件的工厂方法 ──

    /// 构建 StopFailure 事件的上下文。
    pub fn for_stop_failure(
        failure_error: &str,
        failure_category: &str,
        session_id: &str,
        agent_name: &str,
    ) -> Self {
        Self {
            event: HookEvent::StopFailure,
            session_id: session_id.to_string(),
            agent_name: agent_name.to_string(),
            failure_error: Some(failure_error.to_string()),
            failure_category: Some(failure_category.to_string()),
            ..Self::default()
        }
    }

    // ── 演化事件的工厂方法 ──

    /// 构建 PostMemoryWrite 事件的上下文。
    pub fn for_post_memory_write(
        memory_key: &str,
        memory_source: &str,
        session_id: &str,
        agent_name: &str,
    ) -> Self {
        Self {
            event: HookEvent::PostMemoryWrite,
            session_id: session_id.to_string(),
            agent_name: agent_name.to_string(),
            matcher: Some(memory_source.to_string()),
            memory_key: Some(memory_key.to_string()),
            memory_source: Some(memory_source.to_string()),
            ..Self::default()
        }
    }

    /// 构建 MemoryLayerChange 事件的上下文。
    pub fn for_memory_layer_change(
        memory_key: &str,
        from_layer: &str,
        to_layer: &str,
        session_id: &str,
        agent_name: &str,
    ) -> Self {
        Self {
            event: HookEvent::MemoryLayerChange,
            session_id: session_id.to_string(),
            agent_name: agent_name.to_string(),
            matcher: Some(format!("{from_layer}->{to_layer}")),
            memory_key: Some(memory_key.to_string()),
            memory_from_layer: Some(from_layer.to_string()),
            memory_to_layer: Some(to_layer.to_string()),
            ..Self::default()
        }
    }
}

// ── Hook 结果 ─────────────────────────────────────────────────────────

/// 执行一次 hook 得到的结果。
#[derive(Debug, Clone, Default)]
pub struct HookResult {
    /// 若为 true，则应阻断该操作。
    pub block: bool,
    /// 阻断原因（当 block 为 true 时有效）。
    pub block_reason: Option<String>,
    /// 修改后的工具输入（仅 PreToolUse）。
    pub updated_input: Option<Value>,
    /// 要注入到上下文中的消息列表。
    pub messages: Vec<String>,
    /// 若为 true，阻止后续的 hook 继续执行。
    pub stop_propagation: bool,
    /// 权限决策（PreToolUse、PermissionRequest、Notification）。
    pub permission_decision: Option<PermissionDecision>,
    /// 权限模式覆盖值（PreToolUse、PermissionRequest）。
    pub permission_mode_override: Option<PermissionMode>,
    /// 用于 Stop 事件：若设置，agent 将继续而不是停止。
    pub continue_reason: Option<String>,
    /// 用于 UserPromptSubmit/SessionStart：要注入的额外上下文。
    pub injected_context: Option<String>,
    /// 用于 PermissionDenied：若为 true，模型可以用不同参数重试。
    pub retry: bool,
    /// 来自 hook 的任意元数据。
    pub metadata: Option<Value>,
    /// 用于 ActivateSkill hook：要直接激活的 skill（名称 + 原因）。
    /// 当 `ActivateSkill` 动作匹配时由 `execute_action` 填充；
    /// 由 `fire_lifecycle_hook` 消费，用于调用 `ReactAgent::activate_skill`。
    pub activate_skill: Option<(String, String)>,
}

impl HookResult {
    /// 创建一个「允许」结果。
    pub fn allow() -> Self {
        Self {
            permission_decision: Some(PermissionDecision::Allow),
            ..Self::default()
        }
    }

    /// 创建一个「拒绝」结果。
    pub fn deny(reason: String) -> Self {
        Self {
            block: true,
            block_reason: Some(reason.clone()),
            permission_decision: Some(PermissionDecision::Deny { reason }),
            ..Self::default()
        }
    }

    /// 创建一个「询问」结果。
    pub fn ask(suggestions: Vec<String>) -> Self {
        Self {
            permission_decision: Some(PermissionDecision::Ask { suggestions }),
            ..Self::default()
        }
    }

    /// 检查 hook 是否做出了权限决策。
    pub fn has_permission_decision(&self) -> bool {
        self.permission_decision.is_some()
    }

    /// 该结果是否指示 agent 应该继续工作。
    pub fn should_continue(&self) -> bool {
        self.continue_reason.is_some()
    }

    /// 创建一个请求直接激活 skill 的结果。
    pub fn with_activate_skill(skill: String, reason: String) -> Self {
        Self {
            activate_skill: Some((skill, reason)),
            ..Self::default()
        }
    }
}

// ── Hook 来源 ──────────────────────────────────────────────────────────

/// Hook 注册的来源。
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum HookSource {
    /// 来自基于文件的 skill 的 hook。
    Skill(String),
    /// 由宿主应用配置提供的 hook。
    UserConfig,
    /// 来自已安装插件的 hook。
    Plugin(String),
}

impl std::fmt::Display for HookSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HookSource::Skill(name) => write!(f, "skill:{}", name),
            HookSource::UserConfig => write!(f, "user_config"),
            HookSource::Plugin(name) => write!(f, "plugin:{}", name),
        }
    }
}

// ── 统一 Hook 执行器函数 ─────────────────────────────────────────────

/** 类型擦除的回调，用于从没有直接访问 `HookRegistry`（位于
`echo-execution` 中）能力的 orchestration/subagent 层触发统一 hook。

Agent 层在构造时注入此回调，以便带修订版本的任务运行时和
`SubagentExecutor` 能把 `TaskCreated`/`TaskCompleted` 和
`SubagentStart`/`SubagentStop` 事件送入统一系统。

处理 Lifecycle、Subagent 和 Task 事件（不仅限于 Lifecycle）。
*/
pub type UnifiedHookExecutorFn =
    Arc<dyn Fn(HookContext) -> Pin<Box<dyn Future<Output = HookResult> + Send>> + Send + Sync>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hook_event_names_round_trip_for_every_event() {
        for event in HookEvent::ALL {
            assert_eq!(HookEvent::from_name(event.as_str()), Some(*event));
        }
        assert_eq!(HookEvent::from_name("UnknownEvent"), None);
    }

    #[test]
    fn test_hook_event_category() {
        assert_eq!(HookEvent::PreToolUse.category(), HookEventCategory::Tool);
        assert_eq!(
            HookEvent::PermissionDenied.category(),
            HookEventCategory::Tool
        );
        assert_eq!(
            HookEvent::SessionStart.category(),
            HookEventCategory::Lifecycle
        );
        assert_eq!(
            HookEvent::PostToolBatch.category(),
            HookEventCategory::Lifecycle
        );
        assert_eq!(
            HookEvent::InstructionsLoaded.category(),
            HookEventCategory::Lifecycle
        );
        assert_eq!(
            HookEvent::SubagentStart.category(),
            HookEventCategory::Subagent
        );
        assert_eq!(
            HookEvent::SubagentStop.category(),
            HookEventCategory::Subagent
        );
        assert_eq!(HookEvent::TaskCreated.category(), HookEventCategory::Task);
        assert_eq!(HookEvent::TaskCompleted.category(), HookEventCategory::Task);
        assert_eq!(HookEvent::StopFailure.category(), HookEventCategory::Error);
        assert_eq!(
            HookEvent::PostMemoryWrite.category(),
            HookEventCategory::Evolution
        );
        assert_eq!(
            HookEvent::MemoryLayerChange.category(),
            HookEventCategory::Evolution
        );
    }

    #[test]
    fn test_hook_event_is_tool_event() {
        assert!(HookEvent::PreToolUse.is_tool_event());
        assert!(HookEvent::PostToolUse.is_tool_event());
        assert!(HookEvent::PermissionDenied.is_tool_event());
        assert!(!HookEvent::SessionStart.is_tool_event());
        assert!(!HookEvent::StopFailure.is_tool_event());
    }

    #[test]
    fn test_hook_event_supports_matcher() {
        assert!(HookEvent::PreToolUse.supports_matcher());
        assert!(HookEvent::SessionStart.supports_matcher());
        assert!(HookEvent::SubagentStart.supports_matcher());
        assert!(HookEvent::TaskCreated.supports_matcher());
        assert!(!HookEvent::StopFailure.supports_matcher());
    }

    #[test]
    fn test_hook_event_as_str() {
        assert_eq!(HookEvent::PreToolUse.as_str(), "PreToolUse");
        assert_eq!(HookEvent::PostToolUse.as_str(), "PostToolUse");
        assert_eq!(HookEvent::PostToolUseFailure.as_str(), "PostToolUseFailure");
        assert_eq!(HookEvent::PermissionRequest.as_str(), "PermissionRequest");
        assert_eq!(HookEvent::PermissionDenied.as_str(), "PermissionDenied");
        assert_eq!(HookEvent::SessionStart.as_str(), "SessionStart");
        assert_eq!(HookEvent::SessionEnd.as_str(), "SessionEnd");
        assert_eq!(HookEvent::Stop.as_str(), "Stop");
        assert_eq!(HookEvent::Notification.as_str(), "Notification");
        assert_eq!(HookEvent::UserPromptSubmit.as_str(), "UserPromptSubmit");
        assert_eq!(HookEvent::PreCompact.as_str(), "PreCompact");
        assert_eq!(HookEvent::PostCompact.as_str(), "PostCompact");
        assert_eq!(HookEvent::ConfigChange.as_str(), "ConfigChange");
        assert_eq!(HookEvent::InstructionsLoaded.as_str(), "InstructionsLoaded");
        assert_eq!(HookEvent::PostToolBatch.as_str(), "PostToolBatch");
        assert_eq!(HookEvent::SubagentStart.as_str(), "SubagentStart");
        assert_eq!(HookEvent::SubagentStop.as_str(), "SubagentStop");
        assert_eq!(HookEvent::TaskCreated.as_str(), "TaskCreated");
        assert_eq!(HookEvent::TaskCompleted.as_str(), "TaskCompleted");
        assert_eq!(HookEvent::StopFailure.as_str(), "StopFailure");
        assert_eq!(HookEvent::PostMemoryWrite.as_str(), "PostMemoryWrite");
        assert_eq!(HookEvent::MemoryLayerChange.as_str(), "MemoryLayerChange");
    }

    #[test]
    fn test_hook_event_serde() {
        let event = HookEvent::PermissionDenied;
        let json = serde_json::to_string(&event).unwrap();
        assert_eq!(json, "\"PermissionDenied\"");

        let event = HookEvent::SubagentStart;
        let json = serde_json::to_string(&event).unwrap();
        assert_eq!(json, "\"SubagentStart\"");

        let parsed: HookEvent = serde_json::from_str("\"TaskCompleted\"").unwrap();
        assert_eq!(parsed, HookEvent::TaskCompleted);
    }

    #[test]
    fn test_hook_context_for_permission_denied() {
        let ctx = HookContext::for_permission_denied(
            "Bash",
            &serde_json::json!({"command": "rm -rf /"}),
            "unsafe command",
            true,
            "sess-1",
            "agent",
        );
        assert_eq!(ctx.event, HookEvent::PermissionDenied);
        assert_eq!(ctx.tool_name.as_deref(), Some("Bash"));
        assert_eq!(ctx.denial_reason.as_deref(), Some("unsafe command"));
        assert_eq!(ctx.retry_allowed, Some(true));
    }

    #[test]
    fn test_hook_context_for_stop_failure() {
        let ctx = HookContext::for_stop_failure(
            "MaxIterationsExceeded",
            "max_iterations",
            "sess-1",
            "agent",
        );
        assert_eq!(ctx.event, HookEvent::StopFailure);
        assert_eq!(ctx.failure_error.as_deref(), Some("MaxIterationsExceeded"));
        assert_eq!(ctx.failure_category.as_deref(), Some("max_iterations"));
    }

    #[test]
    fn test_hook_context_for_post_tool_batch() {
        let names = vec!["Bash".to_string(), "Read".to_string()];
        let ctx = HookContext::for_post_tool_batch(&names, 1, 1, "sess-1", "agent");
        assert_eq!(ctx.event, HookEvent::PostToolBatch);
        assert_eq!(ctx.batch_tool_names.as_ref().unwrap().len(), 2);
        assert_eq!(ctx.batch_success_count, Some(1));
        assert_eq!(ctx.batch_failure_count, Some(1));
    }

    #[test]
    fn test_hook_context_for_subagent() {
        let ctx = HookContext::for_subagent_start(
            "coder",
            "sync",
            "implement feature X",
            "sess-1",
            "agent",
        );
        assert_eq!(ctx.event, HookEvent::SubagentStart);
        assert_eq!(ctx.subagent_name.as_deref(), Some("coder"));
        assert_eq!(ctx.subagent_mode.as_deref(), Some("sync"));
        assert_eq!(ctx.subagent_task.as_deref(), Some("implement feature X"));
    }

    #[test]
    fn test_hook_context_for_task() {
        let ctx = HookContext::for_task_created("t-1", "build API", "sess-1", "agent");
        assert_eq!(ctx.event, HookEvent::TaskCreated);
        assert_eq!(ctx.task_id.as_deref(), Some("t-1"));
        assert_eq!(ctx.task_subject.as_deref(), Some("build API"));
    }

    #[test]
    fn test_hook_result_retry() {
        let mut result = HookResult::default();
        assert!(!result.retry);
        result.retry = true;
        assert!(result.retry);
    }

    #[test]
    fn test_hook_result_allow_deny_ask() {
        let result = HookResult::allow();
        assert!(result.has_permission_decision());
        assert!(result.permission_decision.unwrap().is_allowed());

        let result = HookResult::deny("unsafe".to_string());
        assert!(result.block);
        assert!(result.permission_decision.unwrap().is_denied());

        let result = HookResult::ask(vec!["Allow".to_string()]);
        assert!(result.permission_decision.unwrap().requires_approval());
    }

    #[test]
    fn test_lifecycle_hook_executor_fn() {
        let executor: UnifiedHookExecutorFn = Arc::new(|ctx: HookContext| {
            Box::pin(async move {
                let mut result = HookResult::default();
                if ctx.event == HookEvent::SubagentStart {
                    result.injected_context = Some("subagent context".to_string());
                }
                result
            })
        });

        let ctx = HookContext::for_subagent_start("test", "sync", "task", "", "");
        let rt = tokio::runtime::Runtime::new().unwrap();
        let result = rt.block_on(executor(ctx));
        assert_eq!(
            result.injected_context,
            Some("subagent context".to_string())
        );
    }

    #[test]
    fn test_evolution_event_category() {
        assert_eq!(
            HookEvent::PostMemoryWrite.category(),
            HookEventCategory::Evolution
        );
        assert_eq!(
            HookEvent::MemoryLayerChange.category(),
            HookEventCategory::Evolution
        );
    }

    #[test]
    fn test_evolution_event_supports_matcher() {
        assert!(HookEvent::PostMemoryWrite.supports_matcher());
        assert!(HookEvent::MemoryLayerChange.supports_matcher());
    }

    #[test]
    fn test_evolution_event_as_str() {
        assert_eq!(HookEvent::PostMemoryWrite.as_str(), "PostMemoryWrite");
        assert_eq!(HookEvent::MemoryLayerChange.as_str(), "MemoryLayerChange");
    }

    #[test]
    fn test_for_post_memory_write() {
        let ctx = HookContext::for_post_memory_write("build_java8", "error_resolution", "s1", "a1");
        assert_eq!(ctx.event, HookEvent::PostMemoryWrite);
        assert_eq!(ctx.memory_key.as_deref(), Some("build_java8"));
        assert_eq!(ctx.memory_source.as_deref(), Some("error_resolution"));
        assert_eq!(ctx.matcher.as_deref(), Some("error_resolution"));
    }

    #[test]
    fn test_for_memory_layer_change() {
        let ctx = HookContext::for_memory_layer_change("build_java8", "warm", "hot", "s1", "a1");
        assert_eq!(ctx.event, HookEvent::MemoryLayerChange);
        assert_eq!(ctx.memory_key.as_deref(), Some("build_java8"));
        assert_eq!(ctx.memory_from_layer.as_deref(), Some("warm"));
        assert_eq!(ctx.memory_to_layer.as_deref(), Some("hot"));
        assert_eq!(ctx.matcher.as_deref(), Some("warm->hot"));
    }

    #[test]
    fn activate_skill_defaults_to_none() {
        let r = HookResult::default();
        assert!(r.activate_skill.is_none());
    }

    #[test]
    fn activate_skill_constructor_sets_field() {
        let r =
            HookResult::with_activate_skill("docx".to_string(), "检测到 .docx 文件".to_string());
        assert_eq!(
            r.activate_skill,
            Some(("docx".to_string(), "检测到 .docx 文件".to_string()))
        );
    }

    #[test]
    fn test_subagent_stop_status_serde_and_str() {
        // 与业界对齐的终止模型：SubagentStop 携带一个状态枚举，
        // 不存在单独的 SubagentCancelled 事件。
        for (status, expected) in [
            (SubagentStopStatus::Completed, "completed"),
            (SubagentStopStatus::Failed, "failed"),
            (SubagentStopStatus::Cancelled, "cancelled"),
            (SubagentStopStatus::TimedOut, "timed_out"),
        ] {
            assert_eq!(status.as_str(), expected);
            let json = serde_json::to_string(&status).unwrap_or_default();
            assert_eq!(json, format!("\"{expected}\""));
        }
    }

    #[test]
    fn test_for_subagent_stop_carries_status() {
        let ctx = HookContext::for_subagent_stop(
            "coder",
            "sync",
            "done",
            SubagentStopStatus::TimedOut,
            "sess-1",
            "agent",
        );
        assert_eq!(ctx.event, HookEvent::SubagentStop);
        assert_eq!(ctx.subagent_stop_status, Some(SubagentStopStatus::TimedOut));
    }

    #[test]
    fn test_task_terminal_status_serde_and_str() {
        // Timeout/Cancelled 是状态值，而不是独立的事件。
        for (status, expected) in [
            (TaskTerminalStatus::Completed, "completed"),
            (TaskTerminalStatus::Failed, "failed"),
            (TaskTerminalStatus::Cancelled, "cancelled"),
            (TaskTerminalStatus::TimedOut, "timed_out"),
            (TaskTerminalStatus::Skipped, "skipped"),
        ] {
            assert_eq!(status.as_str(), expected);
            let json = serde_json::to_string(&status).unwrap_or_default();
            assert_eq!(json, format!("\"{expected}\""));
        }
    }

    #[test]
    fn test_three_stage_task_lifecycle_factories() {
        // TaskCreated → TaskStarted → TaskCompleted(status)
        let created = HookContext::for_task_created("t-1", "build API", "sess-1", "agent");
        assert_eq!(created.event, HookEvent::TaskCreated);
        assert_eq!(created.task_id.as_deref(), Some("t-1"));
        assert_eq!(created.task_terminal_status, None);

        let started = HookContext::for_task_started("t-1", "build API", "sess-1", "agent");
        assert_eq!(started.event, HookEvent::TaskStarted);
        assert_eq!(started.task_id.as_deref(), Some("t-1"));

        let completed = HookContext::for_task_completed(
            "t-1",
            "build API",
            "ok",
            TaskTerminalStatus::Completed,
            "sess-1",
            "agent",
        );
        assert_eq!(completed.event, HookEvent::TaskCompleted);
        assert_eq!(
            completed.task_terminal_status,
            Some(TaskTerminalStatus::Completed)
        );
    }
}
