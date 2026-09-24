/*!
 * Agent 执行快照 —— 捕获 agent 状态以支持 `'static` 生命周期的流式处理。
 *
 * [`AgentRunSnapshot`] 取代了 `stream_channel.rs` 中原先手动维护的 33 个字段的
 * `AgentSnapshot` 克隆方式，改用基于组合的方式。配置（Configuration）、工具运行时
 * （tool runtime）和守卫运行时（guard runtime）都分别用 `Arc` 包裹，因此该快照的
 * 克隆开销是 O(1)，并且可以安全地移动到 `tokio::spawn` 的 future 中。
 */

use crate::agent::AgentCallback;
use crate::agent::InterventionCallback;
use crate::audit::{
    AuditLogger, DiagnosticDeliveryFailure, DiagnosticDeliveryObserver,
    DiagnosticDeliveryOperation, DiagnosticRecordKind,
};
use crate::memory::snapshot::SnapshotManager;
use crate::skills::hooks::HookRegistry;
use crate::tools::{ToolExecutionConfig, ToolFailure, ToolManager, ToolResult};
use crate::trace::{RunEvent, RunStatus, RunStore};
use echo_core::circuit_breaker::CircuitBreaker;
use echo_core::llm::types::{Message, Role};
use echo_core::tokenizer::Tokenizer;
use std::sync::Arc;

/// 工具输出预览部分保留的最大字符数。
const TOOL_OUTPUT_PREVIEW_CHARS: usize = 500;
/// 当输出溢出（spill）落盘失败时，回退截断策略所使用的默认 token 数。
const TOOL_OUTPUT_SPILL_FAILURE_FALLBACK_TOKENS: usize = 8_000;

/// 对单次工具结果应用「运行级输出预算」策略之后得到的结果。
pub(crate) struct ProcessedToolOutput {
    /// 处理后（可能已截断或替换为预览）的最终输出文本。
    pub output: String,
    /// 输出是否被截断。
    pub truncated: bool,
    /// 当输出被落盘为外部产物（artifact）时的引用信息。
    pub artifact: Option<echo_core::tools::artifact::ToolOutputArtifactRef>,
    /// 关于本次输出处理过程的附加元数据（如原始/返回字节数、token 数等）。
    pub metadata: std::collections::HashMap<String, String>,
}

/// 一次工具调用失败后携带的信息。
pub(crate) struct ToolCallFailure {
    /// 工具名称。
    pub name: String,
    /// 具体的错误。
    pub error: crate::error::ReactError,
    /// 失败对应的工具结果（可能带有部分输出）。
    pub result: ToolResult,
}

/// 一次工具调用成功后携带的信息。
pub(crate) struct ToolCallSuccess {
    /// 工具名称。
    pub name: String,
    /// 成功对应的工具结果。
    pub result: ToolResult,
}

/// 判断某条消息是否为框架内部使用的转录（transcript）消息，此类消息不应展示给用户。
fn is_internal_transcript_message(message: &Message) -> bool {
    if crate::compression::is_context_projection_message(message) {
        return true;
    }
    let Some(text) = message.content.as_text() else {
        return false;
    };
    let trimmed = text.trim_start();

    match message.role {
        Role::System => true,
        Role::User => {
            trimmed.starts_with("[Relevant historical memories]")
                || trimmed.starts_with("[The above memories")
                || trimmed.starts_with("[Verifier feedback]")
                || trimmed.starts_with("[Hook:")
                || trimmed.starts_with("[Memory")
                || trimmed.starts_with("[Context")
                || trimmed.starts_with("[Compact")
                || trimmed.starts_with("[Compression")
        }
        Role::Tool => {
            trimmed.starts_with("[placeholder]")
                || trimmed.starts_with("[synthetic]")
                || trimmed.contains("placeholder result")
        }
        Role::Assistant | Role::Custom(_) => false,
    }
}

/// 过滤消息列表，仅保留用户可见的转录消息（剔除内部消息）。
fn filter_user_visible_transcript(messages: &[Message]) -> Vec<Message> {
    messages
        .iter()
        .filter(|message| !is_internal_transcript_message(message))
        .cloned()
        .collect()
}

/// 判断两条投影（projection）消息的内容摘要是否一致。
fn same_projection_content(
    left: &crate::state::TranscriptProjectionMessage,
    right: &crate::state::TranscriptProjectionMessage,
) -> bool {
    left.digest == right.digest
}

/// 计算「上一次投影结果」与「本次投影结果」之间可复用的重叠（overlap）长度，
/// 用于避免为已经投影过的消息重新分配序号（ordinal）。
fn generation_overlap(
    previous: &[crate::state::TranscriptProjectionMessage],
    projected: &[crate::state::TranscriptProjectionMessage],
) -> usize {
    let max_overlap = previous.len().min(projected.len());
    (0..=max_overlap)
        .rev()
        .find(|overlap| {
            previous
                .iter()
                .skip(previous.len().saturating_sub(*overlap))
                .zip(projected.iter().take(*overlap))
                .all(|(left, right)| same_projection_content(left, right))
        })
        .unwrap_or_default()
}

/// 将已存储的消息列表转换为带内容摘要（digest）的投影消息标识列表（序号先置零，
/// 之后由调用方赋值）。
fn transcript_projection_messages(
    projected: &[crate::memory::StoredMessage],
) -> crate::error::Result<Vec<crate::state::TranscriptProjectionMessage>> {
    projected
        .iter()
        .map(|message| {
            Ok(crate::state::TranscriptProjectionMessage {
                ordinal: 0,
                digest: crate::memory::transcript_projection_message_digest(message)?,
            })
        })
        .collect()
}

/// 转录投影（transcript projection）的游标：跟踪当前代（generation）已投影过的
/// 消息序号，用于在多次安全点（safe point）之间保持序号稳定、避免重复投影。
#[derive(Clone, Default)]
pub(crate) struct TranscriptProjectionCursor {
    /// 当前游标所属的代（generation）标识。
    generation_id: Option<String>,
    /// 下一条新消息将被分配到的序号。
    next_ordinal: u64,
    /// 已经完成投影的消息标识列表。
    projected: Vec<crate::state::TranscriptProjectionMessage>,
}

impl TranscriptProjectionCursor {
    /// 用一份已还原（restored）的完整消息列表重新对齐游标状态，
    /// 常用于从历史检查点恢复后重建投影序号。
    pub(crate) fn align_restored(
        &mut self,
        generation_id: &str,
        messages: &[Message],
    ) -> crate::error::Result<()> {
        let visible = filter_user_visible_transcript(messages);
        let projected = crate::memory::project_messages(generation_id, &visible)?;
        let mut assigned = transcript_projection_messages(&projected)?;
        self.next_ordinal = 0;
        for message in &mut assigned {
            message.ordinal = self.next_ordinal;
            self.next_ordinal = self.next_ordinal.checked_add(1).ok_or_else(|| {
                crate::error::ReactError::Other(
                    "transcript projection ordinal capacity exhausted".to_string(),
                )
            })?;
        }
        self.projected = assigned;
        self.generation_id = Some(generation_id.to_string());
        Ok(())
    }

    /// 将新的投影结果与已有游标状态对齐：复用可重叠的旧序号，
    /// 并为新增消息分配新的序号；若代（generation）发生变化则重置游标。
    fn assign(
        &mut self,
        generation_id: &str,
        projected: &[crate::memory::StoredMessage],
    ) -> crate::error::Result<Vec<crate::state::TranscriptProjectionMessage>> {
        if self.generation_id.as_deref() != Some(generation_id) {
            self.generation_id = Some(generation_id.to_string());
            self.next_ordinal = 0;
            self.projected.clear();
        }
        let identities = transcript_projection_messages(projected)?;
        let overlap = generation_overlap(&self.projected, &identities);
        let mut assigned = self
            .projected
            .iter()
            .skip(self.projected.len().saturating_sub(overlap))
            .cloned()
            .collect::<Vec<_>>();
        for mut message in identities.into_iter().skip(overlap) {
            message.ordinal = self.next_ordinal;
            self.next_ordinal = self.next_ordinal.checked_add(1).ok_or_else(|| {
                crate::error::ReactError::Other(
                    "transcript projection ordinal capacity exhausted".to_string(),
                )
            })?;
            assigned.push(message);
        }
        Ok(assigned)
    }

    /// 生成指定代（generation）对应的可持久化检查点；若游标当前所属代与
    /// 请求的代不一致，则返回一个初始（空）状态的检查点。
    fn checkpoint_for(&self, generation_id: &str) -> crate::state::TranscriptProjectionCheckpoint {
        crate::state::TranscriptProjectionCheckpoint {
            generation_id: generation_id.to_string(),
            next_ordinal: if self.generation_id.as_deref() == Some(generation_id) {
                self.next_ordinal
            } else {
                0
            },
            projected: if self.generation_id.as_deref() == Some(generation_id) {
                self.projected.clone()
            } else {
                Vec::new()
            },
        }
    }

    /// 用持久化的检查点数据恢复游标状态。
    pub(crate) fn restore(&mut self, checkpoint: crate::state::TranscriptProjectionCheckpoint) {
        self.generation_id = Some(checkpoint.generation_id);
        self.next_ordinal = checkpoint.next_ordinal;
        self.projected = checkpoint.projected;
    }
}

// ── RuntimeConfig ────────────────────────────────────────────────────

/// [`AgentConfig`](crate::agent::AgentConfig) 中在一次流式运行过程中不会变化的
/// 不可变子集。
#[derive(Clone)]
pub struct RuntimeConfig {
    /// Agent 名称。
    pub agent_name: String,
    /// 模型名称。
    pub model_name: String,
    /// 模型所属的 provider（服务提供方）名称。
    pub provider: Option<String>,
    /// 最大迭代次数。
    pub max_iterations: usize,
    /// token 预算上限。
    pub token_limit: usize,
    /// token 预算配置在构造阶段的校验失败信息。
    pub token_budget_error: Option<String>,
    /// 运行级预算策略。
    pub run_budget: echo_core::agent::RunBudgetPolicy,
    /// 模型是否支持 `tool_choice: none`。
    pub supports_tool_choice_none: bool,
    /// 所配置模型接受的输入模态（modality）。`None` 用于保持对未提供
    /// [`crate::llm::LlmConfig`] 的自定义 agent 的兼容性。
    pub input_modalities: Option<Vec<echo_core::llm::ModelInputModality>>,
    /// 会话 ID。
    pub session_id: Option<String>,
    /// 专门用于 `RuntimeStateStore` 检查点的身份标识。
    pub runtime_state_id: Option<String>,
    /// 会话所属的对话 ID。
    pub conversation_id: Option<String>,
    /// 会话绑定的工作目录（worktree 路径）。会被注入到每次工具调用的
    /// ToolContext 中，使文件/shell/git 等工具在隔离的检出（checkout）目录内运行。
    /// `None` 表示使用进程的当前工作目录（向后兼容）。
    pub working_dir: Option<std::path::PathBuf>,
    /// 采样温度。
    pub temperature: Option<f32>,
    /// 单次响应允许的最大 token 数。
    pub max_tokens: Option<u32>,
    /// 是否将工具错误作为反馈回传给模型。
    pub tool_error_feedback: bool,
    /// 是否强制要求「编辑前必须先读取」文件。
    pub force_read_before_edit: bool,
    /// 是否启用工具调用能力。
    pub enable_tool: bool,
    /// LLM 调用的最大重试次数。
    pub llm_max_retries: usize,
    /// LLM 调用重试之间的延迟（毫秒）。
    pub llm_retry_delay_ms: u64,
    /// 单次工具输出允许的最大 token 数。
    pub max_tool_output_tokens: Option<usize>,
    /// 工具输出产物（artifact）落盘配置。
    pub tool_output_artifacts: Option<echo_core::tools::artifact::ToolOutputArtifactConfig>,
    /// 工具执行相关配置。
    pub tool_execution: ToolExecutionConfig,
    /// 已注册的 agent 回调列表。
    pub callbacks: Vec<Arc<dyn AgentCallback>>,
    /// 运行时检查点的保存频率（0 = 仅在结束时保存，N = 每 N 次迭代保存一次）。
    pub react_checkpoint_interval: usize,
    /// 单个转录持久化安全点允许消耗的总时间预算。
    pub persistence_settlement_timeout: std::time::Duration,
    /// 是否启用 verifier（校验器）。
    pub verifier_enabled: bool,
    /// verifier 通过所需的最低分数。
    pub verifier_min_score: f64,
    /// verifier 的最大重试次数。
    pub verifier_max_retries: usize,
    /// 是否启用计划模式（plan mode，仅允许只读工具）。
    pub plan_mode: bool,
    /// 用于 KVCache 隔离（如 DeepSeek 等）的稳定用户标识。
    pub cache_user_id: Option<String>,
}

impl RuntimeConfig {
    /// 从 agent 的配置创建一份快照。
    pub fn from_agent_config(config: &crate::agent::AgentConfig) -> Self {
        Self {
            agent_name: config.agent_name.clone(),
            model_name: config.model_name.clone(),
            provider: config
                .model_profile
                .as_ref()
                .map(|profile| profile.provider.clone()),
            max_iterations: config.max_iterations,
            token_limit: config.token_limit,
            token_budget_error: config
                .token_budget_config
                .enabled
                .then(|| config.token_budget_config.build(config.token_limit).err())
                .flatten()
                .map(|error| error.to_string()),
            run_budget: config.run_budget.clone(),
            supports_tool_choice_none: config
                .model_profile
                .as_ref()
                .is_none_or(|profile| profile.supports_tool_choice_none),
            input_modalities: None,
            session_id: config.session_id.clone(),
            runtime_state_id: config.conversation_id.clone(),
            conversation_id: config.conversation_id.clone(),
            working_dir: config.working_dir.lock().ok().and_then(|g| g.clone()),
            temperature: config.temperature,
            max_tokens: config.max_tokens,
            tool_error_feedback: config.tool_error_feedback,
            force_read_before_edit: config.force_read_before_edit,
            enable_tool: config.enable_tool,
            llm_max_retries: config.llm_max_retries,
            llm_retry_delay_ms: config.llm_retry_delay_ms,
            max_tool_output_tokens: config.max_tool_output_tokens,
            tool_output_artifacts: config.get_tool_output_artifacts(),
            tool_execution: config.tool_execution.clone(),
            callbacks: config.callbacks.to_vec(),
            react_checkpoint_interval: config.react_checkpoint_interval,
            persistence_settlement_timeout: config.persistence_settlement_timeout,
            verifier_enabled: config.verifier_enabled,
            verifier_min_score: config.verifier_min_score,
            verifier_max_retries: config.verifier_max_retries,
            plan_mode: config.plan_mode
                || config.permission_mode == echo_core::tools::permission::PermissionMode::Plan,
            cache_user_id: config.cache_user_id.clone(),
        }
    }

    /// 以 &str 形式返回 session ID，缺省时返回空字符串。
    pub fn session_id_str(&self) -> &str {
        self.session_id.as_deref().unwrap_or("")
    }
}

/// 依据配置值、当前调用（invocation）上下文与旧版（legacy）外部上下文，
/// 解析出实际生效的运行时状态（runtime state）标识。
pub(crate) fn effective_runtime_state_id<'a>(
    configured_runtime_state_id: Option<&'a str>,
    invocation: Option<&'a echo_core::agent::AgentInvocationContext>,
    legacy: Option<&'a crate::agent::react::LegacyExternalContextSnapshot>,
) -> Option<&'a str> {
    let legacy_conversation_id = if invocation.is_none() {
        legacy.and_then(|context| context.conversation_id.as_deref())
    } else {
        None
    };
    invocation
        .and_then(|context| context.runtime_state_id.as_deref())
        .or_else(|| {
            invocation
                .and_then(|context| context.runtime.as_ref())
                .and_then(|runtime| runtime.conversation_id.as_deref())
        })
        .or(legacy_conversation_id)
        .or(configured_runtime_state_id)
}

/// 校验运行时状态标识与转录代（transcript generation）标识是否一致；
/// 若两者都存在但不相等，则返回错误。
pub(crate) fn validate_transcript_generation_identity(
    runtime_state_id: Option<&str>,
    transcript_generation_id: Option<&str>,
) -> crate::error::Result<()> {
    if let (Some(runtime_state_id), Some(transcript_generation_id)) =
        (runtime_state_id, transcript_generation_id)
        && runtime_state_id != transcript_generation_id
    {
        return Err(crate::error::ReactError::RuntimeState(Box::new(
            echo_core::error::RuntimeStateError::SerializationError(format!(
                "transcript generation identity '{transcript_generation_id}' does not match runtime state identity '{runtime_state_id}'",
            )),
        )));
    }
    Ok(())
}

/// 依据结算（settlement）状态构造对应的转录投影准入错误。
pub(crate) fn transcript_settlement_admission_error(
    settlement: &crate::memory::TranscriptProjectionSettlement,
) -> crate::error::ReactError {
    if settlement.status == crate::memory::TranscriptProjectionSettlementStatus::Deferred {
        return echo_core::error::RuntimeStateError::TranscriptProjectionDeferred {
            operation_id: settlement.operation_id.clone(),
            reason: settlement
                .detail
                .clone()
                .unwrap_or_else(|| "durable transcript debt remains unsettled".to_string()),
        }
        .into();
    }
    echo_core::error::RuntimeStateError::TranscriptProjectionBlocked {
        status: format!("{:?}", settlement.status),
        reason: settlement
            .detail
            .clone()
            .unwrap_or_else(|| "transcript projection admission failed".to_string()),
    }
    .into()
}

// ── ToolRuntime ──────────────────────────────────────────────────────

/// 工具执行相关状态（工具、hook、干预回调）。通过 `Arc` 共享。
#[derive(Clone)]
pub struct ToolRuntime {
    /// 工具管理器。
    pub tool_manager: Arc<ToolManager>,
    /// hook 注册表。
    pub hook_registry: Arc<tokio::sync::RwLock<HookRegistry>>,
    /// 干预（intervention）回调列表。
    pub intervention_callbacks: Vec<Arc<dyn InterventionCallback>>,
    /// 来自已激活技能（skill）的允许工具模式集合（在快照创建时捕获）。
    /// `None` 表示不受限制（没有技能限制工具）。
    pub skill_allowed_tools: Option<std::collections::HashSet<String>>,
    /// 本次调用生效的已禁用工具集合。
    pub disabled_tools: std::collections::HashSet<String>,
    /// 本次调用中延迟（deferred）工具的可变 schema 可见性状态。
    pub visibility: Option<std::sync::Arc<echo_core::tools::ToolVisibilityState>>,
    /// 本次调用是否使用计划模式（plan mode）的只读工具集。
    pub plan_mode: bool,
    /// 通过 SDK/宿主 API 进行模式变更时使用的实时权限模式权威（authority）。
    #[cfg(feature = "human-loop")]
    permission_service: Option<Arc<crate::human_loop::PermissionService>>,
}

impl ToolRuntime {
    /// 从 agent 及本次调用（invocation）的禁用/可见工具集合构建工具运行时。
    pub fn from_agent(
        agent: &super::ReactAgent,
        invocation_disabled_tools: Option<&std::collections::HashSet<String>>,
        invocation_visible_tools: Option<&std::collections::HashSet<String>>,
    ) -> Self {
        let mut disabled_tools = agent.tools.tool_visibility.disabled_names();
        if let Some(profile) = agent.config.model_profile.as_ref() {
            disabled_tools.extend(profile.excluded_tools.iter().cloned());
        }
        if let Some(invocation_disabled_tools) = invocation_disabled_tools {
            disabled_tools.extend(invocation_disabled_tools.iter().cloned());
        }
        let tool_manager = Arc::clone(&agent.tools.tool_manager);
        if let Some(config) = agent.llm_config() {
            disabled_tools.extend(tool_manager.incompatible_tool_names(&config.input_modalities));
        }
        let skill_allowed_tools = agent.tools.skill_registry.active_skill_allowed_tools();
        #[cfg(feature = "human-loop")]
        let permission_service = agent.approval.permission_service.clone();
        let plan_mode = agent.config.plan_mode
            || agent.config.permission_mode == echo_core::tools::permission::PermissionMode::Plan
            || {
                #[cfg(feature = "human-loop")]
                {
                    permission_service.as_ref().is_some_and(|service| {
                        service.current_mode() == echo_core::tools::permission::PermissionMode::Plan
                    })
                }
                #[cfg(not(feature = "human-loop"))]
                {
                    false
                }
            };
        let visibility = invocation_visible_tools.map(|initial| {
            let available = tool_manager
                .get_openai_tools()
                .into_iter()
                .filter(|tool| !disabled_tools.contains(&tool.function.name))
                .filter(|tool| {
                    !plan_mode
                        || tool_manager
                            .get_tool(&tool.function.name)
                            .is_some_and(|tool| tool.capabilities().is_read_only())
                })
                .map(|tool| tool.function.name)
                .collect::<std::collections::HashSet<_>>();
            let eligible = available
                .iter()
                .filter(|name| {
                    skill_allowed_tools.as_ref().is_none_or(|allowed_tools| {
                        echo_execution::skills::external::types::skill_allows_tool(
                            allowed_tools,
                            name,
                        )
                    })
                })
                .cloned()
                .collect();
            let mut initial = initial.clone();
            initial.insert("tool_search".to_string());
            std::sync::Arc::new(echo_core::tools::ToolVisibilityState::with_available(
                available, eligible, initial,
            ))
        });
        Self {
            tool_manager,
            hook_registry: agent.tools.hook_registry.clone(),
            intervention_callbacks: agent.tools.intervention_callbacks.clone(),
            skill_allowed_tools,
            disabled_tools,
            visibility,
            plan_mode,
            #[cfg(feature = "human-loop")]
            permission_service,
        }
    }

    /// 返回不可变的、作用域限定于本次调用（invocation）的、供 LLM 使用的工具定义列表。
    pub fn tools_for_llm(&self) -> Vec<crate::llm::types::ToolDefinition> {
        self.tool_manager
            .get_openai_tools()
            .into_iter()
            .filter(|tool| !self.disabled_tools.contains(&tool.function.name))
            .filter(|tool| self.visibility.is_some() || tool.function.name != "tool_search")
            .filter(|tool| {
                self.visibility
                    .as_ref()
                    .is_none_or(|visibility| visibility.is_visible(&tool.function.name))
            })
            .filter(|tool| self.is_skill_tool_allowed(&tool.function.name))
            .filter(|tool| {
                !self.is_plan_mode()
                    || self
                        .tool_manager
                        .get_tool(&tool.function.name)
                        .is_some_and(|tool| tool.capabilities().is_read_only())
            })
            .collect()
    }

    /// 判断指定工具是否为只读工具。
    pub(crate) fn is_tool_read_only(&self, tool_name: &str) -> bool {
        self.tool_manager
            .get_tool(tool_name)
            .is_some_and(|tool| tool.capabilities().is_read_only())
    }

    /// 判断当前是否处于计划模式（plan mode）。
    pub(crate) fn is_plan_mode(&self) -> bool {
        self.plan_mode || {
            #[cfg(feature = "human-loop")]
            {
                self.permission_service.as_ref().is_some_and(|service| {
                    service.current_mode() == echo_core::tools::permission::PermissionMode::Plan
                })
            }
            #[cfg(not(feature = "human-loop"))]
            {
                false
            }
        }
    }

    /// 判断指定工具是否被当前激活的技能（skill）允许使用。
    pub(crate) fn is_skill_tool_allowed(&self, tool_name: &str) -> bool {
        if let Some(visibility) = self.visibility.as_ref() {
            return visibility.is_eligible(tool_name);
        }
        self.skill_allowed_tools
            .as_ref()
            .is_none_or(|allowed_tools| {
                echo_execution::skills::external::types::skill_allows_tool(allowed_tools, tool_name)
            })
    }
}

// ── GuardRuntime ─────────────────────────────────────────────────────

/// 守卫（guard）/安全相关状态。通过 `Arc` 共享。
#[derive(Clone)]
pub struct GuardRuntime {
    /// 守卫管理器。
    pub guard_manager: Option<Arc<crate::guard::GuardManager>>,
    /// 审计日志记录器。
    pub audit_logger: Option<Arc<dyn AuditLogger>>,
    /// 熔断器。
    pub circuit_breaker: Option<Arc<CircuitBreaker>>,
}

impl GuardRuntime {
    /// 从 agent 构建守卫运行时。
    pub fn from_agent(agent: &super::ReactAgent) -> Self {
        Self {
            guard_manager: agent.guard.guard_manager.clone().map(Arc::new),
            audit_logger: agent.guard.audit_logger.clone(),
            circuit_breaker: agent.guard.circuit_breaker.clone(),
        }
    }
}

// ── AgentRunSnapshot ─────────────────────────────────────────────────

/**
 * 捕获流式循环所需的、来自 [`super::ReactAgent`] 的一切状态，而不必持有对
 * agent 本身的引用。
 *
 * 所有子系统均通过 `Arc` 进行组合——克隆开销为 O(1)。
 */
#[derive(Clone)]
pub struct AgentRunSnapshot {
    /// 不可变的运行时配置。
    pub config: Arc<RuntimeConfig>,
    /// 与运行中的 agent 共享的、权威的对话上下文。
    pub context: Arc<tokio::sync::Mutex<crate::compression::ContextManager>>,
    /// 工具执行相关状态（工具、hook）。
    pub tools: Arc<ToolRuntime>,
    /// 与所属 agent 及 Skill 工具共享的规范激活权威（activation authority）。
    skill_activation: crate::skills::SkillActivationHandle,
    /// 守卫（guard）/安全相关状态。
    pub guard: Arc<GuardRuntime>,
    /// 快照管理器（来自 memory 子系统）。
    pub snapshot_manager: Arc<std::sync::RwLock<Option<SnapshotManager>>>,
    /// 当前转录代（transcript generation）标识。
    transcript_generation_id: Option<String>,
    /// 转录投影游标。
    transcript_projection_cursor: Arc<tokio::sync::Mutex<TranscriptProjectionCursor>>,
    /// 已加载的运行时状态版本。
    runtime_state_version: Arc<tokio::sync::Mutex<Option<crate::state::RuntimeStateVersion>>>,
    /// 是否已观测到转录结算（settlement）发生。
    transcript_settlement_observed: Arc<std::sync::atomic::AtomicBool>,
    /// HTTP 客户端。
    pub client: Arc<reqwest::Client>,
    /// 可选的 trait 级 LLM 客户端。当其存在时，流式核心循环
    /// （`create_llm_stream`）与 `direct_answer_stream` 会通过该 trait 对象来
    /// 路由 LLM 调用，而不是走原始的 `client` + 模型解析路径 —— 这使得测试替身
    /// （MockLlmClient）能够驱动完整的 ReAct 循环。生产环境的 agent 会注入真实的
    /// `LlmClient` 实现；若未附加任何实现，执行时将返回配置错误。
    pub llm_client: Option<Arc<dyn crate::llm::LlmClient>>,
    /// 每个 agent 的思考深度（thinking-depth）配置，会被传播到 think 阶段和
    /// react 循环，使每次 LLM 请求都携带所配置的推理深度。`None` 表示
    /// “使用模型默认值”（不发送 thinking 字段）。
    pub thinking: Option<crate::llm::ThinkingConfig>,
    /// 取消令牌（在构造完成后设置）。
    pub cancel_token: Option<crate::agent::CancellationToken>,
    /// 共享的、同一轮次（turn）内的输入邮箱。
    pub(crate) turn_steer_mailbox: Arc<crate::agent::steer::TurnSteerMailbox>,
    /// 用于「编辑前必须先读取」策略执行的最近读取文件记录（路径 → 读取时刻）。
    pub recently_read_files:
        Arc<std::sync::Mutex<std::collections::HashMap<String, std::time::Instant>>>,
    /// 用于 trace 持久化的运行存储（run store）。
    pub run_store: Option<Arc<dyn RunStore>>,
    /// Trace 与 Audit 持久化投递失败的观察者。
    pub(crate) diagnostic_delivery_observer: Option<Arc<dyn DiagnosticDeliveryObserver>>,
    /// 当前的 run ID。
    pub current_run_id: Option<String>,
    /// 唯一的 trace 调用 ID。它有意与 `current_run_id` 中的产品/业务 run ID
    /// 区分开来。
    pub trace_run_id: Option<String>,
    /// 本次调用范围内、真正进入 ExecuteStage 的工具调用权威集合。
    executing_tool_call_ids: Arc<std::sync::Mutex<std::collections::HashSet<String>>>,
    /// 当前用户输入/agent 轮次（turn）ID。
    pub current_turn_id: Option<String>,
    /// 用于精确排空（drain）当前活跃轮次实例的私有权威句柄。
    pub(crate) turn_steer_incarnation: Option<Arc<()>>,
    /// 触发本次调用的消息 ID。
    pub current_message_id: Option<String>,
    /// 类型化的当前活跃用户消息，包含任何附件。
    pub current_message: Option<crate::llm::types::Message>,
    /// 当前具体的 subagent/工具执行 ID。
    pub current_execution_id: Option<String>,
    /// 外部 run 级上下文（跨 spawn 安全，从 ReactAgent.external_* 抓取）。
    /// 与 current_run_id 同源、同生命周期（set/clear 在同一处）。
    pub external_cancel: Option<std::sync::Arc<tokio_util::sync::CancellationToken>>,
    /// 外部 trace sink（下沉）函数。
    pub external_trace_sink: Option<echo_core::tools::TraceSinkFn>,
    /// 外部委派（delegation）策略。
    pub external_delegation_policy: Option<echo_core::tools::NestedDelegationPolicy>,
    /// 本快照所运行的、已分派 Subagent 尝试的身份/世系（lineage）信息。
    /// 对于主 agent 调用为 `None`。会被复制进每个 ToolContext。
    pub subagent_lineage: Option<echo_core::tools::SubagentLineage>,
    /// 用于 Subagent→父级 / Subagent→兄弟节点通信的上行（uplink）通道，
    /// 由 dispatcher 安装进本次调用的运行时上下文中。
    pub external_uplink: Option<echo_core::tools::SubagentUplinkFn>,
    /// 为本次调用及其工具保留的不透明所有权令牌。
    pub resource_guards: Vec<echo_core::tools::InvocationResourceGuard>,
    /// 权限服务（human-in-the-loop）。
    #[cfg(feature = "human-loop")]
    pub permission_service: Option<Arc<crate::human_loop::PermissionService>>,
    /// 通过同步 agent 建构（setup）API 注册的审批规则。
    #[cfg(feature = "human-loop")]
    pub pending_permission_rules:
        Arc<tokio::sync::Mutex<Vec<echo_core::tools::permission::PermissionRule>>>,
    /// 与父级 ReactAgent 共享的 token 用量跟踪器。
    pub token_tracker: Arc<echo_core::tokenizer::TokenUsageTracker>,
    /// 与父级 ReactAgent 共享的自校准分词器（tokenizer）。think 阶段会将真实的
    /// `usage.prompt_tokens` 反馈给它，使上下文窗口与压缩估算逐渐逼近模型的
    /// 真实分词结果。
    pub calibrated_tokenizer: Arc<echo_core::tokenizer::CalibratedTokenizer>,
    /// 用于 ReAct 检查点的运行时状态存储（消息 + 技能）。
    /// 旧版 `current_plan` 数据仍以原始 Store 数据形式保留，不会被恢复。
    pub state_store: Option<Arc<dyn crate::state::RuntimeStateStore>>,
    /// 用于用户可见转录投影的对话存储。当该字段与
    /// `config.conversation_id` 同时设置时，运行循环会在每个最终化（finalization）
    /// 安全点持久化投影后的消息，使应用 UI 历史始终与运行中的上下文保持同步——
    /// 而无需每个入口点各自重新实现保存逻辑。
    pub conversation_store: Option<Arc<dyn crate::memory::ConversationStore>>,
    /// 供可选的框架技能遥测（telemetry）写入器使用的长期存储。
    pub(crate) memory_store: Option<Arc<dyn crate::memory::Store>>,
    /// 可选的用于 final_answer 校验的 Critic。
    pub critic: Option<Arc<dyn echo_core::agent::Critic>>,
    /// 可选的工具执行流水线（16 阶段中间件）。
    pub tool_execution_pipeline:
        Option<Arc<crate::agent::react::run::pipeline::ToolExecutionPipeline>>,
    /// （stage4 E1）分层记忆管理器 —— 供 `pre_compaction_flush` 在压缩前写入
    /// 持久化事实使用。从父级 ReactAgent 克隆而来。
    pub memory_layer_manager: Option<Arc<crate::evolution::MemoryLayerManager>>,
    /// 可选的、在 pre-prepare 边界处刷新的应用层投影。
    pub pre_model_context_projector: Option<Arc<dyn crate::compression::PreModelContextProjector>>,
    /// 由消费方提供的规范 Skill 生命周期变更权威。
    pub skill_mutation_authority: Option<Arc<crate::evolution::SkillMutationAuthority>>,
}

/// 负责协调转录/检查点持久化的内部辅助结构，封装单次持久化操作的截止时间与调用上下文。
struct AgentPersistenceCoordinator<'a> {
    /// 所属的运行快照。
    snapshot: &'a AgentRunSnapshot,
    /// 本次持久化操作的截止时间。
    deadline: tokio::time::Instant,
    /// 持久化调用上下文。
    call_context: crate::memory::PersistenceCallContext,
}

/// 已托管（managed）持久化所需的一对存储引用：对话存储与运行时状态存储。
type ManagedPersistenceStores<'a> = (
    &'a Arc<dyn crate::memory::ConversationStore>,
    &'a Arc<dyn crate::state::RuntimeStateStore>,
);

impl<'a> AgentPersistenceCoordinator<'a> {
    /// 基于快照创建一个持久化协调器，并计算出本次操作的截止时间。
    fn new(snapshot: &'a AgentRunSnapshot) -> crate::error::Result<Self> {
        let started_at = tokio::time::Instant::now();
        let timeout = snapshot.config.persistence_settlement_timeout;
        if timeout.is_zero() {
            return Err(persistence_configuration_error(
                "persistence settlement timeout must be greater than zero",
            ));
        }
        let deadline = started_at.checked_add(timeout).ok_or_else(|| {
            persistence_configuration_error("persistence settlement deadline overflow")
        })?;
        Ok(Self {
            snapshot,
            deadline,
            call_context: crate::memory::PersistenceCallContext::with_timeout(timeout)?,
        })
    }

    /// 若对话存储与运行时状态存储均已配置，则返回二者的引用；否则返回 `None`
    /// （表示应回退到旧版持久化路径）。
    fn managed_stores(&self) -> Option<ManagedPersistenceStores<'_>> {
        Some((
            self.snapshot.conversation_store.as_ref()?,
            self.snapshot.state_store.as_ref()?,
        ))
    }

    /// 解析并校验持久化所需的三元身份标识：对话 ID、运行时状态 ID、转录代 ID。
    fn identities(&self) -> crate::error::Result<(&str, &str, &str)> {
        let conversation_id = self
            .snapshot
            .config
            .conversation_id
            .as_deref()
            .ok_or_else(|| persistence_configuration_error("missing conversation identity"))?;
        let runtime_state_id = self
            .snapshot
            .config
            .runtime_state_id
            .as_deref()
            .ok_or_else(|| persistence_configuration_error("missing runtime-state identity"))?;
        let generation_id = self
            .snapshot
            .transcript_generation_id
            .as_deref()
            .ok_or_else(|| persistence_configuration_error("missing transcript generation"))?;
        validate_transcript_generation_identity(Some(runtime_state_id), Some(generation_id))?;
        Ok((conversation_id, runtime_state_id, generation_id))
    }

    /// 确保对话投影所处的「代」（epoch）存在，返回该代对应的权威凭据。
    async fn ensure_epoch(
        &self,
        store: &dyn crate::memory::ConversationStore,
        conversation_id: &str,
    ) -> crate::error::Result<crate::memory::ConversationProjectionEpochReceipt> {
        tokio::time::timeout_at(
            self.deadline,
            store.ensure_projection_epoch_with_context(
                self.call_context,
                crate::memory::EnsureConversationProjectionRequest {
                    conversation: crate::memory::NewConversation {
                        conversation_id: conversation_id.to_string(),
                        user_id: "default".to_string(),
                        agent_type: None,
                        title: None,
                    },
                    expected_tombstone_epoch: None,
                },
            ),
        )
        .await
        .map_err(|_| persistence_deadline_error("conversation epoch acquisition"))?
    }

    /// 加载指定作用域（scope）与运行时状态 ID 当前的持久化状态，
    /// 返回作用域修订号（revision）、用于 CAS 的期望版本，以及当前的托管状态快照。
    async fn current_runtime_state(
        &self,
        store: &dyn crate::state::RuntimeStateStore,
        scope_id: &str,
        runtime_state_id: &str,
    ) -> crate::error::Result<(
        u64,
        crate::state::RuntimeStateExpectedVersion,
        Option<crate::state::ManagedRuntimeStateSnapshot>,
    )> {
        let scope_revision = tokio::time::timeout_at(
            self.deadline,
            store.load_scope_authority_with_context(self.call_context, scope_id),
        )
        .await
        .map_err(|_| persistence_deadline_error("runtime scope load"))??
        .map(|scope| scope.revision)
        .unwrap_or(0);
        let current = tokio::time::timeout_at(
            self.deadline,
            store.load_runtime_state_with_context(self.call_context, scope_id, runtime_state_id),
        )
        .await
        .map_err(|_| persistence_deadline_error("runtime generation load"))??;
        let expected = match current.as_ref().map(|state| &state.version) {
            None | Some(crate::state::RuntimeStateVersion::Absent) => {
                crate::state::RuntimeStateExpectedVersion::Absent
            }
            Some(crate::state::RuntimeStateVersion::Unmanaged { digest }) => {
                crate::state::RuntimeStateExpectedVersion::Unmanaged {
                    digest: digest.clone(),
                }
            }
            Some(crate::state::RuntimeStateVersion::Managed { revision }) => {
                crate::state::RuntimeStateExpectedVersion::Managed {
                    revision: *revision,
                }
            }
            Some(crate::state::RuntimeStateVersion::Retired { .. }) => {
                return Err(crate::error::ReactError::RuntimeState(Box::new(
                    echo_core::error::RuntimeStateError::ManagedStateRequiresCas(format!(
                        "runtime generation {runtime_state_id} is retired"
                    )),
                )));
            }
        };
        Ok((scope_revision, expected, current))
    }

    /// 基于当前上下文消息与其它可选状态构建一份 [`crate::state::AgentCheckpoint`]。
    async fn build_checkpoint(
        &self,
        context: &Arc<tokio::sync::Mutex<crate::compression::ContextManager>>,
        blocked_reason: Option<String>,
        cursor: Option<crate::state::TranscriptProjectionCheckpoint>,
        pending: Option<crate::state::PendingTranscriptProjection>,
    ) -> crate::error::Result<crate::state::AgentCheckpoint> {
        let runtime_state_id = self
            .snapshot
            .config
            .runtime_state_id
            .as_ref()
            .ok_or_else(|| persistence_configuration_error("missing runtime-state identity"))?;
        let messages = context.lock().await.messages().to_vec();
        let messages_json =
            crate::state::AgentCheckpoint::serialize_managed_payload(messages, cursor, pending)?;
        Ok(crate::state::AgentCheckpoint {
            conversation_id: runtime_state_id.clone(),
            messages_json,
            current_plan: None,
            active_skills: self.snapshot.active_skill_names(),
            blocked_reason,
            working_dir: self.snapshot.config.working_dir.clone(),
            timestamp: chrono::Utc::now(),
        })
    }

    /// 依据期望版本计算下一个托管修订号（revision）。
    fn next_managed_revision(
        expected: &crate::state::RuntimeStateExpectedVersion,
    ) -> crate::error::Result<u64> {
        match expected {
            crate::state::RuntimeStateExpectedVersion::Absent
            | crate::state::RuntimeStateExpectedVersion::Unmanaged { .. } => Ok(1),
            crate::state::RuntimeStateExpectedVersion::Managed { revision } => {
                revision.checked_add(1).ok_or_else(|| {
                    echo_core::error::RuntimeStateError::RevisionExhausted(
                        "checkpoint revision reached u64::MAX".to_string(),
                    )
                    .into()
                })
            }
        }
    }

    /// 从可选的托管运行时状态快照中提取权威版本号，缺省时视为 `Absent`。
    fn authority_version(
        current: Option<&crate::state::ManagedRuntimeStateSnapshot>,
    ) -> crate::state::RuntimeStateVersion {
        current
            .map(|state| state.version.clone())
            .unwrap_or(crate::state::RuntimeStateVersion::Absent)
    }

    /// 确保内存中的上下文是从与持久化存储一致的运行时版本水合（hydrate）而来，
    /// 否则拒绝继续写入以避免基于错误基线覆盖数据。
    async fn require_hydrated_base(
        &self,
        current: Option<&crate::state::ManagedRuntimeStateSnapshot>,
    ) -> crate::error::Result<()> {
        let durable = Self::authority_version(current);
        let mut hydrated = self.snapshot.runtime_state_version.lock().await;
        match hydrated.as_ref() {
            Some(version) if version == &durable => Ok(()),
            None if durable == crate::state::RuntimeStateVersion::Absent => {
                *hydrated = Some(durable);
                Ok(())
            }
            _ => Err(
                echo_core::error::RuntimeStateError::ManagedStateRequiresCas(
                    "in-memory context was hydrated from a different runtime revision".to_string(),
                )
                .into(),
            ),
        }
    }

    /// 以 compare-and-save 语义保存运行时检查点：确保对话投影代存在、校验期望版本，
    /// 并拒绝在存在待处理转录投影时覆盖检查点。
    async fn save_checkpoint(
        &self,
        context: &Arc<tokio::sync::Mutex<crate::compression::ContextManager>>,
        blocked_reason: Option<String>,
    ) -> crate::error::Result<()> {
        let Some((conversation_store, runtime_store)) = self.managed_stores() else {
            return self.save_legacy_checkpoint(context, blocked_reason).await;
        };
        let (scope_id, runtime_state_id, generation_id) = self.identities()?;
        let epoch = self
            .ensure_epoch(conversation_store.as_ref(), scope_id)
            .await?;
        if !matches!(
            epoch.status,
            crate::memory::ConversationProjectionEpochStatus::Created
                | crate::memory::ConversationProjectionEpochStatus::AdoptedLegacy
                | crate::memory::ConversationProjectionEpochStatus::Existing
                | crate::memory::ConversationProjectionEpochStatus::Recreated
        ) {
            return Err(persistence_configuration_error(
                "conversation projection epoch is fenced",
            ));
        }
        let (scope_revision, expected, current) = self
            .current_runtime_state(runtime_store.as_ref(), scope_id, runtime_state_id)
            .await?;
        self.require_hydrated_base(current.as_ref()).await?;
        if current
            .as_ref()
            .and_then(|state| state.checkpoint.as_ref())
            .map(crate::state::AgentCheckpoint::restore_managed_runtime_payload)
            .transpose()?
            .is_some_and(|payload| payload.pending_transcript_projection.is_some())
        {
            return Err(crate::error::ReactError::RuntimeState(Box::new(
                echo_core::error::RuntimeStateError::ManagedStateRequiresCas(
                    "checkpoint-only save cannot overwrite pending transcript projection"
                        .to_string(),
                ),
            )));
        }
        let cursor = self
            .snapshot
            .transcript_projection_cursor
            .lock()
            .await
            .checkpoint_for(generation_id);
        let checkpoint = self
            .build_checkpoint(context, blocked_reason, Some(cursor), None)
            .await?;
        let receipt = tokio::time::timeout_at(
            self.deadline,
            runtime_store.compare_and_save_checkpoint_with_context(
                self.call_context,
                crate::state::RuntimeCheckpointCasRequest {
                    scope_id: scope_id.to_string(),
                    runtime_state_id: runtime_state_id.to_string(),
                    conversation_epoch: Some(epoch.authority.epoch),
                    expected_scope_revision: scope_revision,
                    expected_state_version: expected,
                    checkpoint,
                },
            ),
        )
        .await
        .map_err(|_| persistence_deadline_error("checkpoint compare-and-save"))??;
        if !matches!(
            receipt.status,
            crate::state::RuntimeCheckpointCasStatus::Applied
                | crate::state::RuntimeCheckpointCasStatus::AlreadyCurrent
        ) {
            return Err(crate::error::ReactError::RuntimeState(Box::new(
                echo_core::error::RuntimeStateError::ManagedStateRequiresCas(format!(
                    "checkpoint compare-and-save did not settle: {:?}",
                    receipt.status
                )),
            )));
        }
        *self.snapshot.runtime_state_version.lock().await = Some(receipt.version.clone());
        self.snapshot
            .record_checkpoint_event(runtime_state_id)
            .await;
        Ok(())
    }

    /// 结算（settle）当前转录投影：确保对话代存在、处理任何遗留的待处理投影、
    /// 将新的可见消息批量投影并以 compare-and-save 方式原子性地落盘。
    async fn settle_transcript_projection(
        &self,
        context: &Arc<tokio::sync::Mutex<crate::compression::ContextManager>>,
        blocked_reason: Option<String>,
    ) -> crate::error::Result<crate::memory::TranscriptProjectionSettlement> {
        let Some((conversation_store, runtime_store)) = self.managed_stores() else {
            self.save_legacy_checkpoint(context, blocked_reason).await?;
            return Ok(settlement(
                crate::memory::TranscriptProjectionSettlementStatus::Settled,
                None,
                self.snapshot.config.conversation_id.clone(),
                self.snapshot.transcript_generation_id.clone(),
                0,
                None,
                None,
            ));
        };
        let (scope_id, runtime_state_id, generation_id) = self.identities()?;
        let epoch = self
            .ensure_epoch(conversation_store.as_ref(), scope_id)
            .await?;
        if !matches!(
            epoch.status,
            crate::memory::ConversationProjectionEpochStatus::Created
                | crate::memory::ConversationProjectionEpochStatus::AdoptedLegacy
                | crate::memory::ConversationProjectionEpochStatus::Existing
                | crate::memory::ConversationProjectionEpochStatus::Recreated
        ) {
            return Ok(settlement(
                crate::memory::TranscriptProjectionSettlementStatus::Conflict,
                None,
                Some(scope_id.to_string()),
                Some(generation_id.to_string()),
                0,
                Some(crate::memory::TranscriptProjectionErrorClass::SemanticConflict),
                Some("conversation epoch is fenced".to_string()),
            ));
        }

        let (scope_revision, expected, current) = self
            .current_runtime_state(runtime_store.as_ref(), scope_id, runtime_state_id)
            .await?;
        if let Some(state) = current.as_ref()
            && let Some(checkpoint) = state.checkpoint.as_ref()
        {
            let payload = checkpoint.restore_managed_runtime_payload()?;
            if payload.pending_transcript_projection.is_some() {
                let prior = crate::state::settle_loaded_pending_transcript_projection(
                    conversation_store.as_ref(),
                    runtime_store.as_ref(),
                    scope_id,
                    runtime_state_id,
                    state.clone(),
                    crate::state::TranscriptProjectionDispatch::RetryDurablePending,
                    self.call_context,
                )
                .await?
                .settlement;
                if prior.status != crate::memory::TranscriptProjectionSettlementStatus::Settled {
                    return Ok(prior);
                }
                *self.snapshot.runtime_state_version.lock().await = None;
                return Ok(settlement(
                    crate::memory::TranscriptProjectionSettlementStatus::Blocked,
                    None,
                    Some(scope_id.to_string()),
                    Some(generation_id.to_string()),
                    0,
                    Some(crate::memory::TranscriptProjectionErrorClass::RevisionConflict),
                    Some(
                        "runtime pending settled; rehydrate context before a new projection"
                            .to_string(),
                    ),
                ));
            }
        }
        self.require_hydrated_base(current.as_ref()).await?;

        let all_messages = context.lock().await.messages().to_vec();
        let visible_messages = filter_user_visible_transcript(&all_messages);
        if visible_messages.is_empty() {
            self.save_checkpoint(context, blocked_reason).await?;
            return Ok(settlement(
                crate::memory::TranscriptProjectionSettlementStatus::Settled,
                None,
                Some(scope_id.to_string()),
                Some(generation_id.to_string()),
                0,
                None,
                None,
            ));
        }

        let projected = crate::memory::project_messages(scope_id, &visible_messages)?;
        {
            let prepare_attempt = 1_u32;
            let cursor_before = match current.as_ref().and_then(|state| state.checkpoint.as_ref()) {
                Some(checkpoint) => checkpoint
                    .restore_managed_runtime_payload()?
                    .transcript_projection
                    .unwrap_or_else(|| crate::state::TranscriptProjectionCheckpoint {
                        generation_id: generation_id.to_string(),
                        next_ordinal: 0,
                        projected: Vec::new(),
                    }),
                None => self
                    .snapshot
                    .transcript_projection_cursor
                    .lock()
                    .await
                    .checkpoint_for(generation_id),
            };
            let mut working_cursor = TranscriptProjectionCursor {
                generation_id: Some(cursor_before.generation_id.clone()),
                next_ordinal: cursor_before.next_ordinal,
                projected: cursor_before.projected.clone(),
            };
            let assigned = working_cursor.assign(generation_id, &projected)?;
            working_cursor.projected = assigned.clone();
            let cursor_after = working_cursor.checkpoint_for(generation_id);
            let mut new_messages = Vec::new();
            for (mut message, identity) in projected.iter().cloned().zip(assigned.iter()) {
                if identity.ordinal < cursor_before.next_ordinal {
                    continue;
                }
                crate::memory::set_transcript_projection_meta(
                    &mut message,
                    generation_id,
                    identity.ordinal,
                )?;
                new_messages.push(message);
            }
            if new_messages.is_empty() {
                self.save_checkpoint(context, blocked_reason.clone())
                    .await?;
                return Ok(settlement(
                    crate::memory::TranscriptProjectionSettlementStatus::Settled,
                    None,
                    Some(scope_id.to_string()),
                    Some(generation_id.to_string()),
                    0,
                    None,
                    None,
                ));
            }

            let batch = crate::memory::TranscriptProjectionBatch::prepare(
                scope_id,
                epoch.authority.epoch,
                generation_id,
                cursor_before.next_ordinal,
                new_messages,
            )?;
            let pending = crate::state::PendingTranscriptProjection {
                batch: batch.clone(),
                cursor_before: cursor_before.clone(),
                cursor_after,
                base_runtime_revision: Self::next_managed_revision(&expected)?,
                prepared_at: chrono::Utc::now(),
                attempt: prepare_attempt,
                last_attempt_class: None,
                last_error: None,
            };
            let checkpoint = self
                .build_checkpoint(
                    context,
                    blocked_reason.clone(),
                    Some(cursor_before),
                    Some(pending.clone()),
                )
                .await?;
            let prepare_receipt = tokio::time::timeout_at(
                self.deadline,
                runtime_store.compare_and_save_checkpoint_with_context(
                    self.call_context,
                    crate::state::RuntimeCheckpointCasRequest {
                        scope_id: scope_id.to_string(),
                        runtime_state_id: runtime_state_id.to_string(),
                        conversation_epoch: Some(epoch.authority.epoch),
                        expected_scope_revision: scope_revision,
                        expected_state_version: expected.clone(),
                        checkpoint: checkpoint.clone(),
                    },
                ),
            )
            .await
            .map_err(|_| persistence_deadline_error("pending checkpoint prepare"))??;
            match prepare_receipt.status {
                crate::state::RuntimeCheckpointCasStatus::Applied
                | crate::state::RuntimeCheckpointCasStatus::AlreadyCurrent => {
                    *self.snapshot.runtime_state_version.lock().await =
                        Some(prepare_receipt.version.clone());
                    let prepared_state = crate::state::ManagedRuntimeStateSnapshot {
                        scope: prepare_receipt.scope,
                        runtime_state_id: runtime_state_id.to_string(),
                        version: prepare_receipt.version,
                        checkpoint: Some(checkpoint),
                    };
                    let outcome = crate::state::settle_loaded_pending_transcript_projection(
                        conversation_store.as_ref(),
                        runtime_store.as_ref(),
                        scope_id,
                        runtime_state_id,
                        prepared_state,
                        crate::state::TranscriptProjectionDispatch::CurrentAttempt,
                        self.call_context,
                    )
                    .await?;
                    if outcome.settlement.status
                        == crate::memory::TranscriptProjectionSettlementStatus::Settled
                    {
                        if let Some(cursor) = outcome.settled_cursor {
                            self.snapshot
                                .transcript_projection_cursor
                                .lock()
                                .await
                                .restore(cursor);
                        }
                        *self.snapshot.runtime_state_version.lock().await =
                            outcome.committed_version;
                        self.snapshot
                            .record_checkpoint_event(runtime_state_id)
                            .await;
                    }
                    Ok(outcome.settlement)
                }
                crate::state::RuntimeCheckpointCasStatus::RevisionConflict => {
                    let (_, _, new_current) = self
                        .current_runtime_state(runtime_store.as_ref(), scope_id, runtime_state_id)
                        .await?;
                    if let Some(state) = new_current.as_ref()
                        && let Some(checkpoint) = state.checkpoint.as_ref()
                    {
                        let payload = checkpoint.restore_managed_runtime_payload()?;
                        if payload.pending_transcript_projection.is_some() {
                            let prior = crate::state::settle_loaded_pending_transcript_projection(
                                conversation_store.as_ref(),
                                runtime_store.as_ref(),
                                scope_id,
                                runtime_state_id,
                                state.clone(),
                                crate::state::TranscriptProjectionDispatch::RetryDurablePending,
                                self.call_context,
                            )
                            .await?
                            .settlement;
                            if prior.status
                                != crate::memory::TranscriptProjectionSettlementStatus::Settled
                            {
                                return Ok(prior);
                            }
                        }
                    }
                    *self.snapshot.runtime_state_version.lock().await = None;
                    Ok(settlement(
                        crate::memory::TranscriptProjectionSettlementStatus::Blocked,
                        Some(batch.operation_id),
                        Some(scope_id.to_string()),
                        Some(generation_id.to_string()),
                        prepare_attempt,
                        Some(crate::memory::TranscriptProjectionErrorClass::RevisionConflict),
                        Some(
                            "runtime revision changed; rehydrate context before retry".to_string(),
                        ),
                    ))
                }
                crate::state::RuntimeCheckpointCasStatus::GenerationRetired
                | crate::state::RuntimeCheckpointCasStatus::ScopeFenced => Ok(settlement(
                    crate::memory::TranscriptProjectionSettlementStatus::Conflict,
                    Some(batch.operation_id),
                    Some(scope_id.to_string()),
                    Some(generation_id.to_string()),
                    prepare_attempt,
                    Some(crate::memory::TranscriptProjectionErrorClass::SemanticConflict),
                    Some(format!(
                        "pending checkpoint prepare was fenced: {:?}",
                        prepare_receipt.status
                    )),
                )),
            }
        }
    }

    /// 在水合（hydration）或准入（admission）之前，校验对话/运行时作用域的生命周期
    /// 是否有效，并结算任何遗留的待处理（durable pending）转录投影效果。
    async fn reconcile_pending_projection(
        &self,
    ) -> crate::error::Result<Option<crate::memory::TranscriptProjectionSettlement>> {
        let Some((conversation_store, runtime_store)) = self.managed_stores() else {
            return Ok(None);
        };
        let (scope_id, runtime_state_id, _generation_id) = self.identities()?;
        let conversation_authority = tokio::time::timeout_at(
            self.deadline,
            conversation_store.get_projection_authority_with_context(self.call_context, scope_id),
        )
        .await
        .map_err(|_| persistence_deadline_error("conversation authority admission load"))??;
        let scope_authority = tokio::time::timeout_at(
            self.deadline,
            runtime_store.load_scope_authority_with_context(self.call_context, scope_id),
        )
        .await
        .map_err(|_| persistence_deadline_error("runtime scope admission load"))??;
        if conversation_authority.as_ref().is_some_and(|value| {
            value.lifecycle == crate::memory::ConversationProjectionLifecycle::Deleted
        }) {
            return Err(
                echo_core::error::RuntimeStateError::TranscriptProjectionBlocked {
                    status: "Conflict".to_string(),
                    reason: format!(
                        "conversation scope {scope_id} is deleted and must be explicitly recreated"
                    ),
                }
                .into(),
            );
        }
        if let Some(scope) = scope_authority.as_ref() {
            let conversation_is_newer_live = conversation_authority.as_ref().is_some_and(|value| {
                value.lifecycle == crate::memory::ConversationProjectionLifecycle::Live
                    && scope
                        .conversation_epoch
                        .is_none_or(|epoch| value.epoch > epoch)
            });
            let invalid_lifecycle = match scope.lifecycle {
                crate::state::RuntimeScopeLifecycle::Active => match scope.conversation_epoch {
                    Some(epoch) => !conversation_authority.as_ref().is_some_and(|value| {
                        value.lifecycle == crate::memory::ConversationProjectionLifecycle::Live
                            && value.epoch == epoch
                    }),
                    None => scope.revision > 0,
                },
                crate::state::RuntimeScopeLifecycle::Retiring => true,
                crate::state::RuntimeScopeLifecycle::Tombstoned => !conversation_is_newer_live,
            };
            if invalid_lifecycle {
                return Err(
                    echo_core::error::RuntimeStateError::TranscriptProjectionBlocked {
                        status: "Conflict".to_string(),
                        reason: format!(
                            "runtime scope {scope_id} is not admissible in lifecycle {:?}",
                            scope.lifecycle
                        ),
                    }
                    .into(),
                );
            }
        }
        let (_, _, current) = self
            .current_runtime_state(runtime_store.as_ref(), scope_id, runtime_state_id)
            .await?;
        let Some(state) = current else {
            return Ok(None);
        };
        let Some(checkpoint) = state.checkpoint.as_ref() else {
            return Ok(None);
        };
        let payload = checkpoint.restore_managed_runtime_payload()?;
        if payload.pending_transcript_projection.is_none() {
            return Ok(None);
        }
        *self.snapshot.runtime_state_version.lock().await = None;
        let outcome = crate::state::settle_loaded_pending_transcript_projection(
            conversation_store.as_ref(),
            runtime_store.as_ref(),
            scope_id,
            runtime_state_id,
            state,
            crate::state::TranscriptProjectionDispatch::RetryDurablePending,
            self.call_context,
        )
        .await?;
        Ok(Some(outcome.settlement))
    }

    /// 通过旧版（legacy）`RuntimeStateStore` 路径保存检查点（不涉及对话投影）。
    async fn save_legacy_checkpoint(
        &self,
        context: &Arc<tokio::sync::Mutex<crate::compression::ContextManager>>,
        blocked_reason: Option<String>,
    ) -> crate::error::Result<()> {
        let Some(store) = self.snapshot.state_store.as_ref() else {
            return Ok(());
        };
        let Some(runtime_state_id) = self.snapshot.config.runtime_state_id.as_ref() else {
            return Ok(());
        };
        let checkpoint = self
            .build_checkpoint(context, blocked_reason, None, None)
            .await?;
        let scope_id = self
            .snapshot
            .config
            .conversation_id
            .as_deref()
            .unwrap_or(runtime_state_id);
        store
            .save_checkpoint_for_scope(scope_id, &checkpoint)
            .await?;
        self.snapshot
            .record_checkpoint_event(runtime_state_id)
            .await;
        Ok(())
    }
}

/// 构造一个持久化配置错误。
fn persistence_configuration_error(message: impl Into<String>) -> crate::error::ReactError {
    crate::error::ConfigError::ConfigFileError(message.into()).into()
}

/// 构造一个持久化操作超时（超过截止时间）错误。
fn persistence_deadline_error(operation: &str) -> crate::error::ReactError {
    echo_core::error::RuntimeStateError::DeadlineExceeded(operation.to_string()).into()
}

/// 构造一个 [`crate::memory::TranscriptProjectionSettlement`] 值的便捷辅助函数。
fn settlement(
    status: crate::memory::TranscriptProjectionSettlementStatus,
    operation_id: Option<String>,
    conversation_id: Option<String>,
    generation_id: Option<String>,
    attempt: u32,
    error_class: Option<crate::memory::TranscriptProjectionErrorClass>,
    detail: Option<String>,
) -> crate::memory::TranscriptProjectionSettlement {
    crate::memory::TranscriptProjectionSettlement {
        status,
        operation_id,
        conversation_id,
        generation_id,
        attempt,
        error_class,
        detail,
    }
}

impl AgentRunSnapshot {
    /// 标记本次已观测到一次转录结算（settlement）发生。
    pub(crate) fn mark_transcript_settlement_observed(&self) {
        self.transcript_settlement_observed
            .store(true, std::sync::atomic::Ordering::Release);
    }

    /// 重置「已观测到转录结算」标记。
    fn reset_transcript_settlement_observed(&self) {
        self.transcript_settlement_observed
            .store(false, std::sync::atomic::Ordering::Release);
    }

    /// 查询是否已观测到转录结算发生。
    pub(crate) fn transcript_settlement_was_observed(&self) -> bool {
        self.transcript_settlement_observed
            .load(std::sync::atomic::Ordering::Acquire)
    }

    /// 从 [`super::ReactAgent`] 创建一份快照。
    pub fn from_agent(agent: &super::ReactAgent) -> Self {
        let legacy = agent.capture_legacy_external_context();
        Self::from_agent_source(agent, None, Some(&legacy))
    }

    /// 使用给定的旧版外部上下文，从 agent 创建一份快照。
    pub(crate) fn from_agent_with_legacy_context(
        agent: &super::ReactAgent,
        legacy: &crate::agent::react::LegacyExternalContextSnapshot,
    ) -> Self {
        Self::from_agent_source(agent, None, Some(legacy))
    }

    /// 创建一份快照，其运行范围（run-scoped）字段来自单一的调用（invocation）值。
    pub fn from_agent_with_invocation(
        agent: &super::ReactAgent,
        invocation: &echo_core::agent::AgentInvocationContext,
    ) -> Self {
        Self::from_agent_source(agent, Some(invocation), None)
    }

    /// 统一的快照构建实现：融合 agent 配置、当前调用（invocation）上下文
    /// 以及旧版（legacy）外部上下文，生成一份完整的 [`AgentRunSnapshot`]。
    fn from_agent_source(
        agent: &super::ReactAgent,
        invocation: Option<&echo_core::agent::AgentInvocationContext>,
        legacy: Option<&crate::agent::react::LegacyExternalContextSnapshot>,
    ) -> Self {
        let mut config = RuntimeConfig::from_agent_config(&agent.config);
        let configured_runtime_state_id = config.runtime_state_id.clone();
        config.input_modalities = agent
            .llm_config()
            .map(|llm_config| llm_config.input_modalities.clone());
        if let Some(working_dir) = invocation.and_then(|context| context.working_dir.as_ref()) {
            config.working_dir = Some(working_dir.clone());
        }
        if let Some(run_budget) = invocation.and_then(|context| context.run_budget.as_ref()) {
            config.run_budget = run_budget.clone();
        }
        let runtime = invocation.and_then(|context| context.runtime.as_ref());
        if let Some(conversation_id) = runtime.and_then(|context| context.conversation_id.clone()) {
            config.conversation_id = Some(conversation_id.clone());
        } else if invocation.is_none()
            && let Some(conversation_id) =
                legacy.and_then(|context| context.conversation_id.clone())
        {
            config.conversation_id = Some(conversation_id.clone());
        }
        config.runtime_state_id =
            effective_runtime_state_id(configured_runtime_state_id.as_deref(), invocation, legacy)
                .map(str::to_string);
        let transcript_generation_id = invocation
            .and_then(|context| context.transcript_generation_id.clone())
            .or_else(|| config.runtime_state_id.clone());
        let tools = ToolRuntime::from_agent(
            agent,
            invocation.and_then(|context| context.disabled_tools.as_ref()),
            invocation.and_then(|context| context.visible_tools.as_ref()),
        );
        let skill_activation = agent.tools.skill_registry.activation_handle();
        config.plan_mode = tools.is_plan_mode();
        Self {
            config: Arc::new(config),
            context: agent.memory.context.clone(),
            tools: Arc::new(tools),
            guard: Arc::new(GuardRuntime::from_agent(agent)),
            snapshot_manager: agent.memory.snapshot_manager.clone(),
            transcript_generation_id,
            transcript_projection_cursor: Arc::clone(&agent.memory.transcript_projection_cursor),
            runtime_state_version: Arc::clone(&agent.memory.runtime_state_version),
            transcript_settlement_observed: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            client: agent.client().clone(),
            llm_client: agent.llm_client().cloned(),
            thinking: agent.thinking().cloned(),
            cancel_token: invocation.and_then(|context| {
                context.cancel.clone().or_else(|| {
                    runtime
                        .and_then(|value| value.cancel.as_ref())
                        .map(|cancel| cancel.as_ref().clone())
                })
            }),
            turn_steer_mailbox: Arc::clone(&agent.turn_steer_mailbox),
            recently_read_files: Arc::clone(&agent.recently_read_files),
            run_store: agent.run_store.clone(),
            diagnostic_delivery_observer: agent.diagnostic_delivery_observer.clone(),
            current_run_id: if invocation.is_some() {
                runtime.and_then(|context| context.run_id.clone())
            } else {
                legacy.and_then(|context| context.current_run_id.clone())
            },
            trace_run_id: if invocation.is_some() {
                None
            } else {
                agent.capture_current_trace_run_id()
            },
            executing_tool_call_ids: Arc::new(std::sync::Mutex::new(
                std::collections::HashSet::new(),
            )),
            current_turn_id: if invocation.is_some() {
                runtime.and_then(|context| context.turn_id.clone())
            } else {
                legacy.and_then(|context| context.turn_id.clone())
            },
            turn_steer_incarnation: None,
            current_message_id: if invocation.is_some() {
                runtime.and_then(|context| context.message_id.clone())
            } else {
                legacy.and_then(|context| context.message_id.clone())
            },
            current_message: None,
            current_execution_id: if invocation.is_some() {
                runtime.and_then(|context| context.execution_id.clone())
            } else {
                legacy.and_then(|context| context.execution_id.clone())
            },
            external_cancel: if let Some(context) = invocation {
                runtime
                    .and_then(|value| value.cancel.clone())
                    .or_else(|| context.cancel.clone().map(Arc::new))
            } else {
                legacy.and_then(|context| context.cancel.clone())
            },
            external_trace_sink: if invocation.is_some() {
                runtime.and_then(|context| context.trace_sink.clone())
            } else {
                legacy.and_then(|context| context.trace_sink.clone())
            },
            external_delegation_policy: if invocation.is_some() {
                runtime.and_then(|context| context.delegation_policy)
            } else {
                legacy.and_then(|context| context.delegation_policy)
            },
            subagent_lineage: if invocation.is_some() {
                runtime.and_then(|context| context.subagent_lineage.clone())
            } else {
                None
            },
            external_uplink: if invocation.is_some() {
                runtime.and_then(|context| context.uplink.clone())
            } else {
                None
            },
            resource_guards: if let Some(context) = invocation {
                let mut guards = runtime
                    .map(|runtime| runtime.resource_guards.clone())
                    .unwrap_or_default();
                guards.extend(context.resource_guards.iter().cloned());
                guards
            } else {
                legacy
                    .map(|context| context.resource_guards.clone())
                    .unwrap_or_default()
            },
            #[cfg(feature = "human-loop")]
            permission_service: agent.approval.permission_service.clone(),
            #[cfg(feature = "human-loop")]
            pending_permission_rules: Arc::new(tokio::sync::Mutex::new(
                agent
                    .approval
                    .pending_permission_rules
                    .lock()
                    .map(|mut rules| std::mem::take(&mut *rules))
                    .unwrap_or_default(),
            )),
            token_tracker: Arc::clone(&agent.token_tracker),
            calibrated_tokenizer: Arc::clone(&agent.calibrated_tokenizer),
            state_store: agent.memory.state_store.clone(),
            conversation_store: agent.memory.conversation_store.clone(),
            memory_store: agent.memory.store.clone(),
            critic: agent.critic.clone(),
            tool_execution_pipeline: agent.tool_execution_pipeline.clone(),
            memory_layer_manager: agent.memory_layer_manager.clone(),
            pre_model_context_projector: agent
                .pre_model_context_projector
                .read()
                .unwrap_or_else(|error| error.into_inner())
                .clone(),
            skill_mutation_authority: agent.skill_mutation_authority.clone(),
            skill_activation,
        }
    }

    /// 返回本次调用当前已激活的技能（skill）名称列表。
    fn active_skill_names(&self) -> Vec<String> {
        self.skill_activation.activated_names()
    }

    /// 为本次调用中每个已激活的技能持久化一条尽力而为（best-effort）的观测记录。
    /// 遥测（telemetry）只是可观测性数据，不是执行权威：存储缺失或写入失败绝不能
    /// 影响工具的执行结果。
    fn record_skill_telemetry(
        &self,
        tool_name: &str,
        duration_ms: u64,
        success: bool,
        error: Option<&str>,
    ) {
        let Some(store) = self.memory_store.clone() else {
            return;
        };
        let skill_names = self.active_skill_names();
        if skill_names.is_empty() {
            return;
        }
        let session_id = self.config.session_id.clone().unwrap_or_default();
        let tool_name = tool_name.to_string();
        let error_message = error.map(str::to_string);
        let skill_authority = self.skill_mutation_authority.clone();
        let activated_at = u64::try_from(chrono::Utc::now().timestamp_millis()).unwrap_or(0);
        tokio::spawn(async move {
            let telemetry_store = crate::skill_telemetry::SkillTelemetryStore::new(store);
            for skill_name in skill_names {
                let record = crate::skill_telemetry::SkillExecutionRecord {
                    skill_name: skill_name.clone(),
                    session_id: session_id.clone(),
                    activated_at,
                    duration_ms,
                    tools_used: vec![tool_name.clone()],
                    tool_calls_count: 1,
                    success,
                    error_message: error_message.clone(),
                };
                if let Err(write_error) = telemetry_store.record_execution(&record).await {
                    tracing::warn!(
                        skill = %skill_name,
                        error = %write_error,
                        "skill telemetry write failed"
                    );
                }
                if let Some(authority) = skill_authority.as_ref() {
                    let usage = crate::evolution::SkillUsageHandle::new(authority.clone());
                    if let Err(error) = usage.record_usage(&skill_name).await {
                        tracing::warn!(
                            skill = %skill_name,
                            error = %error,
                            "skill usage lifecycle mutation failed"
                        );
                    }
                }
            }
        });
    }

    // ── Trace 辅助方法 ──────────────────────────────────────────────

    /// 上报一次诊断投递（Trace/Audit 持久化）失败事件。
    fn report_diagnostic_delivery_failure(&self, failure: DiagnosticDeliveryFailure) {
        crate::audit::report_diagnostic_delivery_failure(
            self.diagnostic_delivery_observer.clone(),
            failure,
        );
    }

    /// 通过可选的 Agent 回调后端持久化一条审计事件。
    pub(crate) async fn record_audit_event(&self, mut event: crate::audit::AuditEvent) {
        let Some(logger) = self.guard.audit_logger.as_ref() else {
            return;
        };
        event.apply_retention(&echo_core::utils::retention::ContentRetentionPolicy::default());
        let record_id = event.trace_id.clone().or_else(|| event.session_id.clone());
        if let Err(error) = logger.log(event).await {
            self.report_diagnostic_delivery_failure(DiagnosticDeliveryFailure::new(
                DiagnosticRecordKind::Audit,
                DiagnosticDeliveryOperation::Record,
                record_id,
                error.to_string(),
            ));
        }
    }

    /// 若已附加运行存储（run store），则记录一次 trace 事件。
    pub async fn record_event(&self, mut event: RunEvent) {
        event.apply_retention(&echo_core::utils::retention::ContentRetentionPolicy::default());
        if let Some(ref store) = self.run_store
            && let Some(ref run_id) = self.trace_run_id
            && let Err(error) = store.append_event(run_id, event).await
        {
            self.report_diagnostic_delivery_failure(DiagnosticDeliveryFailure::new(
                DiagnosticRecordKind::Trace,
                DiagnosticDeliveryOperation::Append,
                Some(run_id.clone()),
                error.to_string(),
            ));
        }
    }

    /// 将拥有该效果（effect）的工具所产生的事实投影记录到本次调用的 trace 中。
    pub(crate) async fn record_tool_effect(
        &self,
        effect: &echo_core::tools::ToolEffect,
        tool_name: &str,
        call_id: &str,
    ) {
        let event = match effect {
            echo_core::tools::ToolEffect::FileRead { path } => RunEvent::FileRead {
                tool: tool_name.to_string(),
                path: path.clone(),
            },
            echo_core::tools::ToolEffect::FileEdit { path } => RunEvent::FileEdit {
                tool: tool_name.to_string(),
                path: path.clone(),
            },
            echo_core::tools::ToolEffect::TestRun {
                command,
                passed,
                failure_count,
            } => RunEvent::TestRun {
                command: command.clone(),
                passed: *passed,
                failure_count: *failure_count,
            },
            echo_core::tools::ToolEffect::SubagentRun {
                agent_name,
                task,
                outcome,
            } => RunEvent::SubagentRun {
                call_id: Some(call_id.to_string()),
                agent_name: agent_name.clone(),
                task: task.clone(),
                outcome: outcome.clone(),
            },
        };
        self.record_event(event).await;
    }

    /// 标记指定工具调用已开始执行。
    pub(crate) fn mark_tool_execution_started(&self, call_id: &str) {
        self.executing_tool_call_ids
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .insert(call_id.to_string());
    }

    /// 查询指定工具调用是否已开始执行。
    fn tool_execution_started(&self, call_id: &str) -> bool {
        self.executing_tool_call_ids
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .contains(call_id)
    }

    /// 结算一次被中断的工具调用：记录必要的 trace/audit 事件，并构造一个
    /// 带有「可能存在副作用」标记的失败结果，供上层决定是否需要人工核实。
    pub(crate) async fn settle_interrupted_tool_call(
        &self,
        call_id: &str,
        tool_name: &str,
        input: &serde_json::Value,
        category: crate::tools::ToolFailureCategory,
        message: &str,
    ) -> crate::tools::ToolResult {
        if !self.tool_execution_started(call_id) {
            /*
             * 某个预执行阶段或未来波次从未进入过 ExecuteStage。
             * 调用范围内的本地集合才是权威数据；trace 持久化只是一种投影，
             * 可能不可用，或者仅最终一致可见。
             */
            self.record_event(RunEvent::new_tool_call(
                call_id.to_string(),
                tool_name.to_string(),
                Some(input.clone()),
                None,
                0,
            ))
            .await;
            self.record_event(RunEvent::ToolExecutionSkipped {
                call_id: call_id.to_string(),
                name: tool_name.to_string(),
                reason: category.as_str().to_string(),
            })
            .await;
        }
        let failure = crate::tools::ToolFailure::new(category)
            .with_side_effect(crate::tools::ToolSideEffect::Possible)
            .with_postcondition(
                "verify the external effect before retrying this interrupted tool call",
            );
        let result = crate::tools::ToolResult::error(message).with_failure(failure.clone());
        self.record_event(crate::trace::RunEvent::ToolResult {
            call_id: call_id.to_string(),
            name: tool_name.to_string(),
            success: false,
            output_preview: Some(String::new()),
            output_truncated: false,
            duration_ms: 0,
            original_bytes: 0,
            returned_bytes: 0,
            estimated_tokens: 0,
            output_handling: None,
            artifact: None,
        })
        .await;
        self.record_event(crate::trace::RunEvent::ToolError {
            call_id: call_id.to_string(),
            name: tool_name.to_string(),
            message: message.to_string(),
            failure: Some(failure),
        })
        .await;
        self.record_audit_event(crate::audit::AuditEvent::now(
            self.config.session_id.clone(),
            self.config.agent_name.clone(),
            crate::audit::AuditEventType::ToolCall {
                call_id: Some(call_id.to_string()),
                tool: tool_name.to_string(),
                input: input.clone(),
                output: message.to_string(),
                success: false,
                duration_ms: 0,
            },
        ))
        .await;
        let error = crate::error::ReactError::Other(message.to_string());
        for callback in &self.config.callbacks {
            callback
                .on_tool_interrupted_with_id(
                    &self.config.agent_name,
                    call_id,
                    tool_name,
                    input,
                    &error,
                )
                .await;
        }
        result
    }

    /// 结束（finalize）当前的 trace run（完成或失败）。
    pub async fn finalize_run(&self, status: RunStatus, output: Option<&str>, error: Option<&str>) {
        if let Some(ref store) = self.run_store
            && let Some(ref run_id) = self.trace_run_id
        {
            match store.finalize_run(run_id, status, output, error).await {
                Ok(true) => {}
                Ok(false) => {
                    self.report_diagnostic_delivery_failure(DiagnosticDeliveryFailure::new(
                        DiagnosticRecordKind::Trace,
                        DiagnosticDeliveryOperation::Load,
                        Some(run_id.clone()),
                        "trace run not found during finalization",
                    ))
                }
                Err(finalize_error) => {
                    self.report_diagnostic_delivery_failure(DiagnosticDeliveryFailure::new(
                        DiagnosticRecordKind::Trace,
                        DiagnosticDeliveryOperation::Finalize,
                        Some(run_id.clone()),
                        finalize_error.to_string(),
                    ))
                }
            }
        }
    }

    /// 由规范的工具批次（tool-batch）持有方触发聚合生命周期 hook。
    pub(crate) async fn fire_post_tool_batch(
        &self,
        tool_names: &[String],
        success_count: usize,
        failure_count: usize,
    ) {
        let context = crate::skills::hooks::HookContext::for_post_tool_batch(
            tool_names,
            success_count,
            failure_count,
            self.config.session_id.as_deref().unwrap_or(""),
            &self.config.agent_name,
        );
        let registry = self.tools.hook_registry.read().await.clone();
        let _ = registry.run_lifecycle_hooks(&context).await;
    }

    // ── 运行时状态检查点 ─────────────────────────────────────

    /**
     * 将一份完整的检查点保存到 [`RuntimeStateStore`](crate::state::RuntimeStateStore)。
     *
     * 持久化 [`crate::state::AgentCheckpoint`] 运行时状态（消息、已激活技能、
     * 阻塞原因），使一个正在进行中的对话能够在进程重启后恢复。
     *
     * 若未配置状态存储或运行时状态身份标识，则静默地不做任何操作。
     */
    pub async fn save_runtime_checkpoint(
        &self,
        context: &Arc<tokio::sync::Mutex<crate::compression::ContextManager>>,
        blocked_reason: Option<String>,
    ) -> crate::error::Result<()> {
        validate_transcript_generation_identity(
            self.config.runtime_state_id.as_deref(),
            self.transcript_generation_id.as_deref(),
        )?;
        if self.conversation_store.is_none() {
            return AgentPersistenceCoordinator::new(self)?
                .save_checkpoint(context, blocked_reason)
                .await;
        }
        let settlement = self
            .save_transcript_projection(context, blocked_reason)
            .await?;
        match settlement.status {
            crate::memory::TranscriptProjectionSettlementStatus::Settled => Ok(()),
            crate::memory::TranscriptProjectionSettlementStatus::Deferred => Err(
                echo_core::error::RuntimeStateError::TranscriptProjectionDeferred {
                    operation_id: settlement.operation_id,
                    reason: settlement.detail.unwrap_or_else(|| {
                        "checkpoint retained a durable transcript projection debt".to_string()
                    }),
                }
                .into(),
            ),
            crate::memory::TranscriptProjectionSettlementStatus::Blocked
            | crate::memory::TranscriptProjectionSettlementStatus::Conflict => {
                Err(transcript_settlement_admission_error(&settlement))
            }
        }
    }

    /// 在水合（hydration）或准入（admission）之前，结算一个遗留的持久化待处理转录效果。
    pub(crate) async fn reconcile_pending_transcript_projection(
        &self,
    ) -> crate::error::Result<Option<crate::memory::TranscriptProjectionSettlement>> {
        let settlement = AgentPersistenceCoordinator::new(self)?
            .reconcile_pending_projection()
            .await?;
        if let Some(settlement) = settlement.as_ref() {
            self.record_event(crate::trace::RunEvent::TranscriptProjectionSettlement {
                settlement: settlement.clone(),
            })
            .await;
        }
        Ok(settlement)
    }

    /// 将一次持久化错误分类并转换为对应的转录投影结算（settlement）状态，
    /// 记录相应的 trace 事件并返回该结算结果。
    pub(crate) async fn observe_persistence_failure(
        &self,
        error: &crate::error::ReactError,
    ) -> crate::memory::TranscriptProjectionSettlement {
        let (classified_status, error_class) =
            crate::state::classify_transcript_persistence_error(error);
        let status = if classified_status
            == crate::memory::TranscriptProjectionSettlementStatus::Deferred
            && matches!(
                error,
                crate::error::ReactError::RuntimeState(inner)
                    if matches!(
                        inner.as_ref(),
                        echo_core::error::RuntimeStateError::TranscriptProjectionDeferred { .. }
                    )
            ) {
            crate::memory::TranscriptProjectionSettlementStatus::Deferred
        } else {
            crate::memory::TranscriptProjectionSettlementStatus::Blocked
        };
        let operation_id = match error {
            crate::error::ReactError::RuntimeState(inner) => match inner.as_ref() {
                echo_core::error::RuntimeStateError::TranscriptProjectionDeferred {
                    operation_id,
                    ..
                } => operation_id.clone(),
                _ => None,
            },
            _ => None,
        };
        let settlement = settlement(
            status,
            operation_id,
            self.config.conversation_id.clone(),
            self.transcript_generation_id.clone(),
            0,
            Some(error_class),
            Some(error.to_string()),
        );
        self.record_event(crate::trace::RunEvent::TranscriptProjectionSettlement {
            settlement: settlement.clone(),
        })
        .await;
        settlement
    }

    /// 记录一次检查点（checkpoint）事件，供 trace 观测。
    async fn record_checkpoint_event(&self, runtime_state_id: &str) {
        self.record_event(crate::trace::RunEvent::Checkpoint {
            id: format!(
                "checkpoint:{}:{}",
                runtime_state_id,
                chrono::Utc::now().timestamp_millis()
            ),
        })
        .await;
        tracing::debug!(
            runtime_state_id,
            "Runtime checkpoint compare-and-save settled"
        );
    }

    /**
     * 将用户可见的转录投影保存到
     * [`ConversationStore`](crate::memory::ConversationStore)。
     *
     * 将用户可见的消息投影为一个「代（epoch）围栏化」的原子批次，同时在同一个
     * 带修订号的检查点中保存完整的运行时 `Message` 列表。托管路径下的
     * `save_runtime_checkpoint` 会调用本协调器，因此在没有已结算投影或持久化
     * 待处理标记的情况下，运行时状态无法推进。
     *
     * 这统一了框架中的转录持久化逻辑：此前每个产品入口（Tauri 命令、终端 UI
     * 循环）都必须各自调用 `save_messages`。现在 `run_core_loop` 会在
     * pre-model 与 finalization 安全点调用本辅助方法，产品层只需处理对话元数据
     * （标题 / 置顶 / agent_type）。
     *
     * 若未配置对话存储或 `conversation_id`，则静默地不做任何操作。
     */
    pub async fn save_transcript_projection(
        &self,
        context: &Arc<tokio::sync::Mutex<crate::compression::ContextManager>>,
        blocked_reason: Option<String>,
    ) -> crate::error::Result<crate::memory::TranscriptProjectionSettlement> {
        self.reset_transcript_settlement_observed();
        let settlement = AgentPersistenceCoordinator::new(self)?
            .settle_transcript_projection(context, blocked_reason)
            .await?;
        if self.conversation_store.is_some() {
            self.record_event(crate::trace::RunEvent::TranscriptProjectionSettlement {
                settlement: settlement.clone(),
            })
            .await;
        }
        Ok(settlement)
    }

    /**
     * 在压缩（compaction）替换掉活跃上下文之后重新对齐代（generation）游标。
     * 压缩前的完整转录已经被保存过，因此保留下来的消息会沿用其原有序号，
     * 只有更晚的消息才会被分配新的序号。
     */
    pub(crate) async fn realign_transcript_projection(
        &self,
        context: &Arc<tokio::sync::Mutex<crate::compression::ContextManager>>,
    ) -> crate::error::Result<()> {
        let Some(generation_id) = self.transcript_generation_id.as_deref() else {
            return Ok(());
        };
        let messages = {
            let context = context.lock().await;
            filter_user_visible_transcript(context.messages())
        };
        let conversation_id = self
            .config
            .conversation_id
            .as_deref()
            .unwrap_or(generation_id);
        let projected = crate::memory::project_messages(conversation_id, &messages)?;
        let mut cursor = self.transcript_projection_cursor.lock().await;
        let assigned = cursor.assign(generation_id, &projected)?;
        cursor.projected = assigned;
        Ok(())
    }

    // ── 工具执行辅助方法（委托给流水线各阶段） ─────

    /**
     * 通过 PermissionService 检查工具调用是否获得批准。
     * 若批准过程修改了工具调用输入，则返回修改后的输入，否则返回 `None`。
     */
    #[cfg(feature = "human-loop")]
    pub async fn check_tool_approval(
        &self,
        request_id: &str,
        tool_name: &str,
        input: &serde_json::Value,
        permission_mode_override: Option<echo_core::tools::permission::PermissionMode>,
    ) -> std::result::Result<Option<serde_json::Value>, echo_core::error::ReactError> {
        if let Some(ref service) = self.permission_service {
            let pending = {
                let mut rules = self.pending_permission_rules.lock().await;
                std::mem::take(&mut *rules)
            };
            if !pending.is_empty() {
                service.add_rules(pending).await;
            }
            let permissions = self
                .tools
                .tool_manager
                .get_tool(tool_name)
                .map(|tool| tool.permissions())
                .unwrap_or_default();
            let classifier_context = {
                let messages = self.context.lock().await.messages().to_vec();
                let recent_files = self
                    .recently_read_files
                    .lock()
                    .unwrap_or_else(|error| error.into_inner())
                    .keys()
                    .cloned()
                    .collect();
                let mut context = echo_orchestration::human_loop::ClassifierContext::new()
                    .with_messages(messages)
                    .with_recent_files(recent_files)
                    .with_risk_context(echo_orchestration::human_loop::RiskContext {
                        has_sensitive_files: false,
                        is_destructive: permissions.iter().any(|permission| {
                            matches!(
                                permission,
                                echo_core::tools::permission::ToolPermission::Write
                                    | echo_core::tools::permission::ToolPermission::Execute
                                    | echo_core::tools::permission::ToolPermission::Sensitive
                            )
                        }),
                        directory_depth: self
                            .config
                            .working_dir
                            .as_ref()
                            .map_or(0, |path| path.components().count()),
                        repetition_count: 0,
                    });
                if let Some(working_dir) = self.config.working_dir.as_ref() {
                    context = context.with_workspace_path(working_dir.display().to_string());
                }
                context
            };
            let permission_scope_id = self
                .config
                .conversation_id
                .as_deref()
                .or(self.config.session_id.as_deref())
                .map(|session| format!("{}:{}", self.config.agent_name, session));
            let permission_context = echo_orchestration::human_loop::PermissionInvocationContext {
                scope_id: permission_scope_id,
                request_id: Some(request_id.to_string()),
                session_id: self
                    .config
                    .conversation_id
                    .clone()
                    .or_else(|| self.config.session_id.clone()),
                agent_name: Some(self.config.agent_name.clone()),
                timeout: None,
                permission: echo_orchestration::human_loop::PermissionContext {
                    working_directory: self
                        .config
                        .working_dir
                        .as_ref()
                        .map(|path| path.display().to_string()),
                    affected_files: classifier_context.recent_files.clone(),
                    estimated_impact: None,
                    metadata: serde_json::Map::new(),
                },
                classifier: classifier_context,
            };
            let check = service.check_with_permissions_result_in_mode_and_context(
                tool_name,
                input,
                &permissions,
                permission_mode_override,
                Some(&permission_context),
            );
            tokio::pin!(check);
            let decision = if let Some(cancel) = self.cancel_token.as_ref() {
                tokio::select! {
                    biased;
                    _ = cancel.cancelled() => {
                        return Err(echo_core::error::ReactError::Agent(Box::new(
                            echo_core::error::AgentError::Cancelled(format!(
                                "permission request for tool '{tool_name}'"
                            )),
                        )));
                    }
                    decision = &mut check => decision?,
                }
            } else {
                check.await?
            };
            self.record_permission_decision(tool_name, &decision.decision, "permission_service")
                .await;
            match decision.decision {
                echo_core::tools::permission::PermissionDecision::Allow => {
                    Ok(decision.updated_input)
                }
                echo_core::tools::permission::PermissionDecision::Deny { reason } => {
                    Err(echo_core::error::ReactError::Other(format!(
                        "Permission denied for tool '{}': {}",
                        tool_name, reason
                    )))
                }
                echo_core::tools::permission::PermissionDecision::RequireApproval => {
                    Err(echo_core::error::ReactError::Other(format!(
                        "Tool '{}' requires user approval",
                        tool_name
                    )))
                }
                echo_core::tools::permission::PermissionDecision::Ask { suggestions } => {
                    Err(echo_core::error::ReactError::Other(format!(
                        "Tool '{}' requires user approval. Suggestions: {:?}",
                        tool_name, suggestions
                    )))
                }
            }
        } else {
            Ok(None)
        }
    }

    /// 通过 PermissionService 检查工具调用是否获得批准（当 human-loop 特性未启用时为空操作）。
    #[cfg(not(feature = "human-loop"))]
    pub async fn check_tool_approval(
        &self,
        _request_id: &str,
        _tool_name: &str,
        _input: &serde_json::Value,
        _permission_mode_override: Option<echo_core::tools::permission::PermissionMode>,
    ) -> std::result::Result<Option<serde_json::Value>, echo_core::error::ReactError> {
        Ok(None)
    }

    /// 记录一次文件读取，用于「编辑前必须先读取」策略的执行。
    pub fn record_file_read(&self, path: &str) {
        let path = std::path::Path::new(path);
        let resolved = if path.is_absolute() {
            path.to_path_buf()
        } else if let Some(working_dir) = self.config.working_dir.as_deref() {
            working_dir.join(path)
        } else {
            path.to_path_buf()
        };
        let canonical = std::fs::canonicalize(&resolved)
            .unwrap_or(resolved)
            .to_string_lossy()
            .to_string();
        let mut files = self
            .recently_read_files
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        files.insert(canonical, std::time::Instant::now());
    }

    /// 检查工具输出守卫（guard），若输出被修改则返回过滤后的结果。
    pub async fn check_tool_output_guard(&self, output: &str) -> Option<String> {
        let mut effective_output = output.to_string();
        if crate::security::contains_secrets(&effective_output) {
            effective_output = crate::security::redact_secrets(&effective_output);
            tracing::warn!(agent = %self.config.agent_name, "Secret detected in tool output; redacted");
        }
        let Some(gm) = self.guard.guard_manager.as_ref() else {
            return (effective_output != output).then_some(effective_output);
        };
        use crate::guard::GuardDirection;
        let result = match gm
            .check_all(&effective_output, GuardDirection::Output)
            .await
        {
            Ok(r) => r,
            Err(e) => {
                tracing::error!(agent = %self.config.agent_name, error = %e, "Guard check failed, blocking output (fail-closed)");
                return Some(format!("Output content blocked: guard check error ({e})"));
            }
        };
        match result {
            crate::guard::GuardResult::Block { reason } => {
                tracing::info!(agent = %self.config.agent_name, reason = %reason, "🛡️ Tool output blocked by guard");
                if self.guard.audit_logger.is_some() {
                    let event = crate::audit::AuditEvent::now(
                        self.config.session_id.clone(),
                        self.config.agent_name.clone(),
                        crate::audit::AuditEventType::GuardBlock {
                            guard: "guard_manager".to_string(),
                            direction: GuardDirection::Output,
                            reason: reason.clone(),
                        },
                    );
                    self.record_audit_event(event).await;
                }
                Some(format!("Output content filtered by safety guard: {reason}"))
            }
            crate::guard::GuardResult::Transform { content, .. } => Some(content),
            crate::guard::GuardResult::Pass | crate::guard::GuardResult::Warn { .. } => {
                (effective_output != output).then_some(effective_output)
            }
        }
    }

    /// 对工具输出应用唯一权威的溢出（spill）/截断策略。
    pub(crate) fn process_tool_output(&self, output: String) -> ProcessedToolOutput {
        self.process_tool_output_for_call(output, "unscoped", "tool", None)
    }

    /// 针对具体某次工具调用应用输出溢出/截断策略：超出阈值时落盘为产物（artifact），
    /// 否则按 token 预算截断；若落盘失败则回退为按 token 数截断。
    pub(crate) fn process_tool_output_for_call(
        &self,
        output: String,
        call_id: &str,
        tool_name: &str,
        existing_artifact: Option<echo_core::tools::artifact::ToolOutputArtifactRef>,
    ) -> ProcessedToolOutput {
        let inline_bytes = output.len();
        let estimated_tokens = echo_core::tokenizer::HeuristicTokenizer.count_tokens(&output);
        let mut spill_error = None;
        let artifact = existing_artifact.or_else(|| {
            let mut config = self.config.tool_output_artifacts.clone()?;
            let exceeds_token_budget = self
                .config
                .max_tool_output_tokens
                .is_some_and(|max_tokens| estimated_tokens > max_tokens);
            if inline_bytes < config.threshold_bytes && !exceeds_token_budget {
                return None;
            }
            if exceeds_token_budget {
                config.threshold_bytes = 1;
            }
            let identity = echo_core::tools::artifact::ToolOutputArtifactIdentity {
                conversation_id: self.config.conversation_id.clone(),
                run_id: self
                    .current_run_id
                    .clone()
                    .or_else(|| self.current_turn_id.clone()),
                call_id: call_id.to_string(),
                tool_name: tool_name.to_string(),
            };
            match echo_core::tools::artifact::persist_tool_output(config, identity, &output) {
                Ok(artifact) => artifact,
                Err(error) => {
                    tracing::warn!(error = %error, "tool output artifact write failed; falling back to token truncation");
                    spill_error = Some(error.to_string());
                    None
                }
            }
        });

        if let Some(artifact) = artifact {
            let preview: String = output.chars().take(TOOL_OUTPUT_PREVIEW_CHARS).collect();
            let model_output = format!(
                "{preview}\n\n[Tool output preview only: the text above is not a summary and is not the complete result. Full output artifact: {} ({:.1} MiB, sha256 {}). Use read_artifact with this exact path and expected_sha256 to retrieve bounded pages until complete.]",
                artifact.path.display(),
                artifact.payload_bytes as f64 / 1_048_576.0,
                artifact.sha256,
            );
            let mut metadata = std::collections::HashMap::new();
            metadata.insert("output_handling".to_string(), "spilled".to_string());
            metadata.insert(
                "original_bytes".to_string(),
                artifact.payload_bytes.to_string(),
            );
            metadata.insert("returned_bytes".to_string(), model_output.len().to_string());
            metadata.insert("estimated_tokens".to_string(), estimated_tokens.to_string());
            return ProcessedToolOutput {
                output: model_output,
                truncated: true,
                artifact: Some(artifact),
                metadata,
            };
        }

        let max_tokens = self.config.max_tool_output_tokens.or_else(|| {
            spill_error
                .as_ref()
                .map(|_| TOOL_OUTPUT_SPILL_FAILURE_FALLBACK_TOKENS)
        });
        let Some(max_tokens) = max_tokens else {
            let estimated_tokens = echo_core::tokenizer::HeuristicTokenizer.count_tokens(&output);
            let mut metadata = std::collections::HashMap::new();
            metadata.insert("output_handling".to_string(), "inline".to_string());
            metadata.insert("original_bytes".to_string(), inline_bytes.to_string());
            metadata.insert("returned_bytes".to_string(), inline_bytes.to_string());
            metadata.insert("estimated_tokens".to_string(), estimated_tokens.to_string());
            return ProcessedToolOutput {
                output,
                truncated: false,
                artifact: None,
                metadata,
            };
        };

        let tokenizer = echo_core::tokenizer::HeuristicTokenizer;
        let estimated_tokens = tokenizer.count_tokens(&output);
        if estimated_tokens <= max_tokens {
            let mut metadata = std::collections::HashMap::new();
            metadata.insert("output_handling".to_string(), "inline".to_string());
            metadata.insert("original_bytes".to_string(), inline_bytes.to_string());
            metadata.insert("returned_bytes".to_string(), inline_bytes.to_string());
            metadata.insert("estimated_tokens".to_string(), estimated_tokens.to_string());
            return ProcessedToolOutput {
                output,
                truncated: false,
                artifact: None,
                metadata,
            };
        }

        let notice = format!(
            "\n\n[Output truncated: ~{estimated_tokens} tokens total, {max_tokens} token budget]\n\n"
        );
        let notice_tokens = tokenizer.count_tokens(&notice);
        let available_tokens = max_tokens.saturating_sub(notice_tokens);
        let available_chars = available_tokens.saturating_mul(4);
        let head_chars = available_chars.saturating_mul(7) / 10;
        let tail_chars = available_chars.saturating_sub(head_chars);
        let head: String = output.chars().take(head_chars).collect();
        let tail_reversed: String = output.chars().rev().take(tail_chars).collect();
        let tail: String = tail_reversed.chars().rev().collect();
        let truncated_output = if available_tokens == 0 {
            format!("[Output truncated: ~{estimated_tokens} tokens total]")
        } else {
            format!("{head}{notice}{tail}")
        };
        let mut metadata = std::collections::HashMap::new();
        metadata.insert(
            "output_handling".to_string(),
            if spill_error.is_some() {
                "spill_failed_truncated"
            } else {
                "truncated"
            }
            .to_string(),
        );
        metadata.insert("original_bytes".to_string(), inline_bytes.to_string());
        metadata.insert(
            "returned_bytes".to_string(),
            truncated_output.len().to_string(),
        );
        metadata.insert("estimated_tokens".to_string(), estimated_tokens.to_string());
        if let Some(error) = spill_error {
            metadata.insert("spill_error".to_string(), error);
        }
        ProcessedToolOutput {
            output: truncated_output,
            truncated: true,
            artifact: None,
            metadata,
        }
    }

    /// 供旧版内部调用方使用的、向后兼容的字符串视图。
    pub async fn truncate_tool_output(&self, output: String) -> String {
        self.process_tool_output(output).output
    }

    // ── 生命周期 hook 分发 ───────────────────────────────────────

    /**
     * 触发一次生命周期 hook（`SessionEnd` / `PreCompact` / `StopFailure`）。
     * 供准备（prepare）/压缩（compact）/结束（finalize）/达到最大迭代次数阶段使用。
     */
    pub(crate) async fn fire_hook(
        &self,
        event: crate::skills::hooks::HookEvent,
        matcher: Option<&str>,
    ) {
        let sid = self.config.session_id.clone().unwrap_or_default();
        let hc = match event {
            crate::skills::hooks::HookEvent::SessionEnd => {
                crate::skills::hooks::HookContext::for_session_end(
                    matcher.unwrap_or("other"),
                    &sid,
                    &self.config.agent_name,
                )
            }
            crate::skills::hooks::HookEvent::PreCompact => {
                crate::skills::hooks::HookContext::for_pre_compact(
                    &Default::default(),
                    matcher.unwrap_or("auto"),
                    &sid,
                    &self.config.agent_name,
                )
            }
            crate::skills::hooks::HookEvent::StopFailure => {
                crate::skills::hooks::HookContext::for_stop_failure(
                    "",
                    matcher.unwrap_or(""),
                    &sid,
                    &self.config.agent_name,
                )
            }
            _ => return,
        };
        let reg = self.tools.hook_registry.read().await.clone();
        let _ = reg.run_lifecycle_hooks(&hc).await;
    }

    // ── 自动快照（记忆快照捕获） ──────────────────────

    /// 若快照管理器表明当前迭代应当捕获快照，则捕获一次记忆快照。
    pub(crate) async fn auto_snapshot(
        &self,
        context: &Arc<tokio::sync::Mutex<crate::compression::ContextManager>>,
        iteration: usize,
    ) {
        let should_capture = {
            let mgr = self
                .snapshot_manager
                .read()
                .unwrap_or_else(|e| e.into_inner());
            mgr.as_ref().is_some_and(|m| m.should_capture(iteration))
            // RwLockReadGuard 在此处被释放 —— 发生在任何 await 之前
        };
        if should_capture {
            let ctx = context.lock().await;
            let ms = ctx.messages().to_vec();
            drop(ctx);
            if let Some(ref mut m) = *self
                .snapshot_manager
                .write()
                .unwrap_or_else(|e| e.into_inner())
            {
                m.capture(iteration, &ms);
            }
        }
    }

    // ── 工具批准（快照语义） ───────────────────────────

    /**
     * 某个工具在执行前是否需要人工批准。此方法镜像了流式路径的语义：它直接查询
     * `self.permission_service`，而不会刷新（flush）待处理的权限规则
     * （`run/approval.rs` 中非流式的 `ReactAgent::tool_needs_approval` 会执行刷新——
     * 这一差异是有意保留的，目的是使流式行为与重构前的实现保持逐字节一致）。
     */
    #[cfg(feature = "human-loop")]
    pub(crate) async fn tool_needs_approval(&self, tool_name: &str) -> bool {
        use crate::tools::permission::PermissionMode;
        if let Some(svc) = &self.permission_service {
            let mode = svc.mode().await;
            if matches!(
                mode,
                PermissionMode::BypassPermissions | PermissionMode::DontAsk | PermissionMode::Plan
            ) {
                return false;
            }
            let perms = self
                .tools
                .tool_manager
                .get_tool(tool_name)
                .map(|t| t.permissions())
                .unwrap_or_default();
            return svc
                .would_request_human_for_permissions(tool_name, &perms)
                .await;
        }
        false
    }

    /// `human-loop` 特性未启用时的桩实现 —— 永远不需要批准。
    #[cfg(not(feature = "human-loop"))]
    #[allow(dead_code)]
    pub(crate) async fn tool_needs_approval(&self, _: &str) -> bool {
        false
    }

    // ── 工具执行（完整流水线） ───────────────────────────────

    /**
     * 执行一次工具调用，走完整的策略流水线：
     * PreToolUse hook → 编辑前读取守卫 → 执行 → PostToolUse hook → 审计。
     *
     * 使用统一的 ToolExecutionPipeline（16 个阶段），以保证流式路径与非流式路径
     * 行为一致。
     */
    pub(crate) fn execute_tool_with_policy<'a>(
        &'a self,
        call_id: String,
        tool_name: &'a str,
        params: &'a crate::tools::ToolParameters,
        input: &'a serde_json::Value,
        stream_tx: Option<
            tokio::sync::mpsc::Sender<crate::agent::react::run::pipeline::ToolPipelineEvent>,
        >,
    ) -> futures::future::BoxFuture<'a, std::result::Result<ToolCallSuccess, ToolCallFailure>> {
        Box::pin(async move {
            // 使用统一的流水线以保证行为一致
            let pipeline = self
                .tool_execution_pipeline
                .as_ref()
                .cloned()
                .unwrap_or_else(|| {
                    std::sync::Arc::new(
                        crate::agent::react::run::pipeline::ToolExecutionPipeline::default_pipeline(
                        ),
                    )
                });

            let mut ctx = crate::agent::react::run::pipeline::ToolExecutionContext {
                call_id,
                requested_tool_name: tool_name.to_string(),
                requested_input: input.clone(),
                tool_name: tool_name.to_string(),
                params: params.clone(),
                input: input.clone(),
                hook_messages: crate::agent::react::run::context::HookMessageBatches::default(),
                result: None,
                output: None,
                audit_error_output: None,
                blocked: false,
                block_reason: None,
                block_failure: None,
                duration_ms: 0,
                plan_mode: self.config.plan_mode,
                permission_decision: None,
                permission_mode_override: None,
                rewrites: Vec::new(),
                invocation_emitted: false,
                callback_started: false,
                interrupted_execution_error: None,
                stream_tx,
            };

            let pipeline_result = pipeline.run(&mut ctx, self).await;
            if !ctx.hook_messages.pre.is_empty() || !ctx.hook_messages.post.is_empty() {
                let mut context = self.context.lock().await;
                for message in &ctx.hook_messages.pre {
                    context.push(crate::agent::react::run::context::runtime_context_note(
                        "Hook:PreToolUse",
                        message,
                    ));
                }
                for message in &ctx.hook_messages.post {
                    context.push(crate::agent::react::run::context::runtime_context_note(
                        "Hook:PostToolUse",
                        message,
                    ));
                }
            }

            match pipeline_result {
                Ok(()) => {
                    // 检查执行是否被阻塞
                    if ctx.blocked {
                        let reason = ctx
                            .block_reason
                            .unwrap_or_else(|| format!("Tool {} blocked", tool_name));
                        let failure = ctx.block_failure.unwrap_or_else(|| {
                            ToolFailure::new(crate::tools::ToolFailureCategory::Permanent)
                        });
                        let mut result = ctx.result.unwrap_or_else(|| {
                            ToolResult::failure(failure.category, reason.clone())
                                .with_failure(failure)
                        });
                        if let Some(output) = ctx.output {
                            result.output = output;
                        }
                        self.record_skill_telemetry(
                            tool_name,
                            ctx.duration_ms,
                            false,
                            Some(reason.as_str()),
                        );
                        return Err(ToolCallFailure {
                            name: ctx.tool_name.clone(),
                            error: crate::error::ToolError::ExecutionFailed {
                                tool: ctx.tool_name.clone(),
                                message: reason,
                            }
                            .into(),
                            result,
                        });
                    }

                    // 在守卫检查与输出预算处理之后，返回完整的结果。
                    if let Some(mut result) = ctx.result {
                        if let Some(output) = ctx.output {
                            result.output = output;
                        }
                        if result.success {
                            let telemetry_tool_name = ctx.tool_name.clone();
                            self.record_skill_telemetry(
                                &telemetry_tool_name,
                                ctx.duration_ms,
                                true,
                                None,
                            );
                            return Ok(ToolCallSuccess {
                                name: ctx.tool_name,
                                result,
                            });
                        }
                        let message = result
                            .error
                            .clone()
                            .unwrap_or_else(|| result.output.clone());
                        let failure = result.failure.clone().unwrap_or_else(|| {
                            ToolFailure::new(crate::tools::ToolFailureCategory::Permanent)
                        });
                        result.error = Some(message.clone());
                        result.failure = Some(failure);
                        let telemetry_tool_name = ctx.tool_name.clone();
                        self.record_skill_telemetry(
                            &telemetry_tool_name,
                            ctx.duration_ms,
                            false,
                            Some(message.as_str()),
                        );
                        return Err(ToolCallFailure {
                            name: ctx.tool_name.clone(),
                            error: crate::error::ToolError::ExecutionFailed {
                                tool: ctx.tool_name.clone(),
                                message,
                            }
                            .into(),
                            result,
                        });
                    }
                    let message = "Pipeline completed without result".to_string();
                    self.record_skill_telemetry(
                        tool_name,
                        ctx.duration_ms,
                        false,
                        Some(message.as_str()),
                    );
                    Err(ToolCallFailure {
                        name: ctx.tool_name,
                        error: crate::error::ReactError::Other(message.clone()),
                        result: ToolResult::failure(
                            crate::tools::ToolFailureCategory::Permanent,
                            message,
                        ),
                    })
                }
                Err(error) => {
                    let may_have_side_effects = self
                        .tools
                        .tool_manager
                        .get_tool(tool_name)
                        .is_none_or(|tool| {
                            tool.risk_level() != crate::tools::ToolRiskLevel::ReadOnly
                        });
                    let failure = ToolFailure::from_error(&error, may_have_side_effects);
                    let result = ToolResult::failure(failure.category, error.to_string())
                        .with_failure(failure);
                    let telemetry_tool_name = ctx.tool_name.clone();
                    let error_message = error.to_string();
                    self.record_skill_telemetry(
                        &telemetry_tool_name,
                        ctx.duration_ms,
                        false,
                        Some(error_message.as_str()),
                    );
                    Err(ToolCallFailure {
                        name: ctx.tool_name,
                        result,
                        error,
                    })
                }
            }
        })
    }
}

#[cfg(test)]
mod transcript_filter_tests {
    use super::{
        AgentRunSnapshot, ToolRuntime, TranscriptProjectionCursor, filter_user_visible_transcript,
    };
    use crate::compression::{ContextManager, ContextProjection};
    use crate::error::{ReactError, Result};
    use echo_core::llm::types::Message;
    use echo_core::tools::{Tool, ToolParameters, ToolResult};
    use std::collections::HashSet;
    use std::sync::Arc;

    struct NamedTool(&'static str);

    struct ReadOnlyNamedTool(&'static str);

    #[cfg(feature = "mcp")]
    struct McpPlanProbe {
        name: String,
        executions: Arc<std::sync::atomic::AtomicUsize>,
    }

    #[cfg(feature = "human-loop")]
    struct ApprovalTool;

    #[cfg(feature = "human-loop")]
    impl Tool for ApprovalTool {
        fn name(&self) -> &str {
            "approval_tool"
        }

        fn description(&self) -> &str {
            "tool that requires write approval"
        }

        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({"type": "object"})
        }

        fn permissions(&self) -> Vec<echo_core::tools::permission::ToolPermission> {
            vec![echo_core::tools::permission::ToolPermission::Write]
        }

        fn execute<'a>(
            &'a self,
            _parameters: ToolParameters,
        ) -> futures::future::BoxFuture<'a, echo_core::error::Result<ToolResult>> {
            Box::pin(async { Ok(ToolResult::success("ok")) })
        }
    }

    #[cfg(feature = "human-loop")]
    struct PendingApprovalProvider;

    #[cfg(feature = "human-loop")]
    impl crate::human_loop::HumanLoopProvider for PendingApprovalProvider {
        fn request(
            &self,
            _request: crate::human_loop::HumanLoopRequest,
        ) -> futures::future::BoxFuture<'_, Result<crate::human_loop::HumanLoopResponse>> {
            Box::pin(std::future::pending())
        }
    }

    impl Tool for NamedTool {
        fn name(&self) -> &str {
            self.0
        }

        fn description(&self) -> &str {
            "snapshot policy test tool"
        }

        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({"type": "object"})
        }

        fn execute<'a>(
            &'a self,
            _parameters: ToolParameters,
        ) -> futures::future::BoxFuture<'a, echo_core::error::Result<ToolResult>> {
            Box::pin(async { Ok(ToolResult::success("ok")) })
        }
    }

    impl Tool for ReadOnlyNamedTool {
        fn name(&self) -> &str {
            self.0
        }

        fn description(&self) -> &str {
            "locally classified read-only snapshot policy test tool"
        }

        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({"type": "object"})
        }

        fn risk_level(&self) -> echo_core::tools::ToolRiskLevel {
            echo_core::tools::ToolRiskLevel::ReadOnly
        }

        fn execute<'a>(
            &'a self,
            _parameters: ToolParameters,
        ) -> futures::future::BoxFuture<'a, echo_core::error::Result<ToolResult>> {
            Box::pin(async { Ok(ToolResult::success("ok")) })
        }
    }

    #[cfg(feature = "mcp")]
    impl Tool for McpPlanProbe {
        fn name(&self) -> &str {
            &self.name
        }

        fn description(&self) -> &str {
            "MCP plan-mode capability probe"
        }

        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({"type": "object"})
        }

        fn permissions(&self) -> Vec<echo_core::tools::permission::ToolPermission> {
            vec![echo_core::tools::permission::ToolPermission::Write]
        }

        fn risk_level(&self) -> echo_core::tools::ToolRiskLevel {
            echo_core::tools::ToolRiskLevel::Standard
        }

        fn execute<'a>(
            &'a self,
            _parameters: ToolParameters,
        ) -> futures::future::BoxFuture<'a, echo_core::error::Result<ToolResult>> {
            self.executions
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Box::pin(async { Ok(ToolResult::success("mutated")) })
        }
    }

    #[test]
    fn invocation_separates_product_conversation_from_runtime_state_identity() {
        let config = crate::agent::AgentConfig::new("test-model", "agent", "system")
            .conversation_id("runtime-incarnation");
        let agent = crate::agent::ReactAgent::new(config);
        let invocation = echo_core::agent::AgentInvocationContext {
            runtime_state_id: Some("runtime-incarnation".to_string()),
            runtime: Some(echo_core::tools::ExternalRunContext {
                conversation_id: Some("product-conversation".to_string()),
                run_id: None,
                turn_id: None,
                execution_id: None,
                isolation_id: None,
                message_id: None,
                cancel: None,
                trace_sink: None,
                delegation_policy: None,
                resource_guards: Vec::new(),
                subagent_lineage: None,
                uplink: None,
            }),
            ..Default::default()
        };

        let snapshot = AgentRunSnapshot::from_agent_with_invocation(&agent, &invocation);
        assert_eq!(
            snapshot.config.conversation_id.as_deref(),
            Some("product-conversation")
        );
        assert_eq!(
            snapshot.config.runtime_state_id.as_deref(),
            Some("runtime-incarnation")
        );
    }

    #[tokio::test]
    async fn transcript_generation_runtime_identity_rejects_checkpoint_write_before_store_mutation()
    -> Result<()> {
        use crate::state::RuntimeStateStore;

        let temp = tempfile::tempdir()?;
        let store = Arc::new(crate::state::FileRuntimeStateStore::new(temp.path())?);
        let config = crate::agent::AgentConfig::new("test-model", "agent", "system")
            .conversation_id("configured-state");
        let mut agent = crate::agent::ReactAgent::new(config);
        agent.set_state_store(store.clone());
        agent
            .memory
            .context
            .lock()
            .await
            .push(Message::user("must not persist".to_string()));
        let invocation = echo_core::agent::AgentInvocationContext {
            runtime_state_id: Some("runtime-a".to_string()),
            transcript_generation_id: Some("runtime-b".to_string()),
            runtime: Some(echo_core::tools::ExternalRunContext {
                conversation_id: Some("product-conversation".to_string()),
                ..Default::default()
            }),
            ..Default::default()
        };
        let snapshot = AgentRunSnapshot::from_agent_with_invocation(&agent, &invocation);

        let error = match snapshot
            .save_runtime_checkpoint(&agent.memory.context, None)
            .await
        {
            Ok(()) => {
                return Err(ReactError::Other(
                    "mismatched transcript generation unexpectedly saved".to_string(),
                ));
            }
            Err(error) => error,
        };
        assert!(matches!(error, ReactError::RuntimeState(_)));
        assert!(error.to_string().contains("transcript generation identity"));
        assert!(store.get_checkpoint("runtime-a").await?.is_none());
        assert!(
            store
                .runtime_state_ids("product-conversation")
                .await?
                .is_empty()
        );
        Ok(())
    }

    #[tokio::test]
    async fn new_projection_safe_point_resets_observation_provenance() -> Result<()> {
        let agent = crate::agent::ReactAgent::new(crate::agent::AgentConfig::new(
            "test-model",
            "agent",
            "system",
        ));
        let snapshot = AgentRunSnapshot::from_agent(&agent);
        snapshot.mark_transcript_settlement_observed();
        assert!(snapshot.transcript_settlement_was_observed());

        snapshot
            .save_transcript_projection(&agent.memory.context, None)
            .await?;

        assert!(!snapshot.transcript_settlement_was_observed());
        Ok(())
    }

    #[tokio::test]
    async fn checkpoint_save_indexes_product_scope_and_exact_reset_reclaims_runtime() -> Result<()>
    {
        use crate::state::RuntimeStateStore;

        let temp = tempfile::tempdir()?;
        let store = Arc::new(crate::state::FileRuntimeStateStore::new(temp.path())?);
        let config = crate::agent::AgentConfig::new("test-model", "agent", "system")
            .conversation_id("runtime-incarnation");
        let mut agent = crate::agent::ReactAgent::new(config);
        agent.set_state_store(store.clone());
        agent
            .memory
            .context
            .lock()
            .await
            .push(Message::user("indexed turn".to_string()));
        let invocation = echo_core::agent::AgentInvocationContext {
            runtime_state_id: Some("runtime-incarnation".to_string()),
            transcript_generation_id: Some("runtime-incarnation".to_string()),
            runtime: Some(echo_core::tools::ExternalRunContext {
                conversation_id: Some("product-conversation".to_string()),
                run_id: None,
                turn_id: None,
                execution_id: None,
                isolation_id: None,
                message_id: None,
                cancel: None,
                trace_sink: None,
                delegation_policy: None,
                resource_guards: Vec::new(),
                subagent_lineage: None,
                uplink: None,
            }),
            ..Default::default()
        };
        let snapshot = AgentRunSnapshot::from_agent_with_invocation(&agent, &invocation);
        snapshot
            .save_runtime_checkpoint(&agent.memory.context, None)
            .await?;

        assert_eq!(
            store.runtime_state_ids("product-conversation").await?,
            vec!["runtime-incarnation".to_string()]
        );
        assert!(store.get_checkpoint("runtime-incarnation").await?.is_some());
        let reset = store
            .clear_runtime_state("product-conversation", "runtime-incarnation")
            .await?;
        assert!(reset.checkpoint_removed);
        assert!(store.get_checkpoint("runtime-incarnation").await?.is_none());
        assert!(
            store
                .runtime_state_ids("product-conversation")
                .await?
                .is_empty()
        );
        Ok(())
    }

    #[tokio::test]
    async fn checkpoint_reads_live_canonical_skill_activation_after_snapshot() -> Result<()> {
        use crate::state::RuntimeStateStore;

        let temp = tempfile::tempdir()?;
        let store = Arc::new(crate::state::FileRuntimeStateStore::new(temp.path())?);
        let config = crate::agent::AgentConfig::new("test-model", "agent", "system")
            .conversation_id("skill-authority");
        let mut agent = crate::agent::ReactAgent::new(config);
        agent.set_state_store(store.clone());
        assert!(agent.tools.skill_registry.mark_activated("before-snapshot"));
        let snapshot = AgentRunSnapshot::from_agent(&agent);

        agent.tools.skill_registry.reset_activation_state();
        assert!(agent.tools.skill_registry.mark_activated("after-snapshot"));
        snapshot
            .save_runtime_checkpoint(&agent.memory.context, None)
            .await?;
        let checkpoint = store
            .get_checkpoint("skill-authority")
            .await?
            .ok_or_else(|| ReactError::Other("skill checkpoint missing".to_string()))?;
        assert_eq!(checkpoint.active_skills, vec!["after-snapshot".to_string()]);

        agent.tools.skill_registry.reset_activation_state();
        snapshot
            .save_runtime_checkpoint(&agent.memory.context, None)
            .await?;
        let reset_checkpoint = store
            .get_checkpoint("skill-authority")
            .await?
            .ok_or_else(|| ReactError::Other("reset checkpoint missing".to_string()))?;
        assert!(reset_checkpoint.active_skills.is_empty());
        Ok(())
    }

    #[test]
    fn invocation_conversation_remains_checkpoint_identity_without_explicit_override() {
        let agent = crate::agent::ReactAgent::new(crate::agent::AgentConfig::new(
            "test-model",
            "agent",
            "system",
        ));
        let invocation = echo_core::agent::AgentInvocationContext {
            runtime: Some(echo_core::tools::ExternalRunContext {
                conversation_id: Some("shared-identity".to_string()),
                run_id: None,
                turn_id: None,
                execution_id: None,
                isolation_id: None,
                message_id: None,
                cancel: None,
                trace_sink: None,
                delegation_policy: None,
                resource_guards: Vec::new(),
                subagent_lineage: None,
                uplink: None,
            }),
            ..Default::default()
        };

        let snapshot = AgentRunSnapshot::from_agent_with_invocation(&agent, &invocation);
        assert_eq!(
            snapshot.config.conversation_id,
            snapshot.config.runtime_state_id
        );
    }

    #[tokio::test]
    async fn runtime_cursor_is_stable_across_safe_points_and_compaction() -> Result<()> {
        let new = crate::memory::project_messages(
            "product-conversation",
            &[
                Message::user("again".to_string()),
                Message::assistant("same answer".to_string()),
            ],
        )?;
        let mut cursor = TranscriptProjectionCursor::default();
        let assigned = cursor.assign("new-runtime-incarnation", &new)?;
        cursor.projected = assigned.clone();
        let same_safe_point = cursor.assign("new-runtime-incarnation", &new)?;
        assert_eq!(same_safe_point, assigned);

        let checkpoint = cursor.checkpoint_for("new-runtime-incarnation");
        let mut restored_cursor = TranscriptProjectionCursor::default();
        restored_cursor.restore(checkpoint);
        let compacted = crate::memory::project_messages(
            "product-conversation",
            &[
                Message::user("again".to_string()),
                Message::assistant("same answer".to_string()),
                Message::user("after compact".to_string()),
            ],
        )?;
        let after_compact = restored_cursor.assign("new-runtime-incarnation", &compacted)?;
        assert_eq!(after_compact.len(), 3);
        assert_eq!(
            after_compact.first().map(|message| message.ordinal),
            Some(0)
        );
        assert_eq!(after_compact.get(1).map(|message| message.ordinal), Some(1));
        assert_eq!(after_compact.get(2).map(|message| message.ordinal), Some(2));

        let repeated = crate::memory::project_messages(
            "product-conversation",
            &[
                Message::user("A".to_string()),
                Message::assistant("B".to_string()),
                Message::user("A".to_string()),
                Message::assistant("B".to_string()),
            ],
        )?;
        let retained = crate::memory::project_messages(
            "product-conversation",
            &[
                Message::user("A".to_string()),
                Message::assistant("B".to_string()),
            ],
        )?;
        let mut repeated_cursor = TranscriptProjectionCursor::default();
        let repeated_projection = repeated_cursor.assign("repeat-generation", &repeated)?;
        repeated_cursor.projected = repeated_projection;
        let retained_projection = repeated_cursor.assign("repeat-generation", &retained)?;
        repeated_cursor.projected = retained_projection;
        let after_repeated_tail = repeated_cursor.assign("repeat-generation", &repeated)?;
        assert_eq!(
            after_repeated_tail
                .iter()
                .map(|message| message.ordinal)
                .collect::<Vec<_>>(),
            vec![2, 3, 4, 5]
        );

        let fresh_agent = crate::agent::ReactAgent::new(
            crate::agent::AgentConfig::new("test-model", "agent", "system")
                .conversation_id("new-runtime-incarnation"),
        );
        assert!(!fresh_agent.get_messages().await.iter().any(|message| {
            message
                .text_content()
                .is_some_and(|text| text.contains("old product turn"))
        }));
        Ok(())
    }

    #[cfg(feature = "human-loop")]
    #[tokio::test]
    async fn permission_wait_stops_when_invocation_is_cancelled() -> Result<()> {
        let permission_service = Arc::new(crate::human_loop::PermissionService::from_provider(
            Arc::new(PendingApprovalProvider),
        ));
        let agent = crate::agent::ReactAgentBuilder::new()
            .model("test-model")
            .permission_service(permission_service)
            .tool(Box::new(ApprovalTool))
            .build()?;
        let cancel = crate::agent::CancellationToken::new();
        let invocation = echo_core::agent::AgentInvocationContext {
            cancel: Some(cancel.clone()),
            ..Default::default()
        };
        let snapshot = AgentRunSnapshot::from_agent_with_invocation(&agent, &invocation);
        let input = serde_json::json!({});
        let approval = snapshot.check_tool_approval("call-1", "approval_tool", &input, None);
        tokio::pin!(approval);

        tokio::task::yield_now().await;
        cancel.cancel();
        let outcome = tokio::time::timeout(std::time::Duration::from_secs(1), approval)
            .await
            .map_err(|_| {
                crate::error::ReactError::Other("approval cancellation timed out".into())
            })?;
        let error = match outcome {
            Ok(_) => {
                return Err(crate::error::ReactError::Other(
                    "cancelled permission wait unexpectedly approved the tool".into(),
                ));
            }
            Err(error) => error,
        };
        assert!(error.to_string().contains("permission request"));
        Ok(())
    }

    fn tool_names(snapshot: &AgentRunSnapshot) -> Vec<String> {
        snapshot
            .tools
            .tools_for_llm()
            .into_iter()
            .map(|tool| tool.function.name)
            .collect()
    }

    #[test]
    fn transcript_filter_excludes_projection_owned_messages_only() {
        let mut context = ContextManager::builder(4096).build();
        context.push(Message::user(
            "ordinary text mentions provider-marker".to_string(),
        ));
        context.apply_projections(&[ContextProjection {
            marker: "provider-marker".to_string(),
            message: Some(Message::user("internal projected state".to_string())),
        }]);
        context.push(Message::user("ordinary visible message".to_string()));

        let visible = filter_user_visible_transcript(context.messages());
        let visible_text: Vec<String> = visible
            .iter()
            .filter_map(|message| message.content.as_text())
            .collect();

        assert_eq!(
            visible_text,
            vec![
                "ordinary text mentions provider-marker".to_string(),
                "ordinary visible message".to_string(),
            ]
        );
    }

    #[test]
    fn invocation_snapshot_derives_runtime_fields_as_one_value() -> Result<()> {
        let agent = crate::agent::ReactAgentBuilder::new()
            .model("test-model")
            .build()?;
        let cancel = std::sync::Arc::new(crate::agent::CancellationToken::new());
        let trace_sink: echo_core::tools::TraceSinkFn = std::sync::Arc::new(|_| {});
        let invocation = echo_core::agent::AgentInvocationContext {
            runtime_state_id: None,
            transcript_generation_id: None,
            runtime: Some(echo_core::tools::ExternalRunContext {
                conversation_id: None,
                run_id: Some("run-atomic".to_string()),
                turn_id: None,
                execution_id: Some("execution-atomic".to_string()),
                isolation_id: None,
                message_id: Some("message-atomic".to_string()),
                cancel: Some(std::sync::Arc::clone(&cancel)),
                trace_sink: Some(std::sync::Arc::clone(&trace_sink)),
                delegation_policy: Some(echo_core::tools::NestedDelegationPolicy {
                    can_spawn_subagents: true,
                    delegate_depth: 3,
                    max_delegate_depth: 4,
                }),
                resource_guards: vec![echo_core::tools::InvocationResourceGuard::new(
                    "runtime-guard".to_string(),
                )],
                subagent_lineage: None,
                uplink: None,
            }),
            working_dir: Some(std::path::PathBuf::from("/tmp/worktree-atomic")),
            cancel: None,
            disabled_tools: None,
            visible_tools: None,
            run_budget: None,
            history: None,
            resource_guards: vec![echo_core::tools::InvocationResourceGuard::new(
                "invocation-guard".to_string(),
            )],
            input_lifecycle: None,
        };

        let snapshot = AgentRunSnapshot::from_agent_with_invocation(&agent, &invocation);
        assert_eq!(snapshot.current_run_id.as_deref(), Some("run-atomic"));
        assert_eq!(
            snapshot.config.working_dir.as_deref(),
            Some(std::path::Path::new("/tmp/worktree-atomic"))
        );
        assert!(
            snapshot
                .external_cancel
                .as_ref()
                .is_some_and(|value| std::sync::Arc::ptr_eq(value, &cancel))
        );
        assert!(
            snapshot
                .external_trace_sink
                .as_ref()
                .is_some_and(|value| std::sync::Arc::ptr_eq(value, &trace_sink))
        );
        assert_eq!(
            snapshot.external_delegation_policy,
            Some(echo_core::tools::NestedDelegationPolicy {
                can_spawn_subagents: true,
                delegate_depth: 3,
                max_delegate_depth: 4,
            })
        );
        assert_eq!(snapshot.resource_guards.len(), 2);
        Ok(())
    }

    #[test]
    fn invocation_tool_exclusions_are_isolated_and_snapshot_immutable() -> Result<()> {
        let agent = crate::agent::ReactAgentBuilder::new()
            .model("test-model")
            .tool(Box::new(NamedTool("alpha")))
            .tool(Box::new(NamedTool("beta")))
            .tool(Box::new(NamedTool("gamma")))
            .tool(Box::new(NamedTool("delta")))
            .build()?;
        agent.set_disabled_tools(Some(HashSet::from(["gamma".to_string()])));

        let invocation_a = echo_core::agent::AgentInvocationContext {
            disabled_tools: Some(HashSet::from(["alpha".to_string()])),
            ..Default::default()
        };
        let invocation_b = echo_core::agent::AgentInvocationContext {
            disabled_tools: Some(HashSet::from(["beta".to_string()])),
            ..Default::default()
        };
        let snapshot_a = AgentRunSnapshot::from_agent_with_invocation(&agent, &invocation_a);
        let snapshot_b = AgentRunSnapshot::from_agent_with_invocation(&agent, &invocation_b);

        agent.set_disabled_tools(Some(HashSet::from(["delta".to_string()])));

        let tools_a: HashSet<String> = tool_names(&snapshot_a).into_iter().collect();
        let tools_b: HashSet<String> = tool_names(&snapshot_b).into_iter().collect();
        assert!(!tools_a.contains("alpha"));
        assert!(!tools_a.contains("gamma"));
        assert!(tools_a.contains("beta"));
        assert!(tools_a.contains("delta"));
        assert!(!tools_b.contains("beta"));
        assert!(!tools_b.contains("gamma"));
        assert!(tools_b.contains("alpha"));
        assert!(tools_b.contains("delta"));
        let snapshot_c = AgentRunSnapshot::from_agent(&agent);
        let tools_c: HashSet<String> = tool_names(&snapshot_c).into_iter().collect();
        assert!(tools_c.contains("alpha"));
        assert!(tools_c.contains("beta"));
        assert!(tools_c.contains("gamma"));
        assert!(!tools_c.contains("delta"));
        Ok(())
    }

    #[test]
    fn invocation_run_budget_overrides_agent_default_without_mutation() {
        let mut agent = crate::agent::ReactAgent::new(
            crate::agent::AgentConfig::new("model", "agent", "system").run_budget(
                echo_core::agent::RunBudgetPolicy {
                    iteration_wind_down_remaining: Some(2),
                    max_model_tokens: Some(1_000),
                },
            ),
        );
        let invocation = echo_core::agent::AgentInvocationContext {
            run_budget: Some(echo_core::agent::RunBudgetPolicy {
                iteration_wind_down_remaining: Some(1),
                max_model_tokens: Some(100),
            }),
            ..Default::default()
        };
        let snapshot = AgentRunSnapshot::from_agent_with_invocation(&agent, &invocation);
        agent.config_mut().run_budget.max_model_tokens = Some(5);

        assert_eq!(snapshot.config.run_budget.max_model_tokens, Some(100));
        assert_eq!(
            snapshot.config.run_budget.iteration_wind_down_remaining,
            Some(1)
        );
    }

    #[test]
    fn model_profile_exclusions_join_effective_tool_policy() -> Result<()> {
        let mut profile =
            echo_core::llm::capabilities::ModelProfile::from_provider_name("model", "openai");
        profile.excluded_tools.insert("shell".to_string());
        let agent = crate::agent::ReactAgentBuilder::new()
            .model("model")
            .model_profile(profile)
            .tool(Box::new(NamedTool("shell")))
            .tool(Box::new(NamedTool("read_file")))
            .build()?;

        let snapshot = AgentRunSnapshot::from_agent(&agent);
        let tools: HashSet<String> = tool_names(&snapshot).into_iter().collect();
        assert!(!tools.contains("shell"));
        assert!(tools.contains("read_file"));
        Ok(())
    }

    #[tokio::test]
    async fn model_profile_prompt_suffix_is_canonical_after_compression() -> Result<()> {
        let mut profile =
            echo_core::llm::capabilities::ModelProfile::from_provider_name("model", "openai");
        profile.prompt_suffix = Some("Use compact tool arguments.".to_string());
        let agent = crate::agent::ReactAgentBuilder::new()
            .model("model")
            .system_prompt("Base prompt")
            .model_profile(profile)
            .token_limit(64)
            .build()?;

        let mut context = agent.memory.context.lock().await;
        context.push(echo_core::llm::types::Message::user(
            "temporary context".to_string(),
        ));
        let _ = context.force_compress(1).await?;
        assert!(context.messages().iter().any(|message| {
            message.text_content().is_some_and(|text| {
                text.contains("Base prompt") && text.contains("Use compact tool arguments.")
            })
        }));
        Ok(())
    }

    #[test]
    fn tool_visibility_combines_skill_plan_and_disabled_policies() {
        let manager = Arc::new(crate::tools::ToolManager::new());
        manager.register(Box::new(ReadOnlyNamedTool("read_file")));
        for name in [
            "write_file",
            "shell",
            "final_answer",
            "custom",
            "mcp__malicious__write",
        ] {
            manager.register(Box::new(NamedTool(name)));
        }
        let runtime = ToolRuntime {
            tool_manager: manager,
            hook_registry: Arc::new(tokio::sync::RwLock::new(Default::default())),
            intervention_callbacks: Vec::new(),
            skill_allowed_tools: Some(HashSet::from([
                "read_file".to_string(),
                "write_file".to_string(),
                "shell".to_string(),
                "final_answer".to_string(),
                "mcp__malicious__write".to_string(),
            ])),
            disabled_tools: HashSet::from(["final_answer".to_string()]),
            visibility: None,
            plan_mode: true,
            #[cfg(feature = "human-loop")]
            permission_service: None,
        };

        let visible: Vec<String> = runtime
            .tools_for_llm()
            .into_iter()
            .map(|tool| tool.function.name)
            .collect();

        assert_eq!(visible, vec!["read_file"]);
    }

    #[cfg(feature = "mcp")]
    #[tokio::test]
    async fn permission_plan_hides_and_blocks_locally_mutating_mcp_tool() -> Result<()> {
        let tool_name = crate::mcp::McpToolAdapter::exposed_name_for("malicious", "write");
        let executions = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut agent = crate::agent::ReactAgent::new(
            crate::agent::AgentConfig::new("test-model", "agent", "system")
                .permission_mode(echo_core::tools::permission::PermissionMode::Plan),
        );
        agent.add_tool(Box::new(McpPlanProbe {
            name: tool_name.clone(),
            executions: Arc::clone(&executions),
        }));

        let snapshot = AgentRunSnapshot::from_agent(&agent);
        assert!(snapshot.config.plan_mode);
        assert!(!tool_names(&snapshot).contains(&tool_name));

        let input = serde_json::json!({});
        let result = snapshot
            .execute_tool_with_policy(
                "call-mcp-plan".to_string(),
                &tool_name,
                &ToolParameters::new(),
                &input,
                None,
            )
            .await;
        let Err(failure) = result else {
            return Err(echo_core::error::ReactError::Other(
                "mutating MCP tool executed in permission plan mode".to_string(),
            ));
        };
        assert!(
            failure
                .result
                .error
                .as_deref()
                .is_some_and(|reason| reason.contains("Plan mode"))
        );
        assert_eq!(executions.load(std::sync::atomic::Ordering::SeqCst), 0);
        Ok(())
    }

    #[cfg(all(feature = "mcp", feature = "human-loop"))]
    #[tokio::test]
    async fn live_permission_plan_precedes_hook_allow_for_mutating_mcp_tool() -> Result<()> {
        use crate::skills::hooks::{HookAction, HookEvent, HookRule, HooksDefinition};

        let tool_name = crate::mcp::McpToolAdapter::exposed_name_for("malicious", "write");
        let executions = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let service = Arc::new(
            crate::human_loop::PermissionService::new()
                .with_mode(echo_core::tools::permission::PermissionMode::Default),
        );
        let agent = crate::agent::ReactAgentBuilder::new()
            .model("test-model")
            .permission_service(Arc::clone(&service))
            .tool(Box::new(McpPlanProbe {
                name: tool_name.clone(),
                executions: Arc::clone(&executions),
            }))
            .build()?;
        let mut hooks = HooksDefinition::default();
        hooks.add_rules(
            HookEvent::PermissionRequest,
            vec![HookRule {
                matcher: "*".to_string(),
                hooks: vec![HookAction::Permission {
                    decision: "allow".to_string(),
                    reason: None,
                    suggestions: Vec::new(),
                }],
            }],
        );
        agent
            .hook_registry()
            .write()
            .await
            .register_user_hooks(hooks);

        let snapshot = AgentRunSnapshot::from_agent(&agent);
        assert!(!snapshot.config.plan_mode);
        service
            .set_mode(echo_core::tools::permission::PermissionMode::Plan)
            .await;
        assert!(snapshot.tools.is_plan_mode());
        assert!(!tool_names(&snapshot).contains(&tool_name));

        let input = serde_json::json!({});
        let result = snapshot
            .execute_tool_with_policy(
                "call-mcp-live-plan".to_string(),
                &tool_name,
                &ToolParameters::new(),
                &input,
                None,
            )
            .await;
        let Err(failure) = result else {
            return Err(echo_core::error::ReactError::Other(
                "hook allow bypassed live permission Plan mode".to_string(),
            ));
        };
        assert!(
            failure
                .result
                .error
                .as_deref()
                .is_some_and(|reason| reason.contains("Plan mode"))
        );
        assert_eq!(executions.load(std::sync::atomic::Ordering::SeqCst), 0);
        Ok(())
    }

    #[test]
    fn invocation_visibility_expands_without_mutating_registry() -> Result<()> {
        let mut agent = crate::agent::ReactAgentBuilder::new()
            .model("test-model")
            .build()?;
        agent.add_tool(Box::new(NamedTool("custom")));
        let invocation = echo_core::agent::AgentInvocationContext {
            visible_tools: Some(HashSet::from([
                "final_answer".to_string(),
                "tool_search".to_string(),
            ])),
            ..Default::default()
        };
        let snapshot = AgentRunSnapshot::from_agent_with_invocation(&agent, &invocation);
        assert!(!tool_names(&snapshot).contains(&"custom".to_string()));

        let activated = snapshot
            .tools
            .visibility
            .as_ref()
            .map(|visibility| visibility.activate(["custom".to_string()]))
            .unwrap_or_default();
        assert_eq!(activated, vec!["custom"]);
        assert!(tool_names(&snapshot).contains(&"custom".to_string()));
        assert!(agent.tool_names().contains(&"custom".to_string()));
        Ok(())
    }

    #[test]
    fn tool_search_is_hidden_when_deferred_visibility_is_disabled() -> Result<()> {
        let agent = crate::agent::ReactAgentBuilder::new()
            .model("test-model")
            .build()?;
        let snapshot = AgentRunSnapshot::from_agent(&agent);

        assert!(!tool_names(&snapshot).contains(&"tool_search".to_string()));
        assert!(agent.tool_names().contains(&"tool_search".to_string()));
        Ok(())
    }
}
