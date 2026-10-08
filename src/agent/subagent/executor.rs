/*! 子代理执行器 —— Sync / Fork / Teammate 三种模式的统一调度引擎。

执行器接收一个 [`DispatchRequest`],并根据定义中的 [`ExecutionMode`] 将其
路由到对应的执行策略。
*/

use crate::error::{AgentError, ReactError, Result};
use echo_core::agent::{
    Agent, AgentEvent, AgentInvocationContext, CancellationToken, EventIdentity,
    ExecutionAdmission, KeyedExecutionLease,
};
use echo_core::error::AgentTerminalKind;
use echo_core::llm::types::Message;
use futures::StreamExt;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::Semaphore;
use tracing::{info, warn};

use super::context::SubagentContext;
use super::control::{
    SubagentAttemptBinding, SubagentAttemptIdentity, SubagentControlError, SubagentControlRegistry,
    SubagentGuidanceQueueReceipt, SubagentInterruptOutcome, SubagentInterruptRequestDisposition,
    SubagentMessageReceipt,
};
use super::events::{SubagentEvent, SubagentEventPublisher, SubagentInvocationIdentity};
use super::hooks::{SubagentHookContext, SubagentHookRegistry};
use super::prompt::{
    CompiledSubagentInvocation, ContextTransferPolicy, SubagentInvocation, SubagentPromptCompiler,
    SubagentTaskContext, ToolCapabilitySnapshot,
};
use super::registry::SubagentRegistry;
use super::types::{
    ExecutionMode, ObservedIsolation, SubagentArtifact, SubagentEvidence, SubagentEvidenceSource,
    SubagentOutcome, SubagentResult, SubagentStatus,
};
use crate::tasks::{
    NestedDelegationPolicy, RuntimeAttemptControlCleanupReceipt,
    RuntimeTaskAttemptInterruptDisposition, RuntimeTaskAttemptInterruptProjectionError, Task,
    TaskClaim, TaskSubagentContext,
};

// ── 派发请求 ──────────────────────────────────────────────────────────

/// 一次向子代理派发任务的请求。
#[derive(Clone)]
pub struct DispatchRequest {
    /// 目标子代理名称。
    pub agent_name: String,
    /// 任务描述。
    pub task: String,
    /// 覆盖定义中默认的执行模式。
    pub mode_override: Option<ExecutionMode>,
    /// 取消令牌（从父级传播而来）。
    pub cancel: CancellationToken,
    /// 父代理名称（用于日志与拓扑关系）。
    pub parent_agent: String,
    /// 用于继承的父级上下文（Fork 模式使用）。
    pub parent_context: Option<SubagentContext>,
    /// 嵌套委派策略（防止无限制的委派链）。
    pub delegation_policy: NestedDelegationPolicy,
    /**
    应用层 run 级上下文（跨 spawn 安全，值传递）。

    dispatch_fork 将它放入本次 `AgentInvocationContext`，使 subagent 内的工具
    能经 ToolContext 读到 run_id/cancel/trace_sink——不修改共享 agent。
    */
    pub runtime_context: Option<echo_core::tools::ExternalRunContext>,
    /**
    可选的多模态消息（图片/文件）。当为 `Some` 时，子代理通过
    `execute_stream_message_with_invocation_context` 而非纯文本 `task`
    路径进行派发，从而能看到用户上传的附件。`None` 表示纯文本派发
    （所有现有调用方的默认行为）。
    */
    pub message: Option<Message>,
    /// 仅由注入的产品编译器消费的不透明结构化负载。
    pub prompt_payload: Option<serde_json::Value>,
    /// 供每个提示词编译器使用的、有类型的任务事实信息。
    pub prompt_context: Option<SubagentTaskContext>,
    /**
    派发方提供的显式任务约束/边界（例如 `agent_tool` 的 `constraints`
    参数）。由调用编译器独立于 `parent_context` 渲染，因此即便是全新
    上下文的派发也能携带边界约束。
    */
    pub constraints: Vec<String>,
    /**
    为 `true` 时表示本次派发是（或将会）通过
    [`SubagentExecutor::dispatch_background`] 启动的。该标记会传播到
    [`SubagentEvent::DispatchStarted`]，供 UI 标记后台卡片。
    */
    pub background: bool,
}

impl std::fmt::Debug for DispatchRequest {
    /// 输出 DispatchRequest 的调试表示。
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DispatchRequest")
            .field("agent_name", &self.agent_name)
            .field("task", &self.task)
            .field("mode_override", &self.mode_override)
            .field("parent_agent", &self.parent_agent)
            .field("delegation_policy", &self.delegation_policy)
            .field(
                "runtime_context",
                &self.runtime_context.as_ref().map(|c| &c.run_id),
            )
            .field("has_prompt_payload", &self.prompt_payload.is_some())
            .field("has_prompt_context", &self.prompt_context.is_some())
            .finish()
    }
}

impl DispatchRequest {
    /// 从旧版公共辅助方法所使用的 depth 参数构造出一个委派策略。
    pub fn policy_from_depth(depth: u32) -> NestedDelegationPolicy {
        NestedDelegationPolicy {
            can_spawn_subagents: true,
            delegate_depth: depth.min(u8::MAX as u32) as u8,
            max_delegate_depth: 3,
        }
    }
}

/**
根据允许工具白名单计算出本次调用需要禁用的工具集合；
若 `allowed_tools` 为空或未提供，则返回 `None`（不做限制）。
*/
fn invocation_disabled_tools(
    tool_names: Vec<String>,
    allowed_tools: Option<&[String]>,
) -> Option<HashSet<String>> {
    let allowed_tools = allowed_tools.filter(|tools| !tools.is_empty())?;
    let allowed = allowed_tools
        .iter()
        .map(String::as_str)
        .collect::<HashSet<_>>();
    Some(
        tool_names
            .into_iter()
            .filter(|tool| !allowed.contains(tool.as_str()))
            .collect(),
    )
}

/// 计算某个代理在应用了可选工具白名单之后的实际工具能力快照。
fn effective_capabilities(
    agent: &dyn Agent,
    allowed_tools: Option<&[String]>,
) -> ToolCapabilitySnapshot {
    let definitions = agent.tool_definitions();
    let mut disabled = agent.disabled_tool_names();
    if let Some(allowed) = allowed_tools.filter(|tools| !tools.is_empty()) {
        let allowed = allowed.iter().map(String::as_str).collect::<HashSet<_>>();
        disabled.extend(
            definitions
                .iter()
                .map(|definition| definition.function.name.clone())
                .filter(|name| !allowed.contains(name.as_str())),
        );
    }
    ToolCapabilitySnapshot::from_definitions(&definitions, &disabled)
}

/// 将框架错误映射为运行时所持有的子代理终态状态。
pub fn subagent_status_from_error(error: &ReactError) -> SubagentStatus {
    match error {
        ReactError::Agent(agent_error) => match agent_error.as_ref() {
            AgentError::Timeout(_) => SubagentStatus::TimedOut,
            AgentError::Interrupted | AgentError::Cancelled(_) => SubagentStatus::Cancelled,
            _ => SubagentStatus::Failed,
        },
        _ => SubagentStatus::Failed,
    }
}

#[cfg(test)]
mod typed_error_mapping_tests {
    use super::*;

    /**
    即使错误文本中偶然包含 "cancelled"/"timeout" 等字样，只要错误的实际
    类型不是取消/超时，映射结果也不应被误导为对应状态。
    */
    #[test]
    fn terminal_status_ignores_misleading_error_text() {
        let error = ReactError::Other("cancelled timeout are ordinary words here".to_string());
        assert_eq!(subagent_status_from_error(&error), SubagentStatus::Failed);
    }

    /**
    验证控制层错误在投影为运行时中断错误类型时，容量超限与身份冲突这两类
    错误的字段信息被完整保留。
    */
    #[test]
    fn attempt_control_projection_preserves_capacity_and_identity_conflict() {
        assert_eq!(
            runtime_interrupt_projection_error(SubagentControlError::PendingCapacityExceeded {
                limit: 17
            }),
            RuntimeTaskAttemptInterruptProjectionError::PendingCapacityExceeded { limit: 17 }
        );
        assert_eq!(
            runtime_interrupt_projection_error(SubagentControlError::IdentityConflict {
                task_id: "task".to_string(),
                attempt: 2,
                expected_execution_id: "expected".to_string(),
                actual_execution_id: "actual".to_string(),
            }),
            RuntimeTaskAttemptInterruptProjectionError::IdentityConflict {
                task_id: "task".to_string(),
                attempt: 2,
                expected_execution_id: "expected".to_string(),
                actual_execution_id: "actual".to_string(),
            }
        );
    }

    /**
    验证 `bind_exact_dispatch_context` 会保留取消令牌、trace_sink、uplink、
    资源守卫等运行时资源，并在遇到不一致的身份字段（如 run_id 冲突）时
    拒绝绑定。
    */
    #[test]
    fn exact_dispatch_context_preserves_resources_and_rejects_identity_conflicts()
    -> std::result::Result<(), String> {
        let trace_sink: echo_core::tools::TraceSinkFn = Arc::new(|_| {});
        let uplink: echo_core::tools::SubagentUplinkFn = Arc::new(|_| {
            Box::pin(async {
                echo_core::tools::SubagentUplinkReceipt {
                    accepted: true,
                    status: "test".to_string(),
                    detail: String::new(),
                }
            })
        });
        let initial_cancel = CancellationToken::new();
        let exact_cancel = CancellationToken::new();
        let exact = ExactTaskDispatchContext {
            run_id: "run".to_string(),
            execution_id: "execution".to_string(),
            task_id: "task".to_string(),
            attempt: 3,
            plan_revision: 9,
            cancel: exact_cancel.clone(),
            delegation_policy: NestedDelegationPolicy {
                can_spawn_subagents: true,
                delegate_depth: 1,
                max_delegate_depth: 4,
            },
        };
        let mut request = DispatchRequest {
            agent_name: "reviewer".to_string(),
            task: "review".to_string(),
            mode_override: None,
            cancel: initial_cancel,
            parent_agent: "parent".to_string(),
            parent_context: None,
            delegation_policy: NestedDelegationPolicy::default(),
            runtime_context: Some(echo_core::tools::ExternalRunContext {
                conversation_id: Some("conversation".to_string()),
                turn_id: Some("turn".to_string()),
                isolation_id: Some("isolation".to_string()),
                message_id: Some("message".to_string()),
                trace_sink: Some(trace_sink.clone()),
                resource_guards: vec![echo_core::tools::InvocationResourceGuard::new_identified(
                    "guard".to_string(),
                    7_u64,
                )],
                subagent_lineage: Some(echo_core::tools::SubagentLineage {
                    agent_name: Some("reviewer".to_string()),
                    parent_agent: Some("parent".to_string()),
                    ..Default::default()
                }),
                uplink: Some(uplink.clone()),
                ..Default::default()
            }),
            message: None,
            prompt_payload: None,
            prompt_context: None,
            constraints: Vec::new(),
            background: false,
        };

        bind_exact_dispatch_context(&mut request, &exact).map_err(|error| error.to_string())?;
        let runtime = request
            .runtime_context
            .as_ref()
            .ok_or_else(|| "runtime context missing".to_string())?;
        assert_eq!(runtime.conversation_id.as_deref(), Some("conversation"));
        assert_eq!(runtime.turn_id.as_deref(), Some("turn"));
        assert_eq!(runtime.isolation_id.as_deref(), Some("isolation"));
        assert_eq!(runtime.message_id.as_deref(), Some("message"));
        assert_eq!(runtime.resource_guards.len(), 1);
        assert!(
            runtime
                .trace_sink
                .as_ref()
                .is_some_and(|value| { Arc::ptr_eq(value, &trace_sink) })
        );
        assert!(
            runtime
                .uplink
                .as_ref()
                .is_some_and(|value| { Arc::ptr_eq(value, &uplink) })
        );
        let lineage = runtime
            .subagent_lineage
            .as_ref()
            .ok_or_else(|| "subagent lineage missing".to_string())?;
        assert_eq!(lineage.agent_name.as_deref(), Some("reviewer"));
        assert_eq!(lineage.run_id.as_deref(), Some("run"));
        assert_eq!(lineage.task_id.as_deref(), Some("task"));
        assert_eq!(lineage.execution_id.as_deref(), Some("execution"));
        assert_eq!(lineage.attempt, Some(3));
        assert_eq!(lineage.plan_revision, Some(9));
        exact_cancel.cancel();
        assert!(request.cancel.is_cancelled());
        assert!(
            runtime
                .cancel
                .as_ref()
                .is_some_and(|cancel| cancel.is_cancelled())
        );
        assert_eq!(request.delegation_policy, exact.delegation_policy);

        let mut conflicting = request.clone();
        if let Some(runtime) = conflicting.runtime_context.as_mut() {
            runtime.run_id = Some("other-run".to_string());
        }
        assert!(
            bind_exact_dispatch_context(&mut conflicting, &exact)
                .is_err_and(|error| error.to_string().contains("run_id mismatch"))
        );
        Ok(())
    }
}

/// 将子代理运行时状态映射为统一钩子系统所使用的 SubagentStop 状态。
fn hook_stop_status(status: SubagentStatus) -> echo_core::hooks::SubagentStopStatus {
    match status {
        SubagentStatus::Running => echo_core::hooks::SubagentStopStatus::Failed,
        SubagentStatus::Completed => echo_core::hooks::SubagentStopStatus::Completed,
        SubagentStatus::Failed => echo_core::hooks::SubagentStopStatus::Failed,
        SubagentStatus::Cancelled => echo_core::hooks::SubagentStopStatus::Cancelled,
        SubagentStatus::TimedOut => echo_core::hooks::SubagentStopStatus::TimedOut,
    }
}

/// 为钩子上下文附加 run_id/execution_id/attempt 关联信息，便于跨系统追踪。
fn correlate_subagent_hook(
    ctx: crate::skills::hooks::HookContext,
    run_id: Option<&str>,
    execution_id: Option<&str>,
    attempt: u32,
) -> crate::skills::hooks::HookContext {
    ctx.with_run_correlation(run_id, None, execution_id, Some(attempt))
}

/// 将文本截断为最多 500 个字符，用于生成长度受限的证据详情。
fn bounded_detail(text: &str) -> String {
    text.chars().take(500).collect()
}

/** 将运行时观测到的证据合并进已解析的结果中。

具有相同 kind、subject 与 attributes 的旧证据会被观测到的新事实替换，
而不同的调用与产出物则会被保留下来。
*/
pub fn merge_observed_evidence(
    outcome: &mut SubagentOutcome,
    evidence: Vec<SubagentEvidence>,
    artifacts: Vec<SubagentArtifact>,
) {
    for observed in evidence {
        outcome.evidence.retain(|existing| {
            existing.kind != observed.kind
                || existing.subject != observed.subject
                || existing.attributes != observed.attributes
        });
        outcome.evidence.push(observed);
    }
    for artifact in artifacts {
        outcome
            .artifacts
            .retain(|existing| existing.path != artifact.path);
        outcome.artifacts.push(artifact);
    }
    super::types::normalize_outcome(outcome);
    outcome.refresh_derived_views();
}

// ── Teammate 句柄 ───────────────────────────────────────────────────────────

/** 指向一个正在运行的 Teammate 代理的句柄。

用于轮询结果或取消其执行。
*/
#[derive(Debug)]
pub struct TeammateHandle {
    /// 唯一的句柄 ID。
    pub id: String,
    /// 该 Teammate 代理的名称。
    pub agent_name: String,
    /// 该 Teammate 的取消令牌。
    pub cancel: CancellationToken,
    /// 所派生任务的 Join 句柄。
    join_handle: tokio::task::JoinHandle<Result<SubagentResult>>,
}

/** [`SubagentExecutor::dispatch_background`] 立即返回的句柄。

子代理会继续在被派生的任务上执行；生命周期事件（`DispatchStarted` /
`DispatchCompleted` / `DispatchFailed`）仍会在注册表的事件总线上触发。
该句柄同时持有取消能力与最终结果，因此直接调用 API 的调用方不必依赖
可能丢失事件的事件流。
*/
#[derive(Debug, Clone)]
pub struct BackgroundSubagentHandle {
    /// 稳定的 execution id（与 `DispatchStarted.execution_id` 一致）。
    pub execution_id: String,
    /// 目标子代理名称。
    pub agent_name: String,
    /// 用于取消本次后台派发的令牌。
    cancel: CancellationToken,
    /// 派生任务的 Join 句柄；结果只能被消费一次。
    join_handle: Arc<Mutex<Option<tokio::task::JoinHandle<Result<SubagentResult>>>>>,
}

/** 针对由 TaskClaim 派生出的子代理执行尝试的、限定作用域的实时控制句柄。

持久化的 claim 校验与 Task 终态落定仍由 [`crate::tasks::RuntimeTaskService`]
拥有。此句柄只是将控制器已校验通过的 claim 投影到派发所使用的同一个
进程内注册表中。
*/
pub struct SubagentAttemptControlHandle {
    /// 底层执行器，用于实际发起派发、预留与中断等操作。
    executor: SubagentExecutor,
    /// 本句柄所限定的控制作用域标识。
    control_scope_id: String,
}

impl Clone for SubagentAttemptControlHandle {
    /// 克隆句柄：底层执行器以可跨 spawn 安全共享的方式克隆，控制作用域 id 保持不变。
    fn clone(&self) -> Self {
        Self {
            executor: self.executor.clone_for_spawn(),
            control_scope_id: self.control_scope_id.clone(),
        }
    }
}

impl std::fmt::Debug for SubagentAttemptControlHandle {
    /// 仅输出 control_scope_id，避免打印内部执行器细节。
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SubagentAttemptControlHandle")
            .field("control_scope_id", &self.control_scope_id)
            .finish_non_exhaustive()
    }
}

impl BackgroundSubagentHandle {
    /// 请求取消该后台派发。
    pub fn cancel(&self) {
        self.cancel.cancel();
    }

    /// 检查被派生的派发是否已到达终态。
    pub fn is_finished(&self) -> bool {
        self.join_handle
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .as_ref()
            .is_none_or(tokio::task::JoinHandle::is_finished)
    }

    /** 等待并消费后台结果。

    被克隆出的多个句柄共享同一个结果槽位；只有第一次调用 `join` 会真正
    消费该结果，之后的调用会返回一个明确的错误。
    */
    pub async fn join(&self) -> Result<SubagentResult> {
        let handle = self
            .join_handle
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take()
            .ok_or_else(|| ReactError::Other("Background result already consumed".to_string()))?;
        handle
            .await
            .map_err(|error| ReactError::Other(format!("Background join error: {error}")))?
    }
}

impl TeammateHandle {
    /// 检查该 Teammate 是否已完成。
    pub fn is_finished(&self) -> bool {
        self.join_handle.is_finished()
    }

    /// 等待该 Teammate 的结果。
    pub async fn join(self) -> Result<SubagentResult> {
        self.join_handle
            .await
            .map_err(|e| ReactError::Other(format!("Teammate join error: {}", e)))?
    }
}

// ── 执行器配置 ───────────────────────────────────────────────────────────

/// 子代理执行器的配置。
pub struct SubagentExecutorConfig {
    /// Fork 模式下允许的最大并发派发数。
    pub max_concurrent_forks: usize,
    /** 所有派发模式（Sync/Fork/Teammate）共用的默认超时时间（秒）。

    0 表示不超时。来源为 `AgentConfig.subagent_timeout_secs`（默认为
    600，即 10 分钟）。若某个子代理定义了 `SubagentDefinition.timeout_secs`
    （且 >0），则以该值覆盖此默认值。
    */
    pub default_timeout_secs: u64,
    /// 是否启用钩子。
    pub enable_hooks: bool,
    /** 可选的、桥接到统一生命周期钩子系统（echo-core）的执行器。

    设置后，SubagentStart/SubagentStop 事件会与基于 trait 的 SubagentHooks
    一起被投递进统一的 HookRegistry。
    */
    pub unified_hook_executor: Option<crate::skills::hooks::UnifiedHookExecutorFn>,
    /// 可选的、由应用层提供的产品自定义隔离类型的提供者。
    pub isolation_provider: Option<super::isolation::SharedIsolationProvider>,
    /// 由直接的 `agent_tool` 调用与程序化委派共用的提示词编译器。
    pub prompt_compiler: Arc<dyn SubagentPromptCompiler>,
    /**
    可选的共享进程级准入控制。缺省时 Fork 使用其自身的
    `max_concurrent_forks` 信号量以保持独立可用。
    */
    pub shared_admission: Option<Arc<ExecutionAdmission>>,
}

impl Default for SubagentExecutorConfig {
    /**
    构造带有合理默认值的执行器配置（Fork 并发上限 5，默认超时 600 秒，
    启用钩子，使用默认提示词编译器，不带隔离提供者/共享准入）。
    */
    fn default() -> Self {
        Self {
            max_concurrent_forks: 5,
            default_timeout_secs: 600,
            enable_hooks: true,
            unified_hook_executor: None,
            isolation_provider: None,
            prompt_compiler: Arc::new(super::prompt::DefaultSubagentPromptCompiler),
            shared_admission: None,
        }
    }
}

// ── 子 Agent 执行器 ─────────────────────────────────────────────────────────

/// 面向所有子代理执行模式的统一调度引擎。
pub struct SubagentExecutor {
    /// 子代理定义与实例注册表。
    registry: Arc<SubagentRegistry>,
    /// 派发生命周期钩子注册表。
    hooks: Arc<SubagentHookRegistry>,
    /// 执行器配置。
    config: SubagentExecutorConfig,
    /// Fork 模式在没有共享准入时使用的本地并发信号量。
    semaphore: Arc<Semaphore>,
    /// 实时控制注册表（追踪进行中的尝试、guidance、interrupt 等）。
    control_registry: Arc<SubagentControlRegistry>,
    /// 可被外部注入的共享进程级准入控制。
    shared_admission: Arc<tokio::sync::RwLock<Option<Arc<ExecutionAdmission>>>>,
    /// 已保留的 Team 运行时句柄索引，按业务 run 分组、限量保留。
    team_runtimes: Arc<Mutex<TeamRuntimeIndex>>,
}

/// Team 运行时索引最多保留的非活跃句柄数量。
const TEAM_RUNTIME_RETENTION: usize = 64;

/**
Team 运行时句柄的进程内索引：既能按 handle_id 查找，也维持一个用于
淘汰非活跃句柄的插入顺序队列。
*/
#[derive(Default)]
struct TeamRuntimeIndex {
    /// 按 handle_id 索引的运行时句柄。
    handles: HashMap<String, Arc<super::team::TeamRuntimeHandle>>,
    /// 插入顺序队列，用于按先入先出的顺序淘汰非活跃句柄。
    order: VecDeque<String>,
}

/// 淘汰超出 [`TEAM_RUNTIME_RETENTION`] 上限的非活跃 Team 运行时句柄。
fn prune_team_runtime_index(index: &Arc<Mutex<TeamRuntimeIndex>>) {
    let mut runtimes = index
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let mut excess = runtimes
        .handles
        .values()
        .filter(|handle| !handle.is_active())
        .count()
        .saturating_sub(TEAM_RUNTIME_RETENTION);
    let scan_count = runtimes.order.len();
    for _ in 0..scan_count {
        let Some(handle_id) = runtimes.order.pop_front() else {
            break;
        };
        let remove = excess > 0
            && runtimes
                .handles
                .get(&handle_id)
                .is_some_and(|handle| !handle.is_active());
        if remove {
            runtimes.handles.remove(&handle_id);
            excess = excess.saturating_sub(1);
        } else if runtimes.handles.contains_key(&handle_id) {
            runtimes.order.push_back(handle_id);
        }
    }
}

/// 作用域守卫：其 `Drop` 会触发一次 Team 运行时索引的淘汰整理。
struct TeamRuntimeRetentionGuard {
    /// 待整理的 Team 运行时索引。
    index: Arc<Mutex<TeamRuntimeIndex>>,
}

impl Drop for TeamRuntimeRetentionGuard {
    /// 守卫离开作用域时触发一次索引淘汰整理。
    fn drop(&mut self) {
        prune_team_runtime_index(&self.index);
    }
}

/** 框架默认的 uplink（上行消息）投递器。

各投递目标的行为：
- `Parent`：当发送方的血缘信息带有 `parent_execution_id`，且该尝试在共享
  控制注册表中仍处于存活状态时，消息会引导父级的活跃回合（仅入队，父级
  会在其下一次模型边界处取出消费）。否则消息仍会被接受，并以
  [`SubagentEvent::UplinkReceived`] 的形式呈现给宿主端监听者。
- `Sibling`：通过共享控制注册表仅入队投递；若目标兄弟节点当前并非活跃
  状态，则报告 `sibling_not_active`，以便模型据此重试或升级处理，而不是
  误以为已投递成功。
*/
pub fn default_uplink_sink(registry: Arc<SubagentRegistry>) -> echo_core::tools::SubagentUplinkFn {
    use echo_core::tools::{SubagentUplinkMessage, SubagentUplinkReceipt, SubagentUplinkTarget};
    Arc::new(move |message: SubagentUplinkMessage| {
        let registry = Arc::clone(&registry);
        Box::pin(async move {
            let from = &message.from;
            let agent = from
                .agent_name
                .clone()
                .unwrap_or_else(|| "<unknown>".to_string());
            let parent = from
                .parent_agent
                .clone()
                .unwrap_or_else(|| "<unknown>".to_string());
            let (direction, intent_prefix, mut receipt) = match &message.target {
                SubagentUplinkTarget::Parent { kind, text } => {
                    let intent = if *kind == echo_core::tools::SubagentUplinkKind::Escalate {
                        "[escalate] "
                    } else {
                        ""
                    };
                    let delivery = match from.parent_execution_id.as_deref() {
                        Some(execution_id) => {
                            registry
                                .control_registry()
                                .send_message_to_active(execution_id, text.clone())
                                .await
                        }
                        None => Err(SubagentControlError::UnknownExecution {
                            execution_id: String::new(),
                        }),
                    };
                    let receipt = match delivery {
                        Ok(receipt) => SubagentUplinkReceipt {
                            accepted: true,
                            status: "parent_steered".to_string(),
                            detail: format!("queued into parent attempt {}", receipt.attempt),
                        },
                        Err(error) => SubagentUplinkReceipt {
                            accepted: true,
                            status: "event_emitted".to_string(),
                            detail: format!(
                                "parent not steerable ({error}); surfaced for host listeners"
                            ),
                        },
                    };
                    ("parent", intent, receipt)
                }
                SubagentUplinkTarget::Sibling { to, text } => {
                    let receipt = match to {
                        echo_core::tools::SubagentPeerAddress::ByExecutionId(execution_id) => {
                            match registry
                                .control_registry()
                                .send_message_to_active(execution_id, text.clone())
                                .await
                            {
                                Ok(receipt) => SubagentUplinkReceipt {
                                    accepted: true,
                                    status: "delivered_to_sibling".to_string(),
                                    detail: format!(
                                        "queued into sibling attempt {}",
                                        receipt.attempt
                                    ),
                                },
                                Err(error) => SubagentUplinkReceipt {
                                    accepted: false,
                                    status: "sibling_not_active".to_string(),
                                    detail: error.to_string(),
                                },
                            }
                        }
                        echo_core::tools::SubagentPeerAddress::ByTaskId(task_id) => {
                            SubagentUplinkReceipt {
                                accepted: false,
                                status: "task_addressing_requires_application_sink".to_string(),
                                detail: format!(
                                    "task '{task_id}' has no live attempt; an application                                      sink owning attempt scheduling must deliver it"
                                ),
                            }
                        }
                    };
                    ("sibling", "", receipt)
                }
            };
            let summary: String = {
                let text = match &message.target {
                    SubagentUplinkTarget::Parent { text, .. }
                    | SubagentUplinkTarget::Sibling { text, .. } => text,
                };
                format!(
                    "{intent_prefix}{}",
                    text.chars().take(200).collect::<String>()
                )
            };
            let uplink_event = SubagentEvent::UplinkReceived {
                parent,
                agent,
                direction: direction.to_string(),
                status: receipt.status.clone(),
                summary,
                execution_id: from.execution_id.clone(),
                run_id: from.run_id.clone(),
            };
            let published = from
                .execution_id
                .as_deref()
                .and_then(|execution_id| registry.event_bus().publisher_for_execution(execution_id))
                .is_some_and(|publisher| match publisher.emit(uplink_event) {
                    Ok(_) => true,
                    Err(error) => {
                        warn!(%error, "failed to publish Subagent uplink envelope");
                        false
                    }
                });
            if !published {
                warn!(
                    execution_id = from.execution_id.as_deref().unwrap_or("<none>"),
                    "Subagent uplink has no active event publisher"
                );
                if receipt.status == "event_emitted" {
                    receipt.accepted = false;
                    receipt.status = "event_unavailable".to_string();
                    receipt.detail =
                        "parent not steerable and no active event publisher exists".to_string();
                }
            }
            receipt
        })
    })
}

impl SubagentAttemptControlHandle {
    /// 返回此句柄所绑定的控制作用域标识。
    pub fn control_scope_id(&self) -> &str {
        &self.control_scope_id
    }

    /// 从任务子代理上下文推导出对应的执行尝试身份标识。
    fn identity_from_context(
        &self,
        context: &TaskSubagentContext,
    ) -> std::result::Result<SubagentAttemptIdentity, SubagentControlError> {
        let claim = context
            .claim()
            .ok_or(SubagentControlError::InvalidIdentity { field: "claim" })?;
        let task_id = context
            .task_id()
            .ok_or(SubagentControlError::InvalidIdentity { field: "task_id" })?;
        SubagentAttemptIdentity::for_runtime_scope(
            self.control_scope_id.clone(),
            context.run_id().to_string(),
            task_id.to_string(),
            claim.execution_id(context.run_id(), task_id),
            claim.attempt,
        )
    }

    /// 根据 run_id、task_id 与 TaskClaim 直接构造出执行尝试身份标识。
    fn identity_from_claim(
        &self,
        run_id: &str,
        task_id: &str,
        claim: &TaskClaim,
    ) -> std::result::Result<SubagentAttemptIdentity, SubagentControlError> {
        SubagentAttemptIdentity::for_runtime_scope(
            self.control_scope_id.clone(),
            run_id.to_string(),
            task_id.to_string(),
            claim.execution_id(run_id, task_id),
            claim.attempt,
        )
    }

    /// 在某个精确的 TaskClaim 执行尝试进入准入等待之前，先为其预留位置。
    pub fn reserve(
        &self,
        context: &TaskSubagentContext,
    ) -> std::result::Result<(), SubagentControlError> {
        let identity = self.identity_from_context(context)?;
        self.executor
            .reserve_attempt(identity, context.cancellation_token().clone())
    }

    /// 通过此句柄所属的注册表，派发某个精确的 TaskClaim 执行尝试。
    pub async fn dispatch(
        &self,
        mut request: DispatchRequest,
        context: TaskSubagentContext,
    ) -> Result<SubagentResult> {
        let identity = self
            .identity_from_context(&context)
            .map_err(SubagentExecutor::control_react_error)?;
        bind_task_dispatch_context(&mut request, &context)?;
        self.executor.dispatch_attempt(request, identity).await
    }

    /** 投影一个已经通过持久化 TaskClaim 校验的中断请求。

    持久化侧调用方只能从
    [`crate::tasks::RuntimeDagController::request_live_interrupt`] 调用此方法。
    */
    pub fn project_interrupt(
        &self,
        run_id: &str,
        task_id: &str,
        claim: &TaskClaim,
    ) -> std::result::Result<
        RuntimeTaskAttemptInterruptDisposition,
        RuntimeTaskAttemptInterruptProjectionError,
    > {
        let identity = self
            .identity_from_claim(run_id, task_id, claim)
            .map_err(runtime_interrupt_projection_error)?;
        self.executor
            .request_interrupt(identity)
            .map(|receipt| runtime_interrupt_disposition(receipt.disposition))
            .map_err(runtime_interrupt_projection_error)
    }

    /// 在某个持久化 claim 状态推进之后，退役对应的实时投影。
    pub fn retire(
        &self,
        run_id: &str,
        task: &Task,
        claim: &TaskClaim,
    ) -> RuntimeAttemptControlCleanupReceipt {
        let identity = match self.identity_from_claim(run_id, &task.spec.id, claim) {
            Ok(identity) => identity,
            Err(error) => {
                return RuntimeAttemptControlCleanupReceipt::RetryableFailure {
                    error: error.to_string(),
                };
            }
        };
        match self.executor.retire_attempt_control(&identity) {
            Ok(true) => RuntimeAttemptControlCleanupReceipt::Retired,
            Ok(false) => RuntimeAttemptControlCleanupReceipt::AlreadyConsumed,
            Err(error) => RuntimeAttemptControlCleanupReceipt::RetryableFailure {
                error: error.to_string(),
            },
        }
    }

    /// 仅对此句柄固定的控制作用域执行一次状态核对。
    pub fn reconcile(
        &self,
        run_id: &str,
        current_execution_ids: &HashSet<String>,
    ) -> std::result::Result<(), SubagentControlError> {
        self.executor.reconcile_attempt_control(
            &self.control_scope_id,
            run_id,
            current_execution_ids,
        )
    }
}

/// 从任务子代理上下文中提取精确的派发身份信息，并将其绑定到派发请求上。
fn bind_task_dispatch_context(
    request: &mut DispatchRequest,
    context: &TaskSubagentContext,
) -> Result<()> {
    let exact = ExactTaskDispatchContext::from_task_context(context)?;
    bind_exact_dispatch_context(request, &exact)
}

/**
一次任务派发所需的精确身份与运行时信息（run/execution/task id、
尝试次数、计划版本、取消令牌与委派策略）。
*/
struct ExactTaskDispatchContext {
    /// 所属的业务 run 标识。
    run_id: String,
    /// 本次执行尝试的 execution id。
    execution_id: String,
    /// 所属任务的 task id。
    task_id: String,
    /// 当前尝试次数。
    attempt: u32,
    /// 当前计划（DAG）版本号。
    plan_revision: u64,
    /// 本次尝试的取消令牌。
    cancel: CancellationToken,
    /// 委派策略（限制嵌套委派深度）。
    delegation_policy: NestedDelegationPolicy,
}

impl ExactTaskDispatchContext {
    /**
    从任务子代理上下文中提取出精确派发上下文；若缺少必要的身份信息
    （execution id、task id 或 claim）则返回错误。
    */
    fn from_task_context(context: &TaskSubagentContext) -> Result<Self> {
        let execution_id = context.execution_id().ok_or_else(|| {
            ReactError::Other("exact task dispatch requires an execution identity".to_string())
        })?;
        let task_id = context.task_id().ok_or_else(|| {
            ReactError::Other("exact task dispatch requires a task identity".to_string())
        })?;
        let claim = context.claim().ok_or_else(|| {
            ReactError::Other("exact task dispatch requires a TaskClaim".to_string())
        })?;
        Ok(Self {
            run_id: context.run_id().to_string(),
            execution_id,
            task_id: task_id.to_string(),
            attempt: claim.attempt,
            plan_revision: claim.revision,
            cancel: context.cancellation_token().clone(),
            delegation_policy: context.delegation_policy(),
        })
    }
}

/**
将精确的派发身份信息绑定进派发请求的运行时上下文与血缘信息中；
若已存在的字段与精确身份不一致，则返回错误而不做覆盖。
*/
fn bind_exact_dispatch_context(
    request: &mut DispatchRequest,
    exact: &ExactTaskDispatchContext,
) -> Result<()> {
    let runtime = request
        .runtime_context
        .get_or_insert_with(echo_core::tools::ExternalRunContext::default);
    ensure_string_identity("run_id", runtime.run_id.as_deref(), &exact.run_id)?;
    ensure_string_identity(
        "execution_id",
        runtime.execution_id.as_deref(),
        &exact.execution_id,
    )?;
    runtime.run_id = Some(exact.run_id.clone());
    runtime.execution_id = Some(exact.execution_id.clone());
    if runtime.turn_id.is_none() {
        runtime.turn_id = Some(exact.execution_id.clone());
    }
    runtime.cancel = Some(Arc::new(exact.cancel.clone()));
    runtime.delegation_policy = Some(exact.delegation_policy);
    let lineage = runtime
        .subagent_lineage
        .get_or_insert_with(Default::default);
    ensure_string_identity("lineage.run_id", lineage.run_id.as_deref(), &exact.run_id)?;
    ensure_string_identity(
        "lineage.task_id",
        lineage.task_id.as_deref(),
        &exact.task_id,
    )?;
    ensure_string_identity(
        "lineage.execution_id",
        lineage.execution_id.as_deref(),
        &exact.execution_id,
    )?;
    ensure_numeric_identity("lineage.attempt", lineage.attempt, exact.attempt)?;
    ensure_numeric_identity(
        "lineage.plan_revision",
        lineage.plan_revision,
        exact.plan_revision,
    )?;
    lineage.run_id = Some(exact.run_id.clone());
    lineage.task_id = Some(exact.task_id.clone());
    lineage.execution_id = Some(exact.execution_id.clone());
    lineage.attempt = Some(exact.attempt);
    lineage.plan_revision = Some(exact.plan_revision);
    request.cancel = exact.cancel.clone();
    request.delegation_policy = exact.delegation_policy;
    Ok(())
}

/// 校验某个字符串身份字段（若已存在）与期望值一致，否则返回不匹配错误。
fn ensure_string_identity(field: &str, actual: Option<&str>, expected: &str) -> Result<()> {
    if let Some(actual) = actual
        && actual != expected
    {
        return Err(ReactError::Other(format!(
            "Subagent {field} mismatch: expected {expected}, actual {actual}"
        )));
    }
    Ok(())
}

/// 校验某个数值型身份字段（若已存在）与期望值一致，否则返回不匹配错误。
fn ensure_numeric_identity<T>(field: &str, actual: Option<T>, expected: T) -> Result<()>
where
    T: Copy + PartialEq + std::fmt::Display,
{
    if let Some(actual) = actual
        && actual != expected
    {
        return Err(ReactError::Other(format!(
            "Subagent {field} mismatch: expected {expected}, actual {actual}"
        )));
    }
    Ok(())
}

/// 将控制层的中断请求处理结果映射为运行时任务尝试中断的处理结果类型。
fn runtime_interrupt_disposition(
    disposition: SubagentInterruptRequestDisposition,
) -> RuntimeTaskAttemptInterruptDisposition {
    match disposition {
        SubagentInterruptRequestDisposition::QueuedBeforeAdmission => {
            RuntimeTaskAttemptInterruptDisposition::QueuedBeforeReservation
        }
        SubagentInterruptRequestDisposition::ReservedRequested => {
            RuntimeTaskAttemptInterruptDisposition::ReservedRequested
        }
        SubagentInterruptRequestDisposition::ActiveRequested => {
            RuntimeTaskAttemptInterruptDisposition::ActiveRequested
        }
        SubagentInterruptRequestDisposition::ActiveAlreadyRequested => {
            RuntimeTaskAttemptInterruptDisposition::ActiveAlreadyRequested
        }
        SubagentInterruptRequestDisposition::AlreadySettled(status) => {
            RuntimeTaskAttemptInterruptDisposition::AlreadySettled {
                status: status.as_str().to_string(),
            }
        }
    }
}

/// 将子代理控制层错误映射为运行时任务尝试中断投影所使用的错误类型。
fn runtime_interrupt_projection_error(
    error: SubagentControlError,
) -> RuntimeTaskAttemptInterruptProjectionError {
    match error {
        SubagentControlError::PendingCapacityExceeded { limit } => {
            RuntimeTaskAttemptInterruptProjectionError::PendingCapacityExceeded { limit }
        }
        SubagentControlError::IdentityConflict {
            task_id,
            attempt,
            expected_execution_id,
            actual_execution_id,
        } => RuntimeTaskAttemptInterruptProjectionError::IdentityConflict {
            task_id,
            attempt,
            expected_execution_id,
            actual_execution_id,
        },
        error => RuntimeTaskAttemptInterruptProjectionError::Unavailable {
            message: error.to_string(),
        },
    }
}

impl SubagentExecutor {
    /// 创建一个由给定注册表支撑的新执行器。
    pub fn new(registry: Arc<SubagentRegistry>, config: SubagentExecutorConfig) -> Self {
        let semaphore = Arc::new(Semaphore::new(config.max_concurrent_forks));
        let shared_admission = config.shared_admission.clone();
        Self {
            control_registry: Arc::clone(registry.control_registry()),
            registry,
            hooks: Arc::new(SubagentHookRegistry::new()),
            config,
            semaphore,
            shared_admission: Arc::new(tokio::sync::RwLock::new(shared_admission)),
            team_runtimes: Arc::new(Mutex::new(TeamRuntimeIndex::default())),
        }
    }

    /// 使用指定的钩子注册表创建执行器。
    pub fn with_hooks(
        registry: Arc<SubagentRegistry>,
        config: SubagentExecutorConfig,
        hooks: SubagentHookRegistry,
    ) -> Self {
        let semaphore = Arc::new(Semaphore::new(config.max_concurrent_forks));
        let shared_admission = config.shared_admission.clone();
        Self {
            control_registry: Arc::clone(registry.control_registry()),
            registry,
            hooks: Arc::new(hooks),
            config,
            semaphore,
            shared_admission: Arc::new(tokio::sync::RwLock::new(shared_admission)),
            team_runtimes: Arc::new(Mutex::new(TeamRuntimeIndex::default())),
        }
    }

    /// 获取钩子注册表（供外部注册钩子使用）。
    pub fn hooks(&self) -> &SubagentHookRegistry {
        &self.hooks
    }

    /// 共享的注册表句柄（供需要查询定义的调用方使用）。
    pub fn registry(&self) -> &Arc<SubagentRegistry> {
        &self.registry
    }

    /// 为外部任务适配器创建一个完整的实时控制能力句柄。
    pub fn attempt_control_handle(
        &self,
        control_scope_id: impl Into<String>,
    ) -> std::result::Result<SubagentAttemptControlHandle, SubagentControlError> {
        let control_scope_id = control_scope_id.into();
        if control_scope_id.trim().is_empty() {
            return Err(SubagentControlError::InvalidIdentity {
                field: "control_scope_id",
            });
        }
        Ok(SubagentAttemptControlHandle {
            executor: self.clone_for_spawn(),
            control_scope_id,
        })
    }

    /// 按唯一的句柄标识返回一个已保留的 Team 运行时。
    pub fn team_runtime_handle(
        &self,
        handle_id: &str,
    ) -> Option<Arc<super::team::TeamRuntimeHandle>> {
        self.team_runtimes
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .handles
            .get(handle_id)
            .cloned()
    }

    /**
    返回某个业务 run 下所有已保留的 Team 运行时。多个嵌套或并发的 Team
    执行完全可能合法地共享同一个 run ID。
    */
    pub fn team_runtime_handles(&self, run_id: &str) -> Vec<Arc<super::team::TeamRuntimeHandle>> {
        let runtimes = self
            .team_runtimes
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        runtimes
            .order
            .iter()
            .filter_map(|handle_id| runtimes.handles.get(handle_id))
            .filter(|handle| handle.run_id() == run_id)
            .cloned()
            .collect()
    }

    /// 保留一个 Team 运行时句柄，并触发一次索引淘汰整理。
    fn retain_team_runtime(&self, handle: Arc<super::team::TeamRuntimeHandle>) {
        let mut runtimes = self
            .team_runtimes
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let handle_id = handle.handle_id().to_string();
        if !runtimes.handles.contains_key(&handle_id) {
            runtimes.order.push_back(handle_id.clone());
        }
        runtimes.handles.insert(handle_id, handle);
        drop(runtimes);
        prune_team_runtime_index(&self.team_runtimes);
    }

    /// 为直接派发注入一个由应用层持有的进程级准入控制器。
    pub async fn set_shared_admission(&self, admission: Arc<ExecutionAdmission>) {
        *self.shared_admission.write().await = Some(admission);
    }

    /** 主派发入口。

    根据定义或覆盖值路由到相应的执行模式。使用循环而非递归来实现
    重试/委派，以避免栈溢出。
    */
    pub async fn dispatch(&self, req: DispatchRequest) -> Result<SubagentResult> {
        self.dispatch_inner(req, None).await
    }

    /** 在实时控制下，派发一个身份已被显式指定的任务执行尝试。

    该身份的作用域仅限于当前进程。持久化的 run/revision/命令校验仍由
    框架的消费方负责。
    */
    pub async fn dispatch_attempt(
        &self,
        req: DispatchRequest,
        identity: SubagentAttemptIdentity,
    ) -> Result<SubagentResult> {
        let (req, admission) = self.admit_attempt(req, identity)?;
        self.dispatch_admitted_attempt(req, admission).await
    }

    /// 将派发请求绑定到给定身份，并向控制注册表申请该尝试的准入。
    fn admit_attempt(
        &self,
        mut req: DispatchRequest,
        identity: SubagentAttemptIdentity,
    ) -> Result<(DispatchRequest, super::control::SubagentAttemptAdmission)> {
        Self::bind_attempt_runtime_identity(&mut req, &identity)?;
        let admission = self
            .control_registry
            .admit(identity, req.cancel.clone())
            .map_err(Self::control_react_error)?;
        req.cancel = admission.cancel.clone();
        if let Some(context) = req.runtime_context.as_mut() {
            context.cancel = Some(Arc::new(req.cancel.clone()));
        }
        Self::append_queued_guidance(&mut req.task, &admission.guidance);
        Ok((req, admission))
    }

    /// 执行一个已获准入的尝试，并在结束后用最终状态结算该准入。
    async fn dispatch_admitted_attempt(
        &self,
        req: DispatchRequest,
        admission: super::control::SubagentAttemptAdmission,
    ) -> Result<SubagentResult> {
        let binding = admission.binding.clone();
        let result = self.dispatch_inner(req, Some(binding)).await;
        let status = match &result {
            Ok(result) => result.outcome.status,
            Err(error) => subagent_status_from_error(error),
        };
        admission.settle(status);
        result
    }

    /// 向某个精确的活跃尝试投递一条实时指令。
    pub async fn send_message_tracked(
        &self,
        execution_id: &str,
        expected_attempt: u32,
        instruction: impl Into<String>,
    ) -> std::result::Result<SubagentMessageReceipt, SubagentControlError> {
        self.control_registry
            .send_message_tracked(execution_id, expected_attempt, instruction)
            .await
    }

    /// 为某个精确的未来尝试排队指导信息；准入时仅认领一次。
    pub fn queue_guidance(
        &self,
        task_id: &str,
        expected_next_attempt: u32,
        instruction: impl Into<String>,
    ) -> std::result::Result<SubagentGuidanceQueueReceipt, SubagentControlError> {
        self.control_registry
            .queue_guidance(task_id, expected_next_attempt, instruction)
    }

    /// 取消某个精确的尝试，并等待其派发结算完成。
    pub async fn interrupt_subagent(
        &self,
        execution_id: &str,
        expected_attempt: u32,
    ) -> std::result::Result<SubagentInterruptOutcome, SubagentControlError> {
        self.control_registry
            .interrupt(execution_id, expected_attempt)
            .await
    }

    /**
    请求中断某个精确的尝试，但不等待其结算。

    这一进程内的确认可能在派发准入之前就已排队。持久化调用方必须
    先校验对应的 TaskClaim，再调用这个注册表层面的投影操作。
    */
    pub fn request_interrupt(
        &self,
        identity: SubagentAttemptIdentity,
    ) -> std::result::Result<super::control::SubagentInterruptRequestReceipt, SubagentControlError>
    {
        self.control_registry.request_interrupt(identity)
    }

    /**
    在已认领的 Team 任务等待执行容量之前，把它绑定到精确的实时控制身份；
    之后的派发会消费该绑定。
    */
    pub(crate) fn reserve_attempt(
        &self,
        identity: SubagentAttemptIdentity,
        cancel: CancellationToken,
    ) -> std::result::Result<(), SubagentControlError> {
        self.control_registry.reserve(identity, cancel)
    }

    /// 从控制注册表中精确退役指定尝试的控制绑定，返回是否确实移除。
    pub(crate) fn retire_attempt_control(
        &self,
        identity: &SubagentAttemptIdentity,
    ) -> std::result::Result<bool, SubagentControlError> {
        self.control_registry.retire_exact(identity)
    }

    /// 按当前仍有效的执行 id 集合，对某控制作用域/run 下的尝试控制进行对账清理。
    pub(crate) fn reconcile_attempt_control(
        &self,
        control_scope_id: &str,
        run_id: &str,
        current_execution_ids: &HashSet<String>,
    ) -> std::result::Result<(), SubagentControlError> {
        self.control_registry
            .reconcile_scope(control_scope_id, run_id, current_execution_ids)
    }

    /**
    为派发请求填充身份/血缘的基础字段，并安装默认的上行 sink。

    优先级：调用方已填写的血缘字段始终优先（`agent_tool` 路径会填写父路径/
    execution id；应用层任务运行时会填写 task/attempt/revision），这里只推导
    缺失的基础字段。应用提供的上行 sink 永远不会被替换——框架默认实现
    （事件总线 + 共享控制注册表投递）只是兜底，而非覆盖。
    */
    fn enrich_dispatch_context(&self, req: &mut DispatchRequest) {
        let Some(context) = req.runtime_context.as_mut() else {
            return;
        };
        let lineage = context
            .subagent_lineage
            .get_or_insert_with(Default::default);
        if lineage.agent_name.is_none() {
            lineage.agent_name = Some(req.agent_name.clone());
        }
        if lineage.parent_agent.is_none() {
            lineage.parent_agent = Some(req.parent_agent.clone());
        }
        if lineage.agent_path.is_none() {
            lineage.agent_path = Some(format!("root/{}", req.agent_name));
        }
        if lineage.execution_id.is_none() {
            lineage.execution_id = context.execution_id.clone();
        }
        if lineage.run_id.is_none() {
            lineage.run_id = context.run_id.clone();
        }
        if context.uplink.is_none() {
            context.uplink = Some(default_uplink_sink(Arc::clone(&self.registry)));
        }
    }

    /// 基于请求的运行时上下文构建本次派发的事件发布器。
    fn event_publisher(&self, req: &DispatchRequest) -> Result<SubagentEventPublisher> {
        let runtime = req.runtime_context.as_ref();
        let transport_identity = EventIdentity::from_runtime_context(runtime)?;
        let invocation_identity = Self::event_invocation_identity(req, &transport_identity)?;
        self.registry
            .event_bus()
            .publisher(transport_identity, invocation_identity)
    }

    /// 根据请求与传输层身份推导子 Agent 调用身份（含血缘信息）。
    fn event_invocation_identity(
        req: &DispatchRequest,
        transport_identity: &EventIdentity,
    ) -> Result<SubagentInvocationIdentity> {
        let runtime = req.runtime_context.as_ref();
        SubagentInvocationIdentity::from_lineage(
            req.parent_agent.clone(),
            req.agent_name.clone(),
            transport_identity,
            runtime.and_then(|context| context.subagent_lineage.as_ref()),
        )
    }

    /// 委派到其它 Agent 后，重写运行时上下文中的血缘信息（agent 名、父 agent、路径）以指向新目标。
    fn retarget_dispatch_context(req: &mut DispatchRequest) {
        let Some(context) = req.runtime_context.as_mut() else {
            return;
        };
        let lineage = context
            .subagent_lineage
            .get_or_insert_with(Default::default);
        lineage.agent_name = Some(req.agent_name.clone());
        lineage.parent_agent = Some(req.parent_agent.clone());
        lineage.agent_path = Some(
            lineage
                .agent_path
                .as_deref()
                .and_then(|path| path.rsplit_once('/').map(|(parent, _)| parent))
                .map(|parent| format!("{parent}/{}", req.agent_name))
                .unwrap_or_else(|| format!("root/{}", req.agent_name)),
        );
    }

    /// 在真正执行前失败时发出终止事件：取消类错误发 DispatchCancelled，其余发 DispatchFailed。
    fn emit_pre_execution_failure(
        publisher: &SubagentEventPublisher,
        req: &DispatchRequest,
        error: &ReactError,
    ) -> Result<()> {
        let status = subagent_status_from_error(error);
        let detail = error.to_string();
        let outcome = SubagentOutcome::terminal(status, detail.clone(), vec![detail.clone()]);
        let runtime = req.runtime_context.as_ref();
        if status == SubagentStatus::Cancelled {
            publisher.emit(SubagentEvent::DispatchCancelled {
                parent: req.parent_agent.clone(),
                agent: req.agent_name.clone(),
                outcome,
                execution_id: runtime.and_then(|context| context.execution_id.clone()),
                run_id: runtime.and_then(|context| context.run_id.clone()),
            })?;
        } else {
            publisher.emit(SubagentEvent::DispatchFailed {
                parent: req.parent_agent.clone(),
                agent: req.agent_name.clone(),
                error: detail,
                status,
                outcome,
                execution_id: runtime.and_then(|context| context.execution_id.clone()),
                run_id: runtime.and_then(|context| context.run_id.clone()),
            })?;
        }
        Ok(())
    }

    /// 根据结果状态构造对应的终止事件；若结果仍处于 Running，则将其改写为 Failed。
    fn terminal_event_for_result(
        parent: &str,
        agent: &str,
        result: &mut SubagentResult,
        execution_id: Option<String>,
        run_id: Option<String>,
    ) -> SubagentEvent {
        match result.outcome.status {
            SubagentStatus::Cancelled => SubagentEvent::DispatchCancelled {
                parent: parent.to_string(),
                agent: agent.to_string(),
                outcome: result.outcome.clone(),
                execution_id,
                run_id,
            },
            SubagentStatus::Completed => SubagentEvent::DispatchCompleted {
                parent: parent.to_string(),
                agent: agent.to_string(),
                duration_ms: u64::try_from(result.duration.as_millis()).unwrap_or(u64::MAX),
                tokens_used: result
                    .tokens_used
                    .map(|tokens| u64::try_from(tokens).unwrap_or(u64::MAX)),
                iterations: Some(u64::try_from(result.iterations).unwrap_or(u64::MAX)),
                output: result.output.clone(),
                outcome: result.outcome.clone(),
                execution_id,
                run_id,
            },
            status => {
                let status = if status == SubagentStatus::Running {
                    result.outcome.status = SubagentStatus::Failed;
                    result.outcome.summary =
                        "Subagent returned without a terminal outcome".to_string();
                    SubagentStatus::Failed
                } else {
                    status
                };
                let error = if result.outcome.summary.is_empty() {
                    format!("Subagent '{agent}' reported {}", status.as_str())
                } else {
                    result.outcome.summary.clone()
                };
                SubagentEvent::DispatchFailed {
                    parent: parent.to_string(),
                    agent: agent.to_string(),
                    error,
                    status,
                    outcome: result.outcome.clone(),
                    execution_id,
                    run_id,
                }
            }
        }
    }

    /// 派发核心循环：发出开始事件、执行前置校验、按模式派发，并处理重试/委派与终止事件。
    #[async_recursion::async_recursion]
    async fn dispatch_inner(
        &self,
        mut req: DispatchRequest,
        control: Option<SubagentAttemptBinding>,
    ) -> Result<SubagentResult> {
        let mut retry_count: u32 = 0;
        let max_retries: u32 = 3; // 防止无限重试循环
        /*
        保存父级取消令牌，使重试/委派路径传播取消信号，
        而不是创建相互独立的令牌（P1 —— CancellationToken 传播）。
        */
        let parent_cancel = req.cancel.clone();

        /*
        在任何已接受的派发能够结算或初始化失败之前，先填充身份并创建
        唯一的、以尝试为作用域的顺序权威。
        */
        self.enrich_dispatch_context(&mut req);
        let event_publisher = self.event_publisher(&req)?;
        let initial_registration = self.registry.get(&req.agent_name).await;
        let initial_mode = req
            .mode_override
            .clone()
            .or_else(|| {
                initial_registration
                    .as_ref()
                    .map(|registered| registered.definition.execution_mode.clone())
            })
            .unwrap_or(ExecutionMode::Sync);
        let runtime = req.runtime_context.as_ref();
        event_publisher.emit(SubagentEvent::DispatchStarted {
            parent: req.parent_agent.clone(),
            agent: req.agent_name.clone(),
            mode: initial_mode.clone(),
            task: req.task.clone(),
            execution_id: runtime.and_then(|context| context.execution_id.clone()),
            run_id: runtime.and_then(|context| context.run_id.clone()),
            conversation_id: runtime.and_then(|context| context.conversation_id.clone()),
            message_id: runtime.and_then(|context| context.message_id.clone()),
            background: req.background,
        })?;

        if parent_cancel.is_cancelled() {
            let result = SubagentResult::cancelled(
                req.agent_name.clone(),
                "Cancelled before execution",
                initial_mode,
            );
            event_publisher.emit(SubagentEvent::DispatchCancelled {
                parent: req.parent_agent.clone(),
                agent: req.agent_name.clone(),
                outcome: result.outcome.clone(),
                execution_id: runtime.and_then(|context| context.execution_id.clone()),
                run_id: runtime.and_then(|context| context.run_id.clone()),
            })?;
            return Ok(result);
        }

        if req.delegation_policy.delegate_depth > req.delegation_policy.max_delegate_depth {
            let error = ReactError::Other(format!(
                "Delegation depth exceeded (max {}): agent '{}'",
                req.delegation_policy.max_delegate_depth, req.agent_name
            ));
            Self::emit_pre_execution_failure(&event_publisher, &req, &error)?;
            return Err(error);
        }
        if initial_registration.is_none() {
            let error = ReactError::Other(format!("Subagent '{}' not found", req.agent_name));
            Self::emit_pre_execution_failure(&event_publisher, &req, &error)?;
            return Err(error);
        }

        loop {
            // 防止重试次数过多
            if retry_count > max_retries {
                let error = ReactError::Agent(Box::new(AgentError::ContextLimitExceeded(format!(
                    "Max retry count exceeded ({}): agent '{}'",
                    max_retries, req.agent_name
                ))));
                Self::emit_pre_execution_failure(&event_publisher, &req, &error)?;
                return Err(error);
            }

            // 查找定义
            let registered = match self.registry.get(&req.agent_name).await {
                Some(registered) => registered,
                None => {
                    let error =
                        ReactError::Other(format!("Subagent '{}' not found", req.agent_name));
                    Self::emit_pre_execution_failure(&event_publisher, &req, &error)?;
                    return Err(error);
                }
            };

            let mode = req
                .mode_override
                .as_ref()
                .unwrap_or(&registered.definition.execution_mode)
                .clone();
            let runtime_run_id = req
                .runtime_context
                .as_ref()
                .and_then(|ctx| ctx.run_id.as_deref())
                .unwrap_or("<none>");
            /*
            为事件负载提取稳定身份（Option<String>）。
            所有 Dispatch* 事件都会使用它，使桥接层/前端无需分配临时 id
            就能路由思考/工具/token 流。
            */
            let event_execution_id = req
                .runtime_context
                .as_ref()
                .and_then(|ctx| ctx.execution_id.clone());
            let event_run_id = req
                .runtime_context
                .as_ref()
                .and_then(|ctx| ctx.run_id.clone());
            let has_trace_sink = req
                .runtime_context
                .as_ref()
                .is_some_and(|ctx| ctx.trace_sink.is_some());
            let has_cancel = req
                .runtime_context
                .as_ref()
                .is_some_and(|ctx| ctx.cancel.is_some());
            info!(
                parent = %req.parent_agent,
                subagent = %req.agent_name,
                mode = ?mode,
                attempt = retry_count + 1,
                delegate_depth = req.delegation_policy.delegate_depth,
                runtime_run_id = %runtime_run_id,
                has_runtime_context = req.runtime_context.is_some(),
                has_trace_sink,
                has_cancel,
                task_chars = req.task.chars().count(),
                "subagent_dispatch_start"
            );

            // 构建钩子上下文
            let hook_ctx = SubagentHookContext {
                parent_agent: req.parent_agent.clone(),
                subagent_name: req.agent_name.clone(),
                execution_mode: mode.clone(),
                task: req.task.clone(),
                attempt: 1 + retry_count,
            };

            if self.config.enable_hooks {
                self.hooks.before_dispatch(&hook_ctx).await;
            }

            // 在 `req` 被移动之前，快照错误路径所需的字段
            let req_agent_name = req.agent_name.clone();
            let req_parent_agent = req.parent_agent.clone();
            let delegation_policy = req.delegation_policy;

            // 为本次具体的派发尝试触发统一的 SubagentStart 钩子。
            if let Some(ref executor) = self.config.unified_hook_executor {
                let ctx = correlate_subagent_hook(
                    crate::skills::hooks::HookContext::for_subagent_start(
                        &req_agent_name,
                        &mode.to_string(),
                        &req.task,
                        "", // 该层拿不到 session_id
                        &req_parent_agent,
                    ),
                    event_run_id.as_deref(),
                    event_execution_id.as_deref(),
                    hook_ctx.attempt,
                );
                executor(ctx).await;
            }

            // 按模式派发
            let start = Instant::now();
            let result = match mode {
                ExecutionMode::Sync => {
                    self.dispatch_sync(&req, control.clone(), event_publisher.clone())
                        .await
                }
                ExecutionMode::Fork => {
                    self.dispatch_fork(&req, control.clone(), event_publisher.clone())
                        .await
                }
                ExecutionMode::Teammate => {
                    // Teammate 模式：独立 spawn，然后等待结果
                    match self
                        .dispatch_teammate_with_control(
                            req.clone(),
                            control.clone(),
                            event_publisher.clone(),
                            false,
                        )
                        .await
                    {
                        Ok(handle) => handle.join().await,
                        Err(e) => Err(e),
                    }
                }
                ExecutionMode::Team => self.dispatch_team(&req, event_publisher.clone()).await,
            };

            let duration = start.elapsed();

            match result {
                Ok(mut sub_result) => {
                    sub_result.duration = duration;
                    sub_result.mode = mode.clone();
                    info!(
                        parent = %req_parent_agent,
                        subagent = %req_agent_name,
                        mode = ?mode,
                        duration_ms = duration.as_millis() as u64,
                        output_chars = sub_result.output.chars().count(),
                        tokens_used = ?sub_result.tokens_used,
                        iterations = sub_result.iterations,
                        "subagent_dispatch_complete"
                    );

                    event_publisher.emit(Self::terminal_event_for_result(
                        &req_parent_agent,
                        &req_agent_name,
                        &mut sub_result,
                        event_execution_id.clone(),
                        event_run_id.clone(),
                    ))?;

                    if self.config.enable_hooks {
                        self.hooks.after_dispatch(&hook_ctx, &sub_result).await;
                    }

                    /*
                    每个 Start 恰好对应一个 Stop。执行器可能在 Ok 中返回
                    结构化的取消/失败结果，因此要按结果映射状态，而不是
                    仅凭 Result 分支就假定为 Completed。
                    */
                    if let Some(ref executor) = self.config.unified_hook_executor {
                        let ctx = correlate_subagent_hook(
                            crate::skills::hooks::HookContext::for_subagent_stop(
                                &req_agent_name,
                                &mode.to_string(),
                                &sub_result.output,
                                hook_stop_status(sub_result.outcome.status),
                                "",
                                &req_parent_agent,
                            ),
                            event_run_id.as_deref(),
                            event_execution_id.as_deref(),
                            hook_ctx.attempt,
                        );
                        executor(ctx).await;
                    }

                    return Ok(sub_result);
                }
                Err(e) => {
                    let error_str = e.to_string();
                    let status = subagent_status_from_error(&e);
                    let terminal_result = SubagentOutcome::terminal(
                        status,
                        error_str.clone(),
                        vec![error_str.clone()],
                    );
                    warn!(
                        parent = %req_parent_agent,
                        subagent = %req_agent_name,
                        mode = ?mode,
                        error = %error_str,
                        "subagent_dispatch_failed"
                    );

                    /*
                    在取消返回或钩子策略启动下一次尝试之前，先关闭本次尝试。
                    以前取消会在唯一的 Stop 调用之前返回，而重试则会发出
                    多个 Start、最后只跟一个终止 Stop。
                    */
                    if let Some(ref executor) = self.config.unified_hook_executor {
                        let ctx = correlate_subagent_hook(
                            crate::skills::hooks::HookContext::for_subagent_stop(
                                &req_agent_name,
                                &mode.to_string(),
                                &format!("error: {error_str}"),
                                hook_stop_status(status),
                                "",
                                &req_parent_agent,
                            ),
                            event_run_id.as_deref(),
                            event_execution_id.as_deref(),
                            hook_ctx.attempt,
                        );
                        executor(ctx).await;
                    }

                    if status == SubagentStatus::Cancelled {
                        event_publisher.emit(SubagentEvent::DispatchCancelled {
                            parent: req_parent_agent.clone(),
                            agent: req_agent_name.clone(),
                            outcome: terminal_result,
                            execution_id: event_execution_id.clone(),
                            run_id: event_run_id.clone(),
                        })?;
                        return Err(e);
                    }

                    if self.config.enable_hooks {
                        let decision = self.hooks.on_failure(&hook_ctx, &error_str).await;
                        match decision {
                            super::hooks::SubagentRetryDecision::Delegate { alternative_agent } => {
                                if let Some(child_policy) = delegation_policy.child_policy() {
                                    info!(
                                        from = %hook_ctx.subagent_name,
                                        to = %alternative_agent,
                                        depth = child_policy.delegate_depth,
                                        "Delegating to alternative subagent"
                                    );
                                    retry_count = retry_count.saturating_add(1);
                                    let rt_ctx = req.runtime_context.clone();
                                    let retry_msg = req.message.clone();
                                    let prompt_payload = req.prompt_payload.clone();
                                    let prompt_context = req.prompt_context.clone();
                                    let parent_context = req.parent_context.clone();
                                    let constraints = req.constraints.clone();
                                    req = DispatchRequest {
                                        agent_name: alternative_agent,
                                        task: hook_ctx.task.clone(),
                                        mode_override: Some(hook_ctx.execution_mode.clone()),
                                        cancel: parent_cancel.child_token(),
                                        parent_agent: hook_ctx.parent_agent.clone(),
                                        parent_context,
                                        delegation_policy: child_policy,
                                        runtime_context: rt_ctx,
                                        message: retry_msg,
                                        prompt_payload,
                                        prompt_context,
                                        constraints,
                                        background: false,
                                    };
                                    Self::retarget_dispatch_context(&mut req);
                                    event_publisher.retarget_from_lineage(
                                        req.parent_agent.clone(),
                                        req.agent_name.clone(),
                                        req.runtime_context
                                            .as_ref()
                                            .and_then(|context| context.subagent_lineage.as_ref()),
                                    )?;
                                    // 本次尝试可恢复，因此不是终止事件。
                                    continue;
                                }
                                warn!(
                                    agent = %hook_ctx.subagent_name,
                                    max_depth = delegation_policy.max_delegate_depth,
                                    "subagent delegation rejected at depth limit"
                                );
                            }
                            super::hooks::SubagentRetryDecision::Retry { delay_secs } => {
                                if retry_count < max_retries {
                                    info!(
                                        delay_secs,
                                        attempt = retry_count.saturating_add(2),
                                        "Retrying subagent dispatch"
                                    );
                                    tokio::time::sleep(Duration::from_secs(delay_secs)).await;
                                    retry_count = retry_count.saturating_add(1);
                                    let rt_ctx = req.runtime_context.clone();
                                    let retry_msg = req.message.clone();
                                    let prompt_payload = req.prompt_payload.clone();
                                    let prompt_context = req.prompt_context.clone();
                                    let parent_context = req.parent_context.clone();
                                    let constraints = req.constraints.clone();
                                    req = DispatchRequest {
                                        agent_name: hook_ctx.subagent_name.clone(),
                                        task: hook_ctx.task.clone(),
                                        mode_override: Some(hook_ctx.execution_mode.clone()),
                                        cancel: parent_cancel.child_token(),
                                        parent_agent: hook_ctx.parent_agent.clone(),
                                        parent_context,
                                        delegation_policy,
                                        runtime_context: rt_ctx,
                                        message: retry_msg,
                                        prompt_payload,
                                        prompt_context,
                                        constraints,
                                        background: false,
                                    };
                                    // 本次尝试可恢复，因此不是终止事件。
                                    continue;
                                }
                                warn!(
                                    agent = %hook_ctx.subagent_name,
                                    max_retries,
                                    "subagent retry limit reached"
                                );
                            }
                            super::hooks::SubagentRetryDecision::Fail => {}
                        }
                    }

                    event_publisher.emit(SubagentEvent::DispatchFailed {
                        parent: req_parent_agent.clone(),
                        agent: req_agent_name.clone(),
                        error: error_str.clone(),
                        status,
                        outcome: terminal_result,
                        execution_id: event_execution_id.clone(),
                        run_id: event_run_id.clone(),
                    })?;

                    return Err(e);
                }
            }
        }
    }

    /// 克隆内部状态，使后台子 Agent 能在 spawn 出的任务中拥有一个执行器。
    fn clone_for_spawn(&self) -> Self {
        Self {
            registry: self.registry.clone(),
            hooks: self.hooks.clone(),
            config: SubagentExecutorConfig {
                max_concurrent_forks: self.config.max_concurrent_forks,
                default_timeout_secs: self.config.default_timeout_secs,
                enable_hooks: self.config.enable_hooks,
                unified_hook_executor: self.config.unified_hook_executor.clone(),
                isolation_provider: self.config.isolation_provider.clone(),
                prompt_compiler: self.config.prompt_compiler.clone(),
                shared_admission: self.config.shared_admission.clone(),
            },
            semaphore: self.semaphore.clone(),
            control_registry: self.control_registry.clone(),
            shared_admission: self.shared_admission.clone(),
            team_runtimes: self.team_runtimes.clone(),
        }
    }

    /// 把尝试身份写入请求的运行时上下文与血缘信息；若已存在不一致的 execution_id 则报错。
    fn bind_attempt_runtime_identity(
        req: &mut DispatchRequest,
        identity: &SubagentAttemptIdentity,
    ) -> Result<()> {
        let context =
            req.runtime_context
                .get_or_insert_with(|| echo_core::tools::ExternalRunContext {
                    conversation_id: None,
                    run_id: None,
                    turn_id: Some(identity.execution_id.clone()),
                    execution_id: Some(identity.execution_id.clone()),
                    isolation_id: None,
                    message_id: None,
                    cancel: None,
                    trace_sink: None,
                    delegation_policy: None,
                    resource_guards: Vec::new(),
                    subagent_lineage: None,
                    uplink: None,
                });
        if let Some(existing) = context.execution_id.as_deref()
            && existing != identity.execution_id.as_str()
        {
            return Err(Self::control_react_error(
                SubagentControlError::ExecutionIdentityMismatch {
                    expected: identity.execution_id.clone(),
                    actual: existing.to_string(),
                },
            ));
        }
        context.execution_id = Some(identity.execution_id.clone());
        if context.turn_id.is_none() && context.run_id.is_none() {
            context.turn_id = Some(identity.execution_id.clone());
        }
        let lineage = context
            .subagent_lineage
            .get_or_insert_with(Default::default);
        lineage.execution_id = Some(identity.execution_id.clone());
        lineage.task_id = Some(identity.task_id.clone());
        lineage.attempt = Some(identity.attempt);
        if lineage.run_id.is_none() {
            lineage.run_id = context.run_id.clone();
        }
        Ok(())
    }

    /// 把排队的指导信息以 `[queued_guidance]` 块的形式追加到任务文本末尾。
    fn append_queued_guidance(task: &mut String, guidance: &[String]) {
        if guidance.is_empty() {
            return;
        }
        task.push_str("\n\n[queued_guidance]");
        for instruction in guidance {
            task.push_str("\n- ");
            task.push_str(instruction);
        }
        task.push_str("\n[/queued_guidance]");
    }

    /// 将子 Agent 控制错误转换为 ReactError。
    fn control_react_error(error: SubagentControlError) -> ReactError {
        ReactError::Other(format!("Subagent control rejected: {error}"))
    }

    /// 确保 `runtime_context.execution_id` 已设置（缺失时生成 `agent_tool-{uuid}`）。
    fn ensure_background_execution_id(req: &mut DispatchRequest) -> String {
        let ctx = req
            .runtime_context
            .get_or_insert_with(|| echo_core::tools::ExternalRunContext {
                conversation_id: None,
                run_id: Some(format!("bg-{}", uuid::Uuid::new_v4().as_simple())),
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
            });
        if ctx
            .execution_id
            .as_ref()
            .map(|id| id.is_empty())
            .unwrap_or(true)
        {
            ctx.execution_id = Some(format!("agent_tool-{}", uuid::Uuid::new_v4()));
        }
        ctx.execution_id
            .clone()
            .unwrap_or_else(|| format!("agent_tool-{}", uuid::Uuid::new_v4()))
    }

    /**
    非阻塞派发：spawn 常规的 `dispatch` 路径并立即返回句柄。
    生命周期事件仍会在注册表事件总线上触发。

    沿用 [`Self::dispatch_teammate`] 的 `tokio::spawn` 模式，但通过
    [`Self::dispatch`] 复用 Sync/Fork/Team 路由，而不是仅限 teammate 的
    流式路径。
    */
    pub async fn dispatch_background(
        &self,
        mut req: DispatchRequest,
    ) -> Result<BackgroundSubagentHandle> {
        // 目标角色不存在时快速失败（与同步派发一致）。
        if self.registry.get(&req.agent_name).await.is_none() {
            return Err(ReactError::Other(format!(
                "Subagent '{}' not found",
                req.agent_name
            )));
        }

        let execution_id = Self::ensure_background_execution_id(&mut req);
        req.background = true;
        let agent_name = req.agent_name.clone();
        let agent_name_for_handle = agent_name.clone();
        let cancel = req.cancel.clone();
        let spawned = self.clone_for_spawn();

        let join_handle = tokio::spawn(async move {
            let result = spawned.dispatch(req).await;
            if let Err(error) = &result {
                warn!(agent = %agent_name, error = %error, "background subagent dispatch failed");
            }
            result
        });

        Ok(BackgroundSubagentHandle {
            execution_id,
            agent_name: agent_name_for_handle,
            cancel,
            join_handle: Arc::new(Mutex::new(Some(join_handle))),
        })
    }

    /// [`Self::dispatch_attempt`] 的非阻塞版本。
    pub async fn dispatch_background_attempt(
        &self,
        req: DispatchRequest,
        identity: SubagentAttemptIdentity,
    ) -> Result<BackgroundSubagentHandle> {
        if self.registry.get(&req.agent_name).await.is_none() {
            return Err(ReactError::Other(format!(
                "Subagent '{}' not found",
                req.agent_name
            )));
        }
        let (mut req, admission) = self.admit_attempt(req, identity)?;
        req.background = true;
        let execution_id = admission.binding.identity().execution_id.clone();
        let agent_name = req.agent_name.clone();
        let agent_name_for_handle = agent_name.clone();
        let cancel = req.cancel.clone();
        let spawned = self.clone_for_spawn();

        let join_handle = tokio::spawn(async move {
            let result = spawned.dispatch_admitted_attempt(req, admission).await;
            if let Err(error) = &result {
                warn!(agent = %agent_name, error = %error, "controlled background subagent dispatch failed");
            }
            result
        });

        Ok(BackgroundSubagentHandle {
            execution_id,
            agent_name: agent_name_for_handle,
            cancel,
            join_handle: Arc::new(Mutex::new(Some(join_handle))),
        })
    }

    /// 派发一个 teammate，返回可供异步轮询的句柄。
    pub async fn dispatch_teammate(&self, mut req: DispatchRequest) -> Result<TeammateHandle> {
        self.enrich_dispatch_context(&mut req);
        let event_publisher = self.event_publisher(&req)?;
        let runtime = req.runtime_context.as_ref();
        event_publisher.emit(SubagentEvent::DispatchStarted {
            parent: req.parent_agent.clone(),
            agent: req.agent_name.clone(),
            mode: ExecutionMode::Teammate,
            task: req.task.clone(),
            execution_id: runtime.and_then(|context| context.execution_id.clone()),
            run_id: runtime.and_then(|context| context.run_id.clone()),
            conversation_id: runtime.and_then(|context| context.conversation_id.clone()),
            message_id: runtime.and_then(|context| context.message_id.clone()),
            background: req.background,
        })?;
        match self
            .dispatch_teammate_with_control(req.clone(), None, event_publisher.clone(), true)
            .await
        {
            Ok(handle) => Ok(handle),
            Err(error) => {
                Self::emit_pre_execution_failure(&event_publisher, &req, &error)?;
                Err(error)
            }
        }
    }

    /// 在可选控制绑定下以 Teammate 模式派发：spawn 独立任务执行并返回句柄；`settle_in_handle` 为 true 时由句柄任务发出终止事件。
    async fn dispatch_teammate_with_control(
        &self,
        req: DispatchRequest,
        control: Option<SubagentAttemptBinding>,
        event_publisher: SubagentEventPublisher,
        settle_in_handle: bool,
    ) -> Result<TeammateHandle> {
        let registered =
            self.registry.get(&req.agent_name).await.ok_or_else(|| {
                ReactError::Other(format!("Subagent '{}' not found", req.agent_name))
            })?;

        let agent_arc = self.isolated_dispatch_agent(&req.agent_name).await?;

        let child_token = req.cancel.child_token();
        let handle_cancel = child_token.clone();
        let compiled = self.compile_invocation(
            &req,
            ExecutionMode::Teammate,
            registered.definition.inherit_history,
            agent_arc.as_ref(),
        );
        let agent_name = req.agent_name.clone();
        let parent_agent = req.parent_agent.clone();
        let timeout_secs = if registered.definition.timeout_secs > 0 {
            registered.definition.timeout_secs
        } else {
            self.config.default_timeout_secs
        };

        let handle_id = format!("tm_{}", uuid::Uuid::new_v4().as_simple());

        // 为事件负载提取稳定身份（会被移入 spawn）。
        let event_execution_id = req
            .runtime_context
            .as_ref()
            .and_then(|ctx| ctx.execution_id.clone());
        let event_run_id = req
            .runtime_context
            .as_ref()
            .and_then(|ctx| ctx.run_id.clone());
        let invocation = AgentInvocationContext {
            runtime: req.runtime_context.clone(),
            working_dir: agent_arc.working_dir(),
            ..AgentInvocationContext::default()
        };

        let join_handle = tokio::spawn(async move {
            let _permit = child_token.clone();
            let start = Instant::now();

            let mut result = if timeout_secs > 0 {
                // 在超时、取消与执行之间竞速
                tokio::select! {
                    biased; // 优先检查取消
                    _ = child_token.cancelled() => {
                        Err(ReactError::Agent(Box::new(AgentError::Interrupted)))
                    }
                    _ = tokio::time::sleep(Duration::from_secs(timeout_secs)) => {
                        child_token.cancel();
                        Err(ReactError::Agent(Box::new(AgentError::Timeout(format!(
                            "Teammate '{}' timed out after {}s",
                            agent_name, timeout_secs
                        )))))
                    }
                    r = Self::execute_agent_streaming(
                        agent_arc.clone(),
                        compiled.clone(),
                        child_token.clone(),
                        invocation.clone(),
                        &parent_agent,
                        &agent_name,
                        ExecutionMode::Teammate,
                        start,
                        event_execution_id.clone(),
                        event_run_id.clone(),
                        control.clone(),
                        event_publisher.clone(),
                    ) => r,
                }
            } else {
                // 在取消与执行之间竞速
                tokio::select! {
                    biased;
                    _ = child_token.cancelled() => {
                        Err(ReactError::Agent(Box::new(AgentError::Interrupted)))
                    }
                    r = Self::execute_agent_streaming(
                        agent_arc.clone(),
                        compiled.clone(),
                        child_token.clone(),
                        invocation.clone(),
                        &parent_agent,
                        &agent_name,
                        ExecutionMode::Teammate,
                        start,
                        event_execution_id.clone(),
                        event_run_id.clone(),
                        control.clone(),
                        event_publisher.clone(),
                    ) => r,
                }
            };
            if settle_in_handle {
                match &mut result {
                    Ok(value) => {
                        event_publisher.emit(Self::terminal_event_for_result(
                            &parent_agent,
                            &agent_name,
                            value,
                            event_execution_id.clone(),
                            event_run_id.clone(),
                        ))?;
                    }
                    Err(error) => {
                        let status = subagent_status_from_error(error);
                        let outcome = SubagentOutcome::terminal(
                            status,
                            error.to_string(),
                            vec![error.to_string()],
                        );
                        if status == SubagentStatus::Cancelled {
                            event_publisher.emit(SubagentEvent::DispatchCancelled {
                                parent: parent_agent.clone(),
                                agent: agent_name.clone(),
                                outcome,
                                execution_id: event_execution_id.clone(),
                                run_id: event_run_id.clone(),
                            })?;
                        } else {
                            event_publisher.emit(SubagentEvent::DispatchFailed {
                                parent: parent_agent.clone(),
                                agent: agent_name.clone(),
                                error: error.to_string(),
                                status,
                                outcome,
                                execution_id: event_execution_id.clone(),
                                run_id: event_run_id.clone(),
                            })?;
                        }
                    }
                }
            }
            result
        });

        Ok(TeammateHandle {
            id: handle_id,
            agent_name: req.agent_name.clone(),
            cancel: handle_cancel,
            join_handle,
        })
    }

    /**
    把 Team 意图编译为一张带修订号的图，并通过标准的
    `RuntimeTaskService` 执行。
    */
    async fn dispatch_team(
        &self,
        req: &DispatchRequest,
        event_publisher: SubagentEventPublisher,
    ) -> Result<SubagentResult> {
        let registered =
            self.registry.get(&req.agent_name).await.ok_or_else(|| {
                ReactError::Other(format!("Subagent '{}' not found", req.agent_name))
            })?;
        let spec = registered.definition.team.clone().ok_or_else(|| {
            ReactError::Other("Team mode requested but definition has no TeamSpec".to_string())
        })?;
        let delegation_policy = req.delegation_policy.child_policy().ok_or_else(|| {
            ReactError::Other(format!(
                "Delegation depth exceeded before Team '{}' (max {})",
                req.agent_name, req.delegation_policy.max_delegate_depth
            ))
        })?;
        /*
        Team 是图协调者而非一次模型调用。保持其目标原样；下面每个具体
        成员的派发都会基于该成员实际的工具、工作区与继承上下文恰好编译一次。
        */
        let team_objective = req.task.clone();
        let run_id = req
            .runtime_context
            .as_ref()
            .and_then(|context| context.run_id.clone())
            .unwrap_or_else(|| format!("team-{}", uuid::Uuid::new_v4().as_simple()));
        let parent_agent = req.agent_name.clone();
        let team_prompt_payload = req.prompt_payload.clone();
        let team_prompt_context = req.prompt_context.clone();
        let team_constraints = req.constraints.clone();
        let team_message = req.message.clone();
        let team_parent_event_id = event_publisher
            .dispatch_started_event_id()
            .map(|event_id| event_id.as_str().to_string());
        let inherited_history = req
            .parent_context
            .clone()
            .filter(|context| !context.messages.is_empty() || context.parent_goal.is_some());
        let control_scope_id = format!(
            "team-control:{}:{}",
            run_id,
            req.runtime_context
                .as_ref()
                .and_then(|context| context.execution_id.as_deref())
                .map(str::to_string)
                .unwrap_or_else(|| uuid::Uuid::new_v4().to_string())
        );
        let attempt_control = Arc::new(
            self.attempt_control_handle(control_scope_id)
                .map_err(Self::control_react_error)?,
        );
        let reserve_control = Arc::clone(&attempt_control);
        let reserve_attempt: super::team::TeamReserveAttemptFn = Arc::new(move |context| {
            reserve_control
                .reserve(&context)
                .map_err(|error| error.to_string())
        });
        let interrupt_control = Arc::clone(&attempt_control);
        let request_interrupt: super::team::TeamRequestInterruptFn =
            Arc::new(move |run_id, task_id, claim| {
                interrupt_control
                    .project_interrupt(&run_id, &task_id, &claim)
                    .map(Some)
            });
        let cleanup_control = Arc::clone(&attempt_control);
        let retire_attempt: super::team::TeamRetireAttemptFn =
            Arc::new(move |run_id, task, claim| {
                let cleanup_control = Arc::clone(&cleanup_control);
                Box::pin(async move { cleanup_control.retire(&run_id, &task, &claim) })
            });
        let reconcile_control = Arc::clone(&attempt_control);
        let reconcile_attempts: super::team::TeamReconcileAttemptFn =
            Arc::new(move |run_id, current_execution_ids| {
                reconcile_control
                    .reconcile(&run_id, &current_execution_ids)
                    .map_err(|error| error.to_string())
            });
        let spawned = self.clone_for_spawn();
        let dispatch_control = attempt_control;
        let dispatch: super::team::TeamDispatchFn = Arc::new(move |request| {
            let super::team::TeamDispatchRequest {
                member: agent_name,
                task,
                context,
            } = request;
            let executor = spawned.clone_for_spawn();
            let attempt_control = Arc::clone(&dispatch_control);
            let parent_agent = parent_agent.clone();
            let parent_context = inherited_history.clone();
            let prompt_payload = team_prompt_payload.clone();
            let message = team_message.clone();
            let mut prompt_context = team_prompt_context.clone();
            if let Some(context) = prompt_context.as_mut() {
                context.task_title = Some(task.clone());
            }
            let constraints = team_constraints.clone();
            let parent_event_id = team_parent_event_id.clone();
            Box::pin(async move {
                let member = executor.registry.get(&agent_name).await.ok_or_else(|| {
                    ReactError::Other(format!("Team Subagent '{agent_name}' not registered"))
                })?;
                if member.definition.execution_mode == ExecutionMode::Team {
                    return Err(ReactError::Other(format!(
                        "nested Team mode is not supported for '{agent_name}'"
                    )));
                }
                let mut runtime_context = context.runtime_context();
                runtime_context.subagent_lineage =
                    runtime_context.subagent_lineage.map(|mut lineage| {
                        lineage.agent_name = Some(agent_name.clone());
                        lineage.parent_agent = Some(parent_agent.clone());
                        lineage.parent_event_id = parent_event_id.clone();
                        lineage
                    });
                attempt_control
                    .dispatch(
                        DispatchRequest {
                            agent_name,
                            task,
                            mode_override: None,
                            cancel: context.cancellation_token().clone(),
                            parent_agent,
                            parent_context,
                            delegation_policy: NestedDelegationPolicy::default(),
                            runtime_context: Some(runtime_context),
                            message,
                            prompt_payload,
                            prompt_context,
                            constraints,
                            background: false,
                        },
                        context,
                    )
                    .await
            })
        });
        let start = Instant::now();
        let control = super::team::TeamDispatchControl {
            reserve_attempt,
            request_interrupt,
            retire_attempt,
            reconcile_attempts,
        };
        let dispatch_controller = super::team::closure_dispatch_controller(dispatch, control);
        let runtime_handle = super::team::runtime_handle_for_controller(
            run_id.clone(),
            dispatch_controller,
            echo_orchestration::tasks::RuntimeTaskServiceConfig {
                max_concurrent_subagents: spec.config.max_concurrent.max(1),
                delegation_policy,
                ..echo_orchestration::tasks::RuntimeTaskServiceConfig::default()
            },
        );
        let active = runtime_handle.start_execution();
        self.retain_team_runtime(runtime_handle.clone());
        let _retention = TeamRuntimeRetentionGuard {
            index: Arc::clone(&self.team_runtimes),
        };
        let result = runtime_handle
            .execute_started(&spec, &team_objective, req.cancel.child_token(), active)
            .await?;
        let tokens_used = result
            .usage
            .as_ref()
            .map(|usage| usize::try_from(usage.total_tokens).unwrap_or(usize::MAX));
        Ok(SubagentResult {
            agent_name: req.agent_name.clone(),
            output: result.output,
            outcome: SubagentOutcome {
                status: SubagentStatus::Completed,
                ..SubagentOutcome::default()
            },
            duration: start.elapsed(),
            iterations: 1,
            tokens_used,
            was_truncated: false,
            mode: ExecutionMode::Team,
            isolation_observed: ObservedIsolation::new("subagent"),
            llm_usage: result.usage,
        }
        .with_structured(
            req.runtime_context
                .as_ref()
                .and_then(|context| context.execution_id.as_deref()),
            std::env::current_dir().ok().as_deref(),
        ))
    }

    // ── 内部派发方法 ──────────────────────────────────────────

    /// 根据派发请求、执行模式与历史继承设置，编译出子 Agent 的调用消息。
    fn compile_invocation(
        &self,
        req: &DispatchRequest,
        mode: ExecutionMode,
        inherit_history: Option<usize>,
        agent: &dyn Agent,
    ) -> CompiledSubagentInvocation {
        let transfer_policy = if matches!(
            mode,
            ExecutionMode::Fork | ExecutionMode::Teammate | ExecutionMode::Team
        ) && req
            .parent_context
            .as_ref()
            .is_some_and(|context| !context.messages.is_empty())
        {
            ContextTransferPolicy::InheritStructured
        } else {
            ContextTransferPolicy::Fresh
        };
        let mut context = req.prompt_context.clone().unwrap_or_default();
        if context.user_goal.is_none() {
            context.user_goal = req
                .parent_context
                .as_ref()
                .and_then(|parent| parent.parent_goal.clone());
        }
        if context.workspace.is_none() {
            context.workspace = agent.working_dir();
        }
        if context.constraints.is_empty() {
            context.constraints = req.constraints.clone();
        }
        let allowed_tools = req
            .parent_context
            .as_ref()
            .and_then(|parent| parent.allowed_tools.as_deref())
            .filter(|allowed| !allowed.is_empty());
        let capability_override =
            allowed_tools.map(|allowed| effective_capabilities(agent, Some(allowed)));
        self.config
            .prompt_compiler
            .compile_invocation(&SubagentInvocation {
                agent_name: &req.agent_name,
                task: &req.task,
                mode,
                transfer_policy,
                history: req
                    .parent_context
                    .as_ref()
                    .map(|context| context.messages.as_slice())
                    .unwrap_or_default(),
                history_limit: inherit_history,
                current_message: req.message.as_ref(),
                context: &context,
                capability_override: capability_override.as_ref(),
                payload: req.prompt_payload.as_ref(),
            })
    }

    /// 以流式方式执行子 Agent，收集输出、用量与观测证据并转发事件，最终汇总为 SubagentResult。
    #[allow(clippy::too_many_arguments)]
    async fn execute_agent_streaming(
        agent: Arc<dyn Agent>,
        compiled: CompiledSubagentInvocation,
        cancel: CancellationToken,
        mut invocation: AgentInvocationContext,
        parent: &str,
        subagent: &str,
        mode: ExecutionMode,
        start: Instant,
        execution_id: Option<String>,
        run_id: Option<String>,
        control: Option<SubagentAttemptBinding>,
        event_publisher: SubagentEventPublisher,
    ) -> Result<SubagentResult> {
        let artifact_base_dir = invocation
            .working_dir
            .clone()
            .or_else(|| std::env::current_dir().ok());
        let control_turn_id = control.as_ref().map(|binding| {
            invocation
                .runtime
                .as_ref()
                .and_then(|runtime| runtime.turn_id.clone().or_else(|| runtime.run_id.clone()))
                .unwrap_or_else(|| binding.identity().execution_id.clone())
        });
        let mut messages = compiled.messages;
        let current_message = messages.pop().ok_or_else(|| {
            ReactError::Other("Subagent prompt compiler returned no invocation message".to_string())
        })?;
        invocation.history = (!messages.is_empty()).then_some(messages);
        let event_identity = echo_core::agent::EventIdentity::from_invocation(&invocation)?;
        /*
        按内容路由：纯文本调用走文本流式路径（只实现了 `execute_stream`
        的 Agent 仍可被派发）；真正的多模态输入才需要消息版本的覆盖实现。
        这与文档中 `DispatchRequest.message` 的约定一致，而不是假定每个
        Agent 都实现了两条路径。
        */
        let raw_stream = match current_message.content.as_text_ref() {
            Some(task) => {
                agent
                    .execute_stream_with_invocation_context(task, cancel, invocation)
                    .await?
            }
            None => {
                agent
                    .execute_stream_message_with_invocation_context(
                        current_message,
                        cancel,
                        invocation,
                    )
                    .await?
            }
        };
        let _steering_lease = if let Some(binding) = control {
            let turn_id = control_turn_id
                .ok_or_else(|| ReactError::Other("Subagent control turn id missing".to_string()))?;
            Some(
                binding
                    .attach(agent.clone(), turn_id)
                    .map_err(Self::control_react_error)?,
            )
        } else {
            None
        };
        let mut stream = echo_core::agent::envelope_event_stream(raw_stream, event_identity);
        let mut output = String::new();
        let mut in_thinking = false;
        let mut prompt_tokens: usize = 0;
        let mut completion_tokens: usize = 0;
        let mut usage_stats = super::usage::LlmUsageStats::default();
        let mut pending_tool_evidence = HashMap::<String, (String, serde_json::Value)>::new();
        let mut observed_evidence = Vec::new();
        let mut observed_artifacts = Vec::new();

        while let Some(event_result) = stream.next().await {
            let event = event_result?.payload;
            match event {
                AgentEvent::Token(content) => {
                    if in_thinking {
                        event_publisher.emit(SubagentEvent::DispatchThinkingDelta {
                            parent: parent.to_string(),
                            agent: subagent.to_string(),
                            content,
                            execution_id: execution_id.clone(),
                            run_id: run_id.clone(),
                        })?;
                    } else {
                        output.push_str(&content);
                        event_publisher.emit(SubagentEvent::DispatchTokenDelta {
                            parent: parent.to_string(),
                            agent: subagent.to_string(),
                            content,
                            execution_id: execution_id.clone(),
                            run_id: run_id.clone(),
                        })?;
                    }
                }
                AgentEvent::ThinkStart => {
                    in_thinking = true;
                    event_publisher.emit(SubagentEvent::DispatchThinkingStarted {
                        parent: parent.to_string(),
                        agent: subagent.to_string(),
                        execution_id: execution_id.clone(),
                        run_id: run_id.clone(),
                    })?;
                }
                AgentEvent::ThinkEnd {
                    prompt_tokens: pt,
                    completion_tokens: ct,
                } => {
                    in_thinking = false;
                    prompt_tokens = prompt_tokens.saturating_add(pt);
                    completion_tokens = completion_tokens.saturating_add(ct);
                    event_publisher.emit(SubagentEvent::DispatchThinkingEnded {
                        parent: parent.to_string(),
                        agent: subagent.to_string(),
                        prompt_tokens: pt,
                        completion_tokens: ct,
                        execution_id: execution_id.clone(),
                        run_id: run_id.clone(),
                    })?;
                }
                AgentEvent::LlmUsage {
                    model,
                    prompt_tokens: pt,
                    completion_tokens: ct,
                    total_tokens: tt,
                    cached_prompt_tokens: cpt,
                    cache_creation_prompt_tokens: ccpt,
                    usage_reported,
                } => {
                    usage_stats.record(&model, pt, ct, tt, cpt, ccpt, usage_reported);
                    event_publisher.emit(SubagentEvent::DispatchLlmUsage {
                        parent: parent.to_string(),
                        agent: subagent.to_string(),
                        model: model.clone(),
                        prompt_tokens: pt,
                        completion_tokens: ct,
                        total_tokens: tt,
                        cached_prompt_tokens: cpt,
                        cache_creation_prompt_tokens: ccpt,
                        usage_reported,
                        execution_id: execution_id.clone(),
                        run_id: run_id.clone(),
                    })?;
                }
                AgentEvent::ToolCall {
                    call_id,
                    invocation,
                } => {
                    pending_tool_evidence.insert(
                        call_id.clone(),
                        (invocation.name.clone(), invocation.args.clone()),
                    );
                    event_publisher.emit(SubagentEvent::DispatchToolStarted {
                        parent: parent.to_string(),
                        agent: subagent.to_string(),
                        call_id,
                        invocation,
                        execution_id: execution_id.clone(),
                        run_id: run_id.clone(),
                    })?;
                }
                AgentEvent::ToolResult {
                    call_id,
                    name,
                    result,
                } => {
                    let detail = result
                        .error
                        .as_deref()
                        .filter(|value| !value.is_empty())
                        .unwrap_or(&result.output);
                    let (subject, args) = pending_tool_evidence
                        .remove(&call_id)
                        .unwrap_or_else(|| (name.clone(), serde_json::Value::Null));
                    observed_evidence.push(SubagentEvidence {
                        kind: "tool_result".to_string(),
                        subject,
                        outcome: Some(if result.success {
                            "succeeded".to_string()
                        } else {
                            "failed".to_string()
                        }),
                        details: bounded_detail(detail),
                        source: SubagentEvidenceSource::Observed,
                        attributes: serde_json::json!({ "args": args }),
                    });
                    if let Some(artifact) = result.artifact.as_ref() {
                        observed_artifacts.push(SubagentArtifact {
                            path: artifact.path.to_string_lossy().to_string(),
                            kind: "tool_log".to_string(),
                            bytes: Some(artifact.artifact_bytes),
                            sha256: Some(artifact.sha256.clone()),
                            producer_execution_id: execution_id.clone(),
                            available: true,
                        });
                    }
                    event_publisher.emit(SubagentEvent::DispatchToolCompleted {
                        parent: parent.to_string(),
                        agent: subagent.to_string(),
                        call_id,
                        name,
                        result,
                        execution_id: execution_id.clone(),
                        run_id: run_id.clone(),
                    })?;
                }
                AgentEvent::FinalAnswer(answer) if !answer.is_empty() => {
                    output = answer;
                }
                AgentEvent::FinalAnswer(_) => {}
                AgentEvent::Cancelled => {
                    return Err(ReactError::Agent(Box::new(AgentError::Cancelled(format!(
                        "Subagent '{subagent}' cancelled"
                    )))));
                }
                AgentEvent::Error {
                    source,
                    message,
                    failure,
                } => {
                    return match failure.terminal_kind {
                        AgentTerminalKind::Cancelled => {
                            Err(ReactError::Agent(Box::new(AgentError::Cancelled(message))))
                        }
                        AgentTerminalKind::TimedOut => {
                            Err(ReactError::Agent(Box::new(AgentError::Timeout(message))))
                        }
                        AgentTerminalKind::PermissionDenied => Err(ReactError::Agent(Box::new(
                            AgentError::PermissionDenied(message),
                        ))),
                        AgentTerminalKind::Failed => {
                            Err(ReactError::Other(format!("{source}: {message}")))
                        }
                    };
                }
                _ => {}
            }
        }

        let tokens_used = Some(prompt_tokens.saturating_add(completion_tokens));
        let usage = if usage_stats.call_count > 0 {
            // 同时以真实用量统计为准，保持与 tokens_used 一致。
            Some(usage_stats)
        } else {
            None
        };

        let status = SubagentStatus::Completed;
        let mut result = SubagentResult {
            agent_name: subagent.to_string(),
            output,
            outcome: SubagentOutcome {
                status,
                ..SubagentOutcome::default()
            },
            duration: start.elapsed(),
            iterations: 1,
            tokens_used,
            was_truncated: false,
            mode,
            isolation_observed: ObservedIsolation::default(),
            llm_usage: usage,
        }
        .with_structured(execution_id.as_deref(), artifact_base_dir.as_deref());
        merge_observed_evidence(&mut result.outcome, observed_evidence, observed_artifacts);
        Ok(result)
    }

    /// 获取本次派发使用的 Agent 实例：优先由工厂创建全新实例，否则回退到注册表中缓存的实例。
    async fn isolated_dispatch_agent(&self, agent_name: &str) -> Result<Arc<dyn Agent>> {
        let agent = match self.registry.create_fresh_agent(agent_name).await? {
            Some(agent) => agent,
            None => self.registry.get_agent(agent_name).await.ok_or_else(|| {
                ReactError::Other(format!(
                    "Subagent '{}' not found or not instantiated",
                    agent_name
                ))
            })?,
        };
        Ok(agent)
    }

    /// Sync 模式：执行一次隔离的调用并返回其结果。
    async fn dispatch_sync(
        &self,
        req: &DispatchRequest,
        control: Option<SubagentAttemptBinding>,
        event_publisher: SubagentEventPublisher,
    ) -> Result<SubagentResult> {
        let agent_arc = self.isolated_dispatch_agent(&req.agent_name).await?;

        /*
        按子 Agent 覆盖（0 = 使用执行器默认值）。Sync 现在同样强制超时
        （以前会无限期阻塞父级）——三种模式统一由一个配置
        （AgentConfig.subagent_timeout_secs）控制。
        */
        let timeout_secs = match self.registry.get(&req.agent_name).await {
            Some(r) if r.definition.timeout_secs > 0 => r.definition.timeout_secs,
            _ => self.config.default_timeout_secs,
        };

        let start = Instant::now();
        let inherit_history = match self.registry.get(&req.agent_name).await {
            Some(r) => r.definition.inherit_history,
            None => None,
        };
        let compiled = self.compile_invocation(
            req,
            ExecutionMode::Sync,
            inherit_history,
            agent_arc.as_ref(),
        );
        let execution_cancel = req.cancel.child_token();
        let event_execution_id = req
            .runtime_context
            .as_ref()
            .and_then(|ctx| ctx.execution_id.clone());
        let event_run_id = req
            .runtime_context
            .as_ref()
            .and_then(|ctx| ctx.run_id.clone());
        let invocation = AgentInvocationContext {
            runtime: req.runtime_context.clone(),
            working_dir: agent_arc.working_dir(),
            ..AgentInvocationContext::default()
        };

        if timeout_secs > 0 {
            tokio::select! {
                biased;
                _ = execution_cancel.cancelled() => Err(ReactError::Agent(Box::new(
                    AgentError::Cancelled(format!("Sync subagent '{}' cancelled", req.agent_name))
                ))),
                r = tokio::time::timeout(
                    Duration::from_secs(timeout_secs),
                    Self::execute_agent_streaming(
                        agent_arc.clone(),
                        compiled.clone(),
                        execution_cancel.clone(),
                        invocation.clone(),
                        &req.parent_agent,
                        &req.agent_name,
                        ExecutionMode::Sync,
                        start,
                        event_execution_id.clone(),
                        event_run_id.clone(),
                        control.clone(),
                        event_publisher.clone(),
                    )
                ) => match r {
                    Ok(r) => r,
                    Err(_) => {
                        execution_cancel.cancel();
                        Err(ReactError::Agent(Box::new(AgentError::Timeout(format!(
                            "Sync subagent '{}' timed out after {}s",
                            req.agent_name, timeout_secs
                        )))))
                    }
                },
            }
        } else {
            Self::execute_agent_streaming(
                agent_arc,
                compiled,
                execution_cancel,
                invocation,
                &req.parent_agent,
                &req.agent_name,
                ExecutionMode::Sync,
                start,
                event_execution_id,
                event_run_id,
                control,
                event_publisher,
            )
            .await
        }
    }

    /// Fork 模式：获取信号量、spawn 任务，并带超时等待。
    async fn dispatch_fork(
        &self,
        req: &DispatchRequest,
        control: Option<SubagentAttemptBinding>,
        event_publisher: SubagentEventPublisher,
    ) -> Result<SubagentResult> {
        let registered =
            self.registry.get(&req.agent_name).await.ok_or_else(|| {
                ReactError::Other(format!("Subagent '{}' not found", req.agent_name))
            })?;
        let timeout_secs = if registered.definition.timeout_secs > 0 {
            registered.definition.timeout_secs
        } else {
            self.config.default_timeout_secs
        };
        let deadline = (timeout_secs > 0)
            .then(|| tokio::time::Instant::now() + Duration::from_secs(timeout_secs));
        /// Fork 模式的并发准入凭证。
        enum ForkAdmission {
            /// 本地信号量许可。
            Local(tokio::sync::OwnedSemaphorePermit),
            /// 共享准入控制器发放的租约。
            Shared(KeyedExecutionLease),
        }
        let shared_admission = self.shared_admission.read().await.clone();
        let permit = if let Some(admission) = shared_admission {
            let admission_key = req
                .runtime_context
                .as_ref()
                .and_then(|context| context.execution_id.as_deref())
                .map(str::to_string)
                .unwrap_or_else(|| uuid::Uuid::new_v4().as_simple().to_string());
            tokio::select! {
                biased;
                _ = req.cancel.cancelled() => {
                    return Err(ReactError::Agent(Box::new(AgentError::Cancelled(format!(
                        "Fork subagent '{}' cancelled while waiting for shared capacity",
                        req.agent_name
                    )))));
                }
                _ = async {
                    match deadline {
                        Some(deadline) => tokio::time::sleep_until(deadline).await,
                        None => std::future::pending().await,
                    }
                } => {
                    return Err(ReactError::Agent(Box::new(AgentError::Timeout(format!(
                        "Fork subagent '{}' timed out after {}s while waiting for shared capacity",
                        req.agent_name, timeout_secs
                    )))));
                }
                permit = admission.issue_wait(format!("subagent:{admission_key}")) => permit
                    .map(ForkAdmission::Shared)
                    .map_err(|error| ReactError::Other(format!("shared execution admission rejected Fork: {error}")))?,
            }
        } else {
            tokio::select! {
            biased;
            _ = req.cancel.cancelled() => {
                return Err(ReactError::Agent(Box::new(AgentError::Cancelled(format!(
                    "Fork subagent '{}' cancelled while waiting for capacity",
                    req.agent_name
                )))));
            }
            _ = async {
                match deadline {
                    Some(deadline) => tokio::time::sleep_until(deadline).await,
                    None => std::future::pending().await,
                }
            } => {
                return Err(ReactError::Agent(Box::new(AgentError::Timeout(format!(
                    "Fork subagent '{}' timed out after {}s while waiting for capacity",
                    req.agent_name, timeout_secs
                )))));
            }
            permit = self.semaphore.clone().acquire_owned() => permit
                .map(ForkAdmission::Local)
                .map_err(|error| ReactError::Other(format!("Semaphore error: {error}")))?,
            }
        };

        let agent_arc = tokio::select! {
            biased;
            _ = req.cancel.cancelled() => {
                return Err(ReactError::Agent(Box::new(AgentError::Cancelled(format!(
                    "Fork subagent '{}' cancelled during initialization",
                    req.agent_name
                )))));
            }
            _ = async {
                match deadline {
                    Some(deadline) => tokio::time::sleep_until(deadline).await,
                    None => std::future::pending().await,
                }
            } => {
                return Err(ReactError::Agent(Box::new(AgentError::Timeout(format!(
                    "Fork subagent '{}' timed out after {}s during initialization",
                    req.agent_name, timeout_secs
                )))));
            }
            agent = self.isolated_dispatch_agent(&req.agent_name) => agent?,
        };

        let agent_name = req.agent_name.clone();
        let parent_agent = req.parent_agent.clone();
        let cancel = req.cancel.clone();
        let message = req.message.clone();
        let prompt_compiler = self.config.prompt_compiler.clone();
        let prompt_task = req.task.clone();
        let prompt_parent_context = req.parent_context.clone();
        let prompt_payload = req.prompt_payload.clone();
        let prompt_context = req.prompt_context.clone();
        let prompt_constraints = req.constraints.clone();
        let prompt_inherit_history = registered.definition.inherit_history;
        /*
        跨 spawn 安全的值传递: 把外部 run context 带进 spawn 块。
        以调用值的形式携带 run 上下文。由工厂支持的 Fork 角色会拿到全新的
        Agent 实例；旧式预构建角色仍使用缓存实例，但不会修改共享的
        runtime/working_dir 字段。
        */
        let mut runtime_context = req.runtime_context.clone();
        if let Some(ctx) = runtime_context.as_mut() {
            ctx.delegation_policy = Some(req.delegation_policy);
        }
        // 为事件负载提取稳定身份（会被移入下方的 spawn）。
        let event_execution_id = runtime_context
            .as_ref()
            .and_then(|ctx| ctx.execution_id.clone());
        let event_run_id = runtime_context.as_ref().and_then(|ctx| ctx.run_id.clone());

        let isolation_kind = registered.definition.isolation.clone();
        let isolation_provider = isolation_kind
            .as_ref()
            .and_then(|_| self.config.isolation_provider.clone());
        if isolation_kind.is_some() && isolation_provider.is_none() {
            /*
            本地执行仍需此防护：悄悄丢弃已请求的隔离边界
            可能会破坏用户数据。
            */
            return Err(ReactError::Other(format!(
                "Subagent '{}' requests isolation but no IsolationProvider is configured; refusing to run without isolation",
                agent_name
            )));
        }
        let runtime_run_id = runtime_context
            .as_ref()
            .and_then(|ctx| ctx.run_id.as_deref())
            .unwrap_or("<none>");
        let has_trace_sink = runtime_context
            .as_ref()
            .is_some_and(|ctx| ctx.trace_sink.is_some());
        info!(
            parent = %parent_agent,
            subagent = %agent_name,
            runtime_run_id = %runtime_run_id,
            has_runtime_context = runtime_context.is_some(),
            has_trace_sink,
            timeout_secs,
            isolation = isolation_kind.as_deref().unwrap_or("context"),
            "subagent_fork_start"
        );
        /*
        优先使用调用方提供的稳定隔离身份，使同一逻辑任务的多次重试能复用
        同一个 worktree/workspace，同时为事件与审计记录保留各自不同的
        execution id。
        */
        let isolation_identity = runtime_context
            .as_ref()
            .and_then(|context| {
                context
                    .isolation_id
                    .as_deref()
                    .or(context.execution_id.as_deref())
                    .or(context.run_id.as_deref())
                    .or(context.turn_id.as_deref())
            })
            .map(str::to_string)
            .unwrap_or_else(|| uuid::Uuid::new_v4().as_simple().to_string());
        let isolation_label = format!("{agent_name}-{isolation_identity}");
        let execution_cancel = cancel.child_token();
        let invocation_allowed_tools = req
            .parent_context
            .as_ref()
            .and_then(|context| context.allowed_tools.clone())
            .filter(|allowed| !allowed.is_empty());

        let result = tokio::spawn(async move {
            let _permit = permit;
            match &_permit {
                ForkAdmission::Local(permit) => {
                    let _ = permit;
                }
                ForkAdmission::Shared(lease) => {
                    let _ = lease;
                }
            }
            let start = Instant::now();

            // 检查取消
            if execution_cancel.is_cancelled() {
                let mut result = SubagentResult::cancelled(
                    agent_name.clone(),
                    "Cancelled before execution",
                    ExecutionMode::Fork,
                );
                result.duration = start.elapsed();
                return Ok(result);
            }

            let mut isolation_handle: Option<super::isolation::IsolationHandle> = None;
            if let (Some(provider), Some(kind)) = (&isolation_provider, &isolation_kind) {
                let request = super::isolation::IsolationRequest {
                    kind: kind.clone(),
                    label: isolation_label,
                };
                match provider.isolate(&request) {
                    Ok(handle) => {
                        isolation_handle = Some(handle);
                    }
                    Err(error) => {
                        return Err(ReactError::Other(format!(
                            "Isolation '{}' for Fork subagent '{agent_name}' failed: {error}",
                            request.kind
                        )));
                    }
                }
            }
            let isolation_observed = isolation_handle
                .as_ref()
                .map(|handle| handle.observed.clone())
                .unwrap_or_else(|| ObservedIsolation::new("context"));
            let mut context = prompt_context.unwrap_or_default();
            if context.user_goal.is_none() {
                context.user_goal = prompt_parent_context
                    .as_ref()
                    .and_then(|parent| parent.parent_goal.clone());
            }
            if context.workspace.is_none() {
                context.workspace = isolation_handle
                    .as_ref()
                    .map(|handle| handle.path.clone())
                    .or_else(|| agent_arc.working_dir());
            }
            if context.constraints.is_empty() {
                context.constraints = prompt_constraints;
            }
            let capability_override = invocation_allowed_tools
                .as_deref()
                .map(|allowed| effective_capabilities(agent_arc.as_ref(), Some(allowed)));
            let compiled = prompt_compiler.compile_invocation(&SubagentInvocation {
                agent_name: &agent_name,
                task: &prompt_task,
                mode: ExecutionMode::Fork,
                transfer_policy: if prompt_parent_context
                    .as_ref()
                    .is_some_and(|context| !context.messages.is_empty())
                {
                    ContextTransferPolicy::InheritStructured
                } else {
                    ContextTransferPolicy::Fresh
                },
                history: prompt_parent_context
                    .as_ref()
                    .map(|context| context.messages.as_slice())
                    .unwrap_or_default(),
                history_limit: prompt_inherit_history,
                current_message: message.as_ref(),
                context: &context,
                capability_override: capability_override.as_ref(),
                payload: prompt_payload.as_ref(),
            });
            let disabled_tools = invocation_disabled_tools(
                agent_arc.tool_names(),
                invocation_allowed_tools.as_deref(),
            );
            let effective_working_dir = isolation_handle
                .as_ref()
                .map(|handle| handle.path.clone())
                .or_else(|| agent_arc.working_dir());
            let invocation = AgentInvocationContext {
                runtime: runtime_context.clone(),
                runtime_state_id: None,
                transcript_generation_id: None,
                working_dir: effective_working_dir,
                cancel: None,
                disabled_tools,
                visible_tools: None,
                run_budget: None,
                history: None,
                resource_guards: Vec::new(),
                input_lifecycle: None,
            };
            event_publisher.emit(SubagentEvent::DispatchIsolationObserved {
                parent: parent_agent.clone(),
                agent: agent_name.clone(),
                isolation: isolation_observed.clone(),
                execution_id: event_execution_id.clone(),
                run_id: event_run_id.clone(),
            })?;
            let mut result = if let Some(deadline) = deadline {
                tokio::select! {
                    biased;
                    _ = execution_cancel.cancelled() => Err(ReactError::Agent(Box::new(
                        AgentError::Cancelled(format!("Fork subagent '{}' cancelled", agent_name))
                    ))),
                    r = tokio::time::timeout_at(
                        deadline,
                        Self::execute_agent_streaming(
                            agent_arc.clone(),
                            compiled.clone(),
                            execution_cancel.clone(),
                            invocation.clone(),
                            &parent_agent,
                            &agent_name,
                            ExecutionMode::Fork,
                            start,
                            event_execution_id.clone(),
                            event_run_id.clone(),
                            control.clone(),
                            event_publisher.clone(),
                        )
                    ) => {
                        match r {
                            Ok(r) => r,
                            Err(_) => {
                                execution_cancel.cancel();
                                Err(ReactError::Agent(Box::new(AgentError::Timeout(format!(
                                    "Fork subagent '{}' timed out after {}s",
                                    agent_name, timeout_secs
                                )))))
                            }
                        }
                    }
                }
            } else {
                tokio::select! {
                    biased;
                    _ = execution_cancel.cancelled() => Err(ReactError::Agent(Box::new(
                        AgentError::Cancelled(format!("Fork subagent '{}' cancelled", agent_name))
                    ))),
                    r = Self::execute_agent_streaming(
                        agent_arc.clone(),
                        compiled.clone(),
                        execution_cancel.clone(),
                        invocation.clone(),
                        &parent_agent,
                        &agent_name,
                        ExecutionMode::Fork,
                        start,
                        event_execution_id.clone(),
                        event_run_id.clone(),
                        control.clone(),
                        event_publisher.clone(),
                    ) => r,
                }
            };
            if let Ok(subagent_result) = &mut result {
                subagent_result.isolation_observed = isolation_observed;
            }

            if let Some(handle) = isolation_handle {
                match (handle.finalize)() {
                    Ok(finalized) => {
                        if let Ok(mut r) = result {
                            if !finalized.summary.trim().is_empty() {
                                r.output = format!(
                                    "{}\n\n--- isolation outcome ---\n{}",
                                    r.output, finalized.summary
                                );
                            }
                            merge_observed_evidence(
                                &mut r.outcome,
                                finalized.evidence,
                                finalized.artifacts,
                            );
                            return Ok(r);
                        }
                    }
                    Err(error) => {
                        tracing::warn!(
                            subagent = %agent_name,
                            error = %error,
                            "Subagent isolation finalization failed; result preserved"
                        );
                    }
                }
            }

            result
        })
        .await
        .map_err(|e| ReactError::Other(format!("Fork task join error: {}", e)))??;

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::subagent::SubagentDefinition;
    use crate::agent::subagent::prompt::{
        CompiledSubagentSystemPrompt, PromptDiagnostics, SubagentSystemPromptInput,
    };
    use crate::agent::subagent::registry::FnAgentFactory;
    use crate::testing::{FailingMockAgent, MockAgent};
    use echo_core::agent::{ToolInvocation, ToolInvocationRewrite};
    use echo_core::tools::{InvocationResourceGuard, ToolResult, ToolResultKind};
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    /**
    在被放行前阻塞事件流、并记录 steer 输入的 Agent——
    用于针对活跃尝试测试默认的上行 sink。
    */
    struct SteerableBlockingAgent {
        /// 用于放行阻塞流的通知器。
        release: Arc<tokio::sync::Notify>,
        /// 记录收到的 steer 输入文本。
        steered: Arc<Mutex<Vec<String>>>,
    }

    impl SteerableBlockingAgent {
        /// 构造一个收到放行通知后才产出最终答案的事件流。
        fn blocking_stream<'a>(
            release: Arc<tokio::sync::Notify>,
        ) -> futures::future::BoxFuture<
            'a,
            Result<futures::stream::BoxStream<'a, Result<AgentEvent>>>,
        > {
            /*
            立即完成创建流的 future：控制平面只有在流存在之后才会挂接
            （实时 steer 也才可能生效），因此等待放行必须放在流的首次
            poll 中，而不是这个 future 里。
            */
            Box::pin(async move {
                Ok(Box::pin(futures::stream::once(async move {
                    release.notified().await;
                    Ok(AgentEvent::FinalAnswer("released".to_string()))
                }))
                    as futures::stream::BoxStream<'a, Result<AgentEvent>>)
            })
        }
    }

    impl Agent for SteerableBlockingAgent {
        /// 返回测试 Agent 的名称。
        fn name(&self) -> &str {
            "steerable-blocking"
        }

        /// 返回测试用的模型名称。
        fn model_name(&self) -> &str {
            "test"
        }

        /// 返回空的系统提示词。
        fn system_prompt(&self) -> &str {
            ""
        }

        /// 直接返回固定结果。
        fn execute<'a>(&'a self, _task: &'a str) -> futures::future::BoxFuture<'a, Result<String>> {
            Box::pin(async { Ok("done".to_string()) })
        }

        /// 返回阻塞直至放行的事件流。
        fn execute_stream<'a>(
            &'a self,
            _task: &'a str,
        ) -> futures::future::BoxFuture<
            'a,
            Result<futures::stream::BoxStream<'a, Result<AgentEvent>>>,
        > {
            Self::blocking_stream(self.release.clone())
        }

        /// 多模态入口同样返回阻塞直至放行的事件流。
        fn execute_stream_message_with_cancel<'a>(
            &'a self,
            _message: Message,
            _cancel: CancellationToken,
        ) -> futures::future::BoxFuture<
            'a,
            Result<futures::stream::BoxStream<'a, Result<AgentEvent>>>,
        > {
            Self::blocking_stream(self.release.clone())
        }

        /// 记录 steer 输入文本并返回固定的轮次 id。
        fn steer_input(
            &self,
            _expected_turn_id: Option<&str>,
            message: Message,
        ) -> std::result::Result<String, echo_core::agent::AgentSteerError> {
            let text = message
                .content
                .as_text()
                .filter(|text| !text.trim().is_empty())
                .ok_or(echo_core::agent::AgentSteerError::EmptyInput)?
                .to_string();
            self.steered
                .lock()
                .map_err(|_| echo_core::agent::AgentSteerError::StateUnavailable)?
                .push(text.clone());
            Ok("turn".to_string())
        }

        /// 记录 steer 输入并返回一个处于已排空状态的回执。
        fn steer_input_tracked(
            &self,
            expected_turn_id: Option<&str>,
            message: Message,
        ) -> std::result::Result<
            echo_core::agent::AgentSteerReceipt,
            echo_core::agent::AgentSteerError,
        > {
            let turn_id = self.steer_input(expected_turn_id, message)?;
            let (_tx, rx) = tokio::sync::watch::channel(echo_core::agent::AgentSteerState::Drained);
            Ok(echo_core::agent::AgentSteerReceipt::new(
                uuid::Uuid::new_v4().to_string(),
                turn_id,
                rx,
            ))
        }
    }

    /// 验证 enrich_dispatch_context 会补全缺失的血缘信息，并保留调用方已设置的值。
    #[test]
    fn enrich_dispatch_context_fills_missing_lineage_and_keeps_caller_values() {
        let registry = Arc::new(SubagentRegistry::new());
        let executor = SubagentExecutor::new(registry, SubagentExecutorConfig::default());

        // 缺少血缘信息 → 从请求推导基础字段。
        let mut req = DispatchRequest {
            agent_name: "explorer".to_string(),
            task: "probe".to_string(),
            mode_override: None,
            cancel: CancellationToken::new(),
            parent_agent: "primary".to_string(),
            parent_context: None,
            delegation_policy: DispatchRequest::policy_from_depth(0),
            runtime_context: Some(echo_core::tools::ExternalRunContext {
                run_id: Some("run-1".to_string()),
                execution_id: Some("exec-1".to_string()),
                ..Default::default()
            }),
            message: None,
            prompt_payload: None,
            prompt_context: None,
            constraints: Vec::new(),
            background: false,
        };
        executor.enrich_dispatch_context(&mut req);
        let lineage = req
            .runtime_context
            .as_ref()
            .and_then(|context| context.subagent_lineage.clone())
            .unwrap_or_default();
        assert_eq!(lineage.agent_name.as_deref(), Some("explorer"));
        assert_eq!(lineage.parent_agent.as_deref(), Some("primary"));
        assert_eq!(lineage.agent_path.as_deref(), Some("root/explorer"));
        assert_eq!(lineage.execution_id.as_deref(), Some("exec-1"));
        assert_eq!(lineage.run_id.as_deref(), Some("run-1"));
        assert!(
            req.runtime_context
                .as_ref()
                .is_some_and(|c| c.uplink.is_some())
        );

        // 调用方填写的血缘信息与应用层上行 sink 均被保留。
        let mut stamped = req.clone();
        let app_sink: echo_core::tools::SubagentUplinkFn = Arc::new(|_message| {
            Box::pin(async {
                echo_core::tools::SubagentUplinkReceipt {
                    accepted: true,
                    status: "app".to_string(),
                    detail: String::new(),
                }
            })
        });
        if let Some(context) = stamped.runtime_context.as_mut() {
            context.subagent_lineage = Some(echo_core::tools::SubagentLineage {
                agent_path: Some("root/implementer/sub-probe".to_string()),
                parent_execution_id: Some("parent-exec".to_string()),
                ..Default::default()
            });
            context.uplink = Some(app_sink.clone());
        }
        executor.enrich_dispatch_context(&mut stamped);
        let (preserved, kept_sink) = stamped
            .runtime_context
            .as_ref()
            .and_then(|context| {
                context
                    .subagent_lineage
                    .clone()
                    .map(|lineage| (lineage, context.uplink.clone()))
            })
            .unwrap_or_default();
        assert_eq!(
            preserved.agent_path.as_deref(),
            Some("root/implementer/sub-probe")
        );
        assert_eq!(
            preserved.parent_execution_id.as_deref(),
            Some("parent-exec")
        );
        assert!(kept_sink.is_some_and(|sink| Arc::ptr_eq(&sink, &app_sink)));
    }

    /// 验证默认上行 sink 能投递到活跃尝试，并报告找不到的目标。
    #[tokio::test]
    async fn default_uplink_sink_delivers_to_active_attempt_and_reports_missing() -> Result<()> {
        let (registry, executor) = make_executor().await;
        let mut envelopes = registry.event_bus().subscribe_envelopes();
        let release = Arc::new(tokio::sync::Notify::new());
        let steered = Arc::new(Mutex::new(Vec::new()));
        registry
            .register(
                SubagentDefinition::new("target", "Uplink target"),
                Box::new(SteerableBlockingAgent {
                    release: release.clone(),
                    steered: steered.clone(),
                }),
            )
            .await;

        let identity = SubagentAttemptIdentity::new("task-9", "exec-9", 1)
            .map_err(|error| ReactError::Other(error.to_string()))?;
        let dispatch = tokio::spawn(async move {
            executor
                .dispatch_attempt(
                    DispatchRequest {
                        agent_name: "target".to_string(),
                        task: "block until released".to_string(),
                        mode_override: Some(ExecutionMode::Sync),
                        cancel: CancellationToken::new(),
                        parent_agent: "primary".to_string(),
                        parent_context: None,
                        delegation_policy: DispatchRequest::policy_from_depth(0),
                        runtime_context: None,
                        message: None,
                        prompt_payload: None,
                        prompt_context: None,
                        constraints: Vec::new(),
                        background: false,
                    },
                    identity,
                )
                .await
        });

        // 等待该尝试在共享控制平面中变为可寻址。
        let mut became_active = false;
        for _ in 0..100 {
            if registry
                .control_registry()
                .active_snapshot(16)
                .iter()
                .any(|summary| summary.execution_id == "exec-9")
            {
                became_active = true;
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        assert!(became_active, "attempt never became active");

        let sink = default_uplink_sink(Arc::clone(&registry));
        let sender_lineage = echo_core::tools::SubagentLineage {
            agent_name: Some("target".to_string()),
            execution_id: Some("exec-9".to_string()),
            parent_agent: Some("primary".to_string()),
            parent_execution_id: Some("exec-9".to_string()),
            ..Default::default()
        };

        // 父级方向：父尝试处于活跃状态 → 仅以排队方式 steer。
        let parent_receipt = sink(echo_core::tools::SubagentUplinkMessage {
            from: sender_lineage.clone(),
            target: echo_core::tools::SubagentUplinkTarget::Parent {
                kind: echo_core::tools::SubagentUplinkKind::Escalate,
                text: "plan assumption is wrong?".to_string(),
            },
        })
        .await;
        assert!(parent_receipt.accepted);
        assert_eq!(parent_receipt.status, "parent_steered");

        // 兄弟方向：活跃尝试 → 已投递；未知目标 → 报告缺失。
        let sibling_receipt = sink(echo_core::tools::SubagentUplinkMessage {
            from: sender_lineage,
            target: echo_core::tools::SubagentUplinkTarget::Sibling {
                to: echo_core::tools::SubagentPeerAddress::ByExecutionId("exec-9".to_string()),
                text: "dependency output is ready".to_string(),
            },
        })
        .await;
        assert!(sibling_receipt.accepted);
        assert_eq!(sibling_receipt.status, "delivered_to_sibling");

        let missing_receipt =
            default_uplink_sink(Arc::clone(&registry))(echo_core::tools::SubagentUplinkMessage {
                from: echo_core::tools::SubagentLineage::default(),
                target: echo_core::tools::SubagentUplinkTarget::Sibling {
                    to: echo_core::tools::SubagentPeerAddress::ByExecutionId(
                        "exec-missing".to_string(),
                    ),
                    text: "anyone?".to_string(),
                },
            })
            .await;
        assert!(!missing_receipt.accepted);
        assert_eq!(missing_receipt.status, "sibling_not_active");

        release.notify_one();
        let settled = dispatch
            .await
            .map_err(|error| ReactError::Other(error.to_string()))??;
        assert_eq!(settled.outcome.status, SubagentStatus::Completed);
        let steered_now = steered
            .lock()
            .map(|messages| messages.clone())
            .unwrap_or_default();
        assert_eq!(steered_now.len(), 2);
        let mut observed = Vec::new();
        for _ in 0..4 {
            observed.push(
                envelopes
                    .recv()
                    .await
                    .map_err(|error| ReactError::Other(error.to_string()))?,
            );
        }
        assert_eq!(
            observed
                .iter()
                .map(|envelope| envelope.sequence)
                .collect::<Vec<_>>(),
            vec![1, 2, 3, 4]
        );
        assert!(matches!(
            observed.get(1).map(|envelope| &envelope.payload.event),
            Some(SubagentEvent::UplinkReceived { .. })
        ));
        assert!(matches!(
            observed.get(2).map(|envelope| &envelope.payload.event),
            Some(SubagentEvent::UplinkReceived { .. })
        ));
        Ok(())
    }

    /// 测试用提示词编译器：给调用任务加上 `compiled:` 前缀。
    struct PrefixPromptCompiler;

    impl SubagentPromptCompiler for PrefixPromptCompiler {
        /// 直接使用角色提示词作为系统提示词。
        fn compile_system(
            &self,
            input: &SubagentSystemPromptInput<'_>,
        ) -> CompiledSubagentSystemPrompt {
            CompiledSubagentSystemPrompt {
                system_prompt: input.role_prompt.to_string(),
                diagnostics: PromptDiagnostics::default(),
            }
        }

        /// 生成一条带 `compiled:` 前缀的用户消息。
        fn compile_invocation(&self, input: &SubagentInvocation<'_>) -> CompiledSubagentInvocation {
            CompiledSubagentInvocation {
                messages: vec![Message::user(format!("compiled:{}", input.task))],
                diagnostics: PromptDiagnostics::default(),
            }
        }
    }

    /// 验证调用级工具白名单会隐藏所有未列出的工具。
    #[test]
    fn invocation_allowlist_hides_every_unlisted_tool() {
        let tools = vec![
            "read_file".to_string(),
            "write_file".to_string(),
            "shell".to_string(),
        ];
        let allowed = vec!["read_file".to_string()];
        let disabled = invocation_disabled_tools(tools.clone(), Some(&allowed)).unwrap_or_default();
        assert_eq!(
            disabled,
            HashSet::from(["write_file".to_string(), "shell".to_string()])
        );
        assert!(invocation_disabled_tools(tools.clone(), None).is_none());
        assert!(invocation_disabled_tools(tools, Some(&[])).is_none());
    }

    /// 测试用钩子：失败时总是委派给 `recovery` Agent。
    struct DelegateFailedDispatch;

    /// 感知取消的测试 Agent：其流在取消后发出 Cancelled 事件。
    struct CancellationAwareStreamAgent {
        /// 取消被观察到时发出通知。
        cancellation_seen: Arc<tokio::sync::Notify>,
    }

    /// 发出带改写信息的丰富工具事件的测试 Agent。
    struct RichToolEventAgent;

    impl Agent for RichToolEventAgent {
        /// 返回测试 Agent 的名称。
        fn name(&self) -> &str {
            "rich-tool-events"
        }

        /// 返回测试用的模型名称。
        fn model_name(&self) -> &str {
            "test-model"
        }

        /// 返回空的系统提示词。
        fn system_prompt(&self) -> &str {
            ""
        }

        /// 直接返回固定结果。
        fn execute<'a>(&'a self, _task: &'a str) -> futures::future::BoxFuture<'a, Result<String>> {
            Box::pin(async { Ok("done".to_string()) })
        }

        /// 返回包含工具调用、工具结果与最终答案的固定事件流。
        fn execute_stream<'a>(
            &'a self,
            _task: &'a str,
        ) -> futures::future::BoxFuture<
            'a,
            Result<futures::stream::BoxStream<'a, Result<AgentEvent>>>,
        > {
            Box::pin(async {
                let invocation = ToolInvocation {
                    requested_name: "requested_tool".to_string(),
                    requested_args: serde_json::json!({"value": "requested"}),
                    name: "effective_tool".to_string(),
                    args: serde_json::json!({"value": "effective"}),
                    rewrites: vec![ToolInvocationRewrite::Approval],
                };
                let result = ToolResult {
                    kind: ToolResultKind::Json,
                    success: true,
                    output: "{\"ok\":true}".to_string(),
                    error: None,
                    failure: None,
                    data: Some(serde_json::json!({"ok": true})),
                    truncated: true,
                    mime_type: Some("application/json".to_string()),
                    metadata: HashMap::from([("source".to_string(), "fixture".to_string())]),
                    artifact: None,
                    effects: Vec::new(),
                    model_content: Vec::new(),
                };
                let events = vec![
                    Ok(AgentEvent::ToolCall {
                        call_id: "call-rich".to_string(),
                        invocation,
                    }),
                    Ok(AgentEvent::ToolResult {
                        call_id: "call-rich".to_string(),
                        name: "effective_tool".to_string(),
                        result,
                    }),
                    Ok(AgentEvent::FinalAnswer("done".to_string())),
                ];
                Ok(Box::pin(futures::stream::iter(events))
                    as futures::stream::BoxStream<'a, Result<AgentEvent>>)
            })
        }
    }

    impl Agent for CancellationAwareStreamAgent {
        /// 返回测试 Agent 的名称。
        fn name(&self) -> &str {
            "cancellation-aware"
        }

        /// 返回测试用的模型名称。
        fn model_name(&self) -> &str {
            "test-model"
        }

        /// 返回空的系统提示词。
        fn system_prompt(&self) -> &str {
            ""
        }

        /// 永不完成的执行。
        fn execute<'a>(&'a self, _task: &'a str) -> futures::future::BoxFuture<'a, Result<String>> {
            Box::pin(async move { std::future::pending::<Result<String>>().await })
        }

        /// 返回一个永不产出事件的流。
        fn execute_stream<'a>(
            &'a self,
            _task: &'a str,
        ) -> futures::future::BoxFuture<
            'a,
            Result<futures::stream::BoxStream<'a, Result<AgentEvent>>>,
        > {
            Box::pin(async move {
                let (_sender, receiver) = tokio::sync::mpsc::channel(1);
                Ok(
                    Box::pin(tokio_stream::wrappers::ReceiverStream::new(receiver))
                        as futures::stream::BoxStream<'a, Result<AgentEvent>>,
                )
            })
        }

        /// 返回的流会在取消令牌触发后发出通知并产出 Cancelled 事件。
        fn execute_stream_with_invocation_context<'a>(
            &'a self,
            _task: &'a str,
            cancel: CancellationToken,
            _invocation: AgentInvocationContext,
        ) -> futures::future::BoxFuture<
            'a,
            Result<futures::stream::BoxStream<'a, Result<AgentEvent>>>,
        > {
            let cancellation_seen = Arc::clone(&self.cancellation_seen);
            Box::pin(async move {
                let (sender, receiver) = tokio::sync::mpsc::channel(1);
                tokio::spawn(async move {
                    cancel.cancelled().await;
                    cancellation_seen.notify_one();
                    let _ = sender.send(Ok(AgentEvent::Cancelled)).await;
                });
                Ok(
                    Box::pin(tokio_stream::wrappers::ReceiverStream::new(receiver))
                        as futures::stream::BoxStream<'a, Result<AgentEvent>>,
                )
            })
        }
    }

    #[async_trait::async_trait]
    impl super::super::hooks::SubagentHooks for DelegateFailedDispatch {
        /// 失败时返回委派到 `recovery` 的决策。
        async fn on_failure(
            &self,
            _ctx: &super::super::hooks::SubagentHookContext,
            _error: &str,
        ) -> super::super::hooks::SubagentRetryDecision {
            super::super::hooks::SubagentRetryDecision::Delegate {
                alternative_agent: "recovery".to_string(),
            }
        }
    }

    /// 创建使用默认配置的注册表与执行器。
    async fn make_executor() -> (Arc<SubagentRegistry>, SubagentExecutor) {
        let registry = Arc::new(SubagentRegistry::new());
        let executor = SubagentExecutor::new(registry.clone(), SubagentExecutorConfig::default());
        (registry, executor)
    }

    /// 资源守卫：被丢弃时递增计数器，用于验证守卫的生命周期。
    struct DispatchGuardDrop(Arc<AtomicUsize>);

    impl Drop for DispatchGuardDrop {
        /// 被丢弃时计数加一。
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    /// 永远挂起的流，被丢弃时设置标记。
    struct PendingDropStream {
        /// 流被丢弃时置为 true。
        dropped: Arc<AtomicBool>,
    }

    impl futures::Stream for PendingDropStream {
        /// 流产出的事件类型。
        type Item = Result<AgentEvent>;

        /// 始终返回 Pending。
        fn poll_next(
            self: std::pin::Pin<&mut Self>,
            _context: &mut std::task::Context<'_>,
        ) -> std::task::Poll<Option<Self::Item>> {
            std::task::Poll::Pending
        }
    }

    impl Drop for PendingDropStream {
        /// 被丢弃时设置 dropped 标记。
        fn drop(&mut self) {
            self.dropped.store(true, Ordering::SeqCst);
        }
    }

    /// 不响应取消的 Team 成员测试 Agent。
    struct NonCooperativeTeamAgent {
        /// 其事件流被丢弃时置为 true。
        dropped: Arc<AtomicBool>,
    }

    impl Agent for NonCooperativeTeamAgent {
        /// 返回测试 Agent 的名称。
        fn name(&self) -> &str {
            "non-cooperative-team-member"
        }

        /// 返回测试用的模型名称。
        fn model_name(&self) -> &str {
            "test"
        }

        /// 返回空的系统提示词。
        fn system_prompt(&self) -> &str {
            ""
        }

        /// 永不完成的执行。
        fn execute<'a>(&'a self, _task: &'a str) -> futures::future::BoxFuture<'a, Result<String>> {
            Box::pin(std::future::pending())
        }

        /// 返回一个永远挂起、被丢弃时打标记的流。
        fn execute_stream<'a>(
            &'a self,
            _task: &'a str,
        ) -> futures::future::BoxFuture<
            'a,
            Result<futures::stream::BoxStream<'a, Result<AgentEvent>>>,
        > {
            let dropped = Arc::clone(&self.dropped);
            Box::pin(async move {
                Ok(Box::pin(PendingDropStream { dropped })
                    as futures::stream::BoxStream<'a, Result<AgentEvent>>)
            })
        }

        /// 忽略取消令牌，直接委托给 execute_stream。
        fn execute_stream_with_invocation_context<'a>(
            &'a self,
            task: &'a str,
            _cancel: CancellationToken,
            _invocation: AgentInvocationContext,
        ) -> futures::future::BoxFuture<
            'a,
            Result<futures::stream::BoxStream<'a, Result<AgentEvent>>>,
        > {
            self.execute_stream(task)
        }
    }

    /// 断言给定执行模式在派发结算前一直持有运行时资源守卫。
    async fn assert_mode_retains_runtime_guards(
        mode: ExecutionMode,
    ) -> std::result::Result<(), String> {
        let (registry, executor) = make_executor().await;
        let agent = MockAgent::new("guarded").with_response("done");
        registry
            .register(
                SubagentDefinition {
                    execution_mode: mode.clone(),
                    ..SubagentDefinition::new("guarded", "Guarded subagent")
                },
                Box::new(agent.clone()),
            )
            .await;
        let drops = Arc::new(AtomicUsize::new(0));
        let result = executor
            .dispatch(DispatchRequest {
                agent_name: "guarded".to_string(),
                task: "retain guard".to_string(),
                mode_override: Some(mode),
                cancel: CancellationToken::new(),
                parent_agent: "parent".to_string(),
                parent_context: None,
                delegation_policy: DispatchRequest::policy_from_depth(0),
                runtime_context: Some(echo_core::tools::ExternalRunContext {
                    conversation_id: None,
                    run_id: None,
                    turn_id: None,
                    execution_id: None,
                    isolation_id: None,
                    message_id: None,
                    cancel: None,
                    trace_sink: None,
                    delegation_policy: None,
                    resource_guards: vec![InvocationResourceGuard::new(DispatchGuardDrop(
                        Arc::clone(&drops),
                    ))],
                    subagent_lineage: None,
                    uplink: None,
                }),
                message: None,
                prompt_payload: None,
                prompt_context: None,
                constraints: Vec::new(),
                background: false,
            })
            .await
            .map_err(|error| error.to_string())?;
        if result.outcome.status != SubagentStatus::Completed {
            return Err(format!(
                "guarded dispatch did not complete: {:?}",
                result.outcome.status
            ));
        }

        let contexts = agent.take_invocation_contexts();
        let guard_count = contexts
            .first()
            .and_then(|context| context.runtime.as_ref())
            .map(|runtime| runtime.resource_guards.len());
        if guard_count != Some(1) {
            return Err(format!(
                "subagent invocation lost runtime guards: {guard_count:?}"
            ));
        }
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        drop(contexts);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        Ok(())
    }

    /// 验证 Sync 派发在结算前保留运行时守卫。
    #[tokio::test]
    async fn sync_dispatch_retains_runtime_guards_until_settlement()
    -> std::result::Result<(), String> {
        assert_mode_retains_runtime_guards(ExecutionMode::Sync).await
    }

    /// 验证 Fork 派发在结算前保留运行时守卫。
    #[tokio::test]
    async fn fork_dispatch_retains_runtime_guards_until_settlement()
    -> std::result::Result<(), String> {
        assert_mode_retains_runtime_guards(ExecutionMode::Fork).await
    }

    /// 验证 Teammate 派发在结算前保留运行时守卫。
    #[tokio::test]
    async fn teammate_dispatch_retains_runtime_guards_until_settlement()
    -> std::result::Result<(), String> {
        assert_mode_retains_runtime_guards(ExecutionMode::Teammate).await
    }

    /// 验证派发会保留丰富的工具事件，而无需并行适配器。
    #[tokio::test]
    async fn dispatch_preserves_rich_tool_events_without_parallel_adapters()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let (registry, executor) = make_executor().await;
        registry
            .register(
                SubagentDefinition::new("rich-tool-events", "Rich tool events"),
                Box::new(RichToolEventAgent),
            )
            .await;
        let mut events = registry.event_bus().subscribe();

        let dispatched = executor
            .dispatch(DispatchRequest {
                agent_name: "rich-tool-events".to_string(),
                task: "preserve typed events".to_string(),
                mode_override: None,
                cancel: CancellationToken::new(),
                parent_agent: "parent".to_string(),
                parent_context: None,
                delegation_policy: DispatchRequest::policy_from_depth(0),
                runtime_context: None,
                message: None,
                prompt_payload: None,
                prompt_context: None,
                constraints: Vec::new(),
                background: false,
            })
            .await?;
        assert_eq!(dispatched.output, "done");

        let mut started = None;
        let mut completed = None;
        while let Ok(event) = events.try_recv() {
            match event.as_ref() {
                SubagentEvent::DispatchToolStarted { invocation, .. } => {
                    started = Some(invocation.clone());
                }
                SubagentEvent::DispatchToolCompleted { name, result, .. } => {
                    completed = Some((name.clone(), result.clone()));
                }
                _ => {}
            }
        }

        let Some(invocation) = started else {
            return Err(std::io::Error::other("missing typed tool-start event").into());
        };
        assert_eq!(invocation.requested_name, "requested_tool");
        assert_eq!(invocation.name, "effective_tool");
        assert_eq!(invocation.rewrites, vec![ToolInvocationRewrite::Approval]);
        assert_eq!(
            invocation
                .args
                .get("value")
                .and_then(serde_json::Value::as_str),
            Some("effective")
        );

        let Some((name, result)) = completed else {
            return Err(std::io::Error::other("missing typed tool-result event").into());
        };
        assert_eq!(name, "effective_tool");
        assert_eq!(result.kind, ToolResultKind::Json);
        assert_eq!(result.data, Some(serde_json::json!({"ok": true})));
        assert_eq!(result.mime_type.as_deref(), Some("application/json"));
        assert!(result.truncated);
        assert_eq!(
            result.metadata.get("source").map(String::as_str),
            Some("fixture")
        );
        Ok(())
    }

    /// 从事件接收端收集所有终止事件（完成/失败/取消）。
    fn collect_terminal_events(
        events: &mut tokio::sync::broadcast::Receiver<Arc<SubagentEvent>>,
    ) -> Vec<Arc<SubagentEvent>> {
        let mut terminal_events = Vec::new();
        while let Ok(event) = events.try_recv() {
            if matches!(
                event.as_ref(),
                SubagentEvent::DispatchCompleted { .. }
                    | SubagentEvent::DispatchFailed { .. }
                    | SubagentEvent::DispatchCancelled { .. }
            ) {
                terminal_events.push(event);
            }
        }
        terminal_events
    }

    /// 验证结构化结果的终止事件与运行时状态一致。
    #[test]
    fn structured_result_terminal_event_matches_runtime_status() -> Result<()> {
        for status in [
            SubagentStatus::Completed,
            SubagentStatus::Failed,
            SubagentStatus::TimedOut,
            SubagentStatus::Cancelled,
            SubagentStatus::Running,
        ] {
            let mut result = SubagentResult::sync_result(
                "child",
                "## Summary\nresult".to_string(),
                std::time::Duration::ZERO,
            );
            result.outcome.status = status;
            let event = SubagentExecutor::terminal_event_for_result(
                "parent",
                "child",
                &mut result,
                Some("attempt-1".to_string()),
                Some("run-1".to_string()),
            );
            let terminal = match &event {
                SubagentEvent::DispatchCompleted { outcome, .. }
                    if status == SubagentStatus::Completed =>
                {
                    outcome.status
                }
                SubagentEvent::DispatchCancelled { outcome, .. }
                    if status == SubagentStatus::Cancelled =>
                {
                    outcome.status
                }
                SubagentEvent::DispatchFailed {
                    status: emitted,
                    outcome,
                    ..
                } if status == SubagentStatus::Running && *emitted == SubagentStatus::Failed => {
                    outcome.status
                }
                SubagentEvent::DispatchFailed {
                    status: emitted,
                    outcome,
                    ..
                } if *emitted == status => outcome.status,
                other => {
                    return Err(ReactError::Other(format!(
                        "incorrect terminal event for {status:?}: {other:?}"
                    )));
                }
            };
            assert_eq!(terminal, result.outcome.status);
            assert_ne!(terminal, SubagentStatus::Running);
        }
        Ok(())
    }

    /// 验证 DispatchRequest 以嵌套的委派策略作为委派深度的判定依据。
    #[test]
    fn dispatch_request_uses_nested_delegation_policy_as_depth_authority() {
        let policy = DispatchRequest::policy_from_depth(2);

        assert!(policy.can_spawn_subagents);
        assert_eq!(policy.delegate_depth, 2);
        assert_eq!(policy.max_delegate_depth, 3);

        let child = policy.child_policy().unwrap_or_default();
        assert_eq!(child.delegate_depth, 3);
        assert!(!child.can_delegate());
    }

    /// 验证合并观测证据时保留最新的检查结果。
    #[test]
    fn merge_observed_evidence_keeps_latest_check_result() {
        let mut outcome = SubagentOutcome {
            evidence: vec![SubagentEvidence {
                kind: "verification".to_string(),
                subject: "cargo test".to_string(),
                outcome: Some("passed".to_string()),
                details: "model claim".to_string(),
                source: SubagentEvidenceSource::Reported,
                attributes: serde_json::Value::Null,
            }],
            ..SubagentOutcome::default()
        };
        merge_observed_evidence(
            &mut outcome,
            vec![
                SubagentEvidence {
                    kind: "verification".to_string(),
                    subject: "cargo test".to_string(),
                    outcome: Some("failed".to_string()),
                    details: "first run failed".to_string(),
                    source: SubagentEvidenceSource::Observed,
                    attributes: serde_json::Value::Null,
                },
                SubagentEvidence {
                    kind: "verification".to_string(),
                    subject: "cargo test".to_string(),
                    outcome: Some("passed".to_string()),
                    details: "retry passed".to_string(),
                    source: SubagentEvidenceSource::Observed,
                    attributes: serde_json::Value::Null,
                },
            ],
            Vec::new(),
        );
        assert!(matches!(
            outcome.evidence.as_slice(),
            [SubagentEvidence {
                source: SubagentEvidenceSource::Observed,
                ..
            }]
        ));
        assert_eq!(
            outcome
                .evidence
                .first()
                .and_then(|item| item.outcome.as_deref()),
            Some("passed")
        );
    }

    /// 验证合并观测证据时保留参数不同的工具调用。
    #[test]
    fn merge_observed_evidence_preserves_distinct_tool_arguments() {
        let mut outcome = SubagentOutcome::default();
        let evidence = ["src/a.rs", "src/b.rs"]
            .into_iter()
            .map(|path| SubagentEvidence {
                kind: "tool_result".to_string(),
                subject: "write_file".to_string(),
                outcome: Some("succeeded".to_string()),
                details: String::new(),
                source: SubagentEvidenceSource::Observed,
                attributes: serde_json::json!({ "args": { "path": path } }),
            })
            .collect();

        merge_observed_evidence(&mut outcome, evidence, Vec::new());

        assert_eq!(outcome.evidence.len(), 2);
    }

    /// 验证 Sync 模式派发的基本流程。
    #[tokio::test]
    async fn test_dispatch_sync() {
        let (registry, executor) = make_executor().await;

        let agent = MockAgent::new("subagent").with_response("done");
        let def = super::super::types::SubagentDefinition::new("subagent", "Subagent");
        registry.register(def, Box::new(agent)).await;

        let req = DispatchRequest {
            agent_name: "subagent".into(),
            task: "do work".into(),
            mode_override: None,
            cancel: CancellationToken::new(),
            parent_agent: "parent".into(),
            parent_context: None,
            delegation_policy: DispatchRequest::policy_from_depth(0),
            runtime_context: None,
            message: None,
            prompt_payload: None,
            prompt_context: None,
            constraints: Vec::new(),
            background: false,
        };

        let result = executor.dispatch(req).await.unwrap();
        assert_eq!(result.output, "done");
        assert_eq!(result.mode, ExecutionMode::Sync);
    }

    /// 验证在排队期间被取消的 Fork 永不启动 Agent。
    #[tokio::test]
    async fn fork_cancelled_while_queued_never_starts_agent() -> std::result::Result<(), String> {
        let registry = Arc::new(SubagentRegistry::new());
        let agent = MockAgent::new("queued").with_response("should not run");
        let observed = agent.clone();
        let mut definition = SubagentDefinition::new("queued", "Queued agent");
        definition.execution_mode = ExecutionMode::Fork;
        registry.register(definition, Box::new(agent)).await;
        let executor = Arc::new(SubagentExecutor::new(
            registry,
            SubagentExecutorConfig {
                max_concurrent_forks: 1,
                default_timeout_secs: 30,
                ..SubagentExecutorConfig::default()
            },
        ));
        let permit = Arc::clone(&executor.semaphore)
            .acquire_owned()
            .await
            .map_err(|error| error.to_string())?;
        let cancel = CancellationToken::new();
        let request = DispatchRequest {
            agent_name: "queued".to_string(),
            task: "must not start".to_string(),
            mode_override: None,
            cancel: cancel.clone(),
            parent_agent: "parent".to_string(),
            parent_context: None,
            delegation_policy: DispatchRequest::policy_from_depth(0),
            runtime_context: None,
            message: None,
            prompt_payload: None,
            prompt_context: None,
            constraints: Vec::new(),
            background: false,
        };
        let execution = tokio::spawn({
            let executor = Arc::clone(&executor);
            async move { executor.dispatch(request).await }
        });
        tokio::task::yield_now().await;
        cancel.cancel();
        drop(permit);

        let result = tokio::time::timeout(Duration::from_secs(2), execution)
            .await
            .map_err(|_| "cancelled queued dispatch did not terminate".to_string())?
            .map_err(|error| error.to_string())?;
        assert!(result.is_err());
        assert_eq!(observed.call_count(), 0);
        Ok(())
    }

    /// 验证派发使用注入的提示词编译器。
    #[tokio::test]
    async fn dispatch_uses_injected_prompt_compiler() -> Result<()> {
        let registry = Arc::new(SubagentRegistry::new());
        let executor = SubagentExecutor::new(
            registry.clone(),
            SubagentExecutorConfig {
                prompt_compiler: Arc::new(PrefixPromptCompiler),
                ..SubagentExecutorConfig::default()
            },
        );
        let agent = MockAgent::new("compiled").with_response("done");
        registry
            .register(
                super::super::types::SubagentDefinition::new("compiled", "Compiled subagent"),
                Box::new(agent.clone()),
            )
            .await;

        executor
            .dispatch(DispatchRequest {
                agent_name: "compiled".to_string(),
                task: "do work".to_string(),
                mode_override: Some(ExecutionMode::Sync),
                cancel: CancellationToken::new(),
                parent_agent: "parent".to_string(),
                parent_context: None,
                delegation_policy: DispatchRequest::policy_from_depth(0),
                runtime_context: None,
                message: None,
                prompt_payload: None,
                prompt_context: None,
                constraints: Vec::new(),
                background: false,
            })
            .await?;

        assert_eq!(agent.last_task().as_deref(), Some("compiled:do work"));
        Ok(())
    }

    /// 验证后台派发在任务完成前即返回。
    #[tokio::test]
    async fn dispatch_background_returns_before_completion() -> Result<()> {
        let (registry, executor) = make_executor().await;
        let agent = MockAgent::new("slow")
            .with_delay_ms(200)
            .with_response("## Summary\nbg done");
        let def = super::super::types::SubagentDefinition::new("slow", "Slow subagent");
        registry.register(def, Box::new(agent)).await;

        let mut events = registry.event_bus().subscribe();
        let req = DispatchRequest {
            agent_name: "slow".into(),
            task: "take your time".into(),
            mode_override: None,
            cancel: CancellationToken::new(),
            parent_agent: "parent".into(),
            parent_context: None,
            delegation_policy: DispatchRequest::policy_from_depth(0),
            runtime_context: None,
            message: None,
            prompt_payload: None,
            prompt_context: None,
            constraints: Vec::new(),
            background: false,
        };

        let started_at = Instant::now();
        let handle = executor.dispatch_background(req).await?;
        assert!(
            started_at.elapsed() < Duration::from_millis(100),
            "dispatch_background must return before the slow subagent finishes"
        );
        assert!(!handle.execution_id.is_empty());
        assert_eq!(handle.agent_name, "slow");
        assert!(handle.execution_id.starts_with("agent_tool-"));
        assert!(!handle.is_finished());

        let mut saw_started_bg = false;
        let mut saw_completed = false;
        let deadline = tokio::time::Instant::now() + Duration::from_secs(3);
        while tokio::time::Instant::now() < deadline && !(saw_started_bg && saw_completed) {
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            match tokio::time::timeout(remaining, events.recv()).await {
                Ok(Ok(ev)) => match ev.as_ref() {
                    SubagentEvent::DispatchStarted {
                        background: true,
                        execution_id: Some(id),
                        ..
                    } if id == &handle.execution_id => {
                        saw_started_bg = true;
                    }
                    SubagentEvent::DispatchCompleted {
                        output,
                        execution_id: Some(id),
                        ..
                    } if id == &handle.execution_id => {
                        assert!(output.contains("bg done"));
                        saw_completed = true;
                    }
                    _ => {}
                },
                _ => break,
            }
        }
        assert!(
            saw_started_bg,
            "expected DispatchStarted with background=true"
        );
        assert!(
            saw_completed,
            "expected DispatchCompleted after background work"
        );
        let result = handle.join().await?;
        assert!(result.output.contains("bg done"));
        assert!(
            handle.join().await.is_err(),
            "result must only be consumed once"
        );
        Ok(())
    }

    /// 验证受控派发对精确尝试只认领一次排队指导。
    #[tokio::test]
    async fn controlled_dispatch_claims_guidance_once_for_exact_attempt() -> Result<()> {
        let (registry, executor) = make_executor().await;
        let agent = MockAgent::new("guided").with_response("done");
        registry
            .register(
                SubagentDefinition::new("guided", "Guided subagent"),
                Box::new(agent.clone()),
            )
            .await;
        executor
            .queue_guidance("task-guided", 1, "inspect the latest revision")
            .map_err(|error| ReactError::Other(error.to_string()))?;
        let request = DispatchRequest {
            agent_name: "guided".to_string(),
            task: "perform task".to_string(),
            mode_override: Some(ExecutionMode::Sync),
            cancel: CancellationToken::new(),
            parent_agent: "parent".to_string(),
            parent_context: None,
            delegation_policy: DispatchRequest::policy_from_depth(0),
            runtime_context: None,
            message: None,
            prompt_payload: None,
            prompt_context: None,
            constraints: Vec::new(),
            background: false,
        };
        let identity = SubagentAttemptIdentity::new("task-guided", "execution-guided-1", 1)
            .map_err(|error| ReactError::Other(error.to_string()))?;
        executor.dispatch_attempt(request, identity).await?;

        let observed = agent.last_task().unwrap_or_default();
        assert!(observed.contains("perform task"));
        assert!(observed.contains("[queued_guidance]"));
        assert!(observed.contains("inspect the latest revision"));
        assert!(matches!(
            executor.queue_guidance("task-guided", 1, "late"),
            Err(SubagentControlError::AttemptAlreadyStarted { .. })
        ));
        Ok(())
    }

    /// 验证后台句柄可以取消派发。
    #[tokio::test]
    async fn background_handle_cancels_dispatch() -> Result<()> {
        let (registry, executor) = make_executor().await;
        registry
            .register(
                super::super::types::SubagentDefinition::new("slow", "Slow subagent"),
                Box::new(MockAgent::new("slow").with_delay_ms(1_000)),
            )
            .await;
        let handle = executor
            .dispatch_background(DispatchRequest {
                agent_name: "slow".into(),
                task: "wait".into(),
                mode_override: None,
                cancel: CancellationToken::new(),
                parent_agent: "parent".into(),
                parent_context: None,
                delegation_policy: DispatchRequest::policy_from_depth(0),
                runtime_context: None,
                message: None,
                prompt_payload: None,
                prompt_context: None,
                constraints: Vec::new(),
                background: false,
            })
            .await?;
        handle.cancel();
        let result = handle.join().await?;
        assert_eq!(result.outcome.status, SubagentStatus::Cancelled);
        Ok(())
    }

    /// 验证受控后台中断在句柄返回之前已完成注册。
    #[tokio::test]
    async fn controlled_background_interrupt_is_registered_before_handle_returns() -> Result<()> {
        let (registry, executor) = make_executor().await;
        registry
            .register(
                SubagentDefinition::new("controlled", "Controlled subagent"),
                Box::new(MockAgent::new("controlled").with_response("must not escape")),
            )
            .await;
        let request = DispatchRequest {
            agent_name: "controlled".to_string(),
            task: "wait".to_string(),
            mode_override: Some(ExecutionMode::Sync),
            cancel: CancellationToken::new(),
            parent_agent: "parent".to_string(),
            parent_context: None,
            delegation_policy: DispatchRequest::policy_from_depth(0),
            runtime_context: None,
            message: None,
            prompt_payload: None,
            prompt_context: None,
            constraints: Vec::new(),
            background: false,
        };
        let identity = SubagentAttemptIdentity::new("task-controlled", "execution-controlled", 1)
            .map_err(|error| ReactError::Other(error.to_string()))?;
        let handle = executor
            .dispatch_background_attempt(request, identity)
            .await?;

        let interrupted = executor
            .interrupt_subagent("execution-controlled", 1)
            .await
            .map_err(|error| ReactError::Other(error.to_string()))?;
        assert!(interrupted.requested);
        assert!(interrupted.settled);
        assert_eq!(interrupted.terminal_status, Some(SubagentStatus::Cancelled));
        let result = handle.join().await?;
        assert_eq!(result.outcome.status, SubagentStatus::Cancelled);
        assert!(matches!(
            executor
                .send_message_tracked("execution-controlled", 1, "late")
                .await,
            Err(SubagentControlError::AttemptSettled { .. })
        ));
        Ok(())
    }

    /// 验证 DispatchStarted 事件保留聊天身份信息。
    #[tokio::test]
    async fn dispatch_started_preserves_chat_identity() -> std::result::Result<(), String> {
        let (registry, executor) = make_executor().await;
        let agent = MockAgent::new("identity").with_response("done");
        let definition =
            super::super::types::SubagentDefinition::new("identity", "Identity subagent");
        registry.register(definition, Box::new(agent)).await;
        let mut events = registry.event_bus().subscribe();
        let mut envelopes = registry.event_bus().subscribe_envelopes();
        let request = DispatchRequest {
            agent_name: "identity".to_string(),
            task: "preserve identity".to_string(),
            mode_override: None,
            cancel: CancellationToken::new(),
            parent_agent: "parent".to_string(),
            parent_context: None,
            delegation_policy: DispatchRequest::policy_from_depth(0),
            runtime_context: Some(echo_core::tools::ExternalRunContext {
                conversation_id: Some("conversation-identity".to_string()),
                run_id: None,
                turn_id: Some("turn-identity".to_string()),
                execution_id: Some("agent_tool-identity".to_string()),
                isolation_id: None,
                message_id: Some("message-identity".to_string()),
                cancel: None,
                trace_sink: None,
                delegation_policy: None,
                resource_guards: Vec::new(),
                subagent_lineage: Some(echo_core::tools::SubagentLineage {
                    task_id: Some("task-identity".to_string()),
                    attempt: Some(3),
                    plan_revision: Some(7),
                    ..echo_core::tools::SubagentLineage::default()
                }),
                uplink: None,
            }),
            message: None,
            prompt_payload: None,
            prompt_context: None,
            constraints: Vec::new(),
            background: false,
        };

        executor
            .dispatch(request)
            .await
            .map_err(|error| error.to_string())?;
        let started = events.recv().await.map_err(|error| error.to_string())?;
        match started.as_ref() {
            SubagentEvent::DispatchStarted {
                conversation_id,
                message_id,
                run_id,
                ..
            } => {
                assert_eq!(conversation_id.as_deref(), Some("conversation-identity"));
                assert_eq!(message_id.as_deref(), Some("message-identity"));
                assert!(run_id.is_none());
            }
            other => return Err(format!("expected DispatchStarted, got {other:?}")),
        }
        let started_envelope = envelopes.recv().await.map_err(|error| error.to_string())?;
        assert_eq!(started_envelope.sequence, 1);
        assert_eq!(
            started_envelope
                .conversation_id
                .as_ref()
                .map(|value| value.as_str()),
            Some("conversation-identity")
        );
        assert_eq!(
            started_envelope
                .message_id
                .as_ref()
                .map(|value| value.as_str()),
            Some("message-identity")
        );
        assert_eq!(
            started_envelope.payload.invocation.task_id.as_deref(),
            Some("task-identity")
        );
        assert_eq!(started_envelope.payload.invocation.attempt, Some(3));
        assert_eq!(started_envelope.payload.invocation.plan_revision, Some(7));
        let terminal = envelopes.recv().await.map_err(|error| error.to_string())?;
        assert_eq!(terminal.sequence, 2);
        assert_eq!(
            terminal.parent_event_id,
            Some(started_envelope.event_id.clone())
        );
        assert!(matches!(
            terminal.payload.event,
            SubagentEvent::DispatchCompleted { .. }
        ));
        Ok(())
    }

    /*
    注意：无法用 MockAgent 为多模态派发转发编写单元测试（即验证子 Agent
    通过 execute_stream_message_with_cancel 收到 Message）——对于这个新增方法，
    trait 对象的虚表路由到的是 trait 默认方法而非 MockAgent 的覆盖实现，
    因此无法单独测试消息路径。子 Agent 的多模态转发改由以下方式验证：
    (1) 上面的文本路径派发测试；(2) 对 DispatchRequest.message 字段与
    execute_agent_streaming 分支的编译期覆盖；(3) 在桌面 UI 中手动测试
    带附件的复杂任务。MockAgent 保留了覆盖实现与消息记录字段，
    供真实 Agent 与未来的测试框架使用。
    */

    /// 验证派发不存在的子 Agent 时报错。
    #[tokio::test]
    async fn test_dispatch_not_found() -> Result<()> {
        let (registry, executor) = make_executor().await;
        let mut envelopes = registry.event_bus().subscribe_envelopes();

        let req = DispatchRequest {
            agent_name: "missing".into(),
            task: "task".into(),
            mode_override: None,
            cancel: CancellationToken::new(),
            parent_agent: "parent".into(),
            parent_context: None,
            delegation_policy: DispatchRequest::policy_from_depth(0),
            runtime_context: None,
            message: None,
            prompt_payload: None,
            prompt_context: None,
            constraints: Vec::new(),
            background: false,
        };

        let err =
            executor.dispatch(req).await.err().ok_or_else(|| {
                ReactError::Other("missing Subagent unexpectedly ran".to_string())
            })?;
        assert!(err.to_string().contains("not found"));
        let started = envelopes
            .recv()
            .await
            .map_err(|error| ReactError::Other(error.to_string()))?;
        let terminal = envelopes
            .recv()
            .await
            .map_err(|error| ReactError::Other(error.to_string()))?;
        assert_eq!(started.sequence, 1);
        assert_eq!(terminal.sequence, 2);
        assert!(matches!(
            terminal.payload.event,
            SubagentEvent::DispatchFailed { .. }
        ));
        Ok(())
    }

    /// 验证已取消的派发返回取消结果。
    #[tokio::test]
    async fn test_dispatch_cancelled() -> Result<()> {
        let (registry, executor) = make_executor().await;

        let agent = MockAgent::new("c").with_response("ok");
        let def = super::super::types::SubagentDefinition::new("c", "C");
        registry.register(def, Box::new(agent)).await;
        let mut envelopes = registry.event_bus().subscribe_envelopes();

        let cancel = CancellationToken::new();
        cancel.cancel();

        let req = DispatchRequest {
            agent_name: "c".into(),
            task: "task".into(),
            mode_override: None,
            cancel,
            parent_agent: "parent".into(),
            parent_context: None,
            delegation_policy: DispatchRequest::policy_from_depth(0),
            runtime_context: None,
            message: None,
            prompt_payload: None,
            prompt_context: None,
            constraints: Vec::new(),
            background: false,
        };

        let result = executor.dispatch(req).await?;
        assert!(result.output.contains("Cancelled"));
        assert_eq!(result.outcome.status, SubagentStatus::Cancelled);
        let started = envelopes
            .recv()
            .await
            .map_err(|error| ReactError::Other(error.to_string()))?;
        let terminal = envelopes
            .recv()
            .await
            .map_err(|error| ReactError::Other(error.to_string()))?;
        assert_eq!(started.sequence, 1);
        assert_eq!(terminal.sequence, 2);
        assert!(matches!(
            terminal.payload.event,
            SubagentEvent::DispatchCancelled { .. }
        ));
        Ok(())
    }

    /// 验证委派深度被拒时只发出一个失败生命周期事件。
    #[tokio::test]
    async fn rejected_delegation_depth_emits_one_failed_lifecycle() -> Result<()> {
        let (registry, executor) = make_executor().await;
        registry
            .register(
                SubagentDefinition::new("depth", "Depth guard"),
                Box::new(MockAgent::new("depth").with_response("must not run")),
            )
            .await;
        let mut envelopes = registry.event_bus().subscribe_envelopes();
        let error = executor
            .dispatch(DispatchRequest {
                agent_name: "depth".to_string(),
                task: "reject".to_string(),
                mode_override: Some(ExecutionMode::Sync),
                cancel: CancellationToken::new(),
                parent_agent: "parent".to_string(),
                parent_context: None,
                delegation_policy: NestedDelegationPolicy {
                    can_spawn_subagents: true,
                    delegate_depth: 2,
                    max_delegate_depth: 1,
                },
                runtime_context: None,
                message: None,
                prompt_payload: None,
                prompt_context: None,
                constraints: Vec::new(),
                background: false,
            })
            .await
            .err()
            .ok_or_else(|| ReactError::Other("invalid depth unexpectedly ran".to_string()))?;
        assert!(error.to_string().contains("Delegation depth exceeded"));
        let started = envelopes
            .recv()
            .await
            .map_err(|error| ReactError::Other(error.to_string()))?;
        let terminal = envelopes
            .recv()
            .await
            .map_err(|error| ReactError::Other(error.to_string()))?;
        assert_eq!((started.sequence, terminal.sequence), (1, 2));
        assert!(matches!(
            terminal.payload.event,
            SubagentEvent::DispatchFailed { .. }
        ));
        Ok(())
    }

    /// 验证恢复后的派发只发出一个终止事件。
    #[tokio::test]
    async fn recovered_dispatch_emits_only_one_terminal_event() -> Result<()> {
        let registry = Arc::new(SubagentRegistry::new());
        let mut hooks = super::super::hooks::SubagentHookRegistry::new();
        hooks.register(Arc::new(DelegateFailedDispatch));
        let executor = SubagentExecutor::with_hooks(
            registry.clone(),
            SubagentExecutorConfig::default(),
            hooks,
        );
        registry
            .register(
                super::super::types::SubagentDefinition::new("primary", "Primary"),
                Box::new(FailingMockAgent::new("primary", "first attempt failed")),
            )
            .await;
        let recovery = MockAgent::new("recovery").with_response("recovered");
        registry
            .register(
                super::super::types::SubagentDefinition::new("recovery", "Recovery"),
                Box::new(recovery.clone()),
            )
            .await;
        let mut events = registry.event_bus().subscribe();
        let mut envelopes = registry.event_bus().subscribe_envelopes();

        let result = executor
            .dispatch(DispatchRequest {
                agent_name: "primary".to_string(),
                task: "recover this task".to_string(),
                mode_override: None,
                cancel: CancellationToken::new(),
                parent_agent: "parent".to_string(),
                parent_context: None,
                delegation_policy: DispatchRequest::policy_from_depth(0),
                runtime_context: None,
                message: None,
                prompt_payload: None,
                prompt_context: None,
                constraints: vec!["Preserve the caller boundary".to_string()],
                background: false,
            })
            .await?;
        assert_eq!(result.outcome.status, SubagentStatus::Completed);
        assert!(
            recovery
                .last_task()
                .as_deref()
                .is_some_and(|task| task.contains("Preserve the caller boundary")),
            "delegated recovery must receive the original constraints"
        );

        let terminal_events = collect_terminal_events(&mut events);
        assert_eq!(terminal_events.len(), 1);
        assert!(matches!(
            terminal_events.first().map(AsRef::as_ref),
            Some(SubagentEvent::DispatchCompleted { outcome, .. })
                if outcome.status == SubagentStatus::Completed
        ));
        let mut sequences = Vec::new();
        let mut agents = Vec::new();
        for _ in 0..2 {
            let envelope = envelopes
                .recv()
                .await
                .map_err(|error| ReactError::Other(error.to_string()))?;
            sequences.push(envelope.sequence);
            agents.push(envelope.payload.invocation.agent_name.clone());
        }
        assert_eq!(sequences, vec![1, 2]);
        assert_eq!(agents, vec!["primary", "recovery"]);
        assert!(matches!(
            envelopes.try_recv(),
            Err(tokio::sync::broadcast::error::TryRecvError::Empty)
        ));
        Ok(())
    }

    /// 验证失败的派发发出 Failed 终止状态。
    #[tokio::test]
    async fn failed_dispatch_emits_failed_terminal_status() -> std::result::Result<(), String> {
        let (registry, executor) = make_executor().await;
        registry
            .register(
                super::super::types::SubagentDefinition::new("failing", "Failing"),
                Box::new(FailingMockAgent::new("failing", "boom")),
            )
            .await;
        let mut events = registry.event_bus().subscribe();
        let error = executor
            .dispatch(DispatchRequest {
                agent_name: "failing".to_string(),
                task: "fail".to_string(),
                mode_override: None,
                cancel: CancellationToken::new(),
                parent_agent: "parent".to_string(),
                parent_context: None,
                delegation_policy: DispatchRequest::policy_from_depth(0),
                runtime_context: None,
                message: None,
                prompt_payload: None,
                prompt_context: None,
                constraints: Vec::new(),
                background: false,
            })
            .await
            .err()
            .ok_or_else(|| "failing subagent unexpectedly completed".to_string())?;
        assert_eq!(subagent_status_from_error(&error), SubagentStatus::Failed);
        let terminal_events = collect_terminal_events(&mut events);
        assert!(matches!(
            terminal_events.as_slice(),
            [event]
                if matches!(
                    event.as_ref(),
                    SubagentEvent::DispatchFailed {
                        status: SubagentStatus::Failed,
                        ..
                    }
                )
        ));
        Ok(())
    }

    /// 验证运行中的派发被取消时发出 Cancelled 终止状态。
    #[tokio::test]
    async fn running_dispatch_cancel_emits_cancelled_terminal_status()
    -> std::result::Result<(), String> {
        let (registry, executor) = make_executor().await;
        registry
            .register(
                super::super::types::SubagentDefinition::new("slow-cancel", "Slow cancel"),
                Box::new(
                    MockAgent::new("slow-cancel")
                        .with_delay_ms(500)
                        .with_default_success("completed"),
                ),
            )
            .await;
        let mut events = registry.event_bus().subscribe();
        let cancel = CancellationToken::new();
        let trigger = cancel.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(20)).await;
            trigger.cancel();
        });
        let error = executor
            .dispatch(DispatchRequest {
                agent_name: "slow-cancel".to_string(),
                task: "wait".to_string(),
                mode_override: None,
                cancel,
                parent_agent: "parent".to_string(),
                parent_context: None,
                delegation_policy: DispatchRequest::policy_from_depth(0),
                runtime_context: None,
                message: None,
                prompt_payload: None,
                prompt_context: None,
                constraints: Vec::new(),
                background: false,
            })
            .await
            .err()
            .ok_or_else(|| "cancelled subagent unexpectedly completed".to_string())?;
        assert_eq!(
            subagent_status_from_error(&error),
            SubagentStatus::Cancelled
        );
        let terminal_events = collect_terminal_events(&mut events);
        assert!(matches!(
            terminal_events.as_slice(),
            [event]
                if matches!(event.as_ref(), SubagentEvent::DispatchCancelled { outcome, .. }
                    if outcome.status == SubagentStatus::Cancelled)
        ));
        Ok(())
    }

    /// 验证超时的派发发出 TimedOut 终止状态。
    #[tokio::test]
    async fn timed_out_dispatch_emits_timed_out_terminal_status() -> std::result::Result<(), String>
    {
        let (registry, executor) = make_executor().await;
        let mut definition =
            super::super::types::SubagentDefinition::new("slow-timeout", "Slow timeout");
        definition.timeout_secs = 1;
        registry
            .register(
                definition,
                Box::new(
                    MockAgent::new("slow-timeout")
                        .with_delay_ms(1_500)
                        .with_default_success("completed"),
                ),
            )
            .await;
        let mut events = registry.event_bus().subscribe();
        let error = executor
            .dispatch(DispatchRequest {
                agent_name: "slow-timeout".to_string(),
                task: "wait".to_string(),
                mode_override: None,
                cancel: CancellationToken::new(),
                parent_agent: "parent".to_string(),
                parent_context: None,
                delegation_policy: DispatchRequest::policy_from_depth(0),
                runtime_context: None,
                message: None,
                prompt_payload: None,
                prompt_context: None,
                constraints: Vec::new(),
                background: false,
            })
            .await
            .err()
            .ok_or_else(|| "timed-out subagent unexpectedly completed".to_string())?;
        assert_eq!(subagent_status_from_error(&error), SubagentStatus::TimedOut);
        let terminal_events = collect_terminal_events(&mut events);
        assert!(matches!(
            terminal_events.as_slice(),
            [event]
                if matches!(
                    event.as_ref(),
                    SubagentEvent::DispatchFailed {
                        status: SubagentStatus::TimedOut,
                        outcome,
                        ..
                    } if outcome.status == SubagentStatus::TimedOut
                )
        ));
        Ok(())
    }

    /// 验证请求中的模式覆盖优先于定义中的执行模式。
    #[tokio::test]
    async fn test_dispatch_mode_override() {
        let (registry, executor) = make_executor().await;

        let agent = MockAgent::new("forker").with_response("forked");
        let mut def = super::super::types::SubagentDefinition::new("forker", "Fork agent");
        def.execution_mode = ExecutionMode::Sync; // 默认即为 Sync
        registry.register(def, Box::new(agent)).await;

        let req = DispatchRequest {
            agent_name: "forker".into(),
            task: "task".into(),
            mode_override: Some(ExecutionMode::Fork),
            cancel: CancellationToken::new(),
            parent_agent: "parent".into(),
            parent_context: None,
            delegation_policy: DispatchRequest::policy_from_depth(0),
            runtime_context: None,
            message: None,
            prompt_payload: None,
            prompt_context: None,
            constraints: Vec::new(),
            background: false,
        };

        let result = executor.dispatch(req).await.unwrap();
        assert_eq!(result.output, "forked");
        assert_eq!(result.mode, ExecutionMode::Fork);
    }

    /// 验证 Sync 派发为每个请求创建全新的 Agent 实例。
    #[tokio::test]
    async fn sync_dispatch_creates_fresh_agent_per_request() -> Result<()> {
        let (registry, executor) = make_executor().await;
        let mut definition = super::super::types::SubagentDefinition::new("explorer", "Explorer");
        definition.execution_mode = ExecutionMode::Sync;
        registry
            .register(
                definition.clone(),
                Box::new(MockAgent::new("explorer").with_response("cached")),
            )
            .await;

        let creations = Arc::new(AtomicUsize::new(0));
        let factory_creations = Arc::clone(&creations);
        let factory = Arc::new(FnAgentFactory::new(move || {
            let instance = factory_creations
                .fetch_add(1, Ordering::SeqCst)
                .saturating_add(1);
            Box::pin(async move {
                Ok(Box::new(
                    MockAgent::new("explorer")
                        .with_delay_ms(50)
                        .with_response(format!("instance-{instance}")),
                ) as Box<dyn Agent>)
            })
        }));
        assert!(registry.register_factory_sync(definition, factory));
        let mut envelopes = registry.event_bus().subscribe_envelopes();

        let request = || DispatchRequest {
            agent_name: "explorer".to_string(),
            task: "inspect".to_string(),
            mode_override: Some(ExecutionMode::Sync),
            cancel: CancellationToken::new(),
            parent_agent: "parent".to_string(),
            parent_context: None,
            delegation_policy: DispatchRequest::policy_from_depth(0),
            runtime_context: None,
            message: None,
            prompt_payload: None,
            prompt_context: None,
            constraints: Vec::new(),
            background: false,
        };

        let (first, second) =
            tokio::join!(executor.dispatch(request()), executor.dispatch(request()));
        let first = first?;
        let second = second?;

        assert_eq!(creations.load(Ordering::SeqCst), 2);
        assert_ne!(first.output, second.output);
        assert_ne!(first.output, "cached");
        assert_ne!(second.output, "cached");
        let mut sequences_by_stream = HashMap::<String, Vec<u64>>::new();
        for _ in 0..4 {
            let envelope = envelopes
                .recv()
                .await
                .map_err(|error| ReactError::Other(error.to_string()))?;
            sequences_by_stream
                .entry(envelope.stream_id.as_str().to_string())
                .or_default()
                .push(envelope.sequence);
        }
        assert_eq!(sequences_by_stream.len(), 2);
        assert!(
            sequences_by_stream
                .values()
                .all(|sequences| sequences.as_slice() == [1, 2])
        );
        Ok(())
    }

    /// 验证 Sync 超时会取消已分离的流生产者。
    #[tokio::test]
    async fn sync_timeout_cancels_detached_stream_producer() -> Result<()> {
        let (registry, executor) = make_executor().await;
        let cancellation_seen = Arc::new(tokio::sync::Notify::new());
        let mut definition = super::super::types::SubagentDefinition::new("slow", "Slow stream");
        definition.execution_mode = ExecutionMode::Sync;
        definition.timeout_secs = 1;
        registry
            .register(
                definition,
                Box::new(CancellationAwareStreamAgent {
                    cancellation_seen: Arc::clone(&cancellation_seen),
                }),
            )
            .await;

        let error = executor
            .dispatch(DispatchRequest {
                agent_name: "slow".to_string(),
                task: "wait".to_string(),
                mode_override: Some(ExecutionMode::Sync),
                cancel: CancellationToken::new(),
                parent_agent: "parent".to_string(),
                parent_context: None,
                delegation_policy: DispatchRequest::policy_from_depth(0),
                runtime_context: None,
                message: None,
                prompt_payload: None,
                prompt_context: None,
                constraints: Vec::new(),
                background: false,
            })
            .await
            .err()
            .ok_or_else(|| ReactError::Other("slow Sync unexpectedly completed".to_string()))?;

        assert_eq!(subagent_status_from_error(&error), SubagentStatus::TimedOut);
        tokio::time::timeout(Duration::from_secs(1), cancellation_seen.notified())
            .await
            .map_err(|_| {
                ReactError::Other(
                    "Sync timeout did not cancel the detached stream producer".to_string(),
                )
            })?;
        Ok(())
    }

    /// 验证 Fork 派发为每个请求创建全新的 Agent 实例。
    #[tokio::test]
    async fn fork_dispatch_creates_fresh_agent_per_request() -> Result<()> {
        let (registry, executor) = make_executor().await;
        let mut definition = super::super::types::SubagentDefinition::new("explorer", "Explorer");
        definition.execution_mode = ExecutionMode::Fork;
        registry
            .register(
                definition.clone(),
                Box::new(MockAgent::new("explorer").with_response("cached")),
            )
            .await;

        let creations = Arc::new(AtomicUsize::new(0));
        let factory_creations = Arc::clone(&creations);
        let factory = Arc::new(FnAgentFactory::new(move || {
            let instance = factory_creations
                .fetch_add(1, Ordering::SeqCst)
                .saturating_add(1);
            Box::pin(async move {
                Ok(Box::new(
                    MockAgent::new("explorer")
                        .with_delay_ms(50)
                        .with_response(format!("instance-{instance}")),
                ) as Box<dyn Agent>)
            })
        }));
        assert!(registry.register_factory_sync(definition, factory));

        let request = || DispatchRequest {
            agent_name: "explorer".to_string(),
            task: "inspect".to_string(),
            mode_override: Some(ExecutionMode::Fork),
            cancel: CancellationToken::new(),
            parent_agent: "parent".to_string(),
            parent_context: None,
            delegation_policy: DispatchRequest::policy_from_depth(0),
            runtime_context: None,
            message: None,
            prompt_payload: None,
            prompt_context: None,
            constraints: Vec::new(),
            background: false,
        };

        let (first, second) =
            tokio::join!(executor.dispatch(request()), executor.dispatch(request()));
        let first = first?;
        let second = second?;

        assert_eq!(creations.load(Ordering::SeqCst), 2);
        assert_ne!(first.output, second.output);
        assert_ne!(first.output, "cached");
        assert_ne!(second.output, "cached");
        Ok(())
    }

    /// 验证 Fork 超时会取消已分离的流生产者。
    #[tokio::test]
    async fn fork_timeout_cancels_detached_stream_producer() -> Result<()> {
        let (registry, executor) = make_executor().await;
        let cancellation_seen = Arc::new(tokio::sync::Notify::new());
        let mut definition = super::super::types::SubagentDefinition::new("slow", "Slow stream");
        definition.execution_mode = ExecutionMode::Fork;
        definition.timeout_secs = 1;
        registry
            .register(
                definition,
                Box::new(CancellationAwareStreamAgent {
                    cancellation_seen: Arc::clone(&cancellation_seen),
                }),
            )
            .await;

        let error = executor
            .dispatch(DispatchRequest {
                agent_name: "slow".to_string(),
                task: "wait".to_string(),
                mode_override: Some(ExecutionMode::Fork),
                cancel: CancellationToken::new(),
                parent_agent: "parent".to_string(),
                parent_context: None,
                delegation_policy: DispatchRequest::policy_from_depth(0),
                runtime_context: None,
                message: None,
                prompt_payload: None,
                prompt_context: None,
                constraints: Vec::new(),
                background: false,
            })
            .await
            .err()
            .ok_or_else(|| ReactError::Other("slow Fork unexpectedly completed".to_string()))?;

        assert_eq!(subagent_status_from_error(&error), SubagentStatus::TimedOut);
        tokio::time::timeout(Duration::from_secs(1), cancellation_seen.notified())
            .await
            .map_err(|_| {
                ReactError::Other(
                    "Fork timeout did not cancel the detached stream producer".to_string(),
                )
            })?;
        Ok(())
    }

    /// 验证 Teammate 模式派发。
    #[tokio::test]
    async fn test_teammate_dispatch() -> Result<()> {
        let (registry, executor) = make_executor().await;

        let agent = MockAgent::new("tm").with_response("team result");
        let mut def = super::super::types::SubagentDefinition::new("tm", "Teammate");
        def.execution_mode = ExecutionMode::Teammate;
        registry.register(def, Box::new(agent)).await;
        let mut envelopes = registry.event_bus().subscribe_envelopes();

        let req = DispatchRequest {
            agent_name: "tm".into(),
            task: "team task".into(),
            mode_override: None,
            cancel: CancellationToken::new(),
            parent_agent: "leader".into(),
            parent_context: None,
            delegation_policy: DispatchRequest::policy_from_depth(0),
            runtime_context: None,
            message: None,
            prompt_payload: None,
            prompt_context: None,
            constraints: Vec::new(),
            background: false,
        };

        let handle = executor.dispatch_teammate(req).await?;
        assert_eq!(handle.agent_name, "tm");

        let result = handle.join().await?;
        assert_eq!(result.output, "team result");
        let started = envelopes
            .recv()
            .await
            .map_err(|error| ReactError::Other(error.to_string()))?;
        let terminal = envelopes
            .recv()
            .await
            .map_err(|error| ReactError::Other(error.to_string()))?;
        assert_eq!(started.sequence, 1);
        assert_eq!(terminal.sequence, 2);
        assert!(matches!(
            terminal.payload.event,
            SubagentEvent::DispatchCompleted { .. }
        ));
        Ok(())
    }

    /// 验证 Teammate 启动失败时发出 Started 与 Failed 事件。
    #[tokio::test]
    async fn teammate_setup_failure_emits_started_and_failed() -> Result<()> {
        let (registry, executor) = make_executor().await;
        let mut envelopes = registry.event_bus().subscribe_envelopes();
        let error = executor
            .dispatch_teammate(DispatchRequest {
                agent_name: "missing-teammate".to_string(),
                task: "fail setup".to_string(),
                mode_override: Some(ExecutionMode::Teammate),
                cancel: CancellationToken::new(),
                parent_agent: "parent".to_string(),
                parent_context: None,
                delegation_policy: DispatchRequest::policy_from_depth(0),
                runtime_context: None,
                message: None,
                prompt_payload: None,
                prompt_context: None,
                constraints: Vec::new(),
                background: false,
            })
            .await
            .err()
            .ok_or_else(|| ReactError::Other("missing teammate unexpectedly ran".to_string()))?;
        assert!(error.to_string().contains("not found"));
        let started = envelopes
            .recv()
            .await
            .map_err(|error| ReactError::Other(error.to_string()))?;
        let failed = envelopes
            .recv()
            .await
            .map_err(|error| ReactError::Other(error.to_string()))?;
        assert_eq!((started.sequence, failed.sequence), (1, 2));
        assert!(matches!(
            failed.payload.event,
            SubagentEvent::DispatchFailed { .. }
        ));
        Ok(())
    }

    /// 验证 Team 派发拒绝未注册成员，且不会误报完成。
    #[tokio::test]
    async fn team_dispatch_rejects_unregistered_member_without_false_completion() -> Result<()> {
        let (registry, executor) = make_executor().await;
        let definition = super::super::builder::SubagentBuilder::new("pipeline-team")
            .team(super::super::team::TeamSpec {
                strategy: super::super::team::TeamStrategy::Pipeline(vec![
                    "missing-member".to_string(),
                ]),
                manager: String::new(),
                subagents: Vec::new(),
                config: super::super::team::TeamConfig::default(),
            })
            .build();
        registry
            .register(definition, Box::new(MockAgent::new("pipeline-team")))
            .await;

        let error = executor
            .dispatch(DispatchRequest {
                agent_name: "pipeline-team".to_string(),
                task: "run pipeline".to_string(),
                mode_override: None,
                cancel: CancellationToken::new(),
                parent_agent: "parent".to_string(),
                parent_context: None,
                delegation_policy: DispatchRequest::policy_from_depth(0),
                runtime_context: None,
                message: None,
                prompt_payload: None,
                prompt_context: None,
                constraints: Vec::new(),
                background: false,
            })
            .await
            .err()
            .ok_or_else(|| {
                ReactError::Other("unregistered Team member completed successfully".to_string())
            })?;
        assert!(error.to_string().contains("not registered"));
        Ok(())
    }

    /// 验证 Team 成员派发保留由认领推导出的执行身份。
    #[tokio::test]
    async fn team_member_dispatch_preserves_claim_derived_execution_identity() -> Result<()> {
        let (registry, executor) = make_executor().await;
        registry
            .register(
                SubagentDefinition::new("member", "Team member"),
                Box::new(MockAgent::new("member").with_response("member done")),
            )
            .await;
        let definition = super::super::builder::SubagentBuilder::new("pipeline-team")
            .team(super::super::team::TeamSpec {
                strategy: super::super::team::TeamStrategy::Pipeline(vec!["member".to_string()]),
                manager: String::new(),
                subagents: Vec::new(),
                config: super::super::team::TeamConfig {
                    default_timeout_secs: 0,
                    ..super::super::team::TeamConfig::default()
                },
            })
            .build();
        registry
            .register(definition, Box::new(MockAgent::new("pipeline-team")))
            .await;

        let mut events = registry.event_bus().subscribe();
        let request = DispatchRequest {
            agent_name: "pipeline-team".to_string(),
            task: "run the member".to_string(),
            mode_override: None,
            cancel: CancellationToken::new(),
            parent_agent: "parent".to_string(),
            parent_context: None,
            delegation_policy: DispatchRequest::policy_from_depth(0),
            runtime_context: Some(echo_core::tools::ExternalRunContext {
                run_id: Some("team-identity-run".to_string()),
                ..echo_core::tools::ExternalRunContext::default()
            }),
            message: None,
            prompt_payload: None,
            prompt_context: None,
            constraints: Vec::new(),
            background: false,
        };
        let root_identity = SubagentAttemptIdentity::new("root-task", "team-identity-root", 1)
            .map_err(|error| ReactError::Other(error.to_string()))?;
        let result = executor.dispatch_attempt(request, root_identity).await?;
        assert_eq!(result.outcome.status, SubagentStatus::Completed);
        let team_runtime = executor
            .team_runtime_handles("team-identity-run")
            .first()
            .cloned()
            .ok_or_else(|| {
                ReactError::Other("React Team runtime handle was not retained".to_string())
            })?;
        assert_eq!(team_runtime.run_id(), "team-identity-run");
        assert!(
            team_runtime
                .snapshot()
                .await?
                .tasks
                .iter()
                .all(|task| task.execution.status.is_terminal())
        );

        let mut member_execution = None;
        while let Ok(event) = tokio::time::timeout(Duration::from_secs(1), events.recv()).await {
            let event = event.map_err(|error| ReactError::Other(error.to_string()))?;
            if let SubagentEvent::DispatchStarted {
                agent,
                execution_id: Some(execution_id),
                run_id: Some(run_id),
                ..
            } = event.as_ref()
                && agent == "member"
            {
                member_execution = Some((execution_id.clone(), run_id.clone()));
                break;
            }
        }
        let (execution_id, run_id) = member_execution.ok_or_else(|| {
            ReactError::Other("member DispatchStarted event was not observed".to_string())
        })?;
        assert_eq!(run_id, "team-identity-run");
        assert!(execution_id.starts_with("team-identity-run:"));
        assert!(execution_id.contains(":1:"));
        Ok(())
    }

    /// 验证同一 run 的多个 Team 运行时按唯一句柄身份分别保留。
    #[test]
    fn same_run_team_runtimes_are_retained_by_unique_handle_identity() -> Result<()> {
        let registry = Arc::new(SubagentRegistry::new());
        let executor = SubagentExecutor::new(registry, SubagentExecutorConfig::default());
        let make_handle = || {
            let dispatch: super::super::team::TeamDispatchFn = Arc::new(|request| {
                Box::pin(async move {
                    Ok(SubagentResult::sync_result(
                        &request.member,
                        "done".to_string(),
                        Duration::ZERO,
                    ))
                })
            });
            let controller = super::super::team::closure_dispatch_controller(
                dispatch,
                super::super::team::TeamDispatchControl::default(),
            );
            super::super::team::runtime_handle_for_controller(
                "shared-run".to_string(),
                controller,
                echo_orchestration::tasks::RuntimeTaskServiceConfig::default(),
            )
        };
        let handles = (0..=TEAM_RUNTIME_RETENTION)
            .map(|_| make_handle())
            .collect::<Vec<_>>();
        let active = handles
            .iter()
            .map(|handle| handle.start_execution())
            .collect::<Vec<_>>();
        for handle in &handles {
            executor.retain_team_runtime(handle.clone());
        }

        let retained = executor.team_runtime_handles("shared-run");
        assert_eq!(retained.len(), TEAM_RUNTIME_RETENTION.saturating_add(1));
        let first = handles
            .first()
            .ok_or_else(|| ReactError::Other("first Team handle missing".to_string()))?;
        let last = handles
            .last()
            .ok_or_else(|| ReactError::Other("last Team handle missing".to_string()))?;
        assert!(
            executor
                .team_runtime_handle(first.handle_id())
                .is_some_and(|retained| Arc::ptr_eq(&retained, first))
        );
        assert!(
            executor
                .team_runtime_handle(last.handle_id())
                .is_some_and(|retained| Arc::ptr_eq(&retained, last))
        );
        drop(active);
        prune_team_runtime_index(&executor.team_runtimes);
        assert_eq!(
            executor.team_runtime_handles("shared-run").len(),
            TEAM_RUNTIME_RETENTION
        );
        Ok(())
    }

    /// 验证 ReAct Team 的定向中止会丢弃不配合成员的 future。
    #[tokio::test]
    async fn react_team_targeted_abort_drops_non_cooperative_member_future() -> Result<()> {
        let (registry, executor) = make_executor().await;
        let executor = Arc::new(executor);
        let dropped = Arc::new(AtomicBool::new(false));
        registry
            .register(
                SubagentDefinition::new("stuck-member", "Non-cooperative Team member"),
                Box::new(NonCooperativeTeamAgent {
                    dropped: Arc::clone(&dropped),
                }),
            )
            .await;
        let definition = super::super::builder::SubagentBuilder::new("controlled-team")
            .team(super::super::team::TeamSpec {
                strategy: super::super::team::TeamStrategy::Pipeline(vec![
                    "stuck-member".to_string(),
                ]),
                manager: String::new(),
                subagents: Vec::new(),
                config: super::super::team::TeamConfig {
                    default_timeout_secs: 0,
                    ..super::super::team::TeamConfig::default()
                },
            })
            .build();
        registry
            .register(definition, Box::new(MockAgent::new("controlled-team")))
            .await;
        let run_id = "react-team-non-cooperative";
        let request = DispatchRequest {
            agent_name: "controlled-team".to_string(),
            task: "wait forever".to_string(),
            mode_override: None,
            cancel: CancellationToken::new(),
            parent_agent: "parent".to_string(),
            parent_context: None,
            delegation_policy: DispatchRequest::policy_from_depth(0),
            runtime_context: Some(echo_core::tools::ExternalRunContext {
                run_id: Some(run_id.to_string()),
                ..echo_core::tools::ExternalRunContext::default()
            }),
            message: None,
            prompt_payload: None,
            prompt_context: None,
            constraints: Vec::new(),
            background: false,
        };
        let root_identity =
            SubagentAttemptIdentity::for_run(run_id, "root-task", "react-team-root", 1)
                .map_err(|error| ReactError::Other(error.to_string()))?;
        let execution = tokio::spawn({
            let executor = Arc::clone(&executor);
            async move { executor.dispatch_attempt(request, root_identity).await }
        });
        let observed = tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                if let Some(handle) = executor.team_runtime_handles(run_id).first().cloned()
                    && let Ok(snapshot) = handle.snapshot().await
                    && let Some((task_id, claim)) = snapshot.tasks.iter().find_map(|task| {
                        task.execution
                            .claim
                            .clone()
                            .map(|claim| (task.spec.id.clone(), claim))
                    })
                {
                    break (handle, task_id, claim);
                }
                tokio::task::yield_now().await;
            }
        })
        .await;
        let (handle, task_id, claim) = match observed {
            Ok(observed) => observed,
            Err(_) if execution.is_finished() => {
                let result = execution.await.map_err(|error| {
                    ReactError::Other(format!("React Team task panicked: {error}"))
                })?;
                return Err(ReactError::Other(format!(
                    "React Team finished before claim observation: {result:?}"
                )));
            }
            Err(_) => {
                return Err(ReactError::Other(
                    "React Team claim was not observable".to_string(),
                ));
            }
        };
        assert!(
            handle
                .request_attempt_interrupt(&task_id, &claim)
                .await?
                .requested
        );
        let error = tokio::time::timeout(Duration::from_secs(7), execution)
            .await
            .map_err(|_| ReactError::Other("React Team exact abort did not settle".to_string()))?
            .map_err(|error| ReactError::Other(format!("React Team task panicked: {error}")))?
            .err()
            .ok_or_else(|| ReactError::Other("cancelled React Team succeeded".to_string()))?;
        assert!(
            matches!(
                &error,
                ReactError::Agent(agent_error)
                    if matches!(agent_error.as_ref(), AgentError::Cancelled(_))
            ),
            "unexpected React Team terminal: {error:?}"
        );
        assert!(dropped.load(Ordering::SeqCst));
        Ok(())
    }

    /// 验证 Teammate 派发为每个请求创建全新的 Agent 实例。
    #[tokio::test]
    async fn teammate_dispatch_creates_fresh_agent_per_request() -> Result<()> {
        let (registry, executor) = make_executor().await;
        let mut definition = super::super::types::SubagentDefinition::new("explorer", "Explorer");
        definition.execution_mode = ExecutionMode::Teammate;
        registry
            .register(
                definition.clone(),
                Box::new(MockAgent::new("explorer").with_response("cached")),
            )
            .await;

        let creations = Arc::new(AtomicUsize::new(0));
        let factory_creations = Arc::clone(&creations);
        let factory = Arc::new(FnAgentFactory::new(move || {
            let instance = factory_creations
                .fetch_add(1, Ordering::SeqCst)
                .saturating_add(1);
            Box::pin(async move {
                Ok(Box::new(
                    MockAgent::new("explorer")
                        .with_delay_ms(50)
                        .with_response(format!("instance-{instance}")),
                ) as Box<dyn Agent>)
            })
        }));
        assert!(registry.register_factory_sync(definition, factory));

        let request = || DispatchRequest {
            agent_name: "explorer".to_string(),
            task: "inspect".to_string(),
            mode_override: Some(ExecutionMode::Teammate),
            cancel: CancellationToken::new(),
            parent_agent: "parent".to_string(),
            parent_context: None,
            delegation_policy: DispatchRequest::policy_from_depth(0),
            runtime_context: None,
            message: None,
            prompt_payload: None,
            prompt_context: None,
            constraints: Vec::new(),
            background: false,
        };

        let (first, second) =
            tokio::join!(executor.dispatch(request()), executor.dispatch(request()));
        let first = first?;
        let second = second?;

        assert_eq!(creations.load(Ordering::SeqCst), 2);
        assert_ne!(first.output, second.output);
        assert_ne!(first.output, "cached");
        assert_ne!(second.output, "cached");
        Ok(())
    }

    /// 验证 Teammate 超时会取消已分离的流生产者。
    #[tokio::test]
    async fn teammate_timeout_cancels_detached_stream_producer() -> Result<()> {
        let (registry, executor) = make_executor().await;
        let cancellation_seen = Arc::new(tokio::sync::Notify::new());
        let mut definition = super::super::types::SubagentDefinition::new("slow", "Slow stream");
        definition.execution_mode = ExecutionMode::Teammate;
        definition.timeout_secs = 1;
        registry
            .register(
                definition,
                Box::new(CancellationAwareStreamAgent {
                    cancellation_seen: Arc::clone(&cancellation_seen),
                }),
            )
            .await;

        let error = executor
            .dispatch(DispatchRequest {
                agent_name: "slow".to_string(),
                task: "wait".to_string(),
                mode_override: Some(ExecutionMode::Teammate),
                cancel: CancellationToken::new(),
                parent_agent: "parent".to_string(),
                parent_context: None,
                delegation_policy: DispatchRequest::policy_from_depth(0),
                runtime_context: None,
                message: None,
                prompt_payload: None,
                prompt_context: None,
                constraints: Vec::new(),
                background: false,
            })
            .await
            .err()
            .ok_or_else(|| ReactError::Other("slow Teammate unexpectedly completed".to_string()))?;

        assert_eq!(subagent_status_from_error(&error), SubagentStatus::TimedOut);
        tokio::time::timeout(Duration::from_secs(1), cancellation_seen.notified())
            .await
            .map_err(|_| {
                ReactError::Other(
                    "Teammate timeout did not cancel the detached stream producer".to_string(),
                )
            })?;
        Ok(())
    }

    /// 验证取消 Teammate 句柄不会取消父级。
    #[tokio::test]
    async fn teammate_handle_cancel_does_not_cancel_parent() -> Result<()> {
        let (registry, executor) = make_executor().await;
        let definition = super::super::types::SubagentDefinition::new("slow", "Slow stream");
        registry
            .register(
                definition,
                Box::new(CancellationAwareStreamAgent {
                    cancellation_seen: Arc::new(tokio::sync::Notify::new()),
                }),
            )
            .await;

        let parent_cancel = CancellationToken::new();
        let handle = executor
            .dispatch_teammate(DispatchRequest {
                agent_name: "slow".to_string(),
                task: "wait".to_string(),
                mode_override: Some(ExecutionMode::Teammate),
                cancel: parent_cancel.clone(),
                parent_agent: "parent".to_string(),
                parent_context: None,
                delegation_policy: DispatchRequest::policy_from_depth(0),
                runtime_context: None,
                message: None,
                prompt_payload: None,
                prompt_context: None,
                constraints: Vec::new(),
                background: false,
            })
            .await?;

        handle.cancel.cancel();
        if parent_cancel.is_cancelled() {
            return Err(ReactError::Other(
                "cancelling a Teammate handle cancelled its parent run".to_string(),
            ));
        }
        let error = handle
            .join()
            .await
            .err()
            .ok_or_else(|| ReactError::Other("cancelled Teammate completed".to_string()))?;
        assert!(
            matches!(error, ReactError::Agent(inner) if matches!(*inner, AgentError::Interrupted))
        );
        Ok(())
    }

    // ── 通用 Fork 隔离 ─────────────────────────────────────────────

    use crate::agent::subagent::{
        IsolationError, IsolationHandle, IsolationOutcome, IsolationProvider, IsolationRequest,
    };
    use std::sync::Mutex as StdMutex;

    /**
    模拟隔离工厂：`create` 总是成功并记录标签，`finalize` 返回预设的 diff；
    `should_fail` 用于切换为硬失败。
    */
    struct MockIsolationProvider {
        /// 记录每次 isolate 收到的标签。
        labels: StdMutex<Vec<String>>,
        /// 为 true 时 isolate 直接失败。
        should_fail: bool,
    }

    impl IsolationProvider for MockIsolationProvider {
        /// 按需失败；否则记录标签并返回带模拟 diff 的隔离句柄。
        fn isolate(
            &self,
            request: &IsolationRequest,
        ) -> std::result::Result<IsolationHandle, IsolationError> {
            if self.should_fail {
                return Err(IsolationError::new("mock isolation failed"));
            }
            let label = &request.label;
            self.labels.lock().unwrap().push(label.to_string());
            let path = std::path::PathBuf::from(format!("/tmp/mock-wt-{label}"));
            Ok(IsolationHandle {
                path,
                observed: ObservedIsolation::new(&request.kind),
                finalize: Box::new(|| {
                    Ok(IsolationOutcome {
                        summary: "=== mock diff ===\nfoo.rs | 1 +".to_string(),
                        artifacts: Vec::new(),
                        evidence: Vec::new(),
                    })
                }),
            })
        }
    }

    /// 构建一个在配置中接入了 worktree 工厂的执行器。
    fn make_executor_with_factory(
        provider: Arc<dyn IsolationProvider>,
    ) -> (Arc<SubagentRegistry>, SubagentExecutor) {
        let registry = Arc::new(SubagentRegistry::new());
        let executor = SubagentExecutor::new(
            registry.clone(),
            SubagentExecutorConfig {
                isolation_provider: Some(provider),
                ..SubagentExecutorConfig::default()
            },
        );
        (registry, executor)
    }

    /// 验证 Fork 隔离提供者绑定路径并追加结果摘要。
    #[tokio::test]
    async fn fork_provider_binds_path_and_appends_outcome() {
        let factory = Arc::new(MockIsolationProvider {
            labels: StdMutex::new(Vec::new()),
            should_fail: false,
        });
        let factory_obs: Arc<dyn IsolationProvider> = factory.clone();
        let (registry, executor) = make_executor_with_factory(factory_obs);

        let agent = MockAgent::new("writer").with_response("done");
        let def = super::super::types::SubagentDefinition {
            execution_mode: ExecutionMode::Fork,
            isolation: Some("sandbox".to_string()),
            ..super::super::types::SubagentDefinition::new("writer", "Writer")
        };
        registry.register(def, Box::new(agent.clone())).await;

        let req = DispatchRequest {
            agent_name: "writer".into(),
            task: "edit foo".into(),
            mode_override: None,
            cancel: CancellationToken::new(),
            parent_agent: "parent".into(),
            parent_context: None,
            delegation_policy: DispatchRequest::policy_from_depth(0),
            runtime_context: None,
            message: None,
            prompt_payload: None,
            prompt_context: None,
            constraints: Vec::new(),
            background: false,
        };

        let result = executor.dispatch(req).await.unwrap();
        /*
        输出 = 子 Agent 的回答 + 追加的 diff（追加的 diff 可以直接证明
        worktree 已被创建并完成收尾）。
        */
        assert!(result.output.contains("done"));
        assert!(result.output.contains("=== mock diff ==="));
        // 工厂被调用了一次，标签由 Agent 名称推导而来。
        let labels = factory.labels.lock().unwrap().clone();
        assert_eq!(labels.len(), 1);
        assert!(labels[0].starts_with("writer-"));
    }

    /// 验证 Fork worktree 优先使用稳定的隔离身份。
    #[tokio::test]
    async fn fork_worktree_prefers_stable_isolation_identity() -> std::result::Result<(), String> {
        let factory = Arc::new(MockIsolationProvider {
            labels: StdMutex::new(Vec::new()),
            should_fail: false,
        });
        let factory_for_executor: Arc<dyn IsolationProvider> = factory.clone();
        let (registry, executor) = make_executor_with_factory(factory_for_executor);
        let agent = MockAgent::new("writer").with_response("done");
        let definition = super::super::types::SubagentDefinition {
            execution_mode: ExecutionMode::Fork,
            isolation: Some("sandbox".to_string()),
            ..super::super::types::SubagentDefinition::new("writer", "Writer")
        };
        registry.register(definition, Box::new(agent)).await;

        executor
            .dispatch(DispatchRequest {
                agent_name: "writer".to_string(),
                task: "retry the same logical task".to_string(),
                mode_override: None,
                cancel: CancellationToken::new(),
                parent_agent: "parent".to_string(),
                parent_context: None,
                delegation_policy: DispatchRequest::policy_from_depth(0),
                runtime_context: Some(echo_core::tools::ExternalRunContext {
                    conversation_id: None,
                    run_id: Some("run-1".to_string()),
                    turn_id: None,
                    execution_id: Some("task-1:2".to_string()),
                    isolation_id: Some("run-1:task-1".to_string()),
                    message_id: None,
                    cancel: None,
                    trace_sink: None,
                    delegation_policy: None,
                    resource_guards: Vec::new(),
                    subagent_lineage: None,
                    uplink: None,
                }),
                message: None,
                prompt_payload: None,
                prompt_context: None,
                constraints: Vec::new(),
                background: false,
            })
            .await
            .map_err(|error| error.to_string())?;

        let labels = factory
            .labels
            .lock()
            .map_err(|error| error.to_string())?
            .clone();
        if labels != vec!["writer-run-1:task-1".to_string()] {
            return Err(format!("unexpected worktree labels: {labels:?}"));
        }
        Ok(())
    }

    /// 验证 worktree 观测记录先于子 Agent 完成。
    #[tokio::test]
    async fn fork_worktree_observation_precedes_subagent_completion()
    -> std::result::Result<(), String> {
        let factory = Arc::new(MockIsolationProvider {
            labels: StdMutex::new(Vec::new()),
            should_fail: false,
        });
        let factory_obs: Arc<dyn IsolationProvider> = factory;
        let (registry, executor) = make_executor_with_factory(factory_obs);
        let agent = MockAgent::new("writer").with_response("done");
        let def = super::super::types::SubagentDefinition {
            execution_mode: ExecutionMode::Fork,
            isolation: Some("sandbox".to_string()),
            ..super::super::types::SubagentDefinition::new("writer", "Writer")
        };
        registry.register(def, Box::new(agent)).await;
        let mut events = registry.event_bus().subscribe();
        let req = DispatchRequest {
            agent_name: "writer".into(),
            task: "edit foo".into(),
            mode_override: None,
            cancel: CancellationToken::new(),
            parent_agent: "parent".into(),
            parent_context: None,
            delegation_policy: DispatchRequest::policy_from_depth(0),
            runtime_context: None,
            message: None,
            prompt_payload: None,
            prompt_context: None,
            constraints: Vec::new(),
            background: false,
        };

        let result = executor
            .dispatch(req)
            .await
            .map_err(|error| error.to_string())?;
        if result.isolation_observed != ObservedIsolation::new("sandbox") {
            return Err(format!(
                "expected worktree result observation, got {:?}",
                result.isolation_observed
            ));
        }
        let started = events.recv().await.map_err(|error| error.to_string())?;
        let observed = events.recv().await.map_err(|error| error.to_string())?;
        let completed = events.recv().await.map_err(|error| error.to_string())?;
        if !matches!(started.as_ref(), SubagentEvent::DispatchStarted { .. }) {
            return Err(format!("first event was not started: {started:?}"));
        }
        if !matches!(
            observed.as_ref(),
            SubagentEvent::DispatchIsolationObserved { isolation, .. }
                if isolation.as_str() == "sandbox"
        ) {
            return Err(format!(
                "second event was not worktree observation: {observed:?}"
            ));
        }
        if !matches!(completed.as_ref(), SubagentEvent::DispatchCompleted { .. }) {
            return Err(format!("third event was not completed: {completed:?}"));
        }
        Ok(())
    }

    /// 验证并发 Fork 各自携带不同的 worktree 调用。
    #[tokio::test]
    async fn concurrent_forks_carry_distinct_worktree_invocations()
    -> std::result::Result<(), String> {
        let factory = Arc::new(MockIsolationProvider {
            labels: StdMutex::new(Vec::new()),
            should_fail: false,
        });
        let factory_obs: Arc<dyn IsolationProvider> = factory;
        let (registry, executor) = make_executor_with_factory(factory_obs);
        let agent = MockAgent::new("writer").with_responses(["done-a", "done-b"]);
        let def = super::super::types::SubagentDefinition {
            execution_mode: ExecutionMode::Fork,
            isolation: Some("sandbox".to_string()),
            ..super::super::types::SubagentDefinition::new("writer", "Writer")
        };
        registry.register(def, Box::new(agent.clone())).await;
        let executor = Arc::new(executor);
        let request = |run_id: &str| DispatchRequest {
            agent_name: "writer".into(),
            task: format!("edit for {run_id}"),
            mode_override: None,
            cancel: CancellationToken::new(),
            parent_agent: "parent".into(),
            parent_context: None,
            delegation_policy: DispatchRequest::policy_from_depth(0),
            runtime_context: Some(echo_core::tools::ExternalRunContext {
                conversation_id: None,
                run_id: Some(run_id.to_string()),
                turn_id: None,
                execution_id: Some(format!("execution-{run_id}")),
                isolation_id: None,
                message_id: None,
                cancel: None,
                trace_sink: None,
                delegation_policy: None,
                resource_guards: Vec::new(),
                subagent_lineage: None,
                uplink: None,
            }),
            message: None,
            prompt_payload: None,
            prompt_context: None,
            constraints: Vec::new(),
            background: false,
        };

        let first = {
            let executor = Arc::clone(&executor);
            async move { executor.dispatch(request("run-a")).await }
        };
        let second = {
            let executor = Arc::clone(&executor);
            async move { executor.dispatch(request("run-b")).await }
        };
        let (first_result, second_result) = tokio::join!(first, second);
        first_result.map_err(|error| error.to_string())?;
        second_result.map_err(|error| error.to_string())?;

        let invocations = agent.invocation_contexts();
        if invocations.len() != 2 {
            return Err(format!(
                "expected two value-scoped invocations, got {invocations:?}"
            ));
        }
        for invocation in invocations {
            let runtime = invocation
                .runtime
                .ok_or_else(|| "fork invocation missing runtime context".to_string())?;
            let run_id = runtime
                .run_id
                .ok_or_else(|| "fork invocation missing run id".to_string())?;
            let working_dir = invocation
                .working_dir
                .ok_or_else(|| format!("{run_id} missing worktree path"))?;
            let path = working_dir.to_string_lossy();
            if !path.contains(run_id.as_str()) {
                return Err(format!(
                    "cross-worktree invocation: run {} received {}",
                    run_id, path
                ));
            }
        }
        Ok(())
    }

    /// 验证隔离提供者失败会使派发失败。
    #[tokio::test]
    async fn fork_isolation_provider_failure_fails_dispatch() {
        // 工厂硬失败 → 派发必须失败，绝不能在无隔离的情况下运行。
        let factory = Arc::new(MockIsolationProvider {
            labels: StdMutex::new(Vec::new()),
            should_fail: true,
        });
        let factory_obs: Arc<dyn IsolationProvider> = factory;
        let (registry, executor) = make_executor_with_factory(factory_obs);

        let agent = MockAgent::new("writer").with_response("done");
        let def = super::super::types::SubagentDefinition {
            execution_mode: ExecutionMode::Fork,
            isolation: Some("sandbox".to_string()),
            ..super::super::types::SubagentDefinition::new("writer", "Writer")
        };
        registry.register(def, Box::new(agent.clone())).await;

        let req = DispatchRequest {
            agent_name: "writer".into(),
            task: "edit foo".into(),
            mode_override: None,
            cancel: CancellationToken::new(),
            parent_agent: "parent".into(),
            parent_context: None,
            delegation_policy: DispatchRequest::policy_from_depth(0),
            runtime_context: None,
            message: None,
            prompt_payload: None,
            prompt_context: None,
            constraints: Vec::new(),
            background: false,
        };

        let err = executor.dispatch(req).await.unwrap_err();
        assert!(
            err.to_string().contains("Isolation 'sandbox'"),
            "got: {err}"
        );
        /*
        派发硬失败（安全闸门）——绝不会在无隔离的情况下悄悄继续。错误信息
        本身就是证据；我们不对 MockAgent 记录的 working_dir_calls 做断言，
        因为注册表把 Agent 存放在 Arc<dyn Agent> 之后，跨克隆边界共享的
        记录状态在这里无法可靠地观察到。
        */
        let _ = agent; // 抑制未使用变量警告
    }

    /// 验证请求隔离但未配置提供者时硬失败。
    #[tokio::test]
    async fn fork_isolate_without_factory_hard_fails() {
        // 请求了隔离边界却没有提供者时必须硬失败。
        let (registry, executor) = make_executor().await; // 默认配置，无工厂

        let agent = MockAgent::new("writer").with_response("done");
        let def = super::super::types::SubagentDefinition {
            execution_mode: ExecutionMode::Fork,
            isolation: Some("sandbox".to_string()),
            ..super::super::types::SubagentDefinition::new("writer", "Writer")
        };
        registry.register(def, Box::new(agent.clone())).await;

        let req = DispatchRequest {
            agent_name: "writer".into(),
            task: "edit foo".into(),
            mode_override: None,
            cancel: CancellationToken::new(),
            parent_agent: "parent".into(),
            parent_context: None,
            delegation_policy: DispatchRequest::policy_from_depth(0),
            runtime_context: None,
            message: None,
            prompt_payload: None,
            prompt_context: None,
            constraints: Vec::new(),
            background: false,
        };

        let err = executor.dispatch(req).await.unwrap_err();
        assert!(
            err.to_string().contains("no IsolationProvider")
                || err
                    .to_string()
                    .contains("refusing to run without isolation"),
            "got: {err}"
        );
        let _ = agent;
    }

    /// 验证未请求隔离的 Fork 不会调用提供者。
    #[tokio::test]
    async fn fork_without_isolation_does_not_invoke_provider() {
        // 未请求隔离的子 Agent 永远不会调用提供者。
        let factory = Arc::new(MockIsolationProvider {
            labels: StdMutex::new(Vec::new()),
            should_fail: false,
        });
        let factory_obs: Arc<dyn IsolationProvider> = factory.clone();
        let (registry, executor) = make_executor_with_factory(factory_obs);

        let agent = MockAgent::new("reader").with_response("ok");
        let def = super::super::types::SubagentDefinition {
            execution_mode: ExecutionMode::Fork,
            ..super::super::types::SubagentDefinition::new("reader", "Reader")
        };
        registry.register(def, Box::new(agent.clone())).await;

        let req = DispatchRequest {
            agent_name: "reader".into(),
            task: "read foo".into(),
            mode_override: None,
            cancel: CancellationToken::new(),
            parent_agent: "parent".into(),
            parent_context: None,
            delegation_policy: DispatchRequest::policy_from_depth(0),
            runtime_context: None,
            message: None,
            prompt_payload: None,
            prompt_context: None,
            constraints: Vec::new(),
            background: false,
        };

        let result = executor.dispatch(req).await.unwrap();
        assert_eq!(result.output, "ok");
        assert_eq!(result.isolation_observed, ObservedIsolation::new("context"));
        // 工厂从未被调用——只读子 Agent 不请求隔离。
        assert!(factory.labels.lock().unwrap().is_empty());
        let _ = agent;
    }

    // 第二个提供者用于证明框架对隔离类型是不透明处理的。

    /// 第二个模拟提供者，配合另一种不透明的隔离类型使用。
    struct MockWorkspaceFactory {
        /// 记录每次 isolate 收到的标签。
        labels: StdMutex<Vec<String>>,
        /// 为 true 时 isolate 直接失败。
        should_fail: bool,
    }

    impl IsolationProvider for MockWorkspaceFactory {
        /// 按需失败；否则记录标签并返回带模拟产物摘要的工作区句柄。
        fn isolate(
            &self,
            request: &IsolationRequest,
        ) -> std::result::Result<IsolationHandle, IsolationError> {
            if self.should_fail {
                return Err(IsolationError::new("mock workspace create failed"));
            }
            let label = &request.label;
            self.labels.lock().unwrap().push(label.to_string());
            Ok(IsolationHandle {
                path: std::path::PathBuf::from(format!("/tmp/mock-ws-{label}")),
                observed: ObservedIsolation::new(&request.kind),
                finalize: Box::new(|| {
                    Ok(IsolationOutcome {
                        summary: "run_001_clean.parquet\nrun_001_stats.json".to_string(),
                        artifacts: Vec::new(),
                        evidence: Vec::new(),
                    })
                }),
            })
        }
    }

    /// 构建一个在配置中接入了数据工作区工厂的执行器。
    fn make_executor_with_workspace_factory(
        provider: Arc<dyn IsolationProvider>,
    ) -> (Arc<SubagentRegistry>, SubagentExecutor) {
        let registry = Arc::new(SubagentRegistry::new());
        let executor = SubagentExecutor::new(
            registry.clone(),
            SubagentExecutorConfig {
                isolation_provider: Some(provider),
                ..SubagentExecutorConfig::default()
            },
        );
        (registry, executor)
    }

    /// 验证任意隔离类型会追加提供者摘要。
    #[tokio::test]
    async fn fork_arbitrary_isolation_kind_appends_provider_summary() {
        let factory = Arc::new(MockWorkspaceFactory {
            labels: StdMutex::new(Vec::new()),
            should_fail: false,
        });
        let factory_obs: Arc<dyn IsolationProvider> = factory.clone();
        let (registry, executor) = make_executor_with_workspace_factory(factory_obs);

        let agent = MockAgent::new("analyst").with_response("analysis done");
        let def = super::super::types::SubagentDefinition {
            execution_mode: ExecutionMode::Fork,
            isolation: Some("workspace".to_string()),
            ..super::super::types::SubagentDefinition::new("analyst", "Analyst")
        };
        registry.register(def, Box::new(agent)).await;

        let req = DispatchRequest {
            agent_name: "analyst".into(),
            task: "analyze data".into(),
            mode_override: None,
            cancel: CancellationToken::new(),
            parent_agent: "parent".into(),
            parent_context: None,
            delegation_policy: DispatchRequest::policy_from_depth(0),
            runtime_context: None,
            message: None,
            prompt_payload: None,
            prompt_context: None,
            constraints: Vec::new(),
            background: false,
        };

        let result = executor.dispatch(req).await.unwrap();
        assert!(result.output.contains("analysis done"));
        assert!(result.output.contains("run_001_clean.parquet"));
        assert!(result.output.contains("--- isolation outcome ---"));
        assert_eq!(
            result.isolation_observed,
            ObservedIsolation::new("workspace")
        );
        // 工厂被调用了一次，标签由 Agent 名称推导而来。
        let labels = factory.labels.lock().unwrap().clone();
        assert_eq!(labels.len(), 1);
        assert!(labels[0].starts_with("analyst-"));
    }

    /// 验证任意隔离类型会传播提供者失败。
    #[tokio::test]
    async fn fork_arbitrary_isolation_kind_propagates_provider_failure() {
        // 工作区工厂硬失败 → 派发失败（安全闸门）。
        let factory = Arc::new(MockWorkspaceFactory {
            labels: StdMutex::new(Vec::new()),
            should_fail: true,
        });
        let factory_obs: Arc<dyn IsolationProvider> = factory;
        let (registry, executor) = make_executor_with_workspace_factory(factory_obs);

        let agent = MockAgent::new("analyst").with_response("ok");
        let def = super::super::types::SubagentDefinition {
            execution_mode: ExecutionMode::Fork,
            isolation: Some("workspace".to_string()),
            ..super::super::types::SubagentDefinition::new("analyst", "Analyst")
        };
        registry.register(def, Box::new(agent)).await;

        let req = DispatchRequest {
            agent_name: "analyst".into(),
            task: "analyze".into(),
            mode_override: None,
            cancel: CancellationToken::new(),
            parent_agent: "parent".into(),
            parent_context: None,
            delegation_policy: DispatchRequest::policy_from_depth(0),
            runtime_context: None,
            message: None,
            prompt_payload: None,
            prompt_context: None,
            constraints: Vec::new(),
            background: false,
        };

        let err = executor.dispatch(req).await.unwrap_err();
        assert!(
            err.to_string().contains("Isolation 'workspace'"),
            "got: {err}"
        );
    }

    /// 验证只使用配置的那一个提供者。
    #[tokio::test]
    async fn fork_uses_only_the_configured_provider() {
        let ws_factory = Arc::new(MockWorkspaceFactory {
            labels: StdMutex::new(Vec::new()),
            should_fail: false,
        });
        let wt_factory = Arc::new(MockIsolationProvider {
            labels: StdMutex::new(Vec::new()),
            should_fail: false,
        });
        let registry = Arc::new(SubagentRegistry::new());
        let executor = SubagentExecutor::new(
            registry.clone(),
            SubagentExecutorConfig {
                isolation_provider: Some(wt_factory.clone()),
                ..SubagentExecutorConfig::default()
            },
        );

        let agent = MockAgent::new("w").with_response("done");
        let def = super::super::types::SubagentDefinition {
            execution_mode: ExecutionMode::Fork,
            isolation: Some("sandbox".to_string()),
            ..super::super::types::SubagentDefinition::new("w", "Writer")
        };
        registry.register(def, Box::new(agent)).await;

        let req = DispatchRequest {
            agent_name: "w".into(),
            task: "work".into(),
            mode_override: None,
            cancel: CancellationToken::new(),
            parent_agent: "parent".into(),
            parent_context: None,
            delegation_policy: DispatchRequest::policy_from_depth(0),
            runtime_context: None,
            message: None,
            prompt_payload: None,
            prompt_context: None,
            constraints: Vec::new(),
            background: false,
        };

        let result = executor.dispatch(req).await.unwrap();
        // 使用了 worktree（追加了模拟 diff），而没有使用 workspace。
        assert!(result.output.contains("=== mock diff ==="));
        assert!(
            ws_factory.labels.lock().unwrap().is_empty(),
            "workspace factory must not be invoked when worktree is active"
        );
        assert!(
            !wt_factory.labels.lock().unwrap().is_empty(),
            "worktree factory should have been invoked"
        );
    }
}
