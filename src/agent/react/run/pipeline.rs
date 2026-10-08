/*!
 * 工具执行流水线 —— 用于处理工具调用的可组合阶段。
 *
 * 用 `ToolExecutionPipeline` 协调的一组离散内部阶段构成的可配置流水线，
 * 取代了原先单体式的 `execute_tool_feedback_raw`。
 */

use super::context::HookMessageBatches;
use crate::agent::{ToolInvocation, ToolInvocationRewrite};
use crate::error::{ReactError, Result};
use crate::tools::{ToolParameters, ToolResult, ToolStreamEvent, is_write_tool};
use async_trait::async_trait;
use echo_core::tools::permission::{PermissionDecision, PermissionMode};
use serde_json::Value;
use tokio::sync::mpsc;
use tracing::debug;

/*
 * 完整的工具结果保持内联存放，这样流水线可以复用同一个带类型的流事件，
 * 而不必为每个流式条目额外分配内存。
 */
#[allow(clippy::large_enum_variant)]
/** 工具流水线内部事件：既承载“调用已发出”的标准事件，也承载工具执行期间的流式增量事件。 */
pub(crate) enum ToolPipelineEvent {
    /// 一次工具调用的规范化调用事件（模型请求名/参数与实际执行名/参数）。
    Invocation {
        /// 本次调用的唯一 ID。
        call_id: String,
        /// 规范化后的调用信息（请求值与生效值、改写记录等）。
        invocation: ToolInvocation,
    },
    /// 工具执行期间产生的增量流事件（例如流式输出片段）。
    Stream {
        /// 所属调用的唯一 ID。
        call_id: String,
        /// 工具名称。
        name: String,
        /// 具体的流事件内容。
        event: ToolStreamEvent,
    },
}

// ── ToolExecutionContext ────────────────────────────────────────────

/// 在流水线各阶段间流转的可变上下文。
pub(crate) struct ToolExecutionContext {
    /// 唯一调用 ID，用于关联规范化的 ToolCall 与带类型的 ToolResult。
    pub call_id: String,
    /// 模型最初请求的工具名。
    pub requested_tool_name: String,
    /// 模型最初请求的参数。
    pub requested_input: Value,
    /// 实际正在执行的工具名称。
    pub tool_name: String,
    /// 解析后的工具参数。
    pub params: ToolParameters,
    /// 来自 LLM 的原始 JSON 输入。
    pub input: Value,
    /// 由钩子（pre/post）累积的消息。
    pub hook_messages: HookMessageBatches,
    /// 由 Execute 阶段填充。
    pub result: Option<ToolResult>,
    /// 最终输出字符串（经过输出防护与截断处理后）。
    pub output: Option<String>,
    /// 当失败且没有工具输出时，用于审计的经防护处理的诊断信息。
    pub audit_error_output: Option<String>,
    /// 是否有某个阶段阻止了本次执行。
    pub blocked: bool,
    /// 阻止原因（若被阻止）。
    pub block_reason: Option<String>,
    /// 工具尚未开始执行前即被阻止时的结构化终态事实。
    pub block_failure: Option<crate::tools::ToolFailure>,
    /// 执行耗时（毫秒）。
    pub duration_ms: u64,
    /// 是否处于计划模式（仅允许只读工具）。
    pub plan_mode: bool,
    /// 仅对本次调用生效的、由 PreToolUse 钩子返回的权限决策。
    pub permission_decision: Option<PermissionDecision>,
    /// 仅对本次调用生效的、由钩子返回的权限模式覆盖值。
    pub permission_mode_override: Option<PermissionMode>,
    /// 执行前应用的策略改写记录（按顺序保留来源）。
    pub rewrites: Vec<ToolInvocationRewrite>,
    /// 规范化的调用事件是否已经发出。
    pub invocation_emitted: bool,
    /// 回调是否已经观察到本次调用的开始。
    pub callback_started: bool,
    /// 在工具返回结果之前使执行中断的执行错误。
    pub interrupted_execution_error: Option<ReactError>,
    /// 增量工具事件通道，事件均带有稳定的调用身份标记。
    pub stream_tx: Option<mpsc::Sender<ToolPipelineEvent>>,
}

impl ToolExecutionContext {
    /// 将本次执行标记为已阻止，记录阻止原因与结构化失败信息，并把输出设为阻止原因文本。
    fn block(&mut self, category: crate::tools::ToolFailureCategory, reason: String) {
        self.blocked = true;
        self.block_reason = Some(reason.clone());
        self.block_failure = Some(crate::tools::ToolFailure::new(category));
        self.output = Some(reason);
    }

    /// 用改写后的参数替换当前输入；仅当输入确有变化时才记录一条改写来源。
    fn replace_input(&mut self, input: Value, rewrite: ToolInvocationRewrite) -> Result<()> {
        let Value::Object(map) = &input else {
            return Err(ReactError::Other(format!(
                "Tool '{}' rewrite returned non-object arguments",
                self.tool_name
            )));
        };
        if input != self.input {
            self.rewrites.push(rewrite);
        }
        self.params = map
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect();
        self.input = input;
        Ok(())
    }

    /// 发出规范化的调用事件（若尚未发出）；必要时先补全 call_id。
    async fn emit_invocation(&mut self) -> Result<()> {
        if self.call_id.is_empty() {
            self.call_id = format!("call_{}", uuid::Uuid::new_v4());
        }
        if self.invocation_emitted {
            return Ok(());
        }
        if let Some(stream_tx) = self.stream_tx.as_ref() {
            stream_tx
                .send(ToolPipelineEvent::Invocation {
                    call_id: self.call_id.clone(),
                    invocation: ToolInvocation {
                        requested_name: self.requested_tool_name.clone(),
                        requested_args: self.requested_input.clone(),
                        name: self.tool_name.clone(),
                        args: self.input.clone(),
                        rewrites: self.rewrites.clone(),
                    },
                })
                .await
                .map_err(|_| ReactError::Other("Tool event receiver closed".to_string()))?;
        }
        self.invocation_emitted = true;
        Ok(())
    }
}

// ── PipelineStage 特征 ────────────────────────────────────────────

/**
 * 工具执行流水线中的单个阶段。
 *
 * 每个阶段接收一个可变的 [`ToolExecutionContext`] 并可以修改它。
 * 返回 `Err` 会使流水线短路（视为工具执行失败）。
 * 设置 `ctx.blocked = true` 会导致后续阶段被跳过。
 */
#[async_trait]
pub(crate) trait PipelineStage: Send + Sync {
    /// 用于日志和调试的可读名称。
    fn name(&self) -> &str;

    /// 执行该阶段。
    async fn run(
        &self,
        ctx: &mut ToolExecutionContext,
        snapshot: &crate::agent::snapshot::AgentRunSnapshot,
    ) -> Result<()>;

    /**
     * 该阶段在 `ctx.blocked` 被设置之后是否仍需运行。
     *
     * 执行门控类阶段一如既往会终止流水线，但记录“已发生事实”的观察类阶段
     * （如追踪、终态回调）可以选择继续运行，从而让被阻止的调用也留下如实的
     * 记录（#102）。
     */
    fn runs_after_block(&self) -> bool {
        false
    }
}

// ── 各阶段实现 ──────────────────────────────────────────

/**
 * 干预回调阶段 —— 在所有其他阶段之前检查干预回调。
 *
 * 这是优先级最高的决策点。干预回调可以阻止、取消、重定向、修改参数，
 * 或注入上下文。
 */
pub struct InterventionStage;

#[async_trait]
impl PipelineStage for InterventionStage {
    /// 返回该阶段的名称。
    fn name(&self) -> &str {
        "intervention"
    }

    /// 执行干预回调，按其决策阻止、重定向、修改参数或注入上下文。
    async fn run(
        &self,
        ctx: &mut ToolExecutionContext,
        snapshot: &crate::agent::snapshot::AgentRunSnapshot,
    ) -> Result<()> {
        for intervention in &snapshot.tools.intervention_callbacks {
            let result = intervention
                .on_tool_call(&snapshot.config.agent_name, &ctx.tool_name, &ctx.input)
                .await;
            if result.cancel {
                return Err(ReactError::Other(format!(
                    "Agent execution cancelled by intervention: tool {}",
                    ctx.tool_name
                )));
            }
            if result.block {
                let reason = result
                    .block_reason
                    .unwrap_or_else(|| "blocked by intervention callback".into());
                ctx.block(
                    crate::tools::ToolFailureCategory::Permanent,
                    format!("Tool {} blocked by intervention: {}", ctx.tool_name, reason),
                );
                return Ok(());
            }
            if let Some(redirect) = result.redirect_to {
                if redirect != ctx.tool_name {
                    ctx.rewrites
                        .push(ToolInvocationRewrite::InterventionRedirect);
                }
                ctx.tool_name = redirect;
            }
            if let Some(modified) = result.modified_args {
                ctx.replace_input(modified, ToolInvocationRewrite::InterventionArguments)?;
            }
            if let Some(injected) = result.injected_context {
                ctx.hook_messages.pre.push(injected);
            }
        }
        Ok(())
    }
}

/// 在执行时强制生效本次调用范围内的工具可见性约束。
pub struct ToolVisibilityStage;

#[async_trait]
impl PipelineStage for ToolVisibilityStage {
    /// 返回该阶段的名称。
    fn name(&self) -> &str {
        "tool_visibility"
    }

    /// 工具被禁用或在当前调用范围内未激活时阻止调用。
    async fn run(
        &self,
        ctx: &mut ToolExecutionContext,
        snapshot: &crate::agent::snapshot::AgentRunSnapshot,
    ) -> Result<()> {
        if snapshot.tools.disabled_tools.contains(&ctx.tool_name) {
            snapshot.tools.tool_manager.record_tool_selection_failure();
            ctx.block(
                crate::tools::ToolFailureCategory::Unavailable,
                format!(
                    "Tool '{}' is not available in this invocation",
                    ctx.tool_name
                ),
            );
        } else if snapshot
            .tools
            .visibility
            .as_ref()
            .is_some_and(|visibility| !visibility.is_visible(&ctx.tool_name))
        {
            snapshot.tools.tool_manager.record_tool_selection_failure();
            ctx.block(
                crate::tools::ToolFailureCategory::Unavailable,
                format!(
                    "Tool '{}' is not activated in this invocation; use tool_search first",
                    ctx.tool_name
                ),
            );
        }
        Ok(())
    }
}

/// 运行 PreToolUse 钩子。
pub struct PreToolUseHookStage;

#[async_trait]
impl PipelineStage for PreToolUseHookStage {
    /// 返回该阶段的名称。
    fn name(&self) -> &str {
        "pre_tool_use_hook"
    }

    /// 运行 PreToolUse 钩子，并应用其阻止或参数改写结果。
    async fn run(
        &self,
        ctx: &mut ToolExecutionContext,
        snapshot: &crate::agent::snapshot::AgentRunSnapshot,
    ) -> Result<()> {
        let has_hooks = {
            let hook_reg = snapshot.tools.hook_registry.read().await;
            !hook_reg.is_empty()
        };
        if !has_hooks {
            return Ok(());
        }

        let hook_reg = {
            let guard = snapshot.tools.hook_registry.read().await;
            guard.clone()
        };
        let hook_result = hook_reg
            .run_pre_tool_use(
                &ctx.tool_name,
                &ctx.input,
                snapshot.config.session_id.as_deref().unwrap_or(""),
            )
            .await;
        ctx.hook_messages.pre = hook_result.messages.clone();
        ctx.permission_decision = hook_result.permission_decision.clone();
        ctx.permission_mode_override = hook_result.permission_mode_override;

        if hook_result.block {
            let reason = hook_result
                .block_reason
                .unwrap_or_else(|| format!("Tool {} blocked by hook", ctx.tool_name));
            snapshot
                .record_permission_decision(
                    &ctx.tool_name,
                    &PermissionDecision::Deny {
                        reason: reason.clone(),
                    },
                    "pre_tool_use_hook",
                )
                .await;
            ctx.block(crate::tools::ToolFailureCategory::Permanent, reason);
            return Ok(());
        }

        if let Some(updated) = hook_result.updated_input {
            ctx.replace_input(updated, ToolInvocationRewrite::PreToolUseHook)?;
        }
        Ok(())
    }
}

impl crate::agent::snapshot::AgentRunSnapshot {
    /// 将一次权限决策记录为追踪事件，携带工具名、决策结果与来源说明。
    pub(crate) async fn record_permission_decision(
        &self,
        tool: &str,
        decision: &PermissionDecision,
        source: &str,
    ) {
        let (decision, reason) = match decision {
            PermissionDecision::Allow => ("allow", source.to_string()),
            PermissionDecision::Deny { reason } => ("deny", format!("{source}: {reason}")),
            PermissionDecision::RequireApproval => ("ask", format!("{source}: requires approval")),
            PermissionDecision::Ask { .. } => ("ask", format!("{source}: requires user input")),
        };
        self.record_event(crate::trace::RunEvent::PermissionDecision {
            tool: tool.to_string(),
            decision: decision.to_string(),
            reason,
        })
        .await;
    }
}

/// 检查工具审批（PermissionService）。
pub struct PermissionStage;

#[async_trait]
impl PipelineStage for PermissionStage {
    /// 返回该阶段的名称。
    fn name(&self) -> &str {
        "permission"
    }

    /// 通过 PermissionService 检查工具审批，拒绝时阻止调用。
    async fn run(
        &self,
        ctx: &mut ToolExecutionContext,
        snapshot: &crate::agent::snapshot::AgentRunSnapshot,
    ) -> Result<()> {
        if let Some(decision) = ctx.permission_decision.take() {
            snapshot
                .record_permission_decision(&ctx.tool_name, &decision, "pre_tool_use_hook")
                .await;
            match decision {
                PermissionDecision::Allow => return Ok(()),
                PermissionDecision::Deny { reason } => {
                    ctx.block(crate::tools::ToolFailureCategory::Permanent, reason);
                    return Ok(());
                }
                PermissionDecision::Ask { .. } | PermissionDecision::RequireApproval => {}
            }
        }

        let permission_hook = {
            let registry = snapshot.tools.hook_registry.read().await.clone();
            let context = crate::skills::hooks::HookContext::for_permission_request(
                &ctx.tool_name,
                &ctx.input,
                snapshot.config.session_id.as_deref().unwrap_or(""),
                &snapshot.config.agent_name,
            );
            registry.run_lifecycle_hooks(&context).await
        };
        ctx.hook_messages.pre.extend(permission_hook.messages);
        if permission_hook.block {
            let reason = permission_hook
                .block_reason
                .unwrap_or_else(|| format!("Tool {} blocked by permission hook", ctx.tool_name));
            snapshot
                .record_permission_decision(
                    &ctx.tool_name,
                    &PermissionDecision::Deny {
                        reason: reason.clone(),
                    },
                    "permission_request_hook",
                )
                .await;
            ctx.block(crate::tools::ToolFailureCategory::Permanent, reason);
            return Ok(());
        }
        if let Some(mode) = permission_hook.permission_mode_override {
            ctx.permission_mode_override = Some(mode);
        }
        if let Some(decision) = permission_hook.permission_decision {
            snapshot
                .record_permission_decision(&ctx.tool_name, &decision, "permission_request_hook")
                .await;
            match decision {
                PermissionDecision::Allow => return Ok(()),
                PermissionDecision::Deny { reason } => {
                    ctx.block(crate::tools::ToolFailureCategory::Permanent, reason);
                    return Ok(());
                }
                PermissionDecision::Ask { .. } | PermissionDecision::RequireApproval => {}
            }
        }

        #[cfg(feature = "human-loop")]
        let approval_modified = snapshot
            .check_tool_approval(
                &ctx.call_id,
                &ctx.tool_name,
                &ctx.input,
                ctx.permission_mode_override,
            )
            .await
            .map_err(|error| ReactError::Other(error.to_string()))?;

        #[cfg(not(feature = "human-loop"))]
        let approval_modified = snapshot
            .check_tool_approval(
                &ctx.call_id,
                &ctx.tool_name,
                &ctx.input,
                ctx.permission_mode_override,
            )
            .await
            .map_err(|_| ReactError::Other("Permission check failed".into()))?;

        if let Some(modified) = approval_modified {
            ctx.replace_input(modified, ToolInvocationRewrite::Approval)?;
        }
        Ok(())
    }
}

/// 强制执行“先读后写”策略。
pub struct ReadBeforeEditStage;

#[async_trait]
impl PipelineStage for ReadBeforeEditStage {
    /// 返回该阶段的名称。
    fn name(&self) -> &str {
        "read_before_edit"
    }

    /// 对写类工具检查目标路径是否已先被读取，未读取则阻止。
    async fn run(
        &self,
        ctx: &mut ToolExecutionContext,
        snapshot: &crate::agent::snapshot::AgentRunSnapshot,
    ) -> Result<()> {
        if !snapshot.config.force_read_before_edit {
            return Ok(());
        }
        if !is_write_tool(&ctx.tool_name) {
            return Ok(());
        }
        let paths = extract_path_params(&ctx.tool_name, &ctx.params);
        let ttl = std::time::Duration::from_secs(30 * 60);
        let mut files = snapshot
            .recently_read_files
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        for path in paths {
            let canonical = resolve_tracking_path(&path, snapshot.config.working_dir.as_deref());
            let read = match files.get(&canonical) {
                Some(instant) if instant.elapsed() < ttl => true,
                Some(_) => {
                    files.remove(&canonical);
                    false
                }
                None => false,
            };
            if !read {
                ctx.block(crate::tools::ToolFailureCategory::Unavailable, format!(
                    "Read-before-edit is enabled. File '{path}' has not been read. Use read_file first."
                ));
                break;
            }
        }
        Ok(())
    }
}

/// 检查基于技能的工具权限（skill_allowed_tools 白名单）。
pub struct SkillPermissionStage;

#[async_trait]
impl PipelineStage for SkillPermissionStage {
    /// 返回该阶段的名称。
    fn name(&self) -> &str {
        "skill_permission"
    }

    /// 按技能的工具白名单检查权限，不在白名单内则阻止。
    async fn run(
        &self,
        ctx: &mut ToolExecutionContext,
        snapshot: &crate::agent::snapshot::AgentRunSnapshot,
    ) -> Result<()> {
        // 检查是否有已激活的技能对工具做了限制
        if !snapshot.tools.is_skill_tool_allowed(&ctx.tool_name) {
            let reason = format!(
                "Tool '{}' is not permitted by the activated skill's allowed_tools whitelist",
                ctx.tool_name
            );
            snapshot
                .record_permission_decision(
                    &ctx.tool_name,
                    &PermissionDecision::Deny {
                        reason: reason.clone(),
                    },
                    "skill_allowed_tools",
                )
                .await;
            ctx.block(crate::tools::ToolFailureCategory::Unavailable, reason);
        }
        Ok(())
    }
}

/// 将已落定的工具结果记录到配置的审计日志中。
pub struct AuditStage;

#[async_trait]
impl PipelineStage for AuditStage {
    /// 返回该阶段的名称。
    fn name(&self) -> &str {
        "audit"
    }

    /// 被阻止的调用同样需要运行该阶段。
    fn runs_after_block(&self) -> bool {
        true
    }

    /// 将已落定的工具结果写入审计日志。
    async fn run(
        &self,
        ctx: &mut ToolExecutionContext,
        snapshot: &crate::agent::snapshot::AgentRunSnapshot,
    ) -> Result<()> {
        if let (Some(_), Some(result)) = (&snapshot.guard.audit_logger, &ctx.result) {
            /*
             * post-use 钩子可能会拒绝一个已经发生的效果。
             * 这里保留该输出，同时上报最终的失败状态。
             */
            let output = if !result.success && result.output.is_empty() {
                ctx.audit_error_output.as_deref().unwrap_or("")
            } else {
                ctx.output.as_deref().unwrap_or(&result.output)
            };
            let ev = crate::audit::AuditEvent::now(
                snapshot.config.session_id.clone(),
                snapshot.config.agent_name.clone(),
                crate::audit::AuditEventType::ToolCall {
                    call_id: Some(ctx.call_id.clone()),
                    tool: ctx.tool_name.clone(),
                    input: ctx.input.clone(),
                    output: output.to_string(),
                    success: result.success,
                    duration_ms: ctx.duration_ms,
                },
            );
            snapshot.record_audit_event(ev).await;
        }
        Ok(())
    }
}

/// 在执行前发布唯一规范化的“请求/生效”调用记录。
pub struct InvocationStage;

#[async_trait]
impl PipelineStage for InvocationStage {
    /// 返回该阶段的名称。
    fn name(&self) -> &str {
        "invocation"
    }

    /// 发布规范化的调用记录。
    async fn run(
        &self,
        ctx: &mut ToolExecutionContext,
        _snapshot: &crate::agent::snapshot::AgentRunSnapshot,
    ) -> Result<()> {
        ctx.emit_invocation().await
    }
}

/// 通过 ToolManager 执行工具。
pub struct ExecuteStage;

#[async_trait]
impl PipelineStage for ExecuteStage {
    /// 返回该阶段的名称。
    fn name(&self) -> &str {
        "execute"
    }

    /// 通过 ToolManager 执行工具，并把基础设施错误转换为失败结果。
    async fn run(
        &self,
        ctx: &mut ToolExecutionContext,
        snapshot: &crate::agent::snapshot::AgentRunSnapshot,
    ) -> Result<()> {
        let execution_start = std::time::Instant::now();

        if ctx.call_id.is_empty() {
            ctx.call_id = format!("call_{}", uuid::Uuid::new_v4());
        }
        snapshot.mark_tool_execution_started(&ctx.call_id);
        // 记录 ToolCall 追踪事件（脱敏处理由 new_tool_call 负责）
        snapshot
            .record_event(crate::trace::RunEvent::new_tool_call(
                ctx.call_id.clone(),
                ctx.tool_name.clone(),
                Some(ctx.input.clone()),
                None,
                0,
            ))
            .await;

        /*
         * 执行工具；若发生基础设施层面的错误，转换为 ToolResult{success:false}，
         * 这样下游阶段（追踪、post-hook、回调）依然能够正常运行。
         *
         * 从 snapshot 的 RuntimeConfig 为本次调用构建专属的 ToolContext，
         * 使（共享、无状态的）ToolManager 拿到的是“这一个” agent/session 对应的
         * working_dir —— 避免跨会话的当前目录污染。
         */
        let effect_sink: Option<echo_core::tools::ToolEffectSinkFn> =
            if snapshot.run_store.is_some() && snapshot.trace_run_id.is_some() {
                let invocation = snapshot.clone();
                let tool_name = ctx.tool_name.clone();
                let call_id = ctx.call_id.clone();
                Some(std::sync::Arc::new(move |effect| {
                    let invocation = invocation.clone();
                    let tool_name = tool_name.clone();
                    let call_id = call_id.clone();
                    Box::pin(async move {
                        invocation
                            .record_tool_effect(&effect, &tool_name, &call_id)
                            .await;
                    })
                }))
            } else {
                None
            };
        let tool_ctx = echo_core::tools::ToolContext {
            working_dir: snapshot.config.working_dir.clone(),
            conversation_id: snapshot.config.conversation_id.clone(),
            run_id: snapshot.current_run_id.clone(),
            turn_id: snapshot.current_turn_id.clone(),
            message_id: snapshot.current_message_id.clone(),
            execution_id: snapshot.current_execution_id.clone(),
            call_id: Some(ctx.call_id.clone()),
            effect_sink,
            active_message: snapshot.current_message.clone(),
            output_artifacts: snapshot.config.tool_output_artifacts.clone(),
            tool_visibility: snapshot.tools.visibility.clone(),
            script_execution_profile: None,
            cancel: snapshot.external_cancel.clone(),
            trace_sink: snapshot.external_trace_sink.clone(),
            delegation_policy: snapshot.external_delegation_policy,
            resource_guards: snapshot.resource_guards.clone(),
            subagent_lineage: snapshot.subagent_lineage.clone(),
            uplink: snapshot.external_uplink.clone(),
        };
        let execution_result = if snapshot
            .tools
            .tool_manager
            .supports_streaming(&ctx.tool_name)
        {
            if let Some(stream_tx) = ctx.stream_tx.as_ref() {
                let (event_tx, mut event_rx) = mpsc::channel(64);
                let mut execution = Box::pin(
                    snapshot
                        .tools
                        .tool_manager
                        .execute_tool_stream_with_context_draining_started(
                            &ctx.tool_name,
                            ctx.params.clone(),
                            &tool_ctx,
                            Some(event_tx),
                        ),
                );
                let mut stream_open = true;
                let result = loop {
                    tokio::select! {
                        biased;
                        event = event_rx.recv(), if stream_open => {
                            match event {
                                Some(event) => {
                                    if !matches!(event, ToolStreamEvent::Complete(_)) {
                                        stream_tx
                                            .send(ToolPipelineEvent::Stream {
                                                call_id: ctx.call_id.clone(),
                                                name: ctx.tool_name.clone(),
                                                event,
                                            })
                                            .await
                                            .map_err(|_| ReactError::Other("Tool stream receiver closed".into()))?;
                                    }
                                }
                                None => stream_open = false,
                            }
                        }
                        result = &mut execution => break result,
                    }
                };
                while let Some(event) = event_rx.recv().await {
                    if !matches!(event, ToolStreamEvent::Complete(_)) {
                        stream_tx
                            .send(ToolPipelineEvent::Stream {
                                call_id: ctx.call_id.clone(),
                                name: ctx.tool_name.clone(),
                                event,
                            })
                            .await
                            .map_err(|_| ReactError::Other("Tool stream receiver closed".into()))?;
                    }
                }
                result
            } else {
                snapshot
                    .tools
                    .tool_manager
                    .execute_tool_stream_with_context_draining_started(
                        &ctx.tool_name,
                        ctx.params.clone(),
                        &tool_ctx,
                        None,
                    )
                    .await
            }
        } else {
            snapshot
                .tools
                .tool_manager
                .execute_tool_with_context_draining_started(
                    &ctx.tool_name,
                    ctx.params.clone(),
                    &tool_ctx,
                )
                .await
        };

        let may_have_side_effects = snapshot
            .tools
            .tool_manager
            .get_tool(&ctx.tool_name)
            .is_none_or(|tool| tool.risk_level() != crate::tools::ToolRiskLevel::ReadOnly);
        let result = match execution_result {
            Ok(r) => r,
            Err(e) => {
                let err_msg = e.to_string();
                let failure = crate::tools::ToolFailure::from_error(&e, may_have_side_effects);
                if matches!(
                    failure.category,
                    crate::tools::ToolFailureCategory::Cancelled
                        | crate::tools::ToolFailureCategory::Timeout
                ) && failure.side_effect == crate::tools::ToolSideEffect::Possible
                {
                    ctx.interrupted_execution_error = Some(e);
                }
                ToolResult {
                    kind: echo_core::tools::ToolResultKind::StructuredError {
                        error_code: "tool_execution_failed".into(),
                    },
                    success: false,
                    output: String::new(),
                    error: Some(err_msg),
                    failure: Some(failure),
                    data: None,
                    truncated: false,
                    mime_type: None,
                    artifact: None,
                    metadata: std::collections::HashMap::new(),
                    effects: Vec::new(),
                    model_content: Vec::new(),
                }
            }
        };

        ctx.duration_ms = u64::try_from(execution_start.elapsed().as_millis()).unwrap_or(u64::MAX);
        #[cfg(feature = "telemetry")]
        {
            crate::telemetry::Metrics::record_tool_execution(
                &ctx.tool_name,
                if result.success { "success" } else { "error" },
            );
            crate::telemetry::Metrics::record_tool_latency(
                &ctx.tool_name,
                execution_start.elapsed().as_secs_f64() * 1000.0,
            );
        }
        ctx.result = Some(result.clone());

        /*
         * 已确认的效果属于工具本身，而非 post-use 呈现阶段的产物。
         * 必须在钩子有机会卡住或拒绝返回结果之前先把它们持久化。
         */
        for effect in &result.effects {
            snapshot
                .record_tool_effect(effect, &ctx.tool_name, &ctx.call_id)
                .await;
        }

        /*
         * 解析后的路径是工具自身确认的事实；模型给出的参数可能是相对路径
         * 或经过改写，不能单凭它来断定一次成功的读取。
         */
        if result.success {
            for effect in &result.effects {
                if let echo_core::tools::ToolEffect::FileRead { path } = effect {
                    snapshot.record_file_read(path);
                }
            }
        }

        Ok(())
    }
}

/// 根据工具结果运行 PostToolUse 或 PostToolUseFailure 钩子。
pub struct PostToolUseHookStage;

#[async_trait]
impl PipelineStage for PostToolUseHookStage {
    /// 返回该阶段的名称。
    fn name(&self) -> &str {
        "post_tool_use_hook"
    }

    /// 根据工具结果运行 PostToolUse 或 PostToolUseFailure 钩子。
    async fn run(
        &self,
        ctx: &mut ToolExecutionContext,
        snapshot: &crate::agent::snapshot::AgentRunSnapshot,
    ) -> Result<()> {
        let hook_reg = {
            let guard = snapshot.tools.hook_registry.read().await;
            if guard.is_empty() {
                return Ok(());
            }
            guard.clone()
        };
        let Some(tool_result) = ctx.result.as_ref() else {
            return Ok(());
        };
        let post_result = if tool_result.success {
            hook_reg
                .run_post_tool_use(
                    &ctx.tool_name,
                    &ctx.input,
                    &tool_result.output,
                    snapshot.config.session_id.as_deref().unwrap_or(""),
                )
                .await
        } else {
            let error = tool_result
                .error
                .as_deref()
                .filter(|value| !value.is_empty())
                .unwrap_or(&tool_result.output);
            hook_reg
                .run_post_tool_use_failure(
                    &ctx.tool_name,
                    &ctx.input,
                    error,
                    snapshot.config.session_id.as_deref().unwrap_or(""),
                )
                .await
        };
        ctx.hook_messages.post = post_result.messages;
        if post_result.block {
            let reason = post_result.block_reason.unwrap_or_else(|| {
                format!("Tool {} output blocked by post-use hook", ctx.tool_name)
            });
            /*
             * 策略拒绝的是一次已经执行完毕的调用，而不是执行过程本身。
             * 结算时保留其输出/产物以及可恢复性相关的事实。
             */
            let mut failure = tool_result.failure.clone().unwrap_or_else(|| {
                crate::tools::ToolFailure::new(crate::tools::ToolFailureCategory::PartialSideEffect)
            });
            if tool_result.success {
                failure = failure.with_postcondition(
                    "Tool execution succeeded before the post-use hook rejected the result",
                );
            }
            ctx.blocked = true;
            ctx.block_reason = Some(reason.clone());
            ctx.block_failure = Some(failure.clone());
            if let Some(result) = ctx.result.as_mut() {
                result.success = false;
                result.error = Some(reason);
                result.failure = Some(failure);
            }
        }
        Ok(())
    }
}

/// 运行输出防护检查。
pub struct OutputGuardStage;

#[async_trait]
impl PipelineStage for OutputGuardStage {
    /// 返回该阶段的名称。
    fn name(&self) -> &str {
        "output_guard"
    }

    /// 被阻止的调用同样需要运行该阶段。
    fn runs_after_block(&self) -> bool {
        true
    }

    /// 对工具输出运行输出防护检查，必要时改写或拦截输出。
    async fn run(
        &self,
        ctx: &mut ToolExecutionContext,
        snapshot: &crate::agent::snapshot::AgentRunSnapshot,
    ) -> Result<()> {
        let mut guard_replaced_output = false;
        if let Some(ref result) = ctx.result {
            if !result.success && result.output.is_empty() {
                if let Some(error) = result.error.as_deref() {
                    let guarded = snapshot.check_tool_output_guard(error).await;
                    guard_replaced_output = guarded.is_some();
                    ctx.audit_error_output = Some(guarded.unwrap_or_else(|| error.to_string()));
                }
            } else if let Some(guarded) = snapshot.check_tool_output_guard(&result.output).await {
                ctx.output = Some(guarded);
                guard_replaced_output = true;
            }
        }
        if guard_replaced_output && let Some(result) = ctx.result.as_mut() {
            result.artifact = None;
            result.metadata.remove("output_handling");
            result.metadata.remove("original_bytes");
            result.metadata.remove("returned_bytes");
            result.metadata.remove("estimated_tokens");
        }
        Ok(())
    }
}

/// 根据 token 预算截断工具输出。
pub struct TruncationStage;

#[async_trait]
impl PipelineStage for TruncationStage {
    /// 返回该阶段的名称。
    fn name(&self) -> &str {
        "truncation"
    }

    /// 被阻止的调用同样需要运行该阶段。
    fn runs_after_block(&self) -> bool {
        true
    }

    /// 按 token 预算截断工具输出，必要时溢出到产物文件。
    async fn run(
        &self,
        ctx: &mut ToolExecutionContext,
        snapshot: &crate::agent::snapshot::AgentRunSnapshot,
    ) -> Result<()> {
        let raw = ctx
            .output
            .as_deref()
            .or_else(|| ctx.result.as_ref().map(|r| r.output.as_str()))
            .unwrap_or("");
        let existing_artifact = ctx
            .result
            .as_ref()
            .and_then(|result| result.artifact.clone());
        let processed = snapshot.process_tool_output_for_call(
            raw.to_string(),
            &ctx.call_id,
            &ctx.tool_name,
            existing_artifact,
        );
        if let Some(result) = ctx.result.as_mut() {
            result.truncated = result.truncated || processed.truncated;
            result.artifact = processed.artifact;
            result.metadata.extend(processed.metadata);
            snapshot.tools.tool_manager.record_tool_result(
                &ctx.tool_name,
                result,
                processed.output.len(),
                ctx.duration_ms,
            );
        }
        ctx.output = Some(processed.output);
        Ok(())
    }
}

/// 触发回调（on_tool_start、on_tool_end、on_tool_error）。
pub struct CallbackStage {
    /// 本实例对应触发的回调阶段（开始或结束）。
    phase: CallbackPhase,
}

/// 回调触发的时机：开始阶段还是结束阶段。
pub enum CallbackPhase {
    /// 工具调用开始时触发。
    Start,
    /// 工具调用结束（成功、失败或中断）时触发。
    End,
}

impl CallbackStage {
    /// 触发“开始”回调的阶段实例。
    pub const START: Self = Self {
        phase: CallbackPhase::Start,
    };
    /// 触发“结束”回调的阶段实例。
    pub const END: Self = Self {
        phase: CallbackPhase::End,
    };
}

#[async_trait]
impl PipelineStage for CallbackStage {
    /// 返回该阶段的名称。
    fn name(&self) -> &str {
        match self.phase {
            CallbackPhase::Start => "callback_start",
            CallbackPhase::End => "callback_end",
        }
    }

    /**
     * CallbackEnd 是调用方观察到的终态：即便发生了 post-effect 阻止（#102），
     * 它也必须携带真实结果被触发。
     */
    fn runs_after_block(&self) -> bool {
        matches!(self.phase, CallbackPhase::End)
    }

    /// 按阶段触发开始或终态回调。
    async fn run(
        &self,
        ctx: &mut ToolExecutionContext,
        snapshot: &crate::agent::snapshot::AgentRunSnapshot,
    ) -> Result<()> {
        let agent_name = &snapshot.config.agent_name;
        for cb in &snapshot.config.callbacks {
            match self.phase {
                CallbackPhase::Start => {
                    cb.on_tool_start_with_id(agent_name, &ctx.call_id, &ctx.tool_name, &ctx.input)
                        .await;
                }
                CallbackPhase::End => {
                    let output = ctx
                        .output
                        .as_deref()
                        .or_else(|| ctx.result.as_ref().map(|r| r.output.as_str()))
                        .unwrap_or("");
                    /*
                     * 在拿到工具结果之前发生的、带类型的取消/超时属于“被中断的
                     * 尝试”，而不是一次已完成的工具失败。
                     */
                    let failure = ctx.result.as_ref().filter(|result| !result.success);
                    if let Some(result) = failure {
                        if let Some(interruption) = ctx.interrupted_execution_error.as_ref() {
                            let terminal_message = result.error.as_deref().unwrap_or(output);
                            let rewritten = (terminal_message != interruption.to_string())
                                .then(|| ReactError::Other(terminal_message.to_string()));
                            let error = rewritten.as_ref().unwrap_or(interruption);
                            cb.on_tool_interrupted_with_id(
                                agent_name,
                                &ctx.call_id,
                                &ctx.tool_name,
                                &ctx.input,
                                error,
                            )
                            .await;
                        } else {
                            let error = ReactError::Other(
                                result.error.clone().unwrap_or_else(|| output.to_string()),
                            );
                            cb.on_tool_error_with_id(
                                agent_name,
                                &ctx.call_id,
                                &ctx.tool_name,
                                &error,
                            )
                            .await;
                        }
                    } else {
                        cb.on_tool_end_with_id(agent_name, &ctx.call_id, &ctx.tool_name, output)
                            .await;
                    }
                }
            }
        }
        if matches!(self.phase, CallbackPhase::Start) {
            ctx.callback_started = true;
        }
        Ok(())
    }
}

/// 为工具结果/错误记录追踪事件。
pub struct TraceRecordingStage;

#[async_trait]
impl PipelineStage for TraceRecordingStage {
    /// 返回该阶段的名称。
    fn name(&self) -> &str {
        "trace_recording"
    }

    /**
     * 被阻止的调用也可能已经产生了外部效果；追踪记录必须如实反映实际结果
     * （失败时还需记录 ToolError）。
     */
    fn runs_after_block(&self) -> bool {
        true
    }

    /// 为工具结果或错误记录追踪事件。
    async fn run(
        &self,
        ctx: &mut ToolExecutionContext,
        snapshot: &crate::agent::snapshot::AgentRunSnapshot,
    ) -> Result<()> {
        if let Some(ref result) = ctx.result {
            /*
             * 始终记录 ToolResult —— 由 success 字段决定最终结果。
             * 失败时额外记录 ToolError 以保留失败细节。
             */
            snapshot
                .record_event(crate::trace::RunEvent::ToolResult {
                    call_id: ctx.call_id.clone(),
                    name: ctx.tool_name.clone(),
                    success: result.success,
                    output_preview: Some(
                        ctx.output
                            .as_deref()
                            .unwrap_or(&result.output)
                            .chars()
                            .take(200)
                            .collect(),
                    ),
                    output_truncated: result.truncated,
                    duration_ms: ctx.duration_ms,
                    original_bytes: metadata_u64(result, "original_bytes"),
                    returned_bytes: metadata_u64(result, "returned_bytes"),
                    estimated_tokens: metadata_usize(result, "estimated_tokens"),
                    output_handling: result.metadata.get("output_handling").cloned(),
                    artifact: result.artifact.clone(),
                })
                .await;
            if !result.success {
                snapshot
                    .record_event(crate::trace::RunEvent::ToolError {
                        call_id: ctx.call_id.clone(),
                        name: ctx.tool_name.clone(),
                        message: result.error.clone().unwrap_or_default(),
                        failure: result.failure.clone(),
                    })
                    .await;
            }
        }
        Ok(())
    }
}

/// 从工具结果的 metadata 中读取指定键并解析为 u64，缺失或解析失败时返回 0。
fn metadata_u64(result: &crate::tools::ToolResult, key: &str) -> u64 {
    result
        .metadata
        .get(key)
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(0)
}

/// 从工具结果的 metadata 中读取指定键并解析为 usize，缺失或解析失败时返回 0。
fn metadata_usize(result: &crate::tools::ToolResult, key: &str) -> usize {
    result
        .metadata
        .get(key)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0)
}

// ── ToolExecutionPipeline ──────────────────────────────────────────

/**
 * 由多个工具执行阶段组成的可组合流水线。
 *
 * # 示例
 *
 * ```rust,ignore
 * let pipeline = ToolExecutionPipeline::default();
 * let mut ctx = ToolExecutionContext { ... };
 * pipeline.run(&mut ctx, &agent).await?;
 * ```
 */
pub struct ToolExecutionPipeline {
    /// 按执行顺序排列的各个阶段。
    stages: Vec<Box<dyn PipelineStage>>,
}

impl ToolExecutionPipeline {
    /// 按正确顺序构建包含全部标准阶段的流水线。
    pub fn default_pipeline() -> Self {
        Self {
            stages: vec![
                Box::new(InterventionStage),
                Box::new(ToolVisibilityStage),
                Box::new(PlanModeStage),
                Box::new(PreToolUseHookStage),
                Box::new(PermissionStage),
                Box::new(ReadBeforeEditStage),
                Box::new(SkillPermissionStage),
                Box::new(InvocationStage),
                Box::new(CallbackStage::START),
                Box::new(ExecuteStage),
                Box::new(PostToolUseHookStage),
                Box::new(OutputGuardStage),
                Box::new(TruncationStage),
                Box::new(TraceRecordingStage),
                Box::new(AuditStage),
                Box::new(CallbackStage::END),
            ],
        }
    }

    /// 按顺序运行所有阶段；一旦 `ctx.blocked` 被设置则短路跳过后续非观察类阶段。
    pub(crate) async fn run(
        &self,
        ctx: &mut ToolExecutionContext,
        snapshot: &crate::agent::snapshot::AgentRunSnapshot,
    ) -> Result<()> {
        for (index, stage) in self.stages.iter().enumerate() {
            if ctx.blocked && !stage.runs_after_block() {
                ctx.emit_invocation().await?;
                debug!(
                    agent = %snapshot.config.agent_name,
                    stage = stage.name(),
                    reason = ?ctx.block_reason,
                    "Pipeline stage skipped (blocked)"
                );
                if ctx.result.is_none() {
                    break;
                }
                continue;
            }
            debug!(
                agent = %snapshot.config.agent_name,
                stage = stage.name(),
                tool = %ctx.tool_name,
                "Pipeline stage running"
            );
            if let Err(error) = stage.run(ctx, snapshot).await {
                ctx.emit_invocation().await?;
                if ctx.callback_started {
                    let message = error.to_string();
                    let failure = crate::tools::ToolFailure::from_error(&error, true);
                    if let Some(result) = ctx.result.as_mut() {
                        result.success = false;
                        result.error = Some(message.clone());
                        result.failure = Some(failure);
                    } else {
                        ctx.result = Some(crate::tools::ToolResult {
                            kind: echo_core::tools::ToolResultKind::StructuredError {
                                error_code: "tool_pipeline_failed".to_string(),
                            },
                            success: false,
                            output: String::new(),
                            error: Some(message),
                            failure: Some(failure),
                            data: None,
                            truncated: false,
                            mime_type: None,
                            artifact: None,
                            metadata: std::collections::HashMap::new(),
                            effects: Vec::new(),
                            model_content: Vec::new(),
                        });
                    }
                    ctx.blocked = true;
                    for remaining in self.stages.iter().skip(index.saturating_add(1)) {
                        if remaining.runs_after_block() {
                            remaining.run(ctx, snapshot).await?;
                        }
                    }
                    return Ok(());
                }
                return Err(error);
            }
        }
        Ok(())
    }
}

// ── PlanModeStage ──────────────────────────────────────────────────

/// 计划模式下，只允许执行本地分类为只读的工具。
pub struct PlanModeStage;

#[async_trait]
impl PipelineStage for PlanModeStage {
    /// 返回该阶段的名称。
    fn name(&self) -> &str {
        "plan_mode"
    }

    /// 计划模式下阻止非只读工具的执行。
    async fn run(
        &self,
        ctx: &mut ToolExecutionContext,
        snapshot: &crate::agent::snapshot::AgentRunSnapshot,
    ) -> Result<()> {
        if !ctx.plan_mode && !snapshot.tools.is_plan_mode() {
            return Ok(());
        }
        if !snapshot.tools.is_tool_read_only(&ctx.tool_name) {
            let reason = format!(
                "Plan mode: '{}' is blocked. Read and analyze only. Use /plan off to enable writes.",
                ctx.tool_name
            );
            snapshot
                .record_permission_decision(
                    &ctx.tool_name,
                    &PermissionDecision::Deny {
                        reason: reason.clone(),
                    },
                    "plan_mode",
                )
                .await;
            ctx.block(crate::tools::ToolFailureCategory::Unavailable, reason);
        }
        Ok(())
    }
}

impl Default for ToolExecutionPipeline {
    /// 默认使用标准流水线。
    fn default() -> Self {
        Self::default_pipeline()
    }
}

// ── 辅助函数 ────────────────────────────────────────────────────────

/// 从工具参数中提取涉及的文件路径列表，用于“先读后写”策略等场景。
fn extract_path_params(tool_name: &str, params: &ToolParameters) -> Vec<String> {
    if tool_name == "apply_patch" {
        #[cfg(feature = "files")]
        return params
            .get("patch")
            .and_then(Value::as_str)
            .and_then(|patch| echo_tools::files::apply_patch::existing_file_paths(patch).ok())
            .unwrap_or_default();
        #[cfg(not(feature = "files"))]
        return Vec::new();
    }
    params
        .get("path")
        .or_else(|| params.get("file_path"))
        .and_then(Value::as_str)
        .map(|path| vec![path.to_string()])
        .unwrap_or_default()
}

/// 将路径解析为用于“已读文件”跟踪表的规范化字符串（相对路径会先拼接工作目录，再尽量规范化）。
fn resolve_tracking_path(path: &str, working_dir: Option<&std::path::Path>) -> String {
    let path = std::path::Path::new(path);
    let resolved = if path.is_absolute() {
        path.to_path_buf()
    } else if let Some(working_dir) = working_dir {
        working_dir.join(path)
    } else {
        path.to_path_buf()
    };
    std::fs::canonicalize(&resolved)
        .unwrap_or(resolved)
        .to_string_lossy()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::AgentEvent;
    use echo_core::tools::{InvocationResourceGuard, Tool, ToolContext, ToolOutputChannel};
    use futures::Stream;
    use std::collections::HashMap;
    use std::pin::Pin;
    use std::sync::Arc;

    /// 验证写类工具的识别。
    #[test]
    fn test_is_write_tool() {
        assert!(is_write_tool("apply_patch"));
        assert!(is_write_tool("write_file"));
        assert!(!is_write_tool("read_file"));
    }

    /// 验证能从 write_file 参数中提取路径。
    #[test]
    fn test_extract_write_path_param() {
        let params = vec![("path".to_string(), Value::String("src/main.rs".to_string()))]
            .into_iter()
            .collect();
        assert_eq!(
            extract_path_params("write_file", &params),
            vec!["src/main.rs".to_string()]
        );
    }

    /// 验证能从 apply_patch 补丁中提取所有涉及的路径。
    #[cfg(feature = "files")]
    #[test]
    fn test_extract_apply_patch_paths() {
        let patch_params = HashMap::from([(
            "patch".to_string(),
            Value::String(
                "*** Begin Patch\n*** Update File: src/lib.rs\n@@\n-old\n+new\n*** Delete File: stale.txt\n*** End Patch"
                    .to_string(),
            ),
        )]);
        assert_eq!(
            extract_path_params("apply_patch", &patch_params),
            vec!["src/lib.rs".to_string(), "stale.txt".to_string()]
        );
    }

    /// 交错输出流式事件的测试工具，用于验证并发流的顺序。
    struct InterleavingTool;

    /// 返回“参数无效”失败结果的测试工具。
    struct InvalidResultTool;

    /// 返回带输出的失败结果的测试工具。
    struct FailedOutputTool;

    /// 只返回敏感错误信息、不带输出的测试工具。
    struct ErrorOnlyTool;

    impl Tool for ErrorOnlyTool {
        /// 返回工具名称。
        fn name(&self) -> &str {
            "error_only"
        }

        /// 返回工具描述。
        fn description(&self) -> &str {
            "returns a sensitive error without tool output"
        }

        /// 返回参数 JSON Schema。
        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({"type": "object"})
        }

        /// 返回一条包含敏感内容的错误结果。
        fn execute<'a>(
            &'a self,
            _parameters: ToolParameters,
        ) -> futures::future::BoxFuture<'a, crate::error::Result<ToolResult>> {
            Box::pin(async { Ok(ToolResult::error("Bearer sensitive-token")) })
        }
    }

    impl Tool for FailedOutputTool {
        /// 返回工具名称。
        fn name(&self) -> &str {
            "failed_output"
        }

        /// 返回工具描述。
        fn description(&self) -> &str {
            "returns an unsuccessful result with output"
        }

        /// 返回参数 JSON Schema。
        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({"type": "object"})
        }

        /// 返回带敏感原始输出的永久失败结果。
        fn execute<'a>(
            &'a self,
            _parameters: ToolParameters,
        ) -> futures::future::BoxFuture<'a, crate::error::Result<ToolResult>> {
            Box::pin(async {
                Ok(ToolResult::failure(
                    crate::tools::ToolFailureCategory::Permanent,
                    "execution failed",
                )
                .with_output("sensitive raw output"))
            })
        }
    }

    /// 在流式路径上记录收到的调用资源守卫数量的测试工具。
    struct ResourceGuardStreamingTool {
        /// 观察到的资源守卫数量。
        observed_guard_count: Arc<std::sync::atomic::AtomicUsize>,
    }

    impl Tool for InvalidResultTool {
        /// 返回工具名称。
        fn name(&self) -> &str {
            "invalid_result"
        }

        /// 返回工具描述。
        fn description(&self) -> &str {
            "returns a classified unsuccessful result"
        }

        /// 返回参数 JSON Schema。
        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({"type": "object"})
        }

        /// 返回“缺少 query”的参数无效失败结果。
        fn execute<'a>(
            &'a self,
            _parameters: ToolParameters,
        ) -> futures::future::BoxFuture<'a, crate::error::Result<ToolResult>> {
            Box::pin(async {
                Ok(ToolResult::failure(
                    crate::tools::ToolFailureCategory::InvalidArguments,
                    "missing query",
                ))
            })
        }

        /// 声明需要执行权限。
        fn permissions(&self) -> Vec<echo_core::tools::permission::ToolPermission> {
            vec![echo_core::tools::permission::ToolPermission::Execute]
        }
    }

    impl Tool for InterleavingTool {
        /// 返回工具名称。
        fn name(&self) -> &str {
            "interleaving"
        }

        /// 返回工具描述。
        fn description(&self) -> &str {
            "test stream interleaving"
        }

        /// 返回参数 JSON Schema。
        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({"type": "object"})
        }

        /// 非流式路径：返回占位结果。
        fn execute<'a>(
            &'a self,
            _parameters: ToolParameters,
        ) -> futures::future::BoxFuture<'a, crate::error::Result<ToolResult>> {
            Box::pin(async { Ok(ToolResult::success("unused")) })
        }

        /// 按参数中的延迟分两次输出分块，最后发出完成事件。
        fn execute_stream_with_context<'a>(
            &'a self,
            params: ToolParameters,
            _ctx: &ToolContext,
        ) -> futures::future::BoxFuture<
            'a,
            crate::error::Result<Pin<Box<dyn Stream<Item = ToolStreamEvent> + Send + 'a>>>,
        > {
            Box::pin(async move {
                let label = params
                    .get("label")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown")
                    .to_string();
                let initial_delay = params
                    .get("initial_delay")
                    .and_then(Value::as_u64)
                    .unwrap_or(0);
                let finish_delay = params
                    .get("finish_delay")
                    .and_then(Value::as_u64)
                    .unwrap_or(0);
                let events = futures::stream::unfold(
                    (0_u8, label, initial_delay, finish_delay),
                    |(state, label, initial_delay, finish_delay)| async move {
                        match state {
                            0 => {
                                tokio::time::sleep(std::time::Duration::from_millis(initial_delay))
                                    .await;
                                Some((
                                    ToolStreamEvent::Output {
                                        channel: ToolOutputChannel::Stdout,
                                        chunk: format!("{label}-1"),
                                    },
                                    (1, label, initial_delay, finish_delay),
                                ))
                            }
                            1 => {
                                tokio::time::sleep(std::time::Duration::from_millis(finish_delay))
                                    .await;
                                Some((
                                    ToolStreamEvent::Output {
                                        channel: ToolOutputChannel::Stdout,
                                        chunk: format!("{label}-2"),
                                    },
                                    (2, label, initial_delay, finish_delay),
                                ))
                            }
                            2 => Some((
                                ToolStreamEvent::Complete(ToolResult::success(label.clone())),
                                (3, label, initial_delay, finish_delay),
                            )),
                            _ => None,
                        }
                    },
                );
                Ok(Box::pin(events) as Pin<Box<dyn Stream<Item = ToolStreamEvent> + Send + 'a>>)
            })
        }

        /// 声明支持流式执行。
        fn supports_streaming(&self) -> bool {
            true
        }
    }

    impl Tool for ResourceGuardStreamingTool {
        /// 返回工具名称。
        fn name(&self) -> &str {
            "resource_guard_stream"
        }

        /// 返回工具描述。
        fn description(&self) -> &str {
            "captures invocation resource guards on the streaming path"
        }

        /// 返回参数 JSON Schema。
        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({"type": "object"})
        }

        /// 非流式回退路径：返回占位结果。
        fn execute<'a>(
            &'a self,
            _parameters: ToolParameters,
        ) -> futures::future::BoxFuture<'a, crate::error::Result<ToolResult>> {
            Box::pin(async { Ok(ToolResult::success("fallback")) })
        }

        /// 声明支持流式执行。
        fn supports_streaming(&self) -> bool {
            true
        }

        /// 记录上下文中的资源守卫数量，并直接返回完成事件。
        fn execute_stream_with_context<'a>(
            &'a self,
            _parameters: ToolParameters,
            context: &ToolContext,
        ) -> futures::future::BoxFuture<
            'a,
            crate::error::Result<Pin<Box<dyn Stream<Item = ToolStreamEvent> + Send + 'a>>>,
        > {
            self.observed_guard_count.store(
                context.resource_guards.len(),
                std::sync::atomic::Ordering::SeqCst,
            );
            Box::pin(async {
                Ok(Box::pin(futures::stream::iter([ToolStreamEvent::Complete(
                    ToolResult::success("streamed"),
                )]))
                    as Pin<Box<dyn Stream<Item = ToolStreamEvent> + Send>>)
            })
        }
    }

    /// 构造一个使用 InterleavingTool 并带流式发送端的执行上下文。
    fn streaming_context(
        call_id: &str,
        label: &str,
        initial_delay: u64,
        finish_delay: u64,
        stream_tx: mpsc::Sender<ToolPipelineEvent>,
    ) -> ToolExecutionContext {
        let input = serde_json::json!({
            "label": label,
            "initial_delay": initial_delay,
            "finish_delay": finish_delay,
        });
        let params = match &input {
            Value::Object(map) => map.clone().into_iter().collect(),
            _ => ToolParameters::new(),
        };
        ToolExecutionContext {
            call_id: call_id.to_string(),
            requested_tool_name: "interleaving".to_string(),
            requested_input: input.clone(),
            tool_name: "interleaving".to_string(),
            params,
            input,
            hook_messages: HookMessageBatches::default(),
            result: None,
            output: None,
            audit_error_output: None,
            blocked: false,
            block_reason: None,
            block_failure: None,
            duration_ms: 0,
            plan_mode: false,
            permission_decision: None,
            permission_mode_override: None,
            rewrites: Vec::new(),
            invocation_emitted: false,
            callback_started: false,
            interrupted_execution_error: None,
            stream_tx: Some(stream_tx),
        }
    }

    /// 构造一个已包含成功结果的 shell 调用执行上下文。
    fn completed_context(output: String) -> ToolExecutionContext {
        ToolExecutionContext {
            call_id: "call-output-budget".to_string(),
            requested_tool_name: "shell".to_string(),
            requested_input: serde_json::json!({}),
            tool_name: "shell".to_string(),
            params: ToolParameters::new(),
            input: serde_json::json!({}),
            hook_messages: HookMessageBatches::default(),
            result: Some(ToolResult::success(output)),
            output: None,
            audit_error_output: None,
            blocked: false,
            block_reason: None,
            block_failure: None,
            duration_ms: 0,
            plan_mode: false,
            permission_decision: None,
            permission_mode_override: None,
            rewrites: Vec::new(),
            invocation_emitted: false,
            callback_started: false,
            interrupted_execution_error: None,
            stream_tx: None,
        }
    }

    /// 验证权限钩子在主路径上如实追踪允许与拒绝。
    #[tokio::test]
    async fn permission_hooks_trace_real_allow_and_deny_on_the_main_path() -> Result<()> {
        use crate::skills::hooks::{HookAction, HookEvent, HookRule, HooksDefinition};
        use crate::trace::{InMemoryRunStore, RunEvent, RunStore};

        for event in [HookEvent::PreToolUse, HookEvent::PermissionRequest] {
            for decision in ["allow", "deny"] {
                let store = Arc::new(InMemoryRunStore::new());
                let agent = crate::agent::ReactAgentBuilder::new()
                    .model("test-model")
                    .with_run_store(store.clone())
                    .tool(Box::new(EffectTool))
                    .build()?;
                let mut definition = HooksDefinition::default();
                definition.add_rules(
                    event,
                    vec![HookRule {
                        matcher: "shell".to_string(),
                        hooks: vec![HookAction::Permission {
                            decision: decision.to_string(),
                            reason: Some("reviewed decision".to_string()),
                            suggestions: Vec::new(),
                        }],
                    }],
                );
                agent
                    .hook_registry()
                    .write()
                    .await
                    .register_user_hooks(definition);
                let legacy = agent.capture_legacy_external_context();
                let run_id = agent
                    .start_legacy_trace_run("permission trace", &legacy)
                    .await
                    .ok_or_else(|| ReactError::Other("trace did not start".to_string()))?;
                let snapshot = crate::agent::snapshot::AgentRunSnapshot::from_agent(&agent);
                let result = snapshot
                    .execute_tool_with_policy(
                        "permission-call".to_string(),
                        "shell",
                        &ToolParameters::new(),
                        &serde_json::json!({}),
                        None,
                    )
                    .await;
                assert_eq!(result.is_ok(), decision == "allow");
                let run = store
                    .load(&run_id)
                    .await?
                    .ok_or_else(|| ReactError::Other("trace disappeared".to_string()))?;
                let decisions: Vec<_> = run
                    .events
                    .iter()
                    .filter_map(|event| match event {
                        RunEvent::PermissionDecision { decision, .. } => Some(decision.as_str()),
                        _ => None,
                    })
                    .collect();
                assert_eq!(decisions, vec![decision]);
                let permission = run
                    .events
                    .iter()
                    .position(|event| matches!(event, RunEvent::PermissionDecision { .. }));
                let execution = run
                    .events
                    .iter()
                    .position(|event| matches!(event, RunEvent::ToolCall { .. }));
                if decision == "allow" {
                    assert!(matches!((permission, execution), (Some(p), Some(e)) if p < e));
                } else {
                    assert!(execution.is_none());
                }
            }
        }
        Ok(())
    }

    /// 验证工具输出不会凭空推断出文件或测试效果。
    #[tokio::test]
    async fn tool_outputs_do_not_invent_file_or_test_effects() -> Result<()> {
        use crate::trace::{InMemoryRunStore, RunEvent, RunStore};

        let store = Arc::new(InMemoryRunStore::new());
        let agent = crate::agent::ReactAgentBuilder::new()
            .model("test-model")
            .with_run_store(store.clone())
            .build()?;
        let legacy = agent.capture_legacy_external_context();
        let run_id = agent
            .start_legacy_trace_run("effects", &legacy)
            .await
            .ok_or_else(|| ReactError::Other("trace did not start".to_string()))?;
        let snapshot = crate::agent::snapshot::AgentRunSnapshot::from_agent(&agent);
        for (tool, kind, success) in [
            ("write_file", echo_core::tools::ToolResultKind::Text, true),
            (
                "apply_patch",
                echo_core::tools::ToolResultKind::Diff {
                    unified_diff: "--- a/file\n+++ b/file\n-old\n+new".to_string(),
                },
                true,
            ),
            (
                "shell",
                echo_core::tools::ToolResultKind::CommandOutput { exit_code: Some(0) },
                true,
            ),
            (
                "shell",
                echo_core::tools::ToolResultKind::CommandOutput { exit_code: Some(1) },
                false,
            ),
        ] {
            let mut ctx = completed_context("test result: 2 passed; 1 failed".to_string());
            ctx.tool_name = tool.to_string();
            ctx.input = serde_json::json!({"command": "cargo test", "path": "file"});
            if let Some(result) = ctx.result.as_mut() {
                result.kind = kind;
                result.success = success;
            }
            TraceRecordingStage.run(&mut ctx, &snapshot).await?;
        }
        let run = store
            .load(&run_id)
            .await?
            .ok_or_else(|| ReactError::Other("trace disappeared".to_string()))?;
        assert_eq!(
            run.events
                .iter()
                .filter(|event| matches!(event, RunEvent::ToolResult { .. }))
                .count(),
            4
        );
        assert!(!run.events.iter().any(|event| matches!(
            event,
            RunEvent::FileRead { .. }
                | RunEvent::FileEdit { .. }
                | RunEvent::TestRun { .. }
                | RunEvent::SubagentRun { .. }
                | RunEvent::Error { .. }
        )));
        Ok(())
    }

    /// 验证权限钩子的决策会在执行前被追踪。
    #[tokio::test]
    async fn permission_hook_decisions_are_traced_before_execution() -> Result<()> {
        use crate::trace::{InMemoryRunStore, RunEvent, RunStore};

        for (decision, expected, blocked) in [
            (PermissionDecision::Allow, "allow", false),
            (
                PermissionDecision::Deny {
                    reason: "explicit denial 中文".to_string(),
                },
                "deny",
                true,
            ),
            (PermissionDecision::RequireApproval, "ask", false),
            (
                PermissionDecision::Ask {
                    suggestions: Vec::new(),
                },
                "ask",
                false,
            ),
        ] {
            let store = Arc::new(InMemoryRunStore::new());
            let agent = crate::agent::ReactAgentBuilder::new()
                .model("test-model")
                .with_run_store(store.clone())
                .build()?;
            let legacy = agent.capture_legacy_external_context();
            let run_id = agent
                .start_legacy_trace_run("permission trace", &legacy)
                .await
                .ok_or_else(|| ReactError::Other("trace did not start".to_string()))?;
            let snapshot = crate::agent::snapshot::AgentRunSnapshot::from_agent(&agent);
            let mut ctx = completed_context(String::new());
            ctx.result = None;
            ctx.permission_decision = Some(decision);

            PermissionStage.run(&mut ctx, &snapshot).await?;

            assert_eq!(ctx.blocked, blocked);
            let run = store
                .load(&run_id)
                .await?
                .ok_or_else(|| ReactError::Other("trace disappeared".to_string()))?;
            let decisions: Vec<_> = run
                .events
                .iter()
                .filter_map(|event| match event {
                    RunEvent::PermissionDecision {
                        tool,
                        decision,
                        reason,
                    } => Some((tool.as_str(), decision.as_str(), reason.as_str())),
                    _ => None,
                })
                .collect();
            assert_eq!(decisions.len(), 1);
            let (tool, decision, reason) = decisions
                .first()
                .copied()
                .ok_or_else(|| ReactError::Other("missing permission decision".to_string()))?;
            assert_eq!(tool, "shell");
            assert_eq!(decision, expected);
            assert!(reason.starts_with("pre_tool_use_hook"));
            if blocked {
                assert!(reason.contains("explicit denial 中文"));
            }
            assert!(!run.events.iter().any(|event| matches!(
                event,
                RunEvent::ToolCall { .. }
                    | RunEvent::FileRead { .. }
                    | RunEvent::FileEdit { .. }
                    | RunEvent::TestRun { .. }
                    | RunEvent::Error { .. }
            )));
        }
        Ok(())
    }

    /// 验证失败结果运行失败钩子而非成功钩子。
    #[tokio::test]
    async fn unsuccessful_result_runs_failure_hook_instead_of_success_hook() -> Result<()> {
        use crate::skills::hooks::{HookAction, HookEvent, HookRule, HooksDefinition};

        let agent = crate::agent::ReactAgentBuilder::new()
            .model("test-model")
            .build()?;
        let mut definition = HooksDefinition::default();
        definition.add_rules(
            HookEvent::PostToolUse,
            vec![HookRule {
                matcher: "invalid_result".to_string(),
                hooks: vec![HookAction::Prompt {
                    prompt: "success-hook".to_string(),
                }],
            }],
        );
        definition.add_rules(
            HookEvent::PostToolUseFailure,
            vec![HookRule {
                matcher: "invalid_result".to_string(),
                hooks: vec![HookAction::Prompt {
                    prompt: "failure-hook".to_string(),
                }],
            }],
        );
        agent
            .hook_registry()
            .write()
            .await
            .register_user_hooks(definition);
        let snapshot = crate::agent::snapshot::AgentRunSnapshot::from_agent(&agent);
        let mut ctx = completed_context(String::new());
        ctx.tool_name = "invalid_result".to_string();
        ctx.result = Some(ToolResult::failure(
            crate::tools::ToolFailureCategory::InvalidArguments,
            "missing query",
        ));

        PostToolUseHookStage.run(&mut ctx, &snapshot).await?;

        assert_eq!(ctx.hook_messages.post, vec!["failure-hook"]);
        Ok(())
    }

    /// 验证权限阶段会消费调用范围内的模式覆盖。
    #[cfg(feature = "human-loop")]
    #[tokio::test]
    async fn permission_stage_consumes_call_scoped_mode_override() -> Result<()> {
        let service = Arc::new(
            crate::human_loop::PermissionService::new()
                .with_mode(echo_core::tools::permission::PermissionMode::Default),
        );
        let agent = crate::agent::ReactAgentBuilder::new()
            .model("test-model")
            .tool(Box::new(InvalidResultTool))
            .permission_service(service.clone())
            .build()?;
        let snapshot = crate::agent::snapshot::AgentRunSnapshot::from_agent(&agent);
        let mut ctx = completed_context(String::new());
        ctx.tool_name = "invalid_result".to_string();
        ctx.permission_mode_override =
            Some(echo_core::tools::permission::PermissionMode::BypassPermissions);

        PermissionStage.run(&mut ctx, &snapshot).await?;

        assert!(!ctx.blocked);
        assert_eq!(
            service.mode().await,
            echo_core::tools::permission::PermissionMode::Default
        );
        Ok(())
    }

    /// 验证被禁用调用的工具会在执行前被阻止。
    #[tokio::test]
    async fn invocation_disabled_tool_is_blocked_before_execution() -> Result<()> {
        let agent = crate::agent::ReactAgentBuilder::new()
            .model("test-model")
            .build()?;
        let invocation = echo_core::agent::AgentInvocationContext {
            disabled_tools: Some(std::collections::HashSet::from(["hidden_tool".to_string()])),
            ..Default::default()
        };
        let snapshot = crate::agent::snapshot::AgentRunSnapshot::from_agent_with_invocation(
            &agent,
            &invocation,
        );
        let mut ctx = completed_context(String::new());
        ctx.tool_name = "hidden_tool".to_string();

        ToolVisibilityStage.run(&mut ctx, &snapshot).await?;

        assert!(ctx.blocked);
        assert!(
            ctx.block_reason
                .as_deref()
                .is_some_and(|reason| reason.contains("not available in this invocation"))
        );
        Ok(())
    }

    /// 验证截断阶段在无 token 限制时也会溢出，且可通过读取产物恢复。
    #[cfg(feature = "files")]
    #[tokio::test]
    async fn truncation_stage_spills_without_token_limit_and_read_artifact_can_recover()
    -> Result<()> {
        let working_dir =
            tempfile::tempdir().map_err(|error| ReactError::Other(error.to_string()))?;
        let agent = crate::agent::ReactAgentBuilder::new()
            .model("test-model")
            .working_dir(working_dir.path())
            .tool_output_artifacts(
                echo_core::tools::artifact::ToolOutputArtifactConfig::new(
                    working_dir.path(),
                    "test",
                )
                .threshold_bytes(8),
            )
            .build()?;
        let snapshot = crate::agent::snapshot::AgentRunSnapshot::from_agent(&agent);
        let original = format!("{}END", "中文🙂\n".repeat(3_000));
        let mut ctx = completed_context(original.clone());

        TruncationStage.run(&mut ctx, &snapshot).await?;

        let result = ctx
            .result
            .as_ref()
            .ok_or_else(|| ReactError::Other("truncation stage lost tool result".to_string()))?;
        assert!(result.truncated);
        assert_eq!(
            result.metadata.get("output_handling").map(String::as_str),
            Some("spilled")
        );
        let artifact = result
            .artifact
            .as_ref()
            .ok_or_else(|| ReactError::Other("spill result lacks typed artifact".to_string()))?;
        let artifact_path = &artifact.path;
        assert!(artifact_path.starts_with(working_dir.path()));
        let artifact_path_text = artifact_path.to_string_lossy();
        assert!(ctx.output.as_deref().is_some_and(|output| {
            output.contains("Tool output preview only")
                && output.contains("not a summary")
                && output.contains("Full output artifact")
                && output.contains("Use read_artifact with this exact path")
                && output.contains(artifact_path_text.as_ref())
        }));

        let read_tool = crate::tools::files::artifact::ReadArtifactTool;
        let tool_context = ToolContext {
            working_dir: Some(working_dir.path().to_path_buf()),
            output_artifacts: agent.tool_output_artifacts(),
            ..ToolContext::default()
        };
        let mut cursor = None;
        let mut recovered = String::new();
        loop {
            let mut params = ToolParameters::from([
                (
                    "path".to_string(),
                    Value::String(artifact_path_text.to_string()),
                ),
                ("max_tokens".to_string(), Value::from(500_u64)),
            ]);
            if let Some(value) = cursor.clone() {
                params.insert("cursor".to_string(), Value::String(value));
            }
            let page = read_tool
                .execute_with_context(params, &tool_context)
                .await?;
            assert!(page.success);
            let (content, _) = page.output.split_once("\n\n[Artifact ").ok_or_else(|| {
                ReactError::Other("read_artifact page omitted its cursor notice".to_string())
            })?;
            recovered.push_str(content);
            cursor = page.metadata.get("next_cursor").cloned();
            if cursor.is_none() {
                break;
            }
        }
        assert_eq!(recovered, original);
        assert_eq!(
            std::fs::read_to_string(artifact_path).ok().as_deref(),
            Some(original.as_str())
        );
        assert_eq!(artifact.sha256.len(), 64);
        assert_eq!(artifact.retention, "test");
        Ok(())
    }

    /// 验证溢出投影在工作目录之外使用产物读取器。
    #[tokio::test]
    async fn spill_projection_uses_artifact_reader_outside_working_dir() -> Result<()> {
        let working_dir =
            tempfile::tempdir().map_err(|error| ReactError::Other(error.to_string()))?;
        let artifact_dir =
            tempfile::tempdir().map_err(|error| ReactError::Other(error.to_string()))?;
        let agent = crate::agent::ReactAgentBuilder::new()
            .model("test-model")
            .working_dir(working_dir.path())
            .tool_output_artifacts(
                echo_core::tools::artifact::ToolOutputArtifactConfig::new(
                    artifact_dir.path(),
                    "test",
                )
                .threshold_bytes(8),
            )
            .build()?;
        let snapshot = crate::agent::snapshot::AgentRunSnapshot::from_agent(&agent);
        let mut ctx = completed_context("large complete output".to_string());

        TruncationStage.run(&mut ctx, &snapshot).await?;

        let output = ctx
            .output
            .as_deref()
            .ok_or_else(|| ReactError::Other("truncation stage produced no output".to_string()))?;
        assert!(output.contains("Use read_artifact with this exact path"));
        assert!(!output.contains("Use read_file with this exact path"));
        Ok(())
    }

    /// 验证即使低于字节阈值，token 预算仍会触发溢出。
    #[tokio::test]
    async fn token_budget_spills_even_below_byte_threshold() -> Result<()> {
        let artifact_dir =
            tempfile::tempdir().map_err(|error| ReactError::Other(error.to_string()))?;
        let agent = crate::agent::ReactAgentBuilder::new()
            .model("test-model")
            .max_tool_output_tokens(20)
            .tool_output_artifacts(
                echo_core::tools::artifact::ToolOutputArtifactConfig::new(
                    artifact_dir.path(),
                    "test",
                )
                .threshold_bytes(1024 * 1024),
            )
            .build()?;
        let snapshot = crate::agent::snapshot::AgentRunSnapshot::from_agent(&agent);
        let mut ctx = completed_context("small but token-heavy 中文🙂".repeat(100));

        TruncationStage.run(&mut ctx, &snapshot).await?;

        let result = ctx
            .result
            .as_ref()
            .ok_or_else(|| ReactError::Other("truncation stage lost tool result".to_string()))?;
        assert_eq!(
            result.metadata.get("output_handling").map(String::as_str),
            Some("spilled")
        );
        assert!(result.artifact.is_some());
        Ok(())
    }

    /// 验证截断阶段保留工具自身的截断标记。
    #[tokio::test]
    async fn truncation_stage_preserves_tool_level_truncation() -> Result<()> {
        let agent = crate::agent::ReactAgentBuilder::new()
            .model("test-model")
            .build()?;
        let snapshot = crate::agent::snapshot::AgentRunSnapshot::from_agent(&agent);
        let mut ctx = completed_context("partial page".to_string());
        if let Some(result) = ctx.result.as_mut() {
            result.truncated = true;
        }

        TruncationStage.run(&mut ctx, &snapshot).await?;

        assert!(ctx.result.as_ref().is_some_and(|result| result.truncated));
        Ok(())
    }

    /// 验证截断阶段对 UTF-8 安全。
    #[tokio::test]
    async fn truncation_stage_is_utf8_safe() -> Result<()> {
        let config = crate::agent::AgentConfig::new("test-model", "test-agent", "test")
            .max_tool_output_tokens(20)
            .tool_output_artifacts(None);
        let agent = crate::agent::ReactAgent::new(config);
        let snapshot = crate::agent::snapshot::AgentRunSnapshot::from_agent(&agent);
        let mut ctx = completed_context("中文🙂abc".repeat(500));

        TruncationStage.run(&mut ctx, &snapshot).await?;

        let output = ctx
            .output
            .as_deref()
            .ok_or_else(|| ReactError::Other("truncation stage produced no output".to_string()))?;
        assert!(output.contains("Output truncated"));
        assert!(ctx.result.as_ref().is_some_and(|result| result.truncated));
        let result = ctx
            .result
            .as_ref()
            .ok_or_else(|| ReactError::Other("truncation stage lost tool result".to_string()))?;
        assert_eq!(
            result.metadata.get("output_handling").map(String::as_str),
            Some("truncated")
        );
        assert!(metadata_u64(result, "original_bytes") > metadata_u64(result, "returned_bytes"));
        assert!(metadata_usize(result, "estimated_tokens") > 20);
        Ok(())
    }

    /// 验证截断阶段记录内联输出的度量指标。
    #[tokio::test]
    async fn truncation_stage_records_inline_output_metrics() -> Result<()> {
        let agent = crate::agent::ReactAgentBuilder::new()
            .model("test-model")
            .max_tool_output_tokens(100)
            .build()?;
        let snapshot = crate::agent::snapshot::AgentRunSnapshot::from_agent(&agent);
        let mut ctx = completed_context("short output".to_string());

        TruncationStage.run(&mut ctx, &snapshot).await?;

        let result = ctx
            .result
            .as_ref()
            .ok_or_else(|| ReactError::Other("truncation stage lost tool result".to_string()))?;
        assert!(!result.truncated);
        assert_eq!(
            result.metadata.get("output_handling").map(String::as_str),
            Some("inline")
        );
        assert_eq!(metadata_u64(result, "original_bytes"), 12);
        assert_eq!(metadata_u64(result, "returned_bytes"), 12);
        assert!(metadata_usize(result, "estimated_tokens") > 0);
        Ok(())
    }

    /// 验证溢出目录不可写时截断阶段会回退处理。
    #[tokio::test]
    async fn truncation_stage_falls_back_when_spill_directory_is_unwritable() -> Result<()> {
        let working_dir_file =
            tempfile::NamedTempFile::new().map_err(|error| ReactError::Other(error.to_string()))?;
        let agent = crate::agent::ReactAgentBuilder::new()
            .model("test-model")
            .working_dir(working_dir_file.path())
            .tool_output_artifacts(echo_core::tools::artifact::ToolOutputArtifactConfig::new(
                working_dir_file.path().join("artifacts"),
                "test",
            ))
            .build()?;
        let snapshot = crate::agent::snapshot::AgentRunSnapshot::from_agent(&agent);
        let mut ctx = completed_context("失败回退🙂".repeat(300_000));

        TruncationStage.run(&mut ctx, &snapshot).await?;

        let result = ctx
            .result
            .as_ref()
            .ok_or_else(|| ReactError::Other("truncation stage lost tool result".to_string()))?;
        assert!(result.truncated);
        assert_eq!(
            result.metadata.get("output_handling").map(String::as_str),
            Some("spill_failed_truncated")
        );
        assert!(result.metadata.contains_key("spill_error"));
        assert!(
            ctx.output
                .as_deref()
                .is_some_and(|output| output.contains("Output truncated"))
        );
        Ok(())
    }

    /// 验证默认流水线在输出预算处理之后才记录追踪。
    #[test]
    fn default_pipeline_records_trace_after_output_budgeting() {
        let pipeline = ToolExecutionPipeline::default_pipeline();
        let names: Vec<&str> = pipeline.stages.iter().map(|stage| stage.name()).collect();
        let truncation = names.iter().position(|name| *name == "truncation");
        let trace = names.iter().position(|name| *name == "trace_recording");
        assert!(matches!((truncation, trace), (Some(left), Some(right)) if left < right));
    }

    /// 验证失败的工具结果仍保持为终态失败。
    #[tokio::test]
    async fn unsuccessful_tool_result_remains_a_terminal_failure() -> crate::error::Result<()> {
        let agent = crate::agent::ReactAgentBuilder::new()
            .model("test-model")
            .enable_tools()
            .tool(Box::new(InvalidResultTool))
            .build()?;
        let snapshot = crate::agent::snapshot::AgentRunSnapshot::from_agent(&agent);
        let result = snapshot
            .execute_tool_with_policy(
                "call-invalid".to_string(),
                "invalid_result",
                &ToolParameters::new(),
                &serde_json::json!({}),
                None,
            )
            .await;
        let failure = match result {
            Ok(_) => {
                return Err(crate::error::ReactError::Other(
                    "unsuccessful ToolResult was projected as success".to_string(),
                ));
            }
            Err(failure) => failure,
        };

        assert_eq!(
            failure.result.failure.map(|failure| failure.category),
            Some(crate::tools::ToolFailureCategory::InvalidArguments)
        );
        Ok(())
    }

    /// 验证流式流水线会转发调用资源守卫。
    #[tokio::test]
    async fn streaming_pipeline_forwards_invocation_resource_guards() -> crate::error::Result<()> {
        let observed = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let agent = crate::agent::ReactAgentBuilder::new()
            .model("test-model")
            .enable_tools()
            .tool(Box::new(ResourceGuardStreamingTool {
                observed_guard_count: Arc::clone(&observed),
            }))
            .build()?;
        let invocation = echo_core::agent::AgentInvocationContext {
            resource_guards: vec![InvocationResourceGuard::new("stream-lease".to_string())],
            ..echo_core::agent::AgentInvocationContext::default()
        };
        let snapshot = crate::agent::snapshot::AgentRunSnapshot::from_agent_with_invocation(
            &agent,
            &invocation,
        );
        let (stream_tx, _stream_rx) = mpsc::channel(4);
        let execution = snapshot
            .execute_tool_with_policy(
                "guard-stream-call".to_string(),
                "resource_guard_stream",
                &ToolParameters::new(),
                &serde_json::json!({}),
                Some(stream_tx),
            )
            .await;
        if let Err(failure) = execution {
            return Err(crate::error::ReactError::Other(format!(
                "streaming guard tool failed: {}",
                failure.result.output
            )));
        }
        assert_eq!(observed.load(std::sync::atomic::Ordering::SeqCst), 1);
        Ok(())
    }

    /**
     * 并发运行两个单工具的 ExecuteStage 实例；本测试锁定“按流区分身份”，
     * 并验证终态事件确实反映了真实的完成顺序（执行事实）。这不是批处理契约
     * 测试：并发批处理路径（一轮内多次工具调用）按调用顺序发出结果，由
     * stream_channel.rs 中的 `concurrent_batch_results_follow_call_order`
     * 覆盖（F-RCT-04-P1-01）。
     */
    #[tokio::test]
    async fn multiplexed_streams_preserve_identity_and_terminal_order() -> crate::error::Result<()>
    {
        let agent = crate::agent::ReactAgentBuilder::new()
            .model("test-model")
            .enable_tools()
            .tool(Box::new(InterleavingTool))
            .build()?;
        let snapshot = Arc::new(crate::agent::snapshot::AgentRunSnapshot::from_agent(&agent));
        let (stream_tx, mut stream_rx) = mpsc::channel(64);
        let (completion_tx, mut completion_rx) = mpsc::channel(2);

        for mut ctx in [
            streaming_context("call-a", "a", 0, 60, stream_tx.clone()),
            streaming_context("call-b", "b", 10, 10, stream_tx.clone()),
        ] {
            let snapshot = Arc::clone(&snapshot);
            let completion_tx = completion_tx.clone();
            tokio::spawn(async move {
                let result = ExecuteStage.run(&mut ctx, &snapshot).await;
                let _ = completion_tx
                    .send((
                        ctx.call_id.clone(),
                        ctx.tool_name.clone(),
                        result,
                        ctx.result,
                    ))
                    .await;
            });
        }
        drop(stream_tx);
        drop(completion_tx);

        let mut events = Vec::new();
        let mut terminal_count = 0;
        while terminal_count < 2 {
            tokio::select! {
                biased;
                Some(event) = stream_rx.recv() => {
                    if let ToolPipelineEvent::Stream { call_id, name, event } = event {
                        events.push(AgentEvent::ToolStream { call_id, name, event });
                    }
                }
                Some((call_id, name, execution, result)) = completion_rx.recv() => {
                    execution?;
                    let result = result.ok_or_else(|| {
                        ReactError::Other("execute stage did not set a result".to_string())
                    })?;
                    events.push(AgentEvent::ToolResult {
                        call_id,
                        name,
                        result,
                    });
                    terminal_count += 1;
                }
            }
        }
        events.push(AgentEvent::ToolBatchEnd);

        let stream_chunks: Vec<(String, String)> = events
            .iter()
            .filter_map(|event| match event {
                AgentEvent::ToolStream {
                    call_id,
                    event: ToolStreamEvent::Output { chunk, .. },
                    ..
                } => Some((call_id.clone(), chunk.clone())),
                _ => None,
            })
            .collect();
        assert_eq!(
            stream_chunks,
            vec![
                ("call-a".into(), "a-1".into()),
                ("call-b".into(), "b-1".into()),
                ("call-b".into(), "b-2".into()),
                ("call-a".into(), "a-2".into()),
            ]
        );

        let terminal_ids: Vec<String> = events
            .iter()
            .filter_map(|event| match event {
                AgentEvent::ToolResult { call_id, .. } => Some(call_id.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(terminal_ids, vec!["call-b", "call-a"]);
        assert!(matches!(events.last(), Some(AgentEvent::ToolBatchEnd)));
        Ok(())
    }

    /// 记录一次流水线运行中触发了哪些终态回调方法。
    #[derive(Default)]
    struct RecordingCallback {
        /// on_tool_end 被调用的次数。
        tool_end_calls: std::sync::atomic::AtomicUsize,
        /// on_tool_error 被调用的次数。
        tool_error_calls: std::sync::atomic::AtomicUsize,
        /// on_tool_interrupted_with_id 被调用的次数。
        tool_interrupted_calls: std::sync::atomic::AtomicUsize,
        /// 中断回调收到的输入参数。
        interrupted_inputs: std::sync::Mutex<Vec<Value>>,
        /// 中断回调收到的错误文本。
        interrupted_errors: std::sync::Mutex<Vec<String>>,
    }

    impl echo_core::agent::AgentCallback for RecordingCallback {
        /// 记录一次工具成功结束回调。
        fn on_tool_end<'a>(
            &'a self,
            _agent: &'a str,
            _tool: &'a str,
            _result: &'a str,
        ) -> futures::future::BoxFuture<'a, ()> {
            Box::pin(async {
                self.tool_end_calls
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            })
        }

        /// 记录一次工具错误回调。
        fn on_tool_error<'a>(
            &'a self,
            _agent: &'a str,
            _tool: &'a str,
            _err: &'a ReactError,
        ) -> futures::future::BoxFuture<'a, ()> {
            Box::pin(async {
                self.tool_error_calls
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            })
        }

        /// 记录一次工具中断回调及其输入与错误。
        fn on_tool_interrupted_with_id<'a>(
            &'a self,
            _agent: &'a str,
            _call_id: &'a str,
            _tool: &'a str,
            input: &'a Value,
            error: &'a ReactError,
        ) -> futures::future::BoxFuture<'a, ()> {
            Box::pin(async move {
                self.tool_interrupted_calls
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                self.interrupted_inputs
                    .lock()
                    .unwrap_or_else(|error| error.into_inner())
                    .push(input.clone());
                self.interrupted_errors
                    .lock()
                    .unwrap_or_else(|error| error.into_inner())
                    .push(error.to_string());
            })
        }
    }

    /// 模拟被取消执行的测试工具：可返回取消失败结果或直接返回取消错误。
    struct InterruptedErrorTool {
        /// 为 true 时返回失败结果，否则返回错误。
        returns_result: bool,
    }

    impl Tool for InterruptedErrorTool {
        /// 返回工具名称。
        fn name(&self) -> &str {
            "interrupted_error"
        }

        /// 返回工具描述。
        fn description(&self) -> &str {
            "test stub for interrupted execution classification"
        }

        /// 返回参数 JSON Schema。
        fn parameters(&self) -> Value {
            serde_json::json!({"type": "object"})
        }

        /// 按 returns_result 返回取消失败结果或取消错误。
        fn execute<'a>(
            &'a self,
            _parameters: ToolParameters,
        ) -> futures::future::BoxFuture<'a, Result<ToolResult>> {
            Box::pin(async move {
                if self.returns_result {
                    Ok(ToolResult::failure(
                        crate::tools::ToolFailureCategory::Cancelled,
                        "completed cancellation result",
                    ))
                } else {
                    Err(ReactError::from(crate::error::ToolError::Cancelled(
                        "reverse extension call".into(),
                    )))
                }
            })
        }
    }

    /// 验证带类型的执行中断会保留输入，且不会重新分类已返回的结果。
    #[tokio::test]
    async fn typed_execution_interruption_preserves_input_without_reclassifying_returned_result()
    -> Result<()> {
        for returns_result in [false, true] {
            let callback = Arc::new(RecordingCallback::default());
            let agent = crate::agent::ReactAgentBuilder::new()
                .model("test-model")
                .callback(callback.clone())
                .tool(Box::new(InterruptedErrorTool { returns_result }))
                .build()?;
            let snapshot = crate::agent::snapshot::AgentRunSnapshot::from_agent(&agent);
            let mut ctx = completed_context(String::new());
            ctx.result = None;
            ctx.tool_name = "interrupted_error".to_string();
            ctx.input = serde_json::json!({"admitted": "canonical input"});
            ctx.params =
                HashMap::from([("admitted".to_string(), serde_json::json!("canonical input"))]);
            ToolExecutionPipeline::default_pipeline()
                .run(&mut ctx, &snapshot)
                .await?;

            assert_eq!(
                ctx.result.as_ref().map(|result| result.success),
                Some(false)
            );
            assert_eq!(
                callback
                    .tool_interrupted_calls
                    .load(std::sync::atomic::Ordering::SeqCst),
                usize::from(!returns_result)
            );
            assert_eq!(
                callback
                    .tool_error_calls
                    .load(std::sync::atomic::Ordering::SeqCst),
                usize::from(returns_result)
            );
            if !returns_result {
                assert_eq!(
                    callback
                        .interrupted_inputs
                        .lock()
                        .unwrap_or_else(|error| error.into_inner())
                        .as_slice(),
                    &[serde_json::json!({"admitted": "canonical input"})]
                );
            }
        }
        Ok(())
    }

    /// 验证 post-use 拒绝时，中断回调与追踪共享同一个终态原因。
    #[tokio::test]
    async fn post_use_rejection_keeps_interrupted_callback_and_trace_on_one_terminal_reason()
    -> Result<()> {
        use crate::skills::hooks::{HookAction, HookEvent, HookRule, HooksDefinition};
        use crate::trace::{InMemoryRunStore, RunEvent, RunStore};

        let callback = Arc::new(RecordingCallback::default());
        let audit = Arc::new(crate::audit::InMemoryAuditLogger::new());
        let store = Arc::new(InMemoryRunStore::new());
        let agent = crate::agent::ReactAgentBuilder::new()
            .model("test-model")
            .callback(callback.clone())
            .callback(Arc::new(crate::audit::AuditCallback::new(
                audit.clone(),
                "test-agent",
                None,
            )))
            .with_run_store(store.clone())
            .tool(Box::new(InterruptedErrorTool {
                returns_result: false,
            }))
            .build()?;
        let (command, shell) = if cfg!(target_os = "windows") {
            (
                "[Console]::In.ReadToEnd() | Out-Null; exit 2",
                Some("powershell".to_string()),
            )
        } else {
            ("read -r hook_context; exit 2", None)
        };
        let mut hooks = HooksDefinition::default();
        hooks.add_rules(
            HookEvent::PostToolUseFailure,
            vec![HookRule {
                matcher: "interrupted_error".to_string(),
                hooks: vec![HookAction::Command {
                    command: command.to_string(),
                    shell,
                    timeout: 10,
                }],
            }],
        );
        agent
            .hook_registry()
            .write()
            .await
            .register_user_hooks(hooks);
        let legacy = agent.capture_legacy_external_context();
        let run_id = agent
            .start_legacy_trace_run("interrupted post-hook", &legacy)
            .await
            .ok_or_else(|| ReactError::Other("trace run did not start".into()))?;
        let snapshot = crate::agent::snapshot::AgentRunSnapshot::from_agent(&agent);
        let input = serde_json::json!({"admitted": "canonical input"});
        let params =
            HashMap::from([("admitted".to_string(), serde_json::json!("canonical input"))]);
        let failure = snapshot
            .execute_tool_with_policy(
                "post-interruption".to_string(),
                "interrupted_error",
                &params,
                &input,
                None,
            )
            .await
            .err()
            .ok_or_else(|| ReactError::Other("post-use block was not enforced".into()))?;
        let final_reason = failure
            .result
            .error
            .as_deref()
            .ok_or_else(|| ReactError::Other("terminal reason missing".into()))?;
        assert!(failure.error.to_string().contains(final_reason));
        assert_eq!(
            failure.result.failure.as_ref().map(|value| value.category),
            Some(crate::tools::ToolFailureCategory::Cancelled)
        );
        assert_eq!(
            callback
                .tool_interrupted_calls
                .load(std::sync::atomic::Ordering::SeqCst),
            1
        );
        assert_eq!(
            callback
                .tool_error_calls
                .load(std::sync::atomic::Ordering::SeqCst),
            0
        );
        assert_eq!(
            callback
                .interrupted_errors
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .as_slice(),
            &[final_reason.to_string()]
        );
        assert_eq!(
            callback
                .interrupted_inputs
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .as_slice(),
            &[input]
        );
        let run = store
            .load(&run_id)
            .await?
            .ok_or_else(|| ReactError::Other("trace run missing".into()))?;
        assert!(run.events.iter().any(|event| matches!(
            event,
            RunEvent::ToolError { message, failure, .. }
                if message == final_reason
                    && failure.as_ref().is_some_and(|value| value.category == crate::tools::ToolFailureCategory::Cancelled)
        )));
        assert!(
            !run.events
                .iter()
                .any(|event| matches!(event, RunEvent::ToolExecutionSkipped { .. }))
        );
        assert!(audit.snapshot().iter().any(|event| matches!(
            &event.event_type,
            crate::audit::AuditEventType::ToolCall { output, success: false, .. }
                if output == final_reason
        )));
        Ok(())
    }

    /**
     * 确定性成功的工具，使流水线能够在不接触文件系统或 shell 的情况下
     * 演练成功路径下的 PostToolUse。
     */
    struct EffectTool;

    /// 返回已确认的带类型效果（读、编辑、测试）的测试工具。
    struct TypedEffectTool;

    impl Tool for TypedEffectTool {
        /// 返回工具名称。
        fn name(&self) -> &str {
            "shell"
        }

        /// 返回工具描述。
        fn description(&self) -> &str {
            "test stub returning confirmed typed effects"
        }

        /// 返回参数 JSON Schema。
        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({"type": "object"})
        }

        /// 返回不透明输出并附带若干已确认的效果。
        fn execute<'a>(
            &'a self,
            _parameters: ToolParameters,
        ) -> futures::future::BoxFuture<'a, crate::error::Result<ToolResult>> {
            Box::pin(async {
                Ok(ToolResult::success("opaque output")
                    .with_effect(echo_core::tools::ToolEffect::FileRead {
                        path: "src/read.rs".to_string(),
                    })
                    .with_effect(echo_core::tools::ToolEffect::FileEdit {
                        path: "src/lib.rs".to_string(),
                    })
                    .with_effect(echo_core::tools::ToolEffect::TestRun {
                        command: "cargo test".to_string(),
                        passed: false,
                        failure_count: Some(2),
                    }))
            })
        }
    }

    /// 永不完成的 post-use 阶段，用于模拟卡住的钩子。
    struct PendingPostStage;

    #[async_trait::async_trait]
    impl PipelineStage for PendingPostStage {
        /// 冒充 post_tool_use_hook 阶段名称。
        fn name(&self) -> &str {
            "post_tool_use_hook"
        }

        /// 永远挂起，不返回。
        async fn run(
            &self,
            _ctx: &mut ToolExecutionContext,
            _snapshot: &crate::agent::snapshot::AgentRunSnapshot,
        ) -> Result<()> {
            std::future::pending().await
        }
    }

    /// 把输出方向内容清空的测试防护。
    struct EmptyOutputGuard;

    impl crate::guard::Guard for EmptyOutputGuard {
        /// 返回防护名称。
        fn name(&self) -> &str {
            "empty_output"
        }

        /// 输出方向转换为空内容，输入方向直接放行。
        fn check<'a>(
            &'a self,
            _content: &'a str,
            direction: crate::guard::GuardDirection,
        ) -> futures::future::BoxFuture<'a, crate::error::Result<crate::guard::GuardResult>>
        {
            Box::pin(async move {
                if direction == crate::guard::GuardDirection::Output {
                    Ok(crate::guard::GuardResult::Transform {
                        content: String::new(),
                        reasons: vec!["redacted".to_string()],
                    })
                } else {
                    Ok(crate::guard::GuardResult::Pass)
                }
            })
        }
    }

    impl Tool for EffectTool {
        /// 返回工具名称。
        fn name(&self) -> &str {
            "shell"
        }

        /// 返回工具描述。
        fn description(&self) -> &str {
            "test stub producing a successful effect result"
        }

        /// 返回参数 JSON Schema。
        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({"type": "object"})
        }

        /// 返回固定的成功结果。
        fn execute<'a>(
            &'a self,
            _parameters: ToolParameters,
        ) -> futures::future::BoxFuture<'a, crate::error::Result<ToolResult>> {
            Box::pin(async { Ok(ToolResult::success("real effect")) })
        }
    }

    /// 验证审计不会恢复已被防护清空的输出。
    #[tokio::test]
    async fn audit_does_not_restore_output_cleared_by_guard() -> Result<()> {
        let audit = Arc::new(crate::audit::InMemoryAuditLogger::new());
        let agent = crate::agent::ReactAgentBuilder::new()
            .model("test-model")
            .audit_logger(audit.clone())
            .guard(Arc::new(EmptyOutputGuard))
            .tool(Box::new(EffectTool))
            .build()?;
        let snapshot = crate::agent::snapshot::AgentRunSnapshot::from_agent(&agent);
        let outcome = snapshot
            .execute_tool_with_policy(
                "guard-clears-output".to_string(),
                "shell",
                &ToolParameters::new(),
                &serde_json::json!({}),
                None,
            )
            .await
            .map_err(|failure| {
                ReactError::Other(format!(
                    "tool unexpectedly failed: {}",
                    failure.result.output
                ))
            })?;
        assert_eq!(outcome.result.output, "");
        let terminal_outputs: Vec<_> = audit
            .snapshot()
            .into_iter()
            .filter_map(|event| match event.event_type {
                crate::audit::AuditEventType::ToolCall {
                    output, success, ..
                } => Some((output, success)),
                _ => None,
            })
            .collect();
        assert_eq!(terminal_outputs, vec![(String::new(), true)]);
        Ok(())
    }

    /// 验证输出防护会撤销防护前的产物访问能力。
    #[tokio::test]
    async fn output_guard_retires_pre_guard_artifact_capability() -> Result<()> {
        let agent = crate::agent::ReactAgentBuilder::new()
            .model("test-model")
            .guard(Arc::new(EmptyOutputGuard))
            .build()?;
        let snapshot = crate::agent::snapshot::AgentRunSnapshot::from_agent(&agent);
        let mut ctx = completed_context("raw secret output".to_string());
        if let Some(result) = ctx.result.as_mut() {
            result.artifact = Some(echo_core::tools::artifact::ToolOutputArtifactRef {
                path: std::path::PathBuf::from("/private/raw-output.txt"),
                artifact_bytes: 17,
                payload_bytes: 17,
                sha256: "a".repeat(64),
                retention: "test".to_string(),
            });
        }

        OutputGuardStage.run(&mut ctx, &snapshot).await?;
        TruncationStage.run(&mut ctx, &snapshot).await?;

        assert_eq!(ctx.output.as_deref(), Some(""));
        assert!(
            ctx.result
                .as_ref()
                .is_some_and(|result| result.artifact.is_none())
        );
        Ok(())
    }

    /// 验证带类型的工具效果被直接投影，而不依赖名称或输出推断。
    #[tokio::test]
    async fn typed_tool_effects_are_projected_without_name_or_output_inference() -> Result<()> {
        use crate::trace::{InMemoryRunStore, RunEvent, RunStore};

        let store = Arc::new(InMemoryRunStore::new());
        let agent = crate::agent::ReactAgentBuilder::new()
            .model("test-model")
            .with_run_store(store.clone())
            .tool(Box::new(TypedEffectTool))
            .build()?;
        let legacy = agent.capture_legacy_external_context();
        let run_id = agent
            .start_legacy_trace_run("typed effects", &legacy)
            .await
            .ok_or_else(|| ReactError::Other("trace did not start".to_string()))?;
        let snapshot = crate::agent::snapshot::AgentRunSnapshot::from_agent(&agent);
        let mut ctx = completed_context("opaque output".to_string());
        ExecuteStage.run(&mut ctx, &snapshot).await?;
        TraceRecordingStage.run(&mut ctx, &snapshot).await?;
        let run = store
            .load(&run_id)
            .await?
            .ok_or_else(|| ReactError::Other("trace disappeared".to_string()))?;
        assert!(run.events.iter().any(|event| matches!(
            event,
            RunEvent::FileRead { tool, path } if tool == "shell" && path == "src/read.rs"
        )));
        assert!(run.events.iter().any(|event| matches!(
            event,
            RunEvent::FileEdit { tool, path } if tool == "shell" && path == "src/lib.rs"
        )));
        assert!(run.events.iter().any(|event| matches!(
            event,
            RunEvent::TestRun { command, passed: false, failure_count: Some(2) }
            if command == "cargo test"
        )));
        Ok(())
    }

    /// 验证 post-use 阶段卡住时，已确认的效果仍会保留。
    #[tokio::test]
    async fn confirmed_effects_survive_stalled_post_use_stage() -> Result<()> {
        use crate::trace::{InMemoryRunStore, RunEvent, RunStore};

        let store = Arc::new(InMemoryRunStore::new());
        let agent = crate::agent::ReactAgentBuilder::new()
            .model("test-model")
            .with_run_store(store.clone())
            .tool(Box::new(TypedEffectTool))
            .build()?;
        let legacy = agent.capture_legacy_external_context();
        let run_id = agent
            .start_legacy_trace_run("effect before post hook", &legacy)
            .await
            .ok_or_else(|| ReactError::Other("trace did not start".to_string()))?;
        let snapshot = crate::agent::snapshot::AgentRunSnapshot::from_agent(&agent);
        let mut ctx = completed_context(String::new());
        let pipeline = ToolExecutionPipeline {
            stages: vec![Box::new(ExecuteStage), Box::new(PendingPostStage)],
        };
        assert!(
            tokio::time::timeout(
                std::time::Duration::from_millis(50),
                pipeline.run(&mut ctx, &snapshot),
            )
            .await
            .is_err()
        );
        let run = store
            .load(&run_id)
            .await?
            .ok_or_else(|| ReactError::Other("trace disappeared".to_string()))?;
        assert_eq!(
            run.events
                .iter()
                .filter(
                    |event| matches!(event, RunEvent::FileEdit { path, .. } if path == "src/lib.rs")
                )
                .count(),
            1
        );
        assert!(run.events.iter().any(|event| matches!(
            event,
            RunEvent::TestRun {
                failure_count: Some(2),
                ..
            }
        )));
        assert!(
            !run.events
                .iter()
                .any(|event| matches!(event, RunEvent::ToolResult { .. }))
        );
        Ok(())
    }

    /// 验证失败的工具输出在交给调用方和审计前已被防护处理。
    #[tokio::test]
    async fn failed_tool_output_is_guarded_before_caller_and_audit() -> Result<()> {
        let audit = Arc::new(crate::audit::InMemoryAuditLogger::new());
        let agent = crate::agent::ReactAgentBuilder::new()
            .model("test-model")
            .audit_logger(audit.clone())
            .guard(Arc::new(EmptyOutputGuard))
            .tool(Box::new(FailedOutputTool))
            .build()?;
        let snapshot = crate::agent::snapshot::AgentRunSnapshot::from_agent(&agent);
        let failure = snapshot
            .execute_tool_with_policy(
                "failed-output".to_string(),
                "failed_output",
                &ToolParameters::new(),
                &serde_json::json!({}),
                None,
            )
            .await
            .err()
            .ok_or_else(|| ReactError::Other("failed tool unexpectedly succeeded".to_string()))?;
        assert_eq!(failure.result.error.as_deref(), Some("execution failed"));
        assert_eq!(failure.result.output, "");
        let terminal_outputs: Vec<_> = audit
            .snapshot()
            .into_iter()
            .filter_map(|event| match event.event_type {
                crate::audit::AuditEventType::ToolCall {
                    output, success, ..
                } => Some((output, success)),
                _ => None,
            })
            .collect();
        assert_eq!(terminal_outputs, vec![(String::new(), false)]);
        Ok(())
    }

    /// 验证仅有错误的审计使用经过防护的诊断信息。
    #[tokio::test]
    async fn error_only_audit_uses_guarded_diagnostic() -> Result<()> {
        let audit = Arc::new(crate::audit::InMemoryAuditLogger::new());
        let agent = crate::agent::ReactAgentBuilder::new()
            .model("test-model")
            .audit_logger(audit.clone())
            .guard(Arc::new(EmptyOutputGuard))
            .tool(Box::new(ErrorOnlyTool))
            .build()?;
        let snapshot = crate::agent::snapshot::AgentRunSnapshot::from_agent(&agent);
        let failure = snapshot
            .execute_tool_with_policy(
                "error-only".to_string(),
                "error_only",
                &ToolParameters::new(),
                &serde_json::json!({}),
                None,
            )
            .await
            .err()
            .ok_or_else(|| {
                ReactError::Other("error-only tool unexpectedly succeeded".to_string())
            })?;
        assert_eq!(
            failure.result.error.as_deref(),
            Some("Bearer sensitive-token")
        );
        assert_eq!(failure.result.output, "");
        let terminal_outputs: Vec<_> = audit
            .snapshot()
            .into_iter()
            .filter_map(|event| match event.event_type {
                crate::audit::AuditEventType::ToolCall {
                    output, success, ..
                } => Some((output, success)),
                _ => None,
            })
            .collect();
        assert_eq!(terminal_outputs, vec![(String::new(), false)]);
        Ok(())
    }

    /// 验证执行前被阻止的调用不会触发成功回调。
    #[tokio::test]
    async fn pre_execution_block_does_not_emit_success_callback() -> Result<()> {
        let callback = Arc::new(RecordingCallback::default());
        let audit = Arc::new(crate::audit::InMemoryAuditLogger::new());
        let agent = crate::agent::ReactAgentBuilder::new()
            .model("test-model")
            .callback(callback.clone())
            .audit_logger(audit.clone())
            .tool(Box::new(EffectTool))
            .build()?;
        let snapshot = crate::agent::snapshot::AgentRunSnapshot::from_agent(&agent);
        let mut ctx = completed_context(String::new());
        ctx.result = None;
        ctx.block(
            crate::tools::ToolFailureCategory::Permanent,
            "denied".to_string(),
        );
        ToolExecutionPipeline::default_pipeline()
            .run(&mut ctx, &snapshot)
            .await?;
        assert!(ctx.result.is_none());
        assert_eq!(
            callback
                .tool_end_calls
                .load(std::sync::atomic::Ordering::SeqCst),
            0
        );
        assert_eq!(
            callback
                .tool_error_calls
                .load(std::sync::atomic::Ordering::SeqCst),
            0
        );
        assert!(audit.snapshot().is_empty());
        Ok(())
    }

    /// #102：失败的 ToolResult 必须到达 on_tool_error，而不是 on_tool_end。
    #[tokio::test]
    async fn failed_tool_result_routes_to_on_tool_error_not_on_tool_end() -> Result<()> {
        let callback = Arc::new(RecordingCallback::default());
        let audit = Arc::new(crate::audit::InMemoryAuditLogger::new());

        let agent = crate::agent::ReactAgentBuilder::new()
            .model("test-model")
            .callback(callback.clone())
            .audit_logger(audit.clone())
            .tool(Box::new(InvalidResultTool))
            .build()?;
        let snapshot = crate::agent::snapshot::AgentRunSnapshot::from_agent(&agent);
        let mut ctx = completed_context(String::new());
        ctx.result = None;
        ctx.tool_name = "invalid_result".to_string();
        ToolExecutionPipeline::default_pipeline()
            .run(&mut ctx, &snapshot)
            .await?;

        assert_eq!(
            callback
                .tool_error_calls
                .load(std::sync::atomic::Ordering::SeqCst),
            1,
            "failure must reach on_tool_error"
        );
        assert_eq!(
            callback
                .tool_end_calls
                .load(std::sync::atomic::Ordering::SeqCst),
            0,
            "failure must not be reported as on_tool_end success"
        );
        assert_eq!(
            audit
                .snapshot()
                .iter()
                .filter(|event| matches!(
                    event.event_type,
                    crate::audit::AuditEventType::ToolCall { success: false, .. }
                ))
                .count(),
            1,
        );
        assert!(audit.snapshot().iter().any(|event| matches!(
            &event.event_type,
            crate::audit::AuditEventType::ToolCall { output, success: false, .. }
            if output.contains("missing query")
        )));
        Ok(())
    }

    /**
     * #102：真实效果发生之后的 PostToolUse 阻止，不能使剩余的观察类阶段
     * （OutputGuard/Truncation/Trace/End）短路跳过。
     * 使用 hooks.rs:1428 中实现的“退出码 2 表示阻止”约定。
     */
    #[tokio::test]
    async fn post_hook_block_after_effect_still_records_full_observations() -> Result<()> {
        use crate::skills::hooks::{HookAction, HookEvent, HookRule, HooksDefinition};

        use crate::trace::{InMemoryRunStore, Run, RunStatus, RunStore};

        let callback = Arc::new(RecordingCallback::default());
        let audit = Arc::new(crate::audit::InMemoryAuditLogger::new());
        let store = Arc::new(InMemoryRunStore::new());
        store
            .save(Run {
                run_id: "post-block".to_string(),
                parent_run_id: None,
                agent_name: "test".to_string(),
                model: "test-model".to_string(),
                provider: None,
                turn_id: None,
                execution_id: None,
                session_id: "test".to_string(),
                status: RunStatus::Running,
                input: String::new(),
                events: Vec::new(),
                final_output: None,
                error: None,
                token_usage: Default::default(),
                timings: Default::default(),
                started_at: chrono::Utc::now(),
                finished_at: None,
            })
            .await?;
        let agent = crate::agent::ReactAgentBuilder::new()
            .model("test-model")
            .callback(callback.clone())
            .audit_logger(audit.clone())
            .with_run_store(store.clone())
            .tool(Box::new(EffectTool))
            .build()?;
        let mut snapshot = crate::agent::snapshot::AgentRunSnapshot::from_agent(&agent);
        snapshot.trace_run_id = Some("post-block".to_string());
        /*
         * 退出前先把钩子上下文读干净，避免 stdin 的写入与显式阻止信号发生
         * 竞态，从而被误判成沙箱错误。
         */
        let (command, shell) = if cfg!(target_os = "windows") {
            (
                "[Console]::In.ReadToEnd() | Out-Null; exit 2",
                Some("powershell".to_string()),
            )
        } else {
            ("read -r hook_context; exit 2", None)
        };
        let mut definition = HooksDefinition::default();
        definition.add_rules(
            HookEvent::PostToolUse,
            vec![HookRule {
                matcher: "shell".to_string(),
                hooks: vec![HookAction::Command {
                    command: command.to_string(),
                    shell,
                    timeout: 10,
                }],
            }],
        );
        snapshot
            .tools
            .hook_registry
            .write()
            .await
            .register_user_hooks(definition);

        let outcome = snapshot
            .execute_tool_with_policy(
                "call-post-block".to_string(),
                "shell",
                &ToolParameters::new(),
                &serde_json::json!({}),
                None,
            )
            .await;
        let failure = outcome.err().ok_or_else(|| {
            ReactError::Other("post-use rejection incorrectly returned success".to_string())
        })?;
        let result = failure.result;
        assert!(!result.success);
        assert!(
            result
                .failure
                .as_ref()
                .is_some_and(|failure| failure.category
                    == crate::tools::ToolFailureCategory::PartialSideEffect
                    && failure.postcondition.is_some())
        );
        assert_eq!(
            callback
                .tool_error_calls
                .load(std::sync::atomic::Ordering::SeqCst),
            1,
            "post-effect policy failure must fire on_tool_error exactly once"
        );
        assert_eq!(
            callback
                .tool_end_calls
                .load(std::sync::atomic::Ordering::SeqCst),
            0
        );
        assert_eq!(result.output, "real effect");
        assert!(result.metadata.contains_key("returned_bytes"));
        let run = store
            .load("post-block")
            .await?
            .ok_or_else(|| ReactError::Other("missing trace run".to_string()))?;
        let terminals: Vec<_> = run
            .events
            .iter()
            .filter_map(|event| match event {
                crate::trace::RunEvent::ToolResult {
                    call_id, success, ..
                } => Some((call_id.as_str(), *success)),
                _ => None,
            })
            .collect();
        assert_eq!(terminals, vec![("call-post-block", false)]);
        assert!(run.events.iter().any(|event| matches!(
            event,
            crate::trace::RunEvent::ToolResult { output_preview: Some(preview), .. }
            if preview == "real effect"
        )));
        assert!(run.events.iter().any(|event| matches!(event,
            crate::trace::RunEvent::ToolError { failure, .. } if failure == &result.failure
        )));
        let audit_events = audit.snapshot();
        let audit_terminals: Vec<_> = audit_events
            .iter()
            .filter_map(|event| match &event.event_type {
                crate::audit::AuditEventType::ToolCall {
                    success, output, ..
                } => Some((*success, output.as_str())),
                _ => None,
            })
            .collect();
        assert_eq!(audit_terminals, vec![(false, "real effect")]);
        Ok(())
    }
}
