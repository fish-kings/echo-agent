//! 上下文管理 + 长期记忆 + 持久化 + 审计

use super::super::ReactAgent;
use super::types::StreamMode;
use crate::llm::types::{Message, Role};
use crate::skills::hooks::{HookContext, HookEvent};
use echo_core::memory::types::{MemoryMeta, MemorySource, MemoryType};
use tracing::{debug, info, warn};

pub(crate) const TURN_MEMORY_CONTEXT_PROJECTION: &str = "echo-agent:turn-memory-context";
pub(crate) const WORKSPACE_CONTEXT_PROJECTION: &str = "echo-agent:workspace-context";

/// (stage4 E1) 压缩前刷写（pre-compaction flush）LLM 调用所使用的提示词。
const PRE_COMPACTION_FLUSH_PROMPT: &str = "\
You are a memory-compaction flusher. The conversation below is about to be \
compressed (older messages will be summarized away). Identify DURABLE facts \
worth persisting long-term: stable user preferences, project facts, \
architecture decisions, debugging lessons, verified error resolutions. Do NOT \
capture transient/session-specific details or task narratives.\n\
Reply with a JSON array of items to persist, each: \
{\"content\": \"<concise fact>\", \"type\": \"user_preference|project_fact|architecture_decision|debugging_lesson|error_resolution|command_pattern|tool_usage\", \"recall_weight\": <0.0-1.0>}.\n\
If nothing is durable, reply exactly: NO_REPLY";

/// 按钩子执行时机（pre/post）分组收集到的消息文本批次。
#[derive(Clone, Default)]
pub(crate) struct HookMessageBatches {
    /// 钩子执行之前收集到的消息。
    pub pre: Vec<String>,
    /// 钩子执行之后收集到的消息。
    pub post: Vec<String>,
}

impl ReactAgent {
    /** 检测值得记忆的对话触发条件，并通过真实的分层记忆管理器将其持久化。
     *
     * 目前只接入了可靠的“用户纠正”路径；显式保存由 `remember` 工具负责处理。
     */
    pub(crate) async fn detect_and_write_memory_triggers(&self, user_message: &str) {
        let Some(layer_manager) = &self.memory_layer_manager else {
            return;
        };

        let assistant_message = {
            let ctx = self.memory.context.lock().await;
            ctx.messages()
                .iter()
                .rev()
                .find(|m| matches!(m.role, Role::Assistant))
                .and_then(|m| m.content.as_text())
                .map(|s| s.to_string())
        };
        let (last_tool_failure, last_tool_success, tool_sequences) =
            match self.memory_trigger_state.lock() {
                Ok(state) => (
                    state.last_tool_failure.clone(),
                    state.last_tool_success.clone(),
                    state.tool_sequences.clone(),
                ),
                Err(e) => {
                    tracing::warn!("memory trigger state lock poisoned: {}", e);
                    (None, None, Vec::new())
                }
            };

        let trigger_ctx = crate::evolution::TriggerContext {
            user_message: Some(user_message.to_string()),
            assistant_message,
            last_tool_failure,
            last_tool_success,
            tool_sequences,
            ..Default::default()
        };
        let detector = crate::evolution::TriggerDetector::new();
        let triggers = detector.detect(&trigger_ctx);
        let mut consumed_error_resolution = false;
        let mut consumed_repeated_workflow = false;
        for trigger in triggers {
            if let Some(sink) = &self.memory_trigger_sink {
                match sink.on_trigger(&trigger).await {
                    Ok(crate::evolution::MemoryTriggerDisposition::Captured) => {
                        consumed_error_resolution |=
                            matches!(trigger.source, crate::memory::MemorySource::ErrorResolution);
                        consumed_repeated_workflow |= matches!(
                            trigger.source,
                            crate::memory::MemorySource::RepeatedWorkflow
                        );
                        continue;
                    }
                    Ok(crate::evolution::MemoryTriggerDisposition::Persist) => {}
                    Err(error) => {
                        tracing::warn!(
                            key = %trigger.suggested_key,
                            %error,
                            "Memory trigger sink failed; falling back to durable write"
                        );
                    }
                }
            }

            let meta =
                crate::memory::MemoryMeta::new(trigger.memory_type, trigger.source, trigger.topic)
                    .with_confidence(trigger.confidence);

            match layer_manager
                .write_memory(&trigger.suggested_key, &trigger.content, meta)
                .await
            {
                Ok(_) => {
                    consumed_error_resolution |=
                        matches!(trigger.source, crate::memory::MemorySource::ErrorResolution);
                    consumed_repeated_workflow |= matches!(
                        trigger.source,
                        crate::memory::MemorySource::RepeatedWorkflow
                    );
                    tracing::info!(
                        key = %trigger.suggested_key,
                        "Memory trigger persisted through layered memory"
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        key = %trigger.suggested_key,
                        error = %e,
                        "Memory trigger write failed"
                    );
                }
            }
        }

        if (consumed_error_resolution || consumed_repeated_workflow)
            && let Ok(mut state) = self.memory_trigger_state.lock()
        {
            if consumed_error_resolution {
                state.last_tool_failure = None;
                state.last_tool_success = None;
            }
            if consumed_repeated_workflow {
                state.tool_sequences.clear();
            }
        }
    }

    /// 重置消息历史，只保留 system prompt，确保每次执行相互独立
    pub(crate) async fn reset_messages(&self) {
        let mut ctx = self.memory.context.lock().await;
        ctx.clear();
        ctx.push(Message::system(self.current_system_prompt()));
        drop(ctx);
        // 以 matcher "clear" 触发 SessionStart 钩子
        let start_result = self
            .fire_lifecycle_hook(HookEvent::SessionStart, Some("clear"))
            .await;
        if start_result.block {
            warn!(agent = %self.config.agent_name, reason = ?start_result.block_reason, "SessionStart hook blocked new session");
        }
    }

    /// 重置运行时上下文：清空消息历史、重置技能激活状态与快照游标。
    async fn reset_runtime_context(&self) {
        self.reset_messages().await;
        self.tools.skill_registry.reset_activation_state();
        *self.memory.transcript_projection_cursor.lock().await =
            crate::agent::snapshot::TranscriptProjectionCursor::default();
        self.set_working_dir(self.memory.configured_working_dir.clone());
    }

    /** 在输入守卫（input guard）可能产生终态结果之前，恢复受管运行时身份。
     *
     * 这里只恢复持久化的运行时状态；生命周期钩子与原始用户输入仍然会
     * 推迟到常规上下文准备阶段再处理。
     */
    pub(crate) async fn hydrate_managed_runtime_before_input_guard(
        &self,
        scope_id: &str,
        runtime_state_id: &str,
    ) -> crate::error::Result<()> {
        let _previous_hydration = self
            .begin_runtime_state_hydration(Some(runtime_state_id))
            .await;
        self.clear_runtime_snapshots();
        if self
            .resume_from_state_store_scope(scope_id, runtime_state_id)
            .await?
            .is_none()
        {
            self.reset_runtime_context().await;
        }
        self.commit_runtime_state_hydration(Some(runtime_state_id))
            .await;
        Ok(())
    }

    /// 将运行时状态水合标记置为“水合中”，并返回替换前的旧状态。
    pub(crate) async fn begin_runtime_state_hydration(
        &self,
        runtime_state_id: Option<&str>,
    ) -> super::super::subsystems::memory::RuntimeStateHydration {
        let mut hydration = self.memory.runtime_state_hydration.lock().await;
        std::mem::replace(
            &mut *hydration,
            super::super::subsystems::memory::RuntimeStateHydration::Hydrating(
                runtime_state_id.map(str::to_string),
            ),
        )
    }

    /// 将运行时状态水合标记提交为“已水合”。
    pub(crate) async fn commit_runtime_state_hydration(&self, runtime_state_id: Option<&str>) {
        let mut hydration = self.memory.runtime_state_hydration.lock().await;
        *hydration = super::super::subsystems::memory::RuntimeStateHydration::Hydrated(
            runtime_state_id.map(str::to_string),
        );
    }

    /// 清空运行时快照管理器中保存的所有快照。
    pub(crate) fn clear_runtime_snapshots(&self) {
        let mut snapshots = match self.memory.snapshot_manager.write() {
            Ok(snapshots) => snapshots,
            Err(poisoned) => {
                tracing::error!("runtime snapshot manager lock was poisoned");
                poisoned.into_inner()
            }
        };
        if let Some(manager) = snapshots.as_mut() {
            manager.clear();
        }
    }

    /// 测试专用：以当前配置的会话 id 恢复线程上下文。
    #[cfg(test)]
    pub(crate) async fn restore_thread_context(&self) -> crate::error::Result<()> {
        let runtime_state_id = self.config.conversation_id.clone();
        self.restore_thread_context_for(runtime_state_id.as_deref(), runtime_state_id.as_deref())
            .await
    }

    /// 在恢复上下文之前，先协调（reconcile）已配置的、待处理的转录投影。
    async fn reconcile_configured_pending_projection(
        &self,
        runtime_state_id: Option<&str>,
    ) -> crate::error::Result<()> {
        if self.memory.conversation_store.is_none()
            || runtime_state_id != self.config.conversation_id.as_deref()
        {
            return Ok(());
        }
        let snapshot = crate::agent::snapshot::AgentRunSnapshot::from_agent(self);
        if let Some(settlement) = snapshot.reconcile_pending_transcript_projection().await?
            && settlement.status != crate::memory::TranscriptProjectionSettlementStatus::Settled
        {
            return Err(crate::agent::snapshot::transcript_settlement_admission_error(&settlement));
        }
        Ok(())
    }

    /// 从 RuntimeStateStore 检查点恢复线程上下文（消息 + 技能），并触发 SessionStart 钩子。
    pub(crate) async fn restore_thread_context_for(
        &self,
        scope_id: Option<&str>,
        runtime_state_id: Option<&str>,
    ) -> crate::error::Result<()> {
        self.reconcile_configured_pending_projection(runtime_state_id)
            .await?;
        let _previous_hydration = self.begin_runtime_state_hydration(runtime_state_id).await;
        self.clear_runtime_snapshots();
        let agent = self.config.agent_name.clone();
        let mut session_matcher = "startup";

        // 尝试从 RuntimeStateStore 检查点恢复（消息 + 技能）。
        if self.memory.state_store.is_some() {
            let restored = match runtime_state_id {
                Some(runtime_state_id) => {
                    self.resume_from_state_store_scope(
                        scope_id.unwrap_or(runtime_state_id),
                        runtime_state_id,
                    )
                    .await
                }
                None => Ok(None),
            };
            match restored {
                Ok(Some(_cp)) => {
                    info!(agent = %agent, "🔄 Restored from RuntimeStateStore checkpoint");
                    session_matcher = "resume";
                }
                Ok(None) => {
                    debug!(agent = %agent, "New session, starting from empty context");
                    self.reset_runtime_context().await;
                }
                Err(e) => {
                    return Err(crate::error::ReactError::Other(format!(
                        "Runtime checkpoint recovery failed for agent '{agent}': {e}"
                    )));
                }
            }
        } else {
            self.reset_runtime_context().await;
        }

        // 以合适的 matcher 触发 SessionStart 钩子
        let start_result = self
            .fire_lifecycle_hook(HookEvent::SessionStart, Some(session_matcher))
            .await;
        if start_result.block {
            warn!(agent = %self.config.agent_name, reason = ?start_result.block_reason, "SessionStart hook blocked session restore");
        }
        self.commit_runtime_state_hydration(runtime_state_id).await;
        Ok(())
    }

    /** 在“冷”Agent 实例上，仅恢复一次持久化的聊天记录。
     *
     * 进程内已有的历史记录是权威数据，绝不能在多个回合之间被替换。
     */
    #[cfg(test)]
    pub(crate) async fn restore_chat_context_if_cold(&self) -> crate::error::Result<()> {
        let runtime_state_id = self.config.conversation_id.clone();
        self.restore_chat_context_if_cold_for(
            runtime_state_id.as_deref(),
            runtime_state_id.as_deref(),
        )
        .await
    }

    /// 在“冷”Agent 实例上恢复聊天上下文；若已经水合过（非冷启动）则跳过。
    pub(crate) async fn restore_chat_context_if_cold_for(
        &self,
        scope_id: Option<&str>,
        runtime_state_id: Option<&str>,
    ) -> crate::error::Result<()> {
        if self.memory.conversation_store.is_some() {
            return self
                .restore_thread_context_for(scope_id, runtime_state_id)
                .await;
        }
        self.reconcile_configured_pending_projection(runtime_state_id)
            .await?;
        let is_cold = {
            let context = self.memory.context.lock().await;
            context
                .messages()
                .iter()
                .all(|message| matches!(message.role, Role::System))
        };
        let effective_runtime_state_id = runtime_state_id.map(str::to_string);
        let should_restore = {
            let mut hydration = self.memory.runtime_state_hydration.lock().await;
            match &*hydration {
                super::super::subsystems::memory::RuntimeStateHydration::Hydrated(current)
                    if current == &effective_runtime_state_id =>
                {
                    is_cold
                }
                super::super::subsystems::memory::RuntimeStateHydration::Uninitialized
                    if !is_cold && effective_runtime_state_id == self.config.conversation_id =>
                {
                    *hydration = super::super::subsystems::memory::RuntimeStateHydration::Hydrated(
                        effective_runtime_state_id.clone(),
                    );
                    false
                }
                _ => true,
            }
        };
        if should_restore {
            self.restore_thread_context_for(scope_id, runtime_state_id)
                .await?;
        }
        Ok(())
    }

    /** (stage4 D1) 统一召回入口 —— 委托给 `MemoryRecaller`，使得自动召回路径
     * 与工具召回路径（`LayeredRecallTool`）在统一命名空间下共享同一套
     * 综合评分入口（割裂点 3/9）。
     */
    pub(crate) async fn recall_long_term_memories(
        &self,
        query: &str,
    ) -> crate::error::Result<Vec<crate::memory::store::StoreItem>> {
        let Some(store) = &self.memory.store else {
            return Ok(vec![]);
        };
        let reca = crate::evolution::recall::MemoryRecaller::new(store.clone());
        reca.recall(query, 5).await
    }

    /// 触发一个生命周期钩子，并将任何结果注入到 Agent 上下文中。
    pub async fn fire_lifecycle_hook(
        &self,
        event: HookEvent,
        matcher: Option<&str>,
    ) -> crate::skills::hooks::HookResult {
        let session_id = self.config.session_id.clone().unwrap_or_default();
        let agent_name = self.config.agent_name.clone();

        let context = match event {
            HookEvent::SessionStart => HookContext::for_session_start(
                matcher.unwrap_or("startup"),
                &session_id,
                &agent_name,
            ),
            HookEvent::SessionEnd => {
                HookContext::for_session_end(matcher.unwrap_or("other"), &session_id, &agent_name)
            }
            HookEvent::Stop => HookContext::for_stop(None, &session_id, &agent_name, false),
            HookEvent::Notification => {
                HookContext::for_notification(matcher.unwrap_or(""), &session_id, &agent_name)
            }
            HookEvent::UserPromptSubmit => {
                /* 用户输入文本由调用方（prepare_stream_context /
                 * prepare_react_context）通过 `matcher` 参数传入。
                 * 这样一来，基于内容的钩子匹配器（例如 "\\.docx"）就能够
                 * 通过 glob::Pattern 对真实的用户输入进行匹配。 */
                let prompt_text = matcher.unwrap_or("");
                HookContext::for_user_prompt_submit(prompt_text, matcher, &session_id, &agent_name)
            }
            HookEvent::PreCompact | HookEvent::PostCompact => {
                // 压缩统计信息由调用方设置，这里使用默认值
                let stats = Default::default();
                if event == HookEvent::PreCompact {
                    HookContext::for_pre_compact(
                        &stats,
                        matcher.unwrap_or("auto"),
                        &session_id,
                        &agent_name,
                    )
                } else {
                    HookContext::for_post_compact(
                        &stats,
                        matcher.unwrap_or("auto"),
                        &session_id,
                        &agent_name,
                    )
                }
            }
            HookEvent::ConfigChange => HookContext::for_config_change(
                matcher.unwrap_or(""),
                matcher,
                &session_id,
                &agent_name,
            ),
            HookEvent::InstructionsLoaded => HookContext::for_instructions_loaded(
                matcher.unwrap_or("startup"),
                &[],
                &session_id,
                &agent_name,
            ),
            HookEvent::PostToolBatch => {
                // 批次明细由调用方设置；这里使用默认值
                HookContext::for_post_tool_batch(&[], 0, 0, &session_id, &agent_name)
            }
            HookEvent::StopFailure => {
                HookContext::for_stop_failure("", matcher.unwrap_or(""), &session_id, &agent_name)
            }
            HookEvent::SubagentStart => HookContext::for_subagent_start(
                "",
                "",
                matcher.unwrap_or(""),
                &session_id,
                &agent_name,
            ),
            HookEvent::SubagentStop => HookContext::for_subagent_stop(
                "",
                "",
                matcher.unwrap_or(""),
                /* 该回退映射并不知道真实的终态状态；直接触发 SubagentStop
                 * 的调用方会传入真实状态。这里默认使用 Completed，使这个
                 * 占位调用能够编译通过，而不必去猜测真实状态。 */
                echo_core::hooks::SubagentStopStatus::Completed,
                &session_id,
                &agent_name,
            ),
            HookEvent::TaskCreated => {
                HookContext::for_task_created("", matcher.unwrap_or(""), &session_id, &agent_name)
            }
            HookEvent::TaskStarted => {
                HookContext::for_task_started("", matcher.unwrap_or(""), &session_id, &agent_name)
            }
            HookEvent::TaskCompleted => HookContext::for_task_completed(
                "",
                matcher.unwrap_or(""),
                "",
                /* 该回退映射并不知道真实的终态状态；直接调用的调用方会
                 * 传入真实状态。这里默认使用 Completed，使这个占位调用
                 * 能够编译通过，而不必去猜测真实状态。 */
                echo_core::hooks::TaskTerminalStatus::Completed,
                &session_id,
                &agent_name,
            ),
            // 新事件 —— 使用通用生命周期上下文
            HookEvent::PluginLoaded | HookEvent::PluginDisabled => {
                HookContext::for_lifecycle(event, matcher.unwrap_or(""), &session_id, &agent_name)
            }
            // 演化（Evolution）事件 —— 使用专用的工厂方法
            HookEvent::PostMemoryWrite => HookContext::for_post_memory_write(
                "",
                matcher.unwrap_or(""),
                &session_id,
                &agent_name,
            ),
            HookEvent::MemoryLayerChange => HookContext::for_memory_layer_change(
                "",
                "",
                matcher.unwrap_or(""),
                &session_id,
                &agent_name,
            ),
            // 第五阶段演化事件 —— 使用通用生命周期上下文
            HookEvent::SkillCandidateDetected
            | HookEvent::SkillLifecycleTransition
            | HookEvent::SkillHealthCheck
            | HookEvent::SkillPatchApplied
            | HookEvent::SkillMergeApplied
            | HookEvent::RulePromoted => {
                HookContext::for_lifecycle(event, matcher.unwrap_or(""), &session_id, &agent_name)
            }
            // 工具类事件应直接使用 run_pre_tool_use / run_post_tool_use / run_post_tool_use_failure
            HookEvent::PreToolUse
            | HookEvent::PostToolUse
            | HookEvent::PostToolUseFailure
            | HookEvent::PermissionRequest
            | HookEvent::PermissionDenied => {
                warn!(event = ?event, "Tool event dispatched via fire_lifecycle_hook; use dedicated tool hook methods instead");
                return crate::skills::hooks::HookResult::default();
            }
        };

        let registry = self.tools.hook_registry.read().await.clone();
        let mut result = registry.run_lifecycle_hooks(&context).await;

        // 从钩子结果中注入上下文（合并为一次加锁以做批量处理）
        let event_name = event.as_str();
        if result.injected_context.is_some() || !result.messages.is_empty() {
            let mut ctx = self.memory.context.lock().await;
            if let Some(ctx_text) = &result.injected_context {
                ctx.push(runtime_context_note(
                    &format!("Hook:{event_name}"),
                    ctx_text,
                ));
            }
            for msg in &result.messages {
                ctx.push(runtime_context_note(&format!("Hook:{event_name}"), msg));
            }
        }

        /* ActivateSkill 钩子：直接激活被请求的技能。
         * 成功后清空该字段，使调用方（准备阶段 → 缓存写入器）不会再把它
         * 交给 TriggerSupervisor，从而避免重复激活。 */
        if let Some((ref skill, ref reason)) = result.activate_skill {
            match self.activate_skill(skill).await {
                Ok(()) => {
                    let note = format!("已根据上下文自动激活技能 {skill}:{reason}");
                    let mut ctx = self.memory.context.lock().await;
                    ctx.push(runtime_context_note("Hook:ActivateSkill", &note));
                    info!(skill = %skill, reason = %reason, "Hook activated skill");
                    // 已消费 —— 防止被 supervisor（P4）重复激活
                    result.activate_skill = None;
                }
                Err(e) => {
                    warn!(skill = %skill, error = %e, "Hook-requested skill activation failed");
                    // 保留 result 中的 activate_skill，以便 supervisor（P4）重试
                }
            }
        }

        result
    }

    /** 流式执行的通用初始化逻辑。
     *
     * 根据模式决定是重置上下文，还是从检查点恢复。
     * 返回召回的长期记忆条数（0 表示没有注入任何记忆）。
     */
    pub(crate) async fn prepare_stream_context(
        &self,
        mode: StreamMode,
        input: &str,
        history: &[Message],
        scope_id: Option<&str>,
        runtime_state_id: Option<&str>,
    ) -> crate::error::Result<usize> {
        /* 为新的对话回合清空“编辑前必读”跟踪状态
         * （已与 prepare_react_context 合并；入口层不再单独清空，
         * 以避免重复清空）。 */
        self.clear_read_files();
        match mode {
            StreamMode::Execute => {
                self.restore_thread_context_for(scope_id, runtime_state_id)
                    .await?;
            }
            StreamMode::Chat => {
                self.restore_chat_context_if_cold_for(scope_id, runtime_state_id)
                    .await?;
            }
        }

        self.detect_and_write_memory_triggers(input).await;

        // 为本回合召回相关的长期记忆。
        let mut recalled = 0usize;
        let mut memory_context = None;
        if let Ok(items) = self.recall_long_term_memories(input).await
            && !items.is_empty()
        {
            recalled = items.len();
            memory_context = Some(format_memory_context(&items));
        }

        let wd = self.config.working_dir.lock().ok().and_then(|g| g.clone());
        let ws_block = crate::agent::react::ReactAgent::build_workspace_context_block(wd.as_ref());

        let mut context = self.memory.context.lock().await;
        context.replace_projection(
            WORKSPACE_CONTEXT_PROJECTION,
            (!ws_block.trim().is_empty()).then(|| runtime_context_note("workspace", &ws_block)),
        );
        context.replace_tail_projection(
            TURN_MEMORY_CONTEXT_PROJECTION,
            memory_context.map(|body| runtime_context_note("memory", body.as_str())),
        );
        for message in history {
            context.push(message.clone());
        }
        context.push(Message::user(input.to_string()));
        /* 在执行钩子之前先释放 context 锁（避免与 fire_lifecycle_hook
         * 自身获取 context 锁时发生死锁）。 */
        drop(context);

        /* 触发 UserPromptSubmit 钩子：传入用户输入，使基于内容的匹配器
         * （例如 "\\.docx"）能够触发 ActivateSkill 动作，静态的 "*" 匹配器
         * 则会在每个回合都注入强制清单提示词。 */
        let hook_result = self
            .fire_lifecycle_hook(HookEvent::UserPromptSubmit, Some(input))
            .await;
        if hook_result.block {
            return Err(crate::error::ReactError::Other(format!(
                "Blocked by UserPromptSubmit hook: {}",
                hook_result.block_reason.unwrap_or_default()
            )));
        }
        // 缓存钩子激活结果，供 TriggerSupervisor（P4）消费
        if hook_result.activate_skill.is_some()
            && let Ok(mut cache) = self.hook_activation_cache.lock()
        {
            *cache = hook_result.activate_skill;
        }

        Ok(recalled)
    }

    /** 流式执行上下文初始化（多模态消息版本）。
     *
     * 与 `prepare_stream_context` 相同，但接受一个预先构建好的 `Message`
     * 而非字符串，支持多模态内容部分（图片、文件等）。
     */
    pub(crate) async fn prepare_stream_context_with_message(
        &self,
        mode: StreamMode,
        message: &Message,
        history: &[Message],
        scope_id: Option<&str>,
        runtime_state_id: Option<&str>,
    ) -> crate::error::Result<usize> {
        // 清空“编辑前必读”跟踪状态（参见 prepare_stream_context）。
        self.clear_read_files();
        match mode {
            StreamMode::Execute => {
                self.restore_thread_context_for(scope_id, runtime_state_id)
                    .await?;
            }
            StreamMode::Chat => {
                self.restore_chat_context_if_cold_for(scope_id, runtime_state_id)
                    .await?
            }
        }

        // 从消息中提取文本，用于长期记忆检索
        let text = message.content.as_text().unwrap_or_default();
        if !text.is_empty() {
            self.detect_and_write_memory_triggers(&text).await;
        }

        let mut recalled = 0usize;
        let mut memory_context = None;
        if !text.is_empty()
            && let Ok(items) = self.recall_long_term_memories(&text).await
            && !items.is_empty()
        {
            recalled = items.len();
            memory_context = Some(format_memory_context(&items));
        }

        let wd = self.config.working_dir.lock().ok().and_then(|g| g.clone());
        let ws_block = crate::agent::react::ReactAgent::build_workspace_context_block(wd.as_ref());

        let mut context = self.memory.context.lock().await;
        context.replace_projection(
            WORKSPACE_CONTEXT_PROJECTION,
            (!ws_block.trim().is_empty()).then(|| runtime_context_note("workspace", &ws_block)),
        );
        context.replace_tail_projection(
            TURN_MEMORY_CONTEXT_PROJECTION,
            memory_context.map(|body| runtime_context_note("memory", body.as_str())),
        );
        for history_message in history {
            context.push(history_message.clone());
        }
        context.push(message.clone());
        drop(context);

        // 触发 UserPromptSubmit 钩子（原因参见 prepare_stream_context）
        if !text.is_empty() {
            let hook_result = self
                .fire_lifecycle_hook(HookEvent::UserPromptSubmit, Some(&text))
                .await;
            if hook_result.block {
                return Err(crate::error::ReactError::Other(format!(
                    "Blocked by UserPromptSubmit hook: {}",
                    hook_result.block_reason.unwrap_or_default()
                )));
            }
            if hook_result.activate_skill.is_some()
                && let Ok(mut cache) = self.hook_activation_cache.lock()
            {
                *cache = hook_result.activate_skill;
            }
        }

        Ok(recalled)
    }
}

/// 构造一条运行时上下文提示消息，并推送到给定的上下文管理器中。
pub(crate) async fn push_runtime_context_note(
    context: &tokio::sync::Mutex<crate::compression::ContextManager>,
    source: &str,
    body: &str,
) {
    context
        .lock()
        .await
        .push(runtime_context_note(source, body));
}

/// 构造一条带来源标记的运行时上下文提示消息（`user` 角色）。
pub(crate) fn runtime_context_note(source: &str, body: &str) -> Message {
    Message::user(format!(
        "[runtime_context:{source}]\n{body}\n[Use this runtime context to continue the current task. It is dynamic turn state, not stable system policy.]"
    ))
}

/// 将召回到的记忆条目格式化为一段供模型阅读的上下文文本。
pub(crate) fn format_memory_context(items: &[crate::memory::store::StoreItem]) -> String {
    let mut lines = vec!["[memory_context] Relevant historical memories:".to_string()];
    for (i, item) in items.iter().enumerate() {
        let content_str = item
            .value
            .get("content")
            .and_then(|v| v.as_str())
            .map(String::from)
            .unwrap_or_else(|| item.value.to_string());
        lines.push(format!("{}. {}", i + 1, content_str));
    }
    lines.push(
        "[The above memories are for reference; answer the user's CURRENT question.]".to_string(),
    );
    lines.join("\n")
}

/*
(stage4 D1) 召回相关的辅助函数（composite_score / age_days_from_storeitem /
incr_recall_count）已迁移到 `evolution::recall::MemoryRecaller`。
*/

impl crate::agent::snapshot::AgentRunSnapshot {
    /** (stage4 E1) 压缩前刷写（Pre-compaction flush）—— 通过一次有超时限制的
     * LLM 调用，从即将被压缩的对话中识别出持久性的规则/技能/事实，并将其
     * 持久化到统一记忆存储中，使其在压缩后依然留存（割裂点 1/7）。
     *
     * 实现思路：一种务实的“结构化返回式刷写”—— 由 LLM 判断哪些内容值得
     * 长期保留并返回一个 JSON 数组；框架通过共享的 `MemoryLayerManager`
     * （类型化记忆、安全检查、审计钩子）逐条写入。这是用户选定的方案
     * （取代了 D14-G1 中的“真 subagent”方案 —— 对于本地单用户助手场景，
     * 派生一个 subagent fork 来做等价的持久信息抽取被判定为成本过高）。
     * 该流程是尽力而为的 —— 出错或超时都绝不会阻塞压缩流程。受
     * `ContextManager::should_compress()` 门控，因此只在压缩即将发生时
     * 才会触发，而不是每次 ReAct 迭代都触发。
     */
    pub(crate) async fn pre_compaction_flush(
        &self,
        context: &std::sync::Arc<tokio::sync::Mutex<crate::compression::ContextManager>>,
    ) {
        let (Some(llm_client), Some(layer_manager)) =
            (self.llm_client.as_ref(), self.memory_layer_manager.as_ref())
        else {
            return; // 没有 LLM 或没有层管理器 —— 跳过
        };

        // 对即将被压缩的消息做快照（最近约 40 条，按时间顺序）。
        let transcript = {
            let ctx = context.lock().await;
            /* (stage4 E1) 仅在压缩即将发生时才刷写 —— 与
             * `ContextManager::prepare` 的 `needs_compression` 判定逻辑保持
             * 一致。避免在无需压缩时，每次 ReAct 迭代都触发一次 LLM 调用。 */
            if !ctx.should_compress() {
                return;
            }
            let msgs = ctx.messages();
            if msgs.len() < 8 {
                return; // 太短，即使 token 数超标也不值得刷写
            }
            let mut buf = String::new();
            for m in msgs
                .iter()
                .rev()
                .take(40)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
            {
                buf.push_str(&format!(
                    "[{}] {}\n",
                    m.role.as_str(),
                    m.content.as_text_ref().unwrap_or("")
                ));
            }
            buf
        };
        if transcript.trim().is_empty() {
            return;
        }

        let request = crate::llm::ChatRequest {
            messages: vec![
                Message::system(PRE_COMPACTION_FLUSH_PROMPT.to_string()),
                Message::user(transcript),
            ],
            temperature: Some(0.2),
            max_tokens: Some(2048),
            user_id: self.config.cache_user_id.clone(),
            ..Default::default()
        };

        // 限时 15 秒；出错或超时都绝不会阻塞压缩流程。
        let response = match tokio::time::timeout(
            std::time::Duration::from_secs(15),
            llm_client.chat(request),
        )
        .await
        {
            Ok(Ok(r)) => r,
            Ok(Err(e)) => {
                tracing::debug!(error = %e, "pre_compaction_flush LLM call failed");
                return;
            }
            Err(_) => {
                tracing::debug!("pre_compaction_flush timed out (15s)");
                return;
            }
        };
        let content = response.content().unwrap_or_default();
        if content.trim().is_empty() || content.trim().eq_ignore_ascii_case("no_reply") {
            return;
        }

        // 解析出一个 {content, type, recall_weight} 的 JSON 数组，逐条写入。
        let items: Vec<serde_json::Value> = match (content.find('['), content.rfind(']')) {
            (Some(start), Some(end)) if end > start => content
                .get(start..=end)
                .and_then(|json| serde_json::from_str(json).ok())
                .unwrap_or_default(),
            _ => Vec::new(),
        };
        for item in items {
            let Some(fact) = item.get("content").and_then(|v| v.as_str()) else {
                continue;
            };
            let ty = item
                .get("type")
                .and_then(|v| v.as_str())
                .unwrap_or("project_fact");
            let rw = item
                .get("recall_weight")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.5)
                .clamp(0.0, 1.0);
            let memory_type = match ty.to_lowercase().as_str() {
                "user_preference" => MemoryType::UserPreference,
                "project_fact" => MemoryType::ProjectFact,
                "architecture_decision" => MemoryType::ArchitectureDecision,
                "debugging_lesson" => MemoryType::DebuggingLesson,
                "error_resolution" => MemoryType::ErrorResolution,
                "command_pattern" => MemoryType::CommandPattern,
                "tool_usage" => MemoryType::ToolUsage,
                _ => MemoryType::ProjectFact,
            };
            let meta = MemoryMeta::new(memory_type, MemorySource::L3Promotion, "compaction_flush")
                .with_recall_weight(rw as f32);
            let key = crate::memory_promoter::durable_memory_content_key(fact);
            match layer_manager.locate(&key).await {
                Ok(Some((_, existing))) if existing.content.trim() == fact.trim() => continue,
                Err(error) => {
                    tracing::warn!(%error, "pre_compaction_flush cannot read memory while reconciliation is pending");
                    continue;
                }
                _ => {}
            }
            if let Err(e) = layer_manager.write_memory(&key, fact, meta).await {
                tracing::debug!(error = %e, "pre_compaction_flush write_memory failed");
            }
        }
    }
}
