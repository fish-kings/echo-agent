use crate::agent::subagent::context::{ContextInheritance, SubagentContext};
use crate::agent::subagent::executor::SubagentExecutor;
use crate::agent::subagent::{ExecutionMode, SubagentOutcome};
use crate::error::ToolError;
use crate::tools::{Tool, ToolParameters, ToolResult};
use echo_core::agent::CancellationToken;
use echo_core::tools::{ExternalRunContext, NestedDelegationPolicy, ToolContext};
use futures::future::BoxFuture;
use serde_json::{Value, json};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tracing::{debug, info, warn};

/// 将子 Agent 的执行结果（outcome）序列化为 JSON 字符串，返回给父级。
fn serialize_parent_result(
    outcome: &SubagentOutcome,
) -> std::result::Result<String, serde_json::Error> {
    serde_json::to_string(outcome)
}

/** 用于在派发（dispatch）时延迟构建父级上下文的工厂。
 *
 * 持有父 Agent 共享子系统的 Arc 引用，以便在真正发生派发时，
 * 能够用最新的消息和工具定义构建出 `SubagentContext`。
 */
pub struct ParentContextFactory {
    /// 父级的工具管理器（用于获取当前的工具定义）。
    pub tool_manager: Arc<crate::tools::ToolManager>,
    /// 父级的上下文管理器（用于获取最近的消息）。
    pub context: Arc<tokio::sync::Mutex<crate::compression::ContextManager>>,
    /// 父级的记忆存储（用于与子 Agent 共享）。
    pub store: Option<Arc<dyn crate::memory::Store>>,
}

impl ParentContextFactory {
    /// 使用显式的继承策略构建一个 SubagentContext。
    pub async fn build_with_inheritance(
        &self,
        inheritance: &ContextInheritance,
    ) -> SubagentContext {
        let ctx = self.context.lock().await;
        let messages = ctx.messages().to_vec();
        let tool_defs: Vec<_> = self
            .tool_manager
            .get_tool_definitions()
            .into_iter()
            .filter(|d| d.function.name != "final_answer")
            .collect();
        SubagentContext::from_parent(&tool_defs, &messages, self.store.clone(), inheritance)
    }
}

/** 将任务派发给已注册子 Agent 的工具（对应 `agent_tool`）。
 *
 * 封装了子 Agent 执行器、父级身份信息、取消令牌的传播，
 * 以及可选的父级上下文继承逻辑。
 */
pub struct AgentDispatchTool {
    /// 子 Agent 执行器，负责实际执行派发请求。
    executor: Arc<SubagentExecutor>,
    /// 发起派发的父级 Agent 名称。
    parent_agent: String,
    /** 指向父级运行取消令牌的共享、可更新句柄（P1-11）。
     *
     * 该工具是在 Agent 构建期间创建的，此时任何运行都还未开始，
     * 因此父级的取消令牌尚未可知。我们持有一个 `Arc<Mutex<..>>`，
     * 父级运行在开始时会更新它；`execute` 读取最新值并据此派生出
     * `child_token`，这样通过该工具派发的子 Agent 会随父级运行一起被取消。
     * 此前这里固定使用构建期捕获的 `CancellationToken::new()`——一个永远
     * 不会触发的惰性令牌，导致由 LLM 发起的派发完全脱离了父级的取消机制。
     */
    cancel: Arc<tokio::sync::Mutex<Option<CancellationToken>>>,
    /** 用于在派发时构建父级上下文的可选工厂。
     * 设置后，显式的 Fork 模式可以继承经过过滤的会话历史。
     */
    parent_context_factory: Option<Arc<ParentContextFactory>>,
    /// 通过工具 schema 暴露给 LLM 的可用子 Agent 快照。
    catalog: Arc<std::sync::RwLock<Vec<crate::agent::subagent::SubagentDefinition>>>,
    /// 目录（catalog）的版本号，用于检测 schema 是否需要刷新。
    catalog_revision: Arc<AtomicU64>,
}

impl AgentDispatchTool {
    /// 创建一个新的 `AgentDispatchTool`。
    pub fn new(
        executor: Arc<SubagentExecutor>,
        parent_agent: impl Into<String>,
        cancel: CancellationToken,
    ) -> Self {
        Self {
            catalog: executor.registry().executable_catalog_handle(),
            catalog_revision: executor.registry().catalog_revision_handle(),
            executor,
            parent_agent: parent_agent.into(),
            cancel: Arc::new(tokio::sync::Mutex::new(Some(cancel))),
            parent_context_factory: None,
        }
    }

    /// 使用父级上下文工厂创建实例，以支持上下文继承。
    pub fn with_parent_context(mut self, factory: Arc<ParentContextFactory>) -> Self {
        self.parent_context_factory = Some(factory);
        self
    }

    /// 根据调用上下文推导子 Agent 的委派（delegation）策略。
    fn delegation_policy_from_context(
        ctx: Option<&ToolContext>,
    ) -> std::result::Result<NestedDelegationPolicy, String> {
        match ctx.and_then(|ctx| ctx.delegation_policy) {
            Some(policy) => policy.child_policy().ok_or_else(|| {
                format!(
                    "Delegation depth exceeded (max {})",
                    policy.max_delegate_depth
                )
            }),
            None => Ok(crate::agent::subagent::DispatchRequest::policy_from_depth(
                0,
            )),
        }
    }

    /** 构建用于应用 UI 身份定位的 [`ExternalRunContext`]。
     *
     * 存在正式 run id 时优先使用，并始终保留聊天回合（turn）id。
     * `execution_id` 中不含 `:`，因此 Tauri 桥接层会将完整字符串
     * 用作 `subagent_run_id`（避免并行的同角色派发相互冲突）。
     */
    fn runtime_context_from_tool_ctx(ctx: Option<&ToolContext>) -> Option<ExternalRunContext> {
        let c = ctx?;
        if c.conversation_id.is_none()
            && c.run_id.is_none()
            && c.turn_id.is_none()
            && c.message_id.is_none()
            && c.execution_id.is_none()
            && c.cancel.is_none()
            && c.trace_sink.is_none()
            && c.delegation_policy.is_none()
            && c.resource_guards.is_empty()
        {
            return None;
        }
        Some(ExternalRunContext {
            conversation_id: c.conversation_id.clone(),
            run_id: c.run_id.clone(),
            turn_id: c.turn_id.clone(),
            execution_id: Some(format!("agent_tool-{}", uuid::Uuid::new_v4())),
            isolation_id: None,
            message_id: c
                .message_id
                .clone()
                .or_else(|| c.turn_id.clone())
                .or_else(|| c.run_id.clone()),
            cancel: c.cancel.clone(),
            trace_sink: c.trace_sink.clone(),
            delegation_policy: c.delegation_policy,
            resource_guards: c.resource_guards.clone(),
            /* 子级的世系（lineage）信息将在下面根据调用方自身的
             * 上下文世系来标记——父级的值不能在此泄露。 */
            subagent_lineage: None,
            uplink: None,
        })
    }

    /// 根据调用方取消令牌与共享句柄，派生出子 Agent 使用的取消令牌。
    async fn child_cancel_token(
        cancel_handle: &Arc<tokio::sync::Mutex<Option<CancellationToken>>>,
        invocation_cancel: Option<&CancellationToken>,
    ) -> CancellationToken {
        if let Some(parent) = invocation_cancel {
            return parent.child_token();
        }

        cancel_handle
            .lock()
            .await
            .as_ref()
            .map(CancellationToken::child_token)
            .unwrap_or_else(CancellationToken::new)
    }

    /// 携带可选工具上下文执行一次派发，构建请求并调用执行器。
    fn dispatch_with_context(
        &self,
        parameters: ToolParameters,
        ctx: Option<&ToolContext>,
    ) -> BoxFuture<'_, crate::error::Result<ToolResult>> {
        let executor = self.executor.clone();
        let parent_agent = self.parent_agent.clone();
        let cancel_handle = self.cancel.clone();
        let factory = self.parent_context_factory.clone();
        let mut runtime_context = Self::runtime_context_from_tool_ctx(ctx);
        let active_message = ctx.and_then(|context| context.active_message.clone());
        /* 持有所有权的父级世系值，用于在异步块内部标记子级身份
         * （ctx 引用本身不能被捕获进异步块）。 */
        let parent_lineage = ctx.and_then(|context| context.subagent_lineage.clone());
        let parent_execution_id = ctx.and_then(|context| context.execution_id.clone());
        let parent_call_id = ctx.and_then(|context| context.call_id.clone());
        let effect_sink = ctx.and_then(|context| context.effect_sink.clone());
        let invocation_cancel = ctx.and_then(|context| context.cancel.clone());
        let delegation_policy = match Self::delegation_policy_from_context(ctx) {
            Ok(policy) => policy,
            Err(e) => return Box::pin(async move { Ok(ToolResult::error(e)) }),
        };

        Box::pin(async move {
            let agent_name = parameters
                .get("agent_name")
                .and_then(|v| v.as_str())
                .ok_or_else(|| ToolError::MissingParameter("agent_name".to_string()))?;

            /* 根据调用方上下文标记子级的世系：父级路径会不断串联
             * （形如 `root/<parent>/<child>`），父级自身的执行 id 使得默认的
             * 上行链路（uplink）在子级升级时能够定位到父级。父级的
             * uplink 被有意地不被继承——执行器会为该子级安装一个全新的
             * 默认 sink。 */
            if let Some(context) = runtime_context.as_mut() {
                let parent_path = parent_lineage
                    .as_ref()
                    .and_then(|lineage| lineage.agent_path.clone())
                    .unwrap_or_else(|| "root".to_string());
                let parent_event_id = parent_execution_id
                    .as_deref()
                    .zip(parent_call_id.as_deref())
                    .and_then(|(execution_id, call_id)| {
                        executor
                            .registry()
                            .event_bus()
                            .parent_event_id_for_tool(execution_id, call_id)
                    })
                    .map(|event_id| event_id.as_str().to_string());
                context.subagent_lineage = Some(echo_core::tools::SubagentLineage {
                    agent_name: Some(agent_name.to_string()),
                    execution_id: context.execution_id.clone(),
                    run_id: context.run_id.clone(),
                    parent_agent: Some(parent_agent.clone()),
                    parent_execution_id: parent_execution_id.clone(),
                    parent_event_id,
                    agent_path: Some(format!("{parent_path}/{agent_name}")),
                    task_id: None,
                    attempt: None,
                    plan_revision: None,
                });
            }

            let task = parameters
                .get("task")
                .and_then(|v| v.as_str())
                .ok_or_else(|| ToolError::MissingParameter("task".to_string()))?;

            let mode_override = match parameters.get("mode") {
                None | Some(Value::Null) => None,
                Some(Value::String(mode)) => Some(match mode.as_str() {
                    "sync" => ExecutionMode::Sync,
                    "fork" => ExecutionMode::Fork,
                    "teammate" => ExecutionMode::Teammate,
                    "team" => ExecutionMode::Team,
                    _ => {
                        return Ok(ToolResult::error(format!(
                            "Invalid subagent execution mode '{mode}'"
                        )));
                    }
                }),
                Some(_) => {
                    return Ok(ToolResult::error(
                        "Subagent execution mode must be a string",
                    ));
                }
            };

            let param_background = parameters
                .get("background")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);

            let constraints = parameters
                .get("constraints")
                .and_then(|v| v.as_array())
                .map(|items| {
                    items
                        .iter()
                        .filter_map(|item| item.as_str().map(|s| s.trim().to_string()))
                        .filter(|s| !s.is_empty())
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();

            /* 上下文继承与执行模式相互独立：
             * - 省略 / sync → 全新上下文（不含父级 system/历史/记忆）
             * - fork → 继承经过过滤的父级片段
             * （在下面构建 parent_context 时应用） */

            /* 执行模式：保留调用方的选择，但当目标角色声明了
             * worktree/workspace 隔离时强制使用 Fork（目前这些路径
             * 只在 dispatch_fork 上存在）。 */
            let registered = executor.registry().get(agent_name).await.ok_or_else(|| {
                ToolError::ExecutionFailed {
                    tool: "agent_tool".to_string(),
                    message: format!("Subagent '{agent_name}' is not registered"),
                }
            })?;
            let mut exec_mode = mode_override
                .clone()
                .unwrap_or_else(|| registered.definition.execution_mode.clone());
            let def = &registered.definition;
            let isolation_forces_fork = def.isolation.is_some();
            if isolation_forces_fork {
                exec_mode = ExecutionMode::Fork;
            }
            let role_is_background = def.is_background;
            let run_background = param_background || role_is_background;

            info!(
                target_agent = %agent_name,
                task = %task,
                mode = ?exec_mode,
                background = run_background,
                inherit_fork = exec_mode == ExecutionMode::Fork,
                delegate_depth = delegation_policy.delegate_depth,
                max_delegate_depth = delegation_policy.max_delegate_depth,
                "Dispatching task to subagent via SubagentExecutor"
            );

            /* 如果工厂可用，则构建父级上下文。Fork、teammate 和 team
             * 三种模式使用同一套结构化历史策略；由编译器决定最终的
             * 有界消息集合。全新的 sync 派发则有意只携带限定范围内的
             * 父级目标。 */
            let parent_context = if let Some(ref f) = factory {
                let ctx = if matches!(
                    exec_mode,
                    ExecutionMode::Fork | ExecutionMode::Teammate | ExecutionMode::Team
                ) {
                    let inheritance = ContextInheritance {
                        /* 这里保留完整的原始快照。由配置好的提示词编译器
                         * 执行统一的、面向 provider 安全的过滤，并应用
                         * 角色限制。 */
                        inherit_history: Some(0),
                        ..ContextInheritance::fork_default()
                    };
                    f.build_with_inheritance(&inheritance).await
                } else {
                    f.build_with_inheritance(&ContextInheritance::fresh_default())
                        .await
                };
                if ctx.has_content() { Some(ctx) } else { None }
            } else {
                None
            };

            /* ToolContext 是与本次调用绑定的作用域，因此对于池化 Agent
             * 而言它是权威来源。共享句柄仅作为兼容性回退，供那些不带
             * 运行时上下文执行该工具的调用方使用。 */
            let cancel =
                Self::child_cancel_token(&cancel_handle, invocation_cancel.as_deref()).await;

            let req = crate::agent::subagent::DispatchRequest {
                agent_name: agent_name.to_string(),
                task: task.to_string(),
                mode_override: if isolation_forces_fork {
                    Some(ExecutionMode::Fork)
                } else {
                    mode_override
                },
                cancel,
                parent_agent: parent_agent.clone(),
                parent_context,
                delegation_policy,
                runtime_context,
                message: active_message,
                prompt_payload: None,
                prompt_context: None,
                constraints,
                background: run_background,
            };

            if run_background {
                match executor.dispatch_background(req).await {
                    Ok(handle) => {
                        let execution_id = handle.execution_id.clone();
                        let started_agent_name = handle.agent_name.clone();
                        info!(
                            target_agent = %started_agent_name,
                            execution_id = %execution_id,
                            "Background subagent started"
                        );
                        if let Some(sink) = effect_sink {
                            let task = task.to_string();
                            tokio::spawn(async move {
                                let outcome = match handle.join().await {
                                    Ok(result) => result.outcome.status.as_str().to_string(),
                                    Err(error) => {
                                        crate::agent::subagent::subagent_status_from_error(&error)
                                            .as_str()
                                            .to_string()
                                    }
                                };
                                sink(echo_core::tools::ToolEffect::SubagentRun {
                                    agent_name: started_agent_name,
                                    task,
                                    outcome,
                                })
                                .await;
                            });
                        }
                        Ok(ToolResult::success(
                            json!({
                                "status": "started",
                                "execution_id": execution_id,
                                "agent_name": agent_name,
                            })
                            .to_string(),
                        ))
                    }
                    Err(e) => {
                        warn!(target_agent = %agent_name, error = %e, "Background subagent start failed");
                        Ok(ToolResult::error(format!(
                            "Subagent '{}' background start failed: {}",
                            agent_name, e
                        )))
                    }
                }
            } else {
                match executor.dispatch(req).await {
                    Ok(result) => {
                        info!(target_agent = %agent_name, "Subagent completed successfully");
                        debug!(
                            target_agent = %agent_name,
                            summary = %result.outcome.summary,
                            output_chars = result.output.chars().count(),
                            "Subagent result"
                        );
                        let outcome = result.outcome.status.as_str().to_string();
                        Ok(match serialize_parent_result(&result.outcome) {
                            Ok(output) => ToolResult::success(output).with_effect(
                                echo_core::tools::ToolEffect::SubagentRun {
                                    agent_name: agent_name.to_string(),
                                    task: task.to_string(),
                                    outcome,
                                },
                            ),
                            Err(error) => ToolResult::error(format!(
                                "Subagent '{}' result serialization failed: {}",
                                agent_name, error
                            ))
                            .with_effect(
                                echo_core::tools::ToolEffect::SubagentRun {
                                    agent_name: agent_name.to_string(),
                                    task: task.to_string(),
                                    outcome,
                                },
                            ),
                        })
                    }
                    Err(e) => {
                        warn!(target_agent = %agent_name, error = %e, "Subagent execution failed");
                        Ok(ToolResult::error(format!(
                            "Subagent '{}' execution failed: {}",
                            agent_name, e
                        ))
                        .with_effect(
                            echo_core::tools::ToolEffect::SubagentRun {
                                agent_name: agent_name.to_string(),
                                task: task.to_string(),
                                outcome: crate::agent::subagent::subagent_status_from_error(&e)
                                    .as_str()
                                    .to_string(),
                            },
                        ))
                    }
                }
            }
        })
    }
}

impl Tool for AgentDispatchTool {
    /// 返回该工具的名称。
    fn name(&self) -> &str {
        "agent_tool"
    }

    /// 返回该工具提供给 LLM 的描述文本。
    fn description(&self) -> &str {
        "Dispatch tasks to specialized Subagents in an isolated context. \
         Default is fresh context (no parent conversation). Use mode=fork only when \
         the Subagent needs shared background from this session. Set background=true \
         to start the Subagent without blocking (returns started + execution_id; \
         completion arrives via events / chat note). One call delegates one bounded \
         task. When the host provides a formal task planner, use that planner for \
         coordinated, dependent, or parallel multi-task work. Synchronous completion returns a \
         JSON result with status, summary, artifacts, evidence, remaining_work, \
         and evidence. Use only agent_name values listed in the schema.\n\
         The Subagent's result is not visible to the user — summarize it in your reply to the user.\n\
         Tell the Subagent clearly whether to write code or only do research, and the expected result format.\n\
         Do not duplicate work the Subagent is already doing (same searches, edits, or checks).\n\
         To run multiple independent Subagents in parallel, send a single message with multiple agent_tool calls."
    }

    /** `agent_tool` 会派发一个运行自己多步 ReAct 流程的子 Agent
     * （延迟远高于典型的文件/shell 工具）。将其从并行批次超时中豁免，
     * 避免它占满批次预算并过早取消同批的其他任务；它自身另有单独的
     * 单次派发超时（参见 `SubagentExecutor`，默认 600 秒）。
     */
    fn exempt_from_batch_timeout(&self) -> bool {
        true
    }

    /// 返回该工具的调用参数 JSON Schema。
    fn parameters(&self) -> Value {
        let catalog = self
            .catalog
            .read()
            .map(|entries| entries.clone())
            .unwrap_or_default();
        let agent_names: Vec<String> = catalog.iter().map(|entry| entry.name.clone()).collect();
        let catalog_text = if catalog.is_empty() {
            "No Subagents are currently registered.".to_string()
        } else {
            let lines: Vec<String> = catalog
                .iter()
                .map(|entry| format!("{}: {}", entry.name, entry.description))
                .collect();
            format!("Available Subagents: {}", lines.join("; "))
        };

        let agent_name_schema = if agent_names.is_empty() {
            json!({
                "type": "string",
                "description": catalog_text
            })
        } else {
            json!({
                "type": "string",
                "enum": agent_names,
                "description": catalog_text
            })
        };

        json!({
            "type": "object",
            "properties": {
                "agent_name": agent_name_schema,
                "task": {
                    "type": "string",
                    "description": "Specific task description to assign to the Subagent. Write its natural-language prose in the user's current language, then include relevant paths, scope, constraints, and required result format."
                },
                "mode": {
                    "type": "string",
                    "enum": ["sync", "fork", "teammate", "team"],
                    "description": "Optional. Omit or \"sync\" = fresh context. \"fork\", \"teammate\", and \"team\" may receive only filtered recent user/final-assistant messages; parent system prompts, tool traffic, and reasoning are never transferred. Worktree/workspace isolation is automatic for roles that declare it. \"teammate\" returns a join/cancel handle. \"team\" executes the role's TeamSpec through the canonical revisioned task DAG."
                },
                "constraints": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional. Explicit constraints or boundary rules the Subagent must respect — scope limits, files/dirs to avoid, verification expectations, output format. Rendered into the task context even for fresh-context dispatches."
                },
                "background": {
                    "type": "boolean",
                    "description": "Optional. When true, start the Subagent without blocking this turn; returns {status:\"started\", execution_id, agent_name}. Also true when the target role declares is_background."
                }
            },
            "required": ["agent_name", "task"]
        })
    }

    /// 返回当前工具 schema 的版本号。
    fn schema_revision(&self) -> u64 {
        self.catalog_revision.load(Ordering::Acquire)
    }

    /// 在没有工具上下文的情况下执行该工具。
    fn execute(
        &self,
        parameters: ToolParameters,
    ) -> BoxFuture<'_, crate::error::Result<ToolResult>> {
        self.dispatch_with_context(parameters, None)
    }

    /// 携带工具上下文执行该工具。
    fn execute_with_context<'a>(
        &'a self,
        parameters: ToolParameters,
        ctx: &'a ToolContext,
    ) -> BoxFuture<'a, crate::error::Result<ToolResult>> {
        self.dispatch_with_context(parameters, Some(ctx))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::subagent::{SubagentDefinition, SubagentExecutorConfig, SubagentRegistry};
    use crate::testing::{FailingMockAgent, MockAgent, MockAgentFailure};
    use echo_core::llm::types::{ContentPart, Message};
    use std::sync::atomic::AtomicUsize;

    struct BackgroundGuardDrop(Arc<AtomicUsize>);

    impl Drop for BackgroundGuardDrop {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn synchronous_parent_result_preserves_structured_contract() -> Result<(), String> {
        let outcome = SubagentOutcome {
            contract_version: 2,
            status: crate::agent::subagent::SubagentStatus::TimedOut,
            summary: "partial result".to_string(),
            artifacts: Vec::new(),
            evidence: vec![crate::agent::subagent::SubagentEvidence {
                kind: "file_read".to_string(),
                subject: "src/lib.rs".to_string(),
                outcome: Some("succeeded".to_string()),
                details: String::new(),
                source: crate::agent::subagent::SubagentEvidenceSource::Observed,
                attributes: serde_json::Value::Null,
            }],
            verification: Vec::new(),
            remaining_work: vec!["finish verification".to_string()],
            touched_files: crate::agent::subagent::SubagentTouchedFiles::default(),
        };
        let serialized = serialize_parent_result(&outcome).map_err(|error| error.to_string())?;
        let decoded: SubagentOutcome =
            serde_json::from_str(&serialized).map_err(|error| error.to_string())?;

        assert_eq!(decoded, outcome);
        Ok(())
    }

    #[test]
    fn default_dispatch_policy_starts_at_root() {
        let policy = AgentDispatchTool::delegation_policy_from_context(None).unwrap_or_default();
        assert!(policy.can_spawn_subagents);
        assert_eq!(policy.delegate_depth, 0);
        assert_eq!(policy.max_delegate_depth, 3);
    }

    #[test]
    fn context_dispatch_policy_advances_to_child() {
        let ctx = ToolContext {
            delegation_policy: Some(NestedDelegationPolicy {
                can_spawn_subagents: true,
                delegate_depth: 1,
                max_delegate_depth: 2,
            }),
            ..Default::default()
        };
        let policy =
            AgentDispatchTool::delegation_policy_from_context(Some(&ctx)).unwrap_or_default();
        assert_eq!(policy.delegate_depth, 2);
        assert!(!policy.can_delegate());
    }

    #[test]
    fn context_dispatch_policy_rejects_exhausted_depth() {
        let ctx = ToolContext {
            delegation_policy: Some(NestedDelegationPolicy {
                can_spawn_subagents: true,
                delegate_depth: 2,
                max_delegate_depth: 2,
            }),
            ..Default::default()
        };
        assert!(AgentDispatchTool::delegation_policy_from_context(Some(&ctx)).is_err());
    }

    #[test]
    fn runtime_context_none_without_tool_ctx() {
        assert!(AgentDispatchTool::runtime_context_from_tool_ctx(None).is_none());
    }

    #[test]
    fn runtime_context_none_without_run_id() {
        let ctx = ToolContext::default();
        assert!(AgentDispatchTool::runtime_context_from_tool_ctx(Some(&ctx)).is_none());
    }

    #[test]
    fn runtime_context_preserves_guards_without_run_or_turn_id() -> Result<(), String> {
        let ctx = ToolContext {
            resource_guards: vec![echo_core::tools::InvocationResourceGuard::new(
                "anonymous-lease".to_string(),
            )],
            ..ToolContext::default()
        };
        let runtime = AgentDispatchTool::runtime_context_from_tool_ctx(Some(&ctx))
            .ok_or_else(|| "guard-only runtime context was dropped".to_string())?;

        assert!(runtime.run_id.is_none());
        assert!(runtime.turn_id.is_none());
        assert_eq!(runtime.resource_guards.len(), 1);
        Ok(())
    }

    #[test]
    fn runtime_context_preserves_conversation_only() -> Result<(), String> {
        let context = ToolContext {
            conversation_id: Some("conversation-only".to_string()),
            ..ToolContext::default()
        };
        let runtime = AgentDispatchTool::runtime_context_from_tool_ctx(Some(&context))
            .ok_or_else(|| "conversation-only ToolContext was dropped".to_string())?;
        assert_eq!(
            runtime.conversation_id.as_deref(),
            Some("conversation-only")
        );
        Ok(())
    }

    #[test]
    fn runtime_context_preserves_cancel_and_trace_only() -> Result<(), String> {
        let cancel = Arc::new(CancellationToken::new());
        let trace: echo_core::tools::TraceSinkFn = Arc::new(|_| {});
        let context = ToolContext {
            cancel: Some(Arc::clone(&cancel)),
            trace_sink: Some(Arc::clone(&trace)),
            ..ToolContext::default()
        };
        let runtime = AgentDispatchTool::runtime_context_from_tool_ctx(Some(&context))
            .ok_or_else(|| "cancel/trace-only ToolContext was dropped".to_string())?;
        assert!(
            runtime
                .cancel
                .as_ref()
                .is_some_and(|value| Arc::ptr_eq(value, &cancel))
        );
        assert!(
            runtime
                .trace_sink
                .as_ref()
                .is_some_and(|value| Arc::ptr_eq(value, &trace))
        );
        Ok(())
    }

    #[test]
    fn runtime_context_from_run_id_pins_message_and_execution_id() -> Result<(), String> {
        let ctx = ToolContext {
            run_id: Some("msg-key-1".to_string()),
            resource_guards: vec![echo_core::tools::InvocationResourceGuard::new(
                "lease".to_string(),
            )],
            ..Default::default()
        };
        let rt = AgentDispatchTool::runtime_context_from_tool_ctx(Some(&ctx))
            .ok_or_else(|| "runtime_context missing when run_id is set".to_string())?;
        assert_eq!(rt.run_id.as_deref(), Some("msg-key-1"));
        assert_eq!(rt.message_id.as_deref(), Some("msg-key-1"));
        assert_eq!(rt.resource_guards.len(), 1);
        let exec_id = rt
            .execution_id
            .ok_or_else(|| "execution_id missing from runtime context".to_string())?;
        assert!(
            exec_id.starts_with("agent_tool-"),
            "execution_id should be agent_tool-{{uuid}}, got {exec_id}"
        );
        assert!(
            !exec_id.contains(':'),
            "execution_id must not contain ':' so bridge uses full id as subagent_run_id"
        );
        Ok(())
    }

    #[tokio::test]
    async fn invocation_cancel_token_overrides_stale_shared_handle() -> Result<(), String> {
        let stale_parent = CancellationToken::new();
        let invocation_parent = CancellationToken::new();
        let shared = Arc::new(tokio::sync::Mutex::new(Some(stale_parent.clone())));
        let ctx = ToolContext {
            cancel: Some(Arc::new(invocation_parent.clone())),
            ..Default::default()
        };

        let child = AgentDispatchTool::child_cancel_token(&shared, ctx.cancel.as_deref()).await;
        stale_parent.cancel();
        if child.is_cancelled() {
            return Err("stale shared token cancelled the invocation child".to_string());
        }

        invocation_parent.cancel();
        if !child.is_cancelled() {
            return Err("invocation cancellation did not reach the subagent child".to_string());
        }
        Ok(())
    }

    #[tokio::test]
    async fn anonymous_background_dispatch_retains_guard_until_subagent_settles()
    -> Result<(), String> {
        let registry = Arc::new(SubagentRegistry::new());
        let executor = Arc::new(SubagentExecutor::new(
            Arc::clone(&registry),
            SubagentExecutorConfig::default(),
        ));
        let agent = MockAgent::new("background").with_response("## Summary\ncomplete");
        registry
            .register(
                SubagentDefinition::new("background", "Background role"),
                Box::new(agent.clone()),
            )
            .await;
        let tool = AgentDispatchTool::new(executor, "parent", CancellationToken::new());
        let drops = Arc::new(AtomicUsize::new(0));
        let context = ToolContext {
            resource_guards: vec![echo_core::tools::InvocationResourceGuard::new(
                BackgroundGuardDrop(Arc::clone(&drops)),
            )],
            ..ToolContext::default()
        };
        let parameters: ToolParameters = [
            (
                "agent_name".to_string(),
                Value::String("background".to_string()),
            ),
            ("task".to_string(), Value::String("continue".to_string())),
            ("background".to_string(), Value::Bool(true)),
        ]
        .into_iter()
        .collect();

        let started = tool
            .execute_with_context(parameters, &context)
            .await
            .map_err(|error| error.to_string())?;
        if !started.success {
            return Err(format!("background dispatch failed to start: {started:?}"));
        }
        drop(context);
        assert_eq!(drops.load(Ordering::SeqCst), 0);

        tokio::time::timeout(std::time::Duration::from_secs(2), async {
            while agent.invocation_contexts().is_empty() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .map_err(|_| "background invocation did not reach the subagent".to_string())?;
        let invocations = agent.take_invocation_contexts();
        let guard_count = invocations
            .first()
            .and_then(|invocation| invocation.runtime.as_ref())
            .map(|runtime| runtime.resource_guards.len());
        if guard_count != Some(1) {
            return Err(format!(
                "background subagent lost anonymous guards: {guard_count:?}"
            ));
        }
        drop(invocations);
        tokio::time::timeout(std::time::Duration::from_secs(2), async {
            while drops.load(Ordering::SeqCst) == 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .map_err(|_| "background guard outlived subagent settlement".to_string())?;
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        Ok(())
    }

    #[tokio::test]
    async fn background_effect_settles_once_after_start_for_success_failure_and_cancel()
    -> Result<(), String> {
        for expected in ["completed", "failed", "cancelled"] {
            let registry = Arc::new(SubagentRegistry::new());
            let executor = Arc::new(SubagentExecutor::new(
                Arc::clone(&registry),
                SubagentExecutorConfig::default(),
            ));
            if expected == "failed" {
                registry
                    .register(
                        SubagentDefinition::new("child", "Failure"),
                        Box::new(
                            FailingMockAgent::new("child", "failed execution")
                                .with_failure(MockAgentFailure::Subagent),
                        ),
                    )
                    .await;
            } else {
                registry
                    .register(
                        SubagentDefinition::new("child", "Success or cancellation"),
                        Box::new(
                            MockAgent::new("child")
                                .with_delay_ms(if expected == "cancelled" { 1_000 } else { 30 })
                                .with_response("## Summary\ncomplete"),
                        ),
                    )
                    .await;
            }
            let tool = AgentDispatchTool::new(executor, "parent", CancellationToken::new());
            let cancellation = Arc::new(CancellationToken::new());
            let (effect_tx, mut effect_rx) = tokio::sync::mpsc::unbounded_channel();
            let context = ToolContext {
                call_id: Some(format!("call-{expected}")),
                run_id: Some(format!("run-{expected}")),
                cancel: Some(Arc::clone(&cancellation)),
                effect_sink: Some(Arc::new(move |effect| {
                    let effect_tx = effect_tx.clone();
                    Box::pin(async move {
                        let _ = effect_tx.send(effect);
                    })
                })),
                ..ToolContext::default()
            };
            let parameters: ToolParameters = [
                ("agent_name".to_string(), Value::String("child".to_string())),
                ("task".to_string(), Value::String("task".to_string())),
                ("background".to_string(), Value::Bool(true)),
            ]
            .into_iter()
            .collect();

            let started = tool
                .execute_with_context(parameters, &context)
                .await
                .map_err(|error| error.to_string())?;
            if !started.success || !started.effects.is_empty() {
                return Err(format!("background launch is not an effect: {started:?}"));
            }
            if expected == "cancelled" {
                cancellation.cancel();
            }
            let effect = tokio::time::timeout(std::time::Duration::from_secs(3), effect_rx.recv())
                .await
                .map_err(|_| format!("{expected} effect did not settle"))?
                .ok_or_else(|| format!("{expected} effect sink disconnected"))?;
            if effect
                != (echo_core::tools::ToolEffect::SubagentRun {
                    agent_name: "child".to_string(),
                    task: "task".to_string(),
                    outcome: expected.to_string(),
                })
            {
                return Err(format!("wrong {expected} effect: {effect:?}"));
            }
            if effect_rx.try_recv().is_ok() {
                return Err(format!("duplicate {expected} terminal effect"));
            }
        }
        Ok(())
    }

    #[tokio::test]
    async fn cached_schema_tracks_shared_registry_revision() -> Result<(), String> {
        let registry = Arc::new(SubagentRegistry::new());
        let executor = Arc::new(SubagentExecutor::new(
            Arc::clone(&registry),
            SubagentExecutorConfig::default(),
        ));
        let manager = crate::tools::ToolManager::new();
        manager.register(Box::new(AgentDispatchTool::new(
            executor,
            "parent",
            CancellationToken::new(),
        )));

        let initial = manager.get_tool_definitions();
        let initial_schema = initial
            .iter()
            .find(|definition| definition.function.name == "agent_tool")
            .map(|definition| &definition.function.parameters);
        assert!(initial_schema.is_some());

        registry
            .register_definition(SubagentDefinition::new("researcher", "Pending role"))
            .await;
        let pending_schema = manager.get_tool_definitions();
        let pending = pending_schema
            .iter()
            .find(|definition| definition.function.name == "agent_tool")
            .map(|definition| &definition.function.parameters);
        assert!(pending.is_some_and(|parameters| {
            parameters.pointer("/properties/agent_name/enum").is_none()
                && parameters
                    .pointer("/properties/agent_name/description")
                    .and_then(Value::as_str)
                    .is_some_and(|description| !description.contains("researcher"))
        }));

        registry
            .register(
                SubagentDefinition::new("researcher", "Research role"),
                Box::new(MockAgent::new("researcher")),
            )
            .await;

        let refreshed = manager.get_tool_definitions();
        let names = refreshed
            .iter()
            .find(|definition| definition.function.name == "agent_tool")
            .and_then(|definition| {
                definition
                    .function
                    .parameters
                    .pointer("/properties/agent_name/enum")
            })
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        assert_eq!(names, vec![Value::String("researcher".to_string())]);

        registry.remove("researcher").await;
        let removed_schema = manager.get_tool_definitions();
        let removed = removed_schema
            .iter()
            .find(|definition| definition.function.name == "agent_tool")
            .map(|definition| &definition.function.parameters);
        assert!(removed.is_some_and(|parameters| {
            parameters.pointer("/properties/agent_name/enum").is_none()
                && parameters
                    .pointer("/properties/agent_name/description")
                    .and_then(Value::as_str)
                    .is_some_and(|description| !description.contains("researcher"))
        }));
        Ok(())
    }

    #[tokio::test]
    async fn active_message_attachments_reach_dispatched_subagent() -> Result<(), String> {
        let registry = Arc::new(SubagentRegistry::new());
        let executor = Arc::new(SubagentExecutor::new(
            Arc::clone(&registry),
            SubagentExecutorConfig::default(),
        ));
        let agent = MockAgent::new("reader").with_response("## Summary\nread");
        registry
            .register(
                SubagentDefinition::new("reader", "Reads attachments"),
                Box::new(agent.clone()),
            )
            .await;
        let tool = AgentDispatchTool::new(executor, "parent", CancellationToken::new());
        let message = Message::user_multimodal(vec![
            ContentPart::Text {
                text: "inspect this".to_string(),
            },
            ContentPart::File {
                name: "notes.txt".to_string(),
                content: "aGVsbG8=".to_string(),
            },
        ]);
        let ctx = ToolContext {
            active_message: Some(message),
            ..Default::default()
        };
        let parameters: ToolParameters = [
            (
                "agent_name".to_string(),
                Value::String("reader".to_string()),
            ),
            ("task".to_string(), Value::String("inspect".to_string())),
        ]
        .into_iter()
        .collect();
        let result = tool
            .execute_with_context(parameters, &ctx)
            .await
            .map_err(|error| error.to_string())?;
        if !result.success {
            return Err(format!("dispatch failed: {}", result.output));
        }
        let received = agent
            .last_message()
            .ok_or_else(|| "subagent did not receive active message".to_string())?;
        let has_file = received.content.parts().is_some_and(|parts| {
            parts
                .iter()
                .any(|part| matches!(part, ContentPart::File { name, .. } if name == "notes.txt"))
        });
        if !has_file {
            return Err("subagent message lost file attachment".to_string());
        }
        Ok(())
    }
}
