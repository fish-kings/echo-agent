/*!
 * ReactAgent 能力配置 API
 *
 * - 工具注册（`add_tool` / `add_tools` / `add_need_appeal_tool`）
 * - 技能安装（`add_skill` / `add_skills` / `discover_skills` / `load_skills_from_dir`）
 * - MCP 连接（`connect_mcp` / `load_mcp_from_file`）
 * - 子代理、压缩器、回调等
 */

use super::ReactAgent;
#[cfg(feature = "subagent")]
use crate::agent::Agent;
use crate::agent::AgentHandle;
#[cfg(feature = "subagent")]
use crate::agent::subagent::{AgentFactory, SubagentDefinition};
use crate::compression::{
    CompressionCheckpoint, ContextCompressor, ContextManager, ForceCompressStats,
};
use crate::error::Result;
use crate::llm::{LlmClient, LlmConfig, ThinkingConfig};
#[cfg(feature = "mcp")]
use crate::mcp::McpServerEntry;
#[cfg(feature = "mcp")]
use crate::mcp::{McpClient, McpConfigFile, McpServerConfig, McpServerId, McpTargetChange};
use crate::skills::external::SkillDescriptor;
use crate::skills::external::activate_tool::ActivateSkillTool;
use crate::skills::external::loader::{DiscoveryScope, SkillLoader};
use crate::skills::external::resource_tool::ReadSkillResourceTool;
use crate::skills::external::run_script_tool::RunSkillScriptTool;
use crate::skills::{Skill, SkillInfo};
use crate::tools::Tool;
use echo_core::agent::Critic;
use echo_core::budget::TokenBudget;
use std::sync::Arc;
use tokio::sync::{OwnedMutexGuard, RwLock};
use tracing::{info, warn};

const SKILL_CATALOG_PROJECTION: &str = "echo-agent:skill-catalog";

/// 一次待注册的 Skill，可能来自文件系统发现，也可能是已经准备好的文档。
enum SkillRegistration {
    /// 通过文件系统发现得到的 Skill 描述符。
    Filesystem(SkillDescriptor),
    /// 已预先构建完成的 Skill 文档（例如插件场景）。
    Prepared(crate::skills::external::SkillDocument),
}

impl SkillRegistration {
    /// 获取该注册项对应的描述符。
    fn descriptor(&self) -> &SkillDescriptor {
        match self {
            Self::Filesystem(descriptor) => descriptor,
            Self::Prepared(document) => document.descriptor(),
        }
    }

    /// 设置该 Skill 的注册来源标记。
    fn set_registration_source(&mut self, source: &str) {
        match self {
            Self::Filesystem(descriptor) => descriptor.source = Some(source.to_string()),
            Self::Prepared(document) => document.set_registration_source(source),
        }
    }
}

/// 将某个已激活 Skill 的提示块投影到上下文中。
pub(crate) async fn project_skill_activation(
    context: &Arc<tokio::sync::Mutex<ContextManager>>,
    skill_name: &str,
    prompt_block: &str,
) {
    context.lock().await.replace_projection(
        format!("echo-agent:skill:{skill_name}"),
        Some(crate::llm::types::Message::system(prompt_block.to_string())),
    );
}

/**
 * 一次已校验的 token 上限更新，独占持有当前存活的上下文。
 *
 * 持有该值可阻止某次 agent 轮次在“准备”与“提交”之间进入上下文。
 * 丢弃（drop）它会取消这次更新，而不会改变 agent 或其上下文。
 */
pub struct PreparedTokenLimit {
    /// 被独占写锁定的目标 agent。
    agent: tokio::sync::OwnedRwLockWriteGuard<ReactAgent>,
    /// 待提交的 token 上限相关取值。
    values: PreparedTokenLimitValues,
}

/// 待提交的 token 上限取值，含预算与被独占锁定的上下文。
struct PreparedTokenLimitValues {
    /// 新的 token 上限。
    token_limit: usize,
    /// 根据配置构建出的 token 预算（若启用）。
    budget: Option<TokenBudget>,
    /// 独占锁定的上下文管理器句柄。
    context: OwnedMutexGuard<ContextManager>,
}

/// 针对一次已准备好的模型生成，critic（评审器）应如何发布的显式策略。
pub enum PreparedCriticUpdate {
    /// 保持当前已安装的 critic 及其所有权不变。
    Preserve,
    /**
     * 仅当 critic 仍归 `owner` 所有时才替换它。
     *
     * 之后的 [`ReactAgent::set_critic`] 调用会清除具名所有权，
     * 因此调用方不会意外覆盖用户自行安装的自定义 critic。
     */
    ReplaceOwned {
        /// 期望持有当前 critic 所有权的调用方标识。
        owner: String,
        /// 待安装的新 critic 实例。
        critic: Arc<dyn Critic>,
    },
}

/**
 * 面向 [`ReactAgent`] 的一次完整、已预先校验的运行时模型生成。
 *
 * LLM 客户端由调用方在准备阶段之前构建完成，因此提交（commit）过程
 * 不执行任何 I/O、配置解析或 provider 查找。
 */
pub struct PreparedAgentModelGeneration {
    /// 被独占写锁定的目标 agent。
    agent: tokio::sync::OwnedRwLockWriteGuard<ReactAgent>,
    /// 待应用的 LLM 配置。
    llm_config: LlmConfig,
    /// 待安装的 LLM 客户端实例。
    llm_client: Arc<dyn LlmClient>,
    /// 待应用的温度参数。
    temperature: Option<f32>,
    /// 待应用的最大 token 数。
    max_tokens: Option<u32>,
    /// 待应用的思考（thinking）配置。
    thinking: Option<ThinkingConfig>,
    /// critic 的发布策略。
    critic: PreparedCriticUpdate,
    /// 待提交的 token 上限相关取值。
    token_limit: PreparedTokenLimitValues,
}

/// 一次与目标 agent 绑定的回执，用于清除该 agent 当前激活的模型生成。
pub struct PreparedAgentModelDeactivation {
    /// 被独占写锁定的目标 agent。
    agent: tokio::sync::OwnedRwLockWriteGuard<ReactAgent>,
    /// 若清除成功，需要一并释放所有权的 critic 拥有者标识。
    critic_owner: Option<String>,
}

impl PreparedTokenLimit {
    /// 将 token 上限发布到准备阶段锁定的那个 agent 上。
    pub fn commit(self) {
        let Self { mut agent, values } = self;
        agent.commit_prepared_token_limit(values);
    }
}

impl PreparedAgentModelGeneration {
    /// 将该生成发布到准备阶段锁定的那个 agent 上。
    /// 由于不存在 target 参数，某个 agent 上准备的回执不可能被提交到另一个 agent。
    pub fn commit(self) {
        let Self {
            mut agent,
            llm_config,
            llm_client,
            temperature,
            max_tokens,
            thinking,
            critic,
            token_limit,
        } = self;
        agent.commit_prepared_token_limit(token_limit);
        agent.config.model_name = llm_config.model.clone();
        agent.config.temperature = temperature;
        agent.config.max_tokens = max_tokens;
        agent.llm_config = Some(llm_config);
        agent.llm_client = Some(llm_client);
        agent.thinking = thinking;
        match critic {
            PreparedCriticUpdate::ReplaceOwned { owner, critic }
                if agent.critic_owner.as_deref() == Some(owner.as_str()) =>
            {
                agent.critic = Some(critic);
            }
            PreparedCriticUpdate::Preserve | PreparedCriticUpdate::ReplaceOwned { .. } => {}
        }
    }
}

impl PreparedAgentModelDeactivation {
    /// 从准备阶段锁定的那个 agent 上清除模型生成。
    pub fn commit(self) {
        let Self {
            mut agent,
            critic_owner,
        } = self;
        agent.config.model_name.clear();
        agent.config.temperature = None;
        agent.config.max_tokens = None;
        agent.llm_config = None;
        agent.llm_client = None;
        agent.thinking = None;
        if let Some(owner) = critic_owner
            && agent.critic_owner.as_deref() == Some(owner.as_str())
        {
            agent.critic = None;
            agent.critic_owner = None;
        }
    }
}

impl AgentHandle {
    /// 准备一次与目标绑定的 token 上限发布。
    pub async fn prepare_token_limit(
        &self,
        token_limit: usize,
    ) -> crate::error::Result<PreparedTokenLimit> {
        let agent = Arc::clone(self.inner()).write_owned().await;
        let values = agent.prepare_token_limit_values(token_limit).await?;
        Ok(PreparedTokenLimit { agent, values })
    }

    /// 准备一次与目标绑定的完整模型生成。
    #[allow(clippy::too_many_arguments)]
    pub async fn prepare_model_generation(
        &self,
        llm_config: LlmConfig,
        llm_client: Arc<dyn LlmClient>,
        temperature: Option<f32>,
        max_tokens: Option<u32>,
        thinking: Option<ThinkingConfig>,
        token_limit: usize,
        critic: PreparedCriticUpdate,
    ) -> crate::error::Result<PreparedAgentModelGeneration> {
        if llm_client.model_name() != llm_config.model {
            return Err(crate::error::ReactError::Other(format!(
                "prepared LLM client model '{}' does not match config model '{}'",
                llm_client.model_name(),
                llm_config.model
            )));
        }
        let agent = Arc::clone(self.inner()).write_owned().await;
        let token_limit = agent.prepare_token_limit_values(token_limit).await?;
        Ok(PreparedAgentModelGeneration {
            agent,
            llm_config,
            llm_client,
            temperature,
            max_tokens,
            thinking,
            critic,
            token_limit,
        })
    }

    /// 锁定该 agent 以执行一次原子性的“取消当前激活模型”操作。
    pub async fn prepare_model_deactivation(
        &self,
        critic_owner: Option<String>,
    ) -> PreparedAgentModelDeactivation {
        let agent = Arc::clone(self.inner()).write_owned().await;
        PreparedAgentModelDeactivation {
            agent,
            critic_owner,
        }
    }
}

impl ReactAgent {
    // ── 工具注册 ────────────────────────────────────────────────────

    /// 注册单个工具，并自动启用工具能力。
    pub fn add_tool(&mut self, tool: Box<dyn Tool>) {
        self.config.enable_tool = true;
        self.tools.tool_manager.register(tool);
    }

    /// 注册多个工具，并自动启用工具能力。
    pub fn add_tools(&mut self, tools: Vec<Box<dyn Tool>>) {
        if tools.is_empty() {
            return;
        }
        self.config.enable_tool = true;
        let allowed = &self.config.allowed_tools;
        if allowed.is_empty() {
            self.tools.tool_manager.register_tools(tools);
        } else {
            for tool in tools {
                if allowed.contains(&tool.name().to_string()) {
                    self.tools.tool_manager.register(tool);
                }
            }
        }
    }

    /**
     * 按名称移除一个工具。若该工具存在，则返回被移除的工具。
     *
     * 用于支持 ReAct 循环中的动态工具管理——例如，在任务执行过程中
     * 从搜索类工具切换到执行类工具。
     */
    pub fn remove_tool(&mut self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.tool_manager.unregister(name)
    }

    /**
     * 用一个新工具替换同名的现有工具。
     *
     * 如果存在同名工具，会先移除旧工具再注册新工具。
     * 若确实发生了替换，则返回旧工具。
     */
    pub fn replace_tool(&mut self, tool: Box<dyn Tool>) -> Option<Arc<dyn Tool>> {
        self.tools.tool_manager.replace(tool)
    }

    /**
     * 注册一个执行前需要人工审批的工具。
     *
     * 该 API 特意设计为同步接口，以便在 agent 初始化阶段调用。
     * 当 `PermissionService` 存在时，审批规则会先缓冲到
     * `pending_permission_rules` 中，随后转移到下一次调用快照里，
     * 这样可以避免在正在运行的 Tokio runtime 内调用 `block_on()`。
     *
     * 该方法会为此工具注册一条要求审批的权限规则。
     */
    pub fn add_need_appeal_tool(&mut self, tool: Box<dyn Tool>) {
        #[cfg(feature = "human-loop")]
        if self.config.enable_human_in_loop {
            let tool_name = tool.name().to_string();
            self.add_tool(tool);

            use echo_core::tools::permission::{PermissionRule, RuleMatcher, RuleSource};
            let rule = PermissionRule {
                matcher: RuleMatcher::Pattern {
                    pattern: tool_name.clone(),
                },
                behavior: echo_core::tools::permission::RuleBehavior::Ask {
                    suggestions: vec!["Allow".to_string(), "Deny".to_string()],
                },
                source: RuleSource::Session,
                description: Some(format!("Tool {} requires human approval", tool_name)),
            };

            // add_need_appeal_tool 是同步 API，不能在正在运行的 Tokio runtime 内
            // 调用 block_on()。因此先在此处注册规则，之后再在异步执行阶段
            // 安全地将其刷新到 PermissionService。
            if let Ok(mut pending) = self.approval.pending_permission_rules.lock() {
                pending.push(rule);
            } else {
                warn!(
                    agent = %self.config.agent_name,
                    tool = %tool_name,
                    "pending_permission_rules lock poisoned"
                );
            }

            return;
        }
        #[cfg(not(feature = "human-loop"))]
        let _ = &self.config.enable_human_in_loop;
        warn!(
            agent = %self.config.agent_name,
            tool = %tool.name(),
            "human_in_loop disabled, tool registered without approval requirement"
        );
        self.add_tool(tool);
    }

    // ── 上下文压缩 ──────────────────────────────────────────────────

    /**
     * 设置上下文压缩器
     *
     * # 参数
     * * `compressor` - 实现了 `ContextCompressor` trait 的压缩器实例
     *
     * # 说明
     * 上下文压缩器会在 token 数超过限制时自动压缩对话历史，
     * 移除不那么重要的消息以减少 token 消耗。
     */
    pub async fn set_compressor(&self, compressor: impl ContextCompressor + 'static) {
        self.memory.context.lock().await.set_compressor(compressor);
    }

    /**
     * 获取上下文统计信息
     *
     * # 返回值
     * 返回元组 `(消息数量, 预估 token 数)`
     */
    pub async fn context_stats(&self) -> (usize, usize) {
        let ctx = self.memory.context.lock().await;
        (ctx.messages().len(), ctx.token_estimate())
    }

    /**
     * 使用指定压缩器强制压缩上下文
     *
     * # 参数
     * * `compressor` - 用于压缩的压缩器实例
     *
     * # 返回值
     * 返回压缩统计信息，包括压缩前后的 token 数
     *
     * # 说明
     * 该方法绕过自动压缩阈值，立即使用指定的压缩器压缩上下文。
     * 通常用于手动压缩控制或测试压缩效果。
     */
    pub async fn force_compress_with(
        &self,
        compressor: &dyn ContextCompressor,
    ) -> Result<(ForceCompressStats, Option<CompressionCheckpoint>)> {
        self.memory
            .context
            .lock()
            .await
            .force_compress_with(compressor)
            .await
    }

    /**
     * 强制压缩，并在操作前后触发 PreCompact/PostCompact hook。
     *
     * `matcher` 参数决定 hook 匹配的上下文：
     * - `"manual"` 表示用户主动触发的压缩（例如 `/compact` 命令）
     * - `"auto"` 表示由阈值自动触发的压缩
     */
    pub async fn force_compress_with_hooks(
        &self,
        compressor: &dyn ContextCompressor,
        matcher: &str,
    ) -> Result<(ForceCompressStats, Option<CompressionCheckpoint>)> {
        self.fire_lifecycle_hook(crate::skills::hooks::HookEvent::PreCompact, Some(matcher))
            .await;
        let (stats, checkpoint) = self.force_compress_with(compressor).await?;
        self.fire_post_compact_hook(matcher, &stats).await;
        Ok((stats, checkpoint))
    }

    /// 携带聚焦指令强制压缩上下文，并触发 PreCompact/PostCompact hook。
    pub async fn force_compress_with_focus_and_hooks(
        &self,
        focus_instructions: &str,
        window: usize,
        matcher: &str,
    ) -> Result<(ForceCompressStats, Option<CompressionCheckpoint>)> {
        self.fire_lifecycle_hook(crate::skills::hooks::HookEvent::PreCompact, Some(matcher))
            .await;
        let (stats, checkpoint) = self
            .memory
            .context
            .lock()
            .await
            .force_compress_with_focus(focus_instructions, window)
            .await?;
        self.fire_post_compact_hook(matcher, &stats).await;
        Ok((stats, checkpoint))
    }

    /// 触发 PostCompact hook，并将 hook 注入的上下文/消息推送到运行时上下文中。
    async fn fire_post_compact_hook(&self, matcher: &str, stats: &ForceCompressStats) {
        let hook_stats = crate::skills::hooks::CompressHookStats {
            before_count: stats.before_count,
            after_count: stats.after_count,
            before_tokens: stats.before_tokens,
            after_tokens: stats.after_tokens,
        };
        let hook_ctx = crate::skills::hooks::HookContext::for_post_compact(
            &hook_stats,
            matcher,
            self.config.session_id.as_deref().unwrap_or(""),
            &self.config.agent_name,
        );
        let registry = self.tools.hook_registry.read().await.clone();
        let post_result = registry.run_lifecycle_hooks(&hook_ctx).await;
        if let Some(ctx) = &post_result.injected_context {
            super::run::context::push_runtime_context_note(
                &self.memory.context,
                "Hook:PostCompact",
                ctx,
            )
            .await;
        }
        for msg in &post_result.messages {
            super::run::context::push_runtime_context_note(
                &self.memory.context,
                "Hook:PostCompact",
                msg,
            )
            .await;
        }
    }

    /**
     * 使用已安装的压缩器强制压缩上下文（若未安装压缩器，
     * 则回退使用窗口大小为 40 的 SlidingWindowCompressor）。
     *
     * 设计用于手动压缩触发场景（桌面端 UI 按钮、`/compact` CLI 命令）。
     * 会以 matcher `"manual"` 触发 PreCompact/PostCompact hook。
     */
    pub async fn force_compress_context(
        &self,
    ) -> Result<(ForceCompressStats, Option<CompressionCheckpoint>)> {
        self.fire_lifecycle_hook(crate::skills::hooks::HookEvent::PreCompact, Some("manual"))
            .await;
        let (stats, checkpoint) = self.memory.context.lock().await.force_compress(40).await?;
        let (protected_message_count, protected_context_tokens) = {
            let context = self.memory.context.lock().await;
            (
                context.protected_message_count(),
                context.protected_token_estimate(),
            )
        };
        self.record_trace_event(crate::trace::RunEvent::ContextCompression {
            source: "manual".to_string(),
            before_messages: stats.before_count,
            after_messages: stats.after_count,
            before_tokens: stats.before_tokens,
            after_tokens: stats.after_tokens,
            protected_context_tokens,
            protected_message_count,
        })
        .await;
        self.fire_post_compact_hook("manual", &stats).await;
        Ok((stats, checkpoint))
    }

    /**
     * 列出所有已注册的工具名称
     *
     * # 返回值
     * 已注册工具名称的列表
     */
    pub fn list_tools(&self) -> Vec<String> {
        self.tools.tool_manager.list_tools()
    }

    // ── 子代理 ─────────────────────────────────────────────────────────────

    /**
     * 将一个子代理注册到子代理注册表中。
     *
     * 该 agent 会自动包装为默认的同步模式 `SubagentDefinition`。
     * 如需更精细的控制，请使用 `register_subagent_with_definition()`。
     */
    #[cfg(feature = "subagent")]
    pub fn register_agent(&mut self, agent: Box<dyn Agent>) {
        let name = agent.name().to_string();
        let def = SubagentDefinition::simple_sync(&name);
        self.register_subagent_with_definition(def, agent);
    }

    /**
     * 使用显式定义注册一个子代理。
     *
     * 该定义会通过 `agent_tool` 暴露给编排器，因此描述应当简洁且面向任务。
     */
    #[cfg(feature = "subagent")]
    pub fn register_subagent_with_definition(
        &mut self,
        def: SubagentDefinition,
        agent: Box<dyn Agent>,
    ) {
        // 注意：子代理注册与 `enable_subagent` 标志已解耦。
        // `enable_subagent` 历史上同时控制两件事：(1) 此处的子代理注册，
        // 以及 (2) `ReactAgent::new` 中 `AgentDispatchTool` LLM 工具的注册。
        // 现在两者已拆分：子代理注册是无条件的（框架通过
        // `delegate_to_agent*` 进行的调度依赖它），而 LLM 工具注册
        // 则由 `register_agent_dispatch_tool` 控制。
        let name = def.name.clone();
        if self
            .tools
            .subagent_registry
            .register_sync(def.clone(), agent)
        {
            info!(agent = %self.config.agent_name, subagent = %name, "Subagent registered");
        }
    }

    /**
     * 仅注册子代理**定义**——不含实例，也不含工厂。
     *
     * 这是一个用于延迟绑定的底层入口。待注册的定义可以通过注册表的
     * `get` 和 `contains` 查询到，但在提供实例或工厂之前，
     * 会被排除在 `list_available`、`agent_names` 以及面向模型的
     * 调度目录之外。应用层（拥有构建真正 `ReactAgent` 实例所需的
     * prompt-compiler / tool-filter / sandbox 装配逻辑）之后会通过
     * 同名调用 [`register_subagent_with_definition`](Self::register_subagent_with_definition)
     * 或 [`register_subagent_factory`](Self::register_subagent_factory)
     * 提供实例，覆盖该定义。
     *
     * 在完成填充之前进行显式的程序化调度会失败（没有可运行的 agent）。
     */
    #[cfg(feature = "subagent")]
    pub fn register_subagent_definition(&mut self, def: SubagentDefinition) {
        let name = def.name.clone();
        let inserted = self
            .tools
            .subagent_registry
            .register_definition_sync(def.clone());
        if inserted {
            info!(
                agent = %self.config.agent_name,
                subagent = %name,
                "Subagent definition registered (late-binding, no instance)"
            );
        }
    }

    /// 注册一个工厂，用于为每次隔离调度创建独立的 agent。
    #[cfg(feature = "subagent")]
    pub fn register_subagent_factory(
        &mut self,
        def: SubagentDefinition,
        factory: Arc<dyn AgentFactory>,
    ) {
        let name = def.name.clone();
        if self
            .tools
            .subagent_registry
            .register_factory_sync(def.clone(), factory)
        {
            info!(agent = %self.config.agent_name, subagent = %name, "Subagent factory registered");
        }
    }

    /// 注销一个子代理，并将其从面向模型的调度目录中移除。
    /// 返回移除之前该定义是否存在。
    #[cfg(feature = "subagent")]
    pub async fn unregister_subagent(&mut self, name: &str) -> bool {
        let existed = self.tools.subagent_registry.contains(name).await;
        self.tools.subagent_registry.remove(name).await;
        existed
    }

    /// 为旧版调用方刷新缓存的工具 schema。调度目录本身是从共享注册表中
    /// 读取的，会忽略传入的副本。
    #[cfg(feature = "subagent")]
    pub fn sync_subagent_dispatch_catalog(&self, _definitions: &[SubagentDefinition]) {
        self.tools.tool_manager.invalidate_definition_cache();
    }

    /**
     * 批量注册子代理
     *
     * # 参数
     * * `agents` - 子代理实例列表
     *
     * # 说明
     * 每个子代理都会自动包装为默认的同步模式 `SubagentDefinition`。
     * 如需更精细的控制，请使用 `register_subagent_with_definition()` 方法。
     */
    #[cfg(feature = "subagent")]
    pub fn register_agents(&mut self, agents: Vec<Box<dyn Agent>>) {
        for agent in agents {
            self.register_agent(agent)
        }
    }

    // ── 基础配置 ─────────────────────────────────────────────────────────

    /**
     * 在运行时设置 LLM 模型名称
     *
     * # 参数
     * * `model_name` - 新的 LLM 模型名称
     *
     * # 说明
     * 该方法允许在运行时动态切换 Agent 使用的 LLM 模型。
     */
    pub fn set_model(&mut self, model_name: &str) {
        self.config.model_name = model_name.to_string();
    }

    /**
     * 设置 LLM 生成使用的温度参数。
     *
     * # 参数
     * * `temperature` - 温度值（通常为 0.0 - 2.0）
     */
    pub fn set_temperature(&mut self, temperature: Option<f32>) {
        self.config.temperature = temperature;
    }

    /**
     * 设置 LLM 生成的最大 token 数。
     *
     * # 参数
     * * `max_tokens` - 最大生成 token 数
     */
    pub fn set_max_tokens(&mut self, max_tokens: Option<u32>) {
        self.config.max_tokens = max_tokens;
    }

    /**
     * 准备一次 token 上限变更，但不修改 agent 本身。
     *
     * 该方法会等待获取当前上下文的独占访问权，并在返回的回执中
     * 保留这一访问权。丢弃该回执会回滚此次操作；
     * 与目标绑定的发布提交操作不会失败。
     */
    async fn prepare_token_limit_values(
        &self,
        token_limit: usize,
    ) -> crate::error::Result<PreparedTokenLimitValues> {
        let budget = if self.config.token_budget_config.enabled {
            Some(self.config.token_budget_config.build(token_limit)?)
        } else {
            None
        };
        let context = Arc::clone(&self.memory.context).lock_owned().await;
        Ok(PreparedTokenLimitValues {
            token_limit,
            budget,
            context,
        })
    }

    /// 将已准备好的 token 上限取值提交到 agent 的上下文与配置中。
    fn commit_prepared_token_limit(&mut self, prepared: PreparedTokenLimitValues) {
        let PreparedTokenLimitValues {
            token_limit,
            budget,
            mut context,
        } = prepared;
        context.set_token_limit(token_limit, budget);
        self.config.token_limit = token_limit;
    }

    /**
     * 在运行时设置 token 上限（上下文窗口预算）。
     *
     * 该兼容方法始终保持非阻塞。需要事务性多 agent 更新的调用方
     * 应改用 prepare/commit API。
     */
    pub fn set_token_limit(&mut self, token_limit: usize) -> crate::error::Result<()> {
        let mut context = self.memory.context.try_lock().map_err(|_| {
            crate::error::AgentError::ContextLimitExceeded(
                "cannot change the context window while an agent turn is active".to_string(),
            )
        })?;
        let budget = if self.config.token_budget_config.enabled {
            Some(self.config.token_budget_config.build(token_limit)?)
        } else {
            None
        };
        context.set_token_limit(token_limit, budget);
        self.config.token_limit = token_limit;
        Ok(())
    }

    /**
     * 设置在后续运行中对 LLM 隐藏的 agent 级默认工具。
     *
     * 名称出现在 `names` 中的工具会从发送给模型的工具列表中过滤掉，
     * 使模型无法调用它们。这些工具仍保持已注册状态（不同于会
     * 修改共享注册表的 `remove_tool`）。传入 `None` 或空集合即可清空。
     *
     * 该取值会在创建 `AgentRunSnapshot` 时被捕获，已有的运行不受影响。
     * 单次调用级别的排除项应通过 `AgentInvocationContext::disabled_tools` 提供。
     */
    pub fn set_disabled_tools(&self, names: Option<std::collections::HashSet<String>>) {
        self.tools.set_disabled_tools(names);
    }

    /// 快照当前对未来模型调用隐藏的 agent 级工具集合。
    pub fn disabled_tool_names(&self) -> std::collections::HashSet<String> {
        self.tools.disabled_tool_names()
    }

    /// 返回后续运行使用的可共享工具可见性权限对象。
    pub fn tool_visibility_policy(&self) -> echo_core::agent::ToolVisibilityPolicy {
        self.tools.tool_visibility.clone()
    }

    /// 将该 Agent 绑定到一个已有的工具可见性权限对象上。
    pub fn use_tool_visibility_policy(&mut self, policy: echo_core::agent::ToolVisibilityPolicy) {
        self.tools.tool_visibility = policy;
    }

    /**
     * 添加 Agent 回调
     *
     * # 参数
     * * `callback` - 实现了 `AgentCallback` trait 的回调实例
     *
     * # 说明
     * 回调会在 Agent 执行过程中触发不同事件时被调用，用于监控、日志记录等。
     * 与配置构建器的 `with_callback` 不同，该方法可以在运行时动态添加回调。
     */
    pub fn add_callback(&mut self, callback: Arc<dyn crate::agent::AgentCallback>) {
        self.config.callbacks.push(callback);
    }

    /**
     * 移除所有类型名称匹配给定字符串的回调。
     *
     * 适用于任务结束后清理临时回调（例如进度桥接器）。
     */
    pub fn remove_callbacks_by_type_name(&mut self, type_name: &str) {
        self.config.callbacks.retain(|cb| {
            let name = std::any::type_name_of_val(cb.as_ref()).to_string();
            let kind_matches = cb
                .callback_kind()
                .map(|kind| kind.contains(type_name))
                .unwrap_or_else(|| name.contains(type_name));
            !kind_matches
        });
    }

    /**
     * 移除给定类型中，稳定回调 ID 匹配的回调。
     *
     * 适用于像 `ProgressBridge` 这样任务作用域的临时回调，
     * 因为仅按类型移除可能会误删另一个活跃任务的回调。
     */
    pub fn remove_callbacks_by_type_name_and_id(&mut self, type_name: &str, callback_id: &str) {
        self.config.callbacks.retain(|cb| {
            let name = std::any::type_name_of_val(cb.as_ref()).to_string();
            let kind_matches = cb
                .callback_kind()
                .map(|kind| kind.contains(type_name))
                .unwrap_or_else(|| name.contains(type_name));
            !(kind_matches && cb.callback_id() == Some(callback_id))
        });
    }

    /**
     * 访问执行序列化锁。
     *
     * 当你需要在多个 agent 操作之间持有执行锁时使用（例如，先注册一个
     * 临时回调再执行，这两步相对于其他并发执行必须是原子的）。
     *
     * 对于普通的 `execute()` / `chat()` / `chat_stream()` 调用，
     * 该锁会在 `run_react_loop` / `run_loop` 内部自动获取。
     */
    pub fn execution_mutex(&self) -> &Arc<tokio::sync::Mutex<()>> {
        &self.execution_mutex
    }

    // ── 技能（基于代码）──────────────────────────────────────────────────

    /// 安装一个基于代码的技能（立即生效：工具与提示词立刻被注入）。
    pub fn add_skill(&mut self, skill: Box<dyn Skill>) {
        let name = skill.name().to_string();

        if self.tools.skill_registry.is_installed(&name) {
            warn!(
                agent = %self.config.agent_name,
                skill = %name,
                "Skill already installed, skipping"
            );
            return;
        }

        let sandbox = self
            .tools
            .sandbox_manager
            .as_ref()
            .map(|manager| manager.clone() as Arc<dyn crate::sandbox::SandboxExecutor>);
        let tools = skill.tools_with_sandbox(sandbox);
        let tool_names: Vec<String> = tools.iter().map(|t| t.name().to_string()).collect();

        for tool in tools {
            self.tools.tool_manager.register(tool);
        }

        let has_injection = skill.system_prompt_injection().is_some();
        if let Some(injection) = skill.system_prompt_injection() {
            self.config.system_prompt.push_str(&injection);
            self.memory
                .context
                .try_lock()
                .map(|mut ctx| ctx.update_system(self.config.system_prompt.clone()))
                .unwrap_or_else(|e| {
                    tracing::warn!("Failed to acquire context lock for skill injection: {}", e);
                });
        }

        self.tools.skill_registry.record_code_skill(SkillInfo {
            name: name.clone(),
            description: skill.description().to_string(),
            tool_names,
            has_prompt_injection: has_injection,
        });

        info!(
            agent = %self.config.agent_name,
            skill = %name,
            description = %skill.description(),
            "Skill installed"
        );
    }

    /// 批量安装多个基于代码的技能。
    pub fn add_skills(&mut self, skills: Vec<Box<dyn Skill>>) {
        for skill in skills {
            self.add_skill(skill);
        }
    }

    // ── 技能（基于文件，渐进式披露）──────────────────────────────

    /**
     * 从给定的多个作用域中发现基于文件的技能。
     *
     * 实现了 agentskills.io 的渐进式披露模型：
     * 1. 从每个作用域解析 `SKILL.md` 的 frontmatter（名称 + 描述）
     * 2. 构建一份紧凑的**目录**并注入到系统提示词中
     * 3. 注册 `activate_skill` 和 `read_skill_resource` 工具
     *
     * 之后 LLM 就可以按需决定激活哪些技能。
     */
    pub async fn discover_skills(&mut self, scopes: &[DiscoveryScope]) -> Result<Vec<String>> {
        self.discover_skills_inner(scopes, None).await
    }

    /// `discover_skills` 的内部实现，额外支持传入插件来源信息。
    async fn discover_skills_inner(
        &mut self,
        scopes: &[DiscoveryScope],
        plugin: Option<(&str, &crate::plugin::PluginVariables)>,
    ) -> Result<Vec<String>> {
        let mut loader = match &self.skill_load_policy {
            Some(policy) => SkillLoader::new().with_policy(policy.clone()),
            None => SkillLoader::new(),
        };
        if let Some((_, variables)) = plugin {
            loader = loader.with_plugin_variables(variables.clone());
        }
        let descriptors = match plugin {
            Some(_) => {
                let directory = scopes.iter().find_map(|scope| match scope {
                    DiscoveryScope::Custom(path) => Some(path.clone()),
                    DiscoveryScope::Project(_) | DiscoveryScope::User => None,
                });
                match directory {
                    Some(directory) => loader.discover_agent_plugin_skills(directory).await?,
                    None => Vec::new(),
                }
            }
            None => loader.discover(scopes).await?,
        };

        let registrations = descriptors
            .into_iter()
            .map(SkillRegistration::Filesystem)
            .collect();
        self.register_skill_descriptors(registrations, plugin, true)
            .await
    }

    /// 将一批 Skill 注册项写入目录及渐进式披露工具的定义视图。
    async fn register_skill_descriptors(
        &mut self,
        registrations: Vec<SkillRegistration>,
        plugin: Option<(&str, &crate::plugin::PluginVariables)>,
        apply_load_policy: bool,
    ) -> Result<Vec<String>> {
        if registrations.is_empty() {
            info!(
                agent = %self.config.agent_name,
                "No skills found during discovery"
            );
            return Ok(Vec::new());
        }

        let mut names = Vec::new();

        // 为渐进式披露工具构建一个并发的定义视图。描述符/文档保留在
        // 本适配器本地，而激活及激活衍生出的沙箱策略则使用主注册表的
        // 共享运行时权限。
        let shared = if let Some(existing) = &self.tools.progressive_skill_registry {
            existing.clone()
        } else {
            let reg = Arc::new(RwLock::new(self.tools.skill_registry.activation_view()));
            if let Some(ref manager) = self.tools.sandbox_manager {
                if let Ok(mut guard) = reg.try_write() {
                    guard.set_sandbox_manager(manager.clone());
                }
                self.tools
                    .skill_registry
                    .set_sandbox_manager(manager.clone());
            }
            self.tools.progressive_skill_registry = Some(reg.clone());
            reg
        };

        {
            let mut reg = shared.write().await;
            if let Some(ref manager) = self.tools.sandbox_manager {
                reg.set_sandbox_manager(manager.clone());
                self.tools
                    .skill_registry
                    .set_sandbox_manager(manager.clone());
            }
            for mut registration in registrations {
                if apply_load_policy
                    && let Some(policy) = &self.skill_load_policy
                    && !policy.allows(registration.descriptor()).await
                {
                    info!(
                        agent = %self.config.agent_name,
                        skill = %registration.descriptor().name,
                        "Skill excluded by load policy"
                    );
                    continue;
                }
                if self
                    .tools
                    .skill_registry
                    .is_installed(&registration.descriptor().name)
                {
                    warn!(
                        agent = %self.config.agent_name,
                        skill = %registration.descriptor().name,
                        "Skill already installed, skipping duplicate"
                    );
                    continue;
                }

                if let Some((source, _)) = plugin {
                    registration.set_registration_source(source);
                }

                // 若程序化描述符中存在 hook 定义，则一并注册。
                if let Some(hooks_def) = &registration.descriptor().hooks {
                    let skill_dir = registration
                        .descriptor()
                        .location
                        .parent()
                        .map(|p| p.display().to_string())
                        .unwrap_or_default();
                    let mut hook_reg = self.tools.hook_registry.write().await;
                    hook_reg.register(
                        &registration.descriptor().name,
                        &skill_dir,
                        hooks_def.clone(),
                    );
                }

                let name = registration.descriptor().name.clone();
                names.push(name.clone());
                match registration {
                    SkillRegistration::Prepared(document) => {
                        self.tools
                            .skill_registry
                            .register_prepared(document.clone());
                        reg.register_prepared(document);
                    }
                    SkillRegistration::Filesystem(descriptor) => {
                        self.tools
                            .skill_registry
                            .register_descriptor(descriptor.clone());
                        reg.register_descriptor(descriptor);
                    }
                }
                if let Some((source, variables)) = plugin {
                    self.tools.skill_registry.tag_source_with_variables(
                        std::slice::from_ref(&name),
                        source,
                        Some(variables),
                    );
                    reg.tag_source_with_variables(
                        std::slice::from_ref(&name),
                        source,
                        Some(variables),
                    );
                }
            }
        }

        if names.is_empty() {
            return Ok(names);
        }

        // 让目录保持可独立替换。若每次发现都修改根系统提示词，
        // 会导致内容重复，并使 provider 最长可缓存前缀失效。
        if let Some(catalog) = self.tools.skill_registry.catalog_prompt() {
            self.memory.context.lock().await.replace_projection(
                SKILL_CATALOG_PROJECTION,
                Some(crate::llm::types::Message::system(catalog)),
            );
        }

        // 触发 InstructionsLoaded hook
        {
            let hook_ctx = crate::skills::hooks::HookContext::for_instructions_loaded(
                "startup",
                &names,
                self.config.session_id.as_deref().unwrap_or(""),
                &self.config.agent_name,
            );
            let registry = self.tools.hook_registry.read().await.clone();
            let result = registry.run_lifecycle_hooks(&hook_ctx).await;
            if result.block {
                warn!(agent = %self.config.agent_name, reason = ?result.block_reason, "InstructionsLoaded hook blocked");
            }
            if let Some(ctx) = &result.injected_context {
                super::run::context::push_runtime_context_note(
                    &self.memory.context,
                    "Hook:InstructionsLoaded",
                    ctx,
                )
                .await;
            }
            for msg in &result.messages {
                super::run::context::push_runtime_context_note(
                    &self.memory.context,
                    "Hook:InstructionsLoaded",
                    msg,
                )
                .await;
            }
        }

        // 用共享注册表注册渐进式披露工具。
        // 每次发现都会刷新这些工具实例，以确保其内部注册表和
        // 可用技能名称列表，能够与多次 `discover_skills()` 调用中
        // 新发现的技能保持同步。
        let available_names = self.tools.skill_registry.available_names();

        self.replace_tool(Box::new(ActivateSkillTool::new(
            shared.clone(),
            available_names,
        )));
        self.replace_tool(Box::new(ReadSkillResourceTool::new(shared.clone())));

        let mut script_tool = RunSkillScriptTool::new(shared);
        if let Some(ref manager) = self.tools.sandbox_manager {
            script_tool = script_tool.with_sandbox_manager(manager.clone());
        }
        self.replace_tool(Box::new(script_tool));

        info!(
            agent = %self.config.agent_name,
            count = names.len(),
            skills = ?names,
            "Skills discovered and catalog injected"
        );

        Ok(names)
    }

    /// 用一个新的 Skill 注册项替换同名的现有注册（不允许与代码技能冲突）。
    async fn replace_skill_registration(&mut self, registration: SkillRegistration) -> Result<()> {
        let name = registration.descriptor().name.clone();
        if self.tools.skill_registry.has_code_skill(&name) {
            return Err(crate::error::ReactError::Other(format!(
                "skill '{name}' conflicts with an installed code skill"
            )));
        }
        if let Some(policy) = &self.skill_load_policy
            && !policy.allows(registration.descriptor()).await
        {
            return Err(crate::error::ReactError::Other(format!(
                "skill '{name}' was rejected by the active load policy"
            )));
        }
        self.unregister_skill_names(std::slice::from_ref(&name))
            .await;
        let registered = self
            .register_skill_descriptors(vec![registration], None, false)
            .await?;
        if registered.iter().any(|registered| registered == &name) {
            Ok(())
        } else {
            Err(crate::error::ReactError::Other(format!(
                "skill '{name}' was rejected by the active load policy or conflicts with a code skill"
            )))
        }
    }

    /// 在目录与渐进式披露工具定义视图中，注册或替换一个基于文件的描述符。
    pub async fn register_skill_descriptor(&mut self, descriptor: SkillDescriptor) -> Result<()> {
        self.replace_skill_registration(SkillRegistration::Filesystem(descriptor))
            .await
    }

    /// 在每一个运行时定义视图中，注册或替换一份不可变的已预备 Skill 文档。
    pub async fn register_prepared_skill(
        &mut self,
        document: crate::skills::external::SkillDocument,
    ) -> Result<()> {
        self.replace_skill_registration(SkillRegistration::Prepared(document))
            .await
    }

    /// 通过 Agent 权限记录代码技能的元数据。
    pub fn record_code_skill_info(&mut self, info: SkillInfo) {
        self.tools.skill_registry.record_code_skill(info);
    }

    /**
     * 注册由一次不可变的已预备生成所捕获的插件 Skill。
     *
     * 该路径不执行任何发现或文档读取。在应用冻结的描述符之前，
     * 仍会评估针对该目标的加载策略。
     */
    pub async fn register_prepared_plugin_skills(
        &mut self,
        source: &str,
        variables: &crate::plugin::PluginVariables,
        skills: &[crate::plugin::PreparedPluginSkill],
    ) -> Result<Vec<String>> {
        let registrations = skills
            .iter()
            .map(|skill| SkillRegistration::Prepared(skill.document().clone()))
            .collect();
        self.register_skill_descriptors(registrations, Some((source, variables)), true)
            .await
    }

    /// 移除已发现但不再被当前产品策略允许的技能，然后刷新目录与渐进式工具。
    pub async fn reconcile_skill_load_policy(&mut self) -> Vec<String> {
        let Some(policy) = &self.skill_load_policy else {
            return Vec::new();
        };
        let descriptors = self.tools.skill_registry.list_descriptors();
        let mut removed = Vec::new();
        for descriptor in descriptors {
            if !policy.allows(descriptor).await {
                removed.push(descriptor.name.clone());
            }
        }
        if removed.is_empty() {
            return removed;
        }

        self.unregister_skill_names(&removed).await;

        info!(
            agent = %self.config.agent_name,
            skills = ?removed,
            "Skills removed by load policy reconciliation"
        );
        removed
    }

    /**
     * 从每一个运行时投影中移除指定名称的基于文件的 Skill。
     *
     * 这是重新加载与策略协调所使用的替换基础操作。它会一并清除
     * 描述符、Hook、渐进式资源以及提示词投影，从而使后续的发现
     * 能够替换已变更的 `SKILL.md`，而不会遗留过期的元数据。
     */
    pub async fn unregister_skill_names(&mut self, names: &[String]) -> Vec<String> {
        let mut removed = Vec::new();
        for name in names {
            if !self.tools.skill_registry.remove_descriptor(name) {
                continue;
            }
            self.tools
                .hook_registry
                .write()
                .await
                .unregister(&crate::skills::hooks::HookSource::Skill(name.clone()));
            self.memory
                .context
                .lock()
                .await
                .replace_projection(format!("echo-agent:skill:{name}"), None);
            removed.push(name.clone());
        }

        if let Some(shared) = &self.tools.progressive_skill_registry {
            let mut registry = shared.write().await;
            for name in &removed {
                registry.remove_descriptor(name);
            }
        }

        let catalog = self
            .tools
            .skill_registry
            .catalog_prompt()
            .map(crate::llm::types::Message::system);
        self.memory
            .context
            .lock()
            .await
            .replace_projection(SKILL_CATALOG_PROJECTION, catalog);

        if let Some(shared) = self.tools.progressive_skill_registry.clone() {
            let available_names = self.tools.skill_registry.available_names();
            self.replace_tool(Box::new(ActivateSkillTool::new(
                shared.clone(),
                available_names,
            )));
            self.replace_tool(Box::new(ReadSkillResourceTool::new(shared.clone())));
            let mut script_tool = RunSkillScriptTool::new(shared);
            if let Some(manager) = &self.tools.sandbox_manager {
                script_tool = script_tool.with_sandbox_manager(manager.clone());
            }
            self.replace_tool(Box::new(script_tool));
        }
        removed
    }

    /**
     * 替换所有当前来源于 `skills_dir` 的 Skill。
     *
     * 正常的发现流程会保留已存在的同名描述符。这个显式的重新加载 API
     * 会先移除该目录下的旧注册项，使变更后的描述、触发器、资源和
     * Hook 能够在下一次注册时原子地生效。
     */
    pub async fn reload_skills_from_dir(
        &mut self,
        skills_dir: impl Into<std::path::PathBuf>,
    ) -> Result<Vec<String>> {
        let skills_dir = skills_dir.into();
        // 发现流程会将每个 SKILL.md 的路径规范化（解析符号链接路径部分），
        // 因此重新加载的过滤逻辑必须比较规范化后的形式，否则传入
        // 非规范化参数会静默匹配不到任何内容，导致重新加载相对于
        // 已安装的描述符退化为空操作。
        let skills_dir = tokio::fs::canonicalize(&skills_dir)
            .await
            .unwrap_or(skills_dir);
        let names = self
            .skill_descriptors()
            .into_iter()
            .filter(|descriptor| descriptor.location.starts_with(&skills_dir))
            .map(|descriptor| descriptor.name)
            .collect::<Vec<_>>();
        self.unregister_skill_names(&names).await;
        self.load_skills_from_dir(skills_dir).await
    }

    /// 从某一个显式目录中发现技能。
    pub async fn load_skills_from_dir(
        &mut self,
        skills_dir: impl Into<std::path::PathBuf>,
    ) -> Result<Vec<String>> {
        self.discover_skills(&[DiscoveryScope::Custom(skills_dir.into())])
            .await
    }

    /**
     * 发现插件所拥有的技能，在标准 frontmatter 与正文解析之前
     * 先应用变量替换。插件 Hook 由插件 Hook 组件负责注册，
     * 而不是由 Skill 文件负责。
     */
    pub async fn load_plugin_skills_from_dir(
        &mut self,
        skills_dir: impl Into<std::path::PathBuf>,
        source: &str,
        variables: &crate::plugin::PluginVariables,
    ) -> Result<Vec<String>> {
        self.discover_skills_inner(
            &[DiscoveryScope::Custom(skills_dir.into())],
            Some((source, variables)),
        )
        .await
    }

    /**
     * 在两个存活注册表中，为新发现的技能标记同一个组件来源。
     * 渐进式注册表驱动着激活/资源工具；如果只标记目录注册表，
     * 已被禁用的插件技能仍可通过这些工具被执行。
     */
    pub async fn tag_skills_source(&mut self, names: &[String], source: &str) {
        self.tools.skill_registry.tag_source(names, source);
        if let Some(shared) = &self.tools.progressive_skill_registry {
            shared.write().await.tag_source(names, source);
        }
    }

    /// 在每一个存活的注册表视图中，为 Skill 定义标记来源与插件变量。
    pub async fn tag_skills_source_with_variables(
        &mut self,
        names: &[String],
        source: &str,
        variables: Option<&crate::plugin::PluginVariables>,
    ) {
        self.tools
            .skill_registry
            .tag_source_with_variables(names, source, variables);
        if let Some(shared) = &self.tools.progressive_skill_registry {
            shared
                .write()
                .await
                .tag_source_with_variables(names, source, variables);
        }
    }

    /// 移除所有归属于 `source` 的技能，并刷新所有投影/工具。
    pub async fn unregister_skills_by_source(&mut self, source: &str) -> Vec<String> {
        let removed = self.tools.skill_registry.unregister_names_by_source(source);
        if removed.is_empty() {
            return removed;
        }

        if let Some(shared) = &self.tools.progressive_skill_registry {
            shared.write().await.unregister_by_source(source);
        }
        {
            let mut hooks = self.tools.hook_registry.write().await;
            for name in &removed {
                hooks.unregister(&crate::skills::hooks::HookSource::Skill(name.clone()));
            }
        }
        {
            let mut context = self.memory.context.lock().await;
            for name in &removed {
                context.replace_projection(format!("echo-agent:skill:{name}"), None);
            }
            let catalog = self
                .tools
                .skill_registry
                .catalog_prompt()
                .map(crate::llm::types::Message::system);
            context.replace_projection(SKILL_CATALOG_PROJECTION, catalog);
        }

        if let Some(shared) = self.tools.progressive_skill_registry.clone() {
            let available_names = self.tools.skill_registry.available_names();
            self.replace_tool(Box::new(ActivateSkillTool::new(
                shared.clone(),
                available_names,
            )));
            self.replace_tool(Box::new(ReadSkillResourceTool::new(shared.clone())));
            let mut script_tool = RunSkillScriptTool::new(shared);
            if let Some(manager) = &self.tools.sandbox_manager {
                script_tool = script_tool.with_sandbox_manager(manager.clone());
            }
            self.replace_tool(Box::new(script_tool));
        }

        removed
    }

    /**
     * 激活一个已安装的技能，并将其指令注入到上下文中。
     *
     * 所有激活路径都应使用该方法，以避免注册表状态与模型上下文出现分歧。
     */
    pub async fn activate_skill(&self, skill_name: &str) -> Result<()> {
        if !self.tools.skill_registry.is_installed(skill_name) {
            /*
             * 报错而非静默返回：用户命令路径（TUI /skill、GUI 激活按钮）需要
             * 区分“激活成功”与“skill 根本不在注册表里（未发现或被禁用）”。
             * 容忍失败的调用方（如 workspace 路由）自行忽略该错误即可。
             */
            return Err(crate::error::ReactError::Other(format!(
                "skill '{skill_name}' is not installed (not discovered, disabled, or removed)"
            )));
        }

        let projection_marker = format!("echo-agent:skill:{skill_name}");
        if self.tools.skill_registry.is_activated(skill_name)
            && self
                .memory
                .context
                .lock()
                .await
                .has_projection(projection_marker.as_str())
        {
            return Ok(());
        }

        let content = self.tools.skill_registry.activate(skill_name).await?;
        let block = content.to_prompt_block();
        project_skill_activation(&self.memory.context, skill_name, &block).await;

        tracing::info!(
            agent = %self.config.agent_name,
            skill = %skill_name,
            "Skill activated and injected as a replaceable context projection"
        );
        Ok(())
    }

    /**
     * 替换或移除一个具名的系统上下文投影。
     *
     * 产品运行时会用它来处理用户可选择的指令类内容（例如输出风格）。
     * marker 是稳定的所有权边界，因此替换或卸载某个投影
     * 不会修改基础提示词或无关的运行时上下文。
     */
    pub async fn replace_system_context_projection(
        &self,
        marker: impl Into<String>,
        content: Option<String>,
    ) {
        let message = content.map(crate::llm::types::Message::system);
        self.memory
            .context
            .lock()
            .await
            .replace_projection(marker.into(), message);
    }

    /// 列出所有已安装的基于代码的技能。
    pub fn list_skills(&self) -> Vec<SkillInfo> {
        self.tools
            .skill_registry
            .list()
            .into_iter()
            .cloned()
            .collect()
    }

    /// 检查某个技能（代码或文件形式）是否已安装。
    pub fn has_skill(&self, name: &str) -> bool {
        self.tools.skill_registry.is_installed(name)
    }

    /// 已安装技能的总数（代码 + 文件形式）。
    pub fn skill_count(&self) -> usize {
        self.tools.skill_registry.count()
    }

    /**
     * 获取用于读取和激活操作的权威 Skill 注册表。
     *
     * Agent 拥有的定义变更会使用上面提到的协调方法，
     * 以避免渐进式工具保留过期的描述符视图。
     */
    pub fn skill_registry(&self) -> &crate::skills::SkillRegistry {
        &self.tools.skill_registry
    }

    /// 获取共享的 hook 注册表句柄。
    pub fn hook_registry(&self) -> &Arc<tokio::sync::RwLock<crate::skills::hooks::HookRegistry>> {
        &self.tools.hook_registry
    }

    /// 获取共享的工具管理器句柄。
    pub fn tool_manager(&self) -> &Arc<echo_execution::tools::ToolManager> {
        &self.tools.tool_manager
    }

    /// 获取沙箱管理器（如果已配置）。
    pub fn sandbox_manager(&self) -> Option<&Arc<dyn crate::sandbox::SandboxExecutor>> {
        self.tools.sandbox_manager.as_ref()
    }

    /// 获取工具执行流水线（如果已配置）。
    pub fn tool_execution_pipeline(
        &self,
    ) -> &Option<Arc<crate::agent::react::run::pipeline::ToolExecutionPipeline>> {
        &self.tool_execution_pipeline
    }

    /// 获取运行时状态存储（如果已配置）。
    pub fn state_store(&self) -> &Option<Arc<dyn crate::state::RuntimeStateStore>> {
        &self.memory.state_store
    }

    /// 获取对话存储（如果已配置）。
    pub fn conversation_store(&self) -> &Option<Arc<dyn crate::memory::ConversationStore>> {
        &self.memory.conversation_store
    }

    #[cfg(feature = "human-loop")]
    /// 获取权限服务（如果已配置）。
    pub fn permission_service(
        &self,
    ) -> Option<&Arc<crate::human_loop::service::PermissionService>> {
        self.approval.permission_service.as_ref()
    }

    /**
     * 创建一个 `TaskHookBridge`，让应用层拥有的任务运行时能够将其
     * 持久化的生命周期事件转译到中心注册表中。
     *
     * 任务运行时适配器会在 `TaskRevisionService` 提交以及
     * `RuntimeDagController` 执行边界处调用生命周期 hook。
     *
     * 该桥接器会将任务生命周期事件（`TaskCreated`、`TaskStarted`，
     * 以及终态的 `TaskCompleted`）触发到中心 `HookRegistry` 中。
     * 终止原因通过状态字段携带。
     */
    pub fn create_task_hook_bridge(&self) -> crate::hooks_bridge::TaskHookBridge {
        crate::hooks_bridge::TaskHookBridge::new(
            self.tools.hook_registry.clone(),
            self.config
                .session_id
                .clone()
                .unwrap_or_else(|| "default".to_string()),
            self.config.agent_name.clone(),
        )
    }

    /**
     * 创建一个 `SubagentHookBridge`，将子代理生命周期事件
     * （SubagentStart、SubagentStop）触发到中心 `HookRegistry` 中。
     * SubagentStop 携带一个终态状态
     * （completed/failed/cancelled/timed_out）；并不存在单独的
     * SubagentCancelled 事件。
     */
    pub fn create_subagent_hook_bridge(&self) -> crate::hooks_bridge::SubagentHookBridge {
        crate::hooks_bridge::SubagentHookBridge::new(
            self.tools.hook_registry.clone(),
            self.config
                .session_id
                .clone()
                .unwrap_or_else(|| "default".to_string()),
            self.config.agent_name.clone(),
        )
    }

    // ── MCP ──────────────────────────────────────────────────────────────────

    /**
     * 注册 MCP（Model Context Protocol）工具
     *
     * # 参数
     * * `tools` - MCP 工具实例列表
     *
     * # 说明
     * 将 MCP 工具注册到 Agent 的工具管理器中，使其可用于工具调用。
     */
    pub fn register_mcp_tools(&mut self, tools: Vec<Box<dyn Tool>>) {
        self.add_tools(tools);
    }

    /// 移除旧的 MCP 资源工具，并根据当前 MCP 管理器重新注册最新的资源工具。
    #[cfg(feature = "mcp")]
    fn sync_mcp_resource_tools(&mut self) {
        for name in crate::mcp::MCP_RESOURCE_TOOL_NAMES {
            self.remove_tool(name);
        }
        let tools = self.tools.mcp_manager.resource_tools();
        self.add_tools(tools);
    }

    #[cfg(feature = "mcp")]
    /**
     * 根据 MCP 服务器配置连接到 MCP 服务器
     *
     * # 参数
     * * `config` - MCP 服务器配置
     *
     * # 返回值
     * 返回连接成功后创建的 `McpClient` 实例
     *
     * # 说明
     * 该方法根据配置连接到 MCP 服务器，获取该服务器提供的工具，
     * 并将其注册到 Agent 的工具管理器中。
     */
    pub async fn connect_mcp_from_config(
        &mut self,
        config: McpServerConfig,
    ) -> crate::error::Result<Arc<McpClient>> {
        let name = config.name.clone();
        self.reconcile_mcp_target(&name, Some(config)).await?;
        let client = self.tools.mcp_manager.get_client(&name).ok_or_else(|| {
            crate::error::ReactError::Agent(Box::new(
                crate::error::AgentError::InitializationFailed(format!(
                    "MCP client '{}' not found after connection",
                    name
                )),
            ))
        })?;
        Ok(client)
    }

    /// 连接一个由插件拥有的 MCP 服务器，且不会将其权限并入直接的
    /// 服务器命名空间中。
    #[cfg(feature = "mcp")]
    pub async fn connect_mcp_owned(
        &mut self,
        id: McpServerId,
        config: McpServerConfig,
    ) -> crate::error::Result<Arc<McpClient>> {
        let previous_client = self.tools.mcp_manager.get_client_by_id(&id);
        let previous_tool_names = previous_client
            .as_ref()
            .map(|client| {
                client
                    .tools()
                    .iter()
                    .map(|tool| {
                        crate::mcp::McpToolAdapter::exposed_name_for_identity(&id, &tool.name)
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let result = self
            .tools
            .mcp_manager
            .reconcile_server(&id, Some(config))
            .await;
        let withdrawn = result.is_err()
            && previous_client.is_some()
            && !self
                .tools
                .mcp_manager
                .get_client_by_id(&id)
                .is_some_and(|current| {
                    previous_client
                        .as_ref()
                        .is_some_and(|previous| Arc::ptr_eq(&current, previous))
                });
        match (result, withdrawn) {
            (Ok(mut receipt), _) => {
                for tool_name in previous_tool_names {
                    self.remove_tool(&tool_name);
                }
                self.add_tools(std::mem::take(&mut receipt.tools));
                self.sync_mcp_resource_tools();
                self.setup_hook_mcp_executor().await;
                self.tools.mcp_manager.get_client_by_id(&id).ok_or_else(|| {
                    crate::error::ReactError::Agent(Box::new(
                        crate::error::AgentError::InitializationFailed(format!(
                            "MCP client '{}' not found after connection",
                            id
                        )),
                    ))
                })
            }
            (Err(error), true) => {
                for tool_name in previous_tool_names {
                    self.remove_tool(&tool_name);
                }
                self.sync_mcp_resource_tools();
                self.setup_hook_mcp_executor().await;
                Err(error)
            }
            (Err(error), false) => Err(error),
        }
    }

    #[cfg(feature = "mcp")]
    /**
     * 协调一个具名的 MCP 条目，在替换项准备完成之前，不会先拆除
     * 正在工作的目标。
     *
     * `Some(entry)` 会根据框架 `McpManager` 的约定连接、替换或保留该
     * 具名服务器；`None` 表示移除它。Agent 仅负责该管理器操作周边的
     * 已暴露工具注册。
     */
    pub async fn reconcile_mcp_entry(
        &mut self,
        name: &str,
        entry: Option<McpServerEntry>,
    ) -> crate::error::Result<McpTargetChange> {
        let desired = match entry {
            Some(entry) if !entry.disabled => Some(entry.to_server_config(name)?),
            Some(_) | None => None,
        };
        self.reconcile_mcp_target(name, desired).await
    }

    /// 协调一个具名 MCP 目标：连接、替换、保留或移除，并同步暴露的工具投影。
    #[cfg(feature = "mcp")]
    async fn reconcile_mcp_target(
        &mut self,
        name: &str,
        desired: Option<McpServerConfig>,
    ) -> crate::error::Result<McpTargetChange> {
        // McpManager 会在切换客户端之前先准备好替换项。只有该操作成功后
        // 才移除旧的已暴露工具，否则一次失败的替换会让最后已知可用的
        // 目标保持未被触碰的状态。
        let previous_client = self.tools.mcp_manager.get_client(name);
        let previous_tool_names = previous_client
            .as_ref()
            .map(|client| {
                client
                    .tools()
                    .iter()
                    .map(|tool| crate::mcp::McpToolAdapter::exposed_name_for(name, &tool.name))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let result = self.tools.mcp_manager.reconcile_target(name, desired).await;
        // 当清理失败时，管理器会撤下该具名客户端（转入清理欠账状态），
        // 但当目标准备失败时，会保留最后已知可用的客户端。只有“已撤下”
        // 这种情况会留下过期投影，需要被撤销以匹配最终的拓扑结果。
        let withdrawn = result.is_err()
            && previous_client.is_some()
            && !self
                .tools
                .mcp_manager
                .get_client(name)
                .is_some_and(|current| {
                    previous_client
                        .as_ref()
                        .is_some_and(|previous| Arc::ptr_eq(&current, previous))
                });
        match (result, withdrawn) {
            (Ok(mut receipt), _) => {
                let change = receipt.change;
                let tool_count = receipt.tools.len();
                for tool_name in previous_tool_names {
                    self.remove_tool(&tool_name);
                }
                self.add_tools(std::mem::take(&mut receipt.tools));
                self.sync_mcp_resource_tools();
                self.setup_hook_mcp_executor().await;
                tracing::info!(
                    agent = %self.config.agent_name,
                    server = %name,
                    ?change,
                    tools = tool_count,
                    "MCP server target reconciled"
                );
                Ok(change)
            }
            (Err(error), true) => {
                for tool_name in previous_tool_names {
                    self.remove_tool(&tool_name);
                }
                self.sync_mcp_resource_tools();
                self.setup_hook_mcp_executor().await;
                tracing::warn!(
                    agent = %self.config.agent_name,
                    server = %name,
                    error = %error,
                    "MCP target was withdrawn after cleanup failure; stale projections revoked"
                );
                Err(error)
            }
            (Err(error), false) => Err(error),
        }
    }

    #[cfg(feature = "mcp")]
    /**
     * 从 JSON 字符串配置连接到 MCP 服务器
     *
     * # 参数
     * * `name` - MCP 服务器名称
     * * `json_config_str` - JSON 格式的 MCP 服务器配置字符串
     *
     * # 返回值
     * 返回连接成功后创建的 `McpClient` 实例
     *
     * # 说明
     * 该方法解析 JSON 格式的配置字符串，将其转换为 MCP 服务器配置，
     * 然后调用 `connect_mcp_from_config` 进行连接。
     */
    pub async fn connect_mcp_from_json(
        &mut self,
        name: &str,
        json_config_str: &str,
    ) -> crate::error::Result<Arc<McpClient>> {
        let entry = serde_json::from_str::<McpServerEntry>(json_config_str)?;
        let config = entry.to_server_config(name)?;
        self.connect_mcp_from_config(config).await
    }

    #[cfg(feature = "mcp")]
    /**
     * 从配置文件中加载并连接多个 MCP 服务器
     *
     * # 参数
     * * `path` - MCP 配置文件路径（支持 `.json` 或 `.yaml` 格式）
     *
     * # 返回值
     * 返回成功连接的 `McpClient` 实例列表
     *
     * # 说明
     * 1. 解析 MCP 配置文件（支持 JSON 或 YAML 格式）
     * 2. 为每个服务器配置调用 `connect_mcp_from_config`
     * 3. 收集所有连接成功的客户端
     * 4. 连接失败的服务器会被跳过，并记录一条警告日志
     *
     * # 示例
     * ```rust
     * use echo_agent::prelude::*;
     *
     * # #[tokio::main]
     * # async fn main() -> echo_agent::error::Result<()> {
     * let path = std::env::temp_dir().join(format!(
     *     "echo-agent-doctest-mcp-{}.json",
     *     std::process::id()
     * ));
     * std::fs::write(&path, r#"{"mcpServers":{}}"#)?;
     *
     * let config = AgentConfig::new("qwen3-max", "mcp_agent", "You are a helpful assistant");
     * let mut agent = ReactAgent::new(config);
     * let clients = agent.load_mcp_from_file(&path).await?;
     * assert!(clients.is_empty());
     * let _ = std::fs::remove_file(&path);
     * # Ok(())
     * # }
     * ```
     */
    pub async fn load_mcp_from_file(
        &mut self,
        path: impl AsRef<std::path::Path>,
    ) -> crate::error::Result<Vec<Arc<McpClient>>> {
        let config = McpConfigFile::from_file(path)?;
        self.load_mcp_config(config).await
    }

    /// 连接一份已解析的 MCP 配置中所有已启用的服务器。
    #[cfg(feature = "mcp")]
    pub async fn load_mcp_config(
        &mut self,
        config: McpConfigFile,
    ) -> crate::error::Result<Vec<Arc<McpClient>>> {
        let server_configs = config.to_server_configs()?;
        let mut clients = Vec::new();
        for server_config in server_configs {
            let name = server_config.name.clone();
            match self.connect_mcp_from_config(server_config).await {
                Ok(client) => clients.push(client),
                Err(e) => {
                    tracing::warn!(
                        agent = %self.config.agent_name,
                        server = %name,
                        error = %e,
                        "MCP server connection failed, skipping"
                    );
                }
            }
        }
        Ok(clients)
    }

    #[cfg(feature = "mcp")]
    /**
     * 获取一个已连接的 MCP 客户端
     *
     * # 参数
     * * `name` - MCP 服务器名称
     *
     * # 返回值
     * 返回指定名称的共享 MCP 客户端，若不存在则返回 `None`
     *
     * # 说明
     * 该方法获取一个已连接的 MCP 客户端，用于直接调用其方法。
     * 客户端通过 `connect_mcp_from_config` 或 `load_mcp_from_file` 建立连接。
     */
    pub fn mcp_client(&self, name: &str) -> Option<Arc<McpClient>> {
        self.tools.mcp_manager.get_client(name)
    }

    #[cfg(feature = "mcp")]
    /**
     * 列出所有已连接的 MCP 服务器名称
     *
     * # 返回值
     * 已连接的 MCP 服务器名称列表
     *
     * # 说明
     * 该方法返回当前所有成功连接的 MCP 服务器名称，
     * 可用于展示连接状态或让用户选择某个特定服务器。
     */
    pub fn list_mcp_servers(&self) -> Vec<String> {
        self.tools.mcp_manager.server_names()
    }

    #[cfg(feature = "mcp")]
    /**
     * 断开与指定 MCP 服务器的连接
     *
     * # 参数
     * * `name` - 要断开的 MCP 服务器名称
     *
     * # 返回值
     * 若已断开则返回 `Ok(true)`；若未找到该服务器名称则返回 `Ok(false)`；
     * 若传输层未能正常收尾，则返回清理错误。
     *
     * # 说明
     * 该方法断开与指定 MCP 服务器的连接，并移除相关的工具注册。
     * 断开后，该服务器提供的工具将不再可用。
     */
    pub async fn disconnect_mcp(&mut self, name: &str) -> Result<bool> {
        let previous_client = self.tools.mcp_manager.get_client(name);
        let tool_names = previous_client
            .as_ref()
            .map(|client| {
                client
                    .tools()
                    .iter()
                    .map(|tool| crate::mcp::McpToolAdapter::exposed_name_for(name, &tool.name))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let result = self.tools.mcp_manager.disconnect(name).await;
        // 即使清理失败，管理器依然会撤下该客户端，所以旧的投影会
        // 引用一个已不属于当前拓扑的客户端；即使返回 Err，也要尽力
        // 撤销这些投影。
        let withdrawn = result.is_err()
            && !self
                .tools
                .mcp_manager
                .get_client(name)
                .is_some_and(|current| {
                    previous_client
                        .as_ref()
                        .is_some_and(|previous| Arc::ptr_eq(&current, previous))
                });
        match (result, withdrawn) {
            (Ok(disconnected), _) => {
                for tool_name in &tool_names {
                    self.remove_tool(tool_name);
                }
                self.sync_mcp_resource_tools();
                self.setup_hook_mcp_executor().await;
                Ok(disconnected)
            }
            (Err(error), true) => {
                for tool_name in &tool_names {
                    self.remove_tool(tool_name);
                }
                self.sync_mcp_resource_tools();
                self.setup_hook_mcp_executor().await;
                tracing::warn!(
                    agent = %self.config.agent_name,
                    server = %name,
                    error = %error,
                    "MCP target was withdrawn after cleanup failure; stale projections revoked"
                );
                Err(error)
            }
            (Err(error), false) => Err(error),
        }
    }

    /// 断开一个带所有权限定的 MCP 服务器，并仅撤销其已暴露的工具投影。
    #[cfg(feature = "mcp")]
    pub async fn disconnect_mcp_owned(&mut self, id: &McpServerId) -> Result<bool> {
        let previous_client = self.tools.mcp_manager.get_client_by_id(id);
        let tool_names = previous_client
            .as_ref()
            .map(|client| {
                client
                    .tools()
                    .iter()
                    .map(|tool| {
                        crate::mcp::McpToolAdapter::exposed_name_for_identity(id, &tool.name)
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let result = self.tools.mcp_manager.disconnect_server(id).await;
        let withdrawn = result.is_err()
            && previous_client.is_some()
            && !self
                .tools
                .mcp_manager
                .get_client_by_id(id)
                .is_some_and(|current| {
                    previous_client
                        .as_ref()
                        .is_some_and(|previous| Arc::ptr_eq(&current, previous))
                });
        match (result, withdrawn) {
            (Ok(disconnected), _) => {
                for tool_name in &tool_names {
                    self.remove_tool(tool_name);
                }
                self.sync_mcp_resource_tools();
                self.setup_hook_mcp_executor().await;
                Ok(disconnected)
            }
            (Err(error), true) => {
                for tool_name in &tool_names {
                    self.remove_tool(tool_name);
                }
                self.sync_mcp_resource_tools();
                self.setup_hook_mcp_executor().await;
                Err(error)
            }
            (Err(error), false) => Err(error),
        }
    }

    /// 重试已撤下的 MCP 清理操作，且不会影响更新的活跃客户端。
    #[cfg(feature = "mcp")]
    pub async fn retry_mcp_cleanup_debt(&self, name: &str) -> Result<()> {
        self.tools.mcp_manager.retry_cleanup_debt(name).await
    }

    /// 重试一个带所有权限定的 MCP 服务器的清理操作。
    #[cfg(feature = "mcp")]
    pub async fn retry_mcp_cleanup_owned(&self, id: &McpServerId) -> Result<()> {
        self.tools.mcp_manager.retry_cleanup_server(id).await
    }

    // ── 系统提示词 ────────────────────────────────────────────────────────

    /**
     * 设置 Agent 的系统提示词
     *
     * # 参数
     * * `prompt` - 新的系统提示词
     *
     * # 说明
     * 该方法更新 Agent 的系统提示词，并在上下文管理器中同步该更新。
     * 系统提示词定义了 Agent 的角色、行为准则与任务指令。
     *
     * # 注意
     * 设置系统提示词会覆盖之前所有的系统提示词内容。
     * 若之前有通过技能注入的提示词，也会一并被覆盖。
     */
    pub async fn set_system_prompt(&mut self, prompt: String) {
        self.config.system_prompt = prompt.clone();
        self.memory.context.lock().await.update_system(prompt);
        tracing::info!(
            agent = %self.config.agent_name,
            "System prompt updated"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::AgentCallback;
    use crate::agent::config::AgentConfig;
    use crate::llm::types::Message;
    use crate::trace::RunStore;

    struct IdentifiedCallback {
        id: &'static str,
    }

    impl AgentCallback for IdentifiedCallback {
        fn callback_kind(&self) -> Option<&'static str> {
            Some("IdentifiedCallback")
        }

        fn callback_id(&self) -> Option<&str> {
            Some(self.id)
        }
    }

    #[test]
    fn remove_callbacks_by_type_name_and_id_is_precise() {
        let config = AgentConfig::minimal("test-model", "helper");
        let mut agent = ReactAgent::new(config);

        agent.add_callback(Arc::new(IdentifiedCallback { id: "task-a" }));
        agent.add_callback(Arc::new(IdentifiedCallback { id: "task-b" }));

        agent.remove_callbacks_by_type_name_and_id("IdentifiedCallback", "task-a");

        assert_eq!(agent.config.callbacks.len(), 1);
        assert_eq!(agent.config.callbacks[0].callback_id(), Some("task-b"));
    }

    #[tokio::test]
    async fn manual_compression_is_recorded_in_durable_trace() -> crate::error::Result<()> {
        let config = AgentConfig::minimal("test-model", "helper");
        let mut agent = ReactAgent::new(config);
        let store = Arc::new(crate::trace::InMemoryRunStore::new());
        agent.set_run_store(store.clone());
        let legacy = agent.capture_legacy_external_context();
        let trace_run_id = agent
            .start_legacy_trace_run("manual compress", &legacy)
            .await
            .ok_or_else(|| crate::error::ReactError::Other("trace run missing".to_string()))?;
        agent
            .memory
            .context
            .lock()
            .await
            .push(Message::user("manual compression input".to_string()));

        let _ = agent.force_compress_context().await?;
        let run = store
            .load(&trace_run_id)
            .await?
            .ok_or_else(|| crate::error::ReactError::Other("trace data missing".to_string()))?;
        assert!(run.events.iter().any(|event| matches!(
            event,
            crate::trace::RunEvent::ContextCompression { source, .. } if source == "manual"
        )));
        Ok(())
    }

    #[cfg(feature = "mcp")]
    struct FailingThenOkCloseTransport {
        failures_remaining: std::sync::atomic::AtomicUsize,
    }

    #[cfg(feature = "mcp")]
    impl echo_integration::mcp::transport::McpTransport for FailingThenOkCloseTransport {
        fn send(
            &self,
            request: echo_integration::mcp::types::JsonRpcRequest,
        ) -> futures::future::BoxFuture<
            '_,
            crate::error::Result<echo_integration::mcp::types::JsonRpcResponse>,
        > {
            Box::pin(async move {
                use echo_integration::mcp::types::{
                    InitializeResult, McpTool, ServerCapabilities, ToolsCapability,
                };
                let result: serde_json::Value = if request.method == "initialize" {
                    serde_json::to_value(InitializeResult {
                        protocol_version: echo_integration::mcp::types::MCP_PROTOCOL_VERSION
                            .to_string(),
                        capabilities: ServerCapabilities {
                            tools: Some(ToolsCapability::default()),
                            resources: Some(Default::default()),
                            ..ServerCapabilities::default()
                        },
                        server_info: None,
                        instructions: None,
                    })?
                } else if request.method == "resources/list" {
                    serde_json::json!({"resources": []})
                } else {
                    serde_json::to_value(echo_integration::mcp::types::McpToolsListResult {
                        tools: vec![McpTool {
                            name: "probe".to_string(),
                            title: None,
                            description: Some("test probe tool".to_string()),
                            input_schema: serde_json::json!({"type": "object"}),
                            output_schema: None,
                            icons: Vec::new(),
                            annotations: None,
                            execution: None,
                            meta: None,
                        }],
                        next_cursor: None,
                    })?
                };
                Ok(echo_integration::mcp::types::JsonRpcResponse {
                    jsonrpc: "2.0".to_string(),
                    id: request.id,
                    result: Some(result),
                    error: None,
                })
            })
        }

        fn notify(
            &self,
            _notification: echo_integration::mcp::types::JsonRpcNotification,
        ) -> futures::future::BoxFuture<'_, crate::error::Result<()>> {
            Box::pin(async { Ok(()) })
        }

        fn close(&self) -> futures::future::BoxFuture<'_, crate::error::Result<()>> {
            let fail = self
                .failures_remaining
                .fetch_update(
                    std::sync::atomic::Ordering::AcqRel,
                    std::sync::atomic::Ordering::Acquire,
                    |remaining| remaining.checked_sub(1),
                )
                .is_ok();
            Box::pin(async move {
                if fail {
                    Err(crate::error::ReactError::Other(
                        "injected close failure".to_string(),
                    ))
                } else {
                    Ok(())
                }
            })
        }

        fn notification_rx(
            &self,
        ) -> Option<Arc<dyn echo_integration::mcp::types::JsonRpcNotificationReceiver>> {
            None
        }
    }

    #[cfg(feature = "mcp")]
    async fn failing_close_client(
        server: &str,
        failures: usize,
    ) -> Result<Arc<crate::mcp::McpClient>> {
        let transport: Arc<dyn echo_integration::mcp::transport::McpTransport> =
            Arc::new(FailingThenOkCloseTransport {
                failures_remaining: std::sync::atomic::AtomicUsize::new(failures),
            });
        let preparation = match crate::mcp::McpClient::from_transport(server, transport) {
            Ok(preparation) => preparation,
            Err(error) => {
                if let Some(owner) = error.cleanup_owner() {
                    let _ = owner.retry_cleanup().await;
                }
                return Err(crate::error::ReactError::Other(error.to_string()));
            }
        };
        match preparation.await {
            Ok(client) => Ok(client),
            Err(error) => {
                if let Some(owner) = error.cleanup_owner() {
                    let _ = owner.retry_cleanup().await;
                }
                Err(crate::error::ReactError::Other(error.to_string()))
            }
        }
    }

    #[cfg(feature = "mcp")]
    async fn registered_target(
        agent: &mut ReactAgent,
        name: &str,
        client: &Arc<crate::mcp::McpClient>,
    ) -> crate::error::Result<()> {
        let config = crate::mcp::McpServerConfig::stdio(name, "test-command", Vec::<String>::new());
        let receipt = agent
            .tools
            .mcp_manager
            .install_prepared_target(name, config, Arc::clone(client))
            .await?;
        agent.add_tools(receipt.tools);
        agent.sync_mcp_resource_tools();
        agent.setup_hook_mcp_executor().await;
        Ok(())
    }

    #[cfg(feature = "mcp")]
    async fn registered_owned_target(
        agent: &mut ReactAgent,
        id: &crate::mcp::McpServerId,
        client: &Arc<crate::mcp::McpClient>,
    ) -> crate::error::Result<()> {
        let config = crate::mcp::McpServerConfig::stdio(
            &id.local_name,
            "test-command",
            Vec::<String>::new(),
        );
        let receipt = agent
            .tools
            .mcp_manager
            .install_prepared_server(id.clone(), config, Arc::clone(client))
            .await?;
        agent.add_tools(receipt.tools);
        agent.sync_mcp_resource_tools();
        agent.setup_hook_mcp_executor().await;
        Ok(())
    }

    #[cfg(feature = "mcp")]
    async fn assert_mcp_projections(agent: &ReactAgent, server: &str, connected: bool) {
        use crate::skills::hooks::{HookAction, HookEvent, HookRule, HooksDefinition};
        let names = agent.tools.tool_manager.list_tools();
        for name in crate::mcp::MCP_RESOURCE_TOOL_NAMES {
            assert_eq!(names.iter().any(|candidate| candidate == name), connected);
        }
        let mut definition = HooksDefinition::default();
        definition.add_rules(
            HookEvent::PreToolUse,
            vec![HookRule {
                matcher: "probe".into(),
                hooks: vec![HookAction::McpTool {
                    server: server.into(),
                    tool: "probe".into(),
                    arguments: None,
                    timeout: 1,
                }],
            }],
        );
        let mut registry = agent.tools.hook_registry.write().await;
        registry.register_user_hooks(definition);
        let result = registry
            .run_pre_tool_use("probe", &serde_json::json!({}), "")
            .await;
        assert_eq!(
            result
                .messages
                .iter()
                .any(|message| message.contains("server is not connected")),
            !connected
        );
    }

    #[cfg(all(feature = "mcp", unix))]
    #[tokio::test]
    async fn cleanup_failure_withdrawn_target_revokes_stale_projections() -> crate::error::Result<()>
    {
        let mut agent = ReactAgent::new(AgentConfig::minimal("test-model", "helper"));
        let old = failing_close_client("context", 1).await?;
        registered_target(&mut agent, "context", &old).await?;
        let exposed = crate::mcp::McpToolAdapter::exposed_name_for("context", "probe");
        assert!(agent.tools.tool_manager.list_tools().contains(&exposed));
        assert_mcp_projections(&agent, "context", true).await;

        // 使用不同的配置进行替换（这样管理器会准备一次真正的替换），
        // 且旧客户端关闭失败：目标被撤下、返回错误，但过期的工具
        // 投影必须被撤销。
        let mut different = replacement_config("context");
        different.transport = crate::mcp::TransportConfig::Stdio {
            command: "/bin/sh".to_string(),
            args: vec![
                "-c".to_string(),
                "while IFS= read -r line; do id=$(printf '%s' \"$line\" | sed -n 's/.*\"id\": *\\([0-9]*\\).*/\\1/p'); if [ -n \"$id\" ]; then printf '{\"jsonrpc\":\"2.0\",\"id\":%s,\"result\":{\"protocolVersion\":\"2025-11-25\",\"capabilities\":{\"tools\":{}},\"serverInfo\":{\"name\":\"t\",\"version\":\"1\"},\"tools\":[{\"name\":\"replacement\",\"inputSchema\":{\"type\":\"object\"}}]}}\\n' \"$id\"; fi; done".to_string(),
            ],
            env: Vec::new(),
            cwd: None,
        };
        assert!(
            agent
                .reconcile_mcp_target("context", Some(different))
                .await
                .is_err()
        );
        assert!(agent.tools.mcp_manager.get_client("context").is_none());
        assert!(!agent.tools.tool_manager.list_tools().contains(&exposed));
        assert_mcp_projections(&agent, "context", false).await;

        // 关闭失败时进行断开操作，同样会撤销投影。
        let again = failing_close_client("context", 1).await?;
        registered_target(&mut agent, "context", &again).await?;
        assert!(agent.tools.tool_manager.list_tools().contains(&exposed));
        assert_mcp_projections(&agent, "context", true).await;
        assert!(agent.disconnect_mcp("context").await.is_err());
        assert!(!agent.tools.tool_manager.list_tools().contains(&exposed));
        Ok(())
    }

    #[cfg(all(feature = "mcp", unix))]
    #[tokio::test]
    async fn owned_cleanup_failures_revoke_only_the_withdrawn_projections()
    -> crate::error::Result<()> {
        let mut agent = ReactAgent::new(AgentConfig::minimal("test-model", "helper"));
        let id = crate::mcp::McpServerId::plugin("plugin-a", "context");
        let selector = id.selector();
        let old = failing_close_client("context", 1).await?;
        registered_owned_target(&mut agent, &id, &old).await?;
        let exposed = crate::mcp::McpToolAdapter::exposed_name_for_identity(&id, "probe");
        assert!(agent.tools.tool_manager.list_tools().contains(&exposed));
        assert_mcp_projections(&agent, &selector, true).await;

        let mut different = replacement_config("context");
        different.transport = crate::mcp::TransportConfig::Stdio {
            command: "/bin/sh".to_string(),
            args: vec![
                "-c".to_string(),
                "while IFS= read -r line; do id=$(printf '%s' \"$line\" | sed -n 's/.*\"id\": *\\([0-9]*\\).*/\\1/p'); if [ -n \"$id\" ]; then printf '{\"jsonrpc\":\"2.0\",\"id\":%s,\"result\":{\"protocolVersion\":\"2025-11-25\",\"capabilities\":{\"tools\":{}},\"serverInfo\":{\"name\":\"t\",\"version\":\"1\"},\"tools\":[{\"name\":\"replacement\",\"inputSchema\":{\"type\":\"object\"}}]}}\\n' \"$id\"; fi; done".to_string(),
            ],
            env: Vec::new(),
            cwd: None,
        };
        assert!(
            agent
                .connect_mcp_owned(id.clone(), different)
                .await
                .is_err()
        );
        assert!(agent.tools.mcp_manager.get_client_by_id(&id).is_none());
        assert!(!agent.tools.tool_manager.list_tools().contains(&exposed));
        assert_mcp_projections(&agent, &selector, false).await;

        let again = failing_close_client("context", 1).await?;
        registered_owned_target(&mut agent, &id, &again).await?;
        assert!(agent.tools.tool_manager.list_tools().contains(&exposed));
        assert_mcp_projections(&agent, &selector, true).await;
        assert!(agent.disconnect_mcp_owned(&id).await.is_err());
        assert!(!agent.tools.tool_manager.list_tools().contains(&exposed));
        assert_mcp_projections(&agent, &selector, false).await;
        assert!(agent.retry_mcp_cleanup_owned(&id).await.is_ok());
        Ok(())
    }

    #[cfg(feature = "mcp")]
    #[tokio::test]
    async fn cleanup_debt_retry_preserves_new_active_target_projections() -> crate::error::Result<()>
    {
        let mut agent = ReactAgent::new(AgentConfig::minimal("test-model", "helper"));
        let active = failing_close_client("shared", 0).await?;
        registered_target(&mut agent, "shared", &active).await?;
        let exposed = crate::mcp::McpToolAdapter::exposed_name_for("shared", "probe");
        assert!(agent.tools.tool_manager.list_tools().contains(&exposed));

        let withdrawn = failing_close_client("shared", 1).await?;
        let config = replacement_config("shared");
        let unpolled = agent
            .tools
            .mcp_manager
            .install_prepared_target("shared", config, withdrawn);
        drop(unpolled);

        assert!(agent.retry_mcp_cleanup_debt("shared").await.is_err());
        assert!(
            agent
                .mcp_client("shared")
                .is_some_and(|client| Arc::ptr_eq(&client, &active))
        );
        assert!(agent.tools.tool_manager.list_tools().contains(&exposed));

        agent.retry_mcp_cleanup_debt("shared").await?;
        assert!(
            agent
                .mcp_client("shared")
                .is_some_and(|client| Arc::ptr_eq(&client, &active))
        );
        assert!(agent.tools.tool_manager.list_tools().contains(&exposed));
        Ok(())
    }

    #[cfg(feature = "mcp")]
    #[tokio::test]
    async fn agent_close_keeps_failed_target_projections_until_retry_settles()
    -> crate::error::Result<()> {
        let mut agent = ReactAgent::new(AgentConfig::minimal("test-model", "helper"));
        let client = failing_close_client("context", 1).await?;
        registered_target(&mut agent, "context", &client).await?;
        let exposed = crate::mcp::McpToolAdapter::exposed_name_for("context", "probe");
        assert!(agent.tools.tool_manager.list_tools().contains(&exposed));
        assert_mcp_projections(&agent, "context", true).await;

        assert!(crate::agent::Agent::close(&agent).await.is_err());
        let retained = agent
            .tools
            .mcp_manager
            .get_client("context")
            .ok_or_else(|| crate::error::ReactError::Other("failed target was withdrawn".into()))?;
        assert!(Arc::ptr_eq(&retained, &client));
        assert!(agent.tools.tool_manager.list_tools().contains(&exposed));
        assert_mcp_projections(&agent, "context", true).await;

        crate::agent::Agent::close(&agent).await?;
        assert!(agent.tools.mcp_manager.get_client("context").is_none());
        assert!(!agent.tools.tool_manager.list_tools().contains(&exposed));
        assert_mcp_projections(&agent, "context", false).await;
        Ok(())
    }

    #[cfg(feature = "mcp")]
    fn replacement_config(name: &str) -> crate::mcp::McpServerConfig {
        crate::mcp::McpServerConfig::stdio(name, "test-command", Vec::<String>::new())
    }

    #[cfg(feature = "mcp")]
    #[tokio::test]
    async fn preparation_failure_keeps_last_known_good_projections() -> crate::error::Result<()> {
        let mut agent = ReactAgent::new(AgentConfig::minimal("test-model", "helper"));
        let existing = failing_close_client("context", 1).await?;
        registered_target(&mut agent, "context", &existing).await?;
        let exposed = crate::mcp::McpToolAdapter::exposed_name_for("context", "probe");
        assert!(agent.tools.tool_manager.list_tools().contains(&exposed));
        assert_mcp_projections(&agent, "context", true).await;

        // 一次准备失败的替换（stdio 命令为空）应保持旧目标及其
        // 投影不受影响。
        let mut broken = replacement_config("context");
        broken.transport = crate::mcp::TransportConfig::Stdio {
            command: "   ".to_string(),
            args: Vec::new(),
            env: Default::default(),
            cwd: None,
        };
        assert!(
            agent
                .reconcile_mcp_target("context", Some(broken))
                .await
                .is_err()
        );
        assert!(agent.tools.tool_manager.list_tools().contains(&exposed));
        assert_mcp_projections(&agent, "context", true).await;
        Ok(())
    }
}
