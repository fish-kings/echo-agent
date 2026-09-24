//! Agent 构建器

use crate::agent::config::DEFAULT_TOKEN_LIMIT;
use crate::agent::react::run::pipeline::ToolExecutionPipeline;
use crate::agent::{Agent, AgentCallback, AgentConfig, InterventionCallback};
use crate::audit::{AuditLogger, DiagnosticDeliveryObserver};
use crate::error::Result;
use crate::guard::{Guard, GuardManager};
#[cfg(feature = "human-loop")]
use crate::human_loop::{HumanLoopProvider, PermissionService};
use crate::llm::{LlmClient, LlmConfig, ResponseFormat};
use crate::memory::snapshot::{SnapshotManager, SnapshotPolicy};
use crate::memory::store::Store;
use crate::prelude::ReactAgent;
use crate::sandbox::SandboxManager;
use crate::tools::{Tool, ToolExecutionConfig};
use crate::trace::RunStore;
use echo_core::circuit_breaker::CircuitBreakerConfig;
use std::sync::Arc;

/**
 * Agent 构建器。
 *
 * 提供流式 API 来配置并构建一个 Agent。
 * 通过泛型参数指定具体类型，可返回 `Box<dyn Agent>` 抽象。
 */
pub struct ReactAgentBuilder {
    /// Agent 名称。
    name: String,
    /// 模型名称。
    model: String,
    /// 系统提示词。
    system_prompt: String,
    /// 自定义 LLM 客户端（依赖注入）。
    llm_client: Option<Arc<dyn LlmClient>>,
    /// LLM 配置（依赖注入）。
    llm_config: Option<LlmConfig>,
    /// 待注册的自定义工具集合。
    tools: Vec<Box<dyn Tool>>,
    /// 替代内存态任务工具的修订服务（revision service）。
    task_revision_service: Option<Arc<crate::tasks::TaskRevisionService>>,
    /// 是否启用内置工具。
    enable_builtin_tools: bool,
    /// 是否将已注册工具限制为只读。
    readonly_tools: bool,
    /// 是否启用长期记忆。
    enable_memory: bool,
    /// 共享的后台命令 cell 注册表。
    command_cells: Option<Arc<dyn echo_core::tools::cell::CommandCellRegistry>>,
    /// 是否启用人机协同（human-in-the-loop）。
    enable_human_in_loop: bool,
    /// 是否启用子 Agent 派发。
    enable_subagent: bool,
    /// 是否注册可被 LLM 调用的 `agent_tool` 派发工具。
    register_agent_dispatch_tool: bool,
    /// 是否注册内建的 Subagent 通信工具（`subagent_message` + `subagent_list`）。
    #[cfg(feature = "subagent")]
    register_subagent_message_tools: bool,
    /// 供 Fork 派发使用的、可选的应用层自有隔离 Provider。
    #[cfg(feature = "subagent")]
    subagent_isolation_provider:
        Option<std::sync::Arc<dyn crate::agent::subagent::IsolationProvider>>,
    /// 该 agent 创建的每次子代理调用所使用的提示词编译器（compiler）。
    #[cfg(feature = "subagent")]
    subagent_prompt_compiler:
        Option<std::sync::Arc<dyn crate::agent::subagent::SubagentPromptCompiler>>,
    /**
     * 由父级 Agent 与具备委派能力的 Subagent 共享的可选注册表。每个 Agent
     * 仍保留各自独立的执行器/工具表面，同时都通过这一份注册表来解析
     * 派发目标。
     */
    #[cfg(feature = "subagent")]
    subagent_registry: Option<std::sync::Arc<crate::agent::subagent::SubagentRegistry>>,
    /// 是否启用思维链（chain-of-thought）引导。
    enable_cot: bool,
    /// 是否将工具错误反馈给模型。
    tool_error_feedback: bool,
    /// 工具执行配置。
    tool_execution: ToolExecutionConfig,
    /// 最大迭代次数。
    max_iterations: usize,
    /// 一次性迭代收尾（wind-down）策略以及 token 预算。
    run_budget: echo_core::agent::RunBudgetPolicy,
    /// 对话记录持久化安全点的总预算时间。
    persistence_settlement_timeout: std::time::Duration,
    /// 已解析好的 provider/model 档案（profile）。
    model_profile: Option<echo_core::llm::capabilities::ModelProfile>,
    /// token 限制。
    token_limit: usize,
    /// 是否显式设置过 `token_limit`（用于决定是否覆盖默认值）。
    token_limit_explicit: bool,
    /// LLM 生成的最大输出 token 数。
    max_tokens: Option<u32>,
    /// LLM 生成的采样温度。
    temperature: Option<f32>,
    /// 已注册的观测型回调。
    callbacks: Vec<Arc<dyn AgentCallback>>,
    /// 长期记忆 Store。
    store: Option<Arc<dyn Store>>,
    /// session_id（会话/线程标识）。
    session_id: Option<String>,
    /// conversation_id（历史记录投影标识）。
    conversation_id: Option<String>,
    /**
     * 绑定到 Session 的工作目录（worktree 路径）。在构建时会传播到
     * `AgentConfig.working_dir`，再进一步注入每个工具的 ToolContext 中。
     */
    working_dir: Option<std::path::PathBuf>,
    /// 用于项目指令发现的边界目录。
    project_root: Option<std::path::PathBuf>,
    /// 审批 Provider。
    #[cfg(feature = "human-loop")]
    approval_provider: Option<Arc<dyn HumanLoopProvider>>,
    /// 统一权限服务。
    #[cfg(feature = "human-loop")]
    permission_service: Option<Arc<PermissionService>>,
    /// 已注册的防护规则（guard）。
    guards: Vec<Arc<dyn Guard>>,
    /// 审计日志记录器。
    audit_logger: Option<Arc<dyn AuditLogger>>,
    /// Trace 与 Audit 持久化失败情况的观察者。
    diagnostic_delivery_observer: Option<Arc<dyn DiagnosticDeliveryObserver>>,
    /// 状态快照策略。
    snapshot_policy: Option<SnapshotPolicy>,
    /// 最大快照保留数量。
    max_snapshots: usize,
    /// 结构化响应格式。
    response_format: Option<ResponseFormat>,
    /// 单次工具输出的最大 token 数量。
    max_tool_output_tokens: Option<usize>,
    /// 超大工具输出的 artifact 存储配置。
    tool_output_artifacts: Option<echo_core::tools::artifact::ToolOutputArtifactConfig>,
    /// 熔断器（circuit breaker）配置。
    circuit_breaker_config: Option<CircuitBreakerConfig>,
    /// 沙箱管理器，为 skill 脚本执行提供安全隔离。
    sandbox_manager: Option<Arc<SandboxManager>>,
    /// 用于持久化执行轨迹的 RunStore。
    run_store: Option<Arc<dyn RunStore>>,
    /// 自定义的工具执行流水线。
    tool_execution_pipeline: Option<Arc<ToolExecutionPipeline>>,
    /**
     * 能够影响 agent 行为的干预回调（拦截工具调用、注入上下文、重定向
     * 执行、取消运行）。
     */
    intervention_callbacks: Vec<Arc<dyn InterventionCallback>>,
    /// React 循环的检查点（checkpoint）间隔（0 = 仅在结束时保存，N = 每 N 次迭代保存一次）。
    react_checkpoint_interval: usize,
    /// 用于 ReAct 之前意图分类与路由的可选意图路由器。
    intent_router: Option<crate::intent::IntentRouter>,
    /// 用于检查点持久化的可选运行时状态存储。
    state_store: Option<Arc<dyn crate::state::RuntimeStateStore>>,
    /// 已提交对话记录投影的存储。
    conversation_store: Option<Arc<dyn crate::memory::ConversationStore>>,
    /// 用于主动压缩工具调用轨迹的可选可见范围（visibility horizon）配置。
    visibility_horizon: Option<echo_state::compression::horizon::VisibilityHorizonConfig>,
}

impl Default for ReactAgentBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ReactAgentBuilder {
    /// 创建一个新的构建器（默认 ReAct 模式）
    pub fn new() -> Self {
        Self {
            name: "assistant".to_string(),
            model: String::new(),
            system_prompt: crate::config::DEFAULT_AGENT_SYSTEM_PROMPT.to_string(),
            llm_client: None,
            llm_config: None,
            tools: Vec::new(),
            task_revision_service: None,
            enable_builtin_tools: false,
            readonly_tools: false,
            enable_memory: false,
            command_cells: None,
            enable_human_in_loop: false,
            enable_subagent: false,
            register_agent_dispatch_tool: false,
            #[cfg(feature = "subagent")]
            register_subagent_message_tools: false,
            #[cfg(feature = "subagent")]
            subagent_isolation_provider: None,
            #[cfg(feature = "subagent")]
            subagent_prompt_compiler: None,
            #[cfg(feature = "subagent")]
            subagent_registry: None,
            enable_cot: true,
            tool_error_feedback: true,
            tool_execution: ToolExecutionConfig::default(),
            max_iterations: 10,
            run_budget: echo_core::agent::RunBudgetPolicy::default(),
            persistence_settlement_timeout: std::time::Duration::from_secs(10),
            model_profile: None,
            token_limit: DEFAULT_TOKEN_LIMIT,
            token_limit_explicit: false,
            max_tokens: None,
            temperature: None,
            callbacks: Vec::new(),
            store: None,
            session_id: None,
            conversation_id: None,
            working_dir: None,
            project_root: None,
            #[cfg(feature = "human-loop")]
            approval_provider: None,
            #[cfg(feature = "human-loop")]
            permission_service: None,
            guards: Vec::new(),
            audit_logger: None,
            diagnostic_delivery_observer: None,
            snapshot_policy: None,
            max_snapshots: 10,
            response_format: None,
            max_tool_output_tokens: None,
            tool_output_artifacts: None,
            circuit_breaker_config: None,
            sandbox_manager: None,
            run_store: None,
            tool_execution_pipeline: None,
            intervention_callbacks: Vec::new(),
            react_checkpoint_interval: 0,
            intent_router: None,
            state_store: None,
            conversation_store: None,
            visibility_horizon: None,
        }
    }

    // ── 预设配置 ───────────────────────────────────────────────────

    /**
     * 创建一个简单的对话 Agent（无工具、无记忆）
     *
     * 适用于简单的问答场景。
     */
    pub fn simple(model: &str, system_prompt: &str) -> Result<ReactAgent> {
        Self::new()
            .model(model)
            .system_prompt(system_prompt)
            .build_strict()
    }

    /**
     * 创建一个标准 Agent（启用工具与思维链）
     *
     * 适用于大多数 Agent 场景。
     */
    pub fn standard(model: &str, name: &str, system_prompt: &str) -> Result<ReactAgent> {
        Self::new()
            .model(model)
            .name(name)
            .system_prompt(system_prompt)
            .enable_tools()
            .build_strict()
    }

    /**
     * 创建一个全功能 Agent（工具、记忆、规划）
     *
     * 适用于复杂的自主 Agent 场景。
     */
    pub fn full_featured(model: &str, name: &str, system_prompt: &str) -> Result<ReactAgent> {
        Self::new()
            .model(model)
            .name(name)
            .system_prompt(system_prompt)
            .enable_tools()
            .enable_memory()
            .build_strict()
    }
    // ── 基础配置 ─────────────────────────────────────────────────────

    /// 设置 Agent 名称
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// 设置模型名称
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// 设置系统提示词
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = prompt.into();
        self
    }

    // ── LLM 配置 ───────────────────────────────────────────────────────

    /**
     * 设置自定义 LLM 客户端
     *
     * 使用该方法可以：
     * - 注入一个用于测试的 Mock 客户端
     * - 使用自定义的 LLM 实现
     * - 共享同一个 LLM 客户端实例
     */
    pub fn llm_client(mut self, client: Arc<dyn LlmClient>) -> Self {
        self.model = client.model_name().to_string();
        self.llm_client = Some(client);
        self
    }

    /**
     * 设置 LLM 配置（依赖注入）
     *
     * 用于动态配置 API 端点、密钥等，而无需依赖环境变量。
     */
    pub fn llm_config(mut self, config: LlmConfig) -> Self {
        self.model = config.model.clone();
        self.llm_config = Some(config);
        self
    }

    // ── 工具配置 ──────────────────────────────────────────────────────

    /// 启用内置工具（通过 `enable_tool` 标志位）
    pub fn enable_tools(mut self) -> Self {
        self.enable_builtin_tools = true;
        self
    }

    /// 禁用内置工具
    pub fn disable_tools(mut self) -> Self {
        self.enable_builtin_tools = false;
        self
    }

    /**
     * 将已注册的工具限制为只读（禁止 shell、禁止文件写入）。
     * 必须与 `enable_tools()` 搭配使用。供只读子代理（subagent）使用，
     * 使只读约束在工具层面强制生效，而不仅仅依赖提示词。
     */
    pub fn readonly_tools(mut self) -> Self {
        self.readonly_tools = true;
        self
    }

    /// 注册单个工具
    pub fn tool(mut self, tool: Box<dyn Tool>) -> Self {
        self.tools.push(tool);
        self
    }

    /// 批量注册工具
    pub fn tools(mut self, tools: Vec<Box<dyn Tool>>) -> Self {
        self.tools.extend(tools);
        self
    }

    /**
     * 用一个基于应用持久化与策略适配器的修订服务（revision service），
     * 替换默认的内存态任务工具。
     */
    pub fn task_revision_service(
        mut self,
        service: Arc<crate::tasks::TaskRevisionService>,
    ) -> Self {
        self.task_revision_service = Some(service);
        self
    }

    // ── 功能开关 ───────────────────────────────────────────────────────────

    /// 启用长期记忆
    pub fn enable_memory(mut self) -> Self {
        self.enable_memory = true;
        self
    }

    /**
     * 共享一个后台命令 cell 注册表（启用 `shell background=true` 以及
     * wait/stop_cell/list_cells 工具）。将同一份注册表注入主 agent 及其
     * 子代理，使它们观察到相同的 cell 集合。
     */
    pub fn command_cells(
        mut self,
        cells: Arc<dyn echo_core::tools::cell::CommandCellRegistry>,
    ) -> Self {
        self.command_cells = Some(cells);
        self
    }

    /// 启用人机协同（human-in-the-loop）
    pub fn enable_human_in_loop(mut self) -> Self {
        self.enable_human_in_loop = true;
        self
    }

    /// 启用子 Agent 派发
    pub fn enable_subagent(mut self) -> Self {
        self.enable_subagent = true;
        self
    }

    /// 提供供 Fork 派发使用的、应用自有的隔离 Provider。
    #[cfg(feature = "subagent")]
    pub fn subagent_isolation_provider(
        mut self,
        provider: std::sync::Arc<dyn crate::agent::subagent::IsolationProvider>,
    ) -> Self {
        self.subagent_isolation_provider = Some(provider);
        self
    }

    /// 提供该 agent 创建的每次子代理调用所使用的提示词编译器（compiler）。
    #[cfg(feature = "subagent")]
    pub fn subagent_prompt_compiler(
        mut self,
        compiler: std::sync::Arc<dyn crate::agent::subagent::SubagentPromptCompiler>,
    ) -> Self {
        self.subagent_prompt_compiler = Some(compiler);
        self
    }

    /**
     * 在父级 Agent 与其 Subagent 之间共享同一份派发目录（dispatch catalog）。
     *
     * 该注册表仍属于框架运行时原语。具体注册哪些定义、哪些角色暴露
     * `agent_tool`，由产品层决定。
     */
    #[cfg(feature = "subagent")]
    pub fn subagent_registry(
        mut self,
        registry: std::sync::Arc<crate::agent::subagent::SubagentRegistry>,
    ) -> Self {
        self.subagent_registry = Some(registry);
        self
    }

    /**
     * 注册可被 LLM 调用的 `agent_tool` 派发工具。
     *
     * 与 `enable_subagent` 相互独立。调用后会注册 `AgentDispatchTool`，
     * 使 LLM 可以调用 `agent_tool`；不调用时（默认情况），LLM 无法调用
     * `agent_tool`，但框架层通过 `delegate_to_agent*` 发起的派发依然可用。
     */
    pub fn register_agent_dispatch_tool(mut self) -> Self {
        self.register_agent_dispatch_tool = true;
        self
    }

    /**
     * 在构建出的 agent 上注册内建的 Subagent 通信工具
     * （`subagent_message` + `subagent_list`）。
     */
    #[cfg(feature = "subagent")]
    pub fn register_subagent_message_tools(mut self) -> Self {
        self.register_subagent_message_tools = true;
        self
    }

    /// 启用思维链（chain-of-thought）引导
    pub fn enable_cot(mut self) -> Self {
        self.enable_cot = true;
        self
    }

    /// 禁用思维链（chain-of-thought）引导
    pub fn disable_cot(mut self) -> Self {
        self.enable_cot = false;
        self
    }

    // ── 结构化输出 ────────────────────────────────────────────────────────

    /**
     * 声明 Agent 的结构化输出类型
     *
     * 会根据 `T` 的 [`JsonSchema`](schemars::JsonSchema) 自动生成
     * `response_format`，配合 [`ReactAgent::execute_typed`] 可直接获得
     * 反序列化后的结果。
     *
     * # 示例
     *
     * ```rust,no_run
     * use echo_agent::prelude::*;
     * use schemars::JsonSchema;
     * use serde::Deserialize;
     *
     * #[derive(Debug, Deserialize, JsonSchema)]
     * struct Person { name: String, age: u32 }
     *
     * # fn main() -> echo_agent::error::Result<()> {
     * let agent = ReactAgentBuilder::new()
     *     .model("qwen3-max")
     *     .output_type::<Person>()
     *     .build()?;
     * # Ok(())
     * # }
     * ```
     */
    pub fn output_type<T: schemars::JsonSchema>(mut self) -> Self {
        let schema_gen = schemars::r#gen::SchemaGenerator::default();
        let root_schema = schema_gen.into_root_schema_for::<T>();
        let schema_value = serde_json::to_value(root_schema).unwrap_or_default();
        let type_name = std::any::type_name::<T>()
            .rsplit("::")
            .next()
            .unwrap_or("output")
            .to_lowercase();
        self.response_format = Some(ResponseFormat::json_schema(type_name, schema_value));
        self
    }

    /// 手动设置响应格式
    pub fn response_format(mut self, fmt: ResponseFormat) -> Self {
        self.response_format = Some(fmt);
        self
    }

    // ── 执行参数 ────────────────────────────────────────────────────

    /**
     * 设置一个非零的最大迭代次数。
     *
     * [`Self::build`] 会拒绝零值，因为无界的自主循环不是有效的框架执行策略。
     */
    pub fn max_iterations(mut self, max: usize) -> Self {
        self.max_iterations = max;
        self
    }

    /// 配置一次性迭代收尾（wind-down）策略以及由 Provider 上报的 token 预算。
    pub fn run_budget(mut self, policy: echo_core::agent::RunBudgetPolicy) -> Self {
        self.run_budget = policy;
        self
    }

    /// 为一次对话记录持久化安全点设置非零的总预算时间。
    pub fn persistence_settlement_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.persistence_settlement_timeout = timeout;
        self
    }

    /// 安装一个已解析好的、用于控制运行时行为的 provider/model 档案（profile）。
    pub fn model_profile(mut self, profile: echo_core::llm::capabilities::ModelProfile) -> Self {
        self.model_profile = Some(profile);
        self
    }

    /// 设置工具错误反馈开关
    pub fn tool_error_feedback(mut self, enabled: bool) -> Self {
        self.tool_error_feedback = enabled;
        self
    }

    /// 设置工具执行配置
    pub fn tool_execution(mut self, config: ToolExecutionConfig) -> Self {
        self.tool_execution = config;
        self
    }

    /// 设置 token 限制
    pub fn token_limit(mut self, limit: usize) -> Self {
        self.token_limit = limit;
        self.token_limit_explicit = true;
        self
    }

    /// 设置 LLM 生成的最大输出 token 数（`None` 表示使用 API 默认值）。
    pub fn max_tokens(mut self, max: Option<u32>) -> Self {
        self.max_tokens = max;
        self
    }

    /// 设置 LLM 生成的采样温度（`None` 表示使用 API 默认值）。
    pub fn temperature(mut self, temp: Option<f32>) -> Self {
        self.temperature = temp;
        self
    }

    /**
     * 设置单次工具输出的最大 token 数量
     *
     * 当配置了 artifact 存储时，超大输出会被持久化为 artifact；未配置时，
     * 则回退为 UTF-8 安全的截断。
     */
    pub fn max_tool_output_tokens(mut self, max: usize) -> Self {
        self.max_tool_output_tokens = Some(max);
        self
    }

    /**
     * 将超大的工具输出完整存储到应用指定的目录下，同时只在模型上下文中
     * 保留一份有界的投影。
     */
    pub fn tool_output_artifacts(
        mut self,
        config: echo_core::tools::artifact::ToolOutputArtifactConfig,
    ) -> Self {
        self.tool_output_artifacts = Some(config);
        self
    }

    // ── 回调与扩展 ─────────────────────────────────────────────────────

    /// 添加回调
    pub fn callback(mut self, callback: Arc<dyn AgentCallback>) -> Self {
        self.callbacks.push(callback);
        self
    }

    /**
     * 添加一个能够影响 agent 行为的干预回调。
     *
     * 干预回调会在工具执行、LLM 推理以及最终回答之前被检查。它们可以
     * 拦截动作、注入上下文、重定向执行、修改工具参数，或取消整次运行。
     *
     * 与仅用于观测的 `AgentCallback` 不同，`InterventionCallback`
     * 可以*影响*agent 的决策。
     *
     * # 示例
     *
     * ```rust,no_run
     * use echo_agent::prelude::*;
     * use echo_agent::agent::InterventionCallback;
     * use echo_agent::agent::InterventionResult;
     * use std::sync::Arc;
     *
     * struct BlockShell;
     *
     * impl InterventionCallback for BlockShell {
     *     fn on_tool_call<'a>(
     *         &'a self,
     *         _agent: &'a str,
     *         tool: &'a str,
     *         _args: &'a serde_json::Value,
     *     ) -> futures::future::BoxFuture<'a, InterventionResult> {
     *         Box::pin(async move {
     *             if tool == "shell" {
     *                 InterventionResult::block("shell is blocked")
     *             } else {
     *                 InterventionResult::allow()
     *             }
     *         })
     *     }
     * }
     *
     * # fn main() -> echo_agent::error::Result<()> {
     * let agent = ReactAgentBuilder::new()
     *     .model("qwen3-max")
     *     .system_prompt("You are a helpful assistant")
     *     .enable_tools()
     *     .intervention_callback(Arc::new(BlockShell))
     *     .build()?;
     * # Ok(())
     * # }
     * ```
     */
    pub fn intervention_callback(mut self, callback: Arc<dyn InterventionCallback>) -> Self {
        self.intervention_callbacks.push(callback);
        self
    }

    /**
     * 设置一个用于 ReAct 之前意图分类与路由的意图路由器。
     *
     * 设置后，agent 会在进入 ReAct 循环之前先对每条用户输入做分类，
     * 可能直接给出答案或激活某个特定 skill，从而实现快捷路径。
     *
     * # 示例
     *
     * ```rust,no_run
     * use echo_agent::prelude::*;
     * use echo_agent::intent::{IntentRouter, IntentRouterConfig};
     *
     * # fn main() -> echo_agent::error::Result<()> {
     * let router = IntentRouter::new(
     *     Box::new(echo_agent::intent::KeywordClassifier::default()),
     *     IntentRouterConfig::default(),
     * );
     * let agent = ReactAgentBuilder::new()
     *     .model("qwen3-max")
     *     .intent_router(router)
     *     .build()?;
     * # Ok(())
     * # }
     * ```
     */
    pub fn intent_router(mut self, router: crate::intent::IntentRouter) -> Self {
        self.intent_router = Some(router);
        self
    }

    /// 设置用于对 agent 执行状态做检查点（checkpoint）的运行时状态存储。
    pub fn state_store(mut self, store: Arc<dyn crate::state::RuntimeStateStore>) -> Self {
        self.state_store = Some(store);
        self
    }

    /**
     * 设置已提交对话记录投影的存储。
     *
     * 持久化投影需要与 [`Self::state_store`] 搭配使用。若两者不完整或
     * 不是原子配对，调用准入阶段会在产生上下文、模型、trace、guard 或
     * 持久化副作用之前就直接拒绝。
     */
    pub fn conversation_store(mut self, store: Arc<dyn crate::memory::ConversationStore>) -> Self {
        self.conversation_store = Some(store);
        self
    }

    /**
     * 设置用于主动压缩工具调用轨迹的可见范围（visibility horizon）。
     *
     * 配置后，在每次 `prepare()` 调用中、主压缩器运行之前，超出活跃
     * 窗口的工具调用/结果对会被自动压缩为符号化摘要。
     */
    pub fn visibility_horizon(
        mut self,
        config: echo_state::compression::horizon::VisibilityHorizonConfig,
    ) -> Self {
        self.visibility_horizon = Some(config);
        self
    }

    /// 设置长期记忆 Store
    pub fn store(mut self, store: Arc<dyn Store>) -> Self {
        self.store = Some(store);
        self
    }

    /**
     * 注入一个外部 Store，并自动注册四个内置工具：remember / recall / search_memory / forget
     *
     * 这是从“拥有一个记忆存储”到“Agent 能够自主使用记忆”的快捷方式，
     * 等价于 `.store(store).enable_memory()`，但支持任意 `Store` 实现
     * （例如 `EmbeddingStore`），无需依赖默认的 `FileStore`。
     *
     * # 示例
     *
     * ```rust,no_run
     * use echo_agent::prelude::*;
     * use std::sync::Arc;
     *
     * # fn main() -> echo_agent::error::Result<()> {
     * let store = Arc::new(InMemoryStore::new());
     * let agent = ReactAgentBuilder::new()
     *     .model("qwen3-max")
     *     .with_memory_tools(store)
     *     .build()?;
     * # Ok(())
     * # }
     * ```
     */
    pub fn with_memory_tools(mut self, store: Arc<dyn Store>) -> Self {
        self.store = Some(store);
        self.enable_memory = true;
        self
    }

    /// 设置 session_id（会话/线程标识）
    pub fn session_id(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    /**
     * 设置 React 循环的检查点（checkpoint）间隔（以迭代次数计）。
     *
     * 大于 0 时，Agent 会在 React 循环过程中每 N 次迭代保存一次检查点。
     * 这样可以支持崩溃恢复：重启后，Agent 会恢复对话历史，并从中断处
     * 继续执行。
     *
     * 默认值：0（仅在执行结束时保存检查点）。
     */
    pub fn react_checkpoint_interval(mut self, interval: usize) -> Self {
        self.react_checkpoint_interval = interval;
        self
    }

    /**
     * 设置 conversation_id（历史记录投影标识）
     *
     * 与 `session_id` 不同，`conversation_id` 仅用于 `ConversationStore`
     * 的对话记录/历史投影；如果启用了对话历史持久化，应显式设置该值。
     */
    pub fn conversation_id(mut self, conversation_id: impl Into<String>) -> Self {
        self.conversation_id = Some(conversation_id.into());
        self
    }

    /**
     * 设置绑定到 Session 的工作目录（worktree 路径）。会传播到
     * `AgentConfig.working_dir`，再由 `ExecuteStage` 注入每次工具调用的
     * `ToolContext` 中，使 shell/file/git 等工具在该 worktree 内运行。
     */
    pub fn working_dir(mut self, working_dir: impl Into<std::path::PathBuf>) -> Self {
        self.working_dir = Some(working_dir.into());
        self
    }

    /// 设置用于项目指令发现的边界目录。
    pub fn project_root(mut self, project_root: impl Into<std::path::PathBuf>) -> Self {
        self.project_root = Some(project_root.into());
        self
    }

    #[cfg(feature = "human-loop")]
    /// 设置审批 Provider
    pub fn approval_provider(mut self, provider: Arc<dyn HumanLoopProvider>) -> Self {
        self.approval_provider = Some(provider);
        self
    }

    #[cfg(feature = "human-loop")]
    /**
     * 设置统一权限服务
     *
     * 一旦设置，该服务就是唯一的权限检查入口。
     */
    pub fn permission_service(mut self, service: Arc<PermissionService>) -> Self {
        self.permission_service = Some(service);
        self
    }

    // ── 防护规则、权限与审计 ─────────────────────────────────────────────────

    /// 添加防护规则（guard）
    pub fn guard(mut self, guard: Arc<dyn Guard>) -> Self {
        self.guards.push(guard);
        self
    }

    /// 批量添加防护规则
    pub fn guards(mut self, guards: Vec<Arc<dyn Guard>>) -> Self {
        self.guards.extend(guards);
        self
    }

    /// 添加内容安全防护规则（PII 检测/脱敏/拒绝）
    #[cfg(feature = "content-guard")]
    pub fn with_content_guard(mut self, mode: echo_core::guard::content::ContentGuardMode) -> Self {
        let guard = echo_core::guard::content::ContentGuard::new(mode);
        self.guards.push(Arc::new(guard));
        self
    }

    /// 设置审计日志记录器
    pub fn audit_logger(mut self, logger: Arc<dyn AuditLogger>) -> Self {
        self.audit_logger = Some(logger);
        self
    }

    /// 观察 Trace 与 Audit 持久化失败情况，但不改变 Agent 的执行结果。
    pub fn diagnostic_delivery_observer(
        mut self,
        observer: Arc<dyn DiagnosticDeliveryObserver>,
    ) -> Self {
        self.diagnostic_delivery_observer = Some(observer);
        self
    }

    // ── 快照配置 ──────────────────────────────────────────────────────

    /**
     * 设置快照策略，启用状态快照功能
     *
     * 启用后，ReAct 循环的每次迭代都可以自动捕获对话历史快照。
     * 出现异常时，可以通过 `agent.rollback(n)` 回滚到此前的已知良好状态。
     */
    pub fn snapshot_policy(mut self, policy: SnapshotPolicy) -> Self {
        self.snapshot_policy = Some(policy);
        self
    }

    /// 设置最大快照保留数量（默认 10）
    pub fn max_snapshots(mut self, max: usize) -> Self {
        self.max_snapshots = max;
        self
    }

    /**
     * 启用熔断器（circuit breaker）
     *
     * 连续 `failure_threshold` 次 LLM 调用失败后熔断打开，等待 `timeout`
     * 后再进入半开（half-open）探测状态。
     */
    pub fn with_circuit_breaker(mut self, config: CircuitBreakerConfig) -> Self {
        self.circuit_breaker_config = Some(config);
        self
    }

    /**
     * 挂载一个用于持久化执行轨迹的 [`RunStore`]。
     *
     * 设置后，每次 agent 调用都会记录一份完整的 [`Run`](crate::trace::Run)，
     * 包含事件、token 用量与耗时信息。
     */
    pub fn with_run_store(mut self, store: Arc<dyn RunStore>) -> Self {
        self.run_store = Some(store);
        self
    }

    /**
     * 设置自定义的 [`ToolExecutionPipeline`]，对工具执行生命周期
     * （钩子、权限、防护规则等）进行精细控制。
     *
     * 未设置时使用标准流水线。
     */
    pub fn tool_execution_pipeline(mut self, pipeline: ToolExecutionPipeline) -> Self {
        self.tool_execution_pipeline = Some(Arc::new(pipeline));
        self
    }

    /// 设置沙箱管理器，为 skill 脚本执行提供安全隔离
    pub fn sandbox_manager(mut self, manager: Arc<SandboxManager>) -> Self {
        self.sandbox_manager = Some(manager);
        self
    }

    // ── 构建 ──────────────────────────────────────────────────────────────────

    /// 构建 ReAct Agent（内部方法）
    pub fn build(self) -> Result<ReactAgent> {
        // ── 构造期校验 ────────────────────────────────────────────
        /* 客户端也可以由应用层的初始化流程后续再注入。在此之前，agent
         * 仍可以对外暴露配置与非 LLM 工具，正常执行路径会返回既有的
         * “缺少客户端”错误。 */
        if self.max_iterations == 0 {
            return Err(crate::error::ConfigError::ConfigFileError(
                "max_iterations must be greater than zero".to_string(),
            )
            .into());
        }
        if self.persistence_settlement_timeout.is_zero() {
            return Err(crate::error::ConfigError::ConfigFileError(
                "persistence_settlement_timeout must be greater than zero".to_string(),
            )
            .into());
        }
        if self.enable_subagent && !self.enable_builtin_tools {
            return Err(crate::error::ConfigError::ConfigFileError(
                "Enabling subagent dispatch (enable_subagent) requires enabling tool calls (enable_builtin_tools)"
                    .to_string(),
            )
            .into());
        }
        /* 在创建 agent 之前，先校验显式提供的 provider 契约是否合法。
         * `ReactAgent::set_llm_config` 出于兼容性考虑保留了一个不会失败的
         * setter，因此如果没有这项检查，非法的端点或协议只会被记录日志，
         * 而构建器仍会返回一个无法执行回合（turn）的 agent。 */
        if let Some(llm_config) = self.llm_config.as_ref() {
            llm_config.build_client()?;
        }

        let mut config = AgentConfig::new(&self.model, &self.name, &self.system_prompt)
            .enable_tool(self.enable_builtin_tools)
            .readonly_tools(self.readonly_tools)
            .enable_memory(self.enable_memory)
            .enable_human_in_loop(self.enable_human_in_loop)
            .enable_subagent(self.enable_subagent)
            .register_agent_dispatch_tool(self.register_agent_dispatch_tool)
            .enable_cot(self.enable_cot)
            .tool_error_feedback(self.tool_error_feedback)
            .tool_execution(self.tool_execution)
            .max_iterations(self.max_iterations)
            .run_budget(self.run_budget)
            .persistence_settlement_timeout(self.persistence_settlement_timeout)
            .max_tokens(self.max_tokens)
            .temperature(self.temperature);
        #[cfg(feature = "subagent")]
        {
            config.register_subagent_message_tools = self.register_subagent_message_tools;
        }
        if let Some(cells) = self.command_cells.clone() {
            config = config.command_cells(cells);
        }
        if self.token_limit_explicit {
            config = config.token_limit(self.token_limit);
        }
        if let Some(profile) = self.model_profile {
            config = config.model_profile(profile);
        }
        if let Some(project_root) = self.project_root {
            config = config.project_root(project_root);
        }

        #[cfg(feature = "subagent")]
        if let Some(provider) = self.subagent_isolation_provider.clone() {
            config = config.subagent_isolation_provider(provider);
        }
        #[cfg(feature = "subagent")]
        if let Some(compiler) = self.subagent_prompt_compiler.clone() {
            config = config.subagent_prompt_compiler(compiler);
        }

        if let Some(fmt) = self.response_format {
            config = config.response_format(fmt);
        }
        if let Some(max) = self.max_tool_output_tokens {
            config = config.max_tool_output_tokens(max);
        }
        if let Some(artifact_config) = self.tool_output_artifacts {
            config = config.tool_output_artifacts(Some(artifact_config));
        }

        for callback in self.callbacks {
            config = config.with_callback(callback);
        }

        if let Some(session_id) = &self.session_id {
            config = config.session_id(session_id);
        }
        if let Some(conversation_id) = &self.conversation_id {
            config = config.conversation_id(conversation_id);
        }
        if let Some(working_dir) = &self.working_dir {
            // Mutex 中毒时恢复毒锁内部数据继续写入, 不 panic 整个 agent。
            *config.working_dir.lock().unwrap_or_else(|e| e.into_inner()) =
                Some(working_dir.clone());
        }
        if config.token_budget_config.enabled {
            config.token_budget_config.build(config.token_limit)?;
        }
        if self.react_checkpoint_interval > 0 {
            config = config.react_checkpoint_interval(self.react_checkpoint_interval);
        }

        /* 当用户通过 with_memory_tools(store) 传入自定义 Store 时，跳过
         * ReactAgent::new() 内部自动的 FileStore 初始化，改为在 build()
         * 阶段手动注入用户提供的 Store。 */
        let has_external_store = self.store.is_some();
        if has_external_store {
            config = config.enable_memory(false);
        }

        #[cfg(feature = "subagent")]
        let mut agent = if let Some(registry) = self.subagent_registry {
            crate::agent::react::ReactAgent::new_with_subagent_registry(config, registry)
        } else {
            crate::agent::react::ReactAgent::new(config)
        };
        #[cfg(not(feature = "subagent"))]
        let mut agent = crate::agent::react::ReactAgent::new(config);

        if let Some(service) = self.task_revision_service {
            crate::tasks::register_task_tools(&mut agent, service);
        }

        if let Some(llm_client) = self.llm_client {
            agent.set_llm_client(llm_client);
        }

        // 注入 LLM 配置
        if let Some(llm_config) = self.llm_config {
            agent.set_llm_config(llm_config);
        }

        // 注册所有自定义工具
        for tool in self.tools {
            agent.add_tool(tool);
        }

        // 设置 Store（同时会注册 remember/recall/search_memory/forget 工具）
        if let Some(store) = self.store {
            agent.set_memory_store(store);
        }

        #[cfg(feature = "human-loop")]
        if let Some(provider) = self.approval_provider {
            agent.set_approval_provider(provider);
        }

        #[cfg(feature = "human-loop")]
        if let Some(service) = self.permission_service {
            agent.set_permission_service(service);
        }

        // 设置防护规则（guardrails）
        if !self.guards.is_empty() {
            agent.set_guard_manager(GuardManager::from_guards(self.guards));
        }

        // 设置审计日志记录器
        if let Some(logger) = self.audit_logger {
            agent.set_audit_logger(logger);
        }
        if let Some(observer) = self.diagnostic_delivery_observer {
            agent.set_diagnostic_delivery_observer(observer);
        }

        // 设置快照管理器
        if let Some(policy) = self.snapshot_policy {
            agent.set_snapshot_manager(SnapshotManager::new(policy, self.max_snapshots));
        }

        // 设置熔断器
        if let Some(cb_config) = self.circuit_breaker_config {
            agent.set_circuit_breaker(cb_config);
        }

        /* 设置沙箱管理器（框架默认值：本地进程包装器）。
         * 需要操作系统级沙箱的应用应注入自己的管理器实现。 */
        let manager = self
            .sandbox_manager
            .unwrap_or_else(|| std::sync::Arc::new(SandboxManager::local_only()));
        agent.set_sandbox_manager(manager);

        // 设置 run store
        if let Some(store) = self.run_store {
            agent.run_store = Some(store);
        }

        // 设置工具执行流水线
        if let Some(pipeline) = self.tool_execution_pipeline {
            agent.tool_execution_pipeline = Some(pipeline);
        }

        // 设置意图路由器
        if let Some(router) = self.intent_router {
            agent.intent_router = Some(router);
        }

        // 设置运行时状态存储
        if let Some(store) = self.state_store {
            agent.memory.state_store = Some(store);
        }
        if let Some(store) = self.conversation_store {
            agent.memory.conversation_store = Some(store);
        }

        // 在 ContextManager 上设置可见范围（visibility horizon）
        if let Some(horizon_config) = self.visibility_horizon {
            let horizon_compressor =
                echo_state::compression::horizon::VisibilityHorizonCompressor::new(horizon_config);
            // Agent 刚刚创建完成 —— 不会有其他任务持有该锁
            if let Ok(mut ctx) = agent.memory.context.try_lock() {
                ctx.set_visibility_horizon(horizon_compressor);
            } else {
                tracing::warn!("Could not acquire ContextManager lock to set visibility horizon");
            }
        }

        // 设置干预回调
        if !self.intervention_callbacks.is_empty() {
            agent.tools.intervention_callbacks = self.intervention_callbacks;
        }

        Ok(agent)
    }

    /**
     * 构建一个可执行的 agent，并拒绝延迟的 LLM 配置。
     *
     * [`Self::build`] 仍然可供那些先组装完框架能力、之后再解析模型的
     * 应用启动流程使用。公开的快速启动/预设入口则使用这个严格版本，
     * 确保不会在缺少可调用 LLM 依赖的情况下创建出 agent。
     */
    pub fn build_strict(self) -> Result<ReactAgent> {
        if self.llm_client.is_none() && self.llm_config.is_none() {
            return Err(crate::error::ConfigError::MissingConfig(
                "llm_client".to_string(),
                "inject an LlmClient or LlmConfig before using build_strict".to_string(),
            )
            .into());
        }
        self.build()
    }

    /**
     * 构建 agent 并将其作为 trait object 返回。
     *
     * 当你需要多态地处理 agent 时（例如通过统一接口传递不同类型的
     * agent，或者把 agent 存放到集合中），这个方法会很有用。
     *
     * ```rust,no_run
     * use echo_agent::agent::Agent;
     * use echo_agent::prelude::ReactAgentBuilder;
     *
     * # async fn run() -> echo_agent::error::Result<()> {
     * let agent: Box<dyn Agent> = ReactAgentBuilder::new()
     *     .model("qwen3-max")
     *     .system_prompt("You are an assistant")
     *     .build_boxed()?;
     * let answer = agent.execute("Hello").await?;
     * # Ok(())
     * # }
     * ```
     */
    pub fn build_boxed(self) -> Result<Box<dyn Agent>> {
        self.build().map(|a| Box::new(a) as Box<dyn Agent>)
    }
}

// ── AgentBuilder trait 实现 ─────────────────────────────────────────

impl echo_core::agent::builder::AgentBuilder for ReactAgentBuilder {
    type Agent = ReactAgent;

    fn model(self, model: impl Into<String>) -> Self {
        self.model(model)
    }

    fn system_prompt(self, prompt: impl Into<String>) -> Self {
        self.system_prompt(prompt)
    }

    fn name(self, name: impl Into<String>) -> Self {
        self.name(name)
    }

    fn max_iterations(self, max: usize) -> Self {
        self.max_iterations(max)
    }

    fn token_limit(self, limit: usize) -> Self {
        self.token_limit(limit)
    }

    fn tool(self, tool: Box<dyn crate::tools::Tool>) -> Self {
        self.tool(tool)
    }

    fn tools(self, tools: Vec<Box<dyn crate::tools::Tool>>) -> Self {
        self.tools(tools)
    }

    fn build(self) -> Result<ReactAgent> {
        self.build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::MockLlmClient;
    use std::sync::Arc;

    #[test]
    fn test_builder_basic() {
        let builder = ReactAgentBuilder::new()
            .name("test-agent")
            .model("qwen3-max")
            .system_prompt("Test");

        assert_eq!(builder.name, "test-agent");
        assert_eq!(builder.model, "qwen3-max");
        assert_eq!(builder.system_prompt, "Test");
    }

    #[test]
    fn test_builder_chaining() {
        let builder = ReactAgentBuilder::new()
            .model("qwen3-max")
            .enable_tools()
            .enable_memory()
            .max_iterations(20);

        assert!(builder.enable_builtin_tools);
        assert!(builder.enable_memory);
        assert_eq!(builder.max_iterations, 20);
    }

    #[test]
    fn rejects_unbounded_zero_iteration_configuration() {
        let result = ReactAgentBuilder::new()
            .model("test-model")
            .max_iterations(0)
            .build();
        assert!(matches!(result, Err(crate::error::ReactError::Config(_))));
    }

    #[test]
    fn test_react_agent_builder() {
        let builder = ReactAgentBuilder::new()
            .model("qwen3-max")
            .system_prompt("Test")
            .enable_tools();

        assert!(builder.enable_builtin_tools);
    }

    #[test]
    fn default_agent_uses_one_task_relation_api() -> std::result::Result<(), String> {
        let agent = ReactAgentBuilder::new()
            .model("test-model")
            .system_prompt("Test")
            .build()
            .map_err(|error| error.to_string())?;
        let names = agent.tool_names();
        for expected in ["task_create", "task_update", "task_list"] {
            assert!(names.iter().any(|name| name == expected));
        }
        assert!(!names.iter().any(|name| name == "todo_write"));
        Ok(())
    }

    #[test]
    fn builder_allows_deferred_llm_configuration() -> std::result::Result<(), String> {
        let agent = ReactAgentBuilder::new()
            .system_prompt("Test")
            .build()
            .map_err(|error| error.to_string())?;

        assert!(agent.model_name().is_empty());
        assert!(agent.llm_client().is_none());
        assert!(agent.llm_config().is_none());
        Ok(())
    }

    #[test]
    fn builder_applies_nonzero_persistence_settlement_timeout() -> std::result::Result<(), String> {
        let timeout = std::time::Duration::from_millis(750);
        let agent = ReactAgentBuilder::new()
            .persistence_settlement_timeout(timeout)
            .build()
            .map_err(|error| error.to_string())?;
        assert_eq!(agent.config().persistence_settlement_timeout, timeout);
        Ok(())
    }

    #[test]
    fn builder_rejects_zero_persistence_settlement_timeout() {
        let result = ReactAgentBuilder::new()
            .persistence_settlement_timeout(std::time::Duration::ZERO)
            .build();
        assert!(matches!(result, Err(crate::error::ReactError::Config(_))));
    }

    #[test]
    fn test_builder_llm_config_syncs_runtime_model_name() -> std::result::Result<(), String> {
        let config = LlmConfig::for_provider(
            "openai",
            "https://api.example.test/v1",
            "sk-demo",
            "gpt-5.5",
            echo_core::llm::LlmApiProtocol::Responses,
        )
        .map_err(|error| error.to_string())?;
        let agent = ReactAgentBuilder::new()
            .llm_config(config)
            .system_prompt("Test")
            .build()
            .map_err(|error| error.to_string())?;

        assert_eq!(agent.config().get_model_name(), "gpt-5.5");
        assert_eq!(
            agent.llm_config().map(|cfg| cfg.model.as_str()),
            Some("gpt-5.5")
        );
        Ok(())
    }

    #[test]
    fn test_builder_llm_client_syncs_runtime_model_name() {
        let agent = ReactAgentBuilder::new()
            .llm_client(Arc::new(
                MockLlmClient::new().with_model_name("mock-topology"),
            ))
            .system_prompt("Test")
            .build()
            .unwrap();

        assert_eq!(agent.config().get_model_name(), "mock-topology");
    }

    #[test]
    fn test_builder_tool_execution_config_is_applied() {
        let agent = ReactAgentBuilder::new()
            .model("qwen3-max")
            .tool_execution(ToolExecutionConfig {
                timeout_ms: 120_000,
                ..ToolExecutionConfig::default()
            })
            .build()
            .unwrap();

        assert_eq!(agent.config().get_tool_execution().timeout_ms, 120_000);
    }
}
