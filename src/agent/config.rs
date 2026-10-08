//! Agent 配置

use crate::agent::AgentCallback;
use crate::llm::ResponseFormat;
use crate::tools::ToolExecutionConfig;
use echo_core::budget::TokenBudgetConfig;
use echo_core::tools::permission::PermissionMode;
use std::path::PathBuf;
use std::sync::Arc;

/// 在没有模型 profile 或显式限制时使用的保守上下文长度上限。
pub const DEFAULT_TOKEN_LIMIT: usize = 128_000;

/** Agent 运行时配置。

通过链式 builder 调用配置各项参数，然后传给 `ReactAgent::new`。
*/
pub struct AgentConfig {
    /// 模型名称。
    pub(crate) model_name: String,
    /// 系统提示词。
    pub(crate) system_prompt: String,
    /// Agent 名称。
    pub(crate) agent_name: String,
    /// 最大迭代轮数，防止无限循环（默认 100，对大多数任务而言相当于不限）。
    pub(crate) max_iterations: usize,
    /// 可选的软性运行预算。所有字段默认禁用。
    pub(crate) run_budget: echo_core::agent::RunBudgetPolicy,
    /// 已解析的 provider/model 能力，会影响 harness 的运行行为。
    pub(crate) model_profile: Option<echo_core::llm::capabilities::ModelProfile>,
    /// 工具白名单（为空表示不限制，所有已注册工具均可调用）。
    pub(crate) allowed_tools: Vec<String>,
    /// 是否允许注册并调用业务工具（例如数学、天气等）。
    pub(crate) enable_tool: bool,
    /** 当 `enable_tool` 为 true 时，仅注册**只读**工具（不含 shell、
    文件写入、git 变更操作）。供只读 subagent 使用，使"只读"在工具层面
    被物理强制，而不仅仅是提示词层面的约束。
    */
    pub(crate) readonly_tools: bool,
    /** 共享的后台命令 cell 注册表。设置后，`ShellTool` 将支持
    `background=true`，并会注册 wait/stop_cell/list_cells 工具。
    应用层在此注入唯一一个注册表，使主 agent 与其 subagent 观察到
    相同的 cell 集合；`None`（默认）表示 cell 功能不可用。
    */
    pub(crate) command_cells:
        Option<std::sync::Arc<dyn echo_core::tools::cell::CommandCellRegistry>>,
    /// 是否启用 human-in-loop 工具。
    pub(crate) enable_human_in_loop: bool,
    /// 是否启用 subagent 派发工具（agent_tool）。
    pub(crate) enable_subagent: bool,
    /** 适用于任意 subagent 派发模式（Sync/Fork/Teammate）的默认超时时间
    （秒）。0 表示不超时。默认 600（10 分钟）——大型项目中的深度任务
    经常会超出此前 5 分钟/不限制的旧上限。单个 subagent 的
    `SubagentDefinition.timeout_secs`（若 > 0）会覆盖此值。
    */
    pub(crate) subagent_timeout_secs: u64,
    /// 供 Fork 派发使用的、应用层自有的可选隔离 provider。
    #[cfg(feature = "subagent")]
    pub(crate) subagent_isolation_provider:
        Option<std::sync::Arc<dyn crate::agent::subagent::IsolationProvider>>,
    /// 由直接工具派发与程序化委派共享的 prompt 编译器。
    #[cfg(feature = "subagent")]
    pub(crate) subagent_prompt_compiler:
        std::sync::Arc<dyn crate::agent::subagent::SubagentPromptCompiler>,
    /** 是否注册可被 LLM 调用的 `agent_tool` 派发工具。

    与 `enable_subagent`（控制 `delegate_to_agent_with_parent_and_cancel`
    所依赖的 subagent 注册表基础设施）相互解耦。当为 `false` 时，LLM
    无法调用 `agent_tool`；但 subagent 派发仍可通过框架的
    `delegate_to_agent*` 方法（TaskRuntime 路径 A）工作。默认：`false`
    （只有产品层确实需要 LLM 驱动的派发逃生通道时，才应显式开启）。
    */
    pub(crate) register_agent_dispatch_tool: bool,
    /** 注册 `subagent_message` + `subagent_list`，使被派发的 Subagent
    能够向其父级汇报/升级问题，并联系同级尝试。与（负责委派的）
    `register_agent_dispatch_tool` 相互解耦，因为上行消息对所有被派发的
    Subagent 都有用，包括那些不能进行委派的角色。默认：`false`
    （由产品层自行开启）。
    */
    #[cfg(feature = "subagent")]
    pub(crate) register_subagent_message_tools: bool,
    /// 上下文 token 上限；超出时自动触发压缩（`usize::MAX` 表示不限制）。
    pub(crate) token_limit: usize,
    /// `token_limit` 是否由调用方显式设置过。
    pub(crate) token_limit_explicit: bool,
    /// 流式事件通道的缓冲区大小（默认 256）。缓冲区满时事件会被丢弃并记录警告。
    pub(crate) stream_buffer_size: usize,
    /// 已注册的 Agent 回调列表。
    pub(crate) callbacks: Vec<Arc<dyn AgentCallback>>,
    /// LLM 调用失败后的最大重试次数（0 = 不重试，默认 3）。
    pub(crate) llm_max_retries: usize,
    /// LLM 重试的初始延迟（毫秒），按指数退避翻倍（默认 500）。
    pub(crate) llm_retry_delay_ms: u64,
    /// 工具执行失败时，将错误反馈给 LLM 而不是直接让 Agent 失败（默认 true）。
    pub(crate) tool_error_feedback: bool,
    /// 启用思维链（CoT）系统提示词注入（默认 false）。
    pub(crate) enable_cot: bool,
    /** 要求文件在被写入/编辑/删除之前必须先被显式读取过（通过
    `read_file`）。启用后，`apply_patch` 等文件变更工具会拒绝那些
    在当前会话轮次中尚未被读取过的路径。（默认 false）
    */
    pub(crate) force_read_before_edit: bool,
    /// Agent 是否处于 plan 模式（仅能使用只读工具）。
    pub(crate) plan_mode: bool,
    /// 推理强度：low（快速）/medium（标准）/high（深入）。
    pub(crate) _reasoning_effort: String,
    /// 工具执行配置：超时、重试策略、并行并发度。
    pub(crate) tool_execution: ToolExecutionConfig,
    /// 是否启用长期记忆 Store（remember/recall/forget 工具 + 自动上下文注入）。
    pub(crate) enable_memory: bool,
    /// 长期记忆 Store 的文件路径（默认 `~/.echo-agent/store.json`）。
    pub(crate) memory_path: String,
    /** 用于进程内/逻辑运行分组的会话标识符。

    运行时崩溃恢复使用的是 `conversation_id` 配合
    [`crate::state::RuntimeStateStore`]；该字段仅是一个进程内标签，
    不驱动任何恢复行为。
    */
    pub(crate) session_id: Option<String>,
    /// 会话（conversation）标识符，供 ConversationStore 持久化事务记录/历史投影时使用。
    pub(crate) conversation_id: Option<String>,
    /// 结构化输出格式（`None` 表示默认文本格式）。
    pub(crate) response_format: Option<ResponseFormat>,
    /** 单次工具输出允许的最大 token 数（`None` 表示不限制）。
    超出时会自动截断，并附加形如 `[Output truncated, N tokens total]` 的提示。
    */
    pub(crate) max_tool_output_tokens: Option<usize>,
    /** 完整工具输出的 artifact 存储策略。应用层可以把通用的临时默认
    目录重定向到一个持久化的、由产品自持的目录。
    */
    pub(crate) tool_output_artifacts:
        std::sync::Mutex<Option<echo_core::tools::artifact::ToolOutputArtifactConfig>>,
    /// LLM 温度参数（0.0–2.0，`None` 表示使用模型默认值）。
    pub(crate) temperature: Option<f32>,
    /// 最大生成 token 数（`None` 表示使用模型默认值）。
    pub(crate) max_tokens: Option<u32>,
    /** 发送给 provider 用于 KVCache 隔离的稳定用户标识符。
    DeepSeek 等使用它来划分 prompt 缓存分区。一个稳定的、会话级的值
    可以带来 90% 以上的缓存命中率；如果没有它，每个请求都会各自单独缓存。
    */
    pub(crate) cache_user_id: Option<String>,
    /// 是否自动加载项目规则文件（`.echo-agent/AGENT.md`），默认 true。
    pub(crate) auto_project_rules: bool,
    /// 项目指令发现的可选显式边界。
    pub(crate) project_root: Option<std::path::PathBuf>,
    /** 工作目录（用于查找项目规则文件），`None` 表示使用当前目录。
    使用 Mutex 是为了在 resume 期间支持通过 `&self` 进行更新（BUG-3）。
    */
    pub(crate) working_dir: std::sync::Mutex<Option<PathBuf>>,
    /// 用于精细化管理上下文窗口的 token 预算配置。
    pub(crate) token_budget_config: TokenBudgetConfig,
    /// 工具执行的权限模式。
    pub(crate) permission_mode: PermissionMode,
    /** React 循环状态设置检查点（checkpoint）的频率（单位：迭代次数）。
    0 = 仅在执行结束时设置检查点（默认）。
    1 = 每次迭代都设置检查点。
    N = 每 N 次迭代设置一次检查点。

    当后台任务在执行过程中崩溃时，Agent 可以从最后一次检查点的迭代
    处恢复，因为对话历史（包括所有工具调用与结果）都被保留了下来。
    */
    pub(crate) react_checkpoint_interval: usize,
    /** 单次事务记录投影落盘安全点所允许的总预算时间。
    应用层可以缩短这个默认值，且不会改变行为的效果标识（effect identity）。
    */
    pub(crate) persistence_settlement_timeout: std::time::Duration,

    /// verifier（Critic）是否启用，用于校验 final_answer。
    pub(crate) verifier_enabled: bool,
    /// verifier 通过所需的最低分数（0.0-10.0）。
    pub(crate) verifier_min_score: f64,
    /// 在接受答案之前，verifier 的最大重试次数。
    pub(crate) verifier_max_retries: usize,
}

impl AgentConfig {
    /** 创建一个新的 Agent 配置。

    # 参数
    * `model_name` - 要使用的 LLM 模型名称（对应配置中的模型标识符）
    * `agent_name` - Agent 名称，用于标识与日志记录
    * `system_prompt` - 系统提示词，定义 Agent 的角色与能力

    # 返回值
    返回一个默认的 AgentConfig 实例；后续可通过链式调用进一步配置。
    */
    pub fn new(model_name: &str, agent_name: &str, system_prompt: &str) -> Self {
        Self {
            model_name: model_name.to_string(),
            system_prompt: system_prompt.to_string(),
            agent_name: agent_name.to_string(),
            max_iterations: 100,
            run_budget: echo_core::agent::RunBudgetPolicy::default(),
            model_profile: None,
            allowed_tools: Vec::new(),
            enable_tool: false,
            readonly_tools: false,
            command_cells: None,
            enable_human_in_loop: false,
            enable_subagent: false,
            subagent_timeout_secs: 600,
            #[cfg(feature = "subagent")]
            subagent_isolation_provider: None,
            #[cfg(feature = "subagent")]
            subagent_prompt_compiler: std::sync::Arc::new(
                crate::agent::subagent::DefaultSubagentPromptCompiler,
            ),
            register_agent_dispatch_tool: false,
            #[cfg(feature = "subagent")]
            register_subagent_message_tools: false,
            token_limit: DEFAULT_TOKEN_LIMIT,
            token_limit_explicit: false,
            stream_buffer_size: 256,
            callbacks: Vec::new(),
            llm_max_retries: 3,
            llm_retry_delay_ms: 500,
            tool_error_feedback: true,
            enable_cot: false,
            force_read_before_edit: false,
            plan_mode: false,
            _reasoning_effort: "medium".to_string(),
            tool_execution: ToolExecutionConfig::default(),
            enable_memory: false,
            memory_path: String::new(),
            session_id: None,
            conversation_id: None,
            response_format: None,
            max_tool_output_tokens: None,
            tool_output_artifacts: std::sync::Mutex::new(Some(
                echo_core::tools::artifact::ToolOutputArtifactConfig::default(),
            )),
            temperature: None,
            max_tokens: None,
            cache_user_id: None,
            auto_project_rules: true,
            project_root: None,
            working_dir: std::sync::Mutex::new(None),
            token_budget_config: TokenBudgetConfig::default(),
            permission_mode: PermissionMode::Default,
            react_checkpoint_interval: 0,
            persistence_settlement_timeout: std::time::Duration::from_secs(10),
            verifier_enabled: false,
            verifier_min_score: 7.0,
            verifier_max_retries: 2,
        }
    }

    // ── 预设配置（易用性优化） ───────────────────────────────────────────

    /** 创建一个最简 Agent（无工具、无记忆）。

    适用于简单对话场景。
    */
    pub fn minimal(model_name: &str, system_prompt: &str) -> Self {
        Self::new(model_name, "assistant", system_prompt)
            .enable_tool(false)
            .enable_memory(false)
            .enable_cot(false)
    }

    /** 创建一个标准 Agent（启用工具 + 思维链）。

    适用于大多数 Agent 场景。
    */
    pub fn standard(model_name: &str, agent_name: &str, system_prompt: &str) -> Self {
        Self::new(model_name, agent_name, system_prompt)
            .enable_tool(true)
            .enable_cot(true)
    }

    /** 创建一个功能完整的 Agent（工具、记忆、规划）。

    适用于复杂的自治 Agent 场景。
    */
    pub fn full_featured(model_name: &str, agent_name: &str, system_prompt: &str) -> Self {
        Self::new(model_name, agent_name, system_prompt)
            .enable_tool(true)
            .enable_memory(true)
            .enable_cot(true)
    }

    /// 启用全部功能（工具、记忆）——builder 链式调用版本。
    pub fn with_full_features(mut self) -> Self {
        self.enable_tool = true;
        self.enable_memory = true;
        self.enable_cot = true;
        self
    }

    /// 启用基础工具功能（工具 + 思维链）——builder 链式调用版本。
    pub fn with_tools(mut self) -> Self {
        self.enable_tool = true;
        self.enable_cot = true;
        self
    }

    // ── 原始 Builder 方法 ─────────────────────────────────────────────────

    /** 启用或禁用工具调用。

    # 参数
    * `enabled` - `true` 启用工具调用，`false` 禁用

    # 说明
    启用后，Agent 可以调用已注册的业务工具（例如数学运算、文件操作等）。
    */
    pub fn enable_tool(mut self, enabled: bool) -> Self {
        self.enable_tool = enabled;
        self
    }

    /** 将已注册工具限制为**只读**（不含 shell、文件写入）。

    仅在同时设置了 `enable_tool(true)` 时才会生效。此时 agent 会获得
    read_file/list_dir/grep/glob/diff/web_search 等工具，但不会获得
    shell/write_file/delete_file/git-commit。供只读 subagent 使用，
    使"只读"在工具层面被强制执行。
    */
    pub fn readonly_tools(mut self, readonly: bool) -> Self {
        self.readonly_tools = readonly;
        self
    }

    /// 检查是否应当只注册只读工具。
    pub fn is_readonly_tools(&self) -> bool {
        self.readonly_tools
    }

    /** 注入一个共享的后台命令 cell 注册表。

    设置后，`ShellTool` 支持 `background=true`（立即返回一个 `cell_id`），
    并会注册 `wait` / `stop_cell` / `list_cells` 工具。在父 agent 与其
    subagent 之间共享同一个注册表，使它们都能观察到相同的 cell 集合。
    */
    pub fn command_cells(
        mut self,
        cells: std::sync::Arc<dyn echo_core::tools::cell::CommandCellRegistry>,
    ) -> Self {
        self.command_cells = Some(cells);
        self
    }

    /** 启用或禁用 Human-in-the-Loop 功能。

    # 参数
    * `enabled` - `true` 启用人工介入，`false` 禁用

    # 说明
    启用后，Agent 在需要审批或确认时可以通过 `human_in_loop` 工具请求人工介入。
    */
    pub fn enable_human_in_loop(mut self, enabled: bool) -> Self {
        self.enable_human_in_loop = enabled;
        self
    }

    /** 启用或禁用 subagent 派发。

    # 参数
    * `enabled` - `true` 启用 subagent 派发，`false` 禁用

    # 说明
    启用后，Agent 可以使用 `agent_tool` 工具派发其他 subagent 执行任务。
    */
    pub fn enable_subagent(mut self, enabled: bool) -> Self {
        self.enable_subagent = enabled;
        self
    }

    /** 设置适用于所有模式（Sync/Fork/Teammate）的默认 subagent 派发超时
    时间（秒）。0 = 不超时。默认 600（10 分钟）。
    */
    pub fn subagent_timeout_secs(mut self, secs: u64) -> Self {
        self.subagent_timeout_secs = secs;
        self
    }

    /// 提供供 Fork 派发使用的、应用层自有的隔离 provider。
    #[cfg(feature = "subagent")]
    pub fn subagent_isolation_provider(
        mut self,
        provider: std::sync::Arc<dyn crate::agent::subagent::IsolationProvider>,
    ) -> Self {
        self.subagent_isolation_provider = Some(provider);
        self
    }

    /// 提供每条 subagent 派发路径都会使用的编译器。
    #[cfg(feature = "subagent")]
    pub fn subagent_prompt_compiler(
        mut self,
        compiler: std::sync::Arc<dyn crate::agent::subagent::SubagentPromptCompiler>,
    ) -> Self {
        self.subagent_prompt_compiler = compiler;
        self
    }

    /** 设置是否注册可被 LLM 调用的 `agent_tool` 派发工具。

    与 `enable_subagent` 相互独立。当为 `true` 时，`AgentDispatchTool`
    会被注册进工具管理器，使 LLM 可以调用 `agent_tool`。当为 `false`
    （默认）时，LLM 无法调用 `agent_tool`，但框架的 `delegate_to_agent*`
    方法仍然可用（供 TaskRuntime 使用）。
    */
    pub fn register_agent_dispatch_tool(mut self, enabled: bool) -> Self {
        self.register_agent_dispatch_tool = enabled;
        self
    }

    /** 注册内置的 Subagent 通信工具
    （`subagent_message` + `subagent_list`）。
    */
    #[cfg(feature = "subagent")]
    pub fn register_subagent_message_tools(mut self, enabled: bool) -> Self {
        self.register_subagent_message_tools = enabled;
        self
    }

    /** 设置工具白名单。

    # 参数
    * `tools` - 允许的工具名称列表

    # 说明
    - 若列表为空，则不做限制，所有已注册工具均可调用
    - 若列表非空，则只有列表中的工具可以被调用
    */
    pub fn allowed_tools(mut self, tools: Vec<String>) -> Self {
        self.allowed_tools.extend(tools);
        self
    }

    /** 获取工具白名单。

    # 返回值
    返回当前允许的工具名称切片。
    */
    pub fn get_allowed_tools(&self) -> &[String] {
        &self.allowed_tools
    }

    /** 检查是否启用了工具调用。

    # 返回值
    启用返回 `true`，禁用返回 `false`。
    */
    pub fn is_tool_enabled(&self) -> bool {
        self.enable_tool
    }

    /** 检查是否启用了 Human-in-the-Loop。

    # 返回值
    启用返回 `true`，禁用返回 `false`。
    */
    pub fn is_human_in_loop_enabled(&self) -> bool {
        self.enable_human_in_loop
    }

    /** 检查是否启用了 subagent 派发。

    # 返回值
    启用返回 `true`，禁用返回 `false`。
    */
    pub fn is_subagent_enabled(&self) -> bool {
        self.enable_subagent
    }

    /// 适用于所有模式的默认 subagent 派发超时时间（秒）。0 = 不超时。
    pub fn get_subagent_timeout_secs(&self) -> u64 {
        self.subagent_timeout_secs
    }

    /** 设置最大迭代轮数。

    # 参数
    * `max_iterations` - 最大迭代次数，防止无限循环（默认 100）

    # 说明
    Agent 在执行期间最多执行指定次数的迭代；超出该限制将终止执行。
    默认值为 100，对大多数任务而言相当于不限制。可以设置为更小的值来实现更严格的控制。
    */
    pub fn max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// 设置用于迭代收尾与模型 token 收敛的软性运行预算。
    pub fn run_budget(mut self, policy: echo_core::agent::RunBudgetPolicy) -> Self {
        self.run_budget = policy;
        self
    }

    /// 设置事务记录持久化安全点所允许的非零总预算时间。
    pub fn persistence_settlement_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.persistence_settlement_timeout = timeout;
        self
    }

    /// 安装一个已解析的 model profile，用于影响 harness 层面的行为。
    pub fn model_profile(mut self, profile: echo_core::llm::capabilities::ModelProfile) -> Self {
        self.model_profile = Some(profile);
        self
    }

    /** 设置 Agent 名称。

    # 参数
    * `agent_name` - Agent 名称，用于标识与日志记录
    */
    pub fn agent_name(mut self, agent_name: &str) -> Self {
        self.agent_name = agent_name.to_string();
        self
    }

    /** 设置 LLM 模型名称。

    # 参数
    * `model_name` - 要使用的 LLM 模型名称（对应配置中的模型标识符）
    */
    pub fn model_name(mut self, model_name: &str) -> Self {
        self.model_name = model_name.to_string();
        self
    }

    /// 在运行时设置模型名称（可变引用版本）。
    pub fn set_model_name(&mut self, model_name: &str) {
        self.model_name = model_name.to_string();
    }

    /** 设置一个稳定的 cache user_id，用于 KVCache 隔离（DeepSeek 等）。
    一个稳定的、会话级的 id 能让 provider 在多次请求之间复用 prompt
    缓存。如果没有它，缓存命中率可能跌至 1% 以下。
    */
    pub fn cache_user_id(mut self, id: impl Into<String>) -> Self {
        self.cache_user_id = Some(id.into());
        self
    }

    /// 在运行时设置 cache_user_id。
    pub fn set_cache_user_id(&mut self, id: impl Into<String>) {
        self.cache_user_id = Some(id.into());
    }

    /// 读取当前的 cache_user_id（访问器方法，区别于 builder 方法）。
    pub fn get_cache_user_id(&self) -> Option<&str> {
        self.cache_user_id.as_deref()
    }

    /** 设置系统提示词。

    # 参数
    * `system_prompt` - 系统提示词，定义 Agent 的角色与能力
    */
    pub fn system_prompt(mut self, system_prompt: &str) -> Self {
        self.system_prompt = system_prompt.to_string();
        self
    }

    /** 设置上下文 token 上限。

    # 参数
    * `limit` - 上下文 token 上限；超出时自动触发压缩（`usize::MAX` 表示不限制）
    */
    pub fn token_limit(mut self, limit: usize) -> Self {
        self.token_limit = limit;
        self.token_limit_explicit = true;
        self
    }

    /** 添加 Agent 回调。

    # 参数
    * `callback` - 实现了 `AgentCallback` trait 的回调实例

    # 说明
    在 Agent 执行过程中触发不同事件时会调用回调，用于监控、日志记录等场景。
    */
    pub fn with_callback(mut self, callback: Arc<dyn AgentCallback>) -> Self {
        self.callbacks.push(callback);
        self
    }

    /** 设置 LLM 调用失败后的最大重试次数。

    # 参数
    * `retries` - 最大重试次数（0 = 不重试，默认 3）
    */
    pub fn llm_max_retries(mut self, retries: usize) -> Self {
        self.llm_max_retries = retries;
        self
    }

    /** 设置 LLM 重试的初始延迟。

    # 参数
    * `delay_ms` - 初始延迟（毫秒），按指数退避翻倍（默认 500）
    */
    pub fn llm_retry_delay_ms(mut self, delay_ms: u64) -> Self {
        self.llm_retry_delay_ms = delay_ms;
        self
    }

    /** 启用或禁用工具错误反馈。

    # 参数
    * `enabled` - `true` 启用工具错误反馈，`false` 禁用

    # 说明
    启用后，工具执行失败时会将错误反馈给 LLM，而不是直接让 Agent 失败。
    */
    pub fn tool_error_feedback(mut self, enabled: bool) -> Self {
        self.tool_error_feedback = enabled;
        self
    }

    /** 获取会话标识符。

    # 返回值
    会话标识符的引用；若未设置则为 `None`

    # 说明
    会话标识符只是一个轻量的运行/线程标签。持久化的运行时恢复使用的
    是 `conversation_id` 配合 `RuntimeStateStore`。
    */
    pub fn get_session_id(&self) -> Option<&str> {
        self.session_id.as_deref()
    }

    /** 获取会话（conversation）标识符。

    # 返回值
    会话标识符的引用；若未设置则为 `None`

    # 说明
    会话标识符用于 `ConversationStore` 的事务记录/历史投影，以及
    `RuntimeStateStore` 的崩溃恢复检查点。它与仅作为运行/线程标签的
    `session_id` 不同。
    */
    pub fn get_conversation_id(&self) -> Option<&str> {
        self.conversation_id.as_deref()
    }

    /** 获取 LLM 调用失败后的最大重试次数。

    # 返回值
    最大重试次数
    */
    pub fn get_llm_max_retries(&self) -> usize {
        self.llm_max_retries
    }

    /** 获取 LLM 重试的初始延迟。

    # 返回值
    初始延迟（毫秒）
    */
    pub fn get_llm_retry_delay_ms(&self) -> u64 {
        self.llm_retry_delay_ms
    }

    /** 获取工具错误反馈的设置。

    # 返回值
    启用返回 `true`，禁用返回 `false`。
    */
    pub fn get_tool_error_feedback(&self) -> bool {
        self.tool_error_feedback
    }

    /** 获取最大迭代轮数。

    # 返回值
    最大迭代次数
    */
    pub fn get_max_iterations(&self) -> usize {
        self.max_iterations
    }

    /** 获取上下文 token 上限。

    # 返回值
    上下文 token 上限，`usize::MAX` 表示不限制。
    */
    pub fn get_token_limit(&self) -> usize {
        self.token_limit
    }

    /** 检查是否启用了思维链（CoT）。

    # 返回值
    启用返回 `true`，禁用返回 `false`。
    */
    pub fn is_cot_enabled(&self) -> bool {
        self.enable_cot
    }

    /** 检查是否启用了长期记忆。

    # 返回值
    启用返回 `true`，禁用返回 `false`。
    */
    pub fn is_memory_enabled(&self) -> bool {
        self.enable_memory
    }

    /** 获取长期记忆 store 的文件路径。

    # 返回值
    长期记忆 store 的文件路径
    */
    pub fn get_memory_path(&self) -> &str {
        &self.memory_path
    }

    /** 获取工具执行配置。

    # 返回值
    工具执行配置的引用（包含超时、重试策略、并行并发度等）
    */
    pub fn get_tool_execution(&self) -> &crate::tools::ToolExecutionConfig {
        &self.tool_execution
    }

    /** 获取结构化输出格式。

    # 返回值
    结构化输出格式的引用；若未设置则为 `None`
    */
    pub fn get_response_format(&self) -> Option<&crate::llm::ResponseFormat> {
        self.response_format.as_ref()
    }

    /** 获取 LLM 模型名称。

    # 返回值
    LLM 模型名称
    */
    pub fn get_model_name(&self) -> &str {
        &self.model_name
    }

    /** 获取系统提示词。

    # 返回值
    系统提示词
    */
    pub fn get_system_prompt(&self) -> &str {
        &self.system_prompt
    }

    /** 获取 Agent 名称。

    # 返回值
    Agent 名称
    */
    pub fn get_agent_name(&self) -> &str {
        &self.agent_name
    }

    /** 启用或禁用思维链（CoT）。

    # 参数
    * `enabled` - `true` 启用 CoT，`false` 禁用

    # 说明
    启用 CoT 后，Agent 会向系统提示词中注入与 CoT 相关的指令。
    */
    pub fn enable_cot(mut self, enabled: bool) -> Self {
        self.enable_cot = enabled;
        self
    }

    /** 启用或禁用长期记忆。

    # 参数
    * `enabled` - `true` 启用长期记忆，`false` 禁用

    # 说明
    启用后，Agent 可以使用 remember/recall/forget 工具，并支持自动上下文注入。
    */
    pub fn enable_memory(mut self, enabled: bool) -> Self {
        self.enable_memory = enabled;
        self
    }

    /** 设置长期记忆 store 的文件路径。

    # 参数
    * `path` - 长期记忆 store 的文件路径
    */
    pub fn memory_path(mut self, path: &str) -> Self {
        self.memory_path = path.to_string();
        self
    }

    /** 设置会话标识符。

    # 参数
    * `id` - 会话标识符

    # 说明
    会话标识符只是一个轻量的运行/线程标签。持久化的运行时恢复使用的是
    `conversation_id` 配合 `RuntimeStateStore`。
    */
    pub fn session_id(mut self, id: &str) -> Self {
        self.session_id = Some(id.to_string());
        self
    }

    /** 设置 React 循环检查点（checkpoint）的间隔（单位：迭代次数）。

    当该值 > 0 时，Agent 会在 React 循环中每 N 次迭代保存一次检查点。
    这样就支持了崩溃恢复：重启后，Agent 会恢复对话历史并从中断处继续。

    默认值：0（仅在执行结束时设置检查点）。
    */
    pub fn react_checkpoint_interval(mut self, interval: usize) -> Self {
        self.react_checkpoint_interval = interval;
        self
    }

    /// 启用或禁用用于校验 final_answer 的 verifier（Critic）。
    pub fn verifier_enabled(mut self, enabled: bool) -> Self {
        self.verifier_enabled = enabled;
        self
    }

    /// 设置 verifier 通过所需的最低分数（0.0-10.0）。
    pub fn verifier_min_score(mut self, score: f64) -> Self {
        self.verifier_min_score = score;
        self
    }

    /// 设置 verifier 的最大重试次数。
    pub fn verifier_max_retries(mut self, retries: usize) -> Self {
        self.verifier_max_retries = retries;
        self
    }

    /// 设置 verifier（Critic）是否启用（可变引用 setter）。
    pub fn set_verifier_enabled(&mut self, enabled: bool) {
        self.verifier_enabled = enabled;
    }

    /// 设置 verifier 的最低通过分数（可变引用 setter）。
    pub fn set_verifier_min_score(&mut self, score: f64) {
        self.verifier_min_score = score;
    }

    /// 设置 verifier 的最大重试次数（可变引用 setter）。
    pub fn set_verifier_max_retries(&mut self, retries: usize) {
        self.verifier_max_retries = retries;
    }

    /** 设置会话（conversation）标识符。

    # 参数
    * `id` - 会话标识符

    # 说明
    会话标识符供 `ConversationStore` 持久化事务记录/历史投影时使用。
    与 `session_id` 不同，它不负责线程状态的恢复。
    */
    pub fn conversation_id(mut self, id: &str) -> Self {
        self.conversation_id = Some(id.to_string());
        self
    }

    /** 设置工具执行配置。

    # 参数
    * `config` - 工具执行配置（包含超时、重试策略、并行并发度等）
    */
    pub fn tool_execution(mut self, config: ToolExecutionConfig) -> Self {
        self.tool_execution = config;
        self
    }

    /** 设置结构化输出格式。

    # 参数
    * `fmt` - 结构化输出格式
    */
    pub fn response_format(mut self, fmt: ResponseFormat) -> Self {
        self.response_format = Some(fmt);
        self
    }

    /** 设置单次工具输出的内联 token 预算。

    当配置了 artifact 存储时，超大输出会被持久化为一个 artifact；
    未配置时则回退为 UTF-8 安全的截断。
    */
    pub fn max_tool_output_tokens(mut self, max: usize) -> Self {
        self.max_tool_output_tokens = Some(max);
        self
    }

    /** 设置完整工具输出 artifact 所使用的目录与保留策略。
    传入 `None` 会禁用 artifact 持久化。
    */
    pub fn tool_output_artifacts(
        self,
        config: Option<echo_core::tools::artifact::ToolOutputArtifactConfig>,
    ) -> Self {
        *self
            .tool_output_artifacts
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = config;
        self
    }

    /// 获取完整工具输出 artifact 的存储配置。
    pub fn get_tool_output_artifacts(
        &self,
    ) -> Option<echo_core::tools::artifact::ToolOutputArtifactConfig> {
        self.tool_output_artifacts
            .lock()
            .ok()
            .and_then(|config| config.clone())
    }

    /** 获取单次工具输出允许的最大 token 数。

    # 返回值
    最大 token 数，`None` 表示不限制
    */
    pub fn get_max_tool_output_tokens(&self) -> Option<usize> {
        self.max_tool_output_tokens
    }

    /** 启用或禁用项目规则自动加载。

    # 参数
    * `enabled` - `true` 表示自动在工作目录中查找 `.echo-agent/AGENT.md`
      并注入系统提示词
    */
    pub fn auto_project_rules(mut self, enabled: bool) -> Self {
        self.auto_project_rules = enabled;
        self
    }

    /// 设置用于指令发现的项目根目录边界。
    pub fn project_root(mut self, root: impl Into<std::path::PathBuf>) -> Self {
        self.project_root = Some(root.into());
        self
    }

    /** 设置工作目录（用于查找项目规则文件）。

    # 参数
    * `path` - 工作目录路径，`None` 表示使用当前目录
    */
    pub fn working_dir(self, path: Option<PathBuf>) -> Self {
        /*
        Mutex 中毒 (某持有锁的线程 panic) 时不让整个 agent 崩溃:
        用 into_inner() 恢复毒锁内部数据继续写入。对齐 react_loop.rs:811
        的 .ok() 思路 — 配置写入失败也不该 panic。
        */
        *self.working_dir.lock().unwrap_or_else(|e| e.into_inner()) = path;
        self
    }

    /// 设置用于精细化管理上下文窗口的 token 预算配置。
    pub fn token_budget(mut self, config: TokenBudgetConfig) -> Self {
        self.token_budget_config = config;
        self
    }

    /** 设置 LLM 温度参数。

    # 参数
    * `temperature` - 温度值（0.0–2.0，`None` 表示使用模型默认值）
    */
    pub fn temperature(mut self, temperature: Option<f32>) -> Self {
        self.temperature = temperature;
        self
    }

    /** 获取 LLM 温度参数。

    # 返回值
    温度值，`None` 表示使用模型默认值。
    */
    pub fn get_temperature(&self) -> Option<f32> {
        self.temperature
    }

    /** 设置最大生成 token 数。

    # 参数
    * `max_tokens` - 最大 token 数（`None` 表示使用模型默认值）
    */
    pub fn max_tokens(mut self, max_tokens: Option<u32>) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /** 获取最大生成 token 数。

    # 返回值
    最大 token 数，`None` 表示使用模型默认值。
    */
    pub fn get_max_tokens(&self) -> Option<u32> {
        self.max_tokens
    }

    /// 设置类型化的权限模式。
    pub fn permission_mode(mut self, mode: PermissionMode) -> Self {
        self.permission_mode = mode;
        self
    }

    /// 获取当前权限模式。
    pub fn get_permission_mode(&self) -> PermissionMode {
        self.permission_mode
    }

    /// 在运行时设置权限模式（可变引用）。
    pub fn set_permission_mode(&mut self, mode: PermissionMode) {
        self.permission_mode = mode;
    }
}

// ── 单元测试 ──────────────────────────────────────────────────────────────────────

/// `AgentConfig` 的单元测试。
#[cfg(test)]
mod tests {
    use super::*;

    /// 验证 `AgentConfig::new` 的默认字段。
    #[test]
    fn test_agent_config_new() {
        let config = AgentConfig::new("qwen3-max", "assistant", "You are a helpful assistant");

        assert_eq!(config.get_model_name(), "qwen3-max");
        assert_eq!(config.get_agent_name(), "assistant");
        assert_eq!(config.get_system_prompt(), "You are a helpful assistant");
        assert_eq!(config.get_max_iterations(), 100);
        assert_eq!(config.get_token_limit(), DEFAULT_TOKEN_LIMIT);
        assert!(!config.is_tool_enabled());
        assert!(!config.is_human_in_loop_enabled());
        assert!(!config.is_subagent_enabled());
    }

    /// 验证最小化配置预设。
    #[test]
    fn test_agent_config_minimal() {
        let config = AgentConfig::minimal("qwen3-max", "Be helpful");

        assert_eq!(config.get_model_name(), "qwen3-max");
        assert!(!config.is_tool_enabled());
        assert!(!config.is_memory_enabled());
        assert!(!config.is_cot_enabled());
    }

    /// 验证标准配置预设。
    #[test]
    fn test_agent_config_standard() {
        let config = AgentConfig::standard("qwen3-max", "agent1", "You are helpful");

        assert!(config.is_tool_enabled());
        assert!(config.is_cot_enabled());
    }

    /// 验证全功能配置预设。
    #[test]
    fn test_agent_config_full_featured() {
        let config = AgentConfig::full_featured("qwen3-max", "agent1", "You are helpful");

        assert!(config.is_tool_enabled());
        assert!(config.is_memory_enabled());
        assert!(config.is_cot_enabled());
    }

    /// 验证构建器链式调用正确设置各字段。
    #[test]
    fn test_agent_config_builder_chain() {
        let config = AgentConfig::new("model", "agent", "prompt")
            .max_iterations(20)
            .token_limit(DEFAULT_TOKEN_LIMIT)
            .enable_tool(true)
            .enable_human_in_loop(true)
            .enable_subagent(true)
            .enable_memory(true)
            .enable_cot(false)
            .llm_max_retries(5)
            .llm_retry_delay_ms(1000)
            .tool_error_feedback(false);

        assert_eq!(config.get_max_iterations(), 20);
        assert_eq!(config.get_token_limit(), DEFAULT_TOKEN_LIMIT);
        assert!(config.is_tool_enabled());
        assert!(config.is_human_in_loop_enabled());
        assert!(config.is_subagent_enabled());
        assert!(config.is_memory_enabled());
        assert!(!config.is_cot_enabled());
        assert_eq!(config.get_llm_max_retries(), 5);
        assert_eq!(config.get_llm_retry_delay_ms(), 1000);
        assert!(!config.get_tool_error_feedback());
    }

    /// 验证权限模式可无损往返转换。
    #[test]
    fn permission_mode_round_trips_without_loss() {
        let modes = [
            PermissionMode::Default,
            PermissionMode::Plan,
            PermissionMode::AcceptEdits,
            PermissionMode::BypassPermissions,
            PermissionMode::Auto,
            PermissionMode::Bubble,
            PermissionMode::DontAsk,
            PermissionMode::StrictConfirm,
        ];
        for mode in modes {
            let mut config = AgentConfig::new("model", "agent", "prompt").permission_mode(mode);
            assert_eq!(config.get_permission_mode(), mode);
            config.set_permission_mode(mode);
            assert_eq!(config.get_permission_mode(), mode);
        }
    }

    /// 验证允许的工具列表设置。
    #[test]
    fn test_agent_config_allowed_tools() {
        let config = AgentConfig::new("model", "agent", "prompt")
            .allowed_tools(vec!["tool1".to_string(), "tool2".to_string()]);

        assert_eq!(config.get_allowed_tools(), &["tool1", "tool2"]);
    }

    /// 验证会话 ID 设置。
    #[test]
    fn test_agent_config_session_id() {
        let config = AgentConfig::new("model", "agent", "prompt").session_id("session-123");

        assert_eq!(config.get_session_id(), Some("session-123"));
    }

    /// 验证对话 ID 设置。
    #[test]
    fn test_agent_config_conversation_id() {
        let config =
            AgentConfig::new("model", "agent", "prompt").conversation_id("conversation-123");

        assert_eq!(config.get_conversation_id(), Some("conversation-123"));
    }

    /// 验证模型名称可被修改。
    #[test]
    fn test_agent_config_model_name_mutation() {
        let mut config = AgentConfig::new("model1", "agent", "prompt");

        config.set_model_name("model2");
        assert_eq!(config.get_model_name(), "model2");
    }

    /// 验证启用全部功能的配置。
    #[test]
    fn test_agent_config_with_full_features() {
        let config = AgentConfig::new("model", "agent", "prompt").with_full_features();

        assert!(config.is_tool_enabled());
        assert!(config.is_memory_enabled());
        assert!(config.is_cot_enabled());
    }

    /// 验证工具列表配置。
    #[test]
    fn test_agent_config_with_tools() {
        let config = AgentConfig::new("model", "agent", "prompt").with_tools();

        assert!(config.is_tool_enabled());
        assert!(config.is_cot_enabled());
    }

    /// 验证记忆存储路径配置。
    #[test]
    fn test_agent_config_memory_path() {
        let config =
            AgentConfig::new("model", "agent", "prompt").memory_path("/custom/path/store.json");

        assert_eq!(config.get_memory_path(), "/custom/path/store.json");
    }
}
