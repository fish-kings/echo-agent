/*!
提供商能力与模型画像（profile）抽象。

这些类型允许在运行时查询某个 LLM 提供商/模型支持哪些特性，用能力检查取代硬编码的
提供商特定行为。

## ProviderCapabilities

不同提供商（OpenAI 兼容、Anthropic、Ollama）之间存在差异的底层协议特性。
每个提供商返回其静态能力集合。

## ModelProfile

根据模型名称与提供商配置解析出的更高层模型信息。将提供商能力与模型特定知识
（上下文窗口、推理支持、多模态支持等）结合在一起。
*/

use crate::llm::{LlmApiProtocol, ThinkingLevel, ThinkingProtocol};
use std::collections::{HashMap, HashSet};

// ── ProviderCapabilities（提供商能力） ─────────────────────────────────────────────

/// 不同 LLM 提供商之间存在差异的协议层特性。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProviderCapabilities {
    /**
    是否支持增量流式工具调用片段（OpenAI `delta.tool_calls`）。
    为 false 时，工具调用仅在最终消息中可用（Anthropic 块事件）。
    */
    pub streaming_tool_calls: bool,

    /// 使用具名 SSE 事件类型（Anthropic `event: content_block_start` 等），而非裸 `data:` 行（OpenAI 兼容）。
    pub named_sse_events: bool,

    /// 流式响应中是否支持 `reasoning_content` / 思考输出（Qwen3、DeepSeek、OpenAI o 系列）。
    pub reasoning_content: bool,

    /// 是否支持图像输入（多模态内容片段）。
    pub image_input: bool,

    /// 系统提示词作为顶层请求字段发送（Anthropic `system`），而非作为 `messages` 数组中 `role: "system"` 的消息。
    pub system_as_top_level: bool,

    /// 流式传输使用 NDJSON（每行一个 JSON 对象，以 `\n` 分隔），而非标准 SSE `data:` 行（Ollama）。
    pub ndjson_streaming: bool,

    /// 请求中是否支持函数/工具定义。
    pub tool_support: bool,

    /// 是否支持结构化输出（带 JSON Schema 的 `response_format`）。
    pub structured_output: bool,

    /// 是否需要提供商特定的版本请求头（Anthropic `anthropic-version`）。
    pub requires_version_header: bool,

    /**
    是否支持并行工具调用（单个响应中请求多个工具）。
    OpenAI、Anthropic 及大多数云提供商支持；Ollama 与本地模型通常不支持。
    */
    pub supports_parallel_tool_calls: bool,

    /**
    是否支持显式的 `tool_choice=none` 请求控制。
    不支持的提供商仍可通过不暴露任何工具来实现仅最终回答模式。
    */
    pub supports_tool_choice_none: bool,

    /**
    用于精确计数 token 的分词器名称或标识（例如 GPT-4 为 `"cl100k_base"`，GPT-4o 为 `"o200k_base"`）。
    `None` 表示分词器未知，调用方应回退到启发式计数。
    */
    pub tokenizer_name: Option<&'static str>,
}

impl ProviderCapabilities {
    /// OpenAI 及 OpenAI 兼容提供商（DashScope、DeepSeek、Moonshot、智谱等）的默认能力。
    pub const fn openai_compatible() -> Self {
        Self {
            streaming_tool_calls: true,
            named_sse_events: false,
            reasoning_content: true,
            image_input: true,
            system_as_top_level: false,
            ndjson_streaming: false,
            tool_support: true,
            structured_output: true,
            requires_version_header: false,
            supports_parallel_tool_calls: true,
            supports_tool_choice_none: true,
            tokenizer_name: None,
        }
    }

    /// Anthropic Messages API 的能力。
    pub const fn anthropic() -> Self {
        Self {
            streaming_tool_calls: false, // 使用 content_block_start/stop 事件
            named_sse_events: true,
            reasoning_content: false, // 本实现中未做映射
            image_input: true,
            system_as_top_level: true,
            ndjson_streaming: false,
            tool_support: true,
            structured_output: false, // Messages API 无 JSON 模式
            requires_version_header: true,
            supports_parallel_tool_calls: true,
            supports_tool_choice_none: false,
            tokenizer_name: Some("claude"),
        }
    }

    /// 本地 Ollama 的能力。
    pub const fn ollama() -> Self {
        Self {
            streaming_tool_calls: false,
            named_sse_events: false,
            reasoning_content: false,
            image_input: false,
            system_as_top_level: false,
            ndjson_streaming: true,
            tool_support: true,
            structured_output: false,
            requires_version_header: false,
            supports_parallel_tool_calls: false,
            supports_tool_choice_none: false,
            tokenizer_name: None,
        }
    }

    /// 根据提供商名称字符串解析能力集合。
    pub fn from_provider_name(name: &str) -> Self {
        match name.to_ascii_lowercase().as_str() {
            "anthropic" => Self::anthropic(),
            "ollama" => Self::ollama(),
            _ => Self::openai_compatible(),
        }
    }
}

// ── ThinkingProfile（思考能力画像） ──────────────────────────────────────────────────

/// GPT-5.6 支持的思考档位。
const GPT_56_LEVELS: &[ThinkingLevel] = &[
    ThinkingLevel::None,
    ThinkingLevel::Low,
    ThinkingLevel::Medium,
    ThinkingLevel::High,
    ThinkingLevel::Xhigh,
    ThinkingLevel::Max,
];
/// Claude 4.6 支持的思考档位。
const CLAUDE_46_LEVELS: &[ThinkingLevel] = &[
    ThinkingLevel::Low,
    ThinkingLevel::Medium,
    ThinkingLevel::High,
    ThinkingLevel::Xhigh,
    ThinkingLevel::Max,
];
/// DeepSeek V4 支持的思考档位。
const DEEPSEEK_V4_LEVELS: &[ThinkingLevel] = &[
    ThinkingLevel::None,
    ThinkingLevel::Low,
    ThinkingLevel::High,
    ThinkingLevel::Max,
];
/// GLM 5.2 支持的思考档位。
const GLM_52_LEVELS: &[ThinkingLevel] =
    &[ThinkingLevel::None, ThinkingLevel::High, ThinkingLevel::Max];
/// Kimi K3 支持的思考档位。
const KIMI_K3_LEVELS: &[ThinkingLevel] =
    &[ThinkingLevel::Low, ThinkingLevel::High, ThinkingLevel::Max];
/// Gemini 3 支持的思考档位。
const GEMINI_3_LEVELS: &[ThinkingLevel] = &[
    ThinkingLevel::Minimal,
    ThinkingLevel::Low,
    ThinkingLevel::Medium,
    ThinkingLevel::High,
];
/// Gemini 2.5 支持的思考档位。
const GEMINI_25_LEVELS: &[ThinkingLevel] = &[
    ThinkingLevel::None,
    ThinkingLevel::Low,
    ThinkingLevel::Medium,
    ThinkingLevel::High,
];
/// 仅开/关两档的思考开关档位。
const TOGGLE_LEVELS: &[ThinkingLevel] = &[ThinkingLevel::None, ThinkingLevel::High];
/// Ollama 上 gpt-oss 支持的思考档位。
const OLLAMA_GPT_OSS_LEVELS: &[ThinkingLevel] = &[
    ThinkingLevel::Low,
    ThinkingLevel::Medium,
    ThinkingLevel::High,
];

/// 针对某个具体提供商端点、API 协议与模型 ID 集中解析出的请求侧思考能力。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ThinkingProfile {
    /// 该模型使用的思考线路协议。
    pub protocol: ThinkingProtocol,
    /**
    仅包含有效档位。应用层会单独添加 `auto`；
    空切片表示由模型自行决定，或尚无已验证的控制方式。
    */
    pub levels: &'static [ThinkingLevel],
}

impl ThinkingProfile {
    /// 以协议与档位列表构造思考画像。
    pub const fn new(protocol: ThinkingProtocol, levels: &'static [ThinkingLevel]) -> Self {
        Self { protocol, levels }
    }

    /// 未知模型的思考画像：无协议、无档位。
    pub const fn unknown() -> Self {
        Self::new(ThinkingProtocol::None, &[])
    }

    /// 是否支持手动控制思考档位（协议会发出请求字段且档位非空）。
    pub fn supports_manual_control(self) -> bool {
        self.protocol.emits_field() && !self.levels.is_empty()
    }
}

// ── ModelProfile（模型画像） ─────────────────────────────────────────────────────

/// 结合提供商能力与模型特定知识的更高层模型信息。
#[derive(Debug, Clone)]
pub struct ModelProfile {
    /// 提供商名称。
    pub provider: String,

    /// 解析后的模型名称。
    pub model_name: String,

    /// 来自提供商的底层协议能力。
    pub capabilities: ProviderCapabilities,

    /// 该模型是否已知支持 `reasoning_content` / 思考输出。
    pub supports_reasoning: bool,

    /**
    该模型使用的思考线路协议（如有）。它决定各提供商实现如何转换 `ChatRequest::thinking`，
    并避免向会以 400 拒绝该字段的模型（例如 GPT-5-nano、Claude Opus 4.7+）发送思考字段。
    */
    pub thinking_protocol: crate::llm::thinking::ThinkingProtocol,

    /// 所解析思考协议下已验证的有效档位。
    pub thinking_levels: &'static [ThinkingLevel],

    /// 该模型是否具备多模态能力（接受图像）。
    pub supports_images: bool,

    /// 该模型是否能定义并调用工具。
    pub supports_tools: bool,

    /// 已知的最大输出 token 数（未知时为 None）。
    pub max_output_tokens: Option<u32>,

    /// 该模型是否支持流式输出。
    pub supports_streaming: bool,

    /// 该模型是否支持并行工具调用（源自提供商能力，特定模型可覆盖）。
    pub supports_parallel_tool_calls: bool,

    /// 提供商/模型是否接受显式的 `tool_choice=none`。
    pub supports_tool_choice_none: bool,

    /// 已知的上下文窗口大小（token 数）。
    pub context_window: Option<u32>,

    /// 针对该模型在框架层面排除的工具集合。
    pub excluded_tools: HashSet<String>,

    /// 稳定的模型特定系统提示词后缀。
    pub prompt_suffix: Option<String>,

    /// 用于精确计数 token 的分词器名称（未知时为 None）。
    pub tokenizer_name: Option<&'static str>,
}

/// 根据厂商和模型名称推断上下文窗口大小；未匹配到已知模式时返回 None。
pub fn infer_context_window(provider: &str, model_name: &str) -> Option<u32> {
    let provider = provider.trim().to_ascii_lowercase();
    let lower = model_name.to_ascii_lowercase();
    if matches!(provider.as_str(), "openai" | "azure-openai") && lower.starts_with("gpt-5.6") {
        // GPT-5.6 Sol/Terra/Luna 提供 1.05M 的上下文窗口。
        Some(1_050_000)
    } else if provider == "anthropic"
        && (lower.starts_with("claude-fable-5")
            || lower.starts_with("claude-opus-4-8")
            || lower.starts_with("claude-sonnet-5"))
        || provider == "deepseek" && lower.starts_with("deepseek-v4")
        || matches!(
            provider.as_str(),
            "dashscope" | "qwen" | "aliyun" | "alibaba"
        ) && (lower.starts_with("qwen3.7-max") || lower.starts_with("qwen3.7-plus"))
        || provider == "zhipu" && lower.starts_with("glm-5.2")
    {
        Some(1_000_000)
    } else if provider == "moonshot"
        && (lower.starts_with("kimi-k2.7") || lower.starts_with("kimi-k2.6"))
    {
        Some(256_000)
    } else {
        None
    }
}

impl ModelProfile {
    /// 根据提供商能力与模型名称构建画像，并按已知模型名称模式应用模型特定覆盖。
    pub fn new(model_name: &str, provider: &str, capabilities: ProviderCapabilities) -> Self {
        let lower = model_name.to_ascii_lowercase();
        let thinking_profile =
            resolve_thinking_profile(provider, model_name, LlmApiProtocol::ChatCompletions, None);
        let thinking_protocol = thinking_profile.protocol;

        /*
        模型特定的推理检测：只有当模型既使用某种思考协议、其提供商又会在响应中输出
        reasoning_content 时，才认为它"支持推理"。（自适应思考的 Claude 模型仍会上报
        推理内容，只是不接受请求字段。）
        */
        let supports_reasoning =
            !matches!(thinking_protocol, ThinkingProtocol::None) && capabilities.reasoning_content;

        // 模型特定的图像支持检测
        let supports_images = capabilities.image_input
            && !lower.starts_with("o3-mini") // o3-mini 不支持图像
            && !lower.starts_with("o1-mini")
            && !lower.starts_with("o1-preview");

        // 已知的最大输出 token 数
        let max_output_tokens = if lower.contains("qwen3-235b") {
            Some(131_072)
        } else if lower.starts_with("gpt-5") || lower.starts_with("o3") || lower.starts_with("o4") {
            Some(16_384)
        } else if lower.starts_with("claude-") {
            Some(8_192)
        } else {
            None
        };

        // 分词器名称映射
        let tokenizer_name = if lower.starts_with("gpt-5") || lower.starts_with("gpt-4.5") {
            Some("o200k_base")
        } else if lower.starts_with("gpt-4") || lower.starts_with("gpt-3") {
            Some("cl100k_base")
        } else {
            capabilities.tokenizer_name
        };

        Self {
            provider: provider.to_string(),
            model_name: model_name.to_string(),
            capabilities,
            supports_reasoning,
            thinking_protocol,
            thinking_levels: thinking_profile.levels,
            supports_images,
            supports_tools: capabilities.tool_support,
            max_output_tokens,
            supports_streaming: true, // 所有已支持的提供商都支持流式
            supports_parallel_tool_calls: capabilities.supports_parallel_tool_calls,
            supports_tool_choice_none: capabilities.supports_tool_choice_none,
            context_window: infer_context_window(provider, model_name),
            excluded_tools: HashSet::new(),
            prompt_suffix: None,
            tokenizer_name,
        }
    }

    /// 根据提供商名称与模型名称构建画像。
    pub fn from_provider_name(model_name: &str, provider: &str) -> Self {
        let capabilities = ProviderCapabilities::from_provider_name(provider);
        Self::new(model_name, provider, capabilities)
    }
}

/// 由使用方提供、作用于某提供商或某精确模型的框架层覆盖配置。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ModelProfileOverride {
    /// 覆盖是否支持并行工具调用。
    pub supports_parallel_tool_calls: Option<bool>,
    /// 覆盖是否支持显式 `tool_choice=none`。
    pub supports_tool_choice_none: Option<bool>,
    /// 覆盖是否支持结构化输出。
    pub supports_structured_output: Option<bool>,
    /// 覆盖上下文窗口大小（token 数）。
    pub context_window: Option<u32>,
    /// 追加排除的工具名称集合。
    pub excluded_tools: HashSet<String>,
    /// 覆盖系统提示词后缀。
    pub prompt_suffix: Option<String>,
}

impl ModelProfileOverride {
    /// 将本覆盖配置中已设置的项应用到给定画像上。
    fn apply_to(&self, profile: &mut ModelProfile) {
        if let Some(value) = self.supports_parallel_tool_calls {
            profile.supports_parallel_tool_calls = value;
        }
        if let Some(value) = self.supports_tool_choice_none {
            profile.supports_tool_choice_none = value;
        }
        if let Some(value) = self.supports_structured_output {
            profile.capabilities.structured_output = value;
        }
        if let Some(value) = self.context_window {
            profile.context_window = Some(value);
        }
        profile
            .excluded_tools
            .extend(self.excluded_tools.iter().cloned());
        if let Some(value) = &self.prompt_suffix {
            profile.prompt_suffix = Some(value.clone());
        }
    }
}

/**
解析基础 [`ModelProfile`] 并叠加使用方注册的覆盖配置。

先应用提供商默认覆盖，再应用精确匹配的规范化 `provider:model` 条目，因此后者优先级更高。
*/
#[derive(Debug, Clone, Default)]
pub struct ModelProfileResolver {
    /// 以规范化提供商名为键的默认覆盖。
    provider_defaults: HashMap<String, ModelProfileOverride>,
    /// 以规范化 `provider:model` 为键的精确模型覆盖。
    exact_models: HashMap<String, ModelProfileOverride>,
}

impl ModelProfileResolver {
    /// 创建不含任何覆盖的解析器。
    pub fn new() -> Self {
        Self::default()
    }

    /// 注册某提供商的默认覆盖配置。
    pub fn register_provider_default(
        mut self,
        provider: impl AsRef<str>,
        profile: ModelProfileOverride,
    ) -> Self {
        self.provider_defaults
            .insert(normalize_selector_part(provider.as_ref()), profile);
        self
    }

    /// 注册某提供商下精确模型的覆盖配置（优先级高于提供商默认）。
    pub fn register_exact(
        mut self,
        provider: impl AsRef<str>,
        model: impl AsRef<str>,
        profile: ModelProfileOverride,
    ) -> Self {
        self.exact_models
            .insert(selector_key(provider.as_ref(), model.as_ref()), profile);
        self
    }

    /// 构建基础画像后依次应用提供商默认覆盖与精确模型覆盖。
    pub fn resolve(
        &self,
        provider: &str,
        model: &str,
        capabilities: ProviderCapabilities,
    ) -> ModelProfile {
        let mut profile = ModelProfile::new(model, provider, capabilities);
        if let Some(default) = self
            .provider_defaults
            .get(&normalize_selector_part(provider))
        {
            default.apply_to(&mut profile);
        }
        if let Some(exact) = self.exact_models.get(&selector_key(provider, model)) {
            exact.apply_to(&mut profile);
        }
        profile
    }
}

/// 规范化选择器片段：去除首尾空白并转为小写。
fn normalize_selector_part(value: &str) -> String {
    value.trim().to_ascii_lowercase()
}

/// 生成规范化的 `provider:model` 选择器键。
fn selector_key(provider: &str, model: &str) -> String {
    format!(
        "{}:{}",
        normalize_selector_part(provider),
        normalize_selector_part(model)
    )
}

/**
为运行时模型解析唯一权威的思考画像。

先判断端点/提供商方言，再判断模型家族，因为网关可能通过不同的线路字段暴露同一模型。
未知模型仍可完全正常使用，只是不提供手动思考控制。
*/
pub fn resolve_thinking_profile(
    provider: &str,
    model: &str,
    api_protocol: LlmApiProtocol,
    endpoint: Option<&str>,
) -> ThinkingProfile {
    use ThinkingProtocol as T;

    let provider = provider.trim().to_ascii_lowercase();
    let model = model.trim().to_ascii_lowercase();
    let endpoint = endpoint.unwrap_or_default().to_ascii_lowercase();
    let is_dashscope = matches!(
        provider.as_str(),
        "dashscope" | "qwen" | "aliyun" | "alibaba" | "modelstudio" | "bailian"
    ) || endpoint.contains("dashscope.aliyuncs.com");
    let is_ollama = provider == "ollama"
        || endpoint.contains("localhost:11434")
        || endpoint.contains("127.0.0.1:11434");

    if model.starts_with("claude-") {
        let Some((major, minor)) = model_family_version(&model, "claude-") else {
            return ThinkingProfile::unknown();
        };
        if major < 4 || (major == 4 && minor < 6) {
            return ThinkingProfile::unknown();
        }
        if major == 4 && minor == 6 {
            return if api_protocol == LlmApiProtocol::Anthropic {
                ThinkingProfile::new(T::AnthropicEffort, CLAUDE_46_LEVELS)
            } else {
                ThinkingProfile::new(T::OpenaiReasoningEffort, CLAUDE_46_LEVELS)
            };
        }
        return ThinkingProfile::new(T::AnthropicAdaptive, &[]);
    }

    if api_protocol == LlmApiProtocol::Anthropic {
        return ThinkingProfile::unknown();
    }

    if is_ollama && api_protocol == LlmApiProtocol::ChatCompletions {
        if model.starts_with("gpt-oss") {
            return ThinkingProfile::new(T::OllamaThink, OLLAMA_GPT_OSS_LEVELS);
        }
        if model.starts_with("qwen3")
            || model.starts_with("deepseek-r1")
            || model.starts_with("deepseek-v3")
            || model.starts_with("deepseek-v4")
            || model.starts_with("magistral")
        {
            return ThinkingProfile::new(T::OllamaThink, TOGGLE_LEVELS);
        }
        return ThinkingProfile::unknown();
    }

    if model.starts_with("gpt-5.6") || model.starts_with("gpt-5-6") {
        return ThinkingProfile::new(T::OpenaiReasoningEffort, GPT_56_LEVELS);
    }

    if model.starts_with("deepseek-v4") {
        if is_dashscope && api_protocol == LlmApiProtocol::ChatCompletions {
            return ThinkingProfile::new(T::EnableThinkingFlag, TOGGLE_LEVELS);
        }
        return ThinkingProfile::new(T::DeepseekReasoningEffort, DEEPSEEK_V4_LEVELS);
    }

    if let Some((major, minor)) = model_family_version(&model, "glm-")
        && (major > 5 || (major == 5 && minor >= 2))
        && api_protocol == LlmApiProtocol::ChatCompletions
    {
        return ThinkingProfile::new(T::GlmReasoningEffort, GLM_52_LEVELS);
    }

    if model.starts_with("kimi-k3") && api_protocol == LlmApiProtocol::ChatCompletions {
        return ThinkingProfile::new(T::OpenaiReasoningEffort, KIMI_K3_LEVELS);
    }
    if model.starts_with("kimi-k2.7") {
        return ThinkingProfile::new(T::ModelManaged, &[]);
    }
    if model.starts_with("kimi-k2.6") && api_protocol == LlmApiProtocol::ChatCompletions {
        return ThinkingProfile::new(T::ThinkingType, TOGGLE_LEVELS);
    }

    if model.starts_with("qwen3") && api_protocol == LlmApiProtocol::ChatCompletions {
        return ThinkingProfile::new(T::EnableThinkingFlag, TOGGLE_LEVELS);
    }

    if (model.starts_with("gemini-3") || model.starts_with("gemini-3."))
        && api_protocol == LlmApiProtocol::ChatCompletions
    {
        return ThinkingProfile::new(T::OpenaiReasoningEffort, GEMINI_3_LEVELS);
    }
    if model.starts_with("gemini-2.5") && api_protocol == LlmApiProtocol::ChatCompletions {
        return ThinkingProfile::new(T::OpenaiReasoningEffort, GEMINI_25_LEVELS);
    }

    ThinkingProfile::unknown()
}

/// 从带指定前缀的模型名中提取 (主版本, 次版本)，主版本限定在 3..=9。
fn model_family_version(model: &str, prefix: &str) -> Option<(u32, u32)> {
    let rest = model.strip_prefix(prefix)?;
    let mut segments = rest.split('-').peekable();
    while let Some(segment) = segments.next() {
        if let Some((major, minor)) = segment.split_once('.')
            && let (Ok(major), Ok(minor)) = (major.parse::<u32>(), minor.parse::<u32>())
            && (3..=9).contains(&major)
        {
            return Some((major, minor));
        }
        if let Ok(major) = segment.parse::<u32>()
            && (3..=9).contains(&major)
        {
            let minor = segments
                .peek()
                .and_then(|value| value.parse::<u32>().ok())
                .filter(|value| *value <= 9)
                .unwrap_or(0);
            return Some((major, minor));
        }
    }
    None
}

// ── Cache Policy（缓存策略） ──────────────────────────────────────────────────────

/**
影响提示词组织方式的提供商特定缓存行为。

不同提供商以请求的不同部分作为提示词缓存键。此策略告诉提示词组装器需要保持哪些部分稳定。
*/
#[derive(Debug, Clone)]
pub struct CachePolicy {
    /// 发送稳定的 `user_id` 是否能启用 KVCache 分区复用；DeepSeek 为 true，不支持的提供商为 false。
    pub stable_user_id_enables_cache: bool,
    /// 提供商是否在用量元数据中上报缓存命中 token；为 false 表示无法衡量缓存效果。
    pub reports_cache_metrics: bool,
    /// 工具定义是否属于缓存键；为 true 时，工具列表的顺序与内容在各请求间必须确定。
    pub cache_key_includes_tools: bool,
    /// 系统消息是否属于缓存键前缀；为 true 时，系统消息的任何变化都会使缓存失效。
    pub cache_key_includes_system_prompt: bool,
    /**
    推荐的最小稳定前缀长度（token 数）。系统提示词中此位置之后的内容可以变化而不破坏缓存。
    0 表示整个系统提示词都必须保持稳定。
    */
    pub recommended_stable_prefix_tokens: usize,
}

impl Default for CachePolicy {
    /// 通用默认缓存策略：不依赖 user_id，工具与系统提示词均计入缓存键。
    fn default() -> Self {
        Self {
            stable_user_id_enables_cache: false,
            reports_cache_metrics: true,
            cache_key_includes_tools: true,
            cache_key_includes_system_prompt: true,
            recommended_stable_prefix_tokens: 0,
        }
    }
}

impl CachePolicy {
    /// DeepSeek 的缓存策略（KVCache，基于 user_id 隔离）。
    pub fn deepseek() -> Self {
        Self {
            stable_user_id_enables_cache: true,
            reports_cache_metrics: true,
            cache_key_includes_tools: true,
            cache_key_includes_system_prompt: true,
            recommended_stable_prefix_tokens: 2000,
        }
    }

    /// Anthropic 的缓存策略（提示词缓存，显式缓存断点）。
    pub fn anthropic() -> Self {
        Self {
            stable_user_id_enables_cache: false,
            reports_cache_metrics: true,
            cache_key_includes_tools: true,
            cache_key_includes_system_prompt: true,
            recommended_stable_prefix_tokens: 1024,
        }
    }

    /// OpenAI 的缓存策略（自动前缀缓存）。
    pub fn openai() -> Self {
        Self {
            stable_user_id_enables_cache: false,
            reports_cache_metrics: true,
            cache_key_includes_tools: true,
            cache_key_includes_system_prompt: true,
            recommended_stable_prefix_tokens: 1024,
        }
    }

    /// 根据提供商名称解析缓存策略。
    pub fn from_provider(provider: &str) -> Self {
        match provider.to_ascii_lowercase().as_str() {
            "deepseek" => Self::deepseek(),
            "anthropic" => Self::anthropic(),
            "openai" => Self::openai(),
            _ => Self::default(),
        }
    }
}

/// Provider 能力与模型档案的单元测试。
#[cfg(test)]
mod tests {
    use super::*;
    use ThinkingProtocol as T;

    /// 以 ChatCompletions 协议、无端点解析思考画像的测试辅助函数。
    fn chat_profile(provider: &str, model: &str) -> ThinkingProfile {
        resolve_thinking_profile(provider, model, LlmApiProtocol::ChatCompletions, None)
    }

    /// 验证 GPT-5.6 拥有六个不同的思考档位。
    #[test]
    fn gpt_56_has_six_distinct_levels() {
        let profile = chat_profile("openai", "gpt-5.6-sol");
        assert_eq!(profile.protocol, T::OpenaiReasoningEffort);
        assert_eq!(profile.levels, GPT_56_LEVELS);
        assert_eq!(profile.levels.len(), 6);
        assert!(!profile.levels.contains(&ThinkingLevel::Minimal));
    }

    /// 验证当前前沿模型的上下文窗口推断结果。
    #[test]
    fn infers_current_frontier_context_windows() {
        assert_eq!(
            infer_context_window("openai", "gpt-5.6-sol"),
            Some(1_050_000)
        );
        assert_eq!(
            infer_context_window("openai", "gpt-5.6-terra"),
            Some(1_050_000)
        );
        assert_eq!(
            infer_context_window("anthropic", "claude-sonnet-5"),
            Some(1_000_000)
        );
        assert_eq!(
            infer_context_window("anthropic", "claude-opus-4-8"),
            Some(1_000_000)
        );
        assert_eq!(
            infer_context_window("deepseek", "deepseek-v4-pro"),
            Some(1_000_000)
        );
        assert_eq!(
            infer_context_window("dashscope", "qwen3.7-max"),
            Some(1_000_000)
        );
        assert_eq!(
            infer_context_window("dashscope", "qwen3.7-plus"),
            Some(1_000_000)
        );
        assert_eq!(
            infer_context_window("moonshot", "kimi-k2.7-code"),
            Some(256_000)
        );
        assert_eq!(infer_context_window("zhipu", "glm-5.2"), Some(1_000_000));
        assert_eq!(infer_context_window("custom", "gpt-5.6-sol"), None);
        assert_eq!(infer_context_window("openai", "claude-sonnet-5"), None);
        assert_eq!(infer_context_window("custom", "unknown-model"), None);
    }

    /// 验证 Claude 的思考控制从 4.6 版本开始支持。
    #[test]
    fn claude_support_starts_at_46() {
        assert_eq!(
            chat_profile("anthropic", "claude-4.5-sonnet").protocol,
            T::None
        );
        let direct = resolve_thinking_profile(
            "anthropic",
            "claude-opus-4-6",
            LlmApiProtocol::Anthropic,
            Some("https://api.anthropic.com/v1/messages"),
        );
        assert_eq!(direct.protocol, T::AnthropicEffort);
        assert_eq!(direct.levels, CLAUDE_46_LEVELS);
        assert_eq!(
            chat_profile("gateway", "claude-opus-4.6").protocol,
            T::OpenaiReasoningEffort
        );
        assert_eq!(
            resolve_thinking_profile(
                "anthropic",
                "claude-opus-4.7",
                LlmApiProtocol::Anthropic,
                None,
            )
            .protocol,
            T::AnthropicAdaptive
        );
    }

    /// 验证 GLM 的思考控制从 5.2 版本开始支持。
    #[test]
    fn glm_support_starts_at_52() {
        assert_eq!(chat_profile("zhipu", "glm-5.1").protocol, T::None);
        assert_eq!(chat_profile("zhipu", "glm-4.6").protocol, T::None);
        let profile = chat_profile("zhipu", "glm-5.2");
        assert_eq!(profile.protocol, T::GlmReasoningEffort);
        assert_eq!(profile.levels, GLM_52_LEVELS);
    }

    /// 验证常见模型的思考画像均被显式定义。
    #[test]
    fn common_model_profiles_are_explicit() {
        assert_eq!(
            chat_profile("deepseek", "deepseek-v4").protocol,
            T::DeepseekReasoningEffort
        );
        assert_eq!(chat_profile("moonshot", "kimi-k3").levels, KIMI_K3_LEVELS);
        assert_eq!(
            chat_profile("moonshot", "kimi-k2.7-code").protocol,
            T::ModelManaged
        );
        assert_eq!(
            chat_profile("moonshot", "kimi-k2.6").protocol,
            T::ThinkingType
        );
        assert_eq!(
            chat_profile("dashscope", "qwen3-max").protocol,
            T::EnableThinkingFlag
        );
        assert_eq!(
            chat_profile("google", "gemini-3-pro").levels,
            GEMINI_3_LEVELS
        );
        assert_eq!(
            chat_profile("google", "gemini-2.5-pro").levels,
            GEMINI_25_LEVELS
        );
        assert_eq!(
            chat_profile("ollama", "gpt-oss:20b").levels,
            OLLAMA_GPT_OSS_LEVELS
        );
        assert_eq!(chat_profile("ollama", "qwen3:32b").protocol, T::OllamaThink);
        assert_eq!(chat_profile("ollama", "llama3.1").protocol, T::None);
    }

    /// 验证提供商方言优先于模型家族判断。
    #[test]
    fn provider_dialect_precedes_model_family() {
        assert_eq!(
            resolve_thinking_profile(
                "custom",
                "deepseek-v4",
                LlmApiProtocol::ChatCompletions,
                Some("https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"),
            )
            .protocol,
            T::EnableThinkingFlag
        );
        assert_eq!(
            chat_profile("deepseek", "deepseek-v4").protocol,
            T::DeepseekReasoningEffort
        );
    }

    /// 验证未知模型不提供手动思考控制。
    #[test]
    fn unknown_models_remain_uncontrolled() {
        let profile = chat_profile("custom", "future-model");
        assert_eq!(profile, ThinkingProfile::unknown());
        assert!(!profile.supports_manual_control());
    }

    /// 验证精确模型覆盖优先于提供商默认覆盖。
    #[test]
    fn exact_model_override_wins_over_provider_default() {
        let resolver = ModelProfileResolver::new()
            .register_provider_default(
                " OpenAI ",
                ModelProfileOverride {
                    supports_parallel_tool_calls: Some(false),
                    supports_tool_choice_none: Some(false),
                    excluded_tools: HashSet::from(["shell".to_string()]),
                    prompt_suffix: Some("provider suffix".to_string()),
                    ..Default::default()
                },
            )
            .register_exact(
                "openai",
                "GPT-5-CODEX",
                ModelProfileOverride {
                    supports_parallel_tool_calls: Some(true),
                    supports_tool_choice_none: Some(true),
                    context_window: Some(400_000),
                    excluded_tools: HashSet::from(["browser".to_string()]),
                    prompt_suffix: Some("exact suffix".to_string()),
                    ..Default::default()
                },
            );

        let profile = resolver.resolve(
            "OPENAI",
            "gpt-5-codex",
            ProviderCapabilities::openai_compatible(),
        );
        assert!(profile.supports_parallel_tool_calls);
        assert!(profile.supports_tool_choice_none);
        assert_eq!(profile.context_window, Some(400_000));
        assert_eq!(profile.prompt_suffix.as_deref(), Some("exact suffix"));
        assert_eq!(
            profile.excluded_tools,
            HashSet::from(["shell".to_string(), "browser".to_string()])
        );
    }

    /// 验证提供商默认覆盖无需模糊匹配模型即可生效。
    #[test]
    fn provider_default_applies_without_fuzzy_model_matching() {
        let resolver = ModelProfileResolver::new().register_provider_default(
            "ollama",
            ModelProfileOverride {
                supports_structured_output: Some(true),
                ..Default::default()
            },
        );
        let profile = resolver.resolve("ollama", "local/custom", ProviderCapabilities::ollama());
        assert!(profile.capabilities.structured_output);
        assert!(!profile.supports_tool_choice_none);
    }
}
