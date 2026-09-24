//! LLM 客户端核心 trait 及请求/响应类型

pub mod cache;
pub mod capabilities;
pub mod thinking;
pub mod types;

use crate::error::Result;
pub use thinking::{ThinkingConfig, ThinkingLevel, ThinkingProtocol};
pub use types::{
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, DeltaMessage, FunctionCall,
    FunctionSpec, JsonSchemaSpec, Message, ResponseFormat, Role, ThinkingTypeBlock, ToolCall,
    ToolDefinition,
};

use futures::future::BoxFuture;
use futures::stream::BoxStream;
use std::time::Duration;
use tokio_util::sync::CancellationToken;

/// 非流式请求的默认超时时间。
const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(120);
/// 从请求发出到首个响应字节到达的默认超时时间。
const DEFAULT_FIRST_CHUNK_TIMEOUT: Duration = Duration::from_secs(30);
/// 流式响应中相邻数据块之间的默认最大空闲时间。
const DEFAULT_IDLE_TIMEOUT: Duration = Duration::from_secs(60);

/// 用于以毫秒数序列化/反序列化 `Option<Duration>` 的 serde 辅助模块。
mod optional_duration_millis {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::Duration;

    /// 将 `Option<Duration>` 序列化为毫秒数（`None` 序列化为 `null`）。
    pub fn serialize<S>(value: &Option<Duration>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        value
            .map(|duration| u64::try_from(duration.as_millis()).unwrap_or(u64::MAX))
            .serialize(serializer)
    }

    /// 从毫秒数反序列化为 `Option<Duration>`；值为 0 时视为未设置（`None`）。
    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<Duration>, D::Error>
    where
        D: Deserializer<'de>,
    {
        Ok(Option::<u64>::deserialize(deserializer)?
            .filter(|millis| *millis > 0)
            .map(Duration::from_millis))
    }
}

/// 非流式请求超时的默认值提供函数（供 serde `default` 使用）。
fn default_request_timeout() -> Option<Duration> {
    Some(DEFAULT_REQUEST_TIMEOUT)
}

/// 首字节超时的默认值提供函数（供 serde `default` 使用）。
fn default_first_chunk_timeout() -> Option<Duration> {
    Some(DEFAULT_FIRST_CHUNK_TIMEOUT)
}

/// 空闲超时的默认值提供函数（供 serde `default` 使用）。
fn default_idle_timeout() -> Option<Duration> {
    Some(DEFAULT_IDLE_TIMEOUT)
}

/// 将超时值归一化：零值视为“禁用”并转换为 `None`。
fn normalize_timeout(timeout: Duration) -> Option<Duration> {
    (!timeout.is_zero()).then_some(timeout)
}

/// 与具体 provider 无关的 LLM 请求及流式响应超时策略。
///
/// `request` 应用于非流式调用。流式调用则使用独立的首字节、空闲和
/// 整体边界，以避免将一个健康的长流误判为请求卡死。边界值为 `None`
/// 表示禁用该超时；零值会被归一化为 `None`。序列化后的值以毫秒为单位。
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct LlmTimeouts {
    /// 非流式请求的超时时间。
    #[serde(default = "default_request_timeout", with = "optional_duration_millis")]
    request: Option<Duration>,
    /// 从请求发出到首个响应字节到达的超时时间。
    #[serde(
        default = "default_first_chunk_timeout",
        with = "optional_duration_millis"
    )]
    first_chunk: Option<Duration>,
    /// 流式响应中相邻数据块之间的最大空闲时间。
    #[serde(default = "default_idle_timeout", with = "optional_duration_millis")]
    idle: Option<Duration>,
    /// 覆盖整个流（包括启动阶段）的总体超时时间。
    #[serde(default, with = "optional_duration_millis")]
    overall: Option<Duration>,
}

impl Default for LlmTimeouts {
    fn default() -> Self {
        Self {
            request: default_request_timeout(),
            first_chunk: default_first_chunk_timeout(),
            idle: default_idle_timeout(),
            overall: None,
        }
    }
}

impl LlmTimeouts {
    /// 返回非流式请求的超时时间。
    pub fn request_timeout(self) -> Option<Duration> {
        self.request
    }

    /// 返回从请求开始到首个响应字节到达的超时时间。
    pub fn first_chunk_timeout(self) -> Option<Duration> {
        self.first_chunk
    }

    /// 返回流式数据块之间允许的最大空闲时长。
    pub fn idle_timeout(self) -> Option<Duration> {
        self.idle
    }

    /// 返回覆盖整个流（包含启动阶段）的超时时间。
    pub fn overall_timeout(self) -> Option<Duration> {
        self.overall
    }

    /// 设置非流式请求的超时时间；传入零值将禁用该超时。
    pub fn with_request_timeout(mut self, timeout: Duration) -> Self {
        self.request = normalize_timeout(timeout);
        self
    }

    /// 禁用非流式请求的超时。
    pub fn without_request_timeout(mut self) -> Self {
        self.request = None;
        self
    }

    /// 设置请求到首字节的超时时间；传入零值将禁用该超时。
    pub fn with_first_chunk_timeout(mut self, timeout: Duration) -> Self {
        self.first_chunk = normalize_timeout(timeout);
        self
    }

    /// 禁用请求到首字节的超时。
    pub fn without_first_chunk_timeout(mut self) -> Self {
        self.first_chunk = None;
        self
    }

    /// 设置数据块间的空闲超时时间；传入零值将禁用该超时。
    pub fn with_idle_timeout(mut self, timeout: Duration) -> Self {
        self.idle = normalize_timeout(timeout);
        self
    }

    /// 禁用数据块间的空闲超时。
    pub fn without_idle_timeout(mut self) -> Self {
        self.idle = None;
        self
    }

    /// 设置覆盖整个流的超时时间；传入零值将禁用该超时。
    pub fn with_overall_timeout(mut self, timeout: Duration) -> Self {
        self.overall = normalize_timeout(timeout);
        self
    }

    /// 禁用覆盖整个流的超时。
    pub fn without_overall_timeout(mut self) -> Self {
        self.overall = None;
        self
    }
}

#[cfg(test)]
mod timeout_tests {
    use super::LlmTimeouts;
    use std::time::Duration;

    #[test]
    fn timeout_defaults_and_millisecond_wire_shape_are_stable() -> Result<(), String> {
        let defaults = LlmTimeouts::default();
        assert_eq!(defaults.request_timeout(), Some(Duration::from_secs(120)));
        assert_eq!(
            defaults.first_chunk_timeout(),
            Some(Duration::from_secs(30))
        );
        assert_eq!(defaults.idle_timeout(), Some(Duration::from_secs(60)));
        assert_eq!(defaults.overall_timeout(), None);

        let configured = defaults
            .with_request_timeout(Duration::from_millis(2_500))
            .with_first_chunk_timeout(Duration::from_millis(750))
            .without_idle_timeout()
            .with_overall_timeout(Duration::from_secs(10));
        let encoded = serde_json::to_value(configured).map_err(|error| error.to_string())?;
        assert_eq!(
            encoded.get("request").and_then(serde_json::Value::as_u64),
            Some(2_500)
        );
        assert_eq!(
            encoded
                .get("first_chunk")
                .and_then(serde_json::Value::as_u64),
            Some(750)
        );
        assert!(encoded.get("idle").is_some_and(serde_json::Value::is_null));
        assert_eq!(
            encoded.get("overall").and_then(serde_json::Value::as_u64),
            Some(10_000)
        );
        Ok(())
    }

    #[test]
    fn zero_millisecond_timeouts_deserialize_as_disabled() -> Result<(), String> {
        let decoded = serde_json::from_value::<LlmTimeouts>(serde_json::json!({
            "request": 0,
            "first_chunk": 0,
            "idle": 0,
            "overall": 0
        }))
        .map_err(|error| error.to_string())?;
        assert_eq!(decoded.request_timeout(), None);
        assert_eq!(decoded.first_chunk_timeout(), None);
        assert_eq!(decoded.idle_timeout(), None);
        assert_eq!(decoded.overall_timeout(), None);
        Ok(())
    }
}

/// 单个已配置模型所接受的输入模态。
///
/// Provider 传输层描述内容是如何编码的；而这个模型级别的集合
/// 描述的是某个具体模型被允许接收哪些类型的输入。
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum ModelInputModality {
    /// 文本输入。
    Text,
    /// 图像输入。
    Image,
    /// 音频输入。
    Audio,
    /// 视频输入。
    Video,
}

impl ModelInputModality {
    /// 仅文本的默认集合，用于对未知模型保持保守假设。
    pub fn text_only() -> Vec<Self> {
        vec![Self::Text]
    }

    /// 保留框架构造函数历史上的多模态行为（支持全部模态）。
    pub fn all_supported() -> Vec<Self> {
        vec![Self::Text, Self::Image, Self::Audio, Self::Video]
    }
}

/// LLM provider 端点所使用的 HTTP API 协议。
///
/// 这一概念被有意与 provider 身份区分开来。一个 provider 会选择一个
/// 默认协议，而自定义或兼容端点则可以显式选择任意受支持的线上格式。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LlmApiProtocol {
    /// 兼容 OpenAI 的 `/chat/completions` 请求与流式响应模式。
    #[default]
    ChatCompletions,
    /// OpenAI Responses 请求及其语义化 SSE 事件模式。
    Responses,
    /// Anthropic Messages 请求及其语义化 SSE 事件模式。
    Anthropic,
}

impl LlmApiProtocol {
    /// 附加在 provider API 根路径之后的标准端点路径。
    pub const fn endpoint_path(self) -> &'static str {
        match self {
            Self::ChatCompletions => "chat/completions",
            Self::Responses => "responses",
            Self::Anthropic => "messages",
        }
    }

    /// 根据一个完整的端点 URL 识别其所使用的协议。
    ///
    /// 对于像 `https://api.example.com/v1` 这样的 provider 根路径会返回
    /// `None`，因为从一个不完整的路径中猜测线上协议，会把 Chat
    /// Completions 这个兜底选项变成一个误判。
    pub fn try_from_endpoint(endpoint: &str) -> Option<Self> {
        let path = endpoint
            .split(['?', '#'])
            .next()
            .unwrap_or(endpoint)
            .trim_end_matches('/');
        if path.ends_with("/responses") {
            Some(Self::Responses)
        } else if path.ends_with("/messages") {
            Some(Self::Anthropic)
        } else if path.ends_with("/chat/completions") {
            Some(Self::ChatCompletions)
        } else {
            None
        }
    }

    /// 从一个完整的端点 URL 推断协议。
    pub fn from_endpoint(endpoint: &str) -> Self {
        if let Some(protocol) = Self::try_from_endpoint(endpoint) {
            protocol
        } else if endpoint
            .split(['?', '#'])
            .next()
            .unwrap_or(endpoint)
            .contains("anthropic.com/")
        {
            Self::Anthropic
        } else {
            Self::ChatCompletions
        }
    }
}

#[cfg(test)]
mod protocol_tests {
    use super::LlmApiProtocol;

    #[test]
    fn infers_all_supported_wire_protocols() {
        assert_eq!(
            LlmApiProtocol::from_endpoint("https://api.openai.com/v1/responses?trace=true"),
            LlmApiProtocol::Responses
        );
        assert_eq!(
            LlmApiProtocol::from_endpoint("https://api.anthropic.com/v1/messages"),
            LlmApiProtocol::Anthropic
        );
        assert_eq!(
            LlmApiProtocol::from_endpoint("https://gateway.example/v1/chat/completions"),
            LlmApiProtocol::ChatCompletions
        );
    }

    #[test]
    fn complete_endpoint_detection_does_not_guess_from_provider_roots() {
        assert_eq!(
            LlmApiProtocol::try_from_endpoint("https://gateway.example/v1/responses?trace=true"),
            Some(LlmApiProtocol::Responses)
        );
        assert_eq!(
            LlmApiProtocol::try_from_endpoint("https://gateway.example/v1/messages#debug"),
            Some(LlmApiProtocol::Anthropic)
        );
        assert_eq!(
            LlmApiProtocol::try_from_endpoint("https://gateway.example/v1/chat/completions/"),
            Some(LlmApiProtocol::ChatCompletions)
        );
        assert_eq!(
            LlmApiProtocol::try_from_endpoint("https://gateway.example/v1"),
            None
        );
        assert_eq!(
            LlmApiProtocol::from_endpoint(
                "https://gateway.example/v1?upstream=https://api.anthropic.com/v1/"
            ),
            LlmApiProtocol::ChatCompletions
        );
    }
}

/// [`LlmClient::chat_simple_with_options`] 的可选参数。
#[derive(Debug, Clone)]
pub struct SimpleChatOptions {
    /// 采样温度（0.0–2.0）。默认为 `0.7`。
    pub temperature: Option<f32>,
    /// 最大生成 token 数。默认为 `2048`。
    pub max_tokens: Option<u32>,
}

impl Default for SimpleChatOptions {
    fn default() -> Self {
        Self {
            temperature: Some(0.7),
            max_tokens: Some(2048),
        }
    }
}

impl SimpleChatOptions {
    /// 创建带有指定温度值的选项。
    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = Some(temp);
        self
    }

    /// 创建带有指定 max_tokens 的选项。
    pub fn with_max_tokens(mut self, max: u32) -> Self {
        self.max_tokens = Some(max);
        self
    }
}

/// LLM 客户端的统一接口
pub trait LlmClient: Send + Sync {
    /// 执行一次非流式的聊天请求。
    fn chat(&self, request: ChatRequest) -> BoxFuture<'_, Result<ChatResponse>>;

    /// 执行一次流式的聊天请求。
    ///
    /// 返回的流是 `'static` 的（不借用 `&self`）：所有 provider
    /// 实现都会生成一个拥有所有权的、装箱的流，这样调用方就可以将它
    /// 跨 await 点移动 / 移入被 spawn 的任务中。这也使得流式核心循环
    /// 能够通过一个 trait object（`Arc<dyn LlmClient>`）进行路由，
    /// 并让返回的流的生命周期超出该 trait 方法调用本身的借用 ——
    /// 这正是让该循环可以用一个 mock 客户端进行测试的关键所在。
    fn chat_stream(
        &self,
        request: ChatRequest,
    ) -> BoxFuture<'_, Result<BoxStream<'static, Result<ChatChunk>>>>;

    /// 用于简单纯文本调用的便捷辅助方法。
    ///
    /// 可通过 [`SimpleChatOptions`] 传入可选的覆盖参数；默认值为
    /// `temperature: 0.7`、`max_tokens: 2048`。
    fn chat_simple(&self, messages: Vec<Message>) -> BoxFuture<'_, Result<String>> {
        self.chat_simple_with_options(messages, SimpleChatOptions::default())
    }

    /// 带可配置选项的便捷辅助方法。
    fn chat_simple_with_options(
        &self,
        messages: Vec<Message>,
        options: SimpleChatOptions,
    ) -> BoxFuture<'_, Result<String>> {
        Box::pin(async move {
            let response = self
                .chat(ChatRequest {
                    messages,
                    temperature: options.temperature,
                    max_tokens: options.max_tokens,
                    ..Default::default()
                })
                .await?;
            Ok(response.content().unwrap_or_default().to_string())
        })
    }

    /// 该客户端使用的模型标识符。
    fn model_name(&self) -> &str;

    /// 该客户端对应的 provider 能力集。
    ///
    /// 默认返回兼容 OpenAI 的能力集。对于 Anthropic、Ollama 或
    /// 自定义 provider，需要重写此方法。
    fn capabilities(&self) -> capabilities::ProviderCapabilities {
        capabilities::ProviderCapabilities::openai_compatible()
    }
}

/// 控制模型在推理过程中如何使用工具。
///
/// 映射到各 provider 特定的 API：
/// - OpenAI：`"auto"`、`"none"`、`"required"`、`{"type": "function", "function": {"name": "..."}}`
/// - Anthropic：`{"type": "auto"}`、`{"type": "any"}`、`{"type": "tool", "name": "..."}`
#[derive(Debug, Clone, Default, PartialEq)]
pub enum ToolChoice {
    /// 由模型自行决定是否调用工具（默认值）。
    #[default]
    Auto,
    /// 强制模型不调用任何工具。
    None,
    /// 要求模型至少调用一个工具。
    Required,
    /// 强制模型调用指定名称的工具。
    Function {
        /// 被强制调用的工具名称。
        name: String,
    },
}

impl ToolChoice {
    /// 创建一个强制调用指定函数的 ToolChoice。
    pub fn function(name: impl Into<String>) -> Self {
        Self::Function { name: name.into() }
    }

    /// 转换为 OpenAI 的线上格式（`"auto"`、`"none"`、`"required"`，
    /// 或 `{"type":"function","function":{"name":"..."}}`）。
    pub fn to_openai_value(&self) -> serde_json::Value {
        match self {
            Self::Auto => serde_json::Value::String("auto".into()),
            Self::None => serde_json::Value::String("none".into()),
            Self::Required => serde_json::Value::String("required".into()),
            Self::Function { name } => serde_json::json!({
                "type": "function",
                "function": {"name": name}
            }),
        }
    }
}

/// 聊天请求参数
#[derive(Debug, Clone, Default)]
pub struct ChatRequest {
    /// 发送给模型的有序聊天历史。
    pub messages: Vec<Message>,
    /// 可选的采样温度。
    pub temperature: Option<f32>,
    /// 可选的响应 token 数上限。
    pub max_tokens: Option<u32>,
    /// 可选的、暴露给模型的工具定义。
    pub tools: Option<Vec<ToolDefinition>>,
    /// 控制模型如何使用工具（OpenAI 线上格式）。
    /// 优先使用 [`ToolChoice`] 以获得类型安全的构造方式；在构建
    /// [`ChatCompletionRequest`] 时通过 [`ToolChoice::to_openai_value`] 转换。
    pub tool_choice: Option<String>,
    /// 可选的结构化输出格式提示。
    pub response_format: Option<ResponseFormat>,
    /// 可选的推理深度 / “思考”控制。
    ///
    /// 设置该字段后，各 provider 会将其转换为自身原生的思考线上字段
    /// （OpenAI 推理模型对应 `reasoning_effort`，Claude 3.7–4.5 对应
    /// `thinking.budget_tokens`，Qwen3/GLM 对应 `enable_thinking`）。
    /// 不支持思考控制的模型会静默丢弃该字段（并输出一条 `warn!` 日志）。
    /// `None`（默认值）表示“使用模型的默认行为” —— 不发送该字段。
    pub thinking: Option<ThinkingConfig>,
    /// 用于中止进行中请求的可选取消令牌。
    /// 设置并被取消后，流式响应会在下一个 SSE 边界处停止，非流式请求
    /// 则会在等待响应头、读取响应体或解析 JSON 响应的过程中中止。
    pub cancel_token: Option<CancellationToken>,
    /// 本次调用的可选超时覆盖值。
    ///
    /// `None` 表示使用 provider 客户端的默认配置。同一个值同时
    /// 控制非流式与流式的超时边界。
    pub timeouts: Option<LlmTimeouts>,
    /// 用于 KVCache 隔离的可选用户标识符（DeepSeek 等场景使用）。
    /// 一个稳定的、绑定到会话的 ID 可以让 provider 在多次请求间
    /// 复用 prompt 缓存条目。若不设置，缓存命中率可能会降到 1% 以下。
    pub user_id: Option<String>,
    /// 可选的缓存提示（断点目标、稳定前缀哈希、分段范围）。
    /// Provider 使用这些信息来放置缓存断点并记录诊断日志。
    pub cache_hints: Option<crate::llm::cache::CacheHints>,
}

impl ChatRequest {
    /// 从消息列表创建一个请求。
    pub fn new(messages: Vec<Message>) -> Self {
        Self {
            messages,
            ..Default::default()
        }
    }

    /// 为请求附加工具定义。
    pub fn with_tools(mut self, tools: Vec<ToolDefinition>) -> Self {
        self.tools = Some(tools);
        self
    }

    /// 为本次请求覆盖 provider 客户端的超时策略。
    pub fn with_timeouts(mut self, timeouts: LlmTimeouts) -> Self {
        self.timeouts = Some(timeouts);
        self
    }
}

/// 聊天响应
#[derive(Debug, Clone)]
pub struct ChatResponse {
    /// provider 返回的主助手消息。
    pub message: Message,
    /// provider 特定的结束原因。
    pub finish_reason: Option<String>,
    /// 经过 provider 归一化的 token 用量，与 [`ChatChunk::usage`] 对称。
    pub usage: Option<types::Usage>,
    /// 原始的 provider 响应，供需要额外元数据的调用方使用。
    pub raw: ChatCompletionResponse,
}

impl ChatResponse {
    /// 提取助手文本内容。
    pub fn content(&self) -> Option<String> {
        self.message.content.as_text()
    }

    /// 若助手发出了工具调用，则借用其引用。
    pub fn tool_calls(&self) -> Option<&Vec<ToolCall>> {
        self.message.tool_calls.as_ref()
    }

    /// 该响应是否包含至少一个工具调用。
    pub fn has_tool_calls(&self) -> bool {
        self.message
            .tool_calls
            .as_ref()
            .is_some_and(|t| !t.is_empty())
    }
}

/// 流式响应数据块
#[derive(Debug, Clone)]
pub struct ChatChunk {
    /// 增量消息差异。
    pub delta: DeltaMessage,
    /// 随该数据块一起发出的结束原因（若有）。
    pub finish_reason: Option<String>,
    /// token 用量（仅当设置了 stream_options.include_usage 时，出现在最后一个数据块中）。
    pub usage: Option<types::Usage>,
}
