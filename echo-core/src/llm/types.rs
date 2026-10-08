//! OpenAI Chat Completions API 类型定义

use crate::tools::Tool;
use serde::{Deserialize, Serialize};

// ── 多模态内容 ────────────────────────────────────────────────────────

/**
结构化消息内容中与提供方无关的一个组成部分。

当供应商适配层的线上 API 原生不支持某个变体时，由适配层负责转换；
框架自身在会话状态中始终保留其类型化的值。
*/
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    /// 纯文本
    Text {
        /// 文本内容
        text: String,
    },
    /// 图片（Base64 编码或 URL）
    ImageUrl {
        /// 图片 URL 或 Base64 数据
        image_url: ImageUrl,
    },
    /// 文件附件（内联 Base64）
    File {
        /// 文件名
        name: String,
        /// 文件内容（Base64 编码）
        content: String,
    },
    /// 指向 Agent 可检查资源的类型化链接。
    ResourceLink {
        /// 无损、与提供方无关的资源元数据。
        resource: Box<LinkedResource>,
    },
}

/// 与提供方无关的推理块，多轮回放时需要。
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ReasoningBlock {
    /// 附带提供方签名的推理文本。
    Signed {
        /// 推理文本。
        thinking: String,
        /// 提供方对推理文本的签名。
        signature: String,
    },
    /// 必须原样回放的不透明加密推理内容。
    Redacted {
        /// 加密后的推理数据。
        data: String,
    },
    /**
    提供方持有的不透明推理状态，必须原样回放。

    当应用在本地管理会话历史且使用 `store: false` 时，
    Responses 接口要求同时提供 item id 和加密后的 payload。
    */
    Opaque {
        /// 产生该推理状态的提供方标识。
        provider: String,
        /// 提供方分配的推理 item id。
        id: String,
        /// 加密后的推理 payload。
        data: String,
        /// 可选的推理摘要片段。
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        summary: Vec<String>,
    },
}

/// 图片 URL 或 Base64 数据
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ImageUrl {
    /// `data:image/png;base64,...` 或 `https://...`
    pub url: String,
    /// 可选的细节等级：`"auto"` | `"low"` | `"high"`
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/**
结构化用户消息中携带的、与提供方无关的链接资源。

协议适配层在到达供应商边界之前会一直保留其类型化的形态。对于没有原生
resource-link 块的供应商，会渲染成确定性文本；而框架内的 Agent 可以在
不解析嵌入用户文本中的私有标记的情况下检查每一个字段。
*/
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct LinkedResource {
    /// 协议提供的可选展示与受众标注信息。
    pub annotations: Option<serde_json::Value>,
    /// 可选的人类可读详情。
    pub description: Option<String>,
    /// 链接内容的可选 MIME 类型。
    pub mime_type: Option<String>,
    /// 资源的人类可读名称。
    pub name: String,
    /// 资源的可选大小（字节）。
    pub size: Option<i64>,
    /// 可选的展示标题。
    pub title: Option<String>,
    /// 与适配层输入完全一致保留的资源 URI。
    pub uri: String,
    /// 可选的协议扩展元数据。
    pub meta: Option<serde_json::Value>,
}

impl LinkedResource {
    /**
    渲染一段确定性的供应商兜底文本，不改变 Session 历史中存储的
    类型化框架消息。
    */
    #[doc(hidden)]
    pub fn model_text(&self) -> String {
        let encoded = serde_json::to_string(self)
            .unwrap_or_else(|_| "{\"resource_link\":\"serialization_failed\"}".to_string());
        format!("[Linked resource]\n{encoded}")
    }
}

/**
消息内容：同时兼容纯文本和多模态 parts 两种形式。

序列化行为：
- `Text("hello")` → `"hello"`（与旧版 API 完全向后兼容）
- `Parts([...])` → `[{"type":"text","text":"..."},...]`
*/
#[derive(Debug, Clone, Default)]
pub enum MessageContent {
    /// 纯文本内容。
    Text(String),
    /// 多模态内容 parts。
    Parts(Vec<ContentPart>),
    /// 显式的空内容负载。
    #[default]
    Empty,
}

impl Serialize for MessageContent {
    /// 纯文本序列化为字符串，多模态序列化为 parts 数组，空内容序列化为空字符串。
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            MessageContent::Text(s) => serializer.serialize_str(s),
            MessageContent::Parts(parts) => parts.serialize(serializer),
            MessageContent::Empty => serializer.serialize_str(""),
        }
    }
}

impl<'de> Deserialize<'de> for MessageContent {
    /// 字符串解析为文本（空串视为空内容），数组解析为 parts，其余类型视为空内容。
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = serde_json::Value::deserialize(deserializer)?;
        match value {
            serde_json::Value::String(s) => {
                if s.is_empty() {
                    Ok(MessageContent::Empty)
                } else {
                    Ok(MessageContent::Text(s))
                }
            }
            serde_json::Value::Array(_) => {
                let parts: Vec<ContentPart> =
                    serde_json::from_value(value).map_err(serde::de::Error::custom)?;
                Ok(MessageContent::Parts(parts))
            }
            _ => Ok(MessageContent::Empty),
        }
    }
}

impl MessageContent {
    /// 提取纯文本内容（多模态时拼接所有 Text part）
    pub fn as_text(&self) -> Option<String> {
        match self {
            MessageContent::Text(s) => {
                if s.is_empty() {
                    None
                } else {
                    Some(s.clone())
                }
            }
            MessageContent::Parts(parts) => {
                let texts: Vec<String> = parts
                    .iter()
                    .filter_map(|p| match p {
                        ContentPart::Text { text } => Some(text.clone()),
                        ContentPart::ResourceLink { resource } => Some(resource.model_text()),
                        _ => None,
                    })
                    .collect();
                if texts.is_empty() {
                    None
                } else {
                    Some(texts.join(""))
                }
            }
            MessageContent::Empty => None,
        }
    }

    /// 以引用形式借出文本内容（仅对 Text 变体有效）。
    pub fn as_text_ref(&self) -> Option<&str> {
        match self {
            MessageContent::Text(s) if !s.is_empty() => Some(s),
            _ => None,
        }
    }

    /// 与旧版 `Option<String>::as_deref()` 调用点保持向后兼容。
    pub fn as_deref(&self) -> Option<&str> {
        self.as_text_ref()
    }

    /// 当内容以 `Parts` 存储时，借出多模态 parts。
    pub fn parts(&self) -> Option<&[ContentPart]> {
        match self {
            MessageContent::Parts(parts) => Some(parts.as_slice()),
            _ => None,
        }
    }

    /// 消费该内容，若存在则返回多模态 parts。
    pub fn into_parts(self) -> Option<Vec<ContentPart>> {
        match self {
            MessageContent::Parts(parts) => Some(parts),
            _ => None,
        }
    }

    /**
    保守估算提供方可见的输入 token 数，包括被 [`Self::as_text`]
    刻意忽略的多模态负载。
    */
    pub fn estimated_tokens(&self, tokenizer: &dyn crate::tokenizer::Tokenizer) -> usize {
        match self {
            MessageContent::Text(text) => tokenizer.count_tokens(text),
            MessageContent::Empty => 0,
            MessageContent::Parts(parts) => parts.iter().fold(0usize, |total, part| {
                let part_tokens = match part {
                    ContentPart::Text { text } => tokenizer.count_tokens(text),
                    ContentPart::ImageUrl { image_url } => {
                        if image_url.url.starts_with("data:") {
                            /*
                            Base64 大约每 4 个字符对应 3 字节；按每 4 个字符
                            计 1 个 token 是一个保守估计，在受支持的视觉
                            供应商上都适用。
                            */
                            image_url.url.chars().count().div_ceil(4)
                        } else if image_url.detail.as_deref() == Some("low") {
                            100
                        } else {
                            1_100
                        }
                    }
                    ContentPart::File { name, content } => tokenizer
                        .count_tokens(name)
                        .saturating_add(content.chars().count().div_ceil(4)),
                    ContentPart::ResourceLink { resource } => {
                        tokenizer.count_tokens(&resource.model_text())
                    }
                };
                total.saturating_add(part_tokens)
            }),
        }
    }
}

// ── 消息角色 ────────────────────────────────────────────────────────────

/**
带类型的消息角色，兼容旧版序列化格式。

标准角色（`system`、`user`、`assistant`、`tool`）是类型安全的变体。
供应商特有或未知的角色落入 `Custom(String)`。
*/
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum Role {
    /// 系统指令消息
    System,
    /// 用户输入消息
    #[default]
    User,
    /// 助手响应消息
    Assistant,
    /// 工具结果消息
    Tool,
    /// 供应商特有或未知的角色
    Custom(String),
}

impl Role {
    /// 返回用于 API 序列化的角色字符串。
    pub fn as_str(&self) -> &str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
            Role::Custom(s) => s.as_str(),
        }
    }
}

impl Serialize for Role {
    /// 以角色的线上字符串形式序列化。
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for Role {
    /// 从字符串反序列化，标准角色映射为对应变体，其余映射为 `Custom`。
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        match s.as_str() {
            "system" => Ok(Role::System),
            "user" => Ok(Role::User),
            "assistant" => Ok(Role::Assistant),
            "tool" => Ok(Role::Tool),
            _ => Ok(Role::Custom(s)),
        }
    }
}

impl From<Role> for String {
    /// 将角色转换为其线上字符串。
    fn from(role: Role) -> String {
        role.as_str().to_string()
    }
}

impl From<String> for Role {
    /// 从字符串解析角色，未知角色映射为 `Custom`。
    fn from(s: String) -> Self {
        match s.as_str() {
            "system" => Role::System,
            "user" => Role::User,
            "assistant" => Role::Assistant,
            "tool" => Role::Tool,
            _ => Role::Custom(s),
        }
    }
}

impl From<&str> for Role {
    /// 从字符串切片解析角色，未知角色映射为 `Custom`。
    fn from(s: &str) -> Self {
        match s {
            "system" => Role::System,
            "user" => Role::User,
            "assistant" => Role::Assistant,
            "tool" => Role::Tool,
            _ => Role::Custom(s.to_string()),
        }
    }
}

// ── 会话消息 ─────────────────────────────────────────────────────────

/**
会话消息

`content` 字段统一使用 [`MessageContent`] 枚举，支持纯文本与多模态内容。
*/
#[derive(Debug, Clone, Default)]
pub struct Message {
    /// 消息角色：`system`、`user`、`assistant`、`tool`，或 `Custom`。
    pub role: Role,
    /// 文本或多模态负载。
    pub content: MessageContent,
    /// 助手发出的可选工具调用。
    pub tool_calls: Option<Vec<ToolCall>>,
    /// 可选的参与者名称。
    pub name: Option<String>,
    /// tool 消息的可选工具调用标识符。
    pub tool_call_id: Option<String>,
    /// 来自 Qwen3/DeepSeek 等模型的推理内容（思考过程）。
    pub reasoning_content: Option<String>,
    /// 保证供应商侧回放有效所需的签名或已脱敏推理块。
    pub reasoning_blocks: Option<Vec<ReasoningBlock>>,
}

impl Serialize for Message {
    /// 以 OpenAI 兼容的 map 形式序列化，省略值为空的可选字段。
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let mut map = serializer.serialize_map(None)?;
        map.serialize_entry("role", &self.role)?;
        // 使用 MessageContent 自身的序列化逻辑来序列化 content
        if !matches!(&self.content, MessageContent::Empty) {
            map.serialize_entry("content", &self.content)?;
        } else if self.tool_calls.is_some() {
            // assistant 消息带 tool_calls 但没有文本内容时，仍需要 content 字段
            map.serialize_entry("content", &serde_json::Value::Null)?;
        }
        if let Some(ref tc) = self.tool_calls {
            map.serialize_entry("tool_calls", tc)?;
        }
        if let Some(ref name) = self.name {
            map.serialize_entry("name", name)?;
        }
        if let Some(ref id) = self.tool_call_id {
            map.serialize_entry("tool_call_id", id)?;
        }
        if let Some(ref rc) = self.reasoning_content {
            map.serialize_entry("reasoning_content", rc)?;
        }
        if let Some(ref blocks) = self.reasoning_blocks {
            map.serialize_entry("reasoning_blocks", blocks)?;
        }
        map.end()
    }
}

impl<'de> Deserialize<'de> for Message {
    /// 宽松地反序列化消息：缺失字段取默认值，`content` 兼容字符串与 parts 数组。
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        /// 反序列化中间结构，字段与线上格式一一对应。
        #[derive(Deserialize)]
        struct RawMessage {
            #[serde(default)]
            role: String,
            #[serde(default)]
            content: Option<serde_json::Value>,
            #[serde(default)]
            tool_calls: Option<Vec<ToolCall>>,
            #[serde(default)]
            name: Option<String>,
            #[serde(default)]
            tool_call_id: Option<String>,
            #[serde(default)]
            reasoning_content: Option<String>,
            #[serde(default)]
            reasoning_blocks: Option<Vec<ReasoningBlock>>,
        }
        let raw = RawMessage::deserialize(deserializer)?;
        let role: Role = serde_json::from_value(serde_json::Value::String(raw.role))
            .map_err(serde::de::Error::custom)?;
        let content = match raw.content {
            Some(value) => MessageContent::deserialize(value).map_err(serde::de::Error::custom)?,
            None => MessageContent::Empty,
        };
        Ok(Message {
            role,
            content,
            tool_calls: raw.tool_calls,
            name: raw.name,
            tool_call_id: raw.tool_call_id,
            reasoning_content: raw.reasoning_content,
            reasoning_blocks: raw.reasoning_blocks,
        })
    }
}

impl Message {
    /// 创建一条系统消息
    pub fn system(content: String) -> Self {
        Self {
            role: Role::System,
            content: MessageContent::Text(content),
            tool_calls: None,
            name: None,
            tool_call_id: None,
            reasoning_content: None,
            reasoning_blocks: None,
        }
    }

    /// 创建一条用户消息
    pub fn user(content: String) -> Self {
        Self {
            role: Role::User,
            content: MessageContent::Text(content),
            tool_calls: None,
            name: None,
            tool_call_id: None,
            reasoning_content: None,
            reasoning_blocks: None,
        }
    }

    /// 创建一条多模态用户消息
    pub fn user_multimodal(parts: Vec<ContentPart>) -> Self {
        Self {
            role: Role::User,
            content: MessageContent::Parts(parts),
            tool_calls: None,
            name: None,
            tool_call_id: None,
            reasoning_content: None,
            reasoning_blocks: None,
        }
    }

    /**
    创建一条包含图片的用户消息

    # 示例

    ```rust
    use echo_core::llm::types::Message;

    let msg = Message::user_with_image(
        "Please describe this image",
        "image/png",
        "iVBORw0KGgo...", // base64 数据
    );
    ```
    */
    pub fn user_with_image(text: &str, media_type: &str, base64_data: &str) -> Self {
        Self::user_multimodal(vec![
            ContentPart::Text {
                text: text.to_string(),
            },
            ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: format!("data:{media_type};base64,{base64_data}"),
                    detail: None,
                },
            },
        ])
    }

    /// 创建一条带图片 URL 的用户消息
    pub fn user_with_image_url(text: &str, image_url: &str) -> Self {
        Self::user_multimodal(vec![
            ContentPart::Text {
                text: text.to_string(),
            },
            ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: image_url.to_string(),
                    detail: None,
                },
            },
        ])
    }

    /// 创建一条助手消息
    pub fn assistant(content: String) -> Self {
        Self {
            role: Role::Assistant,
            content: MessageContent::Text(content),
            tool_calls: None,
            name: None,
            tool_call_id: None,
            reasoning_content: None,
            reasoning_blocks: None,
        }
    }

    /// 创建一条带工具调用的助手消息
    pub fn assistant_with_tools(tool_calls: Vec<ToolCall>) -> Self {
        Self {
            role: Role::Assistant,
            content: MessageContent::Empty,
            tool_calls: Some(tool_calls),
            name: None,
            tool_call_id: None,
            reasoning_content: None,
            reasoning_blocks: None,
        }
    }

    /// 创建一条工具结果消息
    pub fn tool_result(tool_call_id: String, name: String, content: String) -> Self {
        Self {
            role: Role::Tool,
            content: MessageContent::Text(content),
            tool_calls: None,
            name: Some(name),
            tool_call_id: Some(tool_call_id),
            reasoning_content: None,
            reasoning_blocks: None,
        }
    }

    /**
    获取纯文本内容。

    若消息包含多模态内容，则提取并拼接所有 Text part。
    */
    pub fn text_content(&self) -> Option<String> {
        self.content.as_text()
    }

    /// 与直接读取 `content_parts` 的旧版调用点保持向后兼容。
    pub fn content_parts(&self) -> Option<&[ContentPart]> {
        self.content.parts()
    }

    /// 检查消息是否包含多模态内容
    pub fn is_multimodal(&self) -> bool {
        matches!(&self.content, MessageContent::Parts(_))
    }
}

/// LLM 发起的一次工具调用
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ToolCall {
    /// 该工具调用的唯一标识符
    pub id: String,
    /// 工具调用类型，通常为 "function"
    #[serde(rename = "type")]
    pub call_type: String,
    /// 函数调用详情
    pub function: FunctionCall,
}

/// 一次工具调用中的函数信息
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FunctionCall {
    /// 函数名
    pub name: String,
    /// 函数参数（JSON 字符串）
    pub arguments: String,
}

/// 用于结构化输出的 JSON Schema 规格
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct JsonSchemaSpec {
    /// JSON Schema 名称
    pub name: String,
    /// JSON Schema 定义
    pub schema: serde_json::Value,
    /// 是否启用严格校验（默认 true）
    #[serde(default = "default_true")]
    pub strict: bool,
}

/// 返回 `true`，供 `#[serde(default = "default_true")]` 使用。
fn default_true() -> bool {
    true
}

/// 响应格式控制
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseFormat {
    /// 纯文本响应
    Text,
    /// JSON 对象响应
    JsonObject,
    /// 受 JSON Schema 约束的响应
    JsonSchema {
        /// JSON Schema 规格
        json_schema: JsonSchemaSpec,
    },
}

impl ResponseFormat {
    /// 创建一个 JSON Schema 响应格式
    pub fn json_schema(name: impl Into<String>, schema: serde_json::Value) -> Self {
        Self::JsonSchema {
            json_schema: JsonSchemaSpec {
                name: name.into(),
                schema,
                strict: true,
            },
        }
    }

    /// 检查是否为 JSON 响应格式
    pub fn is_json(&self) -> bool {
        matches!(self, Self::JsonObject | Self::JsonSchema { .. })
    }
}

/// OpenAI `/chat/completions` 请求体
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ChatCompletionRequest {
    /// 模型名称
    pub model: String,
    /// 会话消息列表
    pub messages: Vec<Message>,
    /// 可选的工具定义列表
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolDefinition>>,
    /// 工具调用策略（如 "auto"、"none"，或指定工具名）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<String>,
    /// 采样温度（0.0-2.0）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// 生成的最大 token 数
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// 推理模型的输出上限。与 `max_tokens` 互斥。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,
    /// 是否启用流式响应
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// 流式响应选项（如 include_usage）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<serde_json::Value>,
    /// 响应格式控制
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
    /**
    OpenAI 推理模型（GPT-5 系列、o 系列）的 reasoning-effort。

    取值为 `"minimal"`、`"low"`、`"medium"`、`"high"` 之一。仅在解析出的
    模型支持 `ThinkingProtocol::OpenaiReasoningEffort` 时才会发送；
    发送给非推理模型会返回 400。
    */
    #[serde(skip_serializing_if = "Option::is_none", rename = "reasoning_effort")]
    pub reasoning_effort: Option<String>,
    /**
    Qwen3/GLM 的 `enable_thinking` 开关。仅对支持
    `ThinkingProtocol::EnableThinkingFlag` 的模型发送。
    */
    #[serde(skip_serializing_if = "Option::is_none", rename = "enable_thinking")]
    pub enable_thinking: Option<bool>,
    /**
    Qwen3 的 `thinking_budget`（整数 token 预算）。作为
    `enable_thinking` 的可选配套项，供部分 Qwen3 模型接受显式预算。
    */
    #[serde(skip_serializing_if = "Option::is_none", rename = "thinking_budget")]
    pub thinking_budget: Option<u32>,
    /// 通用的 `thinking:{type:"enabled"|"disabled"}` 块。
    #[serde(skip_serializing_if = "Option::is_none", rename = "thinking")]
    pub thinking_type: Option<ThinkingTypeBlock>,
    /// Ollama 原生的 `think` 扩展：布尔值或表示努力程度的字符串。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub think: Option<serde_json::Value>,
    /**
    用于 KVCache 隔离的用户标识（DeepSeek 等）。
    设置后，供应商用它来划分 prompt cache 条目。
    若不设置，每个请求都可能被视为来自不同用户，
    从而阻止同一会话内多次请求复用缓存。
    */
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
}

/// `thinking:{type:"enabled"|"disabled"}` 线上协议块。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinkingTypeBlock {
    /// 块类型：`"enabled"` 或 `"disabled"`。
    #[serde(rename = "type")]
    pub block_type: String,
}

/// 发送给 LLM 的工具定义
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ToolDefinition {
    /// 工具类型，通常为 "function"
    #[serde(rename = "type")]
    pub tool_type: String,
    /// 函数规格
    pub function: FunctionSpec,
}

/// 一个工具的函数声明
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FunctionSpec {
    /// 函数名
    pub name: String,
    /// 函数描述
    pub description: String,
    /// 函数参数的 JSON Schema
    pub parameters: serde_json::Value,
}

impl ToolDefinition {
    /// 从一个 Tool trait 对象创建工具定义
    pub fn from_tool(tool: &dyn Tool) -> Self {
        Self {
            tool_type: "function".to_string(),
            function: FunctionSpec {
                name: tool.name().to_string(),
                description: tool.description().to_string(),
                parameters: tool.parameters(),
            },
        }
    }
}

/// OpenAI chat completion 响应
#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct ChatCompletionResponse {
    /// 响应 ID
    #[serde(default)]
    pub id: String,
    /// 候选响应列表
    #[serde(default)]
    pub choices: Vec<Choice>,
    /// 创建时间戳（秒）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created: Option<u64>,
    /// 模型名称
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Token 使用统计
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    /// 响应中未被显式建模的额外字段
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extra: Option<serde_json::Value>,
}

/// 候选响应
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Choice {
    /// 消息内容
    pub message: Message,
    /// 结束原因（如 "stop"、"length"、"tool_calls" 等）
    #[serde(default)]
    pub finish_reason: Option<String>,
    /// 候选序号
    #[serde(default)]
    pub index: Option<u32>,
}

/// Token 使用统计
#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct Usage {
    /// Prompt token 数
    #[serde(default)]
    pub prompt_tokens: Option<u32>,
    /// Completion token 数
    #[serde(default)]
    pub completion_tokens: Option<u32>,
    /// 总 token 数
    #[serde(default)]
    pub total_tokens: Option<u32>,
    /// 兼容 OpenAI 格式的 prompt token 细分。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_details: Option<TokenUsageDetails>,
    /// 部分兼容供应商改为报告 input-token 细分。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_tokens_details: Option<TokenUsageDetails>,
    /// 供应商特有的 output-token 细分，包含推理 token。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_tokens_details: Option<TokenUsageDetails>,
    /// Anthropic 本次请求的缓存写入量。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_tokens: Option<u32>,
    /// Anthropic 本次请求的缓存读取量。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_read_input_tokens: Option<u32>,
    /**
    DeepSeek：从上下文缓存（KV cache 命中）提供的 prompt token 数。
    `prompt_tokens = prompt_cache_hit_tokens + prompt_cache_miss_tokens`。
    */
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_cache_hit_tokens: Option<u32>,
    /// DeepSeek：未从缓存提供的 prompt token 数（缓存未命中）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_cache_miss_tokens: Option<u32>,
}

impl Usage {
    /**
    经供应商归一化后、模型实际处理的 prompt token 数，
    包括从 prompt cache 提供的 token。

    兼容 OpenAI 格式的响应以及 DeepSeek 的响应中，`prompt_tokens`
    已经包含了缓存 token。Anthropic 则将缓存读取与缓存写入分开报告，
    因此二者都要加回来，以便计算上下文窗口和缓存相关指标。
    */
    pub fn effective_prompt_tokens(&self) -> u32 {
        let prompt = self.prompt_tokens.unwrap_or(0);
        if self.cache_read_input_tokens.is_some() || self.cache_creation_input_tokens.is_some() {
            prompt
                .saturating_add(self.cached_prompt_tokens())
                .saturating_add(self.cache_creation_prompt_tokens())
        } else {
            prompt
        }
    }

    /// 经供应商归一化后、本次响应处理的总 token 数。
    pub fn effective_total_tokens(&self) -> u32 {
        if self.cache_read_input_tokens.is_some() || self.cache_creation_input_tokens.is_some() {
            self.effective_prompt_tokens()
                .saturating_add(self.completion_tokens.unwrap_or(0))
        } else {
            self.total_tokens.unwrap_or_else(|| {
                self.effective_prompt_tokens()
                    .saturating_add(self.completion_tokens.unwrap_or(0))
            })
        }
    }

    /**
    经供应商归一化后、从缓存读取的 prompt token 数。

    按优先级依次检查：
    1. OpenAI `prompt_tokens_details.cached_tokens`
    2. 兼容供应商的 `input_tokens_details.cached_tokens`
    3. Anthropic 的 `cache_read_input_tokens`
    4. DeepSeek 的 `prompt_cache_hit_tokens`
    */
    pub fn cached_prompt_tokens(&self) -> u32 {
        self.prompt_tokens_details
            .as_ref()
            .and_then(|details| details.cached_tokens)
            .or_else(|| {
                self.input_tokens_details
                    .as_ref()
                    .and_then(|details| details.cached_tokens)
            })
            .or(self.cache_read_input_tokens)
            .or(self.prompt_cache_hit_tokens)
            .unwrap_or(0)
    }

    /// 经供应商归一化后、写入缓存的 prompt token 数。
    pub fn cache_creation_prompt_tokens(&self) -> u32 {
        self.prompt_tokens_details
            .as_ref()
            .and_then(|details| details.cache_write_tokens)
            .or_else(|| {
                self.input_tokens_details
                    .as_ref()
                    .and_then(|details| details.cache_write_tokens)
            })
            .or(self.cache_creation_input_tokens)
            .unwrap_or(0)
    }

    /**
    Prompt 缓存命中率，取值范围 [0.0, 1.0]。

    不同供应商对 `prompt_tokens` 是否包含缓存 token 的语义不同：
    - OpenAI / DeepSeek / 兼容供应商：`prompt_tokens` 是**总量**（含缓存部分）。
      按 DeepSeek 文档：`prompt_tokens = prompt_cache_hit_tokens + prompt_cache_miss_tokens`。
      → 命中率 = cached / prompt_tokens
    - Anthropic：`input_tokens`（映射为 `prompt_tokens`）既不包含缓存读取，
      也不包含缓存写入。
      → 命中率 = cached / (prompt_tokens + cache_read + cache_creation)

    我们通过是否存在任一 Anthropic 缓存字段来判断走哪条路径；
    其余所有供应商都使用「总量已含缓存」的公式。
    */
    pub fn cache_hit_rate(&self) -> Option<f64> {
        let cached = self.cached_prompt_tokens();
        let total = self.effective_prompt_tokens();
        if total == 0 {
            return None;
        }
        Some(cached as f64 / total as f64)
    }
}

/// 嵌套在 usage 下的供应商特有 token 细分信息。
#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct TokenUsageDetails {
    /// 从供应商侧 prompt 缓存提供的 token 数。
    #[serde(default)]
    pub cached_tokens: Option<u32>,
    /// 写入供应商侧 prompt 缓存的 token 数。
    #[serde(default)]
    pub cache_write_tokens: Option<u32>,
    /// 用于模型内部推理的 token 数。
    #[serde(default)]
    pub reasoning_tokens: Option<u32>,
}

// ── 流式响应类型 ───────────────────────────────────────────────────────────

/// SSE 流式响应中的单个数据块
#[derive(Debug, Deserialize, Clone)]
pub struct ChatCompletionChunk {
    /// 响应 ID
    #[serde(default)]
    pub id: String,
    /// 候选响应列表
    #[serde(default)]
    pub choices: Vec<ChunkChoice>,
    /// Token 使用统计（仅在设置了 stream_options.include_usage 时，出现在最后一个 chunk）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

/// 将 `null` 反序列化为 `T::default()`，供 `#[serde(deserialize_with)]` 使用。
fn deserialize_null_as_default<'de, D, T>(deserializer: D) -> Result<T, D::Error>
where
    D: serde::Deserializer<'de>,
    T: serde::Deserialize<'de> + Default,
{
    Option::<T>::deserialize(deserializer).map(|opt| opt.unwrap_or_default())
}

/// 流式候选响应
#[derive(Debug, Deserialize, Clone)]
pub struct ChunkChoice {
    /**
    增量消息内容。

    `deserialize_with = "deserialize_null_as_default"` 至关重要：当设置了
    `stream_options.include_usage` 时，供应商返回携带 usage 的最后一个
    chunk，其 `choices` 数组中的元素可能出现 `delta` 为 `null` 的情况
    （如 `{"delta":null,"finish_reason":"stop"}`）。若没有这个反序列化器，
    null delta 会解析失败，整个 chunk 被丢弃，`usage` 字段也随之悄悄丢失
    —— 导致每次 LLM 调用在可观测性中都显示 `usage_reported: false`。
    */
    #[serde(default, deserialize_with = "deserialize_null_as_default")]
    pub delta: DeltaMessage,
    /// 结束原因
    #[serde(default)]
    pub finish_reason: Option<String>,
    /// 候选序号
    #[serde(default)]
    pub index: u32,
}

/// 流式响应中的增量消息体
#[derive(Debug, Deserialize, Clone, Default)]
pub struct DeltaMessage {
    /// 角色（首次出现时携带）
    #[serde(default)]
    pub role: Option<String>,
    /// 内容增量
    #[serde(default)]
    pub content: Option<String>,
    /// 推理内容增量（来自 Qwen3/DeepSeek 等模型的思考过程）
    #[serde(default)]
    pub reasoning_content: Option<String>,
    /// 用于回放的、已完成的签名/已脱敏推理块
    #[serde(default)]
    pub reasoning_blocks: Option<Vec<ReasoningBlock>>,
    /// 工具调用增量
    #[serde(default)]
    pub tool_calls: Option<Vec<DeltaToolCall>>,
}

/// 流式工具调用的增量片段
#[derive(Debug, Deserialize, Clone)]
pub struct DeltaToolCall {
    /// 工具调用序号
    pub index: u32,
    /// 工具调用 ID（增量出现）
    #[serde(default)]
    pub id: Option<String>,
    /// 工具调用类型（增量出现）
    #[serde(rename = "type", default)]
    pub call_type: Option<String>,
    /// 函数调用增量
    #[serde(default)]
    pub function: Option<DeltaFunctionCall>,
}

/// 流式函数调用的增量片段
#[derive(Debug, Deserialize, Clone, Default)]
pub struct DeltaFunctionCall {
    /// 函数名（增量出现）
    #[serde(default)]
    pub name: Option<String>,
    /// 函数参数（增量出现）
    #[serde(default)]
    pub arguments: Option<String>,
}

// ── Feature 6：多模态支持测试 ────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_part_text_serde_roundtrip() {
        let part = ContentPart::Text {
            text: "hello".to_string(),
        };
        let json = serde_json::to_string(&part).unwrap();
        assert!(json.contains("\"type\":\"text\""));
        let back: ContentPart = serde_json::from_str(&json).unwrap();
        match back {
            ContentPart::Text { text } => assert_eq!(text, "hello"),
            _ => panic!("should be Text"),
        }
    }

    #[test]
    fn content_part_image_serde_roundtrip() {
        let part = ContentPart::ImageUrl {
            image_url: ImageUrl {
                url: "data:image/png;base64,abc123".to_string(),
                detail: Some("high".to_string()),
            },
        };
        let json = serde_json::to_string(&part).unwrap();
        assert!(json.contains("\"type\":\"image_url\""));
        let back: ContentPart = serde_json::from_str(&json).unwrap();
        match back {
            ContentPart::ImageUrl { image_url } => {
                assert!(image_url.url.starts_with("data:image/png;base64,"));
                assert_eq!(image_url.detail, Some("high".to_string()));
            }
            _ => panic!("should be ImageUrl"),
        }
    }

    #[test]
    fn content_part_file_serde_roundtrip() {
        let part = ContentPart::File {
            name: "readme.txt".to_string(),
            content: "SGVsbG8=".to_string(),
        };
        let json = serde_json::to_string(&part).unwrap();
        assert!(json.contains("\"type\":\"file\""));
        let back: ContentPart = serde_json::from_str(&json).unwrap();
        match back {
            ContentPart::File { name, content } => {
                assert_eq!(name, "readme.txt");
                assert_eq!(content, "SGVsbG8=");
            }
            _ => panic!("should be File"),
        }
    }

    #[test]
    fn message_content_text_serializes_as_string() {
        let mc = MessageContent::Text("hello".to_string());
        let json = serde_json::to_value(&mc).unwrap();
        assert_eq!(json, serde_json::Value::String("hello".to_string()));
    }

    #[test]
    fn message_content_parts_serializes_as_array() {
        let mc = MessageContent::Parts(vec![
            ContentPart::Text {
                text: "hi".to_string(),
            },
            ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: "https://example.com/img.png".to_string(),
                    detail: None,
                },
            },
        ]);
        let json = serde_json::to_value(&mc).unwrap();
        assert!(json.is_array());
        assert_eq!(json.as_array().unwrap().len(), 2);
    }

    #[test]
    fn message_content_deserialize_string() {
        let json = serde_json::json!("plain text");
        let mc: MessageContent = serde_json::from_value(json).unwrap();
        assert_eq!(mc.as_text(), Some("plain text".to_string()));
    }

    #[test]
    fn message_content_deserialize_array() {
        let json = serde_json::json!([
            { "type": "text", "text": "describe this" },
            { "type": "image_url", "image_url": { "url": "https://img.png" } }
        ]);
        let mc: MessageContent = serde_json::from_value(json).unwrap();
        match mc {
            MessageContent::Parts(parts) => {
                assert_eq!(parts.len(), 2);
            }
            _ => panic!("should be Parts"),
        }
    }

    #[test]
    fn message_text_serde_backward_compatible() {
        let msg = Message::user("hello".to_string());
        let json = serde_json::to_value(&msg).unwrap();
        assert_eq!(json["content"], "hello");
        assert_eq!(json["role"], "user");
        assert!(json.get("content_parts").is_none());
    }

    #[test]
    fn message_multimodal_serde() {
        let msg = Message::user_with_image("describe", "image/png", "base64data");
        let json = serde_json::to_value(&msg).unwrap();

        assert_eq!(json["role"], "user");
        let content = &json["content"];
        assert!(
            content.is_array(),
            "multimodal content should serialize as array"
        );
        assert_eq!(content.as_array().unwrap().len(), 2);
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[1]["type"], "image_url");
    }

    #[test]
    fn message_multimodal_deserialize() {
        let json = serde_json::json!({
            "role": "user",
            "content": [
                { "type": "text", "text": "what is this?" },
                { "type": "image_url", "image_url": { "url": "https://example.com/cat.jpg" } }
            ]
        });
        let msg: Message = serde_json::from_value(json).unwrap();
        assert_eq!(msg.role, Role::User);
        assert!(
            matches!(msg.content, MessageContent::Parts(_)),
            "content should be Parts for multimodal"
        );
        if let MessageContent::Parts(ref parts) = msg.content {
            assert_eq!(parts.len(), 2);
        }
    }

    #[test]
    fn message_text_content_extracts_text_from_parts() {
        let msg = Message::user_multimodal(vec![
            ContentPart::Text {
                text: "part1".to_string(),
            },
            ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: "https://img.png".to_string(),
                    detail: None,
                },
            },
            ContentPart::Text {
                text: "part2".to_string(),
            },
        ]);
        assert_eq!(msg.text_content(), Some("part1part2".to_string()));
    }

    #[test]
    fn message_is_multimodal() {
        let text_msg = Message::user("hello".to_string());
        assert!(!text_msg.is_multimodal());

        let mm_msg = Message::user_with_image_url("describe", "https://img.png");
        assert!(mm_msg.is_multimodal());
    }

    #[test]
    fn message_user_with_image_url_helper() {
        let msg = Message::user_with_image_url("look at this", "https://example.com/photo.jpg");
        assert_eq!(msg.role, Role::User);
        assert!(msg.is_multimodal());
        let MessageContent::Parts(parts) = &msg.content else {
            panic!("expected Parts content");
        };
        assert_eq!(parts.len(), 2);
        match &parts[0] {
            ContentPart::Text { text } => assert_eq!(text, "look at this"),
            _ => panic!("first part should be text"),
        }
        match &parts[1] {
            ContentPart::ImageUrl { image_url } => {
                assert_eq!(image_url.url, "https://example.com/photo.jpg");
            }
            _ => panic!("second part should be image_url"),
        }
    }

    #[test]
    fn message_deserialize_text_string_content() {
        let json = serde_json::json!({
            "role": "assistant",
            "content": "I am a response"
        });
        let msg: Message = serde_json::from_value(json).unwrap();
        assert_eq!(msg.content.as_text(), Some("I am a response".to_string()));
    }

    #[test]
    fn usage_reads_openai_compatible_cached_tokens() {
        let usage: Usage = serde_json::from_value(serde_json::json!({
            "prompt_tokens": 1000,
            "completion_tokens": 20,
            "total_tokens": 1020,
            "prompt_tokens_details": {
                "cached_tokens": 980
            }
        }))
        .unwrap();

        assert_eq!(usage.cached_prompt_tokens(), 980);
        assert_eq!(usage.cache_creation_prompt_tokens(), 0);
    }

    #[test]
    fn usage_reads_openai_cache_write_tokens() -> Result<(), serde_json::Error> {
        let usage: Usage = serde_json::from_value(serde_json::json!({
            "prompt_tokens": 1200,
            "completion_tokens": 40,
            "prompt_tokens_details": {
                "cached_tokens": 800,
                "cache_write_tokens": 200
            }
        }))?;

        assert_eq!(usage.cached_prompt_tokens(), 800);
        assert_eq!(usage.cache_creation_prompt_tokens(), 200);
        Ok(())
    }

    #[test]
    fn usage_reads_anthropic_cache_tokens() {
        let usage: Usage = serde_json::from_value(serde_json::json!({
            "prompt_tokens": 1000,
            "completion_tokens": 20,
            "total_tokens": 1020,
            "cache_creation_input_tokens": 64,
            "cache_read_input_tokens": 900
        }))
        .unwrap();

        assert_eq!(usage.cached_prompt_tokens(), 900);
        assert_eq!(usage.cache_creation_prompt_tokens(), 64);
        assert_eq!(usage.effective_prompt_tokens(), 1964);
        assert_eq!(usage.effective_total_tokens(), 1984);
    }

    #[test]
    fn usage_reads_deepseek_cache_hit_tokens() {
        /*
        DeepSeek 使用顶层的 `prompt_cache_hit_tokens` / `prompt_cache_miss_tokens`，
        而不是 OpenAI 那种 `prompt_tokens_details.cached_tokens` 的模式。
        */
        let usage: Usage = serde_json::from_value(serde_json::json!({
            "prompt_tokens": 1000,
            "completion_tokens": 20,
            "total_tokens": 1020,
            "prompt_cache_hit_tokens": 800,
            "prompt_cache_miss_tokens": 200
        }))
        .unwrap();

        assert_eq!(usage.prompt_tokens, Some(1000));
        assert_eq!(usage.completion_tokens, Some(20));
        assert_eq!(usage.total_tokens, Some(1020));
        assert_eq!(usage.cached_prompt_tokens(), 800);
        assert_eq!(
            usage.prompt_cache_hit_tokens,
            Some(800),
            "should preserve the raw field for direct access"
        );
        assert_eq!(
            usage.prompt_cache_miss_tokens,
            Some(200),
            "should preserve the raw field for direct access"
        );
    }

    #[test]
    fn cache_hit_rate_deepseek_prompt_tokens_includes_cached() {
        /*
        DeepSeek：prompt_tokens = cache_hit + cache_miss（总量已含）。
        命中率 = 800 / 1000 = 0.8（而不是 800/1800）。
        */
        let usage: Usage = serde_json::from_value(serde_json::json!({
            "prompt_tokens": 1000,
            "prompt_cache_hit_tokens": 800,
            "prompt_cache_miss_tokens": 200
        }))
        .unwrap();
        assert_eq!(usage.cache_hit_rate(), Some(0.8));
    }

    #[test]
    fn cache_hit_rate_openai_prompt_tokens_includes_cached() {
        /*
        OpenAI：prompt_tokens 是总量（已含 cached_tokens）。
        命中率 = 900 / 1000 = 0.9（而不是 900/1900）。
        */
        let usage: Usage = serde_json::from_value(serde_json::json!({
            "prompt_tokens": 1000,
            "prompt_tokens_details": { "cached_tokens": 900 }
        }))
        .unwrap();
        assert_eq!(usage.cache_hit_rate(), Some(0.9));
    }

    #[test]
    fn cache_hit_rate_anthropic_input_tokens_excludes_cached() {
        /*
        Anthropic：input_tokens（映射为 prompt_tokens）不包含 cache_read。
        命中率 = 900 / (100 + 900) = 0.9。
        */
        let usage: Usage = serde_json::from_value(serde_json::json!({
            "prompt_tokens": 100,
            "cache_read_input_tokens": 900,
            "cache_creation_input_tokens": 0
        }))
        .unwrap();
        assert_eq!(usage.cache_hit_rate(), Some(0.9));
        assert_eq!(usage.effective_prompt_tokens(), 1000);
    }

    #[test]
    fn anthropic_cache_creation_counts_toward_effective_context() {
        let usage = Usage {
            prompt_tokens: Some(100),
            completion_tokens: Some(20),
            total_tokens: Some(120),
            cache_creation_input_tokens: Some(900),
            ..Default::default()
        };

        assert_eq!(usage.effective_prompt_tokens(), 1000);
        assert_eq!(usage.effective_total_tokens(), 1020);
        assert_eq!(usage.cache_hit_rate(), Some(0.0));
    }

    #[test]
    fn cache_hit_rate_none_when_no_prompt_tokens() {
        let usage = Usage::default();
        assert_eq!(usage.cache_hit_rate(), None);
    }

    #[test]
    fn cache_hit_rate_zero_when_no_cache() {
        let usage: Usage = serde_json::from_value(serde_json::json!({
            "prompt_tokens": 1000
        }))
        .unwrap();
        assert_eq!(usage.cache_hit_rate(), Some(0.0));
    }

    #[test]
    fn chunk_choice_handles_null_delta_in_usage_chunk() {
        /*
        当设置了 stream_options.include_usage 时，部分供应商（DeepSeek、
        OpenAI 等）会在最后一个 chunk 中因模型已停止生成内容而让
        `delta` 为 null。该 chunk 仍携带 `usage`。
        */
        let json = serde_json::json!({
            "id": "chatcmpl-xxx",
            "choices": [{
                "delta": null,
                "finish_reason": "stop",
                "index": 0
            }],
            "usage": {
                "prompt_tokens": 500,
                "completion_tokens": 80,
                "total_tokens": 580,
                "prompt_cache_hit_tokens": 400
            }
        });
        let chunk: ChatCompletionChunk = serde_json::from_value(json).unwrap();
        assert!(
            chunk.usage.is_some(),
            "usage must be parsed even when delta is null"
        );
        let usage = chunk.usage.unwrap();
        assert_eq!(usage.prompt_tokens, Some(500));
        assert_eq!(usage.cached_prompt_tokens(), 400);
    }

    #[test]
    fn chunk_choice_handles_missing_delta_in_usage_chunk() {
        // 部分供应商会发送一个省略或清空 choices 的、仅含 usage 的 chunk。
        let json = serde_json::json!({
            "usage": {
                "prompt_tokens": 500,
                "completion_tokens": 80,
                "total_tokens": 580,
                "prompt_cache_hit_tokens": 50
            }
        });
        let chunk: ChatCompletionChunk = serde_json::from_value(json).unwrap();
        assert!(
            chunk.usage.is_some(),
            "usage-only chunk without choices must parse"
        );
        assert!(chunk.choices.is_empty());
        assert_eq!(chunk.usage.unwrap().cached_prompt_tokens(), 50);
    }

    #[test]
    fn multimodal_token_estimate_charges_non_text_parts() {
        let tokenizer = crate::tokenizer::HeuristicTokenizer;
        let content = MessageContent::Parts(vec![
            ContentPart::Text {
                text: "hello".to_string(),
            },
            ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: "https://example.com/image.png".to_string(),
                    detail: None,
                },
            },
            ContentPart::File {
                name: "note.txt".to_string(),
                content: "YQ==".repeat(100),
            },
        ]);

        assert!(content.estimated_tokens(&tokenizer) >= 1_200);
    }

    #[test]
    fn linked_resource_stays_typed_and_has_deterministic_model_text() {
        let resource = LinkedResource {
            annotations: Some(serde_json::json!({"audience": ["assistant"]})),
            description: Some("source context".to_string()),
            mime_type: Some("text/rust".to_string()),
            name: "lib.rs".to_string(),
            size: Some(42),
            title: Some("Library".to_string()),
            uri: "file:///workspace/src/lib.rs".to_string(),
            meta: Some(serde_json::json!({"trace": "abc"})),
        };
        let content = MessageContent::Parts(vec![ContentPart::ResourceLink {
            resource: Box::new(resource.clone()),
        }]);
        let text = content.as_text().unwrap_or_default();
        assert!(text.starts_with("[Linked resource]\n"));
        assert!(text.contains("file:///workspace/src/lib.rs"));
        assert!(text.contains("source context"));

        let encoded = serde_json::to_string(&content).unwrap_or_default();
        let decoded = serde_json::from_str::<MessageContent>(&encoded).ok();
        let decoded_resource = decoded
            .and_then(MessageContent::into_parts)
            .and_then(|parts| parts.into_iter().next())
            .and_then(|part| match part {
                ContentPart::ResourceLink { resource } => Some(*resource),
                _ => None,
            });
        assert_eq!(decoded_resource, Some(resource));
    }

    #[test]
    fn message_round_trips_signed_and_redacted_reasoning_blocks() {
        let mut message = Message::assistant("answer".to_string());
        message.reasoning_blocks = Some(vec![
            ReasoningBlock::Signed {
                thinking: "reason".to_string(),
                signature: "signature".to_string(),
            },
            ReasoningBlock::Redacted {
                data: "opaque".to_string(),
            },
        ]);

        let encoded = serde_json::to_string(&message).unwrap_or_default();
        let decoded = serde_json::from_str::<Message>(&encoded).ok();
        assert_eq!(
            decoded.and_then(|value| value.reasoning_blocks),
            message.reasoning_blocks
        );
    }
}
