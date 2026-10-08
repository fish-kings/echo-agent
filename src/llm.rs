/*!
LLM 客户端 façade —— 提供商抽象与聊天 API。

对多个 LLM 提供商的统一接口。所有客户端都实现了
[`LlmClient`] trait，同时提供 `chat()` 与 `chat_stream()` 方法。

# 支持的协议

| 客户端 | 协议 | Feature |
|--------|---------|---------|
| [`OpenAiClient`] | OpenAI 及兼容 API | default |
| [`AnthropicClient`] | 原生 Claude API | `a2a` |
| [`ResponsesClient`] | OpenAI Responses | default |

# 快速开始

```rust,no_run
use echo_agent::prelude::*;

# #[tokio::main]
# async fn main() -> echo_agent::error::Result<()> {
let config = LlmConfig::for_provider(
    "local",
    "http://127.0.0.1:11434/v1",
    "",
    "qwen3",
    LlmApiProtocol::ChatCompletions,
)?;
let client = OpenAiClient::new(config)?;
let response = client.chat(ChatRequest {
    messages: vec![Message::user("Hello".to_string())],
    ..Default::default()
}).await?;
println!("{}", response.content().unwrap_or_default());
# Ok(())
# }
```

# 流式输出

```rust,no_run
use echo_agent::prelude::*;
use futures::StreamExt;

# #[tokio::main]
# async fn main() -> echo_agent::error::Result<()> {
let config = LlmConfig::for_provider(
    "local",
    "http://127.0.0.1:11434/v1",
    "",
    "qwen3",
    LlmApiProtocol::ChatCompletions,
)?;
let client = OpenAiClient::new(config)?;
let mut stream = client.chat_stream(ChatRequest {
    messages: vec![Message::user("Tell me a joke".to_string())],
    ..Default::default()
}).await?;
while let Some(chunk) = stream.next().await {
    if let Some(content) = chunk?.delta.content {
        print!("{content}");
    }
}
# Ok(())
# }
```
*/

/// LLM 底层传输类型（从 echo_core 重新导出）。
pub mod types {
    //! 规范化的底层 LLM 传输类型。
    pub use echo_core::llm::types::*;
}

/// 与具体提供商无关的 prompt-cache（提示词缓存）契约。
pub mod cache {
    pub use echo_core::llm::cache::*;
}

// 来自 echo-core 的核心 trait
pub use echo_core::llm::capabilities::{
    ModelProfile, ModelProfileOverride, ModelProfileResolver, ProviderCapabilities,
    ThinkingProfile, infer_context_window, resolve_thinking_profile,
};
pub use echo_core::llm::{
    ChatChunk, ChatRequest, ChatResponse, LlmApiProtocol, LlmClient, LlmTimeouts,
    ModelInputModality, SimpleChatOptions, ThinkingConfig, ThinkingLevel, ThinkingProtocol,
};

// 提供商具体实现与显式的运行时配置。
pub use echo_integration::providers::{
    AnthropicClient, LlmConfig, OpenAiClient, ResponsesClient, resolve_protocol_endpoint,
};

pub use types::{JsonSchemaSpec, Message as LlmMessage, ResponseFormat, ToolDefinition};
