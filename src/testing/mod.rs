/*!
测试基础设施

提供一组用于测试 echo-agent 各组件的工具，无需依赖
真实的 LLM / 外部服务。

| 类型 | 用途 |
|------|---------|
| [`MockLlmClient`] | 替代真实 LLM，用于测试 `SummaryCompressor` 等依赖 `LlmClient` 的组件 |
| [`MockTool`] | 替代真实工具，用于测试 Agent 的工具调用 / 错误处理行为 |
| [`MockAgent`] | 替代真实 Subagent，用于测试多 Agent 编排逻辑 |
| [`FailingMockAgent`] | 总是返回错误，用于测试编排的容错路径 |

# 设计原则

- **零网络请求**：所有 mock 完全在内存中运行
- **可编排**：通过 `with_response()` / `with_error()` 精确控制返回值
- **可观测**：通过 `call_count()` / `last_args()` 等方法检查调用情况
- **线程安全**：内部使用 `Arc<Mutex<_>>`，可在多任务测试间安全共享

# 使用示例

## 测试压缩器（`MockLlmClient`）

```rust,no_run
use echo_agent::testing::MockLlmClient;
use echo_agent::compression::{ContextManager, CompressionInput, ContextCompressor};
use echo_agent::compression::compressor::SummaryCompressor;
use echo_agent::llm::types::Message;
use std::sync::Arc;

# #[tokio::main]
# async fn main() -> echo_agent::error::Result<()> {
let mock_llm = Arc::new(
    MockLlmClient::new().with_response("This is LLM-generated summary content")
);

let compressor = SummaryCompressor::new(mock_llm.clone(), 2);
let input = CompressionInput {
    messages: vec![
        Message::user("message1".to_string()),
        Message::assistant("reply1".to_string()),
        Message::user("message2".to_string()),
        Message::assistant("reply2".to_string()),
        Message::user("message3".to_string()),
    ],
    token_limit: 100,
    current_query: None,
    focus_instructions: None,
    cancel_token: None,
    tokenizer: None,
};

let output = compressor.compress(input).await?;
assert!(!output.messages.is_empty());
assert_eq!(mock_llm.call_count(), 1);  // 确认 LLM 被调用了一次
# Ok(())
# }
```

## 测试工具行为（`MockTool`）

```rust
use echo_agent::testing::MockTool;
use echo_agent::tools::Tool;
use std::collections::HashMap;

# #[tokio::main]
# async fn main() {
let tool = MockTool::new("weather")
    .with_response(r#"{"city":"Beijing","temp":25}"#)
    .with_failure("API service unavailable");

let r1 = tool.execute(HashMap::new()).await.unwrap();
assert!(r1.success);

let r2 = tool.execute(HashMap::new()).await.unwrap();
assert!(!r2.success);

assert_eq!(tool.call_count(), 2);
# }
```

## 测试多 Agent 编排（`MockAgent`）

```rust
use echo_agent::testing::MockAgent;
use echo_agent::agent::Agent;

# #[tokio::main]
# async fn main() {
let mut agent = MockAgent::new("math_agent")
    .with_response("The result is 42");

let answer = agent.execute("6 * 7 = ?").await.unwrap();
assert_eq!(answer, "The result is 42");
assert_eq!(agent.calls()[0], "6 * 7 = ?");
# }
```
*/

/// Mock Agent 实现。
mod mock_agent;
/// Mock 向量嵌入器实现。
mod mock_embedder;
/// Mock LLM 客户端实现。
mod mock_llm;
/// Mock 工具实现。
mod mock_tool;

pub use mock_agent::{FailingMockAgent, MockAgent, MockAgentFailure, MockAgentStep};
pub use mock_embedder::MockEmbedder;
pub use mock_llm::{MockLlmClient, StreamChunk};
pub use mock_tool::{MockTool, MockToolContext};
