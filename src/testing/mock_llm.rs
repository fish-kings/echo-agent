//! 用于测试的 Mock LLM 客户端 —— 让依赖 [`LlmClient`] 的组件无需发起真实 HTTP 请求即可测试。
//!
//! 典型用途：
//! - 测试 [`SummaryCompressor`]（它通过 `LlmClient` 调用 LLM）
//! - 测试自定义的 [`ContextCompressor`] 实现
//! - 任何以 `Arc<dyn LlmClient>` 形式注入依赖的组件
//!
//! # 示例
//!
//! ```rust
//! use echo_agent::testing::MockLlmClient;
//! use echo_agent::llm::LlmClient;
//! use echo_agent::llm::types::Message;
//! use std::sync::Arc;
//!
//! # #[tokio::main]
//! # async fn main() {
//! let mock = Arc::new(
//!     MockLlmClient::new()
//!         .with_response("First response")
//!         .with_response("Second response")
//! );
//!
//! let r1 = mock.chat_simple(vec![Message::user("hi".to_string())]).await.unwrap();
//! assert_eq!(r1, "First response");
//! assert_eq!(mock.call_count(), 1);
//! # }
//! ```

use crate::error::{LlmError, ReactError, Result};
use crate::llm::types::{
    DeltaFunctionCall, DeltaMessage, DeltaToolCall, FunctionCall, Message, ToolCall,
};
use crate::llm::{ChatChunk, ChatRequest, ChatResponse, LlmClient};
use futures::future::BoxFuture;
use futures::stream::BoxStream;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// 预设响应的枚举（文本、工具调用、脚本化流式响应或错误）。
enum MockLlmResponse {
    /// 一次性文本响应，附带可选的用量统计。
    Content(String, Option<crate::llm::types::Usage>),
    /// 携带工具调用的助手消息，附带可选的用量统计。
    ToolCalls(Message, Option<crate::llm::types::Usage>),
    /// 一段脚本化的流式响应，按顺序回放各个分片。
    Stream(Vec<StreamChunk>),
    /// 该次调用直接返回错误。
    Err(ReactError),
}

/** 一次脚本化流式回合中的单个分片。

真实的 provider 从不会把内容和用量合并进同一个分片：内容以增量（delta）形式到达，
终止分片才携带 `finish_reason` 与用量信息（Anthropic 的 `message_delta.usage` 只携带
`output_tokens`；OpenAI 则在单独的最终分片上返回用量）。`StreamChunk` 让测试能够
复现这种真实的线上格式，而不是沿用旧版 mock 那种不可能出现的单分片合并形状
（F-TST-01-P1-01）。
*/
pub enum StreamChunk {
    /// 内容或工具调用的增量分片，不携带终止信息。
    Delta(DeltaMessage),
    /// 终止分片：结束原因 + provider 报告的用量。
    Terminal {
        finish_reason: Option<String>,
        usage: Option<crate::llm::types::Usage>,
    },
    /// 流中途发生的错误（provider 断连、事件格式错误、超时等）。
    Err(ReactError),
    /// 在发出下一个脚本化分片之前，等待一段可被取消感知的延迟。
    Delay(Duration),
}

/** 可编排脚本的 Mock LLM 客户端。

按顺序返回预设的响应；一旦队列耗尽，则返回 `EmptyResponse` 错误。
所有调用都会被记录，可通过 [`call_count`](MockLlmClient::call_count) /
[`last_messages`](MockLlmClient::last_messages) 等方法查看。
*/
pub struct MockLlmClient {
    /// 该 mock 客户端报告的模型名称。
    model_name: String,
    /// 待消费的预设响应队列。
    responses: Arc<Mutex<VecDeque<MockLlmResponse>>>,
    /// 每次调用收到的消息列表，按调用顺序记录。
    calls: Arc<Mutex<Vec<Vec<Message>>>>,
    /// 每次请求收到的缓存分区标识（user_id）。
    user_ids: Arc<Mutex<Vec<Option<String>>>>,
    /// 每次请求收到的工具选择策略。
    tool_choices: Arc<Mutex<Vec<Option<String>>>>,
    /// 每次请求暴露的工具定义数量。
    tool_counts: Arc<Mutex<Vec<usize>>>,
    /// 返回每个响应前的可选延迟。设置后，`chat` 和
    /// `chat_stream` 会睡眠这段时长，但如果请求的 `cancel_token` 被触发，
    /// 会提前返回 Cancelled 错误。这让测试可以模拟"正在运行中"且可被中断的
    /// LLM 调用（Phase 3）。
    delay: Option<Duration>,
}

impl Default for MockLlmClient {
    /// 等价于 [`MockLlmClient::new`]。
    fn default() -> Self {
        Self::new()
    }
}

impl MockLlmClient {
    /// 创建一个尚无预设响应的空 Mock。
    pub fn new() -> Self {
        Self {
            model_name: "mock-model".to_string(),
            responses: Arc::new(Mutex::new(VecDeque::new())),
            calls: Arc::new(Mutex::new(Vec::new())),
            user_ids: Arc::new(Mutex::new(Vec::new())),
            tool_choices: Arc::new(Mutex::new(Vec::new())),
            tool_counts: Arc::new(Mutex::new(Vec::new())),
            delay: None,
        }
    }

    /// 为每个响应设置一个延迟。与请求的 `cancel_token` 配合使用时，
    /// 可以让测试验证中途取消的行为（Phase 3）。
    pub fn with_delay(mut self, delay: Duration) -> Self {
        self.delay = Some(delay);
        self
    }

    /// 设置模型名称。
    pub fn with_model_name(mut self, name: impl Into<String>) -> Self {
        self.model_name = name.into();
        self
    }

    /// 追加一个成功的文本响应。
    pub fn with_response(self, text: impl Into<String>) -> Self {
        self.responses
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push_back(MockLlmResponse::Content(text.into(), None));
        self
    }

    /// 追加一个带 provider 用量信息的文本响应。
    pub fn with_response_usage(
        self,
        text: impl Into<String>,
        usage: crate::llm::types::Usage,
    ) -> Self {
        self.responses
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push_back(MockLlmResponse::Content(text.into(), Some(usage)));
        self
    }

    /// 批量追加多个成功响应。
    pub fn with_responses(self, texts: impl IntoIterator<Item = impl Into<String>>) -> Self {
        {
            let mut q = self.responses.lock().unwrap_or_else(|e| e.into_inner());
            for t in texts {
                q.push_back(MockLlmResponse::Content(t.into(), None));
            }
        }
        self
    }

    /// 追加一个错误响应（用于测试错误处理路径）。
    pub fn with_error(self, err: ReactError) -> Self {
        self.responses
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push_back(MockLlmResponse::Err(err));
        self
    }

    /** 追加一个工具调用响应（模拟 LLM 发起一次工具调用）。

    # 示例

    ```rust
    use echo_agent::testing::MockLlmClient;

    let mock = MockLlmClient::new()
        .then_tool_call("call_1", "calculator", r#"{"a":1,"b":2}"#)
        .with_response("The answer is 3");
    ```
    */
    pub fn then_tool_call(
        self,
        id: impl Into<String>,
        function_name: impl Into<String>,
        arguments: impl Into<String>,
    ) -> Self {
        let tc = ToolCall {
            id: id.into(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: function_name.into(),
                arguments: arguments.into(),
            },
        };
        self.responses
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push_back(MockLlmResponse::ToolCalls(
                Message::assistant_with_tools(vec![tc]),
                None,
            ));
        self
    }

    /// 追加一个带推理内容（reasoning content）的 DeepSeek 风格工具调用响应。
    #[cfg(test)]
    pub(crate) fn then_reasoning_tool_call(
        self,
        id: impl Into<String>,
        function_name: impl Into<String>,
        arguments: impl Into<String>,
        content: impl Into<String>,
        reasoning_content: impl Into<String>,
    ) -> Self {
        let tc = ToolCall {
            id: id.into(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: function_name.into(),
                arguments: arguments.into(),
            },
        };
        let mut message = Message::assistant_with_tools(vec![tc]);
        message.content = crate::llm::types::MessageContent::Text(content.into());
        message.reasoning_content = Some(reasoning_content.into());
        self.responses
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push_back(MockLlmResponse::ToolCalls(message, None));
        self
    }

    /// 追加一个带 provider 用量信息的工具调用响应。
    pub fn then_tool_call_with_usage(
        self,
        id: impl Into<String>,
        function_name: impl Into<String>,
        arguments: impl Into<String>,
        usage: crate::llm::types::Usage,
    ) -> Self {
        let tc = ToolCall {
            id: id.into(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: function_name.into(),
                arguments: arguments.into(),
            },
        };
        self.responses
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push_back(MockLlmResponse::ToolCalls(
                Message::assistant_with_tools(vec![tc]),
                Some(usage),
            ));
        self
    }

    /// 追加一个多工具调用响应（并行工具调用）。
    pub fn then_tool_calls(self, calls: Vec<ToolCall>) -> Self {
        self.responses
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push_back(MockLlmResponse::ToolCalls(
                Message::assistant_with_tools(calls),
                None,
            ));
        self
    }

    /// 追加一个网络错误（常用的便捷方法）。
    pub fn with_network_error(self, msg: impl Into<String>) -> Self {
        self.with_error(ReactError::Llm(Box::new(LlmError::NetworkError(
            msg.into(),
        ))))
    }

    /// 追加一个限流错误（429），用于测试重试逻辑。
    pub fn with_rate_limit_error(self) -> Self {
        self.with_error(ReactError::Llm(Box::new(LlmError::ApiError {
            status: 429,
            message: "Too Many Requests".to_string(),
        })))
    }

    /** 追加一段脚本化的流式回合。`chat_stream` 会按顺序发出这些分片；
    若遇到 `StreamChunk::Err` 则以该错误结束整个流。

    用它来复现真实 provider 的线上格式：先是若干内容增量，随后是单独的
    终止分片携带 `finish_reason` 和用量信息（普通 builder 生成的单分片形状
    仅用于非流式测试）。

    # 示例

    ```rust
    use echo_agent::testing::{MockLlmClient, StreamChunk};
    use echo_agent::llm::types::{DeltaMessage, Usage};

    let usage = Usage {
        prompt_tokens: Some(10),
        completion_tokens: Some(5),
        ..Default::default()
    };
    let mock = MockLlmClient::new().with_stream_script(vec![
        StreamChunk::Delta(DeltaMessage {
            role: Some("assistant".to_string()),
            content: Some("Hello".to_string()),
            reasoning_content: None,
            reasoning_blocks: None,
            tool_calls: None,
        }),
        StreamChunk::Terminal {
            finish_reason: Some("stop".to_string()),
            usage: Some(usage),
        },
    ]);
    ```
    */
    pub fn with_stream_script(self, chunks: Vec<StreamChunk>) -> Self {
        self.responses
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push_back(MockLlmResponse::Stream(chunks));
        self
    }

    /// 已发生的调用总数。
    pub fn call_count(&self) -> usize {
        self.calls.lock().unwrap_or_else(|e| e.into_inner()).len()
    }

    /// 最近一次调用传入的消息（从未调用过时返回 `None`）。
    pub fn last_messages(&self) -> Option<Vec<Message>> {
        self.calls
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .last()
            .cloned()
    }

    /// 所有历史调用的消息（按时间顺序）。
    pub fn all_calls(&self) -> Vec<Vec<Message>> {
        self.calls.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// 每次请求收到的缓存分区标识列表。
    pub fn all_user_ids(&self) -> Vec<Option<String>> {
        self.user_ids
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// 每次请求收到的工具选择策略列表。
    pub fn all_tool_choices(&self) -> Vec<Option<String>> {
        self.tool_choices
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// 每次请求收到的已暴露工具定义数量列表。
    pub fn all_tool_counts(&self) -> Vec<usize> {
        self.tool_counts
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// 尚未被消费的预设响应数量。
    pub fn remaining(&self) -> usize {
        self.responses
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .len()
    }

    /// 清空所有已记录的调用历史（不影响响应队列）。
    pub fn reset_calls(&self) {
        self.calls.lock().unwrap_or_else(|e| e.into_inner()).clear();
        self.user_ids
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
        self.tool_choices
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
        self.tool_counts
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
    }

    /// 弹出下一个响应（文本或工具调用）。
    fn pop_response(&self) -> Result<PopResult> {
        match self
            .responses
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .pop_front()
        {
            Some(MockLlmResponse::Content(text, usage)) => Ok(PopResult::Content(text, usage)),
            Some(MockLlmResponse::ToolCalls(message, usage)) => {
                Ok(PopResult::ToolCalls(message, usage))
            }
            Some(MockLlmResponse::Stream(chunks)) => Ok(PopResult::Stream(chunks)),
            Some(MockLlmResponse::Err(e)) => Err(e),
            None => Err(ReactError::Llm(Box::new(LlmError::EmptyResponse))),
        }
    }
}

/// `pop_response` 的返回类型，统一表示已弹出的一个响应。
enum PopResult {
    /// 文本响应及其可选用量。
    Content(String, Option<crate::llm::types::Usage>),
    /// 工具调用消息及其可选用量。
    ToolCalls(Message, Option<crate::llm::types::Usage>),
    /// 一段待回放的脚本化流式响应。
    Stream(Vec<StreamChunk>),
}

impl LlmClient for MockLlmClient {
    /// 执行一次非流式聊天调用：记录本次调用信息，处理可选延迟/取消，
    /// 并弹出下一个预设响应（文本、工具调用或折叠后的脚本化流）。
    fn chat(&self, request: ChatRequest) -> BoxFuture<'_, Result<ChatResponse>> {
        Box::pin(async move {
            // 记录本次调用
            self.tool_counts
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(request.tools.as_ref().map_or(0, Vec::len));
            self.user_ids
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(request.user_id.clone());
            self.calls
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(request.messages);
            self.tool_choices
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(request.tool_choice);

            // 可选的、支持取消感知的延迟（Phase 3：让测试能够验证
            // 中途取消的行为）。
            if let Some(d) = self.delay {
                let token = request.cancel_token.clone().unwrap_or_default();
                tokio::select! {
                    _ = token.cancelled() => {
                        return Err(ReactError::Other("mock LLM call cancelled".into()));
                    }
                    _ = tokio::time::sleep(d) => {}
                }
            }

            match self.pop_response()? {
                PopResult::Content(text, usage) => Ok(ChatResponse {
                    message: Message::assistant(text),
                    finish_reason: Some("stop".to_string()),
                    usage: usage.clone(),
                    raw: crate::llm::types::ChatCompletionResponse {
                        usage,
                        ..crate::llm::types::ChatCompletionResponse::default()
                    },
                }),
                PopResult::ToolCalls(message, usage) => Ok(ChatResponse {
                    message,
                    finish_reason: Some("tool_calls".to_string()),
                    usage: usage.clone(),
                    raw: crate::llm::types::ChatCompletionResponse {
                        usage,
                        ..crate::llm::types::ChatCompletionResponse::default()
                    },
                }),
                // 通过非流式入口消费脚本化流：把各个增量折叠为单条
                // 助手消息，并采用最后一个终止分片的结束原因/用量。
                PopResult::Stream(chunks) => {
                    let mut text = String::new();
                    let mut reasoning = String::new();
                    let mut reasoning_blocks = Vec::new();
                    let mut finish_reason = None;
                    let mut usage = None;
                    for chunk in chunks {
                        match chunk {
                            StreamChunk::Delta(delta) => {
                                if let Some(t) = delta.content {
                                    text.push_str(&t);
                                }
                                if let Some(value) = delta.reasoning_content {
                                    reasoning.push_str(&value);
                                }
                                reasoning_blocks.extend(delta.reasoning_blocks.unwrap_or_default());
                            }
                            StreamChunk::Terminal {
                                finish_reason: fr,
                                usage: u,
                            } => {
                                finish_reason = fr;
                                usage = u;
                            }
                            StreamChunk::Err(e) => return Err(e),
                            StreamChunk::Delay(delay) => {
                                let token = request.cancel_token.clone().unwrap_or_default();
                                tokio::select! {
                                    _ = token.cancelled() => {
                                        return Err(ReactError::Agent(Box::new(
                                            crate::error::AgentError::Cancelled(
                                                "mock LLM call cancelled".to_string(),
                                            ),
                                        )));
                                    }
                                    _ = tokio::time::sleep(delay) => {}
                                }
                            }
                        }
                    }
                    let mut message = Message::assistant(text);
                    message.reasoning_content = (!reasoning.is_empty()).then_some(reasoning);
                    message.reasoning_blocks =
                        (!reasoning_blocks.is_empty()).then_some(reasoning_blocks);
                    Ok(ChatResponse {
                        message,
                        finish_reason,
                        usage: usage.clone(),
                        raw: crate::llm::types::ChatCompletionResponse {
                            usage,
                            ..crate::llm::types::ChatCompletionResponse::default()
                        },
                    })
                }
            }
        })
    }

    /// 执行一次流式聊天调用：记录本次调用信息，处理可选延迟/取消，
    /// 并把弹出的预设响应转换为对应的分片流。
    fn chat_stream(
        &self,
        request: ChatRequest,
    ) -> BoxFuture<'_, Result<BoxStream<'static, Result<ChatChunk>>>> {
        Box::pin(async move {
            // 记录本次调用
            self.tool_counts
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(request.tools.as_ref().map_or(0, Vec::len));
            self.user_ids
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(request.user_id.clone());
            self.calls
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(request.messages);
            self.tool_choices
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(request.tool_choice);

            // 可选的、支持取消感知的延迟（Phase 3）。
            if let Some(d) = self.delay {
                let token = request.cancel_token.clone().unwrap_or_default();
                tokio::select! {
                    _ = token.cancelled() => {
                        return Err(ReactError::Other("mock LLM stream cancelled".into()));
                    }
                    _ = tokio::time::sleep(d) => {}
                }
            }

            // 真实 provider 以增量形式流式发送内容，并在单独的终止分片中
            // 报告用量；这里发出同样的形状，而不是那种不可能出现的
            // 单分片合并（F-TST-01-P1-01）。
            match self.pop_response()? {
                PopResult::Content(text, usage) => {
                    let stream = futures::stream::iter([
                        Ok(ChatChunk {
                            delta: DeltaMessage {
                                role: Some("assistant".to_string()),
                                content: Some(text),
                                reasoning_content: None,
                                reasoning_blocks: None,
                                tool_calls: None,
                            },
                            finish_reason: None,
                            usage: None,
                        }),
                        Ok(ChatChunk {
                            delta: DeltaMessage {
                                role: None,
                                content: None,
                                reasoning_content: None,
                                reasoning_blocks: None,
                                tool_calls: None,
                            },
                            finish_reason: Some("stop".to_string()),
                            usage,
                        }),
                    ]);
                    Ok(Box::pin(stream) as BoxStream<'_, Result<ChatChunk>>)
                }
                PopResult::ToolCalls(message, usage) => {
                    // 将 ToolCall → DeltaToolCall 转换以用于流式发送
                    let delta_calls: Vec<DeltaToolCall> = message
                        .tool_calls
                        .unwrap_or_default()
                        .into_iter()
                        .enumerate()
                        .map(|(i, tc)| DeltaToolCall {
                            index: i as u32,
                            id: Some(tc.id),
                            call_type: Some(tc.call_type),
                            function: Some(DeltaFunctionCall {
                                name: Some(tc.function.name),
                                arguments: Some(tc.function.arguments),
                            }),
                        })
                        .collect();
                    let content = message.content.as_text();
                    let reasoning_content = message.reasoning_content;
                    let stream = futures::stream::iter([
                        Ok(ChatChunk {
                            delta: DeltaMessage {
                                role: Some("assistant".to_string()),
                                content,
                                reasoning_content,
                                reasoning_blocks: message.reasoning_blocks,
                                tool_calls: Some(delta_calls),
                            },
                            finish_reason: None,
                            usage: None,
                        }),
                        Ok(ChatChunk {
                            delta: DeltaMessage {
                                role: None,
                                content: None,
                                reasoning_content: None,
                                reasoning_blocks: None,
                                tool_calls: None,
                            },
                            finish_reason: Some("tool_calls".to_string()),
                            usage,
                        }),
                    ]);
                    Ok(Box::pin(stream) as BoxStream<'_, Result<ChatChunk>>)
                }
                PopResult::Stream(chunks) => {
                    let cancel = request.cancel_token.unwrap_or_default();
                    let stream = async_stream::stream! {
                        for chunk in chunks {
                            match chunk {
                            StreamChunk::Delta(delta) => yield Ok(ChatChunk {
                                delta,
                                finish_reason: None,
                                usage: None,
                            }),
                            StreamChunk::Terminal {
                                finish_reason,
                                usage,
                            } => yield Ok(ChatChunk {
                                delta: DeltaMessage {
                                    role: None,
                                    content: None,
                                    reasoning_content: None,
                                    reasoning_blocks: None,
                                    tool_calls: None,
                                },
                                finish_reason,
                                usage,
                            }),
                            StreamChunk::Err(error) => {
                                yield Err(error);
                                return;
                            }
                            StreamChunk::Delay(delay) => {
                                tokio::select! {
                                    _ = cancel.cancelled() => {
                                        yield Err(ReactError::Agent(Box::new(
                                            crate::error::AgentError::Cancelled(
                                                "mock LLM stream cancelled".to_string(),
                                            ),
                                        )));
                                        return;
                                    }
                                    _ = tokio::time::sleep(delay) => {}
                                }
                            }
                            }
                        }
                    };
                    Ok(Box::pin(stream) as BoxStream<'_, Result<ChatChunk>>)
                }
            }
        })
    }

    /// 返回该 mock 客户端报告的模型名称。
    fn model_name(&self) -> &str {
        &self.model_name
    }
}
