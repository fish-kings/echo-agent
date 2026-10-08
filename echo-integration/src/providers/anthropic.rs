/*!
Anthropic Messages API 提供方（provider）

为 Anthropic 的 `/v1/messages` 接口实现 [`LlmClient`]。
系统消息会作为顶层的 `system` 字段发送（而不是放在 messages 数组里）。
*/

use echo_core::error::{LlmError, Result};
use echo_core::llm::capabilities::resolve_thinking_profile;
use echo_core::llm::types::{
    ChatCompletionResponse, ContentPart, DeltaFunctionCall, DeltaMessage, DeltaToolCall,
    FunctionCall, Message, MessageContent, ReasoningBlock, Role, ToolCall, Usage,
};
use echo_core::llm::{
    ChatChunk, ChatRequest, ChatResponse, LlmApiProtocol, LlmClient, LlmTimeouts,
    ModelInputModality, ThinkingProtocol,
};
use futures::StreamExt;
use futures::future::BoxFuture;

use super::anthropic_cache::AnthropicCachePlan;
use super::client::{
    JsonSseEvent, ensure_request_not_cancelled, post_json_request, stream_json_sse,
};
use super::config::validate_model_input_modalities;
use futures::stream::BoxStream;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{Instrument, info_span, warn};

/**
Anthropic Messages API 的客户端实现。

负责把内部通用的 [`ChatRequest`]/[`ChatResponse`] 结构与 Anthropic 的线上协议
相互转换，并处理非流式与流式两种调用方式。
*/
pub struct AnthropicClient {
    /// 共享的 HTTP 客户端。
    client: Arc<Client>,
    /// Anthropic API Key，通过 `x-api-key` 请求头发送。
    api_key: String,
    /// 目标模型名称。
    model: String,
    /// Messages API 的完整请求地址。
    base_url: String,
    /// 该模型支持的输入模态集合，用于请求前的校验。
    input_modalities: Vec<ModelInputModality>,
    /// 该模型使用的扩展思考（thinking）协议方言。
    thinking_protocol: ThinkingProtocol,
    /// 请求与流式响应的超时策略。
    timeouts: LlmTimeouts,
}

impl AnthropicClient {
    /**
    校验请求中是否包含当前尚未支持转换的特性（如 `tool_choice`、
    `response_format`），如果有则直接报错，避免静默丢弃用户的设置。
    同时校验消息内容与模型支持的输入模态是否匹配。
    */
    fn validate_request_features(&self, request: &ChatRequest) -> Result<()> {
        if request.tool_choice.is_some() {
            return Err(LlmError::InvalidResponse(
                "Anthropic tool_choice translation is not implemented; refusing to silently ignore it"
                    .to_string(),
            )
            .into());
        }
        if request.response_format.is_some() {
            return Err(LlmError::InvalidResponse(
                "Anthropic structured response format translation is not implemented; refusing to silently ignore it"
                    .to_string(),
            )
            .into());
        }
        validate_model_input_modalities(&self.model, &self.input_modalities, &request.messages)?;
        Ok(())
    }

    /**
    使用官方默认地址（`https://api.anthropic.com/v1/messages`）创建一个新的
    `AnthropicClient`，并根据模型名称解析出对应的扩展思考协议。
    */
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        let model = model.into();
        let base_url = "https://api.anthropic.com/v1/messages".to_string();
        let thinking_protocol = resolve_thinking_profile(
            "anthropic",
            &model,
            LlmApiProtocol::Anthropic,
            Some(&base_url),
        )
        .protocol;
        Self {
            client: Arc::new(Self::build_http_client()),
            api_key: api_key.into(),
            model,
            base_url,
            input_modalities: ModelInputModality::all_supported(),
            thinking_protocol,
            timeouts: LlmTimeouts::default(),
        }
    }

    /**
    使用自定义的接口地址创建 `AnthropicClient`（例如指向兼容 Anthropic
    协议的第三方端点），其余行为与 [`Self::new`] 相同。
    */
    pub fn with_base_url(
        base_url: impl Into<String>,
        api_key: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        let base_url = base_url.into();
        let model = model.into();
        let thinking_protocol = resolve_thinking_profile(
            "anthropic",
            &model,
            LlmApiProtocol::Anthropic,
            Some(&base_url),
        )
        .protocol;
        Self {
            client: Arc::new(Self::build_http_client()),
            api_key: api_key.into(),
            model,
            base_url,
            input_modalities: ModelInputModality::all_supported(),
            thinking_protocol,
            timeouts: LlmTimeouts::default(),
        }
    }

    /// 设置该模型支持的输入模态；传入空集合时回退为仅支持纯文本。
    pub fn with_input_modalities(mut self, input_modalities: Vec<ModelInputModality>) -> Self {
        self.input_modalities = if input_modalities.is_empty() {
            ModelInputModality::text_only()
        } else {
            input_modalities
        };
        self
    }

    /// 设置请求与流式响应的超时策略。
    pub fn with_timeouts(mut self, timeouts: LlmTimeouts) -> Self {
        self.timeouts = timeouts;
        self
    }

    /// 构造底层使用的 `reqwest::Client`。
    fn build_http_client() -> Client {
        Client::new()
    }

    /**
    将内部通用的 [`ChatRequest`] 转换为 Anthropic Messages API 所需的请求体，
    包括系统消息合并、消息角色映射、工具/思考配置转换，以及提示词缓存
    （prompt caching）断点的放置。
    */
    fn convert_request(&self, request: &ChatRequest) -> AnthropicRequest {
        let mut system_parts: Vec<String> = Vec::new();
        let mut messages = Vec::new();

        for msg in &request.messages {
            if msg.role == Role::System {
                if let Some(text) = msg.content.as_text() {
                    system_parts.push(text);
                }
                continue;
            }

            if msg.role == Role::Tool {
                messages.push(AnthropicMessage {
                    role: "user".to_string(),
                    content: AnthropicContent::Blocks(vec![ContentBlock::ToolResult {
                        tool_use_id: msg.tool_call_id.clone().unwrap_or_default(),
                        content: msg.content.as_text().unwrap_or_default(),
                        cache_control: None,
                    }]),
                });
                continue;
            }

            if msg.role == Role::Assistant
                && let Some(ref tool_calls) = msg.tool_calls
            {
                let mut blocks: Vec<ContentBlock> = Vec::new();
                append_reasoning_blocks(&mut blocks, msg.reasoning_blocks.as_deref());
                if let Some(ref text) = msg.content.as_text()
                    && !text.is_empty()
                {
                    blocks.push(ContentBlock::Text {
                        text: text.clone(),
                        cache_control: None,
                    });
                }
                for tc in tool_calls {
                    let input: serde_json::Value =
                        serde_json::from_str(&tc.function.arguments).unwrap_or_default();
                    blocks.push(ContentBlock::ToolUse {
                        id: tc.id.clone(),
                        name: tc.function.name.clone(),
                        input,
                    });
                }
                messages.push(AnthropicMessage {
                    role: "assistant".to_string(),
                    content: AnthropicContent::Blocks(blocks),
                });
                continue;
            }

            let content = match &msg.content {
                MessageContent::Parts(parts) => {
                    let mut blocks: Vec<ContentBlock> = Vec::new();
                    if msg.role == Role::Assistant {
                        append_reasoning_blocks(&mut blocks, msg.reasoning_blocks.as_deref());
                    }
                    blocks.extend(
                        parts
                            .iter()
                            .map(|part| match part {
                                ContentPart::Text { text } => ContentBlock::Text {
                                    text: text.clone(),
                                    cache_control: None,
                                },
                                ContentPart::ImageUrl { image_url } => ContentBlock::Image {
                                    source: data_url_to_image_source(&image_url.url),
                                    cache_control: None,
                                },
                                ContentPart::ResourceLink { resource } => ContentBlock::Text {
                                    text: resource.model_text(),
                                    cache_control: None,
                                },
                                /* 文件附件：根据推断出的媒体类型进行分发。
                                 *   - application/pdf → document 内容块（Anthropic 唯一
                                 *     接受的 base64 文档来源类型）
                                 *   - 文本类（txt/md/json/xml/...） → 解码后内联为文本，
                                 *     让模型可以直接阅读
                                 *   - 其他二进制类型 → 仅保留文件名的占位符（该接口没有
                                 *     通用的二进制附件内容块） */
                                ContentPart::File { name, content } => {
                                    file_to_content_block(name, content)
                                }
                            })
                            .collect::<Vec<_>>(),
                    );
                    if blocks.is_empty() {
                        AnthropicContent::Text(String::new())
                    } else {
                        AnthropicContent::Blocks(blocks)
                    }
                }
                _ if msg.role == Role::Assistant && msg.reasoning_blocks.is_some() => {
                    let mut blocks = Vec::new();
                    append_reasoning_blocks(&mut blocks, msg.reasoning_blocks.as_deref());
                    if let Some(text) = msg.content.as_text()
                        && !text.is_empty()
                    {
                        blocks.push(ContentBlock::Text {
                            text,
                            cache_control: None,
                        });
                    }
                    AnthropicContent::Blocks(blocks)
                }
                _ => AnthropicContent::Text(msg.content.as_text().unwrap_or_default()),
            };

            messages.push(AnthropicMessage {
                role: msg.role.as_str().to_string(),
                content,
            });
        }

        /* ── 缓存断点策略（委托给 AnthropicCachePlan）──
         *
         * 先根据请求中的 messages + tools 构建一个只读的布局视图，再交由
         * `AnthropicCachePlan` 决定放置哪些断点。这样把策略集中在一处，也便于
         * 做单元测试。
         *
         * 当请求上带有 `cache_hints` 时，说明 agent 层已经预先计算好了布局，
         * 这里直接基于这些 hints 构建 plan；否则就在这里自行构建布局并应用
         * 默认策略。 */

        let tools_ref: &[echo_core::llm::types::ToolDefinition] =
            request.tools.as_deref().unwrap_or(&[]);
        let cache_plan = if let Some(ref hints) = request.cache_hints
            && !hints.breakpoints.is_empty()
        {
            // agent 层已经通过 CacheHints 提供了明确的断点位置。
            use echo_core::llm::cache::BreakpointTarget as BT;
            AnthropicCachePlan {
                breakpoints: hints.breakpoints.clone(),
                has_system_breakpoint: hints
                    .breakpoints
                    .iter()
                    .any(|b| matches!(b, BT::SystemLastBlock)),
                has_tool_breakpoint: hints
                    .breakpoints
                    .iter()
                    .any(|b| matches!(b, BT::ToolsLastTool)),
            }
        } else {
            /*
            没有明确的断点：要么请求上根本没有 `cache_hints`，要么是主 think
            路径发送了 `Some(CacheHints { breakpoints: vec![], .. })`
            —— agent 层只为稳定前缀哈希计算了布局，但把断点推导留给了
            provider。之所以在这里推导（而不是在 agent 层），是为了保持分层
            干净：echo-agent 不应依赖 Anthropic 专属的 `AnthropicCachePlan`。
            如果没有这个回退逻辑，主路径会完全不放置 cache_control
            （has_system/tool=false，history_breakpoint_count=0），导致
            Anthropic 在流量最大的这条路径上永远无法缓存稳定前缀。
            */
            let layout = echo_core::llm::cache::PromptCacheLayout::from_messages(
                &request.messages,
                tools_ref,
            );
            AnthropicCachePlan::from_layout(&layout)
        };

        // 构建工具列表，并在最后一个工具上放置 cache_control。
        let tools: Option<Vec<AnthropicToolDef>> = request.tools.as_ref().map(|tools| {
            let count = tools.len();
            tools
                .iter()
                .enumerate()
                .map(|(i, t)| AnthropicToolDef {
                    name: t.function.name.clone(),
                    description: Some(t.function.description.clone()),
                    input_schema: t.function.parameters.clone(),
                    cache_control: if cache_plan.has_tool_breakpoint && i == count - 1 {
                        Some(CacheControl::ephemeral())
                    } else {
                        None
                    },
                })
                .collect()
        });

        // 将系统提示词转换为带 cache_control 的 blocks 格式。
        let system = (!system_parts.is_empty()).then(|| {
            let text = system_parts.join("\n\n");
            vec![SystemBlock {
                block_type: "text".to_string(),
                text,
                cache_control: if cache_plan.has_system_breakpoint {
                    Some(CacheControl::ephemeral())
                } else {
                    None
                },
            }]
        });

        // 在会话消息上放置缓存断点。
        if cache_plan.history_breakpoint_count() > 0 {
            let used_breakpoints = usize::from(cache_plan.has_system_breakpoint)
                + usize::from(cache_plan.has_tool_breakpoint && tools.is_some());
            let remaining_breakpoints = 4usize.saturating_sub(used_breakpoints);

            if remaining_breakpoints > 0 {
                if cache_plan.breakpoints.is_empty() {
                    // 默认回退方案（没有明确的历史索引）。
                    apply_conversation_cache_breakpoints(&mut messages, remaining_breakpoints);
                } else {
                    // 将 BreakpointTarget 映射为具体的消息索引并应用。
                    let mut msg_indices: Vec<usize> = cache_plan
                        .breakpoints
                        .iter()
                        .filter_map(|bp| match bp {
                            echo_core::llm::cache::BreakpointTarget::HistoryIndex(i) => Some(*i),
                            echo_core::llm::cache::BreakpointTarget::HistoryLastStable => {
                                messages.iter().rposition(|msg| !msg.is_runtime_context())
                            }
                            _ => None,
                        })
                        .collect();
                    msg_indices.sort_unstable();
                    msg_indices.dedup();
                    for &idx in msg_indices.iter().take(remaining_breakpoints) {
                        if let Some(msg) = messages.get_mut(idx) {
                            msg.add_cache_control_ephemeral();
                        }
                    }
                }
            }
        }

        /*
        转换思考（thinking）配置。Claude 3.7–4.5 接受 budget 块；
        4.6+/4.7+ 使用自适应思考，并会拒绝旧版的 budget 块。
        以配置好的模型契约为准来决定线上协议的具体方言。
        */
        let max_tokens = request.max_tokens.unwrap_or(4096);
        let (thinking, effort) = build_anthropic_thinking(
            &self.model,
            self.thinking_protocol,
            &request.thinking,
            max_tokens,
        );

        AnthropicRequest {
            model: self.model.clone(),
            max_tokens,
            system,
            messages,
            temperature: request.temperature,
            tools,
            stream: None,
            thinking,
            effort,
            metadata: request.user_id.as_ref().map(|uid| AnthropicMetadata {
                user_id: uid.clone(),
            }),
        }
    }

    /**
    将 Anthropic 的非流式响应转换为内部通用的 [`ChatResponse`]，
    拆分文本、思考（reasoning）内容与工具调用，并映射停止原因与用量统计。
    */
    fn convert_response(&self, resp: AnthropicResponse) -> ChatResponse {
        let mut content_parts: Vec<String> = Vec::new();
        let mut reasoning_parts: Vec<String> = Vec::new();
        let mut reasoning_blocks: Vec<ReasoningBlock> = Vec::new();
        let mut tool_calls: Vec<ToolCall> = Vec::new();

        for block in &resp.content {
            match block {
                ContentBlock::Text { text, .. } => content_parts.push(text.clone()),
                ContentBlock::ToolUse { id, name, input } => {
                    tool_calls.push(ToolCall {
                        id: id.clone(),
                        call_type: "function".to_string(),
                        function: FunctionCall {
                            name: name.clone(),
                            arguments: serde_json::to_string(input).unwrap_or_default(),
                        },
                    });
                }
                ContentBlock::Thinking {
                    thinking,
                    signature,
                } => {
                    reasoning_parts.push(thinking.clone());
                    reasoning_blocks.push(ReasoningBlock::Signed {
                        thinking: thinking.clone(),
                        signature: signature.clone(),
                    });
                }
                ContentBlock::RedactedThinking { data } => {
                    reasoning_blocks.push(ReasoningBlock::Redacted { data: data.clone() });
                }
                _ => {}
            }
        }

        let finish_reason = match resp.stop_reason.as_deref() {
            Some("end_turn" | "stop_sequence") => Some("stop".to_string()),
            Some("tool_use") => Some("tool_calls".to_string()),
            Some("max_tokens") => Some("length".to_string()),
            other => other.map(String::from),
        };

        let message = Message {
            role: Role::Assistant,
            content: if content_parts.is_empty() {
                echo_core::llm::types::MessageContent::Empty
            } else {
                echo_core::llm::types::MessageContent::Text(content_parts.join(""))
            },
            tool_calls: if tool_calls.is_empty() {
                None
            } else {
                Some(tool_calls)
            },
            tool_call_id: None,
            name: None,
            reasoning_content: (!reasoning_parts.is_empty()).then(|| reasoning_parts.join("")),
            reasoning_blocks: (!reasoning_blocks.is_empty()).then_some(reasoning_blocks),
        };

        // 从 Anthropic 响应中提取 token 用量
        let usage = resp.usage.map(|u| {
            let prompt = u.input_tokens;
            let completion = u.output_tokens;
            Usage {
                prompt_tokens: Some(prompt),
                completion_tokens: Some(completion),
                total_tokens: Some(prompt.saturating_add(completion)),
                cache_creation_input_tokens: u.cache_creation_input_tokens,
                cache_read_input_tokens: u.cache_read_input_tokens,
                ..Default::default()
            }
        });

        ChatResponse {
            message,
            finish_reason,
            usage: usage.clone(),
            raw: ChatCompletionResponse {
                id: String::new(),
                choices: Vec::new(),
                created: None,
                model: None,
                usage,
                extra: None,
            },
        }
    }
}

impl LlmClient for AnthropicClient {
    /**
    发起一次非流式的 chat 请求：校验请求特性、转换请求体、发送 HTTP 请求，
    并将 Anthropic 的响应转换回内部通用的 [`ChatResponse`]。
    */
    fn chat(&self, request: ChatRequest) -> BoxFuture<'_, Result<ChatResponse>> {
        let model = self.model.clone();
        Box::pin(
            async move {
                self.validate_request_features(&request)?;
                let timeouts = request.timeouts.unwrap_or(self.timeouts);
                let body = self.convert_request(&request);
                let request_builder = self
                    .client
                    .post(&self.base_url)
                    .header("x-api-key", &self.api_key)
                    .header("anthropic-version", "2023-06-01")
                    .header("anthropic-beta", "prompt-caching-2024-07-31")
                    .header("content-type", "application/json")
                    .json(&body);
                let request_builder = match timeouts.request_timeout() {
                    Some(timeout) => request_builder.timeout(timeout),
                    None => request_builder,
                };
                let anthropic_resp: AnthropicResponse =
                    post_json_request(request_builder, request.cancel_token.clone()).await?;

                let response = self.convert_response(anthropic_resp);
                ensure_request_not_cancelled(request.cancel_token.as_ref())?;
                Ok(response)
            }
            .instrument(info_span!("anthropic_chat", model = %model)),
        )
    }

    /**
    发起一次流式 chat 请求：解析 Anthropic 的 SSE 事件流，将其增量转换为
    内部通用的 [`ChatChunk`] 流，包括文本、思考内容、工具调用参数的累积拼接，
    以及流式用量统计的跟踪。
    */
    fn chat_stream(
        &self,
        request: ChatRequest,
    ) -> BoxFuture<'_, Result<BoxStream<'static, Result<ChatChunk>>>> {
        let model = self.model.clone();
        Box::pin(
            async move {
            self.validate_request_features(&request)?;
            let timeouts = request.timeouts.unwrap_or(self.timeouts);
            let mut body = self.convert_request(&request);
            body.stream = Some(true);

            let cancel_token = request.cancel_token.clone();
            let stream_request = self.client
                .post(&self.base_url)
                .header("x-api-key", &self.api_key)
                .header("anthropic-version", "2023-06-01")
                .header("anthropic-beta", "prompt-caching-2024-07-31")
                .header("content-type", "application/json")
                .json(&body);
            let raw_stream = stream_json_sse(
                stream_request,
                self.model.clone(),
                timeouts,
                cancel_token,
            ).await?;
            // 跟踪流式过程中尚未完成的工具调用（index → 累积中的参数）
            let mut tool_call_args: std::collections::HashMap<usize, (String, String, String)> =
                std::collections::HashMap::new();
            let mut reasoning_blocks: std::collections::HashMap<usize, ReasoningBlock> =
                std::collections::HashMap::new();

            // 跟踪流式事件间累计的用量统计
            let mut stream_input_tokens: u32 = 0;
            let mut stream_output_tokens: u32 = 0;
            let mut stream_cache_creation_input_tokens: Option<u32> = None;
            let mut stream_cache_read_input_tokens: Option<u32> = None;

            let stream = async_stream::stream! {
                let mut pending_terminal: Option<ChatChunk> = None;
                futures::pin_mut!(raw_stream);
                while let Some(event) = raw_stream.next().await {
                    let event = match event {
                        Ok(JsonSseEvent::Done) => {
                            yield Err(LlmError::InvalidResponse("Anthropic stream reached [DONE] before message_stop".to_string()).into());
                            return;
                        }
                        Ok(JsonSseEvent::Data(value)) => {
                            match serde_json::from_value::<AnthropicStreamEvent>(value) {
                                Ok(event) => event,
                                Err(error) => {
                                    yield Err(LlmError::InvalidResponse(format!(
                                        "invalid Anthropic SSE event: {error}"
                                    )).into());
                                    return;
                                }
                            }
                        }
                        Err(error) => {
                            yield Err(error);
                            return;
                        }
                    };
                    if pending_terminal.is_some() && !matches!(event, AnthropicStreamEvent::MessageStop | AnthropicStreamEvent::Other | AnthropicStreamEvent::Error { .. }) {
                        yield Err(LlmError::InvalidResponse("Anthropic stream emitted content after message_delta".to_string()).into());
                        return;
                    }
                                match event {
                                    AnthropicStreamEvent::MessageStart { message } => {
                                        // 从 message_start 中捕获初始用量（input_tokens）
                                        if let Some(u) = message.usage {
                                            stream_input_tokens = u.input_tokens;
                                            stream_cache_creation_input_tokens = u.cache_creation_input_tokens;
                                            stream_cache_read_input_tokens = u.cache_read_input_tokens;
                                        }
                                    }
                                    AnthropicStreamEvent::ContentBlockStart {
                                        index,
                                        content_block:
                                            ContentBlockStartBody::ToolUse { id, name },
                                    } => {
                                        // 开始跟踪一个新的 tool_use 块
                                        tool_call_args.insert(index, (id, name, String::new()));
                                    }
                                    AnthropicStreamEvent::ContentBlockStart {
                                        index,
                                        content_block: ContentBlockStartBody::Thinking { thinking, signature },
                                    } => {
                                        reasoning_blocks.insert(index, ReasoningBlock::Signed { thinking, signature });
                                    }
                                    AnthropicStreamEvent::ContentBlockStart {
                                        index,
                                        content_block: ContentBlockStartBody::RedactedThinking { data },
                                    } => {
                                        reasoning_blocks.insert(index, ReasoningBlock::Redacted { data });
                                    }
                                    AnthropicStreamEvent::ContentBlockStart { .. } => {
                                        // 文本块开始 —— 无需任何处理
                                    }
                                    AnthropicStreamEvent::ContentBlockDelta {
                                        index,
                                        delta,
                                    } => {
                                        if let Some(text) = delta.text {
                                            yield Ok(ChatChunk {
                                                delta: DeltaMessage {
                                                    role: Some("assistant".to_string()),
                                                    content: Some(text),
                                                    reasoning_content: None,
                                                    reasoning_blocks: None,
                                                    tool_calls: None,
                                                },
                                                finish_reason: None,
                                                usage: None,
                                            });
                                        } else if let Some(thinking) = delta.thinking {
                                            yield Ok(ChatChunk {
                                                delta: DeltaMessage {
                                                    role: Some("assistant".to_string()),
                                                    content: None,
                                                    reasoning_content: Some(thinking),
                                                    reasoning_blocks: None,
                                                    tool_calls: None,
                                                },
                                                finish_reason: None,
                                                usage: None,
                                            });
                                        } else if let Some(signature) = delta.signature {
                                            if let Some(ReasoningBlock::Signed { signature: value, .. }) = reasoning_blocks.get_mut(&index) {
                                                value.push_str(&signature);
                                            }
                                        } else if let Some(partial) = delta.partial_json {
                                            // 累积 tool_use 的参数
                                            if let Some(entry) = tool_call_args.get_mut(&index) {
                                                entry.2.push_str(&partial);
                                            }
                                        }
                                    }
                                    AnthropicStreamEvent::ContentBlockStop { index } => {
                                        if let Some(block) = reasoning_blocks.remove(&index) {
                                            yield Ok(ChatChunk {
                                                delta: DeltaMessage {
                                                    role: None,
                                                    content: None,
                                                    reasoning_content: None,
                                                    reasoning_blocks: Some(vec![block]),
                                                    tool_calls: None,
                                                },
                                                finish_reason: None,
                                                usage: None,
                                            });
                                        }
                                        // 完成工具调用并发出
                                        if let Some((id, name, args)) =
                                            tool_call_args.remove(&index)
                                        {
                                            let parsed_args = match serde_json::from_str::<serde_json::Value>(&args) {
                                                Ok(value) => value,
                                                Err(error) => {
                                                    yield Err(LlmError::InvalidResponse(format!(
                                                        "invalid Anthropic tool arguments for '{name}': {error}"
                                                    )).into());
                                                    return;
                                                }
                                            };
                                            yield Ok(ChatChunk {
                                                delta: DeltaMessage {
                                                    role: None,
                                                    content: None,
                                                    reasoning_content: None,
                                                    reasoning_blocks: None,
                                                    tool_calls: Some(vec![DeltaToolCall {
                                                        index: index as u32,
                                                        id: Some(id),
                                                        call_type: Some("function".to_string()),
                                                        function: Some(DeltaFunctionCall {
                                                            name: Some(name),
                                                            arguments: Some(
                                                                parsed_args.to_string(),
                                                            ),
                                                        }),
                                                    }]),
                                                },
                                                finish_reason: None,
                                                usage: None,
                                            });
                                        }
                                    }
                                    AnthropicStreamEvent::MessageDelta { delta, usage } => {
                                        // 从 message_delta 中捕获 output_tokens
                                        if let Some(u) = usage {
                                            stream_output_tokens = u.output_tokens;
                                        }
                                        let finish = match delta.stop_reason.as_deref() {
                                            Some("end_turn" | "stop_sequence") => Some("stop".to_string()),
                                            Some("tool_use") => Some("tool_calls".to_string()),
                                            other => other.map(String::from),
                                        };
                                        // 该 delta 是临时的，需等待 message_stop 确认完成后才生效。
                                        let usage = if stream_input_tokens > 0 || stream_output_tokens > 0 {
                                            Some(Usage {
                                                prompt_tokens: Some(stream_input_tokens),
                                                completion_tokens: Some(stream_output_tokens),
                                                total_tokens: Some(stream_input_tokens.saturating_add(stream_output_tokens)),
                                                cache_creation_input_tokens: stream_cache_creation_input_tokens,
                                                cache_read_input_tokens: stream_cache_read_input_tokens,
                                                ..Default::default()
                                            })
                                        } else {
                                            None
                                        };
                                        pending_terminal = Some(ChatChunk {
                                            delta: DeltaMessage {
                                                role: None,
                                                content: None,
                                                reasoning_content: None,
                                                reasoning_blocks: None,
                                                tool_calls: None,
                                            },
                                            finish_reason: finish,
                                            usage,
                                        });
                                    }
                                    AnthropicStreamEvent::MessageStop => {
                                        let Some(terminal) = pending_terminal.take() else {
                                            yield Err(LlmError::InvalidResponse("Anthropic message_stop preceded message_delta".to_string()).into());
                                            return;
                                        };
                                        let Some(reason) = terminal.finish_reason.as_deref() else {
                                            yield Err(LlmError::InvalidResponse("Anthropic message_stop omitted a stop reason".to_string()).into());
                                            return;
                                        };
                                        if !matches!(reason, "stop" | "tool_calls") || !tool_call_args.is_empty() || !reasoning_blocks.is_empty() {
                                            yield Err(LlmError::InvalidResponse(format!(
                                                "Anthropic message_stop has non-success or unfinished content: '{reason}'"
                                            )).into());
                                            return;
                                        }
                                        yield Ok(terminal);
                                        return;
                                    }
                                    AnthropicStreamEvent::Error { error } => {
                                        yield Err(LlmError::InvalidResponse(format!(
                                            "Anthropic stream error: {}",
                                            error.message
                                        )).into());
                                        return;
                                    }
                                    AnthropicStreamEvent::Other => {}
                                }
                }
                yield Err(LlmError::InvalidResponse("Anthropic stream reached SSE EOF before message_stop".to_string()).into());
            };

            Ok(Box::pin(stream) as BoxStream<'_, Result<ChatChunk>>)
        }
        .instrument(info_span!("anthropic_chat_stream", model = %model)),
        )
    }

    /// 返回该客户端配置的目标模型名称。
    fn model_name(&self) -> &str {
        &self.model
    }
}

// ── Anthropic API 类型 ──────────────────────────────────────────────────────

/// Anthropic 提示词缓存（prompt caching）使用的缓存控制标记。
#[derive(Serialize, Deserialize, Debug, Clone)]
struct CacheControl {
    /// 缓存类型，目前恒为 `"ephemeral"`。
    #[serde(rename = "type")]
    cache_type: String,
}

impl CacheControl {
    /// 构造一个短期（ephemeral）缓存控制标记。
    fn ephemeral() -> Self {
        Self {
            cache_type: "ephemeral".to_string(),
        }
    }
}

/// 可以携带 cache_control 的系统提示词块。
#[derive(Serialize)]
struct SystemBlock {
    /// 块类型，恒为 `"text"`。
    #[serde(rename = "type")]
    block_type: String,
    /// 系统提示词文本内容。
    text: String,
    /// 该块上的缓存控制标记（如果需要缓存断点）。
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_control: Option<CacheControl>,
}

/// 发送给 Anthropic Messages API 的完整请求体。
#[derive(Serialize)]
struct AnthropicRequest {
    /// 目标模型名称。
    model: String,
    /// 最大生成 token 数。
    max_tokens: u32,
    /// 系统提示词块列表（可携带缓存控制）。
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<Vec<SystemBlock>>,
    /// 会话消息列表。
    messages: Vec<AnthropicMessage>,
    /// 采样温度。
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    /// 可供模型调用的工具定义列表。
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<AnthropicToolDef>>,
    /// 是否以流式（SSE）方式返回响应。
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    /**
    扩展思考（extended-thinking）块。

    - Claude 3.7–4.5（`AnthropicThinkingBudget`）：`{type:"enabled", budget_tokens:N}`。
    - Claude 4.6（`AnthropicEffort`）：`{type:"adaptive"}`（模型根据 `effort`
      字段自行决定思考深度）；此时不会带上 budget_tokens。
    - Claude Opus 4.7+（`AnthropicAdaptive`）：该字段整个省略
      （发送会返回 400）。
    */
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<AnthropicThinking>,
    /**
    思考强度控制（Claude 4.5+，主要用于 4.6）。取值为
    `low`/`medium`/`high`/`xhigh`/`max` 之一。在较新模型上作为推荐的
    深度调节旋钮，取代了 `budget_tokens`。
    */
    #[serde(skip_serializing_if = "Option::is_none")]
    effort: Option<String>,
    /**
    `metadata:{user_id}`，用于在 DeepSeek/Anthropic 端点上做提示词缓存
    （KVCache）隔离。当 `ChatRequest.user_id` 设置时填充；为 `None` 时
    整个字段省略（stage4 P4.1：cache_user_id 单一数据源）。
    */
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<AnthropicMetadata>,
}

/**
Anthropic 端点元数据，携带用于提示词缓存隔离的 `user_id`
（DeepSeek 在其兼容 Anthropic 的端点上要求提供该字段）。
*/
#[derive(Serialize)]
struct AnthropicMetadata {
    /// 用户标识，用于按用户隔离提示词缓存。
    user_id: String,
}

/**
Anthropic 的 `thinking` 块。对 3.7–4.5 而言是
`{type:"enabled", budget_tokens}`；对 4.6 而言是 `{type:"adaptive"}`。
*/
#[derive(Serialize)]
struct AnthropicThinking {
    /// 块类型：`"enabled"` 或 `"adaptive"`。
    #[serde(rename = "type")]
    block_type: &'static str,
    /// 思考预算 token 数（仅 `"enabled"` 类型使用）。
    #[serde(skip_serializing_if = "Option::is_none")]
    budget_tokens: Option<u32>,
}

/**
为一次请求解析出 Anthropic 的思考相关字段。

返回 `(thinking_block, effort)`：
- Claude 3.7–4.5（`AnthropicThinkingBudget`）：`{type:"enabled", budget_tokens:N}`
  + 可选的 `effort`（4.5 允许 budget 与 effort 同时存在）。
- Claude 4.6（`AnthropicEffort`）：`{type:"adaptive"}` + `effort`。4.6 会拒绝
  budget_tokens，所以这里不会带上它。
- Claude Opus 4.7+（`AnthropicAdaptive`）：两个字段都省略（旧版块会返回
  400；模型自行决定思考深度）。
- 其他协议 / `None`：不返回任何字段。
*/
fn build_anthropic_thinking(
    model: &str,
    thinking_protocol: ThinkingProtocol,
    thinking: &Option<echo_core::llm::ThinkingConfig>,
    max_tokens: u32,
) -> (Option<AnthropicThinking>, Option<String>) {
    use echo_core::llm::ThinkingProtocol as T;
    let Some(cfg) = thinking.as_ref() else {
        return (None, None);
    };
    match thinking_protocol {
        T::AnthropicThinkingBudget => {
            // 3.7–4.5：budget_tokens（4.5 上 effort 也同样被接受）。
            let block = cfg
                .to_anthropic_budget(max_tokens)
                .map(|budget| AnthropicThinking {
                    block_type: "enabled",
                    budget_tokens: Some(budget),
                });
            (block, None)
        }
        T::AnthropicEffort => {
            // Claude 4.6：自适应思考块 + effort。带上 budget_tokens 会返回 400。
            let block = if matches!(cfg, echo_core::llm::ThinkingConfig::Disabled) {
                None
            } else {
                Some(AnthropicThinking {
                    block_type: "adaptive",
                    budget_tokens: None,
                })
            };
            (block, cfg.to_anthropic_effort().map(str::to_string))
        }
        T::AnthropicAdaptive => {
            warn!(
                model = model,
                "thinking config ignored: Claude Opus 4.7+ use adaptive-only thinking (any field would 400)"
            );
            (None, None)
        }
        _ => (None, None),
    }
}

/// Anthropic Messages API 中的一条会话消息。
#[derive(Serialize)]
struct AnthropicMessage {
    /// 消息角色（`"user"` 或 `"assistant"`）。
    role: String,
    /// 消息内容，可以是纯文本或内容块列表。
    content: AnthropicContent,
}

impl AnthropicMessage {
    /// 在该消息内容上放置一个短期（ephemeral）缓存断点。
    fn add_cache_control_ephemeral(&mut self) {
        self.content.add_cache_control_ephemeral();
    }

    /**
    判断该消息是否属于易变的运行时上下文（例如包含
    `[runtime_context:...]` 前缀的内容），这类消息不适合作为缓存断点。
    */
    fn is_runtime_context(&self) -> bool {
        self.content.is_runtime_context()
    }
}

/// 消息内容：可以是简单的纯文本，也可以是结构化的内容块列表。
#[derive(Serialize)]
#[serde(untagged)]
enum AnthropicContent {
    /// 纯文本内容。
    Text(String),
    /// 结构化内容块列表（文本、图片、工具调用等）。
    Blocks(Vec<ContentBlock>),
}

impl AnthropicContent {
    /**
    为该内容放置一个短期（ephemeral）缓存断点：纯文本会先转换为文本块，
    结构化内容则在最后一个可缓存的块上设置。
    */
    fn add_cache_control_ephemeral(&mut self) {
        match self {
            AnthropicContent::Text(text) => {
                let text = std::mem::take(text);
                *self = AnthropicContent::Blocks(vec![ContentBlock::Text {
                    text,
                    cache_control: Some(CacheControl::ephemeral()),
                }]);
            }
            AnthropicContent::Blocks(blocks) => {
                if let Some(block) = blocks.iter_mut().rev().find(|block| block.is_cacheable()) {
                    block.set_cache_control(CacheControl::ephemeral());
                }
            }
        }
    }

    /// 判断该内容是否属于易变的运行时上下文，不应作为缓存断点。
    fn is_runtime_context(&self) -> bool {
        match self {
            AnthropicContent::Text(text) => is_runtime_context_text(text),
            AnthropicContent::Blocks(blocks) => blocks.iter().any(ContentBlock::is_runtime_context),
        }
    }
}

/// Anthropic Messages API 中的单个内容块。
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type")]
enum ContentBlock {
    /// 纯文本块。
    #[serde(rename = "text")]
    Text {
        /// 文本内容。
        text: String,
        /// 缓存控制标记。
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    /// 图片块。
    #[serde(rename = "image")]
    Image {
        /// 图片来源（base64 数据或 URL）。
        source: ImageSource,
        /// 缓存控制标记。
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    /**
    PDF 文档内容块。

    Anthropic 的 base64 文档来源只接受 `application/pdf`（其他 mime 类型会
    失败）。文本类文件改为内联为 Text 块 —— 参见下方
    `ContentPart::File` 的处理逻辑。
    */
    #[serde(rename = "document")]
    Document {
        /// 文档来源（base64 数据或 URL）。
        source: ImageSource,
        /// 缓存控制标记。
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    /// 模型发起的工具调用块。
    #[serde(rename = "tool_use")]
    ToolUse {
        /// 工具调用的唯一标识。
        id: String,
        /// 被调用的工具名称。
        name: String,
        /// 调用参数（JSON 值）。
        input: serde_json::Value,
    },
    /// 工具执行结果块，作为 user 消息回传给模型。
    #[serde(rename = "tool_result")]
    ToolResult {
        /// 对应的工具调用 ID。
        tool_use_id: String,
        /// 工具执行结果文本。
        content: String,
        /// 缓存控制标记。
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    /// 带签名的思考（thinking）块。
    #[serde(rename = "thinking")]
    Thinking {
        /// 思考过程文本。
        thinking: String,
        /// 思考内容的签名，用于后续回放校验。
        signature: String,
    },
    /// 被脱敏（redacted）的思考块，原始内容不可见，仅保留不透明数据。
    #[serde(rename = "redacted_thinking")]
    RedactedThinking {
        /// 不透明的脱敏数据。
        data: String,
    },
    /// 未识别的其他块类型，作为兜底忽略处理。
    #[serde(other)]
    Other,
}

impl ContentBlock {
    /// 判断该内容块是否允许放置缓存断点（文本、图片、工具结果）。
    fn is_cacheable(&self) -> bool {
        matches!(
            self,
            ContentBlock::Text { .. }
                | ContentBlock::Image { .. }
                | ContentBlock::ToolResult { .. }
        )
    }

    /**
    为该内容块设置缓存控制标记；对不支持缓存的块类型（工具调用、思考块等）
    不做任何处理。
    */
    fn set_cache_control(&mut self, cache_control: CacheControl) {
        match self {
            ContentBlock::Text {
                cache_control: field,
                ..
            }
            | ContentBlock::Image {
                cache_control: field,
                ..
            }
            | ContentBlock::Document {
                cache_control: field,
                ..
            }
            | ContentBlock::ToolResult {
                cache_control: field,
                ..
            } => *field = Some(cache_control),
            ContentBlock::ToolUse { .. }
            | ContentBlock::Thinking { .. }
            | ContentBlock::RedactedThinking { .. }
            | ContentBlock::Other => {}
        }
    }

    /**
    判断该内容块是否属于易变的运行时上下文（文本或工具结果文本以
    `[runtime_context:` 开头）。
    */
    fn is_runtime_context(&self) -> bool {
        match self {
            ContentBlock::Text { text, .. } => is_runtime_context_text(text),
            ContentBlock::ToolResult { content, .. } => is_runtime_context_text(content),
            _ => false,
        }
    }
}

/**
将助手消息携带的思考块（reasoning blocks）追加到目标内容块列表中，
用于在下一轮请求里回放模型此前的思考过程。
*/
fn append_reasoning_blocks(
    target: &mut Vec<ContentBlock>,
    reasoning_blocks: Option<&[ReasoningBlock]>,
) {
    for block in reasoning_blocks.unwrap_or_default() {
        let content = match block {
            ReasoningBlock::Signed {
                thinking,
                signature,
            } => ContentBlock::Thinking {
                thinking: thinking.clone(),
                signature: signature.clone(),
            },
            ReasoningBlock::Redacted { data } => {
                ContentBlock::RedactedThinking { data: data.clone() }
            }
            /*
            不透明（Opaque）状态归属于另一个 provider，当会话切换模型时
            不能跨协议边界传递，因此直接跳过。
            */
            ReasoningBlock::Opaque { .. } => continue,
        };
        target.push(content);
    }
}

/**
在会话消息上应用默认的缓存断点放置策略：跳过尾部的易变运行时上下文，
并根据剩余可用断点数在稳定前缀的末尾（以及约 3/4 处，如果还有余量）
放置断点。
*/
fn apply_conversation_cache_breakpoints(
    messages: &mut [AnthropicMessage],
    remaining_breakpoints: usize,
) {
    if remaining_breakpoints == 0 || messages.is_empty() {
        return;
    }

    let stable_end = messages
        .iter()
        .rposition(|message| !message.is_runtime_context())
        .map(|index| index + 1)
        .unwrap_or(0);
    if stable_end == 0 {
        return;
    }

    let mut indexes = Vec::new();
    if remaining_breakpoints >= 2 && stable_end >= 4 {
        indexes.push((stable_end - 1) * 3 / 4);
    }
    indexes.push(stable_end - 1);
    indexes.sort_unstable();
    indexes.dedup();

    for index in indexes.into_iter().take(remaining_breakpoints) {
        if let Some(message) = messages.get_mut(index) {
            message.add_cache_control_ephemeral();
        }
    }
}

/// 判断文本是否以 `[runtime_context:` 开头，即是否属于易变的运行时上下文。
fn is_runtime_context_text(text: &str) -> bool {
    text.trim_start().starts_with("[runtime_context:")
}

/// 图片/文档内容的来源：既可以是 base64 编码的数据，也可以是 URL。
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ImageSource {
    /// 以 base64 内联的二进制数据。
    Base64 {
        /// MIME 媒体类型（如 `image/png`、`application/pdf`）。
        media_type: String,
        /// base64 编码后的数据。
        data: String,
    },
    /// 指向图片/文档的远程 URL。
    #[serde(rename = "url")]
    Url_ {
        /// 远程资源地址。
        url: String,
    },
}

/// 将 data URL 解析为 Anthropic 的 ImageSource。
fn data_url_to_image_source(url: &str) -> ImageSource {
    if let Some(rest) = url.strip_prefix("data:")
        && let Some((media_type, b64_data)) = rest.split_once(';')
        && let Some(data) = b64_data.strip_prefix("base64,")
    {
        return ImageSource::Base64 {
            media_type: media_type.to_string(),
            data: data.to_string(),
        };
    }
    ImageSource::Url_ {
        url: url.to_string(),
    }
}

/**
将 `ContentPart::File` 转换为最合适的 Anthropic 内容块。

Anthropic 的 Messages API 只接受 `application/pdf` 作为 base64 `document`
来源 —— 其他二进制类型没有通用的附件块。为了让文本类附件
（txt/md/json/src/...）真正可被模型阅读，这里会解码并内联为文本。
非 PDF 的二进制类型则回退为仅保留文件名的占位符（这是此前所有文件的
默认行为）。
*/
fn file_to_content_block(name: &str, content_base64: &str) -> ContentBlock {
    let ext = name.rsplit('.').next().map(|e| e.to_ascii_lowercase());
    match ext.as_deref() {
        // PDF → document 内容块（Anthropic 唯一接受的 base64 文档类型）。
        Some("pdf") => ContentBlock::Document {
            source: ImageSource::Base64 {
                media_type: "application/pdf".to_string(),
                data: content_base64.to_string(),
            },
            cache_control: None,
        },
        // 文本类文件 → 解码 base64 并内联为文本，让模型可以直接阅读内容。
        Some(
            "txt" | "md" | "markdown" | "json" | "xml" | "yaml" | "yml" | "csv" | "tsv" | "rs"
            | "py" | "js" | "ts" | "tsx" | "jsx" | "go" | "java" | "c" | "cpp" | "h" | "sh"
            | "toml" | "ini" | "log" | "sql",
        ) => {
            let decoded =
                base64::Engine::decode(&base64::engine::general_purpose::STANDARD, content_base64)
                    .ok();
            let text = decoded
                .and_then(|bytes| String::from_utf8(bytes).ok())
                .unwrap_or_else(|| format!("[Attachment: {name}] (binary, undecodable)"));
            ContentBlock::Text {
                text: format!("\n[Attachment: {name}]\n```\n{text}\n```"),
                cache_control: None,
            }
        }
        // 其他二进制类型 → 仅保留文件名的占位符（没有通用的附件块）。
        _ => ContentBlock::Text {
            text: format!("\n[Attachment: {name}]"),
            cache_control: None,
        },
    }
}

/// 发送给 Anthropic 的工具定义。
#[derive(Serialize)]
struct AnthropicToolDef {
    /// 工具名称。
    name: String,
    /// 工具描述。
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    /// 工具入参的 JSON Schema。
    input_schema: serde_json::Value,
    /// 缓存控制标记（放在最后一个工具上）。
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_control: Option<CacheControl>,
}

/// Anthropic Messages API 的非流式响应体。
#[derive(Deserialize)]
struct AnthropicResponse {
    /// 响应内容块列表。
    content: Vec<ContentBlock>,
    /// 停止原因（如 `end_turn`、`tool_use`、`max_tokens`）。
    stop_reason: Option<String>,
    /// token 用量统计。
    #[serde(default)]
    usage: Option<AnthropicUsage>,
}

/// Anthropic 响应中的 token 用量统计。
#[derive(Deserialize, Clone)]
struct AnthropicUsage {
    /// 输入 token 数。
    input_tokens: u32,
    /// 输出 token 数。
    output_tokens: u32,
    /// 因创建缓存而消耗的 token 数。
    #[serde(default)]
    cache_creation_input_tokens: Option<u32>,
    /// 命中缓存而节省的 token 数。
    #[serde(default)]
    cache_read_input_tokens: Option<u32>,
}

/// `message_delta` 流式事件中携带的增量用量统计（仅有输出 token）。
#[derive(Deserialize, Clone)]
struct AnthropicDeltaUsage {
    /// 截至目前的输出 token 数。
    output_tokens: u32,
}

/// Anthropic SSE 流中的单个事件。
#[derive(Deserialize)]
#[serde(tag = "type")]
enum AnthropicStreamEvent {
    /// 流式响应开始，携带初始的消息信息（含用量）。
    #[serde(rename = "message_start")]
    MessageStart {
        /// 消息起始时的元数据。
        message: MessageStartBody,
    },
    /// 一个新的内容块开始。
    #[serde(rename = "content_block_start")]
    ContentBlockStart {
        /// 该内容块在响应中的索引。
        index: usize,
        /// 内容块起始时携带的信息。
        content_block: ContentBlockStartBody,
    },
    /// 某个内容块的增量更新。
    #[serde(rename = "content_block_delta")]
    ContentBlockDelta {
        /// 对应内容块的索引。
        index: usize,
        /// 增量内容。
        delta: ContentDelta,
    },
    /// 某个内容块结束。
    #[serde(rename = "content_block_stop")]
    ContentBlockStop {
        /// 对应内容块的索引。
        index: usize,
    },
    /// 消息级别的增量更新（携带停止原因和用量）。
    #[serde(rename = "message_delta")]
    MessageDelta {
        /// 消息增量内容（停止原因）。
        delta: MessageDeltaBody,
        /// 截至目前的用量统计。
        #[serde(default)]
        usage: Option<AnthropicDeltaUsage>,
    },
    /// 流式响应结束。
    #[serde(rename = "message_stop")]
    MessageStop,
    /// 流中发生错误。
    #[serde(rename = "error")]
    Error {
        /// 错误详情。
        error: AnthropicStreamError,
    },
    /// 未识别的其他事件类型，兜底忽略。
    #[serde(other)]
    Other,
}

/// Anthropic 流式错误事件的详情。
#[derive(Deserialize)]
struct AnthropicStreamError {
    /// 错误信息文本。
    message: String,
}

/// `message_start` 事件携带的消息元数据。
#[derive(Deserialize)]
struct MessageStartBody {
    /// 初始用量统计（主要是 input_tokens）。
    #[serde(default)]
    usage: Option<AnthropicUsage>,
}

/// `content_block_start` 事件中携带的内容块起始信息。
#[derive(Deserialize)]
#[serde(tag = "type")]
enum ContentBlockStartBody {
    /// 工具调用块开始。
    #[serde(rename = "tool_use")]
    ToolUse {
        /// 工具调用 ID。
        id: String,
        /// 工具名称。
        name: String,
    },
    /// 思考块开始。
    #[serde(rename = "thinking")]
    Thinking {
        /// 初始的思考文本（通常为空，后续通过 delta 累积）。
        #[serde(default)]
        thinking: String,
        /// 初始的签名（通常为空，后续通过 delta 累积）。
        #[serde(default)]
        signature: String,
    },
    /// 脱敏思考块开始。
    #[serde(rename = "redacted_thinking")]
    RedactedThinking {
        /// 不透明的脱敏数据。
        data: String,
    },
    /// 未识别的其他内容块类型，兜底忽略。
    #[serde(other)]
    Other,
}

/// 内容块的增量更新，各字段互斥，按事件类型只有其中一个会被填充。
#[derive(Deserialize)]
struct ContentDelta {
    /// 文本增量。
    text: Option<String>,
    /// 思考文本增量。
    thinking: Option<String>,
    /// 思考签名增量。
    signature: Option<String>,
    /// 工具调用参数（JSON）的增量片段。
    #[serde(rename = "partial_json")]
    partial_json: Option<String>,
}

/// `message_delta` 事件携带的消息级增量信息。
#[derive(Deserialize)]
struct MessageDeltaBody {
    /// 停止原因（如 `end_turn`、`tool_use`）。
    stop_reason: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::Engine as _;

    /// 构造一条纯文本用户消息。
    fn text_message(text: &str) -> AnthropicMessage {
        AnthropicMessage {
            role: "user".to_string(),
            content: AnthropicContent::Text(text.to_string()),
        }
    }

    /// 判断消息中是否有内容块带有 cache_control 标记。
    fn has_cache_control(message: &AnthropicMessage) -> bool {
        match &message.content {
            AnthropicContent::Text(_) => false,
            AnthropicContent::Blocks(blocks) => blocks.iter().any(|block| match block {
                ContentBlock::Text { cache_control, .. }
                | ContentBlock::Image { cache_control, .. }
                | ContentBlock::Document { cache_control, .. }
                | ContentBlock::ToolResult { cache_control, .. } => cache_control.is_some(),
                ContentBlock::ToolUse { .. }
                | ContentBlock::Thinking { .. }
                | ContentBlock::RedactedThinking { .. }
                | ContentBlock::Other => false,
            }),
        }
    }

    /// 验证会话缓存断点会跳过末尾的运行时上下文消息。
    #[test]
    fn conversation_cache_breakpoints_skip_trailing_runtime_context() {
        let mut messages = vec![
            text_message("older turn"),
            text_message("middle turn"),
            text_message("current request"),
            text_message("[runtime_context:turn]\nvolatile cwd and memory"),
        ];

        apply_conversation_cache_breakpoints(&mut messages, 2);

        assert!(messages.get(2).is_some_and(has_cache_control));
        assert!(
            messages
                .get(3)
                .is_some_and(|message| !has_cache_control(message))
        );
    }

    /*
    ── stage4 P4.1：cache_user_id 单一数据源 ────────────────────────────
    DeepSeek/Anthropic 端点使用 metadata:{user_id} 做 KVCache 隔离。
    验证 AnthropicClient.convert_request 在 ChatRequest.user_id 设置时
    会填充 metadata，未设置（None）时则完全省略该字段。
    */

    /// 构造带有可选 user_id 的聊天请求。
    fn chat_request_with_user(user: Option<&str>) -> ChatRequest {
        ChatRequest {
            messages: vec![Message::user("hi".to_string())],
            user_id: user.map(str::to_string),
            ..ChatRequest::default()
        }
    }

    /// 验证 Claude 4.6 的 max 推理强度保持为 max。
    #[test]
    fn claude_46_max_effort_remains_max() {
        let (thinking, effort) = build_anthropic_thinking(
            "claude-opus-4-6",
            ThinkingProtocol::AnthropicEffort,
            &Some(echo_core::llm::ThinkingConfig::Level(
                echo_core::llm::ThinkingLevel::Max,
            )),
            8_192,
        );
        assert_eq!(effort.as_deref(), Some("max"));
        assert_eq!(thinking.map(|block| block.block_type), Some("adaptive"));
    }

    /// 验证设置 user_id 时请求包含 metadata.user_id。
    #[test]
    fn metadata_user_id_present_when_set() -> std::result::Result<(), serde_json::Error> {
        let client = AnthropicClient::new("ds-xxx".to_string(), "deepseek-chat".to_string());
        let body =
            serde_json::to_value(client.convert_request(&chat_request_with_user(Some("user-7"))))?;
        assert_eq!(
            body.pointer("/metadata/user_id")
                .and_then(serde_json::Value::as_str),
            Some("user-7")
        );
        Ok(())
    }

    /// 验证 user_id 为 None 时请求不包含 metadata。
    #[test]
    fn metadata_absent_when_user_id_none() -> std::result::Result<(), serde_json::Error> {
        let client = AnthropicClient::new("ds-xxx".to_string(), "deepseek-chat".to_string());
        let body = serde_json::to_value(client.convert_request(&chat_request_with_user(None)))?;
        // metadata 标注了 skip_serializing_if Option::is_none → 应缺省（不是 null）。
        assert!(
            body.get("metadata").is_none(),
            "metadata should be absent when user_id is None, got: {body}"
        );
        Ok(())
    }

    /// 验证多条系统消息按原顺序保留。
    #[test]
    fn multiple_system_messages_are_preserved_in_order()
    -> std::result::Result<(), serde_json::Error> {
        let client = AnthropicClient::new("sk-test", "claude-sonnet-4-6");
        let request = ChatRequest {
            messages: vec![
                Message::system("base rules".to_string()),
                Message::system("restored context".to_string()),
                Message::user("hello".to_string()),
            ],
            ..ChatRequest::default()
        };
        let body = serde_json::to_value(client.convert_request(&request))?;
        assert_eq!(
            body.pointer("/system/0/text")
                .and_then(serde_json::Value::as_str),
            Some("base rules\n\nrestored context")
        );
        Ok(())
    }

    /// 验证 message_delta 的 usage 仅含 output_tokens 时也能解析。
    #[test]
    fn message_delta_usage_accepts_output_tokens_only() -> std::result::Result<(), String> {
        let raw = r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":17}}"#;
        let event =
            serde_json::from_str::<AnthropicStreamEvent>(raw).map_err(|error| error.to_string())?;
        let AnthropicStreamEvent::MessageDelta { usage, .. } = event else {
            return Err("expected message_delta event".to_string());
        };
        assert_eq!(usage.map(|value| value.output_tokens), Some(17));
        Ok(())
    }

    /// 验证工具调用开始事件保留提供商的内容块索引。
    #[test]
    fn tool_start_keeps_provider_block_index() -> std::result::Result<(), String> {
        let raw = r#"{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"call-1","name":"read_file","input":{}}}"#;
        let event =
            serde_json::from_str::<AnthropicStreamEvent>(raw).map_err(|error| error.to_string())?;
        let AnthropicStreamEvent::ContentBlockStart { index, .. } = event else {
            return Err("expected content_block_start event".to_string());
        };
        assert_eq!(index, 1);
        Ok(())
    }

    /// 验证 thinking 响应会投射为 reasoning_content。
    #[test]
    fn thinking_response_is_projected_to_reasoning_content()
    -> std::result::Result<(), serde_json::Error> {
        let raw = r#"{"content":[{"type":"thinking","thinking":"reason","signature":"sig"},{"type":"text","text":"answer"}],"stop_reason":"end_turn","usage":{"input_tokens":2,"output_tokens":3}}"#;
        let response = serde_json::from_str::<AnthropicResponse>(raw)?;
        let converted =
            AnthropicClient::new("sk-test", "claude-sonnet-4-6").convert_response(response);
        assert_eq!(
            converted.message.reasoning_content.as_deref(),
            Some("reason")
        );
        assert_eq!(
            converted.message.content.as_text().as_deref(),
            Some("answer")
        );
        assert_eq!(
            converted.message.reasoning_blocks,
            Some(vec![ReasoningBlock::Signed {
                thinking: "reason".to_string(),
                signature: "sig".to_string(),
            }])
        );
        assert_eq!(
            converted.usage.and_then(|usage| usage.total_tokens),
            Some(5)
        );
        Ok(())
    }

    /// 验证带签名的推理块会在工具调用之前回放。
    #[test]
    fn signed_reasoning_blocks_are_replayed_before_tool_calls() {
        let client = AnthropicClient::new("sk-test", "claude-sonnet-4-6");
        let mut assistant = Message::assistant_with_tools(vec![ToolCall {
            id: "call-1".to_string(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: "read_file".to_string(),
                arguments: "{}".to_string(),
            },
        }]);
        assistant.reasoning_blocks = Some(vec![ReasoningBlock::Signed {
            thinking: "reason".to_string(),
            signature: "sig".to_string(),
        }]);
        let request = client.convert_request(&ChatRequest::new(vec![assistant]));

        let blocks = request
            .messages
            .first()
            .and_then(|message| match &message.content {
                AnthropicContent::Blocks(blocks) => Some(blocks),
                AnthropicContent::Text(_) => None,
            });
        assert!(matches!(
            blocks.and_then(|values| values.first()),
            Some(ContentBlock::Thinking { thinking, signature })
                if thinking == "reason" && signature == "sig"
        ));
        assert!(matches!(
            blocks.and_then(|values| values.get(1)),
            Some(ContentBlock::ToolUse { name, .. }) if name == "read_file"
        ));
    }

    /// 验证不支持的请求特性会直接报错而非被静默丢弃。
    #[test]
    fn unsupported_request_features_fail_instead_of_being_dropped() {
        let client = AnthropicClient::new("test-key", "test-model");
        let mut request = ChatRequest::new(vec![Message::user("hello".to_string())]);
        request.tool_choice = Some("none".to_string());
        let tool_error = client
            .validate_request_features(&request)
            .err()
            .map(|error| error.to_string())
            .unwrap_or_default();
        assert!(tool_error.contains("tool_choice"));

        request.tool_choice = None;
        request.response_format = Some(echo_core::llm::types::ResponseFormat::JsonObject);
        let format_error = client
            .validate_request_features(&request)
            .err()
            .map(|error| error.to_string())
            .unwrap_or_default();
        assert!(format_error.contains("response format"));
    }

    /**
    Sprint 2：主思考路径发送 `cache_hints: Some`，但断点列表**为空**
    （agent 为计算哈希而算出布局，但把断点推导留给供应商）。
    供应商必须回退到 `AnthropicCachePlan::from_layout` 并放置 cache_control，
    而不是静默地输出零个断点（修复前的 bug：has_system/tool=false、
    history_breakpoint_count=0 → 主路径上没有任何 cache_control）。
    */
    #[test]
    fn cache_hints_with_empty_breakpoints_still_places_cache_control()
    -> std::result::Result<(), serde_json::Error> {
        use echo_core::llm::cache::CacheHints;
        use echo_core::llm::types::{FunctionSpec, Message, ToolDefinition};

        let messages = vec![
            Message::system("You are Echo Agent".to_string()),
            Message::user("h1".to_string()),
            Message::user("h2".to_string()),
            Message::user("h3".to_string()),
            Message::user("h4".to_string()),
        ];
        let tools = vec![ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionSpec {
                name: "t".to_string(),
                description: "d".to_string(),
                parameters: serde_json::json!({}),
            },
        }];
        /*
        主路径的形态：Some(CacheHints { breakpoints: vec![], .. })。
        供应商会自行根据 request.messages+tools 重新计算布局
        （不会读取 hints.segments），因此 segments 可以使用默认值。
        */
        let req = ChatRequest {
            messages,
            tools: Some(tools),
            cache_hints: Some(CacheHints {
                breakpoints: vec![],
                stable_prefix_hash: Some("deadbeef".to_string()),
                segments: Default::default(),
            }),
            ..ChatRequest::default()
        };

        let client = AnthropicClient::new("sk-xxx".to_string(), "claude-sonnet-4-6".to_string());
        let body = serde_json::to_value(client.convert_request(&req))?;
        /*
        修复前：没有任何 cache_control。修复后：from_layout 会放置
        system + tools + 历史消息断点。
        */
        let body_str = body.to_string();
        assert!(
            body_str.contains("cache_control"),
            "main-path (empty breakpoints) must still place cache_control via from_layout fallback; got: {body_str}"
        );
        assert!(
            body_str.contains("ephemeral"),
            "cache_control must be ephemeral; got: {body_str}"
        );
        Ok(())
    }

    // ── 2B：file_to_content_block 按推断出的媒体类型分派 ──────────

    /// 验证 PDF 附件会转换为 document 内容块。
    #[test]
    fn pdf_attachment_becomes_document_block() -> std::result::Result<(), String> {
        let b64 = base64::engine::general_purpose::STANDARD.encode(b"%PDF-1.4 fake");
        let block = file_to_content_block("report.pdf", &b64);
        let ContentBlock::Document { source, .. } = block else {
            return Err("expected Document block".to_string());
        };
        let ImageSource::Base64 { media_type, data } = source else {
            return Err("expected base64 document source".to_string());
        };
        assert_eq!(media_type, "application/pdf");
        assert_eq!(data, b64);
        Ok(())
    }

    /// 验证文本附件会以内联文本形式发送。
    #[test]
    fn text_attachment_inlined_as_text() -> std::result::Result<(), String> {
        let b64 = base64::engine::general_purpose::STANDARD.encode(b"hello notes");
        let block = file_to_content_block("notes.txt", &b64);
        let ContentBlock::Text { text, .. } = block else {
            return Err("expected Text block".to_string());
        };
        assert!(
            text.contains("hello notes"),
            "text should contain decoded content"
        );
        assert!(
            text.contains("notes.txt"),
            "text should mention the filename"
        );
        Ok(())
    }

    /// 验证非 PDF 的二进制附件会以占位文本代替。
    #[test]
    fn binary_non_pdf_attachment_is_placeholder() -> std::result::Result<(), String> {
        let b64 = base64::engine::general_purpose::STANDARD.encode(b"\x00\x01binary");
        let block = file_to_content_block("archive.zip", &b64);
        let ContentBlock::Text { text, .. } = block else {
            return Err("expected Text placeholder".to_string());
        };
        assert!(text.contains("archive.zip"));
        // 不应包含解码后的二进制乱码。
        assert!(!text.contains("binary"));
        Ok(())
    }

    /// 验证字面流事件能保留用量、工具身份与部分 JSON。
    #[test]
    fn literal_stream_events_preserve_usage_tool_identity_and_partial_json()
    -> std::result::Result<(), serde_json::Error> {
        let start: AnthropicStreamEvent = serde_json::from_str(
            r#"{"type":"message_start","message":{"usage":{"input_tokens":13,"output_tokens":0,"cache_read_input_tokens":4}}}"#,
        )?;
        assert!(matches!(
            start,
            AnthropicStreamEvent::MessageStart {
                message: MessageStartBody {
                    usage: Some(AnthropicUsage {
                        input_tokens: 13,
                        cache_read_input_tokens: Some(4),
                        ..
                    })
                }
            }
        ));

        let tool: AnthropicStreamEvent = serde_json::from_str(
            r#"{"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"toolu-9","name":"read_file","input":{}}}"#,
        )?;
        assert!(matches!(
            tool,
            AnthropicStreamEvent::ContentBlockStart {
                index: 2,
                content_block: ContentBlockStartBody::ToolUse { id, name }
            } if id == "toolu-9" && name == "read_file"
        ));

        let delta: AnthropicStreamEvent = serde_json::from_str(
            r#"{"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"path\":\"文档.md\"}"}}"#,
        )?;
        assert!(matches!(
            delta,
            AnthropicStreamEvent::ContentBlockDelta {
                index: 2,
                delta: ContentDelta { partial_json: Some(value), .. }
            } if value == "{\"path\":\"文档.md\"}"
        ));

        let terminal: AnthropicStreamEvent = serde_json::from_str(
            r#"{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":7}}"#,
        )?;
        assert!(matches!(
            terminal,
            AnthropicStreamEvent::MessageDelta {
                delta: MessageDeltaBody { stop_reason: Some(reason) },
                usage: Some(AnthropicDeltaUsage { output_tokens: 7 })
            } if reason == "tool_use"
        ));
        Ok(())
    }
}
