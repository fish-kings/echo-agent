use crate::compression::compressor::{SlidingWindowCompressor, SummaryCompressor};
use crate::compression::{
    CompressionCheckpoint, CompressionInput, CompressionOutput, ContextCompressor,
};
use echo_core::error::Result;
use echo_core::llm::types::Message;
use futures::future::BoxFuture;
use std::time::Instant;

/**
 * 多阶段流水线压缩器。
 *
 * 将多个 [`ContextCompressor`] 实现按顺序串联成一条流水线。
 * 每一阶段的输出会作为下一阶段的输入。
 *
 * **短路机制**：启用时（默认启用），一旦估算的 token 数已降到 `token_limit`
 * 以下，流水线会跳过剩余阶段。这样可以避免在前面的阶段已经足够时，
 * 后续阶段仍发起不必要的 LLM 调用。
 *
 * # 示例
 *
 * ```rust,no_run
 * use echo_core::llm::LlmClient;
 * use echo_state::compression::compressor::{
 *     HybridCompressor, SlidingWindowCompressor, SummaryCompressor,
 * };
 * use std::sync::Arc;
 *
 * # async fn example(llm: Arc<dyn LlmClient>) {
 * let compressor = HybridCompressor::builder()
 *     .stage(SlidingWindowCompressor::new(20))
 *     .stage(SummaryCompressor::new(llm, 8))
 *     .build();
 * # }
 * ```
 */
pub struct HybridCompressor {
    /// 按顺序执行的压缩阶段列表。
    stages: Vec<Box<dyn ContextCompressor>>,
    /// 为 true 时，若 token 数已经处于或低于限制，则跳过剩余阶段。
    short_circuit: bool,
}

impl ContextCompressor for HybridCompressor {
    /// 返回该压缩器的名称。
    fn name(&self) -> &str {
        "Hybrid"
    }

    /// 依次执行各压缩阶段，必要时应用短路优化，并汇总生成最终检查点。
    fn compress(&self, input: CompressionInput) -> BoxFuture<'_, Result<CompressionOutput>> {
        Box::pin(async move {
            let start = Instant::now();
            let token_limit = input.token_limit;
            let tokenizer = input.tokenizer();
            let current_query = input.current_query.clone();
            let focus_instructions = input.focus_instructions.clone();
            let mut messages = input.messages;
            let mut all_evicted: Vec<Message> = Vec::new();
            let mut stage_names: Vec<String> = Vec::new();
            let mut last_summary: Option<String> = None;

            let tokens_before: usize = messages
                .iter()
                .filter_map(|m| m.content.as_text())
                .map(|c| tokenizer.count_tokens(&c))
                .sum();

            for (i, stage) in self.stages.iter().enumerate() {
                // 短路：如果 token 数已经处于或低于限制，跳过剩余阶段
                if self.short_circuit && i > 0 {
                    let current_tokens: usize = messages
                        .iter()
                        .filter_map(|m| m.content.as_text())
                        .map(|c| tokenizer.count_tokens(&c))
                        .sum();
                    if current_tokens <= token_limit {
                        tracing::debug!(
                            stage = i,
                            total_stages = self.stages.len(),
                            current_tokens,
                            token_limit,
                            "HybridCompressor: short-circuiting, skipping remaining stages"
                        );
                        break;
                    }
                }

                let output = stage
                    .compress(CompressionInput {
                        messages,
                        token_limit,
                        current_query: current_query.clone(),
                        focus_instructions: focus_instructions.clone(),
                        cancel_token: input.cancel_token.clone(),
                        tokenizer: Some(tokenizer.clone()),
                    })
                    .await?;
                // 如果检查点可用，收集该阶段的信息
                if let Some(ref cp) = output.checkpoint {
                    let checkpoint_preview: String = cp.checkpoint_id.chars().take(8).collect();
                    let stage_info = format!("{}(id={checkpoint_preview})", cp.strategy);
                    stage_names.push(stage_info);
                    if cp.summary.is_some() {
                        last_summary = cp.summary.clone();
                    }
                } else {
                    stage_names.push(stage.name().to_string());
                }
                all_evicted.extend(output.evicted);
                messages = output.messages;
            }

            let tokens_after: usize = messages
                .iter()
                .filter_map(|m| m.content.as_text())
                .map(|c| tokenizer.count_tokens(&c))
                .sum();

            let hybrid_strategy = format!("Hybrid({})", stage_names.join("+"));
            let mut checkpoint = CompressionCheckpoint::new(hybrid_strategy)
                .with_counts(messages.len(), all_evicted.len())
                .with_tokens(tokens_before, tokens_after)
                .with_duration_ms(start.elapsed().as_millis() as u64)
                .with_focus(focus_instructions.or(current_query));
            if let Some(s) = last_summary {
                checkpoint = checkpoint.with_summary(s);
            }

            Ok(CompressionOutput {
                messages,
                evicted: all_evicted,
                checkpoint: Some(checkpoint),
            })
        })
    }
}

impl HybridCompressor {
    /// 创建一个用于构建 [`HybridCompressor`] 的构建器。
    pub fn builder() -> HybridCompressorBuilder {
        HybridCompressorBuilder::default()
    }
}

/// [`HybridCompressor`] 的构建器
#[derive(Default)]
pub struct HybridCompressorBuilder {
    /// 已添加的压缩阶段，按添加顺序执行。
    stages: Vec<Box<dyn ContextCompressor>>,
    /// 是否启用短路优化；`None` 表示使用默认值（启用）。
    short_circuit: Option<bool>,
}

impl HybridCompressorBuilder {
    /// 追加一个压缩阶段（按添加顺序执行）。
    pub fn stage(mut self, compressor: impl ContextCompressor + 'static) -> Self {
        self.stages.push(Box::new(compressor));
        self
    }

    /**
     * 启用或禁用短路优化。
     *
     * 启用时（默认），一旦估算的 token 数降到或低于 `token_limit`，流水线
     * 会跳过剩余阶段。设为 `false` 则始终执行全部阶段。
     */
    pub fn short_circuit(mut self, enabled: bool) -> Self {
        self.short_circuit = Some(enabled);
        self
    }

    /// 消耗构建器，生成最终的 [`HybridCompressor`]。
    pub fn build(self) -> HybridCompressor {
        HybridCompressor {
            stages: self.stages,
            short_circuit: self.short_circuit.unwrap_or(true),
        }
    }
}

impl HybridCompressor {
    /**
     * 对齐 LangChain `ConversationSummaryBufferMemory` 的便捷构造函数：
     * 一个两阶段流水线，先对较早的消息做摘要，再对最近的消息保留原文滑动窗口。
     *
     * 阶段顺序为 `[SummaryCompressor, SlidingWindowCompressor]`——先对旧消息
     * 做摘要（有损），再原样保留最近的 `window` 条消息（无损）。启用了短路：
     * 如果摘要阶段已经把 token 数降到限制以下，滑动窗口阶段会被跳过。
     *
     * 这是业界主流的“混合”方案（摘要上下文 + 滑动窗口），
     * 与 LangChain 生产环境中的混合记忆机制一致。
     */
    pub fn summary_buffer(
        llm: std::sync::Arc<dyn echo_core::llm::LlmClient>,
        window: usize,
    ) -> Self {
        Self {
            stages: vec![
                Box::new(SummaryCompressor::new(llm, window)),
                Box::new(SlidingWindowCompressor::new(window)),
            ],
            short_circuit: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compression::compressor::SlidingWindowCompressor;
    use echo_core::llm::types::{MessageContent, Role};

    /// 构造一条测试用消息。
    fn make_msg(role: Role, text: &str) -> Message {
        Message {
            role,
            content: MessageContent::Text(text.to_string()),
            ..Default::default()
        }
    }

    /// 启用短路时，token 已低于上限则应跳过后续阶段。
    #[tokio::test]
    async fn test_hybrid_short_circuit_skips_second_stage() {
        /*
         * 两个滑动窗口阶段：第一个保留 10 条，第二个保留 2 条
         * 当 short_circuit=true 且 token_limit=100（足够高）时，
         * 第一阶段应当执行，第二阶段应当被跳过，
         * 因为第一阶段执行后的 token 数应已低于该较高的限制。
         */
        let compressor = HybridCompressor::builder()
            .stage(SlidingWindowCompressor::new(10))
            .stage(SlidingWindowCompressor::new(2))
            .short_circuit(true)
            .build();

        let mut messages = vec![make_msg(Role::System, "system")];
        for i in 0..15 {
            messages.push(make_msg(Role::User, &format!("msg {}", i)));
        }

        let input = CompressionInput {
            messages,
            token_limit: 1000, // 非常高——第一阶段的结果会低于此值
            current_query: None,
            focus_instructions: None,
            cancel_token: None,
            tokenizer: None,
        };

        let output = compressor.compress(input).await.unwrap();
        /*
         * 启用 short_circuit 后，第二阶段（保留 2 条）应被跳过。
         * 第一阶段保留 10 条消息，因此输出应为 10 + 1 条 system 消息 = 11。
         * 如果不启用 short_circuit，第二阶段会将结果缩减为 2 + 1 = 3。
         */
        assert!(
            output.messages.len() > 3,
            "Short-circuit should have prevented second stage from running. Got {} messages.",
            output.messages.len()
        );
    }

    /// 禁用短路时，所有阶段都应执行。
    #[tokio::test]
    async fn test_hybrid_no_short_circuit_runs_all_stages() {
        let compressor = HybridCompressor::builder()
            .stage(SlidingWindowCompressor::new(10))
            .stage(SlidingWindowCompressor::new(2))
            .short_circuit(false)
            .build();

        let mut messages = vec![make_msg(Role::System, "system")];
        for i in 0..15 {
            messages.push(make_msg(Role::User, &format!("msg {}", i)));
        }

        let input = CompressionInput {
            messages,
            token_limit: 1000,
            current_query: None,
            focus_instructions: None,
            cancel_token: None,
            tokenizer: None,
        };

        let output = compressor.compress(input).await.unwrap();
        /*
         * 两个阶段都应执行：第一个保留 10 条，第二个保留 2 条
         * 结果：system + 2 = 3 条消息
         */
        assert_eq!(output.messages.len(), 3, "All stages should have run");
    }
}
