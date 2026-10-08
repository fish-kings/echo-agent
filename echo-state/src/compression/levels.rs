/*!
 * 多级自适应压缩。
 *
 * 各级别按开销从低到高排列：
 * L1 Snip：移除过大的工具输出（超过 N 个 token）
 * L2 Micro：截断工具输出，只保留首尾 N 行
 * L3 Context Collapse：概括较早的消息组（不依赖 LLM，基于规则）
 * L4 Auto Compact：完整的 LLM 摘要（由外部处理）
 * L5 Reactive：紧急模式——只保留系统提示词 + 最后 3 条消息
 */

use crate::compression::CompressionCheckpoint;
use echo_core::llm::LlmClient;
use echo_core::llm::types::{Message, MessageContent, Role};
use echo_core::tokenizer::{HeuristicTokenizer, Tokenizer};
use futures::future::BoxFuture;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;

/// 按 Unicode 标量值截断；绝不对外部内容做字节切片。
fn safe_truncate(s: &str, max_chars: usize) -> String {
    s.chars().take(max_chars).collect()
}

/// 多级自适应压缩的配置（各级触发阈值与保留参数）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveCompressionConfig {
    /// 触发 L1（剪除大输出）的 token 阈值
    pub l1_snip_threshold_tokens: usize,
    /// 单个工具输出在被剪除前允许的最大 token 数
    pub l1_max_output_tokens: usize,
    /**
     * L1 是否同时把连续的工具结果折叠为摘要。
     * 为 true 时，在剪除之后运行，用于压缩连续的长串工具消息。
     */
    #[serde(default = "default_true")]
    pub l1_fold_consecutive_tools: bool,
    /// 折叠连续工具结果时，保留最新的 N 条，折叠更早的部分。
    #[serde(default = "default_l1_fold_keep_latest")]
    pub l1_fold_keep_latest: usize,
    /// 触发 L2（截断）的 token 阈值
    pub l2_micro_threshold_tokens: usize,
    /// 截断输出时首尾各保留的行数
    pub l2_keep_lines: usize,
    /// 触发 L3（上下文折叠）的 token 阈值
    pub l3_collapse_threshold_tokens: usize,
    /// 折叠时完整保留的最近消息数
    pub l3_keep_recent: usize,
    /// 触发 L4（LLM 摘要）的 token 阈值
    pub l4_compact_threshold_tokens: usize,
    /// LLM 压缩时保留的最近消息数
    pub l4_keep_recent: usize,
}

/// serde 默认值：返回 `true`。
fn default_true() -> bool {
    true
}
/// serde 默认值：L1 折叠时保留最新的 2 条工具结果。
fn default_l1_fold_keep_latest() -> usize {
    2
}

impl Default for AdaptiveCompressionConfig {
    fn default() -> Self {
        Self {
            l1_snip_threshold_tokens: 80_000,
            l1_max_output_tokens: 4_000,
            l1_fold_consecutive_tools: true,
            l1_fold_keep_latest: 2,
            l2_micro_threshold_tokens: 100_000,
            l2_keep_lines: 50,
            l3_collapse_threshold_tokens: 120_000,
            l3_keep_recent: 10,
            l4_compact_threshold_tokens: 150_000,
            l4_keep_recent: 6,
        }
    }
}

/**
 * 根据模型的上下文窗口大小自动调整压缩阈值。
 *
 * 各阈值按上下文窗口的百分比设置：
 * - L1（Snip）：60%
 * - L2（Micro）：75%
 * - L3（Collapse）：85%
 * - L4（Compact）：90%
 *
 * L5（Reactive）不参与调整——无论如何都在 L4 阈值的 2 倍处触发。
 *
 * # 示例
 *
 * ```rust
 * use echo_state::compression::levels::{AdaptiveCompressionConfig, tune_for_model};
 *
 * let mut config = AdaptiveCompressionConfig::default();
 * tune_for_model(&mut config, 200_000); // Claude's context window
 * // Now L1=120K, L2=150K, L3=170K, L4=180K
 * ```
 */
pub fn tune_for_model(config: &mut AdaptiveCompressionConfig, context_window: usize) {
    let w = context_window;
    config.l1_snip_threshold_tokens = w.saturating_mul(60) / 100;
    config.l2_micro_threshold_tokens = w.saturating_mul(75) / 100;
    config.l3_collapse_threshold_tokens = w.saturating_mul(85) / 100;
    config.l4_compact_threshold_tokens = w.saturating_mul(90) / 100;
    // L5 在 tokens > target * 2 时触发，因此会自动适配
}

/// 自适应压缩的结果，记录实际应用了哪些级别。
#[derive(Debug, Clone)]
pub struct AdaptiveCompressionResult {
    /// 已应用的级别标签（如 `L1:Snip`、`L3:Collapse`）
    pub levels_applied: Vec<String>,
    /// 压缩前的 token 数
    pub tokens_before: usize,
    /// 压缩后的 token 数（估算）
    pub tokens_after: usize,
}

/**
 * 自适应多级压缩器。
 *
 * 按开销从低到高依次应用各压缩级别，token 数降到目标阈值以下即停止。
 *
 * 只有通过 [`with_llm()`](Self::with_llm) 配置了 LLM 客户端时，L4（LLM 摘要）才可用。
 * 未配置时跳过 L4，必要时直接启用 L5（紧急模式）。
 */
pub struct AdaptiveCompressor {
    /// 压缩配置
    config: AdaptiveCompressionConfig,
    /// 用于计算 token 数的分词器
    tokenizer: Arc<dyn Tokenizer>,
    /// 可选的 LLM 客户端，用于 L4 自动压缩。
    llm: Option<Arc<dyn LlmClient>>,
}

impl AdaptiveCompressor {
    /// 使用给定配置创建压缩器，默认使用启发式分词器且不配置 LLM。
    pub fn new(config: AdaptiveCompressionConfig) -> Self {
        Self {
            config,
            tokenizer: Arc::new(HeuristicTokenizer),
            llm: None,
        }
    }

    /**
     * 设置 LLM 客户端以启用 L4 自动压缩（完整的 LLM 摘要）。
     *
     * 没有 LLM 客户端时，自适应流水线会跳过 L4；若 L3 之后 token 仍超过阈值，
     * 则直接进入 L5（紧急模式）。
     */
    pub fn with_llm(mut self, llm: Arc<dyn LlmClient>) -> Self {
        self.llm = Some(llm);
        self
    }

    /// 设置低层原地压缩 API 所用的分词器。
    pub fn with_tokenizer(mut self, tokenizer: Arc<dyn Tokenizer>) -> Self {
        self.tokenizer = tokenizer;
        self
    }

    /**
     * 原地对消息进行自适应压缩。
     *
     * 这是直接修改 `messages` 的低层 API。
     * 如需与 [`ContextManager`](super::ContextManager) 集成，
     * 请改用 [`ContextCompressor`](super::ContextCompressor) trait 实现。
     */
    pub fn compress_in_place(
        &self,
        messages: &mut Vec<Message>,
        current_tokens: usize,
        target_tokens: usize,
    ) -> AdaptiveCompressionResult {
        let (result, _evicted) =
            self.compress_with_evicted(messages, current_tokens, target_tokens);
        result
    }

    /// 内部方法：执行压缩并记录被逐出的消息。
    fn compress_with_evicted(
        &self,
        messages: &mut Vec<Message>,
        current_tokens: usize,
        target_tokens: usize,
    ) -> (AdaptiveCompressionResult, Vec<Message>) {
        let tokens_before = current_tokens;
        let mut levels_applied = Vec::new();
        let mut tokens = current_tokens;
        let mut all_evicted = Vec::new();

        // L1：剪除过大的工具输出
        if tokens > self.config.l1_snip_threshold_tokens && tokens > target_tokens {
            let snipped = self.apply_l1_snip(messages);
            tokens = tokens.saturating_sub(snipped);
            if snipped > 0 {
                levels_applied.push("L1:Snip".to_string());
            }

            // L1 同时折叠连续的工具结果（在剪除之后运行）
            if self.config.l1_fold_consecutive_tools {
                let (folded_tokens, folded_evicted) = self.apply_l1_fold_tools(messages);
                tokens = tokens.saturating_sub(folded_tokens);
                if !folded_evicted.is_empty() {
                    if !levels_applied.contains(&"L1:Fold".to_string()) {
                        levels_applied.push("L1:Fold".to_string());
                    }
                    all_evicted.extend(folded_evicted);
                }
            }
        }

        // L2：Micro——截断工具输出
        if tokens > self.config.l2_micro_threshold_tokens && tokens > target_tokens {
            let truncated = self.apply_l2_micro(messages);
            tokens = tokens.saturating_sub(truncated);
            if truncated > 0 {
                levels_applied.push("L2:Micro".to_string());
            }
        }

        // L3：上下文折叠——移除较早消息，保留最近消息
        if tokens > self.config.l3_collapse_threshold_tokens && tokens > target_tokens {
            let (collapsed, evicted) = self.apply_l3_collapse(messages);
            tokens = tokens.saturating_sub(collapsed);
            if collapsed > 0 {
                levels_applied.push("L3:Collapse".to_string());
                all_evicted.extend(evicted);
            }
        }

        // L5：Reactive——紧急模式（L4 需要 LLM，由外部处理）
        if tokens > target_tokens.saturating_mul(2)
            && tokens > self.config.l4_compact_threshold_tokens
        {
            let (saved, evicted) = self.apply_l5_reactive(messages);
            tokens = tokens.saturating_sub(saved);
            if saved > 0 {
                levels_applied.push("L5:Reactive".to_string());
                all_evicted.extend(evicted);
            }
        }

        (
            AdaptiveCompressionResult {
                levels_applied,
                tokens_before,
                tokens_after: tokens,
            },
            all_evicted,
        )
    }

    /**
     * 包含 L4（LLM 摘要）的异步压缩流水线。
     *
     * 依次运行 L1 → L2 → L3 → L4（如已配置 LLM）→ L5。
     * 这是 [`ContextCompressor`] trait 使用的完整流水线。
     */
    async fn compress_async(
        &self,
        messages: &mut Vec<Message>,
        current_tokens: usize,
        target_tokens: usize,
        focus: Option<&str>,
        cancel_token: Option<echo_core::compression::CancellationToken>,
    ) -> (AdaptiveCompressionResult, Vec<Message>, Option<String>) {
        let tokens_before = current_tokens;
        let mut levels_applied = Vec::new();
        let mut tokens = current_tokens;
        let mut all_evicted = Vec::new();
        let mut l4_summary: Option<String> = None;

        // L1：剪除过大的工具输出 + 折叠连续工具结果
        if tokens > self.config.l1_snip_threshold_tokens && tokens > target_tokens {
            let snipped = self.apply_l1_snip(messages);
            tokens = tokens.saturating_sub(snipped);
            if snipped > 0 {
                levels_applied.push("L1:Snip".to_string());
            }
            if self.config.l1_fold_consecutive_tools {
                let (folded_tokens, folded_evicted) = self.apply_l1_fold_tools(messages);
                tokens = tokens.saturating_sub(folded_tokens);
                if !folded_evicted.is_empty() {
                    if !levels_applied.contains(&"L1:Fold".to_string()) {
                        levels_applied.push("L1:Fold".to_string());
                    }
                    all_evicted.extend(folded_evicted);
                }
            }
        }

        // L2：Micro——截断工具输出
        if tokens > self.config.l2_micro_threshold_tokens && tokens > target_tokens {
            let truncated = self.apply_l2_micro(messages);
            tokens = tokens.saturating_sub(truncated);
            if truncated > 0 {
                levels_applied.push("L2:Micro".to_string());
            }
        }

        // L3：上下文折叠——移除较早消息，保留最近消息
        if tokens > self.config.l3_collapse_threshold_tokens && tokens > target_tokens {
            let (collapsed, evicted) = self.apply_l3_collapse(messages);
            tokens = tokens.saturating_sub(collapsed);
            if collapsed > 0 {
                levels_applied.push("L3:Collapse".to_string());
                all_evicted.extend(evicted);
            }
        }

        // L4：自动压缩——LLM 摘要（仅在配置了 LLM 客户端时）
        if tokens > self.config.l4_compact_threshold_tokens
            && tokens > target_tokens
            && let Some(llm) = &self.llm
        {
            let (saved, evicted, summary) = self
                .apply_l4_compact(messages, llm.as_ref(), focus, cancel_token.clone())
                .await;
            tokens = tokens.saturating_sub(saved);
            if saved > 0 {
                levels_applied.push("L4:Compact".to_string());
                all_evicted.extend(evicted);
                l4_summary = summary;
            }
        }

        // L5：Reactive——紧急模式
        if tokens > target_tokens.saturating_mul(2)
            && tokens > self.config.l4_compact_threshold_tokens
        {
            let (saved, evicted) = self.apply_l5_reactive(messages);
            tokens = tokens.saturating_sub(saved);
            if saved > 0 {
                levels_applied.push("L5:Reactive".to_string());
                all_evicted.extend(evicted);
            }
        }

        (
            AdaptiveCompressionResult {
                levels_applied,
                tokens_before,
                tokens_after: tokens,
            },
            all_evicted,
            l4_summary,
        )
    }

    /// L1：剪除超过 max_output_tokens 的工具输出。
    fn apply_l1_snip(&self, messages: &mut [Message]) -> usize {
        let mut saved: usize = 0;
        let max_tokens = self.config.l1_max_output_tokens;

        for msg in messages.iter_mut() {
            if msg.role != Role::Tool {
                continue;
            }
            let text = msg.content.as_text_ref().unwrap_or("");
            let tokens = self.tokenizer.count_tokens(text);
            if tokens > max_tokens {
                let char_limit = max_tokens.saturating_mul(4); // 约每 token 4 个字符
                let truncated = format!(
                    "{}\n...[output truncated: {} tokens, kept first {}]...",
                    safe_truncate(text, char_limit),
                    tokens,
                    max_tokens
                );
                saved += tokens - max_tokens;
                msg.content = MessageContent::Text(truncated);
            }
        }
        saved
    }

    /**
     * L1 折叠：压缩较早的工具结果内容，同时保留每一条结果消息本身。
     * 返回 (tokens_saved, evicted_messages)。
     */
    fn apply_l1_fold_tools(&self, messages: &mut [Message]) -> (usize, Vec<Message>) {
        let keep = self.config.l1_fold_keep_latest;
        let mut saved: usize = 0;
        let mut evicted = Vec::new();
        let mut i = 0;

        while i < messages.len() {
            if messages[i].role != Role::Tool {
                i += 1;
                continue;
            }
            // 找出连续的工具消息区间
            let start = i;
            while i < messages.len() && messages[i].role == Role::Tool {
                i += 1;
            }
            let count = i - start;
            if count > keep {
                let to_compact = count.saturating_sub(keep);
                for offset in 0..to_compact {
                    let Some(msg) = messages.get_mut(start.saturating_add(offset)) else {
                        break;
                    };
                    let tokens = self
                        .tokenizer
                        .count_tokens(msg.content.as_text_ref().unwrap_or(""));
                    saved = saved.saturating_add(tokens.saturating_sub(12));
                    evicted.push(msg.clone());
                    msg.content = MessageContent::Text(format!(
                        "[L1 fold: tool result payload compacted; original ~{tokens} tokens]"
                    ));
                }
            }
        }

        (saved, evicted)
    }

    /// L2：截断工具输出，只保留首尾各 N 行。
    fn apply_l2_micro(&self, messages: &mut [Message]) -> usize {
        let mut saved = 0;
        let keep = self.config.l2_keep_lines;

        for msg in messages.iter_mut() {
            if msg.role != Role::Tool {
                continue;
            }
            let text = msg.content.as_text_ref().unwrap_or("");
            let lines: Vec<&str> = text.lines().collect();
            let retained_lines = keep.saturating_mul(2);
            if lines.len() > retained_lines {
                let head: String = lines
                    .iter()
                    .take(keep)
                    .copied()
                    .collect::<Vec<_>>()
                    .join("\n");
                let tail: String = lines
                    .iter()
                    .rev()
                    .take(keep)
                    .copied()
                    .collect::<Vec<_>>()
                    .into_iter()
                    .rev()
                    .collect::<Vec<_>>()
                    .join("\n");
                let new_text = format!(
                    "{}\n...[{} lines truncated]...\n{}",
                    head,
                    lines.len().saturating_sub(retained_lines),
                    tail
                );
                let old_tokens = self.tokenizer.count_tokens(text);
                let new_tokens = self.tokenizer.count_tokens(&new_text);
                saved += old_tokens.saturating_sub(new_tokens);
                msg.content = MessageContent::Text(new_text);
            }
        }
        saved
    }

    /**
     * L3：移除较早的消息，只保留最近 N 条。
     * 返回 (tokens_saved, evicted_messages)。
     */
    fn apply_l3_collapse(&self, messages: &mut Vec<Message>) -> (usize, Vec<Message>) {
        let keep = self.config.l3_keep_recent;
        if messages.len() <= keep + 1 {
            return (0, vec![]);
        }

        // 保留系统消息 + 最后 N 条消息
        let system_msgs: Vec<Message> = messages
            .iter()
            .filter(|m| m.role == Role::System)
            .cloned()
            .collect();
        let recent: Vec<Message> = messages.iter().rev().take(keep).rev().cloned().collect();

        let removed: Vec<Message> = messages
            .iter()
            .filter(|m| m.role != Role::System)
            .take(messages.len().saturating_sub(keep + system_msgs.len()))
            .cloned()
            .collect();
        let saved: usize = removed
            .iter()
            .map(|m| {
                self.tokenizer
                    .count_tokens(m.content.as_text_ref().unwrap_or(""))
            })
            .sum();

        // 添加一条说明已移除内容的摘要消息
        let summary = format!(
            "[Context compressed: {} older messages removed to save space]",
            removed.len()
        );
        let mut new_messages = system_msgs;
        new_messages.push(Message::system(summary));
        new_messages.extend(recent);
        *messages = new_messages;

        (saved, removed)
    }

    /**
     * L4：自动压缩——用 LLM 概括较早的消息。
     * 返回 (tokens_saved, evicted_messages, summary_option)。
     *
     * 优先尝试结构化 JSON 输出，失败时回退为自然语言摘要。
     * LLM 调用失败时返回 (0, vec![], None)，使流水线继续进入 L5。
     */
    async fn apply_l4_compact(
        &self,
        messages: &mut Vec<Message>,
        llm: &dyn LlmClient,
        focus: Option<&str>,
        cancel_token: Option<echo_core::compression::CancellationToken>,
    ) -> (usize, Vec<Message>, Option<String>) {
        let keep = self.config.l4_keep_recent;

        // 拆分为系统消息 / 较早消息 / 最近消息
        let system_msgs: Vec<Message> = messages
            .iter()
            .filter(|m| m.role == Role::System)
            .cloned()
            .collect();
        let non_system: Vec<Message> = messages
            .iter()
            .filter(|m| m.role != Role::System)
            .cloned()
            .collect();

        if non_system.len() <= keep {
            return (0, vec![], None);
        }

        let split_at = non_system.len() - keep;
        let to_summarize = &non_system[..split_at];
        let to_keep = &non_system[split_at..];

        // 若提供方支持，优先尝试结构化输出
        let summary = if llm.capabilities().structured_output {
            let prompt =
                crate::compression::compressor::structured_summary_prompt(to_summarize, focus);
            match llm
                .chat(echo_core::llm::ChatRequest {
                    messages: vec![Message::user(prompt)],
                    temperature: Some(0.3),
                    max_tokens: Some(2048),
                    tools: None,
                    tool_choice: None,
                    response_format: Some(echo_core::llm::types::ResponseFormat::JsonObject),
                    thinking: None,
                    cancel_token: cancel_token.clone(),
                    timeouts: None,
                    user_id: None,
                    cache_hints: None,
                })
                .await
            {
                Ok(resp) => {
                    let text = resp.content().unwrap_or_default().to_string();
                    if let Some(parsed) = super::StructuredSummary::from_llm_response(&text) {
                        parsed.to_json()
                    } else {
                        // JSON 解析失败——直接使用原始文本
                        text
                    }
                }
                Err(e) => {
                    tracing::warn!(error = %e, "L4 structured summary failed, falling back to natural language");
                    // 继续走后续回退逻辑
                    String::new()
                }
            }
        } else {
            String::new()
        };

        // 自然语言回退
        let summary = if summary.is_empty() {
            let prompt = crate::compression::compressor::default_summary_prompt_with_focus(
                to_summarize,
                focus,
            );
            match llm
                .chat(echo_core::llm::ChatRequest {
                    messages: vec![Message::user(prompt)],
                    temperature: None,
                    max_tokens: None,
                    tools: None,
                    tool_choice: None,
                    response_format: None,
                    thinking: None,
                    cancel_token,
                    timeouts: None,
                    user_id: None,
                    cache_hints: None,
                })
                .await
            {
                Ok(response) => response.content().unwrap_or_default().to_string(),
                Err(e) => {
                    tracing::warn!(error = %e, "L4 Auto-Compact: LLM summarization failed, falling through to L5");
                    return (0, vec![], None);
                }
            }
        } else {
            summary
        };

        // 计算节省的 token 数
        let evicted_tokens: usize = to_summarize
            .iter()
            .map(|m| {
                self.tokenizer
                    .count_tokens(m.content.as_text_ref().unwrap_or(""))
            })
            .sum();
        let summary_tokens = self.tokenizer.count_tokens(&summary);
        let saved = evicted_tokens.saturating_sub(summary_tokens);

        let evicted = to_summarize.to_vec();

        // 重建消息：系统消息 + 摘要 + 最近消息
        let summary_clone = summary.clone();
        let mut new_messages = system_msgs;
        new_messages.push(Message::system(format!("[对话历史摘要]\n{}", summary)));
        new_messages.extend(to_keep.iter().cloned());
        *messages = new_messages;

        (saved, evicted, Some(summary_clone))
    }

    /**
     * L5：紧急模式——只保留系统提示词 + 最后 3 条消息。
     * 返回 (tokens_saved, evicted_messages)。
     */
    fn apply_l5_reactive(&self, messages: &mut Vec<Message>) -> (usize, Vec<Message>) {
        let system_msgs: Vec<Message> = messages
            .iter()
            .filter(|m| m.role == Role::System)
            .cloned()
            .collect();
        let recent: Vec<Message> = messages.iter().rev().take(3).rev().cloned().collect();

        let old_tokens: usize = messages
            .iter()
            .map(|m| {
                self.tokenizer
                    .count_tokens(m.content.as_text_ref().unwrap_or(""))
            })
            .sum();

        // 收集被逐出的消息：所有非系统且不在最近区间（最后 3 条非系统消息）内的消息
        let non_system: Vec<Message> = messages
            .iter()
            .filter(|m| m.role != Role::System)
            .cloned()
            .collect();
        let evicted = if non_system.len() > 3 {
            non_system[..non_system.len() - 3].to_vec()
        } else {
            vec![]
        };

        let mut new_messages = system_msgs;
        new_messages.push(Message::system(
            "[Emergency compression: context was critically large. Only recent messages retained.]"
                .to_string(),
        ));
        new_messages.extend(recent);

        let new_tokens: usize = new_messages
            .iter()
            .map(|m| {
                self.tokenizer
                    .count_tokens(m.content.as_text_ref().unwrap_or(""))
            })
            .sum();

        *messages = new_messages;
        (old_tokens.saturating_sub(new_tokens), evicted)
    }
}

// ── ContextCompressor trait 实现 ──────────────────────────────────────

impl super::ContextCompressor for AdaptiveCompressor {
    /// 压缩器名称。
    fn name(&self) -> &str {
        "Adaptive"
    }

    /// 通过完整异步流水线（含 L4）压缩输入消息，并生成压缩检查点。
    fn compress(
        &self,
        input: super::CompressionInput,
    ) -> BoxFuture<'_, echo_core::error::Result<super::CompressionOutput>> {
        Box::pin(async move {
            let start = Instant::now();
            let compressor = AdaptiveCompressor {
                config: self.config.clone(),
                tokenizer: input.tokenizer(),
                llm: self.llm.clone(),
            };
            let mut messages = input.messages;
            let target_tokens = input.token_limit;
            let focus = input
                .focus_instructions
                .clone()
                .or(input.current_query.clone());

            // 根据输入消息估算当前 token 数
            let current_tokens: usize = messages
                .iter()
                .filter_map(|m| m.content.as_text())
                .map(|c| compressor.tokenizer.count_tokens(&c))
                .sum();

            let (result, evicted, l4_summary) = compressor
                .compress_async(
                    &mut messages,
                    current_tokens,
                    target_tokens,
                    focus.as_deref(),
                    input.cancel_token.clone(),
                )
                .await;

            let tokens_after: usize = messages
                .iter()
                .filter_map(|m| m.content.as_text())
                .map(|c| compressor.tokenizer.count_tokens(&c))
                .sum();

            let mut checkpoint = CompressionCheckpoint::new(self.name())
                .with_counts(messages.len(), evicted.len())
                .with_tokens(result.tokens_before, tokens_after)
                .with_levels(result.levels_applied)
                .with_duration_ms(start.elapsed().as_millis() as u64)
                .with_focus(focus);
            if let Some(summary) = l4_summary {
                checkpoint = checkpoint.with_summary(summary);
            }

            Ok(super::CompressionOutput {
                messages,
                evicted,
                checkpoint: Some(checkpoint),
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 构造指定角色与文本的测试消息。
    fn make_msg(role: Role, text: &str) -> Message {
        Message {
            role,
            content: MessageContent::Text(text.to_string()),
            ..Default::default()
        }
    }

    #[test]
    fn test_l1_snip() {
        let compressor = AdaptiveCompressor::new(AdaptiveCompressionConfig {
            l1_max_output_tokens: 10,
            l1_snip_threshold_tokens: 0,
            ..Default::default()
        });
        let mut messages = vec![
            make_msg(Role::User, "hello"),
            make_msg(Role::Tool, &"x".repeat(1000)),
        ];
        let result = compressor.compress_in_place(&mut messages, 500, 100);
        assert!(result.levels_applied.contains(&"L1:Snip".to_string()));
    }

    #[test]
    fn test_l3_collapse() {
        let compressor = AdaptiveCompressor::new(AdaptiveCompressionConfig {
            l3_collapse_threshold_tokens: 0,
            l3_keep_recent: 2,
            ..Default::default()
        });
        let mut messages = vec![
            make_msg(Role::System, "system prompt"),
            make_msg(Role::User, "msg1"),
            make_msg(Role::Assistant, "msg2"),
            make_msg(Role::User, "msg3"),
            make_msg(Role::Assistant, "msg4"),
            make_msg(Role::User, "msg5"),
        ];
        let result = compressor.compress_in_place(&mut messages, 1000, 100);
        assert!(result.levels_applied.contains(&"L3:Collapse".to_string()));
        // 系统消息应被保留
        assert!(messages.iter().any(|m| m.role == Role::System));
    }

    #[test]
    fn test_no_compression_needed() {
        let compressor = AdaptiveCompressor::new(AdaptiveCompressionConfig::default());
        let mut messages = vec![
            make_msg(Role::User, "hello"),
            make_msg(Role::Assistant, "hi"),
        ];
        let result = compressor.compress_in_place(&mut messages, 100, 200);
        assert!(result.levels_applied.is_empty());
    }

    #[test]
    fn test_safe_truncate_ascii() {
        assert_eq!(safe_truncate("hello world", 5), "hello");
        assert_eq!(safe_truncate("hello", 10), "hello");
        assert_eq!(safe_truncate("hello", 0), "");
    }

    #[test]
    fn test_safe_truncate_cjk_no_panic() {
        let s = "你好世界";
        let result = safe_truncate(s, 4);
        assert_eq!(result, "你好世界");

        let result = safe_truncate(s, 1);
        assert_eq!(result, "你");

        let result = safe_truncate(s, 2);
        assert_eq!(result, "你好");
    }

    #[test]
    fn test_l1_snip_cjk_no_panic() {
        let compressor = AdaptiveCompressor::new(AdaptiveCompressionConfig {
            l1_max_output_tokens: 5,
            l1_snip_threshold_tokens: 0,
            ..Default::default()
        });
        /* 构造一条含 CJK 内容的工具消息，使 char_limit 落在多字节字符中间 */
        let cjk_text = "你".repeat(100); // 100 个 CJK 字符 = 300 字节
        let mut messages = vec![make_msg(Role::Tool, &cjk_text)];
        // 不应 panic
        let result = compressor.compress_in_place(&mut messages, 500, 50);
        assert!(result.levels_applied.contains(&"L1:Snip".to_string()));
        // 验证截断后的内容是合法 UTF-8
        let text = messages[0].content.as_text_ref().unwrap();
        assert!(text.starts_with("你"));
    }

    #[test]
    fn test_l1_fold_consecutive_tools() {
        let compressor = AdaptiveCompressor::new(AdaptiveCompressionConfig {
            l1_snip_threshold_tokens: 0, // 总是触发 L1
            l1_max_output_tokens: 99999, // 不剪除，只折叠
            l1_fold_consecutive_tools: true,
            l1_fold_keep_latest: 2,
            ..Default::default()
        });

        let mut messages = vec![
            make_msg(Role::User, "do stuff"),
            make_msg(Role::Tool, "result1"),
            make_msg(Role::Tool, "result2"),
            make_msg(Role::Tool, "result3"),
            make_msg(Role::Tool, "result4"),
            make_msg(Role::Tool, "result5"),
            make_msg(Role::Assistant, "done"),
        ];
        // 5 条工具消息且 keep=2 时应折叠 3 条
        let result = compressor.compress_in_place(&mut messages, 500, 50);
        assert!(result.levels_applied.contains(&"L1:Fold".to_string()));
        // 每条结果消息仍保持连续；只有较早的内容被压缩。
        assert_eq!(messages.len(), 7);
        let folded = messages
            .iter()
            .filter(|message| {
                message
                    .content
                    .as_text_ref()
                    .unwrap_or("")
                    .contains("L1 fold")
            })
            .count();
        assert_eq!(folded, 3);
    }

    #[test]
    fn test_l1_fold_disabled() {
        let compressor = AdaptiveCompressor::new(AdaptiveCompressionConfig {
            l1_snip_threshold_tokens: 0,
            l1_max_output_tokens: 99999,
            l1_fold_consecutive_tools: false, // 已禁用
            ..Default::default()
        });
        let mut messages = vec![
            make_msg(Role::Tool, "result1"),
            make_msg(Role::Tool, "result2"),
            make_msg(Role::Tool, "result3"),
        ];
        let result = compressor.compress_in_place(&mut messages, 500, 50);
        assert!(!result.levels_applied.contains(&"L1:Fold".to_string()));
        assert_eq!(messages.len(), 3); // 未折叠
    }

    #[tokio::test]
    async fn test_adaptive_as_context_compressor() {
        use crate::compression::ContextManager;

        let config = AdaptiveCompressionConfig {
            l1_snip_threshold_tokens: 0, // 总是触发 L1
            l1_max_output_tokens: 5,
            l3_collapse_threshold_tokens: 0, // 超过目标时总是触发 L3
            l3_keep_recent: 2,
            ..Default::default()
        };
        let compressor = AdaptiveCompressor::new(config);

        // 通过 ContextCompressor trait 在 ContextManager 中使用 AdaptiveCompressor
        let mut ctx = ContextManager::builder(128).compressor(compressor).build();

        ctx.push(Message::system("system prompt".to_string()));
        for i in 0..10 {
            ctx.push(Message::user(format!("question {i} ").repeat(12)));
            ctx.push(Message::assistant(format!("answer {i} ").repeat(12)));
        }

        // prepare() 应通过 ContextCompressor trait 触发自动压缩
        let result = ctx.prepare(None).await.unwrap();
        /* 消息很多且 token 上限很低（50），因此应当触发压缩 */
        assert!(
            result.compressed.is_some(),
            "compression should have been triggered"
        );
        let stats = result.compressed.unwrap();
        assert!(
            stats.before_count > stats.after_count,
            "messages should have been reduced"
        );
    }

    #[tokio::test]
    async fn test_adaptive_trait_evicted_messages() {
        use super::super::{CompressionInput, CompressionOutput, ContextCompressor};

        let config = AdaptiveCompressionConfig {
            l3_collapse_threshold_tokens: 0, // 总是触发 L3
            l3_keep_recent: 2,
            ..Default::default()
        };
        let compressor = AdaptiveCompressor::new(config);

        let messages = vec![
            make_msg(Role::System, "system"),
            make_msg(Role::User, "msg1"),
            make_msg(Role::Assistant, "msg2"),
            make_msg(Role::User, "msg3"),
            make_msg(Role::Assistant, "msg4"),
            make_msg(Role::User, "msg5"),
        ];

        let input = CompressionInput {
            messages,
            token_limit: 0, // 强制压缩
            current_query: None,
            focus_instructions: None,
            cancel_token: None,
            tokenizer: None,
        };

        let output: CompressionOutput = compressor.compress(input).await.unwrap();
        // L3 应已逐出较早的消息
        assert!(
            !output.evicted.is_empty(),
            "L3 should produce evicted messages"
        );
        assert!(
            output.messages.len() < 6,
            "output should have fewer messages"
        );
        // 系统消息应被保留
        assert!(output.messages.iter().any(|m| m.role == Role::System));
    }
}
