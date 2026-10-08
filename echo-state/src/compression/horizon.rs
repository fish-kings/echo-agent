/*!
可见性视界（Visibility Horizon）—— 基于滑动窗口的工具轨迹压缩。

# 三层可见性模型

| 层级 | 生命周期 | 内容 | 处理方式 |
|-------|----------|---------|-----------|
| **全局目标（Global Objective）** | 永久 | 用户最初的任务目标 | 以系统消息注入，永不压缩 |
| **活跃计划窗口（Active Plan Window）** | 最近 N 轮 | 活跃的对话上下文 | 完整保留 |
| **瞬态工具轨迹（Transient Tool Trace）** | 窗口之外 | 已完成的工具调用与结果对 | 压缩为符号化摘要 |

# 工作原理

```text
Messages: [sys] [u₁] [asst+tc] [tool] [tool] [u₂] [asst+tc] [tool] [u₃] ...
                  ↑ Turn 1  ← ToolGroup 1 →        ← ToolGroup 2 →  ↑ Turn 3

With active_window_turns = 2:
  ToolGroup 1 has 2 user turns after it (u₂, u₃) → ≥ threshold → COMPACT
  ToolGroup 2 has 1 user turn after it (u₃)      → < threshold → KEEP
```

# 集成方式

可作为独立的 [`ContextCompressor`] 使用，也可集成到
[`ContextManager`](crate::compression::ContextManager) 中作为预压缩阶段。
*/

use crate::compression::CompressionCheckpoint;
use echo_core::compression::{CompressionInput, CompressionOutput, ContextCompressor};
use echo_core::error::Result;
use echo_core::llm::types::{Message, MessageContent, Role};
use echo_core::tokenizer::Tokenizer;
use futures::future::BoxFuture;
use serde::{Deserialize, Serialize};
use std::time::Instant;

// ── 配置 ────────────────────────────────────────────────────

/** 可见性视界压缩器的配置。

控制工具轨迹何时以及如何被压缩。 */
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisibilityHorizonConfig {
    /** 需要完整保留的最近用户轮次数。

    位于最新这么多个用户轮次内的工具组始终保留。
    更早的工具组会被立即压缩，除非 `retained_tool_tokens`
    为它们提供了额外的 token 预算空间。 */
    #[serde(default = "default_active_window")]
    pub active_window_turns: usize,

    /** 早于受保护最近轮次的工具组的原始结果总预算。
    `None` 表示保持原有的仅按轮次判断的行为。 */
    #[serde(default)]
    pub retained_tool_tokens: Option<usize>,

    /** 除非选中的原始结果至少能释放这么多估算 token，否则不重写历史。
    用于避免对很小的轨迹做无意义的反复改写。 */
    #[serde(default)]
    pub compact_minimum_tokens: usize,

    /// 每条压缩摘要的最大 token 预算。
    #[serde(default = "default_compact_max_tokens")]
    pub compact_max_tokens: usize,

    /// 是否注入全局目标（Global Objective）系统消息。
    #[serde(default)]
    pub enable_global_objective: bool,

    /** 全局目标文本（用户最初的任务目标）。

    当 `enable_global_objective` 为 true 时，该文本会作为第一条
    系统消息注入，并在所有压缩中保留。 */
    #[serde(default)]
    pub global_objective: Option<String>,
}

/// `active_window_turns` 的默认值。
fn default_active_window() -> usize {
    5
}
/// `compact_max_tokens` 的默认值。
fn default_compact_max_tokens() -> usize {
    50
}

impl Default for VisibilityHorizonConfig {
    fn default() -> Self {
        Self {
            active_window_turns: 5,
            retained_tool_tokens: None,
            compact_minimum_tokens: 0,
            compact_max_tokens: 50,
            enable_global_objective: false,
            global_objective: None,
        }
    }
}

// ── ToolGroup（内部） ─────────────────────────────────────────────

/// 一组助手 tool_calls 及其对应的工具结果。
struct ToolGroup {
    /// 带有 `tool_calls` 的助手消息的索引。
    assistant_idx: usize,
    /// 工具结果消息索引区间 `[start, end)` 的起点。
    result_start: usize,
    /// 工具结果消息索引区间 `[start, end)` 的终点（不含）。
    result_end: usize,
    /// 从助手 `tool_calls` 中提取的工具名称。
    tool_names: Vec<String>,
    /// 本组中成功的工具结果数量。
    success_count: usize,
    /// 本组中工具结果的总数。
    total_count: usize,
    /// 工具结果消息的估算 token 数。
    result_tokens: usize,
    /// 错误信息（如有）。
    errors: Vec<String>,
    /// 本组之后出现的 `Role::User` 消息数量。
    user_turns_after: usize,
}

// ── VisibilityHorizonCompressor ──────────────────────────────────────

/** 将活跃计划窗口之外的工具轨迹压缩为符号化摘要。

实现了 [`ContextCompressor`]，便于与其他策略组合
（例如在流水线中与其他压缩器串联）。 */
pub struct VisibilityHorizonCompressor {
    /// 压缩器配置。
    config: VisibilityHorizonConfig,
}

impl VisibilityHorizonCompressor {
    /// 使用给定配置创建可见性视界压缩器。
    pub fn new(config: VisibilityHorizonConfig) -> Self {
        Self { config }
    }

    /** 核心压缩逻辑：识别窗口之外的工具组，并以摘要替换。

    返回被驱逐的（原始）消息，供下游进行记忆提升。 */
    #[cfg(test)]
    fn compact_horizon(&self, messages: &mut Vec<Message>) -> Vec<Message> {
        self.compact_horizon_with(messages, &echo_core::tokenizer::HeuristicTokenizer)
    }

    /** 使用指定分词器执行核心压缩。

    返回被驱逐的原始工具结果消息。 */
    fn compact_horizon_with(
        &self,
        messages: &mut Vec<Message>,
        tokenizer: &dyn Tokenizer,
    ) -> Vec<Message> {
        let groups = self.identify_tool_groups(messages, tokenizer);

        /* 从最新到最旧遍历。最近的轮次始终受保护；更早的原始结果共享一个
        可选的总 token 预算。选中列表本身已按索引倒序排列，保证原地修改安全。 */
        let mut retained_tokens = 0usize;
        let mut budget_exhausted = false;
        let mut to_compact: Vec<&ToolGroup> = Vec::new();
        for group in groups.iter().rev() {
            if group.user_turns_after < self.config.active_window_turns {
                continue;
            }
            if let Some(budget) = self.config.retained_tool_tokens {
                if !budget_exhausted
                    && retained_tokens.saturating_add(group.result_tokens) <= budget
                {
                    retained_tokens = retained_tokens.saturating_add(group.result_tokens);
                    continue;
                }
                budget_exhausted = true;
            }
            to_compact.push(group);
        }

        if to_compact.is_empty() {
            return vec![];
        }

        let compacted_tokens = to_compact.iter().fold(0usize, |total, group| {
            total.saturating_add(group.result_tokens)
        });
        if compacted_tokens < self.config.compact_minimum_tokens {
            return vec![];
        }

        let mut evicted = Vec::new();

        // 从后往前处理，以保持索引有效
        for group in to_compact {
            // 1. 构建摘要消息
            let summary_text = self.build_compact_summary(group);
            let summary_msg = Message::user(summary_text);

            // 2. 移除助手消息中的 tool_calls（保留推理文本）
            if let Some(asst) = messages.get_mut(group.assistant_idx) {
                // 将助手的文本内容保留为用户可见的说明
                let original_text = asst.content.as_text().unwrap_or_default().to_string();
                asst.tool_calls = None;
                if original_text.trim().is_empty() {
                    // 无文本——转换为简短说明
                    asst.content = MessageContent::Text(format!(
                        "[Used tools: {}]",
                        group.tool_names.join(", ")
                    ));
                }
            }

            // 3. 收集被驱逐的工具结果消息
            for i in group.result_start..group.result_end {
                if i < messages.len() {
                    evicted.push(messages[i].clone());
                }
            }

            // 4. 用单条摘要消息替换工具结果区间
            if group.result_start < messages.len() && group.result_end <= messages.len() {
                let drain_range = group.result_start..group.result_end;
                let drain_len = drain_range.end - drain_range.start;
                // 移除旧的工具结果并插入摘要
                let _: Vec<_> = messages.drain(drain_range).collect();
                messages.insert(group.result_start, summary_msg);

                // 注意：由于从后往前处理，无需调整更早工具组的索引。
                let _ = drain_len; // 抑制未使用变量警告
            }
        }

        // 5. 若已配置，则注入全局目标
        if self.config.enable_global_objective
            && let Some(ref objective) = self.config.global_objective
        {
            self.inject_global_objective(messages, objective);
        }

        evicted
    }

    /// 识别消息列表中的所有工具调用组。
    fn identify_tool_groups(
        &self,
        messages: &[Message],
        tokenizer: &dyn Tokenizer,
    ) -> Vec<ToolGroup> {
        let mut groups = Vec::new();
        let mut i = 0;

        while i < messages.len() {
            // 查找带有 tool_calls 的助手消息
            if messages[i].role == Role::Assistant
                && let Some(ref tool_calls) = messages[i].tool_calls
            {
                if tool_calls.is_empty() {
                    i += 1;
                    continue;
                }

                let assistant_idx = i;
                let tool_names: Vec<String> = tool_calls
                    .iter()
                    .map(|tc| tc.function.name.clone())
                    .collect();

                // 向后扫描工具结果消息
                let result_start = i + 1;
                let mut result_end = result_start;
                let mut success_count = 0;
                let mut total_count = 0;
                let mut result_tokens = 0;
                let mut errors = Vec::new();

                while result_end < messages.len() && messages[result_end].role == Role::Tool {
                    let content = messages[result_end].content.as_text().unwrap_or_default();
                    result_tokens += tokenizer.count_tokens(&content);

                    // 启发式：内容以 "[Error" 或 "[error" 开头则计为失败
                    if content.trim_start().starts_with("[Error")
                        || content.trim_start().starts_with("[error")
                    {
                        errors.push(content.chars().take(100).collect());
                    } else {
                        success_count += 1;
                    }
                    total_count += 1;
                    result_end += 1;
                }

                // 统计本组之后的用户轮次数
                let user_turns_after = messages[result_end..]
                    .iter()
                    .filter(|m| m.role == Role::User)
                    .count();

                groups.push(ToolGroup {
                    assistant_idx,
                    result_start,
                    result_end,
                    tool_names,
                    success_count,
                    total_count,
                    result_tokens,
                    errors,
                    user_turns_after,
                });

                i = result_end;
                continue;
            }
            i += 1;
        }

        groups
    }

    /// 为工具组构建紧凑的摘要字符串。
    fn build_compact_summary(&self, group: &ToolGroup) -> String {
        let tool_names = group.tool_names.join(", ");
        let status = if group.total_count == 0 {
            "no results".to_string()
        } else if group.success_count == group.total_count {
            format!("{}/{} success", group.success_count, group.total_count)
        } else {
            let err_preview = if group.errors.is_empty() {
                String::new()
            } else {
                format!(
                    ", error: \"{}\"",
                    group.errors[0].chars().take(60).collect::<String>()
                )
            };
            format!(
                "{}/{} success{}",
                group.success_count, group.total_count, err_preview
            )
        };

        let summary = format!(
            "[Horizon compact: {} | {} | {}→~{} tokens]",
            tool_names, status, group.result_tokens, self.config.compact_max_tokens
        );

        // 确保摘要不超出 token 预算
        let max_chars = self.config.compact_max_tokens.saturating_mul(4);
        if summary.chars().count() > max_chars {
            format!(
                "[Horizon compact: {} | {} | {} tokens compacted]",
                if group.tool_names.len() > 3 {
                    format!("{} tools", group.tool_names.len())
                } else {
                    tool_names
                },
                status,
                group.result_tokens
            )
        } else {
            summary
        }
    }

    /// 将全局目标作为第一条系统消息注入（若已存在则跳过）。
    fn inject_global_objective(&self, messages: &mut Vec<Message>, objective: &str) {
        let objective_marker = "[Global Objective]";
        let objective_text = format!("{} {}", objective_marker, objective);

        // 检查是否已注入
        if let Some(first) = messages.first()
            && first.role == Role::System
            && first
                .content
                .as_text()
                .is_some_and(|t| t.contains(objective_marker))
        {
            return; // 已存在
        }

        // 插入到开头
        messages.insert(0, Message::system(objective_text));
    }
}

impl ContextCompressor for VisibilityHorizonCompressor {
    /// 执行可见性视界压缩，并返回压缩结果与检查点。
    fn compress(&self, input: CompressionInput) -> BoxFuture<'_, Result<CompressionOutput>> {
        Box::pin(async move {
            let start = Instant::now();
            let tokenizer = input.tokenizer();
            let tokens_before: usize = input
                .messages
                .iter()
                .filter_map(|m| m.content.as_text())
                .map(|c| tokenizer.count_tokens(&c))
                .sum();
            let original_count = input.messages.len();

            let mut messages = input.messages;
            let evicted = self.compact_horizon_with(&mut messages, tokenizer.as_ref());

            let tokens_after: usize = messages
                .iter()
                .filter_map(|m| m.content.as_text())
                .map(|c| tokenizer.count_tokens(&c))
                .sum();

            let checkpoint = CompressionCheckpoint::new(self.name())
                .with_counts(messages.len(), evicted.len())
                .with_tokens(tokens_before, tokens_after)
                .with_covered_range(0, original_count.saturating_sub(1))
                .with_duration_ms(start.elapsed().as_millis() as u64)
                .with_focus(
                    input
                        .focus_instructions
                        .clone()
                        .or(input.current_query.clone()),
                );

            Ok(CompressionOutput {
                messages,
                evicted,
                checkpoint: Some(checkpoint),
            })
        })
    }

    /// 返回压缩器名称。
    fn name(&self) -> &str {
        "VisibilityHorizon"
    }
}

// ── 测试 ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use echo_core::llm::types::ToolCall;
    use echo_core::tokenizer::HeuristicTokenizer;

    /// 辅助函数：创建带有 tool_calls 的助手消息。
    fn assistant_with_tools(calls: &[(&str, &str)]) -> Message {
        let tool_calls: Vec<ToolCall> = calls
            .iter()
            .map(|(id, name)| ToolCall {
                id: id.to_string(),
                call_type: "function".to_string(),
                function: echo_core::llm::types::FunctionCall {
                    name: name.to_string(),
                    arguments: "{}".to_string(),
                },
            })
            .collect();
        Message::assistant_with_tools(tool_calls)
    }

    /// 辅助函数：创建工具结果消息。
    fn tool_result(id: &str, name: &str, content: &str) -> Message {
        Message::tool_result(id.to_string(), name.to_string(), content.to_string())
    }

    /** 构建包含工具调用的多轮对话。
    每轮生成：user → assistant(2 个 tool_calls) → 2 个 tool_results → assistant_answer */
    fn build_conversation(turns: usize) -> Vec<Message> {
        let mut messages = vec![Message::system("You are a helpful assistant.".to_string())];

        for i in 1..=turns {
            // 用户消息
            messages.push(Message::user(format!("Question {}", i)));

            // 带有 2 个并行工具调用的助手消息
            let call_id_a = format!("call_{}a", i);
            let call_id_b = format!("call_{}b", i);
            let tool_a = format!("read_file_{}", i);
            let tool_b = format!("grep_{}", i);
            messages.push(assistant_with_tools(&[
                (&call_id_a, &tool_a),
                (&call_id_b, &tool_b),
            ]));

            // 两条工具结果（模拟并行执行）
            messages.push(tool_result(
                &call_id_a,
                &tool_a,
                &"file content ".repeat(100 * i),
            ));
            messages.push(tool_result(
                &call_id_b,
                &tool_b,
                &"grep output ".repeat(80 * i),
            ));

            // 助手最终回答
            messages.push(Message::assistant(format!("Answer to question {}", i)));
        }

        messages
    }

    /// 窗口内的工具组不应被压缩。
    #[test]
    fn test_no_compaction_within_window() {
        let config = VisibilityHorizonConfig {
            active_window_turns: 5,
            ..Default::default()
        };
        let compressor = VisibilityHorizonCompressor::new(config);

        // 3 轮对话——全部位于大小为 5 的窗口内
        let mut messages = build_conversation(3);
        let original_len = messages.len();

        let evicted = compressor.compact_horizon(&mut messages);

        assert!(evicted.is_empty(), "No messages should be evicted");
        assert_eq!(
            messages.len(),
            original_len,
            "Message count should be unchanged"
        );
    }

    /// 窗口外的工具组应被压缩为摘要。
    #[test]
    fn test_compaction_beyond_window() {
        let config = VisibilityHorizonConfig {
            active_window_turns: 2,
            compact_max_tokens: 50,
            ..Default::default()
        };
        let compressor = VisibilityHorizonCompressor::new(config);

        // 6 轮对话且 window=2 → 第 1-4 轮应被压缩
        let mut messages = build_conversation(6);
        let original_len = messages.len();

        let evicted = compressor.compact_horizon(&mut messages);

        assert!(!evicted.is_empty(), "Some tool results should be evicted");
        assert!(
            messages.len() < original_len,
            "Message count should decrease: was {}, now {}",
            original_len,
            messages.len()
        );

        // 验证压缩后的消息包含视界摘要
        let has_summary = messages.iter().any(|m| {
            m.content
                .as_text()
                .is_some_and(|t| t.contains("[Horizon compact"))
        });
        assert!(has_summary, "Should contain a horizon compact summary");

        // 验证最近两轮未被压缩。
        let has_recent_tool = messages.iter().any(|m| m.role == Role::Tool);
        assert!(has_recent_tool, "Recent tool results should be preserved");
    }

    /// 空消息列表不应产生任何压缩。
    #[test]
    fn test_empty_messages() {
        let config = VisibilityHorizonConfig::default();
        let compressor = VisibilityHorizonCompressor::new(config);

        let mut messages: Vec<Message> = vec![];
        let evicted = compressor.compact_horizon(&mut messages);

        assert!(evicted.is_empty());
        assert!(messages.is_empty());
    }

    /// 不含工具消息的对话应保持不变。
    #[test]
    fn test_no_tool_messages() {
        let config = VisibilityHorizonConfig::default();
        let compressor = VisibilityHorizonCompressor::new(config);

        let mut messages = vec![
            Message::system("sys".to_string()),
            Message::user("hello".to_string()),
            Message::assistant("hi".to_string()),
            Message::user("how are you?".to_string()),
            Message::assistant("fine".to_string()),
        ];
        let original_len = messages.len();

        let evicted = compressor.compact_horizon(&mut messages);

        assert!(evicted.is_empty());
        assert_eq!(messages.len(), original_len);
    }

    /// 并行工具调用应被压缩为同一条摘要。
    #[test]
    fn test_parallel_tool_calls() {
        let config = VisibilityHorizonConfig {
            active_window_turns: 1,
            ..Default::default()
        };
        let compressor = VisibilityHorizonCompressor::new(config);

        let mut messages = vec![
            Message::system("sys".to_string()),
            // 第 1 轮：用户提问，助手并行调用 2 个工具
            Message::user("do stuff".to_string()),
            assistant_with_tools(&[("call_a", "read_file"), ("call_b", "grep")]),
            tool_result("call_a", "read_file", &"file content".repeat(100)),
            tool_result("call_b", "grep", &"grep output".repeat(100)),
            Message::assistant("done".to_string()),
            // 第 2 轮：用户追问
            Message::user("thanks".to_string()),
            Message::assistant("welcome".to_string()),
            // 第 3 轮：再加一轮，把第 1 轮推出窗口
            Message::user("bye".to_string()),
            Message::assistant("bye".to_string()),
        ];

        let evicted = compressor.compact_horizon(&mut messages);

        // 第 1 轮的工具组之后有 2 个用户轮次 → > 1 → 应被压缩
        assert!(
            !evicted.is_empty(),
            "Parallel tool results should be evicted"
        );

        // 验证摘要提及了两个工具
        let has_summary = messages.iter().any(|m| {
            m.content
                .as_text()
                .is_some_and(|t| t.contains("read_file") && t.contains("grep"))
        });
        assert!(has_summary, "Summary should mention both parallel tools");
    }

    /// token 预算与最小节省阈值共同控制旧工具组的压缩。
    #[test]
    fn test_token_budget_and_minimum_savings_control_old_tool_compaction() {
        let original = build_conversation(6);
        let mut budgeted = original.clone();
        let budgeted_compressor = VisibilityHorizonCompressor::new(VisibilityHorizonConfig {
            active_window_turns: 1,
            retained_tool_tokens: Some(usize::MAX),
            ..Default::default()
        });
        assert!(
            budgeted_compressor
                .compact_horizon(&mut budgeted)
                .is_empty(),
            "an ample aggregate budget should retain old raw tool results"
        );

        let mut thresholded = original.clone();
        let thresholded_compressor = VisibilityHorizonCompressor::new(VisibilityHorizonConfig {
            active_window_turns: 1,
            retained_tool_tokens: Some(0),
            compact_minimum_tokens: usize::MAX,
            ..Default::default()
        });
        assert!(
            thresholded_compressor
                .compact_horizon(&mut thresholded)
                .is_empty(),
            "a compaction below the minimum savings threshold should not rewrite history"
        );

        let mut constrained = original;
        let constrained_compressor = VisibilityHorizonCompressor::new(VisibilityHorizonConfig {
            active_window_turns: 1,
            retained_tool_tokens: Some(0),
            compact_minimum_tokens: 0,
            ..Default::default()
        });
        assert!(
            !constrained_compressor
                .compact_horizon(&mut constrained)
                .is_empty(),
            "an exhausted budget should compact tool groups outside the latest turn"
        );
        assert_eq!(
            constrained
                .iter()
                .filter(|message| message.role == Role::Tool)
                .count(),
            2,
            "only the latest turn's two raw tool results should remain"
        );
    }

    /// 成功与失败混合时，摘要应反映部分成功。
    #[test]
    fn test_mixed_success_failure() {
        let config = VisibilityHorizonConfig {
            active_window_turns: 0,
            ..Default::default()
        };
        let compressor = VisibilityHorizonCompressor::new(config);

        let mut messages = vec![
            Message::system("sys".to_string()),
            Message::user("run commands".to_string()),
            assistant_with_tools(&[("call_ok", "shell"), ("call_err", "shell")]),
            tool_result("call_ok", "shell", "command succeeded"),
            tool_result("call_err", "shell", "[Error] Permission denied"),
            Message::assistant("one failed".to_string()),
            // window=0 时，之后至少需要 1 个用户轮次
            Message::user("ok".to_string()),
        ];

        let evicted = compressor.compact_horizon(&mut messages);
        assert!(!evicted.is_empty());

        // 摘要应表明部分成功
        let summary = messages
            .iter()
            .find(|m| {
                m.content
                    .as_text()
                    .is_some_and(|t| t.contains("[Horizon compact"))
            })
            .and_then(|m| m.content.as_text())
            .unwrap_or_default();

        assert!(
            summary.contains("1/2 success"),
            "Summary should show 1/2 success, got: {}",
            summary
        );
    }

    /// 压缩时应保留非工具消息。
    #[test]
    fn test_preserves_non_tool_messages() {
        let config = VisibilityHorizonConfig {
            active_window_turns: 0,
            ..Default::default()
        };
        let compressor = VisibilityHorizonCompressor::new(config);

        let mut messages = vec![
            Message::system("sys".to_string()),
            Message::user("q1".to_string()),
            assistant_with_tools(&[("c1", "tool_a")]),
            tool_result("c1", "tool_a", "result_a"),
            Message::assistant("important reasoning text".to_string()),
            Message::user("q2".to_string()),
            Message::assistant("answer 2".to_string()),
        ];

        let _evicted = compressor.compact_horizon(&mut messages);

        // 助手的 "important reasoning text" 应被保留
        let has_reasoning = messages.iter().any(|m| {
            m.content
                .as_text()
                .is_some_and(|t| t.contains("important reasoning"))
        });
        assert!(
            has_reasoning,
            "Non-tool assistant messages should be preserved"
        );

        // 用户消息应被保留
        let user_count = messages.iter().filter(|m| m.role == Role::User).count();
        assert!(user_count >= 2, "User messages should be preserved");
    }

    /// 启用后应将全局目标注入为第一条消息。
    #[test]
    fn test_global_objective_injection() {
        let config = VisibilityHorizonConfig {
            active_window_turns: 0, // 压缩所有工具组
            enable_global_objective: true,
            global_objective: Some("Build a sorting algorithm".to_string()),
            ..Default::default()
        };
        let compressor = VisibilityHorizonCompressor::new(config);

        let mut messages = vec![
            Message::system("You are a helper.".to_string()),
            Message::user("start".to_string()),
            assistant_with_tools(&[("c1", "tool_a")]),
            tool_result("c1", "tool_a", "result"),
            Message::assistant("done".to_string()),
            Message::user("next".to_string()),
        ];

        let evicted = compressor.compact_horizon(&mut messages);

        // 应已发生压缩（之后有 1 个用户轮次 > 阈值 0）
        assert!(!evicted.is_empty(), "Tool results should be evicted");

        // 全局目标应作为第一条消息注入
        assert!(
            messages[0]
                .content
                .as_text()
                .is_some_and(|t| t.contains("[Global Objective]"))
        );
        assert!(
            messages[0]
                .content
                .as_text()
                .is_some_and(|t| t.contains("Build a sorting algorithm"))
        );
    }

    /// 通过 `ContextCompressor` trait 调用压缩。
    #[tokio::test]
    async fn test_context_compressor_trait() {
        let config = VisibilityHorizonConfig {
            active_window_turns: 1,
            ..Default::default()
        };
        let compressor = VisibilityHorizonCompressor::new(config);

        let messages = build_conversation(4);
        let original_count = messages.len();

        let input = CompressionInput {
            messages,
            token_limit: 100_000,
            current_query: None,
            focus_instructions: None,
            cancel_token: None,
            tokenizer: None,
        };

        let output = compressor.compress(input).await.unwrap();

        assert!(
            output.messages.len() < original_count,
            "Should have fewer messages after compression"
        );
        assert!(!output.evicted.is_empty(), "Should have evicted messages");
        assert_eq!(compressor.name(), "VisibilityHorizon");
    }

    /// 25 轮内 token 增长应为次线性。
    #[test]
    fn test_token_stability_over_25_turns() {
        // RFC 3.3 验收标准：对话超过 20 轮后，上下文 token 数保持稳定。
        let config = VisibilityHorizonConfig {
            active_window_turns: 5,
            compact_max_tokens: 50,
            ..Default::default()
        };
        let compressor = VisibilityHorizonCompressor::new(config);
        let tokenizer = HeuristicTokenizer;

        // 使用固定大小的工具输出构建对话，以单独考察轮次数对 token 增长的影响。
        /// 构建工具输出大小固定的多轮对话。
        fn build_constant_conversation(turns: usize) -> Vec<Message> {
            let mut messages = vec![Message::system("You are a helpful assistant.".to_string())];
            for i in 1..=turns {
                messages.push(Message::user(format!("Question {}", i)));
                let ca = format!("call_{}a", i);
                let cb = format!("call_{}b", i);
                messages.push(assistant_with_tools(&[
                    (&ca, &format!("read_{}", i)),
                    (&cb, &format!("grep_{}", i)),
                ]));
                // 固定大小输出（每条 200 字符）
                messages.push(tool_result(&ca, &format!("read_{}", i), &"x".repeat(200)));
                messages.push(tool_result(&cb, &format!("grep_{}", i), &"y".repeat(200)));
                messages.push(Message::assistant(format!("Answer {}", i)));
            }
            messages
        }

        // 测量不同轮次数下压缩后的 token 数
        let mut token_counts: Vec<(usize, usize)> = Vec::new();

        for turns in [5, 10, 15, 20, 25] {
            let mut messages = build_constant_conversation(turns);
            let _evicted = compressor.compact_horizon(&mut messages);

            let total_tokens: usize = messages
                .iter()
                .filter_map(|m| m.content.as_text())
                .map(|c| tokenizer.count_tokens(&c))
                .sum();

            token_counts.push((turns, total_tokens));
        }

        /* 关键不变量：active_window=5 时，仅最近约 5 轮完整保留，
        更早的轮次被压缩为约 50 token 的摘要。
        因此从 10→25 轮的 token 增长应为次线性。 */
        let tokens_at_10 = token_counts[1].1;
        let tokens_at_25 = token_counts[4].1;

        assert!(
            tokens_at_25 < tokens_at_10 * 2,
            "Token growth should be sub-linear with compaction: 10 turns={} tokens, 25 turns={} tokens",
            tokens_at_10,
            tokens_at_25
        );
    }

    /// 30 轮工具调用的集成测试。
    #[test]
    fn test_30_round_integration() {
        // RFC 3.3 验收标准：在视界压缩下进行 30 轮工具调用。
        let config = VisibilityHorizonConfig {
            active_window_turns: 5,
            compact_max_tokens: 50,
            ..Default::default()
        };
        let compressor = VisibilityHorizonCompressor::new(config);

        let mut messages = build_conversation(30);
        let original_count = messages.len();

        /* 每轮产生 5 条消息：user、assistant(2tc)、tool、tool、assistant_answer，
        因此 30 轮 = 150 条消息 + 1 条系统消息 = 151。 */
        assert_eq!(original_count, 151, "Expected 151 messages for 30 turns");

        let evicted = compressor.compact_horizon(&mut messages);

        // active_window=5 且共 30 轮时，只有第 26-30 轮保持完整。
        assert!(
            evicted.len() >= 40,
            "Should evict at least 40 tool result messages, got {}",
            evicted.len()
        );
        assert!(
            messages.len() < original_count,
            "Message count should decrease: was {}, now {}",
            original_count,
            messages.len()
        );

        // 验证最近轮次保留了真实的工具结果
        let recent_tool_results: usize = messages.iter().filter(|m| m.role == Role::Tool).count();
        // 6 个保留轮次 × 2 条工具结果 = 12
        assert!(
            recent_tool_results >= 10,
            "At least 10 recent tool results should be preserved, got {}",
            recent_tool_results
        );

        // 验证不存在缺少对应结果的孤立助手 tool_calls
        let mut orphaned = 0;
        for (i, msg) in messages.iter().enumerate() {
            if msg.role == Role::Assistant
                && let Some(tool_calls) = msg.tool_calls.as_ref()
            {
                let tc_ids: Vec<&str> = tool_calls.iter().map(|tc| tc.id.as_str()).collect();
                for id in &tc_ids {
                    let has_result = messages[i + 1..]
                        .iter()
                        .take(10)
                        .any(|m| m.role == Role::Tool && m.tool_call_id.as_deref() == Some(id));
                    if !has_result {
                        orphaned += 1;
                    }
                }
            }
        }
        assert_eq!(
            orphaned, 0,
            "No orphaned tool_calls should exist after compaction"
        );

        // 验证存在压缩摘要
        let summary_count = messages
            .iter()
            .filter(|m| {
                m.content
                    .as_text()
                    .is_some_and(|t| t.contains("[Horizon compact"))
            })
            .count();
        assert!(
            summary_count >= 20,
            "Should have at least 20 compact summaries, got {}",
            summary_count
        );
    }

    /// 50 轮对话压缩后 token 数应保持稳定。
    #[test]
    fn test_50_turn_token_stability() {
        // RFC 5.2.1：启用视界压缩时，长对话（>50 轮）应保持稳定的 token 数。
        let config = VisibilityHorizonConfig {
            active_window_turns: 5,
            compact_max_tokens: 50,
            ..Default::default()
        };
        let compressor = VisibilityHorizonCompressor::new(config);
        let tokenizer = HeuristicTokenizer;

        // 使用固定大小输出，便于可预测的测量
        /// 构建工具输出大小固定的多轮对话。
        fn build_constant_conversation(turns: usize) -> Vec<Message> {
            let mut messages = vec![Message::system("You are a helpful assistant.".to_string())];
            for i in 1..=turns {
                messages.push(Message::user(format!(
                    "Question {} about various topics",
                    i
                )));
                let ca = format!("call_{}a", i);
                let cb = format!("call_{}b", i);
                messages.push(assistant_with_tools(&[
                    (&ca, &format!("read_{}", i)),
                    (&cb, &format!("grep_{}", i)),
                ]));
                messages.push(tool_result(&ca, &format!("read_{}", i), &"x".repeat(200)));
                messages.push(tool_result(&cb, &format!("grep_{}", i), &"y".repeat(200)));
                messages.push(Message::assistant(format!("Answer to question {}", i)));
            }
            messages
        }

        // 测量 50 轮对话压缩后的 token 数。
        let mut msgs_50 = build_constant_conversation(50);
        let _ = compressor.compact_horizon(&mut msgs_50);
        let tokens_50: usize = msgs_50
            .iter()
            .filter_map(|m| m.content.as_text())
            .map(|c| tokenizer.count_tokens(&c))
            .sum();

        /* active_window=5 时，仅最近约 5 轮完整保留。
        不压缩时，50 轮 × 约 500 token/轮 ≈ 25,000 token。
        压缩后，被压缩的轮次变为约 50 token 的摘要。
        压缩后的大小应远小于未压缩时。 */
        let uncompressed_50 = 50 * 500; // 粗略估计：50 轮 × 每轮约 500 token
        assert!(
            tokens_50 < uncompressed_50 / 3,
            "Compacted 50-turn conversation ({} tokens) should be <33% of uncompressed estimate ({} tokens)",
            tokens_50,
            uncompressed_50
        );

        // 验证旧轮次已被压缩
        let compact_count = msgs_50
            .iter()
            .filter(|m| {
                m.content
                    .as_text()
                    .is_some_and(|t| t.contains("[Horizon compact"))
            })
            .count();
        assert!(
            compact_count >= 40,
            "At least 40 groups should be compacted in 50 turns, got {}",
            compact_count
        );
    }
}
