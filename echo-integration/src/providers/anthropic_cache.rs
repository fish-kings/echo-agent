/*!
Anthropic 缓存断点辅助模块。

本模块把断点策略从 `AnthropicClient::convert_request` 中抽离为独立、
可测试的辅助逻辑：将 [`PromptCacheLayout`] 映射为具体计划，
再由 Anthropic provider 映射为协议层的 `cache_control` 断点。

# 断点分配（最多 4 个，Anthropic API 限制）

| # | 目标                   | 作用                                  |
|---|------------------------|---------------------------------------|
| 1 | SystemLastBlock        | 跨轮次缓存系统提示词                  |
| 2 | ToolsLastTool          | 缓存工具定义                          |
| 3 | HistoryIndex(~75%)     | 缓存深层历史（长对话）                |
| 4 | HistoryLastStable      | 下一轮缓存最新的稳定消息              |

运行时上下文消息（带 `[runtime_context:` 标记）始终不参与断点——
它们每轮都会变化，会导致缓存失效。
*/

use echo_core::llm::cache::layout::{BreakpointTarget, PromptCacheLayout};

/**
面向 Anthropic Messages API 的已解析缓存断点计划。

由 [`PromptCacheLayout`] 只读视图生成；
[`AnthropicClient`](super::anthropic::AnthropicClient) 会将其映射为请求体中的
`cache_control: { type: "ephemeral" }` 字段。
*/
#[derive(Debug, Clone)]
pub struct AnthropicCachePlan {
    /// 从布局推导出的断点目标（最多 4 个）。
    pub breakpoints: Vec<BreakpointTarget>,
    /// 是否建议设置系统级断点。
    pub has_system_breakpoint: bool,
    /// 是否建议设置工具级断点。
    pub has_tool_breakpoint: bool,
}

impl AnthropicCachePlan {
    /**
    根据只读布局视图构建计划。

    这是统一入口，取代了之前散落在 `convert_request` 与
    `apply_conversation_cache_breakpoints` 中的临时逻辑。
    */
    pub fn from_layout(layout: &PromptCacheLayout<'_>) -> Self {
        let mut breakpoints = Vec::with_capacity(4);

        let has_system = !layout.system.is_empty() || !layout.canonical.is_empty();
        let has_tools = !layout.tools.is_empty();
        let has_history = !layout.history.is_empty();

        if has_system {
            breakpoints.push(BreakpointTarget::SystemLastBlock);
        }
        if has_tools {
            breakpoints.push(BreakpointTarget::ToolsLastTool);
        }
        if has_history && layout.history.len() >= 4 {
            // 长对话在 75% 深度处设置断点
            let idx = (layout.history.len() - 1) * 3 / 4;
            breakpoints.push(BreakpointTarget::HistoryIndex(idx));
        }
        if has_history {
            breakpoints.push(BreakpointTarget::HistoryLastStable);
        }

        // Anthropic 硬性限制：总计最多 4 个断点
        breakpoints.truncate(4);

        let has_system_breakpoint = breakpoints
            .iter()
            .any(|b| matches!(b, BreakpointTarget::SystemLastBlock));
        let has_tool_breakpoint = breakpoints
            .iter()
            .any(|b| matches!(b, BreakpointTarget::ToolsLastTool));

        Self {
            breakpoints,
            has_system_breakpoint,
            has_tool_breakpoint,
        }
    }

    /// 根据已有的 `cache_hints` 构建计划（缺失时回退到硬编码策略，以保持向后兼容）。
    pub fn from_layout_or_default(layout: Option<&PromptCacheLayout<'_>>) -> Self {
        match layout {
            Some(l) => Self::from_layout(l),
            None => Self::default_plan(),
        }
    }

    /// 无可用布局时使用的默认硬编码计划（向后兼容）。
    fn default_plan() -> Self {
        Self {
            breakpoints: vec![],
            has_system_breakpoint: true,
            has_tool_breakpoint: true,
        }
    }

    /// 判断给定的 [`BreakpointTarget`] 是否存在于本计划中。
    pub fn has(&self, target: BreakpointTarget) -> bool {
        self.breakpoints.contains(&target)
    }

    /// 以消息索引形式返回对话历史断点（相对于历史段），调用方需自行加上段基址偏移。
    pub fn history_indices(&self) -> Vec<usize> {
        self.breakpoints
            .iter()
            .filter_map(|bp| match bp {
                BreakpointTarget::HistoryIndex(i) => Some(*i),
                _ => None,
            })
            .collect()
    }

    /// 计划中是否包含 `HistoryLastStable` 断点。
    pub fn has_history_last_stable(&self) -> bool {
        self.breakpoints
            .iter()
            .any(|b| matches!(b, BreakpointTarget::HistoryLastStable))
    }

    /// 对话历史断点总数（HistoryIndex + HistoryLastStable）。
    pub fn history_breakpoint_count(&self) -> usize {
        self.breakpoints
            .iter()
            .filter(|b| {
                matches!(
                    b,
                    BreakpointTarget::HistoryIndex(_) | BreakpointTarget::HistoryLastStable
                )
            })
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use echo_core::llm::types::Message;

    /// 构造系统消息。
    fn sys(t: &str) -> Message {
        Message::system(t.to_string())
    }
    /// 构造用户消息。
    fn user(t: &str) -> Message {
        Message::user(t.to_string())
    }
    /// 构造运行时上下文消息。
    fn rt(t: &str) -> Message {
        Message::user(format!("[runtime_context:{t}]"))
    }

    /// 验证运行时上下文消息不会被选为断点。
    #[test]
    fn plan_skips_runtime_context_in_breakpoints() {
        let msgs = vec![
            sys("You are Echo Agent"),
            user("h1"),
            user("h2"),
            user("h3"),
            user("h4"),
            rt("turn\ncwd: /tmp"),
        ];
        let tools = vec![];
        let layout = PromptCacheLayout::from_messages(&msgs, &tools);
        let plan = AnthropicCachePlan::from_layout(&layout);
        // runtime_context 不应出现在断点中
        assert!(
            !plan.breakpoints.iter().any(
                |b| matches!(b, BreakpointTarget::HistoryIndex(i) if *i >= layout.history.len())
            )
        );
        assert!(plan.has_history_last_stable());
        assert!(!plan.has(BreakpointTarget::ToolsLastTool));
    }

    /// 验证断点数量被截断为最多 4 个。
    #[test]
    fn plan_truncates_to_four_breakpoints() {
        // 10 条历史消息 → 可使用完整断点集合
        let msgs: Vec<Message> = (0..10).map(|i| user(&format!("h{i}"))).collect();
        let tools = vec![];
        let layout = PromptCacheLayout::from_messages(&msgs, &tools);
        let plan = AnthropicCachePlan::from_layout(&layout);
        assert!(plan.breakpoints.len() <= 4);
    }

    /// 验证短对话不会生成 HistoryIndex 断点。
    #[test]
    fn no_history_breakpoints_for_short_conversation() {
        let msgs = vec![sys("S"), user("only one")];
        let tools = vec![];
        let layout = PromptCacheLayout::from_messages(&msgs, &tools);
        let plan = AnthropicCachePlan::from_layout(&layout);
        // 只有 1 条历史消息 → 不设置 HistoryIndex 断点
        assert!(
            !plan
                .breakpoints
                .iter()
                .any(|b| matches!(b, BreakpointTarget::HistoryIndex(_)))
        );
        // 历史非空时 HistoryLastStable 仍应存在
        assert!(plan.has_history_last_stable());
    }

    /// 验证存在系统提示与工具时会生成对应断点。
    #[test]
    fn system_and_tools_breakpoints_when_present() {
        let msgs = vec![sys("S")];
        let tools = vec![echo_core::llm::types::ToolDefinition {
            tool_type: "function".to_string(),
            function: echo_core::llm::types::FunctionSpec {
                name: "test_tool".to_string(),
                description: "A test tool".to_string(),
                parameters: serde_json::json!({}),
            },
        }];
        let layout = PromptCacheLayout::from_messages(&msgs, &tools);
        let plan = AnthropicCachePlan::from_layout(&layout);
        assert!(plan.has_system_breakpoint);
        assert!(plan.has_tool_breakpoint);
    }

    /// 验证无布局时回退到默认计划。
    #[test]
    fn from_layout_or_default_falls_back_when_none() {
        let plan = AnthropicCachePlan::from_layout_or_default(None);
        assert!(plan.breakpoints.is_empty());
        assert!(plan.has_system_breakpoint);
        assert!(plan.has_tool_breakpoint);
    }
}
