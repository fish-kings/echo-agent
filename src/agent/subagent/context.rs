/*!
子 Agent 上下文的隔离与继承。

定义父 Agent 向子 Agent 共享哪些内容，并提供创建隔离执行上下文的工具函数。
*/

use echo_core::llm::ToolDefinition;
use echo_core::llm::types::{Message, Role};
use std::collections::HashMap;
use std::sync::Arc;

use crate::memory::store::Store;

use super::types::ExecutionMode;

// ── 上下文继承 ───────────────────────────────────────────────────────

/**
声明子 Agent 从父 Agent 继承哪些内容。

## Fresh 与 Fork 继承（与 Claude/Cursor 对齐）

- **Fresh**（[`Self::fresh_default`]）：不继承父级历史 / 记忆。
  当前由用户撰写的请求仍会作为限定范围的目标携带，使隔离的子 Agent
  在不继承对话记录的情况下也能保留用户意图。这是 TaskRuntime 与
  `agent_tool`（省略 mode 时）的产品默认值。
- **Fork 继承**（[`Self::fork_default`]）：继承过滤后的近期历史 + 记忆。
  父级系统提示词永远不会以用户文本的形式传递。

`ExecutionMode::Fork` 与此正交：它选择的是并发派发路径
（信号量 + worktree/workspace）。一次 Fork *执行* 仍可使用 Fresh *继承*。

历史模式预设：
- **Sync**：不继承（[`Self::sync_default`] == fresh）。
- **Fork 模式预设**：继承工具 + 近期历史。
- **Teammate/Team**：有界的结构化历史，绝不包含父级系统文本。
*/
#[derive(Debug, Clone)]
pub struct ContextInheritance {
    /// 按名称继承指定工具。`None` 表示继承全部。
    pub inherit_tools: Option<Vec<String>>,
    /// 从对话历史中继承最近 N 条消息。`None` 表示不继承。
    pub inherit_history: Option<usize>,
    /// 是否继承父级记忆存储的引用。
    pub inherit_memory: bool,
    /// 向子 Agent 上下文注入自定义键值元数据。
    pub inject_metadata: HashMap<String, String>,
}

impl ContextInheritance {
    /// Sync 模式默认值：不继承任何内容（通过互斥锁共享状态）。
    pub fn sync_default() -> Self {
        Self {
            inherit_tools: None,
            inherit_history: None,
            inherit_memory: false,
            inject_metadata: HashMap::new(),
        }
    }

    /**
    与 Claude/Cursor 对齐的默认值：不继承父级对话。

    新调用点应优先使用该名称；[`Self::sync_default`] 作为字段完全相同的
    历史别名保留。
    */
    pub fn fresh_default() -> Self {
        Self::sync_default()
    }

    /**
    Fork 模式默认值：继承工具 + 最近 2 条消息。

    Sprint 6b：由 10 降为 2（此前继承过多，陈旧轮次撑大了 Fork 子 Agent 的
    上下文）。`SubagentDefinition.inherit_history`（例如来自 `.md` frontmatter
    或 `.inherit_history(n)`）现在会在派发时由所配置的提示词编译器采纳，
    并覆盖此默认值。
    */
    pub fn fork_default() -> Self {
        Self {
            inherit_tools: None,
            inherit_history: Some(2),
            inherit_memory: true,
            inject_metadata: HashMap::new(),
        }
    }

    /// Teammate/Team 默认值：有界历史，不继承工具与记忆。
    pub fn teammate_default() -> Self {
        Self {
            inherit_tools: Some(Vec::new()),
            inherit_history: Some(2),
            inherit_memory: false,
            inject_metadata: HashMap::new(),
        }
    }

    /// 为给定的执行模式选择默认继承配置。
    pub fn for_mode(mode: &ExecutionMode) -> Self {
        match mode {
            ExecutionMode::Sync => Self::sync_default(),
            ExecutionMode::Fork => Self::fork_default(),
            ExecutionMode::Teammate => Self::teammate_default(),
            ExecutionMode::Team => Self::teammate_default(),
        }
    }
}

/// 默认等同于 [`ContextInheritance::sync_default`]。
impl Default for ContextInheritance {
    fn default() -> Self {
        Self::sync_default()
    }
}

/**
用于继承的父 Agent 上下文快照。

在派生子 Agent 之前从父级提取。这是只读快照 —— 子 Agent 获得的是自己的副本。
*/
#[derive(Clone)]
pub struct SubagentContext {
    /// 父级的工具定义（经 `ContextInheritance::inherit_tools` 过滤）。
    pub tool_definitions: Vec<ToolDefinition>,
    /// 父级的近期对话消息（受 `inherit_history` 限制）。
    pub messages: Vec<Message>,
    /// 父级的记忆存储（若共享）。
    pub store: Option<Arc<dyn Store>>,

    // ── 限定范围的上下文字段（Step 6） ──────────────────────────────────────
    /// 父级的总体目标（作为上下文）。
    pub parent_goal: Option<String>,
    /// 该子 Agent 允许使用的工具（覆盖继承配置）。
    pub allowed_tools: Option<Vec<String>>,
}

/// 自定义 Debug 输出：记忆存储仅显示占位符。
impl std::fmt::Debug for SubagentContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SubagentContext")
            .field("tool_definitions", &self.tool_definitions)
            .field("messages", &self.messages)
            .field("store", &self.store.as_ref().map(|_| "Store { .. }"))
            .field("parent_goal", &self.parent_goal)
            .field("allowed_tools", &self.allowed_tools)
            .finish()
    }
}

impl SubagentContext {
    /// 创建空上下文（不继承任何内容）。
    pub fn empty() -> Self {
        Self {
            tool_definitions: Vec::new(),
            messages: Vec::new(),
            store: None,
            // 限定范围的上下文字段
            parent_goal: None,
            allowed_tools: None,
        }
    }

    /// 将继承规格应用于父级完整上下文，构建子 Agent 上下文。
    pub fn from_parent(
        all_tools: &[ToolDefinition],
        all_messages: &[Message],
        store: Option<Arc<dyn Store>>,
        inheritance: &ContextInheritance,
    ) -> Self {
        let parent_goal = latest_user_request(all_messages);
        let filtered_tools = if let Some(allowed) = &inheritance.inherit_tools {
            // 指定了工具列表：仅继承列出的工具
            all_tools
                .iter()
                .filter(|t| allowed.iter().any(|a| a == &t.function.name))
                .cloned()
                .collect()
        } else if inheritance.inherit_tools.is_none()
            && inheritance.inherit_history.is_none()
            && !inheritance.inherit_memory
        {
            // 未指定工具列表，且未请求其它继承：不继承工具
            Vec::new()
        } else {
            // 未指定工具列表，但请求了其它继承内容：继承全部工具
            all_tools.to_vec()
        };

        let messages = match inheritance.inherit_history {
            Some(0) => all_messages.to_vec(),
            Some(n) => {
                let start = all_messages.len().saturating_sub(n);
                all_messages.get(start..).unwrap_or_default().to_vec()
            }
            None => Vec::new(),
        };

        Self {
            tool_definitions: filtered_tools,
            messages,
            store: if inheritance.inherit_memory {
                store
            } else {
                None
            },
            /* 当前活跃的用户请求属于限定范围的委派上下文，而非继承的对话历史，
            因此 Fresh 子 Agent 也会收到它。 */
            parent_goal,
            allowed_tools: None,
        }
    }

    /// 检查该上下文是否包含任何可继承内容。
    pub fn has_content(&self) -> bool {
        !self.tool_definitions.is_empty()
            || !self.messages.is_empty()
            || self.store.is_some()
            || self.parent_goal.is_some()
            || self.allowed_tools.is_some()
    }
}

/// 查找最近一条真实的用户请求（跳过上下文投影消息与运行时上下文消息）。
fn latest_user_request(messages: &[Message]) -> Option<String> {
    messages.iter().rev().find_map(|message| {
        if message.role != Role::User || crate::compression::is_context_projection_message(message)
        {
            return None;
        }

        let text = message.content.as_text()?;
        let trimmed = text.trim();
        if trimmed.is_empty() || trimmed.starts_with("[runtime_context:") {
            return None;
        }
        Some(trimmed.to_string())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sync 默认值不继承历史与记忆。
    #[test]
    fn test_sync_default_no_inheritance() {
        let inh = ContextInheritance::sync_default();
        assert!(inh.inherit_history.is_none());
        assert!(!inh.inherit_memory);
    }

    /// fresh_default 与 sync_default 字段一致。
    #[test]
    fn fresh_default_is_alias_of_sync_default() {
        let a = ContextInheritance::fresh_default();
        let b = ContextInheritance::sync_default();
        assert_eq!(a.inherit_history, b.inherit_history);
        assert_eq!(a.inherit_memory, b.inherit_memory);
    }

    /// Fresh 继承只保留当前用户请求作为父级目标。
    #[test]
    fn from_parent_fresh_keeps_only_current_user_request() {
        let tools = vec![ToolDefinition {
            tool_type: "function".to_string(),
            function: echo_core::llm::types::FunctionSpec {
                name: "search".into(),
                description: "Search".into(),
                parameters: serde_json::json!({}),
            },
        }];
        let msgs = vec![Message::user("hello".to_string())];
        let ctx =
            SubagentContext::from_parent(&tools, &msgs, None, &ContextInheritance::fresh_default());
        assert!(ctx.messages.is_empty());
        assert!(ctx.tool_definitions.is_empty());
        assert!(ctx.store.is_none());
        assert_eq!(ctx.parent_goal.as_deref(), Some("hello"));
        assert!(ctx.has_content());
    }

    /// 提取父级目标时忽略运行时上下文与投影消息。
    #[test]
    fn from_parent_ignores_runtime_context_and_projection_messages() {
        let mut projected = crate::compression::ContextManager::builder(4096).build();
        projected.push(Message::user("用户请求：核对并发问题 🧭".to_string()));
        projected.apply_projections(&[crate::compression::ContextProjection {
            marker: "test:turn-contract".to_string(),
            message: Some(Message::user("English runtime contract".to_string())),
        }]);
        projected.push(Message::user(
            "[runtime_context:task-runtime]\ninternal state".to_string(),
        ));

        let ctx = SubagentContext::from_parent(
            &[],
            projected.messages(),
            None,
            &ContextInheritance::fresh_default(),
        );

        assert_eq!(
            ctx.parent_goal.as_deref(),
            Some("用户请求：核对并发问题 🧭")
        );
        assert!(ctx.messages.is_empty());
        assert!(ctx.store.is_none());
    }

    /// 父级目标取最近一条真实用户请求。
    #[test]
    fn from_parent_uses_latest_real_user_request() {
        let messages = vec![
            Message::user("first request".to_string()),
            Message::assistant("answer".to_string()),
            Message::user("最后一个真实请求".to_string()),
        ];

        let ctx = SubagentContext::from_parent(
            &[],
            &messages,
            None,
            &ContextInheritance::fresh_default(),
        );

        assert_eq!(ctx.parent_goal.as_deref(), Some("最后一个真实请求"));
    }

    /// Fork 默认值继承 2 条历史与记忆。
    #[test]
    fn test_fork_default_inherits() {
        let inh = ContextInheritance::fork_default();
        // Sprint 6b：fork 默认 inherit_history 由 10 降为 2。
        assert_eq!(inh.inherit_history, Some(2));
        assert!(inh.inherit_memory);
    }

    /// Teammate 默认值使用有界历史且不继承工具与记忆。
    #[test]
    fn test_teammate_default_uses_bounded_history() {
        let inh = ContextInheritance::teammate_default();
        assert_eq!(inh.inherit_history, Some(2));
        assert_eq!(inh.inherit_tools, Some(Vec::new()));
        assert!(!inh.inherit_memory);
    }

    /// 各执行模式映射到对应的默认继承配置。
    #[test]
    fn test_for_mode() {
        assert!(
            ContextInheritance::for_mode(&ExecutionMode::Sync)
                .inherit_history
                .is_none()
        );
        assert_eq!(
            ContextInheritance::for_mode(&ExecutionMode::Fork).inherit_history,
            Some(2)
        );
        assert_eq!(
            ContextInheritance::for_mode(&ExecutionMode::Teammate).inherit_history,
            Some(2)
        );
    }

    /// 空上下文不含任何可继承内容。
    #[test]
    fn test_empty_context() {
        let ctx = SubagentContext::empty();
        assert!(!ctx.has_content());
    }

    /// 指定工具列表时仅继承列出的工具。
    #[test]
    fn test_from_parent_filters_tools() {
        let tools = vec![
            ToolDefinition {
                tool_type: "function".to_string(),
                function: echo_core::llm::types::FunctionSpec {
                    name: "search".into(),
                    description: "Search".into(),
                    parameters: serde_json::json!({}),
                },
            },
            ToolDefinition {
                tool_type: "function".to_string(),
                function: echo_core::llm::types::FunctionSpec {
                    name: "read".into(),
                    description: "Read".into(),
                    parameters: serde_json::json!({}),
                },
            },
        ];

        let inh = ContextInheritance {
            inherit_tools: Some(vec!["search".into()]),
            ..ContextInheritance::sync_default()
        };

        let ctx = SubagentContext::from_parent(&tools, &[], None, &inh);
        assert_eq!(ctx.tool_definitions.len(), 1);
        assert!(matches!(
            ctx.tool_definitions.first(),
            Some(tool) if tool.function.name == "search"
        ));
    }
}
