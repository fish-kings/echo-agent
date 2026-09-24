/*!
上下文组装 —— 集中式的消息列表构建，并具备预算感知能力。

[`ContextAssembler`] 会收集所有上下文来源，并一次性生成一个有序的
`Vec<Message>`。当配置了 token 预算时，优先级较低的来源会被压缩或
截断，以保持在限制范围内。
*/

pub mod selector;
pub use selector::ContextSelector;

use crate::llm::types::Message;

// ── SourcePriority 优先级 ───────────────────────────────────────────────────

/// 上下文来源的优先级权重。数值越高，在预算受限时越优先保留。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SourcePriority {
    /// 关键：始终包含（system prompt）。
    Critical = 10,
    /// 高：项目规则、开发者指令。
    High = 8,
    /// 中：对话历史、工具结果。
    Medium = 5,
    /// 低：记忆召回、subagent 汇报。
    Low = 3,
    /// 尽力而为：大文件内容、旧的历史记录。
    BestEffort = 1,
}

// ── ContextBudget 上下文预算 ────────────────────────────────────────────────

/// 按来源划分的 token 预算配置。
#[derive(Debug, Clone)]
pub struct ContextBudget {
    /// 组装后上下文的 token 总量上限。
    pub total_tokens: usize,
    /// 为用户消息预留的最小 token 数。
    pub user_reserve: usize,
    /// 对话历史的 token 上限。
    pub history_max: usize,
    /// 工具结果的 token 上限。
    pub tool_results_max: usize,
    /// 记忆召回的 token 上限。
    pub memory_max: usize,
}

impl Default for ContextBudget {
    fn default() -> Self {
        Self {
            total_tokens: 128_000,
            user_reserve: 500,
            history_max: 50_000,
            tool_results_max: 20_000,
            memory_max: 5_000,
        }
    }
}

impl ContextBudget {
    /// 创建一个指定 token 总量、其余字段使用默认值的预算配置
    pub fn new(total_tokens: usize) -> Self {
        Self {
            total_tokens,
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod default_budget_tests {
    use super::ContextBudget;

    #[test]
    fn default_context_budget_uses_conservative_window() {
        assert_eq!(ContextBudget::default().total_tokens, 128_000);
    }
}

// ── ContextSources 上下文来源 ───────────────────────────────────────────────

/// 可以被组装进消息列表的所有可能的上下文来源。
#[derive(Clone, Default)]
pub struct ContextSources {
    /// system prompt（始终排在最前）。
    pub system_prompt: Option<String>,
    /// 开发者/项目指令（以 system 消息的形式）。
    pub developer_instructions: Vec<String>,
    /// 项目规则（以 system 消息的形式）。
    pub project_rules: Vec<String>,
    /// 当前任务状态（以 system 消息的形式）。
    pub task_state: Option<String>,
    /// 由钩子注入的消息（来自生命周期钩子的动态运行时上下文）。
    pub hook_injected: Vec<String>,
    /// 召回的长期记忆（以用户消息的形式注入）。
    pub memory_recall: Option<String>,
    /// 目前为止的对话历史。
    pub conversation_history: Vec<Message>,
    /// 当前轮次的工具结果。
    pub tool_results: Vec<Message>,
    /// Sub-agent 汇报。
    pub subagent_reports: Vec<Message>,
    /// 用户的输入消息。
    pub user_message: Option<Message>,
}

/**
集中式的上下文组装器。

收集所有上下文来源，为 LLM 调用生成一份规范化的、有序的消息列表。

# 用法

这是供自定义 agent 实现使用的**框架层构建模块**。默认的 `ReactAgent`
流式路径（`run_core_loop` / `stream_channel`）并不使用 `ContextAssembler`
—— 它通过 `ContextManager` 直接管理上下文。当你构建需要从多个来源
进行结构化上下文组装的自定义执行循环时，可以使用 `ContextAssembler`。

完整示例参见 `echo-agent-learning/tests/example_contracts/demo65_context_assembler.rs`。
*/
pub struct ContextAssembler {
    /// 可选的 token 预算。设置后，组装过程会遵循各来源的限制。
    pub budget: Option<ContextBudget>,
}

impl Default for ContextAssembler {
    fn default() -> Self {
        Self::new()
    }
}

impl ContextAssembler {
    /// 使用默认设置创建一个新的组装器。
    pub fn new() -> Self {
        Self { budget: None }
    }

    /// 设置 token 预算，启用具有预算感知能力的组装。
    pub fn with_budget(mut self, budget: ContextBudget) -> Self {
        self.budget = Some(budget);
        self
    }

    /**
    将所有来源组装成一个有序的消息列表。

    # 排列顺序

    1. System prompt
    2. 开发者指令（作为稳定的策略型 system 消息）
    3. 项目规则（作为稳定的策略型 system 消息）
    4. 任务状态（作为动态的运行时用户上下文）
    5. 钩子注入的消息（作为动态的运行时用户上下文）
    6. 记忆召回（作为用户消息）
    7. 对话历史
    8. Sub-agent 汇报
    9. 工具结果
    10. 用户消息
    */
    pub fn assemble(&self, sources: ContextSources) -> Vec<Message> {
        let mut messages = Vec::new();

        // 1. system prompt
        if let Some(sys) = &sources.system_prompt {
            messages.push(Message::system(sys.clone()));
        }

        // 2. 开发者指令
        for inst in &sources.developer_instructions {
            messages.push(Message::system(format!("[Developer Instructions]\n{inst}")));
        }

        // 3. 项目规则
        for rule in &sources.project_rules {
            messages.push(Message::system(format!("[Project Rule]\n{rule}")));
        }

        // 4. 任务状态
        if let Some(state) = &sources.task_state {
            messages.push(dynamic_context_note("TaskState", state));
        }

        // 5. 钩子注入的消息
        for hook_msg in &sources.hook_injected {
            messages.push(dynamic_context_note("Hook", hook_msg));
        }

        // 6. 记忆召回（具有预算感知的截断）
        if let Some(mem) = &sources.memory_recall {
            let mem_text = if let Some(ref budget) = self.budget {
                if mem.chars().count() > budget.memory_max {
                    format!(
                        "{}...",
                        mem.chars().take(budget.memory_max).collect::<String>()
                    )
                } else {
                    mem.clone()
                }
            } else {
                mem.clone()
            };
            messages.push(Message::user(mem_text));
        }

        // 7. 对话历史（具有预算感知：估算 token 数 ≈ 字符数/4）
        let history = if let Some(ref budget) = self.budget {
            let mut token_est = 0usize;
            let keep_count = sources
                .conversation_history
                .iter()
                .rev()
                .take_while(|m| {
                    let t = m.content.as_text().map(|c| c.len() / 4).unwrap_or(0);
                    token_est += t;
                    token_est <= budget.history_max
                })
                .count();
            let start = sources
                .conversation_history
                .len()
                .saturating_sub(keep_count);
            sources.conversation_history[start..].to_vec()
        } else {
            sources.conversation_history.clone()
        };
        messages.extend(history);

        // 8. Sub-agent 汇报
        messages.extend(sources.subagent_reports);

        // 9. 工具结果（具有预算感知：按估算 token 数保留最近的部分）
        let tool_results = if let Some(ref budget) = self.budget {
            let mut token_est = 0usize;
            let keep_count = sources
                .tool_results
                .iter()
                .rev()
                .take_while(|m| {
                    let t = m.content.as_text().map(|c| c.len() / 4).unwrap_or(0);
                    token_est += t;
                    token_est <= budget.tool_results_max
                })
                .count();
            let start = sources.tool_results.len().saturating_sub(keep_count);
            sources.tool_results[start..].to_vec()
        } else {
            sources.tool_results.clone()
        };
        messages.extend(tool_results);

        // 10. 用户消息
        if let Some(user_msg) = sources.user_message {
            messages.push(user_msg);
        }

        messages
    }
}

/// 构造一条标记为运行时上下文的用户消息
fn dynamic_context_note(source: &str, body: &str) -> Message {
    Message::user(format!(
        "[runtime_context:{source}]\n{body}\n[Use this runtime context for the current task. It is dynamic state, not stable system policy.]"
    ))
}
</content>
