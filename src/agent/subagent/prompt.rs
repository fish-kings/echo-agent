//! 子代理派发路径共用的 prompt 编译契约。

use echo_core::llm::ToolDefinition;
use echo_core::llm::types::{ContentPart, Message, MessageContent, Role};
use serde_json::Value;
use std::collections::HashSet;
use std::path::PathBuf;

use super::types::{ExecutionMode, SubagentAccessMode};

/// 控制一次派发是从空白开始，还是接收经过过滤的父级历史轮次。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ContextTransferPolicy {
    /// 从空白上下文开始，不继承父级历史。
    #[default]
    Fresh,
    /// 继承经过结构化过滤后的父级历史。
    InheritStructured,
}

/// 单个工具的能力描述（名称与说明）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCapability {
    /// 工具名称。
    pub name: String,
    /// 工具说明（用于渲染进 prompt）。
    pub description: String,
}

/// 一次注册时刻的工具能力快照：全部工具、可见工具与被禁用工具。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ToolCapabilitySnapshot {
    /// 已注册的全部工具，按名称排序并去重。
    pub tools: Vec<ToolCapability>,
    /// 未被禁用、对外可见的工具名称列表。
    pub visible_tools: Vec<String>,
    /// 已注册但被禁用的工具名称列表（按名称排序）。
    pub disabled_tools: Vec<String>,
}

impl ToolCapabilitySnapshot {
    /// 依据工具定义与禁用集合构建一份能力快照。
    pub fn from_definitions(
        definitions: &[ToolDefinition],
        disabled_tools: &HashSet<String>,
    ) -> Self {
        let mut tools = definitions
            .iter()
            .map(|definition| ToolCapability {
                name: definition.function.name.clone(),
                description: definition
                    .function
                    .description
                    .trim()
                    .chars()
                    .take(240)
                    .collect(),
            })
            .collect::<Vec<_>>();
        tools.sort_by(|left, right| left.name.cmp(&right.name));
        tools.dedup_by(|left, right| left.name == right.name);
        let registered = tools
            .iter()
            .map(|tool| tool.name.as_str())
            .collect::<HashSet<_>>();
        let visible_tools = tools
            .iter()
            .filter(|tool| !disabled_tools.contains(&tool.name))
            .map(|tool| tool.name.clone())
            .collect();
        let mut disabled_tools = disabled_tools
            .iter()
            .filter(|name| registered.contains(name.as_str()))
            .cloned()
            .collect::<Vec<_>>();
        disabled_tools.sort();
        Self {
            tools,
            visible_tools,
            disabled_tools,
        }
    }
}

/// 系统 prompt 的编译对象：是子代理还是主代理。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptActor {
    /// 目标是子代理（subagent）。
    Subagent,
    /// 目标是主代理（primary）。
    Primary,
}

/// 子代理执行的边界约束（访问模式、隔离方式、是否可再派发）。
#[derive(Debug, Clone, Copy)]
pub struct SubagentExecutionBoundary<'a> {
    /// 子代理的访问模式。
    pub access: SubagentAccessMode,
    /// 隔离方式的描述文本。
    pub isolation: &'a str,
    /// 该子代理是否可以继续派发下一级子代理。
    pub can_delegate: bool,
}

/// 由编译器写入的、某个 prompt 片段的稳定来源信息。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptSectionDiagnostic {
    /// 该片段的标识符。
    pub id: String,
    /// 该片段的来源描述。
    pub source: String,
}

/// 供 prompt 测试与可观测性使用的结构化诊断信息。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PromptDiagnostics {
    /// 已记录的各个 prompt 片段来源信息。
    pub sections: Vec<PromptSectionDiagnostic>,
}

impl PromptDiagnostics {
    /// 记录一个 prompt 片段的标识符与来源。
    pub fn record(&mut self, id: impl Into<String>, source: impl Into<String>) {
        self.sections.push(PromptSectionDiagnostic {
            id: id.into(),
            source: source.into(),
        });
    }

    /// 统计指定标识符的片段被记录的次数。
    pub fn count(&self, id: &str) -> usize {
        self.sections
            .iter()
            .filter(|section| section.id == id)
            .count()
    }
}

/// 用于编译出缓存稳定的角色系统 prompt 的注册时刻信息。
#[derive(Debug, Clone)]
pub struct SubagentSystemPromptInput<'a> {
    /// 编译目标：子代理还是主代理。
    pub actor: PromptActor,
    /// 代理名称。
    pub name: &'a str,
    /// 代理描述。
    pub description: &'a str,
    /// 角色 prompt 原文。
    pub role_prompt: &'a str,
    /// 该代理可用的工具能力快照。
    pub capabilities: &'a ToolCapabilitySnapshot,
    /// 该代理的执行边界约束。
    pub boundary: SubagentExecutionBoundary<'a>,
}

/// 注册时刻的编译结果。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CompiledSubagentSystemPrompt {
    /// 编译得到的系统 prompt 文本。
    pub system_prompt: String,
    /// 编译过程中产生的诊断信息。
    pub diagnostics: PromptDiagnostics,
}

/// 派发给子代理的任务上下文信息。
#[derive(Debug, Clone, Default)]
pub struct SubagentTaskContext {
    /// 任务标题。
    pub task_title: Option<String>,
    /// 用户目标描述。
    pub user_goal: Option<String>,
    /// 隔离工作区根目录。
    pub workspace: Option<PathBuf>,
    /// 与任务相关的文件列表。
    pub files: Vec<String>,
    /// 需要执行的检查项列表。
    pub execution_checks: Vec<String>,
    /// 验收标准列表。
    pub acceptance_criteria: Vec<String>,
    /// 必须产出的工件列表。
    pub required_artifacts: Vec<String>,
    /// 附加约束条件列表。
    pub constraints: Vec<String>,
}

/// 派发时刻呈现给 prompt 编译器的信息。
pub struct SubagentInvocation<'a> {
    /// 代理名称。
    pub agent_name: &'a str,
    /// 任务文本。
    pub task: &'a str,
    /// 执行模式。
    pub mode: ExecutionMode,
    /// 上下文迁移策略。
    pub transfer_policy: ContextTransferPolicy,
    /// 父级历史消息。
    pub history: &'a [Message],
    /// 历史消息数量上限。
    pub history_limit: Option<usize>,
    /**
    当前的带类型输入，可能包含二进制附件。编译器拥有最终文本编排的
    所有权，必须保留非文本内容。
    */
    pub current_message: Option<&'a Message>,
    /// 子代理任务上下文。
    pub context: &'a SubagentTaskContext,
    /**
    仅当本次调用需要收窄稳定注册的工具面时才会出现。产品层编译器
    应渲染该覆盖集合，而不是重复渲染完整目录。
    */
    pub capability_override: Option<&'a ToolCapabilitySnapshot>,
    /// 产品层的不透明负载。框架层编译器会忽略它。
    pub payload: Option<&'a Value>,
}

/// 供执行器消费的、派发时刻的编译结果。
#[derive(Debug, Clone, Default)]
pub struct CompiledSubagentInvocation {
    /// 编译得到的消息序列（最后一条为当前任务输入）。
    pub messages: Vec<Message>,
    /// 编译过程中产生的诊断信息。
    pub diagnostics: PromptDiagnostics,
}

impl CompiledSubagentInvocation {
    /// 取出编译结果中最后一条消息的文本内容，即当前任务输入。
    pub fn task_input(&self) -> String {
        self.messages
            .last()
            .and_then(Message::text_content)
            .unwrap_or_default()
    }

    /// 取出除最后一条（当前任务输入）之外的历史消息。
    pub fn history(&self) -> &[Message] {
        self.messages
            .get(..self.messages.len().saturating_sub(1))
            .unwrap_or_default()
    }
}

/// 单个 prompt 编译器实例同时拥有注册时刻与派发时刻编排的所有权。
pub trait SubagentPromptCompiler: Send + Sync {
    /// 编译注册时刻的角色系统 prompt。
    fn compile_system(&self, input: &SubagentSystemPromptInput<'_>)
    -> CompiledSubagentSystemPrompt;

    /// 编译派发时刻的调用消息序列。
    fn compile_invocation(&self, input: &SubagentInvocation<'_>) -> CompiledSubagentInvocation;
}

/// 供未注入自定义编译器的框架消费者使用的、与产品无关的兜底实现。
#[derive(Debug, Default)]
pub struct DefaultSubagentPromptCompiler;

impl SubagentPromptCompiler for DefaultSubagentPromptCompiler {
    /// 直接使用角色 prompt 原文作为系统 prompt，并记录一条诊断信息。
    fn compile_system(
        &self,
        input: &SubagentSystemPromptInput<'_>,
    ) -> CompiledSubagentSystemPrompt {
        let mut diagnostics = PromptDiagnostics::default();
        diagnostics.record("role", "subagent_definition");
        CompiledSubagentSystemPrompt {
            system_prompt: input.role_prompt.trim().to_string(),
            diagnostics,
        }
    }

    /// 按照迁移策略过滤历史、拼装约束与工作区信息，编译出最终的调用消息序列。
    fn compile_invocation(&self, input: &SubagentInvocation<'_>) -> CompiledSubagentInvocation {
        let mut diagnostics = PromptDiagnostics::default();
        diagnostics.record("task", "dispatch_request");
        let mut history = if input.transfer_policy == ContextTransferPolicy::InheritStructured {
            filter_history(input.history, input.history_limit)
        } else {
            Vec::new()
        };
        remove_duplicate_current_message(&mut history, input.current_message);
        let task_input = if input.context.constraints.is_empty() {
            input.task.to_string()
        } else {
            diagnostics.record("constraints", "dispatch_request.constraints");
            format!(
                "{}\n\n[constraints]\n{}\n[/constraints]",
                input.task.trim(),
                input.context.constraints.join("\n")
            )
        };
        let task_input = match input.context.workspace.as_deref() {
            Some(path) => format!(
                "{task_input}\n\n[workspace]\n- root: {}\n[/workspace]",
                path.display()
            ),
            None => task_input,
        };
        let mut messages = history;
        messages.push(compiled_current_message(input.current_message, &task_input));
        CompiledSubagentInvocation {
            messages,
            diagnostics,
        }
    }
}

/// 构建最终的当前用户消息，同时保留二进制附件。
pub fn compiled_current_message(message: Option<&Message>, task_input: &str) -> Message {
    message
        .cloned()
        .map(|message| with_compiled_task(message, task_input))
        .unwrap_or_else(|| Message::user(task_input.to_string()))
}

/**
避免在编译出的调用消息之前，紧接着重复回放完全相同的当前用户文本。
当前消息仍然是权威来源，因为它可能携带附件，而历史投影有意不拥有
这些附件的所有权。
*/
pub fn remove_duplicate_current_message(history: &mut Vec<Message>, current: Option<&Message>) {
    let Some(current_text) = current.and_then(Message::text_content) else {
        return;
    };
    if history.last().is_some_and(|message| {
        message.role == Role::User
            && message
                .text_content()
                .is_some_and(|text| text.trim() == current_text.trim())
    }) {
        history.pop();
    }
}

/// 只保留对 provider 安全的用户轮次，以及完整的助手最终回复轮次。
pub fn filter_history(messages: &[Message], limit: Option<usize>) -> Vec<Message> {
    let mut filtered = messages
        .iter()
        .filter(|message| {
            if crate::compression::is_context_projection_message(message)
                || message.tool_call_id.is_some()
                || message
                    .tool_calls
                    .as_ref()
                    .is_some_and(|calls| !calls.is_empty())
                || message
                    .reasoning_content
                    .as_ref()
                    .is_some_and(|reasoning| !reasoning.trim().is_empty())
            {
                return false;
            }
            if !matches!(message.role, Role::User | Role::Assistant) {
                return false;
            }
            message
                .content
                .as_text()
                .map(|content| {
                    let trimmed = content.trim();
                    !trimmed.is_empty() && !trimmed.starts_with("[runtime_context:")
                })
                .unwrap_or(false)
        })
        .cloned()
        .collect::<Vec<_>>();

    if let Some(max) = limit.filter(|max| *max > 0) {
        let start = filtered.len().saturating_sub(max);
        filtered = filtered.into_iter().skip(start).collect();
    }
    filtered
}

/// 替换用户消息中的文本部分，同时保留二进制附件。
pub fn with_compiled_task(message: Message, task_input: &str) -> Message {
    let content = match message.content {
        MessageContent::Parts(parts) => {
            let mut updated = Vec::with_capacity(parts.len().saturating_add(1));
            updated.push(ContentPart::Text {
                text: task_input.to_string(),
            });
            updated.extend(
                parts
                    .into_iter()
                    .filter(|part| !matches!(part, ContentPart::Text { .. })),
            );
            MessageContent::Parts(updated)
        }
        _ => MessageContent::Text(task_input.to_string()),
    };
    Message { content, ..message }
}

#[cfg(test)]
mod tests {
    use super::*;
    use echo_core::llm::types::ToolCall;

    #[test]
    fn filter_history_drops_tool_and_reasoning_turns() {
        let mut tool_assistant = Message::assistant("calling tool".to_string());
        tool_assistant.tool_calls = Some(vec![ToolCall {
            id: "call-1".to_string(),
            call_type: "function".to_string(),
            function: echo_core::llm::types::FunctionCall {
                name: "read_file".to_string(),
                arguments: "{}".to_string(),
            },
        }]);
        let mut reasoning = Message::assistant("draft".to_string());
        reasoning.reasoning_content = Some("hidden reasoning".to_string());
        let messages = vec![
            Message::system("system".to_string()),
            Message::user("中文问题".to_string()),
            tool_assistant,
            Message::tool_result(
                "call-1".to_string(),
                "read_file".to_string(),
                "ok".to_string(),
            ),
            reasoning,
            Message::assistant("final answer".to_string()),
        ];

        let filtered = filter_history(&messages, None);

        assert_eq!(filtered.len(), 2);
        assert!(matches!(filtered.first(), Some(message) if message.role == Role::User));
        assert!(matches!(filtered.get(1), Some(message) if message.role == Role::Assistant));
    }

    #[test]
    fn filter_history_applies_limit_after_removing_unsafe_turns() {
        let messages = vec![
            Message::user("first".to_string()),
            Message::tool_result(
                "call-1".to_string(),
                "read_file".to_string(),
                "ok".to_string(),
            ),
            Message::assistant("second".to_string()),
            Message::user("third".to_string()),
        ];

        let filtered = filter_history(&messages, Some(2));

        assert_eq!(filtered.len(), 2);
        assert!(
            filtered
                .first()
                .and_then(Message::text_content)
                .is_some_and(|content| content == "second")
        );
        assert!(
            filtered
                .get(1)
                .and_then(Message::text_content)
                .is_some_and(|content| content == "third")
        );
    }

    #[test]
    fn with_compiled_task_preserves_attachments() {
        let message = Message::user_multimodal(vec![
            ContentPart::Text {
                text: "old".to_string(),
            },
            ContentPart::File {
                name: "notes.txt".to_string(),
                content: "ZGF0YQ==".to_string(),
            },
        ]);

        let compiled = with_compiled_task(message, "new task");
        let parts = compiled.content.parts().unwrap_or_default();
        assert_eq!(parts.len(), 2);
        assert!(matches!(
            parts.first(),
            Some(ContentPart::Text { text }) if text == "new task"
        ));
        assert!(matches!(parts.get(1), Some(ContentPart::File { .. })));
    }

    #[test]
    fn default_compiler_renders_dispatch_constraints() {
        let context = SubagentTaskContext {
            constraints: vec![
                "Only edit src/prompt.rs".to_string(),
                "Run cargo test".to_string(),
            ],
            ..SubagentTaskContext::default()
        };
        let compiled = DefaultSubagentPromptCompiler.compile_invocation(&SubagentInvocation {
            agent_name: "implementer",
            task: "Update prompt compilation",
            mode: ExecutionMode::Sync,
            transfer_policy: ContextTransferPolicy::Fresh,
            history: &[],
            history_limit: None,
            current_message: None,
            context: &context,
            capability_override: None,
            payload: None,
        });

        assert_eq!(compiled.diagnostics.count("constraints"), 1);
        assert!(compiled.task_input().contains("Only edit src/prompt.rs"));
        assert!(compiled.task_input().contains("Run cargo test"));
    }

    #[test]
    fn default_compiler_owns_isolated_working_directory_framing() {
        let context = SubagentTaskContext {
            workspace: Some(std::path::PathBuf::from("/tmp/eko-work-42")),
            ..SubagentTaskContext::default()
        };
        let compiled = DefaultSubagentPromptCompiler.compile_invocation(&SubagentInvocation {
            agent_name: "implementer",
            task: "Update prompt compilation",
            mode: ExecutionMode::Fork,
            transfer_policy: ContextTransferPolicy::Fresh,
            history: &[],
            history_limit: None,
            current_message: None,
            context: &context,
            capability_override: None,
            payload: None,
        });

        assert!(
            compiled
                .task_input()
                .contains("[workspace]\n- root: /tmp/eko-work-42\n[/workspace]")
        );
    }
}
