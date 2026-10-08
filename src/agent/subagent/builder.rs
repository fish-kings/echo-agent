//! Subagent 构建器 —— 符合人体工程学的链式配置

use std::path::PathBuf;

use super::types::{ExecutionMode, SubagentAccessMode, SubagentDefinition, SubagentKind};

/**
使用流式 API 创建 [`SubagentDefinition`] 实例的构建器。

# 示例

```rust
use echo_agent::subagent::SubagentBuilder;

let def = SubagentBuilder::new("researcher")
    .description("Researches topics thoroughly")
    .fork_mode()
    .model("qwen3")
    .system_prompt("You are a research specialist...")
    .tools(vec!["search", "read_file"])
    .inherit_history(10)
    .timeout(120)
    .tag("research")
    .can_delegate()
    .build();

assert_eq!(def.name, "researcher");
```
*/
pub struct SubagentBuilder {
    /// 正在构建的 subagent 定义
    definition: SubagentDefinition,
}

impl SubagentBuilder {
    /// 以给定名称开始构建一个 subagent。
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            definition: SubagentDefinition {
                name: name.into(),
                description: String::new(),
                kind: SubagentKind::BuiltIn,
                access_mode: SubagentAccessMode::Write,
                execution_mode: ExecutionMode::Sync,
                model: None,
                thinking: None,
                system_prompt: None,
                tool_filter: None,
                max_iterations: None,
                token_limit: None,
                inherit_history: None,
                inherit_memory: false,
                timeout_secs: 0,
                can_delegate: false,
                tags: Vec::new(),
                lightweight: false,
                isolation: None,
                team: None,
                is_background: false,
            },
        }
    }

    /// 设置一段人类可读的描述（会显示在展示给 LLM 的工具描述中）。
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.definition.description = desc.into();
        self
    }

    /// 设置 agent 的来源类型。
    pub fn kind(mut self, kind: SubagentKind) -> Self {
        self.definition.kind = kind;
        self
    }

    /// 从自定义的 `.md` 定义文件加载。
    pub fn custom(mut self, path: impl Into<PathBuf>) -> Self {
        self.definition.kind = SubagentKind::Custom { path: path.into() };
        self
    }

    /// 从插件来源加载。
    pub fn plugin(mut self, source: impl Into<String>) -> Self {
        self.definition.kind = SubagentKind::Plugin {
            source: source.into(),
        };
        self
    }

    /// 声明该角色只能检查环境，不能修改环境。
    pub fn read_only(mut self) -> Self {
        self.definition.access_mode = SubagentAccessMode::ReadOnly;
        self
    }

    /// 声明该角色可以在其运行时边界内修改环境。
    pub fn write_access(mut self) -> Self {
        self.definition.access_mode = SubagentAccessMode::Write;
        self
    }

    /// 将执行模式设置为 Sync（默认）。
    pub fn sync_mode(mut self) -> Self {
        self.definition.execution_mode = ExecutionMode::Sync;
        self
    }

    /// 将执行模式设置为 Fork（继承上下文，独立运行）。
    pub fn fork_mode(mut self) -> Self {
        self.definition.execution_mode = ExecutionMode::Fork;
        /* Sprint 6b：默认值从 10 降到 2。Fork 子代理是专注型子代理；
        10 条消息的前缀会让上下文里堆满过时的对话轮次，
        分散对实际任务的注意力。保留 2 条最近的消息，
        刚好够携带足够的父级状态（通常是最后一条用户消息 + 一个工具结果），
        又不会带来噪音。每个子代理单独调用 `.inherit_history(n)` 仍然可以
        覆盖这个默认值，`.md` frontmatter 也可以按子代理单独设置。 */
        self.definition.inherit_history = Some(2);
        self.definition.inherit_memory = true;
        self
    }

    /// 将执行模式设置为 Teammate（并行执行，基于 handle）。
    pub fn teammate_mode(mut self) -> Self {
        self.definition.execution_mode = ExecutionMode::Teammate;
        self
    }

    /// 通过带版本管理的 DAG 运行时编译所提供的 Team 意图。
    pub fn team(mut self, spec: super::team::TeamSpec) -> Self {
        self.definition.execution_mode = ExecutionMode::Team;
        self.definition.team = Some(spec);
        self
    }

    /// 覆盖该 subagent 使用的模型。
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.definition.model = Some(model.into());
        self
    }

    /**
    覆盖该 subagent 的推理深度（使用 `ThinkingConfig::parse_spec` 语法的
    规格字符串，例如 `"low"`）。`None`（未设置）表示继承父级生成时的
    thinking 配置。
    */
    pub fn thinking(mut self, spec: impl Into<String>) -> Self {
        self.definition.thinking = Some(spec.into());
        self
    }

    /// 覆盖 system prompt。
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.definition.system_prompt = Some(prompt.into());
        self
    }

    /// 按名称限制可用的工具。
    pub fn tools(mut self, tools: Vec<impl Into<String>>) -> Self {
        self.definition.tool_filter = Some(tools.into_iter().map(Into::into).collect());
        self
    }

    /// 设置最大迭代次数。
    pub fn max_iterations(mut self, max: usize) -> Self {
        self.definition.max_iterations = Some(max);
        self
    }

    /// 设置 token 限制。
    pub fn token_limit(mut self, limit: usize) -> Self {
        self.definition.token_limit = Some(limit);
        self
    }

    /// 设置从父级继承的最近消息条数（Fork 模式下使用）。
    pub fn inherit_history(mut self, count: usize) -> Self {
        self.definition.inherit_history = Some(count);
        self
    }

    /// 启用记忆继承。
    pub fn inherit_memory(mut self) -> Self {
        self.definition.inherit_memory = true;
        self
    }

    /// 设置超时时间（秒），0 表示不超时。
    pub fn timeout(mut self, secs: u64) -> Self {
        self.definition.timeout_secs = secs;
        self
    }

    /// 允许该 subagent 继续委派给其他 subagent。
    pub fn can_delegate(mut self) -> Self {
        self.definition.can_delegate = true;
        self
    }

    /// 为 Fork 调度请求一种产品层定义的隔离类型。
    pub fn isolation(mut self, kind: impl Into<String>) -> Self {
        let kind = kind.into();
        self.definition.isolation = (!kind.trim().is_empty()).then(|| kind.trim().to_string());
        self
    }

    /// 标记该角色倾向于后台调度（Phase 2）。
    pub fn background(mut self) -> Self {
        self.definition.is_background = true;
        self
    }

    /// 添加一个用于发现/过滤的标签。
    pub fn tag(mut self, tag: impl Into<String>) -> Self {
        self.definition.tags.push(tag.into());
        self
    }

    /// 批量添加多个标签。
    pub fn tags(mut self, tags: Vec<impl Into<String>>) -> Self {
        self.definition
            .tags
            .extend(tags.into_iter().map(Into::into));
        self
    }

    /// 构建最终的定义。
    pub fn build(self) -> SubagentDefinition {
        // 如果描述为空，自动填充一个默认描述
        let mut def = self.definition;
        if def.description.is_empty() {
            def.description = format!("Subagent '{}'", def.name);
        }
        def
    }
}

/// 子 Agent 构建器的单元测试。
#[cfg(test)]
mod tests {
    use super::*;

    /// 验证同步（sync）类型子 Agent 的构建。
    #[test]
    fn test_builder_sync() {
        let def = SubagentBuilder::new("subagent")
            .description("Does work")
            .sync_mode()
            .timeout(60)
            .build();

        assert_eq!(def.name, "subagent");
        assert_eq!(def.execution_mode, ExecutionMode::Sync);
        assert_eq!(def.timeout_secs, 60);
        assert!(def.inherit_history.is_none());
    }

    /// 验证派生（fork）类型子 Agent 的构建。
    #[test]
    fn test_builder_fork() {
        let def = SubagentBuilder::new("researcher")
            .description("Researches")
            .fork_mode()
            .model("qwen3")
            .system_prompt("You are a researcher")
            .tools(vec!["search", "read"])
            .inherit_history(10)
            .timeout(120)
            .tag("research")
            .can_delegate()
            .build();

        assert_eq!(def.execution_mode, ExecutionMode::Fork);
        assert_eq!(def.model.as_deref(), Some("qwen3"));
        // fork_mode() 的默认值是 2（Sprint 6b），随后 .inherit_history(10) 覆盖了它。
        assert_eq!(def.inherit_history, Some(10));
        assert!(def.can_delegate);
        assert_eq!(def.tags, vec!["research"]);
        assert_eq!(
            def.tool_filter.as_deref(),
            Some(["search".to_string(), "read".to_string()].as_slice())
        );
    }

    /// 验证 fork 模式默认的 `inherit_history` 已下调为 2。
    #[test]
    fn test_builder_fork_default_inherit_history_lowered() {
        // Sprint 6b：fork_mode() 的默认 inherit_history 是 2（之前是 10）。
        let def = SubagentBuilder::new("w")
            .description("d")
            .fork_mode()
            .build();
        assert_eq!(def.inherit_history, Some(2));
    }

    /// 验证队友（teammate）类型子 Agent 的构建。
    #[test]
    fn test_builder_teammate() {
        let def = SubagentBuilder::new("tm").teammate_mode().build();

        assert_eq!(def.execution_mode, ExecutionMode::Teammate);
        assert!(def.inherit_history.is_none());
    }

    /// 验证未指定时会自动生成描述。
    #[test]
    fn test_builder_auto_description() {
        let def = SubagentBuilder::new("auto").build();
        assert!(def.description.contains("auto"));
    }

    /// 验证自定义类型子 Agent 的构建。
    #[test]
    fn test_builder_custom_kind() {
        let def = SubagentBuilder::new("custom")
            .custom("/path/to/agent.md")
            .build();
        assert!(matches!(def.kind, SubagentKind::Custom { .. }));
    }

    /// 验证插件类型子 Agent 的构建。
    #[test]
    fn test_builder_plugin_kind() {
        let def = SubagentBuilder::new("plugin")
            .plugin("remote://registry")
            .build();
        assert!(matches!(def.kind, SubagentKind::Plugin { .. }));
    }
}
