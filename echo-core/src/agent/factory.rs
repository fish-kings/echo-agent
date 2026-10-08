/*!
Agent 工厂模式 —— 基于配置创建 agent

提供 [`AgentFactoryConfig`]，用于承载创建 agent 所需的配置，
以及一个从配置生产 agent 的 [`AgentFactory`] trait。

# 示例

```rust,ignore
use echo_core::agent::factory::{AgentFactory, AgentFactoryConfig};

let config = AgentFactoryConfig::new()
    .model("qwen3-max")
    .name("my-agent")
    .with_system_prompt("You are a helpful assistant");

struct ProductFactory;
// 在拥有 Agent builder 的产品/门面代码中实现 `AgentFactory`。
```
*/

use crate::error::Result;
use crate::tools::Tool;

/// 核心 agent 的默认系统提示词，描述其工作方式与响应风格。
pub const DEFAULT_CORE_AGENT_SYSTEM_PROMPT: &str = r#"You are a practical local AI agent for real coding, research, data analysis, and task execution.

Core operating model:
- Establish facts from context before making claims.
- Use tools when they can verify, inspect, execute, or make progress.
- Preserve user work and avoid unrelated changes.
- Prefer root-cause fixes over cosmetic workarounds.
- Validate results when possible and report remaining risk.
- Keep stable policy separate from dynamic per-turn context such as memories, tool results, hooks, and task state.
- For high-stakes or time-sensitive topics, verify with available reliable sources before presenting precise claims.

Response style:
- Language: the current user request, not the language of system prompts, tool descriptions, source text, or logs, determines the response language each turn. Use it for natural-language prose, progress, final answers, plan/task titles and descriptions, and Subagent briefs. An explicit requested output language wins. Keep code, identifiers, file paths, shell commands, schema keys, log keywords, and verbatim evidence unchanged.
- Lead with findings for reviews.
- For implementation, summarize changed behavior and verification."#;

// ── Agent Factory Config ────────────────────────────────────────────────────

/**
通过 [`AgentFactory`] 创建 agent 所需的配置。

承载构建 agent 所需的模型、名称、系统提示词与工具。工厂读取该配置
并委托给相应的 builder 完成构建。

注意：该结构体**没有**实现 `Clone`，因为它持有不可克隆的
`Box<dyn Tool>` 实例。工厂通过 [`AgentFactoryConfig::into_tools`]
完整消费该配置。
*/
pub struct AgentFactoryConfig {
    /// LLM 模型标识符（例如 "qwen3-max"、"gpt-5.5"）。
    model: String,
    /// 用于日志与编排中展示的人类可读 agent 名称。
    name: String,
    /// 用于初始化 agent 行为的系统提示词。
    system_prompt: String,
    /// 要注册到该 agent 上的自定义工具。
    tools: Vec<Box<dyn Tool>>,
}

impl AgentFactoryConfig {
    /**
    创建一个新的工厂配置。

    默认值：model = ""，name = "assistant"，并使用一份实用型 agent 系统提示词。
    */
    pub fn new() -> Self {
        Self {
            model: String::new(),
            name: "assistant".to_string(),
            system_prompt: DEFAULT_CORE_AGENT_SYSTEM_PROMPT.to_string(),
            tools: Vec::new(),
        }
    }

    // ── Builder 风格的 setter ──────────────────────────────────────────────────

    /// 设置 LLM 模型名称。
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// 设置 agent 名称。
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// 设置系统提示词。
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = prompt.into();
        self
    }

    /// 注册单个工具。
    pub fn tool(mut self, tool: Box<dyn Tool>) -> Self {
        self.tools.push(tool);
        self
    }

    /// 批量注册多个工具。
    pub fn tools(mut self, tools: Vec<Box<dyn Tool>>) -> Self {
        self.tools.extend(tools);
        self
    }

    // ── 访问器 ──────────────────────────────────────────────────────────────

    /// LLM 模型标识符。
    pub fn model_name(&self) -> &str {
        &self.model
    }

    /// agent 名称。
    pub fn agent_name(&self) -> &str {
        &self.name
    }

    /// 系统提示词。
    pub fn system_prompt(&self) -> &str {
        &self.system_prompt
    }

    /// 已注册的自定义工具数量。
    pub fn tool_count(&self) -> usize {
        self.tools.len()
    }

    /**
    消费该配置并返回其持有的工具向量。

    由于 `Box<dyn Tool>` 无法克隆，这是工厂从配置中取出工具的主要方式。
    */
    pub fn into_tools(self) -> Vec<Box<dyn Tool>> {
        self.tools
    }
}

impl Default for AgentFactoryConfig {
    /// 等价于 [`AgentFactoryConfig::new`]。
    fn default() -> Self {
        Self::new()
    }
}

// ── Agent Factory Trait ─────────────────────────────────────────────────────

/// 从 [`AgentFactoryConfig`] 创建 agent 的 trait。
pub trait AgentFactory: Send + Sync {
    /// 根据给定的配置创建一个 agent。
    fn create_agent(&self, config: AgentFactoryConfig) -> Result<Box<dyn crate::agent::Agent>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    // 验证新建配置的默认值符合预期
    #[test]
    fn test_factory_config_new() {
        let config = AgentFactoryConfig::new();
        assert_eq!(config.model_name(), "");
        assert_eq!(config.agent_name(), "assistant");
        assert_eq!(config.system_prompt(), DEFAULT_CORE_AGENT_SYSTEM_PROMPT);
    }

    // 验证 builder 风格的链式调用能正确设置各字段
    #[test]
    fn test_factory_config_builder() {
        let config = AgentFactoryConfig::new()
            .model("qwen3-max")
            .name("my-agent")
            .with_system_prompt("You are a coder");

        assert_eq!(config.model_name(), "qwen3-max");
        assert_eq!(config.agent_name(), "my-agent");
        assert_eq!(config.system_prompt(), "You are a coder");
    }

    // 验证 Default 实现与 new() 行为一致
    #[test]
    fn test_factory_config_default() {
        let config = AgentFactoryConfig::default();
        assert_eq!(config.agent_name(), "assistant");
    }
}
