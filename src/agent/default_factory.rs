/*!
具体的 `DefaultAgentFactory` —— 通过 `ReactAgentBuilder` 创建 Agent。

本模块提供 `echo_core::agent::factory` 中定义的 [`AgentFactory`] trait
的 facade 层实现。它读取 [`AgentFactoryConfig`]，
并创建一个配置好的 [`crate::agent::ReactAgent`]。

# 示例

```rust,no_run
use echo_agent::agent::default_factory::DefaultAgentFactory;
use echo_agent::agent::factory::AgentFactoryConfig;
use echo_core::agent::factory::AgentFactory;

# fn main() -> echo_agent::error::Result<()> {
let config = AgentFactoryConfig::new()
    .model("qwen3-max")
    .name("coder")
    .with_system_prompt("You are a coding assistant");

let factory = DefaultAgentFactory;
let agent = factory.create_agent(config)?;

println!("Agent: {}, Model: {}", agent.name(), agent.model_name());
# Ok(())
# }
```
*/

use crate::agent::Agent;
use crate::agent::react::builder::ReactAgentBuilder;
use crate::error::Result;

/// 使用 [`ReactAgentBuilder`] 创建 Agent 的具体工厂。
pub struct DefaultAgentFactory;

impl AgentFactory for DefaultAgentFactory {
    /// 根据工厂配置创建 ReactAgent 并装箱为 `dyn Agent`。
    fn create_agent(&self, config: AgentFactoryConfig) -> Result<Box<dyn Agent>> {
        let model = config.model_name().to_string();
        let name = config.agent_name().to_string();
        let system_prompt = config.system_prompt().to_string();
        let tools = config.into_tools();

        ReactAgentBuilder::new()
            .model(&model)
            .name(&name)
            .system_prompt(&system_prompt)
            .enable_tools()
            .tools(tools)
            .build_boxed()
    }
}

// ── 重新导出 ──────────────────────────────────────────────────────────────

pub use echo_core::agent::factory::{AgentFactory, AgentFactoryConfig};

/// 默认 Agent 工厂的单元测试。
#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::MockLlmClient;
    use std::sync::Arc;

    /// 验证工厂配置构建器正确设置各字段。
    #[test]
    fn test_factory_creates_config() {
        let config = AgentFactoryConfig::new()
            .model("mock-model")
            .name("test-agent")
            .with_system_prompt("Test agent");

        assert_eq!(config.model_name(), "mock-model");
        assert_eq!(config.agent_name(), "test-agent");
    }

    /// 验证可使用 Mock LLM 客户端构建 Agent。
    #[tokio::test]
    async fn test_factory_build_agent_with_mock() {
        let mock_client = Arc::new(MockLlmClient::new().with_model_name("mock-test"));

        let agent = ReactAgentBuilder::new()
            .llm_client(mock_client)
            .name("factory-test")
            .system_prompt("Test")
            .enable_tools()
            .build_boxed()
            .unwrap();

        assert_eq!(agent.name(), "factory-test");
        assert_eq!(agent.model_name(), "mock-test");
    }
}
