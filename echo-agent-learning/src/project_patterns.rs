//! 一个真正可用的 echo-agent 工具，无需 LLM 或 API key 即可在本地运行。

use echo_agent::prelude::{Result, Tool, ToolResult};
use echo_agent::tool;
use serde_json::json;
use std::collections::HashMap;

/// 由 `#[tool]` 宏生成的问候工具：返回对指定贡献者的问候语。
#[tool(name = "greet", description = "Greet a contributor")]
async fn greet(name: String) -> Result<ToolResult> {
    Ok(ToolResult::success(format!("你好，{name}")))
}

/// 构造调用参数并执行 `greet` 工具，返回执行结果。
pub async fn run_greet_tool(name: &str) -> Result<ToolResult> {
    let parameters = HashMap::from([("name".to_string(), json!(name))]);
    GreetTool.execute(parameters).await
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 验证由宏生成的工具无需依赖 LLM 即可正常执行。
    #[tokio::test]
    async fn generated_tool_runs_without_an_llm() -> Result<()> {
        let result = run_greet_tool("小明").await?;
        assert!(result.success);
        assert_eq!(result.output, "你好，小明");
        Ok(())
    }
}
