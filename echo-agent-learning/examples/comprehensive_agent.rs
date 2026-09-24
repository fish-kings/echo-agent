/*! 一个确定性的端到端示例，组合了自定义工具、ReAct 循环
与公共门面（facade）LLM 客户端。将本示例改造为实际应用时，
请把 mock 客户端替换为已配置好的真实 provider。
*/

use std::sync::Arc;

use echo_agent::prelude::*;
use echo_agent::testing::MockLlmClient;
use echo_agent::tool;

/// 计算两个整数之和（使用饱和加法避免溢出）的示例工具。
#[tool(name = "add", description = "Add two integers")]
async fn add(a: i64, b: i64) -> Result<ToolResult> {
    Ok(ToolResult::success((a.saturating_add(b)).to_string()))
}

/// 演示构建带有工具调用能力的 ReAct agent，并使用 mock LLM 客户端完成一次问答。
#[tokio::main]
async fn main() -> Result<()> {
    let llm = Arc::new(
        MockLlmClient::new()
            .then_tool_call("call-1", "add", r#"{"a": 40, "b": 2}"#)
            .with_response("The calculation is complete: 42."),
    );
    let agent = ReactAgentBuilder::new()
        .llm_client(llm)
        .system_prompt("You are a concise assistant. Use the add tool for arithmetic.")
        .tool(Box::new(AddTool))
        .build()?;

    let answer = agent.execute("What is 40 + 2?").await?;
    println!("Answer: {answer}");
    Ok(())
}
