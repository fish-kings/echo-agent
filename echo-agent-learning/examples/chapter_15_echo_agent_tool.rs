use echo_agent::prelude::Result;
use echo_agent_learning::project_patterns::run_greet_tool;

/// 演示如何调用 echo-agent 的 greet 工具并打印其执行结果。
#[tokio::main]
async fn main() -> Result<()> {
    let result = run_greet_tool("echo-agent 贡献者").await?;
    println!("工具成功: {}", result.success);
    println!("工具输出: {}", result.output);
    Ok(())
}
