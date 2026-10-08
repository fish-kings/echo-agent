/*!
统一的 `AgentBuilder` trait —— 构建任意 agent 类型的通用接口。

所有具体的 builder（`ReactAgentBuilder`、`SubagentBuilder` 等）都实现
这个 trait，从而可以编写与具体 agent 类型无关的通用代码。
*/

use crate::agent::Agent;
use crate::error::Result;
use crate::tools::Tool;

/**
所有 agent builder 共享的最小接口。

# 类型参数

* `A` —— 该 builder 所构建的具体 agent 类型。

# 示例

```rust,ignore
fn create<B: AgentBuilder>(builder: B, task: &str) -> Result<B::Agent> {
    builder.model("qwen3-max").system_prompt("...").build()
}
```
*/
pub trait AgentBuilder: Sized {
    /// 该 builder 所构建的具体 agent 类型。
    type Agent: Agent;

    /// 设置模型名称（例如 "qwen3-max"、"gpt-5.5"）。
    fn model(self, model: impl Into<String>) -> Self;

    /// 设置用于初始化 agent 行为的系统提示词。
    fn system_prompt(self, prompt: impl Into<String>) -> Self;

    /// 设置 agent 的人类可读名称。
    fn name(self, name: impl Into<String>) -> Self;

    /// 设置最大推理迭代次数。
    fn max_iterations(self, max: usize) -> Self;

    /// 设置上下文窗口的最大 token 数限制。
    fn token_limit(self, limit: usize) -> Self;

    /// 注册单个工具。
    fn tool(self, tool: Box<dyn Tool>) -> Self;

    /// 批量注册多个工具。
    fn tools(self, tools: Vec<Box<dyn Tool>>) -> Self {
        tools.into_iter().fold(self, |b, t| b.tool(t))
    }

    /// 完成构建并返回 agent 实例。
    fn build(self) -> Result<Self::Agent>;

    /// 构建 agent 并以 trait object 的形式返回。
    fn build_boxed(self) -> Result<Box<dyn Agent>>
    where
        Self::Agent: 'static,
    {
        self.build().map(|a| Box::new(a) as Box<dyn Agent>)
    }
}
