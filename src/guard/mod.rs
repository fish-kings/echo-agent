/*!
内容过滤与安全护栏（Guardrails）。

护栏（Guard）会检查用户输入、LLM 输出以及工具调用，对不安全的内容
进行拦截或修改。支持两种护栏类型：

- **基于规则**（`RuleGuard`、`RuleGuardBuilder`） —— 模式匹配、长度限制、正则表达式
- **基于 LLM**（`LlmGuard`） —— AI 驱动的内容分类

# 快速开始

```rust,ignore
use echo_agent::prelude::*;

# fn main() -> echo_agent::error::Result<()> {
let guard = RuleGuardBuilder::new("safety")
    .max_input_length(10000)
    .build();
let agent = ReactAgentBuilder::new()
    .model("qwen3-max")
    .with_guard(Box::new(guard))
    .build()?;
# Ok(())
# }
```

# 关键类型

| 类型 | 说明 |
|------|-------------|
| [`Guard`] | Trait —— 实现它以创建自定义护栏 |
| [`GuardManager`] | 按顺序运行多个护栏，遇到拦截即短路 |
| [`GuardResult`] | `Pass`（通过）/ `Block { reason }`（拦截）/ `Warn { message }`（警告） |
| [`GuardDirection`] | `Input` / `Output` / `ToolInput` / `ToolOutput` |
*/

/// 基于 LLM 的内容护栏。
#[cfg(feature = "content-guard")]
pub mod llm;
/// 基于规则的内容护栏。
#[cfg(feature = "content-guard")]
pub mod rule;

/// 直接从 `echo_core::guard` 重新导出。
pub mod core {
    pub use echo_core::guard::*;
}

pub use echo_core::guard::*;
