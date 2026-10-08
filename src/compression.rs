/*!
用于管理 token 预算的上下文压缩策略。

当会话增长超出模型上下文窗口时，压缩器会在保留最相关信息的
同时缩减历史记录。

# 策略

| 压缩器 | 策略 | 适用场景 |
|------------|----------|----------|
| `SlidingWindowCompressor` | 保留最近 N 条消息 | 简单、快速 |
| `SummaryCompressor` | 使用 LLM 对较早的消息做摘要 | 高质量（默认） |

# 快速开始

```rust,ignore
use echo_agent::prelude::*;

# fn main() -> echo_agent::error::Result<()> {
let agent = ReactAgentBuilder::new()
    .model("qwen3-max")
    .compressor(Box::new(SlidingWindowCompressor::new(4096)))
    .build()?;
# Ok(())
# }
```
*/

/// 直接从 `echo_state::compression` 重新导出。
pub mod state {
    pub use echo_state::compression::*;
}

pub use echo_state::compression::*;
