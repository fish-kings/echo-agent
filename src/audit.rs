/*!
面向 Agent 行为与决策的结构化审计日志。

每一次工具调用、护栏检查和 Agent 决策都可以记录为一条
[`AuditEvent`]，并通过可插拔的后端进行持久化。

# 快速开始

```rust,ignore
use echo_agent::prelude::*;

# fn main() -> echo_agent::error::Result<()> {
let agent = ReactAgentBuilder::new()
    .model("qwen3-max")
    .audit_logger(Arc::new(InMemoryAuditLogger::new()))
    .build()?;
# Ok(())
# }
```

# 关键类型

| 类型 | 说明 |
|------|-------------|
| [`AuditEvent`] | 单条结构化审计记录 |
| [`AuditLogger`] | 审计后端的 trait |
| [`InMemoryAuditLogger`] | 用于测试的内存日志记录器 |
| [`FileAuditLogger`] | 用于生产环境的基于文件的日志记录器 |
| [`DiagnosticDeliveryFailure`] | Trace/Audit 持久化失败的事实记录 |
| [`DiagnosticDeliveryObserver`] | 不能改变执行流程的额外失败观察者 |
*/

/// 直接从 `echo_state::audit` 重新导出。
pub mod state {
    pub use echo_state::audit::*;
}

pub use echo_state::audit::file;
pub use echo_state::audit::file::FileAuditLogger;
pub use echo_state::audit::memory;
pub use echo_state::audit::memory::InMemoryAuditLogger;
pub use echo_state::audit::*;
