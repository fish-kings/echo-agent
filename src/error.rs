//! echo-agent 框架的统一错误类型。
//!
//! 所有错误都归并在 `Error` 枚举下，其变体涵盖 LLM 调用失败、
//! 工具错误、MCP 问题等情况。
//!
//! `Result<T>` 类型别名为 `std::result::Result<T, Error>`。可通过
//! `use echo_agent::prelude::*` 将其引入作用域。

/// 直接从 `echo_core::error` 重新导出。
pub mod core {
    pub use echo_core::error::*;
}

pub use echo_core::error::*;
