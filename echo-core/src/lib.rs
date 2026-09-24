//! # echo-core
//!
//! [echo-agent](https://crates.io/crates/echo_agent) 框架的核心 trait、类型与错误定义。
//!
//! 本 crate 提供了工作区中其他所有 crate 都依赖的共享接口：
//!
//! | 模块 | 内容 |
//! |--------|----------|
//! | [`agent`] | `Agent` trait、`AgentEvent`、`AgentCallback`、`CancellationToken`、内置了若干实现的 `Critic` trait、内置了若干实现的 `Executor` trait、反思相关类型与存储 |
//! | [`llm`] | `LlmClient` trait、`ChatRequest`/`ChatResponse`/`ChatChunk` 类型 |
//! | [`tools`] | `Tool` trait、`ToolResult`、`ToolPermission`、权限模型 |
//! | [`error`] | 覆盖所有失败场景的统一 `Error` 枚举 |
//! | [`guard`] | `Guard` trait、`GuardResult`、`GuardManager` |
//! | [`audit`] | `AuditEvent`、`AuditLogger` trait |
//! | [`retry`] | 带指数退避的 `RetryPolicy` |
//! | [`tokenizer`] | Token 计数接口 |
//!
//! 大多数用户应当依赖 `echo_agent`（门面 crate），而不是直接依赖 `echo_core`。

pub mod agent;
pub mod audit;
pub mod budget;
pub mod circuit_breaker;
pub mod compression;
pub mod error;
pub mod guard;
pub mod hooks;
pub mod llm;
#[cfg(feature = "lsp")]
#[cfg_attr(docsrs, doc(cfg(feature = "lsp")))]
pub mod lsp;
pub mod memory;
pub mod plugin;
pub mod project_rules;
pub mod retry;
pub mod sandbox;
pub mod tokenizer;
pub mod tools;
pub mod utils;

/// `echo_macros` 生成代码所依赖的实现依赖项。
/// 之所以公开，仅仅是为了让下游宏展开时，用户无需在自己的 manifest 中
/// 重复声明框架的实现依赖。
#[doc(hidden)]
pub mod __macro_support {
    pub use futures;
    pub use schemars;
    pub use serde;
    pub use serde_json;
}
