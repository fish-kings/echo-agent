//! # echo_integration
//!
//! [echo-agent](https://crates.io/crates/echo_agent) 框架的集成层。
//!
//! ## 模块
//!
//! | 模块 | 说明 | 特性开关 |
//! |--------|-------------|---------|
//! | [`providers`] | 协议客户端与显式的 `LlmConfig` 构造 | 默认 |
//! | [`mcp`] | MCP 协议：stdio、SSE、HTTP 传输；`McpManager`、`McpServerConfig` | `mcp` |
//! | [`channels`] | IM 集成：QQ 机器人（WebSocket）与飞书（Webhook） | `channels` |
//!
//! 大多数用户应当依赖 `echo_agent`（门面 crate），而不是直接依赖
//! `echo_integration`。

#[cfg(any(feature = "mcp", feature = "channels"))]
pub(crate) mod redaction;

#[cfg(feature = "mcp")]
#[cfg_attr(docsrs, doc(cfg(feature = "mcp")))]
pub mod mcp;

#[cfg(feature = "lsp")]
#[cfg_attr(docsrs, doc(cfg(feature = "lsp")))]
pub mod lsp;

#[cfg(feature = "channels")]
#[cfg_attr(docsrs, doc(cfg(feature = "channels")))]
pub mod channels;

pub mod providers;
