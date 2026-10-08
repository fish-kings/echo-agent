/*!
# echo_integration

[echo-agent](https://crates.io/crates/echo_agent) 框架的集成层。

## 模块

| 模块 | 说明 | 特性开关 |
|--------|-------------|---------|
| [`providers`] | 协议客户端与显式的 `LlmConfig` 构造 | 默认 |
| [`mcp`] | MCP 协议：stdio、SSE、HTTP 传输；`McpManager`、`McpServerConfig` | `mcp` |
| [`channels`] | IM 集成：QQ 机器人（WebSocket）与飞书（Webhook） | `channels` |

大多数用户应当依赖 `echo_agent`（门面 crate），而不是直接依赖
`echo_integration`。
*/

/// 配置 Debug 输出与诊断信息共用的脱敏工具
#[cfg(any(feature = "mcp", feature = "channels"))]
pub(crate) mod redaction;

/// MCP 协议集成：传输层、客户端与服务端管理
#[cfg(feature = "mcp")]
#[cfg_attr(docsrs, doc(cfg(feature = "mcp")))]
pub mod mcp;

/// LSP 语言服务器集成
#[cfg(feature = "lsp")]
#[cfg_attr(docsrs, doc(cfg(feature = "lsp")))]
pub mod lsp;

/// IM 渠道集成（QQ 机器人、飞书等）
#[cfg(feature = "channels")]
#[cfg_attr(docsrs, doc(cfg(feature = "channels")))]
pub mod channels;

/// LLM 供应商协议客户端与运行时配置
pub mod providers;
