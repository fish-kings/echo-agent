/*!
# echo-core

[echo-agent](https://crates.io/crates/echo_agent) 框架的核心 trait、类型与错误定义。

本 crate 提供了工作区中其他所有 crate 都依赖的共享接口：

| 模块 | 内容 |
|--------|----------|
| [`agent`] | `Agent` trait、`AgentEvent`、`AgentCallback`、`CancellationToken`、内置了若干实现的 `Critic` trait、内置了若干实现的 `Executor` trait、反思相关类型与存储 |
| [`llm`] | `LlmClient` trait、`ChatRequest`/`ChatResponse`/`ChatChunk` 类型 |
| [`tools`] | `Tool` trait、`ToolResult`、`ToolPermission`、权限模型 |
| [`error`] | 覆盖所有失败场景的统一 `Error` 枚举 |
| [`guard`] | `Guard` trait、`GuardResult`、`GuardManager` |
| [`audit`] | `AuditEvent`、`AuditLogger` trait |
| [`retry`] | 带指数退避的 `RetryPolicy` |
| [`tokenizer`] | Token 计数接口 |

大多数用户应当依赖 `echo_agent`（门面 crate），而不是直接依赖 `echo_core`。
*/

/// Agent 核心 trait、事件、回调、Critic/Executor 等
pub mod agent;
/// 审计事件与审计日志接口
pub mod audit;
/// Token 预算管理（LLM 上下文窗口的细粒度分配）
pub mod budget;
/// LLM 熔断器
pub mod circuit_breaker;
/// 上下文压缩 trait 及输入/输出类型
pub mod compression;
/// 统一错误类型
pub mod error;
/// 护栏（Guard）系统
pub mod guard;
/// Hook 类型定义
pub mod hooks;
/// LLM 客户端 trait 与请求/响应类型
pub mod llm;
/// LSP（语言服务器协议）集成类型
#[cfg(feature = "lsp")]
#[cfg_attr(docsrs, doc(cfg(feature = "lsp")))]
pub mod lsp;
/// 长期记忆存储接口
pub mod memory;
/// 插件系统类型
pub mod plugin;
/// 项目级指令发现（带显式仓库边界）
pub mod project_rules;
/// 统一重试策略
pub mod retry;
/// 沙箱执行器核心 trait 与类型
pub mod sandbox;
/// Token 计数接口
pub mod tokenizer;
/// 工具 trait 与权限模型
pub mod tools;
/// 通用工具函数
pub mod utils;

/**
`echo_macros` 生成代码所依赖的实现依赖项。
之所以公开，仅仅是为了让下游宏展开时，用户无需在自己的 manifest 中
重复声明框架的实现依赖。
*/
#[doc(hidden)]
pub mod __macro_support {
    pub use futures;
    pub use schemars;
    pub use serde;
    pub use serde_json;
}
