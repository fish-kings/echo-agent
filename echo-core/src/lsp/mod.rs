/*!
LSP（Language Server Protocol，语言服务器协议）集成类型。

本模块定义了 LSP 客户端的核心 trait，以及 EchoAgent 代码分析工具所使用的
简化版协议类型。

## 架构

```text
echo-core/src/lsp/        ← 本模块：trait + 类型定义
echo-integration/src/lsp/ ← LspManager、进程启动、JSON-RPC
echo-tools/src/lsp/       ← 工具实现（诊断、跳转定义等）
```

## 支持的能力

| 能力 | 方法 | 用途 |
|-----------|--------|----------|
| 诊断 | `lsp_diagnostics` | 文件中的错误、警告 |
| 跳转到定义 | `lsp_goto_definition` | 跳转到符号定义处 |
| 查找引用 | `lsp_find_references` | 查找所有使用处 |
| 悬浮提示 | `lsp_hover` | 悬浮时显示类型信息、文档 |
| 自动补全 | `lsp_completions` | 自动补全建议 |
*/

/// LSP 客户端 trait 与错误类型
pub mod client;
/// EchoAgent 使用的简化版 LSP 协议类型
pub mod types;

pub use client::{LspClient, LspError, LspResult};
pub use types::*;
