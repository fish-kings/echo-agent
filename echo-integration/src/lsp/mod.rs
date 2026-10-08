/*!
LSP（语言服务器协议）集成层。

本模块提供：
- `LspManager` —— 管理多个语言服务器进程
- `StdioLspClient` —— 通过 stdio 与单个服务器通信的 JSON-RPC 客户端
- 从 `.lsp.yaml` 文件加载配置

## 架构

```text
LspManager
  ├── StdioLspClient (python/pyright)
  ├── StdioLspClient (typescript/tsserver)
  └── StdioLspClient (rust/rust-analyzer)
```
*/

/// 基于 stdio 的 LSP 客户端
pub mod client;
/// LSP 配置加载与本机语言服务器发现
pub mod config;
/// LSP 使用的 JSON-RPC 2.0 消息类型与编解码
pub mod jsonrpc;
/// 多语言服务器进程管理
pub mod manager;

pub use client::StdioLspClient;
pub use config::LspConfig;
pub use manager::LspManager;
