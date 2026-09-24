//! LSP（语言服务器协议）集成层。
//!
//! 本模块提供：
//! - `LspManager` —— 管理多个语言服务器进程
//! - `StdioLspClient` —— 通过 stdio 与单个服务器通信的 JSON-RPC 客户端
//! - 从 `.lsp.yaml` 文件加载配置
//!
//! ## 架构
//!
//! ```text
//! LspManager
//!   ├── StdioLspClient (python/pyright)
//!   ├── StdioLspClient (typescript/tsserver)
//!   └── StdioLspClient (rust/rust-analyzer)
//! ```

pub mod client;
pub mod config;
pub mod jsonrpc;
pub mod manager;

pub use client::StdioLspClient;
pub use config::LspConfig;
pub use manager::LspManager;
