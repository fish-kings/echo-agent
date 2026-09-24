//! LSP（Language Server Protocol，语言服务器协议）集成 —— 重新导出。
//!
//! 本模块提供对 LSP 类型以及用于与语言服务器通信的
//! 管理器的访问。

// 重新导出核心类型
pub use echo_core::lsp::*;

// 重新导出集成层
pub use echo_integration::lsp::{LspConfig, LspManager, StdioLspClient};
