//! LSP 客户端 trait —— 与语言服务器通信的接口。

use crate::lsp::types::*;
use futures::future::BoxFuture;

/// LSP 操作过程中可能发生的错误。
#[derive(Debug, thiserror::Error)]
pub enum LspError {
    /// 服务器尚未完成初始化。
    #[error("Server not initialized")]
    NotInitialized,

    /// 指定语言的服务器进程未在运行。
    #[error("Server not running for language: {0}")]
    NotRunning(String),

    /// 请求超时。
    #[error("Request timed out")]
    Timeout,

    /// 服务器返回了错误。
    #[error("Server returned error: {0}")]
    ServerError(String),

    /// 启动服务器进程失败。
    #[error("Failed to spawn server process: {0}")]
    SpawnError(String),

    /// 通信过程中发生错误。
    #[error("Communication error: {0}")]
    CommunicationError(String),

    /// URI 格式非法。
    #[error("Invalid URI: {0}")]
    InvalidUri(String),
}

/// LSP 操作统一使用的结果类型别名。
pub type LspResult<T> = Result<T, LspError>;

/// 与 LSP 语言服务器通信的 trait。
///
/// 实现方负责管理单个语言服务器进程的生命周期，并在 EchoAgent 的
/// 简化类型与完整的 LSP JSON-RPC 协议之间进行转换。
///
/// ## 生命周期
///
/// ```text
/// new() → initialize() → [did_open/did_change/did_save]* → shutdown()
/// ```
///
/// ## 线程安全
///
/// 由于 LSP 操作可能被多个异步任务并发调用，实现方必须满足 `Send + Sync`。
///
/// ## 对象安全
///
/// 本 trait 使用 `BoxFuture` 作为返回类型以保持对象安全，
/// 从而可以作为 `dyn LspClient` 使用。
pub trait LspClient: Send + Sync {
    /// 语言标识符（例如 "python"、"typescript"、"rust"）。
    fn language(&self) -> &str;

    /// 服务器进程当前是否正在运行。
    fn is_running(&self) -> bool;

    /// 服务器是否已完成 `initialize` 握手。
    fn is_initialized(&self) -> bool;

    /// 启动服务器进程并执行初始化握手。
    ///
    /// `root_uri` 是工作区根目录（例如 `file:///path/to/project`）。
    fn initialize<'a>(&'a mut self, root_uri: &'a str) -> BoxFuture<'a, LspResult<()>>;

    /// 优雅地关闭服务器。
    fn shutdown(&mut self) -> BoxFuture<'_, LspResult<()>>;

    /// 获取某个文件的诊断信息。
    ///
    /// 返回服务器报告的错误、警告及其他诊断信息。
    fn diagnostics<'a>(&'a self, uri: &'a str) -> BoxFuture<'a, LspResult<Vec<Diagnostic>>>;

    /// 查找指定位置符号的定义。
    fn goto_definition<'a>(
        &'a self,
        uri: &'a str,
        position: Position,
    ) -> BoxFuture<'a, LspResult<Vec<Location>>>;

    /// 查找指定位置符号的所有引用。
    fn find_references<'a>(
        &'a self,
        uri: &'a str,
        position: Position,
    ) -> BoxFuture<'a, LspResult<Vec<Location>>>;

    /// 获取指定位置的悬浮提示信息。
    fn hover<'a>(
        &'a self,
        uri: &'a str,
        position: Position,
    ) -> BoxFuture<'a, LspResult<Option<HoverInfo>>>;

    /// 获取指定位置的自动补全建议。
    fn completion<'a>(
        &'a self,
        uri: &'a str,
        position: Position,
    ) -> BoxFuture<'a, LspResult<Vec<CompletionItem>>>;

    /// 通知服务器某个文件已被打开。
    fn did_open<'a>(
        &'a self,
        uri: &'a str,
        language_id: &'a str,
        text: &'a str,
    ) -> BoxFuture<'a, LspResult<()>>;

    /// 通知服务器某个文件已被修改。
    fn did_change<'a>(
        &'a self,
        uri: &'a str,
        changes: Vec<TextChange>,
    ) -> BoxFuture<'a, LspResult<()>>;

    /// 通知服务器某个文件已被保存。
    fn did_save<'a>(&'a self, uri: &'a str) -> BoxFuture<'a, LspResult<()>>;

    /// 通知服务器某个文件已被关闭。
    fn did_close<'a>(&'a self, uri: &'a str) -> BoxFuture<'a, LspResult<()>>;

    /// 获取服务器当前的状态。
    fn status(&self) -> LspServerStatus;
}
