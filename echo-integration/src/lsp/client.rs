/*!
 * 基于 stdio 的 LSP 客户端——启动语言服务器子进程，
 * 并通过 stdin/stdout 上的 JSON-RPC 与其通信。
 */

use echo_core::lsp::{
    CompletionItem, Diagnostic, HoverInfo, Location, LspClient, LspError, LspResult,
    LspServerConfig, LspServerStatus, Position, TextChange,
};
use futures::future::BoxFuture;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex as StdMutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::{Mutex, oneshot};

use super::jsonrpc::{self, JsonRpcNotification, JsonRpcRequest, JsonRpcResponse};

/**
 * 由每个派生客户端句柄捕获的、共享的管理器生命周期栅栏。
 *
 * 该栅栏刻意保持为框架内部实现。SDK 句柄保留现有的客户端形态，
 * 而管理器始终是唯一有权维持子进程存活的主体。
 */
pub(crate) struct LspLifecycle {
    /// 该生命周期所属的代次（generation）。
    pub(crate) generation: u64,
    /// 生命周期是否已关闭。
    closed: AtomicBool,
}

impl LspLifecycle {
    /// 以指定代次创建一个尚未关闭的生命周期栅栏。
    pub(crate) fn new(generation: u64) -> Self {
        Self {
            generation,
            closed: AtomicBool::new(false),
        }
    }

    /// 关闭栅栏，此后所有派生句柄均视为失效。
    pub(crate) fn close(&self) {
        self.closed.store(true, Ordering::SeqCst);
    }

    /// 判断栅栏未关闭且代次与给定值一致。
    pub(crate) fn is_live(&self, generation: u64) -> bool {
        !self.closed.load(Ordering::SeqCst) && self.generation == generation
    }
}

/// 单个 JSON-RPC 请求的超时时间。
const REQUEST_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);
/// 允许接收的单条 LSP 消息最大字节数。
const MAX_LSP_MESSAGE_BYTES: usize = 16 * 1024 * 1024;

/// 判断消息体大小是否在允许范围内。
fn accepts_lsp_message_size(bytes: usize) -> bool {
    bytes <= MAX_LSP_MESSAGE_BYTES
}

/// 客户端运行时状态：服务器状态与请求准入共享同一把锁。
struct LspRuntime {
    /// 服务器当前状态。
    status: LspServerStatus,
    /// 等待响应的请求，按请求 ID 索引。
    pending: HashMap<u64, oneshot::Sender<JsonRpcResponse>>,
    /// 是否仍接受新请求。
    accepting: bool,
}

impl LspRuntime {
    /// 登记一个待响应请求；服务器未运行或已停止接受请求时返回错误。
    fn register(&mut self, id: u64, tx: oneshot::Sender<JsonRpcResponse>) -> LspResult<()> {
        if !self.status.running || !self.accepting {
            return Err(LspError::NotInitialized);
        }
        self.pending.insert(id, tx);
        Ok(())
    }

    /// 终结运行时：停止接受请求、标记为未运行，并丢弃所有待响应请求。
    fn settle(&mut self, error: Option<String>) {
        self.accepting = false;
        self.status.running = false;
        self.status.initialized = false;
        self.status.pid = None;
        if self.status.last_error.is_none() {
            self.status.last_error = error;
        }
        self.pending.clear();
    }

    /// 进入终止流程：停止接受请求、记录首个错误并丢弃待响应请求。
    fn begin_terminal(&mut self, error: Option<String>) {
        self.accepting = false;
        if self.status.last_error.is_none() {
            self.status.last_error = error;
        }
        self.pending.clear();
    }
}

/**
 * 面向单个语言服务器的 stdio LSP 客户端。
 *
 * 以子进程方式启动服务器，并通过 stdin（请求）与 stdout（响应）
 * 上的 JSON-RPC 进行通信。
 */
pub struct StdioLspClient {
    /// 语言标识。
    language: String,
    /// 服务器配置。
    config: LspServerConfig,
    /// 子进程句柄。
    child: Option<Arc<Mutex<Child>>>,
    /// 向写入任务发送 JSON-RPC 消息的通道。
    writer_tx: Option<tokio::sync::mpsc::Sender<Vec<u8>>>,
    /// 负责写 stdin 的后台任务。
    writer_task: Option<tokio::task::JoinHandle<()>>,
    /// 负责读 stdout 的后台任务。
    reader_task: Option<tokio::task::JoinHandle<()>>,
    /// 下一个请求 ID。
    next_id: AtomicU64,
    /// 状态与请求准入共享同一个原子失败边界。
    runtime: Arc<StdMutex<LspRuntime>>,
    /// 按文件 URI 缓存的诊断信息。
    diagnostics_cache: Arc<Mutex<HashMap<String, Vec<Diagnostic>>>>,
    /// 创建该派生句柄时捕获的管理器生命周期。
    lifecycle: Arc<LspLifecycle>,
    /// 与该客户端句柄关联的代次。
    lifecycle_generation: u64,
    /// 客户端级关闭栅栏，由显式 stop/shutdown 设置。
    closed: Arc<AtomicBool>,
    /// 在优雅关闭期间抑制预期内的 EOF/错误上报。
    stopping: Arc<AtomicBool>,
}

impl StdioLspClient {
    /// 创建新客户端（此时尚未启动服务器）。
    pub fn new(config: LspServerConfig) -> Self {
        let lifecycle = Arc::new(LspLifecycle::new(0));
        Self::new_bound(config, lifecycle, 0)
    }

    /// 创建绑定到管理器生命周期的客户端。
    pub(crate) fn new_bound(
        config: LspServerConfig,
        lifecycle: Arc<LspLifecycle>,
        lifecycle_generation: u64,
    ) -> Self {
        let language = config.language.clone();
        Self {
            language: language.clone(),
            config,
            child: None,
            writer_tx: None,
            writer_task: None,
            reader_task: None,
            next_id: AtomicU64::new(1),
            runtime: Arc::new(StdMutex::new(LspRuntime {
                status: LspServerStatus {
                    language: language.clone(),
                    running: false,
                    initialized: false,
                    restart_count: 0,
                    last_error: None,
                    pid: None,
                },
                pending: HashMap::new(),
                accepting: false,
            })),
            diagnostics_cache: Arc::new(Mutex::new(HashMap::new())),
            lifecycle,
            lifecycle_generation,
            closed: Arc::new(AtomicBool::new(false)),
            stopping: Arc::new(AtomicBool::new(false)),
        }
    }

    /// 确认客户端与所属生命周期仍然有效，否则返回未初始化错误。
    fn ensure_live(&self) -> LspResult<()> {
        if self.closed.load(Ordering::SeqCst) || !self.lifecycle.is_live(self.lifecycle_generation)
        {
            return Err(LspError::NotInitialized);
        }
        Ok(())
    }

    /// 在运行时锁内修改服务器状态。
    fn update_runtime(&self, update: impl FnOnce(&mut LspServerStatus)) {
        let mut runtime = self
            .runtime
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        update(&mut runtime.status);
    }

    /// 设置重启次数。
    pub(crate) fn set_restart_count(&self, count: u32) {
        self.update_runtime(|status| status.restart_count = count);
    }

    /// 记录最近一次错误。
    pub(crate) fn set_last_error(&self, error: String) {
        self.update_runtime(|status| status.last_error = Some(error));
    }

    /// 仅当服务器仍在运行且接受请求时标记为已初始化，返回是否成功。
    fn mark_initialized_if_running(&self) -> bool {
        let mut runtime = self
            .runtime
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        if !runtime.status.running || !runtime.accepting {
            return false;
        }
        runtime.status.initialized = true;
        true
    }

    /// 不再发送任何协议消息，直接终止进程。
    async fn abort_process(&mut self) {
        self.writer_tx = None;
        if let Some(child) = self.child.take() {
            Self::terminate_child(&child).await;
        }
        for task in [self.writer_task.take(), self.reader_task.take()]
            .into_iter()
            .flatten()
        {
            let mut task = task;
            if tokio::time::timeout(std::time::Duration::from_secs(2), &mut task)
                .await
                .is_err()
            {
                task.abort();
                let _ = task.await;
            }
        }
        self.diagnostics_cache.lock().await.clear();
        self.runtime
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .settle(None);
    }

    /// 强制结束子进程并等待其退出。
    async fn terminate_child(child: &Arc<Mutex<Child>>) {
        let mut child = child.lock().await;
        let _ = child.start_kill();
        let _ = child.wait().await;
    }

    /// 测试辅助：判断子进程是否已退出。
    #[cfg(test)]
    pub(crate) async fn process_has_exited(&self) -> bool {
        let Some(child) = self.child.as_ref() else {
            return true;
        };
        child.lock().await.try_wait().ok().flatten().is_some()
    }

    /// 启动服务器进程并建立通信通道。
    fn spawn_process(&mut self) -> Result<(), LspError> {
        let mut cmd = Command::new(&self.config.command);
        cmd.args(&self.config.args);

        // 设置环境变量
        for (key, value) in &self.config.env {
            cmd.env(key, value);
        }

        cmd.stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true);

        let mut child = cmd
            .spawn()
            .map_err(|e| LspError::SpawnError(format!("{}: {e}", self.config.command)))?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| LspError::SpawnError("Failed to capture stdin".into()))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| LspError::SpawnError("Failed to capture stdout".into()))?;

        // 创建写入通道
        let (writer_tx, mut writer_rx) = tokio::sync::mpsc::channel::<Vec<u8>>(64);

        let child_id = child.id();
        let child = Arc::new(Mutex::new(child));
        {
            let mut runtime = self
                .runtime
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            runtime.status.running = true;
            runtime.status.initialized = false;
            runtime.status.last_error = None;
            runtime.status.pid = child_id;
            runtime.accepting = true;
        }

        /*
         * 写入失败时必须终结所有待响应调用，
         * 即使子进程在 stdin 关闭后仍无限期保持 stdout 打开。
         */
        let writer_runtime = Arc::clone(&self.runtime);
        let writer_cache = Arc::clone(&self.diagnostics_cache);
        let writer_child = Arc::clone(&child);
        let writer_stopping = Arc::clone(&self.stopping);
        let writer_task = tokio::spawn(async move {
            let mut stdin = stdin;
            while let Some(data) = writer_rx.recv().await {
                let result = match stdin.write_all(&data).await {
                    Ok(()) => stdin.flush().await,
                    Err(error) => Err(error),
                };
                if let Err(error) = result {
                    let detail = format!("Language server stdin write failed: {error}");
                    writer_runtime
                        .lock()
                        .unwrap_or_else(|poison| poison.into_inner())
                        .begin_terminal(
                            (!writer_stopping.load(Ordering::SeqCst)).then_some(detail),
                        );
                    writer_cache.lock().await.clear();
                    Self::terminate_child(&writer_child).await;
                    writer_runtime
                        .lock()
                        .unwrap_or_else(|poison| poison.into_inner())
                        .settle(None);
                    break;
                }
            }
        });

        // 启动读取任务
        let diagnostics_cache = self.diagnostics_cache.clone();
        let runtime = Arc::clone(&self.runtime);
        let closed = Arc::clone(&self.closed);
        let stopping = Arc::clone(&self.stopping);
        let reader_child = Arc::clone(&child);
        let reader_task = tokio::spawn(async move {
            let reader = BufReader::new(stdout);
            let error =
                Self::read_loop(reader, Arc::clone(&runtime), diagnostics_cache.clone()).await;
            runtime
                .lock()
                .unwrap_or_else(|poison| poison.into_inner())
                .begin_terminal(
                    (!closed.load(Ordering::SeqCst) && !stopping.load(Ordering::SeqCst))
                        .then_some(error),
                );
            diagnostics_cache.lock().await.clear();
            Self::terminate_child(&reader_child).await;
            runtime
                .lock()
                .unwrap_or_else(|poison| poison.into_inner())
                .settle(None);
        });

        self.child = Some(child);
        self.writer_tx = Some(writer_tx);
        self.writer_task = Some(writer_task);
        self.reader_task = Some(reader_task);
        Ok(())
    }

    /// 读取循环——从 stdout 解析带帧头的 LSP 消息。
    async fn read_loop(
        mut reader: BufReader<tokio::process::ChildStdout>,
        runtime: Arc<StdMutex<LspRuntime>>,
        diagnostics_cache: Arc<Mutex<HashMap<String, Vec<Diagnostic>>>>,
    ) -> String {
        let mut header_line = String::new();

        loop {
            // 读取头部直到遇到空行
            let mut content_length: Option<usize> = None;
            loop {
                header_line.clear();
                match reader.read_line(&mut header_line).await {
                    Ok(0) => return "Language server stdout closed".to_string(),
                    Ok(_) => {}
                    Err(error) => return format!("Language server stdout read failed: {error}"),
                }

                let trimmed = header_line.trim();
                if trimmed.is_empty() {
                    break; // 头部结束
                }

                if let Some((name, _)) = trimmed.split_once(':') {
                    if name.eq_ignore_ascii_case("content-length") {
                        let Some(length) = jsonrpc::parse_content_length(trimmed) else {
                            return "Invalid LSP Content-Length header".to_string();
                        };
                        content_length = Some(length);
                    }
                } else {
                    return "Malformed LSP header".to_string();
                }
            }

            let Some(len) = content_length else {
                return "Missing LSP Content-Length header".to_string();
            };
            if !accepts_lsp_message_size(len) {
                return format!("Language server message exceeds {MAX_LSP_MESSAGE_BYTES} bytes");
            }

            // 读取消息体
            let mut body = vec![0u8; len];
            if let Err(error) = reader.read_exact(&mut body).await {
                return format!("Language server message body truncated: {error}");
            }

            // 解析 JSON
            let Ok(value) = serde_json::from_slice::<serde_json::Value>(&body) else {
                continue;
            };

            // 判断是通知（无 id）还是响应（有 id）
            if let Some(id) = value.get("id").and_then(|v| v.as_u64()) {
                // 对某个请求的响应
                if let Ok(resp) = serde_json::from_value::<JsonRpcResponse>(value) {
                    let mut runtime = runtime.lock().unwrap_or_else(|poison| poison.into_inner());
                    if let Some(tx) = runtime.pending.remove(&id) {
                        let _ = tx.send(resp);
                    }
                }
            } else if let Some(method) = value.get("method").and_then(|v| v.as_str()) {
                // 服务器通知
                if method == "textDocument/publishDiagnostics"
                    && let Some(params) = value.get("params")
                    && let Some(uri) = params.get("uri").and_then(|v| v.as_str())
                {
                    let diagnostics: Vec<Diagnostic> = params
                        .get("diagnostics")
                        .and_then(|v| serde_json::from_value(v.clone()).ok())
                        .unwrap_or_default();
                    if runtime
                        .lock()
                        .unwrap_or_else(|poison| poison.into_inner())
                        .status
                        .running
                    {
                        diagnostics_cache
                            .lock()
                            .await
                            .insert(uri.to_string(), diagnostics);
                    }
                }
            }
        }
    }

    /// 发送 JSON-RPC 请求并等待响应。
    async fn send_request(
        &self,
        method: &str,
        params: Option<serde_json::Value>,
    ) -> LspResult<serde_json::Value> {
        self.ensure_live()?;
        if !self.is_running() {
            return Err(LspError::NotInitialized);
        }
        let writer_tx = self.writer_tx.as_ref().ok_or(LspError::NotInitialized)?;

        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let request = JsonRpcRequest::new(id, method, params);
        let data = jsonrpc::encode_message(&request)
            .map_err(|e| LspError::CommunicationError(e.to_string()))?;

        let (tx, rx) = oneshot::channel();
        {
            self.runtime
                .lock()
                .unwrap_or_else(|poison| poison.into_inner())
                .register(id, tx)?;
        }

        if writer_tx.send(data).await.is_err() {
            self.runtime
                .lock()
                .unwrap_or_else(|poison| poison.into_inner())
                .pending
                .remove(&id);
            return Err(LspError::CommunicationError("Writer channel closed".into()));
        }

        let response = match tokio::time::timeout(REQUEST_TIMEOUT, rx).await {
            Ok(Ok(response)) => response,
            Ok(Err(_)) => {
                self.runtime
                    .lock()
                    .unwrap_or_else(|poison| poison.into_inner())
                    .pending
                    .remove(&id);
                return Err(LspError::CommunicationError(
                    "Response channel closed".into(),
                ));
            }
            Err(_) => {
                self.runtime
                    .lock()
                    .unwrap_or_else(|poison| poison.into_inner())
                    .pending
                    .remove(&id);
                let cancellation = JsonRpcNotification::new(
                    "$/cancelRequest",
                    Some(serde_json::json!({ "id": id })),
                );
                if let Ok(data) = jsonrpc::encode_message(&cancellation) {
                    let _ = writer_tx.send(data).await;
                }
                return Err(LspError::CommunicationError(format!(
                    "LSP request '{method}' timed out after {}s",
                    REQUEST_TIMEOUT.as_secs()
                )));
            }
        };

        self.ensure_live()?;
        if !self.is_running() {
            return Err(LspError::NotInitialized);
        }

        if let Some(err) = response.error {
            return Err(LspError::ServerError(err.to_string()));
        }

        response
            .result
            .ok_or_else(|| LspError::ServerError("Empty response".into()))
    }

    /// 发送 JSON-RPC 通知（无需响应）。
    async fn send_notification(
        &self,
        method: &str,
        params: Option<serde_json::Value>,
    ) -> LspResult<()> {
        self.ensure_live()?;
        if !self.is_running() {
            return Err(LspError::NotInitialized);
        }
        let writer_tx = self.writer_tx.as_ref().ok_or(LspError::NotInitialized)?;

        let notification = JsonRpcNotification::new(method, params);
        let data = jsonrpc::encode_message(&notification)
            .map_err(|e| LspError::CommunicationError(e.to_string()))?;

        writer_tx
            .send(data)
            .await
            .map_err(|_| LspError::CommunicationError("Writer channel closed".into()))?;

        Ok(())
    }
}

impl LspClient for StdioLspClient {
    /// 返回语言标识。
    fn language(&self) -> &str {
        &self.language
    }

    /// 服务器是否正在运行。
    fn is_running(&self) -> bool {
        self.status().running
    }

    /// 服务器是否已完成初始化握手。
    fn is_initialized(&self) -> bool {
        self.status().initialized
    }

    /// 启动服务器进程并完成 initialize/initialized 握手。
    fn initialize<'a>(&'a mut self, root_uri: &'a str) -> BoxFuture<'a, LspResult<()>> {
        Box::pin(async move {
            self.ensure_live()?;
            if self.child.is_some() {
                return Err(LspError::CommunicationError(
                    "Language server is already started".to_string(),
                ));
            }
            // 启动进程
            if let Err(error) = self.spawn_process() {
                self.set_last_error(error.to_string());
                return Err(error);
            }

            /*
             * 关闭操作可能在启动进程期间或子进程启动过程中抢先完成。
             * 此时应中止刚创建的子进程，避免无主进程在创建它的管理器之后继续存活。
             */
            if let Err(error) = self.ensure_live() {
                self.abort_process().await;
                return Err(error);
            }

            // 发送 initialize 请求
            let params = serde_json::json!({
                "processId": std::process::id(),
                "rootUri": root_uri,
                "capabilities": {
                    "textDocument": {
                        "completion": {
                            "completionItem": {
                                "snippetSupport": false
                            }
                        },
                        "hover": {
                            "contentFormat": ["markdown", "plaintext"]
                        },
                        "definition": {},
                        "references": {},
                        "publishDiagnostics": {
                            "relatedInformation": true
                        }
                    }
                }
            });

            if let Err(error) = self.send_request("initialize", Some(params)).await {
                self.set_last_error(error.to_string());
                self.abort_process().await;
                return Err(error);
            }

            // 发送 initialized 通知
            if let Err(error) = self
                .send_notification("initialized", Some(serde_json::json!({})))
                .await
            {
                self.set_last_error(error.to_string());
                self.abort_process().await;
                return Err(error);
            }

            if let Err(error) = self.ensure_live() {
                self.abort_process().await;
                return Err(error);
            }

            if !self.mark_initialized_if_running() {
                self.abort_process().await;
                return Err(LspError::NotInitialized);
            }
            Ok(())
        })
    }

    /// 发送 shutdown/exit 并终止服务器进程。
    fn shutdown(&mut self) -> BoxFuture<'_, LspResult<()>> {
        Box::pin(async move {
            self.stopping.store(true, Ordering::SeqCst);
            if self.is_running() {
                // 发送 shutdown 请求
                let _ = tokio::time::timeout(
                    std::time::Duration::from_secs(2),
                    self.send_request("shutdown", None),
                )
                .await;
                // 发送 exit 通知
                let _ = tokio::time::timeout(
                    std::time::Duration::from_secs(2),
                    self.send_notification("exit", None),
                )
                .await;
            }
            self.closed.store(true, Ordering::SeqCst);
            self.abort_process().await;
            Ok(())
        })
    }

    /// 返回指定文件 URI 的缓存诊断信息。
    fn diagnostics<'a>(&'a self, uri: &'a str) -> BoxFuture<'a, LspResult<Vec<Diagnostic>>> {
        Box::pin(async move {
            self.ensure_live()?;
            if !self.is_initialized() {
                return Err(LspError::NotInitialized);
            }
            let cache = self.diagnostics_cache.lock().await;
            Ok(cache.get(uri).cloned().unwrap_or_default())
        })
    }

    /// 跳转到定义。
    fn goto_definition<'a>(
        &'a self,
        uri: &'a str,
        position: Position,
    ) -> BoxFuture<'a, LspResult<Vec<Location>>> {
        Box::pin(async move {
            let params = serde_json::json!({
                "textDocument": { "uri": uri },
                "position": { "line": position.line, "character": position.character }
            });

            let result = self
                .send_request("textDocument/definition", Some(params))
                .await?;

            // 结果可能是单个 Location 或数组
            let locations: Vec<Location> =
                if let Ok(loc) = serde_json::from_value::<Location>(result.clone()) {
                    vec![loc]
                } else {
                    serde_json::from_value(result).unwrap_or_default()
                };

            Ok(locations)
        })
    }

    /// 查找引用（包含声明本身）。
    fn find_references<'a>(
        &'a self,
        uri: &'a str,
        position: Position,
    ) -> BoxFuture<'a, LspResult<Vec<Location>>> {
        Box::pin(async move {
            let params = serde_json::json!({
                "textDocument": { "uri": uri },
                "position": { "line": position.line, "character": position.character },
                "context": { "includeDeclaration": true }
            });

            let result = self
                .send_request("textDocument/references", Some(params))
                .await?;

            let locations: Vec<Location> = serde_json::from_value(result).unwrap_or_default();
            Ok(locations)
        })
    }

    /// 获取指定位置的悬停信息。
    fn hover<'a>(
        &'a self,
        uri: &'a str,
        position: Position,
    ) -> BoxFuture<'a, LspResult<Option<HoverInfo>>> {
        Box::pin(async move {
            let params = serde_json::json!({
                "textDocument": { "uri": uri },
                "position": { "line": position.line, "character": position.character }
            });

            let result = self
                .send_request("textDocument/hover", Some(params))
                .await?;

            if result.is_null() {
                return Ok(None);
            }

            // 从 hover 结果中提取内容
            let contents = result
                .get("contents")
                .map(|v| {
                    if let Some(s) = v.as_str() {
                        s.to_string()
                    } else if let Some(obj) = v.as_object() {
                        obj.get("value")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string()
                    } else {
                        v.to_string()
                    }
                })
                .unwrap_or_default();

            Ok(Some(HoverInfo {
                contents,
                range: None,
            }))
        })
    }

    /// 获取指定位置的补全项。
    fn completion<'a>(
        &'a self,
        uri: &'a str,
        position: Position,
    ) -> BoxFuture<'a, LspResult<Vec<CompletionItem>>> {
        Box::pin(async move {
            let params = serde_json::json!({
                "textDocument": { "uri": uri },
                "position": { "line": position.line, "character": position.character }
            });

            let result = self
                .send_request("textDocument/completion", Some(params))
                .await?;

            // 结果可能是 CompletionItem[] 或 CompletionList
            let items: Vec<CompletionItem> = if let Some(items) = result.get("items") {
                serde_json::from_value(items.clone()).unwrap_or_default()
            } else {
                serde_json::from_value(result).unwrap_or_default()
            };

            Ok(items)
        })
    }

    /// 通知服务器文件已打开。
    fn did_open<'a>(
        &'a self,
        uri: &'a str,
        language_id: &'a str,
        text: &'a str,
    ) -> BoxFuture<'a, LspResult<()>> {
        Box::pin(async move {
            let params = serde_json::json!({
                "textDocument": {
                    "uri": uri,
                    "languageId": language_id,
                    "version": 1,
                    "text": text
                }
            });
            self.send_notification("textDocument/didOpen", Some(params))
                .await
        })
    }

    /// 通知服务器文件内容发生变更。
    fn did_change<'a>(
        &'a self,
        uri: &'a str,
        changes: Vec<TextChange>,
    ) -> BoxFuture<'a, LspResult<()>> {
        Box::pin(async move {
            let content_changes: Vec<serde_json::Value> = changes
                .into_iter()
                .map(|c| {
                    serde_json::json!({
                        "range": {
                            "start": { "line": c.range.start.line, "character": c.range.start.character },
                            "end": { "line": c.range.end.line, "character": c.range.end.character }
                        },
                        "text": c.text
                    })
                })
                .collect();

            let params = serde_json::json!({
                "textDocument": { "uri": uri, "version": 2 },
                "contentChanges": content_changes
            });
            self.send_notification("textDocument/didChange", Some(params))
                .await
        })
    }

    /// 通知服务器文件已保存。
    fn did_save<'a>(&'a self, uri: &'a str) -> BoxFuture<'a, LspResult<()>> {
        Box::pin(async move {
            let params = serde_json::json!({
                "textDocument": { "uri": uri }
            });
            self.send_notification("textDocument/didSave", Some(params))
                .await
        })
    }

    /// 通知服务器文件已关闭。
    fn did_close<'a>(&'a self, uri: &'a str) -> BoxFuture<'a, LspResult<()>> {
        Box::pin(async move {
            let params = serde_json::json!({
                "textDocument": { "uri": uri }
            });
            self.send_notification("textDocument/didClose", Some(params))
                .await
        })
    }

    /// 返回当前服务器状态快照。
    fn status(&self) -> LspServerStatus {
        self.runtime
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .status
            .clone()
    }
}

#[cfg(test)]
mod message_size_tests {
    use super::*;

    /// 拒绝超过上限的 Content-Length。
    #[test]
    fn rejects_unbounded_content_length() {
        assert!(accepts_lsp_message_size(MAX_LSP_MESSAGE_BYTES));
        assert!(!accepts_lsp_message_size(
            MAX_LSP_MESSAGE_BYTES.saturating_add(1)
        ));
    }

    /// EOF 终结与请求准入共享同一边界：终结后不再接纳新请求。
    #[test]
    fn eof_settlement_and_request_admission_share_one_boundary() -> Result<(), String> {
        let mut runtime = LspRuntime {
            status: LspServerStatus {
                language: "test".to_string(),
                running: true,
                initialized: true,
                restart_count: 0,
                last_error: None,
                pid: Some(42),
            },
            pending: HashMap::new(),
            accepting: true,
        };
        let (before_tx, before_rx) = oneshot::channel();
        runtime
            .register(1, before_tx)
            .map_err(|error| error.to_string())?;
        runtime.settle(Some("EOF".to_string()));
        if before_rx.blocking_recv().is_ok() {
            return Err("settlement did not close the admitted request".to_string());
        }
        let (after_tx, _after_rx) = oneshot::channel();
        if runtime.register(2, after_tx).is_ok() || !runtime.pending.is_empty() {
            return Err("request was admitted after EOF settlement".to_string());
        }
        Ok(())
    }
}
