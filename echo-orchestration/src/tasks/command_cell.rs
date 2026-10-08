/*!
后台命令 cell（Codex 风格的 `shell(background=true)` + `wait`）。

[`BackgroundCommandManager`] 实现了
[`echo_core::tools::cell::CommandCellRegistry`] 契约：长时间运行的命令
（小时级的构建/测试）以 *cell* 形式启动，并立即返回 `cell_id`；调用方通过
`wait(cell_id, cursor, yield_ms)` 长轮询获取增量输出与终态。

# 为什么不复用 `BackgroundTask::wait`？

`BackgroundTask` 的结果槽只能被 `take()` 一次——只有单个消费者能拿到结果。
cell 必须支持**多个等待者**（主 agent + 保留的 watcher），且每个等待者都能
重复读取同一终态，因此 cell 维护自己的可重复读取状态
（`RwLock<CellState>` + 尾部保留的输出缓冲区 + `Notify::notify_waiters` 广播唤醒）。
*/

use dashmap::DashMap;
use echo_core::sandbox::{
    SandboxCommand, SandboxExecutor, SandboxOutputChannel, SandboxStreamEvent,
};
use echo_core::tools::ToolOutputChannel;
use echo_core::tools::artifact::{ToolOutputArtifactRef, ToolOutputArtifactWriter};
use echo_core::tools::cell::{
    CommandCellArtifactStatus, CommandCellDelta, CommandCellError, CommandCellLaunchReceipt,
    CommandCellObservationLease, CommandCellPhase, CommandCellRegistry, CommandCellRequest,
    CommandCellSnapshot, CommandCellTerminalCause, CommandCellWaitReason,
};
use echo_core::utils::utf8::IncrementalUtf8Decoder;
use futures::StreamExt;
use futures::future::BoxFuture;
use std::future::Future;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::io::AsyncReadExt;
use tokio::sync::{Mutex as AsyncMutex, Notify, OwnedSemaphorePermit, RwLock, Semaphore, oneshot};
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;
use tokio_util::task::TaskTracker;

/// cell 显示名称的 UTF-8 安全预览长度（按字符计）。
const NAME_PREVIEW_CHARS: usize = 80;
/**
每轮 wait 中 `new_output` 的字节上限。

游标同样按字节计，因此按实际返回的原始字节数推进游标，
可保证每个被保留的字节都能在多次 wait 中被完整读出。
*/
const MAX_DELTA_BYTES: usize = 16 * 1024;
/// 读取任务的单块大小（字节）。
const READ_CHUNK_BYTES: usize = 16 * 1024;
/// 显式取消/关闭后允许的最长清理宽限时间。
const CANCEL_DRAIN_GRACE: Duration = Duration::from_secs(5);

/// 确定性命令 cell 监视的取消行为。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommandCellWatchCancellation {
    /// 一旦取消胜出，立即返回 [`CommandCellError::Cancelled`]。
    Return,
    /**
    记录取消，但继续短轮询，直到 cell 发布其真实终态且所有可观测输出都已读完。
    */
    DrainToTerminal {
        /// 取消后每轮短轮询的 yield 时长（毫秒）。
        yield_ms: u64,
    },
}

/// [`CommandCellWatcher`] 的策略配置。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommandCellWatchConfig {
    /// 每轮 wait 的 yield 时长（毫秒），必须为正。
    pub yield_ms: u64,
    /// 输出摘录保留的尾部最大字符数。
    pub excerpt_chars: usize,
    /// 监视过程中遇到取消时的行为。
    pub cancellation: CommandCellWatchCancellation,
}

impl Default for CommandCellWatchConfig {
    fn default() -> Self {
        Self {
            yield_ms: 30_000,
            excerpt_chars: 1_000,
            cancellation: CommandCellWatchCancellation::Return,
        }
    }
}

/// 确定性监视产出的 cell 终态事实。
#[derive(Debug, Clone)]
pub struct CommandCellTerminalObservation {
    /// cell 的终态快照。
    pub snapshot: CommandCellSnapshot,
    /// 输出尾部摘录（受 `excerpt_chars` 限制）。
    pub output_excerpt: String,
    /// 读完全部输出后的下一个游标位置。
    pub next_cursor: u64,
    /// 监视期间是否有输出被省略（丢弃或截断）。
    pub output_elided: bool,
    /// 监视期间是否观测到取消。
    pub cancellation_observed: bool,
}

/**
针对单个命令 cell 的保留式确定性观察者。

watcher 在整个生命周期内持有框架的观察租约，并将注册表的类型化快照与游标
视为唯一的终态权威来源。此生命周期中不涉及任何模型调用。
*/
pub struct CommandCellWatcher {
    /// 被观察的 cell 注册表。
    registry: Arc<dyn CommandCellRegistry>,
    /// 被观察的 cell ID。
    cell_id: String,
    /// 监视策略。
    config: CommandCellWatchConfig,
    /// 观察租约，保证 cell 在监视期间不会被保留策略清理。
    _observation: CommandCellObservationLease,
}

impl std::fmt::Debug for CommandCellWatcher {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CommandCellWatcher")
            .field("cell_id", &self.cell_id)
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl CommandCellWatcher {
    /// 在读取任何应用层投影之前，先获取一个保留式监视。
    pub fn acquire(
        registry: Arc<dyn CommandCellRegistry>,
        cell_id: impl Into<String>,
        config: CommandCellWatchConfig,
    ) -> std::result::Result<Self, CommandCellError> {
        let cell_id = cell_id.into();
        if cell_id.trim().is_empty() {
            return Err(CommandCellError::Validation {
                message: "command-cell watcher requires a non-empty cell_id".to_string(),
            });
        }
        if config.yield_ms == 0
            || matches!(
                config.cancellation,
                CommandCellWatchCancellation::DrainToTerminal { yield_ms: 0 }
            )
        {
            return Err(CommandCellError::Validation {
                message: "command-cell watcher yield budgets must be positive".to_string(),
            });
        }
        let observation = registry.observe(&cell_id)?;
        Ok(Self {
            registry,
            cell_id,
            config,
            _observation: observation,
        })
    }

    /// 返回被观察的 cell ID。
    pub fn cell_id(&self) -> &str {
        &self.cell_id
    }

    /// 等待 cell 进入终态，且所有可用字节都已通过注册表游标读完。
    pub async fn wait_terminal(
        self,
        cancel: &CancellationToken,
    ) -> std::result::Result<CommandCellTerminalObservation, CommandCellError> {
        let mut cursor = 0_u64;
        let mut excerpt = String::new();
        let mut output_elided = false;
        let mut cancellation_observed = false;
        loop {
            let yield_ms = match (cancellation_observed, self.config.cancellation) {
                (true, CommandCellWatchCancellation::DrainToTerminal { yield_ms }) => yield_ms,
                _ => self.config.yield_ms,
            };
            let delta = if cancellation_observed {
                self.registry.wait(&self.cell_id, cursor, yield_ms).await?
            } else {
                tokio::select! {
                    biased;
                    _ = cancel.cancelled() => {
                        match self.config.cancellation {
                            CommandCellWatchCancellation::Return => {
                                return Err(CommandCellError::Cancelled);
                            }
                            CommandCellWatchCancellation::DrainToTerminal { .. } => {
                                cancellation_observed = true;
                                continue;
                            }
                        }
                    }
                    delta = self.registry.wait(&self.cell_id, cursor, yield_ms) => delta?,
                }
            };
            append_excerpt_tail(&mut excerpt, &delta.new_output, self.config.excerpt_chars);
            output_elided |= delta.output_elided;
            cursor = delta.next_cursor;
            if !delta.snapshot.phase.is_terminal() || cursor < delta.snapshot.total_output_bytes {
                continue;
            }
            return Ok(CommandCellTerminalObservation {
                snapshot: delta.snapshot,
                output_excerpt: excerpt,
                next_cursor: cursor,
                output_elided,
                cancellation_observed,
            });
        }
    }
}

/// 将新输出追加到摘录，并只保留最后 `max_chars` 个字符。
fn append_excerpt_tail(target: &mut String, chunk: &str, max_chars: usize) {
    if max_chars == 0 {
        target.clear();
        return;
    }
    target.push_str(chunk);
    let count = target.chars().count();
    if count <= max_chars {
        return;
    }
    *target = target
        .chars()
        .skip(count.saturating_sub(max_chars))
        .collect();
}

// ── 配置 ──────────────────────────────────────────────────────────

/// [`BackgroundCommandManager`] 的配置。
#[derive(Debug, Clone)]
pub struct BackgroundCommandManagerConfig {
    /// 同时运行的 cell 最大数量。
    pub max_concurrent: usize,
    /// 默认 cell 生命周期（秒），用于 `timeout_secs: None` 的情况。
    pub default_timeout_secs: u64,
    /// 任意 cell 生命周期的上限（秒）。
    pub max_timeout_secs: u64,
    /// 每个 cell 在内存中保留的输出量（保留尾部字节）。
    pub max_retained_output_bytes: usize,
    /**
    为 `list`/`wait` 保留的终态 cell 最大数量。
    运行中的 cell 永远不会被保留策略移除。
    */
    pub max_terminal_history: usize,
}

impl Default for BackgroundCommandManagerConfig {
    fn default() -> Self {
        Self {
            max_concurrent: 4,
            default_timeout_secs: 3600,
            max_timeout_secs: 86400,
            max_retained_output_bytes: 1024 * 1024,
            max_terminal_history: 256,
        }
    }
}

impl BackgroundCommandManagerConfig {
    /// 拒绝永远无法接纳任何命令 cell 的配置。
    pub fn validate(&self) -> std::result::Result<(), String> {
        if self.max_concurrent == 0 {
            return Err("background command max_concurrent must be greater than zero".to_string());
        }
        if self.default_timeout_secs == 0 || self.max_timeout_secs == 0 {
            return Err("background command timeouts must be greater than zero".to_string());
        }
        self.max_concurrent
            .checked_add(self.max_terminal_history)
            .ok_or_else(|| "background command tracked capacity overflow".to_string())?;
        Ok(())
    }

    /// 总跟踪容量：并发上限 + 终态历史上限（饱和相加）。
    fn tracked_capacity(&self) -> usize {
        self.max_concurrent
            .saturating_add(self.max_terminal_history)
    }
}

// ── 输出缓冲区 ───────────────────────────────────────────────────

/**
尾部保留的输出缓冲区。

`total_bytes` 统计曾写入的所有字节（单调递增的游标空间）；
`retained` 最多保留 `max_retained_output_bytes` 个尾部字节——从头部丢弃的字节
仍计入 `total_bytes`，并通过 `truncated` 标记。
*/
#[derive(Default)]
struct OutputBuffer {
    /// 当前保留的尾部字节。
    retained: Vec<u8>,
    /// 累计写入的总字节数。
    total_bytes: u64,
    /// 是否曾从头部丢弃过字节。
    truncated: bool,
}

impl OutputBuffer {
    /// 追加字节，只保留不超过 `max_retained` 的尾部。
    fn push(&mut self, bytes: &[u8], max_retained: usize) {
        self.total_bytes = self
            .total_bytes
            .saturating_add(u64::try_from(bytes.len()).unwrap_or(u64::MAX));
        self.retained.extend_from_slice(bytes);
        let overflow = self.retained.len().saturating_sub(max_retained);
        if overflow > 0 {
            // 只保留尾部: 丢弃头部 overflow 字节(字节级运算, &mut Vec 不会 panic)
            self.retained.drain(..overflow);
            self.truncated = true;
        }
    }
}

// ── Cell 句柄 ─────────────────────────────────────────────────────

/// 进程终态信息，各字段一起原子更新。
#[derive(Debug, Clone)]
struct CellState {
    /// 当前阶段。
    phase: CommandCellPhase,
    /// 进程退出码（若有）。
    exit_code: Option<i32>,
    /// 进入终态的原因。
    terminal_cause: Option<CommandCellTerminalCause>,
    /// 终态附带的说明信息。
    terminal_message: Option<String>,
}

/// cell 完整输出落盘（artifact）的状态。
struct CellArtifactState {
    /// artifact 写入器；完成或失败后置为 `None`。
    writer: Option<ToolOutputArtifactWriter>,
    /// stdout 的增量 UTF-8 解码器。
    stdout_decoder: IncrementalUtf8Decoder,
    /// stderr 的增量 UTF-8 解码器。
    stderr_decoder: IncrementalUtf8Decoder,
    /// 已完成 artifact 的引用。
    reference: Option<ToolOutputArtifactRef>,
    /// artifact 的类型化状态。
    status: CommandCellArtifactStatus,
    /// 状态附带的说明信息（如失败原因）。
    message: Option<String>,
}

/// 单个命令 cell 的共享状态。
pub struct CommandCellHandle {
    /// cell 唯一标识。
    cell_id: String,
    /// 显示名称（命令的截断预览）。
    name: String,
    /// 终态阶段 + 退出码（完成时只写一次）。
    state: RwLock<CellState>,
    /// 尾部保留的输出；使用 std Mutex（临界区很短，其中不 await）。
    output: Mutex<OutputBuffer>,
    /**
    完整输出落盘状态。锁只在处理单个块或收尾时持有，
    以保证状态/引用的变更是原子的。
    */
    artifact: Mutex<CellArtifactState>,
    /// 向等待者广播唤醒（追加输出 / 阶段定稿时）。
    notify: Notify,
    /// 子进程的终止开关。
    cancel: CancellationToken,
    /// 供历史清理同步读取的终态标记。
    terminal_flag: AtomicBool,
    /// 当前正在读取或快照此 cell 的活跃 wait 调用数。
    waiter_leases: AtomicU64,
    /// 跨多轮 wait 保留此 cell 的活跃观察者数。
    observation_leases: AtomicU64,
    /// 总跟踪容量许可，持有至条目最终被移除。
    _tracked_permit: Mutex<Option<OwnedSemaphorePermit>>,
    /// 单调递增的注册顺序，用于有界的终态保留。
    sequence: u64,
    /// 测试专用：拦截 artifact 收尾过程的钩子。
    #[cfg(test)]
    artifact_finalize_hook: Mutex<Option<Arc<TestArtifactFinalizeHook>>>,
}

/// 测试专用：用于阻塞 artifact 收尾的钩子。
#[cfg(test)]
struct TestArtifactFinalizeHook {
    /// 收尾流程进入钩子时通知。
    entered: Notify,
    /// 通知后放行收尾流程。
    release: Notify,
}

impl CommandCellHandle {
    /// cell 是否已进入终态（同步、无锁）。
    fn is_terminal(&self) -> bool {
        self.terminal_flag.load(Ordering::Acquire)
    }

    /// 当前状态（阶段 + 终态原因），克隆开销低。
    async fn current_state(&self) -> CellState {
        self.state.read().await.clone()
    }

    /// 非阻塞快照（状态 + 输出计数）。
    async fn snapshot(&self) -> CommandCellSnapshot {
        let state = self.current_state().await;
        let output = self
            .output
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let artifact = self
            .artifact
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        build_snapshot(&self.cell_id, &self.name, state, &output, &artifact)
    }
}

/// 根据状态、输出缓冲区与 artifact 状态构建 cell 快照。
fn build_snapshot(
    cell_id: &str,
    name: &str,
    state: CellState,
    output: &OutputBuffer,
    artifact: &CellArtifactState,
) -> CommandCellSnapshot {
    CommandCellSnapshot {
        cell_id: cell_id.to_string(),
        name: name.to_string(),
        phase: state.phase,
        exit_code: state.exit_code,
        terminal_cause: state.terminal_cause,
        terminal_message: state.terminal_message,
        total_output_bytes: output.total_bytes,
        output_truncated: output.truncated,
        artifact_status: artifact.status.clone(),
        artifact_message: artifact.message.clone(),
        output_artifact: artifact.reference.clone(),
    }
}

/**
在锁内读取 cell 的输出，并为 `cursor` 构建增量。
（让 `MutexGuard` 的作用域尽量小，便于调用方留在异步代码中。）
*/
async fn snapshot_delta(
    handle: &CommandCellHandle,
    cursor: u64,
    state: CellState,
    wait_reason: CommandCellWaitReason,
) -> CommandCellDelta {
    let output = handle
        .output
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let artifact = handle
        .artifact
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    build_delta(
        &handle.cell_id,
        &handle.name,
        cursor,
        state,
        wait_reason,
        &output,
        &artifact,
    )
}

/**
基于已加锁的输出视图构建一轮 wait 的增量。

所有切片都在 `&[u8]` 上按字节进行（通过 `get` 从不 panic）。合法的 UTF-8
输出只在字符边界处截断；任意进程字节仍可通过有损转换得到支持。
*/
fn build_delta(
    cell_id: &str,
    name: &str,
    cursor: u64,
    state: CellState,
    wait_reason: CommandCellWaitReason,
    output: &OutputBuffer,
    artifact: &CellArtifactState,
) -> CommandCellDelta {
    let total = output.total_bytes;
    // 非法(超前)cursor 一律收敛为 total, 返回空增量而不是报错。
    let effective_cursor = cursor.min(total);
    let retained_len = u64::try_from(output.retained.len()).unwrap_or(u64::MAX);
    let retained_start = total.saturating_sub(retained_len);

    let mut output_elided = false;
    let mut prefix = String::new();
    let (bytes, output_start): (&[u8], u64) = if effective_cursor < retained_start {
        /* cursor 早于保留窗口起点: 头部字节已被丢弃, 只能从 retained 开头取,
        并显式告知调用方该段增量之前有被丢弃的字节。 */
        output_elided = true;
        prefix = format!(
            "[... {} earlier output bytes were discarded (buffer retention cap) ...]\n",
            retained_start - effective_cursor
        );
        (output.retained.as_slice(), retained_start)
    } else {
        let offset =
            usize::try_from(effective_cursor - retained_start).unwrap_or(output.retained.len());
        (
            output.retained.get(offset..).unwrap_or(&[]),
            effective_cursor,
        )
    };

    let consumed = capped_delta_len(bytes);
    let returned = bytes.get(..consumed).unwrap_or_default();
    let text = String::from_utf8_lossy(returned).into_owned();
    if consumed < bytes.len() {
        output_elided = true;
    }
    let consumed_u64 = u64::try_from(consumed).unwrap_or(u64::MAX);
    let next_cursor = output_start.saturating_add(consumed_u64).min(total);

    CommandCellDelta {
        snapshot: build_snapshot(cell_id, name, state, output, artifact),
        wait_reason,
        new_output: format!("{prefix}{text}"),
        next_cursor,
        output_elided,
    }
}

/// 计算本轮增量可返回的字节数：不超过上限，并尽量落在 UTF-8 字符边界上。
fn capped_delta_len(bytes: &[u8]) -> usize {
    let candidate = bytes.len().min(MAX_DELTA_BYTES);
    let prefix = bytes.get(..candidate).unwrap_or_default();
    match std::str::from_utf8(prefix) {
        Ok(_) => candidate,
        Err(error) if error.error_len().is_none() && error.valid_up_to() > 0 => error.valid_up_to(),
        /* 非法的进程字节，或保留窗口起点位于字符中间时，
        仍必须推进字节游标。 */
        Err(_) => candidate,
    }
}

// ── 管理器 ─────────────────────────────────────────────────────────

/**
后台命令 cell 的注册表（参见模块文档）。

`launch` 必须在 tokio 运行时内调用。它可能会等待有界的跟踪容量；
`wait` 会在调用方的 yield 预算内返回。
*/
pub struct BackgroundCommandManager {
    /// 管理器实例标识，用于校验预留归属。
    manager_id: uuid::Uuid,
    /// 所有已发布的 cell（按 cell ID 索引）。
    cells: Arc<DashMap<String, Arc<CommandCellHandle>>>,
    /// 执行并发许可。
    execution: Arc<Semaphore>,
    /// 总跟踪容量许可（运行中 + 保留的终态 cell）。
    tracked: Arc<Semaphore>,
    /// 管理器配置。
    config: BackgroundCommandManagerConfig,
    /// 可选执行器，用于必须保持前台沙箱语义的启动。
    sandbox: Option<Arc<dyn SandboxExecutor>>,
    /// 单调递增的注册序号，用于确定性的终态保留。
    next_sequence: AtomicU64,
    /// 准入锁，串行化发布/启动与关闭。
    admission: AsyncMutex<()>,
    /// 关闭信号。
    shutdown: CancellationToken,
    /// 是否正在关闭。
    shutting_down: AtomicBool,
    /// 跟踪所有 cell 监管任务。
    tasks: TaskTracker,
}

impl Default for BackgroundCommandManager {
    fn default() -> Self {
        Self::from_validated_config(BackgroundCommandManagerConfig::default(), None)
    }
}

impl BackgroundCommandManager {
    /// 校验配置后创建一个直接在宿主机上执行命令的管理器。
    pub fn new(config: BackgroundCommandManagerConfig) -> std::result::Result<Self, String> {
        config.validate()?;
        Ok(Self::from_validated_config(config, None))
    }

    /// 基于已校验的配置构建管理器。
    fn from_validated_config(
        config: BackgroundCommandManagerConfig,
        sandbox: Option<Arc<dyn SandboxExecutor>>,
    ) -> Self {
        let execution = Arc::new(Semaphore::new(config.max_concurrent));
        let tracked = Arc::new(Semaphore::new(config.tracked_capacity()));
        Self {
            manager_id: uuid::Uuid::new_v4(),
            cells: Arc::new(DashMap::new()),
            execution,
            tracked,
            config,
            sandbox,
            next_sequence: AtomicU64::new(0),
            admission: AsyncMutex::new(()),
            shutdown: CancellationToken::new(),
            shutting_down: AtomicBool::new(false),
            tasks: TaskTracker::new(),
        }
    }

    /// 创建一个通过给定沙箱执行 cell 的管理器。
    pub fn new_with_sandbox(
        config: BackgroundCommandManagerConfig,
        sandbox: Arc<dyn SandboxExecutor>,
    ) -> std::result::Result<Self, String> {
        config.validate()?;
        Ok(Self::from_validated_config(config, Some(sandbox)))
    }

    /**
    最多保留 `max_terminal_history` 个终态 cell（优先移除最旧的）。
    运行中的 cell 永远不会被选中。与 `TaskSpawner::prune_terminal_history` 对应。
    */
    fn prune_terminal_history(&self) {
        prune_terminal_history(&self.cells, self.config.max_terminal_history);
    }

    /// 查找 cell 并获取一个 wait 租约，防止本轮 wait 期间被清理。
    fn acquire_waiter_lease(
        &self,
        cell_id: &str,
    ) -> std::result::Result<(Arc<CommandCellHandle>, CellWaiterLease), CommandCellError> {
        let entry = self
            .cells
            .get(cell_id)
            .ok_or_else(|| CommandCellError::NotFound {
                cell_id: cell_id.to_string(),
            })?;
        let handle = entry.value().clone();
        handle.waiter_leases.fetch_add(1, Ordering::AcqRel);
        drop(entry);
        let lease = CellWaiterLease {
            handle: handle.clone(),
            cells: self.cells.clone(),
            max_terminal_history: self.config.max_terminal_history,
        };
        Ok((handle, lease))
    }

    /// 校验并发布一个 cell，但不允许其运行器启动。
    pub async fn prepare_launch(
        &self,
        request: CommandCellRequest,
    ) -> std::result::Result<CommandCellReservation, CommandCellError> {
        self.validate_request(&request)?;
        let runtime =
            tokio::runtime::Handle::try_current().map_err(|error| CommandCellError::Runtime {
                message: format!("Tokio runtime is unavailable: {error}"),
            })?;
        if self.shutting_down.load(Ordering::Acquire) {
            return Err(CommandCellError::Shutdown);
        }
        self.prune_terminal_history();

        let timeout_secs = request
            .timeout_secs
            .unwrap_or(self.config.default_timeout_secs)
            .min(self.config.max_timeout_secs);
        if timeout_secs == 0 {
            return Err(CommandCellError::Validation {
                message: "timeout must be greater than zero".to_string(),
            });
        }
        let timeout = Duration::from_secs(timeout_secs);
        let accepted_at = chrono::Utc::now();
        let deadline_at = accepted_at
            .checked_add_signed(chrono::Duration::seconds(
                i64::try_from(timeout_secs).map_err(|error| CommandCellError::Validation {
                    message: format!("timeout conversion failed: {error}"),
                })?,
            ))
            .ok_or_else(|| CommandCellError::Validation {
                message: "deadline overflow".to_string(),
            })?;
        let deadline =
            Instant::now()
                .checked_add(timeout)
                .ok_or_else(|| CommandCellError::Validation {
                    message: "deadline overflow".to_string(),
                })?;

        let acquire = self.tracked.clone().acquire_owned();
        tokio::pin!(acquire);
        let tracked_permit = tokio::select! {
            biased;
            _ = self.shutdown.cancelled() => return Err(CommandCellError::Shutdown),
            _ = request_cancelled(request.cancel.as_ref()) => return Err(CommandCellError::Cancelled),
            _ = tokio::time::sleep_until(deadline) => return Err(CommandCellError::CapacityDeadline),
            result = &mut acquire => result.map_err(|_| CommandCellError::Shutdown)?,
        };

        let _admission = self.admission.lock().await;
        if self.shutting_down.load(Ordering::Acquire) {
            return Err(CommandCellError::Shutdown);
        }

        let cell_id = uuid::Uuid::new_v4().to_string();
        if self.cells.contains_key(&cell_id) {
            return Err(CommandCellError::DuplicateIdentity { cell_id });
        }
        let name = request
            .command
            .chars()
            .take(NAME_PREVIEW_CHARS)
            .collect::<String>();
        let handle = Arc::new(CommandCellHandle {
            cell_id: cell_id.clone(),
            name,
            state: RwLock::new(CellState {
                phase: CommandCellPhase::Prepared,
                exit_code: None,
                terminal_cause: None,
                terminal_message: None,
            }),
            output: Mutex::new(OutputBuffer::default()),
            artifact: Mutex::new(
                match (
                    request.output_artifacts.clone(),
                    request.artifact_identity.clone(),
                ) {
                    (Some(config), Some(identity)) => CellArtifactState {
                        writer: Some(ToolOutputArtifactWriter::new(config, identity)),
                        stdout_decoder: IncrementalUtf8Decoder::default(),
                        stderr_decoder: IncrementalUtf8Decoder::default(),
                        reference: None,
                        status: CommandCellArtifactStatus::Writing,
                        message: None,
                    },
                    _ => CellArtifactState {
                        writer: None,
                        stdout_decoder: IncrementalUtf8Decoder::default(),
                        stderr_decoder: IncrementalUtf8Decoder::default(),
                        reference: None,
                        status: CommandCellArtifactStatus::NotRequested,
                        message: None,
                    },
                },
            ),
            notify: Notify::new(),
            cancel: CancellationToken::new(),
            terminal_flag: AtomicBool::new(false),
            waiter_leases: AtomicU64::new(0),
            observation_leases: AtomicU64::new(0),
            _tracked_permit: Mutex::new(Some(tracked_permit)),
            sequence: self.next_sequence.fetch_add(1, Ordering::Relaxed),
            #[cfg(test)]
            artifact_finalize_hook: Mutex::new(None),
        });
        let receipt = CommandCellLaunchReceipt {
            cell_id: cell_id.clone(),
            accepted_at,
            deadline: deadline_at,
        };
        let run_spec = CellRunSpec {
            command: request.command,
            working_dir: request.working_dir.map(PathBuf::from),
            deadline,
            sandbox: self.sandbox.clone(),
            owner_cancel: request.cancel,
            max_retained: self.config.max_retained_output_bytes,
        };
        let (action_tx, action_rx) = oneshot::channel();
        self.cells.insert(cell_id.clone(), handle.clone());
        let execution = self.execution.clone();
        let cells = self.cells.clone();
        let terminal_history = self.config.max_terminal_history;
        let shutdown = self.shutdown.clone();
        drop(self.tasks.spawn_on(
            async move {
                supervise_prepared_cell(
                    handle,
                    execution,
                    run_spec,
                    action_rx,
                    shutdown,
                    cells,
                    terminal_history,
                )
                .await;
            },
            &runtime,
        ));

        Ok(CommandCellReservation {
            manager_id: self.manager_id,
            receipt,
            action: Some(action_tx),
        })
    }

    /// 打开已预备 cell 的执行闸门。
    pub async fn start_prepared(
        &self,
        mut reservation: CommandCellReservation,
    ) -> std::result::Result<CommandCellLaunchReceipt, CommandCellError> {
        if reservation.manager_id != self.manager_id {
            return Err(CommandCellError::Runtime {
                message: "prepared reservation belongs to another manager".to_string(),
            });
        }
        let _admission = self.admission.lock().await;
        if self.shutting_down.load(Ordering::Acquire) {
            return Err(CommandCellError::Shutdown);
        }
        let action = reservation
            .action
            .take()
            .ok_or_else(|| CommandCellError::Runtime {
                message: "prepared reservation was already consumed".to_string(),
            })?;
        let (started_tx, started_rx) = oneshot::channel();
        action
            .send(PreparedCellAction::Start(started_tx))
            .map_err(|_| CommandCellError::Runtime {
                message: "prepared cell settled before start".to_string(),
            })?;
        started_rx.await.map_err(|_| {
            if self.shutting_down.load(Ordering::Acquire) {
                CommandCellError::Shutdown
            } else {
                CommandCellError::Runtime {
                    message: "prepared cell settled before start acknowledgement".to_string(),
                }
            }
        })?;
        Ok(reservation.receipt.clone())
    }

    /// 中止一个已预备的 cell，不启动任何进程。
    pub fn abort_prepared(
        &self,
        mut reservation: CommandCellReservation,
        message: impl Into<String>,
    ) -> std::result::Result<CommandCellLaunchReceipt, CommandCellError> {
        if reservation.manager_id != self.manager_id {
            return Err(CommandCellError::Runtime {
                message: "prepared reservation belongs to another manager".to_string(),
            });
        }
        let action = reservation
            .action
            .take()
            .ok_or_else(|| CommandCellError::Runtime {
                message: "prepared reservation was already consumed".to_string(),
            })?;
        action
            .send(PreparedCellAction::Abort(message.into()))
            .map_err(|_| CommandCellError::Runtime {
                message: "prepared cell settled before abort".to_string(),
            })?;
        Ok(reservation.receipt.clone())
    }

    /// 校验启动请求的基本合法性。
    fn validate_request(
        &self,
        request: &CommandCellRequest,
    ) -> std::result::Result<(), CommandCellError> {
        if request.command.trim().is_empty() {
            return Err(CommandCellError::Validation {
                message: "command must not be empty".to_string(),
            });
        }
        if request.timeout_secs == Some(0) {
            return Err(CommandCellError::Validation {
                message: "timeout must be greater than zero".to_string(),
            });
        }
        if request.require_sandbox && self.sandbox.is_none() {
            return Err(CommandCellError::Validation {
                message: "background command requires a sandbox executor".to_string(),
            });
        }
        if request.output_artifacts.is_some() != request.artifact_identity.is_some() {
            return Err(CommandCellError::Validation {
                message: "artifact config and identity must be supplied together".to_string(),
            });
        }
        Ok(())
    }
}

/// 针对单个已校验、已发布命令 cell 的一次性闸门。
#[must_use]
pub struct CommandCellReservation {
    /// 所属管理器的标识。
    manager_id: uuid::Uuid,
    /// 启动回执。
    receipt: CommandCellLaunchReceipt,
    /// 发往监管任务的动作通道；消费后为 `None`。
    action: Option<oneshot::Sender<PreparedCellAction>>,
}

impl CommandCellReservation {
    /// 返回该预留对应的启动回执。
    pub fn receipt(&self) -> &CommandCellLaunchReceipt {
        &self.receipt
    }
}

impl std::fmt::Debug for CommandCellReservation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CommandCellReservation")
            .field("receipt", &self.receipt)
            .field("pending", &self.action.is_some())
            .finish()
    }
}

impl Drop for CommandCellReservation {
    fn drop(&mut self) {
        if let Some(action) = self.action.take() {
            let _ = action.send(PreparedCellAction::Abort(
                "prepared reservation dropped before start".to_string(),
            ));
        }
    }
}

/// 发给已预备 cell 监管任务的动作。
enum PreparedCellAction {
    /// 开始执行；启动后通过该发送端确认。
    Start(oneshot::Sender<()>),
    /// 以给定消息中止。
    Abort(String),
}

/// 单轮 wait 的租约，释放时重新执行终态保留清理。
struct CellWaiterLease {
    /// 被租用的 cell。
    handle: Arc<CommandCellHandle>,
    /// cell 注册表，用于释放后清理。
    cells: Arc<DashMap<String, Arc<CommandCellHandle>>>,
    /// 终态历史保留上限。
    max_terminal_history: usize,
}

impl Drop for CellWaiterLease {
    fn drop(&mut self) {
        self.handle.waiter_leases.fetch_sub(1, Ordering::AcqRel);
        prune_terminal_history(&self.cells, self.max_terminal_history);
    }
}

/// 按注册顺序移除超出 `max_terminal_history` 的最旧终态 cell。
fn prune_terminal_history(
    cells: &DashMap<String, Arc<CommandCellHandle>>,
    max_terminal_history: usize,
) {
    let mut terminal = terminal_retention_candidates(cells);
    let remove_count = terminal.len().saturating_sub(max_terminal_history);
    if remove_count == 0 {
        return;
    }
    terminal.sort_by_key(|(sequence, _)| *sequence);
    for (_, id) in terminal.into_iter().take(remove_count) {
        remove_terminal_candidate(cells, &id);
    }
}

/// 收集可被清理的终态 cell（无 wait/观察租约）及其注册序号。
fn terminal_retention_candidates(
    cells: &DashMap<String, Arc<CommandCellHandle>>,
) -> Vec<(u64, String)> {
    cells
        .iter()
        .filter(|entry| {
            entry.value().is_terminal()
                && entry.value().waiter_leases.load(Ordering::Acquire) == 0
                && entry.value().observation_leases.load(Ordering::Acquire) == 0
        })
        .map(|entry| (entry.value().sequence, entry.key().clone()))
        .collect()
}

/// 在持有分片锁的情况下复核条件后移除候选 cell。
fn remove_terminal_candidate(cells: &DashMap<String, Arc<CommandCellHandle>>, cell_id: &str) {
    /* 候选扫描只是保留策略的一次快照。等待者/观察者可能在扫描之后获取租约，
    因此移除时必须在 DashMap 持有分片锁的情况下重新检查条件。 */
    let _ = cells.remove_if(cell_id, |_id, handle| {
        handle.is_terminal()
            && handle.waiter_leases.load(Ordering::Acquire) == 0
            && handle.observation_leases.load(Ordering::Acquire) == 0
    });
}

impl CommandCellRegistry for BackgroundCommandManager {
    fn launch(
        &self,
        request: CommandCellRequest,
    ) -> BoxFuture<'_, std::result::Result<CommandCellLaunchReceipt, CommandCellError>> {
        Box::pin(async move {
            let reservation = self.prepare_launch(request).await?;
            self.start_prepared(reservation).await
        })
    }

    fn wait(
        &self,
        cell_id: &str,
        cursor: u64,
        yield_ms: u64,
    ) -> BoxFuture<'_, std::result::Result<CommandCellDelta, CommandCellError>> {
        let cell_id = cell_id.to_string();
        Box::pin(async move {
            /* 在持有 DashMap 条目的同时递增租约，消除了查找与清理之间的竞争。
            丢弃 `_lease` 即确认本轮 wait，并立即重新执行有界的终态保留。 */
            let (handle, _lease) = self.acquire_waiter_lease(&cell_id)?;

            loop {
                /* 先注册 waiter 再快照: 否则 snapshot 与 await 之间的
                notify_waiters() 会丢失唤醒。用 noop waker 强制完成一次
                poll 以完成注册。 */
                let mut notified = std::pin::pin!(handle.notify.notified());
                let registered_pending = {
                    let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
                    notified.as_mut().poll(&mut cx).is_pending()
                };

                let state = handle.current_state().await;
                // 锁只在块内持有, 避免 MutexGuard 跨 await。
                let (terminal, has_output) = {
                    let output = handle
                        .output
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    let bytes_since = output.total_bytes.saturating_sub(cursor);
                    (state.phase.is_terminal(), bytes_since > 0)
                };

                if terminal || has_output {
                    let reason = if terminal {
                        CommandCellWaitReason::Terminal
                    } else {
                        CommandCellWaitReason::Output
                    };
                    return Ok(snapshot_delta(&handle, cursor, state, reason).await);
                }
                if yield_ms == 0 || !registered_pending {
                    // 非阻塞 poll, 或 waiter 已带信号(罕见): 直接返回当前快照。
                    return Ok(snapshot_delta(
                        &handle,
                        cursor,
                        state,
                        CommandCellWaitReason::YieldElapsed,
                    )
                    .await);
                }

                match tokio::time::timeout(Duration::from_millis(yield_ms), notified).await {
                    Ok(()) => continue,
                    Err(_) => {
                        // yield 超时: 重读一次快照(与唤醒可能竞争, 以最后读取为准)。
                        let state = handle.current_state().await;
                        return Ok(snapshot_delta(
                            &handle,
                            cursor,
                            state,
                            CommandCellWaitReason::YieldElapsed,
                        )
                        .await);
                    }
                }
            }
        })
    }

    fn observe(
        &self,
        cell_id: &str,
    ) -> std::result::Result<CommandCellObservationLease, CommandCellError> {
        let entry = self
            .cells
            .get(cell_id)
            .ok_or_else(|| CommandCellError::NotFound {
                cell_id: cell_id.to_string(),
            })?;
        let handle = entry.value().clone();
        handle.observation_leases.fetch_add(1, Ordering::AcqRel);
        drop(entry);
        let weak_handle = Arc::downgrade(&handle);
        let cells = self.cells.clone();
        let terminal_history = self.config.max_terminal_history;
        Ok(CommandCellObservationLease::new(cell_id, move || {
            if let Some(handle) = weak_handle.upgrade() {
                handle.observation_leases.fetch_sub(1, Ordering::AcqRel);
            }
            prune_terminal_history(&cells, terminal_history);
        }))
    }

    fn stop(&self, cell_id: &str) -> bool {
        match self.cells.get(cell_id) {
            Some(entry) => {
                entry.value().cancel.cancel();
                true
            }
            None => false,
        }
    }

    fn list(&self) -> BoxFuture<'_, Vec<CommandCellSnapshot>> {
        Box::pin(async move {
            let mut snapshots = Vec::with_capacity(self.cells.len());
            for entry in self.cells.iter() {
                snapshots.push(entry.value().snapshot().await);
            }
            snapshots
        })
    }

    fn shutdown(&self) -> BoxFuture<'_, std::result::Result<(), CommandCellError>> {
        Box::pin(async move {
            let _admission = self.admission.lock().await;
            if !self.shutting_down.swap(true, Ordering::AcqRel) {
                self.shutdown.cancel();
                self.execution.close();
                self.tracked.close();
                for entry in self.cells.iter() {
                    entry.value().cancel.cancel();
                    entry.value().notify.notify_waiters();
                }
                self.tasks.close();
            }
            drop(_admission);
            self.tasks.wait().await;
            Ok(())
        })
    }
}

impl Drop for BackgroundCommandManager {
    fn drop(&mut self) {
        self.shutting_down.store(true, Ordering::Release);
        self.shutdown.cancel();
        self.execution.close();
        self.tracked.close();
        for entry in self.cells.iter() {
            entry.value().cancel.cancel();
            entry.value().notify.notify_waiters();
        }
        self.tasks.close();
    }
}

/// 等待请求方的取消令牌触发；未提供时永不完成。
async fn request_cancelled(cancel: Option<&Arc<CancellationToken>>) {
    match cancel {
        Some(cancel) => cancel.cancelled().await,
        None => std::future::pending().await,
    }
}

/**
监管一个已预备的 cell：等待启动/中止/取消/超时，运行后完成 artifact 收尾，
并一次性写入终态、唤醒等待者、执行保留清理。
*/
#[allow(clippy::too_many_arguments)]
async fn supervise_prepared_cell(
    handle: Arc<CommandCellHandle>,
    execution: Arc<Semaphore>,
    run_spec: CellRunSpec,
    action: oneshot::Receiver<PreparedCellAction>,
    shutdown: CancellationToken,
    cells: Arc<DashMap<String, Arc<CommandCellHandle>>>,
    terminal_history: usize,
) {
    let deadline = run_spec.deadline;
    let owner_cancel = run_spec.owner_cancel.clone();
    let mut outcome = tokio::select! {
        biased;
        _ = shutdown.cancelled() => CellOutcome::cancelled(),
        _ = cancellation_requested(&handle.cancel, run_spec.owner_cancel.as_ref()) => {
            CellOutcome::cancelled()
        }
        _ = tokio::time::sleep_until(run_spec.deadline) => CellOutcome::timed_out(),
        action = action => match action {
            Ok(PreparedCellAction::Start(started)) => {
                *handle.state.write().await = CellState {
                    phase: CommandCellPhase::Queued,
                    exit_code: None,
                    terminal_cause: None,
                    terminal_message: None,
                };
                handle.notify.notify_waiters();
                let _ = started.send(());
                run_cell(handle.clone(), execution, run_spec).await
            }
            Ok(PreparedCellAction::Abort(message)) => CellOutcome::runtime_failure(
                CommandCellTerminalCause::LaunchFailed,
                message,
            ),
            Err(_) => CellOutcome::runtime_failure(
                CommandCellTerminalCause::LaunchFailed,
                "prepared cell start gate closed".to_string(),
            ),
        },
    };

    let finalization = if deadline <= Instant::now() {
        mark_artifact_finalize_interrupted(&handle, ArtifactFinalizeOutcome::DeadlineElapsed);
        ArtifactFinalizeOutcome::DeadlineElapsed
    } else {
        finish_output_artifact(&handle, deadline, &shutdown, owner_cancel.as_ref()).await
    };
    match finalization {
        ArtifactFinalizeOutcome::DeadlineElapsed
            if outcome.cause == CommandCellTerminalCause::Exited =>
        {
            outcome = CellOutcome::timed_out();
        }
        ArtifactFinalizeOutcome::Shutdown if outcome.cause == CommandCellTerminalCause::Exited => {
            outcome = CellOutcome::cancelled();
        }
        ArtifactFinalizeOutcome::Cancelled if outcome.cause == CommandCellTerminalCause::Exited => {
            outcome = CellOutcome::cancelled();
        }
        ArtifactFinalizeOutcome::Finished
        | ArtifactFinalizeOutcome::DeadlineElapsed
        | ArtifactFinalizeOutcome::Shutdown
        | ArtifactFinalizeOutcome::Cancelled => {}
    }
    *handle.state.write().await = CellState {
        phase: outcome.phase,
        exit_code: outcome.exit_code,
        terminal_cause: Some(outcome.cause),
        terminal_message: outcome.message,
    };
    handle.terminal_flag.store(true, Ordering::Release);
    handle.notify.notify_waiters();
    prune_terminal_history(&cells, terminal_history);
    tracing::info!(
        cell_id = %handle.cell_id,
        phase = outcome.phase.as_str(),
        cause = outcome.cause.as_str(),
        "command cell finished"
    );
}

// ── 运行器 ──────────────────────────────────────────────────────────

/// 运行单个 cell 所需的参数。
struct CellRunSpec {
    /// 要执行的 shell 命令。
    command: String,
    /// 工作目录。
    working_dir: Option<PathBuf>,
    /**
    `launch` 接受 cell 时计算出的绝对截止时间。
    信号量准入与执行共享同一预算。
    */
    deadline: Instant,
    /// 可选的沙箱执行器。
    sandbox: Option<Arc<dyn SandboxExecutor>>,
    /// 所属运行的取消令牌。
    owner_cancel: Option<Arc<CancellationToken>>,
    /// 内存中保留的输出字节上限。
    max_retained: usize,
}

/// 单个 cell 运行的最终结果。
#[derive(Debug)]
struct CellOutcome {
    /// 终态阶段。
    phase: CommandCellPhase,
    /// 进程退出码（若有）。
    exit_code: Option<i32>,
    /// 终态原因。
    cause: CommandCellTerminalCause,
    /// 附加说明信息。
    message: Option<String>,
}

impl CellOutcome {
    /// 进程正常退出，根据是否成功决定阶段。
    fn exited(exit_code: Option<i32>, success: bool) -> Self {
        Self {
            phase: if success {
                CommandCellPhase::Succeeded
            } else {
                CommandCellPhase::Failed
            },
            exit_code,
            cause: CommandCellTerminalCause::Exited,
            message: None,
        }
    }

    /// 超时导致的失败。
    fn timed_out() -> Self {
        Self {
            phase: CommandCellPhase::Failed,
            exit_code: None,
            cause: CommandCellTerminalCause::TimedOut,
            message: None,
        }
    }

    /// 被取消。
    fn cancelled() -> Self {
        Self {
            phase: CommandCellPhase::Cancelled,
            exit_code: None,
            cause: CommandCellTerminalCause::Cancelled,
            message: None,
        }
    }

    /// 运行时失败；启动失败映射为 `LaunchFailed` 阶段，其余为 `Failed`。
    fn runtime_failure(cause: CommandCellTerminalCause, message: String) -> Self {
        let phase = if cause == CommandCellTerminalCause::LaunchFailed {
            CommandCellPhase::LaunchFailed
        } else {
            CommandCellPhase::Failed
        };
        Self {
            phase,
            exit_code: None,
            cause,
            message: Some(message),
        }
    }
}

/**
运行单个 cell 直至结束，返回其最终阶段 + 退出码。

所有路径（正常退出 / 超时 / 取消 / 启动失败 / 信号量关闭）都汇聚到同一个
返回点，使调用方只需写入一次终态。
*/
async fn run_cell(
    handle: Arc<CommandCellHandle>,
    semaphore: Arc<Semaphore>,
    spec: CellRunSpec,
) -> CellOutcome {
    let CellRunSpec {
        command,
        working_dir,
        deadline,
        sandbox,
        owner_cancel,
        max_retained,
    } = spec;
    /* 并发许可: 满则排队等待(不阻塞 launch 调用方)。取消必须能打断
    排队本身,不能等前一个小时级命令释放许可后才进入终态。 */
    let acquire = semaphore.acquire_owned();
    tokio::pin!(acquire);
    let _permit = match tokio::select! {
        biased;
        _ = cancellation_requested(&handle.cancel, owner_cancel.as_ref()) => {
            return CellOutcome::cancelled();
        }
        _ = wait_for_deadline(deadline) => {
            return CellOutcome::timed_out();
        }
        result = &mut acquire => result,
    } {
        Ok(permit) => permit,
        Err(_) => {
            tracing::warn!(cell_id = %handle.cell_id, "cell semaphore closed");
            return CellOutcome::runtime_failure(
                CommandCellTerminalCause::LaunchFailed,
                "background command semaphore closed".to_string(),
            );
        }
    };

    *handle.state.write().await = CellState {
        phase: CommandCellPhase::Running,
        exit_code: None,
        terminal_cause: None,
        terminal_message: None,
    };
    handle.notify.notify_waiters();

    if handle.cancel.is_cancelled()
        || owner_cancel
            .as_ref()
            .is_some_and(|cancel| cancel.is_cancelled())
    {
        return CellOutcome::cancelled();
    }
    if deadline <= Instant::now() {
        return CellOutcome::timed_out();
    }

    if let Some(sandbox) = sandbox {
        return run_sandbox_cell(
            handle,
            command,
            working_dir,
            deadline,
            sandbox,
            owner_cancel,
            max_retained,
        )
        .await;
    }

    /* env_clear + allowlist: 防止把密钥泄漏给后台进程; PATH/HOME 必需,
    LANG/LC_ALL 保证 UTF-8 输出不被 C locale 破坏, TMPDIR/TZ 是安全的
    功能性变量。(与旧 spawn_task 的策略一致。) */
    let mut cmd = tokio::process::Command::new("sh");
    cmd.arg("-c").arg(&command).env_clear();
    for var in ["PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "TZ"] {
        if let Ok(val) = std::env::var(var) {
            cmd.env(var, val);
        }
    }
    cmd.stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    #[cfg(unix)]
    cmd.process_group(0);
    if let Some(dir) = &working_dir {
        cmd.current_dir(dir);
    }

    let mut child = match cmd.spawn() {
        Ok(child) => child,
        Err(error) => {
            tracing::warn!(cell_id = %handle.cell_id, error = %error, "cell spawn failed");
            return CellOutcome::runtime_failure(
                CommandCellTerminalCause::LaunchFailed,
                format!("cell spawn failed: {error}"),
            );
        }
    };
    let process_group_id = child.id();

    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let stdout_join = stdout.map(|pipe| {
        let handle = handle.clone();
        tokio::spawn(async move {
            read_pipe(pipe, handle, max_retained, ToolOutputChannel::Stdout).await
        })
    });
    let stderr_join = stderr.map(|pipe| {
        let handle = handle.clone();
        tokio::spawn(async move {
            read_pipe(pipe, handle, max_retained, ToolOutputChannel::Stderr).await
        })
    });

    let mut outcome = tokio::select! {
        biased;
        _ = cancellation_requested(&handle.cancel, owner_cancel.as_ref()) => {
            kill_process_group(&mut child).await;
            CellOutcome::cancelled()
        }
        _ = wait_for_deadline(deadline) => {
            kill_process_group(&mut child).await;
            CellOutcome::timed_out()
        }
        status = child.wait() => match status {
            Ok(status) => CellOutcome::exited(status.code(), status.success()),
            Err(error) => CellOutcome::runtime_failure(
                CommandCellTerminalCause::WaitFailed,
                format!("waiting for command process failed: {error}"),
            ),
        },
    };

    /* 在同一绝对截止时间下读完读取任务。子孙进程可能在 shell 退出后仍保持
    继承的管道打开，因此仅凭进程状态并不是安全的完成边界。 */
    let drain_deadline = if outcome.cause == CommandCellTerminalCause::Cancelled {
        Instant::now()
            .checked_add(CANCEL_DRAIN_GRACE)
            .unwrap_or(deadline)
            .min(deadline)
    } else {
        deadline
    };
    let stdout_finish = finish_pipe_reader("stdout", stdout_join, drain_deadline).await;
    let stderr_finish = finish_pipe_reader("stderr", stderr_join, drain_deadline).await;
    if matches!(stdout_finish, PipeFinish::TimedOut)
        || matches!(stderr_finish, PipeFinish::TimedOut)
    {
        kill_process_group_id(process_group_id);
        if outcome.cause == CommandCellTerminalCause::Cancelled {
            outcome.message = Some("output drain deadline elapsed after cancellation".to_string());
        } else {
            outcome = CellOutcome::timed_out();
        }
    } else if outcome.cause == CommandCellTerminalCause::Exited {
        let failure = stdout_finish.failure().or_else(|| stderr_finish.failure());
        if let Some(message) = failure {
            outcome =
                CellOutcome::runtime_failure(CommandCellTerminalCause::OutputDrainFailed, message);
        }
    }

    outcome
}

/// 通过沙箱执行器以流式方式运行 cell。
async fn run_sandbox_cell(
    handle: Arc<CommandCellHandle>,
    command: String,
    working_dir: Option<PathBuf>,
    deadline: Instant,
    sandbox: Arc<dyn SandboxExecutor>,
    owner_cancel: Option<Arc<CancellationToken>>,
    max_retained: usize,
) -> CellOutcome {
    /* SandboxCommand 自带具体的超时；外层绝对截止时间才是排队、执行
    与流读取的权威依据。 */
    let backend_timeout = deadline.saturating_duration_since(Instant::now());
    if backend_timeout.is_zero() {
        return CellOutcome::timed_out();
    }
    let mut sandbox_command = SandboxCommand::shell(command);
    sandbox_command.timeout = backend_timeout;
    if let Some(dir) = &working_dir {
        sandbox_command = sandbox_command.with_working_dir(dir);
    }

    let stream = tokio::select! {
        biased;
        _ = cancellation_requested(&handle.cancel, owner_cancel.as_ref()) => {
            return CellOutcome::cancelled();
        }
        _ = wait_for_deadline(deadline) => {
            return CellOutcome::timed_out();
        }
        result = sandbox.execute_stream(sandbox_command) => result,
    };
    let mut stream = match stream {
        Ok(stream) => stream,
        Err(error) => {
            let message = format!("Sandbox execution failed: {error}\n");
            append_output(
                &handle,
                message.as_bytes(),
                max_retained,
                ToolOutputChannel::Stderr,
            );
            return CellOutcome::runtime_failure(
                CommandCellTerminalCause::LaunchFailed,
                format!("sandbox execution failed: {error}"),
            );
        }
    };
    let mut saw_stream_output = false;

    loop {
        let event = tokio::select! {
            biased;
            _ = cancellation_requested(&handle.cancel, owner_cancel.as_ref()) => {
                return CellOutcome::cancelled();
            }
            _ = wait_for_deadline(deadline) => {
                return CellOutcome::timed_out();
            }
            event = stream.next() => event,
        };
        let Some(event) = event else {
            return CellOutcome::runtime_failure(
                CommandCellTerminalCause::OutputDrainFailed,
                "sandbox output stream ended without a completion event".to_string(),
            );
        };
        match event {
            SandboxStreamEvent::Output { channel, chunk } => {
                saw_stream_output = true;
                let channel = match channel {
                    SandboxOutputChannel::Stdout => ToolOutputChannel::Stdout,
                    SandboxOutputChannel::Stderr => ToolOutputChannel::Stderr,
                };
                append_output(&handle, chunk.as_bytes(), max_retained, channel);
            }
            SandboxStreamEvent::Complete(result) => {
                if !saw_stream_output {
                    if !result.stdout.is_empty() {
                        append_output(
                            &handle,
                            result.stdout.as_bytes(),
                            max_retained,
                            ToolOutputChannel::Stdout,
                        );
                    }
                    if !result.stderr.is_empty() {
                        append_output(
                            &handle,
                            result.stderr.as_bytes(),
                            max_retained,
                            ToolOutputChannel::Stderr,
                        );
                    }
                }
                if result.output_truncated {
                    let logical_bytes = result.stdout_bytes.saturating_add(result.stderr_bytes);
                    let mut output = handle
                        .output
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    output.total_bytes = output.total_bytes.max(logical_bytes);
                    output.truncated = true;
                }
                if result.cancelled {
                    return CellOutcome::cancelled();
                }
                if result.timed_out {
                    return CellOutcome::timed_out();
                }
                return CellOutcome::exited(Some(result.exit_code), result.success());
            }
            SandboxStreamEvent::Failed { failure } => {
                let message = failure.message().to_string();
                append_output(
                    &handle,
                    message.as_bytes(),
                    max_retained,
                    ToolOutputChannel::Stderr,
                );
                if failure.is_cancelled() {
                    return CellOutcome::cancelled();
                }
                return CellOutcome::runtime_failure(
                    CommandCellTerminalCause::OutputDrainFailed,
                    message,
                );
            }
        }
    }
}

/// 等待到达截止时间。
async fn wait_for_deadline(deadline: Instant) {
    tokio::time::sleep_until(deadline).await;
}

/// 读取管道直到 EOF，把每块数据追加到 cell 输出缓冲区，并在每次追加后唤醒所有等待者。
async fn read_pipe<R: tokio::io::AsyncRead + Unpin>(
    mut pipe: R,
    handle: Arc<CommandCellHandle>,
    max_retained: usize,
    channel: ToolOutputChannel,
) -> std::result::Result<(), String> {
    let mut buffer = vec![0_u8; READ_CHUNK_BYTES];
    loop {
        match pipe.read(&mut buffer).await {
            Ok(0) => return Ok(()),
            Err(error) => return Err(error.to_string()),
            Ok(count) => {
                let bytes = buffer.get(..count).unwrap_or_default();
                append_output(&handle, bytes, max_retained, channel);
            }
        }
    }
}

/// 管道读取任务的收尾结果。
enum PipeFinish {
    /// 已结束；附带失败信息（若有）。
    Finished(Option<String>),
    /// 截止时间到达前未能结束。
    TimedOut,
}

impl PipeFinish {
    /// 返回失败信息（若有）；超时不视为失败信息。
    fn failure(&self) -> Option<String> {
        match self {
            Self::Finished(failure) => failure.clone(),
            Self::TimedOut => None,
        }
    }
}

/// 在截止时间内等待管道读取任务结束，超时则中止该任务。
async fn finish_pipe_reader(
    channel: &str,
    join: Option<tokio::task::JoinHandle<std::result::Result<(), String>>>,
    deadline: Instant,
) -> PipeFinish {
    let Some(mut join) = join else {
        return PipeFinish::Finished(None);
    };
    let result = tokio::select! {
        biased;
        _ = tokio::time::sleep_until(deadline) => {
            join.abort();
            let _ = join.await;
            return PipeFinish::TimedOut;
        }
        result = &mut join => result,
    };
    match result {
        Ok(Ok(())) => PipeFinish::Finished(None),
        Ok(Err(error)) => PipeFinish::Finished(Some(format!("{channel} drain failed: {error}"))),
        Err(error) => PipeFinish::Finished(Some(format!("{channel} reader task failed: {error}"))),
    }
}

/// 等待 cell 自身或所属运行的取消信号，任一触发即完成。
async fn cancellation_requested(
    cell_cancel: &CancellationToken,
    owner_cancel: Option<&Arc<CancellationToken>>,
) {
    tokio::select! {
        _ = cell_cancel.cancelled() => {}
        _ = async {
            match owner_cancel {
                Some(cancel) => cancel.cancelled().await,
                None => std::future::pending().await,
            }
        } => {}
    }
}

/// 追加输出到保留缓冲区与 artifact，并唤醒所有等待者。
fn append_output(
    handle: &CommandCellHandle,
    bytes: &[u8],
    max_retained: usize,
    channel: ToolOutputChannel,
) {
    {
        let mut output = handle
            .output
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        output.push(bytes, max_retained);
    }
    let mut artifact = handle
        .artifact
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if artifact.writer.is_some() {
        let chunks = match channel {
            ToolOutputChannel::Stdout => artifact.stdout_decoder.push(bytes),
            ToolOutputChannel::Stderr | ToolOutputChannel::Log => {
                artifact.stderr_decoder.push(bytes)
            }
        };
        for text in chunks {
            if !push_artifact_text(&mut artifact, channel, &text, &handle.cell_id) {
                break;
            }
        }
    }
    drop(artifact);
    handle.notify.notify_waiters();
}

/// artifact 收尾的结果。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ArtifactFinalizeOutcome {
    /// 收尾正常完成（无论写入成功与否）。
    Finished,
    /// 截止时间到达，收尾被打断。
    DeadlineElapsed,
    /// 管理器关闭，收尾被打断。
    Shutdown,
    /// cell 被取消，收尾被打断。
    Cancelled,
}

/// 刷新解码器残余内容并完成输出 artifact，可被关闭/取消/截止时间打断。
async fn finish_output_artifact(
    handle: &CommandCellHandle,
    deadline: Instant,
    shutdown: &CancellationToken,
    owner_cancel: Option<&Arc<CancellationToken>>,
) -> ArtifactFinalizeOutcome {
    let writer = {
        let mut artifact = handle
            .artifact
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(text) = artifact.stdout_decoder.finish() {
            push_artifact_text(
                &mut artifact,
                ToolOutputChannel::Stdout,
                &text,
                &handle.cell_id,
            );
        }
        if let Some(text) = artifact.stderr_decoder.finish() {
            push_artifact_text(
                &mut artifact,
                ToolOutputChannel::Stderr,
                &text,
                &handle.cell_id,
            );
        }
        artifact.writer.take()
    };
    let Some(writer) = writer else {
        return ArtifactFinalizeOutcome::Finished;
    };

    #[cfg(test)]
    let finalize_hook = handle
        .artifact_finalize_hook
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone();
    #[cfg(test)]
    if let Some(hook) = finalize_hook {
        hook.entered.notify_one();
        let interrupted = tokio::select! {
            biased;
            _ = shutdown.cancelled() => Some(ArtifactFinalizeOutcome::Shutdown),
            _ = cancellation_requested(&handle.cancel, owner_cancel) => {
                Some(ArtifactFinalizeOutcome::Cancelled)
            }
            _ = tokio::time::sleep_until(deadline) => Some(ArtifactFinalizeOutcome::DeadlineElapsed),
            _ = hook.release.notified() => None,
        };
        if let Some(outcome) = interrupted {
            mark_artifact_finalize_interrupted(handle, outcome);
            return outcome;
        }
    }

    let result = tokio::select! {
        biased;
        _ = shutdown.cancelled() => {
            mark_artifact_finalize_interrupted(handle, ArtifactFinalizeOutcome::Shutdown);
            return ArtifactFinalizeOutcome::Shutdown;
        }
        _ = cancellation_requested(&handle.cancel, owner_cancel) => {
            mark_artifact_finalize_interrupted(handle, ArtifactFinalizeOutcome::Cancelled);
            return ArtifactFinalizeOutcome::Cancelled;
        }
        _ = tokio::time::sleep_until(deadline) => {
            mark_artifact_finalize_interrupted(handle, ArtifactFinalizeOutcome::DeadlineElapsed);
            return ArtifactFinalizeOutcome::DeadlineElapsed;
        }
        result = writer.finish_async() => result,
    };
    let mut artifact = handle
        .artifact
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    match result {
        Ok(Some(reference)) => {
            artifact.reference = Some(reference);
            artifact.status = CommandCellArtifactStatus::Available;
            artifact.message = None;
        }
        Ok(None) => {
            artifact.status = CommandCellArtifactStatus::BelowThreshold;
            artifact.message = None;
        }
        Err(error) => {
            tracing::warn!(cell_id = %handle.cell_id, %error, "cell output artifact finalize failed");
            artifact.status = CommandCellArtifactStatus::Failed;
            artifact.message = Some(error.to_string());
        }
    }
    ArtifactFinalizeOutcome::Finished
}

/// 将仍处于写入中的 artifact 标记为因收尾被打断而失败。
fn mark_artifact_finalize_interrupted(
    handle: &CommandCellHandle,
    outcome: ArtifactFinalizeOutcome,
) {
    let mut artifact = handle
        .artifact
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if artifact.status != CommandCellArtifactStatus::Writing {
        return;
    }
    artifact.status = CommandCellArtifactStatus::Failed;
    artifact.message = Some(match outcome {
        ArtifactFinalizeOutcome::DeadlineElapsed => {
            "artifact finalization exceeded the command-cell deadline".to_string()
        }
        ArtifactFinalizeOutcome::Shutdown => {
            "artifact finalization cancelled during command-cell shutdown".to_string()
        }
        ArtifactFinalizeOutcome::Cancelled => {
            "artifact finalization cancelled with the command cell".to_string()
        }
        ArtifactFinalizeOutcome::Finished => "artifact finalization failed".to_string(),
    });
}

/// 向 artifact 写入一段文本；写入失败时标记失败并返回 `false`。
fn push_artifact_text(
    artifact: &mut CellArtifactState,
    channel: ToolOutputChannel,
    text: &str,
    cell_id: &str,
) -> bool {
    let Some(writer) = artifact.writer.as_mut() else {
        return false;
    };
    if let Err(error) = writer.push_channel(channel, text) {
        tracing::warn!(cell_id, %error, "cell output artifact write failed");
        artifact.writer = None;
        artifact.status = CommandCellArtifactStatus::Failed;
        artifact.message = Some(error.to_string());
        return false;
    }
    true
}

/// 杀死子进程所在的整个进程组并回收子进程。尽力而为：忽略错误（cell 仍会收敛到终态）。
async fn kill_process_group(child: &mut tokio::process::Child) {
    kill_process_group_id(child.id());
    let _ = tokio::time::timeout(CANCEL_DRAIN_GRACE, async {
        let _ = child.kill().await;
        let _ = child.wait().await;
    })
    .await;
}

/// 向指定进程组发送 SIGKILL（仅 Unix）。
fn kill_process_group_id(pid: Option<u32>) {
    #[cfg(unix)]
    if let Some(pid) = pid {
        let _ = std::process::Command::new("kill")
            .args(["-KILL", &format!("-{pid}")])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }
    #[cfg(not(unix))]
    let _ = pid;
}

// ── 测试 ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use echo_core::sandbox::{ExecutionResult, IsolationLevel};
    use echo_core::tools::cell::CommandCellRegistry;

    /// 立即返回成功结果并记录执行次数的测试沙箱。
    struct TestSandbox {
        executions: AtomicU64,
    }

    /// 在被放行前一直阻塞的测试沙箱。
    struct BlockingSandbox {
        executions: AtomicU64,
        started: Notify,
        release: Notify,
    }

    impl SandboxExecutor for TestSandbox {
        fn name(&self) -> &str {
            "cell-test-sandbox"
        }

        fn isolation_level(&self) -> IsolationLevel {
            IsolationLevel::Process
        }

        fn is_available(&self) -> BoxFuture<'_, bool> {
            Box::pin(async { true })
        }

        fn execute(
            &self,
            _command: SandboxCommand,
        ) -> BoxFuture<'_, echo_core::error::Result<ExecutionResult>> {
            self.executions.fetch_add(1, Ordering::Relaxed);
            Box::pin(async {
                Ok(ExecutionResult {
                    exit_code: 0,
                    stdout: "sandbox-cell-ok\n".to_string(),
                    stderr: String::new(),
                    duration: Duration::from_millis(1),
                    sandbox_type: "cell-test-sandbox".to_string(),
                    timed_out: false,
                    cancelled: false,
                    output_truncated: false,
                    stdout_bytes: 16,
                    stderr_bytes: 0,
                })
            })
        }
    }

    impl SandboxExecutor for BlockingSandbox {
        fn name(&self) -> &str {
            "blocking-cell-test-sandbox"
        }

        fn isolation_level(&self) -> IsolationLevel {
            IsolationLevel::Process
        }

        fn is_available(&self) -> BoxFuture<'_, bool> {
            Box::pin(async { true })
        }

        fn execute(
            &self,
            _command: SandboxCommand,
        ) -> BoxFuture<'_, echo_core::error::Result<ExecutionResult>> {
            self.executions.fetch_add(1, Ordering::AcqRel);
            self.started.notify_one();
            Box::pin(async move {
                self.release.notified().await;
                Ok(ExecutionResult {
                    exit_code: 0,
                    stdout: "released\n".to_string(),
                    stderr: String::new(),
                    duration: Duration::from_millis(1),
                    sandbox_type: "blocking-cell-test-sandbox".to_string(),
                    timed_out: false,
                    cancelled: false,
                    output_truncated: false,
                    stdout_bytes: 9,
                    stderr_bytes: 0,
                })
            })
        }
    }

    /// 使用默认配置构建管理器。
    fn manager() -> BackgroundCommandManager {
        BackgroundCommandManager::default()
    }

    /// 构建序号为 0 的测试 cell 句柄。
    fn test_handle(cell_id: &str) -> Arc<CommandCellHandle> {
        test_handle_with_sequence(cell_id, 0)
    }

    /// 构建指定注册序号的测试 cell 句柄。
    fn test_handle_with_sequence(cell_id: &str, sequence: u64) -> Arc<CommandCellHandle> {
        Arc::new(CommandCellHandle {
            cell_id: cell_id.to_string(),
            name: cell_id.to_string(),
            state: RwLock::new(CellState {
                phase: CommandCellPhase::Running,
                exit_code: None,
                terminal_cause: None,
                terminal_message: None,
            }),
            output: Mutex::new(OutputBuffer::default()),
            artifact: Mutex::new(CellArtifactState {
                writer: None,
                stdout_decoder: IncrementalUtf8Decoder::default(),
                stderr_decoder: IncrementalUtf8Decoder::default(),
                reference: None,
                status: CommandCellArtifactStatus::NotRequested,
                message: None,
            }),
            notify: Notify::new(),
            cancel: CancellationToken::new(),
            terminal_flag: AtomicBool::new(false),
            waiter_leases: AtomicU64::new(0),
            observation_leases: AtomicU64::new(0),
            _tracked_permit: Mutex::new(None),
            sequence,
            artifact_finalize_hook: Mutex::new(None),
        })
    }

    /// 构建给定命令的默认启动请求。
    fn request(command: &str) -> CommandCellRequest {
        CommandCellRequest {
            command: command.to_string(),
            working_dir: None,
            timeout_secs: None,
            ..Default::default()
        }
    }

    /// 启动命令并返回 cell ID。
    async fn launch_cell(
        manager: &BackgroundCommandManager,
        request: CommandCellRequest,
    ) -> std::result::Result<String, String> {
        manager
            .launch(request)
            .await
            .map(|receipt| receipt.cell_id)
            .map_err(|error| error.to_string())
    }

    /// 启动一个 artifact 收尾会被测试钩子阻塞的 cell。
    async fn launch_with_blocking_artifact_finalizer(
        manager: &BackgroundCommandManager,
        root: &std::path::Path,
        owner_cancel: Option<Arc<CancellationToken>>,
        call_id: &str,
    ) -> std::result::Result<(String, Arc<TestArtifactFinalizeHook>), String> {
        let mut launch = request("printf blocking-finalizer-output");
        launch.timeout_secs = Some(30);
        launch.cancel = owner_cancel;
        launch.output_artifacts = Some(
            echo_core::tools::artifact::ToolOutputArtifactConfig::new(root, "test")
                .threshold_bytes(1),
        );
        launch.artifact_identity = Some(echo_core::tools::artifact::ToolOutputArtifactIdentity {
            conversation_id: Some("conversation".to_string()),
            run_id: Some("run".to_string()),
            call_id: call_id.to_string(),
            tool_name: "shell".to_string(),
        });
        let reservation = manager
            .prepare_launch(launch)
            .await
            .map_err(|error| error.to_string())?;
        let cell_id = reservation.receipt().cell_id.clone();
        let hook = Arc::new(TestArtifactFinalizeHook {
            entered: Notify::new(),
            release: Notify::new(),
        });
        let entry = manager
            .cells
            .get(&cell_id)
            .ok_or_else(|| "prepared cell was not published".to_string())?;
        *entry
            .artifact_finalize_hook
            .lock()
            .unwrap_or_else(|error| error.into_inner()) = Some(hook.clone());
        drop(entry);
        let _receipt = manager
            .start_prepared(reservation)
            .await
            .map_err(|error| error.to_string())?;
        tokio::time::timeout(Duration::from_secs(2), hook.entered.notified())
            .await
            .map_err(|_| "artifact finalizer did not enter the injected wait".to_string())?;
        Ok((cell_id, hook))
    }

    /**
    循环 wait 直到终态, 拼接全部增量输出。wait 会在"有新输出"时就返回
    (可能尚未终态), 因此测试里用它收集完整结果。
    */
    async fn drain_to_terminal(
        manager: &BackgroundCommandManager,
        cell_id: &str,
    ) -> (String, CommandCellDelta) {
        let mut cursor = 0_u64;
        let mut combined = String::new();
        let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
        loop {
            let delta = manager.wait(cell_id, cursor, 10_000).await.unwrap();
            assert!(
                tokio::time::Instant::now() < deadline,
                "cell {cell_id} did not reach terminal phase in time"
            );
            combined.push_str(&delta.new_output);
            cursor = delta.next_cursor;
            if delta.snapshot.phase.is_terminal() && cursor >= delta.snapshot.total_output_bytes {
                return (combined, delta);
            }
        }
    }

    #[tokio::test]
    async fn echo_cell_completes_and_captures_output() {
        let manager = manager();
        let cell_id = launch_cell(&manager, request("echo hello-cell"))
            .await
            .unwrap();

        let (combined, delta) = drain_to_terminal(&manager, &cell_id).await;
        assert_eq!(delta.snapshot.phase, CommandCellPhase::Succeeded);
        assert_eq!(delta.snapshot.exit_code, Some(0));
        assert_eq!(
            delta.snapshot.terminal_cause,
            Some(CommandCellTerminalCause::Exited)
        );
        assert_eq!(
            delta.snapshot.artifact_status,
            CommandCellArtifactStatus::NotRequested
        );
        assert!(combined.contains("hello-cell"), "got: {combined}");
        assert!(delta.next_cursor > 0);
    }

    #[tokio::test]
    async fn launch_publishes_handle_before_fast_terminal_settlement()
    -> std::result::Result<(), String> {
        let manager = manager();
        let reservation = manager
            .prepare_launch(request("printf fast-terminal"))
            .await
            .map_err(|error| error.to_string())?;
        let cell_id = reservation.receipt().cell_id.clone();
        assert!(manager.cells.contains_key(&cell_id));
        let _receipt = manager
            .start_prepared(reservation)
            .await
            .map_err(|error| error.to_string())?;
        let (output, terminal) = drain_to_terminal(&manager, &cell_id).await;
        assert_eq!(terminal.snapshot.phase, CommandCellPhase::Succeeded);
        assert!(output.contains("fast-terminal"));
        Ok(())
    }

    #[tokio::test]
    async fn wait_short_yield_then_retry_reaches_terminal() {
        let manager = manager();
        let cell_id = launch_cell(&manager, request("sleep 0.4; echo done-retry"))
            .await
            .unwrap();

        // 短 yield: cell 仍在运行, 返回空增量(retry-safe, 不消费任何状态)。
        let first = manager.wait(&cell_id, 0, 50).await.unwrap();
        if !first.snapshot.phase.is_terminal() {
            assert!(first.new_output.is_empty());
            assert_eq!(first.wait_reason, CommandCellWaitReason::YieldElapsed);
        }

        // 再次 wait(长 yield): 继续推进直到终态, 拿到完整输出。
        let (combined, last) = drain_to_terminal(&manager, &cell_id).await;
        assert!(combined.contains("done-retry"));

        // 终态可重复读取(多等待者语义)。
        let third = manager.wait(&cell_id, last.next_cursor, 0).await.unwrap();
        assert_eq!(third.snapshot.phase, CommandCellPhase::Succeeded);
        assert!(third.new_output.is_empty());
    }

    #[tokio::test]
    async fn incremental_output_spliced_across_waits() {
        let manager = manager();
        let cell_id = launch_cell(&manager, request("echo one; sleep 1; echo two"))
            .await
            .unwrap();

        let mut combined = String::new();
        let mut cursor = 0_u64;
        let deadline = tokio::time::Instant::now() + Duration::from_secs(20);
        loop {
            let delta = manager.wait(&cell_id, cursor, 10_000).await.unwrap();
            combined.push_str(&delta.new_output);
            cursor = delta.next_cursor;
            if delta.snapshot.phase.is_terminal() {
                break;
            }
            assert!(
                tokio::time::Instant::now() < deadline,
                "cell did not finish in time"
            );
        }

        assert!(combined.contains("one"), "combined: {combined}");
        assert!(combined.contains("two"), "combined: {combined}");
    }

    #[tokio::test]
    async fn stop_cancels_long_running_cell() {
        let manager = manager();
        let cell_id = launch_cell(&manager, request("sleep 30")).await.unwrap();

        let running = manager.wait(&cell_id, 0, 200).await.unwrap();
        assert_eq!(running.snapshot.phase, CommandCellPhase::Running);

        assert!(manager.stop(&cell_id));
        let cancelled = manager
            .wait(&cell_id, running.next_cursor, 10_000)
            .await
            .unwrap();
        assert_eq!(cancelled.snapshot.phase, CommandCellPhase::Cancelled);
        assert_eq!(
            cancelled.snapshot.terminal_cause,
            Some(CommandCellTerminalCause::Cancelled)
        );

        // 未知 cell → false。
        assert!(!manager.stop("no-such-cell"));
    }

    #[tokio::test]
    async fn unknown_cell_wait_returns_error() {
        let manager = manager();
        let result = manager.wait("does-not-exist", 0, 10).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn long_unicode_output_is_incrementally_drainable_without_panic() {
        let manager = manager();
        /* 多字节输出超过单轮 16 KiB；每一轮都必须推进真实字节游标，不能
        因终态或字符截断直接跳到 total 而永久漏掉中间输出。 */
        let cell_id = launch_cell(
            &manager,
            request("for i in $(seq 1 3000); do echo \"中文输出🦀行$i\"; done"),
        )
        .await
        .unwrap();

        let (combined, last) = drain_to_terminal(&manager, &cell_id).await;
        assert_eq!(last.snapshot.phase, CommandCellPhase::Succeeded);
        assert!(combined.contains('🦀'));

        let first = manager.wait(&cell_id, 0, 0).await.unwrap();
        assert_eq!(first.snapshot.phase, CommandCellPhase::Succeeded);
        assert!(
            first.new_output.len() <= MAX_DELTA_BYTES.saturating_add(80),
            "byte cap exceeded: {}",
            first.new_output.len()
        );
        assert!(
            first.output_elided,
            "expected elided flag for capped output"
        );
        assert!(first.next_cursor < first.snapshot.total_output_bytes);
        assert!(!first.new_output.contains('\u{fffd}'));

        let mut cursor = first.next_cursor;
        let mut rounds = 1_u32;
        while cursor < first.snapshot.total_output_bytes {
            let delta = manager.wait(&cell_id, cursor, 0).await.unwrap();
            assert!(!delta.new_output.contains('\u{fffd}'));
            assert!(
                delta.next_cursor > cursor,
                "cursor must advance on every drain"
            );
            cursor = delta.next_cursor;
            rounds = rounds.saturating_add(1);
        }
        assert!(rounds > 1);
        assert_eq!(cursor, first.snapshot.total_output_bytes);
    }

    #[tokio::test]
    async fn queued_cell_cancellation_does_not_wait_for_a_permit() -> std::result::Result<(), String>
    {
        let manager = BackgroundCommandManager::new(BackgroundCommandManagerConfig {
            max_concurrent: 1,
            ..Default::default()
        })?;
        let occupying = launch_cell(&manager, request("sleep 30")).await.unwrap();
        let first = manager.wait(&occupying, 0, 200).await.unwrap();
        assert_eq!(first.snapshot.phase, CommandCellPhase::Running);

        let queued = launch_cell(&manager, request("echo should-not-run"))
            .await
            .unwrap();
        assert!(manager.stop(&queued));
        let cancelled = manager.wait(&queued, 0, 1_000).await.unwrap();
        assert_eq!(cancelled.snapshot.phase, CommandCellPhase::Cancelled);

        let occupying_state = manager.wait(&occupying, 0, 0).await.unwrap();
        assert_eq!(occupying_state.snapshot.phase, CommandCellPhase::Running);
        manager.stop(&occupying);
        let _ = manager.wait(&occupying, 0, 10_000).await;
        Ok(())
    }

    #[tokio::test]
    async fn owning_run_cancellation_propagates_to_the_cell() {
        let manager = manager();
        let cancel = Arc::new(CancellationToken::new());
        let mut launch = request("sleep 30");
        launch.cancel = Some(cancel.clone());
        let cell_id = launch_cell(&manager, launch).await.unwrap();
        let running = manager.wait(&cell_id, 0, 200).await.unwrap();
        assert_eq!(running.snapshot.phase, CommandCellPhase::Running);

        cancel.cancel();
        let terminal = manager.wait(&cell_id, 0, 10_000).await.unwrap();
        assert_eq!(terminal.snapshot.phase, CommandCellPhase::Cancelled);
        assert_eq!(
            terminal.snapshot.terminal_cause,
            Some(CommandCellTerminalCause::Cancelled)
        );
    }

    #[tokio::test]
    async fn complete_output_spills_to_the_existing_artifact_writer() {
        let root = tempfile::tempdir().unwrap();
        let manager = manager();
        let mut launch = request("printf 'artifact-output-abcdefghijklmnopqrstuvwxyz'");
        launch.output_artifacts = Some(
            echo_core::tools::artifact::ToolOutputArtifactConfig::new(root.path(), "test")
                .threshold_bytes(8),
        );
        launch.artifact_identity = Some(echo_core::tools::artifact::ToolOutputArtifactIdentity {
            conversation_id: Some("conversation".to_string()),
            run_id: Some("run".to_string()),
            call_id: "call".to_string(),
            tool_name: "shell".to_string(),
        });
        let cell_id = launch_cell(&manager, launch).await.unwrap();
        let (_, terminal) = drain_to_terminal(&manager, &cell_id).await;
        let artifact = terminal
            .snapshot
            .output_artifact
            .as_ref()
            .expect("large cell output must spill");
        assert!(artifact.path.exists());
        let persisted = std::fs::read_to_string(&artifact.path).unwrap();
        assert!(persisted.contains("artifact-output-abcdefghijklmnopqrstuvwxyz"));
        assert_eq!(
            terminal.snapshot.artifact_status,
            CommandCellArtifactStatus::Available
        );
    }

    #[tokio::test]
    async fn sandbox_required_launch_uses_the_configured_executor()
    -> std::result::Result<(), String> {
        let sandbox = Arc::new(TestSandbox {
            executions: AtomicU64::new(0),
        });
        let manager = BackgroundCommandManager::new_with_sandbox(
            BackgroundCommandManagerConfig::default(),
            sandbox.clone(),
        )?;
        let mut launch = request("echo ignored-by-test-executor");
        launch.require_sandbox = true;
        let cell_id = launch_cell(&manager, launch).await.unwrap();
        let (output, terminal) = drain_to_terminal(&manager, &cell_id).await;
        assert_eq!(terminal.snapshot.phase, CommandCellPhase::Succeeded);
        assert!(output.contains("sandbox-cell-ok"));
        assert_eq!(sandbox.executions.load(Ordering::Relaxed), 1);
        Ok(())
    }

    #[tokio::test]
    async fn sandbox_required_launch_never_silently_downgrades() {
        let manager = manager();
        let mut launch = request("echo must-not-run-directly");
        launch.require_sandbox = true;
        assert!(launch_cell(&manager, launch).await.is_err());
    }

    #[tokio::test]
    async fn retained_buffer_tail_only_marks_truncation() -> std::result::Result<(), String> {
        let manager = BackgroundCommandManager::new(BackgroundCommandManagerConfig {
            max_retained_output_bytes: 1024,
            ..Default::default()
        })?;
        // ~7 bytes × 400 行 ≈ 2.8 KB > 1 KB retention。
        let cell_id = launch_cell(&manager, request("seq 1 400")).await.unwrap();
        let delta = manager.wait(&cell_id, 0, 20_000).await.unwrap();

        assert!(delta.snapshot.output_truncated);
        assert!(delta.snapshot.total_output_bytes > 1024);
        // cursor=0 早于 retained 起点: 有被丢弃字节的提示。
        assert!(delta.new_output.contains("discarded"));
        assert!(delta.output_elided);

        let (_, last) = drain_to_terminal(&manager, &cell_id).await;
        assert_eq!(last.snapshot.phase, CommandCellPhase::Succeeded);
        Ok(())
    }

    #[tokio::test]
    async fn timeout_marks_cell_failed() -> std::result::Result<(), String> {
        let manager = BackgroundCommandManager::new(BackgroundCommandManagerConfig {
            default_timeout_secs: 1,
            ..Default::default()
        })?;
        let cell_id = launch_cell(&manager, request("sleep 30")).await.unwrap();
        let delta = manager.wait(&cell_id, 0, 10_000).await.unwrap();
        assert_eq!(delta.snapshot.phase, CommandCellPhase::Failed);
        assert_eq!(delta.snapshot.exit_code, None);
        assert_eq!(
            delta.snapshot.terminal_cause,
            Some(CommandCellTerminalCause::TimedOut)
        );
        Ok(())
    }

    #[tokio::test]
    async fn bad_working_dir_marks_launch_failed() {
        let manager = manager();
        let cell_id = launch_cell(
            &manager,
            CommandCellRequest {
                command: "echo x".to_string(),
                working_dir: Some("/nonexistent-cell-dir-xyz".to_string()),
                timeout_secs: None,
                ..Default::default()
            },
        )
        .await
        .unwrap();
        let delta = manager.wait(&cell_id, 0, 10_000).await.unwrap();
        assert_eq!(delta.snapshot.phase, CommandCellPhase::LaunchFailed);
        assert_eq!(
            delta.snapshot.terminal_cause,
            Some(CommandCellTerminalCause::LaunchFailed)
        );
        assert!(delta.snapshot.terminal_message.is_some());
    }

    #[test]
    fn zero_concurrency_configuration_is_rejected() {
        let result = BackgroundCommandManager::new(BackgroundCommandManagerConfig {
            max_concurrent: 0,
            ..Default::default()
        });
        assert!(result.is_err());
        let overflow = BackgroundCommandManager::new(BackgroundCommandManagerConfig {
            max_concurrent: 1,
            max_terminal_history: usize::MAX,
            ..Default::default()
        });
        assert!(overflow.is_err());
    }

    #[tokio::test]
    async fn queued_cell_timeout_uses_the_launch_time_deadline() -> std::result::Result<(), String>
    {
        let semaphore = Arc::new(Semaphore::new(1));
        let permit = semaphore
            .clone()
            .acquire_owned()
            .await
            .map_err(|error| error.to_string())?;
        let deadline = Instant::now()
            .checked_add(Duration::from_millis(25))
            .ok_or_else(|| "test deadline overflow".to_string())?;
        let outcome = run_cell(
            test_handle("queued-timeout"),
            semaphore,
            CellRunSpec {
                command: "echo must-not-run".to_string(),
                working_dir: None,
                deadline,
                sandbox: None,
                owner_cancel: None,
                max_retained: 1024,
            },
        )
        .await;
        drop(permit);

        assert_eq!(outcome.phase, CommandCellPhase::Failed);
        assert_eq!(outcome.cause, CommandCellTerminalCause::TimedOut);
        assert_eq!(outcome.exit_code, None);
        Ok(())
    }

    #[tokio::test]
    async fn artifact_below_threshold_has_typed_status() -> std::result::Result<(), String> {
        let root = tempfile::tempdir().map_err(|error| error.to_string())?;
        let manager = manager();
        let mut launch = request("printf tiny");
        launch.output_artifacts = Some(
            echo_core::tools::artifact::ToolOutputArtifactConfig::new(root.path(), "test")
                .threshold_bytes(1024),
        );
        launch.artifact_identity = Some(echo_core::tools::artifact::ToolOutputArtifactIdentity {
            conversation_id: Some("conversation".to_string()),
            run_id: Some("run".to_string()),
            call_id: "below-threshold".to_string(),
            tool_name: "shell".to_string(),
        });
        let cell_id = launch_cell(&manager, launch).await?;
        let (_, terminal) = drain_to_terminal(&manager, &cell_id).await;
        assert_eq!(
            terminal.snapshot.artifact_status,
            CommandCellArtifactStatus::BelowThreshold
        );
        assert!(terminal.snapshot.output_artifact.is_none());
        Ok(())
    }

    #[tokio::test]
    async fn artifact_write_failure_has_typed_status() -> std::result::Result<(), String> {
        let root = tempfile::tempdir().map_err(|error| error.to_string())?;
        let blocked_root = root.path().join("not-a-directory");
        std::fs::write(&blocked_root, b"file").map_err(|error| error.to_string())?;
        let manager = manager();
        let mut launch = request("printf artifact-write-failure");
        launch.output_artifacts = Some(
            echo_core::tools::artifact::ToolOutputArtifactConfig::new(&blocked_root, "test")
                .threshold_bytes(1),
        );
        launch.artifact_identity = Some(echo_core::tools::artifact::ToolOutputArtifactIdentity {
            conversation_id: Some("conversation".to_string()),
            run_id: Some("run".to_string()),
            call_id: "write-failure".to_string(),
            tool_name: "shell".to_string(),
        });
        let cell_id = launch_cell(&manager, launch).await?;
        let (_, terminal) = drain_to_terminal(&manager, &cell_id).await;
        assert_eq!(terminal.snapshot.phase, CommandCellPhase::Succeeded);
        assert_eq!(
            terminal.snapshot.artifact_status,
            CommandCellArtifactStatus::Failed
        );
        assert!(terminal.snapshot.artifact_message.is_some());
        assert!(terminal.snapshot.output_artifact.is_none());
        Ok(())
    }

    #[tokio::test]
    async fn artifact_config_and_identity_must_be_paired() {
        let manager = manager();
        let mut launch = request("echo invalid-artifact-request");
        launch.output_artifacts =
            Some(echo_core::tools::artifact::ToolOutputArtifactConfig::default());
        assert!(launch_cell(&manager, launch).await.is_err());
    }

    #[tokio::test]
    async fn terminal_retention_converges_on_settlement_without_another_launch()
    -> std::result::Result<(), String> {
        let manager = BackgroundCommandManager::new(BackgroundCommandManagerConfig {
            max_terminal_history: 1,
            ..Default::default()
        })?;
        for sequence in 0_u64..3 {
            let cell_id = format!("terminal-{sequence}");
            let handle = test_handle_with_sequence(&cell_id, sequence);
            *handle.state.write().await = CellState {
                phase: CommandCellPhase::Succeeded,
                exit_code: Some(0),
                terminal_cause: Some(CommandCellTerminalCause::Exited),
                terminal_message: None,
            };
            handle.terminal_flag.store(true, Ordering::Release);
            manager.cells.insert(cell_id, handle);
            manager.prune_terminal_history();
        }

        let snapshots = manager.list().await;
        assert_eq!(snapshots.len(), 1);
        assert_eq!(
            snapshots.first().map(|snapshot| snapshot.cell_id.as_str()),
            Some("terminal-2")
        );
        Ok(())
    }

    #[tokio::test]
    async fn waiter_lease_protects_terminal_until_the_wait_round_is_acknowledged()
    -> std::result::Result<(), String> {
        let manager = BackgroundCommandManager::new(BackgroundCommandManagerConfig {
            max_terminal_history: 0,
            ..Default::default()
        })?;
        let handle = test_handle("leased-terminal");
        manager
            .cells
            .insert("leased-terminal".to_string(), handle.clone());
        let (_, lease) = manager
            .acquire_waiter_lease("leased-terminal")
            .map_err(|error| error.to_string())?;

        *handle.state.write().await = CellState {
            phase: CommandCellPhase::Succeeded,
            exit_code: Some(0),
            terminal_cause: Some(CommandCellTerminalCause::Exited),
            terminal_message: None,
        };
        handle.terminal_flag.store(true, Ordering::Release);
        manager.prune_terminal_history();
        assert!(manager.cells.contains_key("leased-terminal"));

        drop(lease);
        assert!(!manager.cells.contains_key("leased-terminal"));
        Ok(())
    }

    #[tokio::test]
    async fn artifact_decoder_preserves_utf8_split_across_pipe_reads()
    -> std::result::Result<(), String> {
        let root = tempfile::tempdir().map_err(|error| error.to_string())?;
        let handle = test_handle("split-utf8-artifact");
        {
            let mut artifact = handle
                .artifact
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            *artifact = CellArtifactState {
                writer: Some(ToolOutputArtifactWriter::new(
                    echo_core::tools::artifact::ToolOutputArtifactConfig::new(root.path(), "test")
                        .threshold_bytes(1),
                    echo_core::tools::artifact::ToolOutputArtifactIdentity {
                        conversation_id: Some("conversation".to_string()),
                        run_id: Some("run".to_string()),
                        call_id: "split-utf8".to_string(),
                        tool_name: "shell".to_string(),
                    },
                )),
                stdout_decoder: IncrementalUtf8Decoder::default(),
                stderr_decoder: IncrementalUtf8Decoder::default(),
                reference: None,
                status: CommandCellArtifactStatus::Writing,
                message: None,
            };
        }

        append_output(&handle, &[0xe4, 0xb8], 1024, ToolOutputChannel::Stdout);
        append_output(&handle, &[0xad], 1024, ToolOutputChannel::Stdout);
        assert_eq!(
            finish_output_artifact(
                &handle,
                Instant::now() + Duration::from_secs(5),
                &CancellationToken::new(),
                None,
            )
            .await,
            ArtifactFinalizeOutcome::Finished
        );
        let snapshot = handle.snapshot().await;
        let artifact = snapshot
            .output_artifact
            .ok_or_else(|| "split UTF-8 output did not create an artifact".to_string())?;
        let persisted =
            std::fs::read_to_string(&artifact.path).map_err(|error| error.to_string())?;
        assert!(persisted.contains('中'));
        assert!(!persisted.contains('\u{fffd}'));
        assert_eq!(snapshot.total_output_bytes, 3);
        Ok(())
    }

    #[tokio::test]
    async fn launch_rejects_empty_command() {
        let manager = manager();
        assert!(launch_cell(&manager, request("   ")).await.is_err());
        let mut zero_timeout = request("echo invalid-timeout");
        zero_timeout.timeout_secs = Some(0);
        assert!(matches!(
            manager.launch(zero_timeout).await,
            Err(CommandCellError::Validation { .. })
        ));
    }

    #[test]
    fn launch_without_tokio_runtime_returns_typed_error() {
        let manager = manager();
        let result =
            futures::executor::block_on(manager.prepare_launch(request("echo no-runtime")));
        assert!(matches!(result, Err(CommandCellError::Runtime { .. })));
    }

    #[tokio::test]
    async fn prepared_launch_cannot_execute_before_start_and_drop_auto_aborts()
    -> std::result::Result<(), String> {
        let sandbox = Arc::new(TestSandbox {
            executions: AtomicU64::new(0),
        });
        let manager = BackgroundCommandManager::new_with_sandbox(
            BackgroundCommandManagerConfig::default(),
            sandbox.clone(),
        )?;
        let mut first_request = request("prepared-first");
        first_request.require_sandbox = true;
        let first = manager
            .prepare_launch(first_request)
            .await
            .map_err(|error| error.to_string())?;
        let first_id = first.receipt().cell_id.clone();
        let prepared = manager
            .wait(&first_id, 0, 0)
            .await
            .map_err(|error| error.to_string())?;
        assert_eq!(prepared.snapshot.phase, CommandCellPhase::Prepared);
        assert_eq!(sandbox.executions.load(Ordering::Acquire), 0);

        let _receipt = manager
            .start_prepared(first)
            .await
            .map_err(|error| error.to_string())?;
        let (_, terminal) = drain_to_terminal(&manager, &first_id).await;
        assert_eq!(terminal.snapshot.phase, CommandCellPhase::Succeeded);
        assert_eq!(sandbox.executions.load(Ordering::Acquire), 1);

        let foreign_manager = BackgroundCommandManager::default();
        let foreign = foreign_manager
            .prepare_launch(request("foreign-manager"))
            .await
            .map_err(|error| error.to_string())?;
        let foreign_id = foreign.receipt().cell_id.clone();
        assert!(manager.start_prepared(foreign).await.is_err());
        /* 被拒绝的预留会被丢弃，并在其所属管理器上中止。
        它的回执不可能出现在本管理器的注册表中。 */
        assert!(!manager.cells.contains_key(&foreign_id));
        let (_, foreign_terminal) = drain_to_terminal(&foreign_manager, &foreign_id).await;
        assert_eq!(
            foreign_terminal.snapshot.phase,
            CommandCellPhase::LaunchFailed
        );

        let mut dropped_request = request("prepared-dropped");
        dropped_request.require_sandbox = true;
        let dropped = manager
            .prepare_launch(dropped_request)
            .await
            .map_err(|error| error.to_string())?;
        let dropped_id = dropped.receipt().cell_id.clone();
        drop(dropped);
        let (_, dropped_terminal) = drain_to_terminal(&manager, &dropped_id).await;
        assert_eq!(
            dropped_terminal.snapshot.phase,
            CommandCellPhase::LaunchFailed
        );
        assert_eq!(sandbox.executions.load(Ordering::Acquire), 1);
        Ok(())
    }

    #[tokio::test]
    async fn total_tracked_capacity_backpressures_prepared_launches()
    -> std::result::Result<(), String> {
        let manager = Arc::new(BackgroundCommandManager::new(
            BackgroundCommandManagerConfig {
                max_concurrent: 1,
                max_terminal_history: 1,
                ..Default::default()
            },
        )?);
        let first = manager
            .prepare_launch(request("prepared-capacity-1"))
            .await
            .map_err(|error| error.to_string())?;
        let second = manager
            .prepare_launch(request("prepared-capacity-2"))
            .await
            .map_err(|error| error.to_string())?;
        assert_eq!(manager.cells.len(), 2);
        assert_eq!(manager.tracked.available_permits(), 0);

        let third_manager = manager.clone();
        let third = tokio::spawn(async move {
            third_manager
                .prepare_launch(request("prepared-capacity-3"))
                .await
        });
        tokio::task::yield_now().await;
        assert!(!third.is_finished());

        drop(first);
        drop(second);
        let third = tokio::time::timeout(Duration::from_secs(2), third)
            .await
            .map_err(|_| "tracked capacity did not converge".to_string())?
            .map_err(|error| error.to_string())?
            .map_err(|error| error.to_string())?;
        assert!(manager.cells.len() <= 2);
        drop(third);
        manager
            .shutdown()
            .await
            .map_err(|error| error.to_string())?;
        Ok(())
    }

    #[tokio::test]
    async fn queued_launch_timeout_never_enters_the_sandbox() -> std::result::Result<(), String> {
        let sandbox = Arc::new(BlockingSandbox {
            executions: AtomicU64::new(0),
            started: Notify::new(),
            release: Notify::new(),
        });
        let manager = BackgroundCommandManager::new_with_sandbox(
            BackgroundCommandManagerConfig {
                max_concurrent: 1,
                ..Default::default()
            },
            sandbox.clone(),
        )?;
        let mut occupying_request = request("occupying");
        occupying_request.require_sandbox = true;
        occupying_request.timeout_secs = Some(10);
        let occupying = manager
            .launch(occupying_request)
            .await
            .map_err(|error| error.to_string())?;
        tokio::time::timeout(Duration::from_secs(2), sandbox.started.notified())
            .await
            .map_err(|_| "occupying sandbox launch did not start".to_string())?;

        let mut queued_request = request("must-not-enter-sandbox");
        queued_request.require_sandbox = true;
        queued_request.timeout_secs = Some(1);
        let queued = manager
            .launch(queued_request)
            .await
            .map_err(|error| error.to_string())?;
        let (_, queued_terminal) = drain_to_terminal(&manager, &queued.cell_id).await;
        assert_eq!(
            queued_terminal.snapshot.terminal_cause,
            Some(CommandCellTerminalCause::TimedOut)
        );
        assert_eq!(sandbox.executions.load(Ordering::Acquire), 1);

        sandbox.release.notify_waiters();
        let (_, occupying_terminal) = drain_to_terminal(&manager, &occupying.cell_id).await;
        assert_eq!(
            occupying_terminal.snapshot.phase,
            CommandCellPhase::Succeeded
        );
        Ok(())
    }

    #[tokio::test]
    async fn observation_lease_retains_terminal_across_multiple_drain_rounds()
    -> std::result::Result<(), String> {
        let manager = BackgroundCommandManager::new(BackgroundCommandManagerConfig {
            max_terminal_history: 0,
            ..Default::default()
        })?;
        let mut launch_request = request("sleep 0.1; seq 1 5000");
        launch_request.timeout_secs = Some(10);
        let receipt = manager
            .launch(launch_request)
            .await
            .map_err(|error| error.to_string())?;
        let lease = manager
            .observe(&receipt.cell_id)
            .map_err(|error| error.to_string())?;
        let (_, terminal) = drain_to_terminal(&manager, &receipt.cell_id).await;
        assert!(terminal.snapshot.phase.is_terminal());
        assert!(manager.cells.contains_key(&receipt.cell_id));
        let first = manager
            .wait(&receipt.cell_id, 0, 0)
            .await
            .map_err(|error| error.to_string())?;
        assert!(first.next_cursor < first.snapshot.total_output_bytes);

        let mut cursor = first.next_cursor;
        while cursor < first.snapshot.total_output_bytes {
            let delta = manager
                .wait(&receipt.cell_id, cursor, 0)
                .await
                .map_err(|error| error.to_string())?;
            assert!(delta.next_cursor > cursor);
            cursor = delta.next_cursor;
        }
        drop(lease);
        assert!(!manager.cells.contains_key(&receipt.cell_id));
        Ok(())
    }

    #[tokio::test]
    async fn shutdown_terminalizes_prepared_and_running_cells_and_rejects_launch()
    -> std::result::Result<(), String> {
        let manager = BackgroundCommandManager::default();
        let running = manager
            .launch(request("sleep 30"))
            .await
            .map_err(|error| error.to_string())?;
        let prepared = manager
            .prepare_launch(request("echo never-started"))
            .await
            .map_err(|error| error.to_string())?;
        let prepared_id = prepared.receipt().cell_id.clone();

        manager
            .shutdown()
            .await
            .map_err(|error| error.to_string())?;
        manager
            .shutdown()
            .await
            .map_err(|error| error.to_string())?;
        let running_state = manager
            .wait(&running.cell_id, 0, 0)
            .await
            .map_err(|error| error.to_string())?;
        let prepared_state = manager
            .wait(&prepared_id, 0, 0)
            .await
            .map_err(|error| error.to_string())?;
        assert!(running_state.snapshot.phase.is_terminal());
        assert!(prepared_state.snapshot.phase.is_terminal());
        assert!(matches!(
            manager.launch(request("echo rejected")).await,
            Err(CommandCellError::Shutdown)
        ));
        assert!(manager.start_prepared(prepared).await.is_err());
        Ok(())
    }

    #[tokio::test]
    async fn shutdown_aborts_blocking_artifact_finalizer_at_deadline()
    -> std::result::Result<(), String> {
        let root = tempfile::tempdir().map_err(|error| error.to_string())?;
        let manager = BackgroundCommandManager::default();
        let mut launch = request("printf finalizer-output");
        launch.timeout_secs = Some(30);
        launch.output_artifacts = Some(
            echo_core::tools::artifact::ToolOutputArtifactConfig::new(root.path(), "test")
                .threshold_bytes(1),
        );
        launch.artifact_identity = Some(echo_core::tools::artifact::ToolOutputArtifactIdentity {
            conversation_id: Some("conversation".to_string()),
            run_id: Some("run".to_string()),
            call_id: "call".to_string(),
            tool_name: "shell".to_string(),
        });
        let reservation = manager
            .prepare_launch(launch)
            .await
            .map_err(|error| error.to_string())?;
        let cell_id = reservation.receipt().cell_id.clone();
        let hook = Arc::new(TestArtifactFinalizeHook {
            entered: Notify::new(),
            release: Notify::new(),
        });
        let entry = manager
            .cells
            .get(&cell_id)
            .ok_or_else(|| "prepared cell was not published".to_string())?;
        *entry
            .artifact_finalize_hook
            .lock()
            .unwrap_or_else(|error| error.into_inner()) = Some(hook.clone());
        drop(entry);
        let _receipt = manager
            .start_prepared(reservation)
            .await
            .map_err(|error| error.to_string())?;
        tokio::time::timeout(Duration::from_secs(2), hook.entered.notified())
            .await
            .map_err(|_| "artifact finalizer did not enter the injected wait".to_string())?;

        tokio::time::timeout(Duration::from_secs(2), manager.shutdown())
            .await
            .map_err(|_| "shutdown waited for the blocked artifact finalizer".to_string())?
            .map_err(|error| error.to_string())?;
        let terminal = manager
            .wait(&cell_id, 0, 0)
            .await
            .map_err(|error| error.to_string())?;
        assert_eq!(terminal.snapshot.phase, CommandCellPhase::Cancelled);
        assert_eq!(
            terminal.snapshot.terminal_cause,
            Some(CommandCellTerminalCause::Cancelled)
        );
        assert_eq!(
            terminal.snapshot.artifact_status,
            CommandCellArtifactStatus::Failed
        );
        assert!(
            terminal
                .snapshot
                .artifact_message
                .as_deref()
                .is_some_and(|message| message.contains("shutdown"))
        );
        assert_eq!(manager.tasks.len(), 0);
        Ok(())
    }

    #[tokio::test]
    async fn owner_cancellation_aborts_blocking_artifact_finalizer()
    -> std::result::Result<(), String> {
        let root = tempfile::tempdir().map_err(|error| error.to_string())?;
        let manager = BackgroundCommandManager::default();
        let owner_cancel = Arc::new(CancellationToken::new());
        let (cell_id, _hook) = launch_with_blocking_artifact_finalizer(
            &manager,
            root.path(),
            Some(owner_cancel.clone()),
            "owner-cancel",
        )
        .await?;

        let running = manager
            .wait(&cell_id, 0, 0)
            .await
            .map_err(|error| error.to_string())?;
        assert_eq!(running.snapshot.phase, CommandCellPhase::Running);
        owner_cancel.cancel();
        let terminal = tokio::time::timeout(
            Duration::from_secs(2),
            manager.wait(&cell_id, running.next_cursor, 1_000),
        )
        .await
        .map_err(|_| "owner cancellation waited for the blocked artifact finalizer".to_string())?
        .map_err(|error| error.to_string())?;
        assert_eq!(terminal.snapshot.phase, CommandCellPhase::Cancelled);
        assert_eq!(
            terminal.snapshot.terminal_cause,
            Some(CommandCellTerminalCause::Cancelled)
        );
        assert_eq!(
            terminal.snapshot.artifact_status,
            CommandCellArtifactStatus::Failed
        );
        assert!(
            terminal
                .snapshot
                .artifact_message
                .as_deref()
                .is_some_and(|message| message.contains("command cell"))
        );
        assert_eq!(manager.tasks.len(), 0);
        Ok(())
    }

    #[tokio::test]
    async fn stop_aborts_blocking_artifact_finalizer() -> std::result::Result<(), String> {
        let root = tempfile::tempdir().map_err(|error| error.to_string())?;
        let manager = BackgroundCommandManager::default();
        let (cell_id, _hook) =
            launch_with_blocking_artifact_finalizer(&manager, root.path(), None, "cell-stop")
                .await?;
        let running = manager
            .wait(&cell_id, 0, 0)
            .await
            .map_err(|error| error.to_string())?;
        assert_eq!(running.snapshot.phase, CommandCellPhase::Running);

        assert!(manager.stop(&cell_id));
        let terminal = tokio::time::timeout(
            Duration::from_secs(2),
            manager.wait(&cell_id, running.next_cursor, 1_000),
        )
        .await
        .map_err(|_| "cell stop waited for the blocked artifact finalizer".to_string())?
        .map_err(|error| error.to_string())?;
        assert_eq!(terminal.snapshot.phase, CommandCellPhase::Cancelled);
        assert_eq!(
            terminal.snapshot.terminal_cause,
            Some(CommandCellTerminalCause::Cancelled)
        );
        assert_eq!(
            terminal.snapshot.artifact_status,
            CommandCellArtifactStatus::Failed
        );
        assert!(
            terminal
                .snapshot
                .artifact_message
                .as_deref()
                .is_some_and(|message| message.contains("command cell"))
        );
        assert_eq!(manager.tasks.len(), 0);
        Ok(())
    }

    #[tokio::test]
    async fn retention_rechecks_a_new_lease_before_removal() -> std::result::Result<(), String> {
        let manager = BackgroundCommandManager::new(BackgroundCommandManagerConfig {
            max_terminal_history: 0,
            ..Default::default()
        })?;
        let handle = test_handle_with_sequence("racing-terminal", 0);
        *handle.state.write().await = CellState {
            phase: CommandCellPhase::Succeeded,
            exit_code: Some(0),
            terminal_cause: Some(CommandCellTerminalCause::Exited),
            terminal_message: None,
        };
        handle.terminal_flag.store(true, Ordering::Release);
        manager.cells.insert("racing-terminal".to_string(), handle);

        /* 先快照候选，然后在生产清理逻辑“扫描”与“移除”之间的同一窗口内获取租约。 */
        let candidates = terminal_retention_candidates(&manager.cells);
        assert_eq!(
            candidates.first().map(|(_, id)| id.as_str()),
            Some("racing-terminal")
        );
        let (_, lease) = manager
            .acquire_waiter_lease("racing-terminal")
            .map_err(|error| error.to_string())?;
        remove_terminal_candidate(&manager.cells, "racing-terminal");
        assert!(manager.cells.contains_key("racing-terminal"));

        drop(lease);
        assert!(!manager.cells.contains_key("racing-terminal"));
        Ok(())
    }

    #[tokio::test]
    async fn inherited_pipe_cannot_outlive_the_cell_deadline() -> std::result::Result<(), String> {
        let manager = manager();
        let mut launch_request = request("sleep 30 &");
        launch_request.timeout_secs = Some(1);
        let started = Instant::now();
        let cell_id = launch_cell(&manager, launch_request).await?;
        let (_, terminal) = drain_to_terminal(&manager, &cell_id).await;
        assert_eq!(
            terminal.snapshot.terminal_cause,
            Some(CommandCellTerminalCause::TimedOut)
        );
        assert!(started.elapsed() < Duration::from_secs(5));
        Ok(())
    }

    #[tokio::test]
    async fn list_contains_running_and_terminal_cells() {
        let manager = manager();
        let quick = launch_cell(&manager, request("echo quick")).await.unwrap();
        let slow = launch_cell(&manager, request("sleep 30")).await.unwrap();

        let (_, done) = drain_to_terminal(&manager, &quick).await;
        assert_eq!(done.snapshot.phase, CommandCellPhase::Succeeded);

        let listed = manager.list().await;
        let quick_snapshot = listed.iter().find(|s| s.cell_id == quick);
        let slow_snapshot = listed.iter().find(|s| s.cell_id == slow);
        assert!(quick_snapshot.is_some_and(|s| s.phase.is_terminal()));
        assert!(slow_snapshot.is_some_and(|s| !s.phase.is_terminal()));

        manager.stop(&slow);
        let _ = manager.wait(&slow, 0, 10_000).await;
    }
}
