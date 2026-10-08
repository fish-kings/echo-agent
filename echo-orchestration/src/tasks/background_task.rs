/*!
后台任务句柄与派生器，用于支持长时间运行的任务。

本模块为"发射后不管"、可轮询、可取消的后台 future 提供一个非阻塞、进程内的抽象。

# 核心类型

- [`BackgroundTask<T>`] —— 已派生异步任务的句柄。提供非阻塞状态轮询、带超时的阻塞等待以及取消能力。

- [`BackgroundTaskStatus`] —— 生命周期状态：Pending → Running → Completed/Failed/Cancelled。

- [`AnyBackgroundTask`] —— 类型擦除的 trait，用于在同一集合中存放异构任务句柄（例如状态看板）。

- [`TaskSpawner`] —— 进程内派生器，通过信号量（Semaphore）控制并发并跟踪已派生的 future。
  持久化任务图由 [`crate::tasks::TaskRevisionService`] 负责。

# 示例

```rust,ignore
use echo_orchestration::tasks::{TaskSpawner, TaskSpawnerConfig};
use std::sync::Arc;

async fn example() {
    let spawner = Arc::new(TaskSpawner::new(TaskSpawnerConfig::default()));

    // Spawn a background task
    let handle = spawner.spawn("fetch-data", async {
        // Long-running work...
        Ok("result data".to_string())
    });

    // Non-blocking status check
    println!("Status: {:?}", handle.status());

    // Block until done (with optional timeout)
    let result = handle.wait(Some(std::time::Duration::from_secs(30))).await;
    println!("Result: {:?}", result);
}
```
*/

use dashmap::DashMap;
use echo_core::error::{ReactError, Result};
use std::future::Future;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::Notify;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info};

// ── BackgroundTaskStatus ──────────────────────────────────────────

/// 后台任务的生命周期状态。
#[derive(Debug, Clone)]
pub enum BackgroundTaskStatus {
    /// 后台 future 已排队，尚未开始执行。
    Pending,
    /// 后台 future 正在执行。
    Running {
        /// 任务开始执行的时刻。
        started_at: Instant,
    },
    /// 后台 future 已成功完成。
    Completed {
        /// 任务结束的时刻。
        finished_at: Instant,
    },
    /// 后台 future 因错误而失败。
    Failed {
        /// 人类可读的错误描述。
        error: String,
        /// 失败发生的时刻。
        at: Instant,
    },
    /// 后台 future 已通过其取消令牌被取消。
    Cancelled,
}

impl BackgroundTaskStatus {
    /// 该状态是否为终态。
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            BackgroundTaskStatus::Completed { .. }
                | BackgroundTaskStatus::Failed { .. }
                | BackgroundTaskStatus::Cancelled
        )
    }

    /// 用于展示的简短文本描述。
    pub fn as_str(&self) -> &'static str {
        match self {
            BackgroundTaskStatus::Pending => "pending",
            BackgroundTaskStatus::Running { .. } => "running",
            BackgroundTaskStatus::Completed { .. } => "completed",
            BackgroundTaskStatus::Failed { .. } => "failed",
            BackgroundTaskStatus::Cancelled => "cancelled",
        }
    }
}

// ── BackgroundTask<T> ─────────────────────────────────────────────

/// 句柄共享的内部状态：生命周期状态与单消费者结果。
struct BackgroundTaskHandleState<T> {
    /// 当前生命周期状态。
    status: BackgroundTaskStatus,
    /// 任务结果，仅能被第一个到达终态的等待者取走。
    result: Option<Result<T>>,
    /// 执行子任务是否发生 panic。
    panicked: bool,
}

/// 类型擦除的状态来源，用于在不知道 `T` 的情况下读取状态快照。
trait BackgroundTaskStatusSource: Send + Sync {
    /// 返回当前状态的快照。
    fn snapshot(&self) -> BackgroundTaskStatus;
}

impl<T: Send + 'static> BackgroundTaskStatusSource for Mutex<BackgroundTaskHandleState<T>> {
    fn snapshot(&self) -> BackgroundTaskStatus {
        self.lock()
            .unwrap_or_else(|error| error.into_inner())
            .status
            .clone()
    }
}

/**
可被轮询、等待或取消的后台任务句柄。

任务在 tokio 运行时上异步执行。句柄克隆开销很低（内部使用 `Arc`），
同一任务可以同时存在多个句柄。
*/
pub struct BackgroundTask<T: Send + 'static> {
    /// 进程内唯一的任务 ID。
    pub id: String,
    /// 人类可读的名称/描述。
    pub name: String,
    /// 生命周期状态与单消费者结果的唯一权威来源。
    state: Arc<Mutex<BackgroundTaskHandleState<T>>>,
    /// 终态被原子发布后触发的通知器。
    result_notify: Arc<Notify>,
    /// 用于请求取消运行中任务的令牌。
    cancel: CancellationToken,
}

impl<T: Send + 'static> Clone for BackgroundTask<T> {
    fn clone(&self) -> Self {
        Self {
            id: self.id.clone(),
            name: self.name.clone(),
            state: Arc::clone(&self.state),
            result_notify: Arc::clone(&self.result_notify),
            cancel: self.cancel.clone(),
        }
    }
}

impl<T: Send + 'static> BackgroundTask<T> {
    /// 当前状态的非阻塞快照。
    pub async fn status(&self) -> BackgroundTaskStatus {
        self.state
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .status
            .clone()
    }

    /// 便捷方法：检查任务是否仍在运行（非阻塞）。
    pub async fn is_running(&self) -> bool {
        matches!(self.status().await, BackgroundTaskStatus::Running { .. })
    }

    /// 便捷方法：检查任务是否已到达终态。
    pub async fn is_completed(&self) -> bool {
        self.status().await.is_terminal()
    }

    /**
    请求取消运行中的任务。

    该方法只触发取消令牌，并不保证立即终止。监督者会在准入或执行阶段观察到取消；
    若处于执行阶段，会先中止并等待子任务结束，然后再发布终态。
    */
    pub fn cancel(&self) {
        self.cancel.cancel();
    }

    /// 是否已请求取消。
    pub fn is_cancelled(&self) -> bool {
        self.cancel.is_cancelled()
    }

    /**
    检查任务是否发生了 panic。

    捕获到执行子任务 panic 时返回 `Some(true)`，其他终态返回 `Some(false)`，
    尚未到达终态时返回 `None`。
    */
    pub async fn is_panicked(&self) -> Option<bool> {
        let state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        match state.status {
            BackgroundTaskStatus::Pending | BackgroundTaskStatus::Running { .. } => None,
            BackgroundTaskStatus::Completed { .. }
            | BackgroundTaskStatus::Failed { .. }
            | BackgroundTaskStatus::Cancelled => Some(state.panicked),
        }
    }

    /**
    等待任务完成并返回结果。

    若 `timeout` 为 `Some`，任务未在给定时长内完成时返回错误；若为 `None`，则无限等待。

    超时不会消耗结果。第一个到达终态的等待者取得 `T`；
    之后的等待者根据持久化的状态立即返回。
    */
    pub async fn wait(&self, timeout: Option<Duration>) -> Result<T> {
        let deadline = match timeout {
            Some(duration) => Some(
                tokio::time::Instant::now()
                    .checked_add(duration)
                    .ok_or_else(|| {
                        ReactError::Other(format!(
                            "Background task '{}' wait deadline overflow",
                            self.name
                        ))
                    })?,
            ),
            None => None,
        };
        loop {
            let notified = self.result_notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            let status = {
                let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
                if let Some(result) = state.result.take() {
                    return result;
                }
                state.status.clone()
            };
            match status {
                BackgroundTaskStatus::Completed { .. } => {
                    return Err(ReactError::Other(format!(
                        "Background task '{}' completed, but its result was already consumed by another waiter",
                        self.name
                    )));
                }
                BackgroundTaskStatus::Failed { error, .. } => {
                    return Err(ReactError::Other(format!(
                        "Background task '{}' failed: {error}",
                        self.name
                    )));
                }
                BackgroundTaskStatus::Cancelled => {
                    return Err(ReactError::Other(format!(
                        "Background task '{}' was cancelled",
                        self.name
                    )));
                }
                BackgroundTaskStatus::Pending | BackgroundTaskStatus::Running { .. } => {}
            }
            match deadline {
                Some(deadline) => {
                    if tokio::time::timeout_at(deadline, &mut notified)
                        .await
                        .is_err()
                    {
                        let duration = timeout.unwrap_or_default();
                        return Err(ReactError::Other(format!(
                            "Background task '{}' timed out after {:?} (retry-safe: call wait() again)",
                            self.name, duration
                        )));
                    }
                }
                None => (&mut notified).await,
            }
        }
    }
}

// ── AnyBackgroundTask ─────────────────────────────────────────────

/**
用于存放异构后台任务句柄的类型擦除 trait。

适用于状态看板、任务列表和取消管理等不需要具体结果类型 `T` 的场景。
*/
pub trait AnyBackgroundTask: Send + Sync {
    /// 唯一任务 ID。
    fn id(&self) -> &str;
    /// 人类可读的任务名称。
    fn name(&self) -> &str;
    /// 当前状态的文本形式。
    fn status_text(&self) -> &'static str;
    /**
    当前状态的非阻塞快照。

    返回与带类型句柄共享的同一状态中的实际 [`BackgroundTaskStatus`] 变体。
    */
    fn status_snapshot(&self) -> BackgroundTaskStatus;
    /// 请求取消。
    fn cancel(&self);
    /// 任务是否已到达终态（同步检查）。
    fn is_terminal_sync(&self) -> bool;
    /// 单调递增的注册序号，用于有界历史保留。
    fn sequence(&self) -> u64;
}

/// 带类型句柄共享生命周期状态的类型擦除实时视图。
struct TypeErasedTask {
    /// 任务 ID。
    id: String,
    /// 任务名称。
    name: String,
    /// 与 BackgroundTask<T> 持有的同一泛型状态的类型擦除视图。
    state: Arc<dyn BackgroundTaskStatusSource>,
    /// 取消令牌。
    cancel: CancellationToken,
    /// 注册序号。
    sequence: u64,
}

impl AnyBackgroundTask for TypeErasedTask {
    fn id(&self) -> &str {
        &self.id
    }
    fn name(&self) -> &str {
        &self.name
    }
    fn status_text(&self) -> &'static str {
        self.state.snapshot().as_str()
    }
    fn status_snapshot(&self) -> BackgroundTaskStatus {
        self.state.snapshot()
    }
    fn cancel(&self) {
        self.cancel.cancel();
    }
    fn is_terminal_sync(&self) -> bool {
        self.state.snapshot().is_terminal()
    }
    fn sequence(&self) -> u64 {
        self.sequence
    }
}

// ── TaskSummary ───────────────────────────────────────────────────

/// 用于列表/看板的后台任务轻量摘要。
#[derive(Debug, Clone)]
pub struct TaskSummary {
    /// 任务 ID。
    pub id: String,
    /// 任务名称。
    pub name: String,
    /// 当前状态快照。
    pub status: BackgroundTaskStatus,
}

// ── TaskSpawnerConfig ─────────────────────────────────────────────

/// [`TaskSpawner`] 的配置。
#[derive(Debug, Clone)]
pub struct TaskSpawnerConfig {
    /// 同时运行的后台任务数上限。
    pub max_concurrent: usize,
    /// 派生任务的默认超时秒数（0 表示不超时）。
    pub default_timeout_secs: u64,
    /**
    为状态/列表查询保留的终态任务摘要数量上限。
    运行中与等待中的任务永远不会因保留策略被移除。
    */
    pub max_terminal_history: usize,
}

impl Default for TaskSpawnerConfig {
    fn default() -> Self {
        Self {
            max_concurrent: 16,
            default_timeout_secs: 300, // 5 分钟
            max_terminal_history: 256,
        }
    }
}

// ── TaskSpawner ───────────────────────────────────────────────────

/**
带并发控制的系统级后台任务派生器。

所有派生的 future 都记录在并发映射中，可被列出、取消或检查。
这个进程内句柄注册表不负责持久化任务关系，也不负责重启恢复。
*/
pub struct TaskSpawner {
    /// 所有被跟踪的任务（类型擦除）。
    tasks: Arc<DashMap<String, Arc<dyn AnyBackgroundTask>>>,
    /// 并发限制器。
    semaphore: Arc<tokio::sync::Semaphore>,
    /// 配置。
    config: TaskSpawnerConfig,
    /// 单调递增的注册序号，用于确定性地保留终态任务。
    next_sequence: AtomicU64,
}

/// 原子发布终态结果并唤醒所有等待者；若已处于终态则忽略。
fn publish_terminal<T: Send + 'static>(
    state: &Arc<Mutex<BackgroundTaskHandleState<T>>>,
    notify: &Arc<Notify>,
    result: Result<T>,
    status: BackgroundTaskStatus,
    panicked: bool,
) {
    {
        let mut state = state.lock().unwrap_or_else(|error| error.into_inner());
        if state.status.is_terminal() {
            return;
        }
        state.result = Some(result);
        state.status = status;
        state.panicked = panicked;
    }
    notify.notify_waiters();
}

/// 将状态从 Pending 切换为 Running。
fn publish_running<T: Send + 'static>(state: &Arc<Mutex<BackgroundTaskHandleState<T>>>) {
    let mut state = state.lock().unwrap_or_else(|error| error.into_inner());
    if matches!(state.status, BackgroundTaskStatus::Pending) {
        state.status = BackgroundTaskStatus::Running {
            started_at: Instant::now(),
        };
    }
}

/// 将执行子任务的 join 结果转换为 (结果, 终态, 是否 panic)。
fn completion_from_join<T: Send + 'static>(
    joined: std::result::Result<Result<T>, tokio::task::JoinError>,
) -> (Result<T>, BackgroundTaskStatus, bool) {
    match joined {
        Ok(result) => {
            let status = match result.as_ref() {
                Ok(_) => BackgroundTaskStatus::Completed {
                    finished_at: Instant::now(),
                },
                Err(error) => BackgroundTaskStatus::Failed {
                    error: error.to_string(),
                    at: Instant::now(),
                },
            };
            (result, status, false)
        }
        Err(error) => {
            let panicked = error.is_panic();
            let message = if panicked {
                format!("Background execution task panicked: {error}")
            } else {
                format!("Background execution task failed to join: {error}")
            };
            (
                Err(ReactError::Other(message.clone())),
                BackgroundTaskStatus::Failed {
                    error: message,
                    at: Instant::now(),
                },
                panicked,
            )
        }
    }
}

/// 中止执行子任务并等待其真正结束。
async fn abort_execution_task<T: Send + 'static>(task: &mut JoinHandle<Result<T>>) {
    task.abort();
    let _ = task.await;
}

/// 监督执行子任务，处理取消与超时，返回 (结果, 终态, 是否 panic)。
async fn supervise_execution<T: Send + 'static>(
    mut execution_task: JoinHandle<Result<T>>,
    cancel: CancellationToken,
    deadline: Option<tokio::time::Instant>,
    timeout: Option<Duration>,
) -> (Result<T>, BackgroundTaskStatus, bool) {
    if let Some(deadline) = deadline {
        tokio::select! {
            biased;
            _ = cancel.cancelled() => {
                abort_execution_task(&mut execution_task).await;
                (
                    Err(ReactError::Other("Task cancelled".to_string())),
                    BackgroundTaskStatus::Cancelled,
                    false,
                )
            }
            _ = tokio::time::sleep_until(deadline) => {
                abort_execution_task(&mut execution_task).await;
                let duration = timeout.unwrap_or_default();
                let message = format!("Task timed out after {duration:?}");
                (
                    Err(ReactError::Other(message.clone())),
                    BackgroundTaskStatus::Failed {
                        error: message,
                        at: Instant::now(),
                    },
                    false,
                )
            }
            joined = &mut execution_task => completion_from_join(joined),
        }
    } else {
        tokio::select! {
            biased;
            _ = cancel.cancelled() => {
                abort_execution_task(&mut execution_task).await;
                (
                    Err(ReactError::Other("Task cancelled".to_string())),
                    BackgroundTaskStatus::Cancelled,
                    false,
                )
            }
            joined = &mut execution_task => completion_from_join(joined),
        }
    }
}

impl TaskSpawner {
    /// 使用给定配置创建新的任务派生器。
    pub fn new(config: TaskSpawnerConfig) -> Self {
        let semaphore = Arc::new(tokio::sync::Semaphore::new(config.max_concurrent));
        Self {
            tasks: Arc::new(DashMap::new()),
            semaphore,
            config,
            next_sequence: AtomicU64::new(0),
        }
    }

    /**
    将一个 future 作为后台任务派生，并立即返回句柄。

    任务在开始前需获取信号量许可；若许可已耗尽，任务会排队直到有可用许可。

    # 参数
    * `name` —— 人类可读的任务名称（用于日志和列表）
    * `fut` —— 要执行的异步工作
    */
    pub fn spawn<F, T>(&self, name: &str, fut: F) -> BackgroundTask<T>
    where
        F: Future<Output = Result<T>> + Send + 'static,
        T: Send + 'static,
    {
        self.prune_terminal_history();
        let id = uuid::Uuid::new_v4().to_string();
        let name = name.to_string();
        let cancel = CancellationToken::new();
        let state = Arc::new(Mutex::new(BackgroundTaskHandleState {
            status: BackgroundTaskStatus::Pending,
            result: None,
            panicked: false,
        }));
        let result_notify = Arc::new(Notify::new());

        let permit = self.semaphore.clone();
        let cancel_inner = cancel.clone();
        let state_inner = Arc::clone(&state);
        let id_inner = id.clone();
        let name_inner = name.clone();
        let max_concurrent = self.config.max_concurrent;
        let timeout = if self.config.default_timeout_secs > 0 {
            Some(Duration::from_secs(self.config.default_timeout_secs))
        } else {
            None
        };
        let accepted_at = tokio::time::Instant::now();
        let deadline = timeout.and_then(|duration| accepted_at.checked_add(duration));
        let deadline_overflow = timeout.is_some() && deadline.is_none();
        let notify = Arc::clone(&result_notify);

        let supervisor = tokio::spawn(async move {
            if max_concurrent == 0 {
                let message = "TaskSpawner max_concurrent must be greater than zero".to_string();
                publish_terminal(
                    &state_inner,
                    &notify,
                    Err(ReactError::Other(message.clone())),
                    BackgroundTaskStatus::Failed {
                        error: message,
                        at: Instant::now(),
                    },
                    false,
                );
                return;
            }
            if deadline_overflow {
                let message = "Background task deadline overflow".to_string();
                publish_terminal(
                    &state_inner,
                    &notify,
                    Err(ReactError::Other(message.clone())),
                    BackgroundTaskStatus::Failed {
                        error: message,
                        at: Instant::now(),
                    },
                    false,
                );
                return;
            }

            let acquire = permit.acquire_owned();
            tokio::pin!(acquire);
            let _permit = if let Some(deadline) = deadline {
                tokio::select! {
                    biased;
                    _ = cancel_inner.cancelled() => {
                        publish_terminal(
                            &state_inner,
                            &notify,
                            Err(ReactError::Other("Task cancelled before start".to_string())),
                            BackgroundTaskStatus::Cancelled,
                            false,
                        );
                        return;
                    }
                    _ = tokio::time::sleep_until(deadline) => {
                        let duration = timeout.unwrap_or_default();
                        let message = format!("Task timed out during admission after {duration:?}");
                        publish_terminal(
                            &state_inner,
                            &notify,
                            Err(ReactError::Other(message.clone())),
                            BackgroundTaskStatus::Failed {
                                error: message,
                                at: Instant::now(),
                            },
                            false,
                        );
                        return;
                    }
                    acquired = &mut acquire => acquired,
                }
            } else {
                tokio::select! {
                    biased;
                    _ = cancel_inner.cancelled() => {
                        publish_terminal(
                            &state_inner,
                            &notify,
                            Err(ReactError::Other("Task cancelled before start".to_string())),
                            BackgroundTaskStatus::Cancelled,
                            false,
                        );
                        return;
                    }
                    acquired = &mut acquire => acquired,
                }
            };
            let _permit = match _permit {
                Ok(permit) => permit,
                Err(_) => {
                    let message = "Semaphore closed - cannot acquire permit".to_string();
                    publish_terminal(
                        &state_inner,
                        &notify,
                        Err(ReactError::Other(message.clone())),
                        BackgroundTaskStatus::Failed {
                            error: message,
                            at: Instant::now(),
                        },
                        false,
                    );
                    return;
                }
            };

            if cancel_inner.is_cancelled() {
                publish_terminal(
                    &state_inner,
                    &notify,
                    Err(ReactError::Other("Task cancelled before start".to_string())),
                    BackgroundTaskStatus::Cancelled,
                    false,
                );
                return;
            }
            if deadline.is_some_and(|deadline| tokio::time::Instant::now() >= deadline) {
                let duration = timeout.unwrap_or_default();
                let message = format!("Task timed out during admission after {duration:?}");
                publish_terminal(
                    &state_inner,
                    &notify,
                    Err(ReactError::Other(message.clone())),
                    BackgroundTaskStatus::Failed {
                        error: message,
                        at: Instant::now(),
                    },
                    false,
                );
                return;
            }

            publish_running(&state_inner);
            debug!(task_id = %id_inner, name = %name_inner, "Background task started");
            let execution_task = tokio::spawn(fut);
            let (result, final_status, panicked) =
                supervise_execution(execution_task, cancel_inner, deadline, timeout).await;
            publish_terminal(
                &state_inner,
                &notify,
                result,
                final_status.clone(),
                panicked,
            );
            let status_text = final_status.as_str();
            info!(task_id = %id_inner, name = %name_inner, status = %status_text, "Background task finished");
        });
        std::mem::drop(supervisor);

        // 注册到类型擦除的任务映射中
        let erased_state: Arc<dyn BackgroundTaskStatusSource> = state.clone();
        let erased: Arc<dyn AnyBackgroundTask> = Arc::new(TypeErasedTask {
            id: id.clone(),
            name: name.clone(),
            state: erased_state,
            cancel: cancel.clone(),
            sequence: self.next_sequence.fetch_add(1, Ordering::Relaxed),
        });
        self.tasks.insert(id.clone(), erased);

        BackgroundTask {
            id,
            name,
            state,
            result_notify,
            cancel,
        }
    }

    /// 列出所有被跟踪的任务及其当前状态。
    pub async fn list(&self) -> Vec<TaskSummary> {
        let mut summaries = Vec::new();
        for entry in self.tasks.iter() {
            let task = entry.value();
            let status = task.status_snapshot();
            summaries.push(TaskSummary {
                id: task.id().to_string(),
                name: task.name().to_string(),
                status,
            });
        }
        summaries
    }

    /// 按 ID 取消指定任务。
    pub fn cancel(&self, id: &str) -> bool {
        if let Some(entry) = self.tasks.get(id) {
            entry.value().cancel();
            true
        } else {
            false
        }
    }

    /// 取消所有运行中的任务。
    pub fn cancel_all(&self) {
        for entry in self.tasks.iter() {
            entry.value().cancel();
        }
    }

    /// 从跟踪映射中移除已完成/失败的任务。
    pub fn prune_completed(&self) -> usize {
        let before = self.tasks.len();
        self.tasks.retain(|_, v| !v.is_terminal_sync());
        before - self.tasks.len()
    }

    /**
    最多保留配置数量的终态任务摘要。

    该方法在注册前执行，因此新派生的任务总会被跟踪。
    等待中/运行中的任务永远不会被选中移除，即使映射总量超过终态历史上限。
    */
    fn prune_terminal_history(&self) -> usize {
        let max_terminal_history = self.config.max_terminal_history;
        let mut terminal = self
            .tasks
            .iter()
            .filter(|entry| entry.value().is_terminal_sync())
            .map(|entry| (entry.value().sequence(), entry.key().clone()))
            .collect::<Vec<_>>();
        let remove_count = terminal.len().saturating_sub(max_terminal_history);
        if remove_count == 0 {
            return 0;
        }
        terminal.sort_by_key(|(sequence, _)| *sequence);
        for (_, id) in terminal.into_iter().take(remove_count) {
            self.tasks.remove(&id);
        }
        remove_count
    }

    /// 当前被跟踪的任务数量。
    pub fn task_count(&self) -> usize {
        self.tasks.len()
    }
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// 不可克隆的结果类型，用于验证单消费者语义。
    struct NonCloneResult(u32);

    /// Drop 探针：被析构时置位标志。
    struct DropProbe(Arc<std::sync::atomic::AtomicBool>);

    impl Drop for DropProbe {
        fn drop(&mut self) {
            self.0.store(true, Ordering::Release);
        }
    }

    /// 成功完成的任务可通过 wait 取得结果。
    #[tokio::test]
    async fn test_background_task_completes() -> Result<()> {
        let spawner = TaskSpawner::new(TaskSpawnerConfig::default());
        let handle = spawner.spawn("test-task", async { Ok(42) });

        assert_eq!(handle.name, "test-task");
        assert!(!handle.is_cancelled());

        let result = handle.wait(Some(Duration::from_secs(5))).await?;
        assert_eq!(result, 42);
        assert!(handle.is_completed().await);
        Ok(())
    }

    /// 失败任务的 wait 返回错误且状态为终态。
    #[tokio::test]
    async fn test_background_task_failure() {
        let spawner = TaskSpawner::new(TaskSpawnerConfig::default());
        let handle = spawner.spawn("failing-task", async {
            Err::<String, _>(ReactError::Other("intentional failure".into()))
        });

        let result = handle.wait(Some(Duration::from_secs(5))).await;
        assert!(result.is_err());
        assert!(handle.is_completed().await);
    }

    /// 普通失败即使错误信息带 panic 前缀，也不能被误判为 panic。
    #[tokio::test]
    async fn ordinary_failure_cannot_spoof_panic_provenance() -> Result<()> {
        let spawner = TaskSpawner::new(TaskSpawnerConfig::default());
        let handle = spawner.spawn("panic-prefix-failure", async {
            Err::<(), _>(ReactError::Other(
                "Background execution task panicked: forged".to_string(),
            ))
        });

        let result = handle.wait(Some(Duration::from_secs(1))).await;
        if result.is_ok()
            || !matches!(handle.status().await, BackgroundTaskStatus::Failed { .. })
            || handle.is_panicked().await != Some(false)
        {
            return Err(ReactError::Other(
                "ordinary failure was misclassified as an execution panic".to_string(),
            ));
        }
        Ok(())
    }

    /// 运行中的任务被取消后 wait 返回错误。
    #[tokio::test]
    async fn test_background_task_cancel() {
        let spawner = TaskSpawner::new(TaskSpawnerConfig::default());
        let handle = spawner.spawn("cancellable-task", async {
            // 模拟耗时工作
            tokio::time::sleep(Duration::from_secs(60)).await;
            Ok(())
        });

        // 给任务一点时间启动
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(handle.is_running().await);

        handle.cancel();

        let result = handle.wait(Some(Duration::from_secs(5))).await;
        assert!(result.is_err());
    }

    /// wait 超时返回错误。
    #[tokio::test]
    async fn test_background_task_timeout() {
        let spawner = TaskSpawner::new(TaskSpawnerConfig {
            default_timeout_secs: 0, // 无默认超时
            ..Default::default()
        });
        let handle = spawner.spawn("slow-task", async {
            tokio::time::sleep(Duration::from_secs(60)).await;
            Ok("done")
        });

        // 以较短超时等待——应当失败
        let result = handle.wait(Some(Duration::from_millis(100))).await;
        assert!(result.is_err());

        // 取消以便清理
        handle.cancel();
    }

    /**
    N-P2-5 回归测试：超时不得消耗结果。超时后再次以更长超时调用 `wait()`，
    在任务完成后仍须返回真实结果。旧的 oneshot 实现在超时后会永久丢失结果。
    */
    #[tokio::test]
    async fn test_background_task_timeout_then_retry_recovers_result() -> Result<()> {
        let spawner = TaskSpawner::new(TaskSpawnerConfig {
            default_timeout_secs: 0,
            ..Default::default()
        });
        let handle = spawner.spawn("retry-safe-task", async {
            tokio::time::sleep(Duration::from_millis(200)).await;
            Ok("recovered")
        });

        // 第一次等待：超时过短 → 报错，但结果不会丢失。
        let r1 = handle.wait(Some(Duration::from_millis(50))).await;
        assert!(r1.is_err(), "first wait should time out");

        // 第二次等待：时长足够 → 必须拿到真实结果（而非 "already consumed"）。
        let r2 = handle.wait(Some(Duration::from_secs(5))).await;
        assert_eq!(r2?, "recovered");
        Ok(())
    }

    /// 并发等待者都能观察到终态，且仅一个取得结果。
    #[tokio::test]
    async fn concurrent_waiters_both_reach_a_terminal_observation() -> Result<()> {
        use std::task::Poll;

        let spawner = TaskSpawner::new(TaskSpawnerConfig::default());
        let (release_tx, release_rx) = tokio::sync::oneshot::channel();
        let handle = spawner.spawn("multi-waiter", async move {
            release_rx
                .await
                .map_err(|_| ReactError::Other("release sender closed".to_string()))?;
            Ok(NonCloneResult(42))
        });
        let second_handle = handle.clone();
        let waits = async { tokio::join!(handle.wait(None), second_handle.wait(None)) };
        tokio::pin!(waits);
        if !matches!(futures::poll!(&mut waits), Poll::Pending) {
            return Err(ReactError::Other(
                "waiters completed before task release".to_string(),
            ));
        }
        release_tx
            .send(())
            .map_err(|_| ReactError::Other("release receiver closed".to_string()))?;
        let (first, second) = tokio::time::timeout(Duration::from_millis(300), &mut waits)
            .await
            .map_err(|_| ReactError::Other("a terminal waiter remained blocked".to_string()))?;
        let successes = usize::from(first.as_ref().is_ok_and(|value| value.0 == 42))
            .saturating_add(usize::from(
                second.as_ref().is_ok_and(|value| value.0 == 42),
            ));
        if successes != 1 || first.is_ok() == second.is_ok() {
            return Err(ReactError::Other(
                "single-consumer result was not paired with one terminal observer".to_string(),
            ));
        }
        let observer_error = first.err().or_else(|| second.err()).ok_or_else(|| {
            ReactError::Other("terminal observer did not receive an error".to_string())
        })?;
        if !observer_error.to_string().contains("already consumed") {
            return Err(ReactError::Other(format!(
                "terminal observer returned the wrong error: {observer_error}"
            )));
        }
        Ok(())
    }

    /// 排队中的任务被取消后直接结束，不会启动 future。
    #[tokio::test]
    async fn queued_cancellation_settles_without_starting_the_future() -> Result<()> {
        let spawner = TaskSpawner::new(TaskSpawnerConfig {
            max_concurrent: 1,
            default_timeout_secs: 0,
            ..TaskSpawnerConfig::default()
        });
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let release = Arc::new(Notify::new());
        let first_release = Arc::clone(&release);
        let first = spawner.spawn("permit-holder", async move {
            let _ = started_tx.send(());
            first_release.notified().await;
            Ok(())
        });
        started_rx
            .await
            .map_err(|_| ReactError::Other("permit holder did not start".to_string()))?;

        let execution_started = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let execution_started_inner = Arc::clone(&execution_started);
        let queued = spawner.spawn("queued-cancel", async move {
            execution_started_inner.store(true, Ordering::Release);
            Ok(())
        });
        queued.cancel();
        let result = tokio::time::timeout(Duration::from_millis(300), queued.wait(None))
            .await
            .map_err(|_| ReactError::Other("queued cancellation did not settle".to_string()))?;
        if result.is_ok()
            || execution_started.load(Ordering::Acquire)
            || !matches!(queued.status().await, BackgroundTaskStatus::Cancelled)
        {
            return Err(ReactError::Other(
                "queued cancellation started work or published the wrong terminal".to_string(),
            ));
        }
        release.notify_one();
        first.wait(Some(Duration::from_secs(1))).await?;
        Ok(())
    }

    /// 并发上限为 0 时发布配置失败。
    #[tokio::test]
    async fn zero_concurrency_settles_as_configuration_failure() -> Result<()> {
        let spawner = TaskSpawner::new(TaskSpawnerConfig {
            max_concurrent: 0,
            default_timeout_secs: 0,
            ..TaskSpawnerConfig::default()
        });
        let execution_started = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let execution_started_inner = Arc::clone(&execution_started);
        let handle = spawner.spawn("zero-concurrency", async move {
            execution_started_inner.store(true, Ordering::Release);
            Ok(())
        });

        let result = tokio::time::timeout(Duration::from_millis(300), handle.wait(None))
            .await
            .map_err(|_| ReactError::Other("zero concurrency remained pending".to_string()))?;
        if result.is_ok()
            || execution_started.load(Ordering::Acquire)
            || !matches!(handle.status().await, BackgroundTaskStatus::Failed { .. })
        {
            return Err(ReactError::Other(
                "zero concurrency did not publish a configuration failure".to_string(),
            ));
        }
        Ok(())
    }

    /// 排队期间到达截止时间则失败结束，不会启动 future。
    #[tokio::test]
    async fn queued_deadline_settles_without_starting_the_future() -> Result<()> {
        let spawner = TaskSpawner::new(TaskSpawnerConfig {
            max_concurrent: 1,
            default_timeout_secs: 1,
            ..TaskSpawnerConfig::default()
        });
        let permit = spawner
            .semaphore
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| ReactError::Other("test semaphore was closed".to_string()))?;
        let execution_started = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let execution_started_inner = Arc::clone(&execution_started);
        let queued = spawner.spawn("queued-deadline", async move {
            execution_started_inner.store(true, Ordering::Release);
            Ok(())
        });

        let result = queued.wait(Some(Duration::from_secs(2))).await;
        if result.is_ok()
            || execution_started.load(Ordering::Acquire)
            || !matches!(queued.status().await, BackgroundTaskStatus::Failed { .. })
        {
            return Err(ReactError::Other(
                "queued deadline started work or published the wrong terminal".to_string(),
            ));
        }
        drop(permit);
        Ok(())
    }

    /// 执行阶段取消会等待子任务析构后再发布终态。
    #[tokio::test]
    async fn execution_cancel_waits_for_child_task_drop() -> Result<()> {
        let spawner = TaskSpawner::new(TaskSpawnerConfig {
            default_timeout_secs: 0,
            ..TaskSpawnerConfig::default()
        });
        let dropped = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let dropped_inner = Arc::clone(&dropped);
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let handle = spawner.spawn("execution-cancel", async move {
            let _probe = DropProbe(dropped_inner);
            let _ = started_tx.send(());
            std::future::pending::<()>().await;
            Ok(())
        });
        started_rx
            .await
            .map_err(|_| ReactError::Other("execution task did not start".to_string()))?;
        handle.cancel();
        let result = handle.wait(Some(Duration::from_secs(1))).await;
        if result.is_ok()
            || !dropped.load(Ordering::Acquire)
            || !matches!(handle.status().await, BackgroundTaskStatus::Cancelled)
        {
            return Err(ReactError::Other(
                "execution cancellation published before child task drop".to_string(),
            ));
        }
        Ok(())
    }

    /// 执行阶段超时会等待子任务析构后再发布终态。
    #[tokio::test]
    async fn execution_timeout_waits_for_child_task_drop() -> Result<()> {
        let spawner = TaskSpawner::new(TaskSpawnerConfig {
            default_timeout_secs: 1,
            ..TaskSpawnerConfig::default()
        });
        let dropped = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let dropped_inner = Arc::clone(&dropped);
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let handle = spawner.spawn("execution-timeout", async move {
            let _probe = DropProbe(dropped_inner);
            let _ = started_tx.send(());
            std::future::pending::<()>().await;
            Ok(())
        });
        started_rx
            .await
            .map_err(|_| ReactError::Other("execution task did not start".to_string()))?;
        let result = handle.wait(Some(Duration::from_secs(2))).await;
        if result.is_ok()
            || !dropped.load(Ordering::Acquire)
            || !matches!(handle.status().await, BackgroundTaskStatus::Failed { .. })
        {
            return Err(ReactError::Other(
                "execution timeout published before child task drop".to_string(),
            ));
        }
        Ok(())
    }

    /// 类型擦除注册表保留失败与取消的终态含义。
    #[tokio::test]
    async fn type_erased_registry_preserves_failure_and_cancellation() -> Result<()> {
        let spawner = TaskSpawner::new(TaskSpawnerConfig::default());
        let failed = spawner.spawn("listed-failure", async {
            Err::<(), _>(ReactError::Other("listed failure".to_string()))
        });
        let failed_id = failed.id.clone();
        let _ = failed.wait(Some(Duration::from_secs(1))).await;

        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let cancelled = spawner.spawn("listed-cancel", async move {
            let _ = started_tx.send(());
            std::future::pending::<()>().await;
            Ok(())
        });
        let cancelled_id = cancelled.id.clone();
        started_rx
            .await
            .map_err(|_| ReactError::Other("listed cancellation did not start".to_string()))?;
        cancelled.cancel();
        let _ = cancelled.wait(Some(Duration::from_secs(1))).await;

        let summaries = spawner.list().await;
        let failed_status = summaries
            .iter()
            .find(|summary| summary.id == failed_id)
            .map(|summary| &summary.status);
        let cancelled_status = summaries
            .iter()
            .find(|summary| summary.id == cancelled_id)
            .map(|summary| &summary.status);
        if !matches!(failed_status, Some(BackgroundTaskStatus::Failed { .. }))
            || !matches!(cancelled_status, Some(BackgroundTaskStatus::Cancelled))
        {
            return Err(ReactError::Other(
                "type-erased task registry changed terminal meaning".to_string(),
            ));
        }
        Ok(())
    }

    /// list 返回所有被跟踪的任务。
    #[tokio::test]
    async fn test_task_spawner_list() {
        let spawner = TaskSpawner::new(TaskSpawnerConfig::default());

        let _h1 = spawner.spawn("task-a", async { Ok(1) });
        let _h2 = spawner.spawn("task-b", async {
            tokio::time::sleep(Duration::from_secs(60)).await;
            Ok(2)
        });

        assert_eq!(spawner.task_count(), 2);

        let list = spawner.list().await;
        assert_eq!(list.len(), 2);
    }

    /// 终态历史有上限，且不会移除运行中的任务。
    #[tokio::test]
    async fn terminal_history_is_bounded_without_removing_running_tasks() -> Result<()> {
        let spawner = TaskSpawner::new(TaskSpawnerConfig {
            max_concurrent: 4,
            default_timeout_secs: 0,
            max_terminal_history: 2,
        });

        for index in 0..4 {
            let handle = spawner.spawn(&format!("completed-{index}"), async { Ok(()) });
            handle.wait(Some(Duration::from_secs(1))).await?;
        }

        let running = spawner.spawn("still-running", async {
            tokio::time::sleep(Duration::from_secs(60)).await;
            Ok(())
        });
        let trigger = spawner.spawn("retention-trigger", async { Ok(()) });

        let summaries = spawner.list().await;
        let completed_history = summaries
            .iter()
            .filter(|summary| summary.name.starts_with("completed-"))
            .count();
        assert_eq!(completed_history, 2);
        assert!(summaries.iter().any(|summary| summary.id == running.id));
        assert!(summaries.iter().any(|summary| summary.id == trigger.id));

        running.cancel();
        let _ = running.wait(Some(Duration::from_secs(1))).await;
        trigger.wait(Some(Duration::from_secs(1))).await?;
        Ok(())
    }

    /// cancel_all 取消全部任务，任务在清理前仍被跟踪。
    #[tokio::test]
    async fn test_task_spawner_cancel_all() {
        let spawner = TaskSpawner::new(TaskSpawnerConfig::default());

        let _h1 = spawner.spawn("task-1", async {
            tokio::time::sleep(Duration::from_secs(60)).await;
            Ok(())
        });
        let _h2 = spawner.spawn("task-2", async {
            tokio::time::sleep(Duration::from_secs(60)).await;
            Ok(())
        });

        spawner.cancel_all();

        // 两个任务最终都应终止
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert_eq!(spawner.task_count(), 2); // 在清理前仍被跟踪
    }

    /// is_terminal 对各状态的判定。
    #[test]
    fn test_status_is_terminal() {
        assert!(!BackgroundTaskStatus::Pending.is_terminal());
        assert!(
            !BackgroundTaskStatus::Running {
                started_at: Instant::now()
            }
            .is_terminal()
        );
        assert!(
            BackgroundTaskStatus::Completed {
                finished_at: Instant::now()
            }
            .is_terminal()
        );
        assert!(
            BackgroundTaskStatus::Failed {
                error: "test".into(),
                at: Instant::now()
            }
            .is_terminal()
        );
        assert!(BackgroundTaskStatus::Cancelled.is_terminal());
    }

    /// as_str 返回的状态文本。
    #[test]
    fn test_status_as_str() {
        assert_eq!(BackgroundTaskStatus::Pending.as_str(), "pending");
        assert_eq!(
            BackgroundTaskStatus::Running {
                started_at: Instant::now()
            }
            .as_str(),
            "running"
        );
        assert_eq!(
            BackgroundTaskStatus::Completed {
                finished_at: Instant::now()
            }
            .as_str(),
            "completed"
        );
    }
}
