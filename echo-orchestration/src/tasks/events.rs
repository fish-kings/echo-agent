//! 面向 UI 消费者的、允许丢失的进度投影。
//!
//! 使用异步广播 channel 实现非阻塞的事件分发。
//! 监听者可以在各自的异步任务中订阅并接收事件。

use super::progress::TaskProgress;
use serde::Serialize;
use std::sync::Arc;
use tokio::sync::broadcast;

/// 事件总线的默认 channel 容量
///
/// 已提升至 512，以在高吞吐场景下减少事件丢失。
/// 如果你的负载会产生大量事件，可以考虑使用更大的容量。
const DEFAULT_CHANNEL_CAPACITY: usize = 512;

/// 由进度报告器发出的、非权威性的进度投影。
#[derive(Debug, Clone, Serialize)]
pub enum TaskEvent {
    /// 任务内部的进度更新（百分比、当前阶段、预计完成时间）。
    ///
    /// 由 [`ProgressReporter`](super::progress::ProgressReporter) 针对带有
    /// [`PhasePlan`](super::progress::PhasePlan) 的长时间运行任务发出。
    Progress {
        /// 任务 ID
        task_id: String,
        /// 进度详情
        progress: TaskProgress,
    },
}

impl TaskEvent {
    /// 返回该事件所属的任务 ID
    pub fn task_id(&self) -> &str {
        match self {
            TaskEvent::Progress { task_id, .. } => task_id,
        }
    }
}

/// 任务事件的监听器 trait（同步版本，用于简单场景）
pub trait TaskEventListener: Send + Sync {
    /// 处理一个任务事件
    fn on_event(&self, event: &TaskEvent);
}

/// 任务事件的异步监听器 trait
#[async_trait::async_trait]
pub trait AsyncTaskEventListener: Send + Sync {
    /// 异步处理一个任务事件
    async fn on_event(&self, event: Arc<TaskEvent>);
}

/// 日志监听器 —— 通过 tracing 记录事件
pub struct LoggingListener;

impl TaskEventListener for LoggingListener {
    fn on_event(&self, event: &TaskEvent) {
        match event {
            TaskEvent::Progress {
                task_id, progress, ..
            } => {
                tracing::debug!(
                    task_id = %task_id,
                    pct = progress.percentage,
                    phase = %progress.current_phase,
                    "task_progress"
                );
            }
        }
    }
}

/// 事件总线 —— 将事件异步广播给所有已注册的监听者
///
/// 使用 `tokio::sync::broadcast` 高效地向多个订阅者扇出事件。
/// 事件被包装在 `Arc` 中以便低成本克隆。
pub struct TaskEventBus {
    /// 广播 channel 的发送端
    tx: broadcast::Sender<Arc<TaskEvent>>,
    // 保留同步监听者以兼容旧代码（包装在 Arc 中以便安全共享）
    sync_listeners: Vec<Arc<dyn TaskEventListener>>,
    // 限制同步监听者并发执行的数量，防止任务堆积时
    // tokio::spawn 无限增长（P1-2.4）。
    sync_semaphore: Arc<tokio::sync::Semaphore>,
}

impl TaskEventBus {
    /// 同步监听者的最大并发调用数。
    const MAX_SYNC_CONCURRENCY: usize = 32;

    /// 使用默认容量创建一个新的事件总线
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_CHANNEL_CAPACITY)
    }

    /// 使用指定的 channel 容量创建一个新的事件总线
    pub fn with_capacity(capacity: usize) -> Self {
        let (tx, _) = broadcast::channel(capacity);
        Self {
            tx,
            sync_listeners: Vec::new(),
            sync_semaphore: Arc::new(tokio::sync::Semaphore::new(Self::MAX_SYNC_CONCURRENCY)),
        }
    }

    /// 注册一个同步监听者（发出事件时会被派发到后台任务执行）
    pub fn register(&mut self, listener: Arc<dyn TaskEventListener>) {
        self.sync_listeners.push(listener);
    }

    /// 订阅异步事件流
    ///
    /// 返回一个可在异步任务中使用的接收端。
    ///
    /// **警告：消费过慢可能导致滞后。** 当 channel 已满且接收端落后时，
    /// `tokio::sync::broadcast` 会丢弃事件。如果你需要保证送达，
    /// 请及时处理事件，或使用单独的持久化队列。
    pub fn subscribe(&self) -> broadcast::Receiver<Arc<TaskEvent>> {
        self.tx.subscribe()
    }

    /// 向所有监听者发出一个事件
    ///
    /// 同步监听者会通过 `tokio::spawn` 被派发到后台任务，以避免阻塞调用方，
    /// 但会受到一个信号量的并发限制（P1-2.4）。
    /// 异步订阅者则通过各自的接收端收到该事件。
    pub fn emit(&self, event: TaskEvent) {
        // 将同步监听者派发到后台任务（有限并发）
        let event_for_async = Arc::new(event.clone());
        for listener in &self.sync_listeners {
            let listener = listener.clone();
            let event = event.clone();
            let sem = self.sync_semaphore.clone();
            tokio::spawn(async move {
                let _permit = sem.acquire().await;
                listener.on_event(&event);
            });
        }

        // 广播给异步订阅者
        let _ = self.tx.send(event_for_async);
    }

    /// 获取异步订阅者的数量
    pub fn subscriber_count(&self) -> usize {
        self.tx.receiver_count()
    }
}

impl Clone for TaskEventBus {
    fn clone(&self) -> Self {
        // 克隆事件总线会创建一个指向同一 channel 的新发送端
        Self {
            tx: self.tx.clone(),
            sync_listeners: Vec::new(), // 不克隆同步监听者
            sync_semaphore: self.sync_semaphore.clone(),
        }
    }
}

impl Default for TaskEventBus {
    /// 使用 [`TaskEventBus::new`] 创建默认实例
    fn default() -> Self {
        Self::new()
    }
}
