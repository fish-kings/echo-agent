//! 任务调度器 —— 基于 Cron 和基于间隔的任务调度。
//!
//! 提供 [`CronTask`] 定义、用于持久化的 [`CronTaskStore`]，
//! 以及用于周期性执行的 [`SchedulerRunner`]。
//!
//! # 示例
//!
//! ```rust,ignore
//! use echo_orchestration::scheduler::{
//!     CronTask, CronTaskStore, OccurrenceFireFn, SchedulerRunner,
//! };
//! use std::sync::Arc;
//! use tokio_util::sync::CancellationToken;
//!
//! let store = CronTaskStore::new();
//! store.add(CronTask::new("daily-report", "0 9 * * *", "Generate daily report")).await?;
//!
//! let cancel = CancellationToken::new();
//! let fire: OccurrenceFireFn = Arc::new(|invocation| {
//!     Box::pin(async move {
//!         // 将 occurrence_id 作为外部副作用的幂等键使用。
//!         println!("Firing: {}", invocation.occurrence_id);
//!         Ok(format!("Executed: {}", invocation.task.name))
//!     })
//! });
//! let runner = SchedulerRunner::new_with_occurrence_context(store, cancel, fire).await?;
//! runner.spawn();
//! ```

mod cron_task;
mod runner;

pub use cron_task::{CronTask, CronTaskStatus, CronTaskStore};
pub use runner::{
    FireFn, OccurrenceFireFn, SchedulerHandle, SchedulerInvocation, SchedulerRunner,
    SchedulerTrigger,
};
