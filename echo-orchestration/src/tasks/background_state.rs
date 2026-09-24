//! 后台任务状态管理
//!
//! 本模块为后台任务提供状态管理能力，
//! 使其可以被追踪、生成检查点并恢复。

use serde::{Deserialize, Serialize};

use crate::tasks::TaskStatus;

/// 后台任务状态
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackgroundTaskState {
    /// 任务 ID
    pub task_id: String,
    /// 父任务 ID（如果这是一个子任务）
    pub parent_task_id: Option<String>,
    /// 任务状态
    pub status: TaskStatus,
    /// Shell 命令（命令模式下使用）
    pub command: Option<String>,
    /// 退出码（命令执行完成后）
    pub exit_code: Option<i32>,
    /// 标准输出
    pub stdout: String,
    /// 标准错误
    pub stderr: String,
    /// 任务开始时间戳
    pub started_at: u64,
    /// 任务完成时间戳
    pub completed_at: Option<u64>,
    /// 任务耗时（毫秒）
    pub duration_ms: u64,
    /// 检查点时间戳
    pub checkpoint_at: u64,
}

impl BackgroundTaskState {
    /// 创建一个新的后台任务状态
    pub fn new(task_id: impl Into<String>) -> Self {
        let now = super::time::now_secs();
        Self {
            task_id: task_id.into(),
            parent_task_id: None,
            status: TaskStatus::Pending,
            command: None,
            exit_code: None,
            stdout: String::new(),
            stderr: String::new(),
            started_at: now,
            completed_at: None,
            duration_ms: 0,
            checkpoint_at: now,
        }
    }

    /// 设置父任务 ID
    pub fn with_parent_task_id(mut self, parent_id: impl Into<String>) -> Self {
        self.parent_task_id = Some(parent_id.into());
        self
    }

    /// 设置命令
    pub fn with_command(mut self, command: impl Into<String>) -> Self {
        self.command = Some(command.into());
        self
    }

    /// 将任务标记为已启动
    pub fn mark_started(&mut self) {
        self.status = TaskStatus::Running;
        self.started_at = super::time::now_secs();
    }

    /// 将任务标记为已完成
    pub fn mark_completed(&mut self, exit_code: i32, stdout: String, stderr: String) {
        self.status = if exit_code == 0 {
            TaskStatus::Completed
        } else {
            TaskStatus::Failed(format!("Exit code: {}", exit_code))
        };
        self.exit_code = Some(exit_code);
        self.stdout = stdout;
        self.stderr = stderr;
        let completed_at = super::time::now_secs();
        self.completed_at = Some(completed_at);
        self.duration_ms = completed_at
            .saturating_sub(self.started_at)
            .saturating_mul(1000);
        self.checkpoint_at = completed_at;
    }

    /// 将任务标记为失败
    pub fn mark_failed(&mut self, error: impl Into<String>) {
        self.status = TaskStatus::Failed(error.into());
        let completed_at = super::time::now_secs();
        self.completed_at = Some(completed_at);
        self.duration_ms = completed_at
            .saturating_sub(self.started_at)
            .saturating_mul(1000);
        self.checkpoint_at = completed_at;
    }

    /// 将任务标记为已取消
    pub fn mark_cancelled(&mut self) {
        self.status = TaskStatus::Cancelled;
        let completed_at = super::time::now_secs();
        self.completed_at = Some(completed_at);
        self.duration_ms = completed_at
            .saturating_sub(self.started_at)
            .saturating_mul(1000);
        self.checkpoint_at = completed_at;
    }

    /// 检查任务是否已处于终止状态
    pub fn is_terminal(&self) -> bool {
        self.status.is_terminal()
    }
}

/// 用于持久化任务状态的检查点存储 trait。
pub trait CheckpointStore: Send + Sync {
    /// 保存一个任务状态
    fn save_task_state(
        &self,
        state: &BackgroundTaskState,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), String>> + Send + '_>>;

    /// 根据任务 ID 加载任务状态
    fn load_task_state(
        &self,
        task_id: &str,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<Output = Result<Option<BackgroundTaskState>, String>>
                + Send
                + '_,
        >,
    >;

    /// 加载所有任务状态
    fn load_all_states(
        &self,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Vec<BackgroundTaskState>, String>> + Send + '_>,
    >;

    /// 删除一个任务状态
    fn delete_task_state(
        &self,
        task_id: &str,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<bool, String>> + Send + '_>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_background_task_state_new() {
        let state = BackgroundTaskState::new("task1");
        assert_eq!(state.task_id, "task1");
        assert_eq!(state.status, TaskStatus::Pending);
        assert!(state.parent_task_id.is_none());
        assert!(state.command.is_none());
        assert!(state.exit_code.is_none());
        assert!(state.stdout.is_empty());
        assert!(state.stderr.is_empty());
    }

    #[test]
    fn test_background_task_state_mark_started() {
        let mut state = BackgroundTaskState::new("task1");
        state.mark_started();
        assert_eq!(state.status, TaskStatus::Running);
        assert!(state.started_at > 0);
    }

    #[test]
    fn test_background_task_state_mark_completed() {
        let mut state = BackgroundTaskState::new("task1");
        state.mark_started();
        state.mark_completed(0, "success".to_string(), String::new());
        assert_eq!(state.status, TaskStatus::Completed);
        assert_eq!(state.exit_code, Some(0));
        assert_eq!(state.stdout, "success");
        assert!(state.completed_at.is_some());
    }

    #[test]
    fn test_background_task_state_mark_failed() {
        let mut state = BackgroundTaskState::new("task1");
        state.mark_started();
        state.mark_failed("error occurred");
        assert!(matches!(state.status, TaskStatus::Failed(_)));
        assert!(state.completed_at.is_some());
    }
}
