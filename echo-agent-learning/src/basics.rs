//! 变量绑定、领域类型、模式匹配、集合与迭代器。

use crate::errors::LearningError;
use serde::{Deserialize, Serialize};

/// 一个小型的领域枚举，类似于 echo-agent 中各处使用的状态枚举。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskState {
    /// 任务尚未开始。
    Pending,
    /// 任务正在执行中。
    Running,
    /// 任务已完成，携带一段结果摘要。
    Completed { summary: String },
}

/// 一个刻意保持精简的任务类型，贯穿各章教学示例使用。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LearningTask {
    /// 任务标题。
    pub title: String,
    /// 任务当前所处的状态。
    pub state: TaskState,
}

impl LearningTask {
    /// 创建一个处于 `Pending` 状态的新任务；标题为空（去除空白后）时返回错误。
    pub fn new(title: impl Into<String>) -> Result<Self, LearningError> {
        let title = title.into();
        if title.trim().is_empty() {
            return Err(LearningError::EmptyTitle);
        }
        Ok(Self {
            title,
            state: TaskState::Pending,
        })
    }

    /// 将任务标记为已完成，并记录给定的结果摘要。
    pub fn complete(&mut self, summary: impl Into<String>) {
        self.state = TaskState::Completed {
            summary: summary.into(),
        };
    }

    /// 若任务处于 `Pending` 状态，则将其转为 `Running` 并返回 `true`；否则返回 `false`。
    pub fn start(&mut self) -> bool {
        if self.state == TaskState::Pending {
            self.state = TaskState::Running;
            true
        } else {
            false
        }
    }

    /// 若任务已完成，返回其结果摘要；否则返回 `None`。
    pub fn summary(&self) -> Option<&str> {
        match &self.state {
            TaskState::Completed { summary } => Some(summary.as_str()),
            TaskState::Pending | TaskState::Running => None,
        }
    }
}

/// 按 Unicode 标量值截断字符串，而不是按 UTF-8 字节偏移量截断。
pub fn unicode_preview(value: &str, max_chars: usize) -> String {
    let mut preview = value.chars().take(max_chars).collect::<String>();
    if value.chars().count() > max_chars {
        preview.push_str("...");
    }
    preview
}

/// 使用迭代器适配器和模式匹配挑选出所有已完成任务的标题。
pub fn completed_titles(tasks: &[LearningTask]) -> Vec<&str> {
    tasks
        .iter()
        .filter_map(|task| match task.state {
            TaskState::Completed { .. } => Some(task.title.as_str()),
            TaskState::Pending | TaskState::Running => None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 验证按 Unicode 标量值截断字符串对中文与 Emoji 都是安全的。
    #[test]
    fn unicode_preview_handles_chinese_and_emoji() {
        assert_eq!(unicode_preview("你好Rust", 3), "你好R...");
        assert_eq!(unicode_preview("A🦀B", 2), "A🦀...");
    }

    /// 验证只有已完成的任务标题会被筛选出来。
    #[test]
    fn filters_completed_tasks() -> Result<(), LearningError> {
        let pending = LearningTask::new("read")?;
        let mut completed = LearningTask::new("test")?;
        completed.complete("green");
        assert_eq!(completed_titles(&[pending, completed]), vec!["test"]);
        Ok(())
    }
}
