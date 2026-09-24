//! trait、泛型、trait object 以及构建者（builder）模式。

use crate::basics::{LearningTask, TaskState};
use crate::errors::LearningError;

/// 将任务格式化为字符串的通用接口。
pub trait TaskFormatter: Send + Sync {
    /// 把给定任务格式化为字符串表示。
    fn format(&self, task: &LearningTask) -> String;
}

/// 只输出任务标题的简单格式化器。
#[derive(Debug, Default)]
pub struct PlainFormatter;

impl TaskFormatter for PlainFormatter {
    /// 返回形如 `task: <标题>` 的字符串。
    fn format(&self, task: &LearningTask) -> String {
        format!("task: {}", task.title)
    }
}

/// 在任务标题后附加当前状态的格式化器。
#[derive(Debug, Default)]
pub struct StatusFormatter;

impl TaskFormatter for StatusFormatter {
    /// 返回形如 `<标题> [状态]` 的字符串。
    fn format(&self, task: &LearningTask) -> String {
        let status = match task.state {
            TaskState::Pending => "pending",
            TaskState::Running => "running",
            TaskState::Completed { .. } => "completed",
        };
        format!("{} [{status}]", task.title)
    }
}

/// 静态分发：编译器会针对具体的格式化器类型单独生成代码。
pub fn format_static<F: TaskFormatter>(formatter: &F, task: &LearningTask) -> String {
    formatter.format(task)
}

/// 动态分发：不同的格式化器实现可以共存于同一个集合中。
pub fn format_dynamic(formatters: &[Box<dyn TaskFormatter>], task: &LearningTask) -> Vec<String> {
    formatters
        .iter()
        .map(|formatter| formatter.format(task))
        .collect()
}

/// 用于分步构建 `LearningTask` 的构建者。
#[derive(Debug, Default)]
pub struct LearningTaskBuilder {
    /// 待构建任务的标题。
    title: Option<String>,
    /// 待构建任务是否应处于运行中状态。
    running: bool,
}

impl LearningTaskBuilder {
    /// 创建一个空的构建者。
    pub fn new() -> Self {
        Self::default()
    }

    /// 设置任务标题。
    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// 设置任务是否处于运行中状态。
    pub fn running(mut self, running: bool) -> Self {
        self.running = running;
        self
    }

    /// 消费构建者，产出最终的 `LearningTask`；标题缺失时返回错误。
    pub fn build(self) -> Result<LearningTask, LearningError> {
        let title = self.title.ok_or(LearningError::EmptyTitle)?;
        let mut task = LearningTask::new(title)?;
        if self.running {
            task.state = TaskState::Running;
        }
        Ok(task)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 验证静态分发与动态分发都能正确调用格式化器。
    #[test]
    fn supports_static_and_dynamic_dispatch() -> Result<(), LearningError> {
        let task = LearningTaskBuilder::new().title("read traits").build()?;
        assert_eq!(format_static(&PlainFormatter, &task), "task: read traits");
        let formatters: Vec<Box<dyn TaskFormatter>> =
            vec![Box::new(PlainFormatter), Box::new(StatusFormatter)];
        assert_eq!(format_dynamic(&formatters, &task).len(), 2);
        Ok(())
    }
}
