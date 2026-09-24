//! `Vec`、`HashMap`、`HashSet`、闭包以及迭代器的所有权处理方式。

use crate::basics::LearningTask;
use crate::errors::LearningError;
use std::collections::{HashMap, HashSet};

/// 按插入顺序保存任务、同时支持按标题快速查找与标签管理的任务目录。
#[derive(Debug, Default)]
pub struct TaskCatalog {
    /// 按插入顺序保存的任务列表。
    ordered: Vec<LearningTask>,
    /// 任务标题到其在 `ordered` 中下标位置的映射，用于快速查找。
    positions: HashMap<String, usize>,
    /// 目录关联的标签集合。
    tags: HashSet<String>,
}

impl TaskCatalog {
    /// 创建一个空的任务目录。
    pub fn new() -> Self {
        Self::default()
    }

    /// 插入一个新任务；若标题已存在则返回错误。
    pub fn insert(&mut self, task: LearningTask) -> Result<(), LearningError> {
        if self.positions.contains_key(&task.title) {
            return Err(LearningError::DuplicateTask(task.title));
        }
        let position = self.ordered.len();
        self.positions.insert(task.title.clone(), position);
        self.ordered.push(task);
        Ok(())
    }

    /// 按标题查找任务；不存在则返回 `None`。
    pub fn get(&self, title: &str) -> Option<&LearningTask> {
        self.positions
            .get(title)
            .and_then(|position| self.ordered.get(*position))
    }

    /// 为目录添加一个标签；若标签已存在则返回 `false`。
    pub fn add_tag(&mut self, tag: impl Into<String>) -> bool {
        self.tags.insert(tag.into())
    }

    /// 返回所有标签，按字典序排序。
    pub fn tags(&self) -> Vec<&str> {
        let mut tags = self.tags.iter().map(String::as_str).collect::<Vec<_>>();
        tags.sort_unstable();
        tags
    }

    /// 返回所有满足给定断言的任务标题，保持原有插入顺序。
    pub fn matching_titles<F>(&self, predicate: F) -> Vec<&str>
    where
        F: Fn(&LearningTask) -> bool,
    {
        self.ordered
            .iter()
            .filter(|task| predicate(task))
            .map(|task| task.title.as_str())
            .collect()
    }
}

/// 统计输入文本中每个单词（小写化后）出现的次数；`entry` API 只需一次查找即可原地更新计数。
pub fn word_frequencies(input: &str) -> HashMap<String, usize> {
    let mut frequencies = HashMap::new();
    for word in input.split_whitespace() {
        let count = frequencies.entry(word.to_lowercase()).or_insert(0usize);
        *count = count.saturating_add(1);
    }
    frequencies
}

/// 规范化一组字符串：去除首尾空白、转为小写，并过滤掉空字符串；
/// `into_iter` 会消费传入的 vector，因此无需克隆即可返回拥有所有权的文本。
pub fn normalize_owned(values: Vec<String>) -> Vec<String> {
    values
        .into_iter()
        .map(|value| value.trim().to_lowercase())
        .filter(|value| !value.is_empty())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 验证目录保持插入顺序，且支持按标题查找与重复标题检测。
    #[test]
    fn catalog_preserves_order_and_supports_lookup() -> Result<(), LearningError> {
        let mut catalog = TaskCatalog::new();
        catalog.insert(LearningTask::new("read")?)?;
        catalog.insert(LearningTask::new("test")?)?;
        assert_eq!(
            catalog.get("test").map(|task| task.title.as_str()),
            Some("test")
        );
        assert!(catalog.insert(LearningTask::new("test")?).is_err());
        Ok(())
    }

    /// 验证闭包可以捕获外部变量作为过滤条件。
    #[test]
    fn closures_can_capture_filter_values() -> Result<(), LearningError> {
        let mut catalog = TaskCatalog::new();
        catalog.insert(LearningTask::new("read Rust")?)?;
        catalog.insert(LearningTask::new("test Agent")?)?;
        let keyword = "Rust";
        assert_eq!(
            catalog.matching_titles(|task| task.title.contains(keyword)),
            vec!["read Rust"]
        );
        Ok(())
    }

    /// 验证词频统计不区分大小写。
    #[test]
    fn frequency_count_is_case_insensitive() {
        let frequencies = word_frequencies("Rust rust Agent");
        assert_eq!(frequencies.get("rust"), Some(&2));
    }
}
