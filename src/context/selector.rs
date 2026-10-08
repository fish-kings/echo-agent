/*!
上下文选择器 —— 根据文件与当前任务的相关性打分。

通过符号匹配、最近修改时间以及 git 变更检测，
找出对上下文组装最相关的文件。
*/

use std::collections::HashMap;
use std::path::PathBuf;

/// 为给定任务的相关文件打分并进行选择。
pub struct ContextSelector {
    /// 符号名称匹配的权重（较高 —— 与代码直接相关）。
    pub symbol_weight: f64,
    /// 最近修改文件的权重（中等 —— 可能相关）。
    pub recency_weight: f64,
    /// 出现在 git diff 中的文件权重（中等 —— 正在被修改）。
    pub git_diff_weight: f64,
    /// 纳入上下文的最大文件数量。
    pub max_files: usize,
}

impl Default for ContextSelector {
    /// 使用默认权重创建上下文选择器。
    fn default() -> Self {
        Self {
            symbol_weight: 1.0,
            recency_weight: 0.6,
            git_diff_weight: 0.8,
            max_files: 10,
        }
    }
}

impl ContextSelector {
    /// 创建一个使用默认权重配置的 [`ContextSelector`]。
    pub fn new() -> Self {
        Self::default()
    }

    /**
    根据与给定任务描述的相关性为文件打分。

    返回按相关性从高到低排序的文件路径列表。
    `symbols` 是从文件路径到提取出的符号名称的映射。
    `recently_modified` 列出最近发生变更的文件。
    `git_changed` 列出存在 git 修改的文件。
    */
    pub fn score_files(
        &self,
        task: &str,
        symbols: &HashMap<PathBuf, Vec<String>>,
        recently_modified: &[PathBuf],
        git_changed: &[PathBuf],
    ) -> Vec<(PathBuf, f64)> {
        let task_lower = task.to_lowercase();
        let task_words: Vec<&str> = task_lower.split_whitespace().collect();
        let mut scores: HashMap<PathBuf, f64> = HashMap::new();

        // 按符号匹配打分
        for (path, syms) in symbols {
            let mut score = 0.0;
            for sym in syms {
                let sym_lower = sym.to_lowercase();
                // 直接的单词匹配
                for word in &task_words {
                    if sym_lower.contains(word) || word.contains(&sym_lower) {
                        score += self.symbol_weight;
                    }
                }
                // 子串匹配（部分匹配）
                if task_lower.contains(&sym_lower) {
                    score += self.symbol_weight * 0.5;
                }
            }
            if score > 0.0 {
                *scores.entry(path.clone()).or_default() += score;
            }
        }

        // 按最近修改时间打分
        for path in recently_modified {
            *scores.entry(path.clone()).or_default() += self.recency_weight;
        }

        // 按 git 变更打分
        for path in git_changed {
            *scores.entry(path.clone()).or_default() += self.git_diff_weight;
        }

        // 按分数降序排序
        let mut scored: Vec<(PathBuf, f64)> = scores.into_iter().collect();
        scored.sort_by(|a, b| {
            b.1.total_cmp(&a.1)
                .then_with(|| a.0.as_os_str().cmp(b.0.as_os_str()))
        });
        scored
    }

    /// 为一个任务选出相关性最高的前 N 个文件。
    pub fn select_relevant(
        &self,
        task: &str,
        symbols: &HashMap<PathBuf, Vec<String>>,
        recently_modified: &[PathBuf],
        git_changed: &[PathBuf],
    ) -> Vec<PathBuf> {
        let scored = self.score_files(task, symbols, recently_modified, git_changed);
        scored
            .into_iter()
            .take(self.max_files)
            .map(|(path, _)| path)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 验证符号匹配会提升文件得分。
    #[test]
    fn test_score_symbol_match() {
        let selector = ContextSelector::new();
        let mut symbols = HashMap::new();
        symbols.insert(
            PathBuf::from("src/auth.rs"),
            vec!["login".into(), "authenticate".into()],
        );
        symbols.insert(
            PathBuf::from("src/utils.rs"),
            vec!["format".into(), "parse".into()],
        );

        let scored = selector.score_files("fix the login bug", &symbols, &[], &[]);
        assert!(!scored.is_empty());
        // 对于 "login bug"，auth.rs 的得分应高于 utils.rs
        assert_eq!(scored[0].0, PathBuf::from("src/auth.rs"));
    }

    /// 验证最近修改与 Git 变更信号会计入得分。
    #[test]
    fn test_score_recency_and_git() {
        let selector = ContextSelector::new();
        let symbols = HashMap::new();
        let recent = vec![PathBuf::from("src/recent.rs")];
        let git = vec![PathBuf::from("src/changed.rs")];

        let scored = selector.score_files("some task", &symbols, &recent, &git);
        assert_eq!(scored.len(), 2);
    }

    /// 验证得分相同时按路径排序以保证确定性。
    #[test]
    fn equal_scores_are_ordered_by_path() {
        let selector = ContextSelector::new();
        let recent = vec![PathBuf::from("src/z.rs"), PathBuf::from("src/a.rs")];
        let scored = selector.score_files("task", &HashMap::new(), &recent, &[]);
        assert_eq!(
            scored.iter().map(|(path, _)| path).collect::<Vec<_>>(),
            vec![&PathBuf::from("src/a.rs"), &PathBuf::from("src/z.rs")]
        );
    }
}
