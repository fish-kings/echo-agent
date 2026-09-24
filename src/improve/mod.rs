/*!
演化（Evolution）辅助工具，以及可选的、基于评测驱动的改进能力。

# 概述

基础的 `improve` feature 提供显式的执行轨迹（trajectory）导出功能。
只有当同时启用 `eval` feature 时，才能使用基于评测驱动的分析能力。

# 安全性

所有建议都需要人工审核。本模块不会自动执行以下操作：
- 修改核心运行时代码
- 放宽安全策略
- 修改权限规则
- 发布或部署任何内容

# 组成部分

- [`TrajectorySaver`] —— 可选的 ShareGPT JSONL 导出功能，供需要显式构建
  微调数据集的框架使用方使用。
## 基于评测驱动的改进（`improve` + `eval`）

- `Analyzer` —— 静态分析 Run trace，检测失败模式。
- `ImprovementLoop` —— 迭代式的提示词优化循环。
- `EvalDrivenImprovement` —— 统一入口，在 ImprovementLoop 基础上封装了 HTML 报告能力。
- `PromptGenerator` —— 由 LLM 驱动的提示词改进生成器。

# 配套模块：`evolution`

[`evolution`](crate::evolution) 模块提供了带有类型化元数据、变更审计与
安全机制的记忆/技能/规则生命周期管理系统，与本模块基于评测驱动的改进
方式互为补充。

# 数据存储位置

- 执行轨迹（Trajectories）：`~/.echo-agent/trajectories/YYYY-MM-DD.jsonl`
- Curator 状态：`~/.echo-agent/curator_state.json`
- 后台复核候选项：直接返回给调用方；可选的用户偏好持久化使用配置好的
  evolution 记忆分层
*/

pub mod trajectory;

#[cfg(feature = "eval")]
pub mod analyzer;
#[cfg(feature = "eval")]
pub mod eval_improvement;
#[cfg(feature = "eval")]
pub mod generator;
#[cfg(feature = "eval")]
pub mod r#loop;

pub use trajectory::{TrajectoryEntry, TrajectorySaver, TrajectoryStats};

#[cfg(feature = "eval")]
pub use analyzer::Analyzer;
#[cfg(feature = "eval")]
pub use eval_improvement::EvalDrivenImprovement;
#[cfg(feature = "eval")]
pub use generator::PromptGenerator;
#[cfg(feature = "eval")]
pub use r#loop::{ImprovementLoop, LoopResult};

#[cfg(feature = "eval")]
use serde::{Deserialize, Serialize};

// ── CritiqueIssue 问题项 ────────────────────────────────────────────────────

/// 在 run trace 中发现的一个具体问题。
#[cfg(feature = "eval")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CritiqueIssue {
    /// Agent 在未先读取文件的情况下就写入了文件。
    WriteWithoutRead {
        /// 涉及的工具名称
        tool: String,
        /// 触发次数
        count: usize,
    },
    /// 工具被过度重试。
    ExcessiveRetries {
        /// 涉及的工具名称
        tool: String,
        /// 重试次数
        count: usize,
    },
    /// 工具错误模式（同一个工具反复失败）。
    ToolErrorPattern {
        /// 涉及的工具名称
        tool: String,
        /// 错误信息
        message: String,
    },
    /// 上下文溢出 —— 触发了压缩。
    ContextOverflow {
        /// 压缩前的 token 数
        tokens_before: usize,
        /// 压缩后的 token 数
        tokens_after: usize,
    },
    /// Agent 没有使用一个看起来是必要的工具。
    MissingTool {
        /// 本应使用但缺失的工具名称
        needed: String,
    },
    /// 对于一个简单任务调用了过多工具。
    ExcessiveToolCalls {
        /// 工具调用总次数
        total: usize,
    },
}

// ── ImprovementSuggestion 改进建议 ────────────────────────────────────────────

/// 一条具体的、可供人工审核的改进建议。
#[cfg(feature = "eval")]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ImprovementSuggestion {
    /// 建议修改 system prompt。
    PromptChange {
        /// 要修改的部分（例如 "tools"、"behavior"、"constraints"）。
        section: String,
        /// 要新增或修改的具体内容。
        suggestion: String,
    },
    /// 建议新增或修改一条策略规则。
    PolicyChange {
        /// 新规则的描述。
        rule: String,
        /// 为什么需要这条规则（基于观察到的失败情况）。
        reason: String,
    },
    /// 基于本次运行生成一个新的评测用例。
    EvalGeneration {
        /// 建议使用的用例 ID。
        case_id: String,
        /// 评测用例内容（JSON 格式）。
        json: String,
    },
}

// ── RunCritique 运行评述 ──────────────────────────────────────────────────────

/// 对单次运行的完整分析。
#[cfg(feature = "eval")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunCritique {
    /// 被分析的运行。
    pub run_id: String,
    /// 该运行整体上是否成功。
    pub success: bool,
    /// 总体评分（0.0 - 1.0）。
    pub score: f64,
    /// 发现的问题列表。
    pub issues: Vec<CritiqueIssue>,
    /// 改进建议列表。
    pub suggestions: Vec<ImprovementSuggestion>,
}

#[cfg(feature = "eval")]
impl RunCritique {
    /// 为一次运行创建一份空的评述。
    pub fn new(run_id: &str, success: bool) -> Self {
        Self {
            run_id: run_id.to_string(),
            success,
            score: if success { 1.0 } else { 0.0 },
            issues: Vec::new(),
            suggestions: Vec::new(),
        }
    }

    /// 添加一个问题。
    pub fn with_issue(mut self, issue: CritiqueIssue) -> Self {
        self.issues.push(issue);
        self
    }

    /// 添加一条建议。
    pub fn with_suggestion(mut self, suggestion: ImprovementSuggestion) -> Self {
        self.suggestions.push(suggestion);
        self
    }

    /// 格式化为人类可读的报告。
    pub fn format_report(&self) -> String {
        let mut lines = vec![
            format!("Run: {}", self.run_id),
            format!("Success: {} (score: {:.2})", self.success, self.score),
            String::new(),
        ];

        if !self.issues.is_empty() {
            lines.push("Issues Found:".into());
            for issue in &self.issues {
                lines.push(match issue {
                    CritiqueIssue::WriteWithoutRead { tool, count } => {
                        format!("  - Write without read: {tool} was called {count} time(s) without prior read_file")
                    }
                    CritiqueIssue::ExcessiveRetries { tool, count } => {
                        format!("  - Excessive retries: {tool} was retried {count} time(s)")
                    }
                    CritiqueIssue::ToolErrorPattern { tool, message } => {
                        format!("  - Tool error: {tool} — {message}")
                    }
                    CritiqueIssue::ContextOverflow { tokens_before, tokens_after } => {
                        format!("  - Context overflow: compressed from {tokens_before} to {tokens_after} tokens")
                    }
                    CritiqueIssue::MissingTool { needed } => {
                        format!("  - Missing tool: {needed} might have helped")
                    }
                    CritiqueIssue::ExcessiveToolCalls { total } => {
                        format!("  - Excessive tool calls: {total} total")
                    }
                });
            }
            lines.push(String::new());
        }

        if !self.suggestions.is_empty() {
            lines.push("Suggestions:".into());
            for suggestion in &self.suggestions {
                match suggestion {
                    ImprovementSuggestion::PromptChange {
                        section,
                        suggestion,
                    } => {
                        lines.push(format!("  [Prompt] {section}: {suggestion}"));
                    }
                    ImprovementSuggestion::PolicyChange { rule, reason } => {
                        lines.push(format!("  [Policy] {rule} — Reason: {reason}"));
                    }
                    ImprovementSuggestion::EvalGeneration { case_id, .. } => {
                        lines.push(format!("  [Eval] Generate case: {case_id}"));
                    }
                }
            }
        }

        if self.issues.is_empty() && self.suggestions.is_empty() {
            lines.push("No issues or suggestions. The run looks clean.".into());
        }

        lines.join("\n")
    }
}
</content>
