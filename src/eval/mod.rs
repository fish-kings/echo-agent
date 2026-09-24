/*!
Agent 评估框架 —— 可量化、可重复的 Agent 测试。

# 概览

评估系统让你可以定义测试用例，针对某个 Agent 运行这些用例，并对结果
打分。它构建在 trace 系统（[`crate::trace`]）之上，将评估结果与执行
轨迹关联起来，便于调试与回归分析。

# 快速开始

```rust,no_run
use echo_agent::eval::{EvalCase, EvalRunner, SuccessCriteria};

let case = EvalCase {
    id: "test_001".into(),
    name: "Basic tool use".into(),
    description: "Agent should use read_file when asked to read".into(),
    domain: Some("coding".into()),
    task: "Read the file src/main.rs".into(),
    project_fixture: None,
    success_criteria: SuccessCriteria::ToolUsed {
        tool_name: "read_file".into(),
    },
    constraints: Default::default(),
};

// let runner = EvalRunner::new(temp_dir);
// let result = runner.run(&case, &agent).await;
```
*/

pub mod comparator;
pub mod grader;
pub mod regression;
pub mod replay;
pub mod report;
pub mod runner;
pub mod server;
pub mod trigger;

pub use comparator::AbComparator;
pub use grader::LlmGrader;
pub use regression::RegressionSuite;
pub use replay::TrajectoryReplay;
pub use report::generate_html;
pub use runner::EvalRunner;
pub use server::generate_review_html;
pub use trigger::TriggerAccuracy;

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// 最小引用数量的默认值。
fn default_min_citations() -> usize {
    1
}

/// 引用格式的默认值。
fn default_citation_format() -> String {
    "any".to_string()
}

/// 数值容差的默认值。
fn default_tolerance() -> f64 {
    0.05
}

// ── EvalCase ─────────────────────────────────────────────────────────

/// 单个评估测试用例——测什么、以及如何判定成功。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalCase {
    /// 该用例的唯一标识符（例如 "read_before_edit_001"）。
    pub id: String,
    /// 人类可读的名称。
    pub name: String,
    /// 该用例测试内容的描述。
    #[serde(default)]
    pub description: String,
    /// 该用例所属的领域（coding、data、research、medical、general、skill-trigger）。
    #[serde(default)]
    pub domain: Option<String>,
    /// 交给 Agent 的任务/问题。
    pub task: String,
    /// 测试前需要准备的可选项目 fixture 目录。
    /// 文件会在 Agent 运行前被复制到评估工作区。
    #[serde(default)]
    pub project_fixture: Option<PathBuf>,
    /// 判定通过/失败的标准。
    pub success_criteria: SuccessCriteria,
    /// Agent 必须遵守的约束条件。
    #[serde(default)]
    pub constraints: EvalConstraints,
}

// ── SuccessCriteria ──────────────────────────────────────────────────

/// 如何判定一个评估用例是否通过。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SuccessCriteria {
    /// 测试命令成功退出（例如 `cargo test`）。
    TestPass { command: String },
    /// Agent 的最终输出包含指定子串。
    OutputContains { substring: String },
    /// Agent 至少使用过一次指定工具。
    ToolUsed { tool_name: String },
    /// Agent 没有使用指定工具。
    ToolNotUsed { tool_name: String },
    /// 给定的所有条件都必须通过。
    AllOf(Vec<SuccessCriteria>),
    /// 给定条件中至少有一个通过即可。
    AnyOf(Vec<SuccessCriteria>),
    /// LLM 作为评判者：由一个 LLM 评分 Agent 检查各项断言。
    LlmGraded {
        /// 供评分器检查的断言列表。
        assertions: Vec<crate::eval::grader::Assertion>,
    },
    /// SWE-bench 风格：克隆仓库、检出提交、运行 Agent、应用测试补丁、运行测试。
    SweBench {
        /// 待克隆的 Git 仓库地址。
        repo_url: String,
        /// Agent 运行前要检出的基础提交。
        base_commit: String,
        /// Agent 完成后要应用的补丁文件（新增测试）。
        test_patch: String,
        /// 应用测试补丁后要运行的命令。
        test_command: String,
    },
    /**
    安全检查：禁止出现的模式一定不能出现，必须出现的模式一定要出现。
    用于医疗等红线评估场景，某些输出是不可接受的。
    */
    SafetyCheck {
        /// 输出中一定不能出现的模式（例如 "建议你服用"）。
        forbidden_patterns: Vec<String>,
        /// 输出中必须出现的模式（例如 "咨询医生"）。
        required_patterns: Vec<String>,
    },
    /**
    引用有效性：Agent 的引用必须来自工具结果，而不是凭空捏造。
    检查输出中是否包含引用标记（PMID、DOI、URL），并可选地检查
    引用数量是否达到最小值。
    */
    CitationValid {
        /// 输出中要求的最小引用数量。
        #[serde(default = "default_min_citations")]
        min_citations: usize,
        /// 要检查的引用格式："pmid"、"doi"、"url"、"any"。
        #[serde(default = "default_citation_format")]
        format: String,
    },
    /**
    数值匹配：Agent 的数值输出必须在容差范围内匹配预期值。
    用于统计结果必须准确的数据分析评估场景。
    */
    ValueMatch {
        /// 预期的键值对（key = 指标名称，value = 预期数值）。
        expected: std::collections::HashMap<String, f64>,
        /// 每个数值允许的容差（例如 0.01 = 1%）。
        #[serde(default = "default_tolerance")]
        tolerance: f64,
    },
}

// ── EvalConstraints ──────────────────────────────────────────────────

/// 评估过程中 Agent 必须遵守的护栏约束。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EvalConstraints {
    /// Agent 允许修改的最大文件数。
    #[serde(default)]
    pub max_files_changed: Option<usize>,
    /// 允许的最大工具调用次数。
    #[serde(default)]
    pub max_tool_calls: Option<usize>,
    /// Agent 不得修改的路径。
    #[serde(default)]
    pub forbidden_paths: Vec<String>,
    /// Agent 编辑文件前是否必须先读取该文件。
    #[serde(default)]
    pub required_read_before_edit: bool,
}

// ── EvalMetric ───────────────────────────────────────────────────────

/// 评估结果中单个维度的打分。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalMetric {
    /// 指标名称（例如 "tool_accuracy"、"test_pass"、"constraint_compliance"）。
    pub name: String,
    /// 分数，从 0.0（最差）到 1.0（最好）。
    pub score: f64,
    /// 人类可读的详情说明。
    pub detail: String,
}

// ── EvalResult ───────────────────────────────────────────────────────

/// 运行单个评估用例的结果。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalResult {
    /// 该结果对应的用例 ID。
    pub case_id: String,
    /// 是否满足成功标准。
    pub success: bool,
    /// 综合分数（所有指标的平均值）。
    pub score: f64,
    /// 各维度分数。
    pub metrics: Vec<EvalMetric>,
    /// 发现的约束违规项。
    pub violations: Vec<String>,
    /// 关联的执行轨迹链接（run ID）。
    pub run_id: Option<String>,
    /// 墙钟耗时（毫秒）。
    pub duration_ms: u64,
    /// Agent 发起的工具调用总数。
    #[serde(default)]
    pub tool_calls: usize,
    /// 估算的输入 token 数。
    #[serde(default)]
    pub tokens_in: u32,
    /// 估算的输出 token 数。
    #[serde(default)]
    pub tokens_out: u32,
    /// provider 报告的、命中 prompt 缓存的输入 token 数。
    #[serde(default)]
    pub cached_tokens_in: u32,
    /// provider 报告的、写入 prompt 缓存的输入 token 数。
    #[serde(default)]
    pub cache_creation_tokens_in: u32,
    /// 本次运行的 prompt 缓存命中率。
    #[serde(default)]
    pub cache_hit_rate: Option<f64>,
    /// 执行轨迹中观测到的工具错误数。
    #[serde(default)]
    pub tool_errors: usize,
    /// 在一次 LLM 调用之前观测到的、受保护上下文 token 估算最大值。
    #[serde(default)]
    pub max_protected_context_tokens: usize,
    /// 变更的文件数量。
    #[serde(default)]
    pub file_changes: usize,
    /// 编译是否成功（如适用）。
    #[serde(default)]
    pub compile_ok: Option<bool>,
    /// 测试是否通过（如适用）。
    #[serde(default)]
    pub tests_pass: Option<bool>,
}

impl EvalResult {
    /// 创建一个新的评估结果；成功时分数为 1.0，失败时为 0.0。
    pub fn new(case_id: &str, success: bool) -> Self {
        Self {
            case_id: case_id.to_string(),
            success,
            score: if success { 1.0 } else { 0.0 },
            metrics: Vec::new(),
            violations: Vec::new(),
            run_id: None,
            duration_ms: 0,
            tool_calls: 0,
            tokens_in: 0,
            tokens_out: 0,
            cached_tokens_in: 0,
            cache_creation_tokens_in: 0,
            cache_hit_rate: None,
            tool_errors: 0,
            max_protected_context_tokens: 0,
            file_changes: 0,
            compile_ok: None,
            tests_pass: None,
        }
    }

    /// 追加一个维度指标，并重新计算综合分数。
    pub fn with_metric(mut self, name: &str, score: f64, detail: &str) -> Self {
        self.metrics.push(EvalMetric {
            name: name.to_string(),
            score,
            detail: detail.to_string(),
        });
        self.recompute_score();
        self
    }

    /// 追加一条违规记录，并将结果标记为失败。
    pub fn with_violation(mut self, violation: &str) -> Self {
        self.violations.push(violation.to_string());
        self.success = false;
        self
    }

    /// 关联一个执行轨迹的 run ID。
    pub fn with_run_id(mut self, run_id: &str) -> Self {
        self.run_id = Some(run_id.to_string());
        self
    }

    /// 依据各维度指标重新计算综合分数；无指标时退化为通过/失败的 0/1 分。
    fn recompute_score(&mut self) {
        if self.metrics.is_empty() {
            self.score = if self.success { 1.0 } else { 0.0 };
        } else {
            self.score =
                self.metrics.iter().map(|m| m.score).sum::<f64>() / self.metrics.len() as f64;
        }
    }
}

// ── EvalReport ───────────────────────────────────────────────────────

/// 跨多个评估用例的汇总报告。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalReport {
    /// 用例总数。
    pub total: usize,
    /// 通过的数量。
    pub passed: usize,
    /// 失败的数量。
    pub failed: usize,
    /// 所有用例的平均分数。
    pub avg_score: f64,
    /// 分数的标准差。
    #[serde(default)]
    pub std_dev: f64,
    /// 最低分数。
    #[serde(default)]
    pub min_score: f64,
    /// 最高分数。
    #[serde(default)]
    pub max_score: f64,
    /// 相对基线的分数差（用于两次运行的对比）。
    #[serde(default)]
    pub delta_vs_baseline: Option<f64>,
    /// 所有用例的工具调用总数。
    #[serde(default)]
    pub total_tool_calls: usize,
    /// 所有用例的输入 token 总数。
    #[serde(default)]
    pub total_tokens_in: u32,
    /// 所有用例的输出 token 总数。
    #[serde(default)]
    pub total_tokens_out: u32,
    /// 命中 prompt 缓存的输入 token 总数。
    #[serde(default)]
    pub total_cached_tokens_in: u32,
    /// 写入 prompt 缓存的输入 token 总数。
    #[serde(default)]
    pub total_cache_creation_tokens_in: u32,
    /// 综合的 prompt 缓存命中率。
    #[serde(default)]
    pub cache_hit_rate: Option<f64>,
    /// 所有用例的工具错误总数。
    #[serde(default)]
    pub total_tool_errors: usize,
    /// 所有用例中受保护上下文估算的最大值。
    #[serde(default)]
    pub max_protected_context_tokens: usize,
    /// 所有用例变更的文件总数。
    #[serde(default)]
    pub total_file_changes: usize,
    /// 各用例的结果列表。
    pub results: Vec<EvalResult>,
}

impl EvalReport {
    /// 根据一组评估结果聚合出统计报告（总数、通过率、均值/标准差等）。
    pub fn new(results: Vec<EvalResult>) -> Self {
        let total = results.len();
        let passed = results.iter().filter(|r| r.success).count();
        let failed = total - passed;
        let avg_score = if total > 0 {
            results.iter().map(|r| r.score).sum::<f64>() / total as f64
        } else {
            0.0
        };
        let total_tool_calls = results.iter().fold(0usize, |total, result| {
            total.saturating_add(result.tool_calls)
        });
        let total_tokens_in = results
            .iter()
            .fold(0u32, |total, result| total.saturating_add(result.tokens_in));
        let total_tokens_out = results.iter().fold(0u32, |total, result| {
            total.saturating_add(result.tokens_out)
        });
        let total_cached_tokens_in = results.iter().fold(0u32, |total, result| {
            total.saturating_add(result.cached_tokens_in)
        });
        let total_cache_creation_tokens_in = results.iter().fold(0u32, |total, result| {
            total.saturating_add(result.cache_creation_tokens_in)
        });
        let cache_hit_rate = if total_tokens_in == 0 {
            None
        } else {
            Some(total_cached_tokens_in as f64 / total_tokens_in as f64)
        };
        let total_tool_errors = results.iter().fold(0usize, |total, result| {
            total.saturating_add(result.tool_errors)
        });
        let max_protected_context_tokens = results
            .iter()
            .map(|r| r.max_protected_context_tokens)
            .max()
            .unwrap_or(0);
        let total_file_changes = results.iter().fold(0usize, |total, result| {
            total.saturating_add(result.file_changes)
        });
        // 统计聚合
        let scores: Vec<f64> = results.iter().map(|r| r.score).collect();
        let variance = if total > 1 {
            scores.iter().map(|s| (s - avg_score).powi(2)).sum::<f64>() / (total - 1) as f64
        } else {
            0.0
        };
        let std_dev = variance.sqrt();
        let min_score = scores.iter().cloned().fold(f64::MAX, f64::min);
        let max_score = scores.iter().cloned().fold(f64::MIN, f64::max);
        Self {
            total,
            passed,
            failed,
            avg_score,
            std_dev,
            min_score: if total > 0 { min_score } else { 0.0 },
            max_score: if total > 0 { max_score } else { 0.0 },
            delta_vs_baseline: None,
            total_tool_calls,
            total_tokens_in,
            total_tokens_out,
            total_cached_tokens_in,
            total_cache_creation_tokens_in,
            cache_hit_rate,
            total_tool_errors,
            max_protected_context_tokens,
            total_file_changes,
            results,
        }
    }
}
