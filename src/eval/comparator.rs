//! A/B 对比评估器 —— 比较两种 Agent 配置。
//!
//! 思路借鉴自 `skill-creator/agents/comparator.md`。在两种不同的
//! Agent 配置上运行相同的评估用例，并计算每个用例的分数差值。

use crate::eval::{EvalCase, EvalReport, EvalRunner};

/// 两次评估运行的对比结果。
pub struct AbComparison {
    /// 配置 A（baseline，基线）的评估结果。
    pub baseline: EvalReport,
    /// 配置 B（experiment，实验组）的评估结果。
    pub experiment: EvalReport,
    /// 每个用例的分数差值（experiment - baseline）。
    pub per_case_deltas: Vec<CaseDelta>,
    /// 实验组相对基线有所提升的用例数。
    pub improved: usize,
    /// 实验组相对基线出现退化的用例数。
    pub regressed: usize,
    /// 没有变化的用例数。
    pub unchanged: usize,
}

/// 单个用例的对比数据。
pub struct CaseDelta {
    /// 用例 id。
    pub case_id: String,
    /// 基线分数。
    pub baseline_score: f64,
    /// 实验组分数。
    pub experiment_score: f64,
    /// 分数差值。
    pub delta: f64,
    /// 该用例是否有所提升。
    pub improved: bool,
}

/// 在两种 Agent 配置之间运行 A/B 对比。
pub struct AbComparator;

impl AbComparator {
    /// 在同一批评估用例上对比两个 Agent 工厂。
    pub async fn compare(
        cases: &[EvalCase],
        mut baseline_factory: impl FnMut() -> Box<dyn crate::agent::Agent>,
        mut experiment_factory: impl FnMut() -> Box<dyn crate::agent::Agent>,
    ) -> AbComparison {
        let runner = EvalRunner::new(std::env::temp_dir());

        let baseline = runner.run_all(cases, &mut baseline_factory).await;
        let experiment = runner.run_all(cases, &mut experiment_factory).await;

        let mut per_case_deltas = Vec::new();
        let mut improved = 0usize;
        let mut regressed = 0usize;
        let mut unchanged = 0usize;

        for (base, exp) in baseline.results.iter().zip(experiment.results.iter()) {
            let delta = exp.score - base.score;
            let is_improved = delta > 0.001;
            let is_regressed = delta < -0.001;

            if is_improved {
                improved += 1;
            } else if is_regressed {
                regressed += 1;
            } else {
                unchanged += 1;
            }

            per_case_deltas.push(CaseDelta {
                case_id: base.case_id.clone(),
                baseline_score: base.score,
                experiment_score: exp.score,
                delta,
                improved: is_improved,
            });
        }

        // 将差值附加到实验组报告上
        let mut experiment = experiment;
        experiment.delta_vs_baseline = Some(experiment.avg_score - baseline.avg_score);

        AbComparison {
            baseline,
            experiment,
            per_case_deltas,
            improved,
            regressed,
            unchanged,
        }
    }

    /// 将对比结果格式化为可读的摘要文本。
    pub fn format_summary(comparison: &AbComparison) -> String {
        let delta = comparison.experiment.avg_score - comparison.baseline.avg_score;
        let direction = if delta > 0.0 {
            "improved"
        } else if delta < 0.0 {
            "regressed"
        } else {
            "unchanged"
        };
        let mut lines = vec![
            "A/B Comparison Results:".to_string(),
            format!(
                "  Baseline:  {:.4} avg (n={})",
                comparison.baseline.avg_score, comparison.baseline.total
            ),
            format!(
                "  Experiment: {:.4} avg (n={})",
                comparison.experiment.avg_score, comparison.experiment.total
            ),
            format!("  Delta: {delta:+.4} → {direction}"),
            format!(
                "  Improved: {}  Regressed: {}  Unchanged: {}",
                comparison.improved, comparison.regressed, comparison.unchanged
            ),
            String::new(),
            "Per-case deltas:".into(),
        ];
        for cd in &comparison.per_case_deltas {
            let icon = if cd.improved {
                "↑"
            } else if cd.delta < -0.001 {
                "↓"
            } else {
                "="
            };
            lines.push(format!(
                "  {icon} {}: {:+.4} ({:.4} → {:.4})",
                cd.case_id, cd.delta, cd.baseline_score, cd.experiment_score
            ));
        }
        lines.join("\n")
    }
}
