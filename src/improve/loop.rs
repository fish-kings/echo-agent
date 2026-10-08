/*!
自我改进循环 —— 评估 → 检测失败 → 改进 → 重新评估。

思路借鉴自 skill-creator 模式：运行 eval 用例，使用 Analyzer 检测失败
模式，生成改进建议并应用，然后重新测试以衡量改进效果。
*/

use crate::eval::{EvalCase, EvalReport, EvalRunner, SuccessCriteria};
use crate::improve::{Analyzer, ImprovementSuggestion, RunCritique};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

/// 单次改进迭代的结果。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoopIteration {
    /// 迭代序号（从 0 开始）。
    pub iteration: usize,
    /// 本轮在测试集（holdout）上的评估报告。
    pub eval_report: EvalReport,
    /// 本轮在训练集上的平均得分。
    pub train_score: f64,
    /// 对训练集失败运行的评析结果。
    pub critiques: Vec<RunCritique>,
    /// 由评析汇总、去重后的改进建议。
    pub suggestions: Vec<ImprovementSuggestion>,
    /// 本轮迭代耗时（毫秒）。
    pub duration_ms: u64,
}

/// 完整的改进循环历史。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoopResult {
    /// 所有已执行的迭代记录。
    pub iterations: Vec<LoopIteration>,
    /// 测试集上取得的最佳平均得分。
    pub best_score: f64,
    /// 取得最佳得分的迭代序号。
    pub best_iteration: usize,
    /// 整个循环的总耗时（毫秒）。
    pub total_duration_ms: u64,
}

/**
分析循环 —— 执行评估、检测失败并给出改进建议。

注意：本循环只分析失败并生成供人工审阅的建议，**不会**自动将建议应用到
agent 上。若要应用建议，请使用 [`PromptGenerator`](crate::improve::PromptGenerator)
生成更新后的系统提示词，并通过工厂将其传给新的 agent。
*/
#[derive(Clone)]
pub struct ImprovementLoop {
    /// 最大迭代次数。
    pub max_iterations: usize,
    /// 提前停止的得分阈值：测试集最佳得分达到该值即停止。
    pub improvement_threshold: f64,
    /// 划入测试集（holdout）的用例比例。
    pub holdout_ratio: f64,
}

/// 默认配置：最多 5 轮、阈值 0.95、holdout 比例 0.4。
impl Default for ImprovementLoop {
    fn default() -> Self {
        Self {
            max_iterations: 5,
            improvement_threshold: 0.95,
            holdout_ratio: 0.4,
        }
    }
}

impl ImprovementLoop {
    /// 使用默认配置创建改进循环。
    pub fn new() -> Self {
        Self::default()
    }

    /// 返回 SuccessCriteria 的变体名称，用于分层划分。
    fn criteria_variant(criteria: &SuccessCriteria) -> &str {
        match criteria {
            SuccessCriteria::TestPass { .. } => "test_pass",
            SuccessCriteria::OutputContains { .. } => "output_contains",
            SuccessCriteria::ToolUsed { .. } => "tool_used",
            SuccessCriteria::ToolNotUsed { .. } => "tool_not_used",
            SuccessCriteria::AllOf(_) => "all_of",
            SuccessCriteria::AnyOf(_) => "any_of",
            SuccessCriteria::LlmGraded { .. } => "llm_graded",
            SuccessCriteria::SweBench { .. } => "swe_bench",
            SuccessCriteria::SafetyCheck { .. } => "safety_check",
            SuccessCriteria::CitationValid { .. } => "citation_valid",
            SuccessCriteria::ValueMatch { .. } => "value_match",
        }
    }

    /// 分层划分：按判定标准类型对用例分组，再按比例拆分每一组。
    fn stratified_split(
        cases: &[EvalCase],
        holdout_ratio: f64,
    ) -> (Vec<&EvalCase>, Vec<&EvalCase>) {
        let mut groups: HashMap<String, Vec<&EvalCase>> = HashMap::new();
        for case in cases {
            let key = Self::criteria_variant(&case.success_criteria).to_string();
            groups.entry(key).or_default().push(case);
        }

        let mut train = Vec::new();
        let mut test = Vec::new();

        for (_, group) in groups {
            let split_idx = if group.len() == 1 {
                /* 单元素分组无法构成独立的 holdout，否则会把训练样本
                泄漏到评估中。 */
                1
            } else {
                let proportional = ((1.0 - holdout_ratio) * group.len() as f64) as usize;
                proportional.clamp(1, group.len().saturating_sub(1))
            };
            for (index, case) in group.into_iter().enumerate() {
                if index < split_idx {
                    train.push(case);
                } else {
                    test.push(case);
                }
            }
        }

        (train, test)
    }

    /**
    运行分析循环：评估 → 评析 → 建议 → 重新评估。
    返回分析结果；建议需由外部自行应用。
    */
    pub async fn run(
        &self,
        cases: &[EvalCase],
        agent_factory: impl Fn() -> Box<dyn crate::agent::Agent>,
        run_store: &Option<Arc<dyn crate::trace::RunStore>>,
    ) -> LoopResult {
        self.run_async(cases, || std::future::ready(agent_factory()), run_store)
            .await
    }

    /**
    使用惰性调用的异步 Agent 工厂运行循环。远程适配器使用此入口，
    以便提前停止时不会触发多余的预先构造。
    */
    pub async fn run_async<F, Fut>(
        &self,
        cases: &[EvalCase],
        agent_factory: F,
        run_store: &Option<Arc<dyn crate::trace::RunStore>>,
    ) -> LoopResult
    where
        F: Fn() -> Fut,
        Fut: std::future::Future<Output = Box<dyn crate::agent::Agent>>,
    {
        let started = Instant::now();
        if cases.is_empty() {
            return LoopResult {
                iterations: vec![],
                best_score: 0.0,
                best_iteration: 0,
                total_duration_ms: 0,
            };
        }
        let mut iterations = Vec::new();
        let mut best_score = 0.0;
        let mut best_iteration = 0;

        // 按判定标准类型分层划分，防止过拟合
        let (train_cases, test_cases) = Self::stratified_split(cases, self.holdout_ratio);
        let runner = EvalRunner::new(std::env::temp_dir());

        for i in 0..self.max_iterations {
            let iter_start = Instant::now();

            // a. 在训练集上评估
            let train_cases_vec: Vec<EvalCase> = train_cases.iter().map(|c| (*c).clone()).collect();
            let train_report = runner.run_all_async(&train_cases_vec, &agent_factory).await;

            // b. 分析失败 —— 加载运行记录并进行评析
            let mut critiques = Vec::new();
            if let Some(store) = run_store {
                for result in &train_report.results {
                    if !result.success
                        && let Some(ref run_id) = result.run_id
                        && let Ok(Some(run)) = store.load(run_id).await
                    {
                        critiques.push(Analyzer::analyze(&run));
                    }
                }
            }

            // c. 根据评析生成建议
            let mut suggestions = Vec::new();
            for c in &critiques {
                suggestions.extend(c.suggestions.clone());
            }
            // 排序并去重（dedup 只移除相邻重复项，因此需先排序）
            suggestions.sort_by_key(|s| format!("{:?}", s));
            suggestions.dedup_by_key(|s| format!("{:?}", s));

            // d. 在测试集上重新评估（盲测 —— 生成器看不到测试集得分）
            let test_cases_vec: Vec<EvalCase> = test_cases.iter().map(|c| (*c).clone()).collect();
            let test_report = runner.run_all_async(&test_cases_vec, &agent_factory).await;

            // e. 按测试集得分记录最佳结果
            if test_report.avg_score > best_score {
                best_score = test_report.avg_score;
                best_iteration = i;
            }

            let iter = LoopIteration {
                iteration: i,
                eval_report: test_report,
                train_score: train_report.avg_score,
                critiques,
                suggestions: suggestions.clone(),
                duration_ms: iter_start.elapsed().as_millis() as u64,
            };
            iterations.push(iter);

            // 达到阈值则提前停止
            if best_score >= self.improvement_threshold {
                break;
            }
        }

        LoopResult {
            iterations,
            best_score,
            best_iteration,
            total_duration_ms: started.elapsed().as_millis() as u64,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::MockAgent;

    /// 构造一个测试用的 eval 用例。
    fn eval_case(id: &str, success_criteria: SuccessCriteria) -> EvalCase {
        EvalCase {
            id: id.to_string(),
            name: id.to_string(),
            description: String::new(),
            domain: None,
            task: format!("Evaluate {id}"),
            project_fixture: None,
            success_criteria,
            constraints: Default::default(),
        }
    }

    /// 验证默认配置值。
    #[test]
    fn test_loop_defaults() {
        let lp = ImprovementLoop::new();
        assert_eq!(lp.max_iterations, 5);
        assert_eq!(lp.improvement_threshold, 0.95);
    }

    /// 单元素分组只进入训练集。
    #[test]
    fn singleton_criteria_group_is_train_only() {
        let cases = vec![eval_case(
            "single",
            SuccessCriteria::OutputContains {
                substring: "done".to_string(),
            },
        )];

        let (train, holdout) = ImprovementLoop::stratified_split(&cases, 0.4);

        assert_eq!(train.len(), 1);
        assert_eq!(train.first().map(|case| case.id.as_str()), Some("single"));
        assert!(holdout.is_empty());
    }

    /// 混合分组时单元素分组不会进入 holdout。
    #[test]
    fn mixed_groups_keep_singletons_out_of_holdout() {
        let cases = vec![
            eval_case(
                "single",
                SuccessCriteria::OutputContains {
                    substring: "done".to_string(),
                },
            ),
            eval_case(
                "tool-a",
                SuccessCriteria::ToolUsed {
                    tool_name: "read".to_string(),
                },
            ),
            eval_case(
                "tool-b",
                SuccessCriteria::ToolUsed {
                    tool_name: "write".to_string(),
                },
            ),
        ];

        let (train, holdout) = ImprovementLoop::stratified_split(&cases, 0.5);

        assert_eq!(train.len(), 2);
        assert_eq!(holdout.len(), 1);
        assert!(train.iter().any(|case| case.id == "single"));
        assert!(!holdout.iter().any(|case| case.id == "single"));
        assert_eq!(
            train
                .iter()
                .chain(&holdout)
                .filter(|case| case.id.starts_with("tool-"))
                .count(),
            2
        );
    }

    /// 多元素分组在极端比例下仍保持训练集与 holdout 均非空。
    #[test]
    fn multi_case_groups_remain_split_at_ratio_boundaries() {
        let cases = vec![
            eval_case(
                "case-a",
                SuccessCriteria::OutputContains {
                    substring: "a".to_string(),
                },
            ),
            eval_case(
                "case-b",
                SuccessCriteria::OutputContains {
                    substring: "b".to_string(),
                },
            ),
            eval_case(
                "case-c",
                SuccessCriteria::OutputContains {
                    substring: "c".to_string(),
                },
            ),
        ];

        for ratio in [-1.0, 0.0, 1.0, 2.0] {
            let (train, holdout) = ImprovementLoop::stratified_split(&cases, ratio);
            assert!(!train.is_empty());
            assert!(!holdout.is_empty());
            assert_eq!(train.len().saturating_add(holdout.len()), cases.len());
            for case in &cases {
                let occurrences = train
                    .iter()
                    .chain(&holdout)
                    .filter(|candidate| candidate.id == case.id)
                    .count();
                assert_eq!(occurrences, 1);
            }
        }
    }

    /// 并发且提前停止的循环使用各自独立且会被清理的工作目录。
    #[tokio::test]
    async fn concurrent_early_stop_loops_use_distinct_cleaned_generations()
    -> std::result::Result<(), String> {
        let cases = vec![eval_case(
            "early-stop",
            SuccessCriteria::OutputContains {
                substring: "done".to_string(),
            },
        )];
        let loop_config = ImprovementLoop {
            max_iterations: 3,
            improvement_threshold: 0.0,
            holdout_ratio: 0.4,
        };
        let first_observer = MockAgent::new("first-loop").with_default_success("done".to_string());
        let first_factory_agent = first_observer.clone();
        let second_observer =
            MockAgent::new("second-loop").with_default_success("done".to_string());
        let second_factory_agent = second_observer.clone();
        let run_store = None;

        let first_loop = loop_config.run_async(
            &cases,
            move || {
                let agent = first_factory_agent.clone();
                std::future::ready(Box::new(agent) as Box<dyn crate::agent::Agent>)
            },
            &run_store,
        );
        let second_loop = loop_config.run_async(
            &cases,
            move || {
                let agent = second_factory_agent.clone();
                std::future::ready(Box::new(agent) as Box<dyn crate::agent::Agent>)
            },
            &run_store,
        );
        let (first_result, second_result) = tokio::join!(first_loop, second_loop);
        let first_cwd = first_observer
            .invocation_contexts()
            .first()
            .and_then(|context| context.working_dir.clone())
            .ok_or_else(|| "first early-stop loop did not record a workspace".to_string())?;
        let second_cwd = second_observer
            .invocation_contexts()
            .first()
            .and_then(|context| context.working_dir.clone())
            .ok_or_else(|| "second early-stop loop did not record a workspace".to_string())?;

        if first_result.iterations.len() != 1 || second_result.iterations.len() != 1 {
            return Err("improvement loops did not stop after the first iteration".to_string());
        }
        if first_cwd == second_cwd {
            return Err("concurrent improvement loops shared a workspace".to_string());
        }
        if first_cwd.exists() || second_cwd.exists() {
            return Err("early-stop improvement workspace was not cleaned up".to_string());
        }
        Ok(())
    }
}
