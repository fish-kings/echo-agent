/*!
触发准确率评估 —— 衡量子 Agent 的激活表现如何。

针对子 Agent 路由，评估精确率（precision，该触发时是否触发了？）
与召回率（recall，不该触发时是否没有触发？）。
*/

use serde::{Deserialize, Serialize};

/// 用于子 Agent 触发准确率评估的测试用例。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TriggerTestCase {
    /// 用户查询内容。
    pub query: String,
    /// 期望由哪个子 Agent 来处理该查询。
    pub expected_agent: String,
    /// 该查询是否应当触发任意子 Agent（false 表示不应触发）。
    pub should_trigger: bool,
    /// 每条查询用于计算触发率的运行次数，默认值为 1。
    #[serde(default = "default_runs")]
    pub runs_per_query: usize,
}

/// `runs_per_query` 的默认值。
fn default_runs() -> usize {
    1
}

/// 触发准确率评估的结果。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TriggerAccuracy {
    /// 测试用例总数。
    pub total: usize,
    /// 正确触发的用例数（真正例，true positives）。
    pub true_positives: usize,
    /// 错误触发的用例数（假正例，false positives —— 不该触发却触发了）。
    pub false_positives: usize,
    /// 正确跳过的用例数（真负例，true negatives —— 不该触发也确实没有触发）。
    pub true_negatives: usize,
    /// 漏掉的触发（假负例，false negatives —— 该触发却没有触发）。
    pub false_negatives: usize,
    /// 精确率：TP / (TP + FP)。
    pub precision: f64,
    /// 召回率：TP / (TP + FN)。
    pub recall: f64,
    /// F1 分数。
    pub f1: f64,
    /// 当 runs_per_query > 1 时，每个用例的触发率（触发率 = 触发次数 / 总运行次数）。
    #[serde(default)]
    pub trigger_rates: Vec<f64>,
}

impl TriggerAccuracy {
    /// 根据测试用例与实际触发结果评估触发准确率。
    pub fn evaluate(
        cases: &[TriggerTestCase],
        actual_triggers: &[(String, Option<String>)],
    ) -> Self {
        let mut tp = 0usize; // 正确触发
        let mut fp = 0usize; // 不该触发却触发了
        let mut tn = 0usize; // 正确跳过
        let mut fn_count = 0usize; // 该触发却没触发

        for (index, case) in cases.iter().enumerate() {
            let actual = actual_triggers.get(index);
            let query_matches = actual.is_some_and(|(query, _)| query == &case.query);
            let actual_agent = actual.and_then(|(_, agent)| agent.as_deref());
            let did_trigger = query_matches && actual_agent.is_some();
            let correct_agent = query_matches && actual_agent == Some(case.expected_agent.as_str());

            match (case.should_trigger, did_trigger, correct_agent) {
                // 应触发、确实触发、Agent 正确 → TP
                (true, true, true) => tp += 1,
                // 路由错误既是一次预期路由的缺失，也是一次意外路由。
                (true, true, false) => {
                    fp += 1;
                    fn_count += 1;
                }
                // 应触发、未触发 → FN
                (true, false, _) => fn_count += 1,
                // 不应触发、却触发了 → FP
                (false, true, _) => fp += 1,
                // 不应触发、正确跳过 → TN
                (false, false, _) => tn += 1,
            }
        }

        fp = fp.saturating_add(actual_triggers.len().saturating_sub(cases.len()));

        let total = cases.len();
        let precision = if tp + fp > 0 {
            tp as f64 / (tp + fp) as f64
        } else {
            0.0
        };
        let recall = if tp + fn_count > 0 {
            tp as f64 / (tp + fn_count) as f64
        } else {
            0.0
        };
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };

        Self {
            total,
            true_positives: tp,
            false_positives: fp,
            true_negatives: tn,
            false_negatives: fn_count,
            precision,
            recall,
            f1,
            trigger_rates: Vec::new(),
        }
    }

    /**
    根据多次运行的结果评估触发准确率。

    `multi_results` 是一个列表，其中每一项都对应一个测试用例，
    包含该用例多次运行（runs_per_query > 1）的实际触发情况。
    */
    pub fn evaluate_multi_run(
        cases: &[TriggerTestCase],
        multi_results: &[Vec<(String, Option<String>)>],
    ) -> Self {
        // 压平为单次运行结果，以便计算标准指标
        let flat_actuals: Vec<(String, Option<String>)> = cases
            .iter()
            .enumerate()
            .map(|(index, case)| {
                multi_results
                    .get(index)
                    .and_then(|runs| runs.first().cloned())
                    .unwrap_or_else(|| (case.query.clone(), None))
            })
            .collect();
        let mut result = Self::evaluate(cases, &flat_actuals);

        // 计算每个用例的触发率
        result.trigger_rates = cases
            .iter()
            .zip(multi_results.iter())
            .map(|(case, runs)| {
                if case.runs_per_query <= 1 {
                    return if runs.first().is_some_and(|(_, a)| a.is_some()) {
                        1.0
                    } else {
                        0.0
                    };
                }
                if runs.is_empty() || runs.len() != case.runs_per_query {
                    return 0.0;
                }
                let correctly_routed = runs
                    .iter()
                    .filter(|(query, agent)| {
                        query == &case.query
                            && agent.as_deref() == Some(case.expected_agent.as_str())
                    })
                    .count();
                correctly_routed as f64 / runs.len() as f64
            })
            .collect();

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 验证全部正确触发时准确率指标均为满分。
    #[test]
    fn test_trigger_accuracy_perfect() {
        let cases = vec![
            TriggerTestCase {
                query: "read src/main.rs".into(),
                expected_agent: "code-explorer".into(),
                should_trigger: true,
                runs_per_query: 1,
            },
            TriggerTestCase {
                query: "what is 2+2".into(),
                expected_agent: "".into(),
                should_trigger: false,
                runs_per_query: 1,
            },
        ];
        let actuals = vec![
            ("read src/main.rs".into(), Some("code-explorer".into())),
            ("what is 2+2".into(), None),
        ];
        let acc = TriggerAccuracy::evaluate(&cases, &actuals);
        assert_eq!(acc.precision, 1.0);
        assert_eq!(acc.recall, 1.0);
        assert_eq!(acc.f1, 1.0);
    }

    /// 验证不该触发却触发时会被统计为误报。
    #[test]
    fn test_trigger_accuracy_false_positive() {
        let cases = vec![TriggerTestCase {
            query: "what is 2+2".into(),
            expected_agent: "".into(),
            should_trigger: false,
            runs_per_query: 1,
        }];
        let actuals = vec![("what is 2+2".into(), Some("code-explorer".into()))];
        let acc = TriggerAccuracy::evaluate(&cases, &actuals);
        assert_eq!(acc.precision, 0.0); // 不该触发却触发了
    }
}
