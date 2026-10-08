/*!
LLM-as-Judge 评分器 —— 使用一个 agent 来给评测输出打分。

灵感来自 `skill-creator/agents/grader.md`。评分器接收一条断言、任务描述
以及 agent 的输出，然后判断该断言是否通过，并提取支撑证据。
*/

use crate::agent::Agent;
use crate::agent::config::DEFAULT_TOKEN_LIMIT;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// 针对 agent 输出要检查的单条断言。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Assertion {
    /// 该断言的唯一标识符。
    pub id: String,
    /// 要检查的内容（自然语言描述，例如"输出包含一个合法的 JSON 对象"）。
    pub check: String,
    /// 给评分器的预期结果提示。
    pub expected: String,
}

/// 对单条断言评分的结果。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradeResult {
    /// 对应断言的 ID
    pub assertion_id: String,
    /// 该断言是否通过
    pub passed: bool,
    /// 评分器给出的置信度（0.0-1.0）
    pub confidence: f64,
    /// 支撑判断的证据文本
    pub evidence: String,
    /// 评分器的推理说明
    pub reasoning: String,
}

/// 针对一个评测用例的完整评分输出。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradingReport {
    /// 对应的评测用例 ID
    pub case_id: String,
    /// 每条断言的评分结果列表
    pub results: Vec<GradeResult>,
    /// 通过率（0.0-1.0）
    pub pass_rate: f64,
    /// 总体评估摘要
    pub overall_assessment: String,
}

/// 基于 LLM 来评估 agent 输出的评分器。
#[derive(Clone)]
pub struct LlmGrader {
    /// 评分器使用的系统提示词
    grader_prompt: String,
}

impl Default for LlmGrader {
    /// 创建默认实例。
    fn default() -> Self {
        Self::new()
    }
}

impl LlmGrader {
    /// 创建一个使用默认评分提示词的 LlmGrader
    pub fn new() -> Self {
        Self {
            grader_prompt: concat!(
                "You are an expert evaluator. You will receive:\n",
                "1. A TASK that an agent was asked to complete\n",
                "2. The agent's OUTPUT\n",
                "3. A list of ASSERTIONS to check\n\n",
                "For each assertion, determine:\n",
                "- PASSED: true/false\n",
                "- CONFIDENCE: 0.0-1.0\n",
                "- EVIDENCE: quote the specific part of the output that proves or disproves\n",
                "- REASONING: explain your judgment\n\n",
                "Output format (JSON):\n",
                "{\n",
                "  \"results\": [\n",
                "    {\"assertion_id\": \"...\", \"passed\": bool, \"confidence\": 0.0-1.0,\n",
                "     \"evidence\": \"...\", \"reasoning\": \"...\"}\n",
                "  ],\n",
                "  \"overall_assessment\": \"brief summary\"\n",
                "}\n\n",
                "Be strict but fair. If evidence is ambiguous, mark as not passed."
            )
            .to_string(),
        }
    }

    /// 针对一组断言给 agent 的输出打分。
    pub async fn grade(
        &self,
        agent: &dyn Agent,
        task: &str,
        output: &str,
        assertions: &[Assertion],
    ) -> GradingReport {
        self.grade_with_trajectory(agent, task, output, assertions, None)
            .await
    }

    /**
    携带可选的执行轨迹摘要给 agent 的输出打分。

    `trajectory_summary` 可以包含已执行的工具调用、已编辑的文件等信息，
    为评分器提供关于 agent 执行路径的更多上下文。
    */
    pub async fn grade_with_trajectory(
        &self,
        agent: &dyn Agent,
        task: &str,
        output: &str,
        assertions: &[Assertion],
        trajectory_summary: Option<&str>,
    ) -> GradingReport {
        let assertions_text: String = assertions
            .iter()
            .map(|a| {
                format!(
                    "- [{}] Check: {}\n  Expected: {}",
                    a.id, a.check, a.expected
                )
            })
            .collect::<Vec<_>>()
            .join("\n");

        let mut sections = vec![self.grader_prompt.clone(), format!("--- TASK ---\n{task}")];

        if let Some(traj) = trajectory_summary {
            sections.push(format!("--- TRAJECTORY ---\n{traj}"));
        }

        sections.push(format!(
            "--- OUTPUT ---\n{}",
            truncate(output, DEFAULT_TOKEN_LIMIT)
        ));
        sections.push(format!("--- ASSERTIONS ---\n{assertions_text}"));
        sections.push("Provide your grading in the JSON format specified.".to_string());

        let prompt = sections.join("\n\n");

        match agent.execute(&prompt).await {
            Ok(raw) => Self::parse_grading_response(&raw, assertions),
            Err(error) => Self::failed_report(assertions, format!("grader agent failed: {error}")),
        }
    }

    /// 解析评分响应 JSON，并在解析失败时优雅降级。
    fn parse_grading_response(raw: &str, assertions: &[Assertion]) -> GradingReport {
        // 尝试解析结构化响应
        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(raw) {
            let Some(rows) = parsed.get("results").and_then(serde_json::Value::as_array) else {
                return Self::failed_report(assertions, "grader response omitted results".into());
            };
            let expected: HashSet<&str> = assertions.iter().map(|item| item.id.as_str()).collect();
            let mut by_id = HashMap::with_capacity(rows.len());
            for row in rows {
                let Some(assertion_id) =
                    row.get("assertion_id").and_then(serde_json::Value::as_str)
                else {
                    return Self::failed_report(
                        assertions,
                        "grader row omitted assertion_id".into(),
                    );
                };
                if !expected.contains(assertion_id) || by_id.contains_key(assertion_id) {
                    return Self::failed_report(
                        assertions,
                        format!(
                            "grader returned unknown or duplicate assertion id {assertion_id:?}"
                        ),
                    );
                }
                let Some(confidence) = row.get("confidence").and_then(serde_json::Value::as_f64)
                else {
                    return Self::failed_report(
                        assertions,
                        format!("grader row {assertion_id:?} omitted confidence"),
                    );
                };
                if !confidence.is_finite() || !(0.0..=1.0).contains(&confidence) {
                    return Self::failed_report(
                        assertions,
                        format!("grader row {assertion_id:?} has invalid confidence"),
                    );
                }
                by_id.insert(
                    assertion_id.to_string(),
                    GradeResult {
                        assertion_id: assertion_id.to_string(),
                        passed: row
                            .get("passed")
                            .and_then(serde_json::Value::as_bool)
                            .unwrap_or(false),
                        confidence,
                        evidence: row
                            .get("evidence")
                            .and_then(serde_json::Value::as_str)
                            .unwrap_or_default()
                            .to_string(),
                        reasoning: row
                            .get("reasoning")
                            .and_then(serde_json::Value::as_str)
                            .unwrap_or_default()
                            .to_string(),
                    },
                );
            }
            if by_id.len() != assertions.len() {
                return Self::failed_report(
                    assertions,
                    "grader response did not cover every assertion".into(),
                );
            }
            let results: Vec<GradeResult> = assertions
                .iter()
                .filter_map(|assertion| by_id.remove(&assertion.id))
                .collect();
            let pass_count = results.iter().filter(|result| result.passed).count();
            let pass_rate = if assertions.is_empty() {
                1.0
            } else {
                pass_count as f64 / assertions.len() as f64
            };

            GradingReport {
                case_id: String::new(),
                results,
                pass_rate,
                overall_assessment: parsed
                    .get("overall_assessment")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or_default()
                    .into(),
            }
        } else {
            Self::failed_report(assertions, "grader response was not valid JSON".into())
        }
    }

    /// 构造一份统一失败的评分报告（所有断言均标记为未通过）
    fn failed_report(assertions: &[Assertion], reason: String) -> GradingReport {
        GradingReport {
            case_id: String::new(),
            results: assertions
                .iter()
                .map(|assertion| GradeResult {
                    assertion_id: assertion.id.clone(),
                    passed: false,
                    confidence: 0.0,
                    evidence: String::new(),
                    reasoning: reason.clone(),
                })
                .collect(),
            pass_rate: if assertions.is_empty() { 1.0 } else { 0.0 },
            overall_assessment: reason,
        }
    }
}

/// 按字符数截断字符串
fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        s.chars().take(max).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_valid_response() {
        let raw = r#"{"results":[{"assertion_id":"a1","passed":true,"confidence":0.9,"evidence":"found","reasoning":"clear"}],"overall_assessment":"good"}"#;
        let assertions = vec![Assertion {
            id: "a1".into(),
            check: "check".into(),
            expected: "yes".into(),
        }];
        let report = LlmGrader::parse_grading_response(raw, &assertions);
        assert_eq!(report.pass_rate, 1.0);
        assert_eq!(report.results.len(), 1);
        assert!(report.results[0].passed);
    }
}
