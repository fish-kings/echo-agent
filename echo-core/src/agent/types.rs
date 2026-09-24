//! 共享的 agent 类型定义

use serde::{Deserialize, Serialize};

/// 评估结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Critique {
    /// 质量得分（0.0 - 10.0）
    pub score: f64,
    /// 是否通过了质量阈值
    pub passed: bool,
    /// 详细反馈
    pub feedback: String,
    /// 改进建议
    #[serde(default)]
    pub suggestions: Vec<String>,
}

/// 评估结果的结构化输出（用于 LLM 的 JSON 解析）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CritiqueOutput {
    /// 质量得分（0.0 - 10.0）
    pub score: f64,
    /// 是否通过了质量阈值
    pub passed: bool,
    /// 详细反馈
    pub feedback: String,
    /// 改进建议
    #[serde(default)]
    pub suggestions: Vec<String>,
}

impl From<CritiqueOutput> for Critique {
    /// 将 LLM 解析出的结构化输出转换为内部使用的 `Critique`。
    fn from(output: CritiqueOutput) -> Self {
        Self {
            score: output.score,
            passed: output.passed,
            feedback: output.feedback,
            suggestions: output.suggestions,
        }
    }
}

/// 返回 `Critique` 结构对应的 JSON Schema（用于 LLM 结构化输出）
pub fn critique_output_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "score": {
                "type": "number",
                "description": "Quality score (0.0 to 10.0, 10.0 is highest)"
            },
            "passed": {
                "type": "boolean",
                "description": "Whether quality standards were met"
            },
            "feedback": {
                "type": "string",
                "description": "Detailed evaluation feedback, explaining strengths and weaknesses"
            },
            "suggestions": {
                "type": "array",
                "items": { "type": "string" },
                "description": "List of specific improvement suggestions"
            }
        },
        "required": ["score", "passed", "feedback", "suggestions"],
        "additionalProperties": false
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // 验证从 JSON 字符串正确解析出 CritiqueOutput
    #[test]
    fn test_critique_output_parse() {
        let json = r#"{"score": 8.5, "passed": true, "feedback": "Accurate response", "suggestions": ["Could be more detailed"]}"#;
        let output: CritiqueOutput = serde_json::from_str(json).unwrap();
        assert_eq!(output.score, 8.5);
        assert!(output.passed);
        assert_eq!(output.suggestions.len(), 1);
    }

    // 验证缺少 suggestions 字段时使用默认空列表
    #[test]
    fn test_critique_output_default_suggestions() {
        let json = r#"{"score": 6.0, "passed": false, "feedback": "Incomplete"}"#;
        let output: CritiqueOutput = serde_json::from_str(json).unwrap();
        assert!(output.suggestions.is_empty());
    }

    // 验证生成的 JSON Schema 结构与字段符合预期
    #[test]
    fn test_critique_schema() {
        let schema = critique_output_schema();
        assert!(schema.is_object());
        assert!(schema["properties"]["score"].is_object());
        assert!(schema["properties"]["passed"].is_object());
        assert_eq!(schema["additionalProperties"], serde_json::json!(false));
        assert_eq!(
            schema["required"],
            serde_json::json!(["score", "passed", "feedback", "suggestions"])
        );
    }
}
