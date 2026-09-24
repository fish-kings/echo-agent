//! 基于 serde 的配置与线上格式（wire-format）转换。

use crate::errors::{LearningError, required_text};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// 描述一个 agent 配置档案：名称、最大迭代次数以及可用工具列表。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentProfile {
    /// agent 名称。
    pub name: String,
    /// 允许的最大迭代次数。
    pub max_iterations: usize,
    /// 该 agent 可用的工具名称列表；反序列化时缺省为空列表。
    #[serde(default)]
    pub tools: Vec<String>,
}

impl AgentProfile {
    /// 结构化反序列化与领域校验是两个独立的步骤，本方法只负责后者。
    pub fn validate(&self) -> Result<(), LearningError> {
        required_text("name", Some(&self.name))?;
        if self.max_iterations == 0 {
            return Err(LearningError::InvalidLimit(self.max_iterations.to_string()));
        }
        Ok(())
    }
}

/// 解析或校验 `AgentProfile` 时可能出现的错误。
#[derive(Debug, Error)]
pub enum ProfileError {
    /// 输入不是合法的 JSON。
    #[error("agent profile is not valid JSON: {0}")]
    Json(#[from] serde_json::Error),
    /// 配置内容违反了领域校验规则。
    #[error("agent profile violates a domain rule: {0}")]
    Validation(#[from] LearningError),
}

/// 将 `AgentProfile` 序列化为格式化（pretty-printed）的 JSON 字符串。
pub fn profile_to_json(profile: &AgentProfile) -> serde_json::Result<String> {
    serde_json::to_string_pretty(profile)
}

/// 从 JSON 字符串反序列化出 `AgentProfile`，不做额外的领域校验。
pub fn profile_from_json(input: &str) -> serde_json::Result<AgentProfile> {
    serde_json::from_str(input)
}

/// 从 JSON 字符串解析出 `AgentProfile` 并完成领域校验。
pub fn parse_and_validate_profile(input: &str) -> Result<AgentProfile, ProfileError> {
    let profile = profile_from_json(input)?;
    profile.validate()?;
    Ok(profile)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 验证 `AgentProfile` 序列化再反序列化后能还原出等价的值。
    #[test]
    fn profile_round_trips() -> serde_json::Result<()> {
        let profile = AgentProfile {
            name: "assistant".to_string(),
            max_iterations: 8,
            tools: vec!["search".to_string()],
        };
        let json = profile_to_json(&profile)?;
        assert_eq!(profile_from_json(&json)?, profile);
        Ok(())
    }

    /// 验证领域校验与 JSON 解析是相互独立的两类错误来源。
    #[test]
    fn validation_is_distinct_from_json_parsing() -> Result<(), ProfileError> {
        let invalid = r#"{"name":" ","max_iterations":0}"#;
        assert!(matches!(
            parse_and_validate_profile(invalid),
            Err(ProfileError::Validation(LearningError::MissingField(
                "name"
            )))
        ));
        assert!(matches!(
            parse_and_validate_profile(r#"{"name":"assistant","max_iterations":0}"#),
            Err(ProfileError::Validation(LearningError::InvalidLimit(_)))
        ));

        assert!(matches!(
            parse_and_validate_profile("not json"),
            Err(ProfileError::Json(_))
        ));
        Ok(())
    }
}
