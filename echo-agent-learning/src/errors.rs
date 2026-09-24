//! `Option`、`Result`、错误转换以及 `?` 运算符。

use thiserror::Error;

/// 各学习示例共用的错误类型。
#[derive(Debug, Error, PartialEq, Eq)]
pub enum LearningError {
    /// 任务标题为空。
    #[error("task title cannot be empty")]
    EmptyTitle,
    /// 限制值不是合法的正整数。
    #[error("limit must be a positive integer, got {0}")]
    InvalidLimit(String),
    /// 任务已存在，携带重复的任务标识。
    #[error("task already exists: {0}")]
    DuplicateTask(String),
    /// 缺少必填字段，携带字段名称。
    #[error("required field is missing: {0}")]
    MissingField(&'static str),
    /// 共享状态当前正被借用，无法完成本次操作。
    #[error("shared state is temporarily borrowed")]
    BorrowConflict,
    /// 共享状态的锁已被污染（持锁线程发生了 panic）。
    #[error("shared state lock was poisoned")]
    PoisonedLock,
    /// 子任务（subagent）执行失败，携带失败原因。
    #[error("subagent task failed: {0}")]
    SubagentJoin(String),
    /// 结果送达前 channel 已被关闭。
    #[error("channel closed before the result was delivered")]
    ChannelClosed,
    /// 操作被取消。
    #[error("operation was cancelled")]
    Cancelled,
    /// 操作超时，携带超时的毫秒数。
    #[error("operation timed out after {0} ms")]
    TimedOut(u64),
}

/// 解析一个严格为正数的限制值，输入非法时返回错误而不会 panic。
pub fn parse_positive_limit(input: &str) -> Result<usize, LearningError> {
    let value = input
        .parse::<usize>()
        .map_err(|_| LearningError::InvalidLimit(input.to_string()))?;
    if value == 0 {
        return Err(LearningError::InvalidLimit(input.to_string()));
    }
    Ok(value)
}

/// 将一个可选的借用字段转换为经过校验的、拥有所有权的值。
pub fn required_text(field: &'static str, value: Option<&str>) -> Result<String, LearningError> {
    let value = value.ok_or(LearningError::MissingField(field))?.trim();
    if value.is_empty() {
        return Err(LearningError::MissingField(field));
    }
    Ok(value.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 验证正整数被正确解析，而零和非数字输入均被判定为错误。
    #[test]
    fn parses_positive_limit() {
        assert_eq!(parse_positive_limit("3"), Ok(3));
        assert!(parse_positive_limit("0").is_err());
        assert!(parse_positive_limit("三").is_err());
    }

    /// 验证必填文本字段的校验逻辑：去除空白后为空或缺失都应返回错误。
    #[test]
    fn validates_required_text() {
        assert_eq!(
            required_text("name", Some(" Echo ")),
            Ok("Echo".to_string())
        );
        assert_eq!(
            required_text("name", None),
            Err(LearningError::MissingField("name"))
        );
    }
}
