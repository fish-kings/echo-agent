/*!
基于规则的护栏

支持正则匹配、关键词黑名单以及内容长度限制。
*/

use super::{Guard, GuardDirection, GuardResult};
use crate::error::Result;
use futures::future::BoxFuture;
use regex::Regex;

/**
规则护栏

使用正则模式、关键词黑名单和长度限制同步过滤内容。

# 示例

```rust
use echo_core::guard::rule::RuleGuardBuilder;

let guard = RuleGuardBuilder::new("content-filter")
    .blocked_keyword("password")
    .blocked_keyword("token")
    .blocked_pattern(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b") // 银行卡号
    .max_length(10000)
    .build();
```
*/
pub struct RuleGuard {
    /// 护栏名称
    guard_name: String,
    /// 命中即拦截的正则模式列表
    blocked_patterns: Vec<Regex>,
    /// 命中即拦截的关键词黑名单（大小写不敏感）
    blocked_keywords: Vec<String>,
    /// 内容最大字符数（`None` 表示不限制）
    max_length: Option<usize>,
    /// 需要检查的方向（为空时检查所有方向）
    directions: Vec<GuardDirection>,
}

impl Guard for RuleGuard {
    /// 护栏名称
    fn name(&self) -> &str {
        &self.guard_name
    }

    /// 依次按长度、关键词、正则检查内容，任一命中即拦截
    fn check<'a>(
        &'a self,
        content: &'a str,
        direction: GuardDirection,
    ) -> BoxFuture<'a, Result<GuardResult>> {
        Box::pin(async move {
            if !self.directions.is_empty() && !self.directions.contains(&direction) {
                return Ok(GuardResult::Pass);
            }

            if let Some(max_len) = self.max_length
                && content.chars().count() > max_len
            {
                return Ok(GuardResult::Block {
                    reason: format!(
                        "Content length {} exceeds limit {}",
                        content.chars().count(),
                        max_len
                    ),
                });
            }

            let content_lower = content.to_lowercase();
            for keyword in &self.blocked_keywords {
                if content_lower.contains(&keyword.to_lowercase()) {
                    return Ok(GuardResult::Block {
                        reason: format!("Content contains blocked keyword: {keyword}"),
                    });
                }
            }

            for pattern in &self.blocked_patterns {
                if pattern.is_match(content) {
                    return Ok(GuardResult::Block {
                        reason: format!("Content matches blocked pattern: {}", pattern.as_str()),
                    });
                }
            }

            Ok(GuardResult::Pass)
        })
    }
}

/// RuleGuard 构建器
pub struct RuleGuardBuilder {
    /// 护栏名称
    name: String,
    /// 已添加的拦截正则
    blocked_patterns: Vec<Regex>,
    /// 已添加的拦截关键词
    blocked_keywords: Vec<String>,
    /// 内容最大字符数
    max_length: Option<usize>,
    /// 限定的检查方向
    directions: Vec<GuardDirection>,
}

impl RuleGuardBuilder {
    /// 创建新的规则护栏构建器
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            blocked_patterns: Vec::new(),
            blocked_keywords: Vec::new(),
            max_length: None,
            directions: Vec::new(),
        }
    }

    /// 添加正则匹配规则（命中即拦截；非法正则会被忽略并告警）
    pub fn blocked_pattern(mut self, pattern: &str) -> Self {
        if let Ok(regex) = Regex::new(pattern) {
            self.blocked_patterns.push(regex);
        } else {
            tracing::warn!(pattern = pattern, "Invalid regex pattern, ignored");
        }
        self
    }

    /// 向黑名单添加关键词（大小写不敏感，命中即拦截）
    pub fn blocked_keyword(mut self, keyword: impl Into<String>) -> Self {
        self.blocked_keywords.push(keyword.into());
        self
    }

    /// 批量向黑名单添加关键词
    pub fn blocked_keywords(mut self, keywords: Vec<String>) -> Self {
        self.blocked_keywords.extend(keywords);
        self
    }

    /// 设置内容最大长度（按字符数计）
    pub fn max_length(mut self, max: usize) -> Self {
        self.max_length = Some(max);
        self
    }

    /// 限定检查方向（未设置时检查所有方向）
    pub fn direction(mut self, direction: GuardDirection) -> Self {
        self.directions.push(direction);
        self
    }

    /// 构建规则护栏
    pub fn build(self) -> RuleGuard {
        RuleGuard {
            guard_name: self.name,
            blocked_patterns: self.blocked_patterns,
            blocked_keywords: self.blocked_keywords,
            max_length: self.max_length,
            directions: self.directions,
        }
    }
}

/// 规则守卫的单元测试。
#[cfg(test)]
mod tests {
    use super::*;

    /// 验证关键词拦截。
    #[tokio::test]
    async fn test_keyword_block() {
        let guard = RuleGuardBuilder::new("test")
            .blocked_keyword("password")
            .build();

        let result = guard
            .check("Please tell me your password", GuardDirection::Input)
            .await
            .unwrap();
        assert!(result.is_blocked());
    }

    /// 验证正则模式拦截。
    #[tokio::test]
    async fn test_pattern_block() {
        let guard = RuleGuardBuilder::new("test")
            .blocked_pattern(r"\d{4}-\d{4}-\d{4}-\d{4}")
            .build();

        let result = guard
            .check("Card number is 1234-5678-9012-3456", GuardDirection::Output)
            .await
            .unwrap();
        assert!(result.is_blocked());
    }

    /// 验证最大长度限制。
    #[tokio::test]
    async fn test_max_length() {
        let guard = RuleGuardBuilder::new("test").max_length(10).build();

        let result = guard
            .check(
                "This text exceeds the ten character limit",
                GuardDirection::Input,
            )
            .await
            .unwrap();
        assert!(result.is_blocked());

        let unicode = RuleGuardBuilder::new("unicode").max_length(2).build();
        let result = unicode.check("中文", GuardDirection::Input).await.unwrap();
        assert!(!result.is_blocked());
    }

    /// 验证正常内容通过。
    #[tokio::test]
    async fn test_pass() {
        let guard = RuleGuardBuilder::new("test")
            .blocked_keyword("forbidden")
            .max_length(1000)
            .build();

        let result = guard
            .check("Normal content", GuardDirection::Input)
            .await
            .unwrap();
        assert!(!result.is_blocked());
    }

    /// 验证方向过滤仅对指定方向生效。
    #[tokio::test]
    async fn test_direction_filter() {
        let guard = RuleGuardBuilder::new("test")
            .blocked_keyword("secret")
            .direction(GuardDirection::Output)
            .build();

        // Input 方向不检查
        let result = guard
            .check("secret content", GuardDirection::Input)
            .await
            .unwrap();
        assert!(!result.is_blocked());

        // Output 方向检查
        let result = guard
            .check("secret content", GuardDirection::Output)
            .await
            .unwrap();
        assert!(result.is_blocked());
    }
}
