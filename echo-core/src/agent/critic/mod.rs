//! Critic trait 及其内置实现

use crate::agent::types::Critique;
use crate::error::Result;
use futures::future::BoxFuture;

// ── Critic trait ──────────────────────────────────────────────────────────────

/**
Critic trait —— 评估 Agent 生成的响应

与 `Planner` 相对称，这是 Self-Reflection（自我反思）模式中的一个可插拔
评估组件。实现 `Critic` 即可自定义评估策略。
*/
pub trait Critic: Send + Sync {
    /**
    评估一次响应

    - `task`：原始任务描述
    - `answer`：Agent 生成的响应
    - `context`：额外上下文（例如情景记忆、历史结果等）
    */
    fn critique<'a>(
        &'a self,
        task: &'a str,
        answer: &'a str,
        context: &'a str,
    ) -> BoxFuture<'a, Result<Critique>>;

    /// Critic 的人类可读名称
    fn name(&self) -> &str {
        "anonymous"
    }
}

// ── 重新导出 composite ──────────────────────────────────────────────────────────

/// 组合评估器实现
mod composite;
pub use composite::{CompositeCritic, CompositeStrategy};

// ── StaticCritic ──────────────────────────────────────────────────────────────

/// 静态 Critic：始终返回固定的结果（用于测试）
pub struct StaticCritic {
    /// 固定返回的评估得分。
    score: f64,
    /// 固定返回的是否通过标志。
    passed: bool,
    /// 固定返回的反馈文本。
    feedback: String,
    /// 固定返回的改进建议列表。
    suggestions: Vec<String>,
}

impl StaticCritic {
    /**
    创建一个静态评估器

    # 参数
    * `score` - 评估得分（通常范围为 0.0-10.0）
    * `passed` - 是否通过评估
    * `feedback` - 评估反馈文本
    */
    pub fn new(score: f64, passed: bool, feedback: impl Into<String>) -> Self {
        Self {
            score,
            passed,
            feedback: feedback.into(),
            suggestions: Vec::new(),
        }
    }

    /// 设置改进建议列表
    pub fn with_suggestions(mut self, suggestions: Vec<String>) -> Self {
        self.suggestions = suggestions;
        self
    }

    /// 创建一个始终通过的 Critic
    pub fn always_pass() -> Self {
        Self::new(9.0, true, "Response quality is excellent")
    }

    /// 创建一个始终不通过的 Critic
    pub fn always_fail() -> Self {
        Self::new(3.0, false, "Response does not meet requirements")
    }
}

impl Critic for StaticCritic {
    /// 直接返回构造时固定的评估结果，忽略传入的任务、答案与上下文。
    fn critique<'a>(
        &'a self,
        _task: &'a str,
        _answer: &'a str,
        _context: &'a str,
    ) -> BoxFuture<'a, Result<Critique>> {
        Box::pin(async move {
            Ok(Critique {
                score: self.score,
                passed: self.passed,
                feedback: self.feedback.clone(),
                suggestions: self.suggestions.clone(),
            })
        })
    }

    /// 返回固定名称 "static"。
    fn name(&self) -> &str {
        "static"
    }
}

// ── ThresholdCritic ───────────────────────────────────────────────────────────

/**
阈值 Critic：通过内部另一个 Critic 完成评估，再依据阈值判定是否通过

用于包装其他 Critic，改写其 `passed` 判定结果。
*/
pub struct ThresholdCritic {
    /// 实际负责打分与生成反馈的内部评估器。
    inner: Box<dyn Critic>,
    /// 通过阈值：得分大于等于该值才判定为通过。
    threshold: f64,
}

impl ThresholdCritic {
    /**
    创建一个阈值评估器

    # 参数
    * `inner` - 内部评估器，用于实际打分与生成反馈
    * `threshold` - 通过阈值，得分 >= threshold 时结果被标记为通过
    */
    pub fn new(inner: impl Critic + 'static, threshold: f64) -> Self {
        Self {
            inner: Box::new(inner),
            threshold,
        }
    }
}

impl Critic for ThresholdCritic {
    /// 调用内部评估器获取结果，再依据阈值重新判定 `passed` 字段。
    fn critique<'a>(
        &'a self,
        task: &'a str,
        answer: &'a str,
        context: &'a str,
    ) -> BoxFuture<'a, Result<Critique>> {
        Box::pin(async move {
            let mut critique = self.inner.critique(task, answer, context).await?;
            critique.passed = critique.score >= self.threshold;
            Ok(critique)
        })
    }

    /// 透传内部评估器的名称。
    fn name(&self) -> &str {
        self.inner.name()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // 验证 always_pass 构造的 Critic 结果为通过且高分
    #[tokio::test]
    async fn test_static_critic_always_pass() {
        let critic = StaticCritic::always_pass();
        let critique = critic.critique("task", "answer", "").await.unwrap();
        assert!(critique.passed);
        assert!(critique.score >= 8.0);
    }

    // 验证 always_fail 构造的 Critic 结果为不通过
    #[tokio::test]
    async fn test_static_critic_always_fail() {
        let critic = StaticCritic::always_fail();
        let critique = critic.critique("task", "answer", "").await.unwrap();
        assert!(!critique.passed);
    }

    // 验证内部得分低于阈值时最终判定为不通过
    #[tokio::test]
    async fn test_threshold_critic() {
        let inner = StaticCritic::new(6.0, true, "OK");
        let critic = ThresholdCritic::new(inner, 8.0);
        let critique = critic.critique("task", "answer", "").await.unwrap();
        assert!(!critique.passed); // 6.0 < 8.0
        assert_eq!(critique.score, 6.0);
    }

    // 验证内部得分达到阈值时最终判定为通过（覆盖内部原本的 passed 值）
    #[tokio::test]
    async fn test_threshold_critic_passes() {
        let inner = StaticCritic::new(9.0, false, "Very good");
        let critic = ThresholdCritic::new(inner, 8.0);
        let critique = critic.critique("task", "answer", "").await.unwrap();
        assert!(critique.passed); // 9.0 >= 8.0
    }
}
