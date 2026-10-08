/*!
意图路由器（Intent Router）—— 在进入 ReAct 循环之前进行轻量级意图分类。

在每一轮用户输入开始时提供一个决策点：

| 意图取值             | 行为                                           |
|---------------------|-----------------------------------------------|
| `DirectAnswer`      | 跳过 ReAct，直接调用 LLM                       |
| `SkillRequired`     | 先激活技能，再运行 ReAct                        |
| `Fallback`          | 按正常的 ReAct 循环继续执行                     |

# 快速开始

```rust,ignore
use echo_agent::prelude::*;
use echo_agent::intent::{IntentRouter, Intent};

let router = IntentRouter::new(
    Box::new(MyClassifier::default()),
    IntentRouterConfig::default(),
);

let agent = ReactAgentBuilder::new()
    .intent_router(router)
    .build()?;
```
*/

use crate::llm::types::Message;
use futures::future::BoxFuture;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::time::Duration;
use tokio_util::sync::CancellationToken;

// ── Intent ───────────────────────────────────────────────────────────

/// [`IntentClassifier`] 产生的分类结果。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Intent {
    /// 简单问题 —— 不进入 ReAct 循环，直接回答。
    DirectAnswer {
        /// 分类结果的置信度。
        confidence: f32,
    },
    /// 在进入 ReAct 循环之前，先按名称激活某个技能。
    SkillRequired {
        /// 需要激活的技能名称。
        skill_name: String,
        /// 分类结果的置信度。
        confidence: f32,
    },
    /// 置信度过低 —— 回退到正常的 ReAct 循环。
    Fallback,
}

impl Intent {
    /// 返回置信度（如果有的话）。
    pub fn confidence(&self) -> Option<f32> {
        match self {
            Intent::DirectAnswer { confidence } => Some(*confidence),
            Intent::SkillRequired { confidence, .. } => Some(*confidence),
            Intent::Fallback => None,
        }
    }

    /// 如果是 `SkillRequired` 意图，返回其技能名称。
    pub fn skill_name(&self) -> Option<&str> {
        match self {
            Intent::SkillRequired { skill_name, .. } => Some(skill_name.as_str()),
            _ => None,
        }
    }
}

// ── IntentClassifier ───────────────────────────────────────────────────

/**
可插拔意图分类策略的 trait。

具体实现可以使用关键词匹配、基于 LLM 的分类、embedding 相似度，
或者以上几种方式的任意组合。
*/
pub trait IntentClassifier: Send + Sync {
    /// 结合可选的会话上下文，对用户输入进行分类。
    fn classify<'a>(&'a self, user_input: &'a str, context: &'a [Message])
    -> BoxFuture<'a, Intent>;
}

// ── IntentRouterConfig ────────────────────────────────────────────────

/// [`IntentRouter`] 的配置。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentRouterConfig {
    /**
    信任一次分类结果所需的最低置信度。
    低于该阈值的结果会被当作 [`Intent::Fallback`] 处理。
    */
    pub confidence_threshold: f32,
    /// 是否启用 `DirectAnswer` 快捷路径。
    pub enable_direct_answer: bool,
    /// 是否启用 `SkillRequired` 快捷路径。
    pub enable_skill_routing: bool,
    /// 路由前分类阶段允许消耗的最长时间。
    #[serde(default = "default_classification_timeout_ms")]
    pub classification_timeout_ms: u64,
}

/// `classification_timeout_ms` 的默认值（毫秒）。
fn default_classification_timeout_ms() -> u64 {
    5_000
}

impl Default for IntentRouterConfig {
    /// 返回默认路由配置：置信度阈值 0.7，并启用各快捷路径。
    fn default() -> Self {
        Self {
            confidence_threshold: 0.7,
            enable_direct_answer: true,
            enable_skill_routing: true,
            classification_timeout_ms: default_classification_timeout_ms(),
        }
    }
}

// ── IntentRouter ──────────────────────────────────────────────────────

/**
位于 ReAct 循环入口处的核心路由组件。

当挂载到 [`ReactAgent`](crate::agent::ReactAgent) 上时，每条用户
消息都会先经过配置好的 [`IntentClassifier`] 进行分类；根据返回的
[`Intent`]，Agent 会选择直接给出答案、激活某个技能，或者
按标准 ReAct 循环继续执行。
*/
#[derive(Clone)]
pub struct IntentRouter {
    /// 用于对用户输入进行分类的分类器实现。
    classifier: std::sync::Arc<dyn IntentClassifier>,
    /// 路由器的行为配置。
    config: IntentRouterConfig,
    /// 允许被激活的技能名称集合；为 `None` 时不做限制。
    available_skills: Option<std::sync::Arc<HashSet<String>>>,
}

impl IntentRouter {
    /// 基于分类器与配置创建一个新的路由器。
    pub fn new(classifier: Box<dyn IntentClassifier>, config: IntentRouterConfig) -> Self {
        Self {
            classifier: std::sync::Arc::from(classifier),
            config,
            available_skills: None,
        }
    }

    /// 将可执行的技能决策限制在调用方所使用的技能目录范围内。
    pub fn with_available_skills(mut self, skills: impl IntoIterator<Item = String>) -> Self {
        self.available_skills = Some(std::sync::Arc::new(skills.into_iter().collect()));
        self
    }

    /// 对用户输入进行分类，并返回路由决策。
    pub async fn classify(&self, user_input: &str, context: &[Message]) -> Intent {
        self.classify_with_cancel(user_input, context, CancellationToken::new())
            .await
    }

    /// 在调用方的取消域与配置的截止时间内进行分类。
    pub async fn classify_with_cancel(
        &self,
        user_input: &str,
        context: &[Message],
        cancel: CancellationToken,
    ) -> Intent {
        let timeout = Duration::from_millis(self.config.classification_timeout_ms.max(1));
        let raw = tokio::select! {
            _ = cancel.cancelled() => return Intent::Fallback,
            result = tokio::time::timeout(timeout, self.classifier.classify(user_input, context)) => {
                match result {
                    Ok(intent) => intent,
                    Err(_) => return Intent::Fallback,
                }
            }
        };
        match raw {
            Intent::DirectAnswer { confidence } => {
                if self.config.enable_direct_answer
                    && confidence >= self.config.confidence_threshold
                {
                    Intent::DirectAnswer { confidence }
                } else {
                    Intent::Fallback
                }
            }
            Intent::SkillRequired {
                skill_name,
                confidence,
            } => {
                if self.config.enable_skill_routing
                    && confidence >= self.config.confidence_threshold
                    && self
                        .available_skills
                        .as_ref()
                        .is_none_or(|skills| skills.contains(&skill_name))
                {
                    Intent::SkillRequired {
                        skill_name,
                        confidence,
                    }
                } else {
                    Intent::Fallback
                }
            }
            Intent::Fallback => Intent::Fallback,
        }
    }
}

// ── 重新导出便捷类型 ────────────────────────────────────────

pub use self::classifier::{
    ChainedClassifier, KeywordClassifier, KeywordClassifierConfig, LlmIntentClassifier,
    SkillDescription,
};
pub use self::trigger_supervisor::TriggerSupervisor;

/// 意图分类器实现（关键词 / LLM 分类器等）。
mod classifier;
/// 融合多路触发信号的触发监督器。
mod trigger_supervisor;
