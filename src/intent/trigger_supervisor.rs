/*!
三源融合意图分类器。

将关键词匹配（0 token，速度快）、LLM 语义分类
（约 500 token，惰性调用），以及来自 UserPromptSubmit 钩子的
激活建议，融合到单一的 [`IntentClassifier`] 实现中。
*/

use crate::intent::classifier::{KeywordClassifier, LlmIntentClassifier};
use crate::intent::{Intent, IntentClassifier};
use futures::future::BoxFuture;
use std::sync::{Arc, Mutex};

/// 用于快速路径（关键词直接命中）绕过的置信度阈值。
const HIGH_CONFIDENCE: f32 = 0.7;

/// 实现 [`IntentClassifier`] 的三源融合监督器。
pub struct TriggerSupervisor {
    /// 关键词分类器。
    keyword_classifier: KeywordClassifier,
    /**
    可选的 LLM 兜底分类器；为 `None` 时，融合逻辑降级为
    仅使用关键词 + 钩子（纯关键词模式，不消耗 LLM token）。
    */
    llm_classifier: Option<LlmIntentClassifier>,
    /**
    由 prepare 阶段在 UserPromptSubmit 钩子完成之后写入的共享槽位。
    P1 会在此处写入 `fire_lifecycle_hook().activate_skill`；
    该监督器每次分类时读取（并消费掉）它。
    */
    hook_activation_slot: Arc<Mutex<Option<(String, String)>>>,
}

impl TriggerSupervisor {
    /// 创建一个新的 [`TriggerSupervisor`]。
    pub fn new(
        keyword_classifier: KeywordClassifier,
        llm_classifier: Option<LlmIntentClassifier>,
        hook_activation_slot: Arc<Mutex<Option<(String, String)>>>,
    ) -> Self {
        Self {
            keyword_classifier,
            llm_classifier,
            hook_activation_slot,
        }
    }

    /**
    将三个来源融合为最终的意图。

    规则（顺序很重要）：
    1. 关键词高置信度（f32 ≥ 0.7）→ 快速路径，跳过 LLM
    2. LLM 高置信度 → 采用 LLM 的结果
    3. 两者置信度都低，但钩子槽位存在激活请求 → 采用钩子结果
    4. 全部置信度都低 → Fallback
    */
    fn fuse(kw: Intent, llm: Intent, hook: Option<(String, String)>) -> Intent {
        if kw.confidence().unwrap_or(0.0) >= HIGH_CONFIDENCE {
            return kw;
        }
        if llm.confidence().unwrap_or(0.0) >= HIGH_CONFIDENCE {
            return llm;
        }
        if let Some((skill_name, _reason)) = hook {
            return Intent::SkillRequired {
                skill_name,
                /*
                钩子激活是一次显式的运行时决策，而非概率性分类结果。
                将其置信度设置得高于路由器可配置的分类器阈值，
                以避免被第二道置信度门槛丢弃。
                */
                confidence: 1.0,
            };
        }
        Intent::Fallback
    }
}

impl IntentClassifier for TriggerSupervisor {
    /// 先运行关键词分类器，高置信度时直接返回；否则结合 LLM 与钩子信号融合出最终意图。
    fn classify<'a>(
        &'a self,
        user_input: &'a str,
        context: &'a [crate::llm::types::Message],
    ) -> BoxFuture<'a, Intent> {
        Box::pin(async move {
            /*
            无论走哪条路径都要消费掉这个按 turn 划分的槽位，
            避免高置信度决策把上一轮的钩子请求泄漏到下一轮。
            */
            let hook = self
                .hook_activation_slot
                .lock()
                .ok()
                .and_then(|mut guard| guard.take());
            // 第一步：关键词分类器（0 token，速度快）
            let kw = self.keyword_classifier.classify(user_input, context).await;

            // 快速路径：完全跳过 LLM
            if kw.confidence().unwrap_or(0.0) >= HIGH_CONFIDENCE {
                return kw;
            }

            // 第二步：LLM 语义分类器（如果可用）
            let llm = match &self.llm_classifier {
                Some(clf) => clf.classify(user_input, context).await,
                None => Intent::Fallback,
            };

            // 第三步：来自本轮 UserPromptSubmit 的钩子激活结果
            Self::fuse(kw, llm, hook)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 验证高置信度关键词结果优先被采用。
    #[test]
    fn keyword_high_confidence_takes_priority() {
        let kw = Intent::SkillRequired {
            skill_name: "docx".into(),
            confidence: 0.9,
        };
        let llm = Intent::SkillRequired {
            skill_name: "pdf".into(),
            confidence: 0.85,
        };
        let result = TriggerSupervisor::fuse(kw, llm, None);
        assert_eq!(result.skill_name().unwrap(), "docx");
    }

    /// 验证关键词置信度低时采用 LLM 结果。
    #[test]
    fn llm_wins_when_keyword_low() {
        let kw = Intent::SkillRequired {
            skill_name: "pdf".into(),
            confidence: 0.3,
        };
        let llm = Intent::SkillRequired {
            skill_name: "docx".into(),
            confidence: 0.85,
        };
        let result = TriggerSupervisor::fuse(kw, llm, None);
        assert_eq!(result.skill_name().unwrap(), "docx");
    }

    /// 验证关键词与 LLM 都低置信度时采用钩子请求。
    #[test]
    fn hook_adopted_when_both_low() {
        let hook = Some(("docx".into(), "detected .docx".into()));
        let result = TriggerSupervisor::fuse(Intent::Fallback, Intent::Fallback, hook);
        assert_eq!(result.skill_name().unwrap(), "docx");
        assert_eq!(result.confidence(), Some(1.0));
    }

    /// 验证所有信号都缺失时返回 Fallback。
    #[test]
    fn all_low_returns_fallback() {
        let r = TriggerSupervisor::fuse(Intent::Fallback, Intent::Fallback, None);
        assert!(matches!(r, Intent::Fallback));
    }

    /// 验证两者都高置信度时关键词仍优先于 LLM。
    #[test]
    fn keyword_beats_llm_even_when_both_high() {
        let kw = Intent::SkillRequired {
            skill_name: "brainstorming".into(),
            confidence: 0.95,
        };
        let llm = Intent::SkillRequired {
            skill_name: "writing-plans".into(),
            confidence: 0.9,
        };
        let r = TriggerSupervisor::fuse(kw, llm, None);
        assert_eq!(r.skill_name().unwrap(), "brainstorming");
    }
}
