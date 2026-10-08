/*! 内置的意图分类器实现。
 *
 * 提供两种分类策略：
 *
 * | 策略 | 适用场景 | 成本 |
 * |----------|-------------|------|
 * | [`KeywordClassifier`] | 快速、零成本的第一遍过滤；触发词全部由产品层提供 | 0 token |
 * | [`LlmIntentClassifier`] | 语义理解；适配任意技能集合 | 约 500 token |
 *
 * 使用 [`ChainedClassifier`] 将两者组合：先尝试关键词匹配，失败后回退到 LLM。
 */

use super::{Intent, IntentClassifier};
use crate::llm::LlmClient;
use crate::llm::types::Message;
use futures::future::BoxFuture;
use std::collections::HashMap;
use std::sync::Arc;

// ── KeywordClassifier ────────────────────────────────────────────────

/** 零成本的、基于关键词的意图分类器。
 *
 * 将触发词/短语映射为 [`Intent::SkillRequired`] 或 [`Intent::DirectAnswer`]。
 * 如果没有关键词匹配，则返回 [`Intent::Fallback`]。
 *
 * **该分类器不包含任何硬编码关键词。**所有触发词均由产品层提供——
 * 例如从技能的 `description` 文本中派生，因为标准的 `SKILL.md` 格式
 * 本身没有 trigger 字段。
 *
 * # 示例
 *
 * ```
 * use echo_agent::intent::{KeywordClassifier, Intent};
 *
 * let mut classifier = KeywordClassifier::default();
 * // 产品层为程序化的描述符注册显式触发词。
 * // 标准 SKILL.md 文件没有 trigger frontmatter 字段。
 * classifier.add_skill_keywords("coding", &["写代码", "编程", "debug", "refactor"]);
 * classifier.add_skill_keywords("paper-search", &["论文", "文献", "arxiv"]);
 * // 产品层注册直接回答（DirectAnswer）关键词
 * classifier.add_direct_keywords(&["你好", "hello", "thanks"]);
 * ```
 */
#[derive(Clone)]
pub struct KeywordClassifier {
    /// 触发词 -> 所有拥有该触发词的技能名称（大小写不敏感）
    skill_keywords: HashMap<String, Vec<String>>,
    /// DirectAnswer 的触发短语（大小写不敏感）
    direct_keywords: Vec<String>,
    /// 是否启用 DirectAnswer 路由
    enable_direct_answer: bool,
}

impl Default for KeywordClassifier {
    /// 创建默认分类器：无关键词，启用 DirectAnswer 路由。
    fn default() -> Self {
        Self {
            skill_keywords: HashMap::new(),
            direct_keywords: Vec::new(),
            enable_direct_answer: true,
        }
    }
}

impl KeywordClassifier {
    /// 创建一个空的分类器。
    pub fn new() -> Self {
        Self::default()
    }

    /** 注册用于触发某个技能激活的关键词。
     *
     * 通常由产品层为程序化描述符调用。标准的 `SKILL.md` 文件没有
     * trigger frontmatter 字段；当需要关键词快速路径时，调用方可以
     * 从描述文本中派生出这些词条。
     */
    pub fn add_skill_keywords(&mut self, skill_name: &str, trigger_words: &[&str]) {
        for word in trigger_words {
            let owners = self.skill_keywords.entry(word.to_lowercase()).or_default();
            if !owners.iter().any(|owner| owner == skill_name) {
                owners.push(skill_name.to_string());
                owners.sort();
            }
        }
    }

    /** 从 skill_name → triggers 的映射批量注册技能关键词。
     *
     * 便于一次性加载所有技能的便捷方法。
     */
    pub fn add_skill_keyword_map(&mut self, map: &HashMap<String, Vec<String>>) {
        for (skill_name, triggers) in map {
            let refs: Vec<&str> = triggers.iter().map(|s| s.as_str()).collect();
            self.add_skill_keywords(skill_name, &refs);
        }
    }

    /// 注册触发直接回答（不使用技能、不进入 ReAct）的关键词。
    pub fn add_direct_keywords(&mut self, trigger_words: &[&str]) {
        for word in trigger_words {
            self.direct_keywords.push(word.to_lowercase());
        }
    }

    /// 启用或禁用 DirectAnswer 快捷路径。
    pub fn set_enable_direct_answer(&mut self, enable: bool) {
        self.enable_direct_answer = enable;
    }

    /// 对输入文本做关键词匹配，返回分类得到的意图。
    fn classify_inner(&self, input: &str) -> Intent {
        let lower = input.trim().to_lowercase();

        /* 1. SkillRequired：基于加权得分的关键词匹配。
         *    触发词越长，权重越高（越具体 = 越可靠）。
         *    同一技能匹配到多个触发词会提升置信度。
         *    若两个技能得分接近，说明输入存在歧义，置信度会随之下降。 */
        let mut skill_scores: HashMap<&str, f32> = HashMap::new();
        for (trigger, skills) in &self.skill_keywords {
            if let Some(weight) = Self::match_weight(&lower, trigger) {
                for skill in skills {
                    *skill_scores.entry(skill.as_str()).or_insert(0.0) += weight;
                }
            }
        }

        if !skill_scores.is_empty() {
            // 按得分降序排序
            let mut scored: Vec<(&str, f32)> = skill_scores.into_iter().collect();
            scored.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(b.0)));

            let Some((best_skill, best_score)) = scored.first().copied() else {
                return Intent::Fallback;
            };
            let runner_up_score = scored.get(1).map(|(_, s)| *s).unwrap_or(0.0);
            if scored
                .get(1)
                .is_some_and(|(_, score)| (*score - best_score).abs() <= f32::EPSILON)
            {
                return Intent::Fallback;
            }

            /* 置信度取决于：
             * - 绝对得分（越高表示匹配到的触发词越多）
             * - 与次高分的差距（差距越大，歧义越小） */
            let gap_ratio = if best_score > 0.0 {
                (best_score - runner_up_score) / best_score
            } else {
                0.0
            };

            // 基于得分的基础置信度（得分为 1 时是 0.5，随后递增）
            let score_confidence = (0.5 + best_score * 0.1).min(0.95);
            // 歧义惩罚（差距小则存在歧义）
            let ambiguity_penalty = if gap_ratio < 0.3 { 0.15 } else { 0.0 };
            let confidence = (score_confidence - ambiguity_penalty).max(0.3);

            /* 仅当置信度 >= 0.7 时才接受关键词匹配结果。
             * 置信度较低时归入 Fallback —— 交给 LLM 分类器或完整 ReAct 流程处理。 */
            if confidence >= 0.7 {
                return Intent::SkillRequired {
                    skill_name: best_skill.to_string(),
                    confidence,
                };
            }
        }

        // 2. DirectAnswer：由产品层显式注册的关键词
        if self.enable_direct_answer {
            for word in &self.direct_keywords {
                if Self::match_weight(&lower, word).is_some() {
                    return Intent::DirectAnswer { confidence: 0.85 };
                }
            }
        }

        // 3. 未匹配 —— 交由下游决定（LLM 回退或完整 ReAct）
        Intent::Fallback
    }

    /** 将某个触发词与输入做匹配，返回一个权重分数。
     *
     * 未匹配时返回 `None`；匹配时返回 `Some(weight)`，权重反映匹配质量：
     * - 触发词越长 → 权重越高（越具体）
     * - CJK 触发词 ≥2 字符 → 权重 = 字符数
     * - 带词边界的 ASCII 触发词 → 权重 = 单词数
     * - 较短的触发词（1 字符 CJK，<3 字符 ASCII）→ 权重较低
     */
    fn match_weight(input: &str, trigger: &str) -> Option<f32> {
        let has_cjk = trigger.chars().any(|c| c > '\u{2E80}');

        if has_cjk {
            // CJK 触发词：要求至少 2 个字符（过滤单字符噪声）
            let char_count = trigger.chars().count();
            if char_count < 2 {
                return None;
            }
            if input.contains(trigger) {
                // 按字符数加权：越长越具体
                Some(char_count as f32)
            } else {
                None
            }
        } else {
            // ASCII/拉丁字母触发词：要求词边界匹配
            if let Some(pos) = input.find(trigger) {
                let before_ok = pos == 0 || !input.as_bytes()[pos - 1].is_ascii_alphanumeric();
                let end = pos + trigger.len();
                let after_ok = end >= input.len() || !input.as_bytes()[end].is_ascii_alphanumeric();
                if before_ok && after_ok {
                    // 按触发词中的单词数加权（多词更具体）
                    let word_count = trigger.split_whitespace().count().max(1);
                    Some(word_count as f32 * 2.0)
                } else {
                    None
                }
            } else {
                None
            }
        }
    }
}

impl IntentClassifier for KeywordClassifier {
    /// 基于关键词匹配对用户输入进行意图分类。
    fn classify<'a>(
        &'a self,
        user_input: &'a str,
        _context: &'a [Message],
    ) -> BoxFuture<'a, Intent> {
        let intent = self.classify_inner(user_input);
        Box::pin(async move { intent })
    }
}

// ── KeywordClassifierConfig ────────────────────────────────────────────

/// 用于从静态数据构建 [`KeywordClassifier`] 的配置。
#[derive(Debug, Clone, Default)]
pub struct KeywordClassifierConfig {
    /// 技能名称 -> 触发词列表的映射。
    pub skill_keyword_map: Vec<(String, Vec<String>)>,
    /// DirectAnswer 的触发关键词列表。
    pub direct_keywords: Vec<String>,
    /// 是否启用 DirectAnswer 快捷路径。
    pub enable_direct_answer: bool,
}

impl KeywordClassifierConfig {
    /// 根据本配置构建一个 [`KeywordClassifier`]。
    pub fn build(self) -> KeywordClassifier {
        let mut classifier = KeywordClassifier::new();
        classifier.set_enable_direct_answer(self.enable_direct_answer);
        for (skill, triggers) in self.skill_keyword_map {
            let refs: Vec<&str> = triggers.iter().map(|s| s.as_str()).collect();
            classifier.add_skill_keywords(&skill, &refs);
        }
        classifier.add_direct_keywords(
            &self
                .direct_keywords
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>(),
        );
        classifier
    }
}

// ── LlmIntentClassifier ──────────────────────────────────────────────

/// 供 LLM 分类器理解可用技能所使用的技能描述。
#[derive(Debug, Clone)]
pub struct SkillDescription {
    /// 技能标识符（例如 "coding"、"paper-search"）
    pub name: String,
    /// 该技能作用的一句话描述
    pub description: String,
    /// 示例触发短语（帮助 LLM 理解意图模式）
    pub example_triggers: Vec<String>,
}

/** 基于轻量分类提示词的、LLM 驱动的意图分类器。
 *
 * 不再依赖详尽的关键词列表，而是将用户输入与可用技能描述一起发送给
 * LLM，由 LLM 返回结构化的分类结果。
 *
 * # 分类提示词
 *
 * 该分类器会构造类似如下的提示词：
 *
 * ```text
 * You are an intent classifier. Given the user input, decide:
 * - DIRECT: simple greeting/question, answer directly
 * - SKILL: needs a specific skill
 * - FALLBACK: complex task, needs full agent loop
 *
 * Available skills:
 * 1. coding — Code generation, debugging, refactoring
 * 2. paper-search — Academic literature search
 * ...
 *
 * User input: {input}
 * Respond in JSON: {"intent": "DIRECT|SKILL|FALLBACK", "skill": "name", "confidence": 0.0-1.0}
 * ```
 *
 * # 示例
 *
 * ```rust,ignore
 * use echo_agent::intent::classifier::{LlmIntentClassifier, SkillDescription};
 *
 * let skills = vec![
 *     SkillDescription {
 *         name: "coding".into(),
 *         description: "Code generation, debugging, refactoring".into(),
 *         example_triggers: vec!["帮我写个排序算法".into()],
 *     },
 * ];
 * let classifier = LlmIntentClassifier::new(llm_client, skills);
 * ```
 */
pub struct LlmIntentClassifier {
    /// 用于执行分类请求的 LLM 客户端。
    llm: Arc<dyn LlmClient>,
    /// 供分类器参考的可用技能描述列表。
    skills: Vec<SkillDescription>,
}

impl LlmIntentClassifier {
    /// 创建一个新的 LLM 意图分类器。
    pub fn new(llm: Arc<dyn LlmClient>, skills: Vec<SkillDescription>) -> Self {
        Self { llm, skills }
    }

    /// 为给定的用户输入构建分类提示词。
    fn build_prompt(&self, user_input: &str) -> String {
        let skill_list = if self.skills.is_empty() {
            "(no skills available)".to_string()
        } else {
            self.skills
                .iter()
                .enumerate()
                .map(|(i, s)| {
                    let examples = if s.example_triggers.is_empty() {
                        String::new()
                    } else {
                        format!(
                            " (examples: {})",
                            s.example_triggers
                                .iter()
                                .take(3)
                                .cloned()
                                .collect::<Vec<_>>()
                                .join(", ")
                        )
                    };
                    format!("{}. {} — {}{}", i + 1, s.name, s.description, examples)
                })
                .collect::<Vec<_>>()
                .join("\n")
        };

        format!(
            r#"You are an intent classifier for an AI agent system.

Given the user's input, classify it into exactly ONE of:
- DIRECT: simple greeting, casual chat, or a straightforward factual question that can be answered directly
- SKILL: a task that matches one of the available specialized skills
- FALLBACK: a complex or ambiguous task that needs the full agent reasoning loop

Available skills:
{skill_list}

Rules:
- If the input clearly matches a skill's domain, choose SKILL and specify which skill
- If the input is a simple greeting, weather query, time query, or short factual question, choose DIRECT
- If unsure or the task is complex/multi-step, choose FALLBACK
- Confidence should be 0.0-1.0; only return DIRECT or SKILL if confidence >= 0.7

User input: {user_input}

Respond ONLY with valid JSON (no markdown):
{{"intent": "DIRECT|SKILL|FALLBACK", "skill": "skill_name_or_null", "confidence": 0.0}}"#
        )
    }

    /// 将 LLM 的 JSON 响应解析为 Intent。
    fn parse_response(response: &str) -> Intent {
        // 从响应中提取 JSON（处理可能存在的 markdown 包裹）
        let json_str = response
            .trim()
            .trim_start_matches("```json")
            .trim_start_matches("```")
            .trim_end_matches("```")
            .trim();

        // 尝试解析为 JSON
        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(json_str) {
            let confidence = parsed
                .get("confidence")
                .and_then(|v| v.as_f64())
                .filter(|value| value.is_finite() && (0.0..=1.0).contains(value))
                .unwrap_or(0.0) as f32;

            match parsed
                .get("intent")
                .and_then(|v| v.as_str())
                .unwrap_or("FALLBACK")
            {
                "DIRECT" => Intent::DirectAnswer { confidence },
                "SKILL" => {
                    if let Some(skill_name) = parsed
                        .get("skill")
                        .and_then(|v| v.as_str())
                        .filter(|s| *s != "null")
                    {
                        Intent::SkillRequired {
                            skill_name: skill_name.to_string(),
                            confidence,
                        }
                    } else {
                        Intent::Fallback
                    }
                }
                _ => Intent::Fallback,
            }
        } else {
            // JSON 解析失败 —— 回退
            Intent::Fallback
        }
    }
}

impl IntentClassifier for LlmIntentClassifier {
    /// 调用 LLM 对用户输入进行语义意图分类，失败时回退为 Fallback。
    fn classify<'a>(
        &'a self,
        user_input: &'a str,
        _context: &'a [Message],
    ) -> BoxFuture<'a, Intent> {
        let prompt = self.build_prompt(user_input);
        let llm = self.llm.clone();

        Box::pin(async move {
            let messages = vec![Message::user(prompt)];
            match llm.chat_simple(messages).await {
                Ok(response) => Self::parse_response(&response),
                Err(e) => {
                    tracing::warn!(error = %e, "LLM intent classification failed, falling back");
                    Intent::Fallback
                }
            }
        })
    }
}

// ── ChainedClassifier ────────────────────────────────────────────────

/** 串联多个分类器：依次尝试，返回第一个非 `Fallback` 的结果。
 *
 * # 典型用法
 *
 * ```rust,ignore
 * use echo_agent::intent::classifier::{ChainedClassifier, KeywordClassifier, LlmIntentClassifier};
 *
 * let chain = ChainedClassifier::new(vec![
 *     Box::new(keyword_classifier),  // fast, zero-cost
 *     Box::new(llm_classifier),      // semantic fallback
 * ]);
 * ```
 */
pub struct ChainedClassifier {
    /// 按顺序尝试的分类器列表。
    classifiers: Vec<Box<dyn IntentClassifier>>,
}

impl ChainedClassifier {
    /// 创建一个新的串联分类器。
    pub fn new(classifiers: Vec<Box<dyn IntentClassifier>>) -> Self {
        Self { classifiers }
    }
}

impl IntentClassifier for ChainedClassifier {
    /// 依次调用各分类器，返回第一个非 Fallback 的结果。
    fn classify<'a>(
        &'a self,
        user_input: &'a str,
        context: &'a [Message],
    ) -> BoxFuture<'a, Intent> {
        // 收集具有所有权的 future，以便遍历各个分类器
        Box::pin(async move {
            for classifier in &self.classifiers {
                let intent = classifier.classify(user_input, context).await;
                if !matches!(intent, Intent::Fallback) {
                    return intent;
                }
            }
            Intent::Fallback
        })
    }
}

// ── 测试 ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// 验证未注册任何关键词时一律返回 Fallback。
    #[test]
    fn test_keyword_classifier_empty_returns_fallback() {
        let classifier = KeywordClassifier::default();
        // 未注册任何关键词 → 全部归为 Fallback
        assert!(matches!(
            classifier.classify_inner("Hello"),
            Intent::Fallback
        ));
        assert!(matches!(
            classifier.classify_inner("帮我写代码"),
            Intent::Fallback
        ));
    }

    /// 验证命中技能关键词时路由到对应技能。
    #[test]
    fn test_keyword_classifier_skill_match() {
        let mut classifier = KeywordClassifier::default();
        classifier.add_skill_keywords("coding", &["写代码", "编程", "debug"]);
        let intent = classifier.classify_inner("帮我写代码实现一个排序算法");
        assert!(matches!(intent, Intent::SkillRequired { .. }));
        assert_eq!(intent.skill_name(), Some("coding"));
    }

    /// 验证命中直接回答关键词时返回 DirectAnswer。
    #[test]
    fn test_keyword_classifier_direct_match() {
        let mut classifier = KeywordClassifier::default();
        classifier.add_direct_keywords(&["你好", "hello", "thanks"]);
        let intent = classifier.classify_inner("Hello, how are you?");
        assert!(matches!(intent, Intent::DirectAnswer { .. }));
    }

    /// 验证关闭 DirectAnswer 路由后不再返回 DirectAnswer。
    #[test]
    fn test_keyword_classifier_direct_disabled() {
        let mut classifier = KeywordClassifier::default();
        classifier.add_direct_keywords(&["hello"]);
        classifier.set_enable_direct_answer(false);
        // DirectAnswer 已禁用 → 落回 Fallback
        assert!(matches!(
            classifier.classify_inner("hello"),
            Intent::Fallback
        ));
    }

    /// 验证关键词匹配不区分大小写。
    #[test]
    fn test_keyword_classifier_case_insensitive() {
        let mut classifier = KeywordClassifier::default();
        classifier.add_skill_keywords("coding", &["Debug"]);
        // 输入是小写，而触发词注册时是 "Debug" → 会被归一化为 "debug"
        let intent = classifier.classify_inner("please DEBUG this");
        assert!(matches!(intent, Intent::SkillRequired { .. }));
    }

    /// 验证解析 DIRECT 响应。
    #[test]
    fn test_llm_classifier_parse_direct() {
        let response = r#"{"intent": "DIRECT", "skill": null, "confidence": 0.95}"#;
        let intent = LlmIntentClassifier::parse_response(response);
        assert!(matches!(
            intent,
            Intent::DirectAnswer {
                confidence: c
            } if (c - 0.95).abs() < 0.01
        ));
    }

    /// 验证解析 SKILL 响应。
    #[test]
    fn test_llm_classifier_parse_skill() {
        let response = r#"{"intent": "SKILL", "skill": "coding", "confidence": 0.88}"#;
        let intent = LlmIntentClassifier::parse_response(response);
        assert_eq!(intent.skill_name(), Some("coding"));
    }

    /// 验证解析 FALLBACK 响应。
    #[test]
    fn test_llm_classifier_parse_fallback() {
        let response = r#"{"intent": "FALLBACK", "skill": null, "confidence": 0.6}"#;
        let intent = LlmIntentClassifier::parse_response(response);
        assert!(matches!(intent, Intent::Fallback));
    }

    /// 验证非法 JSON 响应回退为 Fallback。
    #[test]
    fn test_llm_classifier_parse_invalid_json() {
        let response = "I think this is a coding task";
        let intent = LlmIntentClassifier::parse_response(response);
        assert!(matches!(intent, Intent::Fallback));
    }

    /// 验证能解析被 Markdown 代码块包裹的 JSON 响应。
    #[test]
    fn test_llm_classifier_parse_markdown_wrapped() {
        let response = r#"```json
{"intent": "SKILL", "skill": "paper-search", "confidence": 0.85}
```"#;
        let intent = LlmIntentClassifier::parse_response(response);
        assert_eq!(intent.skill_name(), Some("paper-search"));
    }

    /// 验证构建的提示词包含所有技能描述。
    #[test]
    fn test_llm_classifier_build_prompt_contains_skills() {
        let classifier = LlmIntentClassifier::new(
            // 使用一个占位实现——这里只测试提示词构建，不涉及真实的 LLM 调用
            Arc::new(crate::testing::MockLlmClient::new()),
            vec![
                SkillDescription {
                    name: "coding".into(),
                    description: "Code generation and debugging".into(),
                    example_triggers: vec!["write a function".into()],
                },
                SkillDescription {
                    name: "paper-search".into(),
                    description: "Academic literature search".into(),
                    example_triggers: vec!["find papers about".into()],
                },
            ],
        );
        let prompt = classifier.build_prompt("帮我写个排序算法");
        assert!(prompt.contains("coding"));
        assert!(prompt.contains("paper-search"));
        assert!(prompt.contains("帮我写个排序算法"));
    }

    /// 验证配置构建器的各项设置生效。
    #[test]
    fn test_config_builder() {
        let config = KeywordClassifierConfig {
            skill_keyword_map: vec![("coding".into(), vec!["debug".into()])],
            direct_keywords: vec!["thanks".into()],
            enable_direct_answer: true,
        };
        let classifier = config.build();

        let intent = classifier.classify_inner("debug this code");
        assert!(matches!(intent, Intent::SkillRequired { .. }));

        let intent = classifier.classify_inner("thanks for your help");
        assert!(matches!(intent, Intent::DirectAnswer { .. }));
    }

    /** RFC 3.1.4：针对 KeywordClassifier 的程序化技能触发词集成测试。
     *
     * 为每个技能至少测试 2 条具有代表性的触发查询。
     * 同时验证负例场景（DirectAnswer、Fallback）。
     */
    #[test]
    fn test_11_skill_trigger_routing() {
        let mut classifier = KeywordClassifier::default();

        /* 程序化描述符仍可以显式注册触发词；
         * 而标准 SKILL.md 文件则是依据其描述文本进行路由的。 */
        classifier.add_skill_keywords(
            "coding",
            &[
                "写代码",
                "编程",
                "调试",
                "重构",
                "bug",
                "报错",
                "代码审查",
                "code review",
                "debug",
                "refactor",
                "stack trace",
                "编译错误",
                "单元测试",
                "写个",
                "写一段",
                "写脚本",
                "写函数",
                "写类",
                "实现",
            ],
        );
        classifier.add_skill_keywords(
            "data-visualization",
            &[
                "图表",
                "可视化",
                "柱状图",
                "折线图",
                "饼图",
                "散点图",
                "chart",
                "plot",
                "visualization",
                "画图",
                "画个图",
            ],
        );
        classifier.add_skill_keywords(
            "data-wrangling",
            &[
                "加载数据",
                "读取csv",
                "读取excel",
                "数据清洗",
                "缺失值处理",
                "eda",
                "探索性分析",
                "数据预处理",
                "data wrangling",
                "data cleaning",
                "dataframe",
                "数据质量",
            ],
        );
        classifier.add_skill_keywords(
            "doc-writing",
            &[
                "写报告",
                "写文档",
                "写文章",
                "撰写",
                "写邮件",
                "写方案",
                "write report",
                "write document",
                "draft",
                "起草",
                "写总结",
            ],
        );
        classifier.add_skill_keywords(
            "evidence-medicine",
            &[
                "医学文献",
                "临床研究",
                "护理研究",
                "临床试验",
                "系统综述",
                "meta分析",
                "pubmed",
                "medical research",
                "clinical trial",
                "systematic review",
                "循证医学",
                "骨科护理",
                "药物治疗证据",
            ],
        );
        classifier.add_skill_keywords(
            "git-workflow",
            &[
                "git",
                "分支",
                "提交",
                "commit",
                "pr",
                "mr",
                "merge",
                "合并",
                "冲突",
                "branch",
                "cherry-pick",
                "rebase",
                "版本控制",
            ],
        );
        classifier.add_skill_keywords(
            "paper-reader",
            &[
                "读论文",
                "分析论文",
                "论文解读",
                "读这篇",
                "paper reading",
                "paper analysis",
                "方法论",
                "实验结果",
            ],
        );
        classifier.add_skill_keywords(
            "paper-search",
            &[
                "论文检索",
                "学术论文",
                "文献检索",
                "arxiv",
                "semantic scholar",
                "引用数",
                "paper search",
                "academic paper",
                "research paper",
                "找论文",
                "文献综述",
            ],
        );
        classifier.add_skill_keywords(
            "statistical-analysis",
            &[
                "假设检验",
                "t检验",
                "卡方检验",
                "回归分析",
                "anova",
                "p值",
                "显著性检验",
                "statistics",
                "hypothesis test",
                "regression",
                "统计建模",
                "置信区间",
                "效应量",
            ],
        );
        classifier.add_skill_keywords(
            "translation",
            &[
                "翻译",
                "translate",
                "翻译一下",
                "翻译成中文",
                "翻译成英文",
                "中英对照",
                "本地化",
                "localization",
                "i18n",
                "多语言",
            ],
        );
        classifier.add_skill_keywords(
            "web-search",
            &[
                "上网查",
                "网上搜索",
                "最新新闻",
                "搜索网页",
                "search web",
                "look up",
                "fact check",
                "事实核查",
                "网络调研",
            ],
        );

        // DirectAnswer 关键词
        classifier.add_direct_keywords(&["你好", "hello", "hi", "thanks", "谢谢"]);

        // ── 正例：每个技能至少 2 条测试查询 ──
        let test_cases: Vec<(&str, &str)> = vec![
            // coding 技能
            ("帮我写代码实现排序", "coding"),
            ("这段代码有个 bug", "coding"),
            ("帮我写个 Python 脚本", "coding"),
            ("实现一个单例模式", "coding"),
            // data-visualization 技能
            ("画一个柱状图", "data-visualization"),
            ("生成可视化图表", "data-visualization"),
            // data-wrangling 技能
            ("读取CSV文件并清洗", "data-wrangling"),
            ("缺失值处理怎么做", "data-wrangling"),
            // doc-writing 技能
            ("帮我写报告", "doc-writing"),
            ("撰写技术文档", "doc-writing"),
            // evidence-medicine 技能
            ("搜索 PubMed 文献", "evidence-medicine"),
            ("系统综述怎么做", "evidence-medicine"),
            // git-workflow 技能
            ("git 怎么提交", "git-workflow"),
            ("合并分支冲突了", "git-workflow"),
            // paper-reader 技能
            ("帮我读论文", "paper-reader"),
            ("分析这篇论文的方法论", "paper-reader"),
            // paper-search 技能
            ("找论文 arxiv", "paper-search"),
            ("学术论文检索", "paper-search"),
            // statistical-analysis 技能
            ("做个 t检验", "statistical-analysis"),
            ("回归分析怎么做", "statistical-analysis"),
            // translation 技能
            ("翻译这段话", "translation"),
            ("翻译成英文", "translation"),
            // web-search 技能
            ("上网查一下最新消息", "web-search"),
            ("搜索网页找资料", "web-search"),
        ];

        let mut passed = 0;
        let mut failed = Vec::new();

        for (query, expected_skill) in &test_cases {
            let intent = classifier.classify_inner(query);
            match intent {
                Intent::SkillRequired { ref skill_name, .. } if skill_name == expected_skill => {
                    passed += 1;
                }
                other => {
                    failed.push(format!(
                        "  '{}' → expected '{}', got {:?}",
                        query, expected_skill, other
                    ));
                }
            }
        }

        assert!(
            failed.is_empty(),
            "{}/{} trigger routing tests failed:\n{}",
            failed.len(),
            test_cases.len(),
            failed.join("\n")
        );
        assert_eq!(
            passed,
            test_cases.len(),
            "All {} trigger tests should pass",
            test_cases.len()
        );

        // ── 负例 ──

        // DirectAnswer：问候语
        let intent = classifier.classify_inner("你好");
        assert!(
            matches!(intent, Intent::DirectAnswer { .. }),
            "Greeting should be DirectAnswer, got {:?}",
            intent
        );

        // Fallback：没有匹配到触发词的普通问题
        let intent = classifier.classify_inner("今天天气怎么样");
        assert!(
            matches!(intent, Intent::Fallback),
            "Weather query should be Fallback, got {:?}",
            intent
        );

        // Fallback：无关问题
        let intent = classifier.classify_inner("推荐一本好看的小说");
        assert!(
            matches!(intent, Intent::Fallback),
            "Novel recommendation should be Fallback, got {:?}",
            intent
        );
    }
}
