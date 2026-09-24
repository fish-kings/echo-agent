/*!
写入触发器检测——识别 Agent 学习到的内存何时应当被持久化。

`TriggerDetector` 会扫描对话上下文，寻找表明“出现了值得记录的时刻”的信号。
所有检测方法都是**同步**的（仅做模式匹配）；调用方负责异步持久化。

## 支持的触发器

| 触发器 | 模式 | 默认置信度 | MemoryType |
|---------|---------|-------------------|------------|
| `UserCorrection` | 助手输出之后出现 "不是这样"/"不对"/"不要"/"wrong"/"actually"/"no," | 0.90 | `UserPreference` 或 `DebuggingLesson` |
| `ErrorResolution` | 工具失败 → 换一种方式后成功 | 0.85 | `ErrorResolution` |
| `RepeatedWorkflow` | 相同工具序列出现 ≥3 次 | 0.75 | `WorkflowPattern` |
| `ExplicitSave` | `/remember` 命令或 `remember` 工具 | 1.00 | `UserPreference` |
*/

use echo_core::memory::types::{MemorySource, MemoryType};
use futures::future::BoxFuture;
use std::collections::HashMap;

use super::security::InputTrustLevel;

// ── 触发器匹配 ──────────────────────────────────────────────────────

/// 一个已检测到的写入触发器，附带相关的内存数据。
#[derive(Debug, Clone)]
pub struct TriggerMatch {
    /// 要存储为内存的内容。
    pub content: String,
    /// 从触发器推断出的内存类型。
    pub memory_type: MemoryType,
    /// 触发本次写入的来源。
    pub source: MemorySource,
    /// 来自该来源默认值的置信度。
    pub confidence: f32,
    /// 从上下文推断出的主题分类。
    pub topic: String,
    /// 内容来源的信任级别。
    pub trust_level: InputTrustLevel,
    /// 建议使用的存储 key。
    pub suggested_key: String,
    /// 支持该触发器的精确原文摘录。
    pub evidence: Vec<TriggerEvidence>,
}

/// 一条支持某个已检测触发器的精确摘录。
#[derive(Debug, Clone)]
pub struct TriggerEvidence {
    /// 摘录的来源，例如 `user`、`assistant` 或 `tool_error`。
    pub source_role: String,
    /// 检测器使用的原文摘录。
    pub quote: String,
}

/// 应用侧提供的触发器接收方所返回的处理结果。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryTriggerDisposition {
    /// 继续执行框架默认的持久化内存写入。
    Persist,
    /// 接收方已捕获该触发器，并自行接管其后续生命周期。
    Captured,
}

/// 可选的应用层钩子，用于在持久化之前对已检测的触发器进行路由。
pub trait MemoryTriggerSink: Send + Sync {
    /// 路由一个已检测到的触发器。
    fn on_trigger<'a>(
        &'a self,
        trigger: &'a TriggerMatch,
    ) -> BoxFuture<'a, std::result::Result<MemoryTriggerDisposition, String>>;
}

// ── 上下文记录 ────────────────────────────────────────────────────

/// 一次工具失败的记录。
#[derive(Debug, Clone)]
pub struct ToolFailureRecord {
    /// 失败的工具名称。
    pub tool_name: String,
    /// 工具输入的摘要。
    pub input_summary: String,
    /// 工具返回的错误信息。
    pub error: String,
    /// ISO 8601 时间戳。
    pub timestamp: String,
}

/// 一次成功的工具使用记录（发生在此前的一次失败之后）。
#[derive(Debug, Clone)]
pub struct ToolSuccessRecord {
    /// 成功的工具名称。
    pub tool_name: String,
    /// 工具输入的摘要（可能与失败记录的输入不同）。
    pub input_summary: String,
    /// 工具输出的摘要。
    pub output_summary: String,
    /// ISO 8601 时间戳。
    pub timestamp: String,
}

/// 一次显式保存命令的记录。
#[derive(Debug, Clone)]
pub struct ExplicitSaveRecord {
    /// 用户希望记住的内容。
    pub content: String,
    /// 用户提供的可选主题提示。
    pub topic: Option<String>,
}

/// 用于序列追踪的一次工具调用记录。
#[derive(Debug, Clone)]
pub struct ToolSequenceRecord {
    /// 被调用的工具名称。
    pub tool_name: String,
    /// 用于跨轮次分组的会话 ID。
    pub session_id: String,
    /// ISO 8601 时间戳。
    pub timestamp: String,
}

/// 检测上下文——检测触发器所需的数据。
#[derive(Debug, Clone, Default)]
pub struct TriggerContext {
    /// 当前轮次中用户的消息。
    pub user_message: Option<String>,
    /// 助手的最后一次回复（用于纠正检测）。
    pub assistant_message: Option<String>,
    /// 最近一次工具失败（用于错误解决检测）。
    pub last_tool_failure: Option<ToolFailureRecord>,
    /// 失败之后最近一次工具成功（用于错误解决检测）。
    pub last_tool_success: Option<ToolSuccessRecord>,
    /// 用户是否发出了显式的保存命令。
    pub explicit_save: Option<ExplicitSaveRecord>,
    /// 用于 RepeatedWorkflow 检测所累积的工具序列。
    pub tool_sequences: Vec<ToolSequenceRecord>,
}

// ── 纠正信号模式 ─────────────────────────────────────────────────

/// 中文纠正信号——表明用户正在纠正 Agent。
const CN_CORRECTION_SIGNALS: &[&str] = &[
    "不是这样",
    "不对",
    "不要",
    "不是",
    "不对的",
    "错了",
    "重新来",
    "别这样",
];

/// 英文纠正信号——表明用户正在纠正 Agent。
const EN_CORRECTION_SIGNALS: &[&str] = &[
    "wrong",
    "actually",
    "no,",
    "no ",
    "incorrect",
    "not like that",
    "that's wrong",
    "that's not right",
    "don't do that",
    "stop doing that",
    "not what i meant",
    "i meant",
    "let me clarify",
];

// ── TriggerDetector ────────────────────────────────────────────────────

/**
从对话上下文中检测写入触发器。

所有检测方法都是同步的（仅做模式匹配）。
调用方负责异步持久化。
*/
pub struct TriggerDetector {
    /// 每轮次最多检测的触发器数量。
    max_per_turn: usize,
}

impl TriggerDetector {
    /// 使用默认设置创建一个新的触发器检测器。
    pub fn new() -> Self {
        Self { max_per_turn: 5 }
    }

    /// 使用自定义的每轮次上限创建一个触发器检测器。
    pub fn with_max_per_turn(max_per_turn: usize) -> Self {
        Self { max_per_turn }
    }

    /**
    从给定上下文中检测全部触发器。

    返回一组 `TriggerMatch` 实例，每一项代表一条应当被持久化的
    内存。结果数量以 `max_per_turn` 为上限。
    */
    pub fn detect(&self, ctx: &TriggerContext) -> Vec<TriggerMatch> {
        let mut matches = Vec::new();

        // 优先级顺序：ExplicitSave > UserCorrection > ErrorResolution > RepeatedWorkflow
        if let Some(m) = self.detect_explicit_save(ctx) {
            matches.push(m);
        }
        if let Some(m) = self.detect_user_correction(ctx) {
            matches.push(m);
        }
        if let Some(m) = self.detect_error_resolution(ctx) {
            matches.push(m);
        }
        if let Some(m) = self.detect_repeated_workflow(ctx) {
            matches.push(m);
        }

        matches.truncate(self.max_per_turn);
        matches
    }

    /**
    检测显式保存命令。

    由 `/remember` 命令或 `remember` 工具调用触发。
    */
    fn detect_explicit_save(&self, ctx: &TriggerContext) -> Option<TriggerMatch> {
        let save = ctx.explicit_save.as_ref()?;
        let content = save.content.clone();
        let topic = save.topic.clone().unwrap_or_else(|| infer_topic(&content));
        let key = generate_key(MemorySource::ExplicitSave, &content);

        Some(TriggerMatch {
            content,
            memory_type: MemoryType::UserPreference,
            source: MemorySource::ExplicitSave,
            confidence: MemorySource::ExplicitSave.default_confidence(),
            topic,
            trust_level: InputTrustLevel::Trusted,
            suggested_key: key,
            evidence: vec![TriggerEvidence {
                source_role: "user".to_string(),
                quote: save.content.clone(),
            }],
        })
    }

    /**
    检测用户纠正信号。

    模式：出现在助手输出之后的中英文纠正短语。
    */
    fn detect_user_correction(&self, ctx: &TriggerContext) -> Option<TriggerMatch> {
        let user_msg = ctx.user_message.as_deref()?;
        let assistant_msg = ctx.assistant_message.as_deref()?;

        // 必须同时存在两条消息，且用户消息中必须包含纠正信号
        let user_lower = user_msg.to_lowercase();
        let has_signal = CN_CORRECTION_SIGNALS.iter().any(|s| user_msg.contains(s))
            || EN_CORRECTION_SIGNALS.iter().any(|s| user_lower.contains(s));

        if !has_signal {
            return None;
        }

        let content = format!(
            "User correction: {} (was: {})",
            truncate_content(user_msg, 200),
            truncate_content(assistant_msg, 100)
        );
        let memory_type = classify_correction(user_msg, assistant_msg);
        let topic = infer_topic(user_msg);
        let key = generate_key(MemorySource::UserCorrection, &content);

        Some(TriggerMatch {
            content,
            memory_type,
            source: MemorySource::UserCorrection,
            confidence: MemorySource::UserCorrection.default_confidence(),
            topic,
            trust_level: InputTrustLevel::Trusted,
            suggested_key: key,
            evidence: vec![
                TriggerEvidence {
                    source_role: "user".to_string(),
                    quote: truncate_content(user_msg, 200),
                },
                TriggerEvidence {
                    source_role: "assistant".to_string(),
                    quote: truncate_content(assistant_msg, 100),
                },
            ],
        })
    }

    /**
    检测错误解决模式。

    模式：工具失败之后，换一种方式取得成功。
    */
    fn detect_error_resolution(&self, ctx: &TriggerContext) -> Option<TriggerMatch> {
        let failure = ctx.last_tool_failure.as_ref()?;
        let success = ctx.last_tool_success.as_ref()?;

        // 该成功必须发生在失败之后，并且使用了不同的方式
        // （不同的工具或不同的输入）
        let different_approach = success.tool_name != failure.tool_name
            || success.input_summary != failure.input_summary;

        if !different_approach {
            return None;
        }

        let content = format!(
            "Error resolved: {} failed ({}) → {} succeeded ({})",
            failure.tool_name,
            truncate_content(&failure.error, 80),
            success.tool_name,
            truncate_content(&success.output_summary, 80)
        );
        let topic = infer_topic(&failure.error);
        let key = generate_key(MemorySource::ErrorResolution, &content);

        Some(TriggerMatch {
            content,
            memory_type: MemoryType::ErrorResolution,
            source: MemorySource::ErrorResolution,
            confidence: MemorySource::ErrorResolution.default_confidence(),
            topic,
            trust_level: InputTrustLevel::Assistant,
            suggested_key: key,
            evidence: vec![
                TriggerEvidence {
                    source_role: "tool_error".to_string(),
                    quote: format!(
                        "{}: {}",
                        failure.tool_name,
                        truncate_content(&failure.error, 160)
                    ),
                },
                TriggerEvidence {
                    source_role: "tool_output".to_string(),
                    quote: format!(
                        "{}: {}",
                        success.tool_name,
                        truncate_content(&success.output_summary, 160)
                    ),
                },
            ],
        })
    }

    /**
    检测重复的工作流模式。

    模式：同一工具序列在跨会话场景中被观察到 ≥3 次。
    */
    fn detect_repeated_workflow(&self, ctx: &TriggerContext) -> Option<TriggerMatch> {
        if ctx.tool_sequences.len() < 3 {
            return None;
        }

        // 按 (tool_name, session_id) 对序列分组，寻找重复模式
        let mut tool_counts: HashMap<String, usize> = HashMap::new();
        for seq in &ctx.tool_sequences {
            *tool_counts.entry(seq.tool_name.clone()).or_insert(0) += 1;
        }

        // 找出出现次数最多的重复工具序列（≥3 次）
        let most_repeated = tool_counts
            .iter()
            .filter(|&(_, &count)| count >= 3)
            .max_by_key(|&(_, &count)| count)?;

        let tool_name = most_repeated.0.clone();
        let count = *most_repeated.1;

        let content = format!(
            "Repeated workflow pattern: tool '{}' used {} times across sessions",
            tool_name, count
        );
        let topic = infer_topic(&tool_name);
        let key = generate_key(MemorySource::RepeatedWorkflow, &content);

        Some(TriggerMatch {
            content,
            memory_type: MemoryType::WorkflowPattern,
            source: MemorySource::RepeatedWorkflow,
            confidence: MemorySource::RepeatedWorkflow.default_confidence(),
            topic,
            trust_level: InputTrustLevel::Assistant,
            suggested_key: key,
            evidence: ctx
                .tool_sequences
                .iter()
                .filter(|record| record.tool_name == tool_name)
                .take(3)
                .map(|record| TriggerEvidence {
                    source_role: "tool_sequence".to_string(),
                    quote: format!(
                        "{} in session {} at {}",
                        record.tool_name, record.session_id, record.timestamp
                    ),
                })
                .collect(),
        })
    }
}

impl Default for TriggerDetector {
    fn default() -> Self {
        Self::new()
    }
}

// ── 辅助函数 ───────────────────────────────────────────────────

/// 根据内容信号，将一次用户纠正分类为某个 MemoryType。
fn classify_correction(user_msg: &str, _assistant_msg: &str) -> MemoryType {
    let lower = user_msg.to_lowercase();

    // 调试/错误纠正
    if lower.contains("bug")
        || lower.contains("error")
        || lower.contains("fix")
        || lower.contains("debug")
        || lower.contains("fail")
        || lower.contains("错误")
        || lower.contains("修复")
        || lower.contains("调试")
    {
        return MemoryType::DebuggingLesson;
    }

    // 风格/偏好纠正
    if lower.contains("prefer")
        || lower.contains("style")
        || lower.contains("format")
        || lower.contains("verbose")
        || lower.contains("concise")
        || lower.contains("always")
        || lower.contains("never")
        || lower.contains("风格")
        || lower.contains("格式")
        || lower.contains("简洁")
    {
        return MemoryType::UserPreference;
    }

    // 工具/命令纠正
    if lower.contains("tool")
        || lower.contains("command")
        || lower.contains("utility")
        || lower.contains("工具")
        || lower.contains("命令")
    {
        return MemoryType::ToolUsage;
    }

    // 默认：用户偏好（纠正本身就是一种偏好信号）
    MemoryType::UserPreference
}

/// 使用关键词启发式方法，从内容中推断主题。
fn infer_topic(content: &str) -> String {
    let lower = content.to_lowercase();

    if lower.contains("build")
        || lower.contains("compile")
        || lower.contains("maven")
        || lower.contains("cargo")
    {
        return "build".to_string();
    }
    if lower.contains("test") || lower.contains("spec") || lower.contains("验证") {
        return "testing".to_string();
    }
    if lower.contains("deploy") || lower.contains("release") || lower.contains("发布") {
        return "deploy".to_string();
    }
    if lower.contains("debug")
        || lower.contains("error")
        || lower.contains("fix")
        || lower.contains("bug")
        || lower.contains("错误")
    {
        return "debugging".to_string();
    }
    if lower.contains("style")
        || lower.contains("format")
        || lower.contains("prefer")
        || lower.contains("风格")
    {
        return "style".to_string();
    }
    if lower.contains("arch")
        || lower.contains("design")
        || lower.contains("架构")
        || lower.contains("设计")
    {
        return "architecture".to_string();
    }
    if lower.contains("tool") || lower.contains("command") || lower.contains("工具") {
        return "tools".to_string();
    }
    if lower.contains("git") || lower.contains("branch") || lower.contains("commit") {
        return "git".to_string();
    }

    "general".to_string()
}

/// 根据来源与内容生成一个确定性的 key。
fn generate_key(source: MemorySource, content: &str) -> String {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    content.hash(&mut hasher);
    let hash = hasher.finish();
    let prefix = match source {
        MemorySource::UserCorrection => "uc",
        MemorySource::ErrorResolution => "er",
        MemorySource::RepeatedWorkflow => "rw",
        MemorySource::ExplicitSave => "es",
        MemorySource::AutoExtracted => "ae",
        MemorySource::L3Promotion => "l3",
    };
    format!("{prefix}_{hash:016x}")
}

/// 将内容截断到最大长度；若发生截断，则追加 "..."。
fn truncate_content(s: &str, max_len: usize) -> String {
    let mut chars = s.chars();
    let prefix: String = chars.by_ref().take(max_len).collect();
    if chars.next().is_some() {
        format!("{prefix}...")
    } else {
        prefix
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_user_correction_chinese() {
        let detector = TriggerDetector::new();
        let ctx = TriggerContext {
            user_message: Some("不是这样做的，应该用 cargo test".to_string()),
            assistant_message: Some("Run the tests with make test".to_string()),
            ..Default::default()
        };
        let matches = detector.detect(&ctx);
        assert_eq!(matches.len(), 1);
        let Some(detected) = matches.first() else {
            return;
        };
        assert_eq!(detected.source, MemorySource::UserCorrection);
        assert_eq!(detected.confidence, 0.90);
        assert_eq!(detected.evidence.len(), 2);
        assert!(
            detected
                .evidence
                .iter()
                .any(|item| item.source_role == "user")
        );
    }

    #[test]
    fn truncate_content_counts_unicode_characters() {
        assert_eq!(truncate_content("中文🙂abc", 3), "中文🙂...");
        assert_eq!(truncate_content("中文", 3), "中文");
    }

    #[test]
    fn test_detect_user_correction_english() {
        let detector = TriggerDetector::new();
        let ctx = TriggerContext {
            user_message: Some("Actually, that's wrong — use cargo test instead".to_string()),
            assistant_message: Some("Run the tests with make test".to_string()),
            ..Default::default()
        };
        let matches = detector.detect(&ctx);
        assert!(
            matches
                .iter()
                .any(|m| m.source == MemorySource::UserCorrection)
        );
    }

    #[test]
    fn test_no_false_positive_correction() {
        let detector = TriggerDetector::new();
        let ctx = TriggerContext {
            user_message: Some("Please add a test for the new feature".to_string()),
            assistant_message: Some("I'll add a test now".to_string()),
            ..Default::default()
        };
        let matches = detector.detect(&ctx);
        assert!(
            !matches
                .iter()
                .any(|m| m.source == MemorySource::UserCorrection)
        );
    }

    #[test]
    fn test_no_correction_without_assistant_message() {
        let detector = TriggerDetector::new();
        let ctx = TriggerContext {
            user_message: Some("不对，应该这样做".to_string()),
            assistant_message: None,
            ..Default::default()
        };
        let matches = detector.detect(&ctx);
        assert!(
            !matches
                .iter()
                .any(|m| m.source == MemorySource::UserCorrection)
        );
    }

    #[test]
    fn test_detect_error_resolution() {
        let detector = TriggerDetector::new();
        let ctx = TriggerContext {
            last_tool_failure: Some(ToolFailureRecord {
                tool_name: "Bash".to_string(),
                input_summary: "make test".to_string(),
                error: "make: *** No rule to make target 'test'".to_string(),
                timestamp: "2026-06-15T10:00:00Z".to_string(),
            }),
            last_tool_success: Some(ToolSuccessRecord {
                tool_name: "Bash".to_string(),
                input_summary: "cargo test".to_string(),
                output_summary: "running 42 tests ... ok".to_string(),
                timestamp: "2026-06-15T10:01:00Z".to_string(),
            }),
            ..Default::default()
        };
        let matches = detector.detect(&ctx);
        assert!(
            matches
                .iter()
                .any(|m| m.source == MemorySource::ErrorResolution)
        );
        let er = matches
            .iter()
            .find(|m| m.source == MemorySource::ErrorResolution)
            .unwrap();
        assert_eq!(er.memory_type, MemoryType::ErrorResolution);
        assert_eq!(er.confidence, 0.85);
    }

    #[test]
    fn test_no_error_resolution_same_approach() {
        let detector = TriggerDetector::new();
        let ctx = TriggerContext {
            last_tool_failure: Some(ToolFailureRecord {
                tool_name: "Bash".to_string(),
                input_summary: "cargo test".to_string(),
                error: "test failed".to_string(),
                timestamp: "2026-06-15T10:00:00Z".to_string(),
            }),
            last_tool_success: Some(ToolSuccessRecord {
                tool_name: "Bash".to_string(),
                input_summary: "cargo test".to_string(), // 相同方式
                output_summary: "all tests passed".to_string(),
                timestamp: "2026-06-15T10:01:00Z".to_string(),
            }),
            ..Default::default()
        };
        let matches = detector.detect(&ctx);
        assert!(
            !matches
                .iter()
                .any(|m| m.source == MemorySource::ErrorResolution)
        );
    }

    #[test]
    fn test_detect_explicit_save() {
        let detector = TriggerDetector::new();
        let ctx = TriggerContext {
            explicit_save: Some(ExplicitSaveRecord {
                content: "This project uses Java 8 with Maven".to_string(),
                topic: Some("build".to_string()),
            }),
            ..Default::default()
        };
        let matches = detector.detect(&ctx);
        assert!(
            matches
                .iter()
                .any(|m| m.source == MemorySource::ExplicitSave)
        );
        let es = matches
            .iter()
            .find(|m| m.source == MemorySource::ExplicitSave)
            .unwrap();
        assert_eq!(es.memory_type, MemoryType::UserPreference);
        assert_eq!(es.confidence, 1.0);
        assert_eq!(es.topic, "build");
    }

    #[test]
    fn test_detect_repeated_workflow() {
        let detector = TriggerDetector::new();
        let sequences: Vec<ToolSequenceRecord> = (0..4)
            .map(|i| ToolSequenceRecord {
                tool_name: "search_paper".to_string(),
                session_id: format!("session-{i}"),
                timestamp: format!("2026-06-15T10:{i:02}:00Z"),
            })
            .collect();
        let ctx = TriggerContext {
            tool_sequences: sequences,
            ..Default::default()
        };
        let matches = detector.detect(&ctx);
        assert!(
            matches
                .iter()
                .any(|m| m.source == MemorySource::RepeatedWorkflow)
        );
        let rw = matches
            .iter()
            .find(|m| m.source == MemorySource::RepeatedWorkflow)
            .unwrap();
        assert_eq!(rw.memory_type, MemoryType::WorkflowPattern);
        assert_eq!(rw.confidence, 0.75);
    }

    #[test]
    fn test_no_repeated_workflow_under_threshold() {
        let detector = TriggerDetector::new();
        let ctx = TriggerContext {
            tool_sequences: vec![
                ToolSequenceRecord {
                    tool_name: "search_paper".to_string(),
                    session_id: "s1".to_string(),
                    timestamp: "2026-06-15T10:00:00Z".to_string(),
                },
                ToolSequenceRecord {
                    tool_name: "search_paper".to_string(),
                    session_id: "s2".to_string(),
                    timestamp: "2026-06-15T10:01:00Z".to_string(),
                },
            ],
            ..Default::default()
        };
        let matches = detector.detect(&ctx);
        assert!(
            !matches
                .iter()
                .any(|m| m.source == MemorySource::RepeatedWorkflow)
        );
    }

    #[test]
    fn test_max_per_turn_limit() {
        let detector = TriggerDetector::with_max_per_turn(1);
        let ctx = TriggerContext {
            user_message: Some("不对，实际上应该用 cargo build".to_string()),
            assistant_message: Some("Run make build".to_string()),
            explicit_save: Some(ExplicitSaveRecord {
                content: "Use cargo build".to_string(),
                topic: None,
            }),
            ..Default::default()
        };
        let matches = detector.detect(&ctx);
        assert!(matches.len() <= 1);
    }

    #[test]
    fn test_classify_correction_debugging() {
        assert_eq!(
            classify_correction("不对，这个 bug 应该用别的方式修复", ""),
            MemoryType::DebuggingLesson
        );
        assert_eq!(
            classify_correction("Wrong, that error needs a different fix", ""),
            MemoryType::DebuggingLesson
        );
    }

    #[test]
    fn test_classify_correction_style() {
        assert_eq!(
            classify_correction("不要这样，我更喜欢简洁的代码风格", ""),
            MemoryType::UserPreference
        );
        assert_eq!(
            classify_correction("Actually, I prefer concise format", ""),
            MemoryType::UserPreference
        );
    }

    #[test]
    fn test_generate_key_deterministic() {
        let key1 = generate_key(MemorySource::UserCorrection, "same content");
        let key2 = generate_key(MemorySource::UserCorrection, "same content");
        assert_eq!(key1, key2);

        let key3 = generate_key(MemorySource::ErrorResolution, "same content");
        assert_ne!(key1, key3); // 不同前缀
    }

    #[test]
    fn test_infer_topic() {
        assert_eq!(infer_topic("cargo build failed"), "build");
        assert_eq!(infer_topic("test error in module"), "testing");
        assert_eq!(infer_topic("deploy to production"), "deploy");
        assert_eq!(infer_topic("debug the error"), "debugging");
        assert_eq!(infer_topic("I prefer this style"), "style");
        assert_eq!(infer_topic("architecture decision"), "architecture");
        assert_eq!(infer_topic("use the grep tool"), "tools");
        assert_eq!(infer_topic("git commit the changes"), "git");
        assert_eq!(infer_topic("something random"), "general");
    }

    #[test]
    fn test_truncate_content_short() {
        assert_eq!(truncate_content("hello", 10), "hello");
    }

    #[test]
    fn test_truncate_content_long() {
        let long = "a".repeat(300);
        let truncated = truncate_content(&long, 200);
        assert!(truncated.len() <= 203); // 200 + "..." 的长度
        assert!(truncated.ends_with("..."));
    }
}
