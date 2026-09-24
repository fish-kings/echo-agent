/*!
基于存储的记忆推广器 —— 将被淘汰的消息推广为 L3 记忆。

实现 [`MemoryPromoter`]，从压缩过程中被淘汰的消息里提取关键事实，
并写入 [`crate::memory::Store`] 以便后续召回。

# 事实提取启发式规则

1. **助手结论** —— 助手消息的最后一段（通常是摘要）
2. **用户问题** —— 原始用户输入对召回而言是有用的上下文
3. **关键词信号** —— 消息中包含 “decided”（已决定）、“important”（重要）、
   “remember”（记住）、“conclusion”（结论）等词
4. **跳过**：工具结果、系统消息、过短的消息（少于 50 字符）

# 去重

事实按内容哈希去重：不使用顺序计数器，而是根据事实文本的哈希值
派生出键。这意味着同一事实被多次提取时会覆盖（upsert）而不是产生重复项。
*/
use echo_core::llm::types::{Message, Role};
use echo_core::memory::types::{MemoryMeta, MemorySource, MemoryType};
use echo_state::compression::{MemoryPromoter, MemoryPromotionReceipt};
use futures::future::BoxFuture;
use std::sync::Arc;

/**
基于存储的记忆推广器。

从压缩过程中被淘汰的消息里提取关键事实，并将其作为带类型的记忆
（`MemoryMeta`）写入统一命名空间
（`crate::evolution::layer::WARM_NAMESPACE` = `["agent","memories"]`），
以便后续召回。（stage4 A2：此前是以原始 JSON 写入 `["l3_promoted"]` ——
割裂点1：推广器写入的事实现在与 agent 写入的记忆走相同的带类型路径，
并且可以通过复合分数召回检索到。）
*/
pub struct StoreMemoryPromoter {
    /// 负责持久化写入的分层记忆管理器。
    layer_manager: Arc<crate::evolution::MemoryLayerManager>,
}

impl StoreMemoryPromoter {
    /// 创建一个通过规范分层记忆权威写入的推广器（内容去重始终开启）。
    pub fn new(layer_manager: Arc<crate::evolution::MemoryLayerManager>) -> Self {
        Self { layer_manager }
    }

    /**
    为去重计算一个确定性的、基于内容的键。

    使用 FNV-1a 哈希（在进程重启后依然保持确定性），确保同一事实
    始终映射到同一个键，从而实现跨会话去重。
    */
    fn content_key(fact: &str) -> String {
        durable_memory_content_key(fact)
    }
}

/// 所有压缩记忆写入方共用的、基于内容派生的稳定键。
pub(crate) fn durable_memory_content_key(content: &str) -> String {
    let hash = echo_core::utils::hash::fnv1a_64(content.trim().as_bytes());
    format!("l3_{hash:016x}")
}

impl MemoryPromoter for StoreMemoryPromoter {
    /// 从被淘汰的消息中提取事实并写入分层记忆管理器，返回推广回执。
    fn promote(
        &self,
        evicted: &[Message],
    ) -> BoxFuture<'_, echo_core::error::Result<MemoryPromotionReceipt>> {
        let facts = extract_key_facts(evicted);
        let submitted = facts.len();
        let layer_manager = self.layer_manager.clone();

        Box::pin(async move {
            let mut promoted = 0usize;
            let mut deduplicated = 0usize;
            for (fact, fact_type) in facts.into_iter() {
                let key = Self::content_key(&fact);
                let memory_type = fact_type_to_memory_type(fact_type);
                let recall_weight = if contains_signal_word(&fact) {
                    0.7
                } else {
                    0.4
                };
                let meta = MemoryMeta::new(memory_type, MemorySource::L3Promotion, "l3_promotion")
                    .with_recall_weight(recall_weight);
                if layer_manager
                    .locate(&key)
                    .await?
                    .is_some_and(|(_, existing)| existing.content.trim() == fact.trim())
                {
                    deduplicated = deduplicated.saturating_add(1);
                    continue;
                }
                layer_manager.write_memory(&key, &fact, meta).await?;
                promoted = promoted.saturating_add(1);
            }
            Ok(MemoryPromotionReceipt {
                submitted,
                promoted,
                deduplicated,
            })
        })
    }
}

/// 将推广器使用的事实类型字符串映射为 `MemoryType`。
fn fact_type_to_memory_type(fact_type: &str) -> MemoryType {
    match fact_type {
        "error" => MemoryType::ErrorResolution,
        "decision" => MemoryType::ArchitectureDecision,
        "user_preference" => MemoryType::UserPreference,
        "pending_task" | "user_query" => MemoryType::WorkflowPattern,
        "tool_output" | "file_reference" | "assistant_conclusion" | "general" => {
            MemoryType::ProjectFact
        }
        _ => MemoryType::ProjectFact,
    }
}

/// 使用启发式规则从被淘汰的消息中提取关键事实。
fn extract_key_facts(messages: &[Message]) -> Vec<(String, &'static str)> {
    let mut facts = Vec::new();

    for msg in messages {
        let text = match msg.content.as_text() {
            Some(t) if t.chars().count() >= 50 => t,
            _ => continue, // 跳过过短或空的消息
        };

        match msg.role {
            Role::System => {
                // 跳过系统消息——它们不属于对话事实
                continue;
            }
            Role::Tool => {
                // 提取工具摘要：工具输出中的关键发现
                if let Some(digest) = tool_digest(&text) {
                    facts.push((truncate_fact(&digest), "tool_output"));
                }
                continue;
            }
            Role::Assistant => {
                // 从助手消息中提取类似结论的内容
                if contains_signal_word(&text) {
                    facts.push((truncate_fact(&text), classify_fact_type(&text, &msg.role)));
                } else {
                    // 取最后一段作为可能的结论
                    if let Some(last_para) = last_paragraph(&text)
                        && last_para.chars().count() >= 50
                    {
                            facts.push((
                                truncate_fact(&last_para),
                                classify_fact_type(&last_para, &msg.role),
                            ));
                        }
                }
            }
            Role::User
                // 用户问题是对召回有用的上下文
                if text.chars().count() >= 50 && !text.starts_with('[') => {
                    facts.push((truncate_fact(&text), classify_fact_type(&text, &msg.role)));
                }
            _ => {}
        }
    }

    facts
}

/// 根据内容模式对事实进行分类。
fn classify_fact_type(text: &str, role: &Role) -> &'static str {
    let lower = text.to_lowercase();
    if lower.contains("todo") || lower.contains("fixme") || lower.contains("待处理") {
        "pending_task"
    } else if lower.contains("error")
        || lower.contains("failed")
        || lower.contains("panic")
        || lower.contains("失败")
        || lower.contains("错误")
    {
        "error"
    } else if lower.contains("decided") || lower.contains("decision") || lower.contains("决定") {
        "decision"
    } else if lower.contains(".rs")
        || lower.contains(".py")
        || lower.contains(".js")
        || lower.contains(".toml")
        || lower.contains("src/")
        || lower.contains("/file")
    {
        "file_reference"
    } else if lower.contains("prefer")
        || lower.contains("don't")
        || lower.contains("must")
        || lower.contains("should")
        || lower.contains("不要")
        || lower.contains("必须")
    {
        "user_preference"
    } else {
        if *role == Role::User {
            "user_query"
        } else if *role == Role::Assistant {
            "assistant_conclusion"
        } else {
            "general"
        }
    }
}

/**
从工具输出中提取简洁摘要——关键发现、错误、文件路径。

若无法提取到任何有意义的内容，则返回 `None`。
*/
fn tool_digest(text: &str) -> Option<String> {
    let mut highlights: Vec<&str> = Vec::new();

    for line in text.lines() {
        let lower = line.to_lowercase();
        if lower.contains("error")
            || lower.contains("fail")
            || lower.contains("panic")
            || lower.contains("test result")
            || lower.contains("pass")
            || lower.contains("失败")
            || lower.contains("错误")
            || lower.contains("异常")
            || lower.contains("测试")
            || lower.contains("通过")
            || ((lower.contains(".rs")
                || lower.contains(".py")
                || lower.contains(".js")
                || lower.contains(".toml")
                || lower.contains("src/"))
                && line.chars().count() < 200)
        {
            highlights.push(line.trim());
        }
    }

    if highlights.is_empty() {
        None
    } else {
        Some(
            highlights
                .iter()
                .take(3)
                .cloned()
                .collect::<Vec<_>>()
                .join(" | "),
        )
    }
}

/// 检查文本中是否包含表示重要事实的信号词。
fn contains_signal_word(text: &str) -> bool {
    let lower = text.to_lowercase();
    const SIGNALS: &[&str] = &[
        "decided",
        "decision",
        "conclusion",
        "important",
        "remember",
        "discovered",
        "found that",
        "result:",
        "summary:",
        "决定",
        "结论",
        "发现",
        "重要",
        "总结",
    ];
    SIGNALS.iter().any(|s| lower.contains(s))
}

/// 从文本中提取最后一段（以双换行符分隔）。
fn last_paragraph(text: &str) -> Option<String> {
    text.rsplit("\n\n").next().map(|s| s.trim().to_string())
}

/// 将事实截断到合理长度（约 300 字符），保证 UTF-8 安全。
fn truncate_fact(text: &str) -> String {
    let trimmed = text.trim();
    if trimmed.chars().count() <= 300 {
        trimmed.to_string()
    } else {
        format!("{}...", trimmed.chars().take(300).collect::<String>())
    }
}

/// FNV-1a 哈希——进程重启后依然保持确定性。
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_key_facts_skips_short_messages() {
        let messages = vec![
            Message::user("Hi".to_string()),      // 太短
            Message::assistant("OK".to_string()), // 太短
        ];
        let facts = extract_key_facts(&messages);
        assert!(facts.is_empty());
    }

    #[test]
    fn test_extract_key_facts_skips_tool_results() {
        let messages = vec![Message::tool_result(
            "call_1".into(),
            "read_file".into(),
            "x".repeat(500), // 较长的工具输出
        )];
        let facts = extract_key_facts(&messages);
        assert!(facts.is_empty(), "Tool results should be skipped");
    }

    #[test]
    fn test_extract_key_facts_signal_words() {
        let messages = vec![Message::assistant(
            "After analysis, we decided to use PostgreSQL for the database layer. \
             This provides better JSON support and full-text search capabilities."
                .to_string(),
        )];
        let facts = extract_key_facts(&messages);
        assert_eq!(facts.len(), 1);
        assert!(
            facts
                .first()
                .is_some_and(|(fact, _)| fact.contains("PostgreSQL"))
        );
    }

    #[test]
    fn test_extract_key_facts_user_questions() {
        let messages = vec![Message::user(
            "How should we handle authentication in the microservice architecture? \
             We need to support both OAuth2 and API keys."
                .to_string(),
        )];
        let facts = extract_key_facts(&messages);
        assert_eq!(facts.len(), 1);
        assert!(
            facts
                .first()
                .is_some_and(|(fact, _)| fact.contains("authentication"))
        );
    }

    #[test]
    fn durable_key_is_shared_for_equivalent_content() {
        assert_eq!(
            durable_memory_content_key("user prefers Rust"),
            durable_memory_content_key("  user prefers Rust\n")
        );
    }

    #[test]
    fn test_truncate_fact() {
        let short = "This is a short fact.";
        assert_eq!(truncate_fact(short), short);

        let long = "x".repeat(500);
        let truncated = truncate_fact(&long);
        assert!(truncated.len() <= 310); // 300 + "..."
        assert!(truncated.ends_with("..."));
    }

    #[test]
    fn test_contains_signal_word() {
        assert!(contains_signal_word("We decided to use Rust"));
        assert!(contains_signal_word("这是一个重要的决定"));
        assert!(!contains_signal_word("Hello world"));
    }

    /**
    RFC 5.2.2：端到端质量验证。

    构建一个包含重要事实的对话 → 提取关键事实 → 写入内存态 Store →
    验证这些事实是可以被召回的。
    */
    #[tokio::test]
    async fn test_e2e_fact_extraction_and_recall() -> echo_core::error::Result<()> {
        use crate::evolution::MemoryLayerManager;
        use crate::evolution::audit::NullChangeLog;
        use echo_core::memory::Store;
        use echo_state::memory::InMemoryStore;

        let store = Arc::new(InMemoryStore::new());
        let memory_dir = tempfile::tempdir()?;
        let layer_manager = Arc::new(MemoryLayerManager::new(
            memory_dir.path().to_path_buf(),
            store.clone(),
            Box::new(NullChangeLog),
        ));
        let promoter = StoreMemoryPromoter::new(layer_manager);

        // 构建一个包含应当被保留的重要事实的对话
        let messages = vec![
            // 这条助手消息包含一个决定（信号词）
            Message::assistant(
                "After thorough analysis, we decided to use PostgreSQL for the database layer \
                 because it provides better JSON support and full-text search capabilities. \
                 This is an important architectural decision."
                    .to_string(),
            ),
            // 这条用户消息是一个有意义的问题
            Message::user(
                "How should we handle authentication in the microservice architecture? \
                 We need to support both OAuth2 and API keys for different clients."
                    .to_string(),
            ),
            // 这条助手消息的最后一段包含结论
            Message::assistant(
                "Let me analyze the authentication options available.\n\n\
                 In conclusion, we recommend using JWT tokens with OAuth2 for external \
                 clients and API keys for internal service-to-service communication. \
                 This provides the best balance of security and simplicity."
                    .to_string(),
            ),
            // 这条应当被跳过（工具结果）
            Message::tool_result(
                "call_1".into(),
                "read_file".into(),
                "file contents here that are not important facts".repeat(20),
            ),
            // 这条应当被跳过（太短）
            Message::user("ok".to_string()),
        ];

        // 第 1 步：提取事实
        let facts = extract_key_facts(&messages);
        assert!(
            facts.len() >= 2,
            "Should extract at least 2 facts (decision + user question), got {}",
            facts.len()
        );

        // 验证提取出的事实包含关键信息
        let all_facts: String = facts
            .iter()
            .map(|(s, _)| s.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        assert!(
            all_facts.contains("PostgreSQL"),
            "Should preserve the database decision fact"
        );
        assert!(
            all_facts.contains("authentication") || all_facts.contains("microservice"),
            "Should preserve the user's question about authentication"
        );

        // 第 2 步：推广到 Store（通过推广器写入）
        promoter.promote(&messages).await?;

        // 第 3 步：验证事实可以通过 Store 搜索被召回
        let results = store.search(&["agent", "memories"], "PostgreSQL", 5).await;
        assert!(results.is_ok(), "Store search should succeed");
        let items = results.unwrap_or_default();
        assert!(
            !items.is_empty(),
            "Should be able to recall the PostgreSQL decision fact from Store"
        );

        // 第 4 步：验证质量——被召回的条目包含有意义的内容
        let content = items
            .first()
            .and_then(|recalled| recalled.value.get("content"))
            .and_then(|value| value.as_str())
            .unwrap_or("");
        assert!(
            content.contains("PostgreSQL"),
            "Recalled content should contain the key fact, got: {}",
            content
        );
        assert!(
            content.len() >= 50,
            "Recalled content should be meaningful (≥50 chars), got {} chars",
            content.len()
        );
        Ok(())
    }
}
