/*!
 * 摘要校验器——压缩后的质量检查。
 *
 * 对压缩输出执行轻量级的基于规则的检查，验证关键信息没有丢失。无需调用 LLM。
 */

use crate::compression::CompressionCheckpoint;
use echo_core::llm::types::{Message, Role};

/// 对一次压缩输出执行全部校验检查后的结果。
#[derive(Debug, Clone)]
pub struct SummaryVerification {
    /// 整体是否通过（所有 P0 检查都必须通过）。
    pub passed: bool,
    /// 各项检查的结果。
    pub checks: Vec<VerificationCheck>,
}

/// 单项校验检查的结果。
#[derive(Debug, Clone)]
pub struct VerificationCheck {
    /// 检查项名称
    pub name: String,
    /// 该检查是否通过
    pub passed: bool,
    /// 检查优先级
    pub priority: CheckPriority,
    /// 检查详情说明
    pub detail: String,
}

/// 校验检查的优先级。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CheckPriority {
    /// 必须通过的检查——若失败则认为压缩结果已损坏。
    P0,
    /// 应当通过的检查——失败仅作为警告，不阻断。
    P1,
}

/**
 * 对压缩后的输出执行全部校验检查。
 *
 * `compressed_messages` — 压缩后的消息（即将发送给 LLM 的内容）。
 * `checkpoint` — 包含 token 统计与摘要的压缩检查点。
 * `original_messages` — 压缩前的完整消息缓冲（作为上下文参考）。
 */
pub fn verify_compression(
    compressed_messages: &[Message],
    checkpoint: &CompressionCheckpoint,
    original_messages: &[Message],
) -> SummaryVerification {
    let mut checks = Vec::new();

    // P0：满足 token 目标
    checks.push(check_token_target(checkpoint));

    // P0：摘要非空（若使用了 LLM 压缩器）
    if checkpoint.summary.is_some() {
        checks.push(check_summary_not_empty(checkpoint));
    }

    // P0：最后一条用户查询的关键词出现在摘要或保留消息中
    checks.push(check_last_query_presence(
        original_messages,
        compressed_messages,
        checkpoint,
    ));

    // P0：保留了最近的文件路径
    checks.push(check_file_paths_presence(
        original_messages,
        compressed_messages,
        checkpoint,
    ));

    // P1：保留了待处理任务
    checks.push(check_pending_tasks(
        original_messages,
        compressed_messages,
        checkpoint,
    ));

    // P1：保留了错误信息
    checks.push(check_error_presence(
        original_messages,
        compressed_messages,
        checkpoint,
    ));

    // P1：保留了用户偏好
    checks.push(check_user_preferences(
        original_messages,
        compressed_messages,
        checkpoint,
    ));

    let passed = checks
        .iter()
        .filter(|c| c.priority == CheckPriority::P0)
        .all(|c| c.passed);

    SummaryVerification { passed, checks }
}

/**
 * P0：压缩后的 token 数不得超过压缩前。
 * （更强的“处于目标上限内”检查需要模型上下文信息，此处无法获取。）
 */
fn check_token_target(checkpoint: &CompressionCheckpoint) -> VerificationCheck {
    let after = checkpoint.token_after;
    let before = checkpoint.token_before;
    let reduction = if before > 0 {
        (before.saturating_sub(after)) as f64 / before as f64
    } else {
        0.0
    };

    VerificationCheck {
        name: "tokens_not_increased".to_string(),
        passed: after <= before,
        priority: CheckPriority::P0,
        detail: format!(
            "Tokens: {} → {} (reduction: {:.1}%)",
            before,
            after,
            reduction * 100.0
        ),
    }
}

/// P0：摘要不应过短。
fn check_summary_not_empty(checkpoint: &CompressionCheckpoint) -> VerificationCheck {
    let len = checkpoint.summary.as_deref().unwrap_or("").len();
    VerificationCheck {
        name: "summary_not_empty".to_string(),
        passed: len > 50,
        priority: CheckPriority::P0,
        detail: format!("Summary length: {} chars", len),
    }
}

/// P0：最后一条用户查询的关键词应出现在压缩输出中。
fn check_last_query_presence(
    original: &[Message],
    compressed: &[Message],
    checkpoint: &CompressionCheckpoint,
) -> VerificationCheck {
    // 在原始消息中查找最后一条用户消息
    let last_user = original.iter().rev().find(|m| m.role == Role::User);

    match last_user {
        Some(msg) => {
            let text = msg.content.as_text().unwrap_or_default();
            // 提取关键词（较长的词更可能具有区分度）
            let keywords: Vec<&str> = text.split_whitespace().filter(|w| w.len() > 3).collect();

            if keywords.is_empty() {
                return VerificationCheck {
                    name: "last_query_presence".to_string(),
                    passed: true, // 没有关键词时无法检查
                    priority: CheckPriority::P0,
                    detail: "No distinctive keywords found in last query".to_string(),
                };
            }

            // 检查是否至少 50% 的关键词出现在压缩消息或摘要中
            let combined = format!(
                "{} {}",
                compressed
                    .iter()
                    .filter_map(|m| m.content.as_text())
                    .collect::<Vec<_>>()
                    .join(" "),
                checkpoint.summary.as_deref().unwrap_or("")
            );

            let found = keywords.iter().filter(|kw| combined.contains(*kw)).count();
            let ratio = found as f64 / keywords.len() as f64;

            VerificationCheck {
                name: "last_query_presence".to_string(),
                passed: ratio >= 0.5,
                priority: CheckPriority::P0,
                detail: format!(
                    "Last query keyword presence: {}/{} ({:.0}%)",
                    found,
                    keywords.len(),
                    ratio * 100.0
                ),
            }
        }
        None => VerificationCheck {
            name: "last_query_presence".to_string(),
            passed: true,
            priority: CheckPriority::P0,
            detail: "No user message found".to_string(),
        },
    }
}

/// P0：工具调用中出现的最近文件路径应被保留。
fn check_file_paths_presence(
    original: &[Message],
    compressed: &[Message],
    checkpoint: &CompressionCheckpoint,
) -> VerificationCheck {
    // 从最近 3 轮工具消息中提取文件路径
    let file_paths: Vec<String> = original
        .iter()
        .rev()
        .filter(|m| m.role == Role::Tool)
        .take(6) // 约最近 3 轮的工具结果
        .filter_map(|m| m.content.as_text())
        .flat_map(|text| extract_file_paths_from_text(&text))
        .collect();

    if file_paths.is_empty() {
        return VerificationCheck {
            name: "file_paths_presence".to_string(),
            passed: true, // 没有需要检查的文件路径
            priority: CheckPriority::P0,
            detail: "No file paths found in recent tool messages".to_string(),
        };
    }

    let combined = format!(
        "{} {}",
        compressed
            .iter()
            .filter_map(|m| m.content.as_text())
            .collect::<Vec<_>>()
            .join(" "),
        checkpoint.summary.as_deref().unwrap_or("")
    );

    let found = file_paths
        .iter()
        .filter(|fp| combined.contains(fp.as_str()))
        .count();
    let ratio = found as f64 / file_paths.len() as f64;

    VerificationCheck {
        name: "file_paths_presence".to_string(),
        passed: ratio >= 0.5,
        priority: CheckPriority::P0,
        detail: format!(
            "File path preservation: {}/{} ({:.0}%)",
            found,
            file_paths.len(),
            ratio * 100.0
        ),
    }
}

/**
 * 从工具输出文本中提取可能的文件路径。
 *
 * 匹配常见的源码文件形式：`src/auth.rs`、`/path/to/file`、`Cargo.toml`。
 * 过滤掉版本号（1.2.3）、URL 以及一般的带点单词。
 */
fn extract_file_paths_from_text(text: &str) -> Vec<String> {
    // 常见源码扩展名，用于减少误报
    let code_extensions = [
        ".rs", ".py", ".js", ".ts", ".go", ".java", ".c", ".cpp", ".h", ".hpp", ".toml", ".yaml",
        ".yml", ".json", ".md", ".txt", ".lock", ".sql", ".sh", ".css", ".html", ".xml", ".rb",
        ".php", ".swift", ".kt", ".scala", ".r", ".vue", ".svelte", ".tsx", ".jsx", ".tf",
        ".proto", ".graphql", ".gql",
    ];
    let mut paths = Vec::new();
    for word in text.split_whitespace() {
        let trimmed = word.trim_matches(|c: char| {
            c == '"'
                || c == '\''
                || c == '`'
                || c == ','
                || c == '('
                || c == ')'
                || c == '['
                || c == ']'
        });
        // 必须包含已知扩展名或路径分隔符
        let has_code_ext = code_extensions.iter().any(|ext| trimmed.ends_with(ext));
        let has_path_sep = trimmed.contains('/');
        if (has_code_ext || has_path_sep)
            && trimmed.len() > 2
            && trimmed.len() < 200
            && !trimmed.starts_with("http")
            && !trimmed.starts_with("https")
            // 过滤版本号和带点的数字
            && !trimmed.chars().all(|c| c.is_ascii_digit() || c == '.')
        {
            paths.push(trimmed.to_string());
        }
    }
    paths
}

/// P1：TODO/待处理任务信息应在压缩后保留。
fn check_pending_tasks(
    original: &[Message],
    compressed: &[Message],
    checkpoint: &CompressionCheckpoint,
) -> VerificationCheck {
    let todo_patterns = ["TODO", "FIXME", "HACK", "PENDING", "待处理", "未完成"];

    // 在原始消息（被驱逐范围）中查找包含 TODO 的消息
    let todos_in_original: Vec<String> = original
        .iter()
        .filter(|m| {
            let text = m.content.as_text().unwrap_or_default();
            todo_patterns.iter().any(|p| text.contains(p))
        })
        .filter_map(|m| m.content.as_text())
        .collect();

    if todos_in_original.is_empty() {
        return VerificationCheck {
            name: "pending_tasks_presence".to_string(),
            passed: true,
            priority: CheckPriority::P1,
            detail: "No TODO items found".to_string(),
        };
    }

    let combined = format!(
        "{} {}",
        compressed
            .iter()
            .filter_map(|m| m.content.as_text())
            .collect::<Vec<_>>()
            .join(" "),
        checkpoint.summary.as_deref().unwrap_or("")
    );

    let found = todos_in_original
        .iter()
        .filter(|todo| combined.contains(todo.as_str()))
        .count();

    VerificationCheck {
        name: "pending_tasks_presence".to_string(),
        passed: found > 0 || combined.contains("TODO"),
        priority: CheckPriority::P1,
        detail: format!(
            "TODO items: {}/{} found in compressed output",
            found,
            todos_in_original.len()
        ),
    }
}

/// P1：错误信息应在压缩后保留。
fn check_error_presence(
    original: &[Message],
    compressed: &[Message],
    checkpoint: &CompressionCheckpoint,
) -> VerificationCheck {
    let error_patterns = [
        "error",
        "Error",
        "ERROR",
        "failed",
        "Failed",
        "panic",
        "Panic",
        "exception",
        "失败",
        "错误",
    ];

    let errors_in_original: Vec<String> = original
        .iter()
        .filter(|m| {
            let text = m.content.as_text().unwrap_or_default();
            error_patterns.iter().any(|p| text.contains(p))
        })
        .filter_map(|m| m.content.as_text())
        .collect();

    if errors_in_original.is_empty() {
        return VerificationCheck {
            name: "error_presence".to_string(),
            passed: true,
            priority: CheckPriority::P1,
            detail: "No errors found in original messages".to_string(),
        };
    }

    let combined = format!(
        "{} {}",
        compressed
            .iter()
            .filter_map(|m| m.content.as_text())
            .collect::<Vec<_>>()
            .join(" "),
        checkpoint.summary.as_deref().unwrap_or("")
    );

    let found = errors_in_original
        .iter()
        .any(|e| combined.contains(e.as_str()));

    VerificationCheck {
        name: "error_presence".to_string(),
        passed: found,
        priority: CheckPriority::P1,
        detail: format!("{} error messages to check", errors_in_original.len()),
    }
}

/// P1：用户偏好/约束应在压缩后保留。
fn check_user_preferences(
    original: &[Message],
    compressed: &[Message],
    checkpoint: &CompressionCheckpoint,
) -> VerificationCheck {
    let pref_patterns = [
        "不要",
        "必须",
        "prefer",
        "important",
        "重要",
        "don't",
        "must",
        "should",
        "always",
        "never",
    ];

    let prefs_in_original: Vec<String> = original
        .iter()
        .filter(|m| m.role == Role::User)
        .filter(|m| {
            let text = m.content.as_text().unwrap_or_default();
            pref_patterns
                .iter()
                .any(|p| text.to_lowercase().contains(&p.to_lowercase()))
        })
        .filter_map(|m| m.content.as_text())
        .collect();

    if prefs_in_original.is_empty() {
        return VerificationCheck {
            name: "user_preference_presence".to_string(),
            passed: true,
            priority: CheckPriority::P1,
            detail: "No preference patterns found".to_string(),
        };
    }

    let combined = format!(
        "{} {}",
        compressed
            .iter()
            .filter_map(|m| m.content.as_text())
            .collect::<Vec<_>>()
            .join(" "),
        checkpoint.summary.as_deref().unwrap_or("")
    );

    let found = prefs_in_original
        .iter()
        .filter(|p| combined.contains(p.as_str()))
        .count();

    VerificationCheck {
        name: "user_preference_presence".to_string(),
        passed: found as f64 / prefs_in_original.len() as f64 >= 0.5,
        priority: CheckPriority::P1,
        detail: format!(
            "Preferences: {}/{} preserved",
            found,
            prefs_in_original.len()
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compression::CompressionCheckpoint;
    use echo_core::llm::types::Message;

    /// 无问题时所有检查都应通过。
    #[test]
    fn test_verifier_all_pass_when_no_issues() {
        let compressed = vec![
            Message::system("system".to_string()),
            Message::user("fix auth".to_string()),
            Message::assistant("done".to_string()),
        ];
        let original = compressed.clone();
        let cp = CompressionCheckpoint::new("SlidingWindow")
            .with_counts(3, 2)
            .with_tokens(500, 200);

        let verification = verify_compression(&compressed, &cp, &original);
        assert!(verification.passed);
    }

    /// token 数增加时 token 目标检查应失败。
    #[test]
    fn test_token_target_check_fails_when_tokens_increase() {
        let msgs = vec![Message::user("hello".to_string())];
        let cp = CompressionCheckpoint::new("test")
            .with_counts(1, 0)
            .with_tokens(100, 200); // token 增加 = 失败

        let verification = verify_compression(&msgs, &cp, &msgs);
        let token_check = verification
            .checks
            .iter()
            .find(|c| c.name == "tokens_not_increased")
            .unwrap();
        assert!(!token_check.passed);
    }

    /// 应能从文本中识别文件路径。
    #[test]
    fn test_file_paths_detected() {
        let paths = extract_file_paths_from_text("Modified src/auth.rs and Cargo.toml");
        assert!(!paths.is_empty());
        assert!(paths.contains(&"src/auth.rs".to_string()));
        assert!(paths.contains(&"Cargo.toml".to_string()));
    }
}
