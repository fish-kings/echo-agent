//! JSON 解析工具函数
//!
//! 提供从 LLM 输出中提取和修复 JSON 的常用函数。

/// 从 Markdown 代码块或纯文本中提取 JSON
///
/// 支持以下格式：
/// - ````json ... ````
/// - ```` ... ````
/// - 纯 JSON 文本
pub fn extract_json_from_markdown(content: &str) -> String {
    if let Some(start) = content.find("```json") {
        let rest = &content[start + 7..];
        if let Some(end) = rest.find("```") {
            return rest[..end].trim().to_string();
        }
    }
    if let Some(start) = content.find("```") {
        let rest = &content[start + 3..];
        if let Some(end) = rest.find("```") {
            return rest[..end].trim().to_string();
        }
    }
    content.trim().to_string()
}

/// 清理常见的 JSON 格式问题
///
/// - 修复 `}` 或 `]` 之前的多余逗号，但仅在引号字符串之外生效。
///
/// 有意不处理单引号伪 JSON：纯词法层面的全局替换无法区分内容中的
/// 分隔符与撇号。
pub fn clean_json(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let mut cleaned = String::with_capacity(s.len());
    let mut in_string = false;
    let mut escaped = false;
    let mut index = 0usize;
    while let Some(ch) = chars.get(index).copied() {
        if in_string {
            cleaned.push(ch);
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
            index = index.saturating_add(1);
            continue;
        }
        if ch == '"' {
            in_string = true;
            cleaned.push(ch);
            index = index.saturating_add(1);
            continue;
        }
        if ch == ',' {
            let mut lookahead = index.saturating_add(1);
            while chars
                .get(lookahead)
                .is_some_and(|next| next.is_whitespace())
            {
                lookahead = lookahead.saturating_add(1);
            }
            if matches!(chars.get(lookahead), Some('}' | ']')) {
                index = index.saturating_add(1);
                continue;
            }
        }
        cleaned.push(ch);
        index = index.saturating_add(1);
    }
    cleaned
}

#[cfg(test)]
mod tests {
    use super::*;

    // 验证从带 ```json 标记的代码块中提取 JSON
    #[test]
    fn test_extract_from_json_block() {
        let input = "Here is the result:\n```json\n{\"key\": \"value\"}\n```\nDone.";
        assert_eq!(extract_json_from_markdown(input), "{\"key\": \"value\"}");
    }

    // 验证从不带语言标记的普通代码块中提取 JSON
    #[test]
    fn test_extract_from_plain_block() {
        let input = "```\n{\"key\": \"value\"}\n```";
        assert_eq!(extract_json_from_markdown(input), "{\"key\": \"value\"}");
    }

    // 验证对没有代码块包裹的纯 JSON 文本直接返回
    #[test]
    fn test_extract_bare_json() {
        let input = r#"{"steps": [{"description": "hello"}]}"#;
        assert_eq!(extract_json_from_markdown(input), input.trim());
    }

    // 验证能正确清除对象和数组中的尾随逗号
    #[test]
    fn test_clean_trailing_commas() {
        assert_eq!(clean_json(r#"{"a": 1,}"#), r#"{"a": 1}"#);
        assert_eq!(clean_json(r#"[1, 2,]"#), r#"[1, 2]"#);
    }

    // 验证单引号伪 JSON 不会被误改写
    #[test]
    fn test_clean_does_not_rewrite_single_quotes() {
        assert_eq!(clean_json("{'a': 'don\'t'}"), "{'a': 'don\'t'}");
    }

    // 验证双引号字符串内容保持不变
    #[test]
    fn test_clean_preserves_double_quotes() {
        let input = r#"{"a": "it's fine"}"#;
        assert_eq!(clean_json(input), input);
    }

    // 验证字符串内部形似结构字符（逗号加括号）的文本不会被误清除
    #[test]
    fn test_clean_preserves_structural_text_inside_strings() {
        let input = r#"{"text":"keep ,} and ,] literal","escaped":"\\\" ,}",}"#;
        assert_eq!(
            clean_json(input),
            r#"{"text":"keep ,} and ,] literal","escaped":"\\\" ,}"}"#
        );
    }

    // 验证尾随逗号前存在 Unicode 空白字符时依然能被正确清除
    #[test]
    fn test_clean_trailing_comma_with_unicode_whitespace() {
        assert_eq!(clean_json("{\"中文\": 1, \n}"), "{\"中文\": 1 \n}");
    }
}
