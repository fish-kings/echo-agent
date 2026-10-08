/*!
集成配置的 Debug 输出与诊断信息共用的脱敏工具。

集成配置天然包含凭据。本模块沿用现有 `LlmConfig` 的 Debug 约定：
让携带密钥的值不出现在派生的 Debug 输出中，并在原始传输错误
变成日志或返回的错误字符串之前，套用框架的内容保留策略。
*/

use echo_core::utils::retention::ContentRetentionPolicy;
use serde_json::Value;
#[cfg(any(feature = "mcp", test))]
use std::collections::HashMap;

/// 脱敏后替换敏感内容所用的占位文本。
pub(crate) const REDACTED: &str = "[REDACTED]";

/// 在记录日志或嵌入错误信息之前，对形似凭据的文本进行脱敏。
pub(crate) fn text(value: &str) -> String {
    ContentRetentionPolicy::default().sanitize_text(value)
}

/**
先精确移除当前配置持有的凭据，再执行基于形态的脱敏与有界保留。

这样的顺序也能覆盖跨越保留截断边界的凭据。
*/
pub(crate) fn text_with_secrets<S>(value: &str, secrets: impl IntoIterator<Item = S>) -> String
where
    S: AsRef<str>,
{
    let mut patterns = secrets
        .into_iter()
        .map(|secret| secret.as_ref().chars().collect::<Vec<_>>())
        .filter(|secret| !secret.is_empty())
        .collect::<Vec<_>>();
    patterns.sort_by(|left, right| right.len().cmp(&left.len()).then_with(|| left.cmp(right)));
    patterns.dedup();

    let source = value.chars().collect::<Vec<_>>();
    let mut sanitized = String::with_capacity(value.len());
    let mut position = 0usize;
    while position < source.len() {
        let matched = patterns.iter().find(|pattern| {
            pattern.iter().enumerate().all(|(offset, expected)| {
                position
                    .checked_add(offset)
                    .and_then(|index| source.get(index))
                    == Some(expected)
            })
        });
        if let Some(pattern) = matched {
            sanitized.push_str(REDACTED);
            position = position.saturating_add(pattern.len());
        } else if let Some(character) = source.get(position) {
            sanitized.push(*character);
            position = position.saturating_add(1);
        } else {
            break;
        }
    }
    text(&sanitized)
}

/// 在格式化诊断信息之前，对嵌套的凭据字段进行脱敏。
#[cfg(any(feature = "channels", test))]
pub(crate) fn json(value: &Value) -> Value {
    let mut sanitized = value.clone();
    ContentRetentionPolicy::default().sanitize_json(&mut sanitized);
    sanitized
}

/// 对结构化诊断信息脱敏，并移除当前扩展实例配置的精确敏感值。
pub(crate) fn json_with_secrets(value: &Value, secrets: &[String]) -> Value {
    /// 递归遍历 JSON，把所有字符串中的精确敏感值替换掉。
    fn redact_strings(value: &mut Value, secrets: &[String]) {
        match value {
            Value::String(text) => {
                *text = text_with_secrets(text, secrets.iter());
            }
            Value::Array(values) => {
                for value in values {
                    redact_strings(value, secrets);
                }
            }
            Value::Object(values) => {
                for value in values.values_mut() {
                    redact_strings(value, secrets);
                }
            }
            Value::Null | Value::Bool(_) | Value::Number(_) => {}
        }
    }

    let mut sanitized = value.clone();
    redact_strings(&mut sanitized, secrets);
    ContentRetentionPolicy::default().sanitize_json(&mut sanitized);
    sanitized
}

/**
移除诊断用 URL 中的用户信息以及查询参数值和片段值。

传输请求本身仍使用原始 URL。
*/
pub(crate) fn url(value: &str) -> String {
    let Ok(mut parsed) = url::Url::parse(value) else {
        return "[REDACTED URL]".to_string();
    };
    if !parsed.username().is_empty() {
        let _ = parsed.set_username(REDACTED);
    }
    if parsed.password().is_some() {
        let _ = parsed.set_password(Some(REDACTED));
    }
    if parsed.query().is_some() {
        let keys = parsed
            .query_pairs()
            .map(|(key, _)| key.into_owned())
            .collect::<Vec<_>>();
        parsed.set_query(None);
        {
            let mut pairs = parsed.query_pairs_mut();
            for key in keys {
                pairs.append_pair(&key, REDACTED);
            }
        }
    }
    if parsed.fragment().is_some() {
        parsed.set_fragment(Some(REDACTED));
    }
    parsed.to_string()
}

/// 格式化 reqwest 错误，防止其附带的请求 URL 绕过 URL 凭据脱敏。
pub(crate) fn request_error<S>(
    error: reqwest::Error,
    secrets: impl IntoIterator<Item = S>,
) -> String
where
    S: AsRef<str>,
{
    let secrets = secrets
        .into_iter()
        .map(|secret| secret.as_ref().to_string())
        .collect::<Vec<_>>();
    let attached_url = error.url().map(|value| url(value.as_str()));
    let mut diagnostic = error.without_url().to_string();
    if let Some(attached_url) = attached_url {
        diagnostic.push_str(" for url (");
        diagnostic.push_str(&attached_url);
        diagnostic.push(')');
    }
    text_with_secrets(&diagnostic, secrets.iter())
}

/// 保留可选密钥“是否已配置”的信息，但不暴露其内容。
#[cfg(any(feature = "channels", test))]
pub(crate) fn optional<T>(value: Option<&T>) -> Option<&'static str> {
    value.map(|_| REDACTED)
}

/// 保留含密钥集合“是否已配置”的信息，但不暴露其中的键或值。
#[cfg(any(feature = "mcp", test))]
pub(crate) fn collection_is_redacted(is_empty: bool) -> &'static str {
    if is_empty { "[]" } else { REDACTED }
}

/**
返回必须从 HTTP 诊断信息中移除的精确值。

对于 Authorization 头，同时包含完整的线上值和认证方案之后的凭据部分，
因为服务端通常只回显凭据本体。
*/
#[cfg(any(feature = "mcp", test))]
pub(crate) fn header_secrets(headers: &HashMap<String, String>) -> Vec<String> {
    let mut secrets = Vec::new();
    for (name, value) in headers {
        if !value.is_empty() {
            secrets.push(value.clone());
        }
        if name.eq_ignore_ascii_case("authorization")
            && let Some((_, credential)) = value.split_once(char::is_whitespace)
        {
            let credential = credential.trim();
            if !credential.is_empty() {
                secrets.push(credential.to_string());
            }
        }
    }
    secrets
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 验证扩展凭据赋值会被脱敏。
    #[test]
    fn redacts_extension_credential_assignments() {
        let secrets = [
            "abc123",
            "client-value",
            "app-value",
            "verify-value",
            "sign-value",
        ];
        let sanitized = text_with_secrets(
            "Authorization: Bearer abc123, client_secret=client-value, \
             app_secret=app-value, verification_token=verify-value, \
             signing_key=sign-value",
            secrets,
        );
        for secret in secrets {
            assert!(!sanitized.contains(secret), "secret survived: {sanitized}");
        }
        assert_eq!(url("://host?ticket=invalid-url-secret"), "[REDACTED URL]");
    }

    /// 验证 Option 与集合的脱敏只保留结构形状。
    #[test]
    fn optional_and_collection_redaction_keep_shape_only() {
        assert_eq!(optional(Some(&"secret")), Some(REDACTED));
        assert_eq!(optional::<&str>(None), None);
        assert_eq!(collection_is_redacted(true), "[]");
        assert_eq!(collection_is_redacted(false), REDACTED);
    }

    /// 验证 Authorization 密钥集合同时包含认证方案与载荷。
    #[test]
    fn authorization_secret_set_includes_scheme_and_payload() {
        let headers = HashMap::from([(
            "authorization".to_string(),
            "Bearer opaque-credential".to_string(),
        )]);
        let secrets = header_secrets(&headers);
        let diagnostic = text_with_secrets("credential opaque-credential rejected", secrets.iter());
        assert!(!diagnostic.contains("opaque-credential"));
        assert!(
            secrets
                .iter()
                .any(|secret| secret == "Bearer opaque-credential")
        );
        assert!(secrets.iter().any(|secret| secret == "opaque-credential"));
    }

    /// 验证配置值即使没有字段名也会被脱敏。
    #[test]
    fn configured_values_are_redacted_without_field_names() {
        let sanitized = text_with_secrets(
            "server echoed opaque-value and another-value",
            ["opaque-value", "another-value"],
        );
        assert!(!sanitized.contains("opaque-value"));
        assert!(!sanitized.contains("another-value"));
        assert_eq!(text_with_secrets("unchanged", [""]), "unchanged");

        let overlapping = text_with_secrets(
            "opaque-credential and RED",
            ["opaque", "opaque-credential", "RED"],
        );
        assert_eq!(overlapping, "[REDACTED] and [REDACTED]");

        let crossing_secret = "opaque-crossing-secret";
        let input = format!("{}{}", "x".repeat(16_380), crossing_secret);
        let sanitized = text_with_secrets(&input, [crossing_secret]);
        assert!(!sanitized.contains("opaque"));
    }

    /// 验证嵌套 JSON 与 URL 中的凭据会被脱敏。
    #[test]
    fn redacts_nested_json_and_url_credentials() {
        let value = serde_json::json!({
            "headers": {"Authorization": "Bearer header-secret"},
            "env": {"TOKEN": "env-secret"}
        });
        let sanitized = json(&value).to_string();
        assert!(!sanitized.contains("header-secret"));
        assert!(!sanitized.contains("env-secret"));

        let exact = json_with_secrets(
            &serde_json::json!({"message": "server echoed opaque-value"}),
            &["opaque-value".to_string()],
        );
        assert!(!exact.to_string().contains("opaque-value"));

        let crossing_secret = "opaque-crossing-secret";
        let exact = json_with_secrets(
            &serde_json::json!({
                "message": format!("{}{}", "x".repeat(16_380), crossing_secret)
            }),
            &[crossing_secret.to_string()],
        );
        let exact = exact.to_string();
        assert!(!exact.contains("opaque"));

        let sanitized_url = url(
            "https://client:password@example.invalid/mcp?session_id=session-secret&ticket=ticket-secret#fragment-secret",
        );
        for secret in [
            "client",
            "password",
            "session-secret",
            "ticket-secret",
            "fragment-secret",
        ] {
            assert!(
                !sanitized_url.contains(secret),
                "URL secret survived: {sanitized_url}"
            );
        }
    }

    /// 验证 reqwest 错误中附带的 URL 会被脱敏。
    #[tokio::test]
    async fn reqwest_error_redacts_attached_url() -> Result<(), Box<dyn std::error::Error>> {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
        let address = listener.local_addr()?;
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await?;
            let mut request = [0_u8; 1024];
            let _ = stream.read(&mut request).await?;
            stream
                .write_all(b"HTTP/1.1 500 Internal Server Error\r\nContent-Length: 0\r\n\r\n")
                .await?;
            Ok::<(), std::io::Error>(())
        });
        let error = reqwest::Client::new()
            .get(format!(
                "http://{address}/mcp?ticket=opaque-ticket&session=opaque-session"
            ))
            .send()
            .await?
            .error_for_status()
            .err()
            .ok_or("expected HTTP 500 to fail")?;
        server.await??;
        let diagnostic = request_error(error, std::iter::empty::<&str>());
        assert!(!diagnostic.contains("opaque-ticket"));
        assert!(!diagnostic.contains("opaque-session"));
        assert!(diagnostic.contains("ticket="));
        Ok(())
    }
}
