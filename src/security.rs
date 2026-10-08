/*!
密钥扫描与脱敏 —— 防止凭据泄露。

TODO(security, v0.3)：将模式拆分为高置信度（AWS/GH/Anthropic 密钥前缀）
与启发式/通用模式（误报率较高的 token/password 模式）两类。
高置信度匹配应当阻断执行；通用匹配仅做告警即可。
优先级：先从 AWS（AKIA 前缀）与 GitHub（ghp_/github_pat_）开始，
它们的误报率接近于零，之后再扩展到其他服务商。

TODO(security, v0.3)："URL 中的密码" 模式会匹配到文档与代码注释中
无害的示例。需要增加边界检查：排除以 "//" 或 "#" 开头的行
（注释行），并要求匹配内容出现在配置语境中
（例如出现在连接字符串或 URI 赋值内）。

对文本进行密钥扫描（API key、token、密码、私钥），
并对其进行脱敏或向用户发出警告。被输出护栏（guard）流水线
以及 /commit 流程使用。
*/

pub use echo_core::utils::retention::{
    SecretMatch, contains_secrets, redact_secrets, scan_secrets,
};

/// 扫描文本并返回适合在 CLI 中展示的摘要信息。
pub fn scan_summary(text: &str) -> Vec<String> {
    let matches = scan_secrets(text);
    if matches.is_empty() {
        return vec!["No secrets detected.".into()];
    }
    let mut by_type: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for m in &matches {
        *by_type.entry(m.secret_type).or_default() += 1;
    }
    let mut summary: Vec<String> = vec![format!("Found {} potential secret(s):", matches.len())];
    for (st, count) in &by_type {
        summary.push(format!("  {} x {}", count, st));
    }
    summary.push("Consider using environment variables or a secrets manager.".into());
    summary
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 验证能识别 OpenAI API Key。
    #[test]
    fn test_scan_openai_key() {
        let text = "export OPENAI_API_KEY=sk-proj12345678901234567890abcdef";
        let matches = scan_secrets(text);
        assert!(matches.iter().any(|m| m.secret_type == "OpenAI API Key"));
    }

    /// 验证密钥会被替换为脱敏占位符。
    #[test]
    fn test_redact_secrets() {
        let text = "token: sk-ant-api03-abc123def456ghi789";
        let redacted = redact_secrets(text);
        assert!(redacted.contains("[REDACTED:"));
        assert!(!redacted.contains("sk-ant-api03"));
    }

    /// 验证普通文本不会被误判为包含密钥。
    #[test]
    fn test_no_secrets() {
        let text = "This is normal code without any secrets.";
        assert!(!contains_secrets(text));
        assert!(scan_secrets(text).is_empty());
    }

    /// 验证能识别 GitHub Token。
    #[test]
    fn test_github_token() {
        let text = "GITHUB_TOKEN=ghp_1234567890abcdefghijklmnopqrstuvwxyz123456";
        let matches = scan_secrets(text);
        assert!(matches.iter().any(|m| m.secret_type == "GitHub Token"));
    }

    /// 验证能识别 PEM 私钥块。
    #[test]
    fn test_private_key() {
        let text =
            "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0B\n-----END PRIVATE KEY-----";
        assert!(contains_secrets(text));
    }

    // ── 回归测试：新增的模式（P1-16 / 2.7） ────────────────────

    /// 验证能识别 HuggingFace Token。
    #[test]
    fn test_huggingface_token() {
        assert!(contains_secrets(
            "HF_TOKEN=hf_abcdefghijklmnopqrstuvwxyz123456789"
        ));
    }

    /// 验证能识别 Google API Key。
    #[test]
    fn test_google_api_key() {
        assert!(contains_secrets(
            "key=AIzaSyDkLmNpQrStUvWxYz1234567890abcdefg"
        ));
    }

    /// 验证能识别 GitLab 个人访问令牌。
    #[test]
    fn test_gitlab_pat() {
        assert!(contains_secrets(
            "GITLAB_TOKEN=glpat-abcdefghijklmnopqrstuvwx"
        ));
    }

    /// 验证能识别 Stripe 正式环境密钥。
    #[test]
    fn test_stripe_live_key() {
        assert!(contains_secrets(concat!(
            "sk_",
            "live_51AbCdEfGhIjKlMnOpQrStUvWxYz"
        )));
    }

    /// 验证能识别 npm Token。
    #[test]
    fn test_npm_token() {
        assert!(contains_secrets(
            "NPM_TOKEN=npm_abcdefghijklmnopqrstuvwxyz123456789"
        ));
    }

    /// 验证能识别带密码的数据库连接串。
    #[test]
    fn test_db_connection_string() {
        assert!(contains_secrets(
            "postgresql://user:secret_password@db.internal:5432/mydb"
        ));
        assert!(contains_secrets("mysql://root:pass123@localhost/mydb"));
    }
}
