//! 搜索提供方通用的工具函数

/// URL 编码（百分号编码，遵循 RFC 3986 的未保留字符集）
pub fn urlencode(input: &str) -> String {
    input
        .bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                char::from(b).to_string()
            }
            _ => format!("%{:02X}", b),
        })
        .collect()
}

/// 按字符数安全截断（不会截断到多字节 UTF-8 字符的中间）
pub fn truncate_chars(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        s.chars().take(max).collect()
    }
}

/// 百分号解码
pub fn percent_decode(input: &str) -> String {
    let mut result = Vec::with_capacity(input.len());
    let bytes = input.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            let high = decode_hex(bytes[i + 1]);
            let low = decode_hex(bytes[i + 2]);
            if let (Some(high), Some(low)) = (high, low) {
                let byte = high.saturating_mul(16).saturating_add(low);
                result.push(byte);
                i += 3;
                continue;
            }
        } else if bytes[i] == b'+' {
            result.push(b' ');
            i += 1;
            continue;
        }
        result.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&result).to_string()
}

/// 将单个十六进制字符（ASCII）解码为其数值（0-15）；非法字符返回 `None`。
fn decode_hex(value: u8) -> Option<u8> {
    match value {
        b'0'..=b'9' => Some(value - b'0'),
        b'a'..=b'f' => Some(value - b'a' + 10),
        b'A'..=b'F' => Some(value - b'A' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 验证 URL 编码函数对空格、连字符及多字节 UTF-8 字符的处理是否正确。
    #[test]
    fn test_urlencode() {
        assert_eq!(urlencode("hello world"), "hello%20world");
        assert_eq!(urlencode("rust-lang"), "rust-lang");
        assert_eq!(urlencode("café"), "caf%C3%A9");
    }

    /// 验证截断函数在长度足够/不足时的行为，且不会破坏多字节字符的完整性。
    #[test]
    fn test_truncate_chars() {
        assert_eq!(truncate_chars("hello", 10), "hello");
        assert_eq!(truncate_chars("Hello World", 2), "He");
        // 不会截断到多字节字符的中间
        let s = "a🌍b".repeat(50);
        let truncated = truncate_chars(&s, 10);
        assert!(std::str::from_utf8(truncated.as_bytes()).is_ok());
    }

    /// 验证百分号解码函数能正确处理转义序列、`+` 转空格，以及非法/未编码的字节序列。
    #[test]
    fn test_percent_decode() {
        assert_eq!(
            percent_decode("https%3A%2F%2Fexample.com"),
            "https://example.com"
        );
        assert_eq!(percent_decode("hello+world"), "hello world");
        assert_eq!(percent_decode("%中文"), "%中文");
        assert_eq!(percent_decode("%E4%B8%AD%E6%96%87"), "中文");
    }
}
