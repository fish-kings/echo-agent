/*!
 * 网页抓取工具
 *
 * 提供 [`WebFetchTool`]，抓取 URL 内容并转换为可读文本。
 * 支持 HTML → 纯文本转换，便于 LLM 使用。
 */

use echo_core::error::{Result, ToolError};
use echo_core::tools::artifact::{ToolOutputArtifactIdentity, persist_tool_output};
use echo_core::tools::permission::ToolPermission;
use echo_core::tools::{Tool, ToolParameters, ToolResult};
use futures::StreamExt;
use futures::future::BoxFuture;
use serde_json::Value;
use std::sync::OnceLock;
use std::time::Duration;

/// 默认最大内容长度（字符数）
const DEFAULT_MAX_LENGTH: usize = 50_000;
/// 默认请求超时（秒）
const DEFAULT_TIMEOUT_SECS: u64 = 30;
/// 默认 HTML 转文本的行宽
const DEFAULT_TEXT_WIDTH: usize = 120;

/**
 * 原始 HTTP 响应体的硬性字节上限（10 MB）。
 * 超过该大小的响应会在读取前被拒绝，防止过大或恶意负载耗尽内存。
 */
const MAX_BODY_BYTES: u64 = 10 * 1024 * 1024;

/// 回退路径中用于剥离 HTML 标签的正则（惰性初始化）
static HTML_TAG_RE: OnceLock<Option<regex::Regex>> = OnceLock::new();
/// 回退路径中用于折叠空白的正则（惰性初始化）
static WHITESPACE_RE: OnceLock<Option<regex::Regex>> = OnceLock::new();

/**
 * 网页抓取工具
 *
 * 从指定 URL 抓取内容，并将 HTML 转换为可读文本。
 */
pub struct WebFetchTool {
    /// 返回内容的最大长度（字符数）
    max_content_length: usize,
    /// HTML 转文本时的行宽
    text_width: usize,
}

impl WebFetchTool {
    /// 创建新的 WebFetchTool
    pub fn new() -> Self {
        Self {
            max_content_length: DEFAULT_MAX_LENGTH,
            text_width: DEFAULT_TEXT_WIDTH,
        }
    }

    /// 设置最大内容长度（字符数）
    pub fn with_max_content_length(mut self, n: usize) -> Self {
        self.max_content_length = n;
        self
    }

    /// 设置 HTML 转文本的行宽
    pub fn with_text_width(mut self, width: usize) -> Self {
        self.text_width = width;
        self
    }

    /// 判断该 Content-Type 是否需要 HTML→文本转换
    fn needs_html_conversion(content_type: &str) -> bool {
        content_type.contains("text/html") || content_type.contains("application/xhtml")
    }

    /// 将 HTML 转换为可读文本
    fn html_to_text(&self, html: &str) -> String {
        match html2text::from_read(html.as_bytes(), self.text_width) {
            Ok(text) => text,
            Err(e) => {
                tracing::warn!(
                    "HTML to text conversion failed ({}), falling back to simple tag stripping: {}",
                    self.text_width,
                    e
                );
                // 回退：用简单正则剥离 HTML 标签，再折叠空白
                let stripped = HTML_TAG_RE
                    .get_or_init(|| regex::Regex::new(r"<[^>]*>").ok())
                    .as_ref()
                    .map_or_else(
                        || html.to_string(),
                        |re| re.replace_all(html, " ").to_string(),
                    );
                WHITESPACE_RE
                    .get_or_init(|| regex::Regex::new(r"[ \t\r\n]+").ok())
                    .as_ref()
                    .map_or_else(
                        || stripped.trim().to_string(),
                        |re| re.replace_all(&stripped, "\n").trim().to_string(),
                    )
            }
        }
    }

    /// 按字符数截断内容（安全处理多字节 UTF-8）
    fn truncate_content(content: &str, max_len: usize) -> String {
        if content.chars().count() <= max_len {
            content.to_string()
        } else {
            let truncated: String = content.chars().take(max_len).collect();
            format!("{}\n\n[... content truncated ...]", truncated)
        }
    }
}

impl Default for WebFetchTool {
    /// 等价于 `new()`
    fn default() -> Self {
        Self::new()
    }
}

impl Tool for WebFetchTool {
    /// 工具名称
    fn name(&self) -> &str {
        "web_fetch"
    }

    /// 工具描述
    fn description(&self) -> &str {
        "Fetch a URL as readable text; large content is stored as a recoverable artifact."
    }

    /// 工具所需权限
    fn permissions(&self) -> Vec<ToolPermission> {
        vec![ToolPermission::Network]
    }

    /// 参数 JSON Schema
    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL to fetch"
                },
                "max_length": {
                    "type": "integer",
                    "description": "Inline characters (default 50000)"
                }
            },
            "required": ["url"]
        })
    }

    /// 抓取 URL 内容并按类型转换为文本
    fn execute_with_context<'a>(
        &'a self,
        parameters: ToolParameters,
        ctx: &'a echo_core::tools::ToolContext,
    ) -> BoxFuture<'a, Result<ToolResult>> {
        Box::pin(async move {
            let url = parameters
                .get("url")
                .and_then(|v| v.as_str())
                .ok_or_else(|| ToolError::MissingParameter("url".to_string()))?;

            if url.trim().is_empty() {
                return Ok(ToolResult::error("URL cannot be empty"));
            }

            // 基本的 URL 格式校验
            if !url.starts_with("http://") && !url.starts_with("https://") {
                return Ok(ToolResult::error("URL must start with http:// or https://"));
            }

            let max_length = match parameters.get("max_length") {
                Some(value) => value
                    .as_u64()
                    .and_then(|value| usize::try_from(value).ok())
                    .filter(|value| *value > 0)
                    .ok_or_else(|| ToolError::InvalidParameter {
                        name: "max_length".to_string(),
                        message: "must be a positive integer supported by this platform"
                            .to_string(),
                    })?,
                None => self.max_content_length,
            };

            /*
             * SSRF 防护：解析 + 校验 + 连接固定 IP，
             * 消除 validate_url+client.get 方式遗留的 DNS 重绑定 TOCTOU 窗口。
             */
            let timeout = Duration::from_secs(DEFAULT_TIMEOUT_SECS);
            let response = match crate::security::local_http_get(url, timeout, 5).await {
                Ok(r) => r,
                Err(e) => {
                    return Ok(ToolResult::error(format!("Request failed: {}", e)));
                }
            };

            tracing::info!("WebFetch: url='{}', max_length={}", url, max_length);

            let status = response.status();
            if !status.is_success() {
                return Ok(ToolResult::error(format!(
                    "HTTP request failed, status code: {}",
                    status
                )));
            }

            let content_type = response
                .headers()
                .get("content-type")
                .and_then(|v| v.to_str().ok())
                .unwrap_or("text/html")
                .to_string();

            // 通过 Content-Length 头提前拒绝过大的响应
            if let Some(content_length) = response.content_length()
                && content_length > MAX_BODY_BYTES
            {
                return Ok(ToolResult::error(format!(
                    "Response body too large: Content-Length {} exceeds limit {} ({} MB)",
                    content_length,
                    MAX_BODY_BYTES,
                    MAX_BODY_BYTES / (1024 * 1024),
                )));
            }

            // 以硬性字节上限流式读取响应体，防止内存耗尽
            let mut body_bytes = Vec::new();
            let mut byte_stream = response.bytes_stream();
            while let Some(chunk_result) = byte_stream.next().await {
                let chunk = match chunk_result {
                    Ok(c) => c,
                    Err(e) => {
                        return Ok(ToolResult::error(format!(
                            "Failed to read response body: {}",
                            e
                        )));
                    }
                };
                if body_bytes.len().saturating_add(chunk.len())
                    > usize::try_from(MAX_BODY_BYTES).unwrap_or(usize::MAX)
                {
                    return Ok(ToolResult::error(format!(
                        "Response body exceeds limit of {} bytes ({} MB)",
                        MAX_BODY_BYTES,
                        MAX_BODY_BYTES / (1024 * 1024),
                    )));
                }
                body_bytes.extend_from_slice(&chunk);
            }

            let body = String::from_utf8(body_bytes).unwrap_or_else(|e| {
                // 非 UTF-8：回退到有损转换
                String::from_utf8_lossy(e.as_bytes()).into_owned()
            });

            // 按内容类型处理：仅转换 HTML/XHTML
            let content = if Self::needs_html_conversion(&content_type) {
                self.html_to_text(&body)
            } else {
                // text/plain、application/json 等直接返回原始内容
                body
            };

            let full_output = format!(
                "URL: {}\nStatus: {}\nContent-Type: {}\n\n{}",
                url, status, content_type, content
            );
            let oversized = content.chars().count() > max_length;
            let mut artifact_error = None;
            if oversized && let Some(mut config) = ctx.output_artifacts.clone() {
                config.threshold_bytes = 1;
                let artifact = match persist_tool_output(
                    config,
                    ToolOutputArtifactIdentity::from_context(ctx, "web_fetch"),
                    &full_output,
                ) {
                    Ok(artifact) => artifact,
                    Err(error) => {
                        tracing::warn!(%error, "web_fetch output artifact write failed");
                        artifact_error = Some(error.to_string());
                        None
                    }
                };
                if let Some(artifact) = artifact {
                    let preview = Self::truncate_content(&content, max_length);
                    let output = format!(
                        "URL: {}\nStatus: {}\nContent-Type: {}\n\n{}\n\nFull content artifact: {} (sha256 {}). Use read_artifact to continue.",
                        url,
                        status,
                        content_type,
                        preview,
                        artifact.path.display(),
                        artifact.sha256
                    );
                    let mut result = ToolResult::success(output)
                        .with_truncated(true)
                        .with_mime_type(content_type)
                        .with_meta("url", url)
                        .with_meta("output_handling", "spilled")
                        .with_meta("original_bytes", full_output.len().to_string());
                    result.metadata.insert(
                        "returned_bytes".to_string(),
                        result.output.len().to_string(),
                    );
                    result.artifact = Some(artifact);
                    return Ok(result);
                }
            }

            let mut result = ToolResult::success(full_output)
                .with_mime_type(content_type)
                .with_meta("url", url);
            if let Some(error) = artifact_error {
                result
                    .metadata
                    .insert("artifact_status".to_string(), "write_failed".to_string());
                result.metadata.insert("artifact_error".to_string(), error);
            }
            Ok(result)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 测试 Content-Type 的 HTML 转换判断
    #[test]
    fn test_needs_html_conversion() {
        assert!(WebFetchTool::needs_html_conversion(
            "text/html; charset=utf-8"
        ));
        assert!(WebFetchTool::needs_html_conversion("application/xhtml+xml"));
        // text/plain 不应做 HTML 转换
        assert!(!WebFetchTool::needs_html_conversion("text/plain"));
        assert!(!WebFetchTool::needs_html_conversion("application/json"));
        assert!(!WebFetchTool::needs_html_conversion("image/png"));
    }

    /// 测试短内容不截断
    #[test]
    fn test_truncate_content_short() {
        let content = "Hello world";
        let truncated = WebFetchTool::truncate_content(content, 100);
        assert_eq!(truncated, content);
    }

    /// 测试长 ASCII 内容截断
    #[test]
    fn test_truncate_content_long_ascii() {
        let content = "a".repeat(200);
        let truncated = WebFetchTool::truncate_content(&content, 100);
        assert!(truncated.contains("truncated"));
        assert!(truncated.starts_with(&"a".repeat(100)));
    }

    /// 测试截断的多字节安全性
    #[test]
    fn test_truncate_content_multibyte_safe() {
        // 多字节字符截断不应 panic
        let content = "HelloWorld".repeat(50); // 200 个字符，600 字节？
        let truncated = WebFetchTool::truncate_content(&content, 10);
        assert!(truncated.contains("truncated"));
        assert!(truncated.starts_with("HelloWorld"));
    }

    /// 测试混合字符内容截断
    #[test]
    fn test_truncate_content_mixed() {
        // ASCII 与 emoji 混合
        let content = "Hello 🌍 World 🚀 Rust 🦀".repeat(20);
        let truncated = WebFetchTool::truncate_content(&content, 10);
        assert!(truncated.contains("truncated"));
        // 确保截断结果仍是合法 UTF-8
        assert!(std::str::from_utf8(truncated.as_bytes()).is_ok());
    }

    /// 测试 HTML 转文本
    #[test]
    fn test_html_to_text() {
        let tool = WebFetchTool::new();
        let html = "<html><body><h1>Title</h1><p>Hello world</p></body></html>";
        let text = tool.html_to_text(html);
        assert!(text.contains("Title"));
        assert!(text.contains("Hello"));
    }
}
