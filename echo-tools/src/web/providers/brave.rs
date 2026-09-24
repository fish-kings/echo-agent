//! Brave 搜索 API 提供方
//!
//! 通过 Brave 搜索 API 提供高质量搜索，需要一个 API Key。
//!
//! # 获取 API Key
//!
//! 访问 <https://brave.com/search/api/> 注册并获取免费 API Key（每月 2000 次免费查询）。

use super::utils::{truncate_chars, urlencode};
use super::{SearchProvider, SearchResult};
use async_trait::async_trait;
use echo_core::error::{Result, ToolError};
use reqwest::Client;
use serde::Deserialize;
use std::time::Duration;

/// Brave 搜索 API 提供方
///
/// 通过 Brave 搜索 API 执行搜索，需要一个 API Key。
pub struct BraveSearchProvider {
    /// 用于发起 HTTP 请求的客户端。
    client: Client,
    /// Brave 搜索 API 的鉴权密钥。
    api_key: String,
}

impl BraveSearchProvider {
    /// 创建一个新的 Brave 搜索提供方
    ///
    /// - `api_key`: Brave 搜索 API Key
    pub fn new(api_key: impl Into<String>) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(15))
            .build()
            .unwrap_or_else(|_| Client::new());
        Self {
            client,
            api_key: api_key.into(),
        }
    }

    /// 从环境变量创建
    ///
    /// 读取 `BRAVE_SEARCH_API_KEY` 环境变量。
    pub fn from_env() -> Option<Self> {
        std::env::var("BRAVE_SEARCH_API_KEY").ok().map(Self::new)
    }
}

/// Brave 搜索 API 的响应结构
#[derive(Debug, Deserialize)]
struct BraveResponse {
    /// 网页搜索结果部分（可能不存在）。
    web: Option<BraveWebResults>,
}

/// Brave 搜索 API 响应中的网页结果容器。
#[derive(Debug, Deserialize)]
struct BraveWebResults {
    /// 具体的网页搜索结果列表。
    results: Option<Vec<BraveResult>>,
}

/// Brave 搜索 API 返回的单条搜索结果。
#[derive(Debug, Deserialize)]
struct BraveResult {
    /// 结果标题（可能缺失）。
    title: Option<String>,
    /// 结果链接（可能缺失）。
    url: Option<String>,
    /// 结果摘要描述（可能缺失）。
    description: Option<String>,
}

#[async_trait]
impl SearchProvider for BraveSearchProvider {
    /// 返回该搜索提供方的标识名称：`"brave"`。
    fn name(&self) -> &str {
        "brave"
    }

    /// 使用 Brave 搜索 API 执行查询，返回最多 `max_results` 条搜索结果。
    async fn search(&self, query: &str, max_results: usize) -> Result<Vec<SearchResult>> {
        let url = format!(
            "https://api.search.brave.com/res/v1/web/search?q={}&count={}",
            urlencode(query),
            max_results
        );

        let response = self
            .client
            .get(&url)
            .header("X-Subscription-Token", &self.api_key)
            .header("Accept", "application/json")
            .send()
            .await
            .map_err(|e| ToolError::ExecutionFailed {
                tool: "web_search".into(),
                message: format!("Brave Search request failed: {}", e),
            })?;

        let status = response.status();
        if status.as_u16() == 401 {
            return Err(ToolError::ExecutionFailed {
                tool: "web_search".into(),
                message: "Brave Search API Key is invalid or expired".into(),
            }
            .into());
        }
        if status.as_u16() == 429 {
            return Err(ToolError::ExecutionFailed {
                tool: "web_search".into(),
                message: "Brave Search API rate limit exceeded".into(),
            }
            .into());
        }
        if !status.is_success() {
            let body = crate::http_body::read_bounded_text(
                response,
                crate::http_body::MAX_API_RESPONSE_BYTES,
                "web_search",
                None,
            )
            .await?;
            return Err(ToolError::ExecutionFailed {
                tool: "web_search".into(),
                message: format!(
                    "Brave Search returned error ({}): {}",
                    status,
                    truncate_chars(&body, 200)
                ),
            }
            .into());
        }

        let brave_resp: BraveResponse = crate::http_body::read_bounded_json(
            response,
            crate::http_body::MAX_API_RESPONSE_BYTES,
            "web_search",
            None,
        )
        .await?;

        let results = brave_resp.web.and_then(|w| w.results).unwrap_or_default();

        Ok(results
            .into_iter()
            .take(max_results)
            .filter_map(|r| {
                let title = r.title?.trim().to_string();
                let url = r.url?.trim().to_string();
                if title.is_empty() || url.is_empty() {
                    return None;
                }
                Some(SearchResult {
                    title,
                    url,
                    snippet: r.description.unwrap_or_default().trim().to_string(),
                })
            })
            .collect())
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

    /// 验证能正确解析 Brave API 的 JSON 响应，并在 `description` 缺失时回退为空字符串。
    #[test]
    fn test_parse_brave_response() {
        let json = r#"{
            "web": {
                "results": [
                    {
                        "title": "Rust Programming Language",
                        "url": "https://www.rust-lang.org/",
                        "description": "A language empowering everyone"
                    },
                    {
                        "title": "Rust Documentation",
                        "url": "https://doc.rust-lang.org/",
                        "description": null
                    }
                ]
            }
        }"#;

        let resp: BraveResponse = serde_json::from_str(json).unwrap();
        let results: Vec<SearchResult> = resp
            .web
            .and_then(|w| w.results)
            .unwrap_or_default()
            .into_iter()
            .filter_map(|r| {
                let title = r.title?.trim().to_string();
                let url = r.url?.trim().to_string();
                if title.is_empty() || url.is_empty() {
                    return None;
                }
                Some(SearchResult {
                    title,
                    url,
                    snippet: r.description.unwrap_or_default().trim().to_string(),
                })
            })
            .collect();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].title, "Rust Programming Language");
        assert_eq!(results[0].url, "https://www.rust-lang.org/");
        assert_eq!(results[0].snippet, "A language empowering everyone");
        assert!(results[1].snippet.is_empty());
    }

    /// 验证 `results` 为空数组时能正确解析为空结果集。
    #[test]
    fn test_parse_brave_empty_response() {
        let json = r#"{"web": {"results": []}}"#;
        let resp: BraveResponse = serde_json::from_str(json).unwrap();
        let results = resp.web.and_then(|w| w.results).unwrap_or_default();
        assert!(results.is_empty());
    }
}
