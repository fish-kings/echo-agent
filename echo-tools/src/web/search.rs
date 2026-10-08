/*!
 * Web 搜索工具
 *
 * 提供 [`WebSearchTool`]，通过 [`SearchProvider`] 支持多种搜索引擎。
 * 默认使用 [`DuckDuckGoProvider`]（免费，无需 API Key）。
 *
 * # 用法
 *
 * ```text
 */

use super::providers::SearchProvider;
use super::providers::brave::BraveSearchProvider;
use super::providers::duckduckgo::DuckDuckGoProvider;
use super::providers::tavily::TavilyProvider;
use echo_core::error::{Result, ToolError};
use echo_core::tools::pagination::PageRequest;
use echo_core::tools::permission::ToolPermission;
use echo_core::tools::{Tool, ToolFailure, ToolFailureCategory, ToolParameters, ToolResult};
use futures::future::BoxFuture;
use serde_json::Value;

/// 默认每页结果数
const DEFAULT_MAX_RESULTS: usize = 5;
/// 允许的最大每页结果数
const MAX_ALLOWED_RESULTS: usize = 10;

/**
 * Web 搜索工具
 *
 * 通过不同的搜索引擎 Provider 进行网页搜索。
 * 内置 DuckDuckGo 作为免费兜底选项。
 */
pub struct WebSearchTool {
    /// 实际执行搜索的 Provider
    provider: Box<dyn SearchProvider>,
    /// 默认最大结果数
    default_max_results: usize,
}

impl WebSearchTool {
    /// 使用自定义 Provider 创建
    pub fn new(provider: Box<dyn SearchProvider>) -> Self {
        Self {
            provider,
            default_max_results: DEFAULT_MAX_RESULTS,
        }
    }

    /// 使用 DuckDuckGo 创建（免费兜底）
    pub fn with_duckduckgo() -> Self {
        Self::new(Box::new(DuckDuckGoProvider::new()))
    }

    /// 使用 Brave Search 创建（需要 API Key）
    pub fn with_brave(api_key: impl Into<String>) -> Self {
        Self::new(Box::new(BraveSearchProvider::new(api_key)))
    }

    /// 使用 Tavily 创建（需要 API Key，面向 AI 优化的搜索）
    pub fn with_tavily(api_key: impl Into<String>) -> Self {
        Self::new(Box::new(TavilyProvider::new(api_key)))
    }

    /**
     * 自动选择最佳可用 Provider
     *
     * 优先级：Tavily > Brave > DuckDuckGo
     *
     * 从环境变量读取 API Key：
     * - `TAVILY_API_KEY` → Tavily（面向 AI 优化，质量最高）
     * - `BRAVE_SEARCH_API_KEY` → Brave Search（高质量）
     * - 无 Key → DuckDuckGo（免费兜底）
     */
    pub fn auto() -> Self {
        if let Some(provider) = TavilyProvider::from_env() {
            tracing::info!("WebSearch: auto-selected Tavily Provider");
            return Self::new(Box::new(provider));
        }
        if let Some(provider) = BraveSearchProvider::from_env() {
            tracing::info!("WebSearch: auto-selected Brave Provider");
            return Self::new(Box::new(provider));
        }
        tracing::info!("WebSearch: no API key, falling back to DuckDuckGo Provider");
        Self::with_duckduckgo()
    }

    /// 设置默认最大结果数
    pub fn with_max_results(mut self, n: usize) -> Self {
        self.default_max_results = n.clamp(1, MAX_ALLOWED_RESULTS);
        self
    }

    /// 获取当前 Provider 名称
    pub fn provider_name(&self) -> &str {
        self.provider.name()
    }
}

impl Tool for WebSearchTool {
    /// 工具名称。
    fn name(&self) -> &str {
        "web_search"
    }

    /// 所需权限：网络访问。
    fn permissions(&self) -> Vec<ToolPermission> {
        vec![ToolPermission::Network]
    }

    /// 工具描述。
    fn description(&self) -> &str {
        "Search for information on the internet. Returns search result titles, links, and snippets. \
         Results use cursor pagination; repeat the same query and limit with page.next_cursor."
    }

    /// 参数的 JSON Schema。
    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search keywords"
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_ALLOWED_RESULTS,
                    "description": format!("Results per page (default {}, max {})", DEFAULT_MAX_RESULTS, MAX_ALLOWED_RESULTS)
                },
                "cursor": {
                    "type": "string",
                    "description": "Cursor from the previous page"
                }
            },
            "required": ["query"]
        })
    }

    /// 执行搜索并返回分页的 JSON 结果；Provider 失败时返回可重试的瞬时失败。
    fn execute(&self, parameters: ToolParameters) -> BoxFuture<'_, Result<ToolResult>> {
        Box::pin(async move {
            let query = parameters
                .get("query")
                .and_then(|v| v.as_str())
                .ok_or_else(|| ToolError::MissingParameter("query".to_string()))?;

            if query.trim().is_empty() {
                return Ok(ToolResult::failure(
                    ToolFailureCategory::InvalidArguments,
                    "Search query cannot be empty",
                ));
            }

            let page_request = match PageRequest::from_parameters(
                &parameters,
                self.default_max_results,
                MAX_ALLOWED_RESULTS,
            ) {
                Ok(request) => request,
                Err(error) => return Ok(ToolResult::invalid_arguments(error.to_string())),
            };

            tracing::info!(
                "WebSearch: query='{}', page_limit={}, provider={}",
                query,
                page_request.limit,
                self.provider.name()
            );
            match self.provider.search(query, MAX_ALLOWED_RESULTS).await {
                Ok(results) => {
                    let query_identity = serde_json::json!({
                        "query": query,
                        "provider": self.provider.name(),
                    });
                    let (page, page_info) = match page_request.paginate(results, &query_identity) {
                        Ok(page) => page,
                        Err(error) => {
                            return Ok(ToolResult::invalid_arguments(error.to_string()));
                        }
                    };
                    let data =
                        serde_json::to_value(page).map_err(|error| ToolError::ExecutionFailed {
                            tool: "web_search".to_string(),
                            message: format!("Failed to serialize search results: {error}"),
                        })?;
                    let mut result = ToolResult::success_json(data);
                    page_info.apply_to(&mut result);
                    Ok(result)
                }
                Err(error) => Ok(ToolResult::error(format!(
                    "Search provider '{}' failed: {error}",
                    self.provider.name()
                ))
                .with_failure(
                    ToolFailure::new(ToolFailureCategory::Transient)
                        .retryable()
                        .with_retry_after(1_000),
                )),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::web::providers::SearchResult;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};

    /// 空结果序列化为空数组。
    #[test]
    fn test_empty_results_json() {
        let results: Vec<SearchResult> = vec![];
        let json = serde_json::to_value(&results).unwrap();
        assert!(json.as_array().unwrap().is_empty());
    }

    /// 结果序列化后的 JSON 结构。
    #[test]
    fn test_results_json_structure() {
        let results = vec![
            SearchResult {
                title: "Rust".into(),
                url: "https://rust-lang.org".into(),
                snippet: "A programming language".into(),
            },
            SearchResult {
                title: "Cargo".into(),
                url: "https://doc.rust-lang.org/cargo".into(),
                snippet: String::new(),
            },
        ];

        let json = serde_json::to_value(&results).unwrap();
        let arr = json.as_array().unwrap();
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[0]["title"], "Rust");
        assert_eq!(arr[0]["url"], "https://rust-lang.org");
        assert_eq!(arr[0]["snippet"], "A programming language");
        assert_eq!(arr[1]["title"], "Cargo");
        assert_eq!(arr[1]["snippet"], "");
    }

    /// 模拟 Provider：前 N 次失败，之后成功
    struct FailNTimesProvider {
        /// 需要失败的次数
        fail_count: u32,
        /// 共享的调用计数器
        call_count: Arc<AtomicU32>,
    }

    impl FailNTimesProvider {
        /// 创建 Provider，并返回其调用计数器。
        fn new(fail_count: u32) -> (Self, Arc<AtomicU32>) {
            let counter = Arc::new(AtomicU32::new(0));
            (
                Self {
                    fail_count,
                    call_count: counter.clone(),
                },
                counter,
            )
        }
    }

    #[async_trait::async_trait]
    impl SearchProvider for FailNTimesProvider {
        /// Provider 名称。
        fn name(&self) -> &str {
            "mock-fail-n"
        }

        /// 前 N 次返回失败，之后返回成功结果。
        async fn search(&self, _query: &str, _max_results: usize) -> Result<Vec<SearchResult>> {
            let call = self.call_count.fetch_add(1, Ordering::SeqCst);
            if call < self.fail_count {
                Err(echo_core::error::ToolError::ExecutionFailed {
                    tool: "mock".to_string(),
                    message: format!("Simulated failure #{}", call + 1),
                }
                .into())
            } else {
                Ok(vec![SearchResult {
                    title: "Success".into(),
                    url: "https://example.com".into(),
                    snippet: format!("Succeeded on attempt {}", call + 1),
                }])
            }
        }
    }

    /// 始终失败的 Provider
    struct AlwaysFailProvider;

    #[async_trait::async_trait]
    impl SearchProvider for AlwaysFailProvider {
        /// Provider 名称。
        fn name(&self) -> &str {
            "mock-always-fail"
        }

        /// 总是返回失败。
        async fn search(&self, _query: &str, _max_results: usize) -> Result<Vec<SearchResult>> {
            Err(echo_core::error::ToolError::ExecutionFailed {
                tool: "mock".to_string(),
                message: "Always fails".to_string(),
            }
            .into())
        }
    }

    /// 瞬时失败直接交由中央重试策略处理，工具内部不重试。
    #[tokio::test]
    async fn transient_failure_is_returned_to_the_central_retry_policy() {
        let (provider, counter) = FailNTimesProvider::new(2);
        let tool = WebSearchTool::new(Box::new(provider));

        let mut params = std::collections::HashMap::new();
        params.insert("query".to_string(), serde_json::json!("test"));

        let result = tool.execute(params).await.unwrap();
        assert!(!result.success);
        assert_eq!(counter.load(Ordering::SeqCst), 1);
        assert!(
            result
                .failure
                .as_ref()
                .is_some_and(ToolFailure::allows_automatic_retry)
        );
    }

    /// Provider 失败被归类为瞬时失败。
    #[tokio::test]
    async fn provider_failure_is_classified_as_transient() {
        let tool = WebSearchTool::new(Box::new(AlwaysFailProvider));

        let mut params = std::collections::HashMap::new();
        params.insert("query".to_string(), serde_json::json!("test"));

        let result = tool.execute(params).await.unwrap();
        assert!(!result.success);
        assert_eq!(
            result.failure.map(|failure| failure.category),
            Some(ToolFailureCategory::Transient)
        );
    }
}
