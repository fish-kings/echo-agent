/*!
 * 搜索引擎 Provider 抽象
 *
 * 定义 [`SearchProvider`] trait 和 [`SearchResult`] 类型，
 * 支持多种搜索引擎后端（DuckDuckGo、Brave、Tavily 等）。
 */

/// Brave 搜索 Provider
pub mod brave;
/// DuckDuckGo 搜索 Provider
pub mod duckduckgo;
/// Tavily 搜索 Provider
pub mod tavily;
/// Provider 共用的工具函数
pub mod utils;

use async_trait::async_trait;
use echo_core::error::Result;
use serde::{Deserialize, Serialize};

/// 搜索结果条目
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// 标题
    pub title: String,
    /// 结果链接 URL
    pub url: String,
    /// 片段 / 摘要
    pub snippet: String,
}

/**
 * 搜索引擎 Provider trait
 *
 * 所有搜索引擎后端都必须实现此 trait。
 * 框架提供 [`duckduckgo::DuckDuckGoProvider`] 作为免费兜底。
 *
 * # 扩展
 *
 * 要实现 `BraveSearchProvider` 或 `TavilyProvider`，只需实现此 trait，
 * 并通过 `WebSearchTool::new(provider)` 传入即可。
 */
#[async_trait]
pub trait SearchProvider: Send + Sync {
    /// Provider 名称
    fn name(&self) -> &str;

    /**
     * 执行搜索查询
     *
     * - `query`：搜索关键词
     * - `max_results`：返回的最大结果数
     */
    async fn search(&self, query: &str, max_results: usize) -> Result<Vec<SearchResult>>;
}
