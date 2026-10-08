/*!
 * Web 工具模块
 *
 * 提供网页搜索、页面抓取与内容提取能力：
 *
 * - [`WebSearchTool`][]：在互联网上搜索信息
 * - [`WebFetchTool`][]：抓取网页内容并转换为可读文本
 * - [`WebExtractTool`][]：从 HTML 中提取结构化内容
 */

/// 从 HTML 中提取结构化内容的工具。
pub mod extract;
/// 网页抓取工具。
pub mod fetch;
/// 网页搜索提供方实现。
pub mod providers;
/// 网页搜索工具。
pub mod search;

pub use extract::WebExtractTool;
pub use fetch::WebFetchTool;
pub use search::WebSearchTool;
