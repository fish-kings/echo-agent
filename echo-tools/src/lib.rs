/*!
 * Echo Tools —— echo-agent 框架的领域工具集
 *
 * 提供按 feature 门控的工具模块，可通过 [`register_all_tools`] 注册到
 * 任意 [`ToolRegistrar`](echo_core::tools::ToolRegistrar) 中。
 *
 * # 特性（Features）
 *
 * | Feature   | 工具                                     |
 * |-----------|-------------------------------------------|
 * | `artifact`| 对溢出工具输出做校验、分页读取 |
 * | `files`   | Artifact 读取器与文件系统工具      |
 * | `web`     | `web`（WebFetchTool、WebSearchTool、WebExtractTool、各搜索提供方） |
 * | `chart`   | `chart`（GenerateChartTool）               |
 * | `data`    | `data`（11 个数据分析工具）、`data_quality`（3 个数据质量工具） |
 * | `statistics`| `statistics`（`exploratory_statistics`；正式的统计推断走沙箱化的 `run_code`） |
 * | `database`| `database`（SqlQueryTool 等）              |
 * | `media`   | `excel`、`image`、`pdf`、`word`、`text`、`media` |
 * | `git`     | `git`（6 个 git CLI 工具）                   |
 * | `rag`     | `rag`（RagIndexTool、RagSearchTool 等）    |
 * | `full`    | 以上全部                          |
 */

/**
 * 沙箱化的内联或持久化脚本执行工具（`run_code`）。归入 `shell` feature 门控，
 * 因为它与 shell 属于相同的执行/风险等级（都会启动一个解释器）。
 */
#[cfg(feature = "shell")]
#[cfg_attr(docsrs, doc(cfg(feature = "shell")))]
pub mod code;
/// 文件系统工具与 Artifact 读取器。
#[cfg(any(feature = "artifact", feature = "files"))]
#[cfg_attr(docsrs, doc(cfg(any(feature = "artifact", feature = "files"))))]
pub mod files;
/// 基于 Git 的文件检查点。
pub mod git_checkpoint;
// HTTP 响应体读取辅助。
#[cfg(any(feature = "web", feature = "media", feature = "research"))]
mod http_body;
// 外部进程执行辅助。
#[cfg(feature = "git")]
mod process;
/// 工具共用的安全校验辅助。
pub mod security;
/// 受控的本地命令执行工具。
#[cfg(feature = "shell")]
#[cfg_attr(docsrs, doc(cfg(feature = "shell")))]
pub mod shell;
/// 内置技能（文件、shell 等）。
#[cfg(any(feature = "files", feature = "shell"))]
pub mod skills;

/// 图表生成工具。
#[cfg(feature = "chart")]
#[cfg_attr(docsrs, doc(cfg(feature = "chart")))]
pub mod chart;
/// 数据分析工具。
#[cfg(feature = "data")]
#[cfg_attr(docsrs, doc(cfg(feature = "data")))]
pub mod data;
/// 数据质量检查工具。
#[cfg(feature = "data")]
#[cfg_attr(docsrs, doc(cfg(feature = "data")))]
pub mod data_quality;
/// 数据库查询工具。
#[cfg(feature = "database")]
#[cfg_attr(docsrs, doc(cfg(feature = "database")))]
pub mod database;
// unified diff 分页辅助。
#[cfg(any(feature = "files", feature = "git"))]
mod diff_pagination;
/// Excel 读写工具。
#[cfg(feature = "media")]
#[cfg_attr(docsrs, doc(cfg(feature = "media")))]
pub mod excel;
/// Git CLI 工具。
#[cfg(feature = "git")]
#[cfg_attr(docsrs, doc(cfg(feature = "git")))]
pub mod git;
/// Git worktree 管理。
#[cfg(feature = "git")]
#[cfg_attr(docsrs, doc(cfg(feature = "git")))]
pub mod git_worktree;
/// 本地图片查看工具。
#[cfg(feature = "media")]
#[cfg_attr(docsrs, doc(cfg(feature = "media")))]
pub mod image;
/// 媒体处理工具。
#[cfg(feature = "media")]
#[cfg_attr(docsrs, doc(cfg(feature = "media")))]
pub mod media;
/// PDF 处理工具。
#[cfg(feature = "media")]
#[cfg_attr(docsrs, doc(cfg(feature = "media")))]
pub mod pdf;
/// RAG 索引与检索工具。
#[cfg(feature = "rag")]
#[cfg_attr(docsrs, doc(cfg(feature = "rag")))]
pub mod rag;
/// 科研与论文检索工具。
#[cfg(feature = "research")]
#[cfg_attr(docsrs, doc(cfg(feature = "research")))]
pub mod research;
/// 探索性统计工具。
#[cfg(feature = "statistics")]
#[cfg_attr(docsrs, doc(cfg(feature = "statistics")))]
pub mod statistics;
/// 文本处理工具。
#[cfg(feature = "media")]
#[cfg_attr(docsrs, doc(cfg(feature = "media")))]
pub mod text;
/// Web 搜索、抓取与提取工具。
#[cfg(feature = "web")]
#[cfg_attr(docsrs, doc(cfg(feature = "web")))]
pub mod web;
/// Word 文档处理工具。
#[cfg(feature = "media")]
#[cfg_attr(docsrs, doc(cfg(feature = "media")))]
pub mod word;
/// Git worktree 工具。
#[cfg(feature = "git")]
#[cfg_attr(docsrs, doc(cfg(feature = "git")))]
pub mod worktree_tool;

// 标准工具包与注册入口。
mod registry;
pub use registry::{
    StandardToolPack, register_all_tools, register_all_tools_with_cells, register_readonly_tools,
};
