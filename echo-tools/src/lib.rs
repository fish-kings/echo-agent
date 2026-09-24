//! Echo Tools —— echo-agent 框架的领域工具集
//!
//! 提供按 feature 门控的工具模块，可通过 [`register_all_tools`] 注册到
//! 任意 [`ToolRegistrar`](echo_core::tools::ToolRegistrar) 中。
//!
//! # Features
//!
//! | Feature   | 工具                                     |
//! |-----------|-------------------------------------------|
//! | `artifact`| 对溢出工具输出做校验、分页读取 |
//! | `files`   | Artifact 读取器与文件系统工具      |
//! | `web`     | `web`（WebFetchTool、WebSearchTool、WebExtractTool、各搜索提供方） |
//! | `chart`   | `chart`（GenerateChartTool）               |
//! | `data`    | `data`（11 个数据分析工具）、`data_quality`（3 个数据质量工具） |
//! | `statistics`| `statistics`（`exploratory_statistics`；正式的统计推断走沙箱化的 `run_code`） |
//! | `database`| `database`（SqlQueryTool 等）              |
//! | `media`   | `excel`、`image`、`pdf`、`word`、`text`、`media` |
//! | `git`     | `git`（6 个 git CLI 工具）                   |
//! | `rag`     | `rag`（RagIndexTool、RagSearchTool 等）    |
//! | `full`    | 以上全部                          |

/// 沙箱化的内联或持久化脚本执行工具（`run_code`）。归入 `shell` feature 门控，
/// 因为它与 shell 属于相同的执行/风险等级（都会启动一个解释器）。
#[cfg(feature = "shell")]
#[cfg_attr(docsrs, doc(cfg(feature = "shell")))]
pub mod code;
#[cfg(any(feature = "artifact", feature = "files"))]
#[cfg_attr(docsrs, doc(cfg(any(feature = "artifact", feature = "files"))))]
pub mod files;
pub mod git_checkpoint;
#[cfg(any(feature = "web", feature = "media", feature = "research"))]
mod http_body;
#[cfg(feature = "git")]
mod process;
pub mod security;
#[cfg(feature = "shell")]
#[cfg_attr(docsrs, doc(cfg(feature = "shell")))]
pub mod shell;
#[cfg(any(feature = "files", feature = "shell"))]
pub mod skills;

#[cfg(feature = "chart")]
#[cfg_attr(docsrs, doc(cfg(feature = "chart")))]
pub mod chart;
#[cfg(feature = "data")]
#[cfg_attr(docsrs, doc(cfg(feature = "data")))]
pub mod data;
#[cfg(feature = "data")]
#[cfg_attr(docsrs, doc(cfg(feature = "data")))]
pub mod data_quality;
#[cfg(feature = "database")]
#[cfg_attr(docsrs, doc(cfg(feature = "database")))]
pub mod database;
#[cfg(any(feature = "files", feature = "git"))]
mod diff_pagination;
#[cfg(feature = "media")]
#[cfg_attr(docsrs, doc(cfg(feature = "media")))]
pub mod excel;
#[cfg(feature = "git")]
#[cfg_attr(docsrs, doc(cfg(feature = "git")))]
pub mod git;
#[cfg(feature = "git")]
#[cfg_attr(docsrs, doc(cfg(feature = "git")))]
pub mod git_worktree;
#[cfg(feature = "media")]
#[cfg_attr(docsrs, doc(cfg(feature = "media")))]
pub mod image;
#[cfg(feature = "media")]
#[cfg_attr(docsrs, doc(cfg(feature = "media")))]
pub mod media;
#[cfg(feature = "media")]
#[cfg_attr(docsrs, doc(cfg(feature = "media")))]
pub mod pdf;
#[cfg(feature = "rag")]
#[cfg_attr(docsrs, doc(cfg(feature = "rag")))]
pub mod rag;
#[cfg(feature = "research")]
#[cfg_attr(docsrs, doc(cfg(feature = "research")))]
pub mod research;
#[cfg(feature = "statistics")]
#[cfg_attr(docsrs, doc(cfg(feature = "statistics")))]
pub mod statistics;
#[cfg(feature = "media")]
#[cfg_attr(docsrs, doc(cfg(feature = "media")))]
pub mod text;
#[cfg(feature = "web")]
#[cfg_attr(docsrs, doc(cfg(feature = "web")))]
pub mod web;
#[cfg(feature = "media")]
#[cfg_attr(docsrs, doc(cfg(feature = "media")))]
pub mod word;
#[cfg(feature = "git")]
#[cfg_attr(docsrs, doc(cfg(feature = "git")))]
pub mod worktree_tool;

mod registry;
pub use registry::{
    StandardToolPack, register_all_tools, register_all_tools_with_cells, register_readonly_tools,
};
