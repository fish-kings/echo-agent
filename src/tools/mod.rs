/*!
工具系统 —— 定义并注册供 Agent 调用的工具。

工具是 Agent 与外部世界交互的主要方式：调用 API、读取
文件、执行代码、搜索网页等等。

# 定义工具 —— `#[tool]` 宏

推荐使用 [`#[tool]`](macro@crate::tool) 宏来定义工具：

```rust,no_run
use echo_agent::{tool, prelude::*};

#[tool(name = "add", description = "Add two numbers")]
async fn add(a: f64, b: f64) -> Result<ToolResult> {
    Ok(ToolResult::success(format!("{}", a + b)))
}
```

这会生成 `AddParams`、`AddTool`，以及一个完整的 `Tool` trait
实现，并自动生成 JSON Schema。

# 注册工具

```rust,ignore
use echo_agent::prelude::*;

# fn main() -> echo_agent::error::Result<()> {
let agent = ReactAgentBuilder::new()
    .model("qwen3-max")
    .register_tool(Arc::new(AddTool))
    .build()?;
# Ok(())
# }
```

# 内置工具

| 模块 | 工具 | Feature |
|--------|-------|---------|
| `builtin` | `ThinkTool`、`FinalAnswerTool`、记忆工具、数据工具、git、RAG、图表 | 多种 |
| `web` | `WebSearchTool`、`WebFetchTool` | `web` |
| `media` | `ImageFetchTool`、PDF/Excel/Word 工具 | `media` |
| `files` | 文件读/写/列举/删除工具 | `files` |
| `shell` | Shell 命令执行（沙箱化） | `shell` |

# 关键类型

| 类型 | 说明 |
|------|-------------|
| [`Tool`] | Trait —— 实现 `name()`、`description()`、`parameters()`、`execute()` |
| [`ToolManager`] | 注册表 —— 注册、列举、执行工具，并支持并发控制 |
| [`ToolResult`] | `ToolResult::success(...)` / `ToolResult::error(...)` |
| [`ToolExecutionConfig`] | 每个工具的超时时间、重试次数、最大并发数 |
*/

/// 内置工具（安全检查、思考等）
pub mod builtin;
/// 文件操作工具（从 echo_tools 重新导出）
#[cfg(feature = "files")]
pub mod files {
    pub use echo_tools::files::*;
}
/**
路径校验 / 限制（从 echo_tools 重新导出）

在此重新导出，使下游 crate 只需依赖 `echo_agent` 即可，
而无需为了使用 `PathValidator` 再直接依赖 `echo_tools`。
*/
pub mod security {
    pub use echo_tools::security::*;
}
pub mod permission;
/// 工具调用风险分级（从 echo_execution 重新导出）。
pub mod risk {
    pub use echo_execution::risk::*;
}
/// 补全超大体积工具输出的 artifact（从 echo_core 重新导出）。
pub mod artifact {
    pub use echo_core::tools::artifact::*;
}
/// 长时间运行命令单元（command-cell）的生命周期契约（从 echo_core 重新导出）。
pub mod cell {
    pub use echo_core::tools::cell::*;
}
/// 带版本控制的显式工具可见性控制（从 echo-core 重新导出）。
pub mod control {
    pub use echo_core::tools::control::*;
}
/// 面向返回集合类工具的游标分页契约。
pub mod pagination {
    pub use echo_core::tools::pagination::*;
}
/// Git checkpoint 原语（从 echo_tools 重新导出）。
pub mod git_checkpoint {
    pub use echo_tools::git_checkpoint::*;
}
/// 隔离的 Git worktree 原语（从 echo_tools 重新导出）。
#[cfg(feature = "git")]
pub mod git_worktree {
    pub use echo_tools::git_worktree::*;
}
/// Shell 工具（从 echo_tools 重新导出）
#[cfg(feature = "shell")]
pub mod shell {
    pub use echo_tools::shell::*;
}

/// Web 工具（从 echo_tools 重新导出）
#[cfg(feature = "web")]
pub mod web {
    pub use echo_tools::web::*;
}

/**
媒体工具（从 echo_tools 重新导出）。分组方式与
`StandardToolPack` 中的 media 条目完全一致：文档读取器
（PDF/Word/Excel）与文本工具都随 media 能力一起提供，
因此 facade 层的使用方无需直接依赖 echo_tools，即可通过
同一个模块访问它们。
*/
#[cfg(feature = "media")]
pub mod media {
    pub use echo_tools::media::*;
    pub use echo_tools::{excel, image, pdf, text, word};
}

/// 学术检索与文献管理客户端（从 echo_tools 重新导出）。
#[cfg(feature = "research")]
pub mod research {
    pub use echo_tools::research::*;
}

/// 图表工具（从 echo_tools 重新导出）。
#[cfg(feature = "chart")]
pub mod chart {
    pub use echo_tools::chart::*;
}

/// 数据处理工具（从 echo_tools 重新导出）。
#[cfg(feature = "data")]
pub mod data {
    pub use echo_tools::data::*;
}

/// 数据质量检查工具（从 echo_tools 重新导出）。
#[cfg(feature = "data")]
pub mod data_quality {
    pub use echo_tools::data_quality::*;
}

/// 数据库访问工具（从 echo_tools 重新导出）。
#[cfg(feature = "database")]
pub mod database {
    pub use echo_tools::database::*;
}

#[cfg(feature = "git")]
pub mod git {
    pub use echo_tools::git::*;
    // Worktree 工具随 git 能力一起提供（分组方式与
    // `StandardToolPack` 相同），因此 facade 层使用方可以通过
    // 同一个模块访问它们。
    pub use echo_tools::worktree_tool::{EnterWorktreeTool, ExitWorktreeTool, ListWorktreesTool};
}

/// RAG（检索增强生成）工具（从 echo_tools 重新导出）。
#[cfg(feature = "rag")]
pub mod rag {
    pub use echo_tools::rag::*;
}

/// 统计分析工具（从 echo_tools 重新导出）。
#[cfg(feature = "statistics")]
pub mod statistics {
    pub use echo_tools::statistics::*;
}

/// LSP 工具 —— 语言服务器集成
#[cfg(feature = "lsp")]
pub mod lsp;

/// 直接从 `echo_execution::tools` 重新导出。
pub mod execution {
    pub use echo_execution::tools::*;
}

pub use echo_core::tools::*;
pub use echo_execution::tools::{
    ToolBudgetMetricsSnapshot, ToolManager, ToolSchemaStats, ToolSearchTool,
};
pub use echo_tools::{StandardToolPack, register_all_tools, register_readonly_tools};

// ── 常见文件工具分类 ──────────────────────────────────────────

/// 会修改文件、因而应当要求先读取的工具集合。
pub const WRITE_TOOLS: &[&str] = &[
    "apply_patch",
    "write_file",
    "append_file",
    "create_file",
    "delete_file",
    "update_file",
    "move_file",
];

/// 读取文件内容的工具集合。
pub const READ_TOOLS: &[&str] = &["read_file", "read_artifact"];

/// 判断给定的工具名称是否属于写入类工具。
pub fn is_write_tool(name: &str) -> bool {
    WRITE_TOOLS.contains(&name)
}

/// 判断给定的工具名称是否属于读取类工具。
pub fn is_read_tool(name: &str) -> bool {
    READ_TOOLS.contains(&name)
}
