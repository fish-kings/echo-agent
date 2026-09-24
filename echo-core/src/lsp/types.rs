//! 简化版 LSP 协议类型。
//!
//! 这些是 `LspClient` trait 与 LSP 工具所使用的类型。我们没有依赖完整的
//! `lsp-types` crate，而是定义了自己的简化版本，以保持核心 crate 的轻量。

use serde::{Deserialize, Serialize};

/// 文本文档中的一个位置（行号与字符偏移均从 0 开始）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Position {
    /// 行号（从 0 开始）。
    pub line: u32,
    /// 该行内的字符偏移（从 0 开始）。
    pub character: u32,
}

/// 文本文档中的一段区间。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Range {
    /// 起始位置（包含）。
    pub start: Position,
    /// 结束位置（不包含）。
    pub end: Position,
}

/// 特定文件中的一个位置。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Location {
    /// 文件 URI（例如 `file:///path/to/file.rs`）。
    pub uri: String,
    /// 文件内的区间。
    pub range: Range,
}

/// 诊断信息的严重级别。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiagnosticSeverity {
    /// 错误。
    Error,
    /// 警告。
    Warning,
    /// 提示信息。
    Information,
    /// 建议（最低级别的提示）。
    Hint,
}

/// 语言服务器报告的一条诊断信息（错误、警告等）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Diagnostic {
    /// 该诊断所适用的文档区间。
    pub range: Range,
    /// 严重级别。
    pub severity: DiagnosticSeverity,
    /// 描述该问题的人类可读消息。
    pub message: String,
    /// 诊断信息的来源（例如 "rust-analyzer"、"pyright"）。
    pub source: Option<String>,
    /// 可选的诊断代码（例如 Rust 类型不匹配的 "E0308"）。
    pub code: Option<String>,
}

/// 语言服务器返回的悬浮提示信息。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HoverInfo {
    /// Markdown 格式的内容。
    pub contents: String,
    /// 该悬浮提示所适用的可选区间。
    pub range: Option<Range>,
}

/// 补全项的种类。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompletionItemKind {
    /// 纯文本。
    Text,
    /// 方法。
    Method,
    /// 函数。
    Function,
    /// 构造函数。
    Constructor,
    /// 字段。
    Field,
    /// 变量。
    Variable,
    /// 类。
    Class,
    /// 接口。
    Interface,
    /// 模块。
    Module,
    /// 属性。
    Property,
    /// 单元类型。
    Unit,
    /// 值。
    Value,
    /// 枚举。
    Enum,
    /// 关键字。
    Keyword,
    /// 代码片段。
    Snippet,
    /// 颜色。
    Color,
    /// 文件。
    File,
    /// 引用。
    Reference,
    /// 文件夹。
    Folder,
    /// 枚举成员。
    EnumMember,
    /// 常量。
    Constant,
    /// 结构体。
    Struct,
    /// 事件。
    Event,
    /// 运算符。
    Operator,
    /// 类型参数。
    TypeParameter,
}

/// 一条补全建议。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionItem {
    /// 在补全列表中展示的标签。
    pub label: String,
    /// 补全项的种类。
    pub kind: CompletionItemKind,
    /// 附加详情（例如函数签名）。
    pub detail: Option<String>,
    /// 要插入的文本（若与 label 不同）。
    pub insert_text: Option<String>,
    /// Markdown 格式的文档说明。
    pub documentation: Option<String>,
}

/// 文档中的一次文本变更。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextChange {
    /// 要替换的区间。
    pub range: Range,
    /// 新的文本内容。
    pub text: String,
}

/// 从 `.lsp.yaml` 加载的 LSP 服务器配置。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LspServerConfig {
    /// 语言标识符（例如 "python"、"typescript"、"rust"）。
    pub language: String,
    /// 启动语言服务器所使用的命令。
    pub command: String,
    /// 命令行参数。
    #[serde(default)]
    pub args: Vec<String>,
    /// 该服务器处理的文件扩展名（例如 [".py", ".pyi"]）。
    #[serde(default)]
    pub extensions: Vec<String>,
    /// 需要设置的环境变量。
    #[serde(default)]
    pub env: std::collections::HashMap<String, String>,
    /// 初始化选项（JSON）。
    #[serde(default)]
    pub initialization_options: Option<serde_json::Value>,
    /// 放弃重启前的最大重启尝试次数。
    #[serde(default = "default_max_restarts")]
    pub max_restarts: u32,
}

/// `max_restarts` 字段的默认值。
fn default_max_restarts() -> u32 {
    3
}

/// 顶层 LSP 配置文件格式（`.lsp.yaml`）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LspConfigFile {
    /// 从语言名称到服务器配置的映射。
    pub languages: std::collections::HashMap<String, LspServerConfig>,
}

/// LSP 服务器的状态。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LspServerStatus {
    /// 语言标识符。
    pub language: String,
    /// 服务器当前是否正在运行。
    pub running: bool,
    /// 服务器是否已完成初始化。
    pub initialized: bool,
    /// 到目前为止的重启尝试次数。
    pub restart_count: u32,
    /// 最近一次的错误消息（如有）。
    pub last_error: Option<String>,
    /// 若正在运行，则为进程 ID。
    pub pid: Option<u32>,
}
