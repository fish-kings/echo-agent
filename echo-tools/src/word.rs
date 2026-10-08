/*!
 * Word 文档处理工具
 *
 * 提供 Word 文档读取能力，支持：
 * - .docx 格式
 * - 提取文本内容
 * - 获取文档结构信息
 */

use futures::future::BoxFuture;
use serde_json::Value;

use crate::security::{ResourceLimits, SecurityConfig};
use echo_core::error::{Result, ToolError};
use echo_core::tools::permission::ToolPermission;
use echo_core::tools::{Tool, ToolParameters, ToolResult};

/// 错误信息中使用的工具名称
const TOOL_NAME: &str = "word_tools";

/// Word 文档读取工具：读取 .docx 并提取文本内容
pub struct WordReadTool;

impl Tool for WordReadTool {
    /// 返回工具名称
    fn name(&self) -> &str {
        "read_word"
    }

    /// 返回工具所需权限
    fn permissions(&self) -> Vec<ToolPermission> {
        vec![ToolPermission::Read]
    }

    /// 返回工具描述
    fn description(&self) -> &str {
        "Read Word document (.docx), extract text content."
    }

    /// 返回参数的 JSON Schema
    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute path to the Word document"
                },
                "include_formatting": {
                    "type": "boolean",
                    "description": "Whether to include formatting info (default false)"
                }
            },
            "required": ["file_path"]
        })
    }

    /// 执行工具逻辑
    fn execute(&self, parameters: ToolParameters) -> BoxFuture<'_, Result<ToolResult>> {
        Box::pin(async move {
            let file_path = parameters
                .get("file_path")
                .and_then(|v| v.as_str())
                .ok_or_else(|| ToolError::MissingParameter("file_path".to_string()))?;

            let include_formatting = parameters
                .get("include_formatting")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);

            let security = SecurityConfig::global();
            let path = security.validate_file(file_path)?;

            // 读取文件内容
            let bytes = std::fs::read(&path).map_err(|e| ToolError::ExecutionFailed {
                tool: TOOL_NAME.to_string(),
                message: format!("Failed to read file: {}", e),
            })?;

            // 解析 docx 文件
            let docx = docx_rs::read_docx(&bytes).map_err(|e| ToolError::ExecutionFailed {
                tool: TOOL_NAME.to_string(),
                message: format!("Failed to read Word document: {:?}", e),
            })?;

            // 提取文本内容
            let content = extract_text_from_docx(&docx, include_formatting, &security.limits);

            Ok(ToolResult::success(content))
        })
    }
}

/// Word 文档信息工具：统计段落数、字数等基础信息
pub struct WordInfoTool;

impl Tool for WordInfoTool {
    /// 返回工具名称
    fn name(&self) -> &str {
        "word_info"
    }

    /// 返回工具所需权限
    fn permissions(&self) -> Vec<ToolPermission> {
        vec![ToolPermission::Read]
    }

    /// 返回工具描述
    fn description(&self) -> &str {
        "Get basic info about a Word document: paragraph count, word count estimate, etc."
    }

    /// 返回参数的 JSON Schema
    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute path to the Word document"
                }
            },
            "required": ["file_path"]
        })
    }

    /// 执行工具逻辑
    fn execute(&self, parameters: ToolParameters) -> BoxFuture<'_, Result<ToolResult>> {
        Box::pin(async move {
            let file_path = parameters
                .get("file_path")
                .and_then(|v| v.as_str())
                .ok_or_else(|| ToolError::MissingParameter("file_path".to_string()))?;

            let security = SecurityConfig::global();
            let path = security.validate_file(file_path)?;

            // 读取文件内容
            let bytes = std::fs::read(&path).map_err(|e| ToolError::ExecutionFailed {
                tool: TOOL_NAME.to_string(),
                message: format!("Failed to read file: {}", e),
            })?;

            // 解析 docx 文件
            let docx = docx_rs::read_docx(&bytes).map_err(|e| ToolError::ExecutionFailed {
                tool: TOOL_NAME.to_string(),
                message: format!("Failed to read Word document: {:?}", e),
            })?;

            // 统计信息
            let mut info = Vec::new();
            info.push(format!("File: {}", file_path));

            // 统计段落数
            let document = &docx.document;
            let mut paragraph_count = 0;

            for child in &document.children {
                if let docx_rs::DocumentChild::Paragraph(_) = child {
                    paragraph_count += 1;
                }
            }

            // 提取全部文本并统计字数/字符数
            let all_text = extract_text_from_docx(&docx, false, &security.limits);
            let total_chars = all_text.chars().count();
            let total_words = all_text.split_whitespace().count();

            info.push(format!("Paragraph count: {}", paragraph_count));
            info.push(format!("Character count: {}", total_chars));
            info.push(format!("Word count (estimated): {}", total_words));

            // 文件大小
            if let Ok(metadata) = std::fs::metadata(&path) {
                let size_kb = metadata.len() as f64 / 1024.0;
                info.push(format!("File size: {:.2} KB", size_kb));
            }

            Ok(ToolResult::success(info.join("\n")))
        })
    }
}

/// Word 文档结构工具：列出段落、表格等结构信息
pub struct WordStructureTool;

impl Tool for WordStructureTool {
    /// 返回工具名称
    fn name(&self) -> &str {
        "word_structure"
    }

    /// 返回工具所需权限
    fn permissions(&self) -> Vec<ToolPermission> {
        vec![ToolPermission::Read]
    }

    /// 返回工具描述
    fn description(&self) -> &str {
        "Get structure info of a Word document: headings, paragraphs, tables, images, etc."
    }

    /// 返回参数的 JSON Schema
    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute path to the Word document"
                }
            },
            "required": ["file_path"]
        })
    }

    /// 执行工具逻辑
    fn execute(&self, parameters: ToolParameters) -> BoxFuture<'_, Result<ToolResult>> {
        Box::pin(async move {
            let file_path = parameters
                .get("file_path")
                .and_then(|v| v.as_str())
                .ok_or_else(|| ToolError::MissingParameter("file_path".to_string()))?;

            let security = SecurityConfig::global();
            let path = security.validate_file(file_path)?;

            // 读取文件内容
            let bytes = std::fs::read(&path).map_err(|e| ToolError::ExecutionFailed {
                tool: TOOL_NAME.to_string(),
                message: format!("Failed to read file: {}", e),
            })?;

            // 解析 docx 文件
            let docx = docx_rs::read_docx(&bytes).map_err(|e| ToolError::ExecutionFailed {
                tool: TOOL_NAME.to_string(),
                message: format!("Failed to read Word document: {:?}", e),
            })?;

            // 分析文档结构
            let mut structure = Vec::new();
            structure.push(format!("File: {}", file_path));
            structure.push(String::new());
            structure.push("Document structure:".to_string());

            let document = &docx.document;
            let mut paragraph_count = 0;
            let mut table_count = 0;

            // 限制展示的段落数量
            let max_preview = security.limits.max_preview_rows;

            for child in &document.children {
                match child {
                    docx_rs::DocumentChild::Paragraph(p) => {
                        paragraph_count += 1;
                        if paragraph_count <= max_preview {
                            // 尝试提取段落文本作为标题预览
                            let text = extract_paragraph_text(p);
                            let preview: String = text.chars().take(50).collect();
                            if !preview.is_empty() {
                                structure
                                    .push(format!("  Paragraph {}: {}", paragraph_count, preview));
                            }
                        }
                    }
                    docx_rs::DocumentChild::Table(_) => {
                        table_count += 1;
                        structure.push(format!("  [Table {}]", table_count));
                    }
                    _ => {}
                }
            }

            structure.push(String::new());
            structure.push(format!(
                "Statistics: {} paragraphs, {} tables",
                paragraph_count, table_count
            ));
            if paragraph_count > max_preview {
                structure.push(format!("(Showing only first {} paragraphs)", max_preview));
            }

            Ok(ToolResult::success(structure.join("\n")))
        })
    }
}

// ── 辅助函数 ──────────────────────────────────────────────────

/// 从 docx 中提取文本（可选附带格式），并受预览字符上限约束
fn extract_text_from_docx(
    docx: &docx_rs::Docx,
    include_formatting: bool,
    limits: &ResourceLimits,
) -> String {
    let mut content = Vec::new();
    let document = &docx.document;
    let mut total_chars = 0;

    for child in &document.children {
        if total_chars >= limits.max_preview_chars {
            content.push(format!(
                "... (Maximum preview character limit reached: {})",
                limits.max_preview_chars
            ));
            break;
        }

        match child {
            docx_rs::DocumentChild::Paragraph(p) => {
                let text = if include_formatting {
                    extract_paragraph_text_with_formatting(p)
                } else {
                    extract_paragraph_text(p)
                };
                if !text.is_empty() {
                    total_chars += text.len();
                    content.push(text);
                }
            }
            docx_rs::DocumentChild::Table(table) => {
                content.push(String::new());
                content.push("[Table content]".to_string());
                // docx-rs 0.4 的 Table 结构：rows: Vec<TableChild>
                for row_child in &table.rows {
                    let docx_rs::TableChild::TableRow(row) = row_child;
                    let mut row_text = Vec::new();
                    // TableRow 的 cells: Vec<TableRowChild>
                    for cell_child in &row.cells {
                        let docx_rs::TableRowChild::TableCell(cell) = cell_child;
                        let cell_text = extract_cell_text(cell);
                        row_text.push(cell_text);
                    }
                    total_chars += row_text.join(" | ").len();
                    content.push(row_text.join(" | "));
                }
                content.push(String::new());
            }
            _ => {}
        }
    }

    content.join("\n")
}

/// 提取段落纯文本
fn extract_paragraph_text(paragraph: &docx_rs::Paragraph) -> String {
    let mut text = Vec::new();

    for child in &paragraph.children {
        if let docx_rs::ParagraphChild::Run(run) = child {
            for run_child in &run.children {
                if let docx_rs::RunChild::Text(t) = run_child {
                    text.push(t.text.clone());
                }
            }
        }
    }

    text.join("")
}

/// 提取带格式（标题、粗体、斜体转为 Markdown 标记）的段落文本
fn extract_paragraph_text_with_formatting(paragraph: &docx_rs::Paragraph) -> String {
    let mut text = Vec::new();

    // 检查段落属性（如标题样式）—— style 类型为 Option<ParagraphStyle>
    let style_prefix = if let Some(style) = &paragraph.property.style {
        if style.val.contains("Heading") || style.val.contains("Title") {
            "## "
        } else {
            ""
        }
    } else {
        ""
    };

    if !style_prefix.is_empty() {
        text.push(style_prefix.to_string());
    }

    for child in &paragraph.children {
        if let docx_rs::ParagraphChild::Run(run) = child {
            let mut run_text = String::new();

            // 检查格式 —— 直接读取 run_property 的值
            let props = &run.run_property;
            let is_bold = props.bold.is_some();
            let is_italic = props.italic.is_some();

            for run_child in &run.children {
                if let docx_rs::RunChild::Text(t) = run_child {
                    run_text.push_str(&t.text);
                }
            }

            if !run_text.is_empty() {
                if is_bold {
                    run_text = format!("**{}**", run_text);
                }
                if is_italic {
                    run_text = format!("*{}*", run_text);
                }
                text.push(run_text);
            }
        }
    }

    text.join("")
}

/// 提取单元格文本 —— TableCell 的 children: Vec<TableCellContent>
fn extract_cell_text(cell: &docx_rs::TableCell) -> String {
    let mut text = Vec::new();

    for child in &cell.children {
        if let docx_rs::TableCellContent::Paragraph(p) = child {
            text.push(extract_paragraph_text(p));
        }
    }

    text.join(" ")
}
