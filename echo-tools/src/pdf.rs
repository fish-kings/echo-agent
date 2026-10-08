/*!
 * PDF 文档处理工具
 *
 * 提供 PDF 文本提取能力，支持：
 * - 提取全部文本内容
 * - 提取指定页码范围
 * - 获取文档元数据（标题、作者、页数等）
 */

use futures::future::BoxFuture;
use serde_json::Value;

use crate::security::{ResourceLimits, SecurityConfig};
use echo_core::error::{Result, ToolError};
use echo_core::tools::permission::ToolPermission;
use echo_core::tools::{Tool, ToolParameters, ToolResult};

/// 错误信息中使用的工具名
const TOOL_NAME: &str = "pdf_tools";

/// PDF 文本提取工具
pub struct PdfExtractTool;

impl Tool for PdfExtractTool {
    /// 工具名称
    fn name(&self) -> &str {
        "extract_pdf"
    }

    /// 工具所需权限
    fn permissions(&self) -> Vec<ToolPermission> {
        vec![ToolPermission::Read]
    }

    /// 工具描述
    fn description(&self) -> &str {
        "Extract text content from PDF documents. Supports extracting all text, specified page ranges, or retrieving document metadata."
    }

    /// 参数 JSON Schema
    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute path to the PDF file"
                },
                "pages": {
                    "type": "string",
                    "description": "Page range to extract (optional), e.g. '1-5', '1,3,7', or 'all' (default)"
                },
                "extract_metadata": {
                    "type": "boolean",
                    "description": "Whether to also extract document metadata (default false)"
                }
            },
            "required": ["file_path"]
        })
    }

    /// 执行 PDF 文本提取（可选附带元数据）
    fn execute(&self, parameters: ToolParameters) -> BoxFuture<'_, Result<ToolResult>> {
        Box::pin(async move {
            let file_path = parameters
                .get("file_path")
                .and_then(|v| v.as_str())
                .ok_or_else(|| ToolError::MissingParameter("file_path".to_string()))?;

            let pages = parameters
                .get("pages")
                .and_then(|v| v.as_str())
                .unwrap_or("all");

            let extract_metadata = parameters
                .get("extract_metadata")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);

            let security = SecurityConfig::global();
            let path = security.validate_file(file_path)?;

            // 使用 lopdf 打开 PDF 文档
            let pdf = lopdf::Document::load(&path).map_err(|e| ToolError::ExecutionFailed {
                tool: TOOL_NAME.to_string(),
                message: format!("Failed to open PDF: {}", e),
            })?;

            // 获取页数
            let total_pages = pdf.get_pages().len();

            // 提取元数据
            let metadata_str = if extract_metadata {
                extract_pdf_metadata(&pdf)?
            } else {
                String::new()
            };

            // 解析页码范围
            let page_numbers = parse_page_range(pages, total_pages, &security.limits)?;

            // 从指定页提取文本
            let text_content = extract_pages_text(&pdf, &page_numbers, &security.limits)?;

            // 构建结果
            let result = if extract_metadata {
                format!(
                    "=== PDF Metadata ===\n{}\n\n=== Text Content (pages {}, total {} pages) ===\n{}",
                    metadata_str, pages, total_pages, text_content
                )
            } else {
                format!(
                    "=== Text Content (pages {}, total {} pages) ===\n{}",
                    pages, total_pages, text_content
                )
            };

            Ok(ToolResult::success(result))
        })
    }
}

/// PDF 信息工具（获取文档概览）
pub struct PdfInfoTool;

impl Tool for PdfInfoTool {
    /// 工具名称
    fn name(&self) -> &str {
        "pdf_info"
    }

    /// 工具所需权限
    fn permissions(&self) -> Vec<ToolPermission> {
        vec![ToolPermission::Read]
    }

    /// 工具描述
    fn description(&self) -> &str {
        "Get basic information about a PDF document: page count, title, author, creation time, and other metadata. Does not extract text content."
    }

    /// 参数 JSON Schema
    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute path to the PDF file"
                }
            },
            "required": ["file_path"]
        })
    }

    /// 执行 PDF 元数据读取
    fn execute(&self, parameters: ToolParameters) -> BoxFuture<'_, Result<ToolResult>> {
        Box::pin(async move {
            let file_path = parameters
                .get("file_path")
                .and_then(|v| v.as_str())
                .ok_or_else(|| ToolError::MissingParameter("file_path".to_string()))?;

            let security = SecurityConfig::global();
            let path = security.validate_file(file_path)?;

            let pdf = lopdf::Document::load(&path).map_err(|e| ToolError::ExecutionFailed {
                tool: TOOL_NAME.to_string(),
                message: format!("Failed to open PDF: {}", e),
            })?;

            let metadata = extract_pdf_metadata(&pdf)?;

            Ok(ToolResult::success(metadata))
        })
    }
}

// ── 辅助函数 ────────────────────────────────────────────────────

/// 提取 PDF 元数据
fn extract_pdf_metadata(pdf: &lopdf::Document) -> Result<String> {
    use lopdf::Object;

    let mut info = Vec::new();

    info.push(format!("Page count: {}", pdf.get_pages().len()));

    // 尝试从 trailer 中获取元数据
    if let Ok(trailer) = pdf.trailer.get(b"Info")
        && let Object::Dictionary(dict) = trailer
    {
        for (key, value) in dict.iter() {
            let key_str = match key.as_slice() {
                b"Title" => "Title",
                b"Author" => "Author",
                b"Subject" => "Subject",
                b"Creator" => "Creator",
                b"Producer" => "PDF Producer",
                b"CreationDate" => "Creation Date",
                b"ModDate" => "Modification Date",
                other => std::str::from_utf8(other).unwrap_or("Unknown"),
            };

            let value_str = match value {
                Object::String(s, _) => {
                    // PDF 日期格式转换
                    if key.as_slice() == b"CreationDate" || key.as_slice() == b"ModDate" {
                        parse_pdf_date(s)
                    } else {
                        String::from_utf8_lossy(s).to_string()
                    }
                }
                Object::Name(n) => String::from_utf8_lossy(n).to_string(),
                Object::Integer(i) => i.to_string(),
                Object::Real(f) => f.to_string(),
                Object::Boolean(b) => b.to_string(),
                _ => "Unknown".to_string(),
            };

            info.push(format!("{}: {}", key_str, value_str));
        }
    }

    // 追加页数作为基础信息
    info.push(format!("Total pages: {}", pdf.get_pages().len()));

    Ok(info.join("\n"))
}

/// 解析 PDF 日期格式
fn parse_pdf_date(date: &[u8]) -> String {
    // PDF 日期格式：D:YYYYMMDDHHmmSS
    let date_str = String::from_utf8_lossy(date);
    if let Some(rest) = date_str.strip_prefix("D:") {
        let mut chars = rest.chars();
        let year = chars.by_ref().take(4).collect::<String>();
        let month = chars.by_ref().take(2).collect::<String>();
        let day = chars.by_ref().take(2).collect::<String>();
        if year.chars().count() == 4
            && month.chars().count() == 2
            && day.chars().count() == 2
            && year
                .chars()
                .chain(month.chars())
                .chain(day.chars())
                .all(|ch| ch.is_ascii_digit())
        {
            return format!("{}-{}-{}", year, month, day);
        }
    }
    date_str.to_string()
}

/// 解析页码范围字符串
fn parse_page_range(range: &str, total_pages: usize, limits: &ResourceLimits) -> Result<Vec<u32>> {
    if range == "all" {
        // 限制最大预览页数
        let max_pages = limits.max_preview_pages.min(total_pages);
        return Ok((1..=max_pages as u32).collect());
    }

    let mut pages = Vec::new();

    // 处理逗号分隔的各个部分
    for part in range.split(',') {
        if part.contains('-') {
            // 处理范围
            let bounds: Vec<&str> = part.split('-').collect();
            if bounds.len() != 2 {
                return Err(ToolError::InvalidParameter {
                    name: "pages".to_string(),
                    message: format!("Invalid page range: {}", part),
                }
                .into());
            }

            let start: u32 = bounds[0].parse().map_err(|_| ToolError::InvalidParameter {
                name: "pages".to_string(),
                message: format!("Invalid start page: {}", bounds[0]),
            })?;

            let end: u32 = bounds[1].parse().map_err(|_| ToolError::InvalidParameter {
                name: "pages".to_string(),
                message: format!("Invalid end page: {}", bounds[1]),
            })?;

            if start > end || end > total_pages as u32 {
                return Err(ToolError::InvalidParameter {
                    name: "pages".to_string(),
                    message: format!(
                        "Page range invalid or exceeds document page count ({} pages)",
                        total_pages
                    ),
                }
                .into());
            }

            // 将提取页数限制在 max_preview_pages 以内
            let limited_end = (end - start + 1).min(limits.max_preview_pages as u32);
            for p in start..(start + limited_end) {
                if !pages.contains(&p) {
                    pages.push(p);
                }
            }
        } else {
            // 单页
            let page: u32 = part.parse().map_err(|_| ToolError::InvalidParameter {
                name: "pages".to_string(),
                message: format!("Invalid page number: {}", part),
            })?;

            if page > total_pages as u32 {
                return Err(ToolError::InvalidParameter {
                    name: "pages".to_string(),
                    message: format!(
                        "Page {} exceeds document page count ({} pages)",
                        page, total_pages
                    ),
                }
                .into());
            }

            if !pages.contains(&page) {
                pages.push(page);
            }
        }
    }

    // 检查总页数限制
    if pages.len() > limits.max_preview_pages {
        pages = pages.into_iter().take(limits.max_preview_pages).collect();
    }

    pages.sort();
    Ok(pages)
}

/// 从指定页提取文本内容
fn extract_pages_text(
    pdf: &lopdf::Document,
    page_numbers: &[u32],
    limits: &ResourceLimits,
) -> Result<String> {
    use lopdf::Object;

    let mut all_text = Vec::new();
    let mut total_chars = 0;

    for page_num in page_numbers {
        if total_chars >= limits.max_preview_chars {
            all_text.push(format!(
                "... (max preview character limit {} reached)",
                limits.max_preview_chars
            ));
            break;
        }

        let page_id = *pdf.get_pages().get(page_num).unwrap_or(&(0, 0));

        if let Ok(page_obj) = pdf.get_object(page_id)
            && let Object::Dictionary(dict) = page_obj
        {
            // 获取页面内容流
            if let Ok(contents_ref) = dict.get(b"Contents") {
                let content_stream: Option<lopdf::Stream> = match contents_ref {
                    Object::Reference(id) => pdf.get_object(*id).ok().and_then(|obj| {
                        if let Object::Stream(stream) = obj {
                            Some(stream.clone())
                        } else {
                            None
                        }
                    }),
                    Object::Array(arr) => {
                        // 多个内容流，合并处理
                        let mut combined = Vec::new();
                        for obj_ref in arr.iter() {
                            if let Object::Reference(id) = obj_ref
                                && let Ok(obj) = pdf.get_object(*id)
                                && let Object::Stream(stream) = obj
                            {
                                combined.extend_from_slice(&stream.content);
                            }
                        }
                        // 解析合并后的内容
                        let text = extract_text_from_stream(&combined, limits);
                        total_chars += text.len();
                        all_text.push(format!("--- Page {} ---\n{}", page_num, text));
                        continue;
                    }
                    Object::Stream(stream) => Some(stream.clone()),
                    _ => None,
                };

                if let Some(stream) = content_stream {
                    let text = extract_text_from_stream(&stream.content, limits);
                    total_chars += text.len();
                    all_text.push(format!("--- Page {} ---\n{}", page_num, text));
                }
            }
        }
    }

    Ok(all_text.join("\n\n"))
}

/// 从 PDF 内容流中提取文本
fn extract_text_from_stream(content: &[u8], limits: &ResourceLimits) -> String {
    // 简化的文本提取：查找 Tj 和 TJ 操作符
    let content_str = String::from_utf8_lossy(content);
    let mut text_parts = Vec::new();

    // 使用安全的正则，限制其大小
    let tj_regex = regex::RegexBuilder::new(r"\(([^)]*)\)\s*Tj")
        .size_limit(limits.regex_max_size)
        .dfa_size_limit(limits.regex_max_size)
        .build()
        .ok();
    if let Some(tj_regex) = tj_regex {
        for cap in tj_regex.captures_iter(&content_str) {
            if let Some(text) = cap.get(1) {
                text_parts.push(text.as_str().to_string());
            }
        }
    }

    // 匹配 <...>Tj 格式的十六进制文本
    let hex_regex = regex::RegexBuilder::new(r"<([0-9a-fA-F]*)>\s*Tj")
        .size_limit(limits.regex_max_size)
        .dfa_size_limit(limits.regex_max_size)
        .build()
        .ok();
    if let Some(hex_regex) = hex_regex {
        for cap in hex_regex.captures_iter(&content_str) {
            if let Some(hex) = cap.get(1) {
                // 尝试将十六进制解码为文本
                if let Ok(decoded) = hex_decode(hex.as_str()) {
                    text_parts.push(decoded);
                }
            }
        }
    }

    // 匹配数组格式 [... TJ]
    let tj_array_regex = regex::RegexBuilder::new(r"\[(.*?)\]\s*TJ")
        .size_limit(limits.regex_max_size)
        .dfa_size_limit(limits.regex_max_size)
        .build()
        .ok();
    let str_regex = regex::RegexBuilder::new(r"\(([^)]*)\)")
        .size_limit(limits.regex_max_size)
        .dfa_size_limit(limits.regex_max_size)
        .build()
        .ok();
    if let (Some(tj_array_regex), Some(str_regex)) = (tj_array_regex, str_regex) {
        for cap in tj_array_regex.captures_iter(&content_str) {
            if let Some(arr_content) = cap.get(1) {
                for str_cap in str_regex.captures_iter(arr_content.as_str()) {
                    if let Some(s) = str_cap.get(1) {
                        text_parts.push(s.as_str().to_string());
                    }
                }
            }
        }
    }

    // 限制输出长度
    let result = text_parts.join(" ");
    if result.len() > limits.max_preview_chars {
        result.chars().take(limits.max_preview_chars).collect()
    } else {
        result
    }
}

/// 解码 PDF 十六进制字符串
fn hex_decode(hex: &str) -> Result<String> {
    if !hex.is_ascii() {
        return Err(ToolError::InvalidParameter {
            name: "hex".to_string(),
            message: "PDF hexadecimal string must be ASCII".to_string(),
        }
        .into());
    }
    let mut bytes = Vec::with_capacity(hex.len().div_ceil(2));
    for pair in hex.as_bytes().chunks(2) {
        let high = (pair.first().copied().unwrap_or(b'0') as char)
            .to_digit(16)
            .ok_or_else(|| ToolError::InvalidParameter {
                name: "hex".to_string(),
                message: "PDF hexadecimal string contains a non-hex digit".to_string(),
            })?;
        let low = (pair.get(1).copied().unwrap_or(b'0') as char)
            .to_digit(16)
            .ok_or_else(|| ToolError::InvalidParameter {
                name: "hex".to_string(),
                message: "PDF hexadecimal string contains a non-hex digit".to_string(),
            })?;
        bytes.push(((high << 4) | low) as u8);
    }

    Ok(String::from_utf8_lossy(&bytes).to_string())
}
