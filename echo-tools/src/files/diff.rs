//! DiffTool —— 对比两个文件或字符串并展示差异
//!
//! 使用 `similar` crate 生成统一格式（unified diff）的差异结果。支持：
//! - 对比两个文件（`path_a` + `path_b`）
//! - 对比一个文件与一段字符串内容（`path_a` + `content_b`）

use super::resolve_path;
use echo_core::error::{Result, ToolError};
use echo_core::tools::pagination::PageRequest;
use echo_core::tools::{Tool, ToolParameters, ToolResult};
use futures::future::BoxFuture;
use serde_json::{Value, json};
use similar::udiff;
use std::path::PathBuf;
use tokio::fs;

// ── DiffTool ────────────────────────────────────────────────────────────────────

/// 对比两个文件，或对比一个文件与一段字符串内容，返回统一格式的差异结果。
pub struct DiffTool {
    /// 路径解析的基准目录；为 `None` 时使用调用上下文提供的工作目录或进程当前目录。
    base_dir: Option<PathBuf>,
}

impl DiffTool {
    /// 创建一个不限定基准目录的 `DiffTool`。
    pub fn new() -> Self {
        Self { base_dir: None }
    }

    /// 创建一个限定了基准目录的 `DiffTool`，所有相对路径都会基于该目录解析。
    pub fn with_base_dir(base: impl Into<PathBuf>) -> Self {
        Self {
            base_dir: Some(base.into()),
        }
    }
}

impl Default for DiffTool {
    /// 等价于 [`DiffTool::new`]，创建不限定基准目录的实例。
    fn default() -> Self {
        Self::new()
    }
}

impl Tool for DiffTool {
    /// 返回该工具在工具注册表中的名称：`"diff"`。
    fn name(&self) -> &str {
        "diff"
    }

    /// 返回展示给模型/调用方的工具功能描述。
    fn description(&self) -> &str {
        "Compare two files or a file against a string, returning a unified diff. Use path_a + path_b to compare two files, or path_a + content_b to compare a file against given content."
    }

    /// 声明该工具所需的权限：仅需要读取文件系统。
    fn permissions(&self) -> Vec<echo_core::tools::permission::ToolPermission> {
        vec![echo_core::tools::permission::ToolPermission::Read]
    }
    /// 声明该工具的风险等级：只读操作，不会修改任何内容。
    fn risk_level(&self) -> echo_core::tools::ToolRiskLevel {
        echo_core::tools::ToolRiskLevel::ReadOnly
    }

    /// 返回该工具的 JSON Schema 参数定义。
    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path_a": {
                    "type": "string",
                    "description": "First file path (the 'before' side)"
                },
                "path_b": {
                    "type": "string",
                    "description": "Second file path (the 'after' side). Either path_b or content_b is required."
                },
                "content_b": {
                    "type": "string",
                    "description": "Alternative to path_b: compare path_a against this string content."
                },
                "context": {
                    "type": "integer",
                    "description": "Number of context lines around changes (default: 3)"
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "description": "Hunks per page (default 10)"
                },
                "cursor": {
                    "type": "string",
                    "description": "Cursor from the previous page"
                }
            },
            "required": ["path_a"]
        })
    }

    /// 执行差异对比：读取 `path_a`（以及 `path_b` 或 `content_b`），生成统一格式差异并分页返回。
    fn execute_with_context<'a>(
        &'a self,
        parameters: ToolParameters,
        ctx: &'a echo_core::tools::ToolContext,
    ) -> BoxFuture<'a, Result<ToolResult>> {
        Box::pin(async move {
            let path_a_str = parameters
                .get("path_a")
                .and_then(|v| v.as_str())
                .ok_or_else(|| ToolError::MissingParameter("path_a".to_string()))?;

            let path_b_str = parameters.get("path_b").and_then(|v| v.as_str());
            let content_b = parameters.get("content_b").and_then(|v| v.as_str());

            let context_lines = match parameters.get("context") {
                Some(value) => match value.as_u64().and_then(|value| usize::try_from(value).ok()) {
                    Some(value) => value,
                    None => {
                        return Ok(ToolResult::invalid_arguments(
                            "context must be a non-negative integer supported by this platform"
                                .to_string(),
                        ));
                    }
                },
                None => 3,
            };
            let page_request = match PageRequest::from_parameters(&parameters, 10, 50) {
                Ok(request) => request,
                Err(error) => return Ok(ToolResult::invalid_arguments(error.to_string())),
            };

            if path_b_str.is_none() && content_b.is_none() {
                return Ok(ToolResult::invalid_arguments(
                    "Either path_b or content_b must be provided.".to_string(),
                ));
            }

            let path_a = resolve_path(
                "diff",
                path_a_str,
                &self.base_dir,
                ctx.working_dir.as_deref(),
            )?;

            if !path_a.exists() {
                return Ok(ToolResult::invalid_arguments(format!(
                    "File does not exist: {}",
                    path_a.display()
                )));
            }

            let content_a =
                fs::read_to_string(&path_a)
                    .await
                    .map_err(|e| ToolError::ExecutionFailed {
                        tool: "diff".to_string(),
                        message: format!("Failed to read {}: {}", path_a.display(), e),
                    })?;

            let (content_b_val, label_b) = if let Some(content) = content_b {
                (content.to_string(), "<provided content>".to_string())
            } else if let Some(path_b_str) = path_b_str {
                let path_b = resolve_path(
                    "diff",
                    path_b_str,
                    &self.base_dir,
                    ctx.working_dir.as_deref(),
                )?;
                if !path_b.exists() {
                    return Ok(ToolResult::invalid_arguments(format!(
                        "File does not exist: {}",
                        path_b.display()
                    )));
                }
                let content =
                    fs::read_to_string(&path_b)
                        .await
                        .map_err(|e| ToolError::ExecutionFailed {
                            tool: "diff".to_string(),
                            message: format!("Failed to read {}: {}", path_b.display(), e),
                        })?;
                (content, path_b.display().to_string())
            } else {
                return Ok(ToolResult::invalid_arguments(
                    "Either path_b or content_b must be provided.".to_string(),
                ));
            };

            let label_a = path_a.display().to_string();
            let diff = udiff::unified_diff(
                similar::Algorithm::default(),
                &content_a,
                &content_b_val,
                context_lines,
                Some((&label_a, &label_b)),
            );

            let chunks = crate::diff_pagination::split_unified_diff(&diff);
            let query = serde_json::json!({
                "path_a": path_a,
                "path_b": path_b_str,
                "content_b_sha256": content_b.map(|content| {
                    use sha2::{Digest, Sha256};
                    format!("{:x}", Sha256::digest(content.as_bytes()))
                }),
                "context": context_lines,
            });
            let (page, page_info) = match page_request.paginate(chunks, &query) {
                Ok(page) => page,
                Err(error) => return Ok(ToolResult::invalid_arguments(error.to_string())),
            };
            let output = if page.is_empty() {
                "Files are identical - no differences found.".to_string()
            } else {
                page.join("\n")
            };
            let mut result = ToolResult::success(output);
            page_info.apply_to(&mut result);
            Ok(result)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 内容完全相同的两个字符串应生成空差异。
    #[test]
    fn test_diff_identical() {
        let content = "hello\nworld\n";
        let diff = udiff::unified_diff(
            similar::Algorithm::default(),
            content,
            content,
            3,
            Some(("a.txt", "b.txt")),
        );
        assert!(diff.is_empty());
    }

    /// 内容有差异时，生成的统一格式差异应包含被删除行和新增行。
    #[test]
    fn test_diff_changed() {
        let a = "hello\nworld\nfoo\n";
        let b = "hello\nrust\nfoo\n";
        let diff = udiff::unified_diff(
            similar::Algorithm::default(),
            a,
            b,
            3,
            Some(("a.txt", "b.txt")),
        );
        assert!(diff.contains("-world"));
        assert!(diff.contains("+rust"));
    }
}
