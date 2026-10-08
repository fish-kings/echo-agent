/*!
Grep 工具：按正则表达式搜索文件内容

使用 `regex` crate 进行模式匹配。支持上下文行、
文件类型过滤以及大小写不敏感模式。
*/

use echo_core::error::{Result, ToolError};
use echo_core::tools::pagination::PageRequest;
use echo_core::tools::{Tool, ToolParameters, ToolResult};
use futures::future::BoxFuture;
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use tokio::fs;

// ── GrepTool ────────────────────────────────────────────────────────────────

/**
按正则表达式搜索文件内容。

从 `path`（默认 `.`）开始遍历目录树，读取每个文件，
返回匹配 `pattern` 正则的行。支持上下文行、
文件 glob 过滤以及大小写不敏感模式。
*/
pub struct GrepTool {
    /// 构造时指定的基础目录；用于解析相对路径并限制搜索范围
    base_dir: Option<PathBuf>,
}

impl GrepTool {
    /// 创建未绑定基础目录的 grep 工具
    pub fn new() -> Self {
        Self { base_dir: None }
    }

    /// 创建以 `base` 为基础目录的 grep 工具
    pub fn with_base_dir(base: impl Into<PathBuf>) -> Self {
        Self {
            base_dir: Some(base.into()),
        }
    }
}

impl Default for GrepTool {
    /// 创建默认实例
    fn default() -> Self {
        Self::new()
    }
}

impl Tool for GrepTool {
    /// 返回工具名称
    fn name(&self) -> &str {
        "grep"
    }

    /// 返回工具描述
    fn description(&self) -> &str {
        "Exact text search over file contents using a regex. Returns matching lines with file paths and line numbers; use code_search for symbol-oriented code discovery."
    }

    /// 返回工具所需权限
    fn permissions(&self) -> Vec<echo_core::tools::permission::ToolPermission> {
        vec![echo_core::tools::permission::ToolPermission::Read]
    }
    /// 返回工具风险等级
    fn risk_level(&self) -> echo_core::tools::ToolRiskLevel {
        echo_core::tools::ToolRiskLevel::ReadOnly
    }

    /// 返回参数的 JSON Schema
    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regular expression pattern to search for"
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file path to search in (default: current directory)"
                },
                "glob": {
                    "type": "string",
                    "description": "File name glob filter (e.g. '*.rs', '*.py', '*.{ts,tsx}')"
                },
                "case_insensitive": {
                    "type": "boolean",
                    "description": "Case insensitive matching (default: false)"
                },
                "context": {
                    "type": "integer",
                    "description": "Number of context lines before and after each match (default: 0)"
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 200,
                    "description": "Results per page (default 50)"
                },
                "cursor": {
                    "type": "string",
                    "description": "Cursor from the previous page"
                }
            },
            "required": ["pattern"]
        })
    }

    /// 结合工具上下文执行工具逻辑
    fn execute_with_context<'a>(
        &'a self,
        parameters: ToolParameters,
        ctx: &'a echo_core::tools::ToolContext,
    ) -> BoxFuture<'a, Result<ToolResult>> {
        Box::pin(async move {
            let pattern_str = parameters
                .get("pattern")
                .and_then(|v| v.as_str())
                .ok_or_else(|| ToolError::MissingParameter("pattern".to_string()))?;

            let path_str = parameters
                .get("path")
                .and_then(|v| v.as_str())
                .unwrap_or(".");

            let glob_filter = parameters.get("glob").and_then(|v| v.as_str());

            let case_insensitive = parameters
                .get("case_insensitive")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);

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
                None => 0,
            };

            let page_request = match PageRequest::from_parameters(&parameters, 50, 200) {
                Ok(request) => request,
                Err(error) => return Ok(ToolResult::invalid_arguments(error.to_string())),
            };

            // 编译正则表达式
            let regex = regex::RegexBuilder::new(pattern_str)
                .case_insensitive(case_insensitive)
                .build()
                .map_err(|e| ToolError::ExecutionFailed {
                    tool: "grep".to_string(),
                    message: format!("Invalid regex pattern '{}': {}", pattern_str, e),
                })?;

            /*
            允许的根目录集合（均会规范化，防止通过符号链接或 `..` 逃逸）：
              1. 构造时的 base_dir（解析时优先级最高）
              2. 运行时的 working_dir
              3. output_artifacts.root_dir：让模型可以按绝对路径 grep
                 溢出的工具输出和用户输入 artifact，即 `read_artifact`
                 已经给出的那些路径。与 `artifact.rs` 中的
                 `resolve_artifact_path` 保持一致。
            相对 `path` 会拼接到第一个可用的根目录（依次为 base_dir、
            working_dir、artifact 根目录），以保留原有的单根目录行为；
            绝对 `path` 原样使用。只要规范化后的路径以*任一*允许的根目录
            开头，即通过范围限制检查。
            */
            let mut allowed_roots: Vec<PathBuf> = Vec::new();
            if let Some(base) = &self.base_dir {
                allowed_roots.push(base.clone());
            }
            if let Some(working) = &ctx.working_dir {
                allowed_roots.push(working.clone());
            }
            if let Some(artifacts) = ctx.output_artifacts.as_ref() {
                allowed_roots.push(artifacts.root_dir.clone());
            }

            let preferred_root = self
                .base_dir
                .clone()
                .or_else(|| ctx.working_dir.as_ref().map(|p| p.to_path_buf()))
                .or_else(|| {
                    ctx.output_artifacts
                        .as_ref()
                        .map(|artifacts| artifacts.root_dir.clone())
                });

            let search_path = if let Some(ref base) = preferred_root {
                let resolved = if Path::new(path_str).is_absolute() {
                    PathBuf::from(path_str)
                } else {
                    base.join(path_str)
                };
                /* 同时规范化候选路径与各根目录，以捕获符号链接或 `..` 逃逸。
                不存在的根目录（如尚未创建的 artifact 目录）会被跳过而不是报错。 */
                let resolved_canon = match tokio::fs::canonicalize(&resolved).await {
                    Ok(p) => p,
                    Err(_) => resolved.clone(),
                };
                let within_scope = canonicalize_roots(&allowed_roots)
                    .await
                    .iter()
                    .any(|root| resolved_canon.starts_with(root));
                if !within_scope {
                    return Ok(ToolResult::invalid_arguments(format!(
                        "Path '{}' is outside the allowed directory scope",
                        path_str
                    )));
                }
                resolved
            } else {
                // 未配置任何根目录：保留旧的无限制行为。
                PathBuf::from(path_str)
            };

            if !search_path.exists() {
                return Ok(ToolResult::invalid_arguments(format!(
                    "Path does not exist: {}",
                    search_path.display()
                )));
            }

            let mut results = Vec::new();

            if search_path.is_file() {
                search_file(
                    &search_path,
                    &regex,
                    glob_filter,
                    context_lines,
                    &mut results,
                )
                .await?;
            } else if search_path.is_dir() {
                walk_and_search(
                    &search_path,
                    &regex,
                    glob_filter,
                    context_lines,
                    &mut results,
                )
                .await?;
            }

            let query = serde_json::json!({
                "pattern": pattern_str,
                "path": search_path,
                "glob": glob_filter,
                "case_insensitive": case_insensitive,
                "context": context_lines,
            });
            let (page, page_info) = match page_request.paginate(results, &query) {
                Ok(page) => page,
                Err(error) => return Ok(ToolResult::invalid_arguments(error.to_string())),
            };
            let output = if page.is_empty() {
                "No matches found.".to_string()
            } else {
                format!(
                    "{}\n\n{} matches returned ({} total)",
                    page.join("\n"),
                    page_info.returned,
                    page_info.total.unwrap_or(0)
                )
            };
            let mut result = ToolResult::success(output);
            page_info.apply_to(&mut result);
            Ok(result)
        })
    }
}

/**
将每个候选根目录解析为规范形式，跳过磁盘上不存在的目录
（如尚未创建的 artifact 目录）。用于范围限制检查，
确保符号链接和 `..` 无法逃逸出允许的范围。
*/
async fn canonicalize_roots(roots: &[PathBuf]) -> Vec<PathBuf> {
    let mut canon: Vec<PathBuf> = Vec::with_capacity(roots.len());
    for root in roots {
        // 不存在的根目录（如尚未创建的 artifact 目录）直接跳过，不报错。
        if let Ok(p) = tokio::fs::canonicalize(root).await {
            canon.push(p);
        }
    }
    canon
}

/// 在单个文件中搜索匹配行，并将格式化结果追加到 `results`
async fn search_file(
    path: &Path,
    regex: &regex::Regex,
    glob_filter: Option<&str>,
    context_lines: usize,
    results: &mut Vec<String>,
) -> Result<()> {
    // 检查 glob 过滤条件
    if let Some(glob) = glob_filter
        && let Some(name) = path.file_name().and_then(|n| n.to_str())
        && !glob_matches(glob, name)
    {
        return Ok(());
    }

    // 跳过二进制文件和大文件
    let metadata = fs::metadata(path)
        .await
        .map_err(|e| ToolError::ExecutionFailed {
            tool: "grep".to_string(),
            message: format!("Cannot read metadata for {}: {}", path.display(), e),
        })?;

    if metadata.len() > 10 * 1024 * 1024 {
        // 跳过大于 10MB 的文件
        return Ok(());
    }

    let content = match fs::read_to_string(path).await {
        Ok(c) => c,
        Err(_) => return Ok(()), // 跳过二进制或不可读文件
    };

    let lines: Vec<&str> = content.lines().collect();
    let path_display = path.display().to_string();

    for (line_idx, line) in lines.iter().enumerate() {
        if regex.is_match(line) {
            let line_num = line_idx.saturating_add(1);
            if context_lines > 0 {
                let start = line_idx.saturating_sub(context_lines);
                let end = line_idx
                    .saturating_add(context_lines)
                    .saturating_add(1)
                    .min(lines.len());
                let mut record = Vec::new();
                for (ctx_idx, context_line) in lines
                    .iter()
                    .enumerate()
                    .skip(start)
                    .take(end.saturating_sub(start))
                {
                    let ctx_num = ctx_idx.saturating_add(1);
                    let prefix = if ctx_idx == line_idx { ">" } else { " " };
                    record.push(format!("{prefix}{path_display}:{ctx_num}:  {context_line}"));
                }
                results.push(record.join("\n"));
            } else {
                results.push(format!("{path_display}:{line_num}:  {line}"));
            }
        }
    }

    Ok(())
}

/// 递归遍历目录并对其中的文件执行搜索
async fn walk_and_search(
    dir: &Path,
    regex: &regex::Regex,
    glob_filter: Option<&str>,
    context_lines: usize,
    results: &mut Vec<String>,
) -> Result<()> {
    let mut entries = match fs::read_dir(dir).await {
        Ok(e) => e,
        Err(_) => return Ok(()),
    };

    // 收集并排序目录项，保证输出顺序确定
    let mut entry_paths: Vec<PathBuf> = Vec::new();
    while let Ok(Some(entry)) = entries.next_entry().await {
        let path = entry.path();
        // 跳过隐藏目录和常见的忽略目录
        if let Some(name) = path.file_name().and_then(|n| n.to_str())
            && (name.starts_with('.')
                || name == "target"
                || name == "node_modules"
                || name == "__pycache__"
                || name == ".git"
                || name == "vendor"
                || name == "build"
                || name == "dist")
        {
            continue;
        }
        entry_paths.push(path);
    }
    entry_paths.sort();

    for path in entry_paths {
        if path.is_dir() {
            Box::pin(walk_and_search(
                &path,
                regex,
                glob_filter,
                context_lines,
                results,
            ))
            .await?;
        } else if path.is_file() {
            search_file(&path, regex, glob_filter, context_lines, results).await?;
        }
    }

    Ok(())
}

/// 简单 glob 匹配，支持 `*` 和 `?` 通配符以及 `{a,b}` 花括号展开
fn glob_matches(pattern: &str, name: &str) -> bool {
    // 处理 *.{rs,ts} 这类花括号模式
    if let Some((prefix, remainder)) = pattern.split_once('{')
        && let Some((alternatives, suffix)) = remainder.split_once('}')
    {
        for alt in alternatives.split(',') {
            let full = format!("{}{}{}", prefix, alt.trim(), suffix);
            if glob_matches_simple(&full, name) {
                return true;
            }
        }
        return false;
    }

    glob_matches_simple(pattern, name)
}

/// 不含花括号的简单 glob 匹配
fn glob_matches_simple(pattern: &str, name: &str) -> bool {
    let pat_chars: Vec<char> = pattern.chars().collect();
    let name_chars: Vec<char> = name.chars().collect();
    glob_match_inner(&pat_chars, &name_chars, 0, 0)
}

/// glob 匹配的递归实现：`pi`、`ni` 分别为模式与名称的当前下标
fn glob_match_inner(pat: &[char], name: &[char], pi: usize, ni: usize) -> bool {
    if pi == pat.len() {
        return ni == name.len();
    }

    if pat.get(pi).is_some_and(|character| *character == '*') {
        // 尝试匹配 0 个或多个字符
        for skip in ni..=name.len() {
            if glob_match_inner(pat, name, pi.saturating_add(1), skip) {
                return true;
            }
        }
        return false;
    }

    if ni >= name.len() {
        return false;
    }

    let pattern_character = pat.get(pi);
    let name_character = name.get(ni);
    if pattern_character.is_some_and(|character| *character == '?')
        || pattern_character == name_character
    {
        return glob_match_inner(pat, name, pi.saturating_add(1), ni.saturating_add(1));
    }

    false
}

// ── 测试 ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// 验证 `*` 通配符匹配
    #[test]
    fn test_glob_matches_star() {
        assert!(glob_matches("*.rs", "main.rs"));
        assert!(glob_matches("*.rs", "lib.rs"));
        assert!(!glob_matches("*.rs", "main.py"));
    }

    /// 验证花括号展开匹配
    #[test]
    fn test_glob_matches_brace() {
        assert!(glob_matches("*.{rs,ts}", "main.rs"));
        assert!(glob_matches("*.{rs,ts}", "app.ts"));
        assert!(!glob_matches("*.{rs,ts}", "main.py"));
    }

    /// 验证 `?` 通配符匹配单个字符
    #[test]
    fn test_glob_matches_question_mark() {
        assert!(glob_matches("?.rs", "a.rs"));
        assert!(!glob_matches("?.rs", "ab.rs"));
    }

    /// 验证无通配符时的精确匹配
    #[test]
    fn test_glob_matches_exact() {
        assert!(glob_matches("main.rs", "main.rs"));
        assert!(!glob_matches("main.rs", "lib.rs"));
    }

    // ── 范围限制 / artifact 根目录扩展（execute_with_context） ──
    /*
    以下测试验证新的候选根目录集合逻辑：grep 可以搜索 artifact 根目录下的文件
    （使模型能按绝对路径 grep 溢出的工具输出或用户输入 artifact），
    同时仍会拒绝位于所有允许根目录之外的路径。
    */

    use echo_core::tools::artifact::ToolOutputArtifactConfig;

    /// 根据给定的路径与模式构造 grep 参数。
    fn grep_params(pattern: &str, path: &str) -> ToolParameters {
        ToolParameters::from([
            ("pattern".to_string(), Value::String(pattern.to_string())),
            ("path".to_string(), Value::String(path.to_string())),
        ])
    }

    /// 仅配置 artifact 根目录时，相对路径可在其下找到内容
    #[tokio::test]
    async fn grep_finds_relative_content_under_artifact_only_root() -> Result<()> {
        let root = tempfile::tempdir()?;
        let log = root.path().join("server.log");
        std::fs::write(&log, "ERROR something failed\nINFO ok\n")?;

        let ctx = echo_core::tools::ToolContext {
            output_artifacts: Some(ToolOutputArtifactConfig::new(root.path(), "test")),
            ..Default::default()
        };
        let tool = GrepTool::new();
        let result = tool
            .execute_with_context(grep_params("ERROR", "server.log"), &ctx)
            .await?;
        assert!(result.success, "{}", result.error.unwrap_or_default());
        assert!(result.output.contains("something failed"));
        Ok(())
    }

    /// 位于所有允许根目录之外的路径会被拒绝
    #[tokio::test]
    async fn grep_rejects_path_outside_all_roots() -> Result<()> {
        // 既不在 working_dir 也不在 artifact 根目录下的临时文件，必须因超出允许范围而被拒绝。
        let outside_root = tempfile::tempdir()?;
        let outside = outside_root.path().join("outside.txt");
        std::fs::write(&outside, "SECRET")?;
        let working = tempfile::tempdir()?;
        let artifact = tempfile::tempdir()?;

        let ctx = echo_core::tools::ToolContext {
            working_dir: Some(working.path().to_path_buf()),
            output_artifacts: Some(ToolOutputArtifactConfig::new(artifact.path(), "test")),
            ..Default::default()
        };
        let tool = GrepTool::new();
        let result = tool
            .execute_with_context(grep_params("SECRET", &outside.display().to_string()), &ctx)
            .await?;
        assert!(!result.success, "path outside all roots should be rejected");
        Ok(())
    }

    /// 相对路径仍基于 working_dir 解析
    #[tokio::test]
    async fn grep_relative_path_still_uses_working_dir() -> Result<()> {
        // 相对路径仍按原有方式基于 working_dir（首选根目录）解析。
        let working = tempfile::tempdir()?;
        std::fs::write(working.path().join("notes.txt"), "TODO fix this\n")?;

        let ctx = echo_core::tools::ToolContext {
            working_dir: Some(working.path().to_path_buf()),
            ..Default::default()
        };
        let tool = GrepTool::new();
        let result = tool
            .execute_with_context(grep_params("TODO", "notes.txt"), &ctx)
            .await?;
        assert!(result.success, "{}", result.error.unwrap_or_default());
        assert!(result.output.contains("fix this"));
        Ok(())
    }
}
