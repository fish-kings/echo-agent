/*!
 * Glob 工具——按名称模式查找文件
 *
 * 遍历目录树并返回匹配 glob 模式的文件路径。
 */

use echo_core::error::{Result, ToolError};
use echo_core::tools::pagination::PageRequest;
use echo_core::tools::{Tool, ToolParameters, ToolResult};
use futures::future::BoxFuture;
use serde_json::{Value, json};
use std::path::PathBuf;
use tokio::fs;

// ── GlobTool ────────────────────────────────────────────────────────────────

/**
 * 按名称模式查找文件。
 *
 * 从 `path`（默认 `.`）开始遍历目录树，返回匹配 `pattern` glob 的文件路径
 * （例如递归匹配所有 `.rs` 文件，或 `src` 下所有 `.ts` 文件）。
 */
pub struct GlobTool {
    /// 构造时指定的基础目录；设置后搜索路径被限制在该目录内
    base_dir: Option<PathBuf>,
}

impl GlobTool {
    /// 创建不限定基础目录的 Glob 工具
    pub fn new() -> Self {
        Self { base_dir: None }
    }

    /// 创建限定在指定基础目录内的 Glob 工具
    pub fn with_base_dir(base: impl Into<PathBuf>) -> Self {
        Self {
            base_dir: Some(base.into()),
        }
    }
}

impl Default for GlobTool {
    /// 等价于 [`GlobTool::new`]
    fn default() -> Self {
        Self::new()
    }
}

impl Tool for GlobTool {
    /// 工具名称
    fn name(&self) -> &str {
        "glob"
    }

    /// 工具描述
    fn description(&self) -> &str {
        "Find files by name pattern. Returns file paths matching a glob pattern (e.g. '**/*.rs', 'src/**/*.ts', '*.md'). Supports ** for recursive directory matching."
    }

    /// 所需权限：只读
    fn permissions(&self) -> Vec<echo_core::tools::permission::ToolPermission> {
        vec![echo_core::tools::permission::ToolPermission::Read]
    }
    /// 风险等级：只读
    fn risk_level(&self) -> echo_core::tools::ToolRiskLevel {
        echo_core::tools::ToolRiskLevel::ReadOnly
    }

    /// 参数 JSON Schema
    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern to match file names (e.g. '**/*.rs', 'src/**/*.ts', '*.md')"
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search in (default: current directory)"
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 200,
                    "description": "Results per page (default 100)"
                },
                "cursor": {
                    "type": "string",
                    "description": "Cursor from the previous page"
                }
            },
            "required": ["pattern"]
        })
    }

    /// 执行 glob 搜索并分页返回匹配的文件路径
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

            let page_request = match PageRequest::from_parameters(&parameters, 100, 200) {
                Ok(request) => request,
                Err(error) => return Ok(ToolResult::invalid_arguments(error.to_string())),
            };

            /*
             * 有效基础目录：构造时的 base_dir（带范围限制）优先；
             * 否则回退到运行时的 working_dir（仅覆盖 CWD，不做范围检查）。
             */
            let effective_base: Option<PathBuf> = self
                .base_dir
                .clone()
                .or_else(|| ctx.working_dir.as_ref().map(|p| p.to_path_buf()));

            let search_path = if let Some(ref base) = effective_base {
                let resolved = if std::path::Path::new(path_str).is_absolute() {
                    PathBuf::from(path_str)
                } else {
                    base.join(path_str)
                };
                if !resolved.starts_with(base) {
                    return Ok(ToolResult::invalid_arguments(format!(
                        "Path '{}' is outside the allowed directory scope",
                        path_str
                    )));
                }
                resolved
            } else {
                PathBuf::from(path_str)
            };

            if !search_path.exists() {
                return Ok(ToolResult::invalid_arguments(format!(
                    "Path does not exist: {}",
                    search_path.display()
                )));
            }

            if !search_path.is_dir() {
                return Ok(ToolResult::invalid_arguments(format!(
                    "Path is not a directory: {}",
                    search_path.display()
                )));
            }

            let mut results = Vec::new();
            walk_glob(&search_path, pattern_str, &mut results).await;
            let query = serde_json::json!({
                "pattern": pattern_str,
                "path": search_path,
            });
            let (page, page_info) = match page_request.paginate(results, &query) {
                Ok(page) => page,
                Err(error) => return Ok(ToolResult::invalid_arguments(error.to_string())),
            };
            let output = if page.is_empty() {
                format!("No files matching '{pattern_str}' found.")
            } else {
                format!(
                    "{}\n\n{} files returned ({} total)",
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

/// 递归遍历目录，把匹配模式的文件路径收集到 `results`
async fn walk_glob(dir: &std::path::Path, pattern: &str, results: &mut Vec<String>) {
    let mut entries = match fs::read_dir(dir).await {
        Ok(e) => e,
        Err(_) => return,
    };

    let mut entry_paths: Vec<PathBuf> = Vec::new();
    while let Ok(Some(entry)) = entries.next_entry().await {
        entry_paths.push(entry.path());
    }
    entry_paths.sort();

    for path in entry_paths {
        // 跳过隐藏目录和常见的忽略目录
        if path.is_dir()
            && let Some(name) = path.file_name().and_then(|n| n.to_str())
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

        if path.is_dir() {
            Box::pin(walk_glob(&path, pattern, results)).await;
        } else if path.is_file()
            && let Some(name) = path.file_name().and_then(|n| n.to_str())
            && glob_pattern_matches(pattern, name, &path)
        {
            results.push(path.display().to_string());
        }
    }
}

/**
 * 用 glob 模式匹配文件名。
 *
 * 支持：
 * - `*` 匹配任意字符序列（不含路径分隔符）
 * - `**` 匹配任意路径前缀（由遍历处理）
 * - `?` 匹配任意单个字符
 * - `{a,b}` 匹配 a 或 b
 * - `[abc]` 匹配集合中的任意字符
 */
fn glob_pattern_matches(pattern: &str, name: &str, full_path: &std::path::Path) -> bool {
    // 处理 ** 模式（递归）——可在任意目录层级匹配
    if pattern.contains("**") {
        return glob_with_doublestar(pattern, full_path);
    }

    // 简单模式只匹配文件名
    let name_pattern = if pattern.contains('/') {
        // 模式包含 / 时，按相对路径组件匹配
        pattern.to_string()
    } else {
        pattern.to_string()
    };

    if pattern.contains('/') {
        // 按路径后缀匹配
        glob_match_advanced(&name_pattern, &full_path.display().to_string())
    } else {
        glob_match_advanced(&name_pattern, name)
    }
}

/// 处理包含 `**` 的模式，按路径前缀/后缀进行匹配
fn glob_with_doublestar(pattern: &str, full_path: &std::path::Path) -> bool {
    let path_str = full_path.display().to_string();

    // 以 ** 拆分模式，并尝试在每一层目录匹配
    let parts: Vec<&str> = pattern.split("**").collect();

    if parts.len() == 2 {
        let Some(prefix) = parts.first().copied() else {
            return false;
        };
        let Some(suffix) = parts.get(1).copied() else {
            return false;
        };

        // 尝试在每一层匹配
        if prefix.is_empty() && suffix.is_empty() {
            return true;
        }

        if prefix.is_empty() {
            // **/*.rs——匹配任何以后缀结尾的路径
            return path_str.ends_with(suffix.trim_start_matches('/'))
                || path_str.ends_with(suffix.trim_start_matches('/').trim_start_matches('\\'));
        }

        if suffix.is_empty() {
            return path_str.contains(prefix.trim_end_matches('/'));
        }

        // src/**/*.rs——匹配以前缀开头且以后缀结尾的路径
        let suffix_clean = suffix.trim_start_matches('/');
        let prefix_clean = prefix.trim_end_matches('/');
        if let Some(pos) = path_str.find(prefix_clean)
            && let Some(rest) = path_str
                .get(pos..)
                .and_then(|tail| tail.strip_prefix(prefix_clean))
        {
            return rest.ends_with(suffix_clean);
        }
        return false;
    }

    // 回退：简单匹配
    glob_match_advanced(pattern, &path_str)
}

/// 支持 * ? {a,b} [abc] 的高级 glob 匹配
fn glob_match_advanced(pattern: &str, text: &str) -> bool {
    let mut pi = 0;
    let mut ti = 0;
    let p_chars: Vec<char> = pattern.chars().collect();
    let t_chars: Vec<char> = text.chars().collect();

    while let (Some(pattern_character), Some(text_character)) =
        (p_chars.get(pi).copied(), t_chars.get(ti).copied())
    {
        match pattern_character {
            '*' => {
                // 匹配任意序列
                pi = pi.saturating_add(1);
                if pi == p_chars.len() {
                    return true;
                }
                for skip in ti..=t_chars.len() {
                    /*
                     * UTF-8 安全切片：pi/skip 是字符索引（来自 p_chars/t_chars），
                     * 而非字节索引。使用 &pattern[pi..] 会在多字节字符上 panic
                     * （例如 screenshot_北京.png 这类路径）。因此通过字符迭代器重建。
                     */
                    let pat_rest: String = p_chars.get(pi..).unwrap_or_default().iter().collect();
                    let txt_rest: String = t_chars.get(skip..).unwrap_or_default().iter().collect();
                    if glob_match_advanced(&pat_rest, &txt_rest) {
                        return true;
                    }
                }
                return false;
            }
            '?' => {
                pi = pi.saturating_add(1);
                ti = ti.saturating_add(1);
            }
            '[' => {
                // 字符类
                pi = pi.saturating_add(1);
                let mut found = false;
                let negated = p_chars.get(pi).is_some_and(|character| *character == '!');
                if negated {
                    pi = pi.saturating_add(1);
                }
                while p_chars.get(pi).is_some_and(|character| *character != ']') {
                    if p_chars.get(pi) == t_chars.get(ti) {
                        found = true;
                    }
                    pi = pi.saturating_add(1);
                }
                if pi < p_chars.len() {
                    pi = pi.saturating_add(1); // 跳过 ]
                }
                if negated {
                    found = !found;
                }
                if !found {
                    return false;
                }
                ti = ti.saturating_add(1);
            }
            '{' => {
                // 花括号展开
                let close = p_chars
                    .get(pi..)
                    .and_then(|remainder| remainder.iter().position(|character| *character == '}'));
                if let Some(close_offset) = close {
                    let close_index = pi.saturating_add(close_offset);
                    let brace_content: String = p_chars
                        .get(pi.saturating_add(1)..close_index)
                        .unwrap_or_default()
                        .iter()
                        .collect();
                    let alternatives: Vec<&str> = brace_content.split(',').collect();
                    let after_brace: String = p_chars
                        .get(close_index.saturating_add(1)..)
                        .unwrap_or_default()
                        .iter()
                        .collect();
                    let before_brace: String =
                        p_chars.get(..pi).unwrap_or_default().iter().collect();
                    for alt in alternatives {
                        let full = format!("{}{}{}", before_brace, alt.trim(), after_brace);
                        if glob_match_advanced(&full, text) {
                            return true;
                        }
                    }
                    return false;
                }
                // 没有闭合花括号，按字面量处理
                if pattern_character == text_character {
                    pi = pi.saturating_add(1);
                    ti = ti.saturating_add(1);
                } else {
                    return false;
                }
            }
            c => {
                if c == text_character {
                    pi = pi.saturating_add(1);
                    ti = ti.saturating_add(1);
                } else {
                    return false;
                }
            }
        }
    }

    // 消耗模式末尾的 *
    while p_chars.get(pi).is_some_and(|character| *character == '*') {
        pi = pi.saturating_add(1);
    }

    pi == p_chars.len() && ti == t_chars.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 测试 * 通配
    #[test]
    fn test_glob_match_star() {
        assert!(glob_match_advanced("*.rs", "main.rs"));
        assert!(glob_match_advanced("*.rs", "lib.rs"));
        assert!(!glob_match_advanced("*.rs", "main.py"));
    }

    /// 测试花括号展开
    #[test]
    fn test_glob_match_brace() {
        assert!(glob_match_advanced("*.{rs,ts}", "main.rs"));
        assert!(glob_match_advanced("*.{rs,ts}", "app.ts"));
        assert!(!glob_match_advanced("*.{rs,ts}", "main.py"));
    }

    /// 测试 ? 单字符匹配
    #[test]
    fn test_glob_match_question() {
        assert!(glob_match_advanced("?.rs", "a.rs"));
        assert!(!glob_match_advanced("?.rs", "ab.rs"));
    }

    /// 测试字符类匹配
    #[test]
    fn test_glob_match_bracket() {
        assert!(glob_match_advanced("[abc].rs", "a.rs"));
        assert!(!glob_match_advanced("[abc].rs", "d.rs"));
    }

    /// 测试精确匹配
    #[test]
    fn test_glob_match_exact() {
        assert!(glob_match_advanced("main.rs", "main.rs"));
        assert!(!glob_match_advanced("main.rs", "lib.rs"));
    }

    /// 测试包含多字节字符的路径匹配
    #[test]
    fn test_glob_match_multibyte_path() {
        /*
         * 回归测试：包含多字节 UTF-8 字符（如中文）的路径不得 panic。
         * 此前 `&pattern[pi..]` / `&text[skip..]` 用字符索引做字节切片，
         * 会在 screenshot_北京.png 这类路径上 panic。
         */
        assert!(glob_match_advanced("*.png", "screenshot_北京.png"));
        assert!(glob_match_advanced("screenshot_*", "screenshot_北京.png"));
        assert!(glob_match_advanced(
            "*/北京*",
            "/Users/ls/照片/北京/景山.png"
        ));
        assert!(!glob_match_advanced("*.rs", "screenshot_北京.png"));
        // 健全性检查：修复后纯 ASCII 仍然正常。
        assert!(glob_match_advanced("*.png", "screenshot_shanghai.png"));
    }
}
