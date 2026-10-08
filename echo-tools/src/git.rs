/*!
 * Git 版本控制工具
 *
 * 对齐 Claude Code/Cursor，提供完整的 Git 操作能力：
 * - git_status：工作区状态
 * - git_diff：差异对比（未暂存 + 已暂存）
 * - git_log：提交历史
 * - git_blame：逐行溯源
 * - git_branch：分支操作
 * - git_commit：创建提交
 */

use futures::future::BoxFuture;
use serde_json::Value;
use std::path::Path;
use std::time::Duration;

use echo_core::error::{Result, ToolError};
use echo_core::tools::pagination::PageRequest;
use echo_core::tools::permission::ToolPermission;
use echo_core::tools::{Tool, ToolContext, ToolParameters, ToolResult, ToolRiskLevel};

/// 单次 git 命令的超时时间
const GIT_TIMEOUT: Duration = Duration::from_secs(60);
/// git 输出的最大字节数，超出部分被截断
const MAX_GIT_OUTPUT_BYTES: usize = 1024 * 1024;

// ── Git 状态 ──────────────────────────────────────────────────────────────

#[derive(Default)]
/// 查看工作区状态的 git_status 工具
pub struct GitStatusTool;

impl Tool for GitStatusTool {
    /// 工具名称
    fn name(&self) -> &str {
        "git_status"
    }

    /// 工具描述
    fn description(&self) -> &str {
        "View working directory status of the current repo: modified, staged, untracked files"
    }

    /// 参数 JSON Schema
    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "repo_path": {
                    "type": "string",
                    "description": "Repository path (defaults to current working directory)"
                }
            },
            "required": []
        })
    }

    /// 工具所需权限
    fn permissions(&self) -> Vec<ToolPermission> {
        vec![ToolPermission::Read]
    }

    /// 工具风险等级
    fn risk_level(&self) -> ToolRiskLevel {
        ToolRiskLevel::ReadOnly
    }

    /// 执行对应的 git 命令
    fn execute_with_context<'a>(
        &'a self,
        parameters: ToolParameters,
        ctx: &'a ToolContext,
    ) -> BoxFuture<'a, Result<ToolResult>> {
        Box::pin(async move {
            let repo_path = parameters
                .get("repo_path")
                .and_then(Value::as_str)
                .unwrap_or(".");
            let repo_path = effective_repo_path(repo_path, ctx);
            let output = run_git(&repo_path, &["status", "--short"], ctx).await?;
            if output.is_empty() {
                Ok(ToolResult::success("Working directory clean, no changes"))
            } else {
                Ok(ToolResult::success(format!("Git status:\n{output}")))
            }
        })
    }
}

// ── Git 差异 ────────────────────────────────────────────────────────────────

/// 查看差异的 git_diff 工具
pub struct GitDiffTool;

impl Tool for GitDiffTool {
    /// 工具名称
    fn name(&self) -> &str {
        "git_diff"
    }

    /// 工具描述
    fn description(&self) -> &str {
        "View code diffs: unstaged changes, staged changes, or diffs between specified branches/commits"
    }

    /// 工具所需权限
    fn permissions(&self) -> Vec<ToolPermission> {
        vec![ToolPermission::Read]
    }
    /// 工具风险等级
    fn risk_level(&self) -> ToolRiskLevel {
        ToolRiskLevel::ReadOnly
    }

    /// 参数 JSON Schema
    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "repo_path": {
                    "type": "string",
                    "description": "Repository path (defaults to current working directory)"
                },
                "staged": {
                    "type": "boolean",
                    "description": "Whether to view staged changes (default false)"
                },
                "target": {
                    "type": "string",
                    "description": "Comparison target: branch name, commit hash, or HEAD~1"
                },
                "file_path": {
                    "type": "string",
                    "description": "Show diff for specified file only"
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "description": "Maximum file/hunk blocks per page (default: 10)"
                },
                "cursor": {
                    "type": "string",
                    "description": "Opaque cursor from page.next_cursor; reuse only with identical parameters"
                }
            },
            "required": []
        })
    }

    /// 执行对应的 git 命令
    fn execute_with_context<'a>(
        &'a self,
        parameters: ToolParameters,
        ctx: &'a echo_core::tools::ToolContext,
    ) -> BoxFuture<'a, Result<ToolResult>> {
        Box::pin(async move {
            let repo_path = parameters
                .get("repo_path")
                .and_then(|v| v.as_str())
                .unwrap_or(".");
            let repo_path = effective_repo_path(repo_path, ctx);
            let page_request = match PageRequest::from_parameters(&parameters, 10, 50) {
                Ok(request) => request,
                Err(error) => return Ok(ToolResult::invalid_arguments(error.to_string())),
            };

            let mut args = vec!["diff"];
            let staged = parameters
                .get("staged")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            if staged {
                args.push("--staged");
            }

            let target = parameters.get("target").and_then(|v| v.as_str());
            let file_path = parameters.get("file_path").and_then(|v| v.as_str());

            if let Some(t) = target {
                if t.starts_with('-') {
                    return Ok(ToolResult::invalid_arguments(
                        "target must be a branch or commit name, not an option",
                    ));
                }
                args.push(t);
            }
            if let Some(fp) = file_path {
                args.push("--");
                args.push(fp);
            }

            let output = run_git(&repo_path, &args, ctx).await?;
            let chunks = crate::diff_pagination::split_unified_diff(&output);
            let query = serde_json::json!({
                "repo_path": repo_path,
                "staged": staged,
                "target": target,
                "file_path": file_path,
            });
            let (page, page_info) = match page_request.paginate(chunks, &query) {
                Ok(page) => page,
                Err(error) => return Ok(ToolResult::invalid_arguments(error.to_string())),
            };
            let output = if page.is_empty() {
                "No differences".to_string()
            } else {
                format!("```diff\n{}\n```", page.join("\n"))
            };
            let mut result = ToolResult::success(output);
            page_info.apply_to(&mut result);
            Ok(result)
        })
    }
}

// ── Git 日志 ─────────────────────────────────────────────────────────────────

/// 查看提交历史的 git_log 工具
pub struct GitLogTool;

impl Tool for GitLogTool {
    /// 工具名称
    fn name(&self) -> &str {
        "git_log"
    }

    /// 工具描述
    fn description(&self) -> &str {
        "View Git commit history, with options for count limit and format"
    }

    /// 工具所需权限
    fn permissions(&self) -> Vec<ToolPermission> {
        vec![ToolPermission::Read]
    }
    /// 工具风险等级
    fn risk_level(&self) -> ToolRiskLevel {
        ToolRiskLevel::ReadOnly
    }

    /// 参数 JSON Schema
    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "repo_path": {
                    "type": "string",
                    "description": "Repository path (defaults to current working directory)"
                },
                "count": {
                    "type": "integer",
                    "description": "Number of commits to show (default 20)"
                },
                "oneline": {
                    "type": "boolean",
                    "description": "Single-line mode (default true)"
                },
                "author": {
                    "type": "string",
                    "description": "Filter by author"
                },
                "since": {
                    "type": "string",
                    "description": "Start date, e.g. '2024-01-01'"
                }
            },
            "required": []
        })
    }

    /// 执行对应的 git 命令
    fn execute_with_context<'a>(
        &'a self,
        parameters: ToolParameters,
        ctx: &'a echo_core::tools::ToolContext,
    ) -> BoxFuture<'a, Result<ToolResult>> {
        Box::pin(async move {
            let repo_path = parameters
                .get("repo_path")
                .and_then(|v| v.as_str())
                .unwrap_or(".");
            let repo_path = effective_repo_path(repo_path, ctx);

            let count = parameters
                .get("count")
                .and_then(|v| v.as_u64())
                .unwrap_or(20);
            let oneline = parameters
                .get("oneline")
                .and_then(|v| v.as_bool())
                .unwrap_or(true);

            let mut args: Vec<&str> = vec!["log"];
            if oneline {
                args.push("--oneline");
            }
            let count_str = count.to_string();
            args.extend_from_slice(&["-n", &count_str]);

            let author = parameters.get("author").and_then(|v| v.as_str());
            let since = parameters.get("since").and_then(|v| v.as_str());
            let mut extra_args: Vec<String> = Vec::new();
            if let Some(a) = author {
                extra_args.push(format!("--author={}", a));
            }
            if let Some(s) = since {
                extra_args.push(format!("--since={}", s));
            }
            let extra_strs: Vec<&str> = extra_args.iter().map(|s| s.as_str()).collect();
            args.extend(&extra_strs);

            let output = run_git(&repo_path, &args, ctx).await?;
            if output.is_empty() {
                Ok(ToolResult::success(
                    "Repository has no commit history".to_string(),
                ))
            } else {
                Ok(ToolResult::success(format!("Commit history:\n{}", output)))
            }
        })
    }
}

// ── Git 溯源 ───────────────────────────────────────────────────────────────

/// 逐行溯源的 git_blame 工具
pub struct GitBlameTool;

impl Tool for GitBlameTool {
    /// 工具名称
    fn name(&self) -> &str {
        "git_blame"
    }

    /// 工具描述
    fn description(&self) -> &str {
        "View line-by-line annotations for a file, showing the last author and commit for each line"
    }

    /// 工具所需权限
    fn permissions(&self) -> Vec<ToolPermission> {
        vec![ToolPermission::Read]
    }
    /// 工具风险等级
    fn risk_level(&self) -> ToolRiskLevel {
        ToolRiskLevel::ReadOnly
    }

    /// 参数 JSON Schema
    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "File path to inspect (relative to repo root)"
                },
                "repo_path": {
                    "type": "string",
                    "description": "Repository path (defaults to current working directory)"
                },
                "start_line": {
                    "type": "integer",
                    "description": "Start line number"
                },
                "end_line": {
                    "type": "integer",
                    "description": "End line number"
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 300,
                    "description": "Maximum blame lines per page (default: 100)"
                },
                "cursor": {
                    "type": "string",
                    "description": "Opaque cursor from page.next_cursor; reuse only with identical parameters"
                }
            },
            "required": ["file_path"]
        })
    }

    /// 执行对应的 git 命令
    fn execute_with_context<'a>(
        &'a self,
        parameters: ToolParameters,
        ctx: &'a echo_core::tools::ToolContext,
    ) -> BoxFuture<'a, Result<ToolResult>> {
        Box::pin(async move {
            let file_path = parameters
                .get("file_path")
                .and_then(|v| v.as_str())
                .ok_or_else(|| ToolError::MissingParameter("file_path".to_string()))?;
            let repo_path = parameters
                .get("repo_path")
                .and_then(|v| v.as_str())
                .unwrap_or(".");
            let repo_path = effective_repo_path(repo_path, ctx);
            let page_request = match PageRequest::from_parameters(&parameters, 100, 300) {
                Ok(request) => request,
                Err(error) => return Ok(ToolResult::invalid_arguments(error.to_string())),
            };

            let mut args: Vec<String> = vec!["blame".to_string()];
            if let (Some(start), Some(end)) = (
                parameters.get("start_line").and_then(|v| v.as_u64()),
                parameters.get("end_line").and_then(|v| v.as_u64()),
            ) {
                args.push("-L".to_string());
                args.push(format!("{},{}", start, end));
            }
            // 在用户提供的文件路径前加 -- 分隔符，防止参数注入
            args.push("--".to_string());
            args.push(file_path.to_string());

            let str_args: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
            let output = run_git(&repo_path, &str_args, ctx).await?;
            let lines = output.lines().map(str::to_string).collect();
            let query = serde_json::json!({
                "repo_path": repo_path,
                "file_path": file_path,
                "start_line": parameters.get("start_line"),
                "end_line": parameters.get("end_line"),
            });
            let (page, page_info) = match page_request.paginate(lines, &query) {
                Ok(page) => page,
                Err(error) => return Ok(ToolResult::invalid_arguments(error.to_string())),
            };
            let mut result = ToolResult::success(page.join("\n"));
            page_info.apply_to(&mut result);
            Ok(result)
        })
    }
}

// ── Git 分支 ──────────────────────────────────────────────────────────────

/// 分支操作的 git_branch 工具
pub struct GitBranchTool;

impl Tool for GitBranchTool {
    /// 工具名称
    fn name(&self) -> &str {
        "git_branch"
    }

    /// 工具描述
    fn description(&self) -> &str {
        "View, create, or switch Git branches. No args lists all branches; with name creates a new branch"
    }

    /// 工具所需权限
    fn permissions(&self) -> Vec<ToolPermission> {
        vec![
            ToolPermission::Read,
            ToolPermission::Write,
            ToolPermission::Execute,
        ]
    }
    /// 工具风险等级
    fn risk_level(&self) -> ToolRiskLevel {
        ToolRiskLevel::Dangerous
    }

    /// 参数 JSON Schema
    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "repo_path": {
                    "type": "string",
                    "description": "Repository path (defaults to current working directory)"
                },
                "name": {
                    "type": "string",
                    "description": "New branch name (creates a branch if provided)"
                },
                "switch": {
                    "type": "string",
                    "description": "Switch to the specified branch"
                },
                "delete": {
                    "type": "string",
                    "description": "Delete the specified branch (must be merged)"
                }
            },
            "required": []
        })
    }

    /// 执行对应的 git 命令
    fn execute_with_context<'a>(
        &'a self,
        parameters: ToolParameters,
        ctx: &'a echo_core::tools::ToolContext,
    ) -> BoxFuture<'a, Result<ToolResult>> {
        Box::pin(async move {
            let repo_path = parameters
                .get("repo_path")
                .and_then(|v| v.as_str())
                .unwrap_or(".");
            let repo_path = effective_repo_path(repo_path, ctx);

            let action;
            let args: Vec<String>;

            if let Some(name) = parameters.get("name").and_then(|v| v.as_str()) {
                // -- 分隔符：以 - 开头的分支名被视为名称而非选项
                args = vec!["branch".to_string(), "--".to_string(), name.to_string()];
                action = format!("Create branch '{}'", name);
            } else if let Some(target) = parameters.get("switch").and_then(|v| v.as_str()) {
                args = vec!["checkout".to_string(), "--".to_string(), target.to_string()];
                action = format!("Switch to branch '{}'", target);
            } else if let Some(target) = parameters.get("delete").and_then(|v| v.as_str()) {
                args = vec![
                    "branch".to_string(),
                    "-d".to_string(),
                    "--".to_string(),
                    target.to_string(),
                ];
                action = format!("Delete branch '{}'", target);
            } else {
                args = vec!["branch".to_string(), "-a".to_string()];
                action = "List all branches".to_string();
            }

            let str_args: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
            let output = run_git(&repo_path, &str_args, ctx).await?;
            Ok(ToolResult::success(format!("{}:\n{}", action, output)))
        })
    }
}

// ── Git 提交 ──────────────────────────────────────────────────────────────

/// 创建提交的 git_commit 工具
pub struct GitCommitTool;

impl Tool for GitCommitTool {
    /// 工具名称
    fn name(&self) -> &str {
        "git_commit"
    }

    /// 工具描述
    fn description(&self) -> &str {
        "Create a Git commit. Files must be staged via git add first. Only call this tool when explicitly requested by the user."
    }

    /// 工具所需权限
    fn permissions(&self) -> Vec<ToolPermission> {
        vec![ToolPermission::Write, ToolPermission::Execute]
    }
    /// 工具风险等级
    fn risk_level(&self) -> ToolRiskLevel {
        ToolRiskLevel::Dangerous
    }

    /// 参数 JSON Schema
    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "repo_path": {
                    "type": "string",
                    "description": "Repository path (defaults to current working directory)"
                },
                "message": {
                    "type": "string",
                    "description": "Commit message"
                },
                "files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of files to stage and commit (empty = commit all staged)"
                }
            },
            "required": ["message"]
        })
    }

    /// 执行对应的 git 命令
    fn execute_with_context<'a>(
        &'a self,
        parameters: ToolParameters,
        ctx: &'a echo_core::tools::ToolContext,
    ) -> BoxFuture<'a, Result<ToolResult>> {
        Box::pin(async move {
            let repo_path = parameters
                .get("repo_path")
                .and_then(|v| v.as_str())
                .unwrap_or(".");
            let repo_path = effective_repo_path(repo_path, ctx);
            let message = parameters
                .get("message")
                .and_then(|v| v.as_str())
                .ok_or_else(|| ToolError::MissingParameter("message".to_string()))?;

            // 如果指定了文件列表，先执行 git add
            if let Some(files) = parameters.get("files").and_then(|v| v.as_array()) {
                for f_val in files {
                    if let Some(f) = f_val.as_str() {
                        /*
                         * `--` 把路径与选项分隔开，使以 `-` 开头的文件名
                         * （例如 `--upload-pack=...`）不会被解释为 git 参数（P1-2 参数注入）。
                         */
                        let add_args = ["add", "--", f];
                        run_git(&repo_path, &add_args, ctx).await?;
                    }
                }
            }

            let commit_args = ["commit", "-m", message];
            let output = run_git(&repo_path, &commit_args, ctx).await?;
            Ok(ToolResult::success(format!(
                "Commit succeeded:\n{}",
                output
            )))
        })
    }
}

// ── 辅助函数 ──────────────────────────────────────────────────────────────────

/**
 * 解析 git 工具调用的有效仓库路径。
 *
 * 当 LLM 未传入 `repo_path`（默认 `"."`）且会话绑定了 `working_dir`（例如 worktree）时，
 * 在该目录中执行 git 命令，使 git 操作作用于隔离的检出目录而非进程 cwd。
 * 用户显式提供的 `repo_path` 始终优先。
 */
fn effective_repo_path(user_repo_path: &str, ctx: &echo_core::tools::ToolContext) -> String {
    /*
     * 设置了 working_dir（worktree 绑定）时，相对路径基于它解析——与文件工具语义一致。
     * 绝对路径和 "." 直接透传（设置了 working_dir 时 "." 解析为 working_dir）。
     */
    if user_repo_path == "." {
        if let Some(dir) = &ctx.working_dir {
            return dir.to_string_lossy().to_string();
        }
        return ".".to_string();
    }
    // 相对路径：若设置了 working_dir 则与之拼接
    if !std::path::Path::new(user_repo_path).is_absolute()
        && let Some(dir) = &ctx.working_dir
    {
        return dir.join(user_repo_path).to_string_lossy().to_string();
    }
    user_repo_path.to_string()
}

/// 在指定仓库路径下执行 git 命令，返回标准输出或错误
async fn run_git(repo_path: &str, args: &[&str], context: &ToolContext) -> Result<String> {
    // 校验 repo_path：拒绝路径穿越并确保路径存在
    let path = Path::new(repo_path);
    if path
        .components()
        .any(|c| c == std::path::Component::ParentDir)
    {
        return Err(ToolError::ExecutionFailed {
            tool: "git".to_string(),
            message: format!(
                "Invalid repo_path: path traversal not allowed: {}",
                repo_path
            ),
        }
        .into());
    }
    if repo_path != "." && !path.is_dir() {
        return Err(ToolError::ExecutionFailed {
            tool: "git".to_string(),
            message: format!(
                "repo_path does not exist or is not a directory: {}",
                repo_path
            ),
        }
        .into());
    }

    let output = crate::process::run_bounded_command(
        "git",
        "git",
        args,
        path,
        context,
        GIT_TIMEOUT,
        MAX_GIT_OUTPUT_BYTES,
    )
    .await?;

    if output.status.success() {
        let mut text = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if output.truncated {
            text.push_str("\n[git output truncated at 1 MiB]");
        }
        Ok(text)
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        Err(ToolError::ExecutionFailed {
            tool: "git".to_string(),
            message: stderr.trim().to_string(),
        }
        .into())
    }
}
