/*!
 * 用于并行子代理隔离的 Git worktree 管理。
 *
 * 当多个子代理在同一仓库中工作时，每个子代理都拥有独立的 worktree，
 * 以避免文件冲突。各 worktree 共享同一个 .git 对象库，因此开销很小。
 */

use serde::{Deserialize, Serialize};
use std::ffi::OsStr;
use std::path::{Path, PathBuf};
#[cfg(test)]
use std::process::Command;
use std::time::Duration;

use echo_core::tools::ToolContext;

use crate::process::BoundedProcessOutput;

/// 标记 worktree 由 echo-agent 创建并管理的元数据文件名。
const MANAGED_MARKER: &str = "echo-agent-managed";
/// 单条 git 命令的超时时间。
const GIT_TIMEOUT: Duration = Duration::from_secs(60);
/// git 命令输出的最大捕获字节数（1 MiB）。
const MAX_GIT_OUTPUT_BYTES: usize = 1024 * 1024;

/// 当前检出位置的引用，用于合并失败后恢复原检出状态。
enum CheckoutRef {
    /// 检出在某个分支上（分支名）。
    Branch(String),
    /// 处于分离 HEAD 状态（提交哈希）。
    Commit(String),
}

/// 新建 worktree 的配置。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorktreeConfig {
    /// worktree 使用的分支名（不存在时自动创建）
    pub branch: String,
    /// 创建时基于的分支/提交（默认：HEAD）
    pub base: Option<String>,
    /// worktree 目录的可选路径后缀
    pub path_suffix: Option<String>,
}

/// 一个受管理的 git worktree。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManagedWorktree {
    /// worktree 目录路径
    pub path: PathBuf,
    /// 分支名
    pub branch: String,
    /// 该 worktree 是否由本程序创建（用于清理追踪）
    pub managed: bool,
}

/**
 * 创建新的 git worktree，用于隔离的并行工作。
 *
 * 返回 worktree 信息。调用方在使用完毕后需负责通过
 * `remove_worktree()` 进行清理。
 */
pub async fn create_worktree(
    repo_path: &Path,
    config: &WorktreeConfig,
) -> Result<ManagedWorktree, String> {
    create_worktree_with_context(repo_path, config, &ToolContext::default()).await
}

/// 使用指定工具上下文创建受管理的 git worktree，并写入归属标记文件。
pub async fn create_worktree_with_context(
    repo_path: &Path,
    config: &WorktreeConfig,
    context: &ToolContext,
) -> Result<ManagedWorktree, String> {
    let git_root = find_git_root(repo_path, context).await?;

    // 生成 worktree 路径
    let worktrees_root = git_root.join(".worktrees");
    let worktree_dir = if let Some(ref suffix) = config.path_suffix {
        echo_core::utils::fs::join_path_segment(&worktrees_root, suffix)
            .map_err(|error| format!("Invalid worktree path suffix: {error}"))?
    } else {
        let branch_safe = config
            .branch
            .replace(|c: char| !c.is_alphanumeric() && c != '-' && c != '_', "_");
        git_root.join(".worktrees").join(&branch_safe)
    };

    // 按需创建 .worktrees 目录
    if let Some(parent) = worktree_dir.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Failed to create worktrees directory: {e}"))?;
    }

    // 构建 git worktree add 命令
    let mut args = vec!["worktree".to_string(), "add".to_string()];
    if let Some(ref base) = config.base {
        args.extend([
            "-b".to_string(),
            config.branch.clone(),
            worktree_dir.to_string_lossy().to_string(),
            "--".to_string(),
            base.clone(),
        ]);
    } else {
        args.extend([
            "-b".to_string(),
            config.branch.clone(),
            worktree_dir.to_string_lossy().to_string(),
        ]);
    }

    let output = run_git(&git_root, &args, context).await?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        // 若分支已存在，则去掉 -b 重试
        if stderr.contains("already exists") {
            let output2 = run_git(
                &git_root,
                &[
                    "worktree".to_string(),
                    "add".to_string(),
                    worktree_dir.to_string_lossy().to_string(),
                    "--".to_string(),
                    config.branch.clone(),
                ],
                context,
            )
            .await?;

            if !output2.status.success() {
                return Err(format!(
                    "git worktree add failed: {}",
                    String::from_utf8_lossy(&output2.stderr)
                ));
            }
        } else {
            return Err(format!("git worktree add failed: {stderr}"));
        }
    }

    let marker = worktree_git_path(&worktree_dir, MANAGED_MARKER, context).await?;
    echo_core::utils::fs::atomic_write(&marker, config.branch.as_bytes()).map_err(|error| {
        format!(
            "Worktree was created but its ownership marker could not be written at {}: {error}",
            marker.display()
        )
    })?;

    Ok(ManagedWorktree {
        path: worktree_dir,
        branch: config.branch.clone(),
        managed: true,
    })
}

/// 移除受管理的 worktree 并完成清理。
pub async fn remove_worktree(repo_path: &Path, worktree: &ManagedWorktree) -> Result<(), String> {
    remove_worktree_with_context(repo_path, worktree, &ToolContext::default()).await
}

/// 使用指定工具上下文移除受管理的 worktree；存在未提交改动时拒绝移除。
pub async fn remove_worktree_with_context(
    repo_path: &Path,
    worktree: &ManagedWorktree,
    context: &ToolContext,
) -> Result<(), String> {
    let git_root = find_git_root(repo_path, context).await?;
    let canonical_worktree = verify_managed_worktree(&git_root, worktree, context).await?;

    let status = run_git(
        &canonical_worktree,
        &["status", "--porcelain=v1", "--untracked-files=all"],
        context,
    )
    .await?;
    if !status.status.success() {
        return Err(format!(
            "Failed to inspect worktree status: {}",
            String::from_utf8_lossy(&status.stderr)
        ));
    }
    if !status.stdout.is_empty() {
        return Err("Refusing to remove a worktree with uncommitted changes".to_string());
    }

    let output = run_git(
        &git_root,
        &[
            "worktree".to_string(),
            "remove".to_string(),
            canonical_worktree.to_string_lossy().to_string(),
        ],
        context,
    )
    .await?;

    if !output.status.success() {
        return Err(format!(
            "git worktree remove failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    Ok(())
}

/// 列出仓库中的所有 worktree。
pub async fn list_worktrees(repo_path: &Path) -> Result<Vec<ManagedWorktree>, String> {
    list_worktrees_with_context(repo_path, &ToolContext::default()).await
}

/// 使用指定工具上下文列出仓库中的所有 worktree，并识别其是否受管理。
pub async fn list_worktrees_with_context(
    repo_path: &Path,
    context: &ToolContext,
) -> Result<Vec<ManagedWorktree>, String> {
    let git_root = find_git_root(repo_path, context).await?;

    let output = run_git(&git_root, &["worktree", "list", "--porcelain"], context).await?;
    if !output.status.success() {
        return Err(command_error(&output));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut worktrees = Vec::new();
    let mut current_path: Option<PathBuf> = None;
    let mut current_branch = String::new();

    for line in stdout.lines() {
        if let Some(path) = line.strip_prefix("worktree ") {
            current_path = Some(PathBuf::from(path));
            current_branch.clear();
        } else if let Some(branch) = line.strip_prefix("branch ") {
            current_branch = branch.replace("refs/heads/", "");
        } else if line.is_empty()
            && let Some(path) = current_path.take()
        {
            let managed = worktree_git_path(&path, MANAGED_MARKER, context)
                .await
                .is_ok_and(|marker| marker.is_file());
            worktrees.push(ManagedWorktree {
                path,
                branch: current_branch.clone(),
                managed,
            });
        }
    }
    // 处理最后一个条目
    if let Some(path) = current_path {
        let managed = worktree_git_path(&path, MANAGED_MARKER, context)
            .await
            .is_ok_and(|marker| marker.is_file());
        worktrees.push(ManagedWorktree {
            path,
            branch: current_branch,
            managed,
        });
    }

    Ok(worktrees)
}

/// 将 worktree 分支上的改动合并回目标分支。
pub async fn merge_worktree(
    repo_path: &Path,
    worktree: &ManagedWorktree,
    target_branch: &str,
) -> Result<String, String> {
    merge_worktree_with_context(repo_path, worktree, target_branch, &ToolContext::default()).await
}

/**
 * 使用指定工具上下文将 worktree 分支合并到目标分支。
 *
 * 合并失败时会中止合并并恢复原检出位置。
 */
pub async fn merge_worktree_with_context(
    repo_path: &Path,
    worktree: &ManagedWorktree,
    target_branch: &str,
    context: &ToolContext,
) -> Result<String, String> {
    let git_root = find_git_root(repo_path, context).await?;
    verify_managed_worktree(&git_root, worktree, context).await?;
    ensure_clean_checkout(&git_root, context).await?;
    let original_ref = current_checkout(&git_root, context).await?;
    let merge_in_progress = run_git(
        &git_root,
        &["rev-parse", "--verify", "-q", "MERGE_HEAD"],
        context,
    )
    .await?;
    if merge_in_progress.status.success() {
        return Err("Refusing to start a worktree merge while another merge is active".to_string());
    }

    let co = run_git(&git_root, &["switch", "--", target_branch], context).await?;

    if !co.status.success() {
        return Err(format!(
            "Failed to checkout {}: {}",
            target_branch,
            String::from_utf8_lossy(&co.stderr)
        ));
    }

    let merge = match run_git(
        &git_root,
        &[
            "-c",
            "commit.gpgsign=false",
            "merge",
            "--no-edit",
            &worktree.branch,
            "--",
        ],
        context,
    )
    .await
    {
        Ok(output) => output,
        Err(error) => {
            restore_checkout(&git_root, &original_ref, context)
                .await
                .map_err(|restore_error| {
                    format!(
                        "Failed to start merge ({error}); checkout restoration failed: {restore_error}"
                    )
                })?;
            return Err(format!("Failed to start merge: {error}"));
        }
    };

    if !merge.status.success() {
        let merge_error = String::from_utf8_lossy(&merge.stderr).to_string();
        abort_merge_if_active(&git_root, &merge_error, context).await?;
        restore_checkout(&git_root, &original_ref, context)
            .await
            .map_err(|restore_error| {
                format!("Merge failed ({merge_error}); merge was aborted but checkout restoration failed: {restore_error}")
            })?;
        return Err(format!("Merge conflict or error: {merge_error}"));
    }

    Ok(format!("Merged {} into {}", worktree.branch, target_branch))
}

/// 校验 worktree 确实由 echo-agent 管理且位于 .worktrees 下，返回其规范化路径。
async fn verify_managed_worktree(
    git_root: &Path,
    worktree: &ManagedWorktree,
    context: &ToolContext,
) -> Result<PathBuf, String> {
    if !worktree.managed {
        return Err("Refusing to operate on a worktree not owned by echo-agent".to_string());
    }
    let canonical_root = std::fs::canonicalize(git_root.join(".worktrees"))
        .map_err(|error| format!("Failed to resolve worktrees root: {error}"))?;
    let canonical_worktree = std::fs::canonicalize(&worktree.path)
        .map_err(|error| format!("Failed to resolve worktree path: {error}"))?;
    if !canonical_worktree.starts_with(&canonical_root) {
        return Err(format!(
            "Refusing to operate on worktree outside {}: {}",
            canonical_root.display(),
            canonical_worktree.display()
        ));
    }
    let marker = worktree_git_path(&canonical_worktree, MANAGED_MARKER, context).await?;
    let marker_branch = std::fs::read_to_string(&marker)
        .map_err(|error| format!("Worktree ownership marker is missing or unreadable: {error}"))?;
    let actual_branch =
        run_git(&canonical_worktree, &["branch", "--show-current"], context).await?;
    if !actual_branch.status.success() {
        return Err(String::from_utf8_lossy(&actual_branch.stderr).to_string());
    }
    let actual_branch = String::from_utf8_lossy(&actual_branch.stdout);
    if marker_branch.trim() != worktree.branch || actual_branch.trim() != worktree.branch {
        return Err(
            "Worktree ownership metadata does not match its checked-out branch".to_string(),
        );
    }
    Ok(canonical_worktree)
}

/// 确保目标检出没有未提交的改动。
async fn ensure_clean_checkout(git_root: &Path, context: &ToolContext) -> Result<(), String> {
    let status = run_git(
        git_root,
        &["status", "--porcelain=v1", "--untracked-files=all"],
        context,
    )
    .await?;
    if !status.status.success() {
        return Err(String::from_utf8_lossy(&status.stderr).to_string());
    }
    if status.stdout.is_empty() {
        Ok(())
    } else {
        Err("Refusing to merge while the target checkout has uncommitted changes".to_string())
    }
}

/// 若存在进行中的合并（MERGE_HEAD），则执行 `git merge --abort`。
async fn abort_merge_if_active(
    git_root: &Path,
    merge_error: &str,
    context: &ToolContext,
) -> Result<(), String> {
    let active = run_git(
        git_root,
        &["rev-parse", "--verify", "-q", "MERGE_HEAD"],
        context,
    )
    .await
    .map_err(|error| {
        format!("Merge failed ({merge_error}); merge-state inspection failed: {error}")
    })?;
    if !active.status.success() {
        return Ok(());
    }
    let abort = run_git(git_root, &["merge", "--abort"], context)
        .await
        .map_err(|error| format!("Merge failed ({merge_error}); abort could not start: {error}"))?;
    if abort.status.success() {
        Ok(())
    } else {
        Err(format!(
            "Merge failed ({merge_error}); abort also failed: {}",
            String::from_utf8_lossy(&abort.stderr)
        ))
    }
}

/// 解析 worktree 内指定 git 元数据文件的绝对路径。
async fn worktree_git_path(
    worktree_path: &Path,
    name: &str,
    context: &ToolContext,
) -> Result<PathBuf, String> {
    let output = run_git(
        worktree_path,
        &["rev-parse", "--path-format=absolute", "--git-path", name],
        context,
    )
    .await?;
    if !output.status.success() {
        return Err(format!(
            "Failed to resolve worktree metadata path: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let path = String::from_utf8(output.stdout)
        .map_err(|error| format!("Worktree metadata path is not UTF-8: {error}"))?;
    Ok(PathBuf::from(path.trim()))
}

/// 获取当前检出位置（分支或分离 HEAD 提交）。
async fn current_checkout(git_root: &Path, context: &ToolContext) -> Result<CheckoutRef, String> {
    let branch = run_git(
        git_root,
        &["symbolic-ref", "--quiet", "--short", "HEAD"],
        context,
    )
    .await?;
    if branch.status.success() {
        return String::from_utf8(branch.stdout)
            .map(|value| CheckoutRef::Branch(value.trim().to_string()))
            .map_err(|error| format!("Current branch is not UTF-8: {error}"));
    }
    let commit = run_git(git_root, &["rev-parse", "HEAD"], context).await?;
    if !commit.status.success() {
        return Err(String::from_utf8_lossy(&commit.stderr).to_string());
    }
    String::from_utf8(commit.stdout)
        .map(|value| CheckoutRef::Commit(value.trim().to_string()))
        .map_err(|error| format!("Detached HEAD is not UTF-8: {error}"))
}

/// 恢复到之前记录的检出位置。
async fn restore_checkout(
    git_root: &Path,
    checkout: &CheckoutRef,
    context: &ToolContext,
) -> Result<(), String> {
    let args = match checkout {
        CheckoutRef::Branch(branch) => vec!["switch", "--", branch.as_str()],
        CheckoutRef::Commit(commit) => vec!["switch", "--detach", commit.as_str()],
    };
    let output = run_git(git_root, &args, context).await?;
    if output.status.success() {
        Ok(())
    } else {
        Err(String::from_utf8_lossy(&output.stderr).to_string())
    }
}

/// 查找给定路径所在 git 仓库的根目录。
async fn find_git_root(path: &Path, context: &ToolContext) -> Result<PathBuf, String> {
    let output = run_git(path, &["rev-parse", "--show-toplevel"], context).await?;

    if output.status.success() {
        Ok(PathBuf::from(
            String::from_utf8_lossy(&output.stdout).trim(),
        ))
    } else {
        Err("Not a git repository".to_string())
    }
}

/// 在指定目录下以受限（超时、输出上限）方式运行 git 命令。
async fn run_git<S: AsRef<OsStr>>(
    working_dir: &Path,
    args: &[S],
    context: &ToolContext,
) -> Result<BoundedProcessOutput, String> {
    crate::process::run_bounded_command(
        "git_worktree",
        "git",
        args,
        working_dir,
        context,
        GIT_TIMEOUT,
        MAX_GIT_OUTPUT_BYTES,
    )
    .await
    .map_err(|error| error.to_string())
}

/// 从命令输出中提取错误信息，输出被截断时附加提示。
fn command_error(output: &BoundedProcessOutput) -> String {
    let mut error = String::from_utf8_lossy(&output.stderr).trim().to_string();
    if output.truncated {
        error.push_str("\n[git output truncated at 1 MiB]");
    }
    error
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 测试辅助：同步运行 git 命令并返回标准输出。
    fn run_git(repo: &Path, args: &[&str]) -> Result<String, String> {
        let output = Command::new("git")
            .args(args)
            .current_dir(repo)
            .output()
            .map_err(|error| error.to_string())?;
        if !output.status.success() {
            return Err(String::from_utf8_lossy(&output.stderr).to_string());
        }
        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    }

    /// 测试辅助：创建带有初始提交的临时仓库。
    fn test_repo() -> Result<tempfile::TempDir, String> {
        let dir = tempfile::tempdir().map_err(|error| error.to_string())?;
        run_git(dir.path(), &["init", "-b", "main"])?;
        run_git(dir.path(), &["config", "user.email", "test@example.com"])?;
        run_git(dir.path(), &["config", "user.name", "Test User"])?;
        std::fs::write(dir.path().join("value.txt"), "base\n")
            .map_err(|error| error.to_string())?;
        run_git(dir.path(), &["add", "value.txt"])?;
        run_git(
            dir.path(),
            &["-c", "commit.gpgsign=false", "commit", "-m", "base"],
        )?;
        Ok(dir)
    }

    /// 测试能在当前 crate 目录找到 git 根目录。
    #[tokio::test]
    async fn test_find_git_root() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"));
        let root = find_git_root(path, &ToolContext::default()).await;
        assert!(root.is_ok());
    }

    /// 测试在非仓库目录查找 git 根目录不会 panic。
    #[tokio::test]
    async fn test_find_git_root_non_repo() {
        let path = Path::new("/tmp");
        let root = find_git_root(path, &ToolContext::default()).await;
        // 是否失败取决于 /tmp 是否位于某个 git 仓库中
        let _ = root;
    }

    /// 测试非受管理的 worktree 不允许被移除。
    #[tokio::test]
    async fn unmanaged_worktree_is_not_removable() -> Result<(), String> {
        let repo = test_repo()?;
        let path = repo.path().join(".worktrees").join("external");
        std::fs::create_dir_all(path.parent().ok_or_else(|| "missing parent".to_string())?)
            .map_err(|error| error.to_string())?;
        run_git(
            repo.path(),
            &[
                "worktree",
                "add",
                "-b",
                "external",
                path.to_string_lossy().as_ref(),
            ],
        )?;
        let worktree = ManagedWorktree {
            path,
            branch: "external".to_string(),
            managed: false,
        };
        assert!(remove_worktree(repo.path(), &worktree).await.is_err());
        Ok(())
    }

    /// 测试合并冲突时会中止合并并恢复原分支。
    #[tokio::test]
    async fn merge_conflict_aborts_and_restores_original_branch() -> Result<(), String> {
        let repo = test_repo()?;
        let worktree = create_worktree(
            repo.path(),
            &WorktreeConfig {
                branch: "feature".to_string(),
                base: None,
                path_suffix: Some("feature".to_string()),
            },
        )
        .await?;
        std::fs::write(worktree.path.join("value.txt"), "feature\n")
            .map_err(|error| error.to_string())?;
        run_git(&worktree.path, &["add", "value.txt"])?;
        run_git(
            &worktree.path,
            &["-c", "commit.gpgsign=false", "commit", "-m", "feature"],
        )?;

        run_git(repo.path(), &["switch", "-c", "target"])?;
        std::fs::write(repo.path().join("value.txt"), "target\n")
            .map_err(|error| error.to_string())?;
        run_git(repo.path(), &["add", "value.txt"])?;
        run_git(
            repo.path(),
            &["-c", "commit.gpgsign=false", "commit", "-m", "target"],
        )?;
        run_git(repo.path(), &["switch", "main"])?;

        assert!(
            merge_worktree(repo.path(), &worktree, "target")
                .await
                .is_err()
        );
        assert_eq!(run_git(repo.path(), &["branch", "--show-current"])?, "main");
        assert!(run_git(repo.path(), &["rev-parse", "--verify", "-q", "MERGE_HEAD"]).is_err());
        assert_eq!(
            std::fs::read_to_string(repo.path().join("value.txt"))
                .map_err(|error| error.to_string())?,
            "base\n"
        );
        Ok(())
    }

    /// 测试合并会拒绝伪造的受管理标记以及存在未提交改动的检出。
    #[tokio::test]
    async fn merge_rejects_forged_managed_flag_and_dirty_checkout() -> Result<(), String> {
        let repo = test_repo()?;
        let external_path = repo.path().join(".worktrees").join("external");
        std::fs::create_dir_all(
            external_path
                .parent()
                .ok_or_else(|| "missing parent".to_string())?,
        )
        .map_err(|error| error.to_string())?;
        run_git(
            repo.path(),
            &[
                "worktree",
                "add",
                "-b",
                "external",
                external_path.to_string_lossy().as_ref(),
            ],
        )?;
        let forged = ManagedWorktree {
            path: external_path,
            branch: "external".to_string(),
            managed: true,
        };
        assert!(merge_worktree(repo.path(), &forged, "main").await.is_err());

        let managed = create_worktree(
            repo.path(),
            &WorktreeConfig {
                branch: "feature".to_string(),
                base: None,
                path_suffix: Some("feature".to_string()),
            },
        )
        .await?;
        std::fs::write(repo.path().join("untracked.txt"), "preserve\n")
            .map_err(|error| error.to_string())?;
        assert!(merge_worktree(repo.path(), &managed, "main").await.is_err());
        assert_eq!(
            std::fs::read_to_string(repo.path().join("untracked.txt"))
                .map_err(|error| error.to_string())?,
            "preserve\n"
        );
        Ok(())
    }
}
