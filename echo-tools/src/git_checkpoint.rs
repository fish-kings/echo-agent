//! 按文件粒度的 Git 工作区检查点，在破坏性文件修改之前创建。

use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};

/// 检查点序号，保证同一进程内生成的 ID 唯一
static CHECKPOINT_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// 文件在 Git index 中的条目
#[derive(Serialize, Deserialize)]
struct IndexEntry {
    /// 文件模式（如 100644）
    mode: String,
    /// 对象 ID
    object_id: String,
    /// 暂存阶段号
    stage: String,
}

/// 检查点清单，记录恢复文件所需的信息
#[derive(Serialize, Deserialize)]
struct CheckpointManifest {
    /// 文件相对 Git 根目录的路径
    relative_path: String,
    /// 创建检查点时工作区中该文件是否存在
    worktree_existed: bool,
    /// 文件权限模式（仅 Unix）
    #[cfg(unix)]
    worktree_mode: Option<u32>,
    /// 修改前的 index 条目，None 表示未被跟踪
    index_entry: Option<IndexEntry>,
}

/**
 * 为单个文件快照其工作区中的精确字节内容及 index 条目。
 *
 * `Ok(None)` 表示目标不在 Git 仓库中。一旦找到 Git 根目录，
 * 任何快照失败都会返回错误，以便调用方中止待执行的破坏性修改。
 */
pub fn create_checkpoint(file_path: &Path) -> Result<Option<String>, String> {
    let Some(git_root) = find_git_root(file_path)? else {
        return Ok(None);
    };
    let target = fs::canonicalize(file_path)
        .map_err(|error| format!("failed to resolve checkpoint target: {error}"))?;
    let relative = target.strip_prefix(&git_root).map_err(|_| {
        format!(
            "checkpoint target '{}' is outside Git root '{}'",
            target.display(),
            git_root.display()
        )
    })?;
    let relative_path = relative
        .to_str()
        .ok_or_else(|| "checkpoint path is not valid UTF-8".to_string())?
        .to_string();
    let id = checkpoint_id();
    let checkpoint_dir = checkpoint_root(&git_root)?.join(&id);
    fs::create_dir_all(&checkpoint_dir)
        .map_err(|error| format!("failed to create checkpoint directory: {error}"))?;

    let result = (|| {
        let bytes = fs::read(&target)
            .map_err(|error| format!("failed to read checkpoint target: {error}"))?;
        echo_core::utils::fs::atomic_write(&checkpoint_dir.join("worktree.bin"), &bytes)
            .map_err(|error| format!("failed to persist checkpoint bytes: {error}"))?;
        let manifest = CheckpointManifest {
            relative_path,
            worktree_existed: true,
            #[cfg(unix)]
            worktree_mode: file_mode(&target)?,
            index_entry: read_index_entry(&git_root, relative)?,
        };
        let encoded = serde_json::to_vec(&manifest)
            .map_err(|error| format!("failed to encode checkpoint manifest: {error}"))?;
        echo_core::utils::fs::atomic_write(&checkpoint_dir.join("manifest.json"), &encoded)
            .map_err(|error| format!("failed to persist checkpoint manifest: {error}"))?;
        Ok(Some(id))
    })();
    if result.is_err() {
        let _ = fs::remove_dir_all(&checkpoint_dir);
    }
    result
}

/**
 * 仅恢复 `checkpoint_id` 所对应的文件，包括其修改前的精确 Git index 条目。
 * 其他工作区变更和已暂存的变更不受影响。
 */
pub fn rollback_to_checkpoint(file_path: &Path, checkpoint_id: &str) -> Result<(), String> {
    echo_core::utils::fs::validate_path_segment(checkpoint_id)
        .map_err(|error| format!("invalid checkpoint id: {error}"))?;
    let git_root = find_git_root(file_path)?
        .ok_or_else(|| "checkpoint target is not inside a Git repository".to_string())?;
    let checkpoint_dir = checkpoint_root(&git_root)?.join(checkpoint_id);
    let manifest_bytes = fs::read(checkpoint_dir.join("manifest.json"))
        .map_err(|error| format!("failed to read checkpoint manifest: {error}"))?;
    let manifest: CheckpointManifest = serde_json::from_slice(&manifest_bytes)
        .map_err(|error| format!("invalid checkpoint manifest: {error}"))?;
    let target = git_root.join(&manifest.relative_path);
    let expected = absolute_target(file_path)?;
    if target != expected {
        return Err("checkpoint does not belong to the requested file".to_string());
    }

    if manifest.worktree_existed {
        let bytes = fs::read(checkpoint_dir.join("worktree.bin"))
            .map_err(|error| format!("failed to read checkpoint bytes: {error}"))?;
        echo_core::utils::fs::atomic_write(&target, &bytes)
            .map_err(|error| format!("failed to restore checkpoint bytes: {error}"))?;
        #[cfg(unix)]
        if let Some(mode) = manifest.worktree_mode {
            set_file_mode(&target, mode)?;
        }
    } else if target.exists() {
        fs::remove_file(&target)
            .map_err(|error| format!("failed to remove restored-absent file: {error}"))?;
    }
    restore_index_entry(
        &git_root,
        &manifest.relative_path,
        manifest.index_entry.as_ref(),
    )
}

/// 删除较旧的单文件检查点，保留最新的 `keep` 个。
pub fn cleanup_old_checkpoints(file_path: &Path, keep: usize) {
    let Ok(Some(git_root)) = find_git_root(file_path) else {
        return;
    };
    let Ok(root) = checkpoint_root(&git_root) else {
        return;
    };
    let Ok(entries) = fs::read_dir(root) else {
        return;
    };
    let mut entries: Vec<_> = entries.filter_map(Result::ok).collect();
    entries.sort_by_key(|entry| std::cmp::Reverse(entry.file_name()));
    for entry in entries.into_iter().skip(keep) {
        let _ = fs::remove_dir_all(entry.path());
    }
}

/// 生成唯一的检查点 ID（时间戳 + 进程 ID + 序号）。
fn checkpoint_id() -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    let sequence = CHECKPOINT_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    format!("{nanos:032x}-{:08x}-{sequence:016x}", std::process::id())
}

/// 获取存放检查点的目录（位于 Git 元数据目录下）。
fn checkpoint_root(git_root: &Path) -> Result<PathBuf, String> {
    let output = Command::new("git")
        .args(["rev-parse", "--git-path", "echo-checkpoints"])
        .current_dir(git_root)
        .output()
        .map_err(|error| format!("failed to locate Git metadata directory: {error}"))?;
    if !output.status.success() {
        return Err("git rev-parse --git-path failed".to_string());
    }
    let path = String::from_utf8(output.stdout)
        .map_err(|error| format!("Git metadata path is not valid UTF-8: {error}"))?;
    let path = PathBuf::from(path.trim());
    Ok(if path.is_absolute() {
        path
    } else {
        git_root.join(path)
    })
}

/// 将目标路径解析为绝对路径。
fn absolute_target(file_path: &Path) -> Result<PathBuf, String> {
    if let Ok(path) = fs::canonicalize(file_path) {
        return Ok(path);
    }
    if file_path.is_absolute() {
        return Ok(file_path.to_path_buf());
    }
    std::env::current_dir()
        .map(|directory| directory.join(file_path))
        .map_err(|error| format!("failed to resolve checkpoint target: {error}"))
}

/// 查找文件所在的 Git 仓库根目录，不在仓库中时返回 None。
fn find_git_root(file_path: &Path) -> Result<Option<PathBuf>, String> {
    let dir = if file_path.is_file() {
        file_path.parent().unwrap_or(file_path)
    } else {
        file_path
    };
    let output = Command::new("git")
        .args(["rev-parse", "--show-toplevel"])
        .current_dir(dir)
        .output()
        .map_err(|error| format!("failed to run git rev-parse: {error}"))?;
    if !output.status.success() {
        return Ok(None);
    }
    let root = String::from_utf8(output.stdout)
        .map_err(|error| format!("Git root is not valid UTF-8: {error}"))?;
    let root = fs::canonicalize(root.trim())
        .map_err(|error| format!("failed to resolve Git root: {error}"))?;
    Ok(Some(root))
}

/// 读取文件在 Git index 中的条目。
fn read_index_entry(git_root: &Path, relative: &Path) -> Result<Option<IndexEntry>, String> {
    let output = Command::new("git")
        .args(["ls-files", "--stage", "--"])
        .arg(relative)
        .current_dir(git_root)
        .output()
        .map_err(|error| format!("failed to inspect Git index: {error}"))?;
    if !output.status.success() {
        return Err("git ls-files failed while creating checkpoint".to_string());
    }
    let line = String::from_utf8(output.stdout)
        .map_err(|error| format!("Git index output is not valid UTF-8: {error}"))?;
    let Some(metadata) = line.split_once('\t').map(|(metadata, _)| metadata) else {
        return Ok(None);
    };
    let mut fields = metadata.split_whitespace();
    let mode = fields
        .next()
        .ok_or_else(|| "missing index mode".to_string())?;
    let object_id = fields
        .next()
        .ok_or_else(|| "missing index object id".to_string())?;
    let stage = fields
        .next()
        .ok_or_else(|| "missing index stage".to_string())?;
    Ok(Some(IndexEntry {
        mode: mode.to_string(),
        object_id: object_id.to_string(),
        stage: stage.to_string(),
    }))
}

/// 将文件的 index 条目恢复为给定值；为 None 时从 index 中移除。
fn restore_index_entry(
    git_root: &Path,
    relative_path: &str,
    entry: Option<&IndexEntry>,
) -> Result<(), String> {
    let mut child = Command::new("git")
        .args(["update-index", "--index-info"])
        .current_dir(git_root)
        .stdin(Stdio::piped())
        .spawn()
        .map_err(|error| format!("failed to start git update-index: {error}"))?;
    let line = match entry {
        Some(entry) => format!(
            "{} {} {}\t{}\n",
            entry.mode, entry.object_id, entry.stage, relative_path
        ),
        None => format!("0 {} 0\t{}\n", "0".repeat(40), relative_path),
    };
    let mut stdin = child
        .stdin
        .take()
        .ok_or_else(|| "git update-index stdin unavailable".to_string())?;
    stdin
        .write_all(line.as_bytes())
        .map_err(|error| format!("failed to restore Git index entry: {error}"))?;
    drop(stdin);
    let status = child
        .wait()
        .map_err(|error| format!("failed to wait for git update-index: {error}"))?;
    if status.success() {
        Ok(())
    } else {
        Err("git update-index rejected checkpoint entry".to_string())
    }
}

/// 读取文件权限模式（仅 Unix）。
#[cfg(unix)]
fn file_mode(path: &Path) -> Result<Option<u32>, String> {
    use std::os::unix::fs::PermissionsExt;
    fs::metadata(path)
        .map(|metadata| Some(metadata.permissions().mode()))
        .map_err(|error| format!("failed to inspect checkpoint permissions: {error}"))
}

/// 设置文件权限模式（仅 Unix）。
#[cfg(unix)]
fn set_file_mode(path: &Path, mode: u32) -> Result<(), String> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(mode))
        .map_err(|error| format!("failed to restore checkpoint permissions: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 回滚只恢复目标文件的工作区内容和 index 条目。
    #[test]
    fn rollback_restores_only_target_worktree_and_index() -> Result<(), Box<dyn std::error::Error>>
    {
        let temp = tempfile::tempdir()?;
        let root = temp.path();
        run_git(root, &["init"])?;
        run_git(root, &["config", "user.email", "test@example.com"])?;
        run_git(root, &["config", "user.name", "Test"])?;
        run_git(root, &["config", "commit.gpgsign", "false"])?;
        let target = root.join("target.txt");
        let unrelated = root.join("unrelated.txt");
        fs::write(&target, "base")?;
        fs::write(&unrelated, "base")?;
        run_git(root, &["add", "."])?;
        run_git(root, &["commit", "-m", "base"])?;
        fs::write(&target, "staged")?;
        run_git(root, &["add", "target.txt"])?;
        fs::write(&target, "worktree")?;
        fs::write(&unrelated, "keep me")?;

        let id = create_checkpoint(&target)?.ok_or("missing checkpoint")?;
        fs::write(&target, "mutated")?;
        rollback_to_checkpoint(&target, &id)?;

        assert_eq!(fs::read_to_string(&target)?, "worktree");
        assert_eq!(fs::read_to_string(&unrelated)?, "keep me");
        let staged = Command::new("git")
            .args(["show", ":target.txt"])
            .current_dir(root)
            .output()?;
        assert!(staged.status.success());
        assert_eq!(String::from_utf8(staged.stdout)?, "staged");
        Ok(())
    }

    /// 在指定目录中运行 git 命令，失败时返回错误。
    fn run_git(root: &Path, args: &[&str]) -> Result<(), Box<dyn std::error::Error>> {
        let status = Command::new("git").args(args).current_dir(root).status()?;
        if status.success() {
            Ok(())
        } else {
            Err(format!("git command failed: {args:?}").into())
        }
    }
}
