//! 文件系统相关工具与共享的路径解析辅助。

/// 事务性多文件补丁工具。
#[cfg(feature = "files")]
pub mod apply_patch;
/// 溢出工具输出的 Artifact 读取工具。
pub mod artifact;
/// 代码搜索工具。
#[cfg(feature = "files")]
pub mod code_search;
/// 文件/内容差异对比工具。
#[cfg(feature = "files")]
pub mod diff;
/// 文件读取与目录列举工具。
#[cfg(feature = "files")]
#[allow(clippy::module_inception)]
pub mod files;
/// 按 glob 模式匹配文件的工具。
#[cfg(feature = "files")]
pub mod glob;
/// 按正则搜索文件内容的工具。
#[cfg(feature = "files")]
pub mod grep;
/// 仓库目录树与符号地图工具。
#[cfg(feature = "files")]
pub mod repo_map;

#[cfg(feature = "files")]
use std::path::{Component, Path, PathBuf};

#[cfg(feature = "files")]
use echo_core::error::{Result, ToolError};

/**
 * 将用户提供的相对/绝对路径解析为一个安全的绝对路径。
 *
 * 相对路径解析所使用的基准目录，按以下优先级选取：
 * 1. `base_dir`（构造时设定的限制范围，例如 `with_base_dir`）
 * 2. `working_dir`（运行时按调用传入的目录，例如某个会话绑定的工作树）
 * 3. 进程的 `current_dir`（兜底方案）
 *
 * 当设置了 `base_dir` 时，解析结果会被限制在该目录内（绝对路径必须
 * 落在其中；相对路径会拼接到该目录下；并会对符号链接做防御性的
 * canonicalize 校验）。当只设置了 `working_dir` 时，相对路径会拼接到
 * 该目录下，但不会做限制范围校验（它仅作为 CWD 的覆盖值）。当两者
 * 都未设置时，行为与之前保持一致（直接使用原始路径 + 尽力而为的
 * canonicalize）。
 *
 * - 绝对路径：先归一化，然后（若设置了 base_dir）校验其仍落在 base_dir 内
 * - 相对路径：基于选定的基准目录展开，然后校验
 *
 * 完成文本层面的归一化之后，会使用 `std::fs::canonicalize()` 解析符号链接，
 * 并校验解析后的真实路径仍落在允许的目录范围内。对于目标文件尚不存在的
 * 写入操作，则改为对其父目录做 canonicalize 校验。
 */
#[cfg(feature = "files")]
fn resolve_path(
    tool: &str,
    path_str: &str,
    base_dir: &Option<PathBuf>,
    working_dir: Option<&Path>,
) -> Result<PathBuf> {
    let requested = Path::new(path_str);

    /*
     * 有效基准目录：构造时设定的 base_dir 优先（限制范围）；
     * 否则回退到运行时的 working_dir（仅作为 CWD 覆盖，不做限制范围校验）。
     */
    let effective_base: Option<PathBuf> = base_dir
        .clone()
        .or_else(|| working_dir.map(|p| p.to_path_buf()));

    let resolved = if let Some(ref base) = effective_base {
        let normalized_base = normalize_path(base);

        // 相对路径：以 base_dir 为根目录展开；绝对路径：直接归一化
        let normalized = if requested.is_absolute() {
            normalize_path(requested)
        } else {
            normalize_path(&normalized_base.join(requested))
        };

        if !normalized.starts_with(&normalized_base) {
            return Err(ToolError::ExecutionFailed {
                tool: tool.to_string(),
                message: format!(
                    "Path '{}' is outside the allowed working directory '{}'. \
                     Use a relative path, or an absolute path under the working directory.",
                    path_str,
                    normalized_base.display()
                ),
            }
            .into());
        }

        /*
         * 防御符号链接绕过：canonicalize 解析符号链接，
         * 并校验解析后的真实路径仍落在基准目录内。
         */
        match std::fs::canonicalize(&normalized) {
            Ok(canonical) => {
                let canonical_base = std::fs::canonicalize(&normalized_base).map_err(|e| {
                    ToolError::ExecutionFailed {
                        tool: tool.to_string(),
                        message: format!("Cannot resolve base directory: {}", e),
                    }
                })?;
                if !canonical.starts_with(&canonical_base) {
                    return Err(ToolError::ExecutionFailed {
                        tool: tool.to_string(),
                        message: format!(
                            "Path '{}' resolves via symlink to location outside allowed scope",
                            path_str
                        ),
                    }
                    .into());
                }
                canonical
            }
            Err(_) => {
                // 文件尚不存在（写入操作）—— 改为对父目录做 canonicalize
                if let Some(parent) = normalized.parent()
                    && parent != Path::new("")
                    && let Ok(canonical_parent) = std::fs::canonicalize(parent)
                {
                    let canonical_base = std::fs::canonicalize(&normalized_base)
                        .unwrap_or_else(|_| normalized_base.clone());
                    if !canonical_parent.starts_with(&canonical_base) {
                        return Err(ToolError::ExecutionFailed {
                            tool: tool.to_string(),
                            message: format!(
                                "Parent directory of '{}' resolves outside allowed scope",
                                path_str
                            ),
                        }
                        .into());
                    }
                    let filename = normalized.file_name().unwrap_or_default();
                    return Ok(canonical_parent.join(filename));
                }
                // 兜底方案：父目录也不存在时，返回文本层面归一化后的路径
                normalized
            }
        }
    } else {
        let normalized = normalize_path(requested);
        // normalize_path 会把 "." 归一化为空路径 —— 此时回退使用 CWD
        let normalized = if normalized.as_os_str().is_empty() {
            std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
        } else {
            normalized
        };
        // 未设置 base_dir 限制时，尽力而为地做 canonicalize
        std::fs::canonicalize(&normalized).unwrap_or(normalized)
    };

    Ok(resolved)
}

/// 不依赖文件系统的路径归一化（解析 `.` 与 `..`）
#[cfg(feature = "files")]
fn normalize_path(path: &Path) -> PathBuf {
    let mut components = Vec::new();
    for component in path.components() {
        match component {
            Component::ParentDir => {
                if let Some(Component::Normal(_)) = components.last() {
                    components.pop();
                }
            }
            Component::CurDir => {}
            c => components.push(c),
        }
    }
    components.iter().collect()
}
