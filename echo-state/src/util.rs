//! echo-state crate 的公共工具函数。

#[cfg(feature = "sqlite")]
use echo_core::error::MemoryError;
use std::path::{Path, PathBuf};

/**
将路径开头的 `~` 或 `~/` 展开为用户主目录。

优先读取 `HOME` 环境变量，若不存在则回退到 `USERPROFILE`（Windows）。
*/
pub fn expand_tilde(path: &Path) -> PathBuf {
    let s = path.to_string_lossy();
    if s.starts_with("~/")
        && let Some(home) = std::env::var("HOME")
            .ok()
            .or_else(|| std::env::var("USERPROFILE").ok())
    {
        return PathBuf::from(home).join(&s[2..]);
    }
    path.to_path_buf()
}

/// 创建一个带有上下文说明文本的 `MemoryError::IoError`，保证错误信息格式统一。
#[cfg(feature = "sqlite")]
pub fn memory_io_error(context: &str, err: impl std::fmt::Display) -> MemoryError {
    MemoryError::IoError(format!("{context}: {err}"))
}
