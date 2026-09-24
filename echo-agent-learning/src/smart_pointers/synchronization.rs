//! `Arc<tokio::sync::RwLock<T>>` 用于异步场景下的共享状态。

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::sync::RwLock;

/// 使用异步读写锁保护的、按任务名记录进度百分比的共享状态。
#[derive(Debug, Clone, Default)]
pub struct SharedProgress {
    /// 任务名称到进度百分比（0-100）的映射。
    inner: Arc<RwLock<HashMap<String, u8>>>,
}

/// 对于单个独立的数值，使用原子类型比使用锁更合适。
#[derive(Debug, Clone, Default)]
pub struct AtomicProgress {
    /// 以原子方式存储的进度百分比。
    percent: Arc<AtomicUsize>,
}

impl AtomicProgress {
    /// 设置当前进度百分比，超过 100 会被截断为 100。
    pub fn set(&self, percent: usize) {
        self.percent.store(percent.min(100), Ordering::Release);
    }

    /// 读取当前进度百分比。
    pub fn get(&self) -> usize {
        self.percent.load(Ordering::Acquire)
    }
}

impl SharedProgress {
    /// 创建一个空的共享进度表。
    pub fn new() -> Self {
        Self::default()
    }

    /// 设置指定任务的进度百分比，超过 100 会被截断为 100。
    pub async fn set(&self, task: impl Into<String>, percent: u8) {
        self.inner
            .write()
            .await
            .insert(task.into(), percent.min(100));
    }

    /// 读取指定任务的进度百分比，任务不存在时返回 `None`。
    pub async fn get(&self, task: &str) -> Option<u8> {
        self.inner.read().await.get(task).copied()
    }

    /// 返回当前所有任务进度的完整快照。
    pub async fn snapshot(&self) -> HashMap<String, u8> {
        self.inner.read().await.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 验证克隆出的共享进度表与原表共享同一份底层数据。
    #[tokio::test]
    async fn clones_share_async_state() {
        let first = SharedProgress::new();
        let second = first.clone();
        second.set("compile", 120).await;
        assert_eq!(first.get("compile").await, Some(100));
    }

    /// 验证原子进度值无需加锁即可在克隆之间共享。
    #[test]
    fn atomic_progress_is_shared_without_a_lock() {
        let first = AtomicProgress::default();
        let second = first.clone();
        second.set(75);
        assert_eq!(first.get(), 75);
    }
}
