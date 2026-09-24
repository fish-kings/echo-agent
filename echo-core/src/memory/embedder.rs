//! 文本向量化接口
//!
//! 将文本映射为稠密浮点向量，用于语义检索。
//! 具体实现（`HttpEmbedder`）位于 `echo_state` 中。

use crate::error::Result;
use futures::future::BoxFuture;

/// 文本向量化接口：将文本映射为稠密浮点向量
pub trait Embedder: Send + Sync {
    /// 计算给定文本的嵌入向量
    fn embed<'a>(&'a self, text: &'a str) -> BoxFuture<'a, Result<Vec<f32>>>;
}
