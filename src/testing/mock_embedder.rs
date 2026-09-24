//! 用于测试的 MockEmbedder
//!
//! 基于文本哈希生成确定性的归一化向量，无需真实 API，
//! 适用于单元测试与集成测试。

use crate::error::Result;
use crate::memory::embedder::Embedder;
use futures::future::BoxFuture;

/// 基于字节哈希生成确定性伪嵌入向量的测试用 Embedder
///
/// - 相同文本总是产生相同的向量（确定性）
/// - 越相似的文本得到的余弦距离越接近（语义感知能力有限，但足以用于测试流程）
/// - 零网络请求，适合 CI 环境
///
/// # 示例
///
/// ```rust
/// use echo_agent::testing::MockEmbedder;
/// use echo_agent::memory::Embedder;
///
/// # #[tokio::main]
/// # async fn main() {
/// let embedder = MockEmbedder::new(8).unwrap();
/// let vec = embedder.embed("hello world").await.unwrap();
/// assert_eq!(vec.len(), 8);
/// // 归一化向量，模长约等于 1.0
/// let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
/// assert!((norm - 1.0).abs() < 1e-5);
/// # }
/// ```
pub struct MockEmbedder {
    /// 生成向量的维度。
    dimension: usize,
}

impl MockEmbedder {
    /// 创建一个指定维度的 MockEmbedder（建议 4~64，足以测试相似度逻辑）
    pub fn new(dimension: usize) -> Result<Self> {
        if dimension == 0 {
            return Err(crate::error::ReactError::Other(
                "mock embedder dimension must be greater than zero".to_string(),
            ));
        }
        Ok(Self { dimension })
    }
}

impl Embedder for MockEmbedder {
    fn embed<'a>(&'a self, text: &'a str) -> BoxFuture<'a, Result<Vec<f32>>> {
        Box::pin(async move {
            let mut vec = vec![0.0f32; self.dimension];
            // 将各字节值累加到对应维度上（确定性）
            for (i, b) in text.bytes().enumerate() {
                vec[i % self.dimension] += b as f32;
            }
            // 做 L2 归一化，使余弦相似度有意义
            let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for v in &mut vec {
                    *v /= norm;
                }
            }
            Ok(vec)
        })
    }
}
