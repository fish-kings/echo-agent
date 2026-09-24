//! 记忆系统
//!
//! 分层架构，各层职责各不相同：
//!
//! | 层级 | 实现 | 作用范围 |
//! |------|------|--------|
//! | 短期上下文 | [`crate::compression::ContextManager`] | 单次 `execute()` 调用内 |
//! | 对话历史 | [`ConversationStore`] / `SqliteConversationStore` | 会话记录投影、历史浏览、多用户隔离 |
//! | 长期记忆 | [`Store`] / [`FileStore`] / `SqliteStore` | 跨会话、跨用户共享 |
//!
//! 运行时检查点（用于在进程重启后恢复正在进行的对话）存放在
//! `echo_agent::state::RuntimeStateStore` 中——不在本模块内。
//!
//! **注意**：trait 定义与数据类型现已迁移至 `echo_core::memory`。
//! 此处仅为向后兼容而重新导出。

pub mod conversation;
pub mod embedder;
pub mod embedding_store;
pub mod file_conversation;
pub mod snapshot;
#[cfg(feature = "sqlite")]
pub mod sqlite_conversation;
#[cfg(feature = "sqlite")]
pub mod sqlite_store;
pub mod store;
pub mod typed_store;

// 从 echo-core 重新导出 trait 与数据类型（向后兼容）
pub use echo_core::memory::MemoryScope;
pub use echo_core::memory::conversation::{
    Conversation, ConversationFilter, ConversationMeta, ConversationProjectionAuthority,
    ConversationProjectionCapability, ConversationProjectionEpochReceipt,
    ConversationProjectionEpochStatus, ConversationProjectionLifecycle, ConversationStore,
    EnsureConversationProjectionRequest, ManagedConversationDelete,
    ManagedConversationDeleteReceipt, ManagedConversationDeleteStatus, ManagedConversationImport,
    ManagedConversationMetadataUpdate, ManagedConversationMetadataUpdateReceipt,
    ManagedConversationMetadataUpdateStatus, NewConversation, PersistenceCallCapability,
    PersistenceCallContext, StoredMessage, TranscriptProjectionApplyReceipt,
    TranscriptProjectionApplyStatus, TranscriptProjectionBatch, TranscriptProjectionConflictKind,
    TranscriptProjectionErrorClass, TranscriptProjectionItem, TranscriptProjectionSettlement,
    TranscriptProjectionSettlementStatus,
};
pub use echo_core::memory::embedder::Embedder;
pub use echo_core::memory::store::{SearchMode, SearchQuery, Store, StoreItem};
pub use echo_core::memory::types::{
    MemoryMeta, MemoryRisk, MemorySource, MemoryStatus, MemoryType, TypedMemoryValue,
};

// 从子模块重新导出具体实现
pub use conversation::{
    TranscriptProjectionMeta, normalized_transcript_attachments, project_message, project_messages,
    restore_message, restore_messages, set_transcript_projection_meta,
    transcript_projection_message_digest, transcript_projection_meta,
};
pub use embedder::HttpEmbedder;
pub use embedding_store::EmbeddingStore;
pub use file_conversation::FileConversationStore;
pub use snapshot::{SnapshotManager, SnapshotPolicy, StateSnapshot};
#[cfg(feature = "sqlite")]
pub use sqlite_conversation::SqliteConversationStore;
#[cfg(feature = "sqlite")]
pub use sqlite_store::SqliteStore;
pub use store::{FileStore, InMemoryStore};
pub use typed_store::{MemoryFilter, TypedMemoryEntry, TypedMemoryStore};

#[cfg(test)]
pub use test_utils::MockEmbedder;

/// 测试用嵌入器（仅在测试环境下可见）
#[cfg(test)]
mod test_utils {
    use echo_core::error::Result;
    use echo_core::memory::embedder::Embedder;
    use futures::future::BoxFuture;

    /// 用于测试的模拟嵌入器，通过字节求和生成确定性的归一化向量。
    pub struct MockEmbedder {
        /// 生成的嵌入向量的维度。
        dimension: usize,
    }

    impl MockEmbedder {
        /// 创建一个指定维度的模拟嵌入器；维度必须大于零。
        pub fn new(dimension: usize) -> Result<Self> {
            if dimension == 0 {
                return Err(echo_core::error::ReactError::Other(
                    "mock embedder dimension must be greater than zero".to_string(),
                ));
            }
            Ok(Self { dimension })
        }
    }

    impl Embedder for MockEmbedder {
        /// 将文本的字节值按维度取模累加，再做 L2 归一化，生成确定性的伪嵌入向量。
        fn embed<'a>(&'a self, text: &'a str) -> BoxFuture<'a, Result<Vec<f32>>> {
            Box::pin(async move {
                let mut vec = vec![0.0f32; self.dimension];
                for (i, b) in text.bytes().enumerate() {
                    vec[i % self.dimension] += b as f32;
                }
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
}
