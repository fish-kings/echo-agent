//! 记忆系统的 trait 与数据类型
//!
//! 记忆子系统的核心 trait 定义。具体实现位于 `echo_state` 中。
//!
//! | Trait | 用途 |
//! |-------|---------|
//! | [`Store`] | 具备命名空间隔离的长期 KV 存储 |
//! | [`Embedder`] | 文本转向量的嵌入接口 |
//! | [`ConversationStore`] | 会话持久化（对话记录的读模型） |
//!
//! 运行时检查点（跨进程重启恢复）由 `echo_agent::state` 中的
//! `RuntimeStateStore` 负责，不属于本模块的职责。

pub mod conversation;
pub mod embedder;
pub mod scope;
pub mod store;
pub mod types;

pub use scope::MemoryScope;

pub use conversation::{
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
pub use embedder::Embedder;
pub use store::{SearchMode, SearchQuery, Store, StoreItem};
pub use types::{MemoryMeta, MemoryRisk, MemorySource, MemoryStatus, MemoryType, TypedMemoryValue};
