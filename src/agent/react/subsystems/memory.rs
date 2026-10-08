/*!
记忆与持久化子系统

集中管理上下文窗口、长期记忆存储、运行时 checkpoint、
状态快照以及会话历史投影。
*/

use crate::compression::ContextManager;
use crate::memory::{SnapshotManager, Store};
use std::sync::Arc;

/**
运行时状态的水合（hydration）阶段，描述 `context` 与某个
checkpoint 身份之间的同步状态。
*/
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) enum RuntimeStateHydration {
    /// 尚未初始化，`context` 还未关联任何 checkpoint 身份。
    #[default]
    Uninitialized,
    /// 正在从指定（可选）的 checkpoint 身份水合中，此阶段可被取消。
    Hydrating(Option<String>),
    /// 已完成水合，`context` 当前对应指定（可选）的 checkpoint 身份。
    Hydrated(Option<String>),
}

/**
记忆与持久化子系统

汇聚了会话上下文、长期记忆 Store、状态快照、
运行时 checkpoint 存储以及会话历史投影 Store。
*/
pub(crate) struct MemorySubsystem {
    /// 当前会话的上下文管理器。
    pub(crate) context: Arc<tokio::sync::Mutex<ContextManager>>,
    /// 可选的长期记忆存储。
    pub(crate) store: Option<Arc<dyn Store>>,
    /// 可选的状态快照管理器。
    pub(crate) snapshot_manager: Arc<std::sync::RwLock<Option<SnapshotManager>>>,
    /// 可选的会话历史投影存储。
    pub(crate) conversation_store: Option<Arc<dyn crate::memory::ConversationStore>>,
    /// 可选的运行时 checkpoint 存储。
    pub(crate) state_store: Option<Arc<dyn crate::state::RuntimeStateStore>>,
    /**
    `context` 当前所代表的运行时 checkpoint 身份。

    在任何可取消的恢复变更操作之前都会先发布 `Hydrating`，因此
    一次被取消的切换绝不会让部分替换后的上下文，看起来像是
    仍对应此前那个身份的已就绪状态。
    */
    pub(crate) runtime_state_hydration: Arc<tokio::sync::Mutex<RuntimeStateHydration>>,
    /// 内存中上下文所代表的运行时权威（authority）版本号。
    pub(crate) runtime_state_version:
        Arc<tokio::sync::Mutex<Option<crate::state::RuntimeStateVersion>>>,
    /// 构造时的工作目录，当新的运行时身份没有自己的 checkpoint 时用于恢复。
    pub(crate) configured_working_dir: Option<std::path::PathBuf>,
    /// 会话记录（transcript）投影的读取游标。
    pub(crate) transcript_projection_cursor:
        Arc<tokio::sync::Mutex<crate::agent::snapshot::TranscriptProjectionCursor>>,
}
