//! 用于 Agent 持久化状态的分层记忆系统。
//!
//! 该记忆系统提供三层，各自承担不同职责：
//!
//! - **Store** —— 带命名空间隔离的长期键值存储。
//!   由 [`InMemoryStore`]、[`FileStore`] 或 `SqliteStore`（需要 `sqlite` feature）支持。
//!   用于 L3 记忆晋升（压缩淘汰 → 写入此处 → 后续召回）。
//! - **ConversationStore** —— 面向用户可见的会话记录投影（每条消息
//!   一行，`StoredMessage` 结构）。驱动应用 UI 的历史记录面板。
//!   框架会在 `run_core_loop` 结束时自动持久化此数据。
//! - **RuntimeStateStore** —— ReAct 运行时 checkpoint（messages +
//!   active_skills + blocked_reason），用于跨进程重启恢复正在进行中
//!   的会话。公开的 `current_plan` 字段仅用于兼容旧版 Store 的读写
//!   往返，ReactAgent 不会恢复该字段。参见 [`crate::state`]。
//!
//! # 快速开始
//!
//! ```rust,no_run
//! use echo_agent::prelude::*;
//! use std::sync::Arc;
//!
//! # fn main() -> echo_agent::error::Result<()> {
//! let store = Arc::new(InMemoryStore::new());
//! let agent = ReactAgentBuilder::new()
//!     .model("qwen3-max")
//!     .with_memory_tools(store)  // 注册 remember、recall、search_memory、forget 工具
//!     .build()?;
//! # Ok(())
//! # }
//! ```
//!
//! # 关键类型
//!
//! | 类型 | 说明 |
//! |------|-------------|
//! | [`Store`] | 长期记忆后端的 trait |
//! | [`InMemoryStore`] / [`FileStore`] | 内置的 store 实现 |
//! | `SqliteStore` | 基于 SQLite 的 store（feature `sqlite`） |
//! | [`SnapshotManager`] | 在任意时刻捕获并恢复 Agent 状态 |
//! | [`crate::state::RuntimeStateStore`] | 用于崩溃恢复的完整运行时 checkpoint |
//! | [`ConversationStore`] | 面向用户可见的会话记录投影 |

/// 直接从 `echo_state::memory` 重新导出。
pub mod state {
    pub use echo_state::memory::*;
}

/// 长期记忆存储契约（contract）。
pub mod store {
    pub use echo_state::memory::store::*;
}

pub use echo_state::memory::*;
