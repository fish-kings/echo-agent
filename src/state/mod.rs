/*!
 * 用于支持 Agent 运行可恢复的运行时检查点（checkpoint）。
 *
 * [`RuntimeStateStore`] 仅持久化 ReAct 的检查点。任务依赖关系与生命周期状态归
 * `echo_orchestration::tasks::TaskRevisionService` 与 `RuntimeTaskService` 所有，
 * 因此本模块刻意不提供任务图相关的 API。
 *
 * # 快速开始
 *
 * ```rust,ignore
 * use echo_agent::state::{AgentCheckpoint, RuntimeStateStore};
 *
 * # async fn example(store: &dyn RuntimeStateStore) -> echo_agent::error::Result<()> {
 * let checkpoint = AgentCheckpoint::new("conv-123");
 * store.save_checkpoint(&checkpoint).await?;
 * # Ok(())
 * # }
 * ```
 */

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const RUNTIME_STATE_OPERATION_SCHEMA_VERSION: u16 = 1;

/**
 * 有序事件日志（journal）与检查点归约器（reducer）的基础组件。
 *
 * 检查点包含一个 [`JournalIdentity`](crate::state::journal::JournalIdentity)；
 * 恢复时只接受由产生该检查点的确切 Journal 代次（generation）所提供的状态。
 *
 * 结果未知时会保留原始的 [`PreparedJournalBatch`](crate::state::journal::PreparedJournalBatch)。
 * 重新打开一个文件后端的权威存储后，应先执行一次只读查找，再把同一个已准备好的值
 * 传给 `apply_batch`：已存在的身份会返回 `AlreadyCommitted`，归约器只会折叠尚未
 * 应用的那部分后缀。
 *
 * ```
 * use echo_agent::state::journal::{
 *     ApplyBatchReceipt, CheckpointStore, CheckpointedReducer, EventJournal, EventReducer,
 *     JournalBatchLookup, MemoryCheckpointStore, MemoryEventJournal, PreparedJournalBatch,
 * };
 * use std::sync::Arc;
 *
 * #[derive(Default, serde::Serialize, serde::Deserialize)]
 * struct Count(u64);
 * impl EventReducer for Count {
 *     type Event = String;
 *     fn apply(&mut self, _event: &String) {
 *         self.0 = self.0.saturating_add(1);
 *     }
 * }
 *
 * fn resume_prepared<J, R>(
 *     journal: &J,
 *     reducer: &CheckpointedReducer<J, R>,
 *     prepared: PreparedJournalBatch<R::Event>,
 * ) -> Result<ApplyBatchReceipt, String>
 * where
 *     J: EventJournal<R::Event>,
 *     R: EventReducer,
 * {
 *     if let JournalBatchLookup::Conflict { error } = journal
 *         .lookup_batch(&prepared)
 *         .map_err(|error| error.to_string())?
 *     {
 *         return Err(error);
 *     }
 *     reducer.apply_batch(prepared).map_err(|error| error.to_string())
 * }
 *
 * # fn main() -> Result<(), String> {
 * let journal = Arc::new(MemoryEventJournal::<String>::new());
 * let checkpoints: Arc<dyn CheckpointStore<Count>> = Arc::new(MemoryCheckpointStore::new());
 * let reducer = CheckpointedReducer::new(Arc::clone(&journal), checkpoints, 8);
 * let prepared = PreparedJournalBatch::new(vec!["one".to_string(), "two".to_string()])
 *     .map_err(|error| error.to_string())?;
 * let receipt = resume_prepared(journal.as_ref(), &reducer, prepared)?;
 * assert_eq!(receipt.record_count, 2);
 * # Ok(())
 * # }
 * ```
 */
pub mod journal {
    pub use echo_state::journal::*;
}

/**
 * 带类型的持久化投递（delivery）生命周期基础组件。
 *
 * 这是有序消息投递的稳定框架门面（facade）。路由与负载（payload）的类型仍由
 * 调用方持有；框架只负责生命周期身份、尝试次数、保留期与恢复。
 */
pub mod delivery {
    pub use echo_state::delivery::*;
}

// ── AgentCheckpoint ────────────────────────────────────────────────────

/// Agent 运行时状态的完整检查点，可用于序列化并在之后恢复（hydration）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentCheckpoint {
    /// 会话 / 对话标识符。
    pub conversation_id: String,
    /// 序列化后的消息历史。
    pub messages_json: String,
    /**
     * 为兼容旧版检查点解码与 Store 往返而保留的旧版 plan 文本。
     * ReactAgent 既不会把它恢复为任务状态，也不会写入新值；
     * 带版本号的 Task 图才是规划的权威来源。
     */
    pub current_plan: Option<String>,
    /// 当前处于激活状态的 skill 名称列表。
    pub active_skills: Vec<String>,
    /// 如果 agent 处于阻塞状态，记录阻塞原因。
    pub blocked_reason: Option<String>,
    /**
     * 会话绑定的工作目录（worktree 路径）。恢复（hydration）时会一并恢复，
     * 使绑定了 worktree 的会话能在同一个隔离的检出（checkout）中继续运行。
     */
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub working_dir: Option<std::path::PathBuf>,
    /// 捕获该检查点时的时间戳。
    #[serde(with = "crate::utils::time::local_rfc3339")]
    pub timestamp: DateTime<Utc>,
}

/// 为某一代（generation）transcript 投影持久化的稳定消息身份。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TranscriptProjectionMessage {
    /// 消息在投影序列中的序号。
    pub ordinal: u64,
    /// 归一化后的用户可见消息投影的 SHA-256 摘要。
    pub digest: String,
}

/// 用于对某一代模型上下文做仅追加（append-only）投影的持久化游标。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TranscriptProjectionCheckpoint {
    /// 所属的 transcript 投影代次标识。
    pub generation_id: String,
    /// 下一条待分配的序号。
    pub next_ordinal: u64,
    /// 已投影的消息列表。
    pub projected: Vec<TranscriptProjectionMessage>,
}

/// 一次持久化 transcript 副作用最近一次已知的重试分类。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TranscriptProjectionAttemptClass {
    /// 结果未知（例如超时后无法确认是否已提交）。
    OutcomeUnknown,
    /// 瞬时失败且未提交，可安全重试。
    TransientNoCommit,
    /// 版本号冲突。
    RevisionConflict,
    /// 超过截止时间。
    DeadlineExceeded,
    /// 后端不支持该操作。
    Unsupported,
    /// 配置无效。
    InvalidConfiguration,
    /// 持久化状态已损坏。
    CorruptState,
    /// 语义冲突（例如目标已被删除或已变更）。
    SemanticConflict,
}

/// 在 transcript 后端能够观察到该副作用之前，预先写入的持久化意图记录。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PendingTranscriptProjection {
    /// 待应用的 transcript 投影批次。
    pub batch: crate::memory::TranscriptProjectionBatch,
    /// 应用该批次之前的游标状态。
    pub cursor_before: TranscriptProjectionCheckpoint,
    /// 应用该批次之后期望达到的游标状态。
    pub cursor_after: TranscriptProjectionCheckpoint,
    /// 准备该待处理效果时所基于的运行时版本号。
    pub base_runtime_revision: u64,
    /// 该待处理效果被准备（prepare）的时间。
    #[serde(with = "crate::utils::time::local_rfc3339")]
    pub prepared_at: DateTime<Utc>,
    /// 当前尝试次数。
    pub attempt: u32,
    /// 最近一次尝试的分类结果（如果有）。
    pub last_attempt_class: Option<TranscriptProjectionAttemptClass>,
    /// 最近一次尝试的错误信息（如果有）。
    pub last_error: Option<String>,
}

/// 解析后的检查点负载，包含尚未结算（unsettled）的 transcript 意图（如果有）。
pub struct ManagedAgentCheckpointPayload {
    /// 恢复出的消息历史。
    pub messages: Vec<crate::llm::types::Message>,
    /// 恢复出的 transcript 投影游标（如果存在）。
    pub transcript_projection: Option<TranscriptProjectionCheckpoint>,
    /// 恢复出的待结算 transcript 投影（如果存在）。
    pub pending_transcript_projection: Option<PendingTranscriptProjection>,
}

/// 从一次 `AgentCheckpoint` 解析中恢复出的、已通过校验的运行时负载。
pub struct RestoredAgentCheckpoint {
    /// 恢复出的消息历史。
    pub messages: Vec<crate::llm::types::Message>,
    /// 恢复出的 transcript 投影游标（如果存在）。
    pub transcript_projection: Option<TranscriptProjectionCheckpoint>,
}

/// 检查点在磁盘 / Store 中的序列化负载结构（带 schema 版本号）。
#[derive(Serialize, Deserialize)]
struct AgentCheckpointPayload {
    #[serde(default = "default_agent_checkpoint_payload_version")]
    schema_version: u16,
    messages: Vec<crate::llm::types::Message>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    transcript_projection: Option<TranscriptProjectionCheckpoint>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pending_transcript_projection: Option<PendingTranscriptProjection>,
}

/// `schema_version` 字段缺失时使用的默认值（旧数据视为版本 1）。
fn default_agent_checkpoint_payload_version() -> u16 {
    1
}

/**
 * 兼容新旧两种检查点负载格式：新版带 schema_version 的结构，或旧版仅有
 * 消息数组的格式。
 */
#[derive(Deserialize)]
#[serde(untagged)]
enum AgentCheckpointPayloadCompat {
    /// 当前（新版）带 schema_version 的负载格式。
    Current(Box<AgentCheckpointPayload>),
    /// 旧版仅包含消息数组的负载格式。
    Legacy(Vec<crate::llm::types::Message>),
}

impl AgentCheckpoint {
    /// 创建一个新的检查点。
    pub fn new(conversation_id: impl Into<String>) -> Self {
        Self {
            conversation_id: conversation_id.into(),
            messages_json: String::new(),
            current_plan: None,
            active_skills: Vec::new(),
            blocked_reason: None,
            working_dir: None,
            timestamp: Utc::now(),
        }
    }

    /**
     * 反序列化并校验检查点中的消息历史。
     *
     * 只有当每一次 assistant 发起的工具调用都恰好有一条按顺序匹配的工具结果时，
     * 该检查点才被认为是可恢复的。在这里拒绝格式错误的历史记录，可以避免向
     * provider 发送非法上下文，或在重启后重放一个已经完成过的副作用。
     */
    pub fn restore_messages(&self) -> crate::error::Result<Vec<crate::llm::types::Message>> {
        self.restore_runtime_payload()
            .map(|payload| payload.messages)
    }

    /// 恢复持久化的 transcript 代次游标（如果存在）。
    pub fn restore_transcript_projection(
        &self,
    ) -> crate::error::Result<Option<TranscriptProjectionCheckpoint>> {
        self.restore_runtime_payload()
            .map(|payload| payload.transcript_projection)
    }

    /// 一次性解析并校验消息与 transcript 游标。
    pub fn restore_runtime_payload(&self) -> crate::error::Result<RestoredAgentCheckpoint> {
        let payload = self.restore_managed_runtime_payload()?;
        if payload.pending_transcript_projection.is_some() {
            return Err(invalid_checkpoint(
                "checkpoint has unsettled transcript projection; managed recovery is required"
                    .to_string(),
            ));
        }
        Ok(RestoredAgentCheckpoint {
            messages: payload.messages,
            transcript_projection: payload.transcript_projection,
        })
    }

    /// 解析并校验运行时消息、游标以及持久化的待处理效果。
    pub fn restore_managed_runtime_payload(
        &self,
    ) -> crate::error::Result<ManagedAgentCheckpointPayload> {
        let payload = self.restore_payload()?;
        validate_tool_message_pairing(&payload.messages)?;
        self.validate_transcript_projection(payload.transcript_projection.as_ref())?;
        self.validate_pending_transcript_projection(
            payload.pending_transcript_projection.as_ref(),
        )?;
        Ok(payload)
    }

    /**
     * 校验 transcript 投影游标是否与本检查点的会话身份一致、序号严格递增且
     * 摘要格式合法。
     */
    fn validate_transcript_projection(
        &self,
        projection: Option<&TranscriptProjectionCheckpoint>,
    ) -> crate::error::Result<()> {
        if let Some(projection) = projection.as_ref() {
            if projection.generation_id != self.conversation_id {
                return Err(invalid_checkpoint(
                    "transcript projection generation does not match checkpoint identity"
                        .to_string(),
                ));
            }
            let mut previous = None;
            for message in &projection.projected {
                if message.ordinal >= projection.next_ordinal
                    || previous.is_some_and(|previous| message.ordinal <= previous)
                    || message.digest.len() != 64
                    || !message.digest.bytes().all(|byte| byte.is_ascii_hexdigit())
                {
                    return Err(invalid_checkpoint(
                        "transcript projection cursor is corrupt".to_string(),
                    ));
                }
                previous = Some(message.ordinal);
            }
        }
        Ok(())
    }

    /**
     * 校验待结算 transcript 投影的身份、游标前后一致性，以及批次摘要是否与
     * cursor_after 中记录的摘要相符。
     */
    fn validate_pending_transcript_projection(
        &self,
        pending: Option<&PendingTranscriptProjection>,
    ) -> crate::error::Result<()> {
        let Some(pending) = pending else {
            return Ok(());
        };
        pending.batch.validate()?;
        let invalid_attempt_metadata = match (
            pending.last_attempt_class.as_ref(),
            pending.last_error.as_deref(),
        ) {
            (None, None) => false,
            (Some(_), Some(error)) => error.trim().is_empty(),
            _ => true,
        };
        if pending.batch.generation_id != self.conversation_id
            || pending.cursor_before.generation_id != self.conversation_id
            || pending.cursor_after.generation_id != self.conversation_id
            || pending.base_runtime_revision == 0
            || pending.attempt == 0
            || invalid_attempt_metadata
            || pending.cursor_before.next_ordinal
                != pending
                    .batch
                    .items
                    .first()
                    .map(|item| item.ordinal)
                    .unwrap_or(pending.cursor_before.next_ordinal)
            || pending.cursor_after.next_ordinal
                != pending
                    .batch
                    .items
                    .last()
                    .and_then(|item| item.ordinal.checked_add(1))
                    .unwrap_or(pending.cursor_after.next_ordinal)
        {
            return Err(invalid_checkpoint(
                "pending transcript projection is not aligned with checkpoint identity and cursor"
                    .to_string(),
            ));
        }
        self.validate_transcript_projection(Some(&pending.cursor_before))?;
        self.validate_transcript_projection(Some(&pending.cursor_after))?;
        let before = pending
            .cursor_before
            .projected
            .iter()
            .map(|message| (message.ordinal, message.digest.clone()))
            .collect::<std::collections::BTreeMap<_, _>>();
        let mut batch = std::collections::BTreeMap::new();
        for item in &pending.batch.items {
            batch.insert(
                item.ordinal,
                crate::memory::transcript_projection_message_digest(&item.message)?,
            );
        }
        let first_batch_ordinal = pending
            .batch
            .items
            .first()
            .map(|item| item.ordinal)
            .ok_or_else(|| invalid_checkpoint("pending transcript batch is empty".to_string()))?;
        for message in &pending.cursor_after.projected {
            let expected = if message.ordinal < first_batch_ordinal {
                before.get(&message.ordinal).map(String::as_str)
            } else {
                batch.get(&message.ordinal).map(String::as_str)
            };
            if expected != Some(message.digest.as_str()) {
                return Err(invalid_checkpoint(
                    "pending cursor-after is not derived from cursor-before and batch".to_string(),
                ));
            }
        }
        if batch.iter().any(|(ordinal, digest)| {
            pending
                .cursor_after
                .projected
                .iter()
                .find(|message| message.ordinal == *ordinal)
                .is_none_or(|message| message.digest.as_str() != digest)
        }) {
            return Err(invalid_checkpoint(
                "pending cursor-after omits a prepared transcript item".to_string(),
            ));
        }
        Ok(())
    }

    /**
     * 将消息以及它们精确的 transcript 投影游标序列化到现有的检查点负载
     * 列 / 文件中。
     */
    pub fn serialize_payload(
        messages: Vec<crate::llm::types::Message>,
        transcript_projection: Option<TranscriptProjectionCheckpoint>,
    ) -> crate::error::Result<String> {
        Self::serialize_managed_payload(messages, transcript_projection, None)
    }

    /// 序列化消息、游标以及一个持久化的待处理 transcript 效果。
    pub fn serialize_managed_payload(
        messages: Vec<crate::llm::types::Message>,
        transcript_projection: Option<TranscriptProjectionCheckpoint>,
        pending_transcript_projection: Option<PendingTranscriptProjection>,
    ) -> crate::error::Result<String> {
        serde_json::to_string(&AgentCheckpointPayload {
            schema_version: 2,
            messages,
            transcript_projection,
            pending_transcript_projection,
        })
        .map_err(|error| {
            crate::error::ReactError::RuntimeState(Box::new(
                echo_core::error::RuntimeStateError::SerializationError(format!(
                    "Failed to serialize checkpoint payload: {error}"
                )),
            ))
        })
    }

    /// 反序列化 `messages_json`（兼容新旧两种负载格式），并校验 schema 版本号。
    fn restore_payload(&self) -> crate::error::Result<ManagedAgentCheckpointPayload> {
        let payload: AgentCheckpointPayloadCompat = serde_json::from_str(&self.messages_json)
            .map_err(|error| {
                crate::error::ReactError::RuntimeState(Box::new(
                    echo_core::error::RuntimeStateError::SerializationError(format!(
                        "Failed to deserialize checkpoint messages: {error}"
                    )),
                ))
            })?;
        match payload {
            AgentCheckpointPayloadCompat::Current(payload) => {
                if !(1..=2).contains(&payload.schema_version) {
                    return Err(invalid_checkpoint(format!(
                        "unsupported checkpoint payload version {}",
                        payload.schema_version
                    )));
                }
                if payload.schema_version == 1 && payload.pending_transcript_projection.is_some() {
                    return Err(invalid_checkpoint(
                        "legacy checkpoint payload cannot contain pending transcript projection"
                            .to_string(),
                    ));
                }
                Ok(ManagedAgentCheckpointPayload {
                    messages: payload.messages,
                    transcript_projection: payload.transcript_projection,
                    pending_transcript_projection: payload.pending_transcript_projection,
                })
            }
            AgentCheckpointPayloadCompat::Legacy(messages) => Ok(ManagedAgentCheckpointPayload {
                messages,
                transcript_projection: None,
                pending_transcript_projection: None,
            }),
        }
    }

    /// 该检查点中已完成的工具调用 ID 列表，按消息顺序排列。
    pub fn completed_tool_call_ids(&self) -> crate::error::Result<Vec<String>> {
        let messages = self.restore_messages()?;
        Ok(messages
            .into_iter()
            .filter_map(|message| {
                if message.role == crate::llm::types::Role::Tool {
                    message.tool_call_id
                } else {
                    None
                }
            })
            .collect())
    }
}

/**
 * 校验消息历史中每一个 assistant 发起的工具调用都恰好有一条对应的、名称匹配
 * 的工具结果消息，且没有重复或孤立（orphan）的工具结果。
 */
fn validate_tool_message_pairing(
    messages: &[crate::llm::types::Message],
) -> crate::error::Result<()> {
    let mut pending = std::collections::HashMap::<String, String>::new();
    for message in messages {
        if message.role == crate::llm::types::Role::Assistant
            && let Some(tool_calls) = message.tool_calls.as_ref()
        {
            for call in tool_calls {
                if pending
                    .insert(call.id.clone(), call.function.name.clone())
                    .is_some()
                {
                    return Err(invalid_checkpoint(format!(
                        "duplicate in-flight tool call id {}",
                        call.id
                    )));
                }
            }
        }
        if message.role == crate::llm::types::Role::Tool {
            let call_id = message.tool_call_id.as_deref().ok_or_else(|| {
                invalid_checkpoint("tool result is missing tool_call_id".to_string())
            })?;
            let expected_name = pending.remove(call_id).ok_or_else(|| {
                invalid_checkpoint(format!("orphan or duplicate tool result for {call_id}"))
            })?;
            if message.name.as_deref() != Some(expected_name.as_str()) {
                return Err(invalid_checkpoint(format!(
                    "tool result name mismatch for {call_id}: expected {expected_name}, got {}",
                    message.name.as_deref().unwrap_or("<missing>")
                )));
            }
        }
    }
    if pending.is_empty() {
        Ok(())
    } else {
        let mut ids = pending.into_keys().collect::<Vec<_>>();
        ids.sort();
        Err(invalid_checkpoint(format!(
            "checkpoint has tool calls without results: {}",
            ids.join(", ")
        )))
    }
}

/// 构造一个表示"检查点数据无效"的 `ReactError`。
fn invalid_checkpoint(message: String) -> crate::error::ReactError {
    crate::error::ReactError::RuntimeState(Box::new(
        echo_core::error::RuntimeStateError::SerializationError(message),
    ))
}

// ── RuntimeStateStore 特征 ────────────────────────────────────────────

/// 无需执行 I/O 即可对外声明的、带版本号的运行时状态支持能力。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeStateCapability {
    /// 后端不支持带版本号的运行时状态。
    Unsupported,
    /// 支持版本号为 V1 的运行时状态。
    RevisionedV1,
}

/// 一次运行时代次（generation）的持久化版本状态。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeStateVersion {
    /// 尚不存在任何状态。
    Absent,
    /// 存在但未纳入版本管理，仅以摘要（digest）标识内容。
    Unmanaged {
        /// 内容摘要。
        digest: String,
    },
    /// 已纳入版本管理，带有明确的修订号（revision）。
    Managed {
        /// 当前修订号。
        revision: u64,
    },
    /// 已被退休（retired），保留退休时的修订号与对应操作 ID。
    Retired {
        /// 退休时的修订号。
        revision: u64,
        /// 执行退休的操作 ID。
        operation_id: String,
    },
}

/// compare-and-save 请求期望替换掉的版本。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeStateExpectedVersion {
    /// 期望当前不存在任何状态。
    Absent,
    /// 期望当前状态是未纳入版本管理、带有指定摘要的内容。
    Unmanaged {
        /// 期望的内容摘要。
        digest: String,
    },
    /// 期望当前状态处于指定的修订号。
    Managed {
        /// 期望的修订号。
        revision: u64,
    },
}

/// 一个稳定运行时作用域（scope）的持久化生命周期。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeScopeLifecycle {
    /// 作用域处于活跃状态。
    Active,
    /// 作用域正在退休（删除流程进行中）。
    Retiring,
    /// 作用域已被完全退休并打上墓碑（tombstone）标记。
    Tombstoned,
}

/// 供准入判断与检查点 CAS 操作查询的稳定运行时作用域权威信息。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeScopeAuthority {
    /// 作用域标识符。
    pub scope_id: String,
    /// 该作用域当前关联的会话 epoch（如果有）。
    pub conversation_epoch: Option<u64>,
    /// 作用域的当前修订号。
    pub revision: u64,
    /// 作用域当前所处的生命周期阶段。
    pub lifecycle: RuntimeScopeLifecycle,
}

/// 一次运行时检查点或退休墓碑的、带版本号的视图。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManagedRuntimeStateSnapshot {
    /// 所属作用域的权威信息。
    pub scope: RuntimeScopeAuthority,
    /// 运行时状态标识符。
    pub runtime_state_id: String,
    /// 当前的持久化版本状态。
    pub version: RuntimeStateVersion,
    /// 当前的检查点内容（如果存在）。
    pub checkpoint: Option<AgentCheckpoint>,
}

/// 原子性检查点写入请求。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeCheckpointCasRequest {
    /// 目标作用域标识符。
    pub scope_id: String,
    /// 目标运行时状态标识符。
    pub runtime_state_id: String,
    /// 期望的会话 epoch。
    pub conversation_epoch: Option<u64>,
    /// 期望的作用域修订号。
    pub expected_scope_revision: u64,
    /// 期望被替换掉的运行时状态版本。
    pub expected_state_version: RuntimeStateExpectedVersion,
    /// 待写入的检查点内容。
    pub checkpoint: AgentCheckpoint,
}

/// compare-and-save 操作的业务结果。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeCheckpointCasStatus {
    /// 写入已成功应用。
    Applied,
    /// 当前值已与请求写入的内容一致，无需再次应用。
    AlreadyCurrent,
    /// 期望的版本号与实际不符，发生冲突。
    RevisionConflict,
    /// 目标代次已被退休，无法写入。
    GenerationRetired,
    /// 目标作用域已被围栏（fenced），拒绝写入。
    ScopeFenced,
}

/// compare-and-save 操作的稳定回执。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeCheckpointCasReceipt {
    /// 操作完成后的作用域权威信息。
    pub scope: RuntimeScopeAuthority,
    /// 运行时状态标识符。
    pub runtime_state_id: String,
    /// 操作完成后的版本状态。
    pub version: RuntimeStateVersion,
    /// 操作的业务结果。
    pub status: RuntimeCheckpointCasStatus,
}

/// 携带证明的请求，用于清除一个已应用的 transcript 待处理标记。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeTranscriptAckRequest {
    /// 目标作用域标识符。
    pub scope_id: String,
    /// 目标运行时状态标识符。
    pub runtime_state_id: String,
    /// 期望的会话 epoch。
    pub conversation_epoch: u64,
    /// 期望的作用域修订号。
    pub expected_scope_revision: u64,
    /// 期望被替换掉的运行时状态版本。
    pub expected_state_version: RuntimeStateExpectedVersion,
    /// 证明该批次已被应用的投影回执。
    pub projection_receipt: crate::memory::TranscriptProjectionApplyReceipt,
    /// 确认后要写入的检查点内容。
    pub checkpoint: AgentCheckpoint,
}

impl RuntimeTranscriptAckRequest {
    /// 校验该确认请求的身份、epoch 与所附带的投影回执是否一致、有效。
    pub fn validate(&self) -> crate::error::Result<()> {
        validate_runtime_operation_identity(&self.scope_id, &self.runtime_state_id)?;
        if self.conversation_epoch == 0 || self.checkpoint.conversation_id != self.runtime_state_id
        {
            return Err(invalid_runtime_operation(
                "invalid transcript acknowledgement identity",
            ));
        }
        if !matches!(
            self.projection_receipt.status,
            crate::memory::TranscriptProjectionApplyStatus::Applied
                | crate::memory::TranscriptProjectionApplyStatus::AlreadyApplied
        ) || self.projection_receipt.authority.conversation_id != self.scope_id
            || self.projection_receipt.authority.epoch != self.conversation_epoch
        {
            return Err(invalid_runtime_operation(
                "transcript acknowledgement lacks a matching applied receipt",
            ));
        }
        if self
            .checkpoint
            .restore_managed_runtime_payload()?
            .pending_transcript_projection
            .is_some()
        {
            return Err(invalid_runtime_operation(
                "transcript acknowledgement checkpoint still contains pending debt",
            ));
        }
        Ok(())
    }
}

/// 用于退休某一确切运行时代次的幂等请求。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeGenerationRetireRequest {
    /// 请求负载的 schema 版本号。
    pub schema_version: u16,
    /// 该操作的幂等标识符（由负载摘要派生）。
    pub operation_id: String,
    /// 请求负载的内容摘要。
    pub payload_digest: String,
    /// 目标作用域标识符。
    pub scope_id: String,
    /// 目标运行时状态标识符。
    pub runtime_state_id: String,
    /// 期望的作用域修订号。
    pub expected_scope_revision: u64,
    /// 期望被替换掉的运行时状态版本。
    pub expected_state_version: RuntimeStateExpectedVersion,
}

impl RuntimeGenerationRetireRequest {
    /**
     * 根据目标身份与期望版本构造一个幂等的退休请求（自动派生
     * `operation_id` 与 `payload_digest`）。
     */
    pub fn prepare(
        scope_id: impl Into<String>,
        runtime_state_id: impl Into<String>,
        expected_scope_revision: u64,
        expected_state_version: RuntimeStateExpectedVersion,
    ) -> crate::error::Result<Self> {
        let scope_id = scope_id.into();
        let runtime_state_id = runtime_state_id.into();
        validate_runtime_operation_identity(&scope_id, &runtime_state_id)?;
        let payload_digest = runtime_operation_digest(&(
            "runtime-generation-retire-v1",
            RUNTIME_STATE_OPERATION_SCHEMA_VERSION,
            &scope_id,
            &runtime_state_id,
            expected_scope_revision,
            &expected_state_version,
        ))?;
        Ok(Self {
            schema_version: RUNTIME_STATE_OPERATION_SCHEMA_VERSION,
            operation_id: format!("runtime-generation-retire-v1:{payload_digest}"),
            payload_digest,
            scope_id,
            runtime_state_id,
            expected_scope_revision,
            expected_state_version,
        })
    }

    /**
     * 校验请求身份合法，且 `operation_id` / `payload_digest` 确实是从其余字段
     * 派生出来的（防止被篡改或错误重放）。
     */
    pub fn validate(&self) -> crate::error::Result<()> {
        validate_runtime_operation_identity(&self.scope_id, &self.runtime_state_id)?;
        if self.schema_version != RUNTIME_STATE_OPERATION_SCHEMA_VERSION {
            return Err(invalid_runtime_operation(
                "unsupported runtime generation retirement schema",
            ));
        }
        let digest = runtime_operation_digest(&(
            "runtime-generation-retire-v1",
            self.schema_version,
            &self.scope_id,
            &self.runtime_state_id,
            self.expected_scope_revision,
            &self.expected_state_version,
        ))?;
        if digest != self.payload_digest
            || self.operation_id != format!("runtime-generation-retire-v1:{digest}")
        {
            return Err(invalid_runtime_operation(
                "runtime generation retirement identity does not match its payload",
            ));
        }
        Ok(())
    }
}

/// 确切代次退休操作的业务结果。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeGenerationRetireStatus {
    /// 已成功退休。
    Retired,
    /// 该代次此前已被退休（幂等重放）。
    AlreadyRetired,
    /// 版本或作用域校验冲突，退休未生效。
    Conflict,
    /// 该代次获得了新的、尚未结算的 transcript 投影，无法退休。
    PendingProjection,
    /// 目标作用域已被围栏，拒绝操作。
    ScopeFenced,
}

/// 确切代次退休操作的稳定回执。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeGenerationRetireReceipt {
    /// 操作完成后的作用域权威信息。
    pub scope: RuntimeScopeAuthority,
    /// 运行时状态标识符。
    pub runtime_state_id: String,
    /// 操作完成后的版本状态。
    pub version: RuntimeStateVersion,
    /// 操作的业务结果。
    pub status: RuntimeGenerationRetireStatus,
}

/// 作用域退休清单（manifest）中记录的单个代次的进度状态。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScopeRetirementItemStatus {
    /// 尚待处理。
    Pending,
    /// 已随删除操作一并被丢弃。
    DroppedByDelete,
}

/// 在产品层删除开始之前捕获的、一条不可变的代次条目。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScopeRetirementItem {
    /// 该代次的运行时状态标识符。
    pub runtime_state_id: String,
    /// 捕获时该代次的版本状态。
    pub state_version: RuntimeStateVersion,
    /// 关联的待处理退休操作 ID（如果有）。
    pub pending_operation_id: Option<String>,
    /// 该代次目前的退休进度。
    pub status: ScopeRetirementItemStatus,
}

/// 为一个稳定作用域中所有代次统一加上围栏（fence）的持久化清单。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScopeRetirementManifest {
    /// 触发本次退休的删除操作 ID。
    pub delete_operation_id: String,
    /// 删除请求负载的摘要。
    pub payload_digest: String,
    /// 期望的会话 epoch。
    pub expected_conversation_epoch: u64,
    /// 清单中记录的每个代次条目。
    pub items: Vec<ScopeRetirementItem>,
    /// 会话删除操作的回执（尚未产生时为 `None`）。
    pub conversation_delete_receipt: Option<crate::memory::ManagedConversationDeleteReceipt>,
}

impl ScopeRetirementManifest {
    /**
     * 校验清单内容自洽：删除请求本身合法、各代次条目身份不重复且状态有效，
     * 以及会话删除回执（若存在）与本清单的身份、epoch 匹配。
     */
    pub fn validate(&self, scope_id: &str) -> crate::error::Result<()> {
        let delete = crate::memory::ManagedConversationDelete {
            schema_version: RUNTIME_STATE_OPERATION_SCHEMA_VERSION,
            operation_id: self.delete_operation_id.clone(),
            payload_digest: self.payload_digest.clone(),
            conversation_id: scope_id.to_string(),
            expected_epoch: self.expected_conversation_epoch,
        };
        delete.validate()?;
        let mut runtime_ids = std::collections::BTreeSet::new();
        for item in &self.items {
            if item.runtime_state_id.trim().is_empty()
                || !runtime_ids.insert(item.runtime_state_id.as_str())
                || matches!(item.state_version, RuntimeStateVersion::Absent)
                || item
                    .pending_operation_id
                    .as_deref()
                    .is_some_and(str::is_empty)
            {
                return Err(invalid_runtime_operation(
                    "scope retirement manifest contains invalid generation identity",
                ));
            }
        }
        if let Some(receipt) = self.conversation_delete_receipt.as_ref()
            && (!matches!(
                receipt.status,
                crate::memory::ManagedConversationDeleteStatus::Deleted
                    | crate::memory::ManagedConversationDeleteStatus::AlreadyDeleted
            ) || receipt.operation_id != self.delete_operation_id
                || receipt.payload_digest != self.payload_digest
                || receipt.deleted_epoch != self.expected_conversation_epoch
                || receipt.retention_floor_epoch >= receipt.deleted_epoch)
        {
            return Err(invalid_runtime_operation(
                "scope retirement manifest contains invalid conversation delete receipt",
            ));
        }
        Ok(())
    }
}

/// 开始或重放产品作用域退休流程的请求。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScopeRetirementRequest {
    /// 请求负载的 schema 版本号。
    pub schema_version: u16,
    /// 触发本次退休的删除操作 ID。
    pub delete_operation_id: String,
    /// 请求负载的内容摘要。
    pub payload_digest: String,
    /// 目标作用域标识符。
    pub scope_id: String,
    /// 期望的作用域修订号。
    pub expected_scope_revision: u64,
    /// 期望的会话 epoch。
    pub expected_conversation_epoch: u64,
}

impl ScopeRetirementRequest {
    /**
     * 基于一个已校验过的 [`crate::memory::ManagedConversationDelete`] 请求，
     * 构造对应的作用域退休请求。
     */
    pub fn prepare(
        scope_id: impl Into<String>,
        expected_scope_revision: u64,
        delete: &crate::memory::ManagedConversationDelete,
    ) -> crate::error::Result<Self> {
        delete.validate()?;
        let scope_id = scope_id.into();
        if scope_id != delete.conversation_id {
            return Err(invalid_runtime_operation(
                "scope retirement must share the managed conversation identity",
            ));
        }
        Ok(Self {
            schema_version: RUNTIME_STATE_OPERATION_SCHEMA_VERSION,
            delete_operation_id: delete.operation_id.clone(),
            payload_digest: delete.payload_digest.clone(),
            scope_id,
            expected_scope_revision,
            expected_conversation_epoch: delete.expected_epoch,
        })
    }

    /// 校验请求 schema 版本，并复用对应删除请求的校验逻辑。
    pub fn validate(&self) -> crate::error::Result<()> {
        if self.schema_version != RUNTIME_STATE_OPERATION_SCHEMA_VERSION {
            return Err(invalid_runtime_operation(
                "unsupported runtime scope retirement schema",
            ));
        }
        let delete = crate::memory::ManagedConversationDelete {
            schema_version: self.schema_version,
            operation_id: self.delete_operation_id.clone(),
            payload_digest: self.payload_digest.clone(),
            conversation_id: self.scope_id.clone(),
            expected_epoch: self.expected_conversation_epoch,
        };
        delete.validate()
    }
}

/// 校验作用域 ID 与运行时状态 ID 均非空白字符串。
fn validate_runtime_operation_identity(
    scope_id: &str,
    runtime_state_id: &str,
) -> crate::error::Result<()> {
    if scope_id.trim().is_empty() || runtime_state_id.trim().is_empty() {
        return Err(invalid_runtime_operation(
            "runtime operation identities must not be empty",
        ));
    }
    Ok(())
}

/// 对给定值做 JSON 序列化后计算 SHA-256 摘要，作为幂等操作的内容指纹。
fn runtime_operation_digest(value: &impl Serialize) -> crate::error::Result<String> {
    let encoded = serde_json::to_vec(value).map_err(|error| {
        invalid_runtime_operation(format!("failed to serialize runtime operation: {error}"))
    })?;
    Ok(format!("{:x}", Sha256::digest(encoded)))
}

/// 构造一个表示"运行时操作无效"的 `ReactError`。
fn invalid_runtime_operation(message: impl Into<String>) -> crate::error::ReactError {
    echo_core::error::RuntimeStateError::SerializationError(message.into()).into()
}

/// 对一个已存在的退休清单所做的一次持久化推进（advance）。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScopeRetirementAdvance {
    /// 会话（transcript）已被删除，附带删除回执。
    ConversationDeleted {
        /// 会话删除回执。
        receipt: crate::memory::ManagedConversationDeleteReceipt,
    },
    /// 某个运行时代次已被丢弃。
    GenerationDropped {
        /// 被丢弃代次的运行时状态 ID。
        runtime_state_id: String,
        /// 丢弃时仍挂起的操作 ID（如有）。
        pending_operation_id: Option<String>,
    },
    /// 整个退休流程已完成。
    Complete,
}

/// 作用域退休流程的稳定生命周期结果。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScopeRetirementStatus {
    /// 退休流程刚刚开始。
    Begun,
    /// 退休流程正在进行中。
    InProgress,
    /// 退休流程已完成。
    Completed,
    /// 退休流程此前已完成（幂等重放）。
    AlreadyCompleted,
    /// 修订号冲突。
    RevisionConflict,
    /// 会话 epoch 冲突。
    EpochConflict,
    /// 回执已过期（保留的删除证明已超出有效期）。
    ReceiptExpired,
    /// 身份不匹配冲突。
    IdentityConflict,
}

/// 产品作用域退休流程的持久化回执，附带当前清单快照。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScopeRetirementReceipt {
    /// 作用域的权威信息。
    pub scope: RuntimeScopeAuthority,
    /// 当前的退休清单。
    pub manifest: ScopeRetirementManifest,
    /// 已被丢弃的操作 ID 列表。
    pub dropped_operation_ids: Vec<String>,
    /// 当前的保留下限 epoch。
    pub retention_floor_epoch: u64,
    /// 当前的退休流程状态。
    pub status: ScopeRetirementStatus,
}

impl ScopeRetirementReceipt {
    /**
     * 校验回执内部自洽：清单本身合法、已丢弃操作 ID 无重复且与清单记录一致、
     * 保留下限 epoch 与生命周期状态匹配，且"已完成"状态下作用域确已被打上
     * 墓碑标记。
     */
    pub fn validate(&self) -> crate::error::Result<()> {
        self.manifest.validate(&self.scope.scope_id)?;
        let mut dropped = self.dropped_operation_ids.clone();
        dropped.sort();
        if dropped.windows(2).any(|pair| pair.first() == pair.get(1)) {
            return Err(invalid_runtime_operation(
                "scope retirement receipt contains duplicate dropped operation identity",
            ));
        }
        let mut expected = self
            .manifest
            .items
            .iter()
            .filter(|item| item.status == ScopeRetirementItemStatus::DroppedByDelete)
            .filter_map(|item| item.pending_operation_id.clone())
            .collect::<Vec<_>>();
        expected.sort();
        if dropped != expected {
            return Err(invalid_runtime_operation(
                "scope retirement dropped-operation receipt does not match manifest",
            ));
        }
        let expected_retention_floor = self
            .manifest
            .conversation_delete_receipt
            .as_ref()
            .map(|receipt| receipt.retention_floor_epoch)
            .unwrap_or(0);
        let retention_floor_valid = match self.status {
            ScopeRetirementStatus::Completed => {
                self.retention_floor_epoch == expected_retention_floor
            }
            ScopeRetirementStatus::AlreadyCompleted => {
                self.retention_floor_epoch >= expected_retention_floor
                    && self.retention_floor_epoch < self.manifest.expected_conversation_epoch
            }
            ScopeRetirementStatus::ReceiptExpired => {
                self.retention_floor_epoch >= expected_retention_floor
                    && self.retention_floor_epoch >= self.manifest.expected_conversation_epoch
            }
            _ => self.retention_floor_epoch == expected_retention_floor,
        };
        if !retention_floor_valid {
            return Err(invalid_runtime_operation(
                "scope retirement retention floor is inconsistent with lifecycle status",
            ));
        }
        let completed_history = matches!(
            self.status,
            ScopeRetirementStatus::Completed | ScopeRetirementStatus::AlreadyCompleted
        ) || (self.status == ScopeRetirementStatus::ReceiptExpired
            && self.manifest.conversation_delete_receipt.is_some());
        if completed_history
            && (self.scope.lifecycle != RuntimeScopeLifecycle::Tombstoned
                || self.scope.conversation_epoch != Some(self.manifest.expected_conversation_epoch)
                || self.manifest.conversation_delete_receipt.is_none()
                || self
                    .manifest
                    .items
                    .iter()
                    .any(|item| item.status != ScopeRetirementItemStatus::DroppedByDelete))
        {
            return Err(invalid_runtime_operation(
                "completed scope retirement receipt is incomplete",
            ));
        }
        if self.status == ScopeRetirementStatus::ReceiptExpired
            && self.manifest.conversation_delete_receipt.is_none()
            && (self.scope.lifecycle != RuntimeScopeLifecycle::Retiring
                || self
                    .manifest
                    .items
                    .iter()
                    .any(|item| item.status != ScopeRetirementItemStatus::Pending))
        {
            return Err(invalid_runtime_operation(
                "in-progress expired receipt advanced beyond its retained delete proof",
            ));
        }
        Ok(())
    }
}

/// 从一个稳定作用域中删除某一确切运行时状态实例的结果。
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct RuntimeStateClearReceipt {
    /// 作用域标识符。
    pub scope_id: String,
    /// 被清除的运行时状态标识符。
    pub runtime_state_id: String,
    /// 是否确实移除了一个已存在的检查点。
    pub checkpoint_removed: bool,
}

/// 清除一个作用域内所有已建立索引的运行时状态实例的结果。
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct RuntimeStateScopeClearReceipt {
    /// 作用域标识符。
    pub scope_id: String,
    /// 被清除的运行时状态标识符列表。
    pub runtime_state_ids: Vec<String>,
}

/// 删除一个稳定 transcript 及其关联的全部运行时状态谱系的结果。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PersistedConversationDeleteReceipt {
    /// 被删除的会话标识符。
    pub conversation_id: String,
    /// 被清理的运行时状态标识符列表。
    pub runtime_state_ids: Vec<String>,
    /**
     * 完整的持久化托管退休（managed-retirement）结果。旧版存储没有带版本号
     * 的退休权威机制，因此返回 `None`。
     */
    pub retirement: Option<ScopeRetirementReceipt>,
}

/// 决定结算逻辑走"当前尝试"还是"重试已持久化的待处理效果"路径。
#[allow(dead_code)] // Agent 集成切片会把全新派发切换为 CurrentAttempt。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TranscriptProjectionDispatch {
    /// 首次派发，使用当前刚准备好的待处理效果。
    CurrentAttempt,
    /// 恢复流程中重试一个此前已持久化的待处理效果。
    RetryDurablePending,
}

/**
 * transcript 投影结算的内部结果：结算状态本身，以及（如已提交）新的运行时
 * 版本号和已结算的游标。
 */
pub(crate) struct TranscriptProjectionSettlementOutcome {
    /// 本次投影的结算状态。
    pub(crate) settlement: crate::memory::TranscriptProjectionSettlement,
    /// 若结算已提交，则为提交后的新运行时状态版本号。
    pub(crate) committed_version: Option<RuntimeStateVersion>,
    /// 若结算已提交，则为已结算的 transcript 投影游标。
    pub(crate) settled_cursor: Option<TranscriptProjectionCheckpoint>,
}

/// 持久化状态操作的默认超时时长。
const PERSISTED_STATE_OPERATION_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);
/// 为记录持久化失败预留的时间余量，确保截止时间到达前仍有机会写入失败标记。
const PERSISTENCE_FAILURE_RECORD_RESERVE: std::time::Duration = std::time::Duration::from_secs(2);

/**
 * 一次持久化操作的截止时间预算：既有对外暴露的绝对截止时间 `context`，
 * 也有本地 Tokio 时钟下换算出的 `deadline`，以及为记录失败预留余量后的
 * `effect_deadline`。
 */
#[derive(Debug, Clone, Copy)]
struct PersistenceOperationDeadline {
    context: crate::memory::PersistenceCallContext,
    deadline: tokio::time::Instant,
    effect_deadline: tokio::time::Instant,
}

impl PersistenceOperationDeadline {
    /// 仅供测试使用：基于默认超时时长构造一个截止时间预算。
    #[cfg(test)]
    fn new() -> crate::error::Result<Self> {
        Self::from_context(crate::memory::PersistenceCallContext::with_timeout(
            PERSISTED_STATE_OPERATION_TIMEOUT,
        )?)
    }

    /**
     * 将调用方传入的绝对 Unix 截止时间换算为本地 Tokio 时钟下的 `Instant`，
     * 并预留出失败记录所需的时间余量。
     */
    fn from_context(context: crate::memory::PersistenceCallContext) -> crate::error::Result<Self> {
        let absolute_deadline_ms = u64::try_from(context.absolute_deadline_unix_ms)
            .map_err(|_| persistence_deadline_error("invalid persistence deadline"))?;
        let system_deadline = std::time::UNIX_EPOCH
            .checked_add(std::time::Duration::from_millis(absolute_deadline_ms))
            .ok_or_else(|| persistence_deadline_error("persistence deadline overflow"))?;
        let remaining = system_deadline
            .duration_since(std::time::SystemTime::now())
            .map_err(|_| persistence_deadline_error("persistence deadline already elapsed"))?;
        let now = tokio::time::Instant::now();
        let deadline = now
            .checked_add(remaining)
            .ok_or_else(|| persistence_deadline_error("local persistence deadline overflow"))?;
        /*
         * 从正常操作预算中预留一部分时间给持久化失败的 CAS 记录。调用方给的
         * 预算本身很短时，直接使用完整的截止时间，而不是让某个 effect
         * 在第一次被轮询之前就先超时。
         */
        let reserved = deadline
            .checked_sub(PERSISTENCE_FAILURE_RECORD_RESERVE)
            .filter(|reserved| *reserved > now)
            .unwrap_or(deadline);
        Ok(Self {
            context,
            deadline,
            effect_deadline: reserved,
        })
    }
}

/**
 * 在给定的截止时间预算内等待一个持久化 future 完成，超时则返回
 * `DeadlineExceeded` 错误。
 */
async fn await_persistence<T>(
    budget: PersistenceOperationDeadline,
    operation: &str,
    future: impl std::future::Future<Output = crate::error::Result<T>>,
) -> crate::error::Result<T> {
    tokio::time::timeout_at(budget.deadline, future)
        .await
        .map_err(|_| persistence_deadline_error(operation))?
}

/// 构造一个表示"持久化操作超过截止时间"的 `ReactError`。
fn persistence_deadline_error(operation: &str) -> crate::error::ReactError {
    echo_core::error::RuntimeStateError::DeadlineExceeded(operation.to_string()).into()
}

/// 从退休回执中提取出所涉及的运行时状态 ID 列表，构造对外的会话删除回执。
fn persisted_delete_receipt(
    conversation_id: String,
    retirement: ScopeRetirementReceipt,
) -> PersistedConversationDeleteReceipt {
    let runtime_state_ids = retirement
        .manifest
        .items
        .iter()
        .map(|item| item.runtime_state_id.clone())
        .collect();
    PersistedConversationDeleteReceipt {
        conversation_id,
        runtime_state_ids,
        retirement: Some(retirement),
    }
}

/**
 * 用于持久化运行时状态的 trait。
 *
 * 实现可以基于 SQLite、JSON 文件或其他持久化后端。
 */
pub trait RuntimeStateStore: Send + Sync {
    /// 在不执行任何 I/O 的前提下，报告是否支持带版本号的检查点。
    fn runtime_state_capability(&self) -> RuntimeStateCapability {
        RuntimeStateCapability::Unsupported
    }

    /**
     * 报告带上下文的调用是否真正强制执行了传入的绝对截止时间，而不是在
     * 适配器的默认实现中被默默丢弃。
     */
    fn persistence_call_capability(&self) -> crate::memory::PersistenceCallCapability {
        crate::memory::PersistenceCallCapability::Unsupported
    }

    /// 加载一个受管理的检查点 / 墓碑，及其所属稳定作用域的当前修订号。
    fn load_runtime_state<'a>(
        &'a self,
        _scope_id: &'a str,
        _runtime_state_id: &'a str,
    ) -> futures::future::BoxFuture<'a, crate::error::Result<Option<ManagedRuntimeStateSnapshot>>>
    {
        Box::pin(async {
            Err(echo_core::error::RuntimeStateError::Unsupported(
                "revisioned runtime-state load".to_string(),
            )
            .into())
        })
    }

    /**
     * 带调用上下文（携带绝对截止时间）的 [`Self::load_runtime_state`] 变体。
     * 默认实现直接委托给不带上下文的版本。
     */
    fn load_runtime_state_with_context<'a>(
        &'a self,
        _context: crate::memory::PersistenceCallContext,
        scope_id: &'a str,
        runtime_state_id: &'a str,
    ) -> futures::future::BoxFuture<'a, crate::error::Result<Option<ManagedRuntimeStateSnapshot>>>
    {
        self.load_runtime_state(scope_id, runtime_state_id)
    }

    /// 加载用于围栏准入判断与删除操作的稳定作用域权威信息。
    fn load_scope_authority<'a>(
        &'a self,
        _scope_id: &'a str,
    ) -> futures::future::BoxFuture<'a, crate::error::Result<Option<RuntimeScopeAuthority>>> {
        Box::pin(async {
            Err(echo_core::error::RuntimeStateError::Unsupported(
                "runtime scope authority load".to_string(),
            )
            .into())
        })
    }

    /**
     * 带调用上下文的 [`Self::load_scope_authority`] 变体。默认实现直接委托
     * 给不带上下文的版本。
     */
    fn load_scope_authority_with_context<'a>(
        &'a self,
        _context: crate::memory::PersistenceCallContext,
        scope_id: &'a str,
    ) -> futures::future::BoxFuture<'a, crate::error::Result<Option<RuntimeScopeAuthority>>> {
        self.load_scope_authority(scope_id)
    }

    /// 仅当作用域与代次版本号都匹配时，原子性地写入一个检查点。
    fn compare_and_save_checkpoint<'a>(
        &'a self,
        _request: RuntimeCheckpointCasRequest,
    ) -> futures::future::BoxFuture<'a, crate::error::Result<RuntimeCheckpointCasReceipt>> {
        Box::pin(async {
            Err(echo_core::error::RuntimeStateError::Unsupported(
                "runtime checkpoint compare-and-save".to_string(),
            )
            .into())
        })
    }

    /**
     * 带调用上下文的 [`Self::compare_and_save_checkpoint`] 变体。默认实现
     * 直接委托给不带上下文的版本。
     */
    fn compare_and_save_checkpoint_with_context<'a>(
        &'a self,
        _context: crate::memory::PersistenceCallContext,
        request: RuntimeCheckpointCasRequest,
    ) -> futures::future::BoxFuture<'a, crate::error::Result<RuntimeCheckpointCasReceipt>> {
        self.compare_and_save_checkpoint(request)
    }

    /// 仅当提供了匹配的已应用回执时，清除一个待处理的 transcript 效果标记。
    fn acknowledge_transcript_projection<'a>(
        &'a self,
        _request: RuntimeTranscriptAckRequest,
    ) -> futures::future::BoxFuture<'a, crate::error::Result<RuntimeCheckpointCasReceipt>> {
        Box::pin(async {
            Err(echo_core::error::RuntimeStateError::Unsupported(
                "runtime transcript acknowledgement".to_string(),
            )
            .into())
        })
    }

    /**
     * 带调用上下文的 [`Self::acknowledge_transcript_projection`] 变体。默认
     * 实现直接委托给不带上下文的版本。
     */
    fn acknowledge_transcript_projection_with_context<'a>(
        &'a self,
        _context: crate::memory::PersistenceCallContext,
        request: RuntimeTranscriptAckRequest,
    ) -> futures::future::BoxFuture<'a, crate::error::Result<RuntimeCheckpointCasReceipt>> {
        self.acknowledge_transcript_projection(request)
    }

    /// 用一个持久化墓碑退休某一代次。
    fn retire_runtime_generation<'a>(
        &'a self,
        _request: RuntimeGenerationRetireRequest,
    ) -> futures::future::BoxFuture<'a, crate::error::Result<RuntimeGenerationRetireReceipt>> {
        Box::pin(async {
            Err(echo_core::error::RuntimeStateError::Unsupported(
                "runtime generation retirement".to_string(),
            )
            .into())
        })
    }

    /**
     * 带调用上下文的 [`Self::retire_runtime_generation`] 变体。默认实现
     * 直接委托给不带上下文的版本。
     */
    fn retire_runtime_generation_with_context<'a>(
        &'a self,
        _context: crate::memory::PersistenceCallContext,
        request: RuntimeGenerationRetireRequest,
    ) -> futures::future::BoxFuture<'a, crate::error::Result<RuntimeGenerationRetireReceipt>> {
        self.retire_runtime_generation(request)
    }

    /// 为一个稳定作用域加上围栏，并原子性地捕获其完整的删除清单。
    fn begin_scope_retirement<'a>(
        &'a self,
        _request: ScopeRetirementRequest,
    ) -> futures::future::BoxFuture<'a, crate::error::Result<ScopeRetirementReceipt>> {
        Box::pin(async {
            Err(echo_core::error::RuntimeStateError::Unsupported(
                "runtime scope retirement".to_string(),
            )
            .into())
        })
    }

    /**
     * 带调用上下文的 [`Self::begin_scope_retirement`] 变体。默认实现直接
     * 委托给不带上下文的版本。
     */
    fn begin_scope_retirement_with_context<'a>(
        &'a self,
        _context: crate::memory::PersistenceCallContext,
        request: ScopeRetirementRequest,
    ) -> futures::future::BoxFuture<'a, crate::error::Result<ScopeRetirementReceipt>> {
        self.begin_scope_retirement(request)
    }

    /// 持久化一次针对作用域退休 saga 的幂等推进。
    fn continue_scope_retirement<'a>(
        &'a self,
        _scope_id: &'a str,
        _delete_operation_id: &'a str,
        _expected_scope_revision: u64,
        _advance: ScopeRetirementAdvance,
    ) -> futures::future::BoxFuture<'a, crate::error::Result<ScopeRetirementReceipt>> {
        Box::pin(async {
            Err(echo_core::error::RuntimeStateError::Unsupported(
                "runtime scope retirement advancement".to_string(),
            )
            .into())
        })
    }

    /**
     * 带调用上下文的 [`Self::continue_scope_retirement`] 变体。默认实现
     * 直接委托给不带上下文的版本。
     */
    fn continue_scope_retirement_with_context<'a>(
        &'a self,
        _context: crate::memory::PersistenceCallContext,
        scope_id: &'a str,
        delete_operation_id: &'a str,
        expected_scope_revision: u64,
        advance: ScopeRetirementAdvance,
    ) -> futures::future::BoxFuture<'a, crate::error::Result<ScopeRetirementReceipt>> {
        self.continue_scope_retirement(
            scope_id,
            delete_operation_id,
            expected_scope_revision,
            advance,
        )
    }

    /// 获取某个会话最近的一次检查点（如果存在）。
    fn get_checkpoint<'a>(
        &'a self,
        conversation_id: &'a str,
    ) -> futures::future::BoxFuture<'a, crate::error::Result<Option<AgentCheckpoint>>>;

    /// 保存一个检查点。
    fn save_checkpoint<'a>(
        &'a self,
        checkpoint: &'a AgentCheckpoint,
    ) -> futures::future::BoxFuture<'a, crate::error::Result<()>>;

    /**
     * 保存一个检查点，并将其全局唯一的运行时身份持久化绑定到一个稳定的
     * 产品 / 会话作用域上。
     *
     * 实现必须将检查点写入与全局唯一的作用域归属关系，作为同一个可恢复的
     * 权威一起提交。只有在崩溃后能够无歧义地从该权威重建时，二级索引才
     * 允许暂时滞后。
     */
    fn save_checkpoint_for_scope<'a>(
        &'a self,
        scope_id: &'a str,
        checkpoint: &'a AgentCheckpoint,
    ) -> futures::future::BoxFuture<'a, crate::error::Result<()>>;

    /// 列出持久化绑定到 `scope_id` 的、已排序的运行时状态标识符。
    fn runtime_state_ids<'a>(
        &'a self,
        scope_id: &'a str,
    ) -> futures::future::BoxFuture<'a, crate::error::Result<Vec<String>>>;

    /// 删除某一确切的运行时状态实例，并移除其与作用域的绑定关系。
    fn clear_runtime_state<'a>(
        &'a self,
        scope_id: &'a str,
        runtime_state_id: &'a str,
    ) -> futures::future::BoxFuture<'a, crate::error::Result<RuntimeStateClearReceipt>>;

    /// 删除 `scope_id` 下建立了索引的全部运行时状态实例。
    fn clear_runtime_state_scope<'a>(
        &'a self,
        scope_id: &'a str,
    ) -> futures::future::BoxFuture<'a, crate::error::Result<RuntimeStateScopeClearReceipt>>;

    /// 删除某个会话的全部状态。
    fn clear_conversation<'a>(
        &'a self,
        conversation_id: &'a str,
    ) -> futures::future::BoxFuture<'a, crate::error::Result<()>>;
}

/**
 * 恢复并结算某一运行时代次所持有的、唯一的持久化 transcript 效果。
 *
 * 恢复流程始终复用检查点中已保存的批次身份。它首先通过运行时 CAS 推进
 * 持久化的尝试计数，然后派发这个确切的批次并用后端回执做确认。所有 Store
 * 调用都携带 `context`；框架会从绝对 Unix 截止时间换算出本地 Tokio 截止
 * 时间，而不会通过 API 暴露进程本地时间。
 */
pub async fn settle_pending_transcript_projection(
    conversation_store: &dyn crate::memory::ConversationStore,
    runtime_state_store: &dyn RuntimeStateStore,
    scope_id: &str,
    runtime_state_id: &str,
    context: crate::memory::PersistenceCallContext,
) -> crate::error::Result<crate::memory::TranscriptProjectionSettlement> {
    if !managed_persistence_pair(conversation_store, runtime_state_store)? {
        return Err(echo_core::error::RuntimeStateError::Unsupported(
            "transcript settlement requires atomic conversation and revisioned runtime stores"
                .to_string(),
        )
        .into());
    }
    let budget = PersistenceOperationDeadline::from_context(context)?;
    await_persistence(
        budget,
        "durable pending transcript settlement",
        settle_pending_transcript_projection_with_budget(
            conversation_store,
            runtime_state_store,
            scope_id,
            runtime_state_id,
            budget,
        ),
    )
    .await
}

/**
 * [`settle_pending_transcript_projection`] 的内部实现：先加载运行时状态，
 * 再在给定的截止时间预算内结算其中的待处理效果。
 */
async fn settle_pending_transcript_projection_with_budget(
    conversation_store: &dyn crate::memory::ConversationStore,
    runtime_state_store: &dyn RuntimeStateStore,
    scope_id: &str,
    runtime_state_id: &str,
    budget: PersistenceOperationDeadline,
) -> crate::error::Result<crate::memory::TranscriptProjectionSettlement> {
    let state = await_persistence(
        budget,
        "runtime generation load for transcript settlement",
        runtime_state_store.load_runtime_state_with_context(
            budget.context,
            scope_id,
            runtime_state_id,
        ),
    )
    .await?
    .ok_or_else(|| {
        echo_core::error::RuntimeStateError::NotFound(format!(
            "runtime state {runtime_state_id} is not owned by scope {scope_id}"
        ))
    })?;
    let outcome = await_persistence(
        budget,
        "loaded durable pending transcript settlement",
        settle_loaded_pending_transcript_projection_with_budget(
            conversation_store,
            runtime_state_store,
            scope_id,
            runtime_state_id,
            state,
            TranscriptProjectionDispatch::RetryDurablePending,
            budget,
        ),
    )
    .await?;
    Ok(outcome.settlement)
}

/**
 * 供"全新 Agent 派发"与"持久化恢复"两条路径共用的规范化结算核心逻辑
 * （已加载状态版本）。调用方不得自行实现 apply/ack 的先后顺序。
 */
pub(crate) async fn settle_loaded_pending_transcript_projection(
    conversation_store: &dyn crate::memory::ConversationStore,
    runtime_state_store: &dyn RuntimeStateStore,
    scope_id: &str,
    runtime_state_id: &str,
    state: ManagedRuntimeStateSnapshot,
    dispatch: TranscriptProjectionDispatch,
    context: crate::memory::PersistenceCallContext,
) -> crate::error::Result<TranscriptProjectionSettlementOutcome> {
    if !managed_persistence_pair(conversation_store, runtime_state_store)? {
        return Err(echo_core::error::RuntimeStateError::Unsupported(
            "loaded transcript settlement requires managed stores with absolute deadlines"
                .to_string(),
        )
        .into());
    }
    let budget = PersistenceOperationDeadline::from_context(context)?;
    await_persistence(
        budget,
        "loaded pending transcript settlement",
        settle_loaded_pending_transcript_projection_with_budget(
            conversation_store,
            runtime_state_store,
            scope_id,
            runtime_state_id,
            state,
            dispatch,
            budget,
        ),
    )
    .await
}

/**
 * [`settle_loaded_pending_transcript_projection`] 的内部实现，携带截止时间
 * 预算。校验已加载状态的身份匹配后，按情况处理"无检查点""无待处理效果"
 * 或实际需要结算的待处理效果三种情形。
 */
async fn settle_loaded_pending_transcript_projection_with_budget(
    conversation_store: &dyn crate::memory::ConversationStore,
    runtime_state_store: &dyn RuntimeStateStore,
    scope_id: &str,
    runtime_state_id: &str,
    state: ManagedRuntimeStateSnapshot,
    dispatch: TranscriptProjectionDispatch,
    budget: PersistenceOperationDeadline,
) -> crate::error::Result<TranscriptProjectionSettlementOutcome> {
    if state.scope.scope_id != scope_id || state.runtime_state_id != runtime_state_id {
        return Err(echo_core::error::RuntimeStateError::SerializationError(
            "loaded runtime state does not match transcript settlement identity".to_string(),
        )
        .into());
    }
    let Some(checkpoint) = state.checkpoint.as_ref() else {
        return Ok(transcript_settlement_outcome(transcript_settlement(
            crate::memory::TranscriptProjectionSettlementStatus::Settled,
            None,
            Some(scope_id.to_string()),
            Some(runtime_state_id.to_string()),
            0,
            None,
            None,
        )));
    };
    let payload = checkpoint.restore_managed_runtime_payload()?;
    let Some(pending) = payload.pending_transcript_projection else {
        return Ok(transcript_settlement_outcome(transcript_settlement(
            crate::memory::TranscriptProjectionSettlementStatus::Settled,
            None,
            Some(scope_id.to_string()),
            Some(runtime_state_id.to_string()),
            0,
            None,
            None,
        )));
    };
    if pending.batch.conversation_id != scope_id || pending.batch.generation_id != runtime_state_id
    {
        return Ok(transcript_settlement_outcome(pending_settlement(
            &pending,
            crate::memory::TranscriptProjectionSettlementStatus::Blocked,
            crate::memory::TranscriptProjectionErrorClass::CorruptState,
            "durable pending transcript identity does not match runtime ownership".to_string(),
        )));
    }
    if let Some(attempt_class) = pending.last_attempt_class.as_ref()
        && !transcript_attempt_is_retryable(attempt_class)
    {
        let status = if *attempt_class == TranscriptProjectionAttemptClass::SemanticConflict {
            crate::memory::TranscriptProjectionSettlementStatus::Conflict
        } else {
            crate::memory::TranscriptProjectionSettlementStatus::Blocked
        };
        return Ok(transcript_settlement_outcome(pending_settlement(
            &pending,
            status,
            transcript_attempt_error_class(attempt_class),
            pending
                .last_error
                .clone()
                .unwrap_or_else(|| "durable transcript failure is not retryable".to_string()),
        )));
    }
    if dispatch == TranscriptProjectionDispatch::CurrentAttempt
        && pending.last_attempt_class.is_some()
    {
        return Ok(transcript_settlement_outcome(pending_settlement(
            &pending,
            crate::memory::TranscriptProjectionSettlementStatus::Blocked,
            crate::memory::TranscriptProjectionErrorClass::CorruptState,
            "current-attempt dispatch requires a freshly prepared pending effect".to_string(),
        )));
    }

    let (state, messages, pending) = match dispatch {
        TranscriptProjectionDispatch::CurrentAttempt => (state, payload.messages, pending),
        TranscriptProjectionDispatch::RetryDurablePending => {
            match prepare_transcript_projection_retry(
                runtime_state_store,
                state,
                payload.messages,
                pending,
                budget,
            )
            .await?
            {
                TranscriptProjectionDispatchPreparation::Ready {
                    state,
                    messages,
                    pending,
                } => (state, messages, *pending),
                TranscriptProjectionDispatchPreparation::Terminal(settlement) => {
                    return Ok(transcript_settlement_outcome(settlement));
                }
            }
        }
    };
    dispatch_transcript_projection(
        conversation_store,
        runtime_state_store,
        state,
        messages,
        pending,
        budget,
    )
    .await
}

/// 为重试一个已持久化的待处理 transcript 投影所做准备的结果。
enum TranscriptProjectionDispatchPreparation {
    /// 已成功推进持久化尝试计数，可以继续派发。
    Ready {
        state: ManagedRuntimeStateSnapshot,
        messages: Vec<crate::llm::types::Message>,
        pending: Box<PendingTranscriptProjection>,
    },
    /// 准备阶段已得出最终结算结果，无需继续派发。
    Terminal(crate::memory::TranscriptProjectionSettlement),
}

/**
 * 在重试一个持久化的待处理 transcript 投影之前，先推进其尝试计数并把新的
 * 检查点写回运行时存储（通过 CAS），为后续实际派发做准备。
 */
async fn prepare_transcript_projection_retry(
    runtime_state_store: &dyn RuntimeStateStore,
    state: ManagedRuntimeStateSnapshot,
    messages: Vec<crate::llm::types::Message>,
    mut pending: PendingTranscriptProjection,
    budget: PersistenceOperationDeadline,
) -> crate::error::Result<TranscriptProjectionDispatchPreparation> {
    let current_revision = match state.version {
        RuntimeStateVersion::Managed { revision } if revision == pending.base_runtime_revision => {
            revision
        }
        _ => {
            return Ok(TranscriptProjectionDispatchPreparation::Terminal(
                pending_settlement(
                    &pending,
                    crate::memory::TranscriptProjectionSettlementStatus::Blocked,
                    crate::memory::TranscriptProjectionErrorClass::CorruptState,
                    "durable pending transcript revision does not match runtime authority"
                        .to_string(),
                ),
            ));
        }
    };
    let next_revision = current_revision.checked_add(1).ok_or_else(|| {
        echo_core::error::RuntimeStateError::RevisionExhausted(
            "pending retry revision reached u64::MAX".to_string(),
        )
    })?;
    let next_attempt = pending.attempt.checked_add(1).ok_or_else(|| {
        echo_core::error::RuntimeStateError::RevisionExhausted(
            "pending transcript attempt reached u32::MAX".to_string(),
        )
    })?;
    pending.attempt = next_attempt;
    pending.last_attempt_class = None;
    pending.last_error = None;
    pending.base_runtime_revision = next_revision;
    let mut checkpoint = state.checkpoint.clone().ok_or_else(|| {
        echo_core::error::RuntimeStateError::SerializationError(
            "durable pending runtime state lost its checkpoint".to_string(),
        )
    })?;
    checkpoint.messages_json = AgentCheckpoint::serialize_managed_payload(
        messages.clone(),
        Some(pending.cursor_before.clone()),
        Some(pending.clone()),
    )?;
    checkpoint.timestamp = Utc::now();
    let retry = tokio::time::timeout_at(
        budget.effect_deadline,
        runtime_state_store.compare_and_save_checkpoint_with_context(
            budget.context,
            RuntimeCheckpointCasRequest {
                scope_id: state.scope.scope_id.clone(),
                runtime_state_id: state.runtime_state_id.clone(),
                conversation_epoch: state.scope.conversation_epoch,
                expected_scope_revision: state.scope.revision,
                expected_state_version: RuntimeStateExpectedVersion::Managed {
                    revision: current_revision,
                },
                checkpoint: checkpoint.clone(),
            },
        ),
    )
    .await;
    let retry = match retry {
        Err(_) => {
            return Ok(TranscriptProjectionDispatchPreparation::Terminal(
                pending_settlement(
                    &pending,
                    crate::memory::TranscriptProjectionSettlementStatus::Deferred,
                    crate::memory::TranscriptProjectionErrorClass::DeadlineExceeded,
                    "pending retry checkpoint deadline elapsed with unknown outcome".to_string(),
                ),
            ));
        }
        Ok(Err(error)) => {
            let (status, error_class) = classify_transcript_persistence_error(&error);
            let reported_attempt = if error_class
                == crate::memory::TranscriptProjectionErrorClass::TransientNoCommit
            {
                pending.attempt.saturating_sub(1)
            } else {
                pending.attempt
            };
            let mut settlement =
                pending_settlement(&pending, status, error_class, error.to_string());
            settlement.attempt = reported_attempt;
            return Ok(TranscriptProjectionDispatchPreparation::Terminal(
                settlement,
            ));
        }
        Ok(Ok(receipt)) => receipt,
    };
    match retry.status {
        RuntimeCheckpointCasStatus::Applied | RuntimeCheckpointCasStatus::AlreadyCurrent => {
            Ok(TranscriptProjectionDispatchPreparation::Ready {
                state: ManagedRuntimeStateSnapshot {
                    scope: retry.scope,
                    runtime_state_id: state.runtime_state_id,
                    version: retry.version,
                    checkpoint: Some(checkpoint),
                },
                messages,
                pending: Box::new(pending),
            })
        }
        RuntimeCheckpointCasStatus::RevisionConflict => Ok(
            TranscriptProjectionDispatchPreparation::Terminal(pending_settlement(
                &pending,
                crate::memory::TranscriptProjectionSettlementStatus::Deferred,
                crate::memory::TranscriptProjectionErrorClass::RevisionConflict,
                "pending retry checkpoint revision changed".to_string(),
            )),
        ),
        RuntimeCheckpointCasStatus::GenerationRetired | RuntimeCheckpointCasStatus::ScopeFenced => {
            Ok(TranscriptProjectionDispatchPreparation::Terminal(
                pending_settlement(
                    &pending,
                    crate::memory::TranscriptProjectionSettlementStatus::Conflict,
                    crate::memory::TranscriptProjectionErrorClass::SemanticConflict,
                    format!("pending retry checkpoint was fenced: {:?}", retry.status),
                ),
            ))
        }
    }
}

/**
 * 实际派发一个已准备好的待处理 transcript 投影：先把批次应用到会话存储，
 * 再用应用回执向运行时存储确认，最终结算成功或记录失败。
 */
async fn dispatch_transcript_projection(
    conversation_store: &dyn crate::memory::ConversationStore,
    runtime_state_store: &dyn RuntimeStateStore,
    state: ManagedRuntimeStateSnapshot,
    messages: Vec<crate::llm::types::Message>,
    pending: PendingTranscriptProjection,
    budget: PersistenceOperationDeadline,
) -> crate::error::Result<TranscriptProjectionSettlementOutcome> {
    if !matches!(
        state.version,
        RuntimeStateVersion::Managed { revision } if revision == pending.base_runtime_revision
    ) {
        return Ok(transcript_settlement_outcome(pending_settlement(
            &pending,
            crate::memory::TranscriptProjectionSettlementStatus::Blocked,
            crate::memory::TranscriptProjectionErrorClass::CorruptState,
            "pending transcript revision does not match runtime authority".to_string(),
        )));
    }
    let apply = tokio::time::timeout_at(
        budget.effect_deadline,
        conversation_store
            .apply_transcript_projection_with_context(budget.context, pending.batch.clone()),
    )
    .await;
    let projection_receipt = match apply {
        Err(_) => {
            let detail = "transcript apply deadline elapsed with unknown outcome";
            let settlement = recorded_transcript_failure_settlement(
                runtime_state_store,
                &state,
                &messages,
                &pending,
                TranscriptProjectionFailure {
                    status: crate::memory::TranscriptProjectionSettlementStatus::Deferred,
                    error_class: crate::memory::TranscriptProjectionErrorClass::DeadlineExceeded,
                    detail,
                },
                budget,
            )
            .await;
            return Ok(transcript_settlement_outcome(settlement));
        }
        Ok(Err(error)) => {
            let (status, error_class) = classify_transcript_persistence_error(&error);
            let detail = error.to_string();
            let settlement = recorded_transcript_failure_settlement(
                runtime_state_store,
                &state,
                &messages,
                &pending,
                TranscriptProjectionFailure {
                    status,
                    error_class,
                    detail: &detail,
                },
                budget,
            )
            .await;
            return Ok(transcript_settlement_outcome(settlement));
        }
        Ok(Ok(receipt)) => receipt,
    };
    if !matches!(
        projection_receipt.status,
        crate::memory::TranscriptProjectionApplyStatus::Applied
            | crate::memory::TranscriptProjectionApplyStatus::AlreadyApplied
    ) {
        let detail = "conversation store rejected transcript projection";
        let settlement = recorded_transcript_failure_settlement(
            runtime_state_store,
            &state,
            &messages,
            &pending,
            TranscriptProjectionFailure {
                status: crate::memory::TranscriptProjectionSettlementStatus::Conflict,
                error_class: crate::memory::TranscriptProjectionErrorClass::SemanticConflict,
                detail,
            },
            budget,
        )
        .await;
        return Ok(transcript_settlement_outcome(settlement));
    }
    if projection_receipt.operation_id != pending.batch.operation_id
        || projection_receipt.payload_digest != pending.batch.payload_digest
    {
        let detail = "conversation store returned a mismatched projection receipt";
        let settlement = recorded_transcript_failure_settlement(
            runtime_state_store,
            &state,
            &messages,
            &pending,
            TranscriptProjectionFailure {
                status: crate::memory::TranscriptProjectionSettlementStatus::Conflict,
                error_class: crate::memory::TranscriptProjectionErrorClass::SemanticConflict,
                detail,
            },
            budget,
        )
        .await;
        return Ok(transcript_settlement_outcome(settlement));
    }

    let mut checkpoint = state.checkpoint.clone().ok_or_else(|| {
        echo_core::error::RuntimeStateError::SerializationError(
            "prepared pending runtime state lost its checkpoint".to_string(),
        )
    })?;
    checkpoint.messages_json = AgentCheckpoint::serialize_managed_payload(
        messages.clone(),
        Some(pending.cursor_after.clone()),
        None,
    )?;
    checkpoint.timestamp = Utc::now();
    let current_revision = match state.version {
        RuntimeStateVersion::Managed { revision } => revision,
        ref other => {
            return Err(
                echo_core::error::RuntimeStateError::SerializationError(format!(
                    "prepared pending checkpoint has non-managed version {other:?}"
                ))
                .into(),
            );
        }
    };
    let conversation_epoch = state.scope.conversation_epoch.ok_or_else(|| {
        echo_core::error::RuntimeStateError::SerializationError(
            "managed transcript acknowledgement is missing conversation epoch".to_string(),
        )
    })?;
    let ack = tokio::time::timeout_at(
        budget.effect_deadline,
        runtime_state_store.acknowledge_transcript_projection_with_context(
            budget.context,
            RuntimeTranscriptAckRequest {
                scope_id: state.scope.scope_id.clone(),
                runtime_state_id: state.runtime_state_id.clone(),
                conversation_epoch,
                expected_scope_revision: state.scope.revision,
                expected_state_version: RuntimeStateExpectedVersion::Managed {
                    revision: current_revision,
                },
                projection_receipt,
                checkpoint,
            },
        ),
    )
    .await;
    let ack = match ack {
        Err(_) => {
            let detail = "checkpoint acknowledgement deadline elapsed with unknown outcome";
            let settlement = recorded_transcript_failure_settlement(
                runtime_state_store,
                &state,
                &messages,
                &pending,
                TranscriptProjectionFailure {
                    status: crate::memory::TranscriptProjectionSettlementStatus::Deferred,
                    error_class: crate::memory::TranscriptProjectionErrorClass::DeadlineExceeded,
                    detail,
                },
                budget,
            )
            .await;
            return Ok(transcript_settlement_outcome(settlement));
        }
        Ok(Err(error)) => {
            let (status, error_class) = classify_transcript_persistence_error(&error);
            let detail = error.to_string();
            let settlement = recorded_transcript_failure_settlement(
                runtime_state_store,
                &state,
                &messages,
                &pending,
                TranscriptProjectionFailure {
                    status,
                    error_class,
                    detail: &detail,
                },
                budget,
            )
            .await;
            return Ok(transcript_settlement_outcome(settlement));
        }
        Ok(Ok(receipt)) => receipt,
    };
    match ack.status {
        RuntimeCheckpointCasStatus::Applied | RuntimeCheckpointCasStatus::AlreadyCurrent => {
            Ok(TranscriptProjectionSettlementOutcome {
                settlement: transcript_settlement(
                    crate::memory::TranscriptProjectionSettlementStatus::Settled,
                    Some(pending.batch.operation_id),
                    Some(pending.batch.conversation_id),
                    Some(pending.batch.generation_id),
                    pending.attempt,
                    None,
                    None,
                ),
                committed_version: Some(ack.version),
                settled_cursor: Some(pending.cursor_after),
            })
        }
        RuntimeCheckpointCasStatus::RevisionConflict => {
            let detail = "checkpoint acknowledgement revision changed";
            let settlement = recorded_transcript_failure_settlement(
                runtime_state_store,
                &state,
                &messages,
                &pending,
                TranscriptProjectionFailure {
                    status: crate::memory::TranscriptProjectionSettlementStatus::Deferred,
                    error_class: crate::memory::TranscriptProjectionErrorClass::RevisionConflict,
                    detail,
                },
                budget,
            )
            .await;
            Ok(transcript_settlement_outcome(settlement))
        }
        RuntimeCheckpointCasStatus::GenerationRetired | RuntimeCheckpointCasStatus::ScopeFenced => {
            let detail = format!("checkpoint acknowledgement was fenced: {:?}", ack.status);
            let settlement = recorded_transcript_failure_settlement(
                runtime_state_store,
                &state,
                &messages,
                &pending,
                TranscriptProjectionFailure {
                    status: crate::memory::TranscriptProjectionSettlementStatus::Conflict,
                    error_class: crate::memory::TranscriptProjectionErrorClass::SemanticConflict,
                    detail: &detail,
                },
                budget,
            )
            .await;
            Ok(transcript_settlement_outcome(settlement))
        }
    }
}

/// 描述一次 transcript 投影派发失败的分类结果与详情。
struct TranscriptProjectionFailure<'a> {
    status: crate::memory::TranscriptProjectionSettlementStatus,
    error_class: crate::memory::TranscriptProjectionErrorClass,
    detail: &'a str,
}

/**
 * 尝试把本次失败的分类结果持久化记录下来，再据此构造最终的结算结果；
 * 如果连失败记录本身都未能持久化，则退化为"结果未知"（OutcomeUnknown），
 * 确保绝不会在无法证明的情况下错误地上报成功或错误的失败分类。
 */
async fn recorded_transcript_failure_settlement(
    runtime_state_store: &dyn RuntimeStateStore,
    state: &ManagedRuntimeStateSnapshot,
    messages: &[crate::llm::types::Message],
    pending: &PendingTranscriptProjection,
    failure: TranscriptProjectionFailure<'_>,
    budget: PersistenceOperationDeadline,
) -> crate::memory::TranscriptProjectionSettlement {
    let attempt_class = transcript_error_attempt_class(failure.error_class);
    let retryable = transcript_attempt_is_retryable(&attempt_class);
    if persist_transcript_attempt_failure(
        runtime_state_store,
        state,
        messages,
        pending,
        attempt_class,
        failure.detail,
        budget,
    )
    .await
    {
        pending_settlement(
            pending,
            failure.status,
            failure.error_class,
            failure.detail.to_string(),
        )
    } else {
        pending_settlement(
            pending,
            if retryable {
                crate::memory::TranscriptProjectionSettlementStatus::Deferred
            } else {
                failure.status
            },
            crate::memory::TranscriptProjectionErrorClass::OutcomeUnknown,
            format!(
                "failure classification is not durable; recovery must re-prove it: {}",
                failure.detail
            ),
        )
    }
}

/**
 * 尝试用 CAS 把本次失败的尝试分类写回运行时检查点；写入本身失败时，退回
 * 去检查该分类是否已经通过其他方式被持久化记录下来。
 */
async fn persist_transcript_attempt_failure(
    runtime_state_store: &dyn RuntimeStateStore,
    state: &ManagedRuntimeStateSnapshot,
    messages: &[crate::llm::types::Message],
    pending: &PendingTranscriptProjection,
    attempt_class: TranscriptProjectionAttemptClass,
    detail: &str,
    budget: PersistenceOperationDeadline,
) -> bool {
    let current_revision = match state.version {
        RuntimeStateVersion::Managed { revision } => revision,
        _ => return false,
    };
    let Some(next_revision) = current_revision.checked_add(1) else {
        return false;
    };
    let Some(mut checkpoint) = state.checkpoint.clone() else {
        return false;
    };
    let mut recorded = pending.clone();
    recorded.base_runtime_revision = next_revision;
    recorded.last_attempt_class = Some(attempt_class.clone());
    recorded.last_error = Some(detail.to_string());
    let Ok(messages_json) = AgentCheckpoint::serialize_managed_payload(
        messages.to_vec(),
        Some(recorded.cursor_before.clone()),
        Some(recorded),
    ) else {
        return false;
    };
    checkpoint.messages_json = messages_json;
    checkpoint.timestamp = Utc::now();
    let persisted = tokio::time::timeout_at(
        budget.deadline,
        runtime_state_store.compare_and_save_checkpoint_with_context(
            budget.context,
            RuntimeCheckpointCasRequest {
                scope_id: state.scope.scope_id.clone(),
                runtime_state_id: state.runtime_state_id.clone(),
                conversation_epoch: state.scope.conversation_epoch,
                expected_scope_revision: state.scope.revision,
                expected_state_version: RuntimeStateExpectedVersion::Managed {
                    revision: current_revision,
                },
                checkpoint,
            },
        ),
    )
    .await;
    if matches!(
        persisted,
        Ok(Ok(RuntimeCheckpointCasReceipt {
            status: RuntimeCheckpointCasStatus::Applied
                | RuntimeCheckpointCasStatus::AlreadyCurrent,
            ..
        }))
    ) {
        return true;
    }
    transcript_failure_record_is_durable(
        runtime_state_store,
        state,
        pending,
        &attempt_class,
        detail,
        budget,
    )
    .await
}

/**
 * 重新加载运行时状态，检查其中记录的失败分类是否确实与本次尝试匹配，
 * 用于确认某次失败记录是否已经被持久化下来。
 */
async fn transcript_failure_record_is_durable(
    runtime_state_store: &dyn RuntimeStateStore,
    state: &ManagedRuntimeStateSnapshot,
    pending: &PendingTranscriptProjection,
    attempt_class: &TranscriptProjectionAttemptClass,
    detail: &str,
    budget: PersistenceOperationDeadline,
) -> bool {
    let loaded = tokio::time::timeout_at(
        budget.deadline,
        runtime_state_store.load_runtime_state_with_context(
            budget.context,
            &state.scope.scope_id,
            &state.runtime_state_id,
        ),
    )
    .await;
    let Ok(Ok(Some(loaded))) = loaded else {
        return false;
    };
    let Some(checkpoint) = loaded.checkpoint.as_ref() else {
        return false;
    };
    let Ok(payload) = checkpoint.restore_managed_runtime_payload() else {
        return false;
    };
    let Some(recorded) = payload.pending_transcript_projection else {
        return false;
    };
    matches!(
        loaded.version,
        RuntimeStateVersion::Managed { revision }
            if revision == recorded.base_runtime_revision
    ) && recorded.batch.operation_id == pending.batch.operation_id
        && recorded.batch.payload_digest == pending.batch.payload_digest
        && recorded.attempt == pending.attempt
        && recorded.last_attempt_class.as_ref() == Some(attempt_class)
        && recorded.last_error.as_deref() == Some(detail)
}

/// 判断某个尝试分类是否值得重试。
fn transcript_attempt_is_retryable(attempt_class: &TranscriptProjectionAttemptClass) -> bool {
    matches!(
        attempt_class,
        TranscriptProjectionAttemptClass::OutcomeUnknown
            | TranscriptProjectionAttemptClass::TransientNoCommit
            | TranscriptProjectionAttemptClass::RevisionConflict
            | TranscriptProjectionAttemptClass::DeadlineExceeded
    )
}

/**
 * 将公开的 `TranscriptProjectionErrorClass` 映射为内部持久化用的
 * `TranscriptProjectionAttemptClass`。
 */
fn transcript_error_attempt_class(
    error_class: crate::memory::TranscriptProjectionErrorClass,
) -> TranscriptProjectionAttemptClass {
    match error_class {
        crate::memory::TranscriptProjectionErrorClass::OutcomeUnknown => {
            TranscriptProjectionAttemptClass::OutcomeUnknown
        }
        crate::memory::TranscriptProjectionErrorClass::TransientNoCommit => {
            TranscriptProjectionAttemptClass::TransientNoCommit
        }
        crate::memory::TranscriptProjectionErrorClass::RevisionConflict => {
            TranscriptProjectionAttemptClass::RevisionConflict
        }
        crate::memory::TranscriptProjectionErrorClass::DeadlineExceeded => {
            TranscriptProjectionAttemptClass::DeadlineExceeded
        }
        crate::memory::TranscriptProjectionErrorClass::Unsupported => {
            TranscriptProjectionAttemptClass::Unsupported
        }
        crate::memory::TranscriptProjectionErrorClass::InvalidConfiguration => {
            TranscriptProjectionAttemptClass::InvalidConfiguration
        }
        crate::memory::TranscriptProjectionErrorClass::CorruptState => {
            TranscriptProjectionAttemptClass::CorruptState
        }
        crate::memory::TranscriptProjectionErrorClass::SemanticConflict => {
            TranscriptProjectionAttemptClass::SemanticConflict
        }
    }
}

/**
 * 将内部持久化用的 `TranscriptProjectionAttemptClass` 映射回公开的
 * `TranscriptProjectionErrorClass`。
 */
fn transcript_attempt_error_class(
    attempt_class: &TranscriptProjectionAttemptClass,
) -> crate::memory::TranscriptProjectionErrorClass {
    match attempt_class {
        TranscriptProjectionAttemptClass::OutcomeUnknown => {
            crate::memory::TranscriptProjectionErrorClass::OutcomeUnknown
        }
        TranscriptProjectionAttemptClass::TransientNoCommit => {
            crate::memory::TranscriptProjectionErrorClass::TransientNoCommit
        }
        TranscriptProjectionAttemptClass::RevisionConflict => {
            crate::memory::TranscriptProjectionErrorClass::RevisionConflict
        }
        TranscriptProjectionAttemptClass::DeadlineExceeded => {
            crate::memory::TranscriptProjectionErrorClass::DeadlineExceeded
        }
        TranscriptProjectionAttemptClass::Unsupported => {
            crate::memory::TranscriptProjectionErrorClass::Unsupported
        }
        TranscriptProjectionAttemptClass::InvalidConfiguration => {
            crate::memory::TranscriptProjectionErrorClass::InvalidConfiguration
        }
        TranscriptProjectionAttemptClass::CorruptState => {
            crate::memory::TranscriptProjectionErrorClass::CorruptState
        }
        TranscriptProjectionAttemptClass::SemanticConflict => {
            crate::memory::TranscriptProjectionErrorClass::SemanticConflict
        }
    }
}

/// 依据具体的错误类型，将一次持久化失败归类为结算状态与错误分类的组合。
pub(crate) fn classify_transcript_persistence_error(
    error: &crate::error::ReactError,
) -> (
    crate::memory::TranscriptProjectionSettlementStatus,
    crate::memory::TranscriptProjectionErrorClass,
) {
    match error {
        crate::error::ReactError::Config(_) => (
            crate::memory::TranscriptProjectionSettlementStatus::Blocked,
            crate::memory::TranscriptProjectionErrorClass::InvalidConfiguration,
        ),
        crate::error::ReactError::Memory(error) => match error.as_ref() {
            echo_core::error::MemoryError::Unsupported(_) => (
                crate::memory::TranscriptProjectionSettlementStatus::Blocked,
                crate::memory::TranscriptProjectionErrorClass::Unsupported,
            ),
            echo_core::error::MemoryError::SerializationError(_)
            | echo_core::error::MemoryError::NotFound(_)
            | echo_core::error::MemoryError::ManagedConversationRequiresProjection(_)
            | echo_core::error::MemoryError::ProjectionEpochExhausted(_) => (
                crate::memory::TranscriptProjectionSettlementStatus::Blocked,
                crate::memory::TranscriptProjectionErrorClass::CorruptState,
            ),
            echo_core::error::MemoryError::TransientNoCommit(_) => (
                crate::memory::TranscriptProjectionSettlementStatus::Deferred,
                crate::memory::TranscriptProjectionErrorClass::TransientNoCommit,
            ),
            echo_core::error::MemoryError::DeadlineExceeded(_) => (
                crate::memory::TranscriptProjectionSettlementStatus::Deferred,
                crate::memory::TranscriptProjectionErrorClass::DeadlineExceeded,
            ),
            _ => (
                crate::memory::TranscriptProjectionSettlementStatus::Deferred,
                crate::memory::TranscriptProjectionErrorClass::OutcomeUnknown,
            ),
        },
        crate::error::ReactError::RuntimeState(error) => match error.as_ref() {
            echo_core::error::RuntimeStateError::Unsupported(_) => (
                crate::memory::TranscriptProjectionSettlementStatus::Blocked,
                crate::memory::TranscriptProjectionErrorClass::Unsupported,
            ),
            echo_core::error::RuntimeStateError::SerializationError(_)
            | echo_core::error::RuntimeStateError::NotFound(_)
            | echo_core::error::RuntimeStateError::ManagedStateRequiresCas(_)
            | echo_core::error::RuntimeStateError::RevisionExhausted(_) => (
                crate::memory::TranscriptProjectionSettlementStatus::Blocked,
                crate::memory::TranscriptProjectionErrorClass::CorruptState,
            ),
            echo_core::error::RuntimeStateError::DeadlineExceeded(_)
            | echo_core::error::RuntimeStateError::TranscriptProjectionDeferred { .. } => (
                crate::memory::TranscriptProjectionSettlementStatus::Deferred,
                crate::memory::TranscriptProjectionErrorClass::DeadlineExceeded,
            ),
            echo_core::error::RuntimeStateError::TransientNoCommit(_) => (
                crate::memory::TranscriptProjectionSettlementStatus::Deferred,
                crate::memory::TranscriptProjectionErrorClass::TransientNoCommit,
            ),
            _ => (
                crate::memory::TranscriptProjectionSettlementStatus::Deferred,
                crate::memory::TranscriptProjectionErrorClass::OutcomeUnknown,
            ),
        },
        _ => (
            crate::memory::TranscriptProjectionSettlementStatus::Deferred,
            crate::memory::TranscriptProjectionErrorClass::OutcomeUnknown,
        ),
    }
}

/// 基于一个待处理 transcript 投影的身份信息，构造对应的结算结果。
fn pending_settlement(
    pending: &PendingTranscriptProjection,
    status: crate::memory::TranscriptProjectionSettlementStatus,
    error_class: crate::memory::TranscriptProjectionErrorClass,
    detail: String,
) -> crate::memory::TranscriptProjectionSettlement {
    transcript_settlement(
        status,
        Some(pending.batch.operation_id.clone()),
        Some(pending.batch.conversation_id.clone()),
        Some(pending.batch.generation_id.clone()),
        pending.attempt,
        Some(error_class),
        Some(detail),
    )
}

/// 将一个结算结果包装为不带新版本号 / 新游标的 `TranscriptProjectionSettlementOutcome`。
fn transcript_settlement_outcome(
    settlement: crate::memory::TranscriptProjectionSettlement,
) -> TranscriptProjectionSettlementOutcome {
    TranscriptProjectionSettlementOutcome {
        settlement,
        committed_version: None,
        settled_cursor: None,
    }
}

/// 便捷构造函数：按字段拼装出一个 `TranscriptProjectionSettlement`。
fn transcript_settlement(
    status: crate::memory::TranscriptProjectionSettlementStatus,
    operation_id: Option<String>,
    conversation_id: Option<String>,
    generation_id: Option<String>,
    attempt: u32,
    error_class: Option<crate::memory::TranscriptProjectionErrorClass>,
    detail: Option<String>,
) -> crate::memory::TranscriptProjectionSettlement {
    crate::memory::TranscriptProjectionSettlement {
        status,
        operation_id,
        conversation_id,
        generation_id,
        attempt,
        error_class,
        detail,
    }
}

/// 要求给定的结算结果必须处于"已结算"状态，否则将其转换为对应的错误。
fn require_settled_transcript_projection(
    settlement: crate::memory::TranscriptProjectionSettlement,
) -> crate::error::Result<()> {
    match settlement.status {
        crate::memory::TranscriptProjectionSettlementStatus::Settled => Ok(()),
        crate::memory::TranscriptProjectionSettlementStatus::Deferred => Err(
            echo_core::error::RuntimeStateError::TranscriptProjectionDeferred {
                operation_id: settlement.operation_id,
                reason: settlement
                    .detail
                    .unwrap_or_else(|| "transcript settlement remains deferred".to_string()),
            }
            .into(),
        ),
        crate::memory::TranscriptProjectionSettlementStatus::Blocked
        | crate::memory::TranscriptProjectionSettlementStatus::Conflict => Err(
            echo_core::error::RuntimeStateError::TranscriptProjectionBlocked {
                status: format!("{:?}", settlement.status),
                reason: settlement
                    .detail
                    .unwrap_or_else(|| "transcript settlement is blocked".to_string()),
            }
            .into(),
        ),
    }
}

/**
 * 删除一个已退休的运行时实例，但不删除其对应的稳定产品 transcript。
 *
 * 托管状态会先结算其持久化的 transcript 效果，确认待处理标记已清除，然后
 * 用墓碑退休该代次。因为这里没有保留稳定的 transcript 删除身份，所以它绝不
 * 会删除一个恰好共享同一实例 ID 的独立 transcript。旧版非托管配对则保留其
 * 历史上的"实例-transcript 一并清理"行为。
 */
pub async fn clear_persisted_runtime_incarnation(
    conversation_store: &dyn crate::memory::ConversationStore,
    runtime_state_store: &dyn RuntimeStateStore,
    scope_id: &str,
    runtime_state_id: &str,
) -> crate::error::Result<RuntimeStateClearReceipt> {
    if managed_persistence_pair(conversation_store, runtime_state_store)? {
        return clear_persisted_runtime_incarnation_with_context(
            conversation_store,
            runtime_state_store,
            scope_id,
            runtime_state_id,
            crate::memory::PersistenceCallContext::with_timeout(PERSISTED_STATE_OPERATION_TIMEOUT)?,
        )
        .await;
    }
    clear_legacy_runtime_incarnation(
        conversation_store,
        runtime_state_store,
        scope_id,
        runtime_state_id,
    )
    .await
}

/**
 * 带调用上下文的托管精确清理。旧版 Store API 不携带调用上下文，因此这里
 * 直接失败（fail closed），而不是冒险执行一次可能过期的删除。
 */
pub async fn clear_persisted_runtime_incarnation_with_context(
    conversation_store: &dyn crate::memory::ConversationStore,
    runtime_state_store: &dyn RuntimeStateStore,
    scope_id: &str,
    runtime_state_id: &str,
    context: crate::memory::PersistenceCallContext,
) -> crate::error::Result<RuntimeStateClearReceipt> {
    let budget = PersistenceOperationDeadline::from_context(context)?;
    if !managed_persistence_pair(conversation_store, runtime_state_store)? {
        return Err(echo_core::error::RuntimeStateError::Unsupported(
            "context-aware exact clear is unsupported for legacy stores".to_string(),
        )
        .into());
    }
    await_persistence(
        budget,
        "managed runtime incarnation clear",
        clear_managed_runtime_incarnation(
            conversation_store,
            runtime_state_store,
            scope_id,
            runtime_state_id,
            budget,
        ),
    )
    .await
}

/**
 * 判断会话存储与运行时存储是否共同构成一对"托管持久化"组合：两者都必须
 * 同时支持原子投影 / 版本化能力，且都强制执行绝对截止时间。
 */
fn managed_persistence_pair(
    conversation_store: &dyn crate::memory::ConversationStore,
    runtime_state_store: &dyn RuntimeStateStore,
) -> crate::error::Result<bool> {
    let conversation_managed = conversation_store.projection_capability()
        == crate::memory::ConversationProjectionCapability::AtomicV1;
    let runtime_managed =
        runtime_state_store.runtime_state_capability() == RuntimeStateCapability::RevisionedV1;
    match (conversation_managed, runtime_managed) {
        (false, false) => Ok(false),
        (true, true)
            if conversation_store.persistence_call_capability()
                == crate::memory::PersistenceCallCapability::AbsoluteDeadlineV1
                && runtime_state_store.persistence_call_capability()
                    == crate::memory::PersistenceCallCapability::AbsoluteDeadlineV1 =>
        {
            Ok(true)
        }
        (true, true) => Err(echo_core::error::RuntimeStateError::Unsupported(
            "managed persistence stores must enforce absolute deadline call context".to_string(),
        )
        .into()),
        _ => Err(echo_core::error::RuntimeStateError::Unsupported(
            "conversation and runtime stores must advertise compatible managed persistence capabilities"
                .to_string(),
        )
        .into()),
    }
}

/**
 * 旧版（非托管）存储对下的精确清理实现：直接删除会话历史与运行时状态，
 * 不涉及托管的结算 / 退休流程。
 */
async fn clear_legacy_runtime_incarnation(
    conversation_store: &dyn crate::memory::ConversationStore,
    runtime_state_store: &dyn RuntimeStateStore,
    scope_id: &str,
    runtime_state_id: &str,
) -> crate::error::Result<RuntimeStateClearReceipt> {
    if runtime_state_id != scope_id {
        let indexed = runtime_state_store.runtime_state_ids(scope_id).await?;
        if !indexed
            .iter()
            .any(|indexed_id| indexed_id == runtime_state_id)
        {
            return Err(echo_core::error::RuntimeStateError::NotFound(format!(
                "runtime state {runtime_state_id} is not owned by scope {scope_id}"
            ))
            .into());
        }
        conversation_store
            .delete_conversation(runtime_state_id)
            .await?;
    }
    runtime_state_store
        .clear_runtime_state(scope_id, runtime_state_id)
        .await
}

/**
 * 托管存储对下的精确清理实现：先结算待处理的 transcript 投影，确认清除
 * 干净后再将该代次退休为墓碑。
 */
async fn clear_managed_runtime_incarnation(
    conversation_store: &dyn crate::memory::ConversationStore,
    runtime_state_store: &dyn RuntimeStateStore,
    scope_id: &str,
    runtime_state_id: &str,
    budget: PersistenceOperationDeadline,
) -> crate::error::Result<RuntimeStateClearReceipt> {
    let settlement = settle_pending_transcript_projection_with_budget(
        conversation_store,
        runtime_state_store,
        scope_id,
        runtime_state_id,
        budget,
    )
    .await?;
    require_settled_transcript_projection(settlement)?;

    let state = await_persistence(
        budget,
        "runtime generation load after transcript settlement",
        runtime_state_store.load_runtime_state_with_context(
            budget.context,
            scope_id,
            runtime_state_id,
        ),
    )
    .await?
    .ok_or_else(|| {
        echo_core::error::RuntimeStateError::NotFound(format!(
            "runtime state {runtime_state_id} is not owned by scope {scope_id}"
        ))
    })?;
    if let Some(pending) = pending_transcript_projection(&state)? {
        return Err(
            echo_core::error::RuntimeStateError::TranscriptProjectionDeferred {
                operation_id: Some(pending.batch.operation_id),
                reason:
                    "canonical transcript settlement returned without clearing durable pending state"
                        .to_string(),
            }
            .into(),
        );
    }

    let checkpoint_removed = if matches!(state.version, RuntimeStateVersion::Retired { .. }) {
        false
    } else {
        let request = RuntimeGenerationRetireRequest::prepare(
            scope_id,
            runtime_state_id,
            state.scope.revision,
            expected_runtime_state_version(&state.version),
        )?;
        let receipt = await_persistence(
            budget,
            "runtime generation retirement",
            runtime_state_store.retire_runtime_generation_with_context(budget.context, request),
        )
        .await?;
        match receipt.status {
            RuntimeGenerationRetireStatus::Retired
            | RuntimeGenerationRetireStatus::AlreadyRetired => true,
            RuntimeGenerationRetireStatus::PendingProjection => {
                let operation_id = await_persistence(
                    budget,
                    "runtime generation reload after pending retirement",
                    runtime_state_store.load_runtime_state_with_context(
                        budget.context,
                        scope_id,
                        runtime_state_id,
                    ),
                )
                .await?
                .as_ref()
                .map(pending_transcript_projection)
                .transpose()?
                .flatten()
                .map(|pending| pending.batch.operation_id);
                return Err(
                    echo_core::error::RuntimeStateError::TranscriptProjectionDeferred {
                        operation_id,
                        reason: "runtime generation acquired a new unsettled transcript projection"
                            .to_string(),
                    }
                    .into(),
                );
            }
            RuntimeGenerationRetireStatus::Conflict
            | RuntimeGenerationRetireStatus::ScopeFenced => {
                return Err(
                    echo_core::error::RuntimeStateError::ManagedStateRequiresCas(format!(
                        "runtime generation retirement did not settle: {:?}",
                        receipt.status
                    ))
                    .into(),
                );
            }
        }
    };

    Ok(RuntimeStateClearReceipt {
        scope_id: scope_id.to_string(),
        runtime_state_id: runtime_state_id.to_string(),
        checkpoint_removed,
    })
}

/// 从一个已加载的运行时状态快照中提取出其待处理的 transcript 投影（如果有）。
fn pending_transcript_projection(
    state: &ManagedRuntimeStateSnapshot,
) -> crate::error::Result<Option<PendingTranscriptProjection>> {
    state
        .checkpoint
        .as_ref()
        .map(AgentCheckpoint::restore_managed_runtime_payload)
        .transpose()
        .map(|payload| payload.and_then(|payload| payload.pending_transcript_projection))
}

/**
 * 将一个已持久化的 `RuntimeStateVersion` 转换为对应的
 * `RuntimeStateExpectedVersion`，用于后续 CAS 请求。
 */
fn expected_runtime_state_version(version: &RuntimeStateVersion) -> RuntimeStateExpectedVersion {
    match version {
        RuntimeStateVersion::Absent => RuntimeStateExpectedVersion::Absent,
        RuntimeStateVersion::Unmanaged { digest } => RuntimeStateExpectedVersion::Unmanaged {
            digest: digest.clone(),
        },
        RuntimeStateVersion::Managed { revision } => RuntimeStateExpectedVersion::Managed {
            revision: *revision,
        },
        RuntimeStateVersion::Retired { .. } => RuntimeStateExpectedVersion::Absent,
    }
}

/**
 * 删除一个稳定的、用户可见的 transcript，以及绑定到同一作用域的每一个
 * 运行时检查点。
 *
 * 这个兼容性入口只对旧版非托管存储生效。托管存储必须使用
 * [`delete_persisted_conversation_managed`]，以便重试时保留原始的、
 * 带 epoch 围栏的删除身份。
 */
pub async fn delete_persisted_conversation(
    conversation_store: &dyn crate::memory::ConversationStore,
    runtime_state_store: &dyn RuntimeStateStore,
    conversation_id: &str,
) -> crate::error::Result<PersistedConversationDeleteReceipt> {
    if managed_persistence_pair(conversation_store, runtime_state_store)? {
        return Err(echo_core::error::RuntimeStateError::ManagedStateRequiresCas(
            "managed conversation deletion requires delete_persisted_conversation_managed with a stable ManagedConversationDelete request"
                .to_string(),
        )
        .into());
    }
    delete_legacy_persisted_conversation(conversation_store, runtime_state_store, conversation_id)
        .await
}

/**
 * 带调用上下文的删除兼容性边界。托管存储必须使用稳定请求 API；旧版存储
 * 会直接失败（fail closed），因为它们的原始删除 API 无法在调用方取消后
 * 对一次阻塞写入做围栏。
 */
pub async fn delete_persisted_conversation_with_context(
    conversation_store: &dyn crate::memory::ConversationStore,
    runtime_state_store: &dyn RuntimeStateStore,
    _conversation_id: &str,
    context: crate::memory::PersistenceCallContext,
) -> crate::error::Result<PersistedConversationDeleteReceipt> {
    let _budget = PersistenceOperationDeadline::from_context(context)?;
    if managed_persistence_pair(conversation_store, runtime_state_store)? {
        return Err(echo_core::error::RuntimeStateError::ManagedStateRequiresCas(
            "managed conversation deletion requires delete_persisted_conversation_managed with a stable ManagedConversationDelete request"
                .to_string(),
        )
        .into());
    }
    Err(echo_core::error::RuntimeStateError::Unsupported(
        "context-aware conversation delete is unsupported for legacy stores".to_string(),
    )
    .into())
}

/// 旧版 Store 的删除路径：先删除附属运行时状态对应的会话，再清空运行时作用域并删除主会话。
async fn delete_legacy_persisted_conversation(
    conversation_store: &dyn crate::memory::ConversationStore,
    runtime_state_store: &dyn RuntimeStateStore,
    conversation_id: &str,
) -> crate::error::Result<PersistedConversationDeleteReceipt> {
    let runtime_state_ids = runtime_state_store
        .runtime_state_ids(conversation_id)
        .await?;
    for runtime_state_id in &runtime_state_ids {
        if runtime_state_id != conversation_id {
            conversation_store
                .delete_conversation(runtime_state_id)
                .await?;
        }
    }
    let runtime = runtime_state_store
        .clear_runtime_state_scope(conversation_id)
        .await?;
    conversation_store
        .delete_conversation(conversation_id)
        .await?;
    Ok(PersistedConversationDeleteReceipt {
        conversation_id: conversation_id.to_string(),
        runtime_state_ids: runtime.runtime_state_ids,
        retirement: None,
    })
}

/**
 * 使用调用方持有的、可重试稳定的请求身份，删除一个托管会话。
 *
 * 这是唯一的托管入口。重试时必须传入同一个
 * [`crate::memory::ManagedConversationDelete`]，无论是在确认丢失之后，
 * 还是在同一个会话 ID 被重新创建之后。运行时退休权威会在任何 transcript
 * 变更之前先被查询，因此一份保留下来的完成回执会优先于当前的会话 epoch。
 */
pub async fn delete_persisted_conversation_managed(
    conversation_store: &dyn crate::memory::ConversationStore,
    runtime_state_store: &dyn RuntimeStateStore,
    delete: crate::memory::ManagedConversationDelete,
) -> crate::error::Result<PersistedConversationDeleteReceipt> {
    delete_persisted_conversation_managed_with_context(
        conversation_store,
        runtime_state_store,
        delete,
        crate::memory::PersistenceCallContext::with_timeout(PERSISTED_STATE_OPERATION_TIMEOUT)?,
    )
    .await
}

/**
 * 带调用上下文的托管删除。重试时提供同一个持久化请求，只刷新非权威性的
 * 绝对调用截止时间。
 */
pub async fn delete_persisted_conversation_managed_with_context(
    conversation_store: &dyn crate::memory::ConversationStore,
    runtime_state_store: &dyn RuntimeStateStore,
    delete: crate::memory::ManagedConversationDelete,
    context: crate::memory::PersistenceCallContext,
) -> crate::error::Result<PersistedConversationDeleteReceipt> {
    let budget = PersistenceOperationDeadline::from_context(context)?;
    delete.validate()?;
    if !managed_persistence_pair(conversation_store, runtime_state_store)? {
        return Err(echo_core::error::RuntimeStateError::Unsupported(
            "managed conversation deletion requires atomic conversation and revisioned runtime stores"
                .to_string(),
        )
        .into());
    }
    await_persistence(
        budget,
        "managed persisted conversation delete",
        delete_managed_persisted_conversation(
            conversation_store,
            runtime_state_store,
            delete,
            budget,
        ),
    )
    .await
}

/// 托管删除的核心流程：在给定截止时间预算内推进作用域退休，并返回删除回执。
async fn delete_managed_persisted_conversation(
    conversation_store: &dyn crate::memory::ConversationStore,
    runtime_state_store: &dyn RuntimeStateStore,
    delete: crate::memory::ManagedConversationDelete,
    budget: PersistenceOperationDeadline,
) -> crate::error::Result<PersistedConversationDeleteReceipt> {
    let conversation_id = delete.conversation_id.clone();
    let scope_revision = await_persistence(
        budget,
        "runtime scope load before managed delete",
        runtime_state_store.load_scope_authority_with_context(budget.context, &conversation_id),
    )
    .await?
    .map(|scope| scope.revision)
    .unwrap_or(0);
    let request = ScopeRetirementRequest::prepare(&conversation_id, scope_revision, &delete)?;
    let mut retirement = await_persistence(
        budget,
        "runtime scope retirement begin",
        runtime_state_store.begin_scope_retirement_with_context(budget.context, request),
    )
    .await?;
    retirement.validate()?;
    if matches!(
        retirement.status,
        ScopeRetirementStatus::Completed | ScopeRetirementStatus::AlreadyCompleted
    ) || !scope_retirement_can_advance(retirement.status)
    {
        return Ok(persisted_delete_receipt(conversation_id, retirement));
    }

    if retirement.manifest.conversation_delete_receipt.is_none() {
        let conversation_receipt = await_persistence(
            budget,
            "managed conversation transcript delete",
            conversation_store
                .delete_managed_conversation_with_context(budget.context, delete.clone()),
        )
        .await?;
        retirement = await_persistence(
            budget,
            "managed conversation delete receipt persistence",
            runtime_state_store.continue_scope_retirement_with_context(
                budget.context,
                &conversation_id,
                &delete.operation_id,
                retirement.scope.revision,
                ScopeRetirementAdvance::ConversationDeleted {
                    receipt: conversation_receipt,
                },
            ),
        )
        .await?;
        retirement.validate()?;
        if !scope_retirement_can_advance(retirement.status) {
            return Ok(persisted_delete_receipt(conversation_id, retirement));
        }
    }

    let pending_items = retirement
        .manifest
        .items
        .iter()
        .filter(|item| item.status == ScopeRetirementItemStatus::Pending)
        .map(|item| {
            (
                item.runtime_state_id.clone(),
                item.pending_operation_id.clone(),
            )
        })
        .collect::<Vec<_>>();
    for (runtime_state_id, pending_operation_id) in pending_items {
        retirement = await_persistence(
            budget,
            "managed runtime generation drop",
            runtime_state_store.continue_scope_retirement_with_context(
                budget.context,
                &conversation_id,
                &delete.operation_id,
                retirement.scope.revision,
                ScopeRetirementAdvance::GenerationDropped {
                    runtime_state_id,
                    pending_operation_id,
                },
            ),
        )
        .await?;
        retirement.validate()?;
        if !scope_retirement_can_advance(retirement.status) {
            return Ok(persisted_delete_receipt(conversation_id, retirement));
        }
    }
    retirement = await_persistence(
        budget,
        "managed runtime scope retirement completion",
        runtime_state_store.continue_scope_retirement_with_context(
            budget.context,
            &conversation_id,
            &delete.operation_id,
            retirement.scope.revision,
            ScopeRetirementAdvance::Complete,
        ),
    )
    .await?;
    retirement.validate()?;
    Ok(persisted_delete_receipt(conversation_id, retirement))
}

/// 判断作用域退休是否仍处于可继续推进的状态（已开始或进行中）。
fn scope_retirement_can_advance(status: ScopeRetirementStatus) -> bool {
    matches!(
        status,
        ScopeRetirementStatus::Begun | ScopeRetirementStatus::InProgress
    )
}

// ── 重新导出各实现 ─────────────────────────────────────────

/// 基于文件的运行时状态存储实现（默认实现，不依赖 SQLite）。
pub mod file;

/// 基于 SQLite 的运行时状态存储实现（`sqlite` feature）。
#[cfg(feature = "sqlite")]
pub mod sqlite;

pub use file::FileRuntimeStateStore;
#[cfg(feature = "sqlite")]
pub use sqlite::SqliteRuntimeStateStore;

#[cfg(test)]
mod checkpoint_tests {
    use super::*;
    use crate::llm::types::{FunctionCall, Message, ToolCall};

    /// 构造一个指定 ID 与名称、参数为空对象的测试用工具调用。
    fn tool_call(id: &str, name: &str) -> ToolCall {
        ToolCall {
            id: id.to_string(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: name.to_string(),
                arguments: "{}".to_string(),
            },
        }
    }

    /// 用给定消息列表构造一个测试用检查点。
    fn checkpoint(messages: Vec<Message>) -> crate::error::Result<AgentCheckpoint> {
        let mut checkpoint = AgentCheckpoint::new("conversation-1");
        checkpoint.messages_json = serde_json::to_string(&messages)
            .map_err(|error| crate::error::ReactError::Other(error.to_string()))?;
        Ok(checkpoint)
    }

    /// 构造托管会话投影请求，可指定期望的墓碑 epoch。
    fn managed_projection_request(
        conversation_id: &str,
        expected_tombstone_epoch: Option<u64>,
    ) -> crate::memory::EnsureConversationProjectionRequest {
        crate::memory::EnsureConversationProjectionRequest {
            conversation: crate::memory::NewConversation {
                conversation_id: conversation_id.to_string(),
                user_id: "default".to_string(),
                agent_type: None,
                title: None,
            },
            expected_tombstone_epoch,
        }
    }

    /// 返回一个截止时间早已过期的持久化调用上下文。
    fn expired_persistence_context() -> crate::memory::PersistenceCallContext {
        crate::memory::PersistenceCallContext {
            absolute_deadline_unix_ms: 1,
        }
    }

    /// 构造一个携带待处理 transcript 投影的检查点及对应的待处理投影。
    fn pending_projection_checkpoint(
        conversation_id: &str,
        runtime_state_id: &str,
        epoch: u64,
    ) -> crate::error::Result<(AgentCheckpoint, PendingTranscriptProjection)> {
        let batch = crate::memory::TranscriptProjectionBatch::prepare(
            conversation_id,
            epoch,
            runtime_state_id,
            0,
            vec![crate::memory::StoredMessage {
                id: None,
                conversation_id: conversation_id.to_string(),
                role: "user".to_string(),
                content: Some("pending transcript".to_string()),
                attachments_json: None,
                tool_calls_json: None,
                tool_result_json: None,
                created_at: "2026-09-17T00:00:00Z".to_string(),
            }],
        )?;
        let projected_item = batch.items.first().ok_or_else(|| {
            crate::error::ReactError::Other("prepared batch is empty".to_string())
        })?;
        let projected = TranscriptProjectionMessage {
            ordinal: projected_item.ordinal,
            digest: crate::memory::transcript_projection_message_digest(&projected_item.message)?,
        };
        let pending = PendingTranscriptProjection {
            batch,
            cursor_before: TranscriptProjectionCheckpoint {
                generation_id: runtime_state_id.to_string(),
                next_ordinal: 0,
                projected: Vec::new(),
            },
            cursor_after: TranscriptProjectionCheckpoint {
                generation_id: runtime_state_id.to_string(),
                next_ordinal: 1,
                projected: vec![projected],
            },
            base_runtime_revision: 1,
            prepared_at: Utc::now(),
            attempt: 1,
            last_attempt_class: None,
            last_error: None,
        };
        let mut checkpoint = AgentCheckpoint::new(runtime_state_id);
        checkpoint.messages_json = AgentCheckpoint::serialize_managed_payload(
            vec![Message::user("pending transcript".to_string())],
            Some(pending.cursor_before.clone()),
            Some(pending.clone()),
        )?;
        Ok((checkpoint, pending))
    }

    /// 测试用运行时状态存储：拒绝写入带失败分类的待处理投影的 CAS 请求，其余委托给内部 Store。
    #[derive(Clone)]
    struct RejectFailureCasRuntimeStore {
        /// 被包装的真实文件存储。
        inner: FileRuntimeStateStore,
    }

    impl RejectFailureCasRuntimeStore {
        /// 判断该 CAS 请求是否携带已记录失败分类的待处理投影，是则应被拒绝。
        fn rejects(request: &RuntimeCheckpointCasRequest) -> bool {
            request
                .checkpoint
                .restore_managed_runtime_payload()
                .map(|payload| {
                    payload
                        .pending_transcript_projection
                        .is_some_and(|pending| pending.last_attempt_class.is_some())
                })
                .unwrap_or(false)
        }

        /// 构造注入的“未提交的瞬时失败”错误。
        fn rejected_receipt() -> crate::error::ReactError {
            echo_core::error::RuntimeStateError::TransientNoCommit(
                "injected failure-class CAS rejection".to_string(),
            )
            .into()
        }
    }

    impl RuntimeStateStore for RejectFailureCasRuntimeStore {
        /// 直接委托给内部 Store。
        fn runtime_state_capability(&self) -> RuntimeStateCapability {
            RuntimeStateCapability::RevisionedV1
        }

        /// 直接委托给内部 Store。
        fn persistence_call_capability(&self) -> crate::memory::PersistenceCallCapability {
            crate::memory::PersistenceCallCapability::AbsoluteDeadlineV1
        }

        /// 直接委托给内部 Store。
        fn load_runtime_state<'a>(
            &'a self,
            scope_id: &'a str,
            runtime_state_id: &'a str,
        ) -> futures::future::BoxFuture<'a, crate::error::Result<Option<ManagedRuntimeStateSnapshot>>>
        {
            self.inner.load_runtime_state(scope_id, runtime_state_id)
        }

        /// 直接委托给内部 Store。
        fn load_runtime_state_with_context<'a>(
            &'a self,
            context: crate::memory::PersistenceCallContext,
            scope_id: &'a str,
            runtime_state_id: &'a str,
        ) -> futures::future::BoxFuture<'a, crate::error::Result<Option<ManagedRuntimeStateSnapshot>>>
        {
            self.inner
                .load_runtime_state_with_context(context, scope_id, runtime_state_id)
        }

        /// 直接委托给内部 Store。
        fn load_scope_authority<'a>(
            &'a self,
            scope_id: &'a str,
        ) -> futures::future::BoxFuture<'a, crate::error::Result<Option<RuntimeScopeAuthority>>>
        {
            self.inner.load_scope_authority(scope_id)
        }

        /// 直接委托给内部 Store。
        fn load_scope_authority_with_context<'a>(
            &'a self,
            context: crate::memory::PersistenceCallContext,
            scope_id: &'a str,
        ) -> futures::future::BoxFuture<'a, crate::error::Result<Option<RuntimeScopeAuthority>>>
        {
            self.inner
                .load_scope_authority_with_context(context, scope_id)
        }

        /// 若请求应被拒绝则返回注入错误，否则委托给内部 Store。
        fn compare_and_save_checkpoint<'a>(
            &'a self,
            request: RuntimeCheckpointCasRequest,
        ) -> futures::future::BoxFuture<'a, crate::error::Result<RuntimeCheckpointCasReceipt>>
        {
            if Self::rejects(&request) {
                return Box::pin(async { Err(Self::rejected_receipt()) });
            }
            self.inner.compare_and_save_checkpoint(request)
        }

        /// 若请求应被拒绝则返回注入错误，否则委托给内部 Store。
        fn compare_and_save_checkpoint_with_context<'a>(
            &'a self,
            context: crate::memory::PersistenceCallContext,
            request: RuntimeCheckpointCasRequest,
        ) -> futures::future::BoxFuture<'a, crate::error::Result<RuntimeCheckpointCasReceipt>>
        {
            if Self::rejects(&request) {
                return Box::pin(async { Err(Self::rejected_receipt()) });
            }
            self.inner
                .compare_and_save_checkpoint_with_context(context, request)
        }

        /// 直接委托给内部 Store。
        fn acknowledge_transcript_projection<'a>(
            &'a self,
            request: RuntimeTranscriptAckRequest,
        ) -> futures::future::BoxFuture<'a, crate::error::Result<RuntimeCheckpointCasReceipt>>
        {
            self.inner.acknowledge_transcript_projection(request)
        }

        /// 直接委托给内部 Store。
        fn acknowledge_transcript_projection_with_context<'a>(
            &'a self,
            context: crate::memory::PersistenceCallContext,
            request: RuntimeTranscriptAckRequest,
        ) -> futures::future::BoxFuture<'a, crate::error::Result<RuntimeCheckpointCasReceipt>>
        {
            self.inner
                .acknowledge_transcript_projection_with_context(context, request)
        }

        /// 直接委托给内部 Store。
        fn retire_runtime_generation<'a>(
            &'a self,
            request: RuntimeGenerationRetireRequest,
        ) -> futures::future::BoxFuture<'a, crate::error::Result<RuntimeGenerationRetireReceipt>>
        {
            self.inner.retire_runtime_generation(request)
        }

        /// 直接委托给内部 Store。
        fn get_checkpoint<'a>(
            &'a self,
            conversation_id: &'a str,
        ) -> futures::future::BoxFuture<'a, crate::error::Result<Option<AgentCheckpoint>>> {
            self.inner.get_checkpoint(conversation_id)
        }

        /// 直接委托给内部 Store。
        fn save_checkpoint<'a>(
            &'a self,
            checkpoint: &'a AgentCheckpoint,
        ) -> futures::future::BoxFuture<'a, crate::error::Result<()>> {
            self.inner.save_checkpoint(checkpoint)
        }

        /// 直接委托给内部 Store。
        fn save_checkpoint_for_scope<'a>(
            &'a self,
            scope_id: &'a str,
            checkpoint: &'a AgentCheckpoint,
        ) -> futures::future::BoxFuture<'a, crate::error::Result<()>> {
            self.inner.save_checkpoint_for_scope(scope_id, checkpoint)
        }

        /// 直接委托给内部 Store。
        fn runtime_state_ids<'a>(
            &'a self,
            scope_id: &'a str,
        ) -> futures::future::BoxFuture<'a, crate::error::Result<Vec<String>>> {
            self.inner.runtime_state_ids(scope_id)
        }

        /// 直接委托给内部 Store。
        fn clear_runtime_state<'a>(
            &'a self,
            scope_id: &'a str,
            runtime_state_id: &'a str,
        ) -> futures::future::BoxFuture<'a, crate::error::Result<RuntimeStateClearReceipt>>
        {
            self.inner.clear_runtime_state(scope_id, runtime_state_id)
        }

        /// 直接委托给内部 Store。
        fn clear_runtime_state_scope<'a>(
            &'a self,
            scope_id: &'a str,
        ) -> futures::future::BoxFuture<'a, crate::error::Result<RuntimeStateScopeClearReceipt>>
        {
            self.inner.clear_runtime_state_scope(scope_id)
        }

        /// 直接委托给内部 Store。
        fn clear_conversation<'a>(
            &'a self,
            conversation_id: &'a str,
        ) -> futures::future::BoxFuture<'a, crate::error::Result<()>> {
            self.inner.clear_conversation(conversation_id)
        }
    }

    /// 旧版会话 Store 探针：任何调用都会计数并返回错误，用于断言未被调用。
    #[derive(Default)]
    struct LegacyConversationProbe {
        /// 被调用的次数。
        calls: std::sync::atomic::AtomicUsize,
    }

    impl LegacyConversationProbe {
        /// 记录一次调用并返回“不应被调用”的错误。
        fn unexpected<T: Send + 'static>(
            &self,
        ) -> futures::future::BoxFuture<'static, crate::error::Result<T>> {
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Box::pin(async {
                Err(crate::error::ReactError::Other(
                    "legacy conversation probe was called".to_string(),
                ))
            })
        }
    }

    impl crate::memory::ConversationStore for LegacyConversationProbe {
        /// 探针实现：计数并返回错误。
        fn create_conversation<'a>(
            &'a self,
            _conv: crate::memory::NewConversation,
        ) -> futures::future::BoxFuture<'a, crate::error::Result<crate::memory::Conversation>>
        {
            self.unexpected()
        }

        /// 探针实现：计数并返回错误。
        fn get_conversation<'a>(
            &'a self,
            _conversation_id: &'a str,
        ) -> futures::future::BoxFuture<'a, crate::error::Result<Option<crate::memory::Conversation>>>
        {
            self.unexpected()
        }

        /// 探针实现：计数并返回错误。
        fn list_conversations<'a>(
            &'a self,
            _filter: crate::memory::ConversationFilter,
        ) -> futures::future::BoxFuture<
            'a,
            crate::error::Result<Vec<crate::memory::ConversationMeta>>,
        > {
            self.unexpected()
        }

        /// 探针实现：计数并返回错误。
        fn update_conversation<'a>(
            &'a self,
            _conversation_id: &'a str,
            _title: Option<&'a str>,
            _summary: Option<&'a str>,
            _compressed_before_id: Option<i64>,
        ) -> futures::future::BoxFuture<'a, crate::error::Result<()>> {
            self.unexpected()
        }

        /// 探针实现：计数并返回错误。
        fn delete_conversation<'a>(
            &'a self,
            _conversation_id: &'a str,
        ) -> futures::future::BoxFuture<'a, crate::error::Result<()>> {
            self.unexpected()
        }

        /// 探针实现：计数并返回错误。
        fn save_messages<'a>(
            &'a self,
            _conversation_id: &'a str,
            _messages: &'a [crate::memory::StoredMessage],
        ) -> futures::future::BoxFuture<'a, crate::error::Result<()>> {
            self.unexpected()
        }

        /// 探针实现：计数并返回错误。
        fn get_messages<'a>(
            &'a self,
            _conversation_id: &'a str,
        ) -> futures::future::BoxFuture<'a, crate::error::Result<Vec<crate::memory::StoredMessage>>>
        {
            self.unexpected()
        }

        /// 探针实现：计数并返回错误。
        fn count_messages<'a>(
            &'a self,
            _conversation_id: &'a str,
        ) -> futures::future::BoxFuture<'a, crate::error::Result<usize>> {
            self.unexpected()
        }
    }

    /// 旧版运行时状态 Store 探针：任何调用都会计数并返回错误，用于断言未被调用。
    #[derive(Default)]
    struct LegacyRuntimeProbe {
        /// 被调用的次数。
        calls: std::sync::atomic::AtomicUsize,
    }

    impl LegacyRuntimeProbe {
        /// 记录一次调用并返回“不应被调用”的错误。
        fn unexpected<T: Send + 'static>(
            &self,
        ) -> futures::future::BoxFuture<'static, crate::error::Result<T>> {
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Box::pin(async {
                Err(crate::error::ReactError::Other(
                    "legacy runtime probe was called".to_string(),
                ))
            })
        }
    }

    impl RuntimeStateStore for LegacyRuntimeProbe {
        /// 探针实现：计数并返回错误。
        fn get_checkpoint<'a>(
            &'a self,
            _conversation_id: &'a str,
        ) -> futures::future::BoxFuture<'a, crate::error::Result<Option<AgentCheckpoint>>> {
            self.unexpected()
        }

        /// 探针实现：计数并返回错误。
        fn save_checkpoint<'a>(
            &'a self,
            _checkpoint: &'a AgentCheckpoint,
        ) -> futures::future::BoxFuture<'a, crate::error::Result<()>> {
            self.unexpected()
        }

        /// 探针实现：计数并返回错误。
        fn save_checkpoint_for_scope<'a>(
            &'a self,
            _scope_id: &'a str,
            _checkpoint: &'a AgentCheckpoint,
        ) -> futures::future::BoxFuture<'a, crate::error::Result<()>> {
            self.unexpected()
        }

        /// 探针实现：计数并返回错误。
        fn runtime_state_ids<'a>(
            &'a self,
            _scope_id: &'a str,
        ) -> futures::future::BoxFuture<'a, crate::error::Result<Vec<String>>> {
            self.unexpected()
        }

        /// 探针实现：计数并返回错误。
        fn clear_runtime_state<'a>(
            &'a self,
            _scope_id: &'a str,
            _runtime_state_id: &'a str,
        ) -> futures::future::BoxFuture<'a, crate::error::Result<RuntimeStateClearReceipt>>
        {
            self.unexpected()
        }

        /// 探针实现：计数并返回错误。
        fn clear_runtime_state_scope<'a>(
            &'a self,
            _scope_id: &'a str,
        ) -> futures::future::BoxFuture<'a, crate::error::Result<RuntimeStateScopeClearReceipt>>
        {
            self.unexpected()
        }

        /// 探针实现：计数并返回错误。
        fn clear_conversation<'a>(
            &'a self,
            _conversation_id: &'a str,
        ) -> futures::future::BoxFuture<'a, crate::error::Result<()>> {
            self.unexpected()
        }
    }

    /// 验证检查点能恢复工具调用与结果成对的历史。
    #[test]
    fn checkpoint_restores_paired_tool_history() -> crate::error::Result<()> {
        let checkpoint = checkpoint(vec![
            Message::assistant_with_tools(vec![tool_call("call-1", "write_file")]),
            Message::tool_result(
                "call-1".to_string(),
                "write_file".to_string(),
                "written".to_string(),
            ),
        ])?;
        assert_eq!(checkpoint.restore_messages()?.len(), 2);
        assert_eq!(checkpoint.completed_tool_call_ids()?, vec!["call-1"]);
        Ok(())
    }

    /// 验证检查点拒绝未配对或重复的工具结果。
    #[test]
    fn checkpoint_rejects_unpaired_or_duplicate_tool_results() -> crate::error::Result<()> {
        let unpaired = checkpoint(vec![Message::assistant_with_tools(vec![tool_call(
            "call-1",
            "write_file",
        )])])?;
        assert!(unpaired.restore_messages().is_err());

        let duplicate = checkpoint(vec![
            Message::assistant_with_tools(vec![tool_call("call-1", "write_file")]),
            Message::tool_result(
                "call-1".to_string(),
                "write_file".to_string(),
                "written".to_string(),
            ),
            Message::tool_result(
                "call-1".to_string(),
                "write_file".to_string(),
                "written twice".to_string(),
            ),
        ])?;
        assert!(duplicate.restore_messages().is_err());
        Ok(())
    }

    /// 验证 transcript 投影游标能往返序列化，并能拒绝损坏的数据。
    #[test]
    fn transcript_projection_cursor_round_trips_and_rejects_corruption() -> crate::error::Result<()>
    {
        let projection = TranscriptProjectionCheckpoint {
            generation_id: "conversation-1".to_string(),
            next_ordinal: 2,
            projected: vec![
                TranscriptProjectionMessage {
                    ordinal: 0,
                    digest: "a".repeat(64),
                },
                TranscriptProjectionMessage {
                    ordinal: 1,
                    digest: "b".repeat(64),
                },
            ],
        };
        let mut checkpoint = AgentCheckpoint::new("conversation-1");
        checkpoint.messages_json = AgentCheckpoint::serialize_payload(
            vec![Message::user("hello".to_string())],
            Some(projection.clone()),
        )?;
        assert_eq!(
            checkpoint.restore_transcript_projection()?,
            Some(projection)
        );

        let wrong_generation = TranscriptProjectionCheckpoint {
            generation_id: "other".to_string(),
            next_ordinal: 1,
            projected: vec![TranscriptProjectionMessage {
                ordinal: 0,
                digest: "c".repeat(64),
            }],
        };
        checkpoint.messages_json =
            AgentCheckpoint::serialize_payload(Vec::new(), Some(wrong_generation))?;
        assert!(checkpoint.restore_transcript_projection().is_err());

        let duplicate_ordinal = TranscriptProjectionCheckpoint {
            generation_id: "conversation-1".to_string(),
            next_ordinal: 2,
            projected: vec![
                TranscriptProjectionMessage {
                    ordinal: 1,
                    digest: "d".repeat(64),
                },
                TranscriptProjectionMessage {
                    ordinal: 1,
                    digest: "e".repeat(64),
                },
            ],
        };
        checkpoint.messages_json =
            AgentCheckpoint::serialize_payload(Vec::new(), Some(duplicate_ordinal))?;
        assert!(checkpoint.restore_transcript_projection().is_err());
        Ok(())
    }

    /// 验证带 transcript 游标的检查点其摘要开销是有界的。
    #[test]
    fn transcript_cursor_checkpoint_has_bounded_digest_overhead() -> crate::error::Result<()> {
        let projection = TranscriptProjectionCheckpoint {
            generation_id: "conversation-1".to_string(),
            next_ordinal: 100_000,
            projected: (0_u64..100_000)
                .map(|ordinal| TranscriptProjectionMessage {
                    ordinal,
                    digest: "f".repeat(64),
                })
                .collect(),
        };
        let started = std::time::Instant::now();
        let payload = AgentCheckpoint::serialize_payload(
            vec![Message::user("x".repeat(1_000_000))],
            Some(projection),
        )?;
        let serialize_elapsed = started.elapsed();
        let mut checkpoint = AgentCheckpoint::new("conversation-1");
        checkpoint.messages_json = payload;
        let restore_started = std::time::Instant::now();
        let restored = checkpoint.restore_runtime_payload()?;
        let restore_elapsed = restore_started.elapsed();
        if checkpoint.messages_json.len() > 13_000_000
            || serialize_elapsed > std::time::Duration::from_secs(5)
            || restore_elapsed > std::time::Duration::from_secs(5)
            || restored
                .transcript_projection
                .as_ref()
                .map(|projection| projection.projected.len())
                != Some(100_000)
        {
            return Err(crate::error::ReactError::Other(format!(
                "checkpoint cursor exceeded budget: bytes={}, serialize={serialize_elapsed:?}, restore={restore_elapsed:?}",
                checkpoint.messages_json.len(),
            )));
        }
        Ok(())
    }

    /// 验证待处理 transcript 投影能往返序列化，且必须经由托管恢复处理。
    #[test]
    fn pending_transcript_projection_round_trips_and_requires_managed_recovery()
    -> crate::error::Result<()> {
        let batch = crate::memory::TranscriptProjectionBatch::prepare(
            "product-conversation",
            1,
            "conversation-1",
            0,
            vec![crate::memory::StoredMessage {
                id: None,
                conversation_id: "product-conversation".to_string(),
                role: "user".to_string(),
                content: Some("hello".to_string()),
                attachments_json: None,
                tool_calls_json: None,
                tool_result_json: None,
                created_at: "2026-09-16T00:00:00Z".to_string(),
            }],
        )?;
        let projected_item = batch.items.first().ok_or_else(|| {
            crate::error::ReactError::Other("prepared batch is empty".to_string())
        })?;
        let projected = vec![TranscriptProjectionMessage {
            ordinal: 0,
            digest: crate::memory::transcript_projection_message_digest(&projected_item.message)?,
        }];
        let pending = PendingTranscriptProjection {
            batch,
            cursor_before: TranscriptProjectionCheckpoint {
                generation_id: "conversation-1".to_string(),
                next_ordinal: 0,
                projected: Vec::new(),
            },
            cursor_after: TranscriptProjectionCheckpoint {
                generation_id: "conversation-1".to_string(),
                next_ordinal: 1,
                projected,
            },
            base_runtime_revision: 1,
            prepared_at: Utc::now(),
            attempt: 1,
            last_attempt_class: None,
            last_error: None,
        };
        assert_ne!(
            pending.batch.items.first().map(|item| item.digest.as_str()),
            pending
                .cursor_after
                .projected
                .first()
                .map(|message| message.digest.as_str())
        );
        let mut checkpoint = AgentCheckpoint::new("conversation-1");
        checkpoint.messages_json = AgentCheckpoint::serialize_managed_payload(
            vec![Message::user("hello".to_string())],
            Some(pending.cursor_before.clone()),
            Some(pending.clone()),
        )?;

        assert!(checkpoint.restore_runtime_payload().is_err());
        let restored = checkpoint.restore_managed_runtime_payload()?;
        assert_eq!(
            restored.pending_transcript_projection,
            Some(pending.clone())
        );

        for (attempt_class, last_error) in [
            (None, Some("missing class".to_string())),
            (
                Some(TranscriptProjectionAttemptClass::TransientNoCommit),
                None,
            ),
            (
                Some(TranscriptProjectionAttemptClass::DeadlineExceeded),
                Some("   ".to_string()),
            ),
        ] {
            let mut corrupt = pending.clone();
            corrupt.last_attempt_class = attempt_class;
            corrupt.last_error = last_error;
            checkpoint.messages_json = AgentCheckpoint::serialize_managed_payload(
                vec![Message::user("hello".to_string())],
                Some(corrupt.cursor_before.clone()),
                Some(corrupt),
            )?;
            assert!(checkpoint.restore_managed_runtime_payload().is_err());
        }
        Ok(())
    }

    /// 验证运行时退休身份拒绝被重新绑定到其它修订版本。
    #[test]
    fn runtime_retirement_identity_rejects_revision_rebinding() -> crate::error::Result<()> {
        let mut request = RuntimeGenerationRetireRequest::prepare(
            "scope",
            "runtime",
            3,
            RuntimeStateExpectedVersion::Managed { revision: 7 },
        )?;
        request.expected_scope_revision = 4;

        assert!(request.validate().is_err());
        Ok(())
    }

    /// 验证作用域退休复用托管删除的请求身份。
    #[test]
    fn scope_retirement_reuses_managed_delete_identity() -> crate::error::Result<()> {
        let delete = crate::memory::ManagedConversationDelete::prepare("scope", 5)?;
        let request = ScopeRetirementRequest::prepare("scope", 9, &delete)?;

        assert_eq!(request.delete_operation_id, delete.operation_id);
        assert_eq!(request.payload_digest, delete.payload_digest);
        request.validate()?;
        Ok(())
    }

    /// 验证“已完成”的退休要求作用域已处于墓碑状态。
    #[test]
    fn already_completed_retirement_requires_tombstoned_scope() -> crate::error::Result<()> {
        let delete = crate::memory::ManagedConversationDelete::prepare("scope", 5)?;
        let receipt = ScopeRetirementReceipt {
            scope: RuntimeScopeAuthority {
                scope_id: "scope".to_string(),
                conversation_epoch: Some(6),
                revision: 12,
                lifecycle: RuntimeScopeLifecycle::Active,
            },
            manifest: ScopeRetirementManifest {
                delete_operation_id: delete.operation_id.clone(),
                payload_digest: delete.payload_digest.clone(),
                expected_conversation_epoch: delete.expected_epoch,
                items: Vec::new(),
                conversation_delete_receipt: Some(
                    crate::memory::ManagedConversationDeleteReceipt {
                        operation_id: delete.operation_id,
                        payload_digest: delete.payload_digest,
                        deleted_epoch: delete.expected_epoch,
                        retention_floor_epoch: 0,
                        status: crate::memory::ManagedConversationDeleteStatus::AlreadyDeleted,
                    },
                ),
            },
            dropped_operation_ids: Vec::new(),
            retention_floor_epoch: 0,
            status: ScopeRetirementStatus::AlreadyCompleted,
        };

        assert!(receipt.validate().is_err());
        Ok(())
    }

    /// 验证过期的退休会投影出新的下限，而不修改历史。
    #[test]
    fn expired_retirement_projects_new_floor_without_mutating_history() -> crate::error::Result<()>
    {
        let delete = crate::memory::ManagedConversationDelete::prepare("scope", 5)?;
        let receipt = ScopeRetirementReceipt {
            scope: RuntimeScopeAuthority {
                scope_id: "scope".to_string(),
                conversation_epoch: Some(5),
                revision: 13,
                lifecycle: RuntimeScopeLifecycle::Tombstoned,
            },
            manifest: ScopeRetirementManifest {
                delete_operation_id: delete.operation_id.clone(),
                payload_digest: delete.payload_digest.clone(),
                expected_conversation_epoch: delete.expected_epoch,
                items: Vec::new(),
                conversation_delete_receipt: Some(
                    crate::memory::ManagedConversationDeleteReceipt {
                        operation_id: delete.operation_id,
                        payload_digest: delete.payload_digest,
                        deleted_epoch: delete.expected_epoch,
                        retention_floor_epoch: 0,
                        status: crate::memory::ManagedConversationDeleteStatus::AlreadyDeleted,
                    },
                ),
            },
            dropped_operation_ids: Vec::new(),
            retention_floor_epoch: 5,
            status: ScopeRetirementStatus::ReceiptExpired,
        };

        receipt.validate()?;
        let mut stale_floor = receipt.clone();
        stale_floor.retention_floor_epoch = 4;
        assert!(stale_floor.validate().is_err());
        let mut active_scope = receipt.clone();
        active_scope.scope.lifecycle = RuntimeScopeLifecycle::Active;
        assert!(active_scope.validate().is_err());
        let mut missing_delete = receipt.clone();
        missing_delete.manifest.conversation_delete_receipt = None;
        assert!(missing_delete.validate().is_err());
        let mut incomplete_manifest = receipt;
        incomplete_manifest
            .manifest
            .items
            .push(ScopeRetirementItem {
                runtime_state_id: "runtime".to_string(),
                state_version: RuntimeStateVersion::Managed { revision: 1 },
                pending_operation_id: None,
                status: ScopeRetirementItemStatus::Pending,
            });
        assert!(incomplete_manifest.validate().is_err());
        Ok(())
    }

    /// 验证进行中且已过期的退休会让所有代次保持待处理状态。
    #[test]
    fn in_progress_expired_retirement_keeps_all_generations_pending() -> crate::error::Result<()> {
        let delete = crate::memory::ManagedConversationDelete::prepare("scope", 5)?;
        let receipt = ScopeRetirementReceipt {
            scope: RuntimeScopeAuthority {
                scope_id: "scope".to_string(),
                conversation_epoch: Some(5),
                revision: 9,
                lifecycle: RuntimeScopeLifecycle::Retiring,
            },
            manifest: ScopeRetirementManifest {
                delete_operation_id: delete.operation_id,
                payload_digest: delete.payload_digest,
                expected_conversation_epoch: delete.expected_epoch,
                items: vec![ScopeRetirementItem {
                    runtime_state_id: "runtime".to_string(),
                    state_version: RuntimeStateVersion::Managed { revision: 2 },
                    pending_operation_id: None,
                    status: ScopeRetirementItemStatus::Pending,
                }],
                conversation_delete_receipt: None,
            },
            dropped_operation_ids: Vec::new(),
            retention_floor_epoch: 5,
            status: ScopeRetirementStatus::ReceiptExpired,
        };

        receipt.validate()?;
        let mut mixed_items = receipt.clone();
        if let Some(item) = mixed_items.manifest.items.first_mut() {
            item.status = ScopeRetirementItemStatus::DroppedByDelete;
        }
        assert!(mixed_items.validate().is_err());
        let mut active_scope = receipt.clone();
        active_scope.scope.lifecycle = RuntimeScopeLifecycle::Active;
        assert!(active_scope.validate().is_err());
        let mut tombstoned_without_receipt = receipt;
        tombstoned_without_receipt.scope.lifecycle = RuntimeScopeLifecycle::Tombstoned;
        assert!(tombstoned_without_receipt.validate().is_err());
        Ok(())
    }

    /// 验证会话重建后重试托管删除，能取回最初的删除回执。
    #[tokio::test]
    async fn managed_delete_retry_after_recreate_recovers_original_receipt()
    -> crate::error::Result<()> {
        use crate::memory::{ConversationProjectionLifecycle, ConversationStore};

        let temp = tempfile::tempdir()?;
        let conversations = crate::memory::FileConversationStore::new(temp.path())?;
        let runtime = FileRuntimeStateStore::new(temp.path())?;
        let acquired = conversations
            .ensure_projection_epoch(managed_projection_request("delete-retry", None))
            .await?;
        let (checkpoint, pending) =
            pending_projection_checkpoint("delete-retry", "delete-generation", 1)?;
        let prepared = runtime
            .compare_and_save_checkpoint(RuntimeCheckpointCasRequest {
                scope_id: "delete-retry".to_string(),
                runtime_state_id: "delete-generation".to_string(),
                conversation_epoch: Some(acquired.authority.epoch),
                expected_scope_revision: 0,
                expected_state_version: RuntimeStateExpectedVersion::Absent,
                checkpoint,
            })
            .await?;
        assert_eq!(prepared.status, RuntimeCheckpointCasStatus::Applied);

        let delete = crate::memory::ManagedConversationDelete::prepare(
            "delete-retry",
            acquired.authority.epoch,
        )?;
        assert!(
            delete_persisted_conversation(&conversations, &runtime, "delete-retry")
                .await
                .is_err()
        );
        let deleted =
            delete_persisted_conversation_managed(&conversations, &runtime, delete.clone()).await?;
        let retirement = deleted.retirement.as_ref().ok_or_else(|| {
            crate::error::ReactError::Other(
                "managed delete lost its retirement receipt".to_string(),
            )
        })?;
        assert_eq!(retirement.status, ScopeRetirementStatus::Completed);
        assert_eq!(retirement.manifest.delete_operation_id, delete.operation_id);
        assert_eq!(retirement.manifest.expected_conversation_epoch, 1);
        assert_eq!(
            retirement.dropped_operation_ids,
            vec![pending.batch.operation_id.clone()]
        );
        assert_eq!(deleted.runtime_state_ids, vec!["delete-generation"]);
        let transcript_receipt = retirement
            .manifest
            .conversation_delete_receipt
            .as_ref()
            .ok_or_else(|| {
                crate::error::ReactError::Other(
                    "managed delete lost its transcript receipt".to_string(),
                )
            })?;
        assert_eq!(transcript_receipt.deleted_epoch, 1);
        assert_eq!(transcript_receipt.retention_floor_epoch, 0);
        assert_eq!(
            transcript_receipt.status,
            crate::memory::ManagedConversationDeleteStatus::Deleted
        );

        let recreated = conversations
            .ensure_projection_epoch(managed_projection_request("delete-retry", Some(1)))
            .await?;
        assert_eq!(recreated.authority.epoch, 2);
        assert_eq!(
            recreated.authority.lifecycle,
            ConversationProjectionLifecycle::Live
        );

        let replayed =
            delete_persisted_conversation_managed(&conversations, &runtime, delete).await?;
        assert_eq!(
            replayed.retirement.map(|receipt| receipt.status),
            Some(ScopeRetirementStatus::AlreadyCompleted)
        );
        let current = conversations
            .get_projection_authority("delete-retry")
            .await?
            .ok_or_else(|| {
                crate::error::ReactError::Other(
                    "recreated conversation authority disappeared".to_string(),
                )
            })?;
        assert_eq!(current.epoch, 2);
        assert_eq!(current.lifecycle, ConversationProjectionLifecycle::Live);
        assert!(
            conversations
                .get_conversation("delete-retry")
                .await?
                .is_some()
        );
        Ok(())
    }

    /// 验证调用方截止时间已过期时，清理与删除都不会产生副作用。
    #[tokio::test]
    async fn expired_caller_deadline_has_no_clear_or_delete_side_effects()
    -> crate::error::Result<()> {
        use crate::memory::ConversationStore;

        let temp = tempfile::tempdir()?;
        let conversations = crate::memory::FileConversationStore::new(temp.path())?;
        let runtime = FileRuntimeStateStore::new(temp.path())?;
        let acquired = conversations
            .ensure_projection_epoch(managed_projection_request("expired-managed", None))
            .await?;
        let (checkpoint, _) =
            pending_projection_checkpoint("expired-managed", "expired-generation", 1)?;
        let prepared = runtime
            .compare_and_save_checkpoint(RuntimeCheckpointCasRequest {
                scope_id: "expired-managed".to_string(),
                runtime_state_id: "expired-generation".to_string(),
                conversation_epoch: Some(acquired.authority.epoch),
                expected_scope_revision: 0,
                expected_state_version: RuntimeStateExpectedVersion::Absent,
                checkpoint,
            })
            .await?;
        let before = runtime
            .load_runtime_state("expired-managed", "expired-generation")
            .await?
            .ok_or_else(|| {
                crate::error::ReactError::Other("prepared runtime state is absent".to_string())
            })?;
        let expired = expired_persistence_context();
        assert!(
            clear_persisted_runtime_incarnation_with_context(
                &conversations,
                &runtime,
                "expired-managed",
                "expired-generation",
                expired,
            )
            .await
            .is_err()
        );
        assert!(
            delete_persisted_conversation_managed_with_context(
                &conversations,
                &runtime,
                crate::memory::ManagedConversationDelete::prepare("expired-managed", 1)?,
                expired,
            )
            .await
            .is_err()
        );
        assert!(
            delete_persisted_conversation_with_context(
                &conversations,
                &runtime,
                "expired-managed",
                expired,
            )
            .await
            .is_err()
        );
        let after = runtime
            .load_runtime_state("expired-managed", "expired-generation")
            .await?
            .ok_or_else(|| {
                crate::error::ReactError::Other(
                    "runtime state disappeared after expired call".to_string(),
                )
            })?;
        assert_eq!(after.scope, before.scope);
        assert_eq!(after.version, before.version);
        assert_eq!(
            after
                .checkpoint
                .as_ref()
                .map(|checkpoint| checkpoint.messages_json.as_str()),
            before
                .checkpoint
                .as_ref()
                .map(|checkpoint| checkpoint.messages_json.as_str())
        );
        assert_eq!(
            runtime.load_scope_authority("expired-managed").await?,
            Some(prepared.scope)
        );
        assert!(
            conversations
                .get_conversation("expired-managed")
                .await?
                .is_some()
        );

        let legacy_conversations = LegacyConversationProbe::default();
        let legacy_runtime = LegacyRuntimeProbe::default();
        let legacy_context = crate::memory::PersistenceCallContext::with_timeout(
            std::time::Duration::from_secs(10),
        )?;
        assert!(
            clear_persisted_runtime_incarnation_with_context(
                &legacy_conversations,
                &legacy_runtime,
                "legacy-scope",
                "legacy-runtime",
                legacy_context,
            )
            .await
            .is_err()
        );
        assert!(
            delete_persisted_conversation_with_context(
                &legacy_conversations,
                &legacy_runtime,
                "legacy-scope",
                legacy_context,
            )
            .await
            .is_err()
        );
        assert_eq!(
            legacy_conversations
                .calls
                .load(std::sync::atomic::Ordering::SeqCst),
            0
        );
        assert_eq!(
            legacy_runtime
                .calls
                .load(std::sync::atomic::Ordering::SeqCst),
            0
        );

        Ok(())
    }

    /// 验证未持久化失败分类的终态失败绝不会被发布为成功。
    #[tokio::test]
    async fn terminal_failure_without_durable_classification_never_publishes_success()
    -> crate::error::Result<()> {
        use crate::memory::{ConversationStore, TranscriptProjectionErrorClass};

        let temp = tempfile::tempdir()?;
        let conversations = crate::memory::FileConversationStore::new(temp.path())?;
        let runtime = RejectFailureCasRuntimeStore {
            inner: FileRuntimeStateStore::new(temp.path())?,
        };
        let acquired = conversations
            .ensure_projection_epoch(managed_projection_request("failure-cas", None))
            .await?;
        let (checkpoint, _) = pending_projection_checkpoint("failure-cas", "failure-runtime", 1)?;
        let prepared = runtime
            .compare_and_save_checkpoint(RuntimeCheckpointCasRequest {
                scope_id: "failure-cas".to_string(),
                runtime_state_id: "failure-runtime".to_string(),
                conversation_epoch: Some(acquired.authority.epoch),
                expected_scope_revision: 0,
                expected_state_version: RuntimeStateExpectedVersion::Absent,
                checkpoint,
            })
            .await?;
        assert_eq!(prepared.status, RuntimeCheckpointCasStatus::Applied);
        conversations
            .delete_managed_conversation(crate::memory::ManagedConversationDelete::prepare(
                "failure-cas",
                acquired.authority.epoch,
            )?)
            .await?;
        conversations
            .ensure_projection_epoch(managed_projection_request(
                "failure-cas",
                Some(acquired.authority.epoch),
            ))
            .await?;

        for expected_attempt in [2_u32, 3_u32] {
            let context = crate::memory::PersistenceCallContext::with_timeout(
                std::time::Duration::from_secs(10),
            )?;
            let settlement = settle_pending_transcript_projection(
                &conversations,
                &runtime,
                "failure-cas",
                "failure-runtime",
                context,
            )
            .await?;
            assert_eq!(
                settlement.status,
                crate::memory::TranscriptProjectionSettlementStatus::Conflict
            );
            assert_eq!(
                settlement.error_class,
                Some(TranscriptProjectionErrorClass::OutcomeUnknown)
            );
            assert!(
                settlement
                    .detail
                    .is_some_and(|detail| detail.contains("not durable"))
            );
            let durable = runtime
                .load_runtime_state("failure-cas", "failure-runtime")
                .await?
                .ok_or_else(|| {
                    crate::error::ReactError::Other(
                        "runtime pending disappeared after rejected failure CAS".to_string(),
                    )
                })?;
            let payload = durable
                .checkpoint
                .as_ref()
                .ok_or_else(|| {
                    crate::error::ReactError::Other(
                        "runtime pending checkpoint disappeared".to_string(),
                    )
                })?
                .restore_managed_runtime_payload()?;
            let pending = payload.pending_transcript_projection.ok_or_else(|| {
                crate::error::ReactError::Other(
                    "runtime pending debt was cleared without acknowledgement".to_string(),
                )
            })?;
            assert_eq!(pending.attempt, expected_attempt);
            assert!(pending.last_attempt_class.is_none());
            assert!(pending.last_error.is_none());
        }

        let durable = runtime
            .load_runtime_state("failure-cas", "failure-runtime")
            .await?
            .ok_or_else(|| {
                crate::error::ReactError::Other("runtime pending disappeared".to_string())
            })?;
        let payload = durable
            .checkpoint
            .as_ref()
            .ok_or_else(|| {
                crate::error::ReactError::Other(
                    "runtime pending checkpoint disappeared".to_string(),
                )
            })?
            .restore_managed_runtime_payload()?;
        let pending = payload.pending_transcript_projection.ok_or_else(|| {
            crate::error::ReactError::Other("runtime pending debt disappeared".to_string())
        })?;
        let blocked = recorded_transcript_failure_settlement(
            &runtime,
            &durable,
            &payload.messages,
            &pending,
            TranscriptProjectionFailure {
                status: crate::memory::TranscriptProjectionSettlementStatus::Blocked,
                error_class: TranscriptProjectionErrorClass::Unsupported,
                detail: "injected unsupported apply",
            },
            PersistenceOperationDeadline::new()?,
        )
        .await;
        assert_eq!(
            blocked.status,
            crate::memory::TranscriptProjectionSettlementStatus::Blocked
        );
        assert_eq!(
            blocked.error_class,
            Some(TranscriptProjectionErrorClass::OutcomeUnknown)
        );
        Ok(())
    }

    /// 验证精确清理会在退休前先结算待处理的投影。
    #[tokio::test]
    async fn exact_clear_settles_pending_projection_before_retirement() -> crate::error::Result<()>
    {
        use crate::memory::ConversationStore;

        let temp = tempfile::tempdir()?;
        let conversations = crate::memory::FileConversationStore::new(temp.path())?;
        let runtime = FileRuntimeStateStore::new(temp.path())?;
        let acquired = conversations
            .ensure_projection_epoch(managed_projection_request("clear-pending", None))
            .await?;
        let (checkpoint, pending) =
            pending_projection_checkpoint("clear-pending", "clear-pending", 1)?;
        let operation_id = pending.batch.operation_id.clone();
        let prepared = runtime
            .compare_and_save_checkpoint(RuntimeCheckpointCasRequest {
                scope_id: "clear-pending".to_string(),
                runtime_state_id: "clear-pending".to_string(),
                conversation_epoch: Some(acquired.authority.epoch),
                expected_scope_revision: 0,
                expected_state_version: RuntimeStateExpectedVersion::Absent,
                checkpoint,
            })
            .await?;
        assert_eq!(prepared.status, RuntimeCheckpointCasStatus::Applied);

        let cleared = clear_persisted_runtime_incarnation(
            &conversations,
            &runtime,
            "clear-pending",
            "clear-pending",
        )
        .await?;
        assert!(cleared.checkpoint_removed);
        let retired = runtime
            .load_runtime_state("clear-pending", "clear-pending")
            .await?
            .ok_or_else(|| {
                crate::error::ReactError::Other("retired runtime tombstone is absent".to_string())
            })?;
        assert!(matches!(
            retired.version,
            RuntimeStateVersion::Retired { .. }
        ));
        assert_eq!(
            conversations
                .apply_transcript_projection(pending.batch)
                .await?
                .operation_id,
            operation_id
        );
        Ok(())
    }

    /// 验证精确清理的重试不会删除重建后新实例的 transcript。
    #[tokio::test]
    async fn exact_clear_retry_does_not_delete_recreated_incarnation_transcript()
    -> crate::error::Result<()> {
        use crate::memory::{ConversationProjectionLifecycle, ConversationStore};

        let temp = tempfile::tempdir()?;
        let conversations = crate::memory::FileConversationStore::new(temp.path())?;
        let runtime = FileRuntimeStateStore::new(temp.path())?;
        let scope = conversations
            .ensure_projection_epoch(managed_projection_request("clear-scope", None))
            .await?;
        let incarnation = conversations
            .ensure_projection_epoch(managed_projection_request("clear-generation", None))
            .await?;
        let (checkpoint, _) = pending_projection_checkpoint(
            "clear-scope",
            "clear-generation",
            scope.authority.epoch,
        )?;
        let prepared = runtime
            .compare_and_save_checkpoint(RuntimeCheckpointCasRequest {
                scope_id: "clear-scope".to_string(),
                runtime_state_id: "clear-generation".to_string(),
                conversation_epoch: Some(scope.authority.epoch),
                expected_scope_revision: 0,
                expected_state_version: RuntimeStateExpectedVersion::Absent,
                checkpoint,
            })
            .await?;
        assert_eq!(prepared.status, RuntimeCheckpointCasStatus::Applied);

        let first = clear_persisted_runtime_incarnation(
            &conversations,
            &runtime,
            "clear-scope",
            "clear-generation",
        )
        .await?;
        assert!(first.checkpoint_removed);
        assert!(
            conversations
                .get_conversation("clear-generation")
                .await?
                .is_some()
        );

        let incarnation_delete = crate::memory::ManagedConversationDelete::prepare(
            "clear-generation",
            incarnation.authority.epoch,
        )?;
        assert_eq!(
            conversations
                .delete_managed_conversation(incarnation_delete)
                .await?
                .status,
            crate::memory::ManagedConversationDeleteStatus::Deleted
        );
        let recreated = conversations
            .ensure_projection_epoch(managed_projection_request(
                "clear-generation",
                Some(incarnation.authority.epoch),
            ))
            .await?;
        assert_eq!(recreated.authority.epoch, 2);

        let replayed = clear_persisted_runtime_incarnation(
            &conversations,
            &runtime,
            "clear-scope",
            "clear-generation",
        )
        .await?;
        assert!(!replayed.checkpoint_removed);
        let current = conversations
            .get_projection_authority("clear-generation")
            .await?
            .ok_or_else(|| {
                crate::error::ReactError::Other(
                    "recreated incarnation transcript disappeared".to_string(),
                )
            })?;
        assert_eq!(current.epoch, 2);
        assert_eq!(current.lifecycle, ConversationProjectionLifecycle::Live);
        assert!(
            conversations
                .get_conversation("clear-generation")
                .await?
                .is_some()
        );
        Ok(())
    }
}
