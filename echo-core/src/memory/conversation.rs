//! 会话持久化 trait 与数据类型
//!
//! 提供支持多用户、多 Agent 隔离的会话历史结构化存储。具体实现
//! （`SqliteConversationStore`）位于 `echo_state` 中。

use crate::error::{MemoryError, Result};
use futures::future::BoxFuture;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const TRANSCRIPT_PROJECTION_SCHEMA_VERSION: u16 = 1;
const TRANSCRIPT_PROJECTION_EPOCH_MAX: u64 = i64::MAX as u64;
const TRANSCRIPT_PROJECTION_ORDINAL_MAX: u64 = i64::MAX as u64;

/** 通过 Host/扩展适配器传播的非权威性调用元数据。

截止时间不参与操作摘要（digest）的计算，因此持久化效果可以在获得一个全新的、
有限的恢复预算后被重试。
*/
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PersistenceCallContext {
    /// 该调用的绝对截止时间（Unix 毫秒时间戳）。
    pub absolute_deadline_unix_ms: i64,
}

impl PersistenceCallContext {
    /// 依据给定的超时时长，构造一个以当前时间为起点的调用上下文。
    pub fn with_timeout(timeout: std::time::Duration) -> Result<Self> {
        let deadline = std::time::SystemTime::now()
            .checked_add(timeout)
            .ok_or_else(|| projection_error("persistence deadline overflow"))?;
        let since_epoch = deadline
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|error| projection_error(format!("invalid persistence deadline: {error}")))?;
        let absolute_deadline_unix_ms =
            i64::try_from(since_epoch.as_millis()).map_err(|error| {
                projection_error(format!(
                    "persistence deadline is outside i64 range: {error}"
                ))
            })?;
        Ok(Self {
            absolute_deadline_unix_ms,
        })
    }

    /// 返回该调用到期前剩余的墙钟（wall-clock）时间预算。
    pub fn remaining_duration(self) -> std::result::Result<std::time::Duration, MemoryError> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|error| {
                MemoryError::SerializationError(format!("invalid system time: {error}"))
            })?;
        let now_unix_ms = i64::try_from(now.as_millis()).map_err(|error| {
            MemoryError::SerializationError(format!("system time is outside i64 range: {error}"))
        })?;
        let remaining_ms = self
            .absolute_deadline_unix_ms
            .checked_sub(now_unix_ms)
            .filter(|remaining| *remaining > 0)
            .ok_or_else(|| {
                MemoryError::DeadlineExceeded(format!(
                    "deadline {}ms elapsed at {now_unix_ms}ms",
                    self.absolute_deadline_unix_ms
                ))
            })?;
        let remaining_ms = u64::try_from(remaining_ms).map_err(|error| {
            MemoryError::SerializationError(format!(
                "remaining deadline is outside u64 range: {error}"
            ))
        })?;
        Ok(std::time::Duration::from_millis(remaining_ms))
    }

    /** 在一次托管持久化调用开始持久化工作之前，判断是否应予以拒绝。

    后端不得将一个已完成的持久化操作重新映射为截止时间错误，因为那样会掩盖
    该操作究竟是否已提交成功。
    */
    pub fn ensure_not_expired(self) -> std::result::Result<(), MemoryError> {
        self.remaining_duration().map(|_| ())
    }
}

/// 创建新会话所需的参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewConversation {
    /// 外部唯一 ID（由前端生成，例如 `conv-1709000000-abc123`）
    pub conversation_id: String,
    /// 用户 ID（用于多用户隔离）
    #[serde(default = "default_user_id")]
    pub user_id: String,
    /// 该会话所属的 Agent 类型
    pub agent_type: Option<String>,
    /// 会话标题（首轮摘要）
    pub title: Option<String>,
}

/// 返回默认用户 ID（`"default"`），用于 `user_id` 字段的序列化默认值。
fn default_user_id() -> String {
    "default".to_string()
}

/// 完整的会话记录
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conversation {
    /// 数据库自增 ID
    pub id: i64,
    /// 外部唯一 ID
    pub conversation_id: String,
    /// 用户 ID
    pub user_id: String,
    /// Agent 类型
    pub agent_type: Option<String>,
    /// 会话标题
    pub title: Option<String>,
    /// 压缩后的上下文摘要
    pub summary: Option<String>,
    /// 压缩边界：摘要覆盖到该 id（含）为止的消息
    pub compressed_before_id: Option<i64>,
    /// 创建时间
    pub created_at: String,
    /// 更新时间
    pub updated_at: String,
}

/// 会话列表条目（轻量级，不包含消息内容或摘要）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationMeta {
    /// 数据库自增 ID
    pub id: i64,
    /// 外部唯一 ID
    pub conversation_id: String,
    /// 用户 ID
    pub user_id: String,
    /// 会话标题
    pub title: Option<String>,
    /// 该会话包含的消息数量
    pub message_count: usize,
    /// 创建时间
    pub created_at: String,
    /// 更新时间
    pub updated_at: String,
}

/// 已持久化的消息（独立于 LLM 的 Message 类型，附带持久化相关字段）
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StoredMessage {
    /// 数据库自增 ID（新建消息为 None）
    pub id: Option<i64>,
    /// 所属会话的外部 ID
    pub conversation_id: String,
    /// 角色：user / assistant / tool / system
    pub role: String,
    /// 消息内容
    pub content: Option<String>,
    /// 附件元数据（JSON）
    pub attachments_json: Option<String>,
    /// 工具调用记录（JSON）
    pub tool_calls_json: Option<String>,
    /// 工具执行结果（JSON）
    pub tool_result_json: Option<String>,
    /// 创建时间
    pub created_at: String,
}

/// 无需变更存储即可对外声明的原子化转录投影（transcript projection）能力等级。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConversationProjectionCapability {
    /// 不支持原子化转录投影。
    Unsupported,
    /// 支持第一版原子化转录投影协议。
    AtomicV1,
}

/// 托管持久化操作所支持的截止时间契约等级。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PersistenceCallCapability {
    /// 不支持截止时间契约。
    Unsupported,
    /// 拒绝那些在持久化工作开始前绝对截止时间已过期的调用。
    AbsoluteDeadlineV1,
}

/// 托管会话转录（transcript）的持久化生命周期状态。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConversationProjectionLifecycle {
    /// 转录当前存活（未被删除）。
    Live,
    /// 转录已被删除（墓碑状态）。
    Deleted,
}

/// 已提交转录权威（authority）当前的 epoch 与 revision。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConversationProjectionAuthority {
    /// 该权威所属的会话 ID。
    pub conversation_id: String,
    /// 当前的 epoch（每次重建会话时递增）。
    pub epoch: u64,
    /// 当前 epoch 内的 revision（每次成功变更时递增）。
    pub revision: u64,
    /// 当前生命周期状态（存活或已删除）。
    pub lifecycle: ConversationProjectionLifecycle,
    /// 已删除且其精确操作回执不再保留的最高 epoch。
    pub delete_receipt_retention_floor_epoch: u64,
}

/// 获取某一投影生成（projection generation）所用 epoch 的请求。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsureConversationProjectionRequest {
    /// 待创建或确认的会话参数。
    pub conversation: NewConversation,
    /// 当重建一个已存在托管墓碑的会话时，必须提供的期望墓碑 epoch。
    pub expected_tombstone_epoch: Option<u64>,
}

/// 获取托管转录 epoch 的结果状态。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConversationProjectionEpochStatus {
    /// 新建了一个会话及其 epoch。
    Created,
    /// 采纳了一个既有的、非托管的历史会话数据。
    AdoptedLegacy,
    /// 会话及其 epoch 已存在，直接复用。
    Existing,
    /// 覆盖已删除的墓碑，重新创建了会话。
    Recreated,
    /// 会话当前处于已删除（墓碑）状态。
    Tombstoned,
    /// 请求的期望 epoch 与当前权威状态冲突。
    EpochConflict,
}

/// 托管转录 epoch 获取操作的稳定回执。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConversationProjectionEpochReceipt {
    /// 获取后的当前权威状态。
    pub authority: ConversationProjectionAuthority,
    /// 本次获取操作的结果状态。
    pub status: ConversationProjectionEpochStatus,
}

/// 转录投影批次中一条完整、不可变的消息效果（effect）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TranscriptProjectionItem {
    /// 该消息在转录中的序号。
    pub ordinal: u64,
    /// 该消息内容的摘要（digest）。
    pub digest: String,
    /// 消息本体。
    pub message: StoredMessage,
}

/// 一次运行时生成（generation）对应的、规范化且可安全重试的转录效果。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TranscriptProjectionBatch {
    /// 该批次所遵循的 schema 版本。
    pub schema_version: u16,
    /// 该批次的幂等操作标识。
    pub operation_id: String,
    /// 该批次载荷（payload）的摘要。
    pub payload_digest: String,
    /// 目标会话 ID。
    pub conversation_id: String,
    /// 目标会话的 epoch。
    pub conversation_epoch: u64,
    /// 产生该批次的运行时生成标识。
    pub generation_id: String,
    /// 该批次包含的消息效果列表。
    pub items: Vec<TranscriptProjectionItem>,
}

impl TranscriptProjectionBatch {
    /// 一次性构造一个规范化批次。恢复流程会持久化并复用该值；
    /// 不得重新生成消息时间戳或其它效果字段。
    pub fn prepare(
        conversation_id: impl Into<String>,
        conversation_epoch: u64,
        generation_id: impl Into<String>,
        first_ordinal: u64,
        messages: Vec<StoredMessage>,
    ) -> Result<Self> {
        let conversation_id = conversation_id.into();
        let generation_id = generation_id.into();
        validate_projection_identity(&conversation_id, conversation_epoch, &generation_id)?;
        if messages.is_empty() {
            return Err(projection_error(
                "transcript projection batch must not be empty",
            ));
        }

        let mut items = Vec::with_capacity(messages.len());
        for (offset, message) in messages.into_iter().enumerate() {
            if message.id.is_some() {
                return Err(projection_error(
                    "prepared transcript projection messages must not carry backend ids",
                ));
            }
            if message.conversation_id != conversation_id {
                return Err(projection_error(
                    "prepared transcript projection message belongs to another conversation",
                ));
            }
            if message.created_at.trim().is_empty() {
                return Err(projection_error(
                    "prepared transcript projection message is missing created_at",
                ));
            }
            let offset = u64::try_from(offset).map_err(|error| {
                projection_error(format!("transcript projection ordinal overflow: {error}"))
            })?;
            let ordinal = first_ordinal
                .checked_add(offset)
                .ok_or_else(|| projection_error("transcript projection ordinal overflow"))?;
            if ordinal > TRANSCRIPT_PROJECTION_ORDINAL_MAX {
                return Err(projection_error(
                    "transcript projection ordinal exceeds backend-compatible range",
                ));
            }
            let digest = digest_serialized(&message)?;
            items.push(TranscriptProjectionItem {
                ordinal,
                digest,
                message,
            });
        }

        let payload_digest = projection_payload_digest(
            TRANSCRIPT_PROJECTION_SCHEMA_VERSION,
            &conversation_id,
            conversation_epoch,
            &generation_id,
            &items,
        )?;
        Ok(Self {
            schema_version: TRANSCRIPT_PROJECTION_SCHEMA_VERSION,
            operation_id: format!("transcript-projection-v1:{payload_digest}"),
            payload_digest,
            conversation_id,
            conversation_epoch,
            generation_id,
            items,
        })
    }

    /// 在后端观察到任何效果之前，校验一个已解码的批次。
    pub fn validate(&self) -> Result<()> {
        validate_projection_identity(
            &self.conversation_id,
            self.conversation_epoch,
            &self.generation_id,
        )?;
        if self.schema_version != TRANSCRIPT_PROJECTION_SCHEMA_VERSION || self.items.is_empty() {
            return Err(projection_error(
                "unsupported or empty transcript projection batch",
            ));
        }
        let mut previous: Option<u64> = None;
        for item in &self.items {
            let contiguous = match previous {
                Some(previous) => previous
                    .checked_add(1)
                    .is_some_and(|next| item.ordinal == next),
                None => true,
            };
            if item.message.id.is_some()
                || item.message.conversation_id != self.conversation_id
                || item.message.created_at.trim().is_empty()
                || item.ordinal > TRANSCRIPT_PROJECTION_ORDINAL_MAX
                || !contiguous
                || digest_serialized(&item.message)? != item.digest
            {
                return Err(projection_error(
                    "transcript projection batch payload is not canonical",
                ));
            }
            previous = Some(item.ordinal);
        }
        let digest = projection_payload_digest(
            self.schema_version,
            &self.conversation_id,
            self.conversation_epoch,
            &self.generation_id,
            &self.items,
        )?;
        if digest != self.payload_digest
            || self.operation_id != format!("transcript-projection-v1:{digest}")
        {
            return Err(projection_error(
                "transcript projection operation identity does not match its payload",
            ));
        }
        Ok(())
    }
}

/// 原子化应用转录投影时得到的冲突种类。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TranscriptProjectionConflictKind {
    /// 操作标识（operation identity）不一致。
    OperationIdentity,
    /// 序号或摘要不一致。
    OrdinalDigest,
    /// revision 不一致。
    Revision,
    /// 与托管导入（managed import）冲突。
    ManagedImport,
}

/// 原子化应用一次转录投影后的业务结果。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TranscriptProjectionApplyStatus {
    /// 本次应用成功写入。
    Applied,
    /// 该批次此前已被应用过（幂等重放）。
    AlreadyApplied,
    /// 因当前 epoch 或生命周期状态被拦截（fenced）。
    Fenced {
        /// 当前实际的 epoch。
        current_epoch: u64,
        /// 当前实际的生命周期状态。
        lifecycle: ConversationProjectionLifecycle,
    },
    /// 与当前权威状态发生冲突。
    Conflict {
        /// 当前实际的 epoch。
        current_epoch: u64,
        /// 当前实际的 revision。
        current_revision: u64,
        /// 冲突的具体种类。
        kind: TranscriptProjectionConflictKind,
    },
}

/// 一次原子化投影尝试的稳定回执。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TranscriptProjectionApplyReceipt {
    /// 本次操作的幂等标识。
    pub operation_id: String,
    /// 本次操作载荷的摘要。
    pub payload_digest: String,
    /// 应用后的当前权威状态。
    pub authority: ConversationProjectionAuthority,
    /// 本次应用操作的结果状态。
    pub status: TranscriptProjectionApplyStatus,
}

/// 在一次调用终态（terminal）之前观察到的框架层结算状态。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TranscriptProjectionSettlementStatus {
    /// 已完成结算。
    Settled,
    /// 结算被推迟。
    Deferred,
    /// 结算被阻塞。
    Blocked,
    /// 结算发生冲突。
    Conflict,
}

/// 投影结算失败的稳定分类。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TranscriptProjectionErrorClass {
    /// 操作结果未知（可能提交也可能未提交）。
    OutcomeUnknown,
    /// 已知失败且未产生任何持久化提交，可安全重试。
    TransientNoCommit,
    /// revision 冲突。
    RevisionConflict,
    /// 截止时间已超时。
    DeadlineExceeded,
    /// 后端不支持该操作。
    Unsupported,
    /// 配置无效。
    InvalidConfiguration,
    /// 存储状态已损坏。
    CorruptState,
    /// 语义层面的冲突。
    SemanticConflict,
}

/// 对一次转录投影安全点（safe point）的类型化观测结果。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TranscriptProjectionSettlement {
    /// 结算状态。
    pub status: TranscriptProjectionSettlementStatus,
    /// 关联的操作标识（若已知）。
    pub operation_id: Option<String>,
    /// 关联的会话 ID（若已知）。
    pub conversation_id: Option<String>,
    /// 关联的运行时生成标识（若已知）。
    pub generation_id: Option<String>,
    /// 当前重试次数。
    pub attempt: u32,
    /// 失败时的错误分类（若适用）。
    pub error_class: Option<TranscriptProjectionErrorClass>,
    /// 附加的诊断说明文本。
    pub detail: Option<String>,
}

/// 用于替换一个托管转录的显式 CAS（比较并交换）导入请求。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManagedConversationImport {
    /// 该请求所遵循的 schema 版本。
    pub schema_version: u16,
    /// 该请求的幂等操作标识。
    pub operation_id: String,
    /// 该请求载荷的摘要。
    pub payload_digest: String,
    /// 目标会话 ID。
    pub conversation_id: String,
    /// 期望的当前 epoch（用于 CAS 校验）。
    pub expected_epoch: u64,
    /// 期望的当前 revision（用于 CAS 校验）。
    pub expected_revision: u64,
    /// 用于替换的完整消息集合。
    pub messages: Vec<StoredMessage>,
}

impl ManagedConversationImport {
    /// 构造一个规范化的托管会话导入请求，并计算其幂等标识与载荷摘要。
    pub fn prepare(
        conversation_id: impl Into<String>,
        expected_epoch: u64,
        expected_revision: u64,
        messages: Vec<StoredMessage>,
    ) -> Result<Self> {
        let conversation_id = conversation_id.into();
        validate_managed_messages(&conversation_id, &messages)?;
        validate_managed_epoch(expected_epoch)?;
        let payload_digest = digest_serialized(&ManagedImportIdentity {
            schema_version: TRANSCRIPT_PROJECTION_SCHEMA_VERSION,
            conversation_id: &conversation_id,
            expected_epoch,
            expected_revision,
            messages: &messages,
        })?;
        Ok(Self {
            schema_version: TRANSCRIPT_PROJECTION_SCHEMA_VERSION,
            operation_id: format!("managed-conversation-import-v1:{payload_digest}"),
            payload_digest,
            conversation_id,
            expected_epoch,
            expected_revision,
            messages,
        })
    }

    /// 在后端观察到任何效果之前，校验该导入请求的完整性与身份一致性。
    pub fn validate(&self) -> Result<()> {
        validate_managed_messages(&self.conversation_id, &self.messages)?;
        validate_managed_epoch(self.expected_epoch)?;
        if self.schema_version != TRANSCRIPT_PROJECTION_SCHEMA_VERSION {
            return Err(projection_error(
                "unsupported managed conversation import schema",
            ));
        }
        let digest = digest_serialized(&ManagedImportIdentity {
            schema_version: self.schema_version,
            conversation_id: &self.conversation_id,
            expected_epoch: self.expected_epoch,
            expected_revision: self.expected_revision,
            messages: &self.messages,
        })?;
        if digest != self.payload_digest
            || self.operation_id != format!("managed-conversation-import-v1:{digest}")
        {
            return Err(projection_error(
                "managed conversation import identity does not match its payload",
            ));
        }
        Ok(())
    }
}

/// 受 epoch 与 revision 拦截保护的托管会话元数据修改请求。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManagedConversationMetadataUpdate {
    /// 该请求所遵循的 schema 版本。
    pub schema_version: u16,
    /// 该请求的幂等操作标识。
    pub operation_id: String,
    /// 该请求载荷的摘要。
    pub payload_digest: String,
    /// 目标会话 ID。
    pub conversation_id: String,
    /// 期望的当前 epoch（用于拦截校验）。
    pub expected_epoch: u64,
    /// 期望的当前 revision（用于拦截校验）。
    pub expected_revision: u64,
    /// 新的会话标题（`None` 表示不修改）。
    pub title: Option<String>,
    /// 新的会话摘要（`None` 表示不修改）。
    pub summary: Option<String>,
    /// 新的压缩边界（`None` 表示不修改）。
    pub compressed_before_id: Option<i64>,
}

impl ManagedConversationMetadataUpdate {
    /// 构造一个规范化的托管会话元数据修改请求，并计算其幂等标识与载荷摘要。
    #[allow(clippy::too_many_arguments)]
    pub fn prepare(
        conversation_id: impl Into<String>,
        expected_epoch: u64,
        expected_revision: u64,
        title: Option<String>,
        summary: Option<String>,
        compressed_before_id: Option<i64>,
    ) -> Result<Self> {
        let conversation_id = conversation_id.into();
        if conversation_id.trim().is_empty() {
            return Err(projection_error(
                "managed conversation metadata identity must not be empty",
            ));
        }
        validate_managed_epoch(expected_epoch)?;
        let payload_digest = digest_serialized(&ManagedMetadataIdentity {
            schema_version: TRANSCRIPT_PROJECTION_SCHEMA_VERSION,
            conversation_id: &conversation_id,
            expected_epoch,
            expected_revision,
            title: title.as_deref(),
            summary: summary.as_deref(),
            compressed_before_id,
        })?;
        Ok(Self {
            schema_version: TRANSCRIPT_PROJECTION_SCHEMA_VERSION,
            operation_id: format!("managed-conversation-metadata-v1:{payload_digest}"),
            payload_digest,
            conversation_id,
            expected_epoch,
            expected_revision,
            title,
            summary,
            compressed_before_id,
        })
    }

    /// 在后端观察到任何效果之前，校验该元数据修改请求的完整性与身份一致性。
    pub fn validate(&self) -> Result<()> {
        if self.conversation_id.trim().is_empty()
            || self.schema_version != TRANSCRIPT_PROJECTION_SCHEMA_VERSION
        {
            return Err(projection_error(
                "invalid managed conversation metadata update",
            ));
        }
        validate_managed_epoch(self.expected_epoch)?;
        let digest = digest_serialized(&ManagedMetadataIdentity {
            schema_version: self.schema_version,
            conversation_id: &self.conversation_id,
            expected_epoch: self.expected_epoch,
            expected_revision: self.expected_revision,
            title: self.title.as_deref(),
            summary: self.summary.as_deref(),
            compressed_before_id: self.compressed_before_id,
        })?;
        if digest != self.payload_digest
            || self.operation_id != format!("managed-conversation-metadata-v1:{digest}")
        {
            return Err(projection_error(
                "managed conversation metadata identity does not match its payload",
            ));
        }
        Ok(())
    }
}

/// 托管会话元数据修改操作的结果状态。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ManagedConversationMetadataUpdateStatus {
    /// 本次修改成功写入。
    Updated,
    /// 该请求此前已被应用过（幂等重放）。
    AlreadyUpdated,
    /// epoch 不匹配。
    EpochConflict,
    /// revision 不匹配。
    RevisionConflict,
    /// 身份标识不匹配。
    IdentityConflict,
}

/// 托管会话元数据修改操作的稳定回执。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ManagedConversationMetadataUpdateReceipt {
    /// 本次操作的幂等标识。
    pub operation_id: String,
    /// 本次操作载荷的摘要。
    pub payload_digest: String,
    /// 修改后的当前权威状态。
    pub authority: ConversationProjectionAuthority,
    /// 本次修改操作的结果状态。
    pub status: ManagedConversationMetadataUpdateStatus,
}

/// 拦截（fence）并删除一个托管会话具体实例（incarnation）的请求。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManagedConversationDelete {
    /// 该请求所遵循的 schema 版本。
    pub schema_version: u16,
    /// 该请求的幂等操作标识。
    pub operation_id: String,
    /// 该请求载荷的摘要。
    pub payload_digest: String,
    /// 目标会话 ID。
    pub conversation_id: String,
    /// 期望的当前 epoch（用于拦截校验）。
    pub expected_epoch: u64,
}

impl ManagedConversationDelete {
    /// 构造一个规范化的托管会话删除请求，并计算其幂等标识与载荷摘要。
    pub fn prepare(conversation_id: impl Into<String>, expected_epoch: u64) -> Result<Self> {
        let conversation_id = conversation_id.into();
        if conversation_id.trim().is_empty() {
            return Err(projection_error(
                "managed conversation delete identity must not be empty",
            ));
        }
        validate_managed_epoch(expected_epoch)?;
        let payload_digest = digest_serialized(&ManagedDeleteIdentity {
            schema_version: TRANSCRIPT_PROJECTION_SCHEMA_VERSION,
            conversation_id: &conversation_id,
            expected_epoch,
        })?;
        Ok(Self {
            schema_version: TRANSCRIPT_PROJECTION_SCHEMA_VERSION,
            operation_id: format!("managed-conversation-delete-v1:{payload_digest}"),
            payload_digest,
            conversation_id,
            expected_epoch,
        })
    }

    /// 在后端观察到任何效果之前，校验该删除请求的完整性与身份一致性。
    pub fn validate(&self) -> Result<()> {
        if self.conversation_id.trim().is_empty()
            || self.schema_version != TRANSCRIPT_PROJECTION_SCHEMA_VERSION
        {
            return Err(projection_error(
                "invalid managed conversation delete request",
            ));
        }
        validate_managed_epoch(self.expected_epoch)?;
        let digest = digest_serialized(&ManagedDeleteIdentity {
            schema_version: self.schema_version,
            conversation_id: &self.conversation_id,
            expected_epoch: self.expected_epoch,
        })?;
        if digest != self.payload_digest
            || self.operation_id != format!("managed-conversation-delete-v1:{digest}")
        {
            return Err(projection_error(
                "managed conversation delete identity does not match its payload",
            ));
        }
        Ok(())
    }
}

/// 托管会话删除请求的业务结果。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ManagedConversationDeleteStatus {
    /// 本次删除成功执行。
    Deleted,
    /// 该会话此前已被删除（幂等重放）。
    AlreadyDeleted,
    /// epoch 不匹配。
    EpochConflict,
    /// 相关回执已过期，无法确认结果。
    ReceiptExpired,
    /// 身份标识不匹配。
    IdentityConflict,
}

/// 受 epoch 拦截保护的转录删除操作的稳定回执。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ManagedConversationDeleteReceipt {
    /// 本次操作的幂等标识。
    pub operation_id: String,
    /// 本次操作载荷的摘要。
    pub payload_digest: String,
    /// 被删除的 epoch。
    pub deleted_epoch: u64,
    /// 删除后的回执保留下限 epoch。
    pub retention_floor_epoch: u64,
    /// 本次删除操作的结果状态。
    pub status: ManagedConversationDeleteStatus,
}

/// 用于计算转录投影载荷摘要的内部身份结构。
#[derive(Serialize)]
struct ProjectionPayloadIdentity<'a> {
    schema_version: u16,
    conversation_id: &'a str,
    conversation_epoch: u64,
    generation_id: &'a str,
    items: &'a [TranscriptProjectionItem],
}

/// 用于计算托管会话导入载荷摘要的内部身份结构。
#[derive(Serialize)]
struct ManagedImportIdentity<'a> {
    schema_version: u16,
    conversation_id: &'a str,
    expected_epoch: u64,
    expected_revision: u64,
    messages: &'a [StoredMessage],
}

/// 用于计算托管会话删除载荷摘要的内部身份结构。
#[derive(Serialize)]
struct ManagedDeleteIdentity<'a> {
    schema_version: u16,
    conversation_id: &'a str,
    expected_epoch: u64,
}

/// 用于计算托管会话元数据修改载荷摘要的内部身份结构。
#[derive(Serialize)]
struct ManagedMetadataIdentity<'a> {
    schema_version: u16,
    conversation_id: &'a str,
    expected_epoch: u64,
    expected_revision: u64,
    title: Option<&'a str>,
    summary: Option<&'a str>,
    compressed_before_id: Option<i64>,
}

/// 校验一组待导入消息是否规范：属于目标会话、尚未分配后端 ID、时间戳非空。
fn validate_managed_messages(conversation_id: &str, messages: &[StoredMessage]) -> Result<()> {
    if conversation_id.trim().is_empty() {
        return Err(projection_error(
            "managed conversation identity must not be empty",
        ));
    }
    for message in messages {
        if message.id.is_some()
            || message.conversation_id != conversation_id
            || message.created_at.trim().is_empty()
        {
            return Err(projection_error(
                "managed conversation import messages are not canonical",
            ));
        }
    }
    Ok(())
}

/// 校验 epoch 是否落在受支持的取值范围（1 至 `TRANSCRIPT_PROJECTION_EPOCH_MAX`）内。
fn validate_managed_epoch(epoch: u64) -> Result<()> {
    if epoch == 0 || epoch > TRANSCRIPT_PROJECTION_EPOCH_MAX {
        return Err(MemoryError::ProjectionEpochExhausted(format!(
            "epoch {epoch} is outside 1..={TRANSCRIPT_PROJECTION_EPOCH_MAX}"
        ))
        .into());
    }
    Ok(())
}

/// 校验转录投影的身份三元组（会话 ID、epoch、生成标识）是否均有效。
fn validate_projection_identity(
    conversation_id: &str,
    conversation_epoch: u64,
    generation_id: &str,
) -> Result<()> {
    if conversation_id.trim().is_empty() || generation_id.trim().is_empty() {
        return Err(projection_error(
            "transcript projection identities must not be empty",
        ));
    }
    if conversation_epoch == 0 || conversation_epoch > TRANSCRIPT_PROJECTION_EPOCH_MAX {
        return Err(projection_error(
            "transcript projection epoch is outside the supported range",
        ));
    }
    Ok(())
}

/// 计算转录投影批次载荷的摘要。
fn projection_payload_digest(
    schema_version: u16,
    conversation_id: &str,
    conversation_epoch: u64,
    generation_id: &str,
    items: &[TranscriptProjectionItem],
) -> Result<String> {
    digest_serialized(&ProjectionPayloadIdentity {
        schema_version,
        conversation_id,
        conversation_epoch,
        generation_id,
        items,
    })
}

/// 将任意可序列化的值编码为规范 JSON 后计算其 SHA-256 摘要（十六进制字符串）。
fn digest_serialized(value: &impl Serialize) -> Result<String> {
    let encoded = serde_json::to_vec(value).map_err(|error| {
        projection_error(format!(
            "failed to serialize transcript projection identity: {error}"
        ))
    })?;
    Ok(format!("{:x}", Sha256::digest(encoded)))
}

/// 构造一个表示序列化/摘要计算失败的 `ReactError`。
fn projection_error(message: impl Into<String>) -> crate::error::ReactError {
    MemoryError::SerializationError(message.into()).into()
}

/// 会话列表查询的过滤条件
#[derive(Debug, Clone, Default)]
pub struct ConversationFilter {
    /// 按用户 ID 过滤。
    pub user_id: Option<String>,
    /// 按 Agent 类型过滤。
    pub agent_type: Option<String>,
    /// 返回结果数量上限。
    pub limit: Option<usize>,
    /// 结果偏移量（用于分页）。
    pub offset: Option<usize>,
}

// ── Trait ─────────────────────────────────────────────────────────────────────

/// 会话持久化存储 trait
///
/// 提供会话与消息的增删改查操作，支持不同的存储后端实现。
pub trait ConversationStore: Send + Sync {
    /// 在不执行 I/O 的前提下，报告是否支持原子化转录投影。
    fn projection_capability(&self) -> ConversationProjectionCapability {
        ConversationProjectionCapability::Unsupported
    }

    /// 在不执行 I/O 的前提下，报告是否支持托管持久化截止时间。
    fn persistence_call_capability(&self) -> PersistenceCallCapability {
        PersistenceCallCapability::Unsupported
    }

    /// 创建一个新会话
    fn create_conversation<'a>(
        &'a self,
        conv: NewConversation,
    ) -> BoxFuture<'a, Result<Conversation>>;

    /// 获取会话详情
    fn get_conversation<'a>(
        &'a self,
        conversation_id: &'a str,
    ) -> BoxFuture<'a, Result<Option<Conversation>>>;

    /// 列出会话（支持过滤，按 updated_at 降序排列）
    fn list_conversations<'a>(
        &'a self,
        filter: ConversationFilter,
    ) -> BoxFuture<'a, Result<Vec<ConversationMeta>>>;

    /// 更新会话（标题 / 摘要 / 压缩边界）
    fn update_conversation<'a>(
        &'a self,
        conversation_id: &'a str,
        title: Option<&'a str>,
        summary: Option<&'a str>,
        compressed_before_id: Option<i64>,
    ) -> BoxFuture<'a, Result<()>>;

    /// 删除会话（级联删除其消息）
    fn delete_conversation<'a>(&'a self, conversation_id: &'a str) -> BoxFuture<'a, Result<()>>;

    /// 保存消息（upsert 模式：先删除旧消息，再插入新消息）
    fn save_messages<'a>(
        &'a self,
        conversation_id: &'a str,
        messages: &'a [StoredMessage],
    ) -> BoxFuture<'a, Result<()>>;

    /// 获取某个会话的全部消息（按 id 升序排列）
    fn get_messages<'a>(
        &'a self,
        conversation_id: &'a str,
    ) -> BoxFuture<'a, Result<Vec<StoredMessage>>>;

    /// 获取消息数量
    fn count_messages<'a>(&'a self, conversation_id: &'a str) -> BoxFuture<'a, Result<usize>>;

    /// 若会话不存在则创建，若已存在则返回既有记录。
    fn ensure_conversation<'a>(
        &'a self,
        conv: NewConversation,
    ) -> BoxFuture<'a, Result<Conversation>> {
        Box::pin(async move {
            let conversation_id = conv.conversation_id.clone();
            if let Some(existing) = self.get_conversation(&conversation_id).await? {
                Ok(existing)
            } else {
                self.create_conversation(conv).await
            }
        })
    }

    /// 获取或查询某个会话的原子投影 epoch。
    fn ensure_projection_epoch<'a>(
        &'a self,
        _request: EnsureConversationProjectionRequest,
    ) -> BoxFuture<'a, Result<ConversationProjectionEpochReceipt>> {
        Box::pin(async {
            Err(MemoryError::Unsupported(
                "atomic conversation projection epoch acquisition".to_string(),
            )
            .into())
        })
    }

    /// 带调用上下文（截止时间）的 [`ensure_projection_epoch`] 版本；
    /// 默认实现直接委托给不带上下文的版本。
    fn ensure_projection_epoch_with_context<'a>(
        &'a self,
        _context: PersistenceCallContext,
        request: EnsureConversationProjectionRequest,
    ) -> BoxFuture<'a, Result<ConversationProjectionEpochReceipt>> {
        self.ensure_projection_epoch(request)
    }

    /// 在不创建数据的前提下，读取托管会话的 epoch/墓碑状态。
    fn get_projection_authority<'a>(
        &'a self,
        _conversation_id: &'a str,
    ) -> BoxFuture<'a, Result<Option<ConversationProjectionAuthority>>> {
        Box::pin(async {
            Err(MemoryError::Unsupported("managed conversation authority query".to_string()).into())
        })
    }

    /// 带调用上下文（截止时间）的 [`get_projection_authority`] 版本；
    /// 默认实现直接委托给不带上下文的版本。
    fn get_projection_authority_with_context<'a>(
        &'a self,
        _context: PersistenceCallContext,
        conversation_id: &'a str,
    ) -> BoxFuture<'a, Result<Option<ConversationProjectionAuthority>>> {
        self.get_projection_authority(conversation_id)
    }

    /// 原子性地合并一个规范化的、限定在某个运行时生成范围内的转录批次。
    fn apply_transcript_projection<'a>(
        &'a self,
        _batch: TranscriptProjectionBatch,
    ) -> BoxFuture<'a, Result<TranscriptProjectionApplyReceipt>> {
        Box::pin(async {
            Err(MemoryError::Unsupported("atomic transcript projection apply".to_string()).into())
        })
    }

    /// 带调用上下文（截止时间）的 [`apply_transcript_projection`] 版本；
    /// 默认实现直接委托给不带上下文的版本。
    fn apply_transcript_projection_with_context<'a>(
        &'a self,
        _context: PersistenceCallContext,
        batch: TranscriptProjectionBatch,
    ) -> BoxFuture<'a, Result<TranscriptProjectionApplyReceipt>> {
        self.apply_transcript_projection(batch)
    }

    /// 仅当 epoch 与 revision 仍然匹配时，才替换一个托管转录。
    fn import_managed_messages<'a>(
        &'a self,
        _request: ManagedConversationImport,
    ) -> BoxFuture<'a, Result<TranscriptProjectionApplyReceipt>> {
        Box::pin(async {
            Err(MemoryError::Unsupported("managed conversation import".to_string()).into())
        })
    }

    /// 带调用上下文（截止时间）的 [`import_managed_messages`] 版本；
    /// 默认实现直接委托给不带上下文的版本。
    fn import_managed_messages_with_context<'a>(
        &'a self,
        _context: PersistenceCallContext,
        request: ManagedConversationImport,
    ) -> BoxFuture<'a, Result<TranscriptProjectionApplyReceipt>> {
        self.import_managed_messages(request)
    }

    /// 仅当 epoch 与 revision 仍然匹配时，才更新托管元数据。
    fn update_managed_conversation<'a>(
        &'a self,
        _request: ManagedConversationMetadataUpdate,
    ) -> BoxFuture<'a, Result<ManagedConversationMetadataUpdateReceipt>> {
        Box::pin(async {
            Err(MemoryError::Unsupported("managed conversation metadata update".to_string()).into())
        })
    }

    /// 带调用上下文（截止时间）的 [`update_managed_conversation`] 版本；
    /// 默认实现直接委托给不带上下文的版本。
    fn update_managed_conversation_with_context<'a>(
        &'a self,
        _context: PersistenceCallContext,
        request: ManagedConversationMetadataUpdate,
    ) -> BoxFuture<'a, Result<ManagedConversationMetadataUpdateReceipt>> {
        self.update_managed_conversation(request)
    }

    /// 在 epoch 拦截与稳定回执的保护下，删除一个托管的具体实例（incarnation）。
    fn delete_managed_conversation<'a>(
        &'a self,
        _request: ManagedConversationDelete,
    ) -> BoxFuture<'a, Result<ManagedConversationDeleteReceipt>> {
        Box::pin(async {
            Err(MemoryError::Unsupported("managed conversation deletion".to_string()).into())
        })
    }

    /// 带调用上下文（截止时间）的 [`delete_managed_conversation`] 版本；
    /// 默认实现直接委托给不带上下文的版本。
    fn delete_managed_conversation_with_context<'a>(
        &'a self,
        _context: PersistenceCallContext,
        request: ManagedConversationDelete,
    ) -> BoxFuture<'a, Result<ManagedConversationDeleteReceipt>> {
        self.delete_managed_conversation(request)
    }

    /** 按标题和消息内容搜索会话。

    默认实现会扫描所有会话及其消息（较朴素）。支持全文搜索的后端
    （例如 SQLite）应当重写该方法。
    */
    fn search_conversations<'a>(
        &'a self,
        query: &'a str,
        limit: usize,
    ) -> BoxFuture<'a, Result<Vec<ConversationMeta>>> {
        Box::pin(async move {
            let all = self
                .list_conversations(ConversationFilter {
                    limit: Some(100),
                    ..Default::default()
                })
                .await?;
            let query_lower = query.to_lowercase();
            let mut results = Vec::new();
            for meta in all {
                // 匹配标题
                let title_match = meta
                    .title
                    .as_ref()
                    .is_some_and(|t| t.to_lowercase().contains(&query_lower));
                // 匹配消息内容
                let content_match = if title_match {
                    true
                } else {
                    let messages = self.get_messages(&meta.conversation_id).await?;
                    messages.iter().any(|m| {
                        m.content
                            .as_ref()
                            .is_some_and(|c| c.to_lowercase().contains(&query_lower))
                    })
                };
                if title_match || content_match {
                    results.push(meta);
                    if results.len() >= limit {
                        break;
                    }
                }
            }
            Ok(results)
        })
    }
}

#[cfg(test)]
mod transcript_projection_contract_tests {
    use super::*;

    /// 仅用于测试的会话存储：不覆盖任何默认能力方法，用来验证默认值。
    struct DefaultCapabilityStore;

    /// 测试专用：返回一个总是失败的 `Unsupported` 结果 future。
    fn unsupported_store_call<'a, T>() -> BoxFuture<'a, Result<T>> {
        Box::pin(async {
            Err(MemoryError::Unsupported("test conversation store".to_string()).into())
        })
    }

    impl ConversationStore for DefaultCapabilityStore {
        fn create_conversation<'a>(
            &'a self,
            _conv: NewConversation,
        ) -> BoxFuture<'a, Result<Conversation>> {
            unsupported_store_call()
        }

        fn get_conversation<'a>(
            &'a self,
            _conversation_id: &'a str,
        ) -> BoxFuture<'a, Result<Option<Conversation>>> {
            unsupported_store_call()
        }

        fn list_conversations<'a>(
            &'a self,
            _filter: ConversationFilter,
        ) -> BoxFuture<'a, Result<Vec<ConversationMeta>>> {
            unsupported_store_call()
        }

        fn update_conversation<'a>(
            &'a self,
            _conversation_id: &'a str,
            _title: Option<&'a str>,
            _summary: Option<&'a str>,
            _compressed_before_id: Option<i64>,
        ) -> BoxFuture<'a, Result<()>> {
            unsupported_store_call()
        }

        fn delete_conversation<'a>(
            &'a self,
            _conversation_id: &'a str,
        ) -> BoxFuture<'a, Result<()>> {
            unsupported_store_call()
        }

        fn save_messages<'a>(
            &'a self,
            _conversation_id: &'a str,
            _messages: &'a [StoredMessage],
        ) -> BoxFuture<'a, Result<()>> {
            unsupported_store_call()
        }

        fn get_messages<'a>(
            &'a self,
            _conversation_id: &'a str,
        ) -> BoxFuture<'a, Result<Vec<StoredMessage>>> {
            unsupported_store_call()
        }

        fn count_messages<'a>(&'a self, _conversation_id: &'a str) -> BoxFuture<'a, Result<usize>> {
            unsupported_store_call()
        }
    }

    /// 测试专用：构造一条用于测试的示例消息。
    fn message(conversation_id: &str, created_at: &str) -> StoredMessage {
        StoredMessage {
            id: None,
            conversation_id: conversation_id.to_string(),
            role: "user".to_string(),
            content: Some("hello".to_string()),
            attachments_json: Some(r#"{"kind":"text"}"#.to_string()),
            tool_calls_json: None,
            tool_result_json: None,
            created_at: created_at.to_string(),
        }
    }

    #[test]
    fn persistence_call_capability_requires_explicit_opt_in() {
        assert_eq!(
            DefaultCapabilityStore.persistence_call_capability(),
            PersistenceCallCapability::Unsupported
        );
    }

    #[test]
    fn persistence_call_context_reports_future_budget_and_typed_expiration() -> Result<()> {
        let remaining = PersistenceCallContext {
            absolute_deadline_unix_ms: i64::MAX,
        }
        .remaining_duration()?;
        assert!(remaining > std::time::Duration::ZERO);

        let expired = PersistenceCallContext {
            absolute_deadline_unix_ms: i64::MIN,
        }
        .remaining_duration();
        assert!(matches!(expired, Err(MemoryError::DeadlineExceeded(_))));
        Ok(())
    }

    #[test]
    fn transcript_projection_batch_identity_binds_complete_prepared_effect() -> Result<()> {
        let first = TranscriptProjectionBatch::prepare(
            "conversation",
            7,
            "generation",
            3,
            vec![message("conversation", "2026-09-16T00:00:00Z")],
        )?;
        let repeated = TranscriptProjectionBatch::prepare(
            "conversation",
            7,
            "generation",
            3,
            vec![message("conversation", "2026-09-16T00:00:00Z")],
        )?;
        let changed_created_at = TranscriptProjectionBatch::prepare(
            "conversation",
            7,
            "generation",
            3,
            vec![message("conversation", "2026-09-16T00:00:01Z")],
        )?;

        assert_eq!(first.operation_id, repeated.operation_id);
        assert_eq!(first.payload_digest, repeated.payload_digest);
        assert_ne!(first.operation_id, changed_created_at.operation_id);
        assert_ne!(first.payload_digest, changed_created_at.payload_digest);
        Ok(())
    }

    #[test]
    fn transcript_projection_batch_rejects_cross_conversation_message() {
        let result = TranscriptProjectionBatch::prepare(
            "conversation-a",
            1,
            "generation",
            0,
            vec![message("conversation-b", "2026-09-16T00:00:00Z")],
        );

        assert!(result.is_err());
    }

    #[test]
    fn transcript_projection_batch_rejects_backend_incompatible_ordinals() {
        let result = TranscriptProjectionBatch::prepare(
            "conversation",
            1,
            "generation",
            (i64::MAX as u64).saturating_add(1),
            vec![message("conversation", "2026-09-16T00:00:00Z")],
        );
        assert!(result.is_err());
    }

    #[test]
    fn managed_delete_identity_rejects_epoch_rebinding() -> Result<()> {
        let mut request = ManagedConversationDelete::prepare("conversation", 4)?;
        request.expected_epoch = 5;

        assert!(request.validate().is_err());
        Ok(())
    }

    #[test]
    fn managed_import_identity_binds_complete_replacement() -> Result<()> {
        let first = ManagedConversationImport::prepare(
            "conversation",
            2,
            9,
            vec![message("conversation", "2026-09-16T00:00:00Z")],
        )?;
        let changed = ManagedConversationImport::prepare(
            "conversation",
            2,
            9,
            vec![message("conversation", "2026-09-16T00:00:01Z")],
        )?;

        assert_ne!(first.operation_id, changed.operation_id);
        assert_ne!(first.payload_digest, changed.payload_digest);
        first.validate()?;
        changed.validate()?;
        Ok(())
    }

    #[test]
    fn managed_metadata_identity_rejects_late_revision_rebinding() -> Result<()> {
        let mut request = ManagedConversationMetadataUpdate::prepare(
            "conversation",
            2,
            7,
            Some("title".to_string()),
            Some("summary".to_string()),
            Some(42),
        )?;
        request.expected_revision = 8;

        assert!(request.validate().is_err());
        Ok(())
    }
}
