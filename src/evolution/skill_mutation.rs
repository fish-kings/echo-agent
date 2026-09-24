//! Canonical durable mutation owner for Skill lifecycle resources.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::ffi::OsString;
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use chrono::{DateTime, Utc};
use echo_core::error::{MemoryError, ReactError};
use echo_core::utils::fs::FileDurability;
use echo_state::journal::file::FileEventJournal;
use echo_state::journal::{EventJournal, JournalDurabilityStatus, PreparedJournalBatch};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::audit::{
    ChangeEntry, ChangeEntryBuilder, ChangeFilter, ChangeLog, ChangeType, EntityType,
};
use super::curator::{Curator, CuratorState};
use super::security::{PromptInjectionDetector, SecretScanner};
use crate::error::Result;

/// ChangeLog 中用于记录 Skill 生命周期权威绑定关系的固定审计键。
const SKILL_AUTHORITY_AUDIT_KEY: &str = "__skill_lifecycle_authority_binding__";

/// 将 Skill 操作日志（journal）与某个 ChangeLog 权威实例绑定的标记。
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct SkillAuditBinding {
    /// 绑定的审计权威（authority）唯一标识。
    authority_id: String,
    /// 绑定时 ChangeLog 的持久化目的地身份标识。
    destination_identity: String,
}

/// Skill 生命周期变更的操作种类。
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SkillMutationKind {
    /// 起草一个新的 Skill。
    Draft,
    /// 将 Skill 提升到更高的生命周期阶段。
    Promote,
    /// 记录一次不改变生命周期阶段的使用（usage）触达。
    Touch,
    /// 弃用一个 Skill。
    Deprecate,
    /// 合并多个 Skill。
    Merge,
    /// 对已有 Skill 内容打补丁。
    Patch,
    /// 回滚到之前的 Skill 状态。
    Rollback,
}

/// 单个 Skill 资源文件在一次变更前后的快照。
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SkillFileMutation {
    /// 规范化（canonical）之后的文件路径。
    pub path: PathBuf,
    /// 变更前的文件内容；`None` 表示该文件此前不存在。
    pub before: Option<Vec<u8>>,
    /// 变更后的文件内容；`None` 表示该文件将被删除。
    pub after: Option<Vec<u8>>,
}

impl SkillFileMutation {
    /// 将给定路径规范化为用于比较与存储的绝对规范路径。
    pub fn canonical_path(path: impl AsRef<Path>) -> Result<PathBuf> {
        canonical_resource_path(path.as_ref())
    }

    /// 构造一条文件变更记录，路径会先被规范化。
    pub fn new(
        path: impl AsRef<Path>,
        before: Option<Vec<u8>>,
        after: Option<Vec<u8>>,
    ) -> Result<Self> {
        Ok(Self {
            path: Self::canonical_path(path)?,
            before,
            after,
        })
    }
}

/// 一次完整的 Skill 生命周期变更请求。
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SkillMutationRequest {
    /// 该请求的幂等（idempotent）唯一标识。
    pub request_id: String,
    /// 受影响的 Skill 实体键（通常是 Skill 名称）。
    pub entity_key: String,
    /// 本次变更的种类。
    pub kind: SkillMutationKind,
    /// 发起本次变更的原因说明。
    pub reason: String,
    /// 本次变更涉及的文件级别改动列表。
    pub files: Vec<SkillFileMutation>,
    /// 变更前预期的 Curator 状态，用于乐观并发校验。
    pub curator_before: CuratorState,
    /// 变更后期望达到的 Curator 状态。
    pub curator_after: CuratorState,
    /// 若本次请求是一次回滚，指向被回滚目标的血缘信息。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rollback_of: Option<SkillRollbackLineage>,
}

/// 回滚请求指向的目标批次血缘信息。
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SkillRollbackLineage {
    /// 被回滚目标所在的操作批次 ID。
    pub target_batch_id: String,
    /// 触发本次回滚的单个变更 ID（若适用）。
    pub target_change_id: Option<String>,
    /// 目标批次对应的全部审计变更 ID。
    pub target_change_ids: Vec<String>,
    /// 目标批次的生成号（generation）。
    pub target_generation: u64,
}

/// 对某次操作摘要（digest）的审批凭证。
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SkillApprovalArtifact {
    /// 该审批凭证的唯一标识。
    pub approval_id: String,
    /// 被审批操作的内容摘要，必须与预览阶段计算出的摘要一致。
    pub operation_digest: String,
    /// 审批人标识。
    pub approver: String,
    /// 审批完成的时间。
    pub approved_at: DateTime<Utc>,
}

impl SkillApprovalArtifact {
    /// 构造一条审批凭证。
    pub fn new(
        approval_id: impl Into<String>,
        operation_digest: impl Into<String>,
        approver: impl Into<String>,
        approved_at: DateTime<Utc>,
    ) -> Self {
        Self {
            approval_id: approval_id.into(),
            operation_digest: operation_digest.into(),
            approver: approver.into(),
            approved_at,
        }
    }
}

/// 一次变更请求在应用前的预览结果，供审批环节比对使用。
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SkillMutationPreview {
    /// 对应的请求 ID。
    pub request_id: String,
    /// 本次操作的内容摘要。
    pub operation_digest: String,
    /// 受影响的文件路径列表。
    pub affected_paths: Vec<PathBuf>,
    /// 受影响的 Skill 名称列表。
    pub affected_skills: Vec<String>,
}

/// 一次变更成功落地后返回的回执。
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SkillMutationReceipt {
    /// 对应的请求 ID。
    pub request_id: String,
    /// 本次变更落地所属的操作批次 ID。
    pub batch_id: String,
    /// 该批次在操作日志中的生成号。
    pub generation: u64,
    /// 本次变更写入 ChangeLog 的审计变更 ID 列表。
    pub change_ids: Vec<String>,
    /// 本次变更所使用的审批凭证 ID。
    pub approval_id: String,
}

/// 一次变更请求应用后的结果。
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SkillMutationOutcome {
    /// 变更已成功应用。
    Applied(SkillMutationReceipt),
    /// 该请求此前已被应用过（幂等重放）。
    AlreadyApplied(SkillMutationReceipt),
    /// 变更因与当前状态冲突而被拒绝。
    Conflict {
        /// 冲突原因说明。
        reason: String,
    },
    /// 由于历史记录不可用，无法判断请求状态。
    HistoryUnavailable {
        /// 不可用原因说明。
        reason: String,
    },
}

/// 回滚操作所指向的目标定位方式。
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SkillRollbackTarget {
    /// 按审计变更 ID 定位目标。
    ChangeId(String),
    /// 按操作批次 ID 定位目标。
    BatchId(String),
    /// 按宿主自有（host-owned）规则名定位目标。
    Rule(String),
}

/// 描述某个回滚目标是否受本权威支持。
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SkillRollbackSupport {
    /// 该目标受本权威直接支持。
    Supported,
    /// 该目标由宿主自行拥有并处理。
    HostOwned {
        /// 宿主侧的目标标识。
        target: String,
    },
}

/// 回滚预览的结果。
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SkillRollbackPreviewOutcome {
    /// 预览已就绪，可提交审批。
    Ready(SkillMutationPreview),
    /// 预览因冲突而无法生成。
    Conflict {
        /// 冲突原因说明。
        reason: String,
    },
    /// 历史记录不可用，无法生成预览。
    HistoryUnavailable {
        /// 不可用原因说明。
        reason: String,
    },
    /// 该回滚目标由宿主自行拥有。
    HostOwned {
        /// 宿主侧的目标标识。
        target: String,
    },
}

/// 在每次 Skill 变更成功落地后接收通知的观察者。
pub trait SkillMutationObserver: Send + Sync {
    /// 在一次变更成功应用后被调用。
    fn on_skill_mutation(&self, receipt: &SkillMutationReceipt);
}

/// 记录 Skill 使用情况的结果。
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SkillUsageOutcome {
    /// 使用记录已更新。
    Updated,
    /// 目标 Skill 未知（不存在）。
    UnknownSkill,
}

/**
 * 用于记录已有 Skill 使用情况的窄接口运行时句柄。
 * 它不能创建、提升或以其他方式改变生命周期状态。
 */
#[derive(Clone)]
pub struct SkillUsageHandle {
    /// 底层的 Skill 变更权威实例。
    authority: Arc<SkillMutationAuthority>,
}

impl SkillUsageHandle {
    /// 基于给定的变更权威构造一个使用记录句柄。
    pub fn new(authority: Arc<SkillMutationAuthority>) -> Self {
        Self { authority }
    }

    /// 记录一次对指定 Skill 的使用，内部会重试直至冲突消解或次数耗尽。
    pub async fn record_usage(&self, skill_name: &str) -> Result<SkillUsageOutcome> {
        let mut last_error = None;
        for _ in 0..3 {
            let before = self.authority.curator.load_state()?;
            let Some(_) = before.skills.get(skill_name) else {
                return Ok(SkillUsageOutcome::UnknownSkill);
            };
            let mut after = before.clone();
            let meta = after.skills.get_mut(skill_name).ok_or_else(|| {
                ReactError::Other("skill disappeared while planning usage mutation".into())
            })?;
            meta.last_used_at = Utc::now();
            let request = SkillMutationRequest {
                request_id: uuid::Uuid::new_v4().to_string(),
                entity_key: skill_name.to_owned(),
                kind: SkillMutationKind::Touch,
                reason: "runtime skill usage".into(),
                files: Vec::new(),
                curator_before: before,
                curator_after: after,
                rollback_of: None,
            };
            let preview = match self.authority.preview(&request) {
                Ok(preview) => preview,
                Err(error) => {
                    last_error = Some(error);
                    continue;
                }
            };
            let approval = SkillApprovalArtifact::new(
                uuid::Uuid::new_v4().to_string(),
                &preview.operation_digest,
                "framework:skill-usage",
                Utc::now(),
            );
            match self.authority.apply(request, approval).await {
                Ok(SkillMutationOutcome::Applied(_))
                | Ok(SkillMutationOutcome::AlreadyApplied(_)) => {
                    return Ok(SkillUsageOutcome::Updated);
                }
                Ok(SkillMutationOutcome::Conflict { reason }) => {
                    last_error = Some(ReactError::Other(reason));
                    continue;
                }
                Ok(outcome) => {
                    return Err(ReactError::Other(format!(
                        "skill usage mutation did not settle: {outcome:?}"
                    )));
                }
                Err(error) => return Err(error),
            }
        }
        Err(last_error
            .unwrap_or_else(|| ReactError::Other("skill usage mutation exhausted retries".into())))
    }
}

/// 一次已获批准并即将/已经落地的 Skill 操作批次。
#[derive(Clone, Debug, Serialize, Deserialize)]
struct SkillOperationBatch {
    /// 批次唯一 ID。
    id: String,
    /// 原始变更请求。
    request: SkillMutationRequest,
    /// 对应的审批凭证。
    approval: SkillApprovalArtifact,
    /// 本批次写入 ChangeLog 的审计条目。
    audits: Vec<ChangeEntry>,
    /// 若本批次是一次回滚，记录被回滚目标的血缘信息。
    #[serde(default)]
    rollback_of: Option<SkillRollbackLineage>,
}

/// 写入 Skill 操作日志的事件类型。
#[derive(Clone, Debug, Serialize, Deserialize)]
enum SkillOperationEvent {
    /// 日志已绑定到某个审计权威。
    Bound {
        /// 绑定的审计权威 ID。
        audit_authority_id: String,
    },
    /// 一个操作批次已准备（prepare）但尚未结算。
    Prepared(Box<SkillOperationBatch>),
    /// 指定 ID 的操作批次已结算（settle）。
    Settled {
        /// 已结算批次的 ID。
        id: String,
    },
}

/// 通过重放日志得到的单条历史操作条目。
#[derive(Clone)]
struct SkillHistoryItem {
    /// 该条目对应的操作批次。
    batch: SkillOperationBatch,
    /// 该批次是否已完成结算。
    settled: bool,
    /// 该批次在日志中的序号（generation）。
    generation: u64,
}

/// 重放操作日志后得到的完整历史视图。
struct SkillJournalHistory {
    /// 日志当前绑定的审计权威 ID（若已绑定）。
    audit_authority_id: Option<String>,
    /// 已重放出的全部历史操作条目。
    operations: Vec<SkillHistoryItem>,
}

/// Skill 操作的持久化事件日志（journal）。
struct SkillOperationJournal {
    /// 底层的文件事件日志。
    journal: Arc<FileEventJournal<SkillOperationEvent>>,
    /// 用于串行化写入的进程内互斥锁。
    serial: Arc<tokio::sync::Mutex<()>>,
}

impl SkillOperationJournal {
    /// 打开（或创建）位于指定路径的操作日志。
    fn open(path: &Path) -> Result<Self> {
        let journal = Arc::new(FileEventJournal::open(path, FileDurability::SyncData)?);
        static SERIALS: OnceLock<Mutex<HashMap<PathBuf, Weak<tokio::sync::Mutex<()>>>>> =
            OnceLock::new();
        let mut serials = SERIALS
            .get_or_init(|| Mutex::new(HashMap::new()))
            .lock()
            .map_err(|error| ReactError::Other(format!("skill journal lock poisoned: {error}")))?;
        let serial = serials
            .get(journal.path())
            .and_then(Weak::upgrade)
            .unwrap_or_else(|| {
                let created = Arc::new(tokio::sync::Mutex::new(()));
                serials.insert(journal.path().to_path_buf(), Arc::downgrade(&created));
                created
            });
        Ok(Self { journal, serial })
    }

    /// 将一个事件追加写入日志，并根据落盘状态确保持久性。
    fn append(&self, event: SkillOperationEvent) -> Result<()> {
        let receipt = self
            .journal
            .append_batch(
                PreparedJournalBatch::new(vec![event])
                    .map_err(|error| ReactError::Other(error.to_string()))?,
            )
            .map_err(|error| ReactError::Other(error.to_string()))?;
        match receipt.durability() {
            JournalDurabilityStatus::Confirmed => Ok(()),
            JournalDurabilityStatus::Degraded { .. } => self.journal.sync_data(),
            JournalDurabilityStatus::Unconfirmed => Err(ReactError::Other(
                "skill operation journal durability is unconfirmed".into(),
            )),
        }
    }

    /// 记录一个刚准备好、尚未结算的操作批次。
    fn prepare(&self, batch: SkillOperationBatch) -> Result<()> {
        self.append(SkillOperationEvent::Prepared(Box::new(batch)))
    }

    /// 记录指定批次已完成结算。
    fn settle(&self, id: &str) -> Result<()> {
        self.append(SkillOperationEvent::Settled { id: id.to_owned() })
    }

    /// 将本日志绑定到指定的审计权威 ID；重复绑定同一 ID 是幂等的。
    fn bind(&self, audit_authority_id: &str) -> Result<()> {
        if audit_authority_id.is_empty() {
            return Err(ReactError::Other(
                "skill audit authority identity must not be empty".into(),
            ));
        }
        let history = self.history_state()?;
        match history.audit_authority_id {
            Some(existing) if existing == audit_authority_id => Ok(()),
            Some(existing) => Err(ReactError::Other(format!(
                "skill journal is bound to audit authority {existing}, not {audit_authority_id}"
            ))),
            None if history.operations.is_empty() => self.append(SkillOperationEvent::Bound {
                audit_authority_id: audit_authority_id.to_owned(),
            }),
            None => Err(ReactError::Other(
                "skill journal has operations without an audit authority binding".into(),
            )),
        }
    }

    /// 返回已绑定日志的全部历史操作条目；若日志未绑定则报错。
    fn history(&self) -> Result<Vec<SkillHistoryItem>> {
        let history = self.history_state()?;
        if history.audit_authority_id.is_none() && !history.operations.is_empty() {
            return Err(ReactError::Other(
                "skill journal operation history is unbound".into(),
            ));
        }
        Ok(history.operations)
    }

    /// 从头重放底层文件日志，重建完整的历史状态。
    fn history_state(&self) -> Result<SkillJournalHistory> {
        self.journal.sync_data()?;
        let mut audit_authority_id = None;
        let mut pending = HashMap::<String, usize>::new();
        let mut known = HashSet::<String>::new();
        let mut history = Vec::<SkillHistoryItem>::new();
        let mut sequence = 0_u64;
        loop {
            let records = self.journal.replay_after(sequence, 512)?;
            if records.is_empty() {
                break;
            }
            for record in records {
                match record.event.as_ref() {
                    SkillOperationEvent::Bound {
                        audit_authority_id: observed,
                    } => {
                        if observed.is_empty()
                            || audit_authority_id.is_some()
                            || !history.is_empty()
                        {
                            return Err(ReactError::Other(
                                "skill journal has an invalid audit authority binding".into(),
                            ));
                        }
                        audit_authority_id = Some(observed.clone());
                    }
                    SkillOperationEvent::Prepared(batch) => {
                        if audit_authority_id.is_none() {
                            return Err(ReactError::Other(
                                "skill prepare precedes audit authority binding".into(),
                            ));
                        }
                        if !known.insert(batch.id.clone()) {
                            return Err(ReactError::Other(format!(
                                "duplicate skill operation batch {}",
                                batch.id
                            )));
                        }
                        let index = history.len();
                        pending.insert(batch.id.clone(), index);
                        history.push(SkillHistoryItem {
                            batch: batch.as_ref().clone(),
                            settled: false,
                            generation: record.sequence,
                        });
                    }
                    SkillOperationEvent::Settled { id } => {
                        let index = pending.remove(id).ok_or_else(|| {
                            ReactError::Other(format!("skill settlement {id} has no prepare"))
                        })?;
                        let item = history.get_mut(index).ok_or_else(|| {
                            ReactError::Other(format!("skill history index missing for {id}"))
                        })?;
                        item.settled = true;
                    }
                }
                sequence = record.sequence;
            }
        }
        Ok(SkillJournalHistory {
            audit_authority_id,
            operations: history,
        })
    }
}

/// Skill 生命周期资源的唯一（canonical）持久化变更权威。
///
/// 所有对 Skill 文件与 Curator 状态的变更都必须经由本权威完成，
/// 以保证操作日志、文件系统与 ChangeLog 三者之间的一致性。
pub struct SkillMutationAuthority {
    /// Skill 元数据（Curator 状态）的读写入口。
    curator: Curator,
    /// 持久化的操作日志。
    journal: SkillOperationJournal,
    /// 审计变更日志。
    change_log: Arc<dyn ChangeLog>,
    /// 本权威绑定的审计权威 ID。
    audit_authority_id: String,
    /// 用于检测内容中是否包含密钥的扫描器。
    scanner: SecretScanner,
    /// 用于检测提示词注入攻击的检测器。
    injector: PromptInjectionDetector,
    /// 可选的变更观察者，用于在变更落地后接收通知。
    observer: Option<Arc<dyn SkillMutationObserver>>,
    /// 仅测试使用：用于在指定阶段注入失败的开关。
    #[cfg(test)]
    fail_point: Mutex<Option<SkillMutationFailPoint>>,
    /// 仅测试使用：在文件投影完成后执行的钩子。
    #[cfg(test)]
    after_files_hook: Option<Arc<dyn Fn() -> Result<()> + Send + Sync>>,
}

/// 仅测试使用：标记应在应用流程的哪个阶段注入失败。
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SkillMutationFailPoint {
    /// 在批次已准备（prepare）之后失败。
    Prepared,
    /// 在文件投影完成之后失败。
    FilesProjected,
    /// 在 Curator 状态投影完成之后失败。
    CuratorProjected,
    /// 在审计记录写入完成之后失败。
    AuditRecorded,
}

impl SkillMutationAuthority {
    /// 打开（或首次创建）一个绑定到指定 Curator 与 ChangeLog 的变更权威。
    ///
    /// 会校验操作日志与 ChangeLog 中的审计权威标记是否一致；
    /// 若两者都为空，则会创建一条新的绑定标记并写入 ChangeLog。
    pub fn open(curator: Curator, change_log: Arc<dyn ChangeLog>) -> Result<Self> {
        let journal = SkillOperationJournal::open(&curator.skill_operation_journal_path())?;
        let _serial = journal.serial.try_lock().map_err(|_| {
            ReactError::Other("skill authority is busy during destination binding".into())
        })?;
        let history = journal.history_state()?;
        let destination_identity = change_log.durable_destination_identity()?.ok_or_else(|| {
            ReactError::from(MemoryError::Unsupported(
                "Skill mutation authority requires a durable ChangeLog destination identity".into(),
            ))
        })?;
        let audit_marker = Self::read_audit_binding(change_log.as_ref())?;
        if let Some(marker) = audit_marker.as_ref()
            && marker.destination_identity != destination_identity
        {
            return Err(ReactError::Other(format!(
                "Skill ChangeLog marker destination {:?} does not match actual destination {:?}",
                marker.destination_identity, destination_identity
            )));
        }
        let audit_authority_id = match (history.audit_authority_id, audit_marker) {
            (Some(journal_id), Some(binding)) if journal_id == binding.authority_id => journal_id,
            (Some(journal_id), Some(binding)) => {
                return Err(ReactError::Other(format!(
                    "skill journal authority {journal_id} does not match ChangeLog authority {}",
                    binding.authority_id
                )));
            }
            (Some(_), None) => {
                return Err(ReactError::Other(
                    "skill journal is bound but ChangeLog has no matching authority marker".into(),
                ));
            }
            (None, Some(binding)) if history.operations.is_empty() => {
                journal.bind(&binding.authority_id)?;
                binding.authority_id
            }
            (None, Some(_)) => {
                return Err(ReactError::Other(
                    "unbound skill journal already contains operations".into(),
                ));
            }
            (None, None) if history.operations.is_empty() && change_log.is_empty() => {
                let created = uuid::Uuid::new_v4().to_string();
                let binding = SkillAuditBinding {
                    authority_id: created.clone(),
                    destination_identity,
                };
                let marker = ChangeEntryBuilder::new(
                    EntityType::Skill,
                    SKILL_AUTHORITY_AUDIT_KEY,
                    ChangeType::Create,
                )
                .after(serde_json::to_value(&binding)?)
                .reason("bind Skill lifecycle journal to one ChangeLog authority")
                .trigger("skill_mutation_authority_binding")
                .build_with(format!("skill_authority_{created}"), Utc::now());
                change_log.record_idempotent(marker)?;
                if Self::read_audit_binding(change_log.as_ref())?.as_ref() != Some(&binding) {
                    return Err(ReactError::Other(
                        "Skill ChangeLog authority marker is not durably observable".into(),
                    ));
                }
                if change_log.durable_destination_identity()?.as_ref()
                    != Some(&binding.destination_identity)
                {
                    return Err(ReactError::Other(
                        "Skill ChangeLog destination changed during authority binding".into(),
                    ));
                }
                journal.bind(&created)?;
                created
            }
            (None, None) => {
                return Err(ReactError::Other(
                    "unbound Skill authority has existing journal operations or ChangeLog records"
                        .into(),
                ));
            }
        };
        drop(_serial);
        Ok(Self {
            curator,
            journal,
            change_log,
            audit_authority_id,
            scanner: SecretScanner::new(),
            injector: PromptInjectionDetector,
            observer: None,
            #[cfg(test)]
            fail_point: Mutex::new(None),
            #[cfg(test)]
            after_files_hook: None,
        })
    }

    /// 附加一个变更观察者，返回配置后的自身（构建器风格）。
    pub fn with_observer(mut self, observer: Arc<dyn SkillMutationObserver>) -> Self {
        self.observer = Some(observer);
        self
    }

    /// 仅测试使用：配置在指定阶段注入一次失败。
    #[cfg(test)]
    fn with_fail_point(self, point: SkillMutationFailPoint) -> Self {
        if let Ok(mut configured) = self.fail_point.lock() {
            *configured = Some(point);
        }
        self
    }

    /// 仅测试使用：配置在文件投影完成后执行的钩子。
    #[cfg(test)]
    fn with_after_files_hook(mut self, hook: Arc<dyn Fn() -> Result<()> + Send + Sync>) -> Self {
        self.after_files_hook = Some(hook);
        self
    }

    /// 仅测试使用：若当前失败点与给定阶段匹配，则消费该失败点并返回错误。
    #[cfg(test)]
    fn fail_if(&self, point: SkillMutationFailPoint) -> Result<()> {
        let mut configured = self.fail_point.lock().map_err(|error| {
            ReactError::Other(format!("skill failpoint lock poisoned: {error}"))
        })?;
        if configured.as_ref() == Some(&point) {
            *configured = None;
            return Err(ReactError::Other(format!(
                "injected skill mutation failure at {point:?}"
            )));
        }
        Ok(())
    }

    /// 返回底层的 Curator 引用。
    pub fn curator(&self) -> &Curator {
        &self.curator
    }

    /// 返回本权威绑定的审计权威 ID。
    pub fn audit_authority_id(&self) -> &str {
        &self.audit_authority_id
    }

    /// 从 ChangeLog 中读取当前的审计权威绑定标记（若存在）。
    fn read_audit_binding(change_log: &dyn ChangeLog) -> Result<Option<SkillAuditBinding>> {
        let markers = change_log
            .query(&ChangeFilter::new().with_key_prefix(SKILL_AUTHORITY_AUDIT_KEY))?
            .into_iter()
            .filter(|entry| {
                entry.entity_type == EntityType::Skill
                    && entry.entity_key == SKILL_AUTHORITY_AUDIT_KEY
            })
            .collect::<Vec<_>>();
        match markers.as_slice() {
            [] => Ok(None),
            [marker] if marker.change_type == ChangeType::Create => {
                let binding = marker
                    .after
                    .as_ref()
                    .cloned()
                    .ok_or_else(|| {
                        ReactError::Other("Skill ChangeLog authority marker is invalid".into())
                    })
                    .and_then(|value| {
                        serde_json::from_value::<SkillAuditBinding>(value).map_err(|error| {
                            ReactError::Other(format!(
                                "Skill ChangeLog authority marker is invalid: {error}"
                            ))
                        })
                    })?;
                if binding.authority_id.is_empty() || binding.destination_identity.is_empty() {
                    return Err(ReactError::Other(
                        "Skill ChangeLog authority marker is invalid".into(),
                    ));
                }
                Ok(Some(binding))
            }
            _ => Err(ReactError::Other(
                "Skill ChangeLog has duplicate or invalid authority markers".into(),
            )),
        }
    }

    /// 构造一个表示“该回滚目标由宿主自行拥有”的支持声明。
    pub fn rule_rollback_support(target: impl Into<String>) -> SkillRollbackSupport {
        SkillRollbackSupport::HostOwned {
            target: target.into(),
        }
    }

    /// 预览一次回滚操作，但不实际应用；用于生成待审批的摘要。
    pub async fn preview_rollback(
        &self,
        request_id: impl Into<String>,
        target: &SkillRollbackTarget,
    ) -> Result<SkillRollbackPreviewOutcome> {
        let _serial = self.journal.serial.lock().await;
        self.reconcile_locked()?;
        let request_id = request_id.into();
        if let SkillRollbackTarget::Rule(target) = target {
            return Ok(SkillRollbackPreviewOutcome::HostOwned {
                target: target.clone(),
            });
        }
        let history = self.journal.history()?;
        let Some(target_item) = resolve_target(target, &history, self.change_log.as_ref())? else {
            return Ok(SkillRollbackPreviewOutcome::HistoryUnavailable {
                reason: "rollback target is not present in retained skill history".into(),
            });
        };
        if !target_item.settled {
            return Ok(SkillRollbackPreviewOutcome::HistoryUnavailable {
                reason: "rollback target is not settled".into(),
            });
        }
        if let Some(reason) = non_tip_reason(target_item, &history) {
            return Ok(SkillRollbackPreviewOutcome::Conflict { reason });
        }
        let request = inverse_request(request_id, target_item, target_change_id(target));
        Ok(SkillRollbackPreviewOutcome::Ready(self.preview(&request)?))
    }

    /// 对指定目标执行一次回滚，需要提供已通过审批的凭证。
    pub async fn rollback(
        &self,
        request_id: impl Into<String>,
        target: SkillRollbackTarget,
        approval: SkillApprovalArtifact,
    ) -> Result<SkillMutationOutcome> {
        let _serial = self.journal.serial.lock().await;
        self.reconcile_locked()?;
        let request_id = request_id.into();
        let history = self.journal.history()?;
        if let Some(existing) = history
            .iter()
            .find(|item| item.batch.request.request_id == request_id)
        {
            let requested_target = match &target {
                SkillRollbackTarget::BatchId(batch_id) => Some(batch_id.clone()),
                SkillRollbackTarget::ChangeId(_) => {
                    resolve_target(&target, &history, self.change_log.as_ref())?
                        .map(|item| item.batch.id.clone())
                }
                SkillRollbackTarget::Rule(_) => None,
            };
            if existing
                .batch
                .rollback_of
                .as_ref()
                .map(|lineage| lineage.target_batch_id.as_str())
                == requested_target.as_deref()
            {
                return Ok(SkillMutationOutcome::AlreadyApplied(receipt(existing)));
            }
            return Ok(SkillMutationOutcome::Conflict {
                reason: "rollback request ID was reused for a different target".into(),
            });
        }
        let Some(target_item) = resolve_target(&target, &history, self.change_log.as_ref())? else {
            return match target {
                SkillRollbackTarget::Rule(target) => Ok(SkillMutationOutcome::Conflict {
                    reason: format!("rule rollback is host-owned: {target}"),
                }),
                _ => Ok(SkillMutationOutcome::HistoryUnavailable {
                    reason: "rollback target is not present in retained skill history".into(),
                }),
            };
        };
        if let Some(reason) = non_tip_reason(target_item, &history) {
            return Ok(SkillMutationOutcome::Conflict { reason });
        }
        self.apply_locked(
            inverse_request(request_id, target_item, target_change_id(&target)),
            approval,
        )
    }

    /// 计算某个请求内容的摘要（digest），供审批环节校验一致性使用。
    pub fn digest(request: &SkillMutationRequest) -> Result<String> {
        let bytes = serde_json::to_vec(request)?;
        Ok(hex_digest(&bytes))
    }

    /// 校验一个请求，并生成对应的预览（不产生任何持久化副作用）。
    pub fn preview(&self, request: &SkillMutationRequest) -> Result<SkillMutationPreview> {
        self.validate_request(request)?;
        self.validate_current(request)?;
        Ok(SkillMutationPreview {
            request_id: request.request_id.clone(),
            operation_digest: Self::digest(request)?,
            affected_paths: request.files.iter().map(|file| file.path.clone()).collect(),
            affected_skills: changed_skill_names(request),
        })
    }

    /// 应用一次已获审批的变更请求，加锁后委托给 `apply_locked` 完成实际落地。
    pub async fn apply(
        &self,
        request: SkillMutationRequest,
        approval: SkillApprovalArtifact,
    ) -> Result<SkillMutationOutcome> {
        let _serial = self.journal.serial.lock().await;
        self.apply_locked(request, approval)
    }

    /// 在已持有串行锁的前提下应用一次变更请求：先做幂等/冲突检测，再落地。
    fn apply_locked(
        &self,
        request: SkillMutationRequest,
        approval: SkillApprovalArtifact,
    ) -> Result<SkillMutationOutcome> {
        self.reconcile_locked()?;
        let history = self.journal.history()?;
        if let Some(existing) = history
            .iter()
            .find(|item| item.batch.request.request_id == request.request_id)
        {
            if existing.batch.request != request || existing.batch.approval != approval {
                return Ok(SkillMutationOutcome::Conflict {
                    reason: "request ID was reused with different content or approval".into(),
                });
            }
            return Ok(SkillMutationOutcome::AlreadyApplied(receipt(existing)));
        }
        if history.iter().any(|item| {
            item.batch.approval.approval_id == approval.approval_id
                && item.batch.request.request_id != request.request_id
        }) {
            return Ok(SkillMutationOutcome::Conflict {
                reason: "approval artifact was already consumed by another request".into(),
            });
        }
        let preview = match self.preview(&request) {
            Ok(preview) => preview,
            Err(error) => {
                return Ok(SkillMutationOutcome::Conflict {
                    reason: error.to_string(),
                });
            }
        };
        validate_approval(&approval, &preview)?;
        let batch_id = uuid::Uuid::new_v4().to_string();
        let audits = build_audits(&batch_id, &request, &approval);
        let rollback_of = request.rollback_of.clone();
        let batch = SkillOperationBatch {
            id: batch_id,
            request,
            approval,
            audits,
            rollback_of,
        };
        self.journal.prepare(batch.clone())?;
        #[cfg(test)]
        self.fail_if(SkillMutationFailPoint::Prepared)?;
        self.apply_batch(&batch)?;
        let item = self
            .journal
            .history()?
            .into_iter()
            .find(|item| item.batch.id == batch.id)
            .ok_or_else(|| ReactError::Other("skill receipt batch disappeared".into()))?;
        let receipt = receipt(&item);
        if let Some(observer) = &self.observer {
            observer.on_skill_mutation(&receipt);
        }
        Ok(SkillMutationOutcome::Applied(receipt))
    }

    /// 补齐日志中所有已准备但尚未结算的批次（例如进程崩溃恢复后调用）。
    pub fn reconcile(&self) -> Result<()> {
        let guard = self.journal.serial.try_lock().map_err(|_| {
            ReactError::Other("skill mutation is active; retry reconciliation".into())
        })?;
        let result = self.reconcile_locked();
        drop(guard);
        result
    }

    /// 在已持有串行锁的前提下补齐所有未结算的批次。
    fn reconcile_locked(&self) -> Result<()> {
        for item in self.journal.history()? {
            if !item.settled {
                self.apply_batch(&item.batch)?;
            }
        }
        Ok(())
    }

    /// 将一个已准备好的批次实际投影到文件系统、Curator 状态与 ChangeLog，并标记为已结算。
    fn apply_batch(&self, batch: &SkillOperationBatch) -> Result<()> {
        self.validate_current_or_after(&batch.request)?;
        for file in &batch.request.files {
            let current = read_optional(&file.path)?;
            if current == file.after {
                continue;
            }
            if current != file.before {
                return Err(ReactError::Other(format!(
                    "skill file {} changed after prepare",
                    file.path.display()
                )));
            }
            write_optional(&file.path, file.after.as_deref())?;
        }
        #[cfg(test)]
        {
            if let Some(hook) = &self.after_files_hook {
                hook()?;
            }
            self.fail_if(SkillMutationFailPoint::FilesProjected)?;
        }
        let affected_names = changed_skill_names(&batch.request);
        self.curator.replace_skill_entries_if_current(
            &batch.request.curator_before,
            &batch.request.curator_after,
            &affected_names,
        )?;
        #[cfg(test)]
        self.fail_if(SkillMutationFailPoint::CuratorProjected)?;
        for audit in &batch.audits {
            self.change_log.record_idempotent(audit.clone())?;
        }
        #[cfg(test)]
        self.fail_if(SkillMutationFailPoint::AuditRecorded)?;
        self.journal.settle(&batch.id)
    }

    /// 对请求本身做静态校验（结构完整性、路径规范化、安全扫描等），不涉及当前系统状态。
    fn validate_request(&self, request: &SkillMutationRequest) -> Result<()> {
        if request.request_id.is_empty()
            || request.entity_key.is_empty()
            || request.reason.is_empty()
        {
            return Err(ReactError::Other(
                "skill mutation identity, entity key, and reason are required".into(),
            ));
        }
        if request.entity_key == SKILL_AUTHORITY_AUDIT_KEY {
            return Err(ReactError::Other(
                "skill mutation entity key is reserved for authority binding".into(),
            ));
        }
        let mut paths = HashSet::new();
        for file in &request.files {
            let canonical = canonical_resource_path(&file.path)?;
            if canonical != file.path {
                return Err(ReactError::Other(format!(
                    "skill resource path is not canonical: {}",
                    file.path.display()
                )));
            }
            if !paths.insert(canonical) || file.before == file.after {
                return Err(ReactError::Other(
                    "skill mutation has duplicate or unchanged file projections".into(),
                ));
            }
            if let Some(after) = &file.after {
                let content = std::str::from_utf8(after).map_err(|error| {
                    ReactError::Other(format!(
                        "SKILL.md projection is not valid UTF-8 at {}: {error}",
                        file.path.display()
                    ))
                })?;
                if request.kind != SkillMutationKind::Rollback
                    && (self.injector.detect(content) || self.scanner.contains_secrets(content))
                {
                    return Err(ReactError::Other(
                        "skill security check rejected content before prepare".into(),
                    ));
                }
            }
        }
        if request.files.is_empty() && request.curator_before == request.curator_after {
            return Err(ReactError::Other("skill mutation has no effect".into()));
        }
        if request.curator_before.last_run_at != request.curator_after.last_run_at {
            return Err(ReactError::Other(
                "skill lifecycle request cannot mutate Curator run metadata".into(),
            ));
        }
        if request.kind != SkillMutationKind::Rollback {
            for name in changed_skill_names(request) {
                if let Some(path) = request
                    .curator_after
                    .skills
                    .get(&name)
                    .and_then(|meta| meta.path.as_ref())
                    && SkillFileMutation::canonical_path(path)? != *path
                {
                    return Err(ReactError::Other(format!(
                        "skill lifecycle {name} has a non-canonical path"
                    )));
                }
            }
        }
        Ok(())
    }

    /// 校验请求所引用的“变更前”状态是否与当前实际状态精确一致（乐观并发控制）。
    fn validate_current(&self, request: &SkillMutationRequest) -> Result<()> {
        for file in &request.files {
            if read_optional(&file.path)? != file.before {
                return Err(ReactError::Other(format!(
                    "skill file {} does not match reviewed bytes",
                    file.path.display()
                )));
            }
        }
        let current = self.curator.load_state()?;
        for name in changed_skill_names(request) {
            if current.skills.get(&name) != request.curator_before.skills.get(&name) {
                return Err(ReactError::Other(format!(
                    "skill lifecycle {name} does not match reviewed generation"
                )));
            }
        }
        Ok(())
    }

    /// 校验当前实际状态是幂等重放场景下允许的“变更前”或“变更后”之一（容忍重复应用）。
    fn validate_current_or_after(&self, request: &SkillMutationRequest) -> Result<()> {
        for file in &request.files {
            let current = read_optional(&file.path)?;
            if current != file.before && current != file.after {
                return Err(ReactError::Other(format!(
                    "skill file {} has an external edit",
                    file.path.display()
                )));
            }
        }
        let state = self.curator.load_state()?;
        for name in changed_skill_names(request) {
            let current = state.skills.get(&name);
            let before = request.curator_before.skills.get(&name);
            let after = request.curator_after.skills.get(&name);
            if current != before && current != after {
                return Err(ReactError::Other(format!(
                    "skill lifecycle {name} has an external edit"
                )));
            }
        }
        Ok(())
    }
}

/// 校验审批凭证是否完整，且其摘要与预览的操作摘要精确匹配。
fn validate_approval(
    approval: &SkillApprovalArtifact,
    preview: &SkillMutationPreview,
) -> Result<()> {
    if approval.approval_id.is_empty()
        || approval.approver.is_empty()
        || approval.operation_digest != preview.operation_digest
    {
        return Err(ReactError::Other(
            "skill approval is missing or does not match the exact mutation digest".into(),
        ));
    }
    Ok(())
}

/// 在历史记录中定位回滚目标对应的操作批次；`Rule` 目标始终返回 `None`。
fn resolve_target<'a>(
    target: &SkillRollbackTarget,
    history: &'a [SkillHistoryItem],
    change_log: &dyn ChangeLog,
) -> Result<Option<&'a SkillHistoryItem>> {
    match target {
        SkillRollbackTarget::BatchId(batch_id) => {
            Ok(history.iter().find(|item| item.batch.id == *batch_id))
        }
        SkillRollbackTarget::ChangeId(change_id) => {
            let audit_exists = change_log
                .query(&ChangeFilter::new())?
                .into_iter()
                .any(|entry| entry.change_id == *change_id);
            if !audit_exists {
                return Ok(None);
            }
            Ok(history.iter().find(|item| {
                item.batch
                    .audits
                    .iter()
                    .any(|entry| entry.change_id == *change_id)
            }))
        }
        SkillRollbackTarget::Rule(_) => Ok(None),
    }
}

/// 若目标批次不再是其所涉及文件/Skill 的最新（tip）生成，返回冲突原因；否则返回 `None`。
fn non_tip_reason(target: &SkillHistoryItem, history: &[SkillHistoryItem]) -> Option<String> {
    for file in &target.batch.request.files {
        let latest = history
            .iter()
            .filter(|item| {
                item.batch
                    .request
                    .files
                    .iter()
                    .any(|candidate| candidate.path == file.path)
            })
            .max_by_key(|item| item.generation);
        if latest.map(|item| item.batch.id.as_str()) != Some(target.batch.id.as_str()) {
            return Some(format!(
                "skill file {} is no longer at target generation",
                file.path.display()
            ));
        }
    }
    for name in changed_skill_names(&target.batch.request) {
        let latest = history
            .iter()
            .filter(|item| changed_skill_names(&item.batch.request).contains(&name))
            .max_by_key(|item| item.generation);
        if latest.map(|item| item.batch.id.as_str()) != Some(target.batch.id.as_str()) {
            return Some(format!(
                "skill lifecycle {name} is no longer at target generation"
            ));
        }
    }
    None
}

/// 若回滚目标是按审计变更 ID 定位的，返回该 ID；否则返回 `None`。
fn target_change_id(target: &SkillRollbackTarget) -> Option<String> {
    match target {
        SkillRollbackTarget::ChangeId(change_id) => Some(change_id.clone()),
        SkillRollbackTarget::BatchId(_) | SkillRollbackTarget::Rule(_) => None,
    }
}

/// 基于目标批次构造其反向（逆）变更请求，用于回滚操作。
fn inverse_request(
    request_id: String,
    target: &SkillHistoryItem,
    target_change_id: Option<String>,
) -> SkillMutationRequest {
    SkillMutationRequest {
        request_id,
        entity_key: target.batch.request.entity_key.clone(),
        kind: SkillMutationKind::Rollback,
        reason: format!("rollback skill batch {}", target.batch.id),
        files: target
            .batch
            .request
            .files
            .iter()
            .map(|file| SkillFileMutation {
                path: file.path.clone(),
                before: file.after.clone(),
                after: file.before.clone(),
            })
            .collect(),
        curator_before: target.batch.request.curator_after.clone(),
        curator_after: target.batch.request.curator_before.clone(),
        rollback_of: Some(SkillRollbackLineage {
            target_batch_id: target.batch.id.clone(),
            target_change_id,
            target_change_ids: target
                .batch
                .audits
                .iter()
                .map(|audit| audit.change_id.clone())
                .collect(),
            target_generation: target.generation,
        }),
    }
}

/// 计算请求在 `curator_before` 与 `curator_after` 之间发生变化的 Skill 名称，按字典序排序。
fn changed_skill_names(request: &SkillMutationRequest) -> Vec<String> {
    let mut names = HashSet::new();
    for name in request
        .curator_before
        .skills
        .keys()
        .chain(request.curator_after.skills.keys())
    {
        if request.curator_before.skills.get(name) != request.curator_after.skills.get(name) {
            names.insert(name.clone());
        }
    }
    let mut names = names.into_iter().collect::<Vec<_>>();
    names.sort();
    names
}

/// 为一次变更请求构造要写入 ChangeLog 的审计条目。
fn build_audits(
    batch_id: &str,
    request: &SkillMutationRequest,
    approval: &SkillApprovalArtifact,
) -> Vec<ChangeEntry> {
    let change_type = match request.kind {
        SkillMutationKind::Draft => ChangeType::Create,
        SkillMutationKind::Promote => ChangeType::Promote,
        SkillMutationKind::Touch | SkillMutationKind::Patch | SkillMutationKind::Rollback => {
            ChangeType::Update
        }
        SkillMutationKind::Deprecate => ChangeType::Demote,
        SkillMutationKind::Merge => ChangeType::Merge,
    };
    let operation = serde_json::json!({
        "mutation_kind": request.kind.clone(),
        "request_id": request.request_id.clone(),
        "batch_id": batch_id,
        "approval": approval,
        "rollback_of": request.rollback_of.clone(),
    });
    vec![
        ChangeEntryBuilder::new(EntityType::Skill, &request.entity_key, change_type)
            .before(serde_json::json!({
                "files": request.files.iter().map(file_summary_before).collect::<Vec<_>>(),
                "curator": curator_summary(request, true),
                "operation": operation.clone(),
            }))
            .after(serde_json::json!({
                "files": request.files.iter().map(file_summary_after).collect::<Vec<_>>(),
                "curator": curator_summary(request, false),
                "operation": operation,
            }))
            .reason(request.reason.clone())
            .trigger("skill_mutation_authority")
            .build_with(uuid::Uuid::new_v4().to_string(), Utc::now()),
    ]
}

/// 提取请求中受影响 Skill 在变更前或变更后的元数据快照，用于审计摘要。
fn curator_summary(
    request: &SkillMutationRequest,
    before: bool,
) -> BTreeMap<String, Option<super::curator::SkillMeta>> {
    let state = if before {
        &request.curator_before
    } else {
        &request.curator_after
    };
    changed_skill_names(request)
        .into_iter()
        .map(|name| {
            let meta = state.skills.get(&name).cloned();
            (name, meta)
        })
        .collect()
}

/// 生成文件变更“前”状态的摘要（用于审计记录）。
fn file_summary_before(file: &SkillFileMutation) -> serde_json::Value {
    file_summary(&file.path, file.before.as_deref())
}

/// 生成文件变更“后”状态的摘要（用于审计记录）。
fn file_summary_after(file: &SkillFileMutation) -> serde_json::Value {
    file_summary(&file.path, file.after.as_deref())
}

/// 为文件内容生成脱敏后的摘要 JSON（路径、哈希、长度、内容预览、是否含密钥）。
fn file_summary(path: &Path, bytes: Option<&[u8]>) -> serde_json::Value {
    let scanner = SecretScanner::new();
    let (summary, redacted) = bytes
        .map(|value| {
            let scanned = scanner.scan(&String::from_utf8_lossy(value));
            (
                Some(scanned.content.chars().take(160).collect::<String>()),
                scanned.has_secrets,
            )
        })
        .unwrap_or((None, false));
    serde_json::json!({
        "path": path,
        "sha256": bytes.map(hex_digest),
        "length": bytes.map(<[u8]>::len),
        "summary": summary,
        "redacted": redacted,
    })
}

/// 由一条历史条目构造对外返回的变更回执。
fn receipt(item: &SkillHistoryItem) -> SkillMutationReceipt {
    SkillMutationReceipt {
        request_id: item.batch.request.request_id.clone(),
        batch_id: item.batch.id.clone(),
        generation: item.generation,
        change_ids: item
            .batch
            .audits
            .iter()
            .map(|audit| audit.change_id.clone())
            .collect(),
        approval_id: item.batch.approval.approval_id.clone(),
    }
}

/// 读取文件内容；文件不存在时返回 `None` 而不是报错。
fn read_optional(path: &Path) -> Result<Option<Vec<u8>>> {
    match std::fs::read(path) {
        Ok(bytes) => Ok(Some(bytes)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error.into()),
    }
}

/// 将路径规范化为绝对规范路径，拒绝任何包含父目录穿越（`..`）的路径；
/// 若路径尚不存在，则规范化其存在的最近祖先目录后拼回剩余部分。
fn canonical_resource_path(path: &Path) -> Result<PathBuf> {
    if path
        .components()
        .any(|component| matches!(component, Component::ParentDir))
    {
        return Err(ReactError::Other(format!(
            "skill resource path contains parent traversal: {}",
            path.display()
        )));
    }
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()?.join(path)
    };
    let mut normalized = PathBuf::new();
    for component in absolute.components() {
        match component {
            Component::CurDir => {}
            Component::Prefix(prefix) => normalized.push(prefix.as_os_str()),
            Component::RootDir => normalized.push(Path::new(std::path::MAIN_SEPARATOR_STR)),
            Component::Normal(value) => normalized.push(value),
            Component::ParentDir => {
                return Err(ReactError::Other(
                    "normalized skill resource path contains parent traversal".into(),
                ));
            }
        }
    }
    if normalized.exists() {
        return std::fs::canonicalize(normalized).map_err(Into::into);
    }
    let mut probe = normalized.as_path();
    let mut suffix = Vec::<OsString>::new();
    while !probe.exists() {
        let name = probe.file_name().ok_or_else(|| {
            ReactError::Other(format!(
                "skill resource path has no existing ancestor: {}",
                normalized.display()
            ))
        })?;
        suffix.push(name.to_os_string());
        probe = probe.parent().ok_or_else(|| {
            ReactError::Other(format!(
                "skill resource path has no parent: {}",
                normalized.display()
            ))
        })?;
    }
    let mut canonical = std::fs::canonicalize(probe)?;
    for segment in suffix.into_iter().rev() {
        canonical.push(segment);
    }
    Ok(canonical)
}

/// 原子写入文件；`bytes` 为 `None` 时删除文件（文件不存在也视为成功）。
fn write_optional(path: &Path, bytes: Option<&[u8]>) -> Result<()> {
    match bytes {
        Some(bytes) => echo_core::utils::fs::atomic_write(path, bytes).map_err(Into::into),
        None => match std::fs::remove_file(path) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error.into()),
        },
    }
}

/// 计算字节内容的 SHA-256 十六进制摘要字符串。
fn hex_digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evolution::audit::JsonlChangeLog;
    use crate::evolution::audit::{ChangeRecordOutcome, EntityType};
    use crate::evolution::curator::{CuratorConfig, SkillLifecycle, SkillMeta};
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct FailOnceLog {
        inner: JsonlChangeLog,
        fail_next: std::sync::atomic::AtomicBool,
    }

    impl FailOnceLog {
        fn new(path: PathBuf) -> Result<Self> {
            Ok(Self {
                inner: JsonlChangeLog::new(path)?,
                fail_next: std::sync::atomic::AtomicBool::new(false),
            })
        }

        fn fail_next(&self) {
            self.fail_next.store(true, Ordering::SeqCst);
        }
    }

    impl ChangeLog for FailOnceLog {
        fn durable_destination_identity(&self) -> Result<Option<String>> {
            self.inner.durable_destination_identity()
        }

        fn record(&self, entry: ChangeEntry) -> Result<()> {
            self.record_idempotent(entry).map(|_| ())
        }

        fn record_idempotent(&self, entry: ChangeEntry) -> Result<ChangeRecordOutcome> {
            if self.fail_next.swap(false, Ordering::SeqCst) {
                return Err(ReactError::Other("injected audit failure".into()));
            }
            self.inner.record_idempotent(entry)
        }

        fn query(&self, filter: &ChangeFilter) -> Result<Vec<ChangeEntry>> {
            self.inner.query(filter)
        }

        fn latest_for(
            &self,
            entity_type: EntityType,
            entity_key: &str,
        ) -> Result<Option<ChangeEntry>> {
            self.inner.latest_for(entity_type, entity_key)
        }

        fn len(&self) -> usize {
            self.inner.len()
        }
    }

    struct UnsupportedIdentityLog {
        inner: JsonlChangeLog,
    }

    impl UnsupportedIdentityLog {
        fn new(path: PathBuf) -> Result<Self> {
            Ok(Self {
                inner: JsonlChangeLog::new(path)?,
            })
        }
    }

    impl ChangeLog for UnsupportedIdentityLog {
        fn record(&self, entry: ChangeEntry) -> Result<()> {
            self.inner.record(entry)
        }

        fn record_idempotent(&self, entry: ChangeEntry) -> Result<ChangeRecordOutcome> {
            self.inner.record_idempotent(entry)
        }

        fn query(&self, filter: &ChangeFilter) -> Result<Vec<ChangeEntry>> {
            self.inner.query(filter)
        }

        fn latest_for(
            &self,
            entity_type: EntityType,
            entity_key: &str,
        ) -> Result<Option<ChangeEntry>> {
            self.inner.latest_for(entity_type, entity_key)
        }

        fn len(&self) -> usize {
            self.inner.len()
        }
    }

    #[derive(Default)]
    struct CountingObserver(AtomicUsize);

    impl SkillMutationObserver for CountingObserver {
        fn on_skill_mutation(&self, _receipt: &SkillMutationReceipt) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    fn skill_meta(name: &str, path: &Path, lifecycle: SkillLifecycle) -> SkillMeta {
        let now = Utc::now();
        let canonical_path =
            SkillFileMutation::canonical_path(path).unwrap_or_else(|_| path.to_path_buf());
        SkillMeta {
            name: name.to_owned(),
            path: Some(canonical_path),
            lifecycle,
            created_at: now,
            last_used_at: now,
            last_modified_at: now,
            pinned: false,
            agent_created: true,
            superseded_by: None,
        }
    }

    fn approval(preview: &SkillMutationPreview, id: &str) -> SkillApprovalArtifact {
        SkillApprovalArtifact::new(id, &preview.operation_digest, "reviewer", Utc::now())
    }

    fn open_json_authority(
        curator: Curator,
        path: impl AsRef<Path>,
    ) -> Result<(SkillMutationAuthority, Arc<JsonlChangeLog>)> {
        let log = Arc::new(JsonlChangeLog::new(path.as_ref().to_path_buf())?);
        let authority = SkillMutationAuthority::open(curator, log.clone())?;
        Ok((authority, log))
    }

    #[tokio::test]
    async fn approved_skill_mutation_is_durable_idempotent_and_rollbackable() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let state_path = dir.path().join("curator.json");
        let skill_path = dir.path().join("skills/a/SKILL.md");
        let curator = Curator::new(CuratorConfig::default(), &state_path);
        let observer = Arc::new(CountingObserver::default());
        let (authority, _log) =
            open_json_authority(curator.clone(), dir.path().join("changes.jsonl"))?;
        let authority = authority.with_observer(observer.clone());
        let before = curator.load_state()?;
        let mut after = before.clone();
        after.skills.insert(
            "a".into(),
            skill_meta("a", &skill_path, SkillLifecycle::Draft),
        );
        let request = SkillMutationRequest {
            request_id: "draft-a".into(),
            entity_key: "a".into(),
            kind: SkillMutationKind::Draft,
            reason: "approved draft".into(),
            files: vec![SkillFileMutation::new(
                &skill_path,
                None,
                Some(b"---\nname: a\n---\nbody\n".to_vec()),
            )?],
            curator_before: before,
            curator_after: after,
            rollback_of: None,
        };
        let preview = authority.preview(&request)?;
        let artifact = approval(&preview, "approval-a");
        let applied = authority.apply(request.clone(), artifact.clone()).await?;
        let receipt = match applied {
            SkillMutationOutcome::Applied(receipt) => receipt,
            other => return Err(ReactError::Other(format!("unexpected outcome: {other:?}"))),
        };
        assert_eq!(std::fs::read(&skill_path)?, b"---\nname: a\n---\nbody\n");
        assert_eq!(
            curator
                .load_state()?
                .skills
                .get("a")
                .map(|meta| meta.lifecycle),
            Some(SkillLifecycle::Draft)
        );
        assert_eq!(observer.0.load(Ordering::SeqCst), 1);
        assert!(matches!(
            authority.apply(request, artifact).await?,
            SkillMutationOutcome::AlreadyApplied(_)
        ));
        assert_eq!(observer.0.load(Ordering::SeqCst), 1);

        let rollback_target = SkillRollbackTarget::BatchId(receipt.batch_id);
        let rollback_preview = authority
            .preview_rollback("rollback-a", &rollback_target)
            .await?;
        let SkillRollbackPreviewOutcome::Ready(rollback_preview) = rollback_preview else {
            return Err(ReactError::Other("rollback preview was not ready".into()));
        };
        let rollback_approval = approval(&rollback_preview, "approval-rollback-a");
        let rollback = authority
            .rollback(
                "rollback-a",
                rollback_target.clone(),
                rollback_approval.clone(),
            )
            .await?;
        let rollback_receipt = match rollback {
            SkillMutationOutcome::Applied(receipt) => receipt,
            other => return Err(ReactError::Other(format!("unexpected rollback: {other:?}"))),
        };
        assert!(!skill_path.exists());
        assert!(!curator.load_state()?.skills.contains_key("a"));
        assert_eq!(observer.0.load(Ordering::SeqCst), 2);
        assert!(matches!(
            authority
                .rollback("rollback-a", rollback_target, rollback_approval)
                .await?,
            SkillMutationOutcome::AlreadyApplied(_)
        ));

        let redo_target = SkillRollbackTarget::BatchId(rollback_receipt.batch_id);
        let SkillRollbackPreviewOutcome::Ready(redo_preview) =
            authority.preview_rollback("redo-a", &redo_target).await?
        else {
            return Err(ReactError::Other(
                "rollback-of-rollback was not ready".into(),
            ));
        };
        assert!(matches!(
            authority
                .rollback(
                    "rollback-a",
                    redo_target.clone(),
                    approval(&redo_preview, "wrong-target-approval"),
                )
                .await?,
            SkillMutationOutcome::Conflict { .. }
        ));
        assert!(matches!(
            authority
                .rollback(
                    "redo-a",
                    redo_target,
                    approval(&redo_preview, "approval-redo-a"),
                )
                .await?,
            SkillMutationOutcome::Applied(_)
        ));
        assert_eq!(std::fs::read(&skill_path)?, b"---\nname: a\n---\nbody\n");
        Ok(())
    }

    #[tokio::test]
    async fn approval_mismatch_replay_and_external_edit_fail_closed() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let curator = Curator::new(CuratorConfig::default(), dir.path().join("curator.json"));
        let (authority, _log) =
            open_json_authority(curator.clone(), dir.path().join("changes.jsonl"))?;
        let path = dir.path().join("SKILL.md");
        std::fs::write(&path, "old")?;
        let state = curator.load_state()?;
        let request = SkillMutationRequest {
            request_id: "patch-a".into(),
            entity_key: "a".into(),
            kind: SkillMutationKind::Patch,
            reason: "patch".into(),
            files: vec![SkillFileMutation::new(
                &path,
                Some(b"old".to_vec()),
                Some(b"new".to_vec()),
            )?],
            curator_before: state.clone(),
            curator_after: state,
            rollback_of: None,
        };
        let preview = authority.preview(&request)?;
        let wrong = SkillApprovalArtifact::new("approval", "wrong", "reviewer", Utc::now());
        assert!(authority.apply(request.clone(), wrong).await.is_err());
        assert_eq!(std::fs::read(&path)?, b"old");
        std::fs::write(&path, "external")?;
        assert!(matches!(
            authority
                .apply(request, approval(&preview, "approval"))
                .await?,
            SkillMutationOutcome::Conflict { .. }
        ));
        assert_eq!(std::fs::read(&path)?, b"external");
        assert!(matches!(
            SkillMutationAuthority::rule_rollback_support("rules/policy"),
            SkillRollbackSupport::HostOwned { .. }
        ));
        Ok(())
    }

    #[test]
    fn security_rejection_happens_before_prepare() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let curator = Curator::new(CuratorConfig::default(), dir.path().join("curator.json"));
        let (authority, _log) =
            open_json_authority(curator.clone(), dir.path().join("changes.jsonl"))?;
        let state = curator.load_state()?;
        let request = SkillMutationRequest {
            request_id: "secret".into(),
            entity_key: "secret".into(),
            kind: SkillMutationKind::Patch,
            reason: "secret rejection".into(),
            files: vec![SkillFileMutation::new(
                dir.path().join("SKILL.md"),
                None,
                Some(b"API_KEY='abcdefghijklmnop'".to_vec()),
            )?],
            curator_before: state.clone(),
            curator_after: state,
            rollback_of: None,
        };
        assert!(authority.preview(&request).is_err());
        assert!(authority.journal.history()?.is_empty());
        Ok(())
    }

    #[tokio::test]
    async fn reserved_authority_entity_key_cannot_poison_marker_or_journal() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let state_path = dir.path().join("curator.json");
        let skill_path = dir.path().join("malicious/SKILL.md");
        let log_path = dir.path().join("changes.jsonl");
        let curator = Curator::new(CuratorConfig::default(), &state_path);
        let (authority, log) = open_json_authority(curator.clone(), &log_path)?;
        let state = curator.load_state()?;
        let request = SkillMutationRequest {
            request_id: "reserved-entity-key".into(),
            entity_key: SKILL_AUTHORITY_AUDIT_KEY.into(),
            kind: SkillMutationKind::Patch,
            reason: "attempt to collide with authority marker".into(),
            files: vec![SkillFileMutation::new(
                &skill_path,
                None,
                Some(b"---\nname: malicious\n---\nbody\n".to_vec()),
            )?],
            curator_before: state.clone(),
            curator_after: state,
            rollback_of: None,
        };
        let approval = SkillApprovalArtifact::new(
            "reserved-key-approval",
            SkillMutationAuthority::digest(&request)?,
            "reviewer",
            Utc::now(),
        );

        assert!(matches!(
            authority.apply(request, approval).await?,
            SkillMutationOutcome::Conflict { .. }
        ));
        assert!(!skill_path.exists());
        assert!(authority.journal.history()?.is_empty());
        assert_eq!(log.len(), 1);
        drop(authority);

        let reopened = SkillMutationAuthority::open(curator, log.clone())?;
        assert!(reopened.journal.history()?.is_empty());
        let markers = log
            .query(&ChangeFilter::new().with_key_prefix(SKILL_AUTHORITY_AUDIT_KEY))?
            .into_iter()
            .filter(|entry| entry.entity_key == SKILL_AUTHORITY_AUDIT_KEY)
            .count();
        assert_eq!(markers, 1);
        assert_eq!(log.len(), 1);
        Ok(())
    }

    #[test]
    fn path_alias_and_invalid_utf8_requests_fail_before_prepare() -> Result<()> {
        let dir = tempfile::tempdir()?;
        assert!(
            SkillFileMutation::new(
                dir.path().join("nested/../SKILL.md"),
                None,
                Some(b"valid".to_vec()),
            )
            .is_err()
        );
        let curator = Curator::new(CuratorConfig::default(), dir.path().join("curator.json"));
        let (authority, _log) =
            open_json_authority(curator.clone(), dir.path().join("changes.jsonl"))?;
        let state = curator.load_state()?;
        let invalid = SkillMutationRequest {
            request_id: "invalid-utf8".into(),
            entity_key: "invalid".into(),
            kind: SkillMutationKind::Patch,
            reason: "invalid utf8".into(),
            files: vec![SkillFileMutation::new(
                dir.path().join("SKILL.md"),
                None,
                Some(vec![0xff, 0xfe]),
            )?],
            curator_before: state.clone(),
            curator_after: state,
            rollback_of: None,
        };
        assert!(authority.preview(&invalid).is_err());
        let state = curator.load_state()?;
        let relative = SkillMutationRequest {
            request_id: "relative".into(),
            entity_key: "relative".into(),
            kind: SkillMutationKind::Patch,
            reason: "relative".into(),
            files: vec![SkillFileMutation {
                path: PathBuf::from("SKILL.md"),
                before: None,
                after: Some(b"valid".to_vec()),
            }],
            curator_before: state.clone(),
            curator_after: state,
            rollback_of: None,
        };
        assert!(authority.preview(&relative).is_err());

        #[cfg(unix)]
        {
            use std::os::unix::fs::symlink;
            let real = dir.path().join("real.md");
            let alias = dir.path().join("alias.md");
            std::fs::write(&real, "old")?;
            symlink(&real, &alias)?;
            let first =
                SkillFileMutation::new(&real, Some(b"old".to_vec()), Some(b"one".to_vec()))?;
            let second =
                SkillFileMutation::new(&alias, Some(b"old".to_vec()), Some(b"two".to_vec()))?;
            assert_eq!(first.path, second.path);
            let state = curator.load_state()?;
            let duplicate = SkillMutationRequest {
                request_id: "alias".into(),
                entity_key: "alias".into(),
                kind: SkillMutationKind::Patch,
                reason: "alias".into(),
                files: vec![first, second],
                curator_before: state.clone(),
                curator_after: state,
                rollback_of: None,
            };
            assert!(authority.preview(&duplicate).is_err());
        }
        assert!(authority.journal.history()?.is_empty());
        Ok(())
    }

    #[tokio::test]
    async fn secret_removal_audit_is_redacted_but_private_rollback_is_exact() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("SKILL.md");
        let secret = "API_KEY='abcdefghijklmnop'\n";
        std::fs::write(&path, secret)?;
        let curator = Curator::new(CuratorConfig::default(), dir.path().join("curator.json"));
        let log_path = dir.path().join("changes.jsonl");
        let (authority, _log) = open_json_authority(curator.clone(), &log_path)?;
        let state = curator.load_state()?;
        let request = SkillMutationRequest {
            request_id: "remove-secret".into(),
            entity_key: "secret".into(),
            kind: SkillMutationKind::Patch,
            reason: "remove secret".into(),
            files: vec![SkillFileMutation::new(
                &path,
                Some(secret.as_bytes().to_vec()),
                Some(b"clean\n".to_vec()),
            )?],
            curator_before: state.clone(),
            curator_after: state,
            rollback_of: None,
        };
        let preview = authority.preview(&request)?;
        let receipt = match authority
            .apply(request, approval(&preview, "remove-secret-approval"))
            .await?
        {
            SkillMutationOutcome::Applied(receipt) => receipt,
            other => return Err(ReactError::Other(format!("unexpected outcome: {other:?}"))),
        };
        let audit = std::fs::read_to_string(&log_path)?;
        assert!(!audit.contains("abcdefghijklmnop"));
        assert!(audit.contains("[REDACTED"));
        let target = SkillRollbackTarget::BatchId(receipt.batch_id);
        let SkillRollbackPreviewOutcome::Ready(rollback_preview) = authority
            .preview_rollback("restore-secret", &target)
            .await?
        else {
            return Err(ReactError::Other("secret rollback was not ready".into()));
        };
        authority
            .rollback(
                "restore-secret",
                target,
                approval(&rollback_preview, "restore-secret-approval"),
            )
            .await?;
        assert_eq!(std::fs::read_to_string(&path)?, secret);
        assert!(!std::fs::read_to_string(log_path)?.contains("abcdefghijklmnop"));
        Ok(())
    }

    #[tokio::test]
    async fn business_audit_exposes_approval_and_rollback_lineage_without_payload() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let curator = Curator::new(CuratorConfig::default(), dir.path().join("curator.json"));
        let log = Arc::new(JsonlChangeLog::new(dir.path().join("changes.jsonl"))?);
        let authority = SkillMutationAuthority::open(curator.clone(), log.clone())?;
        let mut receipts = Vec::new();
        for (index, kind) in [
            SkillMutationKind::Promote,
            SkillMutationKind::Merge,
            SkillMutationKind::Patch,
        ]
        .into_iter()
        .enumerate()
        {
            let key = format!("audit-{index}");
            let path = dir.path().join(format!("{key}.md"));
            std::fs::write(&path, "before")?;
            let before = curator.load_state()?;
            let mut after = before.clone();
            if kind == SkillMutationKind::Promote {
                after
                    .skills
                    .insert(key.clone(), skill_meta(&key, &path, SkillLifecycle::Active));
            }
            let request = SkillMutationRequest {
                request_id: format!("request-{index}"),
                entity_key: key.clone(),
                kind: kind.clone(),
                reason: format!("audit {kind:?}"),
                files: if kind == SkillMutationKind::Promote {
                    Vec::new()
                } else {
                    vec![SkillFileMutation::new(
                        &path,
                        Some(b"before".to_vec()),
                        Some(format!("after-{index}").into_bytes()),
                    )?]
                },
                curator_before: before,
                curator_after: after,
                rollback_of: None,
            };
            let preview = authority.preview(&request)?;
            let approval_id = format!("approval-{index}");
            let receipt = match authority
                .apply(request, approval(&preview, &approval_id))
                .await?
            {
                SkillMutationOutcome::Applied(receipt) => receipt,
                other => return Err(ReactError::Other(format!("unexpected outcome: {other:?}"))),
            };
            assert_eq!(receipt.approval_id, approval_id);
            let audit = log
                .latest_for(EntityType::Skill, &key)?
                .ok_or_else(|| ReactError::Other("missing business audit".into()))?;
            let operation = audit
                .after
                .as_ref()
                .and_then(|value| value.get("operation"))
                .ok_or_else(|| ReactError::Other("missing operation envelope".into()))?;
            assert_eq!(
                operation
                    .get("mutation_kind")
                    .and_then(serde_json::Value::as_str),
                Some(match kind {
                    SkillMutationKind::Promote => "promote",
                    SkillMutationKind::Merge => "merge",
                    SkillMutationKind::Patch => "patch",
                    _ => return Err(ReactError::Other("unexpected test mutation kind".into())),
                })
            );
            assert_eq!(
                operation
                    .get("approval")
                    .and_then(|value| value.get("approval_id"))
                    .and_then(serde_json::Value::as_str),
                Some(approval_id.as_str())
            );
            assert_eq!(
                operation
                    .get("approval")
                    .and_then(|value| value.get("operation_digest"))
                    .and_then(serde_json::Value::as_str),
                Some(preview.operation_digest.as_str())
            );
            assert_eq!(
                operation
                    .get("approval")
                    .and_then(|value| value.get("approver"))
                    .and_then(serde_json::Value::as_str),
                Some("reviewer")
            );
            assert!(
                operation
                    .get("approval")
                    .and_then(|value| value.get("approved_at"))
                    .and_then(serde_json::Value::as_str)
                    .is_some()
            );
            assert!(operation.get("request_id").is_some());
            assert!(operation.get("batch_id").is_some());
            receipts.push((key, receipt));
        }

        let (patch_key, patch_receipt) = receipts
            .pop()
            .ok_or_else(|| ReactError::Other("missing patch receipt".into()))?;
        let target_change_id = patch_receipt
            .change_ids
            .first()
            .cloned()
            .ok_or_else(|| ReactError::Other("missing patch change id".into()))?;
        let target = SkillRollbackTarget::ChangeId(target_change_id.clone());
        let SkillRollbackPreviewOutcome::Ready(preview) = authority
            .preview_rollback("audit-rollback", &target)
            .await?
        else {
            return Err(ReactError::Other("audit rollback was not ready".into()));
        };
        authority
            .rollback(
                "audit-rollback",
                target,
                approval(&preview, "audit-rollback-approval"),
            )
            .await?;
        let rollback_audit = log
            .latest_for(EntityType::Skill, &patch_key)?
            .ok_or_else(|| ReactError::Other("missing rollback business audit".into()))?;
        let lineage = rollback_audit
            .after
            .as_ref()
            .and_then(|value| value.get("operation"))
            .and_then(|value| value.get("rollback_of"))
            .ok_or_else(|| ReactError::Other("missing rollback lineage".into()))?;
        assert_eq!(
            lineage
                .get("target_batch_id")
                .and_then(serde_json::Value::as_str),
            Some(patch_receipt.batch_id.as_str())
        );
        assert_eq!(
            lineage
                .get("target_change_id")
                .and_then(serde_json::Value::as_str),
            Some(target_change_id.as_str())
        );
        assert_eq!(
            lineage
                .get("target_generation")
                .and_then(serde_json::Value::as_u64),
            Some(patch_receipt.generation)
        );
        Ok(())
    }

    #[tokio::test]
    async fn audit_failure_reconciles_exact_projection_once_after_restart() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let state_path = dir.path().join("curator.json");
        let skill_path = dir.path().join("SKILL.md");
        let curator = Curator::new(CuratorConfig::default(), &state_path);
        let state = curator.load_state()?;
        let request = SkillMutationRequest {
            request_id: "crash-request".into(),
            entity_key: "crash-skill".into(),
            kind: SkillMutationKind::Patch,
            reason: "crash recovery".into(),
            files: vec![SkillFileMutation::new(
                &skill_path,
                None,
                Some(b"durable skill".to_vec()),
            )?],
            curator_before: state.clone(),
            curator_after: state,
            rollback_of: None,
        };
        {
            let log = Arc::new(FailOnceLog::new(dir.path().join("changes-a.jsonl"))?);
            let authority = SkillMutationAuthority::open(curator.clone(), log.clone())?;
            let preview = authority.preview(&request)?;
            log.fail_next();
            assert!(
                authority
                    .apply(request, approval(&preview, "crash-approval"))
                    .await
                    .is_err()
            );
            assert_eq!(std::fs::read(&skill_path)?, b"durable skill");
        }
        let log_b_path = dir.path().join("changes-b.jsonl");
        let log_b = Arc::new(JsonlChangeLog::new(log_b_path.clone())?);
        assert!(SkillMutationAuthority::open(curator.clone(), log_b).is_err());
        assert_eq!(JsonlChangeLog::new(log_b_path)?.len(), 0);

        let log_path = dir.path().join("changes-a.jsonl");
        let (authority, _log) = open_json_authority(curator, &log_path)?;
        authority.reconcile()?;
        assert_eq!(std::fs::read(&skill_path)?, b"durable skill");
        assert_eq!(JsonlChangeLog::new(log_path)?.len(), 2);
        Ok(())
    }

    #[tokio::test]
    async fn every_prepared_stage_retries_to_one_settled_receipt() -> Result<()> {
        for point in [
            SkillMutationFailPoint::Prepared,
            SkillMutationFailPoint::FilesProjected,
            SkillMutationFailPoint::CuratorProjected,
            SkillMutationFailPoint::AuditRecorded,
        ] {
            let dir = tempfile::tempdir()?;
            let path = dir.path().join("SKILL.md");
            let curator = Curator::new(CuratorConfig::default(), dir.path().join("curator.json"));
            let log = Arc::new(JsonlChangeLog::new(dir.path().join("changes.jsonl"))?);
            let before = curator.load_state()?;
            let mut after = before.clone();
            after.skills.insert(
                "stage".into(),
                skill_meta("stage", &path, SkillLifecycle::Draft),
            );
            let request = SkillMutationRequest {
                request_id: "stage-request".into(),
                entity_key: "stage".into(),
                kind: SkillMutationKind::Draft,
                reason: "stage recovery".into(),
                files: vec![SkillFileMutation::new(
                    &path,
                    None,
                    Some(b"stage-content".to_vec()),
                )?],
                curator_before: before,
                curator_after: after,
                rollback_of: None,
            };
            let artifact;
            {
                let authority = SkillMutationAuthority::open(curator.clone(), log.clone())?
                    .with_fail_point(point);
                let preview = authority.preview(&request)?;
                artifact = approval(&preview, "stage-approval");
                assert!(
                    authority
                        .apply(request.clone(), artifact.clone())
                        .await
                        .is_err()
                );
            }
            let authority = SkillMutationAuthority::open(curator.clone(), log.clone())?;
            assert!(matches!(
                authority.apply(request, artifact).await?,
                SkillMutationOutcome::AlreadyApplied(_)
            ));
            assert_eq!(std::fs::read(&path)?, b"stage-content");
            assert_eq!(
                curator
                    .load_state()?
                    .skills
                    .get("stage")
                    .map(|meta| meta.lifecycle),
                Some(SkillLifecycle::Draft)
            );
            assert_eq!(log.len(), 2);
        }
        Ok(())
    }

    #[tokio::test]
    async fn unrelated_candidate_insert_between_file_and_curator_is_preserved() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("SKILL.md");
        let curator = Curator::new(CuratorConfig::default(), dir.path().join("curator.json"));
        let log = Arc::new(JsonlChangeLog::new(dir.path().join("changes.jsonl"))?);
        let before = curator.load_state()?;
        let mut after = before.clone();
        after.skills.insert(
            "skill-a".into(),
            skill_meta("skill-a", &path, SkillLifecycle::Draft),
        );
        let request = SkillMutationRequest {
            request_id: "skill-a-draft".into(),
            entity_key: "skill-a".into(),
            kind: SkillMutationKind::Draft,
            reason: "draft with candidate interleave".into(),
            files: vec![SkillFileMutation::new(
                &path,
                None,
                Some(b"skill-a".to_vec()),
            )?],
            curator_before: before,
            curator_after: after,
            rollback_of: None,
        };
        let hook_curator = curator.clone();
        let authority = SkillMutationAuthority::open(curator.clone(), log)?.with_after_files_hook(
            Arc::new(move || {
                hook_curator.register_candidate_with_authority("candidate-b", "candidate-authority")
            }),
        );
        let preview = authority.preview(&request)?;
        authority
            .apply(request, approval(&preview, "interleave-approval"))
            .await?;
        let state = curator.load_state()?;
        assert!(state.skills.contains_key("skill-a"));
        assert!(state.skills.contains_key("candidate-b"));
        Ok(())
    }

    #[tokio::test]
    async fn unsettled_inverse_retry_reconciles_before_already_applied() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("SKILL.md");
        std::fs::write(&path, "before")?;
        let curator = Curator::new(CuratorConfig::default(), dir.path().join("curator.json"));
        let log = Arc::new(FailOnceLog::new(dir.path().join("changes.jsonl"))?);
        let state = curator.load_state()?;
        let request = SkillMutationRequest {
            request_id: "forward".into(),
            entity_key: "inverse".into(),
            kind: SkillMutationKind::Patch,
            reason: "forward".into(),
            files: vec![SkillFileMutation::new(
                &path,
                Some(b"before".to_vec()),
                Some(b"after".to_vec()),
            )?],
            curator_before: state.clone(),
            curator_after: state,
            rollback_of: None,
        };
        let receipt;
        let rollback_approval;
        let target;
        {
            let authority = SkillMutationAuthority::open(curator.clone(), log.clone())?;
            let preview = authority.preview(&request)?;
            receipt = match authority
                .apply(request, approval(&preview, "forward-approval"))
                .await?
            {
                SkillMutationOutcome::Applied(receipt) => receipt,
                other => return Err(ReactError::Other(format!("unexpected outcome: {other:?}"))),
            };
            target = SkillRollbackTarget::BatchId(receipt.batch_id.clone());
            let SkillRollbackPreviewOutcome::Ready(preview) = authority
                .preview_rollback("inverse-request", &target)
                .await?
            else {
                return Err(ReactError::Other("inverse preview was not ready".into()));
            };
            rollback_approval = approval(&preview, "inverse-approval");
            log.fail_next();
            assert!(
                authority
                    .rollback("inverse-request", target.clone(), rollback_approval.clone(),)
                    .await
                    .is_err()
            );
        }
        let authority = SkillMutationAuthority::open(curator, log.clone())?;
        assert!(matches!(
            authority
                .rollback("inverse-request", target, rollback_approval)
                .await?,
            SkillMutationOutcome::AlreadyApplied(_)
        ));
        assert_eq!(std::fs::read(path)?, b"before");
        assert_eq!(log.len(), 3);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn rollback_holds_serial_across_revalidation_prepare_and_projection() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("SKILL.md");
        std::fs::write(&path, "zero")?;
        let curator = Curator::new(CuratorConfig::default(), dir.path().join("curator.json"));
        let log = Arc::new(JsonlChangeLog::new(dir.path().join("changes.jsonl"))?);
        let calls = Arc::new(AtomicUsize::new(0));
        let (entered_tx, entered_rx) = std::sync::mpsc::sync_channel(1);
        let (release_tx, release_rx) = std::sync::mpsc::sync_channel(1);
        let release_rx = Arc::new(Mutex::new(release_rx));
        let hook_calls = calls.clone();
        let hook_release = release_rx.clone();
        let hook = Arc::new(move || {
            if hook_calls.fetch_add(1, Ordering::SeqCst) == 1 {
                entered_tx.send(()).map_err(|error| {
                    ReactError::Other(format!("rollback hook signal failed: {error}"))
                })?;
                hook_release
                    .lock()
                    .map_err(|error| {
                        ReactError::Other(format!("rollback hook lock failed: {error}"))
                    })?
                    .recv()
                    .map_err(|error| {
                        ReactError::Other(format!("rollback hook release failed: {error}"))
                    })?;
            }
            Ok(())
        });
        let authority = Arc::new(
            SkillMutationAuthority::open(curator.clone(), log)?.with_after_files_hook(hook),
        );
        let state = curator.load_state()?;
        let first = SkillMutationRequest {
            request_id: "serial-forward".into(),
            entity_key: "serial".into(),
            kind: SkillMutationKind::Patch,
            reason: "forward".into(),
            files: vec![SkillFileMutation::new(
                &path,
                Some(b"zero".to_vec()),
                Some(b"one".to_vec()),
            )?],
            curator_before: state.clone(),
            curator_after: state.clone(),
            rollback_of: None,
        };
        let preview = authority.preview(&first)?;
        let receipt = match authority
            .apply(first, approval(&preview, "serial-forward-approval"))
            .await?
        {
            SkillMutationOutcome::Applied(receipt) => receipt,
            other => return Err(ReactError::Other(format!("unexpected outcome: {other:?}"))),
        };
        let target = SkillRollbackTarget::BatchId(receipt.batch_id);
        let SkillRollbackPreviewOutcome::Ready(rollback_preview) = authority
            .preview_rollback("serial-rollback", &target)
            .await?
        else {
            return Err(ReactError::Other("serial rollback was not ready".into()));
        };
        let rollback_approval = approval(&rollback_preview, "serial-rollback-approval");
        let newer = SkillMutationRequest {
            request_id: "serial-newer".into(),
            entity_key: "serial".into(),
            kind: SkillMutationKind::Patch,
            reason: "newer".into(),
            files: vec![SkillFileMutation::new(
                &path,
                Some(b"one".to_vec()),
                Some(b"two".to_vec()),
            )?],
            curator_before: state.clone(),
            curator_after: state,
            rollback_of: None,
        };
        let newer_preview = authority.preview(&newer)?;
        let newer_approval = approval(&newer_preview, "serial-newer-approval");
        let rollback_task = {
            let authority = authority.clone();
            tokio::spawn(async move {
                authority
                    .rollback("serial-rollback", target, rollback_approval)
                    .await
            })
        };
        entered_rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .map_err(|error| {
                ReactError::Other(format!("rollback did not enter projection hook: {error}"))
            })?;
        let newer_task = {
            let authority = authority.clone();
            tokio::spawn(async move { authority.apply(newer, newer_approval).await })
        };
        tokio::task::yield_now().await;
        release_tx.send(()).map_err(|error| {
            ReactError::Other(format!("rollback release signal failed: {error}"))
        })?;
        let rollback_outcome = rollback_task
            .await
            .map_err(|error| ReactError::Other(format!("rollback task failed: {error}")))??;
        assert!(matches!(rollback_outcome, SkillMutationOutcome::Applied(_)));
        let newer_outcome = newer_task
            .await
            .map_err(|error| ReactError::Other(format!("newer task failed: {error}")))??;
        assert!(matches!(
            newer_outcome,
            SkillMutationOutcome::Conflict { .. }
        ));
        assert_eq!(std::fs::read(path)?, b"zero");
        Ok(())
    }

    #[tokio::test]
    async fn non_tip_aba_rollback_is_rejected_by_generation() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("SKILL.md");
        std::fs::write(&path, "same")?;
        let curator = Curator::new(CuratorConfig::default(), dir.path().join("curator.json"));
        let (authority, _log) =
            open_json_authority(curator.clone(), dir.path().join("changes.jsonl"))?;
        let state = curator.load_state()?;
        let first = SkillMutationRequest {
            request_id: "aba-1".into(),
            entity_key: "aba".into(),
            kind: SkillMutationKind::Patch,
            reason: "first".into(),
            files: vec![SkillFileMutation::new(
                &path,
                Some(b"same".to_vec()),
                Some(b"different".to_vec()),
            )?],
            curator_before: state.clone(),
            curator_after: state.clone(),
            rollback_of: None,
        };
        let first_preview = authority.preview(&first)?;
        let first_receipt = match authority
            .apply(first, approval(&first_preview, "aba-approval-1"))
            .await?
        {
            SkillMutationOutcome::Applied(receipt) => receipt,
            other => return Err(ReactError::Other(format!("unexpected outcome: {other:?}"))),
        };
        let second = SkillMutationRequest {
            request_id: "aba-2".into(),
            entity_key: "aba".into(),
            kind: SkillMutationKind::Patch,
            reason: "second".into(),
            files: vec![SkillFileMutation::new(
                path,
                Some(b"different".to_vec()),
                Some(b"same".to_vec()),
            )?],
            curator_before: state.clone(),
            curator_after: state,
            rollback_of: None,
        };
        let second_preview = authority.preview(&second)?;
        authority
            .apply(second, approval(&second_preview, "aba-approval-2"))
            .await?;
        assert!(matches!(
            authority
                .preview_rollback(
                    "rollback-old",
                    &SkillRollbackTarget::BatchId(first_receipt.batch_id),
                )
                .await?,
            SkillRollbackPreviewOutcome::Conflict { .. }
        ));
        Ok(())
    }

    #[test]
    fn same_stem_curator_paths_have_isolated_journals() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let json = Curator::new(CuratorConfig::default(), dir.path().join("state.json"));
        let toml = Curator::new(CuratorConfig::default(), dir.path().join("state.toml"));
        assert_ne!(
            json.skill_operation_journal_path(),
            toml.skill_operation_journal_path()
        );
        let (_json, _) = open_json_authority(json, dir.path().join("json.jsonl"))?;
        let (_toml, _) = open_json_authority(toml, dir.path().join("toml.jsonl"))?;
        Ok(())
    }

    #[test]
    fn authorities_for_same_root_share_one_in_process_serial() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let curator = Curator::new(CuratorConfig::default(), dir.path().join("state.json"));
        let log = Arc::new(JsonlChangeLog::new(dir.path().join("changes.jsonl"))?);
        let owner = SkillMutationAuthority::open(curator.clone(), log.clone())?;
        let peer = SkillMutationAuthority::open(curator, log)?;
        assert!(Arc::ptr_eq(&owner.journal.serial, &peer.journal.serial));
        Ok(())
    }

    #[test]
    fn concurrent_open_binds_exactly_one_destination_marker() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let state_path = dir.path().join("state.json");
        let curator = Curator::new(CuratorConfig::default(), &state_path);
        let log_a = Arc::new(JsonlChangeLog::new(dir.path().join("a.jsonl"))?);
        let log_b = Arc::new(JsonlChangeLog::new(dir.path().join("b.jsonl"))?);
        let barrier = Arc::new(std::sync::Barrier::new(3));
        let first = {
            let curator = curator.clone();
            let log = log_a.clone();
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                barrier.wait();
                SkillMutationAuthority::open(curator, log).is_ok()
            })
        };
        let second = {
            let curator = curator.clone();
            let log = log_b.clone();
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                barrier.wait();
                SkillMutationAuthority::open(curator, log).is_ok()
            })
        };
        barrier.wait();
        let first_ok = first
            .join()
            .map_err(|_| ReactError::Other("first authority open panicked".into()))?;
        let second_ok = second
            .join()
            .map_err(|_| ReactError::Other("second authority open panicked".into()))?;
        assert_ne!(first_ok, second_ok);
        assert_eq!(log_a.len() + log_b.len(), 1);

        let same_dir = tempfile::tempdir()?;
        let same_curator =
            Curator::new(CuratorConfig::default(), same_dir.path().join("state.json"));
        let same_log = Arc::new(JsonlChangeLog::new(same_dir.path().join("changes.jsonl"))?);
        let barrier = Arc::new(std::sync::Barrier::new(3));
        let mut joins = Vec::new();
        for _ in 0..2 {
            let curator = same_curator.clone();
            let log = same_log.clone();
            let barrier = barrier.clone();
            joins.push(std::thread::spawn(move || {
                barrier.wait();
                SkillMutationAuthority::open(curator, log).is_ok()
            }));
        }
        barrier.wait();
        let mut successes = 0;
        for join in joins {
            if join
                .join()
                .map_err(|_| ReactError::Other("same-log authority open panicked".into()))?
            {
                successes += 1;
            }
        }
        assert!(successes >= 1);
        assert_eq!(same_log.len(), 1);
        let reopened = SkillMutationAuthority::open(same_curator.clone(), same_log.clone())?;
        assert!(!reopened.audit_authority_id().is_empty());

        let copied_path = same_dir.path().join("copied.jsonl");
        std::fs::copy(same_dir.path().join("changes.jsonl"), &copied_path)?;
        let copied_log = Arc::new(JsonlChangeLog::new(copied_path)?);
        let copied_result = SkillMutationAuthority::open(same_curator.clone(), copied_log);
        assert!(matches!(copied_result, Err(ReactError::Other(_))));
        let same_path_reopened = SkillMutationAuthority::open(same_curator, same_log)?;
        assert_eq!(
            same_path_reopened.audit_authority_id(),
            reopened.audit_authority_id()
        );
        Ok(())
    }

    #[test]
    fn authority_rejects_changelog_without_durable_destination_identity() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let curator = Curator::new(CuratorConfig::default(), dir.path().join("state.json"));
        let log = Arc::new(UnsupportedIdentityLog::new(
            dir.path().join("changes.jsonl"),
        )?);
        assert!(matches!(
            SkillMutationAuthority::open(curator, log),
            Err(ReactError::Memory(error))
                if matches!(error.as_ref(), MemoryError::Unsupported(_))
        ));
        Ok(())
    }

    #[test]
    fn authority_rejects_unmarked_existing_business_log() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let curator = Curator::new(CuratorConfig::default(), dir.path().join("state.json"));
        let log = Arc::new(JsonlChangeLog::new(dir.path().join("changes.jsonl"))?);
        log.record(
            ChangeEntryBuilder::new(EntityType::Skill, "existing", ChangeType::Update)
                .reason("existing business record")
                .trigger("test")
                .build(log.as_ref()),
        )?;
        assert!(matches!(
            SkillMutationAuthority::open(curator, log),
            Err(ReactError::Other(_))
        ));
        Ok(())
    }

    #[test]
    fn skill_journal_lease_child_probe() -> Result<()> {
        let Some(path) = std::env::var_os("ECHO_TEST_SKILL_JOURNAL_STATE") else {
            return Ok(());
        };
        let curator = Curator::new(CuratorConfig::default(), PathBuf::from(&path));
        let log = Arc::new(JsonlChangeLog::new(
            PathBuf::from(&path).with_extension("changes.jsonl"),
        )?);
        if SkillMutationAuthority::open(curator, log).is_ok() {
            return Err(ReactError::Other(
                "competing process opened the live skill journal".into(),
            ));
        }
        Ok(())
    }

    #[test]
    fn second_process_fails_closed_while_skill_authority_is_live() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let state_path = dir.path().join("state.json");
        let curator = Curator::new(CuratorConfig::default(), &state_path);
        let log = Arc::new(JsonlChangeLog::new(dir.path().join("changes.jsonl"))?);
        let _owner = SkillMutationAuthority::open(curator, log)?;
        let executable = std::env::current_exe()?;
        let child = std::process::Command::new(executable)
            .arg("--exact")
            .arg("evolution::skill_mutation::tests::skill_journal_lease_child_probe")
            .env("ECHO_TEST_SKILL_JOURNAL_STATE", state_path)
            .output()?;
        if !child.status.success() {
            return Err(ReactError::Other(format!(
                "competing skill journal process probe failed: {}",
                String::from_utf8_lossy(&child.stdout)
            )));
        }
        Ok(())
    }

    #[tokio::test]
    async fn usage_handle_never_creates_unknown_and_audits_existing_skill() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let curator = Curator::new(CuratorConfig::default(), dir.path().join("curator.json"));
        let (authority, log) =
            open_json_authority(curator.clone(), dir.path().join("changes.jsonl"))?;
        let authority = Arc::new(authority);
        let usage = SkillUsageHandle::new(authority.clone());
        assert_eq!(
            usage.record_usage("unknown").await?,
            SkillUsageOutcome::UnknownSkill
        );
        assert!(curator.load_state()?.skills.is_empty());

        let before = curator.load_state()?;
        let mut after = before.clone();
        let skill_path = dir.path().join("known/SKILL.md");
        after.skills.insert(
            "known".into(),
            skill_meta("known", &skill_path, SkillLifecycle::Active),
        );
        let request = SkillMutationRequest {
            request_id: "register-known".into(),
            entity_key: "known".into(),
            kind: SkillMutationKind::Promote,
            reason: "register known skill".into(),
            files: Vec::new(),
            curator_before: before,
            curator_after: after,
            rollback_of: None,
        };
        let preview = authority.preview(&request)?;
        authority
            .apply(request, approval(&preview, "register-known-approval"))
            .await?;
        let before_usage = curator
            .load_state()?
            .skills
            .get("known")
            .map(|meta| meta.last_used_at)
            .ok_or_else(|| ReactError::Other("known skill missing".into()))?;
        assert_eq!(
            usage.record_usage("known").await?,
            SkillUsageOutcome::Updated
        );
        let after_usage = curator
            .load_state()?
            .skills
            .get("known")
            .map(|meta| meta.last_used_at)
            .ok_or_else(|| ReactError::Other("known skill missing after usage".into()))?;
        assert!(after_usage >= before_usage);
        assert_eq!(log.len(), 3);
        Ok(())
    }

    #[tokio::test]
    async fn concurrent_usage_updates_serialize_without_creating_parallel_authority() -> Result<()>
    {
        let dir = tempfile::tempdir()?;
        let curator = Curator::new(CuratorConfig::default(), dir.path().join("curator.json"));
        let (authority, log) =
            open_json_authority(curator.clone(), dir.path().join("changes.jsonl"))?;
        let authority = Arc::new(authority);
        let before = curator.load_state()?;
        let mut after = before.clone();
        after.skills.insert(
            "known".into(),
            skill_meta(
                "known",
                &dir.path().join("SKILL.md"),
                SkillLifecycle::Active,
            ),
        );
        let request = SkillMutationRequest {
            request_id: "register-concurrent".into(),
            entity_key: "known".into(),
            kind: SkillMutationKind::Promote,
            reason: "register concurrent skill".into(),
            files: Vec::new(),
            curator_before: before,
            curator_after: after,
            rollback_of: None,
        };
        let preview = authority.preview(&request)?;
        authority
            .apply(request, approval(&preview, "register-concurrent-approval"))
            .await?;
        let first = SkillUsageHandle::new(authority.clone());
        let second = SkillUsageHandle::new(authority);
        let (first, second) =
            tokio::join!(first.record_usage("known"), second.record_usage("known"));
        assert_eq!(first?, SkillUsageOutcome::Updated);
        assert_eq!(second?, SkillUsageOutcome::Updated);
        assert_eq!(log.len(), 4);
        Ok(())
    }

    #[tokio::test]
    async fn consumed_approval_cannot_authorize_another_request() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("SKILL.md");
        std::fs::write(&path, "one")?;
        let curator = Curator::new(CuratorConfig::default(), dir.path().join("curator.json"));
        let (authority, _log) =
            open_json_authority(curator.clone(), dir.path().join("changes.jsonl"))?;
        let state = curator.load_state()?;
        let first = SkillMutationRequest {
            request_id: "approval-request-1".into(),
            entity_key: "approval".into(),
            kind: SkillMutationKind::Patch,
            reason: "first".into(),
            files: vec![SkillFileMutation::new(
                &path,
                Some(b"one".to_vec()),
                Some(b"two".to_vec()),
            )?],
            curator_before: state.clone(),
            curator_after: state.clone(),
            rollback_of: None,
        };
        let first_preview = authority.preview(&first)?;
        let artifact = approval(&first_preview, "single-use-approval");
        authority.apply(first, artifact.clone()).await?;
        let second = SkillMutationRequest {
            request_id: "approval-request-2".into(),
            entity_key: "approval".into(),
            kind: SkillMutationKind::Patch,
            reason: "second".into(),
            files: vec![SkillFileMutation::new(
                &path,
                Some(b"two".to_vec()),
                Some(b"three".to_vec()),
            )?],
            curator_before: state.clone(),
            curator_after: state,
            rollback_of: None,
        };
        assert!(matches!(
            authority.apply(second, artifact).await?,
            SkillMutationOutcome::Conflict { .. }
        ));
        assert_eq!(std::fs::read(path)?, b"two");
        Ok(())
    }

    #[tokio::test]
    async fn cancellation_while_waiting_for_authority_prepares_nothing() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let curator = Curator::new(CuratorConfig::default(), dir.path().join("curator.json"));
        let log = Arc::new(JsonlChangeLog::new(dir.path().join("changes.jsonl"))?);
        let authority = Arc::new(SkillMutationAuthority::open(curator.clone(), log)?);
        let state = curator.load_state()?;
        let request = SkillMutationRequest {
            request_id: "cancelled".into(),
            entity_key: "cancelled".into(),
            kind: SkillMutationKind::Patch,
            reason: "cancelled".into(),
            files: vec![SkillFileMutation::new(
                dir.path().join("SKILL.md"),
                None,
                Some(b"content".to_vec()),
            )?],
            curator_before: state.clone(),
            curator_after: state,
            rollback_of: None,
        };
        let preview = authority.preview(&request)?;
        let artifact = approval(&preview, "cancelled-approval");
        let held = authority.journal.serial.lock().await;
        let task = {
            let authority = authority.clone();
            tokio::spawn(async move { authority.apply(request, artifact).await })
        };
        tokio::task::yield_now().await;
        task.abort();
        let _ = task.await;
        drop(held);
        assert!(authority.journal.history()?.is_empty());
        Ok(())
    }
}
