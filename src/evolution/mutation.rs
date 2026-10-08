//! 分层记忆与其审计之间的持久化、可前向恢复的边界。

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use echo_core::error::ReactError;
use echo_core::utils::fs::FileDurability;
use echo_state::journal::file::FileEventJournal;
use echo_state::journal::{EventJournal, JournalDurabilityStatus, PreparedJournalBatch};
use serde::{Deserialize, Serialize};

use super::audit::ChangeEntry;
use super::layer::HotEntryMeta;

/// 本模块统一使用的结果类型。
type Result<T> = std::result::Result<T, ReactError>;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
/// 热层记忆条目的完整值（元数据加内容）。
pub(super) struct HotValue {
    /// 热层条目元数据。
    pub meta: HotEntryMeta,
    /// 热层条目内容。
    pub content: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
/// 针对单个记忆键的一次变更操作，记录温层/热层的前后值及审计条目。
pub(super) struct MemoryOperation {
    /// 操作 ID，须与审计条目的 `change_id` 一致。
    pub id: String,
    /// 被修改的记忆键。
    pub key: String,
    /// 变更前的温层值（不存在时为 `None`）。
    pub warm_before: Option<serde_json::Value>,
    /// 变更后的温层值（删除时为 `None`）。
    pub warm_after: Option<serde_json::Value>,
    /// 变更前的热层值。
    pub hot_before: Option<HotValue>,
    /// 变更后的热层值。
    pub hot_after: Option<HotValue>,
    /// 与本操作对应的审计条目。
    pub audit: ChangeEntry,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
/// 一批需要原子地准备与结算的记忆操作。
pub(super) struct MemoryOperationBatch {
    /// 批次 ID。
    pub id: String,
    /// 批次内的操作列表（键与 ID 均不得重复）。
    pub operations: Vec<MemoryOperation>,
    /**
     * 前向/回滚谱系是日志私有的元数据。`None` 表示旧版的前向形式，
     * 以确保旧日志记录仍可解码。
     */
    #[serde(default)]
    pub origin: Option<MemoryOperationOrigin>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
/// 批次的来源谱系（例如回滚请求所针对的目标批次）。
pub(super) struct MemoryOperationOrigin {
    /// 触发该批次的请求 ID。
    pub request_id: String,
    /// 目标批次 ID。
    pub target_batch_id: String,
    /// 目标批次在日志中的代次（序号）。
    pub target_generation: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
/// 写入日志的记忆操作事件。
enum MemoryOperationEvent {
    /// 批次已准备（尚未结算）。
    Prepared(MemoryOperationBatch),
    /// 指定 ID 的批次已结算。
    Settled {
        /// 已结算的批次 ID。
        id: String,
    },
}

#[derive(Clone, Debug)]
/// 从日志重放得到的单个批次历史记录。
pub(super) struct MemoryOperationHistory {
    /// 批次内容。
    pub batch: MemoryOperationBatch,
    /// 该批次是否已结算。
    pub settled: bool,
    /// 批次准备事件在日志中的序号（代次）。
    pub generation: u64,
}

/// 基于文件事件日志的记忆操作日志，提供持久化的准备/结算记录。
pub(super) struct MemoryOperationJournal {
    /// 底层文件事件日志。
    journal: Arc<FileEventJournal<MemoryOperationEvent>>,
    /// 同一日志路径共享的串行化锁。
    serial: Arc<tokio::sync::Mutex<()>>,
}

impl MemoryOperationJournal {
    /// 打开（或创建）根目录下的记忆操作日志，并获取该路径共享的串行锁。
    pub fn open(root: &Path) -> Result<Self> {
        let path = root.join("evolution").join("memory-operations.jsonl");
        let journal = Arc::new(FileEventJournal::open(path, FileDurability::SyncData)?);
        static SERIALS: OnceLock<Mutex<HashMap<PathBuf, Weak<tokio::sync::Mutex<()>>>>> =
            OnceLock::new();
        let mut serials = SERIALS
            .get_or_init(|| Mutex::new(HashMap::new()))
            .lock()
            .map_err(|error| {
                ReactError::Other(format!(
                    "memory operation lock registry is poisoned: {error}"
                ))
            })?;
        let serial = match serials.get(journal.path()).and_then(Weak::upgrade) {
            Some(existing) => existing,
            None => {
                let created = Arc::new(tokio::sync::Mutex::new(()));
                serials.insert(journal.path().to_path_buf(), Arc::downgrade(&created));
                created
            }
        };
        Ok(Self { journal, serial })
    }

    /// 返回用于串行化记忆操作的共享锁。
    pub fn serial(&self) -> Arc<tokio::sync::Mutex<()>> {
        Arc::clone(&self.serial)
    }

    /// 追加事件并确保其持久性已确认，否则返回错误。
    fn append_confirmed(&self, event: MemoryOperationEvent) -> Result<()> {
        let batch = PreparedJournalBatch::new(vec![event])
            .map_err(|error| ReactError::Other(error.to_string()))?;
        let receipt = self
            .journal
            .append_batch(batch)
            .map_err(|error| ReactError::Other(error.to_string()))?;
        match receipt.durability() {
            JournalDurabilityStatus::Confirmed => Ok(()),
            JournalDurabilityStatus::Degraded { .. } => self.journal.sync_data(),
            JournalDurabilityStatus::Unconfirmed => Err(ReactError::Other(
                "memory operation journal append has unconfirmed durability".into(),
            )),
        }
    }

    /// 将单个操作作为独立批次写入准备记录。
    pub fn prepare(&self, operation: MemoryOperation) -> Result<()> {
        self.prepare_batch(MemoryOperationBatch {
            id: operation.id.clone(),
            operations: vec![operation],
            origin: None,
        })
    }

    /// 校验批次（非空、键与 ID 不重复、审计身份一致）后写入准备记录。
    pub fn prepare_batch(&self, batch: MemoryOperationBatch) -> Result<()> {
        if batch.id.is_empty() || batch.operations.is_empty() {
            return Err(ReactError::Other(
                "memory operation batch must not be empty".into(),
            ));
        }
        let mut keys = std::collections::HashSet::new();
        let mut ids = std::collections::HashSet::new();
        for operation in &batch.operations {
            if operation.id.is_empty()
                || operation.id != operation.audit.change_id
                || !keys.insert(operation.key.as_str())
                || !ids.insert(operation.id.as_str())
            {
                return Err(ReactError::Other(
                    "memory operation batch has duplicate keys or invalid audit identity".into(),
                ));
            }
        }
        self.append_confirmed(MemoryOperationEvent::Prepared(batch))
    }

    /// 写入指定批次的结算记录。
    pub fn settle(&self, id: &str) -> Result<()> {
        self.append_confirmed(MemoryOperationEvent::Settled { id: id.to_owned() })
    }

    /// 返回所有批次及其是否已结算。
    pub fn history(&self) -> Result<Vec<(MemoryOperationBatch, bool)>> {
        Ok(self
            .history_with_generations()?
            .into_iter()
            .map(|item| (item.batch, item.settled))
            .collect())
    }

    /// 重放日志，返回带代次信息的完整批次历史，并校验准备/结算事实的一致性。
    pub fn history_with_generations(&self) -> Result<Vec<MemoryOperationHistory>> {
        self.journal.sync_data()?;
        let mut identities = HashMap::<String, usize>::new();
        let mut known_ids = HashSet::<String>::new();
        let mut history = Vec::<MemoryOperationHistory>::new();
        let mut sequence = 0_u64;
        loop {
            let records = self.journal.replay_after(sequence, 512)?;
            if records.is_empty() {
                break;
            }
            for record in &records {
                match record.event.as_ref() {
                    MemoryOperationEvent::Prepared(batch) => {
                        let index = history.len();
                        if !known_ids.insert(batch.id.clone()) {
                            return Err(ReactError::Other(format!(
                                "memory operation {} has duplicate prepare facts",
                                batch.id
                            )));
                        }
                        identities.insert(batch.id.clone(), index);
                        history.push(MemoryOperationHistory {
                            batch: batch.clone(),
                            settled: false,
                            generation: record.sequence,
                        });
                    }
                    MemoryOperationEvent::Settled { id } => {
                        let Some(index) = identities.remove(id) else {
                            return Err(ReactError::Other(format!(
                                "memory operation {id} has no pending prepare fact"
                            )));
                        };
                        let Some(item) = history.get_mut(index) else {
                            return Err(ReactError::Other(format!(
                                "memory operation {id} history index is invalid"
                            )));
                        };
                        item.settled = true;
                    }
                }
                sequence = record.sequence;
            }
        }
        Ok(history)
    }
}

#[cfg(test)]
mod tests {
    use super::MemoryOperationBatch;

    /// 验证不含谱系字段的旧版批次可解码为前向历史。
    #[test]
    fn legacy_batch_without_lineage_decodes_as_forward_history() -> Result<(), serde_json::Error> {
        let legacy = serde_json::json!({
            "id": "legacy-batch",
            "operations": []
        });
        let decoded = serde_json::from_value::<MemoryOperationBatch>(legacy)?;
        assert!(decoded.origin.is_none());
        Ok(())
    }
}
