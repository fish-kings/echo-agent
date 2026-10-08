/*!
变更审计日志——记录对 memories、skills、rules 的所有变更操作。

每一次 create、update、delete、promote、demote 与 merge 操作都会作为一条
`ChangeEntry` 记录进 JSONL 文件。这带来以下能力：

- **审计（Auditing）**：回顾发生了什么变更、为什么变更
- **对账（Reconciliation）**：稳定的 change ID 让一次已准备好的内存操作
  可以在重启后继续完成其审计；标准内存日志之后还可以追加一条带类型的
  逆向变更，而本日志始终保持仅追加（append-only）。
- **趋势分析（Trending）**：检测演化活动中的模式
*/

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::error::Result;

/// 进程内计数器，用于消除同一纳秒内产生的 change_id 的歧义。
static CHANGE_COUNTER: AtomicU64 = AtomicU64::new(0);

// ── ChangeType ──────────────────────────────────────────────────────────

/// 所执行的变更种类。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChangeType {
    /// 创建了一个新实体。
    Create,
    /// 更新了一个已有实体。
    Update,
    /// 删除了一个实体。
    Delete,
    /// 一个实体被提升到更高的层级或状态。
    Promote,
    /// 一个实体被降级到更低的层级或状态。
    Demote,
    /// 两个或多个实体被合并为一个。
    Merge,
}

// ── EntityType ──────────────────────────────────────────────────────────

/// 被变更的实体类型。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EntityType {
    /// 一条内存条目。
    Memory,
    /// 一个技能（SKILL.md 或候选技能）。
    Skill,
    /// 一条规则（AGENTS.md 或指令文件）。
    Rule,
}

// ── ChangeEntry ─────────────────────────────────────────────────────────

/// 演化审计日志中记录的一次变更。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChangeEntry {
    /// 该变更的唯一标识符。
    pub change_id: String,
    /// 变更发生的时间。
    #[serde(with = "crate::utils::time::local_rfc3339")]
    pub timestamp: DateTime<Utc>,
    /// 被变更的实体类型。
    pub entity_type: EntityType,
    /// 被变更实体的 key/path。
    pub entity_key: String,
    /// 变更的种类。
    pub change_type: ChangeType,
    /// 变更前的实体状态（若可获得）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub before: Option<serde_json::Value>,
    /// 变更后的实体状态。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub after: Option<serde_json::Value>,
    /// 该变更的可读原因说明。
    pub reason: String,
    /// 触发本次变更的来源（例如 "memory_reviewer"、"user_command"、"auto_trigger"）。
    pub trigger: String,
}

// ── ChangeLog 特征 ─────────────────────────────────────────────────────

/// 用于记录与查询演化变更的 trait。
pub trait ChangeLog: Send + Sync {
    /**
    具体审计目的地的稳定持久标识。

    变更（mutation）日志用它来防止一个待处理操作被重放进另一个
    ChangeLog。没有稳定目的地的实现必须拒绝绑定持久变更权威。
    */
    fn durable_destination_identity(&self) -> Result<Option<String>> {
        Ok(None)
    }

    /// 记录一条变更。
    fn record(&self, entry: ChangeEntry) -> Result<()>;

    /**
    以其稳定的 `change_id` 为标识，原子地记录一条条目。

    重放一条完全相同的条目会成功且不会追加第二行；
    对同一 ID 复用为不同内容则会失败关闭（fail closed）。
    */
    fn record_idempotent(&self, entry: ChangeEntry) -> Result<ChangeRecordOutcome>;

    /// 查询匹配给定过滤条件的变更。
    fn query(&self, filter: &ChangeFilter) -> Result<Vec<ChangeEntry>>;

    /// 查找某个实体最近一次的变更。
    fn latest_for(&self, entity_type: EntityType, entity_key: &str) -> Result<Option<ChangeEntry>>;

    /// 获取已记录变更的总数。
    fn len(&self) -> usize;

    /// 该日志是否为空。
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// 一次幂等审计追加操作的结果。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChangeRecordOutcome {
    /// 该条目已被持久化追加。
    Appended,
    /// 拥有相同稳定 ID 的相同条目此前已经持久化过。
    AlreadyRecorded,
}

// ── ChangeFilter ────────────────────────────────────────────────────────

/// 查询变更日志时使用的过滤条件。
#[derive(Debug, Clone, Default)]
pub struct ChangeFilter {
    /// 按实体类型过滤。
    pub entity_type: Option<EntityType>,
    /// 按变更类型过滤。
    pub change_type: Option<ChangeType>,
    /// 按实体 key 前缀过滤。
    pub entity_key_prefix: Option<String>,
    /// 只保留晚于该时间戳的变更。
    pub after: Option<DateTime<Utc>>,
    /// 只保留早于该时间戳的变更。
    pub before: Option<DateTime<Utc>>,
    /// 结果数量上限。
    pub limit: Option<usize>,
}

impl ChangeFilter {
    /// 创建一个空过滤条件（匹配所有内容）。
    pub fn new() -> Self {
        Self::default()
    }

    /// 按实体类型过滤。
    pub fn with_entity_type(mut self, entity_type: EntityType) -> Self {
        self.entity_type = Some(entity_type);
        self
    }

    /// 按变更类型过滤。
    pub fn with_change_type(mut self, change_type: ChangeType) -> Self {
        self.change_type = Some(change_type);
        self
    }

    /// 按实体 key 前缀过滤。
    pub fn with_key_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.entity_key_prefix = Some(prefix.into());
        self
    }

    /// 限制结果数量。
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    /// 检查一条变更是否匹配该过滤条件。
    pub fn matches(&self, entry: &ChangeEntry) -> bool {
        if let Some(ref et) = self.entity_type
            && entry.entity_type != *et
        {
            return false;
        }
        if let Some(ref ct) = self.change_type
            && entry.change_type != *ct
        {
            return false;
        }
        if let Some(ref prefix) = self.entity_key_prefix
            && !entry.entity_key.starts_with(prefix)
        {
            return false;
        }
        if let Some(after) = self.after
            && entry.timestamp < after
        {
            return false;
        }
        if let Some(before) = self.before
            && entry.timestamp > before
        {
            return false;
        }
        true
    }
}

// ── JsonlChangeLog ──────────────────────────────────────────────────────

/**
以 JSONL 文件为后端的变更日志。每一行是一个 `ChangeEntry` JSON 对象。

通过内部锁实现线程安全。追加操作是原子的（每次写入一行）。
*/
pub struct JsonlChangeLog {
    /// 日志文件路径。
    path: PathBuf,
    /// 内存中的条目缓存，用于快速查询。
    entries: std::sync::Mutex<Vec<ChangeEntry>>,
}

impl JsonlChangeLog {
    /**
    在给定路径创建一个新的 JSONL 变更日志。

    加载并校验已有条目。数据文件会在第一次持久化记录时创建。
    */
    pub fn new(path: PathBuf) -> Result<Self> {
        let log = Self {
            path,
            entries: std::sync::Mutex::new(Vec::new()),
        };
        let entries = log.with_disk_lock(|| Self::load_from_disk(&log.path))?;
        *log.entries
            .lock()
            .unwrap_or_else(|error| error.into_inner()) = entries;
        Ok(log)
    }

    /**
    从 JSONL 文件加载条目。

    每一条以换行符结尾的记录都必须能被解析。只有最后一条没有换行符的
    记录才可能是崩溃写坏的追加：一条有效记录会补上缺失的换行符，
    而无效的结尾会被截断到最后一条持久化记录处。内部（非末尾）的损坏
    永远不会被跳过。
    */
    fn load_from_disk(path: &Path) -> std::io::Result<Vec<ChangeEntry>> {
        let content = match echo_core::utils::fs::read_existing(path) {
            Ok(content) => content,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => return Err(error),
        };
        let mut entries = Vec::new();
        let mut change_ids = HashSet::new();
        let mut committed_len = 0_u64;

        for (line_index, segment) in content.split_inclusive(|byte| *byte == b'\n').enumerate() {
            let complete = segment.last().is_some_and(|byte| *byte == b'\n');
            let line = if complete {
                segment.strip_suffix(b"\n").unwrap_or(segment)
            } else {
                segment
            };
            match serde_json::from_slice::<ChangeEntry>(line) {
                Ok(entry) => {
                    if !change_ids.insert(entry.change_id.clone()) {
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            format!(
                                "audit log {} repeats change_id {:?} at line {}",
                                path.display(),
                                entry.change_id,
                                line_index.saturating_add(1)
                            ),
                        ));
                    }
                    entries.push(entry);
                }
                Err(error) if !complete => {
                    tracing::warn!(
                        path = ?path,
                        error = %error,
                        tail_preview = %String::from_utf8_lossy(line).chars().take(120).collect::<String>(),
                        "audit log: truncating crash-torn final JSONL record"
                    );
                    echo_core::utils::fs::truncate_existing(
                        path,
                        committed_len,
                        echo_core::utils::fs::FileDurability::SyncData,
                    )?;
                    return Ok(entries);
                }
                Err(error) => {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!(
                            "audit log {} has corrupt complete record at line {}: {}; preview={:?}",
                            path.display(),
                            line_index.saturating_add(1),
                            error,
                            String::from_utf8_lossy(line)
                                .chars()
                                .take(120)
                                .collect::<String>()
                        ),
                    ));
                }
            }

            if complete {
                committed_len = committed_len
                    .checked_add(u64::try_from(segment.len()).map_err(|error| {
                        std::io::Error::other(format!("audit record length overflow: {error}"))
                    })?)
                    .ok_or_else(|| std::io::Error::other("audit log length overflow"))?;
            } else {
                echo_core::utils::fs::append_existing(
                    path,
                    b"\n",
                    echo_core::utils::fs::FileDurability::SyncData,
                )?;
            }
        }
        Ok(entries)
    }

    /// 持有跨进程的建议性文件锁（flock），在锁保护下执行给定操作。
    fn with_disk_lock<T>(
        &self,
        operation: impl FnOnce() -> std::io::Result<T>,
    ) -> std::io::Result<T> {
        if let Some(parent) = self
            .path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent)?;
        }

        let lock_path = self.path.with_extension("jsonl.lock");
        let lock_file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&lock_path)?;
        use fs2::FileExt;
        // 获取文件锁的最大尝试次数。
        const MAX_ATTEMPTS: u32 = 600;
        // 每次尝试获取文件锁失败后的等待间隔。
        const BACKOFF: std::time::Duration = std::time::Duration::from_secs(1);
        let mut got_lock = false;
        for _ in 0..MAX_ATTEMPTS {
            match lock_file.try_lock_exclusive() {
                Ok(()) => {
                    got_lock = true;
                    break;
                }
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(BACKOFF);
                }
                Err(error) => return Err(error),
            }
        }
        if !got_lock {
            return Err(std::io::Error::new(
                std::io::ErrorKind::WouldBlock,
                format!("audit log lock unavailable after {MAX_ATTEMPTS} attempts"),
            ));
        }

        let result = operation();
        let unlock_result = lock_file.unlock();
        match (result, unlock_result) {
            (Err(error), _) | (Ok(_), Err(error)) => Err(error),
            (Ok(value), Ok(())) => Ok(value),
        }
    }

    /// 若日志文件尚不存在，则创建一个空文件。
    fn create_log_file_if_missing(&self) -> std::io::Result<()> {
        match std::fs::symlink_metadata(&self.path) {
            Ok(_) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                echo_core::utils::fs::atomic_write(&self.path, b"")
            }
            Err(error) => Err(error),
        }
    }

    /**
    向 JSONL 文件追加一条条目。

    在一个旁路（sidecar）文件上持有跨进程的建议性 `flock`，并在返回前
    调用 `sync_all`，因此：
    - 并发的追加者（本进程或其他进程）不会把两条较大的 JSON 行交织成
      一行损坏的数据；
    - 掉电崩溃不会丢失刚刚追加的那一行（操作系统会在我们报告成功之前
      刷新这次持久化写入）。

    每次追加都执行 `sync_all`，这是与*审计*日志的目标（完整、持久的
    记录）相匹配的持久性/成本权衡。
    */
    fn record_idempotent_inner(&self, entry: ChangeEntry) -> Result<ChangeRecordOutcome> {
        if entry.change_id.is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "audit change_id must not be empty",
            )
            .into());
        }

        let mut cached = self
            .entries
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let (entries, outcome) = self.with_disk_lock(|| {
            let mut entries = Self::load_from_disk(&self.path)?;
            if let Some(existing) = entries
                .iter()
                .find(|existing| existing.change_id == entry.change_id)
            {
                if existing == &entry {
                    return Ok((entries, ChangeRecordOutcome::AlreadyRecorded));
                }
                return Err(std::io::Error::new(
                    std::io::ErrorKind::AlreadyExists,
                    format!(
                        "audit change_id {:?} already identifies different content",
                        entry.change_id
                    ),
                ));
            }

            let mut line = serde_json::to_vec(&entry).map_err(std::io::Error::other)?;
            line.push(b'\n');
            self.create_log_file_if_missing()?;
            echo_core::utils::fs::append_existing(
                &self.path,
                &line,
                echo_core::utils::fs::FileDurability::SyncData,
            )?;
            entries.push(entry);
            Ok((entries, ChangeRecordOutcome::Appended))
        })?;
        *cached = entries;
        Ok(outcome)
    }
}

impl ChangeLog for JsonlChangeLog {
    /// 返回该日志规范化后的持久化目标标识。
    fn durable_destination_identity(&self) -> Result<Option<String>> {
        Ok(Some(format!(
            "jsonl:{}",
            canonical_destination_path(&self.path)?.display()
        )))
    }

    /// 记录一条变更条目。
    fn record(&self, entry: ChangeEntry) -> Result<()> {
        self.record_idempotent_inner(entry)?;
        Ok(())
    }

    /// 以幂等方式记录一条变更条目。
    fn record_idempotent(&self, entry: ChangeEntry) -> Result<ChangeRecordOutcome> {
        self.record_idempotent_inner(entry)
    }

    /// 按过滤条件查询变更条目。
    fn query(&self, filter: &ChangeFilter) -> Result<Vec<ChangeEntry>> {
        let mut entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        *entries = self.with_disk_lock(|| Self::load_from_disk(&self.path))?;
        let mut results: Vec<ChangeEntry> = entries
            .iter()
            .rev() // 最近的排在前面
            .filter(|e| filter.matches(e))
            .cloned()
            .collect();

        if let Some(limit) = filter.limit {
            results.truncate(limit);
        }
        Ok(results)
    }

    /// 返回指定实体的最新变更条目。
    fn latest_for(&self, entity_type: EntityType, entity_key: &str) -> Result<Option<ChangeEntry>> {
        let mut entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        *entries = self.with_disk_lock(|| Self::load_from_disk(&self.path))?;
        Ok(entries
            .iter()
            .rev()
            .find(|e| e.entity_type == entity_type && e.entity_key == entity_key)
            .cloned())
    }

    /// 返回已记录的条目数量。
    fn len(&self) -> usize {
        self.entries.lock().unwrap_or_else(|e| e.into_inner()).len()
    }
}

/**
计算该路径的规范化形式；若路径尚不存在，则沿祖先目录向上查找存在的
那一级再拼回后缀部分。
*/
fn canonical_destination_path(path: &Path) -> std::io::Result<PathBuf> {
    if path.exists() {
        return std::fs::canonicalize(path);
    }
    let absolute = std::path::absolute(path)?;
    let mut probe = absolute.as_path();
    let mut suffix = Vec::new();
    while !probe.exists() {
        suffix.push(
            probe
                .file_name()
                .ok_or_else(|| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        format!(
                            "ChangeLog path has no existing ancestor: {}",
                            path.display()
                        ),
                    )
                })?
                .to_os_string(),
        );
        probe = probe.parent().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("ChangeLog path has no parent: {}", path.display()),
            )
        })?;
    }
    let mut canonical = std::fs::canonicalize(probe)?;
    for segment in suffix.into_iter().rev() {
        canonical.push(segment);
    }
    Ok(canonical)
}

// ── 创建变更条目的辅助工具 ──────────────────────────────────────────────

/// 用于创建 `ChangeEntry` 实例的 builder 辅助工具。
pub struct ChangeEntryBuilder {
    /// 实体类型。
    entity_type: EntityType,
    /// 实体 key。
    entity_key: String,
    /// 变更类型。
    change_type: ChangeType,
    /// 变更前状态。
    before: Option<serde_json::Value>,
    /// 变更后状态。
    after: Option<serde_json::Value>,
    /// 变更原因。
    reason: String,
    /// 触发来源。
    trigger: String,
}

impl ChangeEntryBuilder {
    /// 开始构建一条变更记录。
    pub fn new(
        entity_type: EntityType,
        entity_key: impl Into<String>,
        change_type: ChangeType,
    ) -> Self {
        Self {
            entity_type,
            entity_key: entity_key.into(),
            change_type,
            before: None,
            after: None,
            reason: String::new(),
            trigger: String::new(),
        }
    }

    /// 设置变更前状态。
    pub fn before(mut self, value: serde_json::Value) -> Self {
        self.before = Some(value);
        self
    }

    /// 设置变更后状态。
    pub fn after(mut self, value: serde_json::Value) -> Self {
        self.after = Some(value);
        self
    }

    /// 设置变更原因。
    pub fn reason(mut self, reason: impl Into<String>) -> Self {
        self.reason = reason.into();
        self
    }

    /// 设置触发来源。
    pub fn trigger(mut self, trigger: impl Into<String>) -> Self {
        self.trigger = trigger.into();
        self
    }

    /**
    构建该变更记录，分配新的 ID 与时间戳。

    使用纳秒精度的时间戳来生成 ID，以避免此前 `log.len() + 1`
    （在锁之外读取）所存在的 TOCTOU 竞态。
    */
    pub fn build(self, _log: &dyn ChangeLog) -> ChangeEntry {
        let now = Utc::now();
        /* timestamp_nanos_opt() 只有在极端溢出时才会返回 None；
        此时回退为 0，让 id 仍然是良构的（唯一性仍然由 `now` 保证）。 */
        let nanos = now.timestamp_nanos_opt().unwrap_or(0);
        ChangeEntry {
            change_id: format!(
                "chg_{:016x}_{}",
                nanos,
                CHANGE_COUNTER.fetch_add(1, Ordering::Relaxed)
            ),
            timestamp: now,
            entity_type: self.entity_type,
            entity_key: self.entity_key,
            change_type: self.change_type,
            before: self.before,
            after: self.after,
            reason: self.reason,
            trigger: self.trigger,
        }
    }

    /**
    使用调用方持有的稳定 ID 与时间戳来构建。

    恢复流程会持久化这些值，并通过 [`ChangeLog::record_idempotent`]
    重放同一条完整记录。
    */
    pub fn build_with(self, change_id: impl Into<String>, timestamp: DateTime<Utc>) -> ChangeEntry {
        ChangeEntry {
            change_id: change_id.into(),
            timestamp,
            entity_type: self.entity_type,
            entity_key: self.entity_key,
            change_type: self.change_type,
            before: self.before,
            after: self.after,
            reason: self.reason,
            trigger: self.trigger,
        }
    }
}

// ── 测试/mock 共享实现 ──────────────────────────────────────────────

/**
供测试与关闭相关 feature 时构建使用的空操作 `ChangeLog`。请直接使用它，
而不要在各个模块中各自定义一份（P1 —— nullChangeLog 去重）。
*/
pub struct NullChangeLog;

impl ChangeLog for NullChangeLog {
    /// 空操作：直接丢弃条目。
    fn record(&self, _entry: ChangeEntry) -> Result<()> {
        Ok(())
    }
    /// 空操作：总是视为已记录。
    fn record_idempotent(&self, _entry: ChangeEntry) -> Result<ChangeRecordOutcome> {
        Ok(ChangeRecordOutcome::AlreadyRecorded)
    }
    /// 空操作：总是返回空列表。
    fn query(&self, _filter: &ChangeFilter) -> Result<Vec<ChangeEntry>> {
        Ok(vec![])
    }
    /// 空操作：总是返回 `None`。
    fn latest_for(
        &self,
        _entity_type: EntityType,
        _entity_key: &str,
    ) -> Result<Option<ChangeEntry>> {
        Ok(None)
    }
    /// 空操作：总是返回 `0`。
    fn len(&self) -> usize {
        0
    }
}

// ── 测试 ───────────────────────────────────────────────────────────────

/// 审计日志的单元测试。
#[cfg(test)]
mod tests {
    use super::*;

    /// 构造一条用于测试的示例变更条目。
    fn sample_entry(change_id: &str, entity_key: &str) -> ChangeEntry {
        ChangeEntryBuilder::new(EntityType::Memory, entity_key, ChangeType::Create)
            .reason("Test")
            .trigger("test")
            .build_with(change_id, Utc::now())
    }

    /// 验证记录后可以查询到条目。
    #[test]
    fn test_record_and_query() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let log = JsonlChangeLog::new(dir.path().join("change-log.jsonl"))?;

        let entry = ChangeEntryBuilder::new(EntityType::Memory, "mem_001", ChangeType::Create)
            .reason("Auto-extracted observation")
            .trigger("auto_memory")
            .build_with("chg_000001", Utc::now());

        log.record(entry)?;
        assert_eq!(log.len(), 1);

        let results = log.query(
            &ChangeFilter::new()
                .with_entity_type(EntityType::Memory)
                .with_limit(10),
        )?;
        assert_eq!(results.len(), 1);
        assert_eq!(
            results.first().map(|entry| entry.entity_key.as_str()),
            Some("mem_001")
        );
        Ok(())
    }

    /// 验证两个句柄都能查询到当前的持久化审计记录。
    #[test]
    fn two_handles_query_current_durable_audit()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("change-log.jsonl");
        let first = JsonlChangeLog::new(path.clone())?;
        let second = JsonlChangeLog::new(path)?;
        first.record_idempotent(sample_entry("first", "one"))?;
        second.record_idempotent(sample_entry("second", "two"))?;
        assert_eq!(first.query(&ChangeFilter::new())?.len(), 2);
        assert_eq!(second.query(&ChangeFilter::new())?.len(), 2);
        assert_eq!(
            first
                .latest_for(EntityType::Memory, "two")?
                .map(|entry| entry.change_id),
            Some("second".into())
        );
        Ok(())
    }

    /// 验证 `latest_for` 返回实体的最新条目。
    #[test]
    fn test_latest_for() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let log = JsonlChangeLog::new(dir.path().join("change-log.jsonl"))?;

        let entry1 = ChangeEntryBuilder::new(EntityType::Memory, "mem_001", ChangeType::Create)
            .reason("Created")
            .trigger("test")
            .build_with("chg_000001", Utc::now());

        let entry2 = ChangeEntryBuilder::new(EntityType::Memory, "mem_001", ChangeType::Update)
            .reason("Updated confidence")
            .trigger("memory_reviewer")
            .build_with("chg_000002", Utc::now());

        log.record(entry1)?;
        log.record(entry2)?;

        let latest = log.latest_for(EntityType::Memory, "mem_001")?;
        assert_eq!(
            latest.map(|entry| entry.change_type),
            Some(ChangeType::Update)
        );
        Ok(())
    }

    /// 验证按变更类型过滤。
    #[test]
    fn test_filter_by_change_type() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let log = JsonlChangeLog::new(dir.path().join("change-log.jsonl"))?;

        for (i, ct) in [ChangeType::Create, ChangeType::Update, ChangeType::Promote]
            .iter()
            .enumerate()
        {
            let entry = ChangeEntryBuilder::new(EntityType::Skill, format!("skill_{i}"), *ct)
                .reason("Test")
                .trigger("test")
                .build_with(format!("chg_{:06}", i + 1), Utc::now());
            log.record(entry)?;
        }

        let promotes = log.query(&ChangeFilter::new().with_change_type(ChangeType::Promote))?;
        assert_eq!(promotes.len(), 1);
        assert_eq!(
            promotes.first().map(|entry| entry.change_type),
            Some(ChangeType::Promote)
        );
        Ok(())
    }

    /// 验证按键前缀过滤。
    #[test]
    fn test_filter_by_key_prefix() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let log = JsonlChangeLog::new(dir.path().join("change-log.jsonl"))?;

        let entry1 = ChangeEntryBuilder::new(EntityType::Memory, "build/java8", ChangeType::Create)
            .reason("Test")
            .trigger("test")
            .build_with("chg_000001", Utc::now());
        let entry2 =
            ChangeEntryBuilder::new(EntityType::Memory, "style/concise", ChangeType::Create)
                .reason("Test")
                .trigger("test")
                .build_with("chg_000002", Utc::now());

        log.record(entry1)?;
        log.record(entry2)?;

        let build_entries = log.query(&ChangeFilter::new().with_key_prefix("build/"))?;
        assert_eq!(build_entries.len(), 1);
        assert_eq!(
            build_entries.first().map(|entry| entry.entity_key.as_str()),
            Some("build/java8")
        );
        Ok(())
    }

    /// 验证中间记录损坏时以失败关闭方式处理且不修改文件。
    #[test]
    fn interior_corruption_fails_closed_without_mutation()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("change-log.jsonl");
        let first = serde_json::to_vec(&sample_entry("promotion-1", "rule/one"))?;
        let second = serde_json::to_vec(&sample_entry("promotion-2", "rule/two"))?;
        let mut bytes = first;
        bytes.extend_from_slice(b"\n{not-json}\n");
        bytes.extend_from_slice(&second);
        bytes.push(b'\n');
        std::fs::write(&path, &bytes)?;

        assert!(JsonlChangeLog::new(path.clone()).is_err());
        assert_eq!(std::fs::read(path)?, bytes);
        Ok(())
    }

    /// 验证被截断的最后一条记录会被截掉，只保留最后一条完整条目。
    #[test]
    fn torn_final_record_is_truncated_to_last_complete_entry()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("change-log.jsonl");
        let first = serde_json::to_vec(&sample_entry("promotion-1", "rule/one"))?;
        let mut durable = first;
        durable.push(b'\n');
        let mut bytes = durable.clone();
        bytes.extend_from_slice(b"{\"change_id\":\"promotion-2\"");
        std::fs::write(&path, bytes)?;

        let log = JsonlChangeLog::new(path.clone())?;
        assert_eq!(log.len(), 1);
        assert_eq!(std::fs::read(path)?, durable);
        Ok(())
    }

    /// 验证缺少换行符但有效的最后一条记录会被保留并补齐换行。
    #[test]
    fn valid_final_record_without_newline_is_preserved_and_terminated()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("change-log.jsonl");
        let mut expected = serde_json::to_vec(&sample_entry("promotion-1", "rule/one"))?;
        std::fs::write(&path, &expected)?;

        let log = JsonlChangeLog::new(path.clone())?;
        expected.push(b'\n');
        assert_eq!(log.len(), 1);
        assert_eq!(std::fs::read(path)?, expected);
        Ok(())
    }

    /// 验证稳定变更 ID 具备幂等性，并拒绝内容冲突。
    #[test]
    fn stable_change_id_is_idempotent_and_rejects_content_collision()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("change-log.jsonl");
        let log = JsonlChangeLog::new(path.clone())?;
        let entry = sample_entry("promotion-stable", "rule/one");

        assert_eq!(
            log.record_idempotent(entry.clone())?,
            ChangeRecordOutcome::Appended
        );
        assert_eq!(
            log.record_idempotent(entry.clone())?,
            ChangeRecordOutcome::AlreadyRecorded
        );
        assert_eq!(log.len(), 1);

        let mut conflicting = entry;
        conflicting.reason = "different promotion payload".to_string();
        assert!(log.record_idempotent(conflicting).is_err());
        assert_eq!(log.len(), 1);
        assert_eq!(JsonlChangeLog::new(path.clone())?.len(), 1);

        let durable = std::fs::read(&path)?;
        let mut duplicated = durable.clone();
        duplicated.extend_from_slice(&durable);
        std::fs::write(&path, duplicated)?;
        assert!(JsonlChangeLog::new(path).is_err());
        Ok(())
    }
}
