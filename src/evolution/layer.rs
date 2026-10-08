/*! 内存分层管理器 —— 双层（热/温）内存组织结构。
 *
 * （stage4）冷层已被移除：陈旧度现在体现为一种召回衰减权重，
 * 而不是一个独立的命名空间。已归档的记忆（`MemoryStatus::Archived`）
 * 仍留在温/统一命名空间中，并保持可被召回（伴随衰减）。
 *
 * # 分层
 *
 * | 层级 | 存储 | 命名空间 | 用途 |
 * |-------|---------|-----------|---------|
 * | **热层（Hot）** | `.echo-agent/MEMORY.md`（YAML frontmatter + markdown 正文） | 文件 | 始终加载进上下文，最多约 2000 tokens |
 * | **温层（Warm）** | Store KV | `["agent", "memories"]` | 按需通过搜索获取；已归档条目也留在这里（stage4：冷层已移除） |
 *
 * # 热层 MEMORY.md 格式
 *
 * ```markdown
 * ---
 * entries:
 *   - key: build_java8
 *     memory_type: debugging_lesson
 *     confidence: 0.90
 *     stability: 0.80
 *     source: error_resolution
 *     topic: build
 *     risk: low
 *     last_promoted: "2026-06-15T10:30:00Z"
 * ---
 *
 * - **[build/java8]** Maven compile requires JAVA_HOME pointing to JDK 8.
 * - **[style/concise]** User prefers concise code comments.
 * ```
 *
 * # 晋升/降级
 *
 * 当记忆满足晋升资格条件（`MemoryMeta::is_hot_eligible()`）且热层仍有容量时，
 * 会从温层晋升到热层。当热层超出其 token 预算时，
 * 优先级最低的条目会依据降级分数被降回温层。
 */

use echo_core::memory::store::Store;
use echo_core::memory::types::{
    MemoryMeta, MemoryRisk, MemorySource, MemoryStatus, MemoryType, TypedMemoryValue,
};
use echo_state::memory::typed_store::{MemoryFilter, TypedMemoryEntry, TypedMemoryStore};
use futures::future::BoxFuture;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::sync::Mutex;

use super::audit::{ChangeEntryBuilder, ChangeFilter, ChangeLog, ChangeType, EntityType};
use super::mutation::{
    HotValue, MemoryOperation, MemoryOperationBatch, MemoryOperationHistory,
    MemoryOperationJournal, MemoryOperationOrigin,
};
use super::review::{
    AppliedMemoryMerge, ConflictDetector, ConflictGroup, MemoryConflictProposal,
    MemoryMergeSnapshot, MemoryMerger, MergeResult, ordered_conflict_entries,
};
use super::security::{EvolutionSecurityGuard, InputTrustLevel};
use echo_core::error::{ConfigError, ReactError};

/// 分层操作结果的别名。
type Result<T> = std::result::Result<T, ReactError>;

/// 构造一个合并计划相关的配置错误。
fn merge_plan_error(message: impl Into<String>) -> ReactError {
    ReactError::Config(Box::new(ConfigError::ConfigFileError(message.into())))
}

/// 构造一个"审阅过的提案已过期"错误。
fn stale_merge_plan_error(message: impl Into<String>) -> ReactError {
    ReactError::Memory(Box::new(echo_core::error::MemoryError::StaleProposal(
        message.into(),
    )))
}

/// 判断某个错误是否意味着一个已审阅的记忆提案已不再是最新状态。
pub fn is_stale_memory_proposal_error(error: &ReactError) -> bool {
    matches!(
        error,
        ReactError::Memory(memory_error)
            if matches!(
                memory_error.as_ref(),
                echo_core::error::MemoryError::StaleProposal(_)
            )
    )
}

// ── 常量 ──────────────────────────────────────────────────────────

/**
 * 统一类型化记忆存储所使用的命名空间（stage4：温层+冷层+L3 均已合并到这里；
 * Archived 状态取代了原先的冷层命名空间）。
 */
pub const WARM_NAMESPACE: &[&str] = &["agent", "memories"];

/**
 * 独立冷层/归档层的可选命名空间（作为 pub API 保留，供对齐 Letta/MemGPT
 * 归档记忆理念——按需召回、不主动加载进上下文——的消费者使用）。
 * 默认产品路径改用 `WARM_NAMESPACE` + `MemoryStatus::Archived`；
 * 保留此常量是为了让需要独立冷存储的消费者可以选择接入。
 */
pub const COLD_NAMESPACE: &[&str] = &["agent", "cold_memories"];

/// 热层的最大 token 预算（MEMORY.md 正文）。
const HOT_TOKEN_BUDGET: usize = 2000;

// ── MemoryLayer ────────────────────────────────────────────────────────

/// 一个内存层级。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryLayer {
    /// 始终加载进上下文（MEMORY.md）。最多约 2000 tokens。
    Hot,
    /**
     * 通过 Store KV 搜索按需获取。已归档的记忆（`MemoryStatus::Archived`）
     * 同样存放在这个统一命名空间中——stage4 移除了独立的 Cold 层，
     * 陈旧度是一种召回衰减权重，而不是层级迁移。
     */
    Warm,
    /**
     * 用于长期归档的可选第三层（很少被加载，置信度低/已陈旧）。
     * stage4 已将冷层折叠进 `Warm`+`Archived`，但此变体以及
     * [`COLD_NAMESPACE`] 作为 pub API 被保留下来，供需要独立冷层
     * （对齐 Letta/MemGPT 归档记忆理念）的消费者使用。默认产品路径不使用它。
     */
    Cold,
}

impl std::fmt::Display for MemoryLayer {
    /// 以小写字符串形式输出层级名称。
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Hot => write!(f, "hot"),
            Self::Warm => write!(f, "warm"),
            Self::Cold => write!(f, "cold"),
        }
    }
}

// ── HotEntryMeta ───────────────────────────────────────────────────────

/**
 * MEMORY.md 热层中单条记忆条目的元数据。
 *
 * 与 markdown 正文一起存储在 YAML frontmatter 中。
 */
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HotEntryMeta {
    /// 该记忆的键标识符。
    pub key: String,
    /// 记忆类型分类。
    pub memory_type: MemoryType,
    /// 置信度分数（0.0-1.0）。
    pub confidence: f32,
    /// 稳定性分数（0.0-1.0）。
    pub stability: f32,
    /// 温层排序所使用的召回重要性。
    #[serde(default = "default_hot_recall_weight")]
    pub recall_weight: f32,
    /// 该记忆的来源。
    pub source: MemorySource,
    /// 主题分类。
    pub topic: String,
    /// 风险等级。
    #[serde(default)]
    pub risk: MemoryRisk,
    /// 晋升前累积的语义修订次数。
    #[serde(default)]
    pub revision_count: u32,
    /// 晋升前累积的成功召回次数。
    #[serde(default)]
    pub recall_count: u32,
    /// 最近一次成功召回的时间戳。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_recalled_at: Option<u64>,
    /** 非平凡内容会以 JSON 字符串形式存储在其正文的列表项中，使空白与换行
     * 在 MEMORY.md 往返读写中得以保留。没有该标记的旧列表项仍按纯文本处理。
     */
    #[serde(default, skip_serializing_if = "is_false")]
    pub content_json: bool,
    /// 该条目被晋升到热层的时间（ISO 8601）。
    pub last_promoted: String,
}

/// 判断布尔值是否为 `false`（用于 serde 的 `skip_serializing_if`）。
fn is_false(value: &bool) -> bool {
    !*value
}

impl HotEntryMeta {
    /// 转换为 MemoryMeta 以便重建。
    fn to_memory_meta(&self) -> MemoryMeta {
        let mut meta = MemoryMeta::new(self.memory_type, self.source, &self.topic)
            .with_confidence(self.confidence)
            .with_stability(self.stability)
            .with_recall_weight(self.recall_weight)
            .with_risk(self.risk);
        meta.revision_count = self.revision_count;
        meta.recall_count = self.recall_count;
        meta.last_recalled_at = self.last_recalled_at;
        meta
    }
}

/// 默认的热层召回权重。
fn default_hot_recall_weight() -> f32 {
    0.5
}

// ── MemoryFile ─────────────────────────────────────────────────────────

/// MEMORY.md 的解析结构。
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MemoryFile {
    /// 来自 YAML frontmatter 的条目。
    pub entries: Vec<HotEntryMeta>,
    /// markdown 正文（人类可读，会被加载进上下文）。
    pub body: String,
}

// ── LayerChangeResult ──────────────────────────────────────────────────

/// 一次晋升或降级操作的结果。
#[derive(Debug, Clone)]
pub struct LayerChangeResult {
    /// 被移动的键。
    pub key: String,
    /// 移动的方向（来源层）。
    pub from_layer: MemoryLayer,
    /// 移动的方向（目标层）。
    pub to_layer: MemoryLayer,
    /// 本次变更的原因。
    pub reason: String,
}

/// 一个类型化、稳定的目标，用于描述之后的一次记忆回滚。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryRollbackTarget {
    /// 回滚包含该 ChangeLog 条目的那个持久化批次。
    ChangeId(String),
    /// 回滚一个完整的持久化日志批次。
    BatchId(String),
}

impl MemoryRollbackTarget {
    /// 以某个 ChangeLog 条目 ID 构造一个回滚目标。
    pub fn change_id(value: impl Into<String>) -> Self {
        Self::ChangeId(value.into())
    }

    /// 以某个日志批次 ID 构造一个回滚目标。
    pub fn batch_id(value: impl Into<String>) -> Self {
        Self::BatchId(value.into())
    }
}

/// 对一个无法安全应用的回滚目标给出的稳定说明。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryRollbackConflict {
    /// 冲突所指向的目标批次 ID（若可确定）。
    pub target_batch_id: Option<String>,
    /// 受此次冲突影响的记忆键列表。
    pub affected_keys: Vec<String>,
    /// 冲突原因说明。
    pub reason: String,
}

/// 对不可用或已过期的日志历史给出的稳定说明。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryRollbackHistoryUnavailable {
    /// 本次查询所请求的回滚目标。
    pub target: MemoryRollbackTarget,
    /// 历史不可用的原因说明。
    pub reason: String,
}

/// 在创建逆向批次之前，对之后的一次回滚给出的预览。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryRollbackPreview {
    /// 将被回滚的目标批次 ID。
    pub target_batch_id: String,
    /// 目标批次所处的日志世代号。
    pub target_generation: u64,
    /// 受此次回滚影响的记忆键列表。
    pub affected_keys: Vec<String>,
}

/// 回滚预览的结果：就绪、存在冲突，或历史不可用。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryRollbackPreviewOutcome {
    /// 可以安全执行回滚。
    Ready(MemoryRollbackPreview),
    /// 目标与当前状态冲突，无法安全回滚。
    Conflict(MemoryRollbackConflict),
    /// 目标所需的历史记录不可用。
    HistoryUnavailable(MemoryRollbackHistoryUnavailable),
}

/// 一次已提交的逆向批次的持久化回执。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryRollbackReceipt {
    /// 发起本次回滚请求的幂等请求 ID。
    pub request_id: String,
    /// 被回滚的目标批次 ID。
    pub target_batch_id: String,
    /// 本次回滚生成的逆向批次 ID。
    pub inverse_batch_id: String,
    /// 目标批次所处的日志世代号。
    pub target_generation: u64,
    /// 受此次回滚影响的记忆键列表。
    pub affected_keys: Vec<String>,
}

/// 一次记忆回滚请求的最终结果。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryRollbackOutcome {
    /// 本次回滚已成功应用。
    Applied(MemoryRollbackReceipt),
    /// 相同的请求 ID 此前已经应用过，直接返回原回执。
    AlreadyApplied(MemoryRollbackReceipt),
    /// 请求的目标与当前状态冲突，回滚未执行。
    Conflict(MemoryRollbackConflict),
    /// 目标所需的历史记录不可用，回滚未执行。
    HistoryUnavailable(MemoryRollbackHistoryUnavailable),
}

// ── MemoryLayerManager ─────────────────────────────────────────────────

/**
 * 管理双层内存层级系统（stage4：冷层已移除）。
 *
 * - **热层**：`.echo-agent/MEMORY.md`（YAML frontmatter + markdown 正文）。
 *   始终加载进上下文，最多约 2000 tokens。
 * - **温层**：[`WARM_NAMESPACE`] = `["agent", "memories"]` 下的 Store KV。
 *   按需通过搜索获取；已归档条目也留在这里（可被召回，伴随衰减）。
 *   关于冷层为何消失，参见模块级文档。
 */
pub struct MemoryLayerManager {
    /// MEMORY.md 文件的路径。
    hot_path: PathBuf,
    /**
     * 进程内锁，序列化所有对 MEMORY.md 的读-改-写操作，使同一个
     * `MemoryLayerManager` 上的两次调用不会交错执行、丢失更新（P0-3）。
     * 跨进程安全性由 `with_locked_hot_file` 中的 fs2 建议锁提供。
     */
    hot_lock: Mutex<()>,
    /// 温/冷层所使用的类型化存储。
    typed_store: TypedMemoryStore,
    /// 用于审计追踪的变更日志。
    change_log: Box<dyn ChangeLog>,
    /// 打开日志失败仍然是一个硬性的变更错误，尽管这个旧式构造函数无法返回 Result。
    operation_journal: std::result::Result<MemoryOperationJournal, String>,
    /// 序列化所有操作日志读写的进程内锁。
    operation_lock: Arc<Mutex<()>>,
    /// 标记是否已完成一次待处理操作的对账。
    reconciled: AtomicBool,
    /// 用于写入时检查（密钥扫描、注入检测、限流）的安全守卫。
    security_guard: EvolutionSecurityGuard,
    /// 在持久化演化变更后被调用的可选观察者。
    evolution_observer: Option<Arc<dyn EvolutionObserver>>,
}

/// 用于持久化记忆与技能演化事件的观察者。
pub trait EvolutionObserver: Send + Sync {
    /// 在一条记忆到达分层存储之后被调用。
    fn on_memory_write<'a>(&'a self, _key: &'a str, _source: &'a str) -> BoxFuture<'a, ()> {
        Box::pin(async {})
    }

    /// 在一次内存层级迁移被完全持久化之后被调用。
    fn on_memory_layer_change<'a>(
        &'a self,
        _key: &'a str,
        _from_layer: &'a str,
        _to_layer: &'a str,
    ) -> BoxFuture<'a, ()> {
        Box::pin(async {})
    }

    /// 在一个新的技能候选被持久化之后被调用。
    fn on_skill_candidate_detected<'a>(&'a self, _skill_name: &'a str) -> BoxFuture<'a, ()> {
        Box::pin(async {})
    }

    /// 在一次技能健康检查报告被计算出来之后被调用。
    fn on_skill_health_check<'a>(&'a self, _skill_name: &'a str) -> BoxFuture<'a, ()> {
        Box::pin(async {})
    }
}

/**
 * 持有 MEMORY.md 热层排他锁以及刚加载出的 `MemoryFile` 的守卫。
 *
 * 由 [`MemoryLayerManager::lock_hot_file`] 创建。在异步操作期间持有它，
 * 然后调用 [`commit`](Self::commit) 持久化，或者直接丢弃以放弃变更。
 * 进程内 Mutex 与跨进程 flock 都会在丢弃时释放（flock 在进程被
 * kill -9 杀死时也会由操作系统自动释放，因此不会卡死其他进程）。
 */
struct HotFileGuard<'a> {
    /// 所属的管理器（仅测试使用）。
    #[cfg(test)]
    manager: &'a MemoryLayerManager,
    /// 已加载、可能会被修改的 MEMORY.md 结构。
    file: MemoryFile,
    /**
     * 进程内 Mutex 守卫（`tokio::sync::MutexGuard`，它是 `Send` 的，
     * 因此该守卫可以跨越 `.await` 点持有）。在该守卫释放之前，
     * 阻止同一管理器上的其他任务进入临界区。
     */
    _inproc: tokio::sync::MutexGuard<'a, ()>,
    /**
     * 持有跨进程建议锁；丢弃时释放。该 flock 在文件描述符关闭时
     * 也会由操作系统释放（对 kill 是安全的）。
     */
    _lock_file: std::fs::File,
}

impl<'a> HotFileGuard<'a> {
    /// 在仍持有锁的情况下，将（可能已被修改的）`MemoryFile` 持久化回磁盘。
    #[cfg(test)]
    fn commit(self) -> std::io::Result<()> {
        self.manager.write_memory_file(&self.file)
        // self 在此处被丢弃：进程内守卫与锁文件的 fd 均被释放 → 两把锁都被释放。
    }
}

impl MemoryLayerManager {
    /**
     * 创建一个新的分层管理器。
     *
     * # 参数
     * * `echo_agent_dir` — `.echo-agent/` 目录的路径（热层 MEMORY.md 将位于其中）。
     * * `store` — 温/冷层所使用的底层 Store 实现。
     * * `change_log` — 用于记录所有变更的审计日志。
     */
    pub fn new(
        echo_agent_dir: PathBuf,
        store: Arc<dyn Store>,
        change_log: Box<dyn ChangeLog>,
    ) -> Self {
        let hot_path = echo_agent_dir.join("MEMORY.md");
        let operation_journal =
            MemoryOperationJournal::open(&echo_agent_dir).map_err(|error| error.to_string());
        let operation_lock = operation_journal
            .as_ref()
            .map(MemoryOperationJournal::serial)
            .unwrap_or_else(|_| Arc::new(Mutex::new(())));
        Self {
            hot_path,
            hot_lock: Mutex::new(()),
            typed_store: TypedMemoryStore::new(store),
            change_log,
            operation_journal,
            operation_lock,
            reconciled: AtomicBool::new(false),
            security_guard: EvolutionSecurityGuard::default_config(),
            evolution_observer: None,
        }
    }

    /**
     * 构造一个在恢复日志缺失或损坏时会直接拒绝的管理器。
     * 在把 Store 交给读取方之前，请先调用 [`Self::reconcile_pending`]。
     */
    pub fn try_new(
        echo_agent_dir: PathBuf,
        store: Arc<dyn Store>,
        change_log: Box<dyn ChangeLog>,
    ) -> Result<Self> {
        let manager = Self::new(echo_agent_dir, store, change_log);
        manager.operation_journal()?;
        Ok(manager)
    }

    /// 配置一个在持久化演化事件成功后被调用的观察者。
    pub fn with_evolution_observer(mut self, observer: Arc<dyn EvolutionObserver>) -> Self {
        self.evolution_observer = Some(observer);
        self
    }

    /// 获取操作日志的引用；日志不可用时返回错误。
    fn operation_journal(&self) -> Result<&MemoryOperationJournal> {
        self.operation_journal.as_ref().map_err(|error| {
            ReactError::Other(format!("memory operation journal unavailable: {error}"))
        })
    }

    /// 将内容与元数据序列化为一份类型化记忆投影值。
    fn typed_value(content: &str, meta: MemoryMeta) -> Result<serde_json::Value> {
        TypedMemoryValue::new(content, meta)
            .to_value()
            .map_err(|error| {
                ReactError::Other(format!("failed to serialize memory projection: {error}"))
            })
    }

    /// 在已解析的热层文件中查找指定键的当前值。
    fn hot_value(file: &MemoryFile, key: &str) -> Option<HotValue> {
        file.entries
            .iter()
            .find(|entry| entry.key == key)
            .map(|meta| HotValue {
                meta: meta.clone(),
                content: extract_hot_value_content(&file.body, meta),
            })
    }

    /// 在热层文件中写入、替换或删除（`value` 为 `None` 时）指定键的条目。
    fn set_hot_value(file: &mut MemoryFile, key: &str, value: Option<&HotValue>) -> Result<()> {
        file.entries.retain(|entry| entry.key != key);
        let pattern = format!("- **[{key}]**");
        file.body = file
            .body
            .lines()
            .filter(|line| !line.starts_with(&pattern))
            .collect::<Vec<_>>()
            .join("\n");
        if !file.body.is_empty() && !file.body.ends_with('\n') {
            file.body.push('\n');
        }
        if let Some(value) = value {
            let content = if value.meta.content_json {
                serde_json::to_string(&value.content).map_err(|error| {
                    ReactError::Other(format!("failed to encode hot memory content: {error}"))
                })?
            } else {
                value.content.clone()
            };
            file.entries.push(value.meta.clone());
            file.body.push_str(&format!("- **[{key}]** {content}\n"));
        }
        Ok(())
    }

    /// 归一化一份温层投影，剔除召回遥测字段以便做语义比较。
    fn semantic_warm(value: &serde_json::Value) -> serde_json::Value {
        let mut normalized = value.clone();
        if let Some(meta) = normalized
            .get_mut("meta")
            .and_then(serde_json::Value::as_object_mut)
        {
            meta.remove("recall_count");
            meta.remove("last_recalled_at");
        }
        normalized
    }

    /// 判断两份温层投影在剔除召回遥测字段后是否语义相等。
    fn warm_matches(
        current: Option<&serde_json::Value>,
        expected: Option<&serde_json::Value>,
    ) -> bool {
        match (current, expected) {
            (Some(current), Some(expected)) => {
                Self::semantic_warm(current) == Self::semantic_warm(expected)
            }
            (None, None) => true,
            _ => false,
        }
    }

    /// 将当前投影中的召回遥测字段（取较大值）合并进目标投影，避免被覆盖丢失。
    fn preserve_recall_telemetry(
        current: Option<&serde_json::Value>,
        desired: &serde_json::Value,
    ) -> serde_json::Value {
        let mut result = desired.clone();
        if let (Some(current_meta), Some(target_meta)) = (
            current
                .and_then(|value| value.get("meta"))
                .and_then(serde_json::Value::as_object),
            result
                .get_mut("meta")
                .and_then(serde_json::Value::as_object_mut),
        ) {
            for field in ["recall_count", "last_recalled_at"] {
                let old = current_meta.get(field).and_then(serde_json::Value::as_u64);
                let new = target_meta.get(field).and_then(serde_json::Value::as_u64);
                if let Some(value) = old.into_iter().chain(new).max() {
                    target_meta.insert(field.to_owned(), serde_json::Value::from(value));
                }
            }
        }
        result
    }

    /// 应用单个记忆操作（以其自身作为唯一的历史记录）。
    async fn apply_operation(&self, operation: &MemoryOperation) -> Result<()> {
        self.apply_operation_with_history(operation, std::slice::from_ref(operation))
            .await
    }

    /// 在给定操作历史的校验下，将一个记忆操作应用到热层与温层。
    async fn apply_operation_with_history(
        &self,
        operation: &MemoryOperation,
        history: &[MemoryOperation],
    ) -> Result<()> {
        let mut hot = self.lock_hot_file().await.map_err(ReactError::from)?;
        let current_hot = Self::hot_value(&hot.file, &operation.key);
        let current_warm = self
            .typed_store
            .inner()
            .get(WARM_NAMESPACE, &operation.key)
            .await?
            .map(|entry| entry.value);
        if !history
            .iter()
            .any(|prior| current_hot == prior.hot_before || current_hot == prior.hot_after)
        {
            return Err(ReactError::Other(format!(
                "memory operation {} conflicts with hot entry {}",
                operation.id, operation.key
            )));
        }
        if !history.iter().any(|prior| {
            Self::warm_matches(current_warm.as_ref(), prior.warm_before.as_ref())
                || Self::warm_matches(current_warm.as_ref(), prior.warm_after.as_ref())
        }) {
            return Err(ReactError::Other(format!(
                "memory operation {} conflicts with warm entry {}",
                operation.id, operation.key
            )));
        }

        /* 层级迁移时优先写入目标端。每一个中间状态都可以从持久化的意图中
         * 修复回来，包括一个被取消的 future。 */
        if operation.hot_after.is_some() && current_hot != operation.hot_after {
            Self::set_hot_value(&mut hot.file, &operation.key, operation.hot_after.as_ref())?;
            self.write_memory_file(&hot.file)
                .map_err(ReactError::from)?;
        }
        if let Some(after) = &operation.warm_after
            && !Self::warm_matches(current_warm.as_ref(), Some(after))
        {
            self.typed_store
                .inner()
                .put(
                    WARM_NAMESPACE,
                    &operation.key,
                    Self::preserve_recall_telemetry(current_warm.as_ref(), after),
                )
                .await?;
        }
        if operation.hot_after.is_none() && current_hot.is_some() {
            Self::set_hot_value(&mut hot.file, &operation.key, None)?;
            self.write_memory_file(&hot.file)
                .map_err(ReactError::from)?;
        }
        if operation.warm_after.is_none() && current_warm.is_some() {
            self.typed_store
                .inner()
                .delete(WARM_NAMESPACE, &operation.key)
                .await?;
        }
        drop(hot);
        Ok(())
    }

    /// 结算一个操作批次：应用每个操作、幂等记录审计日志，并将批次标记为已结算。
    async fn settle_batch(&self, batch: &MemoryOperationBatch) -> Result<()> {
        for operation in &batch.operations {
            self.apply_operation(operation).await?;
            self.change_log.record_idempotent(operation.audit.clone())?;
        }
        self.operation_journal()?.settle(&batch.id)
    }

    /// 在已持有序列化锁的前提下，对账所有待处理的记忆操作。
    async fn reconcile_pending_locked(&self) -> Result<()> {
        let history = self.operation_journal()?.history()?;
        let mut by_key = BTreeMap::<String, Vec<MemoryOperation>>::new();
        for (batch, _) in &history {
            for operation in &batch.operations {
                by_key
                    .entry(operation.key.clone())
                    .or_default()
                    .push(operation.clone());
            }
        }
        for operations in by_key.values() {
            let latest = operations.last().ok_or_else(|| {
                ReactError::Other("memory operation history has an empty key series".into())
            })?;
            self.apply_operation_with_history(latest, operations)
                .await?;
        }
        for (batch, settled) in history {
            if !settled {
                for operation in &batch.operations {
                    self.change_log.record_idempotent(operation.audit.clone())?;
                }
                self.operation_journal()?.settle(&batch.id)?;
            }
        }
        self.reconciled.store(true, Ordering::Release);
        Ok(())
    }

    /**
     * 在重启或发生不确定失败后，完成已准备但尚未结算的记忆操作。
     * 重复调用会以相同的身份重放，绝不会追加重复的审计记录。
     */
    pub async fn reconcile_pending(&self) -> Result<()> {
        let _serial = self.operation_lock.lock().await;
        self.reconcile_pending_locked().await
    }

    /// 解析一个回滚目标当前对应的日志批次 ID。
    fn rollback_target_batch_id(
        &self,
        target: &MemoryRollbackTarget,
        history: &[MemoryOperationHistory],
    ) -> Result<Option<String>> {
        match target {
            MemoryRollbackTarget::BatchId(batch_id) => Ok(history
                .iter()
                .find(|item| item.batch.id == *batch_id)
                .map(|item| item.batch.id.clone())),
            MemoryRollbackTarget::ChangeId(change_id) => {
                let known = self
                    .change_log
                    .query(&ChangeFilter::new())?
                    .into_iter()
                    .any(|entry| entry.change_id == *change_id);
                if !known {
                    return Ok(None);
                }
                Ok(history
                    .iter()
                    .find(|item| {
                        item.batch
                            .operations
                            .iter()
                            .any(|operation| operation.audit.change_id == *change_id)
                    })
                    .map(|item| item.batch.id.clone()))
            }
        }
    }

    /// 解析一个回滚目标：返回可回滚的历史条目，或说明为何无法回滚的结果。
    fn rollback_resolution<'a>(
        &self,
        target: &MemoryRollbackTarget,
        history: &'a [MemoryOperationHistory],
    ) -> Result<std::result::Result<&'a MemoryOperationHistory, MemoryRollbackPreviewOutcome>> {
        let Some(batch_id) = self.rollback_target_batch_id(target, history)? else {
            return Ok(Err(MemoryRollbackPreviewOutcome::HistoryUnavailable(
                MemoryRollbackHistoryUnavailable {
                    target: target.clone(),
                    reason: "target is absent from the retained ChangeLog or operation journal"
                        .to_string(),
                },
            )));
        };
        let Some(item) = history.iter().find(|item| item.batch.id == batch_id) else {
            return Ok(Err(MemoryRollbackPreviewOutcome::HistoryUnavailable(
                MemoryRollbackHistoryUnavailable {
                    target: target.clone(),
                    reason: "target journal batch is unavailable".to_string(),
                },
            )));
        };
        let keys = item
            .batch
            .operations
            .iter()
            .map(|operation| operation.key.clone())
            .collect::<Vec<_>>();
        if !item.settled {
            return Ok(Err(MemoryRollbackPreviewOutcome::HistoryUnavailable(
                MemoryRollbackHistoryUnavailable {
                    target: target.clone(),
                    reason: "target batch is not settled".to_string(),
                },
            )));
        }
        for key in &keys {
            let latest = history
                .iter()
                .filter(|candidate| {
                    candidate
                        .batch
                        .operations
                        .iter()
                        .any(|operation| operation.key == *key)
                })
                .max_by_key(|candidate| candidate.generation);
            if latest.map(|candidate| candidate.batch.id.as_str()) != Some(batch_id.as_str()) {
                return Ok(Err(MemoryRollbackPreviewOutcome::Conflict(
                    MemoryRollbackConflict {
                        target_batch_id: Some(batch_id),
                        affected_keys: keys.clone(),
                        reason: format!(
                            "target is not the latest journal generation for memory key {key}"
                        ),
                    },
                )));
            }
        }
        Ok(Ok(item))
    }

    /// 预览之后的一次回滚，但不会创建逆向批次。
    pub async fn preview_rollback(
        &self,
        target: &MemoryRollbackTarget,
    ) -> Result<MemoryRollbackPreviewOutcome> {
        let _serial = self.operation_lock.lock().await;
        self.reconcile_pending_locked().await?;
        let history = self.operation_journal()?.history_with_generations()?;
        match self.rollback_resolution(target, &history)? {
            Ok(item) => Ok(MemoryRollbackPreviewOutcome::Ready(MemoryRollbackPreview {
                target_batch_id: item.batch.id.clone(),
                target_generation: item.generation,
                affected_keys: item
                    .batch
                    .operations
                    .iter()
                    .map(|operation| operation.key.clone())
                    .collect(),
            })),
            Err(outcome) => Ok(outcome),
        }
    }

    /// 为偏好使用更短回滚词汇的调用方提供的别名。
    pub async fn preview(
        &self,
        target: &MemoryRollbackTarget,
    ) -> Result<MemoryRollbackPreviewOutcome> {
        self.preview_rollback(target).await
    }

    /// 构造一份回滚回执。
    fn rollback_receipt(
        request_id: &str,
        target_batch_id: &str,
        target_generation: u64,
        inverse: &MemoryOperationHistory,
    ) -> MemoryRollbackReceipt {
        MemoryRollbackReceipt {
            request_id: request_id.to_owned(),
            target_batch_id: target_batch_id.to_owned(),
            inverse_batch_id: inverse.batch.id.clone(),
            target_generation,
            affected_keys: inverse
                .batch
                .operations
                .iter()
                .map(|operation| operation.key.clone())
                .collect(),
        }
    }

    /// 计算与给定变更类型相对应的逆向变更类型。
    fn inverse_change_type(change_type: ChangeType) -> ChangeType {
        match change_type {
            ChangeType::Create => ChangeType::Delete,
            ChangeType::Delete => ChangeType::Create,
            ChangeType::Promote => ChangeType::Demote,
            ChangeType::Demote => ChangeType::Promote,
            ChangeType::Update | ChangeType::Merge => ChangeType::Update,
        }
    }

    /// 构造一份可机读的回滚前/后投影摘要，用于审计记录。
    fn rollback_projection_summary(
        warm: &Option<serde_json::Value>,
        hot: &Option<HotValue>,
        original_change_id: &str,
        target_batch_id: &str,
        target_generation: u64,
    ) -> serde_json::Value {
        let layer = if hot.is_some() {
            "hot"
        } else if warm.is_some() {
            "warm"
        } else {
            "absent"
        };
        serde_json::json!({
            "layer": layer,
            "warm": warm,
            "hot": hot,
            "rollback_of": original_change_id,
            "target_batch_id": target_batch_id,
            "target_generation": target_generation,
        })
    }

    /**
     * 为一个已结算的记忆批次创建并结算一个持久化的逆向批次。
     *
     * 请求 ID 会被持久化在逆向批次的血统信息中。重放同一个请求会返回
     * `AlreadyApplied`；将其复用于另一个目标则会直接失败拒绝。每一次逆向
     * 都是一个新批次，因此"回滚的回滚"只是一次使用新请求 ID 的普通回滚。
     */
    pub async fn rollback_memory(
        &self,
        request_id: &str,
        target: MemoryRollbackTarget,
    ) -> Result<MemoryRollbackOutcome> {
        if request_id.is_empty() {
            return Err(ReactError::Other(
                "rollback request ID must not be empty".into(),
            ));
        }
        let _serial = self.operation_lock.lock().await;
        self.reconcile_pending_locked().await?;
        let history = self.operation_journal()?.history_with_generations()?;

        if let Some(existing) = history.iter().find(|item| {
            item.batch
                .origin
                .as_ref()
                .is_some_and(|origin| origin.request_id == request_id)
        }) {
            let requested_batch = self.rollback_target_batch_id(&target, &history)?;
            let origin = existing.batch.origin.as_ref().ok_or_else(|| {
                ReactError::Other("rollback lineage disappeared from journal history".into())
            })?;
            if requested_batch.as_deref() != Some(origin.target_batch_id.as_str()) {
                return Ok(MemoryRollbackOutcome::Conflict(MemoryRollbackConflict {
                    target_batch_id: requested_batch,
                    affected_keys: existing
                        .batch
                        .operations
                        .iter()
                        .map(|operation| operation.key.clone())
                        .collect(),
                    reason: "request ID was already used for a different rollback target"
                        .to_string(),
                }));
            }
            let receipt = Self::rollback_receipt(
                request_id,
                &origin.target_batch_id,
                origin.target_generation,
                existing,
            );
            return if existing.settled {
                Ok(MemoryRollbackOutcome::AlreadyApplied(receipt))
            } else {
                Err(ReactError::Other(
                    "rollback request remained unsettled after reconciliation".into(),
                ))
            };
        }

        let item = match self.rollback_resolution(&target, &history)? {
            Ok(item) => item,
            Err(MemoryRollbackPreviewOutcome::Conflict(conflict)) => {
                return Ok(MemoryRollbackOutcome::Conflict(conflict));
            }
            Err(MemoryRollbackPreviewOutcome::HistoryUnavailable(unavailable)) => {
                return Ok(MemoryRollbackOutcome::HistoryUnavailable(unavailable));
            }
            Err(MemoryRollbackPreviewOutcome::Ready(_)) => {
                return Err(ReactError::Other(
                    "rollback resolution returned an invalid ready state".into(),
                ));
            }
        };
        let inverse_batch_id = uuid::Uuid::new_v4().to_string();
        let mut operations = Vec::with_capacity(item.batch.operations.len());
        for operation in &item.batch.operations {
            let operation_id = uuid::Uuid::new_v4().to_string();
            let audit = ChangeEntryBuilder::new(
                EntityType::Memory,
                &operation.key,
                Self::inverse_change_type(operation.audit.change_type),
            )
            .before(Self::rollback_projection_summary(
                &operation.warm_after,
                &operation.hot_after,
                &operation.audit.change_id,
                &item.batch.id,
                item.generation,
            ))
            .after(Self::rollback_projection_summary(
                &operation.warm_before,
                &operation.hot_before,
                &operation.audit.change_id,
                &item.batch.id,
                item.generation,
            ))
            .reason(format!(
                "rollback of {:?} memory change in batch {}",
                operation.audit.change_type, item.batch.id
            ))
            .trigger("memory_rollback")
            .build_with(operation_id.clone(), chrono::Utc::now());
            operations.push(MemoryOperation {
                id: operation_id,
                key: operation.key.clone(),
                warm_before: operation.warm_after.clone(),
                warm_after: operation.warm_before.clone(),
                hot_before: operation.hot_after.clone(),
                hot_after: operation.hot_before.clone(),
                audit,
            });
        }
        let inverse = MemoryOperationBatch {
            id: inverse_batch_id,
            operations,
            origin: Some(MemoryOperationOrigin {
                request_id: request_id.to_owned(),
                target_batch_id: item.batch.id.clone(),
                target_generation: item.generation,
            }),
        };
        self.operation_journal()?.prepare_batch(inverse.clone())?;
        self.settle_batch(&inverse).await?;
        let inverse_history = self
            .operation_journal()?
            .history_with_generations()?
            .into_iter()
            .find(|candidate| candidate.batch.id == inverse.id)
            .ok_or_else(|| ReactError::Other("rollback receipt batch disappeared".into()))?;
        Ok(MemoryRollbackOutcome::Applied(Self::rollback_receipt(
            request_id,
            &item.batch.id,
            item.generation,
            &inverse_history,
        )))
    }

    /// 为偏好使用更短回滚词汇的调用方提供的别名。
    pub async fn rollback(
        &self,
        request_id: &str,
        target: MemoryRollbackTarget,
    ) -> Result<MemoryRollbackOutcome> {
        self.rollback_memory(request_id, target).await
    }

    /// 在校验期望的当前状态后，对一个记忆键执行一次层级/内容迁移。
    async fn transition(
        &self,
        key: &str,
        warm_after: Option<serde_json::Value>,
        hot_after: Option<HotValue>,
        builder: ChangeEntryBuilder,
        expected: Option<(MemoryLayer, &TypedMemoryEntry)>,
    ) -> Result<()> {
        let _serial = self.operation_lock.lock().await;
        self.reconcile_pending_locked().await?;
        let hot = self.lock_hot_file().await.map_err(ReactError::from)?;
        let hot_before = Self::hot_value(&hot.file, key);
        let warm_before = self
            .typed_store
            .inner()
            .get(WARM_NAMESPACE, key)
            .await?
            .map(|entry| entry.value);
        drop(hot);

        if let Some((layer, entry)) = expected {
            let matches = if entry.key != key {
                false
            } else {
                match layer {
                    MemoryLayer::Warm => {
                        hot_before.is_none()
                            && Self::warm_matches(warm_before.as_ref(), Some(&entry.raw.value))
                    }
                    MemoryLayer::Hot => match hot_before.as_ref() {
                        Some(current) if current.content == entry.content => {
                            let current_value =
                                Self::typed_value(&current.content, current.meta.to_memory_meta())?;
                            let expected_value =
                                Self::typed_value(&entry.content, entry.meta.clone())?;
                            Self::warm_matches(Some(&current_value), Some(&expected_value))
                        }
                        _ => false,
                    },
                    MemoryLayer::Cold => false,
                }
            };
            if !matches {
                return Err(ReactError::Other(format!(
                    "memory key {key} changed before transition; retry from a fresh read"
                )));
            }
        }

        let id = uuid::Uuid::new_v4().to_string();
        let mut audit = builder.build_with(id.clone(), chrono::Utc::now());
        if audit.trigger == "write_memory" {
            audit.change_type = if warm_before.is_some() || hot_before.is_some() {
                ChangeType::Update
            } else {
                ChangeType::Create
            };
            audit.before = (warm_before.is_some() || hot_before.is_some())
                .then(|| serde_json::json!({ "layer": if hot_before.is_some() { "hot" } else { "warm" } }));
        }
        let operation = MemoryOperation {
            audit,
            id,
            key: key.to_owned(),
            warm_before,
            warm_after,
            hot_before,
            hot_after,
        };
        self.operation_journal()?.prepare(operation.clone())?;
        self.settle_batch(&MemoryOperationBatch {
            id: operation.id.clone(),
            operations: vec![operation],
            origin: None,
        })
        .await
    }

    // ── 读取 ─────────────────────────────────────────────────────

    /**
     * 读取热层内容（MEMORY.md 正文，去除 frontmatter）。
     *
     * 若仍有操作在等待对账，会返回错误。
     */
    pub fn read_hot_content(&self) -> Result<String> {
        let _serial = self.clean_read_guard()?;
        Ok(self.parse_memory_file().body)
    }

    /**
     * 读取热层元数据（MEMORY.md 的 YAML frontmatter 条目）。
     *
     * 若仍有操作在等待对账，会返回错误。
     */
    pub fn read_hot_meta(&self) -> Result<Vec<HotEntryMeta>> {
        let _serial = self.clean_read_guard()?;
        Ok(self.parse_memory_file().entries)
    }

    /// 获取一个"读取安全"守卫：要求没有未结算的操作，且已完成过对账。
    fn clean_read_guard(&self) -> Result<tokio::sync::MutexGuard<'_, ()>> {
        let guard = self.operation_lock.try_lock().map_err(|_| {
            ReactError::Other("memory mutation in progress; read after settlement".into())
        })?;
        let history = self.operation_journal()?.history()?;
        if history.iter().any(|(_, settled)| !settled)
            || (!history.is_empty() && !self.reconciled.load(Ordering::Acquire))
        {
            return Err(ReactError::Other(
                "memory mutation pending; call reconcile_pending before reading".into(),
            ));
        }
        Ok(guard)
    }

    /**
     * 判断一个记忆键当前所处的层级。
     *
     * 依次检查热层（MEMORY.md）与温层（Store）。stage4 移除了冷层——
     * 已归档的记忆继续存活在温/统一命名空间中。
     * 若该键在任何层都未找到，返回 `None`。
     */
    pub async fn locate(&self, key: &str) -> Result<Option<(MemoryLayer, TypedMemoryEntry)>> {
        let _serial = self.operation_lock.lock().await;
        self.reconcile_pending_locked().await?;
        // 检查热层
        let file = self.parse_memory_file();
        if let Some(entry) = self.find_in_hot(&file, key) {
            return Ok(Some((MemoryLayer::Hot, entry)));
        }

        // 检查温层（统一命名空间；包含已归档条目）
        if let Some(entry) = self.typed_store.get_typed(WARM_NAMESPACE, key).await? {
            return Ok(Some((MemoryLayer::Warm, entry)));
        }

        Ok(None)
    }

    /// 获取全部热层条目，重建为 TypedMemoryEntries（来自 MEMORY.md）。
    pub fn list_hot(&self) -> Result<Vec<TypedMemoryEntry>> {
        let _serial = self.clean_read_guard()?;
        let file = self.parse_memory_file();
        Ok(file
            .entries
            .iter()
            .filter_map(|meta| self.hot_meta_to_entry(meta, &file.body))
            .collect())
    }

    /// 获取所有匹配给定过滤条件的温层条目。
    pub async fn list_warm(&self, filter: &MemoryFilter) -> Result<Vec<TypedMemoryEntry>> {
        let _serial = self.operation_lock.lock().await;
        self.reconcile_pending_locked().await?;
        self.typed_store.list_typed(WARM_NAMESPACE, filter).await
    }

    // ── 晋升 / 降级 ────────────────────────────────────────
    /* （stage4 B1）`promote`（冷层→温层的调度函数）已被移除——它没有任何
     * 调用方，且冷层已不复存在。温层→热层的晋升仍然通过
     * `consider_promotion` → `promote_warm_to_hot` 进行。 */

    /**
     * 降级一条记忆：热层→温层，或温层→已归档（原地标记）。
     * 将一条记忆向上晋升一层：冷层→温层，或温层→热层。
     *
     * 这是 [`demote`](Self::demote) 的逆操作。stage4 已将独立的冷层
     * 折叠进 `Warm`+`Archived`，因此：
     * - `Warm → Hot`：委托给 [`consider_promotion`](Self::consider_promotion)。
     * - `Cold → Warm`：仅对维护独立冷存储的消费者（在
     *   [`COLD_NAMESPACE`] 下）有意义；默认路径返回 `Ok(None)`。
     * - `Hot`：已经处于顶层，返回 `Ok(None)`。
     *
     * 作为 pub API 保留下来（对齐 Letta/MemGPT 的晋升理念以及
     * OpenClaw Dreaming），供采用三层结构的消费者接入。
     */
    pub async fn promote(&self, key: &str) -> Result<Option<LayerChangeResult>> {
        let Some((layer, _entry)) = self.locate(key).await? else {
            return Ok(None);
        };
        match layer {
            MemoryLayer::Hot => Ok(None),
            MemoryLayer::Warm => self.consider_promotion(key).await,
            MemoryLayer::Cold => {
                tracing::debug!(
                    key = %key,
                    "promote() on cold-layer entry; default path has no cold store, no-op"
                );
                Ok(None)
            }
        }
    }

    /**
     * （stage4 B1）温层降级不再把条目搬到一个冷命名空间；
     * 而是原地把 `status` 标记为 `Archived`，使该记忆在统一命名空间中
     * 继续保持可召回（伴随衰减）。从 Archived 复活为 Active 由
     * Dreaming（stage 2）负责处理。`update_meta` 接收的是完整的
     * `MemoryMeta`（而不是闭包），因此这里采用先读后写（get-modify-put）。
     */
    pub async fn demote(&self, key: &str, reason: &str) -> Result<LayerChangeResult> {
        let Some((layer, entry)) = self.locate(key).await? else {
            return Err(ReactError::Config(Box::new(ConfigError::ConfigFileError(
                format!("Memory key '{key}' not found in any layer"),
            ))));
        };

        match layer {
            MemoryLayer::Hot => self.demote_hot_to_warm(key, entry, reason).await,
            MemoryLayer::Warm => {
                let mut meta = entry.meta.clone();
                meta.status = MemoryStatus::Archived;
                self.transition(
                    key,
                    Some(Self::typed_value(&entry.content, meta)?),
                    None,
                    Self::change_builder(
                        key,
                        ChangeType::Demote,
                        Some("warm"),
                        Some("archived"),
                        reason,
                        "demote",
                    ),
                    Some((MemoryLayer::Warm, &entry)),
                )
                .await?;
                self.notify_memory_layer_change(key, "warm", "archived")
                    .await;
                Ok(LayerChangeResult {
                    key: key.to_string(),
                    from_layer: MemoryLayer::Warm,
                    to_layer: MemoryLayer::Warm,
                    reason: reason.to_string(),
                })
            }
            MemoryLayer::Cold => Err(ReactError::Config(Box::new(ConfigError::ConfigFileError(
                format!(
                    "Memory key '{key}' is in the optional cold layer; the default product path has no cold store, cannot demote further"
                ),
            )))),
        }
    }

    /**
     * 原地将一条已归档的温层记忆复活为 Active（stage4 F1 Dreaming, G2）。
     *
     * `MemoryMeta::is_hot_eligible()` 要求 `status == Active`，因此已归档
     * 的记忆必须先被复活，`consider_promotion` 才能将其晋升到热层。
     * 采用先读后写（`update_meta` 接收的是完整的 `MemoryMeta`，而不是
     * 闭包）。仅当条目此前为 Archived 且现在变为 Active 时才返回 `true`。
     */
    pub async fn revive_archived(&self, key: &str) -> Result<bool> {
        if let Some(entry) = self.typed_store.get_typed(WARM_NAMESPACE, key).await?
            && entry.meta.status == MemoryStatus::Archived
        {
            let mut meta = entry.meta.clone();
            meta.status = MemoryStatus::Active;
            self.transition(
                key,
                Some(Self::typed_value(&entry.content, meta)?),
                None,
                Self::change_builder(
                    key,
                    ChangeType::Promote,
                    Some("archived"),
                    Some("warm"),
                    "recent high-recall activity revived archived memory",
                    "dreaming",
                ),
                Some((MemoryLayer::Warm, &entry)),
            )
            .await?;
            self.notify_memory_layer_change(key, "archived", "warm")
                .await;
            return Ok(true);
        }
        Ok(false)
    }

    /**
     * 从任何持有该记忆的层级中永久删除它（C4 修复）。
     *
     * 通过 [`locate`](Self::locate) 查找 `key`：热层条目从 MEMORY.md
     * 中移除（frontmatter + 正文）；温层条目通过 `delete_typed` 从类型化
     * 存储中删除。如果找到并删除了记忆则返回 `true`，若该键在任何
     * 层都不存在则返回 `false`。
     *
     * 这修复了一个命名空间缺口：产品的 `forget` 工具（旧版
     * `ForgetTool`）曾从按 agent 划分的命名空间 `[agent_name,
     * "memories"]` 删除，而 `remember` 写入的却是统一的
     * `["agent", "memories"]`——导致 `forget` 永远删不掉 `remember`
     * 存下的内容。分层版本（`LayeredForgetTool`）改为走这里。
     */
    pub async fn delete_memory(&self, key: &str) -> Result<bool> {
        let Some((layer, entry)) = self.locate(key).await? else {
            return Ok(false);
        };

        match layer {
            MemoryLayer::Hot | MemoryLayer::Warm => {}
            MemoryLayer::Cold => {
                return Err(ReactError::Other(format!(
                    "memory key {key} belongs to an unsupported cold store"
                )));
            }
        }

        let layer_name = match layer {
            MemoryLayer::Hot => "hot",
            MemoryLayer::Warm => "warm",
            MemoryLayer::Cold => "cold",
        };
        self.transition(
            key,
            None,
            None,
            Self::change_builder(
                key,
                ChangeType::Delete,
                Some(layer_name),
                None,
                "user forget",
                "delete_memory",
            ),
            Some((layer, &entry)),
        )
        .await?;
        if layer == MemoryLayer::Hot {
            self.notify_memory_layer_change(key, "hot", "deleted").await;
        }
        Ok(true)
    }

    /**
     * 列出所有匹配 `filter` 的温层类型化记忆（stage4 F1 Dreaming）。
     *
     * Dreaming 会扫描统一的 `["agent","memories"]` 命名空间，找出
     * 值得晋升的高召回记忆，以及应当降级的陈旧低召回记忆。
     */
    pub async fn list_warm_memories(&self, filter: &MemoryFilter) -> Result<Vec<TypedMemoryEntry>> {
        self.list_warm(filter).await
    }

    /**
     * 应用一个此前已审阅过的语义冲突提案。
     *
     * 当前温层的冲突状态必须与提案精确匹配；过期的提案会在任何
     * 变更发生之前失败。返回的快照是调用方审阅日志中权威的撤销负载。
     */
    pub async fn apply_merge_proposal(
        &self,
        proposal: &MemoryConflictProposal,
    ) -> Result<AppliedMemoryMerge> {
        if proposal.members.len() < 2 {
            return Err(merge_plan_error(
                "memory merge proposal requires at least two members",
            ));
        }

        let entries = self
            .typed_store
            .list_typed(WARM_NAMESPACE, &MemoryFilter::new())
            .await?;
        let expected_keys: std::collections::HashSet<&str> = proposal
            .members
            .iter()
            .map(|member| member.key.as_str())
            .collect();
        if expected_keys.len() != proposal.members.len() {
            return Err(merge_plan_error(
                "memory merge proposal contains duplicate member keys",
            ));
        }

        let group = ConflictDetector::new()
            .detect(&entries)
            .into_iter()
            .find(|group| {
                if group.topic != proposal.topic || group.memory_type != proposal.memory_type {
                    return false;
                }
                let current_keys: std::collections::HashSet<&str> = group
                    .entries
                    .iter()
                    .map(|entry| entry.key.as_str())
                    .collect();
                current_keys == expected_keys
            })
            .ok_or_else(|| {
                stale_merge_plan_error("memory conflict changed; refresh Review Inbox")
            })?;

        for expected in &proposal.members {
            let current = group
                .entries
                .iter()
                .find(|entry| entry.key == expected.key)
                .ok_or_else(|| merge_plan_error("memory conflict member disappeared"))?;
            if current.content != expected.content
                || current.meta.status != expected.status
                || (current.meta.confidence - expected.confidence).abs() > f32::EPSILON
            {
                return Err(stale_merge_plan_error(
                    "memory conflict content or metadata changed; refresh Review Inbox",
                ));
            }
        }

        let current_proposal = MemoryConflictProposal::from_group(&group)
            .ok_or_else(|| merge_plan_error("memory conflict has no primary candidate"))?;
        if current_proposal.recommended_primary_key != proposal.recommended_primary_key {
            return Err(stale_merge_plan_error(
                "memory conflict recommendation changed; refresh Review Inbox",
            ));
        }

        let before = group
            .entries
            .iter()
            .map(|entry| MemoryMergeSnapshot {
                key: entry.key.clone(),
                content: entry.content.clone(),
                meta: entry.meta.clone(),
            })
            .collect();
        let result = MemoryMerger::new(self).merge_group(&group).await?;
        Ok(AppliedMemoryMerge {
            primary_key: result.primary_key,
            superseded_keys: result.superseded_keys,
            before,
            batch_id: result.batch_id,
        })
    }

    /// 提交一个合并组：主条目保留，其余成员被标记为 Superseded。
    pub(super) async fn commit_merge_group(&self, group: &ConflictGroup) -> Result<MergeResult> {
        let ordered = ordered_conflict_entries(group);
        let Some(primary) = ordered.first() else {
            return Ok(MergeResult {
                primary_key: String::new(),
                superseded_keys: Vec::new(),
                batch_id: None,
            });
        };
        if ordered.len() < 2 {
            return Ok(MergeResult {
                primary_key: primary.key.clone(),
                superseded_keys: Vec::new(),
                batch_id: None,
            });
        }
        let _serial = self.operation_lock.lock().await;
        self.reconcile_pending_locked().await?;
        let revision_count = ordered.iter().fold(0_u32, |sum, entry| {
            sum.saturating_add(entry.meta.revision_count)
        });
        let mut operations = Vec::with_capacity(ordered.len());
        for (index, entry) in ordered.iter().enumerate() {
            let current = self
                .typed_store
                .inner()
                .get(WARM_NAMESPACE, &entry.key)
                .await?
                .ok_or_else(|| stale_merge_plan_error("memory merge member disappeared"))?;
            if current.value != entry.raw.value
                || Self::hot_value(&self.parse_memory_file(), &entry.key).is_some()
            {
                return Err(stale_merge_plan_error(
                    "memory merge member changed; refresh Review Inbox",
                ));
            }
            let mut meta = entry.meta.clone();
            let superseded_by = if index == 0 {
                None
            } else {
                Some(primary.key.clone())
            };
            if index == 0 {
                meta.revision_count = revision_count.max(meta.revision_count);
            } else {
                meta.status = MemoryStatus::Superseded;
                meta.superseded_by = superseded_by.clone();
            }
            let id = uuid::Uuid::new_v4().to_string();
            let reason = if index == 0 {
                format!(
                    "primary survivor of {}-way merge on topic '{}'",
                    ordered.len(),
                    group.topic
                )
            } else {
                format!(
                    "superseded by '{}' during merge on topic '{}'",
                    primary.key, group.topic
                )
            };
            let audit = ChangeEntryBuilder::new(EntityType::Memory, &entry.key, ChangeType::Merge)
                .reason(reason).trigger("explicit_memory_merge")
                .after(serde_json::json!({ "superseded_by": superseded_by, "group_size": ordered.len() }))
                .build_with(id.clone(), chrono::Utc::now());
            operations.push(MemoryOperation {
                id,
                key: entry.key.clone(),
                warm_before: Some(current.value),
                warm_after: Some(Self::typed_value(&entry.content, meta)?),
                hot_before: None,
                hot_after: None,
                audit,
            });
        }
        let batch = MemoryOperationBatch {
            id: uuid::Uuid::new_v4().to_string(),
            operations,
            origin: None,
        };
        self.operation_journal()?.prepare_batch(batch.clone())?;
        self.settle_batch(&batch).await?;
        Ok(MergeResult {
            primary_key: primary.key.clone(),
            superseded_keys: ordered
                .iter()
                .skip(1)
                .map(|entry| entry.key.clone())
                .collect(),
            batch_id: Some(batch.id),
        })
    }

    /**
     * 旧版合并撤销入口。
     *
     * 仅基于快照的恢复无法证明日志尖端（journal tip），因此已被淘汰。
     * 请改用 [`Self::rollback_memory`]，配合 [`Self::apply_merge_proposal`]
     * 返回的 `batch_id`，通过权威的日志机制还原完整的合并组。
     */
    #[deprecated(note = "use rollback_memory with AppliedMemoryMerge::batch_id")]
    pub async fn restore_merge_snapshots(&self, _snapshots: &[MemoryMergeSnapshot]) -> Result<()> {
        Err(merge_plan_error(
            "snapshot-only merge restore is retired; use rollback_memory with the merge batch ID",
        ))
    }

    /**
     * 若温层条目满足晋升资格且热层仍有容量，则考虑将其晋升到热层。
     *
     * 在每次新的记忆写入之后被调用。
     */
    pub async fn consider_promotion(&self, key: &str) -> Result<Option<LayerChangeResult>> {
        let Some((layer, entry)) = self.locate(key).await? else {
            return Ok(None);
        };

        if layer != MemoryLayer::Warm {
            return Ok(None);
        }

        // 检查是否满足晋升资格
        if !entry.meta.is_hot_eligible() {
            return Ok(None);
        }

        // 检查信任级别 —— 不可信内容不能自动晋升到热层
        use super::security::InputTrustLevel;
        if !InputTrustLevel::from_source(entry.meta.source).can_auto_promote() {
            return Ok(None);
        }

        self.promote_warm_to_hot(key, entry).await
    }

    /**
     * 通过降级优先级最低的条目来控制热层的 token 预算。
     *
     * 在每次热层写入之后被调用。返回降级结果列表。
     */
    pub async fn enforce_hot_budget(&self) -> Result<Vec<LayerChangeResult>> {
        let mut results = Vec::new();
        loop {
            let file = self.parse_memory_file();
            if estimate_tokens(&file.body) <= HOT_TOKEN_BUDGET {
                break;
            }
            let candidate = file.entries.iter().max_by(|a, b| {
                Self::demotion_score(a)
                    .partial_cmp(&Self::demotion_score(b))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let Some(candidate) = candidate else { break };
            let key = candidate.key.clone();
            let entry = self
                .find_in_hot(&file, &key)
                .ok_or_else(|| ReactError::Other(format!("missing hot entry {key}")))?;
            results.push(
                self.demote_hot_to_warm(&key, entry, "hot layer budget enforcement")
                    .await?,
            );
        }

        Ok(results)
    }

    /** 计算一个热层条目的降级优先级分数。
     * 分数越高，越容易被降级。 */
    fn demotion_score(meta: &HotEntryMeta) -> f32 {
        let confidence_factor = 1.0 - meta.confidence;
        let stability_factor = 1.0 - meta.stability;

        /* 自上次晋升以来的陈旧度（简化处理——MEMORY.md 中没有访问追踪）。
         * 由于脱离 Store 就无法追踪访问情况，这里使用一个固定的中等陈旧度。 */
        let staleness_factor = 0.3;

        // 新近度 —— 没有 last_accessed 字段，用 stability 作为替代指标
        let recency_factor = 1.0 - meta.stability;

        confidence_factor * 0.35
            + stability_factor * 0.25
            + staleness_factor * 0.20
            + recency_factor * 0.20
    }

    // ── 写入温层 ─────────────────────────────────────────

    /**
     * 向温层写入一条新的类型化记忆，并考虑是否晋升。
     *
     * 若该记忆被晋升到热层，返回 `Ok(Some(result))`。
     */
    pub async fn write_memory(
        &self,
        key: &str,
        content: &str,
        meta: MemoryMeta,
    ) -> Result<Option<LayerChangeResult>> {
        // 安全检查：扫描密钥、检测注入、限流、判定信任级别。
        let trust = InputTrustLevel::from_source(meta.source);
        let verdict = self.security_guard.check_memory_write(content, trust);

        if !verdict.allowed {
            return Err(ReactError::Config(Box::new(ConfigError::ConfigFileError(
                format!(
                    "Memory write blocked by security guard: {}",
                    verdict
                        .reason
                        .unwrap_or_else(|| "unknown reason".to_string())
                ),
            ))));
        }

        // 若密钥已被脱敏，则使用脱敏后的内容，否则使用原始内容。
        let safe_content = verdict
            .sanitized_content
            .unwrap_or_else(|| content.to_string());

        self.transition(
            key,
            Some(Self::typed_value(&safe_content, meta.clone())?),
            None,
            Self::change_builder(
                key,
                ChangeType::Create,
                None,
                Some("warm"),
                &format!("memory via {}", meta.source.as_str().unwrap_or("unknown")),
                "write_memory",
            ),
            None,
        )
        .await?;

        self.notify_memory_write(key, meta.source.as_str().unwrap_or("unknown"))
            .await;

        // 考虑是否晋升
        self.consider_promotion(key).await
    }

    // ── 搜索 ──────────────────────────────────────────────────────

    /**
     * 跨热层与温层搜索。
     *
     * 热层结果排在前面（优先级更高），随后是温层结果。
     */
    pub async fn search_layered(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<(MemoryLayer, TypedMemoryEntry)>> {
        let _serial = self.operation_lock.lock().await;
        self.reconcile_pending_locked().await?;
        let mut results = Vec::new();

        // 搜索热层（对正文做关键词匹配）
        let file = self.parse_memory_file();
        let query_lower = query.to_lowercase();
        for meta in &file.entries {
            if results.len() >= limit {
                break;
            }
            // 简单的关键词匹配：检查键或主题是否包含查询词
            if (meta.key.to_lowercase().contains(&query_lower)
                || meta.topic.to_lowercase().contains(&query_lower))
                && let Some(entry) = self.hot_meta_to_entry(meta, &file.body)
            {
                results.push((MemoryLayer::Hot, entry));
            }
        }

        /* （stage4 D1）温层通过统一的复合分数召回入口获取——排序方式、
         * Superseded 过滤、recall_count 均与自动路径保持一致。 */
        let remaining = limit.saturating_sub(results.len());
        if remaining > 0 {
            let reca =
                crate::evolution::recall::MemoryRecaller::new(self.typed_store.inner().clone());
            if let Ok(warm_items) = reca.recall(query, remaining).await {
                for item in warm_items {
                    let entry = TypedMemoryEntry::from_store_item(item);
                    results.push((MemoryLayer::Warm, entry));
                }
            }
        }

        Ok(results)
    }

    // ── 热层文件 I/O ──────────────────────────────────────────

    /**
     * 将 MEMORY.md 解析为结构化形式。
     *
     * 若文件不存在或无法解析，返回一个默认（空）的 MemoryFile。
     *
     * 注意：这是一次不加锁的读取。任何需要原子性的读-改-写操作，
     * 请改用 [`with_locked_hot_file`]。
     */
    fn parse_memory_file(&self) -> MemoryFile {
        let raw = match std::fs::read_to_string(&self.hot_path) {
            Ok(content) => content,
            Err(e) => {
                if e.kind() != std::io::ErrorKind::NotFound {
                    tracing::warn!(
                        path = ?self.hot_path,
                        error = %e,
                        "MEMORY.md could not be read; falling back to default"
                    );
                }
                return MemoryFile::default();
            }
        };

        parse_memory_md(&raw)
    }

    /**
     * 将 MemoryFile 持久化写回磁盘。
     *
     * 采用原子写入（临时文件 + rename）并执行 `fsync`，因此崩溃不会
     * 导致 MEMORY.md 出现半写或被清空的情况。
     */
    fn write_memory_file(&self, file: &MemoryFile) -> std::io::Result<()> {
        // 确保父目录存在
        if let Some(parent) = self.hot_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let content = format_memory_md(file);
        // 原子写入：先写入临时文件，fsync，再 rename。
        let tmp_path = self.hot_path.with_extension("md.tmp");
        {
            use std::io::Write;
            let mut tmp = std::fs::File::create(&tmp_path)?;
            tmp.write_all(content.as_bytes())?;
            /* rename 之前先 fsync，这样 rename 之后发生崩溃也不会
             * 暴露出一个空文件（旧版本没有做 fsync）。 */
            tmp.sync_all()?;
        }
        std::fs::rename(&tmp_path, &self.hot_path)?;
        if let Some(parent) = self.hot_path.parent() {
            std::fs::File::open(parent)?.sync_all()?;
        }

        Ok(())
    }

    /**
     * 以独占方式访问 MEMORY.md 热层来运行一个同步闭包。在完整的
     * 加载 → 修改 → 保存 过程中，同时持有进程内 Mutex 与跨进程 flock。
     *
     * 对于需要在修改与保存之间 `.await`（例如 `enforce_hot_budget`，
     * 它会在临界区中途向温层 Store 写入）的异步调用方，请改用
     * [`lock_hot_file`] 获取一个守卫，并让它跨越这些 await 点持有。
     * 同时获取 MEMORY.md 文件的进程内 Mutex 与跨进程 flock，
     * 返回一个持有刚加载出的 `MemoryFile` 的守卫。
     *
     * 该守卫可以跨越 `.await` 点持有（它只包含 `tokio::sync::MutexGuard`
     * 和一个 `File`，两者都是 `Send`），这使得类似 `enforce_hot_budget`
     * 这样的函数可以在临界区内部执行异步的 Store 写入。调用
     * [`HotFileGuard::commit`] 持久化，或直接丢弃以放弃变更。
     *
     * 注意：下面的 flock 获取会阻塞调用线程。这是可以接受的，
     * 因为 curator/记忆写入并不频繁、竞争程度低；有界重试循环
     * 可以防止一个卡死的对端把我们永远挂起。
     */
    async fn lock_hot_file(&self) -> std::io::Result<HotFileGuard<'_>> {
        /* 先做进程内序列化。使用 tokio::sync::Mutex 使守卫是
         * `Send` 的，可以跨越 `.await`。 */
        let inproc = self.hot_lock.lock().await;

        // 基于旁路文件的跨进程建议锁。
        let lock_path = self.hot_path.with_extension("md.lock");
        let lock_file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&lock_path)
            .map_err(|e| {
                tracing::warn!(error = %e, "hot-file lock sidecar open failed; continuing lockless");
                e
            })?;
        /* 有界的 flock 尝试。flock 每次调用都会阻塞；限制总尝试次数，
         * 以免一个卡死的对端把我们永远挂起。 */
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
                    tokio::time::sleep(BACKOFF).await;
                }
                Err(error) => return Err(error),
            }
        }
        if !got_lock {
            return Err(std::io::Error::new(
                std::io::ErrorKind::WouldBlock,
                format!("MEMORY.md lock unavailable after {MAX_ATTEMPTS} attempts"),
            ));
        }

        let file = self.parse_memory_file();
        Ok(HotFileGuard {
            #[cfg(test)]
            manager: self,
            file,
            _inproc: inproc,
            _lock_file: lock_file,
        })
    }

    /// 向 MEMORY.md 热层添加一条条目。
    #[cfg(test)]
    async fn add_to_hot(&self, entry: &TypedMemoryEntry) -> std::io::Result<()> {
        let mut guard = self.lock_hot_file().await?;
        let value = HotValue {
            meta: Self::hot_meta(entry),
            content: entry.content.clone(),
        };
        Self::set_hot_value(&mut guard.file, &entry.key, Some(&value))
            .map_err(std::io::Error::other)?;

        guard.commit()
    }

    /// 在已解析的文件中按键查找一条热层条目。
    fn find_in_hot(&self, file: &MemoryFile, key: &str) -> Option<TypedMemoryEntry> {
        let meta = file.entries.iter().find(|e| e.key == key)?;
        self.hot_meta_to_entry(meta, &file.body)
    }

    /// 从 HotEntryMeta 与文件正文重建出一个 TypedMemoryEntry。
    fn hot_meta_to_entry(&self, meta: &HotEntryMeta, body: &str) -> Option<TypedMemoryEntry> {
        let content = extract_hot_value_content(body, meta);
        Some(TypedMemoryEntry {
            key: meta.key.clone(),
            content,
            meta: meta.to_memory_meta(),
            raw: echo_core::memory::store::StoreItem::new(
                WARM_NAMESPACE.iter().map(|s| s.to_string()).collect(),
                meta.key.clone(),
                serde_json::Value::Null,
            ),
        })
    }

    // ── 内部晋升/降级方法 ─────────────────────────

    /// 将一条温层条目晋升到热层。
    async fn promote_warm_to_hot(
        &self,
        key: &str,
        entry: TypedMemoryEntry,
    ) -> Result<Option<LayerChangeResult>> {
        let hot = HotValue {
            meta: Self::hot_meta(&entry),
            content: entry.content.clone(),
        };
        self.transition(
            key,
            None,
            Some(hot),
            Self::change_builder(
                key,
                ChangeType::Promote,
                Some("warm"),
                Some("hot"),
                "warm-to-hot promotion (eligible)",
                "promote",
            ),
            Some((MemoryLayer::Warm, &entry)),
        )
        .await?;

        self.notify_memory_layer_change(key, "warm", "hot").await;

        /* 在发布这次晋升之后再执行预算控制。这样一来，由此产生的
         * 任何"热层→温层"事件都能保留真实的时间先后顺序。 */
        self.enforce_hot_budget().await?;

        Ok(Some(LayerChangeResult {
            key: key.to_string(),
            from_layer: MemoryLayer::Warm,
            to_layer: MemoryLayer::Hot,
            reason: "warm→hot promotion (eligible)".to_string(),
        }))
    }

    /// 将一条热层条目降级到温层。
    async fn demote_hot_to_warm(
        &self,
        key: &str,
        entry: TypedMemoryEntry,
        reason: &str,
    ) -> Result<LayerChangeResult> {
        self.transition(
            key,
            Some(Self::typed_value(&entry.content, entry.meta.clone())?),
            None,
            Self::change_builder(
                key,
                ChangeType::Demote,
                Some("hot"),
                Some("warm"),
                reason,
                "demote",
            ),
            Some((MemoryLayer::Hot, &entry)),
        )
        .await?;

        self.notify_memory_layer_change(key, "hot", "warm").await;

        Ok(LayerChangeResult {
            key: key.to_string(),
            from_layer: MemoryLayer::Hot,
            to_layer: MemoryLayer::Warm,
            reason: reason.to_string(),
        })
    }

    /// 从一条 TypedMemoryEntry 构造对应的热层元数据。
    fn hot_meta(entry: &TypedMemoryEntry) -> HotEntryMeta {
        HotEntryMeta {
            key: entry.key.clone(),
            memory_type: entry.meta.memory_type,
            confidence: entry.meta.confidence,
            stability: entry.meta.stability,
            recall_weight: entry.meta.recall_weight,
            source: entry.meta.source,
            topic: entry.meta.topic.clone(),
            risk: entry.meta.risk,
            revision_count: entry.meta.revision_count,
            recall_count: entry.meta.recall_count,
            last_recalled_at: entry.meta.last_recalled_at,
            content_json: entry.content.trim() != entry.content
                || entry.content.contains('\n')
                || entry.content.contains('\r'),
            last_promoted: crate::utils::time::now_local().to_rfc3339(),
        }
    }

    /// 构造一个已填充来源层/目标层/原因/触发方式的变更条目构建器。
    fn change_builder(
        key: &str,
        change_type: ChangeType,
        from: Option<&str>,
        to: Option<&str>,
        reason: &str,
        trigger: &str,
    ) -> ChangeEntryBuilder {
        let mut builder = ChangeEntryBuilder::new(EntityType::Memory, key, change_type);

        if let Some(f) = from {
            builder = builder.before(serde_json::json!({ "layer": f }));
        }
        if let Some(t) = to {
            builder = builder.after(serde_json::json!({ "layer": t }));
        }
        builder = builder.reason(reason.to_string());
        builder = builder.trigger(trigger.to_string());

        builder.reason(reason).trigger(trigger)
    }

    /// 若配置了观察者，通知一次记忆写入事件。
    async fn notify_memory_write(&self, key: &str, source: &str) {
        if let Some(observer) = &self.evolution_observer {
            observer.on_memory_write(key, source).await;
        }
    }

    /// 若配置了观察者，通知一次内存层级变更事件。
    async fn notify_memory_layer_change(&self, key: &str, from_layer: &str, to_layer: &str) {
        if let Some(observer) = &self.evolution_observer {
            observer
                .on_memory_layer_change(key, from_layer, to_layer)
                .await;
        }
    }
}

// ── 解析函数 ──────────────────────────────────────────────────────

/** 将一份 MEMORY.md 文件解析为结构化形式。
 * 该文件预期在 `---` 标记之间有 YAML frontmatter，
 * 随后是包含列表条目的 markdown 正文。 */
fn parse_memory_md(raw: &str) -> MemoryFile {
    let trimmed = raw.trim_start();

    // 检查是否存在 YAML frontmatter
    if !trimmed.starts_with("---") {
        // 没有 frontmatter —— 将全部内容当作正文处理
        return MemoryFile {
            entries: Vec::new(),
            body: raw.to_string(),
        };
    }

    // 查找结束的 ---
    let rest = &trimmed[3..]; // 跳过开头的 ---
    let end_marker = rest.find("\n---").or_else(|| rest.find("\r\n---"));

    let (frontmatter_str, body_str) = match end_marker {
        Some(pos) => {
            let fm = &rest[..pos];
            let after_marker = &rest[pos + 4..]; // 跳过 \n---
            let body = after_marker
                .trim_start_matches('\n')
                .trim_start_matches('\r');
            (fm, body)
        }
        None => {
            // 没有结束标记 —— 全部当作正文处理
            return MemoryFile {
                entries: Vec::new(),
                body: raw.to_string(),
            };
        }
    };

    let entries = match serde_yaml_ng::from_str::<MemoryFileFrontmatter>(frontmatter_str) {
        Ok(fm) => fm.entries.unwrap_or_default(),
        Err(_) => Vec::new(), // frontmatter 格式有误 —— 优雅降级
    };

    MemoryFile {
        entries,
        body: body_str.to_string(),
    }
}

/// 将一个 MemoryFile 格式化回 MEMORY.md 格式。
fn format_memory_md(file: &MemoryFile) -> String {
    let mut out = String::new();

    if !file.entries.is_empty() {
        out.push_str("---\n");
        let fm = MemoryFileFrontmatter {
            entries: Some(file.entries.clone()),
        };
        match serde_yaml_ng::to_string(&fm) {
            Ok(yaml) => {
                // serde_yaml 会添加 "---\n" 前缀，这里将其剥离
                let yaml = yaml.trim_start_matches("---\n");
                out.push_str(yaml);
            }
            Err(_) => {
                // 兜底：使用空 frontmatter
                out.push_str("entries: []\n");
            }
        }
        out.push_str("---\n\n");
    }

    out.push_str(&file.body);

    out
}

/// 用于 YAML frontmatter 反序列化的辅助结构体。
#[derive(Debug, Serialize, Deserialize)]
struct MemoryFileFrontmatter {
    /// 热层条目元数据列表。
    #[serde(default)]
    entries: Option<Vec<HotEntryMeta>>,
}

/// 按键从正文中提取某条热层条目的内容。
fn extract_hot_entry_content(body: &str, key: &str) -> String {
    let pattern = format!("- **[{key}]** ");
    body.lines()
        .find_map(|line| line.strip_prefix(&pattern).map(str::to_owned))
        .unwrap_or_default()
}

/// 提取热层条目内容，必要时按 JSON 字符串解码。
fn extract_hot_value_content(body: &str, meta: &HotEntryMeta) -> String {
    let raw = extract_hot_entry_content(body, &meta.key);
    if meta.content_json {
        serde_json::from_str::<String>(&raw).unwrap_or(raw)
    } else {
        raw
    }
}

/** 估算一个字符串的 token 数量。
 * 采用一种保守的启发式方法：拉丁字符约每 4 个字符 1 个 token，
 * 中日韩字符约每 1.5 个字符 1 个 token。我们取 3 字符/token 作为折中。 */
fn estimate_tokens(text: &str) -> usize {
    let latin_chars = text.chars().filter(|c| c.is_ascii()).count();
    let cjk_chars = text.chars().filter(|c| !c.is_ascii()).count();
    // 拉丁字符：约 4 字符/token，中日韩字符：约 1.5 字符/token
    let latin_tokens = latin_chars / 4;
    let cjk_tokens = (cjk_chars as f32 / 1.5).ceil() as usize;
    latin_tokens + cjk_tokens
}

/// 为 MemorySource 提供 as_str 的扩展 trait。
trait MemorySourceExt {
    /// 返回该来源对应的字符串表示。
    fn as_str(&self) -> Option<&'static str>;
}

impl MemorySourceExt for MemorySource {
    /// 返回该来源对应的字符串表示。
    fn as_str(&self) -> Option<&'static str> {
        match self {
            MemorySource::UserCorrection => Some("user_correction"),
            MemorySource::ErrorResolution => Some("error_resolution"),
            MemorySource::RepeatedWorkflow => Some("repeated_workflow"),
            MemorySource::ExplicitSave => Some("explicit_save"),
            MemorySource::AutoExtracted => Some("auto_extracted"),
            MemorySource::L3Promotion => Some("l3_promotion"),
        }
    }
}

/// 为 InputTrustLevel 提供从 MemorySource 推导的扩展。
mod security_ext {
    use super::super::security::InputTrustLevel;
    use echo_core::memory::types::MemorySource;

    impl InputTrustLevel {
        /// 根据记忆来源推导出信任级别。
        pub fn from_source(source: MemorySource) -> Self {
            match source {
                MemorySource::ExplicitSave | MemorySource::UserCorrection => {
                    InputTrustLevel::Trusted
                }
                MemorySource::AutoExtracted | MemorySource::L3Promotion => {
                    InputTrustLevel::Assistant
                }
                MemorySource::ErrorResolution | MemorySource::RepeatedWorkflow => {
                    InputTrustLevel::Assistant
                }
            }
        }
    }
}

/// 记忆分层管理的单元测试。
#[cfg(test)]
mod tests {
    use super::super::audit::{ChangeEntry, ChangeRecordOutcome, JsonlChangeLog};
    use super::*;
    use echo_core::memory::types::{MemoryMeta, MemorySource, MemoryType};
    use echo_state::memory::store::{FileStore, InMemoryStore};
    use std::sync::Mutex as StdMutex;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    /// 用于测试的一个空操作（no-op）ChangeLog。
    struct NullChangeLog;
    impl ChangeLog for NullChangeLog {
        /// 空操作：直接丢弃条目。
        fn record(&self, _entry: ChangeEntry) -> Result<()> {
            Ok(())
        }
        /// 空操作：总是视为已记录。
        fn record_idempotent(
            &self,
            _entry: ChangeEntry,
        ) -> Result<super::super::audit::ChangeRecordOutcome> {
            Ok(super::super::audit::ChangeRecordOutcome::AlreadyRecorded)
        }
        /// 空操作：总是返回空列表。
        fn query(&self, _filter: &super::super::audit::ChangeFilter) -> Result<Vec<ChangeEntry>> {
            Ok(Vec::new())
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

    /// 记录层级变更通知的测试观察者。
    #[derive(Default)]
    struct RecordingObserver {
        /// 已收到的 (键, 源层, 目标层) 变更列表。
        changes: StdMutex<Vec<(String, String, String)>>,
    }

    impl RecordingObserver {
        /// 返回已记录变更的快照。
        fn changes(&self) -> Vec<(String, String, String)> {
            self.changes
                .lock()
                .map(|changes| changes.clone())
                .unwrap_or_default()
        }
    }

    impl EvolutionObserver for RecordingObserver {
        /// 记录一次记忆层级变更。
        fn on_memory_layer_change<'a>(
            &'a self,
            key: &'a str,
            from_layer: &'a str,
            to_layer: &'a str,
        ) -> BoxFuture<'a, ()> {
            Box::pin(async move {
                if let Ok(mut changes) = self.changes.lock() {
                    changes.push((
                        key.to_string(),
                        from_layer.to_string(),
                        to_layer.to_string(),
                    ));
                }
            })
        }
    }

    /// 所有写入都失败的 ChangeLog。
    struct FailingChangeLog;

    /// 下一次写入失败一次、之后委托给内部日志的 ChangeLog。
    struct FailOnceChangeLog {
        /// 实际存储条目的内部日志。
        inner: JsonlChangeLog,
        /// 下一次写入是否应失败。
        fail_next: AtomicBool,
    }

    /// 第二次写入失败、其余委托给内部日志的 ChangeLog。
    struct FailSecondChangeLog {
        /// 实际存储条目的内部日志。
        inner: JsonlChangeLog,
        /// 已发生的写入调用次数。
        calls: AtomicUsize,
    }

    impl ChangeLog for FailSecondChangeLog {
        /// 记录一条变更条目。
        fn record(&self, entry: ChangeEntry) -> Result<()> {
            self.record_idempotent(entry).map(|_| ())
        }

        /// 以幂等方式记录一条变更条目，第二次调用时失败。
        fn record_idempotent(&self, entry: ChangeEntry) -> Result<ChangeRecordOutcome> {
            if self.calls.fetch_add(1, Ordering::SeqCst) == 1 {
                return Err(merge_plan_error("injected second audit failure"));
            }
            self.inner.record_idempotent(entry)
        }

        /// 委托给内部日志查询。
        fn query(&self, filter: &super::super::audit::ChangeFilter) -> Result<Vec<ChangeEntry>> {
            self.inner.query(filter)
        }

        /// 委托给内部日志返回最新条目。
        fn latest_for(&self, entity_type: EntityType, key: &str) -> Result<Option<ChangeEntry>> {
            self.inner.latest_for(entity_type, key)
        }

        /// 返回内部日志的条目数量。
        fn len(&self) -> usize {
            self.inner.len()
        }
    }

    impl ChangeLog for FailOnceChangeLog {
        /// 记录一条变更条目。
        fn record(&self, entry: ChangeEntry) -> Result<()> {
            self.record_idempotent(entry).map(|_| ())
        }

        /// 以幂等方式记录一条变更条目，按需注入一次失败。
        fn record_idempotent(&self, entry: ChangeEntry) -> Result<ChangeRecordOutcome> {
            if self.fail_next.swap(false, Ordering::SeqCst) {
                return Err(merge_plan_error("injected change-log failure"));
            }
            self.inner.record_idempotent(entry)
        }

        /// 委托给内部日志查询。
        fn query(&self, filter: &super::super::audit::ChangeFilter) -> Result<Vec<ChangeEntry>> {
            self.inner.query(filter)
        }

        /// 委托给内部日志返回最新条目。
        fn latest_for(
            &self,
            entity_type: EntityType,
            entity_key: &str,
        ) -> Result<Option<ChangeEntry>> {
            self.inner.latest_for(entity_type, entity_key)
        }

        /// 返回内部日志的条目数量。
        fn len(&self) -> usize {
            self.inner.len()
        }
    }

    impl ChangeLog for FailingChangeLog {
        /// 总是返回错误。
        fn record(&self, _entry: ChangeEntry) -> Result<()> {
            Err(merge_plan_error("injected change-log failure"))
        }

        /// 总是返回错误。
        fn record_idempotent(
            &self,
            _entry: ChangeEntry,
        ) -> Result<super::super::audit::ChangeRecordOutcome> {
            Err(merge_plan_error("injected change-log failure"))
        }

        /// 总是返回空列表。
        fn query(&self, _filter: &super::super::audit::ChangeFilter) -> Result<Vec<ChangeEntry>> {
            Ok(Vec::new())
        }

        /// 总是返回 `None`。
        fn latest_for(
            &self,
            _entity_type: EntityType,
            _entity_key: &str,
        ) -> Result<Option<ChangeEntry>> {
            Ok(None)
        }

        /// 总是返回 `0`。
        fn len(&self) -> usize {
            0
        }
    }

    /// 创建一个用于测试的记忆分层管理器。
    fn make_manager() -> MemoryLayerManager {
        let dir = tempfile::tempdir().expect("tempdir");
        let dir_path = dir.keep();
        let store = Arc::new(InMemoryStore::new());
        let change_log = Box::new(NullChangeLog);
        MemoryLayerManager::new(dir_path, store, change_log)
    }

    /// 读取指定键最新的记忆审计条目。
    fn latest_memory_audit(path: &std::path::Path, key: &str) -> Result<ChangeEntry> {
        JsonlChangeLog::new(path.to_path_buf())?
            .latest_for(EntityType::Memory, key)?
            .ok_or_else(|| merge_plan_error(format!("expected audit for memory key {key}")))
    }

    /// 从审计条目中取出变更前或变更后的层级投影。
    fn audit_projection_layer(entry: &ChangeEntry, before: bool) -> Option<&str> {
        let projection = if before {
            entry.before.as_ref()
        } else {
            entry.after.as_ref()
        };
        projection
            .and_then(|value| value.get("layer"))
            .and_then(serde_json::Value::as_str)
    }

    /// 断言回滚审计条目包含机器可读的摘要。
    fn assert_machine_readable_rollback_summary(entry: &ChangeEntry) -> Result<()> {
        let before = entry.before.as_ref().and_then(serde_json::Value::as_object);
        let after = entry.after.as_ref().and_then(serde_json::Value::as_object);
        for summary in [before, after] {
            let summary = summary.ok_or_else(|| {
                merge_plan_error("rollback audit must contain an object projection summary")
            })?;
            for field in [
                "layer",
                "warm",
                "hot",
                "rollback_of",
                "target_batch_id",
                "target_generation",
            ] {
                assert!(
                    summary.contains_key(field),
                    "missing rollback field {field}"
                );
            }
        }
        assert_eq!(
            before.and_then(|summary| summary.get("rollback_of")),
            after.and_then(|summary| summary.get("rollback_of"))
        );
        assert_eq!(
            before.and_then(|summary| summary.get("target_batch_id")),
            after.and_then(|summary| summary.get("target_batch_id"))
        );
        assert_eq!(
            before.and_then(|summary| summary.get("target_generation")),
            after.and_then(|summary| summary.get("target_generation"))
        );
        Ok(())
    }

    /// 验证解析带 frontmatter 的记忆文件。
    #[test]
    fn test_parse_memory_file_with_frontmatter() {
        let raw = "\
---
entries:
  - key: build_java8
    memory_type: debugging_lesson
    confidence: 0.90
    stability: 0.80
    source: error_resolution
    topic: build
    risk: low
    last_promoted: \"2026-06-15T10:30:00Z\"
---

- **[build_java8]** Maven compile requires JAVA_HOME pointing to JDK 8.
- **[style/concise]** User prefers concise code comments.
";
        let file = parse_memory_md(raw);
        assert_eq!(file.entries.len(), 1);
        assert_eq!(file.entries[0].key, "build_java8");
        assert_eq!(file.entries[0].memory_type, MemoryType::DebuggingLesson);
        assert!(file.body.contains("**[build_java8]**"));
        assert!(file.body.contains("**[style/concise]**"));
    }

    /// 验证解析不带 frontmatter 的记忆文件。
    #[test]
    fn test_parse_memory_file_without_frontmatter() {
        let raw = "- Simple memory without frontmatter\n";
        let file = parse_memory_md(raw);
        assert!(file.entries.is_empty());
        assert!(file.body.contains("Simple memory without frontmatter"));
    }

    /// 验证解析空记忆文件。
    #[test]
    fn test_parse_memory_file_empty() {
        let file = parse_memory_md("");
        assert!(file.entries.is_empty());
        assert!(file.body.is_empty());
    }

    /// 验证 MEMORY.md 格式化与解析可往返。
    #[test]
    fn test_format_memory_md_roundtrip() {
        let file = MemoryFile {
            entries: vec![HotEntryMeta {
                key: "test_key".to_string(),
                memory_type: MemoryType::UserPreference,
                confidence: 0.95,
                stability: 0.85,
                recall_weight: 0.8,
                source: MemorySource::ExplicitSave,
                topic: "style".to_string(),
                risk: MemoryRisk::Low,
                revision_count: 2,
                recall_count: 4,
                last_recalled_at: Some(1_750_000_000),
                content_json: false,
                last_promoted: "2026-06-15T10:00:00Z".to_string(),
            }],
            body: "- **[test_key]** User prefers concise output.\n".to_string(),
        };

        let formatted = format_memory_md(&file);
        let reparsed = parse_memory_md(&formatted);
        assert_eq!(reparsed.entries.len(), 1);
        assert_eq!(reparsed.entries[0].key, "test_key");
        assert!(reparsed.body.contains("**[test_key]**"));
    }

    /// 验证空热层读取返回空内容。
    #[test]
    fn test_read_hot_content_empty() -> Result<()> {
        let manager = make_manager();
        assert!(manager.read_hot_content()?.is_empty());
        Ok(())
    }

    /// 验证预算内可以写入热层。
    #[tokio::test]
    async fn test_add_to_hot_under_budget() -> Result<()> {
        let manager = make_manager();
        let entry = TypedMemoryEntry {
            key: "test_key".to_string(),
            content: "User prefers concise output.".to_string(),
            meta: MemoryMeta::new(
                MemoryType::UserPreference,
                MemorySource::ExplicitSave,
                "style",
            ),
            raw: echo_core::memory::store::StoreItem::new(
                vec!["agent".to_string(), "typed_memories".to_string()],
                "test_key".to_string(),
                serde_json::Value::Null,
            ),
        };

        manager.add_to_hot(&entry).await.expect("add to hot");

        let content = manager.read_hot_content()?;
        assert!(content.contains("**[test_key]**"));
        assert!(content.contains("User prefers concise output"));

        let meta = manager.read_hot_meta()?;
        assert_eq!(meta.len(), 1);
        assert_eq!(meta[0].key, "test_key");
        Ok(())
    }

    /// 验证热层删除成功时只通知一次，重试删除缺失条目不再通知。
    #[tokio::test]
    async fn successful_hot_delete_notifies_once_and_missing_retry_does_not_notify() -> Result<()> {
        let dir = tempfile::tempdir()
            .map_err(|error| merge_plan_error(format!("tempdir failed: {error}")))?
            .keep();
        let observer = Arc::new(RecordingObserver::default());
        let manager =
            MemoryLayerManager::new(dir, Arc::new(InMemoryStore::new()), Box::new(NullChangeLog))
                .with_evolution_observer(observer.clone());
        let entry = TypedMemoryEntry {
            key: "delete_hot".to_string(),
            content: "Delete this hot memory".to_string(),
            meta: MemoryMeta::new(
                MemoryType::UserPreference,
                MemorySource::ExplicitSave,
                "test",
            ),
            raw: echo_core::memory::store::StoreItem::new(
                vec!["agent".to_string()],
                "delete_hot".to_string(),
                serde_json::Value::Null,
            ),
        };
        manager.add_to_hot(&entry).await.map_err(ReactError::from)?;

        assert!(manager.delete_memory("delete_hot").await?);
        assert!(!manager.delete_memory("delete_hot").await?);
        assert_eq!(
            observer.changes(),
            vec![(
                "delete_hot".to_string(),
                "hot".to_string(),
                "deleted".to_string(),
            )]
        );
        Ok(())
    }

    /// 验证热层删除审计失败时不发出层级变更通知。
    #[tokio::test]
    async fn failed_hot_delete_audit_emits_no_layer_change() -> Result<()> {
        let dir = tempfile::tempdir()
            .map_err(|error| merge_plan_error(format!("tempdir failed: {error}")))?
            .keep();
        let observer = Arc::new(RecordingObserver::default());
        let manager = MemoryLayerManager::new(
            dir,
            Arc::new(InMemoryStore::new()),
            Box::new(FailingChangeLog),
        )
        .with_evolution_observer(observer.clone());
        let entry = TypedMemoryEntry {
            key: "failed_delete".to_string(),
            content: "Audit failure".to_string(),
            meta: MemoryMeta::new(MemoryType::ProjectFact, MemorySource::ExplicitSave, "test"),
            raw: echo_core::memory::store::StoreItem::new(
                vec!["agent".to_string()],
                "failed_delete".to_string(),
                serde_json::Value::Null,
            ),
        };
        manager.add_to_hot(&entry).await.map_err(ReactError::from)?;

        assert!(manager.delete_memory("failed_delete").await.is_err());
        assert!(observer.changes().is_empty());
        assert!(manager.read_hot_content().is_err());
        assert!(manager.list_hot().is_err());
        assert!(manager.locate("failed_delete").await.is_err());
        Ok(())
    }

    /// 验证两个管理器共享同一根目录下的操作顺序。
    #[tokio::test]
    async fn two_managers_share_one_root_operation_order() -> Result<()> {
        let dir = tempfile::tempdir().map_err(ReactError::from)?;
        let root = dir.path().join(".echo-agent");
        let audit_path = root.join("evolution/change-log.jsonl");
        let store = Arc::new(FileStore::new(dir.path().join("store.json"))?);
        let first = MemoryLayerManager::new(
            root.clone(),
            store.clone(),
            Box::new(JsonlChangeLog::new(audit_path.clone())?),
        );
        let second = MemoryLayerManager::new(
            root,
            store.clone(),
            Box::new(JsonlChangeLog::new(audit_path.clone())?),
        );
        let meta = MemoryMeta::new(
            MemoryType::ProjectFact,
            MemorySource::AutoExtracted,
            "build",
        );
        let (a, b) = tokio::join!(
            first.write_memory("shared", "A", meta.clone()),
            second.write_memory("shared", "B", meta),
        );
        assert!(a?.is_none());
        assert!(b?.is_none());
        let audit =
            JsonlChangeLog::new(audit_path)?.query(&super::super::audit::ChangeFilter::new())?;
        assert_eq!(audit.len(), 2);
        assert_eq!(
            audit
                .iter()
                .filter(|entry| entry.change_type == ChangeType::Create)
                .count(),
            1
        );
        assert_eq!(
            audit
                .iter()
                .filter(|entry| entry.change_type == ChangeType::Update)
                .count(),
            1
        );
        assert_eq!(store.list(WARM_NAMESPACE).await?.len(), 1);
        Ok(())
    }

    /// 子进程探针：验证日志租约被占用时无法打开。
    #[test]
    fn journal_lease_child_probe() -> Result<()> {
        let Some(root) = std::env::var_os("ECHO_TEST_MEMORY_JOURNAL_ROOT") else {
            return Ok(());
        };
        let manager = MemoryLayerManager::try_new(
            PathBuf::from(root),
            Arc::new(InMemoryStore::new()),
            Box::new(NullChangeLog),
        );
        if manager.is_ok() {
            return Err(merge_plan_error(
                "competing process opened the live memory operation journal",
            ));
        }
        Ok(())
    }

    /// 验证管理器持有日志时第二个进程以失败关闭方式处理。
    #[test]
    fn second_process_fails_closed_while_manager_owns_journal() -> Result<()> {
        let dir = tempfile::tempdir().map_err(ReactError::from)?;
        let root = dir.path().join(".echo-agent");
        let _manager = MemoryLayerManager::try_new(
            root.clone(),
            Arc::new(InMemoryStore::new()),
            Box::new(NullChangeLog),
        )?;
        let executable = std::env::current_exe().map_err(ReactError::from)?;
        let child = std::process::Command::new(executable)
            .arg("--exact")
            .arg("evolution::layer::tests::journal_lease_child_probe")
            .env("ECHO_TEST_MEMORY_JOURNAL_ROOT", root)
            .output()
            .map_err(ReactError::from)?;
        if !child.status.success() {
            return Err(ReactError::Other(format!(
                "competing memory journal process probe failed: {}",
                String::from_utf8_lossy(&child.stdout)
            )));
        }
        Ok(())
    }

    /// 验证持久化写入在重启后只对账出一条审计。
    #[tokio::test]
    async fn durable_write_reconciles_one_audit_after_restart() -> Result<()> {
        let dir = tempfile::tempdir().map_err(ReactError::from)?;
        let root = dir.path().join(".echo-agent");
        let store_path = dir.path().join("store.json");
        let audit_path = root.join("evolution/change-log.jsonl");
        let store = Arc::new(FileStore::new(&store_path)?);
        let log = FailOnceChangeLog {
            inner: JsonlChangeLog::new(audit_path.clone())?,
            fail_next: AtomicBool::new(true),
        };
        let manager = MemoryLayerManager::new(root.clone(), store.clone(), Box::new(log));
        let meta = MemoryMeta::new(
            MemoryType::ProjectFact,
            MemorySource::AutoExtracted,
            "build",
        );

        assert!(
            manager
                .write_memory("durable", "Use cargo", meta)
                .await
                .is_err()
        );
        assert!(store.get(WARM_NAMESPACE, "durable").await?.is_some());
        assert_eq!(JsonlChangeLog::new(audit_path.clone())?.len(), 0);
        drop(manager);
        drop(store);

        let reopened = MemoryLayerManager::new(
            root,
            Arc::new(FileStore::new(store_path)?),
            Box::new(JsonlChangeLog::new(audit_path.clone())?),
        );
        reopened.reconcile_pending().await?;
        reopened.reconcile_pending().await?;
        let entries =
            JsonlChangeLog::new(audit_path)?.query(&super::super::audit::ChangeFilter::new())?;
        assert_eq!(entries.len(), 1);
        assert_eq!(
            entries.first().map(|entry| entry.entity_key.as_str()),
            Some("durable")
        );
        Ok(())
    }

    /// 验证已结算历史可修复旧的存储投影且不重复审计。
    #[tokio::test]
    async fn settled_history_repairs_older_store_projection_without_duplicate_audit() -> Result<()>
    {
        let dir = tempfile::tempdir().map_err(ReactError::from)?;
        let root = dir.path().join(".echo-agent");
        let store_path = dir.path().join("store.json");
        let audit_path = root.join("evolution/change-log.jsonl");
        let store = Arc::new(FileStore::new(&store_path)?);
        let manager = MemoryLayerManager::new(
            root.clone(),
            store.clone(),
            Box::new(JsonlChangeLog::new(audit_path.clone())?),
        );
        let meta = MemoryMeta::new(
            MemoryType::ProjectFact,
            MemorySource::AutoExtracted,
            "build",
        );
        manager
            .write_memory("settled", "Old value", meta.clone())
            .await?;
        let old = store
            .get(WARM_NAMESPACE, "settled")
            .await?
            .ok_or_else(|| merge_plan_error("missing old projection"))?
            .value;
        manager.write_memory("settled", "New value", meta).await?;
        drop(manager);
        store.put(WARM_NAMESPACE, "settled", old).await?;
        drop(store);

        let store = Arc::new(FileStore::new(&store_path)?);
        let reopened = MemoryLayerManager::new(
            root,
            store.clone(),
            Box::new(JsonlChangeLog::new(audit_path.clone())?),
        );
        assert!(reopened.read_hot_content().is_err());
        reopened.reconcile_pending().await?;
        reopened.reconcile_pending().await?;
        let restored = reopened
            .typed_store
            .get_typed(WARM_NAMESPACE, "settled")
            .await?
            .ok_or_else(|| merge_plan_error("missing repaired projection"))?;
        assert_eq!(restored.content, "New value");
        assert_eq!(JsonlChangeLog::new(audit_path)?.len(), 2);
        store
            .put(
                WARM_NAMESPACE,
                "settled",
                serde_json::json!({ "external": "different" }),
            )
            .await?;
        assert!(reopened.reconcile_pending().await.is_err());
        Ok(())
    }

    /// 验证准备阶段失败不会修改温层存储。
    #[tokio::test]
    async fn failed_prepare_does_not_mutate_warm_store() -> Result<()> {
        let dir = tempfile::tempdir().map_err(ReactError::from)?;
        let root = dir.path().join(".echo-agent");
        std::fs::create_dir_all(root.join("evolution/memory-operations.jsonl"))
            .map_err(ReactError::from)?;
        let store = Arc::new(InMemoryStore::new());
        let manager = MemoryLayerManager::new(root, store.clone(), Box::new(NullChangeLog));
        let meta = MemoryMeta::new(
            MemoryType::ProjectFact,
            MemorySource::AutoExtracted,
            "build",
        );
        assert!(
            manager
                .write_memory("no_prepare", "No mutation", meta)
                .await
                .is_err()
        );
        assert!(store.get(WARM_NAMESPACE, "no_prepare").await?.is_none());
        Ok(())
    }

    /// 验证重启后对进行到一半的晋升与降级只对账一次。
    #[tokio::test]
    async fn restart_reconciles_mid_promotion_and_demotion_once() -> Result<()> {
        for promote in [true, false] {
            let dir = tempfile::tempdir().map_err(ReactError::from)?;
            let root = dir.path().join(".echo-agent");
            let store_path = dir.path().join("store.json");
            let audit_path = root.join("evolution/change-log.jsonl");
            let store = Arc::new(FileStore::new(&store_path)?);
            let manager = MemoryLayerManager::new(
                root.clone(),
                store.clone(),
                Box::new(JsonlChangeLog::new(audit_path.clone())?),
            );
            let meta = MemoryMeta::new(
                MemoryType::UserPreference,
                MemorySource::ExplicitSave,
                "style",
            )
            .with_confidence(0.95)
            .with_stability(0.90);
            let entry = TypedMemoryEntry {
                key: "moving".into(),
                content: "Keep this memory".into(),
                meta: meta.clone(),
                raw: echo_core::memory::store::StoreItem::new(
                    WARM_NAMESPACE
                        .iter()
                        .map(|part| (*part).to_owned())
                        .collect(),
                    "moving".into(),
                    serde_json::Value::Null,
                ),
            };
            if promote {
                manager
                    .typed_store
                    .put_typed(WARM_NAMESPACE, "moving", &entry.content, meta.clone())
                    .await?;
            } else {
                manager.add_to_hot(&entry).await.map_err(ReactError::from)?;
            }
            let hot = HotValue {
                meta: MemoryLayerManager::hot_meta(&entry),
                content: entry.content.clone(),
            };
            let id = uuid::Uuid::new_v4().to_string();
            let operation = MemoryOperation {
                id: id.clone(),
                key: "moving".into(),
                warm_before: store
                    .get(WARM_NAMESPACE, "moving")
                    .await?
                    .map(|item| item.value),
                warm_after: if promote {
                    None
                } else {
                    Some(MemoryLayerManager::typed_value(&entry.content, meta)?)
                },
                hot_before: MemoryLayerManager::hot_value(&manager.parse_memory_file(), "moving"),
                hot_after: if promote { Some(hot.clone()) } else { None },
                audit: MemoryLayerManager::change_builder(
                    "moving",
                    if promote {
                        ChangeType::Promote
                    } else {
                        ChangeType::Demote
                    },
                    Some(if promote { "warm" } else { "hot" }),
                    Some(if promote { "hot" } else { "warm" }),
                    "crash window",
                    "test",
                )
                .build_with(id, chrono::Utc::now()),
            };
            manager.operation_journal()?.prepare(operation)?;
            if promote {
                let mut guard = manager.lock_hot_file().await.map_err(ReactError::from)?;
                MemoryLayerManager::set_hot_value(&mut guard.file, "moving", Some(&hot))?;
                guard.commit().map_err(ReactError::from)?;
            } else {
                manager
                    .typed_store
                    .put_typed(WARM_NAMESPACE, "moving", &entry.content, entry.meta)
                    .await?;
            }
            drop(manager);
            drop(store);

            let reopened = MemoryLayerManager::new(
                root,
                Arc::new(FileStore::new(store_path)?),
                Box::new(JsonlChangeLog::new(audit_path.clone())?),
            );
            reopened.reconcile_pending().await?;
            reopened.reconcile_pending().await?;
            let actual = reopened.locate("moving").await?.map(|(layer, _)| layer);
            assert_eq!(
                actual,
                Some(if promote {
                    MemoryLayer::Hot
                } else {
                    MemoryLayer::Warm
                })
            );
            assert_eq!(JsonlChangeLog::new(audit_path)?.len(), 1);
        }
        Ok(())
    }

    /// 验证重启后对进行到一半的温层删除与元数据更新进行对账。
    #[tokio::test]
    async fn restart_reconciles_mid_warm_delete_and_meta_update() -> Result<()> {
        for delete in [true, false] {
            let dir = tempfile::tempdir().map_err(ReactError::from)?;
            let root = dir.path().join(".echo-agent");
            let store_path = dir.path().join("store.json");
            let audit_path = root.join("evolution/change-log.jsonl");
            let store = Arc::new(FileStore::new(&store_path)?);
            let manager = MemoryLayerManager::new(
                root.clone(),
                store.clone(),
                Box::new(JsonlChangeLog::new(audit_path.clone())?),
            );
            let meta = MemoryMeta::new(
                MemoryType::ProjectFact,
                MemorySource::AutoExtracted,
                "build",
            );
            manager
                .typed_store
                .put_typed(WARM_NAMESPACE, "warm", "Original", meta.clone())
                .await?;
            let mut after_meta = meta;
            after_meta.status = MemoryStatus::Archived;
            let after = if delete {
                None
            } else {
                Some(MemoryLayerManager::typed_value("Original", after_meta)?)
            };
            let id = uuid::Uuid::new_v4().to_string();
            manager.operation_journal()?.prepare(MemoryOperation {
                id: id.clone(),
                key: "warm".into(),
                warm_before: store
                    .get(WARM_NAMESPACE, "warm")
                    .await?
                    .map(|item| item.value),
                warm_after: after.clone(),
                hot_before: None,
                hot_after: None,
                audit: MemoryLayerManager::change_builder(
                    "warm",
                    if delete {
                        ChangeType::Delete
                    } else {
                        ChangeType::Demote
                    },
                    Some("warm"),
                    if delete { None } else { Some("archived") },
                    "crash window",
                    "test",
                )
                .build_with(id, chrono::Utc::now()),
            })?;
            match after {
                Some(value) => store.put(WARM_NAMESPACE, "warm", value).await?,
                None => {
                    store.delete(WARM_NAMESPACE, "warm").await?;
                }
            }
            drop(manager);
            drop(store);

            let reopened = MemoryLayerManager::new(
                root,
                Arc::new(FileStore::new(store_path)?),
                Box::new(JsonlChangeLog::new(audit_path.clone())?),
            );
            reopened.reconcile_pending().await?;
            reopened.reconcile_pending().await?;
            let entry = reopened
                .typed_store
                .get_typed(WARM_NAMESPACE, "warm")
                .await?;
            if delete {
                assert!(entry.is_none());
            } else {
                assert_eq!(
                    entry.map(|item| item.meta.status),
                    Some(MemoryStatus::Archived)
                );
            }
            assert_eq!(JsonlChangeLog::new(audit_path)?.len(), 1);
        }
        Ok(())
    }

    /// 验证第二次失败后合并分组会重放所有成员的审计。
    #[tokio::test]
    async fn merge_group_replays_all_member_audits_after_second_failure() -> Result<()> {
        let dir = tempfile::tempdir().map_err(ReactError::from)?;
        let root = dir.path().join(".echo-agent");
        let store_path = dir.path().join("store.json");
        let audit_path = root.join("evolution/change-log.jsonl");
        let store = Arc::new(FileStore::new(&store_path)?);
        let manager = MemoryLayerManager::new(
            root.clone(),
            store.clone(),
            Box::new(FailSecondChangeLog {
                inner: JsonlChangeLog::new(audit_path.clone())?,
                calls: AtomicUsize::new(0),
            }),
        );
        let primary = MemoryMeta::new(MemoryType::ProjectFact, MemorySource::ExplicitSave, "build")
            .with_confidence(0.95);
        let secondary = MemoryMeta::new(
            MemoryType::ProjectFact,
            MemorySource::AutoExtracted,
            "build",
        )
        .with_confidence(0.55);
        manager
            .typed_store
            .put_typed(WARM_NAMESPACE, "primary", "Use cargo", primary)
            .await?;
        manager
            .typed_store
            .put_typed(WARM_NAMESPACE, "secondary", "Use make", secondary)
            .await?;
        let entries = manager.list_warm_memories(&MemoryFilter::new()).await?;
        let group = ConflictDetector::new()
            .detect(&entries)
            .into_iter()
            .next()
            .ok_or_else(|| merge_plan_error("expected conflict group"))?;
        assert!(
            MemoryMerger::new(&manager)
                .merge_group(&group)
                .await
                .is_err()
        );
        assert_eq!(JsonlChangeLog::new(audit_path.clone())?.len(), 1);
        assert!(manager.read_hot_content().is_err());
        drop(manager);
        drop(store);

        let reopened = MemoryLayerManager::new(
            root,
            Arc::new(FileStore::new(store_path)?),
            Box::new(JsonlChangeLog::new(audit_path.clone())?),
        );
        reopened.reconcile_pending().await?;
        reopened.reconcile_pending().await?;
        let state = reopened.list_warm_memories(&MemoryFilter::new()).await?;
        assert_eq!(
            state
                .iter()
                .find(|entry| entry.key == "secondary")
                .map(|entry| entry.meta.status),
            Some(MemoryStatus::Superseded)
        );
        let audit =
            JsonlChangeLog::new(audit_path)?.query(&super::super::audit::ChangeFilter::new())?;
        assert_eq!(audit.len(), 2);
        assert_eq!(
            audit
                .iter()
                .filter(|entry| entry.change_type == ChangeType::Merge)
                .count(),
            2
        );
        Ok(())
    }

    /// 验证投影中断后已准备的合并会重放剩余成员。
    #[tokio::test]
    async fn prepared_merge_replays_remaining_member_after_interrupted_projection() -> Result<()> {
        let dir = tempfile::tempdir().map_err(ReactError::from)?;
        let root = dir.path().join(".echo-agent");
        let store_path = dir.path().join("store.json");
        let audit_path = root.join("evolution/change-log.jsonl");
        let store = Arc::new(FileStore::new(&store_path)?);
        let manager = MemoryLayerManager::new(
            root.clone(),
            store.clone(),
            Box::new(JsonlChangeLog::new(audit_path.clone())?),
        );
        let mut operations = Vec::new();
        for key in ["first", "second"] {
            let before = MemoryLayerManager::typed_value(
                "Original",
                MemoryMeta::new(
                    MemoryType::ProjectFact,
                    MemorySource::AutoExtracted,
                    "merge",
                ),
            )?;
            store.put(WARM_NAMESPACE, key, before.clone()).await?;
            let mut meta = MemoryMeta::new(
                MemoryType::ProjectFact,
                MemorySource::AutoExtracted,
                "merge",
            );
            meta.status = MemoryStatus::Superseded;
            let after = MemoryLayerManager::typed_value("Original", meta)?;
            let id = uuid::Uuid::new_v4().to_string();
            operations.push(MemoryOperation {
                id: id.clone(),
                key: key.into(),
                warm_before: Some(before),
                warm_after: Some(after),
                hot_before: None,
                hot_after: None,
                audit: MemoryLayerManager::change_builder(
                    key,
                    ChangeType::Merge,
                    Some("warm"),
                    Some("superseded"),
                    "approved merge",
                    "test",
                )
                .build_with(id, chrono::Utc::now()),
            });
        }
        let batch = MemoryOperationBatch {
            id: uuid::Uuid::new_v4().to_string(),
            operations,
            origin: None,
        };
        manager.operation_journal()?.prepare_batch(batch.clone())?;
        let first = batch
            .operations
            .first()
            .ok_or_else(|| merge_plan_error("missing first member"))?;
        manager.apply_operation(first).await?;
        assert_eq!(JsonlChangeLog::new(audit_path.clone())?.len(), 0);
        drop(manager);
        drop(store);

        let reopened = MemoryLayerManager::new(
            root,
            Arc::new(FileStore::new(store_path)?),
            Box::new(JsonlChangeLog::new(audit_path.clone())?),
        );
        reopened.reconcile_pending().await?;
        reopened.reconcile_pending().await?;
        for key in ["first", "second"] {
            assert_eq!(
                reopened
                    .typed_store
                    .get_typed(WARM_NAMESPACE, key)
                    .await?
                    .map(|entry| entry.meta.status),
                Some(MemoryStatus::Superseded)
            );
        }
        assert_eq!(JsonlChangeLog::new(audit_path)?.len(), 2);
        Ok(())
    }

    /// 验证降级评分的排序。
    #[test]
    fn test_demotion_score_ordering() {
        let high_quality = HotEntryMeta {
            key: "hq".to_string(),
            memory_type: MemoryType::UserPreference,
            confidence: 0.95,
            stability: 0.90,
            recall_weight: 0.8,
            source: MemorySource::ExplicitSave,
            topic: "style".to_string(),
            risk: MemoryRisk::Low,
            revision_count: 0,
            recall_count: 0,
            last_recalled_at: None,
            content_json: false,
            last_promoted: "2026-06-15T10:00:00Z".to_string(),
        };

        let low_quality = HotEntryMeta {
            key: "lq".to_string(),
            memory_type: MemoryType::CommandPattern,
            confidence: 0.50,
            stability: 0.30,
            recall_weight: 0.4,
            source: MemorySource::AutoExtracted,
            topic: "build".to_string(),
            risk: MemoryRisk::Low,
            revision_count: 0,
            recall_count: 0,
            last_recalled_at: None,
            content_json: false,
            last_promoted: "2026-06-15T10:00:00Z".to_string(),
        };

        // 置信度/稳定性越高 → 降级分数越低
        assert!(
            MemoryLayerManager::demotion_score(&high_quality)
                < MemoryLayerManager::demotion_score(&low_quality)
        );
    }

    /// 验证提取热层条目内容。
    #[test]
    fn test_extract_hot_entry_content() {
        let body = "- **[build_java8]** Maven needs Java 8\n- **[style/concise]** Be brief\n";
        assert_eq!(
            extract_hot_entry_content(body, "build_java8"),
            "Maven needs Java 8"
        );
        assert_eq!(extract_hot_entry_content(body, "style/concise"), "Be brief");
        assert_eq!(extract_hot_entry_content(body, "nonexistent"), "");
    }

    /// 验证 token 数量估算。
    #[test]
    fn test_estimate_tokens() {
        // 纯拉丁字符
        let latin = "a".repeat(100);
        assert!(estimate_tokens(&latin) <= 30);

        // 混合字符
        let mixed = format!("Hello 世界 {}", "x".repeat(50));
        let tokens = estimate_tokens(&mixed);
        assert!(tokens > 0);
    }

    /// 验证写入记忆并考虑晋升。
    #[tokio::test]
    async fn test_write_memory_and_consider_promotion() -> Result<()> {
        let manager = make_manager();

        // 高置信度、高稳定性 → 应当符合晋升热层的资格
        let meta = MemoryMeta::new(
            MemoryType::UserPreference,
            MemorySource::ExplicitSave,
            "style",
        )
        .with_confidence(0.95)
        .with_stability(0.90);

        let result = manager
            .write_memory("test_pref", "User prefers concise output", meta)
            .await
            .expect("write_memory");

        // ExplicitSave 且高置信度/高稳定性 → 应当自动晋升
        assert!(result.is_some());
        let change = result.unwrap();
        assert_eq!(change.from_layer, MemoryLayer::Warm);
        assert_eq!(change.to_layer, MemoryLayer::Hot);

        // 验证它确实位于热层
        let content = manager.read_hot_content()?;
        assert!(content.contains("**[test_pref]**"));
        Ok(())
    }

    /// 验证晋升后的空白与多行内容在对账后保持不变。
    #[tokio::test]
    async fn promoted_whitespace_and_multiline_content_survives_reconcile() -> Result<()> {
        for content in [
            "  leading and trailing  ",
            "first line\nsecond line\n",
            "- **[formatted]** literal content",
        ] {
            let dir = tempfile::tempdir().map_err(ReactError::from)?;
            let root = dir.path().join(".echo-agent");
            let store_path = dir.path().join("store.json");
            let audit_path = root.join("evolution/change-log.jsonl");
            let store = Arc::new(FileStore::new(&store_path)?);
            let manager = MemoryLayerManager::try_new(
                root.clone(),
                store,
                Box::new(JsonlChangeLog::new(audit_path.clone())?),
            )?;
            let meta = MemoryMeta::new(
                MemoryType::UserPreference,
                MemorySource::ExplicitSave,
                "format",
            )
            .with_confidence(0.95)
            .with_stability(0.90);
            manager
                .write_memory("formatted", content, meta.clone())
                .await?;
            let (_, live) = manager
                .locate("formatted")
                .await?
                .ok_or_else(|| merge_plan_error("promoted memory missing"))?;
            assert_eq!(live.content, content);
            drop(manager);

            let reopened = MemoryLayerManager::try_new(
                root,
                Arc::new(FileStore::new(store_path)?),
                Box::new(JsonlChangeLog::new(audit_path)?),
            )?;
            reopened.reconcile_pending().await?;
            let (_, recovered) = reopened
                .locate("formatted")
                .await?
                .ok_or_else(|| merge_plan_error("recovered memory missing"))?;
            assert_eq!(recovered.content, content);
            reopened.demote("formatted", "format roundtrip").await?;
            let (_, warm) = reopened
                .locate("formatted")
                .await?
                .ok_or_else(|| merge_plan_error("demoted memory missing"))?;
            assert_eq!(warm.content, content);
            reopened
                .write_memory("subsequent", "another fact", meta)
                .await?;
        }
        Ok(())
    }

    /// 验证过期的晋升快照不能覆盖更新的温层写入。
    #[tokio::test]
    async fn stale_promotion_snapshot_cannot_overwrite_newer_warm_write() -> Result<()> {
        let dir = tempfile::tempdir().map_err(ReactError::from)?;
        let store = Arc::new(InMemoryStore::new());
        let first = MemoryLayerManager::new(
            dir.path().to_path_buf(),
            store.clone(),
            Box::new(NullChangeLog),
        );
        let second = MemoryLayerManager::new(
            dir.path().to_path_buf(),
            store.clone(),
            Box::new(NullChangeLog),
        );
        let eligible = MemoryMeta::new(
            MemoryType::UserPreference,
            MemorySource::ExplicitSave,
            "style",
        )
        .with_confidence(0.95)
        .with_stability(0.90);
        first
            .typed_store
            .put_typed(WARM_NAMESPACE, "shared", "Old fact", eligible)
            .await?;
        let (_, stale) = first
            .locate("shared")
            .await?
            .ok_or_else(|| merge_plan_error("missing initial memory"))?;
        let newer = MemoryMeta::new(
            MemoryType::ProjectFact,
            MemorySource::AutoExtracted,
            "style",
        );
        second.write_memory("shared", "New fact", newer).await?;

        assert!(first.promote_warm_to_hot("shared", stale).await.is_err());
        let (layer, current) = first
            .locate("shared")
            .await?
            .ok_or_else(|| merge_plan_error("newer memory disappeared"))?;
        assert_eq!(layer, MemoryLayer::Warm);
        assert_eq!(current.content, "New fact");
        Ok(())
    }

    /// 验证过期的降级快照不能覆盖更新的热层写入。
    #[tokio::test]
    async fn stale_demotion_snapshot_cannot_overwrite_newer_hot_write() -> Result<()> {
        let dir = tempfile::tempdir().map_err(ReactError::from)?;
        let store = Arc::new(InMemoryStore::new());
        let first = MemoryLayerManager::new(
            dir.path().to_path_buf(),
            store.clone(),
            Box::new(NullChangeLog),
        );
        let second =
            MemoryLayerManager::new(dir.path().to_path_buf(), store, Box::new(NullChangeLog));
        let eligible = MemoryMeta::new(
            MemoryType::UserPreference,
            MemorySource::ExplicitSave,
            "style",
        )
        .with_confidence(0.95)
        .with_stability(0.90);
        first
            .write_memory("shared", "Old fact", eligible.clone())
            .await?;
        let (_, stale) = first
            .locate("shared")
            .await?
            .ok_or_else(|| merge_plan_error("missing initial hot memory"))?;
        second.write_memory("shared", "New fact", eligible).await?;

        assert!(
            first
                .demote_hot_to_warm("shared", stale, "stale decision")
                .await
                .is_err()
        );
        let (layer, current) = first
            .locate("shared")
            .await?
            .ok_or_else(|| merge_plan_error("newer hot memory disappeared"))?;
        assert_eq!(layer, MemoryLayer::Hot);
        assert_eq!(current.content, "New fact");
        Ok(())
    }

    /// 验证过期的温层归档快照不能覆盖更新的写入。
    #[tokio::test]
    async fn stale_warm_archive_snapshot_cannot_overwrite_newer_write() -> Result<()> {
        let dir = tempfile::tempdir().map_err(ReactError::from)?;
        let store = Arc::new(InMemoryStore::new());
        let first = MemoryLayerManager::new(
            dir.path().to_path_buf(),
            store.clone(),
            Box::new(NullChangeLog),
        );
        let second =
            MemoryLayerManager::new(dir.path().to_path_buf(), store, Box::new(NullChangeLog));
        let meta = MemoryMeta::new(
            MemoryType::ProjectFact,
            MemorySource::AutoExtracted,
            "build",
        );
        first
            .write_memory("shared", "Old fact", meta.clone())
            .await?;
        let (_, stale) = first
            .locate("shared")
            .await?
            .ok_or_else(|| merge_plan_error("missing initial warm memory"))?;
        second.write_memory("shared", "New fact", meta).await?;
        let mut archived = stale.meta.clone();
        archived.status = MemoryStatus::Archived;

        let result = first
            .transition(
                "shared",
                Some(MemoryLayerManager::typed_value(&stale.content, archived)?),
                None,
                MemoryLayerManager::change_builder(
                    "shared",
                    ChangeType::Demote,
                    Some("warm"),
                    Some("archived"),
                    "stale decision",
                    "demote",
                ),
                Some((MemoryLayer::Warm, &stale)),
            )
            .await;
        assert!(result.is_err());
        let (_, current) = first
            .locate("shared")
            .await?
            .ok_or_else(|| merge_plan_error("newer warm memory disappeared"))?;
        assert_eq!(current.content, "New fact");
        assert_eq!(current.meta.status, MemoryStatus::Active);
        Ok(())
    }

    /// 验证不满足热层资格的记忆不会被晋升。
    #[tokio::test]
    async fn test_write_memory_not_eligible_for_hot() -> Result<()> {
        let manager = make_manager();

        // 低置信度 → 不应被晋升到热层
        let meta = MemoryMeta::new(
            MemoryType::ProjectFact,
            MemorySource::AutoExtracted,
            "project",
        )
        .with_confidence(0.50)
        .with_stability(0.30);

        let result = manager
            .write_memory("low_conf", "Some low-confidence fact", meta)
            .await
            .expect("write_memory");

        // 不符合资格 → 应保留在温层
        assert!(result.is_none());
        assert!(manager.read_hot_content()?.is_empty());
        Ok(())
    }

    /// 验证已批准的合并可以被精确恢复。
    #[tokio::test]
    async fn approved_merge_can_be_restored_exactly() -> crate::error::Result<()> {
        let store = Arc::new(InMemoryStore::new());
        let dir = tempfile::tempdir()
            .map_err(|error| merge_plan_error(format!("tempdir failed: {error}")))?
            .keep();
        let manager = MemoryLayerManager::new(dir, store, Box::new(NullChangeLog));
        let high = MemoryMeta::new(MemoryType::ProjectFact, MemorySource::ExplicitSave, "build")
            .with_confidence(0.95);
        let low = MemoryMeta::new(
            MemoryType::ProjectFact,
            MemorySource::AutoExtracted,
            "build",
        )
        .with_confidence(0.55);
        manager
            .typed_store
            .put_typed(WARM_NAMESPACE, "build_a", "Build uses cargo", high)
            .await?;
        manager
            .typed_store
            .put_typed(WARM_NAMESPACE, "build_b", "Build uses make", low)
            .await?;
        let entries = manager.list_warm_memories(&MemoryFilter::new()).await?;
        let group = ConflictDetector::new()
            .detect(&entries)
            .into_iter()
            .next()
            .ok_or_else(|| merge_plan_error("expected conflict"))?;
        let proposal = MemoryConflictProposal::from_group(&group)
            .ok_or_else(|| merge_plan_error("expected proposal"))?;

        let applied = manager.apply_merge_proposal(&proposal).await?;
        let secondary = manager
            .typed_store
            .get_typed(WARM_NAMESPACE, "build_b")
            .await?
            .ok_or_else(|| merge_plan_error("expected secondary"))?;
        assert_eq!(secondary.meta.status, MemoryStatus::Superseded);

        let batch_id = applied
            .batch_id
            .clone()
            .ok_or_else(|| merge_plan_error("merge did not return a durable batch ID"))?;
        let rollback = manager
            .rollback_memory("merge-undo-1", MemoryRollbackTarget::batch_id(batch_id))
            .await?;
        assert!(matches!(rollback, MemoryRollbackOutcome::Applied(_)));
        let primary = manager
            .typed_store
            .get_typed(WARM_NAMESPACE, "build_a")
            .await?
            .ok_or_else(|| merge_plan_error("expected restored primary"))?;
        assert_eq!(primary.meta.status, MemoryStatus::Active);
        assert_eq!(primary.content, "Build uses cargo");
        let restored = manager
            .typed_store
            .get_typed(WARM_NAMESPACE, "build_b")
            .await?
            .ok_or_else(|| merge_plan_error("expected restored secondary"))?;
        assert_eq!(restored.meta.status, MemoryStatus::Active);
        assert_eq!(restored.content, "Build uses make");
        Ok(())
    }

    /// 验证后续回滚能幂等地恢复已结算的记忆变更。
    #[tokio::test]
    async fn later_rollback_restores_a_settled_memory_change_idempotently()
    -> crate::error::Result<()> {
        let store = Arc::new(InMemoryStore::new());
        let dir = tempfile::tempdir()
            .map_err(|error| merge_plan_error(format!("tempdir failed: {error}")))?
            .keep();
        let audit_path = dir.join("evolution").join("change-log.jsonl");
        let manager = MemoryLayerManager::new(
            dir,
            store,
            Box::new(JsonlChangeLog::new(audit_path.clone())?),
        );
        let meta = MemoryMeta::new(
            MemoryType::ProjectFact,
            MemorySource::ExplicitSave,
            "rollback",
        )
        .with_confidence(0.40);
        manager
            .write_memory("rollback/key", "before", meta.clone())
            .await?;
        manager.write_memory("rollback/key", "after", meta).await?;
        let changes =
            JsonlChangeLog::new(audit_path)?.query(&super::super::audit::ChangeFilter::new())?;
        let target = MemoryRollbackTarget::change_id(
            changes
                .first()
                .ok_or_else(|| merge_plan_error("expected update audit"))?
                .change_id
                .clone(),
        );

        let preview = manager.preview_rollback(&target).await?;
        assert!(matches!(preview, MemoryRollbackPreviewOutcome::Ready(_)));
        let applied = manager.rollback_memory("request-1", target.clone()).await?;
        assert!(matches!(applied, MemoryRollbackOutcome::Applied(_)));
        let (_, restored) = manager
            .locate("rollback/key")
            .await?
            .ok_or_else(|| merge_plan_error("expected restored memory"))?;
        assert_eq!(restored.content, "before");

        let retry = manager.rollback_memory("request-1", target).await?;
        assert!(matches!(retry, MemoryRollbackOutcome::AlreadyApplied(_)));
        Ok(())
    }

    /// 验证后续回滚会拒绝非最新目标与 ABA 目标。
    #[tokio::test]
    async fn later_rollback_rejects_non_tip_and_aba_targets() -> crate::error::Result<()> {
        let store = Arc::new(InMemoryStore::new());
        let dir = tempfile::tempdir()
            .map_err(|error| merge_plan_error(format!("tempdir failed: {error}")))?
            .keep();
        let audit_path = dir.join("evolution").join("change-log.jsonl");
        let manager = MemoryLayerManager::new(
            dir,
            store,
            Box::new(JsonlChangeLog::new(audit_path.clone())?),
        );
        let meta = MemoryMeta::new(
            MemoryType::ProjectFact,
            MemorySource::ExplicitSave,
            "rollback",
        )
        .with_confidence(0.40);
        manager.write_memory("aba", "same", meta.clone()).await?;
        manager
            .write_memory("aba", "different", meta.clone())
            .await?;
        manager.write_memory("aba", "same", meta).await?;
        let changes = JsonlChangeLog::new(audit_path)?.query(&ChangeFilter::new())?;
        let first = changes
            .get(2)
            .ok_or_else(|| merge_plan_error("expected three memory changes"))?;
        let outcome = manager
            .preview_rollback(&MemoryRollbackTarget::change_id(&first.change_id))
            .await?;
        assert!(matches!(
            outcome,
            MemoryRollbackPreviewOutcome::Conflict(MemoryRollbackConflict { .. })
        ));
        Ok(())
    }

    /// 验证回滚的回滚是一个新批次，且请求冲突时失败关闭。
    #[tokio::test]
    async fn rollback_of_rollback_is_a_new_batch_and_request_conflicts_fail_closed()
    -> crate::error::Result<()> {
        let store = Arc::new(InMemoryStore::new());
        let dir = tempfile::tempdir()
            .map_err(|error| merge_plan_error(format!("tempdir failed: {error}")))?
            .keep();
        let audit_path = dir.join("evolution").join("change-log.jsonl");
        let manager = MemoryLayerManager::new(
            dir,
            store,
            Box::new(JsonlChangeLog::new(audit_path.clone())?),
        );
        let meta = MemoryMeta::new(
            MemoryType::ProjectFact,
            MemorySource::ExplicitSave,
            "rollback",
        )
        .with_confidence(0.40);
        manager
            .write_memory("lineage", "before", meta.clone())
            .await?;
        manager.write_memory("lineage", "after", meta).await?;
        let changes = JsonlChangeLog::new(audit_path)?.query(&ChangeFilter::new())?;
        let target = MemoryRollbackTarget::change_id(
            changes
                .first()
                .ok_or_else(|| merge_plan_error("expected target change"))?
                .change_id
                .clone(),
        );
        let applied = manager.rollback_memory("rollback-a", target).await?;
        let receipt = match applied {
            MemoryRollbackOutcome::Applied(receipt) => receipt,
            other => return Err(merge_plan_error(format!("unexpected outcome: {other:?}"))),
        };
        let conflict = manager
            .rollback_memory(
                "rollback-a",
                MemoryRollbackTarget::batch_id(receipt.inverse_batch_id.clone()),
            )
            .await?;
        assert!(matches!(
            conflict,
            MemoryRollbackOutcome::Conflict(MemoryRollbackConflict { .. })
        ));

        let reverted = manager
            .rollback_memory(
                "rollback-b",
                MemoryRollbackTarget::batch_id(receipt.inverse_batch_id),
            )
            .await?;
        assert!(matches!(reverted, MemoryRollbackOutcome::Applied(_)));
        let (_, current) = manager
            .locate("lineage")
            .await?
            .ok_or_else(|| merge_plan_error("expected lineage memory"))?;
        assert_eq!(current.content, "after");
        Ok(())
    }

    /// 验证回滚回执在管理器重启后仍然保留。
    #[tokio::test]
    async fn rollback_receipt_survives_manager_restart() -> crate::error::Result<()> {
        let dir = tempfile::tempdir()
            .map_err(|error| merge_plan_error(format!("tempdir failed: {error}")))?;
        let root = dir.path().join(".echo-agent");
        let store_path = dir.path().join("store.json");
        let audit_path = root.join("evolution").join("change-log.jsonl");
        let meta = MemoryMeta::new(
            MemoryType::ProjectFact,
            MemorySource::ExplicitSave,
            "rollback",
        )
        .with_confidence(0.40);
        let target;
        {
            let store = Arc::new(FileStore::new(&store_path)?);
            let manager = MemoryLayerManager::try_new(
                root.clone(),
                store,
                Box::new(JsonlChangeLog::new(audit_path.clone())?),
            )?;
            manager
                .write_memory("restart", "before", meta.clone())
                .await?;
            manager.write_memory("restart", "after", meta).await?;
            let changes = JsonlChangeLog::new(audit_path.clone())?.query(&ChangeFilter::new())?;
            target = MemoryRollbackTarget::change_id(
                changes
                    .first()
                    .ok_or_else(|| merge_plan_error("expected restart target"))?
                    .change_id
                    .clone(),
            );
            let applied = manager
                .rollback_memory("restart-request", target.clone())
                .await?;
            assert!(matches!(applied, MemoryRollbackOutcome::Applied(_)));
        }
        let store = Arc::new(FileStore::new(&store_path)?);
        let manager =
            MemoryLayerManager::try_new(root, store, Box::new(JsonlChangeLog::new(audit_path)?))?;
        manager.reconcile_pending().await?;
        let (_, restored) = manager
            .locate("restart")
            .await?
            .ok_or_else(|| merge_plan_error("expected restarted memory"))?;
        assert_eq!(restored.content, "before");
        let retry = manager.rollback_memory("restart-request", target).await?;
        assert!(matches!(retry, MemoryRollbackOutcome::AlreadyApplied(_)));
        Ok(())
    }

    /// 验证回滚审计失败后重启会对账逆操作。
    #[tokio::test]
    async fn rollback_audit_failure_reconciles_inverse_after_restart() -> crate::error::Result<()> {
        let dir = tempfile::tempdir()
            .map_err(|error| merge_plan_error(format!("tempdir failed: {error}")))?;
        let root = dir.path().join(".echo-agent");
        let store_path = dir.path().join("store.json");
        let audit_path = root.join("evolution").join("change-log.jsonl");
        let meta = MemoryMeta::new(
            MemoryType::ProjectFact,
            MemorySource::ExplicitSave,
            "rollback",
        )
        .with_confidence(0.40);
        let target;
        {
            let manager = MemoryLayerManager::try_new(
                root.clone(),
                Arc::new(FileStore::new(&store_path)?),
                Box::new(JsonlChangeLog::new(audit_path.clone())?),
            )?;
            manager
                .write_memory("audit-crash", "before", meta.clone())
                .await?;
            manager.write_memory("audit-crash", "after", meta).await?;
            let changes = JsonlChangeLog::new(audit_path.clone())?.query(&ChangeFilter::new())?;
            target = MemoryRollbackTarget::change_id(
                changes
                    .first()
                    .ok_or_else(|| merge_plan_error("expected audit-crash target"))?
                    .change_id
                    .clone(),
            );
        }
        {
            let failing_log = FailOnceChangeLog {
                inner: JsonlChangeLog::new(audit_path.clone())?,
                fail_next: AtomicBool::new(true),
            };
            let manager = MemoryLayerManager::try_new(
                root.clone(),
                Arc::new(FileStore::new(&store_path)?),
                Box::new(failing_log),
            )?;
            assert!(
                manager
                    .rollback_memory("audit-crash-request", target.clone())
                    .await
                    .is_err()
            );
        }
        let manager = MemoryLayerManager::try_new(
            root,
            Arc::new(FileStore::new(&store_path)?),
            Box::new(JsonlChangeLog::new(audit_path.clone())?),
        )?;
        manager.reconcile_pending().await?;
        let (_, restored) = manager
            .locate("audit-crash")
            .await?
            .ok_or_else(|| merge_plan_error("expected audit-crash memory"))?;
        assert_eq!(restored.content, "before");
        let retry = manager
            .rollback_memory("audit-crash-request", target)
            .await?;
        assert!(matches!(retry, MemoryRollbackOutcome::AlreadyApplied(_)));
        Ok(())
    }

    /// 验证逆投影与审计矩阵与真实记忆结果一致。
    #[tokio::test]
    async fn inverse_projection_and_audit_matrix_matches_real_memory_outcomes()
    -> crate::error::Result<()> {
        let store = Arc::new(InMemoryStore::new());
        let root = tempfile::tempdir()
            .map_err(|error| merge_plan_error(format!("tempdir failed: {error}")))?
            .keep();
        let audit_path = root.join("evolution").join("change-log.jsonl");
        let manager = MemoryLayerManager::new(
            root,
            store,
            Box::new(JsonlChangeLog::new(audit_path.clone())?),
        );
        let warm_meta = MemoryMeta::new(
            MemoryType::ProjectFact,
            MemorySource::ExplicitSave,
            "rollback-matrix",
        )
        .with_confidence(0.40);
        let hot_meta = MemoryMeta::new(
            MemoryType::UserPreference,
            MemorySource::ExplicitSave,
            "rollback-matrix",
        )
        .with_confidence(0.95)
        .with_stability(0.90);

        manager
            .write_memory("matrix-create", "created", warm_meta.clone())
            .await?;
        let create = latest_memory_audit(&audit_path, "matrix-create")?;
        manager
            .rollback_memory(
                "matrix-create-request",
                MemoryRollbackTarget::change_id(create.change_id),
            )
            .await?;
        let inverse_create = latest_memory_audit(&audit_path, "matrix-create")?;
        assert_machine_readable_rollback_summary(&inverse_create)?;
        assert_eq!(inverse_create.change_type, ChangeType::Delete);
        assert_eq!(audit_projection_layer(&inverse_create, true), Some("warm"));
        assert_eq!(
            audit_projection_layer(&inverse_create, false),
            Some("absent")
        );
        assert!(manager.locate("matrix-create").await?.is_none());

        manager
            .write_memory("matrix-update", "before", warm_meta.clone())
            .await?;
        manager
            .write_memory("matrix-update", "after", warm_meta.clone())
            .await?;
        let update = latest_memory_audit(&audit_path, "matrix-update")?;
        manager
            .rollback_memory(
                "matrix-update-request",
                MemoryRollbackTarget::change_id(update.change_id),
            )
            .await?;
        let inverse_update = latest_memory_audit(&audit_path, "matrix-update")?;
        assert_machine_readable_rollback_summary(&inverse_update)?;
        assert_eq!(inverse_update.change_type, ChangeType::Update);
        assert_eq!(audit_projection_layer(&inverse_update, true), Some("warm"));
        assert_eq!(audit_projection_layer(&inverse_update, false), Some("warm"));
        let (_, updated) = manager
            .locate("matrix-update")
            .await?
            .ok_or_else(|| merge_plan_error("matrix update memory disappeared"))?;
        assert_eq!(updated.content, "before");

        manager
            .write_memory("matrix-delete", "deleted", warm_meta.clone())
            .await?;
        assert!(manager.delete_memory("matrix-delete").await?);
        let delete = latest_memory_audit(&audit_path, "matrix-delete")?;
        manager
            .rollback_memory(
                "matrix-delete-request",
                MemoryRollbackTarget::change_id(delete.change_id),
            )
            .await?;
        let inverse_delete = latest_memory_audit(&audit_path, "matrix-delete")?;
        assert_machine_readable_rollback_summary(&inverse_delete)?;
        assert_eq!(inverse_delete.change_type, ChangeType::Create);
        assert_eq!(
            audit_projection_layer(&inverse_delete, true),
            Some("absent")
        );
        assert_eq!(audit_projection_layer(&inverse_delete, false), Some("warm"));
        let (_, restored_delete) = manager
            .locate("matrix-delete")
            .await?
            .ok_or_else(|| merge_plan_error("matrix delete memory was not restored"))?;
        assert_eq!(restored_delete.content, "deleted");

        manager
            .write_memory("matrix-promote", "promoted", hot_meta.clone())
            .await?;
        let promote = latest_memory_audit(&audit_path, "matrix-promote")?;
        assert_eq!(promote.change_type, ChangeType::Promote);
        manager
            .rollback_memory(
                "matrix-promote-request",
                MemoryRollbackTarget::change_id(promote.change_id),
            )
            .await?;
        let inverse_promote = latest_memory_audit(&audit_path, "matrix-promote")?;
        assert_machine_readable_rollback_summary(&inverse_promote)?;
        assert_eq!(inverse_promote.change_type, ChangeType::Demote);
        assert_eq!(audit_projection_layer(&inverse_promote, true), Some("hot"));
        assert_eq!(
            audit_projection_layer(&inverse_promote, false),
            Some("warm")
        );
        assert!(matches!(
            manager.locate("matrix-promote").await?,
            Some((MemoryLayer::Warm, _))
        ));

        manager
            .write_memory("matrix-demote", "demoted", hot_meta)
            .await?;
        manager.demote("matrix-demote", "matrix demotion").await?;
        let demote = latest_memory_audit(&audit_path, "matrix-demote")?;
        assert_eq!(demote.change_type, ChangeType::Demote);
        manager
            .rollback_memory(
                "matrix-demote-request",
                MemoryRollbackTarget::change_id(demote.change_id),
            )
            .await?;
        let inverse_demote = latest_memory_audit(&audit_path, "matrix-demote")?;
        assert_machine_readable_rollback_summary(&inverse_demote)?;
        assert_eq!(inverse_demote.change_type, ChangeType::Promote);
        assert_eq!(audit_projection_layer(&inverse_demote, true), Some("warm"));
        assert_eq!(audit_projection_layer(&inverse_demote, false), Some("hot"));
        assert!(matches!(
            manager.locate("matrix-demote").await?,
            Some((MemoryLayer::Hot, _))
        ));

        manager
            .write_memory("matrix-meta", "metadata", warm_meta.clone())
            .await?;
        manager.demote("matrix-meta", "archive metadata").await?;
        let meta_change = latest_memory_audit(&audit_path, "matrix-meta")?;
        manager
            .rollback_memory(
                "matrix-meta-request",
                MemoryRollbackTarget::change_id(meta_change.change_id),
            )
            .await?;
        let inverse_meta = latest_memory_audit(&audit_path, "matrix-meta")?;
        assert_machine_readable_rollback_summary(&inverse_meta)?;
        assert_eq!(inverse_meta.change_type, ChangeType::Promote);
        assert_eq!(audit_projection_layer(&inverse_meta, true), Some("warm"));
        assert_eq!(audit_projection_layer(&inverse_meta, false), Some("warm"));
        let (_, restored_meta) = manager
            .locate("matrix-meta")
            .await?
            .ok_or_else(|| merge_plan_error("matrix metadata memory disappeared"))?;
        assert_eq!(restored_meta.meta.status, MemoryStatus::Active);

        let multiline = "  first line\nsecond line  \n";
        manager
            .write_memory("matrix-multiline", multiline, warm_meta)
            .await?;
        let multiline_create = latest_memory_audit(&audit_path, "matrix-multiline")?;
        manager
            .rollback_memory(
                "matrix-multiline-delete",
                MemoryRollbackTarget::change_id(multiline_create.change_id),
            )
            .await?;
        let multiline_delete = latest_memory_audit(&audit_path, "matrix-multiline")?;
        manager
            .rollback_memory(
                "matrix-multiline-restore",
                MemoryRollbackTarget::change_id(multiline_delete.change_id),
            )
            .await?;
        let multiline_restore = latest_memory_audit(&audit_path, "matrix-multiline")?;
        assert_machine_readable_rollback_summary(&multiline_restore)?;
        assert_eq!(multiline_restore.change_type, ChangeType::Create);
        assert_eq!(
            audit_projection_layer(&multiline_restore, true),
            Some("absent")
        );
        assert_eq!(
            audit_projection_layer(&multiline_restore, false),
            Some("warm")
        );
        let (_, restored_multiline) = manager
            .locate("matrix-multiline")
            .await?
            .ok_or_else(|| merge_plan_error("matrix multiline memory disappeared"))?;
        assert_eq!(restored_multiline.content, multiline);
        Ok(())
    }

    /// 验证过期的合并提案在修改前即失败。
    #[tokio::test]
    async fn stale_merge_proposal_fails_before_mutation() -> crate::error::Result<()> {
        let store = Arc::new(InMemoryStore::new());
        let dir = tempfile::tempdir()
            .map_err(|error| merge_plan_error(format!("tempdir failed: {error}")))?
            .keep();
        let manager = MemoryLayerManager::new(dir, store, Box::new(NullChangeLog));
        let meta = MemoryMeta::new(MemoryType::ProjectFact, MemorySource::ExplicitSave, "build");
        manager
            .typed_store
            .put_typed(WARM_NAMESPACE, "a", "cargo", meta.clone())
            .await?;
        manager
            .typed_store
            .put_typed(WARM_NAMESPACE, "b", "make", meta.clone())
            .await?;
        let entries = manager.list_warm_memories(&MemoryFilter::new()).await?;
        let group = ConflictDetector::new()
            .detect(&entries)
            .into_iter()
            .next()
            .ok_or_else(|| merge_plan_error("expected conflict"))?;
        let proposal = MemoryConflictProposal::from_group(&group)
            .ok_or_else(|| merge_plan_error("expected proposal"))?;
        manager
            .typed_store
            .put_typed(WARM_NAMESPACE, "b", "ninja", meta)
            .await?;

        assert!(manager.apply_merge_proposal(&proposal).await.is_err());
        let a = manager
            .typed_store
            .get_typed(WARM_NAMESPACE, "a")
            .await?
            .ok_or_else(|| merge_plan_error("expected primary"))?;
        assert_eq!(a.meta.status, MemoryStatus::Active);
        Ok(())
    }

    /// 验证能在温层中定位条目。
    #[tokio::test]
    async fn test_locate_in_warm() -> Result<()> {
        let manager = make_manager();

        let meta = MemoryMeta::new(
            MemoryType::ProjectFact,
            MemorySource::AutoExtracted,
            "project",
        );
        manager
            .typed_store
            .put_typed(WARM_NAMESPACE, "warm_key", "A warm memory", meta)
            .await
            .expect("put_typed");

        let location = manager.locate("warm_key").await?;
        assert!(location.is_some());
        let (layer, _) = location.unwrap();
        assert_eq!(layer, MemoryLayer::Warm);
        Ok(())
    }

    /// 验证定位不存在的条目返回空。
    #[tokio::test]
    async fn test_locate_not_found() -> Result<()> {
        let manager = make_manager();
        let location = manager.locate("nonexistent").await?;
        assert!(location.is_none());
        Ok(())
    }

    /// 验证将热层条目降级到温层。
    #[tokio::test]
    async fn test_demote_hot_to_warm() -> Result<()> {
        let manager = make_manager();

        let entry = TypedMemoryEntry {
            key: "demote_test".to_string(),
            content: "Test memory to demote".to_string(),
            meta: MemoryMeta::new(
                MemoryType::UserPreference,
                MemorySource::ExplicitSave,
                "style",
            )
            .with_confidence(0.95)
            .with_stability(0.90),
            raw: echo_core::memory::store::StoreItem::new(
                vec!["agent".to_string()],
                "demote_test".to_string(),
                serde_json::Value::Null,
            ),
        };

        manager.add_to_hot(&entry).await.expect("add to hot");
        assert!(manager.read_hot_content()?.contains("**[demote_test]**"));

        let result = manager
            .demote("demote_test", "test demotion")
            .await
            .expect("demote");
        assert_eq!(result.from_layer, MemoryLayer::Hot);
        assert_eq!(result.to_layer, MemoryLayer::Warm);

        // 不应再位于热层
        assert!(!manager.read_hot_content()?.contains("**[demote_test]**"));

        // 应位于温层
        let location = manager.locate("demote_test").await?;
        assert!(location.is_some());
        let (layer, _) = location.unwrap();
        assert_eq!(layer, MemoryLayer::Warm);
        Ok(())
    }

    /// 验证分层搜索时热层结果排在前面。
    #[tokio::test]
    async fn test_search_layered_returns_hot_first() {
        let manager = make_manager();

        // 添加到热层
        let hot_entry = TypedMemoryEntry {
            key: "hot_build".to_string(),
            content: "Hot build memory".to_string(),
            meta: MemoryMeta::new(
                MemoryType::DebuggingLesson,
                MemorySource::ExplicitSave,
                "build",
            )
            .with_confidence(0.95)
            .with_stability(0.90),
            raw: echo_core::memory::store::StoreItem::new(
                vec!["agent".to_string()],
                "hot_build".to_string(),
                serde_json::Value::Null,
            ),
        };
        manager.add_to_hot(&hot_entry).await.expect("add to hot");

        // 添加到温层
        let warm_meta = MemoryMeta::new(
            MemoryType::ProjectFact,
            MemorySource::AutoExtracted,
            "build",
        );
        manager
            .typed_store
            .put_typed(WARM_NAMESPACE, "warm_build", "Warm build memory", warm_meta)
            .await
            .expect("put_typed");

        let results = manager
            .search_layered("build", 10)
            .await
            .expect("search_layered");

        // 热层结果应排在前面
        if results.len() >= 2 {
            assert_eq!(results[0].0, MemoryLayer::Hot);
        }
        assert!(results.iter().any(|(l, _)| *l == MemoryLayer::Hot));
    }

    /// 验证 MemoryLayer 的 Display 输出。
    #[test]
    fn test_memory_layer_display() {
        assert_eq!(MemoryLayer::Hot.to_string(), "hot");
        assert_eq!(MemoryLayer::Warm.to_string(), "warm");
    }

    /** P0-3（MEMORY.md 无锁）的回归测试：在同一个 `MemoryLayerManager` 上
     * 并发调用两次 `add_to_hot`，两次写入都必须生效。在加锁之前，
     * 每次调用都会读取文件、追加自己的条目、再写回——第二次写回会
     * 悄无声息地丢弃第一个写入者的条目。 */
    #[tokio::test]
    async fn test_concurrent_add_to_hot_does_not_lose_entries() -> Result<()> {
        use std::sync::Arc;
        let dir = tempfile::tempdir().expect("tempdir");
        let dir_path = dir.keep();
        let store = Arc::new(InMemoryStore::new());
        let manager = Arc::new(MemoryLayerManager::new(
            dir_path,
            store,
            Box::new(NullChangeLog),
        ));

        let mk_entry = |key: &'static str| TypedMemoryEntry {
            key: key.to_string(),
            content: format!("memory {key}"),
            meta: MemoryMeta::new(
                MemoryType::UserPreference,
                MemorySource::ExplicitSave,
                "topic",
            ),
            raw: echo_core::memory::store::StoreItem::new(
                vec!["agent".to_string()],
                key.to_string(),
                serde_json::Value::Null,
            ),
        };

        // 同时触发两次添加。
        let m1 = manager.clone();
        let m2 = manager.clone();
        let e1 = mk_entry("concurrent-a");
        let e2 = mk_entry("concurrent-b");
        let (r1, r2) = tokio::join!(async move { m1.add_to_hot(&e1).await }, async move {
            m2.add_to_hot(&e2).await
        },);
        r1.expect("add a");
        r2.expect("add b");

        let content = manager.read_hot_content()?;
        assert!(
            content.contains("**[concurrent-a]**"),
            "concurrent-a lost (MEMORY.md TOCTOU regression)"
        );
        assert!(
            content.contains("**[concurrent-b]**"),
            "concurrent-b lost (MEMORY.md TOCTOU regression)"
        );
        assert_eq!(manager.read_hot_meta()?.len(), 2);
        Ok(())
    }
}
