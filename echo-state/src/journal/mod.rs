/*!
带检查点复合归约（checkpoint-compounded reduction）的有序事件日志。

# 目的

本模块是框架侧的通用原语，用于实现“仅追加的有序日志 + 检查点投影”模式；
EKO 目前将该模式实现了两次（普通聊天的 `ChatEventLog`，以及 TaskRuntime 的
`events.jsonl` + `checkpoint.json` 组合）。本模块刻意不依赖任何 EKO 产品类型：
任何需要持久、可重放、有序事实且热路径读取有界的消费方，都可以组合这些构件。

# 业界依据

- OpenAI Codex rollout 记录器：完整 rollout 独立且持久地追加；消费方从稳定
  边界派生有界视图，而不是从头重新推导状态。
- LangGraph 持久化：执行进度以 checkpoint / 线程标识对的形式保存，恢复时从
  已持久化的事实继续，而不是重放全部历史。
- EventStoreDB/KurrentDB 的 expected-version 追加：一个批次要么作为一次有序
  提交被接受，要么被拒绝，且不会暴露任何记录前缀。
- RocksDB `WriteBatch`：一个批次就是物理原子性边界，而不是对若干各自可见的
  写入进行循环。
- Apache Kafka 的日志与提交标记模型：不可变分段划分物理 I/O，而提交边界在
  同一个全局偏移空间内控制可见性。
- Rust [`std::fs::File::sync_data`]：写入完成与其请求的持久性屏障是两个独立
  结果，这也是回执可能“已提交但持久性降级”的原因。
- 经典事件溯源快照：将事件折叠为状态，并周期性持久化
  `(state, applied_sequence)`，使恢复时只需重放尾部。

# 分层决策

日志排序、持久追加、容错重放和检查点复合是任何 agent 运行时都需要的通用机制，
因此它们位于框架层（`echo-state`）。分段实现只负责其内部文件命名与完整性契约。
数据根目录与流目录的选择、保留数量、固定（pin）以及产品投影均由消费方负责。
将 EKO 中重复的文件算法迁移到本模块是计划中的后续工作，届时会删除重复算法；
本模块不依赖于该迁移。

# 不变量

- 序号从 1 开始且连续（`1, 2, 3, ...`）。日志加载若发现缺口或非单调记录，
  会显式失败，而不是静默地错误重放。
- 日志实现会串行化其追加权限。基于文件的实现还会额外持有一个进程生命周期内
  独占的租约。
- 每个物理行都是一个受摘要保护的批次帧。被撕裂的尾部帧会使整个批次不可见，
  并在下次打开时被截断；任何完整帧中的损坏都会以失败关闭（fail closed）处理。
- [`EventReducer::apply`] 不会失败：日志事件在追加前已校验，投影代码必须对其
  自身事件类型保持全函数（total）。
- 一个 [`CheckpointedReducer`] 是其日志唯一的投影所有者。它将追加、折叠、
  检查点和恢复串行化为一个事务，使并发调用方无法打乱已提交序号的顺序。
- 每个检查点都携带产生它的事实所属的确切 [`JournalIdentity`]。序号绝不会在该
  日志代（generation）之外被信任。

# 示例

```
use echo_state::journal::{
    CheckpointStore, CheckpointedReducer, EventReducer, MemoryCheckpointStore,
    MemoryEventJournal, PreparedJournalBatch,
};
use std::sync::Arc;

#[derive(Default, serde::Serialize, serde::Deserialize)]
struct CountingReducer {
    applied: u64,
}

impl EventReducer for CountingReducer {
    type Event = String;
    fn apply(&mut self, _event: &String) {
        self.applied = self.applied.saturating_add(1);
    }
}

# fn main() -> echo_core::error::Result<()> {
let journal = Arc::new(MemoryEventJournal::<String>::new());
let checkpoints: Arc<dyn CheckpointStore<CountingReducer>> =
    Arc::new(MemoryCheckpointStore::new());
let reducer = CheckpointedReducer::new(Arc::clone(&journal), Arc::clone(&checkpoints), 2);

let batch = PreparedJournalBatch::new(vec!["one".to_string(), "two".to_string()])
    .map_err(|error| echo_core::error::ReactError::Other(error.to_string()))?;
let committed = reducer
    .apply_batch(batch)
    .map_err(|error| echo_core::error::ReactError::Other(error.to_string()))?;
assert_eq!(committed.record_count, 2); // 一个物理提交帧
reducer
    .apply("three".to_string())
    .map_err(|error| echo_core::error::ReactError::Other(error.to_string()))?;
assert_eq!(reducer.last_applied_sequence(), 3);

// 恢复时加载检查点，并只重放尾部。
let recovered = CheckpointedReducer::new(journal, checkpoints, 2);
assert_eq!(recovered.recover()?.last_applied_sequence, 3);
recovered.with_state(|state| assert_eq!(state.applied, 3));
# Ok(())
}
```
*/

/// 基于单文件的日志与检查点存储实现。
pub mod file;
/// 基于分段文件的日志实现。
pub mod segmented;

pub use file::{FileCheckpointStore, FileEventJournal};
pub use segmented::{
    JournalPhysicalCleanupStatus, JournalPhysicalSegmentMetadata, JournalPruneCommitStatus,
    JournalPruneReceipt, JournalRetentionMetadata, JournalSegmentMetadata,
    SegmentedFileEventJournal,
};

use echo_core::error::{ReactError, Result};
use echo_core::utils::canonical_json::canonical_json_bytes;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fmt;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, Weak};

/// 测试专用的通用错误类型。
#[cfg(test)]
pub(crate) type TestError = Box<dyn std::error::Error + Send + Sync>;

/// 测试专用的结果类型。
#[cfg(test)]
pub(crate) type TestResult<T = ()> = std::result::Result<T, TestError>;

/// 以给定消息构造一个测试失败错误。
#[cfg(test)]
pub(crate) fn test_failure(message: impl Into<String>) -> TestError {
    Box::new(std::io::Error::other(message.into()))
}

/// 为测试中的 `Result` / `Option` 附加上下文信息的辅助 trait。
#[cfg(test)]
pub(crate) trait TestContext<T> {
    /// 将值转换为 [`TestResult`]，失败时附带 `context` 描述。
    fn test_context(self, context: &str) -> TestResult<T>;
}

#[cfg(test)]
impl<T, E: fmt::Debug> TestContext<T> for std::result::Result<T, E> {
    /// `Err` 时附带上下文转换为测试错误。
    fn test_context(self, context: &str) -> TestResult<T> {
        match self {
            Ok(value) => Ok(value),
            Err(error) => Err(test_failure(format!("{context}: {error:?}"))),
        }
    }
}

#[cfg(test)]
impl<T> TestContext<T> for Option<T> {
    /// `None` 时附带上下文转换为测试错误。
    fn test_context(self, context: &str) -> TestResult<T> {
        match self {
            Some(value) => Ok(value),
            None => Err(test_failure(format!("{context}: value was absent"))),
        }
    }
}

/// 日志批次帧的 schema 版本号。
const JOURNAL_BATCH_SCHEMA_VERSION: u16 = 2;

/*
基于文件的运行时缓存通常只保留少得多的存活权限对象。软阈值区间按固定节奏扫描；
硬上限则立即扫描，以限制失效路径的数量。这属于生命周期卫生管理，而非淘汰策略：
存活的权限对象永远不会被移除。
*/
/// 触发按节奏清理失效条目的软阈值。
pub(super) const WEAK_REGISTRY_PRUNE_THRESHOLD: usize = 128;
/// 软阈值区间内两次清理之间的打开次数间隔。
pub(super) const WEAK_REGISTRY_PRUNE_INTERVAL: usize = 32;
/// 达到后立即清理失效条目的硬上限。
pub(super) const WEAK_REGISTRY_HARD_LIMIT: usize = 256;

/// 弱引用注册表中的单个条目。
struct WeakRegistryEntry<T> {
    /// 指向共享权限对象的弱引用。
    authority: Weak<T>,
    /// 当前已成功返回且尚未释放的日志句柄数量。
    journal_handle_count: usize,
}

/// 按路径索引共享权限对象的弱引用注册表，用于同一路径复用同一权限对象。
pub(super) struct WeakRegistry<T> {
    /// 路径到注册条目的映射。
    entries: HashMap<PathBuf, WeakRegistryEntry<T>>,
    /// 自上次清理以来的打开次数。
    opens_since_prune: usize,
}

impl<T> WeakRegistry<T> {
    /// 创建一个空注册表。
    pub(super) fn new() -> Self {
        Self {
            entries: HashMap::new(),
            opens_since_prune: 0,
        }
    }

    /// 在达到软节奏或硬上限时，清理已失效（无强引用）的条目。
    pub(super) fn prune_dead_if_due(&mut self) {
        self.opens_since_prune = self.opens_since_prune.saturating_add(1);
        let len = self.entries.len();
        let hard_limit_due = len >= WEAK_REGISTRY_HARD_LIMIT;
        let cadence_due = (WEAK_REGISTRY_PRUNE_THRESHOLD..WEAK_REGISTRY_HARD_LIMIT).contains(&len)
            && self.opens_since_prune >= WEAK_REGISTRY_PRUNE_INTERVAL;
        if !hard_limit_due && !cadence_due {
            return;
        }
        self.entries
            .retain(|_, entry| entry.authority.strong_count() > 0);
        self.opens_since_prune = 0;
    }

    /// 尝试将路径对应的弱引用升级为强引用。
    pub(super) fn upgrade(&self, path: &Path) -> Option<Arc<T>> {
        self.entries
            .get(path)
            .and_then(|entry| entry.authority.upgrade())
    }

    /// 为路径插入新的权限对象条目，初始句柄计数为 1。
    pub(super) fn insert(&mut self, path: PathBuf, authority: &Arc<T>) {
        self.entries.insert(
            path,
            WeakRegistryEntry {
                authority: Arc::downgrade(authority),
                journal_handle_count: 1,
            },
        );
    }

    /// 若路径条目恰好指向同一权限对象，则将其句柄计数加一；成功返回 `true`。
    pub(super) fn add_handle(&mut self, path: &Path, authority: &Arc<T>) -> bool {
        let authority = Arc::downgrade(authority);
        let Some(entry) = self.entries.get_mut(path) else {
            return false;
        };
        if !entry.authority.ptr_eq(&authority) {
            return false;
        }
        let Some(next) = entry.journal_handle_count.checked_add(1) else {
            return false;
        };
        entry.journal_handle_count = next;
        true
    }

    /** 释放一个已成功返回的日志句柄。返回 `true` 表示该确切注册条目的计数归零并已被移除。 */
    pub(super) fn release_handle(&mut self, path: &Path, authority: &Arc<T>) -> bool {
        let authority = Arc::downgrade(authority);
        let Some(entry) = self.entries.get_mut(path) else {
            return false;
        };
        if !entry.authority.ptr_eq(&authority) || entry.journal_handle_count == 0 {
            return false;
        }
        entry.journal_handle_count = entry.journal_handle_count.saturating_sub(1);
        if entry.journal_handle_count != 0 {
            return false;
        }
        self.entries.remove(path);
        true
    }

    /// 统计位于 `root` 之下的注册路径数量（仅测试使用）。
    #[cfg(test)]
    pub(super) fn paths_beneath(&self, root: &Path) -> usize {
        self.entries
            .keys()
            .filter(|path| path.starts_with(root))
            .count()
    }
}

/**
[`EventJournal`] 接受的事件负载。

一个 blanket impl 覆盖了所有支持 `serde` 且线程安全的类型。事件无需实现
[`Clone`]：追加会消费该值，记录则通过 [`Arc`] 共享它，用于回执和内存重放。
*/
pub trait JournalEvent:
    Serialize + DeserializeOwned + Send + Sync + std::fmt::Debug + 'static
{
}

/**
单个日志代（generation）的稳定标识。

序号只在该标识内有意义。基于文件的日志会将其持久化到每个物理批次帧中，
从而替换或混用日志代时，不会让无关的检查点看起来是最新的。
*/
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
#[serde(transparent)]
pub struct JournalIdentity(String);

impl JournalIdentity {
    /// 分配一个新的不透明日志代标识。
    pub fn new() -> Self {
        Self(uuid::Uuid::new_v4().to_string())
    }

    /// 校验并规范化一个已持久化的日志代标识。
    pub fn parse(value: impl AsRef<str>) -> Result<Self> {
        let parsed = uuid::Uuid::parse_str(value.as_ref()).map_err(|error| {
            ReactError::Other(format!("journal identity is not a UUID: {error}"))
        })?;
        Ok(Self(parsed.to_string()))
    }

    /// 该标识的规范文本表示。
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for JournalIdentity {
    /// 返回一个新生成的随机日志身份。
    fn default() -> Self {
        Self::new()
    }
}

impl<'de> Deserialize<'de> for JournalIdentity {
    /// 反序列化日志身份并校验其格式。
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::parse(&value).map_err(serde::de::Error::custom)
    }
}

impl fmt::Display for JournalIdentity {
    /// 以字符串形式显示日志身份。
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

impl<T> JournalEvent for T where
    T: Serialize + DeserializeOwned + Send + Sync + std::fmt::Debug + 'static
{
}

/// 一条带有已分配序号的日志条目。
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JournalRecord<E> {
    /// 包含该记录的物理批次帧标识。
    batch_id: String,
    /// 由日志分配的从 1 开始的连续序号。
    pub sequence: u64,
    /// 已持久化的事件负载。
    pub event: Arc<E>,
}

impl<E> Clone for JournalRecord<E> {
    /// 克隆日志记录（不要求 `E: Clone`）。
    fn clone(&self) -> Self {
        Self {
            batch_id: self.batch_id.clone(),
            sequence: self.sequence,
            event: Arc::clone(&self.event),
        }
    }
}

impl<E> JournalRecord<E> {
    /** 构造一条记录，同时保留由另一个类型化日志适配器分配的物理批次标识与序号。 */
    pub fn from_parts(batch_id: impl Into<String>, sequence: u64, event: E) -> Self {
        Self {
            batch_id: batch_id.into(),
            sequence,
            event: Arc::new(event),
        }
    }

    /// 包含该记录的物理批次帧的稳定标识。
    pub fn batch_id(&self) -> &str {
        &self.batch_id
    }
}

/// 参与批次摘要计算的完整性视图（不含摘要本身）。
#[derive(Debug, Serialize)]
struct BatchIntegrity<'a, E> {
    /// 批次帧 schema 版本。
    schema_version: u16,
    /// 所属日志代标识。
    journal_identity: &'a JournalIdentity,
    /// 批次标识。
    batch_id: &'a str,
    /// 批次中第一条记录的序号。
    first_sequence: u64,
    /// 批次内的记录。
    records: &'a [JournalRecord<E>],
}

/// 用于序列化写出的批次帧借用视图（含摘要）。
#[derive(Debug, Serialize)]
struct StoredBatchFrameRef<'a, E> {
    /// 批次帧 schema 版本。
    schema_version: u16,
    /// 所属日志代标识。
    journal_identity: &'a JournalIdentity,
    /// 批次标识。
    batch_id: &'a str,
    /// 批次中第一条记录的序号。
    first_sequence: u64,
    /// 批次内的记录。
    records: &'a [JournalRecord<E>],
    /// 批次内容的完整性摘要。
    digest: &'a str,
}

/// 从持久化行反序列化得到的批次帧。
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct StoredBatchFrame<E> {
    /// 批次帧 schema 版本。
    schema_version: u16,
    /// 所属日志代标识。
    journal_identity: JournalIdentity,
    /// 批次标识。
    batch_id: String,
    /// 批次中第一条记录的序号。
    first_sequence: u64,
    /// 批次内的记录。
    records: Vec<JournalRecord<E>>,
    /// 持久化的完整性摘要。
    digest: String,
    /// 对原始负载计算得到的摘要（不参与序列化）。
    #[serde(skip)]
    payload_digest: String,
}

/// 已提交批次的身份信息，用于幂等性比对与冲突检测。
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct BatchIdentity {
    /// 批次中第一条记录的序号。
    pub first_sequence: u64,
    /// 批次包含的记录数。
    pub record_count: u64,
    /// 预处理负载的摘要。
    pub payload_digest: String,
    /// 物理帧的完整性摘要。
    pub frame_digest: String,
}

/**
一个稳定、已预校验、持有不可克隆事件负载的批次。

只需准备一次，然后将该值传给 [`EventJournal::append_batch`]。
[`JournalBatchAppendError::NotCommitted`] 会原样返回该值（包括其原始 `batch_id`
和负载 `Arc`），因此重试时绝不会生成新的标识，也不要求 `E: Clone`。
*/
pub struct PreparedJournalBatch<E> {
    /// 跨重试与对账保持不变的批次幂等标识。
    batch_id: String,
    /// 有序的事件负载。
    events: Arc<[Arc<E>]>,
    /// 预校验时计算得到的负载摘要。
    payload_digest: String,
}

impl<E> Clone for PreparedJournalBatch<E> {
    /// 克隆已准备的批次（不要求 `E: Clone`）。
    fn clone(&self) -> Self {
        Self {
            batch_id: self.batch_id.clone(),
            events: Arc::clone(&self.events),
            payload_digest: self.payload_digest.clone(),
        }
    }
}

impl<E: fmt::Debug> fmt::Debug for PreparedJournalBatch<E> {
    /// 输出批次的调试信息。
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PreparedJournalBatch")
            .field("batch_id", &self.batch_id)
            .field("event_count", &self.events.len())
            .field("payload_digest", &self.payload_digest)
            .finish()
    }
}

/// 已编码、可直接写入的日志批次帧。
pub(super) struct PreparedJournalFrame<E> {
    /// 该批次提交后的下一个可用序号。
    pub next_sequence: u64,
    /// 已分配序号的记录。
    pub records: Arc<[JournalRecord<E>]>,
    /// 以换行结尾的序列化帧字节。
    pub line: Vec<u8>,
    /// 该批次的身份信息。
    pub identity: BatchIdentity,
}

/// 将 0..=15 的半字节转换为小写十六进制字符。
fn lower_hex_digit(nibble: u8) -> char {
    match nibble {
        0..=9 => char::from(b'0'.saturating_add(nibble)),
        _ => char::from(b'a'.saturating_add(nibble.saturating_sub(10))),
    }
}

/// 计算批次帧（含日志代标识、批次标识、起始序号与记录）的 SHA-256 十六进制摘要。
fn batch_digest<E: Serialize>(
    journal_identity: &JournalIdentity,
    batch_id: &str,
    first_sequence: u64,
    records: &[JournalRecord<E>],
) -> Result<String> {
    let bytes = canonical_json_bytes(&BatchIntegrity {
        schema_version: JOURNAL_BATCH_SCHEMA_VERSION,
        journal_identity,
        batch_id,
        first_sequence,
        records,
    })
    .map_err(|error| {
        ReactError::Other(format!(
            "failed to encode journal batch integrity input: {error}"
        ))
    })?;
    let digest = Sha256::digest(bytes);
    let mut encoded = String::with_capacity(digest.len().saturating_mul(2));
    for byte in digest {
        encoded.push(lower_hex_digit(byte >> 4));
        encoded.push(lower_hex_digit(byte & 0x0f));
    }
    Ok(encoded)
}

/// 在尝试任何日志变更之前，未能构造出稳定批次的错误。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JournalBatchPrepareError {
    /// 预检失败批次所分配的标识。
    batch_id: String,
    /// 可读的失败描述。
    error: String,
}

impl JournalBatchPrepareError {
    /// 预检失败批次所分配的稳定标识。
    pub fn batch_id(&self) -> &str {
        &self.batch_id
    }

    /// 可读的预检失败描述。
    pub fn error(&self) -> &str {
        &self.error
    }
}

impl fmt::Display for JournalBatchPrepareError {
    /// 显示批次准备错误的描述。
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "journal batch {} failed preflight: {}",
            self.batch_id, self.error
        )
    }
}

impl std::error::Error for JournalBatchPrepareError {}

/// 参与负载摘要计算的视图。
#[derive(Serialize)]
struct PreparedPayloads<'a, E> {
    /// 批次标识。
    batch_id: &'a str,
    /// 有序事件负载。
    events: &'a [Arc<E>],
}

/// 计算批次标识与事件负载的 SHA-256 十六进制摘要。
fn payload_digest<E: Serialize>(batch_id: &str, events: &[Arc<E>]) -> Result<String> {
    let bytes = canonical_json_bytes(&PreparedPayloads { batch_id, events }).map_err(|error| {
        ReactError::Other(format!("failed to encode journal batch payloads: {error}"))
    })?;
    let digest = Sha256::digest(bytes);
    let mut encoded = String::with_capacity(digest.len().saturating_mul(2));
    for byte in digest {
        encoded.push(lower_hex_digit(byte >> 4));
        encoded.push(lower_hex_digit(byte & 0x0f));
    }
    Ok(encoded)
}

impl<E: JournalEvent> PreparedJournalBatch<E> {
    /// 持有一个非空的有序事件集合，并预校验其序列化。
    pub fn new(events: Vec<E>) -> std::result::Result<Self, JournalBatchPrepareError> {
        Self::with_identity(uuid::Uuid::new_v4().to_string(), events)
    }

    /**
    以已有的批次标识持有一个事件集合。

    适配器在类型边界转换事件负载时，用它来保留日志帧标识。负载摘要会针对转换后
    的事件类型重新计算，因此查找仍保持确定性。
    */
    pub fn with_identity(
        batch_id: impl Into<String>,
        events: Vec<E>,
    ) -> std::result::Result<Self, JournalBatchPrepareError> {
        let batch_id = batch_id.into();
        if batch_id.trim().is_empty() {
            return Err(JournalBatchPrepareError {
                batch_id,
                error: "journal batch identity must not be empty".to_string(),
            });
        }
        if events.is_empty() {
            return Err(JournalBatchPrepareError {
                batch_id,
                error: "journal batch must contain at least one event".to_string(),
            });
        }
        u64::try_from(events.len()).map_err(|_| JournalBatchPrepareError {
            batch_id: batch_id.clone(),
            error: "journal batch event count exceeds supported range".to_string(),
        })?;
        let events: Arc<[Arc<E>]> = events.into_iter().map(Arc::new).collect::<Vec<_>>().into();
        let payload_digest = payload_digest(&batch_id, events.as_ref()).map_err(|error| {
            JournalBatchPrepareError {
                batch_id: batch_id.clone(),
                error: error.to_string(),
            }
        })?;
        Ok(Self {
            batch_id,
            events,
            payload_digest,
        })
    }

    /// 跨重试与对账保持不变的稳定幂等标识。
    pub fn batch_id(&self) -> &str {
        &self.batch_id
    }

    /// 该不可分割批次中有序负载的数量。
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// 该批次是否为空。成功准备的批次总是返回 false。
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// 预校验负载视图与批次标识的摘要。
    pub fn payload_digest(&self) -> &str {
        &self.payload_digest
    }

    /**
    只读的有序负载视图。

    内部可变性不得改变序列化结果；追加和查找会重新计算摘要，并在幂等判断之前
    拒绝已被修改的负载。
    */
    pub fn events(&self) -> &[Arc<E>] {
        self.events.as_ref()
    }

    /**
    校验已提交回执是否恰好包含本批次预处理的负载。

    物理提交权限对象可以独立地重试或重新打开，但归约器绝不能折叠属于其他批次
    标识的回执。
    */
    pub fn matches_receipt(&self, receipt: &JournalBatchAppendReceipt<E>) -> Result<bool> {
        if self.batch_id != receipt.batch_id() || self.len() != receipt.records().len() {
            return Ok(false);
        }
        let events = receipt
            .records()
            .iter()
            .map(|record| Arc::clone(&record.event))
            .collect::<Vec<_>>();
        Ok(payload_digest(&self.batch_id, events.as_slice())? == self.payload_digest)
    }

    /// 重新计算负载摘要，确认预检之后负载未被修改。
    fn validate_payload_integrity(&self) -> Result<()> {
        let current = payload_digest(&self.batch_id, self.events.as_ref())?;
        if current != self.payload_digest {
            return Err(ReactError::Other(format!(
                "prepared journal batch {} payload changed after preflight",
                self.batch_id
            )));
        }
        Ok(())
    }

    /// 以指定批次标识构造批次（仅测试使用）。
    #[cfg(test)]
    fn with_test_identity(
        batch_id: String,
        events: Vec<E>,
    ) -> std::result::Result<Self, JournalBatchPrepareError> {
        Self::with_identity(batch_id, events)
    }
}

/// 为批次分配从 `first_sequence` 开始的序号，并编码为可写入的物理帧。
pub(super) fn prepare_journal_frame<E: JournalEvent>(
    prepared: &PreparedJournalBatch<E>,
    journal_identity: &JournalIdentity,
    first_sequence: u64,
) -> Result<PreparedJournalFrame<E>> {
    prepared.validate_payload_integrity()?;
    let count = u64::try_from(prepared.events.len()).map_err(|_| {
        ReactError::Other("journal batch event count exceeds supported range".to_string())
    })?;
    let next_sequence = first_sequence.checked_add(count).ok_or_else(|| {
        ReactError::Other("journal sequence exhausted before batch append".to_string())
    })?;
    let mut records = Vec::with_capacity(prepared.events.len());
    for (index, event) in prepared.events.iter().enumerate() {
        let index = u64::try_from(index).map_err(|_| {
            ReactError::Other("journal batch index exceeds supported range".to_string())
        })?;
        let sequence = first_sequence.checked_add(index).ok_or_else(|| {
            ReactError::Other("journal sequence exhausted while preparing batch".to_string())
        })?;
        records.push(JournalRecord {
            batch_id: prepared.batch_id.clone(),
            sequence,
            event: Arc::clone(event),
        });
    }
    let records: Arc<[JournalRecord<E>]> = records.into();
    let digest = batch_digest(
        journal_identity,
        &prepared.batch_id,
        first_sequence,
        records.as_ref(),
    )?;
    let stored = StoredBatchFrameRef {
        schema_version: JOURNAL_BATCH_SCHEMA_VERSION,
        journal_identity,
        batch_id: &prepared.batch_id,
        first_sequence,
        records: records.as_ref(),
        digest: &digest,
    };
    let mut line = serde_json::to_vec(&stored)
        .map_err(|error| ReactError::Other(format!("failed to encode journal batch: {error}")))?;
    line.push(b'\n');
    u64::try_from(line.len())
        .map_err(|_| ReactError::Other("journal batch frame exceeds supported size".to_string()))?;
    Ok(PreparedJournalFrame {
        next_sequence,
        records,
        line,
        identity: BatchIdentity {
            first_sequence,
            record_count: count,
            payload_digest: prepared.payload_digest.clone(),
            frame_digest: digest,
        },
    })
}

/// 解码一行持久化批次帧，并校验 schema、标识、序号连续性与摘要。
pub(super) fn decode_journal_batch<E: JournalEvent>(
    context: &str,
    line: &[u8],
) -> Result<StoredBatchFrame<E>> {
    let frame: StoredBatchFrame<E> = serde_json::from_slice(line)
        .map_err(|error| ReactError::Other(format!("{context}: corrupt journal batch: {error}")))?;
    if frame.schema_version != JOURNAL_BATCH_SCHEMA_VERSION {
        return Err(ReactError::Other(format!(
            "{context}: unsupported journal batch schema {}",
            frame.schema_version
        )));
    }
    if uuid::Uuid::parse_str(&frame.batch_id).is_err() {
        return Err(ReactError::Other(format!(
            "{context}: journal batch id is not a UUID"
        )));
    }
    if frame.records.is_empty() {
        return Err(ReactError::Other(format!(
            "{context}: journal batch contains no records"
        )));
    }
    for (index, record) in frame.records.iter().enumerate() {
        if record.batch_id != frame.batch_id {
            return Err(ReactError::Other(format!(
                "{context}: record batch identity does not match frame {}",
                frame.batch_id
            )));
        }
        let index = u64::try_from(index).map_err(|_| {
            ReactError::Other(format!(
                "{context}: journal batch index exceeds supported range"
            ))
        })?;
        let expected = frame.first_sequence.checked_add(index).ok_or_else(|| {
            ReactError::Other(format!("{context}: journal sequence exhausted in batch"))
        })?;
        if record.sequence != expected {
            return Err(ReactError::Other(format!(
                "{context}: journal batch sequence gap, expected {expected} but found {}",
                record.sequence
            )));
        }
    }
    let expected_digest = batch_digest(
        &frame.journal_identity,
        &frame.batch_id,
        frame.first_sequence,
        &frame.records,
    )?;
    if frame.digest != expected_digest {
        return Err(ReactError::Other(format!(
            "{context}: journal batch digest mismatch at sequence {}",
            frame.first_sequence
        )));
    }
    let events = frame
        .records
        .iter()
        .map(|record| Arc::clone(&record.event))
        .collect::<Vec<_>>();
    let computed_payload_digest = payload_digest(&frame.batch_id, &events)?;
    Ok(StoredBatchFrame {
        payload_digest: computed_payload_digest,
        ..frame
    })
}

impl<E> StoredBatchFrame<E> {
    /// 帧所属的日志代标识。
    pub(super) fn journal_identity(&self) -> &JournalIdentity {
        &self.journal_identity
    }

    /// 帧的批次标识。
    pub(super) fn batch_id(&self) -> &str {
        &self.batch_id
    }

    /// 从帧推导出批次身份信息。
    pub(super) fn identity(&self) -> Result<BatchIdentity> {
        let record_count = u64::try_from(self.records.len()).map_err(|_| {
            ReactError::Other("journal batch count exceeds supported range".to_string())
        })?;
        Ok(BatchIdentity {
            first_sequence: self.first_sequence,
            record_count,
            payload_digest: self.payload_digest.clone(),
            frame_digest: self.digest.clone(),
        })
    }
    /// 帧内记录的借用视图。
    pub(super) fn records(&self) -> &[JournalRecord<E>] {
        &self.records
    }

    /// 消费帧并返回其记录。
    pub(super) fn into_records(self) -> Vec<JournalRecord<E>> {
        self.records
    }
}

/// 校验帧的起始序号是否等于期望的下一个序号。
pub(super) fn verify_journal_batch_sequence<E>(
    context: &str,
    expected_sequence: u64,
    frame: &StoredBatchFrame<E>,
) -> Result<()> {
    if frame.first_sequence != expected_sequence {
        return Err(ReactError::Other(format!(
            "{context}: journal sequence gap, expected {expected_sequence} but batch starts at {}",
            frame.first_sequence
        )));
    }
    Ok(())
}

/// 带有明确重试契约的批次追加失败。
#[derive(Debug)]
pub enum JournalBatchAppendError<E> {
    /// 批次中没有任何记录被提交；重试整个批次是安全的。
    NotCommitted {
        /// 原样返回的批次。
        batch: PreparedJournalBatch<E>,
        /// 失败描述。
        error: String,
    },
    /** 写入方无法证明批次是否已提交。权限对象已被毒化，调用方必须重新打开并检查恢复结果后才能重试。 */
    OutcomeUnknown {
        /// 原样返回的批次。
        batch: PreparedJournalBatch<E>,
        /// 失败描述。
        error: String,
    },
    /** 该标识已绑定到不同的负载或形状。复用它永远无法安全重试，且存活的权限对象已被毒化。 */
    IdentityConflict {
        /// 原样返回的批次。
        batch: PreparedJournalBatch<E>,
        /// 已存在批次的起始序号。
        existing_first_sequence: u64,
        /// 已存在批次的记录数。
        existing_record_count: u64,
        /// 失败描述。
        error: String,
    },
    /** 内部可变性在预检之后修改了预处理负载。该稳定标识仍为原始摘要保留，此值不得重试。 */
    PreparedMutation {
        /// 原样返回的批次。
        batch: PreparedJournalBatch<E>,
        /// 失败描述。
        error: String,
    },
    /// 存活的权限对象在关闭并重新打开之前无法接受写入。
    AuthorityPoisoned {
        /// 原样返回的批次。
        batch: PreparedJournalBatch<E>,
        /// 失败描述。
        error: String,
    },
}

impl<E> JournalBatchAppendError<E> {
    /// 构造一个可重试、发生在变更之前、并归还所有权的失败。
    pub fn not_committed(batch: PreparedJournalBatch<E>, error: impl Into<String>) -> Self {
        Self::NotCommitted {
            batch,
            error: error.into(),
        }
    }

    /// 构造一个结果未知、保留所有权且需要重新打开的失败。
    pub fn outcome_unknown(batch: PreparedJournalBatch<E>, error: impl Into<String>) -> Self {
        Self::OutcomeUnknown {
            batch,
            error: error.into(),
        }
    }

    /// 构造一个与已提交标识发生永久冲突的失败。
    pub fn identity_conflict(
        batch: PreparedJournalBatch<E>,
        existing_first_sequence: u64,
        existing_record_count: u64,
        error: impl Into<String>,
    ) -> Self {
        Self::IdentityConflict {
            batch,
            existing_first_sequence,
            existing_record_count,
            error: error.into(),
        }
    }

    /// 构造一个不可重试的负载变更失败。
    pub fn prepared_mutation(batch: PreparedJournalBatch<E>, error: impl Into<String>) -> Self {
        Self::PreparedMutation {
            batch,
            error: error.into(),
        }
    }

    /// 构造一个来自必须重新打开的存活权限对象的拒绝。
    pub fn authority_poisoned(batch: PreparedJournalBatch<E>, error: impl Into<String>) -> Self {
        Self::AuthorityPoisoned {
            batch,
            error: error.into(),
        }
    }

    /// 此次失败涉及的稳定批次标识。
    pub fn batch_id(&self) -> &str {
        match self {
            Self::NotCommitted { batch, .. } => &batch.batch_id,
            Self::OutcomeUnknown { batch, .. }
            | Self::IdentityConflict { batch, .. }
            | Self::PreparedMutation { batch, .. }
            | Self::AuthorityPoisoned { batch, .. } => &batch.batch_id,
        }
    }

    /// 返回的预处理批次是否可以立即重试。
    pub fn is_retry_safe(&self) -> bool {
        matches!(self, Self::NotCommitted { .. })
    }

    /// 借用保留的预处理批次，用于诊断或查找。
    pub fn prepared(&self) -> Option<&PreparedJournalBatch<E>> {
        match self {
            Self::NotCommitted { batch, .. }
            | Self::OutcomeUnknown { batch, .. }
            | Self::IdentityConflict { batch, .. }
            | Self::PreparedMutation { batch, .. }
            | Self::AuthorityPoisoned { batch, .. } => Some(batch),
        }
    }

    /** 取回所有权。不可重试的变体仍需执行 [`Self::requires_reopen`] 所指示的操作，或使用新标识。 */
    pub fn into_prepared(self) -> Option<PreparedJournalBatch<E>> {
        match self {
            Self::NotCommitted { batch, .. }
            | Self::OutcomeUnknown { batch, .. }
            | Self::IdentityConflict { batch, .. }
            | Self::PreparedMutation { batch, .. }
            | Self::AuthorityPoisoned { batch, .. } => Some(batch),
        }
    }

    /** 该结果在任何重试之前，是否必须使用新的权限对象并通过 [`EventJournal::lookup_batch`] 解析。 */
    pub fn requires_reopen(&self) -> bool {
        matches!(
            self,
            Self::OutcomeUnknown { .. }
                | Self::IdentityConflict { .. }
                | Self::AuthorityPoisoned { .. }
        )
    }
}

impl<E> fmt::Display for JournalBatchAppendError<E> {
    /// 显示批次追加错误的描述。
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotCommitted { batch, error } => {
                write!(
                    formatter,
                    "journal batch {} was not committed: {error}",
                    batch.batch_id
                )
            }
            Self::OutcomeUnknown { batch, error } => write!(
                formatter,
                "journal batch {} has unknown commit outcome: {error}; reopen before retry",
                batch.batch_id
            ),
            Self::IdentityConflict {
                batch,
                existing_first_sequence,
                existing_record_count,
                error,
            } => write!(
                formatter,
                "journal batch {} conflicts with the committed batch at sequence {} ({} records): {error}",
                batch.batch_id, existing_first_sequence, existing_record_count
            ),
            Self::PreparedMutation { batch, error } => write!(
                formatter,
                "prepared journal batch {} changed after preflight: {error}",
                batch.batch_id
            ),
            Self::AuthorityPoisoned { batch, error } => write!(
                formatter,
                "journal authority is poisoned for batch {}: {error}; reopen required",
                batch.batch_id
            ),
        }
    }
}

impl<E: fmt::Debug> std::error::Error for JournalBatchAppendError<E> {}

/// 批次追加的结果类型。
pub type JournalBatchAppendResult<E> =
    std::result::Result<JournalBatchAppendReceipt<E>, JournalBatchAppendError<E>>;

/// 日志中一个完整批次帧的持久性结果。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum JournalDurabilityStatus {
    /// 只读查找证明了帧标识，但有意没有执行新的持久性屏障。
    Unconfirmed,
    /// 追加与请求的持久性操作均已完成。
    Confirmed,
    /**
    完整批次帧已存在并占有所有已分配序号，但请求的持久性屏障报告了错误。
    调用方不得重试该批次。
    */
    Degraded {
        /// 持久性屏障报告的错误描述。
        error: String,
    },
}

/// 一次追加是新建了帧，还是解析到了已存在的标识。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JournalBatchCommitStatus {
    /// 本次调用提交了物理帧。
    Committed,
    /// 该标识已被提交过；未写入第二个帧。
    AlreadyCommitted,
}

/// 单条日志追加提交后的回执。
#[derive(Debug)]
pub struct JournalAppendReceipt<E> {
    /// 提交该记录的日志代标识。
    journal_identity: JournalIdentity,
    /// 该单条记录回执对应的批次标识。
    batch_id: String,
    /// 已提交的记录。
    pub record: JournalRecord<E>,
    /// 持久性结果。
    pub durability: JournalDurabilityStatus,
    /// 提交状态（新提交或已提交）。
    pub commit: JournalBatchCommitStatus,
}

impl<E> Clone for JournalAppendReceipt<E> {
    /// 克隆追加回执（不要求 `E: Clone`）。
    fn clone(&self) -> Self {
        Self {
            journal_identity: self.journal_identity.clone(),
            batch_id: self.batch_id.clone(),
            record: self.record.clone(),
            durability: self.durability.clone(),
            commit: self.commit,
        }
    }
}

impl<E> JournalAppendReceipt<E> {
    /// 提交该记录的日志代。
    pub fn journal_identity(&self) -> &JournalIdentity {
        &self.journal_identity
    }

    /// 该单条记录回执的稳定批次标识。
    pub fn batch_id(&self) -> &str {
        &self.batch_id
    }
}

/// 一个原子提交的日志批次的回执。
#[derive(Debug)]
pub struct JournalBatchAppendReceipt<E> {
    /// 提交该帧的日志代。
    journal_identity: JournalIdentity,
    /// 物理提交帧的稳定标识。
    batch_id: String,
    /** 该帧提交的有序记录。负载通过 [`Arc`] 共享，绝不要求 `E: Clone`。 */
    records: Arc<[JournalRecord<E>]>,
    /// 覆盖整个帧的单一持久性结果。
    durability: JournalDurabilityStatus,
    /// 提交状态（新提交或已提交）。
    commit: JournalBatchCommitStatus,
}

impl<E> Clone for JournalBatchAppendReceipt<E> {
    /// 克隆批次追加回执（不要求 `E: Clone`）。
    fn clone(&self) -> Self {
        Self {
            journal_identity: self.journal_identity.clone(),
            batch_id: self.batch_id.clone(),
            records: Arc::clone(&self.records),
            durability: self.durability.clone(),
            commit: self.commit,
        }
    }
}

impl<E> JournalBatchAppendReceipt<E> {
    /// 在跨类型化适配器边界映射记录后重建回执。
    pub fn from_parts(
        journal_identity: JournalIdentity,
        batch_id: impl Into<String>,
        records: Vec<JournalRecord<E>>,
        durability: JournalDurabilityStatus,
        commit: JournalBatchCommitStatus,
    ) -> std::result::Result<Self, String> {
        if records.is_empty() {
            return Err("journal receipt must contain at least one record".to_string());
        }
        Ok(Self {
            journal_identity,
            batch_id: batch_id.into(),
            records: records.into(),
            durability,
            commit,
        })
    }

    /// 已提交或经幂等解析得到的稳定批次标识。
    pub fn batch_id(&self) -> &str {
        &self.batch_id
    }

    /// 提交或解析出该帧的日志代。
    pub fn journal_identity(&self) -> &JournalIdentity {
        &self.journal_identity
    }

    /// 以只读共享切片形式返回有序的已提交记录。
    pub fn records(&self) -> &[JournalRecord<E>] {
        self.records.as_ref()
    }

    /// 由追加或只读查找得出的持久性信息。
    pub fn durability(&self) -> &JournalDurabilityStatus {
        &self.durability
    }

    /// 本次调用是新建了帧，还是幂等地解析到了已有帧。
    pub fn commit_status(&self) -> JournalBatchCommitStatus {
        self.commit
    }
}

/** 只读的标识对账结果。基于文件的查找从不写入，也从不执行持久性屏障。 */
#[derive(Debug)]
pub enum JournalBatchLookup<E> {
    /// 没有保留的帧占有该标识与摘要。
    Absent,
    /// 唯一保留的帧匹配，并提供其原始序号。
    AlreadyCommitted(JournalBatchAppendReceipt<E>),
    /// 标识、形状、摘要或当前预处理序列化存在冲突。
    Conflict {
        /// 冲突描述。
        error: String,
    },
}

/// 类型化的单事件失败，在变更开始后仍保留预处理批次的所有权。
#[derive(Debug)]
pub enum JournalAppendError<E> {
    /// 在到达日志权限对象之前，准备阶段就已失败。
    Prepare(JournalBatchPrepareError),
    /// 单元素预处理批次已到达日志权限对象。
    Commit(JournalBatchAppendError<E>),
}

impl<E> JournalAppendError<E> {
    /// 为该单个事件分配的稳定标识。
    pub fn batch_id(&self) -> &str {
        match self {
            Self::Prepare(error) => &error.batch_id,
            Self::Commit(error) => error.batch_id(),
        }
    }

    /// 保留的预处理值是否可以立即重试。
    pub fn is_retry_safe(&self) -> bool {
        matches!(self, Self::Commit(error) if error.is_retry_safe())
    }

    /// 在提交阶段出错后取回单元素预处理批次。
    pub fn into_prepared(self) -> Option<PreparedJournalBatch<E>> {
        match self {
            Self::Prepare(_) => None,
            Self::Commit(error) => error.into_prepared(),
        }
    }

    /// 在对账之前是否必须重新打开存活的权限对象。
    pub fn requires_reopen(&self) -> bool {
        matches!(self, Self::Commit(error) if error.requires_reopen())
    }
}

impl<E> fmt::Display for JournalAppendError<E> {
    /// 显示追加错误的描述。
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Prepare(error) => error.fmt(formatter),
            Self::Commit(error) => error.fmt(formatter),
        }
    }
}

impl<E: fmt::Debug> std::error::Error for JournalAppendError<E> {}

/// 单事件追加的结果类型。
pub type JournalAppendResult<E> =
    std::result::Result<JournalAppendReceipt<E>, JournalAppendError<E>>;

/// 类型化的检查点归约器失败。
#[derive(Debug)]
pub enum CheckpointedApplyError<E> {
    /// 在日志变更之前准备阶段就已失败。
    Prepare(JournalBatchPrepareError),
    /// 日志返回了一个保留所有权的类型化结果。
    Journal(JournalBatchAppendError<E>),
    /// 已提交的回执违反了投影序号不变量。
    CommittedInvariant {
        /// 涉及的批次标识。
        batch_id: String,
        /// 违反不变量的描述。
        error: String,
    },
}

impl<E> CheckpointedApplyError<E> {
    /// 此次归约器应用涉及的稳定标识。
    pub fn batch_id(&self) -> &str {
        match self {
            Self::Prepare(error) => &error.batch_id,
            Self::Journal(error) => error.batch_id(),
            Self::CommittedInvariant { batch_id, .. } => batch_id,
        }
    }

    /// 从日志阶段的失败中取回预处理批次。
    pub fn into_prepared(self) -> Option<PreparedJournalBatch<E>> {
        match self {
            Self::Journal(error) => error.into_prepared(),
            Self::Prepare(_) | Self::CommittedInvariant { .. } => None,
        }
    }

    /// 在对账之前是否必须重新打开日志权限对象。
    pub fn requires_reopen(&self) -> bool {
        matches!(self, Self::Journal(error) if error.requires_reopen())
    }
}

impl<E> fmt::Display for CheckpointedApplyError<E> {
    /// 显示检查点化应用错误的描述。
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Prepare(error) => error.fmt(formatter),
            Self::Journal(error) => error.fmt(formatter),
            Self::CommittedInvariant { batch_id, error } => write!(
                formatter,
                "committed journal batch {batch_id} violated reducer receipt invariant: {error}"
            ),
        }
    }
}

impl<E: fmt::Debug> std::error::Error for CheckpointedApplyError<E> {}

/**
支持原子批次提交的仅追加有序日志。

实现在追加时分配从 1 开始的连续序号，并能按顺序重放日志的任意后缀。持久化实现
必须在整个日志代内保持同一个 [`JournalIdentity`]，并在底层事实历史被替换时替换它。
*/
pub trait EventJournal<E: JournalEvent>: Send + Sync {
    /// 该日志代的稳定标识。
    fn journal_identity(&self) -> &JournalIdentity;

    /**
    将一个非空的有序事件批次作为一个物理帧提交。

    [`JournalBatchAppendError::NotCommitted`] 允许重试整个批次。
    [`JournalBatchAppendError::OutcomeUnknown`] 会毒化基于文件的权限对象，
    在经过验证的重新打开之前禁止盲目重试。
    */
    fn append_batch(&self, batch: PreparedJournalBatch<E>) -> JournalBatchAppendResult<E>;

    /**
    在提交结果未知后，以不写入的方式对账预处理标识。`Absent` 允许重试保留的
    预处理值；`AlreadyCommitted` 返回原始序号范围。
    */
    fn lookup_batch(&self, batch: &PreparedJournalBatch<E>) -> Result<JournalBatchLookup<E>>;

    /**
    消费一个事件，并同时报告其共享持有的记录与持久性结果。
    [`JournalDurabilityStatus::Degraded`] 回执表示完整记录已占有其序号，不得重试。
    */
    fn append(&self, event: E) -> JournalAppendResult<E> {
        let batch = PreparedJournalBatch::new(vec![event]).map_err(JournalAppendError::Prepare)?;
        let appended = self
            .append_batch(batch)
            .map_err(JournalAppendError::Commit)?;
        let record = appended.records.first().cloned().ok_or_else(|| {
            JournalAppendError::Prepare(JournalBatchPrepareError {
                batch_id: appended.batch_id.clone(),
                error: format!(
                    "journal batch {} committed without a record",
                    appended.batch_id
                ),
            })
        })?;
        Ok(JournalAppendReceipt {
            journal_identity: appended.journal_identity,
            batch_id: appended.batch_id,
            record,
            durability: appended.durability,
            commit: appended.commit,
        })
    }

    /// 下一次追加将分配的序号。
    fn next_sequence(&self) -> u64;

    /**
    日志中最后一个已提交的序号（为空时为 `0`）。需查阅追加回执，以区分已确认的
    持久性与完整写入但持久性降级的提交。
    */
    fn last_sequence(&self) -> u64;

    /**
    逻辑上仍可重放的最早序号。

    不裁剪的日志保留完整历史，默认返回 `1`。
    */
    fn retained_floor(&self) -> u64 {
        1
    }

    /// 按顺序重放最多 `limit` 条满足 `sequence > after_sequence` 的记录。
    fn replay_after(&self, after_sequence: u64, limit: usize) -> Result<Vec<JournalRecord<E>>>;
}

/// 用于测试和临时消费方的内存版 [`EventJournal`]。
#[derive(Debug)]
pub struct MemoryEventJournal<E> {
    /// 该内存日志的日志代标识。
    identity: JournalIdentity,
    /// 受互斥锁保护的内部状态。
    inner: Mutex<MemoryInner<E>>,
}

/// 内存日志的内部可变状态。
#[derive(Debug, Default)]
struct MemoryInner<E> {
    /// 下一次追加将分配的序号。
    next_sequence: u64,
    /// 已提交的全部记录。
    records: Vec<JournalRecord<E>>,
    /// 按批次标识索引的已提交批次。
    batches: HashMap<String, MemoryCommittedBatch<E>>,
    /// 若已被毒化，记录原因。
    poison: Option<String>,
}

/// 内存日志中一个已提交批次的记录。
#[derive(Debug)]
struct MemoryCommittedBatch<E> {
    /// 提交时的负载摘要。
    payload_digest: String,
    /// 提交时生成的回执。
    receipt: JournalBatchAppendReceipt<E>,
}

impl<E: JournalEvent> Default for MemoryEventJournal<E> {
    /// 创建一个空的内存日志。
    fn default() -> Self {
        Self::new()
    }
}

impl<E: JournalEvent> MemoryEventJournal<E> {
    /// 创建一个新的空内存日志，序号从 1 开始。
    pub fn new() -> Self {
        Self {
            identity: JournalIdentity::new(),
            inner: Mutex::new(MemoryInner {
                next_sequence: 1,
                records: Vec::new(),
                batches: HashMap::new(),
                poison: None,
            }),
        }
    }
}

impl<E: JournalEvent> EventJournal<E> for MemoryEventJournal<E> {
    /// 返回该内存日志的身份。
    fn journal_identity(&self) -> &JournalIdentity {
        &self.identity
    }

    /// 以一次有序提交追加整个批次，并保证幂等与冲突检测。
    fn append_batch(&self, batch: PreparedJournalBatch<E>) -> JournalBatchAppendResult<E> {
        if let Err(error) = batch.validate_payload_integrity() {
            return Err(JournalBatchAppendError::prepared_mutation(
                batch,
                error.to_string(),
            ));
        }
        let mut inner = self.inner.lock().unwrap_or_else(|error| error.into_inner());
        if let Some(reason) = &inner.poison {
            return Err(JournalBatchAppendError::authority_poisoned(
                batch,
                format!("memory journal is poisoned: {reason}"),
            ));
        }
        if let Some(existing) = inner.batches.get(batch.batch_id()) {
            if existing.payload_digest == batch.payload_digest()
                && existing.receipt.records().len() == batch.len()
            {
                let mut receipt = existing.receipt.clone();
                receipt.commit = JournalBatchCommitStatus::AlreadyCommitted;
                return Ok(receipt);
            }
            let reason = format!(
                "batch identity {} conflicts with an existing committed payload",
                batch.batch_id()
            );
            let existing_first_sequence = existing
                .receipt
                .records()
                .first()
                .map(|record| record.sequence)
                .unwrap_or(0);
            let existing_record_count =
                u64::try_from(existing.receipt.records().len()).unwrap_or(u64::MAX);
            inner.poison = Some(reason.clone());
            return Err(JournalBatchAppendError::identity_conflict(
                batch,
                existing_first_sequence,
                existing_record_count,
                reason,
            ));
        }
        let prepared = match prepare_journal_frame(&batch, &self.identity, inner.next_sequence) {
            Ok(prepared) => prepared,
            Err(error) => {
                return Err(JournalBatchAppendError::not_committed(
                    batch,
                    error.to_string(),
                ));
            }
        };
        inner.next_sequence = prepared.next_sequence;
        inner.records.extend(prepared.records.iter().cloned());
        let receipt = JournalBatchAppendReceipt {
            journal_identity: self.identity.clone(),
            batch_id: batch.batch_id,
            records: prepared.records,
            durability: JournalDurabilityStatus::Confirmed,
            commit: JournalBatchCommitStatus::Committed,
        };
        inner.batches.insert(
            receipt.batch_id.clone(),
            MemoryCommittedBatch {
                payload_digest: batch.payload_digest,
                receipt: receipt.clone(),
            },
        );
        Ok(receipt)
    }

    /// 按批次身份查找已提交的批次。
    fn lookup_batch(&self, batch: &PreparedJournalBatch<E>) -> Result<JournalBatchLookup<E>> {
        if let Err(error) = batch.validate_payload_integrity() {
            return Ok(JournalBatchLookup::Conflict {
                error: error.to_string(),
            });
        }
        let mut inner = self.inner.lock().unwrap_or_else(|error| error.into_inner());
        let Some(existing) = inner.batches.get(batch.batch_id()) else {
            return Ok(JournalBatchLookup::Absent);
        };
        if existing.payload_digest == batch.payload_digest()
            && existing.receipt.records().len() == batch.len()
        {
            let mut receipt = existing.receipt.clone();
            receipt.commit = JournalBatchCommitStatus::AlreadyCommitted;
            return Ok(JournalBatchLookup::AlreadyCommitted(receipt));
        }
        let reason = format!(
            "batch identity {} conflicts with an existing committed payload",
            batch.batch_id()
        );
        inner.poison = Some(reason.clone());
        Ok(JournalBatchLookup::Conflict { error: reason })
    }

    /// 返回下一条记录将分配的序号。
    fn next_sequence(&self) -> u64 {
        self.inner
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .next_sequence
    }

    /// 返回最后一条已提交记录的序号。
    fn last_sequence(&self) -> u64 {
        self.inner
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .next_sequence
            .saturating_sub(1)
    }

    /// 重放 `after_sequence` 之后最多 `limit` 条记录。
    fn replay_after(&self, after_sequence: u64, limit: usize) -> Result<Vec<JournalRecord<E>>> {
        let inner = self.inner.lock().unwrap_or_else(|error| error.into_inner());
        let start = usize::try_from(after_sequence).map_err(|_| {
            ReactError::Other("journal sequence exceeds supported memory index".to_string())
        })?;
        let suffix = inner.records.get(start..).unwrap_or(&[]);
        Ok(suffix.iter().take(limit).cloned().collect())
    }
}

/**
由日志事件折叠得到的状态投影。

`apply` 必须对该归约器领域产生的所有事件保持全函数（total）；参见模块不变量。
*/
pub trait EventReducer: Default + Serialize + DeserializeOwned + Send + Sync + 'static {
    /// 该归约器处理的事件类型。
    type Event: JournalEvent;

    /// 将单个事件折叠进当前状态。
    fn apply(&mut self, event: &Self::Event);

    /**
    折叠一条权威日志记录（包括其已分配的序号）。

    只需要负载的归约器直接继承该默认实现。需要感知序号的投影应覆盖此钩子，
    而不是在追加之前将序号预分配到事件中。
    */
    fn apply_record(&mut self, record: &JournalRecord<Self::Event>) {
        self.apply(record.event.as_ref());
    }
}

/// 追加之后自动检查点复合的结果。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CheckpointApplyStatus {
    /// 尚未达到配置的检查点节奏。
    NotDue,
    /// 投影已在本次追加处保存了检查点。
    Saved,
    /**
    事件与投影均已提交，但派生检查点未能保存。调用方不得重试该事件；后续的
    追加或恢复会重试这个可丢弃的检查点。
    */
    Degraded {
        /// 检查点保存失败的描述。
        error: String,
    },
}

/// 单次日志化投影更新的提交回执。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApplyReceipt {
    /// 单记录物理批次帧的标识。
    pub batch_id: String,
    /// 提交到权威日志的序号。
    pub sequence: u64,
    /// 权威日志追加的结果。
    pub journal: JournalDurabilityStatus,
    /// 提交状态（新提交或已提交）。
    pub commit: JournalBatchCommitStatus,
    /// 可选检查点复合的结果。
    pub checkpoint: CheckpointApplyStatus,
}

/// 单个日志化投影批次的提交回执。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApplyBatchReceipt {
    /// 物理批次帧的标识。
    pub batch_id: String,
    /// 批次中第一条记录的序号。
    pub first_sequence: u64,
    /// 批次中最后一条记录的序号。
    pub last_sequence: u64,
    /// 批次包含的记录数。
    pub record_count: u64,
    /// 权威日志追加的结果。
    pub journal: JournalDurabilityStatus,
    /// 提交状态（新提交或已提交）。
    pub commit: JournalBatchCommitStatus,
    /// 可选检查点复合的结果。
    pub checkpoint: CheckpointApplyStatus,
}

/// 恢复过程如何获取并修复其派生检查点。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CheckpointRecoveryStatus {
    /// 加载了有效检查点，无需修复。
    Loaded,
    /// 不存在检查点，且重放尾部未达到检查点节奏。
    Missing,
    /// 缺失、损坏、过期或落后的检查点已根据权威日志重建。
    Rebuilt {
        /// 需要重建的原因。
        reason: String,
    },
    /// 日志恢复成功，但修复派生检查点失败。
    Degraded {
        /// 需要修复的原因。
        reason: String,
        /// 修复失败的描述。
        error: String,
    },
}

/// 检查点投影的恢复结果。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecoveryReceipt {
    /// 已折叠进恢复状态的最后一个日志序号。
    pub last_applied_sequence: u64,
    /// 检查点加载 / 修复结果。
    pub checkpoint: CheckpointRecoveryStatus,
}

/// 持久化的检查点，将归约器状态与序号及其所属日志代配对。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointFrame<S> {
    /// 派生该投影的确切日志代。
    pub journal_identity: JournalIdentity,
    /// 该状态快照所覆盖的日志序号。
    pub sequence: u64,
    /// 该序号处的归约器状态。
    pub state: S,
}

/// 存储由 [`CheckpointedReducer`] 写入、绑定日志代的 [`CheckpointFrame`]。
pub trait CheckpointStore<S>: Send + Sync {
    /// 保存覆盖至 `through_sequence` 的状态检查点。
    fn save(
        &self,
        journal_identity: &JournalIdentity,
        state: &S,
        through_sequence: u64,
    ) -> Result<()>;

    /// 加载最新的检查点；不存在时返回 `None`。
    fn load(&self) -> Result<Option<CheckpointFrame<S>>>;
}

/// 以序列化形式存储的检查点。
#[derive(Debug)]
struct StoredCheckpoint {
    /// 所属日志代标识。
    journal_identity: JournalIdentity,
    /// 覆盖到的序号。
    sequence: u64,
    /// 序列化后的状态。
    state: serde_json::Value,
}

/**
用于测试和临时消费方的内存版 [`CheckpointStore`]。

状态以序列化形式存储，因此投影无需仅为了保存检查点而实现 `Clone`。
*/
#[derive(Debug)]
pub struct MemoryCheckpointStore<S> {
    /// 最新的检查点。
    latest: Mutex<Option<StoredCheckpoint>>,
    /// 状态类型标记。
    _state: PhantomData<S>,
}

impl<S> Default for MemoryCheckpointStore<S> {
    /// 创建一个空的内存检查点存储。
    fn default() -> Self {
        Self {
            latest: Mutex::new(None),
            _state: PhantomData,
        }
    }
}

impl<S> MemoryCheckpointStore<S> {
    /// 创建一个空的内存检查点存储。
    pub fn new() -> Self {
        Self::default()
    }
}

impl<S: Serialize + DeserializeOwned + Send + Sync + 'static> CheckpointStore<S>
    for MemoryCheckpointStore<S>
{
    /// 保存检查点帧（覆盖之前的检查点）。
    fn save(
        &self,
        journal_identity: &JournalIdentity,
        state: &S,
        through_sequence: u64,
    ) -> Result<()> {
        let encoded = serde_json::to_value(state).map_err(|error| {
            ReactError::Other(format!("failed to encode checkpoint state: {error}"))
        })?;
        *self
            .latest
            .lock()
            .unwrap_or_else(|error| error.into_inner()) = Some(StoredCheckpoint {
            journal_identity: journal_identity.clone(),
            sequence: through_sequence,
            state: encoded,
        });
        Ok(())
    }

    /// 加载最近保存的检查点帧。
    fn load(&self) -> Result<Option<CheckpointFrame<S>>> {
        let latest = self
            .latest
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        match latest.as_ref() {
            None => Ok(None),
            Some(stored) => {
                let state = serde_json::from_value(stored.state.clone()).map_err(|error| {
                    ReactError::Other(format!("failed to decode checkpoint state: {error}"))
                })?;
                Ok(Some(CheckpointFrame {
                    journal_identity: stored.journal_identity.clone(),
                    sequence: stored.sequence,
                    state,
                }))
            }
        }
    }
}

/// 检查点归约器的内部可变状态。
struct ReducerInner<R> {
    /// 当前投影状态。
    state: R,
    /// 已折叠的最后一个序号。
    last_applied: u64,
    /// 自上次检查点以来折叠的记录数。
    since_checkpoint: u64,
}

/// 单次恢复重放操作最多物化的记录数。
const RECOVER_BATCH: usize = 512;

/**
将日志事件折叠进归约器状态，并复合检查点。

`apply` 先追加到日志（持久性先于投影），再按顺序折叠所有返回的记录。当一个批次
跨越 `checkpoint_every` 时，会在批次边界保存一次当前状态，使后续恢复只需重放尾部。
`checkpoint_every == 0` 会禁用自动复合；此时应手动调用 [`Self::checkpoint`]。
*/
pub struct CheckpointedReducer<J, R> {
    /// 权威日志。
    journal: Arc<J>,
    /// 检查点存储。
    checkpoints: Arc<dyn CheckpointStore<R>>,
    /// 自动保存检查点的记录间隔（0 表示禁用）。
    checkpoint_every: u64,
    /// 受互斥锁保护的投影状态。
    inner: Mutex<ReducerInner<R>>,
}

impl<J, R> CheckpointedReducer<J, R>
where
    R: EventReducer,
    J: EventJournal<R::Event>,
{
    /// 以给定日志、检查点存储和检查点间隔创建归约器，初始状态为默认值。
    pub fn new(
        journal: Arc<J>,
        checkpoints: Arc<dyn CheckpointStore<R>>,
        checkpoint_every: u64,
    ) -> Self {
        Self {
            journal,
            checkpoints,
            checkpoint_every,
            inner: Mutex::new(ReducerInner {
                state: R::default(),
                last_applied: 0,
                since_checkpoint: 0,
            }),
        }
    }

    /**
    追加一个非空批次，按顺序折叠已提交记录，并在批次边界最多复合一个检查点。

    并发调用方在追加与折叠上被串行化为一个事务，因此已提交序号不会被重排。
    */
    pub fn apply_batch(
        &self,
        batch: PreparedJournalBatch<R::Event>,
    ) -> std::result::Result<ApplyBatchReceipt, CheckpointedApplyError<R::Event>> {
        let expected = batch.clone();
        /*
        追加与折叠必须在投影锁上构成一个事务。若在两者之间释放锁，持有更晚序号的
        线程可能先折叠，并因虚假的投影缺口而失败，从而违反本模块记录的单一所有者
        不变量。加锁顺序始终是 投影 -> 日志，与 `recover` 保持一致。
        */
        let mut inner = self.inner.lock().unwrap_or_else(|error| error.into_inner());
        let appended = self
            .journal
            .append_batch(batch)
            .map_err(CheckpointedApplyError::Journal)?;
        self.fold_committed_locked(&mut inner, &expected, appended)
    }

    /**
    折叠由外部物理权限对象提交的批次回执。这样既能将投影所有权保留在框架内，
    又允许应用提供具有自定义重新打开和持久性处理的日志。
    */
    pub fn apply_committed(
        &self,
        expected: &PreparedJournalBatch<R::Event>,
        appended: JournalBatchAppendReceipt<R::Event>,
    ) -> std::result::Result<ApplyBatchReceipt, CheckpointedApplyError<R::Event>> {
        let mut inner = self.inner.lock().unwrap_or_else(|error| error.into_inner());
        self.fold_committed_locked(&mut inner, expected, appended)
    }

    /// 在已持有投影锁的前提下，校验回执与预处理批次一致后按序折叠记录并按需保存检查点。
    fn fold_committed_locked(
        &self,
        inner: &mut ReducerInner<R>,
        expected: &PreparedJournalBatch<R::Event>,
        appended: JournalBatchAppendReceipt<R::Event>,
    ) -> std::result::Result<ApplyBatchReceipt, CheckpointedApplyError<R::Event>> {
        let batch_id = appended.batch_id().to_string();
        if appended.journal_identity() != self.journal.journal_identity() {
            return Err(CheckpointedApplyError::CommittedInvariant {
                batch_id,
                error: format!(
                    "journal receipt belongs to generation {}, expected {}",
                    appended.journal_identity(),
                    self.journal.journal_identity()
                ),
            });
        }
        let receipt_matches = expected.matches_receipt(&appended).map_err(|error| {
            CheckpointedApplyError::CommittedInvariant {
                batch_id: batch_id.clone(),
                error: error.to_string(),
            }
        })?;
        if !receipt_matches {
            return Err(CheckpointedApplyError::CommittedInvariant {
                batch_id,
                error: "journal receipt does not match the prepared batch".to_string(),
            });
        }
        let records = appended.records();
        let record_count = u64::try_from(records.len()).map_err(|_| {
            CheckpointedApplyError::CommittedInvariant {
                batch_id: batch_id.clone(),
                error: "record count exceeds supported range".to_string(),
            }
        })?;
        let first_sequence = records
            .first()
            .map(|record| record.sequence)
            .ok_or_else(|| CheckpointedApplyError::CommittedInvariant {
                batch_id: batch_id.clone(),
                error: "receipt contains no records".to_string(),
            })?;
        let last_sequence = records
            .last()
            .map(|record| record.sequence)
            .ok_or_else(|| CheckpointedApplyError::CommittedInvariant {
                batch_id: batch_id.clone(),
                error: "receipt contains no records".to_string(),
            })?;
        if first_sequence == 0 {
            return Err(CheckpointedApplyError::CommittedInvariant {
                batch_id,
                error: "receipt sequence must be positive".to_string(),
            });
        }
        let mut expected_sequence = first_sequence;
        for record in records {
            if record.batch_id() != appended.batch_id() || record.sequence != expected_sequence {
                return Err(CheckpointedApplyError::CommittedInvariant {
                    batch_id: batch_id.clone(),
                    error: format!(
                        "receipt is not one contiguous batch at sequence {expected_sequence}"
                    ),
                });
            }
            expected_sequence = expected_sequence.checked_add(1).ok_or_else(|| {
                CheckpointedApplyError::CommittedInvariant {
                    batch_id: batch_id.clone(),
                    error: "receipt sequence exhausted".to_string(),
                }
            })?;
        }
        if last_sequence <= inner.last_applied
            && appended.commit_status() != JournalBatchCommitStatus::AlreadyCommitted
        {
            return Err(CheckpointedApplyError::CommittedInvariant {
                batch_id: batch_id.clone(),
                error: "new commit is entirely behind the applied projection".to_string(),
            });
        }
        let first_unapplied = records
            .iter()
            .position(|record| record.sequence > inner.last_applied);
        let folded_count = match first_unapplied {
            None => 0,
            Some(index) => {
                let required = inner.last_applied.checked_add(1).ok_or_else(|| {
                    CheckpointedApplyError::CommittedInvariant {
                        batch_id: batch_id.clone(),
                        error: "projection sequence exhausted before receipt suffix".to_string(),
                    }
                })?;
                let first_suffix = records.get(index).ok_or_else(|| {
                    CheckpointedApplyError::CommittedInvariant {
                        batch_id: batch_id.clone(),
                        error: "receipt suffix index is missing".to_string(),
                    }
                })?;
                if first_suffix.sequence != required {
                    return Err(CheckpointedApplyError::CommittedInvariant {
                        batch_id: batch_id.clone(),
                        error: format!(
                            "projection gap: expected sequence {required} but receipt suffix starts at {}",
                            first_suffix.sequence
                        ),
                    });
                }
                if index > 0
                    && appended.commit_status() != JournalBatchCommitStatus::AlreadyCommitted
                {
                    return Err(CheckpointedApplyError::CommittedInvariant {
                        batch_id: batch_id.clone(),
                        error: "new commit illegally overlaps the applied projection".to_string(),
                    });
                }
                let mut folded = 0_u64;
                for record in records.iter().skip(index) {
                    inner.state.apply_record(record);
                    inner.last_applied = record.sequence;
                    folded = folded.saturating_add(1);
                }
                folded
            }
        };
        inner.since_checkpoint = inner.since_checkpoint.saturating_add(folded_count);
        let checkpoint = if folded_count != 0
            && self.checkpoint_every != 0
            && inner.since_checkpoint >= self.checkpoint_every
        {
            match self.checkpoints.save(
                self.journal.journal_identity(),
                &inner.state,
                inner.last_applied,
            ) {
                Ok(()) => {
                    inner.since_checkpoint = 0;
                    CheckpointApplyStatus::Saved
                }
                Err(error) => CheckpointApplyStatus::Degraded {
                    error: error.to_string(),
                },
            }
        } else {
            CheckpointApplyStatus::NotDue
        };
        Ok(ApplyBatchReceipt {
            batch_id,
            first_sequence,
            last_sequence,
            record_count,
            journal: appended.durability().clone(),
            commit: appended.commit_status(),
            checkpoint,
        })
    }

    /**
     * 通过批次权限追加单个事件并将其折叠，必要时复合生成检查点。
     */
    pub fn apply(
        &self,
        event: R::Event,
    ) -> std::result::Result<ApplyReceipt, CheckpointedApplyError<R::Event>> {
        let batch =
            PreparedJournalBatch::new(vec![event]).map_err(CheckpointedApplyError::Prepare)?;
        let receipt = self.apply_batch(batch)?;
        Ok(ApplyReceipt {
            batch_id: receipt.batch_id,
            sequence: receipt.first_sequence,
            journal: receipt.journal,
            commit: receipt.commit,
            checkpoint: receipt.checkpoint,
        })
    }

    /// 将当前状态持久化为截至已应用序号的检查点。
    pub fn checkpoint(&self) -> Result<()> {
        let mut inner = self.inner.lock().unwrap_or_else(|error| error.into_inner());
        self.checkpoints.save(
            self.journal.journal_identity(),
            &inner.state,
            inner.last_applied,
        )?;
        inner.since_checkpoint = 0;
        Ok(())
    }

    /**
     * 加载属于本日志确切代（generation）的检查点，并重放尾部。
     *
     * 仅当完整日志仍可重放时，才会重建来自其他日志的检查点；
     * 一旦前缀被裁剪，身份不匹配将以失败关闭处理。
     */
    pub fn recover(&self) -> Result<RecoveryReceipt> {
        let mut inner = self.inner.lock().unwrap_or_else(|error| error.into_inner());
        let journal_last = self.journal.last_sequence();
        let retained_floor = self.journal.retained_floor();
        let required_checkpoint = retained_floor.saturating_sub(1);
        let loaded = self.checkpoints.load();
        let journal_identity = self.journal.journal_identity();
        let (mut state, mut last_applied, mut checkpoint_sequence, mut repair_reason) = match loaded
        {
            Ok(Some(frame))
                if frame.journal_identity != *journal_identity && retained_floor > 1 =>
            {
                return Err(ReactError::Other(format!(
                    "checkpoint journal identity {} does not match source journal {journal_identity}, and retained journal floor {retained_floor} prevents rebuilding the missing prefix",
                    frame.journal_identity
                )));
            }
            Ok(Some(frame)) if frame.journal_identity != *journal_identity => (
                R::default(),
                0,
                0,
                Some(format!(
                    "checkpoint journal identity {} does not match source journal {journal_identity}",
                    frame.journal_identity
                )),
            ),
            Ok(Some(frame)) if frame.sequence < required_checkpoint => {
                return Err(ReactError::Other(format!(
                    "checkpoint sequence {} is behind retained journal floor {retained_floor}; expected at least {required_checkpoint}",
                    frame.sequence
                )));
            }
            Ok(Some(frame)) if retained_floor > 1 && frame.sequence > journal_last => {
                return Err(ReactError::Other(format!(
                    "checkpoint sequence {} is ahead of journal sequence {journal_last} after prefix pruning",
                    frame.sequence
                )));
            }
            Ok(Some(frame)) if frame.sequence <= journal_last => {
                (frame.state, frame.sequence, frame.sequence, None)
            }
            Ok(Some(frame)) => (
                R::default(),
                0,
                0,
                Some(format!(
                    "checkpoint sequence {} is ahead of journal sequence {journal_last}",
                    frame.sequence
                )),
            ),
            Ok(None) if retained_floor > 1 => {
                return Err(ReactError::Other(format!(
                    "checkpoint is required through sequence {required_checkpoint} because journal retention starts at {retained_floor}"
                )));
            }
            Ok(None) => (R::default(), 0, 0, None),
            Err(error) if retained_floor > 1 => {
                return Err(ReactError::Other(format!(
                    "checkpoint load failed for retained journal floor {retained_floor}: {error}"
                )));
            }
            Err(error) => (
                R::default(),
                0,
                0,
                Some(format!("checkpoint load failed: {error}")),
            ),
        };
        loop {
            let records = self.journal.replay_after(last_applied, RECOVER_BATCH)?;
            if records.is_empty() {
                break;
            }
            for record in records {
                state.apply_record(&record);
                last_applied = record.sequence;
            }
        }
        let replayed_since_checkpoint = last_applied.saturating_sub(checkpoint_sequence);
        let checkpoint_due =
            self.checkpoint_every != 0 && replayed_since_checkpoint >= self.checkpoint_every;
        let checkpoint = if repair_reason.is_some() || checkpoint_due {
            let reason = repair_reason.take().unwrap_or_else(|| {
                format!(
                    "replayed {replayed_since_checkpoint} events after checkpoint sequence {checkpoint_sequence}"
                )
            });
            match self
                .checkpoints
                .save(journal_identity, &state, last_applied)
            {
                Ok(()) => {
                    checkpoint_sequence = last_applied;
                    CheckpointRecoveryStatus::Rebuilt { reason }
                }
                Err(error) => CheckpointRecoveryStatus::Degraded {
                    reason,
                    error: error.to_string(),
                },
            }
        } else if checkpoint_sequence == 0 {
            CheckpointRecoveryStatus::Missing
        } else {
            CheckpointRecoveryStatus::Loaded
        };
        inner.state = state;
        inner.last_applied = last_applied;
        inner.since_checkpoint = last_applied.saturating_sub(checkpoint_sequence);
        Ok(RecoveryReceipt {
            last_applied_sequence: last_applied,
            checkpoint,
        })
    }

    /// 已折叠进当前状态的最后一个日志序号。
    pub fn last_applied_sequence(&self) -> u64 {
        self.inner
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .last_applied
    }

    /// 在归约器锁保护下读取投影状态。
    pub fn with_state<T>(&self, f: impl FnOnce(&R) -> T) -> T {
        let inner = self.inner.lock().unwrap_or_else(|error| error.into_inner());
        f(&inner.state)
    }

    /// 在不绕过归约器锁的前提下，以可变方式访问内存中的投影。
    pub fn with_state_mut<T>(&self, f: impl FnOnce(&mut R) -> T) -> T {
        let mut inner = self.inner.lock().unwrap_or_else(|error| error.into_inner());
        f(&mut inner.state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Barrier;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// 弱引用注册表应按软节奏清理，并在达到硬上限时立即清理。
    #[test]
    fn weak_registry_prunes_on_soft_cadence_and_immediate_hard_limit() {
        let root = PathBuf::from("registry-test");
        let mut registry = WeakRegistry::<()>::new();
        for index in 0..WEAK_REGISTRY_PRUNE_THRESHOLD {
            let authority = Arc::new(());
            registry.insert(root.join(format!("soft-{index}")), &authority);
        }
        for _ in 1..WEAK_REGISTRY_PRUNE_INTERVAL {
            registry.prune_dead_if_due();
        }
        assert_eq!(registry.paths_beneath(&root), WEAK_REGISTRY_PRUNE_THRESHOLD);
        registry.prune_dead_if_due();
        assert_eq!(registry.paths_beneath(&root), 0);

        for index in 0..WEAK_REGISTRY_HARD_LIMIT {
            let authority = Arc::new(());
            registry.insert(root.join(format!("hard-{index}")), &authority);
        }
        registry.prune_dead_if_due();
        assert_eq!(registry.paths_beneath(&root), 0);

        let mut live = Vec::new();
        for index in 0..WEAK_REGISTRY_HARD_LIMIT.saturating_add(8) {
            let authority = Arc::new(());
            registry.insert(root.join(format!("live-{index}")), &authority);
            live.push(authority);
        }
        registry.prune_dead_if_due();
        assert_eq!(
            registry.paths_beneath(&root),
            WEAK_REGISTRY_HARD_LIMIT.saturating_add(8)
        );
        drop(live);
        registry.prune_dead_if_due();
        assert_eq!(registry.paths_beneath(&root), 0);
    }

    /// 测试用归约器：累加事件值并记录事件序列。
    #[derive(Default, Serialize, Deserialize, Debug)]
    struct SumReducer {
        /// 事件值之和
        total: i64,
        /// 已应用的事件列表
        events: Vec<i32>,
    }

    /// 序列化必定失败的测试事件。
    #[derive(Debug, Deserialize)]
    struct FailingEvent;

    impl Serialize for FailingEvent {
        /// 始终返回注入的序列化错误。
        fn serialize<S>(&self, _serializer: S) -> std::result::Result<S::Ok, S::Error>
        where
            S: serde::Serializer,
        {
            Err(<S::Error as serde::ser::Error>::custom(
                "injected event serialization failure",
            ))
        }
    }

    /// 内部可变的测试事件，用于检测追加前的内部修改。
    #[derive(Debug)]
    struct MutableEvent {
        /// 可被原子修改的事件值
        value: AtomicUsize,
    }

    impl Serialize for MutableEvent {
        /// 将当前值序列化为 u64。
        fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
        where
            S: serde::Serializer,
        {
            serializer
                .serialize_u64(u64::try_from(self.value.load(Ordering::SeqCst)).unwrap_or(u64::MAX))
        }
    }

    impl<'de> Deserialize<'de> for MutableEvent {
        /// 从 usize 反序列化事件。
        fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
        where
            D: serde::Deserializer<'de>,
        {
            let value = usize::deserialize(deserializer)?;
            Ok(Self {
                value: AtomicUsize::new(value),
            })
        }
    }

    impl EventReducer for SumReducer {
        /// 事件类型为 i32。
        type Event = i32;

        /// 累加事件值并记录事件。
        fn apply(&mut self, event: &i32) {
            self.total = self.total.saturating_add(i64::from(*event));
            self.events.push(*event);
        }
    }

    /// 基于内存日志的求和归约器测试装置类型。
    type SumHarness = CheckpointedReducer<MemoryEventJournal<i32>, SumReducer>;

    /// 测试夹具：归约器及其共享的日志与检查点存储。
    struct SumFixture {
        /// 被测归约器
        reducer: SumHarness,
        /// 共享的内存日志
        journal: Arc<MemoryEventJournal<i32>>,
        /// 共享的内存检查点存储
        checkpoints: Arc<MemoryCheckpointStore<SumReducer>>,
    }

    /// 记录每次重放 limit 参数的包装日志。
    struct TrackingJournal {
        /// 被包装的内存日志
        inner: MemoryEventJournal<i32>,
        /// 每次 replay_after 调用的 limit 记录
        replay_limits: Mutex<Vec<usize>>,
    }

    impl TrackingJournal {
        /// 创建空的跟踪日志。
        fn new() -> Self {
            Self {
                inner: MemoryEventJournal::new(),
                replay_limits: Mutex::new(Vec::new()),
            }
        }
    }

    impl EventJournal<i32> for TrackingJournal {
        /// 透传日志身份。
        fn journal_identity(&self) -> &JournalIdentity {
            self.inner.journal_identity()
        }

        /// 透传批次追加。
        fn append_batch(&self, batch: PreparedJournalBatch<i32>) -> JournalBatchAppendResult<i32> {
            self.inner.append_batch(batch)
        }

        /// 透传批次查找。
        fn lookup_batch(
            &self,
            batch: &PreparedJournalBatch<i32>,
        ) -> Result<JournalBatchLookup<i32>> {
            self.inner.lookup_batch(batch)
        }

        /// 透传下一个序号。
        fn next_sequence(&self) -> u64 {
            self.inner.next_sequence()
        }

        /// 透传最后一个序号。
        fn last_sequence(&self) -> u64 {
            self.inner.last_sequence()
        }

        /// 记录 limit 后透传重放。
        fn replay_after(
            &self,
            after_sequence: u64,
            limit: usize,
        ) -> Result<Vec<JournalRecord<i32>>> {
            self.replay_limits
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .push(limit);
            self.inner.replay_after(after_sequence, limit)
        }
    }

    /// 创建带指定检查点间隔的求和测试夹具。
    fn reducer_with(checkpoint_every: u64) -> SumFixture {
        let journal = Arc::new(MemoryEventJournal::new());
        let checkpoints = Arc::new(MemoryCheckpointStore::new());
        let reducer = CheckpointedReducer::new(
            Arc::clone(&journal),
            Arc::clone(&checkpoints) as Arc<dyn CheckpointStore<SumReducer>>,
            checkpoint_every,
        );
        SumFixture {
            reducer,
            journal,
            checkpoints,
        }
    }

    /// 由事件列表构造测试批次。
    fn batch<E: JournalEvent>(events: Vec<E>) -> TestResult<PreparedJournalBatch<E>> {
        PreparedJournalBatch::new(events).test_context("prepare test batch")
    }

    /// 内存日志应从 1 开始分配连续序号。
    #[test]
    fn memory_journal_assigns_contiguous_sequences_from_one() {
        let journal = MemoryEventJournal::new();
        assert_eq!(journal.last_sequence(), 0);
        assert_eq!(journal.next_sequence(), 1);
        let first = journal.append(10).expect("append");
        let second = journal.append(20).expect("append");
        assert_eq!(first.record.sequence, 1);
        assert_eq!(second.record.sequence, 2);
        assert_eq!(first.durability, JournalDurabilityStatus::Confirmed);
        assert_eq!(second.durability, JournalDurabilityStatus::Confirmed);
        assert_eq!(journal.last_sequence(), 2);
        assert_eq!(journal.next_sequence(), 3);
    }

    /// 内存批次应为一次有序提交，空批次不产生写入。
    #[test]
    fn memory_batch_is_one_ordered_commit_and_empty_is_zero_write() -> TestResult {
        let journal = MemoryEventJournal::new();
        let Err(empty) = PreparedJournalBatch::new(Vec::<i32>::new()) else {
            return Err(test_failure("empty batch unexpectedly passed preflight"));
        };
        assert!(empty.error.contains("at least one"));
        assert_eq!(journal.next_sequence(), 1);

        let receipt = journal
            .append_batch(batch(vec![10, 20, 30])?)
            .test_context("append memory batch")?;
        assert!(uuid::Uuid::parse_str(&receipt.batch_id).is_ok());
        assert_eq!(receipt.records.len(), 3);
        assert_eq!(
            receipt
                .records
                .iter()
                .map(|record| record.sequence)
                .collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
        let replay = journal
            .replay_after(0, usize::MAX)
            .test_context("replay batch")?;
        assert_eq!(replay.len(), 3);
        assert_eq!(
            journal
                .replay_after(1, 1)
                .test_context("replay within batch")?
                .first()
                .map(|record| *record.event),
            Some(20)
        );
        assert!(
            receipt
                .records
                .iter()
                .zip(replay.iter())
                .all(|(committed, replayed)| Arc::ptr_eq(&committed.event, &replayed.event))
        );
        Ok(())
    }

    /// 批次身份应幂等，身份冲突应使日志中毒。
    #[test]
    fn memory_batch_identity_is_idempotent_and_conflicts_poison() -> TestResult {
        let journal = MemoryEventJournal::new();
        let original = batch(vec![10, 20])?;
        let batch_id = original.batch_id().to_string();
        let committed = journal
            .append_batch(original)
            .test_context("initial commit")?;
        assert_eq!(
            committed.commit_status(),
            JournalBatchCommitStatus::Committed
        );

        let duplicate = PreparedJournalBatch::with_test_identity(batch_id.clone(), vec![10, 20])
            .test_context("same identity payload")?;
        let idempotent = journal
            .append_batch(duplicate)
            .test_context("idempotent append")?;
        assert_eq!(
            idempotent.commit_status(),
            JournalBatchCommitStatus::AlreadyCommitted
        );
        assert_eq!(
            idempotent
                .records()
                .iter()
                .map(|record| record.sequence)
                .collect::<Vec<_>>(),
            vec![1, 2]
        );
        assert_eq!(journal.last_sequence(), 2);

        let conflict = PreparedJournalBatch::with_test_identity(batch_id, vec![10, 99])
            .test_context("conflicting identity payload")?;
        let Err(error) = journal.append_batch(conflict) else {
            return Err(test_failure("identity conflict unexpectedly committed"));
        };
        assert!(error.to_string().contains("conflicts"));
        assert!(!error.is_retry_safe());
        let Err(refused) = journal.append_batch(batch(vec![30])?) else {
            return Err(test_failure(
                "poisoned memory authority unexpectedly accepted a retry",
            ));
        };
        assert!(refused.to_string().contains("poisoned"));
        assert!(!refused.is_retry_safe());
        assert!(refused.requires_reopen());
        Ok(())
    }

    /// 幂等查找前应拒绝内部被修改的事件。
    #[test]
    fn memory_rejects_interior_mutation_before_idempotent_lookup() -> TestResult {
        let journal = MemoryEventJournal::new();
        let original = batch(vec![MutableEvent {
            value: AtomicUsize::new(1),
        }])?;
        let batch_id = original.batch_id().to_string();
        journal
            .append_batch(original)
            .test_context("commit original payload")?;

        let mutated = PreparedJournalBatch::with_test_identity(
            batch_id.clone(),
            vec![MutableEvent {
                value: AtomicUsize::new(1),
            }],
        )
        .test_context("prepare mutable duplicate")?;
        if let Some(event) = mutated.events().first() {
            event.value.store(2, Ordering::SeqCst);
        }
        assert!(matches!(
            journal
                .lookup_batch(&mutated)
                .test_context("lookup mutated payload")?,
            JournalBatchLookup::Conflict { .. }
        ));
        let Err(mutation) = journal.append_batch(mutated) else {
            return Err(test_failure(
                "mutated prepared payload unexpectedly matched the committed batch",
            ));
        };
        assert!(matches!(
            mutation,
            JournalBatchAppendError::PreparedMutation { .. }
        ));
        assert!(!mutation.is_retry_safe());
        assert_eq!(journal.last_sequence(), 1);

        let original_lookup = PreparedJournalBatch::with_test_identity(
            batch_id,
            vec![MutableEvent {
                value: AtomicUsize::new(1),
            }],
        )
        .test_context("prepare original digest lookup")?;
        assert!(matches!(
            journal
                .lookup_batch(&original_lookup)
                .test_context("lookup original payload")?,
            JournalBatchLookup::AlreadyCommitted(_)
        ));
        Ok(())
    }

    /// 序号溢出时批次预检应拒绝且不修改状态。
    #[test]
    fn memory_batch_preflight_rejects_sequence_overflow_without_mutation() -> TestResult {
        let journal = MemoryEventJournal::new();
        journal
            .inner
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .next_sequence = u64::MAX;
        let Err(error) = journal.append_batch(batch(vec![1])?) else {
            return Err(test_failure("sequence overflow unexpectedly committed"));
        };
        assert!(matches!(
            error,
            JournalBatchAppendError::NotCommitted { .. }
        ));
        assert_eq!(journal.next_sequence(), u64::MAX);
        assert!(
            journal
                .replay_after(0, usize::MAX)
                .test_context("replay")?
                .is_empty()
        );
        Ok(())
    }

    /// 序列化失败的批次不应被提交。
    #[test]
    fn memory_batch_serialization_failure_is_not_committed() -> TestResult {
        let journal = MemoryEventJournal::<FailingEvent>::new();
        let Err(error) = PreparedJournalBatch::new(vec![FailingEvent]) else {
            return Err(test_failure(
                "serialization failure unexpectedly produced a prepared batch",
            ));
        };
        assert!(error.error.contains("serialization failure"));
        assert_eq!(journal.next_sequence(), 1);
        assert!(
            journal
                .replay_after(0, usize::MAX)
                .test_context("replay")?
                .is_empty()
        );
        Ok(())
    }

    /// 重放应遵守边界与数量上限。
    #[test]
    fn replay_after_respects_boundary_and_limit() {
        let journal = MemoryEventJournal::new();
        for value in 0..10 {
            journal.append(value).expect("append");
        }
        let records = journal.replay_after(7, 2).expect("replay");
        assert_eq!(
            records.iter().map(|r| r.sequence).collect::<Vec<_>>(),
            vec![8, 9]
        );
        let rest = journal.replay_after(7, usize::MAX).expect("replay");
        assert_eq!(rest.len(), 3);
        let all = journal.replay_after(0, usize::MAX).expect("replay");
        assert_eq!(all.len(), 10);
    }

    /// apply 应折叠状态并返回序号。
    #[test]
    fn apply_folds_state_and_returns_sequence() {
        let fixture = reducer_with(0);
        assert_eq!(fixture.reducer.apply(5).expect("apply").sequence, 1);
        assert_eq!(fixture.reducer.apply(7).expect("apply").sequence, 2);
        assert_eq!(fixture.journal.last_sequence(), 2);
        fixture.reducer.with_state(|state| {
            assert_eq!(state.total, 12);
            assert_eq!(state.events, vec![5, 7]);
        });
    }

    /// 自动检查点应按节奏触发。
    #[test]
    fn automatic_checkpoint_fires_on_cadence() {
        let fixture = reducer_with(2);
        fixture.reducer.apply(1).expect("apply");
        assert!(fixture.checkpoints.load().expect("load").is_none());
        fixture.reducer.apply(2).expect("apply");
        let frame = fixture
            .checkpoints
            .load()
            .expect("load")
            .expect("checkpoint saved");
        assert_eq!(frame.sequence, 2);
        assert_eq!(frame.state.total, 3);
    }

    /// 跨越检查点节奏的批次应先折叠，再在批次末尾保存检查点。
    #[test]
    fn batch_crossing_checkpoint_cadence_folds_then_saves_at_batch_end() -> TestResult {
        let fixture = reducer_with(3);
        fixture.reducer.apply(1).test_context("seed event")?;
        let receipt = fixture
            .reducer
            .apply_batch(batch(vec![2, 3, 4])?)
            .test_context("apply batch")?;
        assert_eq!(receipt.first_sequence, 2);
        assert_eq!(receipt.last_sequence, 4);
        assert_eq!(receipt.record_count, 3);
        assert_eq!(receipt.checkpoint, CheckpointApplyStatus::Saved);
        let frame = fixture
            .checkpoints
            .load()
            .test_context("load checkpoint")?
            .test_context("batch checkpoint")?;
        assert_eq!(frame.sequence, 4);
        assert_eq!(frame.state.events, vec![1, 2, 3, 4]);
        Ok(())
    }

    /// 已提交的部分重叠批次只折叠连续的后缀。
    #[test]
    fn already_committed_partial_overlap_folds_only_contiguous_suffix() -> TestResult {
        let journal = Arc::new(MemoryEventJournal::new());
        let original = batch(vec![1, 2, 3])?;
        let batch_id = original.batch_id().to_string();
        let committed = journal
            .append_batch(original)
            .test_context("commit source batch")?;
        let reducer = CheckpointedReducer::<_, SumReducer>::new(
            Arc::clone(&journal),
            Arc::new(MemoryCheckpointStore::new()),
            0,
        );
        {
            let mut inner = reducer
                .inner
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            if let Some(first) = committed.records().first() {
                inner.state.apply_record(first);
                inner.last_applied = first.sequence;
                inner.since_checkpoint = 1;
            }
        }
        let duplicate = PreparedJournalBatch::with_test_identity(batch_id, vec![1, 2, 3])
            .test_context("prepare idempotent overlap")?;
        let receipt = reducer
            .apply_batch(duplicate)
            .test_context("fold idempotent suffix")?;
        assert_eq!(receipt.commit, JournalBatchCommitStatus::AlreadyCommitted);
        assert_eq!(reducer.last_applied_sequence(), 3);
        reducer.with_state(|state| assert_eq!(state.events, vec![1, 2, 3]));
        Ok(())
    }

    /// 恢复时应从检查点开始并只重放尾部。
    #[test]
    fn recover_replays_from_checkpoint_then_tail() {
        let fixture = reducer_with(2);
        for value in 1..=5 {
            fixture.reducer.apply(value).expect("apply");
        }
        assert_eq!(fixture.reducer.last_applied_sequence(), 5);

        /* 第二个归约器共享同一日志和检查点存储，
        通过加载检查点并仅重放尾部完成恢复。 */
        let second: SumHarness = CheckpointedReducer::new(
            fixture.journal,
            fixture.checkpoints as Arc<dyn CheckpointStore<SumReducer>>,
            2,
        );
        assert_eq!(second.recover().expect("recover").last_applied_sequence, 5);
        second.with_state(|state| {
            assert_eq!(state.total, 15);
            assert_eq!(state.events, (1..=5).collect::<Vec<_>>());
        });
    }

    /// 无检查点时恢复应重放全部事件。
    #[test]
    fn recover_without_checkpoint_replays_everything() {
        let fixture = reducer_with(0);
        for value in 1..=4 {
            fixture.reducer.apply(value).expect("apply");
        }
        let second: SumHarness = CheckpointedReducer::new(
            fixture.journal,
            Arc::new(MemoryCheckpointStore::<SumReducer>::new()),
            0,
        );
        assert_eq!(second.recover().expect("recover").last_applied_sequence, 4);
        second.with_state(|state| assert_eq!(state.total, 10));
    }

    /// 来自其他日志的同序号检查点应被重建。
    #[test]
    fn recover_rebuilds_same_sequence_checkpoint_from_another_journal() -> TestResult {
        let source = Arc::new(MemoryEventJournal::new());
        let foreign = Arc::new(MemoryEventJournal::new());
        let checkpoints = Arc::new(MemoryCheckpointStore::new());
        let foreign_reducer = CheckpointedReducer::new(
            Arc::clone(&foreign),
            Arc::clone(&checkpoints) as Arc<dyn CheckpointStore<SumReducer>>,
            1,
        );
        foreign_reducer
            .apply(99)
            .test_context("checkpoint foreign journal")?;
        source.append(7).test_context("append source journal")?;

        let recovered = CheckpointedReducer::new(
            Arc::clone(&source),
            Arc::clone(&checkpoints) as Arc<dyn CheckpointStore<SumReducer>>,
            1,
        );
        let receipt = recovered.recover().test_context("recover source journal")?;
        let CheckpointRecoveryStatus::Rebuilt { reason } = receipt.checkpoint else {
            return Err(test_failure("foreign checkpoint was accepted as loaded"));
        };
        assert!(reason.contains("does not match source journal"));
        recovered.with_state(|state| {
            assert_eq!(state.total, 7);
            assert_eq!(state.events, vec![7]);
        });
        let repaired = checkpoints
            .load()
            .test_context("load repaired checkpoint")?
            .test_context("repaired checkpoint exists")?;
        assert_eq!(repaired.journal_identity, *source.journal_identity());
        assert_eq!(repaired.sequence, 1);
        Ok(())
    }

    /// apply_committed 应拒绝来自其他日志的回执。
    #[test]
    fn apply_committed_rejects_receipt_from_another_journal() -> TestResult {
        let source = Arc::new(MemoryEventJournal::new());
        let foreign = MemoryEventJournal::new();
        let prepared = batch(vec![11])?;
        let receipt = foreign
            .append_batch(prepared.clone())
            .test_context("commit foreign receipt")?;
        let reducer = CheckpointedReducer::<_, SumReducer>::new(
            Arc::clone(&source),
            Arc::new(MemoryCheckpointStore::new()),
            1,
        );

        let Err(error) = reducer.apply_committed(&prepared, receipt) else {
            return Err(test_failure("foreign receipt was folded"));
        };
        assert!(error.to_string().contains("receipt belongs to generation"));
        assert_eq!(source.last_sequence(), 0);
        assert_eq!(reducer.last_applied_sequence(), 0);
        reducer.with_state(|state| {
            assert_eq!(state.total, 0);
            assert!(state.events.is_empty());
        });
        Ok(())
    }

    /// 缺失检查点的大量尾部应按固定批次重放。
    #[test]
    fn recovery_replays_large_missing_checkpoint_tail_in_fixed_batches() {
        /// 测试事件总数。
        const EVENTS: i32 = 1_537;
        let journal = Arc::new(TrackingJournal::new());
        for value in 0..EVENTS {
            journal.append(value).expect("append");
        }
        let reducer = CheckpointedReducer::<_, SumReducer>::new(
            Arc::clone(&journal),
            Arc::new(MemoryCheckpointStore::new()),
            0,
        );

        let recovered = reducer.recover().expect("recover");
        assert_eq!(
            recovered.last_applied_sequence,
            u64::try_from(EVENTS).unwrap_or(u64::MAX)
        );
        let limits = journal
            .replay_limits
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        assert!(limits.len() >= 4);
        assert!(limits.iter().all(|limit| *limit == RECOVER_BATCH));
    }

    /// 手动检查点应能通过存储往返。
    #[test]
    fn manual_checkpoint_round_trips_through_store() {
        let fixture = reducer_with(0);
        for value in 1..=3 {
            fixture.reducer.apply(value).expect("apply");
        }
        fixture.reducer.checkpoint().expect("checkpoint");
        let frame = fixture
            .checkpoints
            .load()
            .expect("load")
            .expect("checkpoint present");
        assert_eq!(frame.sequence, 3);
        assert_eq!(frame.state.total, 6);
    }

    /// 第一次保存失败、之后正常的检查点存储。
    struct FailOnceCheckpointStore {
        /// 剩余需要注入失败的次数
        remaining_failures: AtomicUsize,
        /// 被包装的内存检查点存储
        inner: MemoryCheckpointStore<SumReducer>,
    }

    impl FailOnceCheckpointStore {
        /// 创建只失败一次的检查点存储。
        fn new() -> Self {
            Self {
                remaining_failures: AtomicUsize::new(1),
                inner: MemoryCheckpointStore::new(),
            }
        }
    }

    impl CheckpointStore<SumReducer> for FailOnceCheckpointStore {
        /// 在剩余失败次数内返回注入错误，否则正常保存。
        fn save(
            &self,
            journal_identity: &JournalIdentity,
            state: &SumReducer,
            through_sequence: u64,
        ) -> Result<()> {
            if self
                .remaining_failures
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |remaining| {
                    remaining.checked_sub(1)
                })
                .is_ok()
            {
                return Err(ReactError::Other("injected checkpoint failure".to_string()));
            }
            self.inner.save(journal_identity, state, through_sequence)
        }

        /// 透传加载检查点。
        fn load(&self) -> Result<Option<CheckpointFrame<SumReducer>>> {
            self.inner.load()
        }
    }

    /// 检查点失败时应返回已提交批次，并标记检查点降级。
    #[test]
    fn checkpoint_failure_returns_committed_batch_with_degraded_checkpoint() -> TestResult {
        let journal = Arc::new(MemoryEventJournal::new());
        let checkpoints = Arc::new(FailOnceCheckpointStore::new());
        let reducer = CheckpointedReducer::new(
            Arc::clone(&journal),
            Arc::clone(&checkpoints) as Arc<dyn CheckpointStore<SumReducer>>,
            2,
        );

        let first = reducer
            .apply_batch(batch(vec![7, 8, 9])?)
            .test_context("event batch must commit")?;
        assert_eq!(first.first_sequence, 1);
        assert_eq!(first.last_sequence, 3);
        assert!(matches!(
            first.checkpoint,
            CheckpointApplyStatus::Degraded { .. }
        ));
        assert_eq!(journal.last_sequence(), 3);
        reducer.with_state(|state| assert_eq!(state.events, vec![7, 8, 9]));

        let second = reducer
            .apply(10)
            .test_context("later append retries checkpoint")?;
        assert_eq!(second.sequence, 4);
        assert_eq!(second.checkpoint, CheckpointApplyStatus::Saved);
        let frame = checkpoints
            .load()
            .test_context("checkpoint load")?
            .test_context("checkpoint repaired")?;
        assert_eq!(frame.sequence, 4);
        assert_eq!(frame.state.events, vec![7, 8, 9, 10]);
        Ok(())
    }

    /// 并发 apply 应保持投影与日志顺序一致。
    #[test]
    fn concurrent_apply_preserves_journal_order_in_projection() {
        /// 并发线程数。
        const THREADS: usize = 32;
        let journal = Arc::new(MemoryEventJournal::new());
        let reducer = Arc::new(CheckpointedReducer::<_, SumReducer>::new(
            Arc::clone(&journal),
            Arc::new(MemoryCheckpointStore::new()),
            0,
        ));
        let barrier = Arc::new(Barrier::new(THREADS));
        let mut handles = Vec::new();
        for value in 0..THREADS {
            let reducer = Arc::clone(&reducer);
            let barrier = Arc::clone(&barrier);
            handles.push(std::thread::spawn(move || {
                barrier.wait();
                let value = i32::try_from(value).unwrap_or(i32::MAX);
                reducer.apply(value).map(|receipt| receipt.sequence)
            }));
        }
        for handle in handles {
            let joined = handle.join();
            let ok = matches!(&joined, Ok(Ok(_)));
            assert!(ok, "subagent failed: {joined:?}");
        }

        let journal_events = journal
            .replay_after(0, usize::MAX)
            .expect("journal replay")
            .into_iter()
            .map(|record| *record.event)
            .collect::<Vec<_>>();
        reducer.with_state(|state| assert_eq!(state.events, journal_events));
        assert_eq!(
            reducer.last_applied_sequence(),
            u64::try_from(THREADS).unwrap_or(u64::MAX)
        );
    }

    /// 并发的内存批次记录不应交错。
    #[test]
    fn concurrent_memory_batches_never_interleave_records() -> TestResult {
        /// 并发线程数。
        const THREADS: usize = 16;
        let journal = Arc::new(MemoryEventJournal::new());
        let barrier = Arc::new(Barrier::new(THREADS));
        let mut handles = Vec::new();
        for index in 0..THREADS {
            let journal = Arc::clone(&journal);
            let barrier = Arc::clone(&barrier);
            handles.push(std::thread::spawn(move || -> TestResult<_> {
                barrier.wait();
                let base = i32::try_from(index).unwrap_or(i32::MAX).saturating_mul(10);
                journal
                    .append_batch(batch(vec![
                        base,
                        base.saturating_add(1),
                        base.saturating_add(2),
                    ])?)
                    .test_context("append concurrent memory batch")
            }));
        }
        let mut receipts = Vec::new();
        for handle in handles {
            let receipt = handle.join().test_context("join concurrent batch")??;
            receipts.push(receipt);
        }
        let replay = journal.replay_after(0, usize::MAX).test_context("replay")?;
        for receipt in receipts {
            let first = receipt
                .records
                .first()
                .map(|record| record.sequence)
                .test_context("batch first sequence")?;
            let record_count = u64::try_from(receipt.records.len()).unwrap_or(u64::MAX);
            let values = replay
                .iter()
                .filter(|record| {
                    record.sequence >= first && record.sequence < first.saturating_add(record_count)
                })
                .map(|record| *record.event)
                .collect::<Vec<_>>();
            assert_eq!(values.len(), 3);
            assert!(values.windows(2).all(|pair| {
                pair.first()
                    .zip(pair.get(1))
                    .is_some_and(|(left, right)| left.saturating_add(1) == *right)
            }));
        }
        Ok(())
    }

    /// 领先于日志的检查点应从权威事件重建。
    #[test]
    fn checkpoint_ahead_of_journal_is_rebuilt_from_authoritative_events() {
        let journal = Arc::new(MemoryEventJournal::new());
        journal.append(3).expect("append");
        let checkpoints = Arc::new(MemoryCheckpointStore::new());
        checkpoints
            .save(
                journal.journal_identity(),
                &SumReducer {
                    total: 999,
                    events: vec![999],
                },
                99,
            )
            .expect("seed invalid checkpoint");
        let reducer = CheckpointedReducer::new(
            journal,
            Arc::clone(&checkpoints) as Arc<dyn CheckpointStore<SumReducer>>,
            10,
        );

        let receipt = reducer.recover().expect("recover from journal");
        assert_eq!(receipt.last_applied_sequence, 1);
        assert!(matches!(
            receipt.checkpoint,
            CheckpointRecoveryStatus::Rebuilt { .. }
        ));
        reducer.with_state(|state| {
            assert_eq!(state.total, 3);
            assert_eq!(state.events, vec![3]);
        });
        assert_eq!(
            checkpoints
                .load()
                .expect("load")
                .expect("repaired checkpoint")
                .sequence,
            1
        );
    }

    /// 记录所收到权威序号的测试归约器。
    #[derive(Default, Serialize, Deserialize)]
    struct SequenceAwareReducer {
        /// 收到的记录序号列表
        sequences: Vec<u64>,
    }

    impl EventReducer for SequenceAwareReducer {
        /// 事件类型为 String。
        type Event = String;

        /// 忽略事件内容。
        fn apply(&mut self, _event: &String) {}

        /// 记录权威序号。
        fn apply_record(&mut self, record: &JournalRecord<Self::Event>) {
            self.sequences.push(record.sequence);
        }
    }

    /// 归约器在 apply 与恢复时都应收到权威序号。
    #[test]
    fn reducer_receives_authoritative_sequence_on_apply_and_recovery() {
        let journal = Arc::new(MemoryEventJournal::new());
        let writer = CheckpointedReducer::<_, SequenceAwareReducer>::new(
            Arc::clone(&journal),
            Arc::new(MemoryCheckpointStore::new()),
            0,
        );
        writer.apply("one".to_string()).expect("first apply");
        writer.apply("two".to_string()).expect("second apply");
        writer.with_state(|state| assert_eq!(state.sequences, vec![1, 2]));

        let recovered = CheckpointedReducer::<_, SequenceAwareReducer>::new(
            journal,
            Arc::new(MemoryCheckpointStore::new()),
            0,
        );
        recovered.recover().expect("recover");
        recovered.with_state(|state| assert_eq!(state.sequences, vec![1, 2]));
    }
}
