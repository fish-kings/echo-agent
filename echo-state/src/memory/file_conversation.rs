/*!
 * 文件后端的 [`ConversationStore`] —— 一个无额外依赖的 JSON 文件存储实现。
 *
 * 每个会话都有一份小型清单（manifest）和一份追加式的消息日志，另外在
 * `<base>/conversations/_meta.json` 中维护一个单调递增的 id 计数器。这是在不使用
 * SQLite 时替代 `SqliteConversationStore`（`sqlite` feature）的方案。
 *
 * ## 目录结构
 *
 * - `<base>/conversations/<safe_id>.json` —— 会话元数据，以及已提交的消息日志的
 *   代（generation）编号/字节边界，原子写入。
 * - `<base>/conversations/<safe_id>.messages.<generation>.jsonl` —— 消息内容。
 *   纯粹的转录增长只追加新的后缀；如果发生替换，则会先原子写入一个新的
 *   generation，再提交清单。
 * - `<base>/conversations/_meta.json` —— 单调递增的 id 计数器，用来替代 SQLite 的
 *   自增主键。每新增一个会话/消息都会递增该计数器，从而保证 id 在整个存储中唯一；
 *   重启后可自我修复（self-heal）。
 *
 * ## 健壮性设计
 *
 * - **路径安全的 id。** 会话 id 在拼入路径前会被规整为文件系统安全的片段（拒绝
 *   `/`、`\`、`..`），从而防止路径穿越（`../foo`）以及意外的目录逃逸。
 * - **损坏的 JSON 视为错误。** 格式错误的记录/元数据文件会返回
 *   `MemoryError::SerializationError`，而不是静默返回 `None`/空列表（以前这种做法
 *   会让"数据损坏"和"没有数据"难以区分）。
 * - **唯一临时文件名 + 父目录同步。** 每次原子写入都使用带 uuid 后缀的临时文件
 *   （避免并发写入冲突），在 Unix 上重命名后还会对父目录执行 `fsync`，确保重命名
 *   能在崩溃后仍然生效。
 * - **已提交日志边界。** 消息字节会在清单推进前先落盘同步。读取方会忽略未提交的
 *   尾部数据，下一次追加会在写入前先清理这部分尾部，因此崩溃不会暴露出不完整的
 *   消息。
 */

use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, RwLock, Weak};

use echo_core::error::{MemoryError, Result};
use echo_core::memory::conversation::{
    Conversation, ConversationFilter, ConversationMeta, ConversationProjectionAuthority,
    ConversationProjectionCapability, ConversationProjectionEpochReceipt,
    ConversationProjectionEpochStatus, ConversationProjectionLifecycle, ConversationStore,
    EnsureConversationProjectionRequest, ManagedConversationDelete,
    ManagedConversationDeleteReceipt, ManagedConversationDeleteStatus, ManagedConversationImport,
    ManagedConversationMetadataUpdate, ManagedConversationMetadataUpdateReceipt,
    ManagedConversationMetadataUpdateStatus, NewConversation, PersistenceCallCapability,
    PersistenceCallContext, StoredMessage, TranscriptProjectionApplyReceipt,
    TranscriptProjectionApplyStatus, TranscriptProjectionBatch, TranscriptProjectionConflictKind,
};
use echo_core::utils::blocking::{
    BlockingFileOperationKey, BlockingFileOperationScope, run_keyed_file_operation,
};
use echo_core::utils::fs::{ExclusiveFileLease, try_exclusive_file_lease};
use futures::future::BoxFuture;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// 返回 `Result<T>` 的装箱异步 future 简写。
type BoxFut<'a, T> = BoxFuture<'a, Result<T>>;

/// 搜索候选过滤器位图所含的 `u64` 字数（共 64 × 64 = 4096 位）。
const SEARCH_FILTER_WORDS: usize = 64;
/// 投影 epoch 允许的最大值（与 SQLite 的 `i64` 上限保持一致）。
const MAX_PROJECTION_EPOCH: u64 = i64::MAX as u64;
/// 每个会话最多保留的删除回执数量，超出部分按 epoch 从旧到新淘汰。
const DELETE_RECEIPT_RETENTION: usize = 32;

/**
 * 一份会话清单，加上从其已提交日志中加载出的消息。
 *
 * `messages` 字段依然可以反序列化最初单文件布局所写出的记录。后续第一次调用
 * `save_messages` 时，会把这类旧记录迁移到追加式布局。
 */
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ConversationRecord {
    /// 会话本身的元数据（标题、用户、时间戳等）。
    conversation: Conversation,
    /// 旧版单文件布局内嵌的消息；一旦迁移到 `message_log`，此字段清空。
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    messages: Vec<StoredMessage>,
    /// 追加式消息日志的提交状态（generation、已提交字节数等）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    message_log: Option<MessageLogMeta>,
    /// 用于快速排除不匹配会话的搜索候选过滤器。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    search_filter: Option<SearchFilter>,
    /// 托管会话的投影（projection）状态；`None` 表示这是一个非托管会话。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    projection: Option<FileProjectionState>,
}

/**
 * 单个会话的投影（projection）权威状态：epoch、revision、生命周期，
 * 以及去重/回执所需的簿记信息。
 */
#[derive(Debug, Clone, Serialize, Deserialize)]
struct FileProjectionState {
    /// 当前投影所处的 epoch；每次(重新)创建会话都会递增。
    epoch: u64,
    /// 当前投影的修订号；每次成功变更都会递增。
    revision: u64,
    /// 会话当前的生命周期状态（存活/已删除等）。
    lifecycle: ConversationProjectionLifecycle,
    /// 删除回执的保留下限 epoch；低于此值的回执可被回收。
    #[serde(default)]
    retention_floor_epoch: u64,
    /// 已应用的操作 id → payload 摘要，用于幂等去重。
    #[serde(default)]
    applied_operations: HashMap<String, String>,
    /// 每个 generation 内，序号（ordinal）到摘要的映射，用于校验转录顺序与内容一致性。
    #[serde(default)]
    ordinals: BTreeMap<String, BTreeMap<u64, String>>,
    /// 已保留的删除回执，按操作 id 索引。
    #[serde(default)]
    delete_receipts: HashMap<String, FileDeleteReceipt>,
}

impl FileProjectionState {
    /// 构造一个全新的、处于存活状态的投影（epoch/revision 均从 1 开始）。
    fn live() -> Self {
        Self {
            epoch: 1,
            revision: 1,
            lifecycle: ConversationProjectionLifecycle::Live,
            retention_floor_epoch: 0,
            applied_operations: HashMap::new(),
            ordinals: BTreeMap::new(),
            delete_receipts: HashMap::new(),
        }
    }

    /// 把内部状态导出为对外暴露的权威（authority）快照。
    fn authority(&self, conversation_id: &str) -> ConversationProjectionAuthority {
        ConversationProjectionAuthority {
            conversation_id: conversation_id.to_string(),
            epoch: self.epoch,
            revision: self.revision,
            lifecycle: self.lifecycle,
            delete_receipt_retention_floor_epoch: self.retention_floor_epoch,
        }
    }

    /// 计算下一个修订号；到达 `u64` 上限时返回错误而不是溢出。
    fn next_revision(&self, conversation_id: &str) -> Result<u64> {
        self.revision.checked_add(1).ok_or_else(|| {
            MemoryError::Unsupported(format!(
                "conversation projection revision exhausted: {conversation_id}"
            ))
            .into()
        })
    }

    /**
     * 裁剪已保留的删除回执，使其数量不超过 [`DELETE_RECEIPT_RETENTION`]；
     * 被淘汰的回执会把 `retention_floor_epoch` 上移到其 `deleted_epoch`。
     */
    fn retain_delete_receipts(&mut self) {
        while self.delete_receipts.len() > DELETE_RECEIPT_RETENTION {
            let oldest = self
                .delete_receipts
                .iter()
                .min_by_key(|(_, receipt)| receipt.deleted_epoch)
                .map(|(operation_id, receipt)| (operation_id.clone(), receipt.deleted_epoch));
            let Some((operation_id, deleted_epoch)) = oldest else {
                break;
            };
            self.delete_receipts.remove(&operation_id);
            self.retention_floor_epoch = self.retention_floor_epoch.max(deleted_epoch);
        }
    }
}

/**
 * 一次托管会话删除操作的回执：payload 摘要 + 发生删除时的 epoch，
 * 用于对重复的删除请求做幂等判断。
 */
#[derive(Debug, Clone, Serialize, Deserialize)]
struct FileDeleteReceipt {
    /// 触发本次删除的请求 payload 的摘要。
    payload_digest: String,
    /// 执行删除时所处的 epoch。
    deleted_epoch: u64,
}

/// 单个消息日志 generation 的持久化提交标记。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct MessageLogMeta {
    /// 当前生效的消息日志文件代号（generation），对应文件名中的数字后缀。
    generation: u64,
    /// 该 generation 文件中已提交（可安全读取）的字节数。
    committed_bytes: u64,
    /// 已提交消息的数量。
    message_count: usize,
    /// 已提交消息中的最大 id（如果有的话），用于恢复单调 id 计数器。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    max_message_id: Option<i64>,
}

impl MessageLogMeta {
    /// 构造一个空的日志元数据（generation 从 1 开始，尚无已提交内容）。
    fn empty() -> Self {
        Self {
            generation: 1,
            committed_bytes: 0,
            message_count: 0,
            max_message_id: None,
        }
    }
}

/**
 * 用于大小写不敏感的 Unicode 子串搜索的固定大小候选过滤器（布隆过滤器思路）。
 *
 * 它从不作为权威判定：命中的候选还需要针对会话标题和已提交消息再做一次真实校验。
 * 旧版清单没有此字段，因此会被保守地当作候选保留下来（即不做过滤）。
 */
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SearchFilter {
    /// 位图存储，每个 `u64` 表示 64 位。
    bits: Vec<u64>,
}

impl SearchFilter {
    /// 构造一个全零的空过滤器。
    fn empty() -> Self {
        Self {
            bits: vec![0; SEARCH_FILTER_WORDS],
        }
    }

    /// 根据会话标题和当前消息列表构建一个新的过滤器。
    fn from_conversation(conversation: &Conversation, messages: &[StoredMessage]) -> Self {
        let mut filter = Self::empty();
        if let Some(title) = conversation.title.as_deref() {
            filter.insert_text(title);
        }
        for content in messages
            .iter()
            .filter_map(|message| message.content.as_deref())
        {
            filter.insert_text(content);
        }
        filter
    }

    /// 把单条消息的内容并入过滤器（用于增量追加场景）。
    fn insert_message(&mut self, message: &StoredMessage) {
        if let Some(content) = message.content.as_deref() {
            self.insert_text(content);
        }
    }

    /// 把一段文本按 1~3 个字符宽度切出所有 n-gram 并写入过滤器。
    fn insert_text(&mut self, text: &str) {
        let normalized = text.to_lowercase();
        let chars = normalized.chars().collect::<Vec<_>>();
        for width in 1..=3 {
            for window in chars.windows(width) {
                let gram = window.iter().collect::<String>();
                self.insert_gram(&gram);
            }
        }
    }

    /// 把一个 n-gram 对应的比特位置 1。
    fn insert_gram(&mut self, gram: &str) {
        for bit in search_filter_bits(gram) {
            let word = bit / u64::BITS as usize;
            let offset = bit % u64::BITS as usize;
            if let Some(value) = self.bits.get_mut(word) {
                *value |= 1_u64 << offset;
            }
        }
    }

    /// 判断查询串是否*可能*出现在该会话中（假阳性允许，假阴性不允许）。
    fn might_contain(&self, query: &str) -> bool {
        if query.is_empty() {
            return true;
        }
        let normalized = query.to_lowercase();
        let chars = normalized.chars().collect::<Vec<_>>();
        if chars.is_empty() {
            return true;
        }
        let width = chars.len().min(3);
        chars.windows(width).all(|window| {
            let gram = window.iter().collect::<String>();
            search_filter_bits(&gram).into_iter().all(|bit| {
                let word = bit / u64::BITS as usize;
                let offset = bit % u64::BITS as usize;
                self.bits
                    .get(word)
                    .is_some_and(|value| value & (1_u64 << offset) != 0)
            })
        })
    }
}

/// 把一个 n-gram 哈希映射为过滤器位图中的两个比特位下标。
fn search_filter_bits(value: &str) -> [usize; 2] {
    let digest = Sha256::digest(value.as_bytes());
    let first = digest
        .get(..8)
        .and_then(|bytes| <[u8; 8]>::try_from(bytes).ok())
        .map(u64::from_le_bytes)
        .unwrap_or(0);
    let second = digest
        .get(8..16)
        .and_then(|bytes| <[u8; 8]>::try_from(bytes).ok())
        .map(u64::from_le_bytes)
        .unwrap_or(0);
    let bit_count = SEARCH_FILTER_WORDS.saturating_mul(u64::BITS as usize);
    [
        usize::try_from(first).unwrap_or(0) % bit_count,
        usize::try_from(second).unwrap_or(0) % bit_count,
    ]
}

/**
 * 持久化为 `_meta.json` 的单调递增 id 计数器，用来替代 SQLite 的自增主键。
 * 每新建一个会话/消息，`next_id` 都会递增，从而保证 id 在整个存储中唯一。
 */
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
struct StoreMeta {
    /// 下一个可分配的 id。
    next_id: i64,
}

impl StoreMeta {
    /// 分配并返回下一个 id（自增后返回新值）。
    fn take_id(&mut self) -> i64 {
        self.next_id = self.next_id.saturating_add(1);
        self.next_id
    }
}

/**
 * 内存缓存中的单条消息摘要信息，用于快速判断追加写入的前缀是否匹配，
 * 而无需重新读取/反序列化整段消息内容。
 */
#[derive(Debug)]
struct CachedMessage {
    /// 消息 id（如果已分配）。
    id: Option<i64>,
    /// 消息创建时间。
    created_at: String,
    /// 消息语义字段（角色、内容、附件等）的摘要，用于内容比对。
    semantic_digest: [u8; 32],
}

/**
 * 单个会话的内存缓存：当前消息日志状态及每条消息的摘要，
 * 用来加速 `save_messages` 的前缀匹配判断。
 */
#[derive(Debug)]
struct ConversationCache {
    /// 缓存所对应的消息日志元数据快照，用于判断缓存是否仍然有效。
    log: Option<MessageLogMeta>,
    /// 已缓存的消息摘要列表。
    messages: Vec<CachedMessage>,
}

/**
 * 文件后端的会话存储。
 *
 * 在同一个规范化（canonical）根路径下打开的多个句柄，共享同一个进程内权威状态，
 * 并对每个会话的操作保持顺序。不同会话之间使用一个进程级、有界的阻塞线程池。
 * 一个贯穿生命周期持有的旁路文件锁（sidecar lease）会拒绝另一个竞争进程接入，
 * 而不是允许出现未同步的并发文件写入。
 *
 * [`Self::new`] 是同步的初始化过程：获取进程锁、协调元数据、清理孤儿 generation
 * 文件。异步调用方应当在对延迟敏感的工作开始之前，或在一个阻塞的初始化任务中
 * 构造该存储。[`ConversationStore`] trait 的各个方法会把文件操作转移到阻塞线程执行。
 */
#[derive(Clone)]
pub struct FileConversationStore {
    /// 会话数据所在的根目录（`<base>/conversations`），已规范化。
    base: PathBuf,
    /// 进程内共享的权威状态（缓存、元数据、进程锁等）。
    authority: Arc<FileConversationAuthority>,
    /// 可选的持久化调用上下文（用于超时/截止时间检查）。
    persistence_call_context: Option<PersistenceCallContext>,
}

/// 同一根目录下所有 [`FileConversationStore`] 句柄共享的进程内权威状态。
struct FileConversationAuthority {
    /// 单调 id 计数器。
    meta: Mutex<StoreMeta>,
    /// 按会话 id 索引的内存缓存。
    conversations: Mutex<HashMap<String, ConversationCache>>,
    /// 用于串行化「扫描全部清单」与「单会话写入」之间关系的读写锁。
    scan_barrier: RwLock<()>,
    /// 测试专用：用于在搜索扫描到某个快照点时暂停，便于测试并发场景。
    #[cfg(test)]
    search_snapshot_hook: Mutex<Option<SearchSnapshotHook>>,
    /// 测试专用：置位后使下一次清单写入注入失败，用于测试错误路径。
    #[cfg(test)]
    fail_next_manifest_write: Mutex<bool>,
    /// 进程级独占文件锁，防止多个进程实例并发写入同一存储目录。
    _lease: ExclusiveFileLease,
}

/// 测试专用：搜索扫描的暂停/继续钩子。
#[cfg(test)]
struct SearchSnapshotHook {
    /// 通知观察者「已捕获到快照点」的单发信号。
    captured: tokio::sync::oneshot::Sender<()>,
    /// 等待观察者放行、恢复执行的接收端。
    release: std::sync::mpsc::Receiver<()>,
}

/**
 * 进程内的全局注册表：按规范化根路径复用同一个 [`FileConversationAuthority`]，
 * 使得指向同一目录的多个 store 句柄共享缓存与进程锁。
 */
fn file_conversation_registry() -> &'static Mutex<HashMap<PathBuf, Weak<FileConversationAuthority>>>
{
    // 规范化根路径 → 权威状态的弱引用；所有句柄释放后条目自然失效。
    static REGISTRY: OnceLock<Mutex<HashMap<PathBuf, Weak<FileConversationAuthority>>>> =
        OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

impl FileConversationStore {
    /**
     * 创建一个以 `base/conversations/` 为根目录的文件后端会话存储。
     *
     * 这是一个同步的初始化过程：获取进程锁并扫描已有文件。
     */
    pub fn new(base: impl AsRef<Path>) -> Result<Self> {
        let base = base.as_ref().join("conversations");
        std::fs::create_dir_all(&base)
            .map_err(|e| MemoryError::IoError(format!("create conversations dir: {e}")))?;
        let base = std::fs::canonicalize(&base).map_err(|error| {
            MemoryError::IoError(format!("canonicalize conversations dir: {error}"))
        })?;
        let mut registry = file_conversation_registry().lock().map_err(|error| {
            MemoryError::IoError(format!("FileConversationStore registry poisoned: {error}"))
        })?;
        if let Some(authority) = registry.get(&base).and_then(Weak::upgrade) {
            return Ok(Self {
                base,
                authority,
                persistence_call_context: None,
            });
        }
        let lease = try_exclusive_file_lease(&base).map_err(|error| {
            MemoryError::IoError(format!("acquire FileConversationStore lease: {error}"))
        })?;
        let meta = Self::read_meta(&base)?;
        Self::cleanup_orphaned_message_logs(&base)?;
        let authority = Arc::new(FileConversationAuthority {
            meta: Mutex::new(meta),
            conversations: Mutex::new(HashMap::new()),
            scan_barrier: RwLock::new(()),
            #[cfg(test)]
            search_snapshot_hook: Mutex::new(None),
            #[cfg(test)]
            fail_next_manifest_write: Mutex::new(false),
            _lease: lease,
        });
        registry.insert(base.clone(), Arc::downgrade(&authority));
        Ok(Self {
            base,
            authority,
            persistence_call_context: None,
        })
    }

    /// 克隆出一个携带指定持久化调用上下文的句柄（用于截止时间检查）。
    fn with_persistence_call_context(&self, context: PersistenceCallContext) -> Self {
        Self {
            base: self.base.clone(),
            authority: Arc::clone(&self.authority),
            persistence_call_context: Some(context),
        }
    }

    /// 若当前持有调用上下文，则检查其是否已过期。
    fn ensure_persistence_call_not_expired(&self) -> Result<()> {
        if let Some(context) = self.persistence_call_context {
            context.ensure_not_expired()?;
        }
        Ok(())
    }

    /// 为某个会话 id 构造对应的阻塞文件操作作用域标识，用于串行化同一会话的并发操作。
    fn conversation_scope(conversation_id: impl Into<String>) -> BlockingFileOperationScope {
        let conversation_id = conversation_id.into();
        BlockingFileOperationScope::Entity(echo_core::utils::fs::encode_utf8_path_identity(
            &conversation_id,
        ))
    }

    /// 测试专用：安装一个搜索快照钩子，使搜索扫描在完成快照采集后暂停等待放行。
    #[cfg(test)]
    fn install_search_snapshot_hook(
        &self,
        captured: tokio::sync::oneshot::Sender<()>,
        release: std::sync::mpsc::Receiver<()>,
    ) -> Result<()> {
        let mut hook = self.authority.search_snapshot_hook.lock().map_err(poison)?;
        if hook.is_some() {
            return Err(
                MemoryError::IoError("search snapshot hook is already installed".into()).into(),
            );
        }
        *hook = Some(SearchSnapshotHook { captured, release });
        Ok(())
    }

    /// 测试专用：若已安装钩子，通知观察者快照已捕获，并阻塞等待其放行（最多 2 秒）。
    #[cfg(test)]
    fn pause_search_after_snapshot(&self) -> Result<()> {
        let hook = self
            .authority
            .search_snapshot_hook
            .lock()
            .map_err(poison)?
            .take();
        let Some(hook) = hook else {
            return Ok(());
        };
        hook.captured
            .send(())
            .map_err(|_| MemoryError::IoError("search snapshot observer was dropped".into()))?;
        hook.release
            .recv_timeout(std::time::Duration::from_secs(2))
            .map_err(|error| MemoryError::IoError(format!("release search snapshot: {error}")))?;
        Ok(())
    }

    /// 测试专用：置位标志，使下一次清单写入注入一次失败（用于测试错误恢复路径）。
    #[cfg(test)]
    fn fail_next_manifest_write(&self) -> Result<()> {
        let mut fail = self
            .authority
            .fail_next_manifest_write
            .lock()
            .map_err(poison)?;
        *fail = true;
        Ok(())
    }

    /// 把给定的同步操作放到进程级阻塞线程池上执行，并按 `scope` 与其他操作排队/串行化。
    fn run_blocking<'a, T, F>(
        &'a self,
        scope: BlockingFileOperationScope,
        operation: F,
    ) -> BoxFut<'a, T>
    where
        T: Send + 'static,
        F: FnOnce(Self) -> Result<T> + Send + 'static,
    {
        let store = self.clone();
        Box::pin(async move {
            let key =
                BlockingFileOperationKey::new("conversation-store", store.base.clone(), scope);
            run_keyed_file_operation(key, move || {
                store.ensure_persistence_call_not_expired()?;
                operation(store)
            })
            .await
            .map_err(|error| {
                MemoryError::IoError(format!(
                    "FileConversationStore blocking operation failed: {error}"
                ))
            })?
        })
    }

    /// 计算某个会话清单文件的路径。
    fn conv_path(&self, conversation_id: &str) -> Result<PathBuf> {
        let safe = safe_segment(conversation_id)?;
        Ok(self.base.join(format!("{safe}.json")))
    }

    /// 计算某个会话在指定 generation 下的消息日志文件路径。
    fn message_log_path(&self, conversation_id: &str, generation: u64) -> Result<PathBuf> {
        let safe = safe_segment(conversation_id)?;
        Ok(self
            .base
            .join(format!("{safe}.messages.{generation}.jsonl")))
    }

    /// 计算 `_meta.json` 的路径。
    fn meta_path(base: &Path) -> PathBuf {
        base.join("_meta.json")
    }

    /**
     * 读取 `_meta.json`，并与已持久化的记录做协调（reconcile）。
     *
     * 崩溃场景下，一次记录重命名可能恰好在对应的 meta 写入之前变为持久化状态。
     * 扫描现有记录可以防止下一次启动的进程复用一个已经被持久化过的
     * 会话/消息 id。
     */
    fn read_meta(base: &Path) -> Result<StoreMeta> {
        let mut meta = match std::fs::read_to_string(Self::meta_path(base)) {
            Ok(s) => {
                let meta: StoreMeta = serde_json::from_str(&s).map_err(|e| {
                    MemoryError::SerializationError(format!("parse _meta.json: {e}"))
                })?;
                meta
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => StoreMeta::default(),
            Err(e) => return Err(MemoryError::IoError(format!("read _meta.json: {e}")).into()),
        };

        let entries =
            std::fs::read_dir(base).map_err(|e| MemoryError::IoError(format!("readdir: {e}")))?;
        for entry in entries {
            let entry = entry.map_err(|e| MemoryError::IoError(format!("readdir entry: {e}")))?;
            let path = entry.path();
            if path.extension().and_then(|value| value.to_str()) != Some("json")
                || path.file_name().and_then(|value| value.to_str()) == Some("_meta.json")
            {
                continue;
            }
            let record = Self::read_manifest_path(&path)?;
            meta.next_id = meta.next_id.max(record.conversation.id);
            let max_message_id = record
                .message_log
                .as_ref()
                .and_then(|log| log.max_message_id)
                .or_else(|| {
                    record
                        .messages
                        .iter()
                        .filter_map(|message| message.id)
                        .max()
                });
            if let Some(message_id) = max_message_id {
                meta.next_id = meta.next_id.max(message_id);
            }
        }
        Ok(meta)
    }

    /**
     * 扫描目录，删除不再被任何清单引用的消息日志文件（例如替换操作中被淘汰的旧
     * generation，在崩溃后未能及时清理的情况）。
     */
    fn cleanup_orphaned_message_logs(base: &Path) -> Result<()> {
        let mut referenced = HashSet::new();
        let entries =
            std::fs::read_dir(base).map_err(|e| MemoryError::IoError(format!("readdir: {e}")))?;
        for entry in entries {
            let entry = entry.map_err(|e| MemoryError::IoError(format!("readdir entry: {e}")))?;
            let path = entry.path();
            if path.extension().and_then(|value| value.to_str()) != Some("json")
                || path.file_name().and_then(|value| value.to_str()) == Some("_meta.json")
            {
                continue;
            }
            let record = Self::read_manifest_path(&path)?;
            if let Some(log) = record.message_log.as_ref() {
                let safe = safe_segment(&record.conversation.conversation_id)?;
                referenced.insert(base.join(format!("{safe}.messages.{}.jsonl", log.generation)));
            }
        }

        let entries =
            std::fs::read_dir(base).map_err(|e| MemoryError::IoError(format!("readdir: {e}")))?;
        for entry in entries {
            let entry = entry.map_err(|e| MemoryError::IoError(format!("readdir entry: {e}")))?;
            let path = entry.path();
            let Some(file_name) = path.file_name().and_then(|value| value.to_str()) else {
                continue;
            };
            if referenced.contains(&path) || !is_message_log_file_name(file_name) {
                continue;
            }
            let metadata = std::fs::symlink_metadata(&path).map_err(|error| {
                MemoryError::IoError(format!(
                    "stat orphan message log {}: {error}",
                    path.display()
                ))
            })?;
            if !metadata.file_type().is_file() {
                continue;
            }
            std::fs::remove_file(&path).map_err(|error| {
                MemoryError::IoError(format!(
                    "remove orphan message log {}: {error}",
                    path.display()
                ))
            })?;
        }
        Ok(())
    }

    /// 把 id 计数器原子写入 `_meta.json`。
    fn persist_meta(&self, meta: &StoreMeta) -> Result<()> {
        let json = serde_json::to_string(meta)
            .map_err(|e| MemoryError::SerializationError(format!("serialize meta: {e}")))?;
        echo_core::utils::fs::atomic_write(&Self::meta_path(&self.base), json.as_bytes())
            .map_err(|e| MemoryError::IoError(format!("write meta: {e}")))?;
        Ok(())
    }

    /**
     * 解析清单 JSON 内容，并校验其内部一致性（消息日志/内嵌消息互斥、搜索过滤器
     * 长度、投影权威状态的各项不变量）。
     */
    fn parse_manifest(path: &Path, content: &str) -> Result<ConversationRecord> {
        let record: ConversationRecord = serde_json::from_str(content).map_err(|e| {
            MemoryError::SerializationError(format!("parse {}: {e}", path.display()))
        })?;
        if record.message_log.is_some() && !record.messages.is_empty() {
            return Err(MemoryError::SerializationError(format!(
                "manifest {} contains both embedded and logged messages",
                path.display()
            ))
            .into());
        }
        if record
            .search_filter
            .as_ref()
            .is_some_and(|filter| filter.bits.len() != SEARCH_FILTER_WORDS)
        {
            return Err(MemoryError::SerializationError(format!(
                "manifest {} has an invalid search filter",
                path.display()
            ))
            .into());
        }
        if let Some(projection) = record.projection.as_ref() {
            let invalid_authority = projection.epoch == 0
                || projection.epoch > MAX_PROJECTION_EPOCH
                || projection.revision == 0
                || projection.retention_floor_epoch > projection.epoch;
            let invalid_operations =
                projection
                    .applied_operations
                    .iter()
                    .any(|(operation_id, payload_digest)| {
                        operation_id.trim().is_empty() || payload_digest.trim().is_empty()
                    });
            let invalid_ordinals = projection.ordinals.iter().any(|(generation_id, ordinals)| {
                generation_id.trim().is_empty()
                    || ordinals.values().any(|digest| digest.trim().is_empty())
            });
            let invalid_delete_receipts =
                projection
                    .delete_receipts
                    .iter()
                    .any(|(operation_id, receipt)| {
                        operation_id.trim().is_empty()
                            || receipt.payload_digest.trim().is_empty()
                            || receipt.deleted_epoch == 0
                            || receipt.deleted_epoch > projection.epoch
                            || receipt.deleted_epoch <= projection.retention_floor_epoch
                    });
            if invalid_authority
                || invalid_operations
                || invalid_ordinals
                || invalid_delete_receipts
            {
                return Err(MemoryError::SerializationError(format!(
                    "manifest {} has invalid conversation projection authority",
                    path.display()
                ))
                .into());
            }
        }
        Ok(record)
    }

    /// 从给定路径读取并解析一份清单文件。
    fn read_manifest_path(path: &Path) -> Result<ConversationRecord> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| MemoryError::IoError(format!("read {}: {e}", path.display())))?;
        Self::parse_manifest(path, &content)
    }

    /// 读取单个会话的清单。文件不存在 → `Ok(None)`；内容损坏 → `Err`。
    fn read_manifest(&self, conversation_id: &str) -> Result<Option<ConversationRecord>> {
        let path = self.conv_path(conversation_id)?;
        match std::fs::read_to_string(&path) {
            Ok(content) => Self::parse_manifest(&path, &content).map(Some),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(MemoryError::IoError(format!(
                "read conversation {conversation_id}: {error}"
            ))
            .into()),
        }
    }

    /// 读取单个会话及其已提交的消息。
    fn read_record(&self, conversation_id: &str) -> Result<Option<ConversationRecord>> {
        let mut record = match self.read_manifest(conversation_id)? {
            Some(record) => record,
            None => return Ok(None),
        };
        if let Some(log) = record.message_log.as_ref() {
            record.messages = self.read_message_log(conversation_id, log)?;
        }
        Ok(Some(record))
    }

    /**
     * 按照清单记录的已提交字节边界，读取并解析该会话的消息日志文件，
     * 忽略未提交的尾部数据；发现截断或格式不一致时返回错误。
     */
    fn read_message_log(
        &self,
        conversation_id: &str,
        log: &MessageLogMeta,
    ) -> Result<Vec<StoredMessage>> {
        let path = self.message_log_path(conversation_id, log.generation)?;
        let file = std::fs::File::open(&path)
            .map_err(|e| MemoryError::IoError(format!("read {}: {e}", path.display())))?;
        let actual_bytes = file
            .metadata()
            .map_err(|e| MemoryError::IoError(format!("stat {}: {e}", path.display())))?
            .len();
        if actual_bytes < log.committed_bytes {
            return Err(MemoryError::SerializationError(format!(
                "message log {} is truncated: committed {} bytes, found {actual_bytes}",
                path.display(),
                log.committed_bytes
            ))
            .into());
        }

        let committed_capacity = usize::try_from(log.committed_bytes).map_err(|error| {
            MemoryError::Unsupported(format!("message log length is unsupported: {error}"))
        })?;
        let mut committed = Vec::with_capacity(committed_capacity);
        file.take(log.committed_bytes)
            .read_to_end(&mut committed)
            .map_err(|error| MemoryError::IoError(format!("read {}: {error}", path.display())))?;
        if committed.len() != committed_capacity
            || committed.last().is_some_and(|byte| *byte != b'\n')
        {
            return Err(MemoryError::SerializationError(format!(
                "message log {} has an incomplete committed record",
                path.display()
            ))
            .into());
        }

        let mut messages = Vec::with_capacity(log.message_count);
        let reader = BufReader::new(committed.as_slice());
        for (line_index, line) in reader.lines().enumerate() {
            let line = line.map_err(|e| {
                MemoryError::SerializationError(format!(
                    "read message log {} line {}: {e}",
                    path.display(),
                    line_index.saturating_add(1)
                ))
            })?;
            if line.is_empty() {
                return Err(MemoryError::SerializationError(format!(
                    "message log {} contains an empty committed line at {}",
                    path.display(),
                    line_index.saturating_add(1)
                ))
                .into());
            }
            messages.push(serde_json::from_str(&line).map_err(|e| {
                MemoryError::SerializationError(format!(
                    "parse message log {} line {}: {e}",
                    path.display(),
                    line_index.saturating_add(1)
                ))
            })?);
        }
        if messages.len() != log.message_count {
            return Err(MemoryError::SerializationError(format!(
                "message log {} count mismatch: committed {}, parsed {}",
                path.display(),
                log.message_count,
                messages.len()
            ))
            .into());
        }
        Ok(messages)
    }

    /**
     * 原子写入会话清单；若已采用日志式消息存储，写入前会清空内嵌的 `messages`
     * 字段（避免同时持有两种存储形式）。
     */
    fn write_manifest(&self, record: &ConversationRecord) -> Result<()> {
        #[cfg(test)]
        {
            let mut fail = self
                .authority
                .fail_next_manifest_write
                .lock()
                .map_err(poison)?;
            if *fail {
                *fail = false;
                return Err(MemoryError::IoError(
                    "injected conversation manifest publish failure".to_string(),
                )
                .into());
            }
        }
        let mut manifest = record.clone();
        if manifest.message_log.is_some() {
            manifest.messages.clear();
        }
        let json = serde_json::to_string_pretty(&manifest)
            .map_err(|e| MemoryError::SerializationError(format!("serialize conversation: {e}")))?;
        let path = self.conv_path(&record.conversation.conversation_id)?;
        echo_core::utils::fs::atomic_write(&path, json.as_bytes())
            .map_err(|e| MemoryError::IoError(format!("write conversation: {e}")))?;
        Ok(())
    }

    /// 把消息列表序列化为 JSON Lines 字节流（每条消息一行，以 `\n` 分隔）。
    fn serialize_message_log(messages: &[StoredMessage]) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        for message in messages {
            serde_json::to_writer(&mut bytes, message).map_err(|error| {
                MemoryError::SerializationError(format!("serialize conversation message: {error}"))
            })?;
            bytes.push(b'\n');
        }
        Ok(bytes)
    }

    /**
     * 计算一条消息「语义字段」（角色、内容、附件、工具调用等）的摘要，
     * 用于判断两条消息在内容上是否等价，而不考虑 id/时间戳等元数据差异。
     */
    fn semantic_digest(message: &StoredMessage) -> Result<[u8; 32]> {
        let semantic_fields = (
            &message.role,
            &message.content,
            &message.attachments_json,
            &message.tool_calls_json,
            &message.tool_result_json,
        );
        let encoded = serde_json::to_vec(&semantic_fields).map_err(|error| {
            MemoryError::SerializationError(format!("fingerprint conversation message: {error}"))
        })?;
        Ok(Sha256::digest(encoded).into())
    }

    /// 根据已加载的会话记录构建内存缓存（计算每条消息的语义摘要）。
    fn cache_from_record(record: &ConversationRecord) -> Result<ConversationCache> {
        let messages = record
            .messages
            .iter()
            .map(|message| {
                Ok(CachedMessage {
                    id: message.id,
                    created_at: message.created_at.clone(),
                    semantic_digest: Self::semantic_digest(message)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(ConversationCache {
            log: record.message_log.clone(),
            messages,
        })
    }

    /**
     * 判断缓存中的已提交消息是否构成传入消息列表的前缀：
     * 语义摘要必须一致；若传入消息带有显式 id，则 id 与创建时间也须一致。
     */
    fn prefix_matches(
        cache: &ConversationCache,
        messages: &[StoredMessage],
        digests: &[[u8; 32]],
    ) -> bool {
        /*
         * 规范的运行时投影有意携带 `id = None`，并在每次定稿时重新生成
         * `created_at`，因此它们的时间戳变化不算作更新。显式 id 表示这是一条
         * 导入的已持久化记录；对于这种形态，时间戳变化属于替换语义的一部分。
         */
        messages.len() >= cache.messages.len()
            && cache
                .messages
                .iter()
                .zip(messages.iter())
                .zip(digests.iter())
                .all(|((cached, incoming), digest)| {
                    cached.semantic_digest == *digest
                        && incoming.id.is_none_or(|id| {
                            cached.id == Some(id) && cached.created_at == incoming.created_at
                        })
                })
    }

    /// 为单条消息绑定会话 id 并分配消息 id：已有 id 时推进计数器，否则从计数器取新 id。
    fn assign_message(
        conversation_id: &str,
        incoming: &StoredMessage,
        meta: &mut StoreMeta,
    ) -> StoredMessage {
        let mut assigned = incoming.clone();
        assigned.conversation_id = conversation_id.to_string();
        match assigned.id {
            Some(id) => meta.next_id = meta.next_id.max(id),
            None => assigned.id = Some(meta.take_id()),
        }
        assigned
    }

    /// 在持有 id 计数器锁的情况下，为一批消息统一分配 id。
    fn reserve_messages(
        &self,
        conversation_id: &str,
        incoming: &[StoredMessage],
    ) -> Result<Vec<StoredMessage>> {
        let mut meta = self.authority.meta.lock().map_err(poison)?;
        Ok(incoming
            .iter()
            .map(|message| Self::assign_message(conversation_id, message, &mut meta))
            .collect())
    }

    /// 把当前内存中的 id 计数器持久化到 `_meta.json`。
    fn persist_current_meta(&self) -> Result<()> {
        let meta = self.authority.meta.lock().map_err(poison)?;
        self.persist_meta(&meta)
    }

    /// 用给定的语义摘要为一条消息构造缓存条目。
    fn cached_message(message: &StoredMessage, digest: [u8; 32]) -> CachedMessage {
        CachedMessage {
            id: message.id,
            created_at: message.created_at.clone(),
            semantic_digest: digest,
        }
    }

    /**
     * 为追加写入准备消息日志文件：校验文件未被截断，
     * 并截掉超出已提交边界的未提交尾部，返回日志文件路径。
     */
    fn prepare_log_for_append(
        &self,
        conversation_id: &str,
        log: &MessageLogMeta,
    ) -> Result<PathBuf> {
        let path = self.message_log_path(conversation_id, log.generation)?;
        let actual_bytes = std::fs::metadata(&path)
            .map_err(|error| {
                MemoryError::IoError(format!("stat message log {}: {error}", path.display()))
            })?
            .len();
        if actual_bytes < log.committed_bytes {
            return Err(MemoryError::SerializationError(format!(
                "message log {} is truncated: committed {} bytes, found {actual_bytes}",
                path.display(),
                log.committed_bytes
            ))
            .into());
        }
        if actual_bytes > log.committed_bytes {
            echo_core::utils::fs::truncate_existing(
                &path,
                log.committed_bytes,
                echo_core::utils::fs::FileDurability::SyncData,
            )
            .map_err(|error| {
                MemoryError::IoError(format!(
                    "remove uncommitted message-log tail {}: {error}",
                    path.display()
                ))
            })?;
        }
        Ok(path)
    }

    /// 计算下一个消息日志 generation；无日志时从 1 开始，溢出时返回错误。
    fn next_generation(log: Option<&MessageLogMeta>) -> Result<u64> {
        match log {
            Some(log) => log.generation.checked_add(1).ok_or_else(|| {
                MemoryError::Unsupported("conversation message-log generation exhausted".into())
                    .into()
            }),
            None => Ok(1),
        }
    }

    /// 尽力删除已被替换的旧 generation 日志文件；失败时仅记录告警。
    fn remove_replaced_log(&self, conversation_id: &str, replaced: Option<&MessageLogMeta>) {
        let Some(replaced) = replaced else {
            return;
        };
        let Ok(path) = self.message_log_path(conversation_id, replaced.generation) else {
            return;
        };
        if let Err(error) = std::fs::remove_file(&path)
            && error.kind() != std::io::ErrorKind::NotFound
        {
            tracing::warn!(
                path = %path.display(),
                error = %error,
                "failed to remove replaced conversation message log"
            );
        }
    }

    /**
     * 用给定消息整体替换会话的消息日志：写入新 generation、持久化 id 计数器、
     * 提交清单，然后删除旧 generation 并刷新内存缓存。
     */
    fn replace_record_messages(
        &self,
        record: &mut ConversationRecord,
        messages: &[StoredMessage],
    ) -> Result<()> {
        let conversation_id = record.conversation.conversation_id.clone();
        let assigned = self.reserve_messages(&conversation_id, messages)?;
        let generation = Self::next_generation(record.message_log.as_ref())?;
        let bytes = Self::serialize_message_log(&assigned)?;
        let path = self.message_log_path(&conversation_id, generation)?;
        echo_core::utils::fs::atomic_write(&path, &bytes).map_err(|error| {
            MemoryError::IoError(format!(
                "replace conversation message log {}: {error}",
                path.display()
            ))
        })?;
        let committed_bytes = u64::try_from(bytes.len()).map_err(|error| {
            MemoryError::Unsupported(format!("message log length is unsupported: {error}"))
        })?;
        let log = MessageLogMeta {
            generation,
            committed_bytes,
            message_count: assigned.len(),
            max_message_id: assigned.iter().filter_map(|message| message.id).max(),
        };
        let replaced = record.message_log.clone();
        record.messages.clear();
        record.message_log = Some(log.clone());
        record.search_filter = Some(SearchFilter::from_conversation(
            &record.conversation,
            &assigned,
        ));
        self.persist_current_meta()?;
        self.write_manifest(record)?;
        self.remove_replaced_log(&conversation_id, replaced.as_ref());
        self.authority.conversations.lock().map_err(poison)?.insert(
            conversation_id,
            ConversationCache {
                log: Some(log),
                messages: assigned
                    .iter()
                    .map(|message| {
                        Ok(Self::cached_message(
                            message,
                            Self::semantic_digest(message)?,
                        ))
                    })
                    .collect::<Result<Vec<_>>>()?,
            },
        );
        Ok(())
    }

    /// 构造「托管会话必须通过投影接口操作」的错误。
    fn managed_conversation_error(conversation_id: &str) -> echo_core::error::ReactError {
        MemoryError::ManagedConversationRequiresProjection(conversation_id.to_string()).into()
    }

    /// 根据转录投影批次构造应用回执。
    fn apply_receipt(
        batch: &TranscriptProjectionBatch,
        authority: ConversationProjectionAuthority,
        status: TranscriptProjectionApplyStatus,
    ) -> TranscriptProjectionApplyReceipt {
        TranscriptProjectionApplyReceipt {
            operation_id: batch.operation_id.clone(),
            payload_digest: batch.payload_digest.clone(),
            authority,
            status,
        }
    }

    /// 根据托管导入请求构造应用回执。
    fn import_receipt(
        request: &ManagedConversationImport,
        authority: ConversationProjectionAuthority,
        status: TranscriptProjectionApplyStatus,
    ) -> TranscriptProjectionApplyReceipt {
        TranscriptProjectionApplyReceipt {
            operation_id: request.operation_id.clone(),
            payload_digest: request.payload_digest.clone(),
            authority,
            status,
        }
    }

    /// 同步创建一个非托管会话（不带投影状态）。
    fn create_conversation_sync(&self, conv: NewConversation) -> Result<Conversation> {
        self.create_conversation_sync_with_projection(conv, None)
    }

    /**
     * 同步创建会话，可选附带投影状态：分配会话 id、创建空的第 1 代消息日志、
     * 持久化 id 计数器并提交清单。会话已存在时返回错误。
     */
    fn create_conversation_sync_with_projection(
        &self,
        conv: NewConversation,
        projection: Option<FileProjectionState>,
    ) -> Result<Conversation> {
        if self.read_manifest(&conv.conversation_id)?.is_some() {
            return Err(MemoryError::IoError(format!(
                "conversation already exists: {}",
                conv.conversation_id
            ))
            .into());
        }
        let id = {
            let mut meta = self.authority.meta.lock().map_err(poison)?;
            meta.take_id()
        };
        let now = now_rfc3339();
        let conversation = Conversation {
            id,
            conversation_id: conv.conversation_id,
            user_id: conv.user_id,
            agent_type: conv.agent_type,
            title: conv.title,
            summary: None,
            compressed_before_id: None,
            created_at: now.clone(),
            updated_at: now,
        };
        let record = ConversationRecord {
            conversation: conversation.clone(),
            messages: Vec::new(),
            message_log: Some(MessageLogMeta::empty()),
            search_filter: Some(SearchFilter::from_conversation(&conversation, &[])),
            projection,
        };
        let log_path = self.message_log_path(&record.conversation.conversation_id, 1)?;
        echo_core::utils::fs::atomic_write(&log_path, &[])
            .map_err(|error| MemoryError::IoError(format!("create message log: {error}")))?;
        self.persist_current_meta()?;
        self.write_manifest(&record)?;
        Ok(conversation)
    }

    /**
     * 枚举磁盘上的所有会话清单，但不读取消息正文。
     *
     * 任意一条记录损坏都会作为错误返回（以前的行为是静默跳过，从而掩盖数据丢失）。
     */
    fn read_all_manifests(&self) -> Result<Vec<ConversationRecord>> {
        let mut records = Vec::new();
        let entries = std::fs::read_dir(&self.base)
            .map_err(|e| MemoryError::IoError(format!("readdir: {e}")))?;
        for entry in entries {
            let entry = entry.map_err(|e| MemoryError::IoError(format!("readdir entry: {e}")))?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("json") {
                continue;
            }
            // 跳过 meta 文件。
            if path
                .file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n == "_meta.json")
            {
                continue;
            }
            records.push(Self::read_manifest_path(&path)?);
        }
        Ok(records)
    }
}

/// 把锁中毒错误转换为 `MemoryError::IoError`。
fn poison<T>(_: std::sync::PoisonError<T>) -> MemoryError {
    MemoryError::IoError("store lock poisoned".into())
}

impl ConversationStore for FileConversationStore {
    /// 声明本存储支持原子投影协议 `AtomicV1`。
    fn projection_capability(&self) -> ConversationProjectionCapability {
        ConversationProjectionCapability::AtomicV1
    }

    /// 声明本存储支持绝对截止时间的持久化调用上下文。
    fn persistence_call_capability(&self) -> PersistenceCallCapability {
        PersistenceCallCapability::AbsoluteDeadlineV1
    }

    /// 带调用上下文的 [`Self::ensure_projection_epoch`]：先检查截止时间再执行。
    fn ensure_projection_epoch_with_context<'a>(
        &'a self,
        context: PersistenceCallContext,
        request: EnsureConversationProjectionRequest,
    ) -> BoxFut<'a, ConversationProjectionEpochReceipt> {
        let store = self.with_persistence_call_context(context);
        Box::pin(async move {
            context.ensure_not_expired()?;
            store.ensure_projection_epoch(request).await
        })
    }

    /// 带调用上下文的 [`Self::get_projection_authority`]：先检查截止时间再执行。
    fn get_projection_authority_with_context<'a>(
        &'a self,
        context: PersistenceCallContext,
        conversation_id: &'a str,
    ) -> BoxFut<'a, Option<ConversationProjectionAuthority>> {
        let store = self.with_persistence_call_context(context);
        Box::pin(async move {
            context.ensure_not_expired()?;
            store.get_projection_authority(conversation_id).await
        })
    }

    /// 带调用上下文的 [`Self::apply_transcript_projection`]：先检查截止时间再执行。
    fn apply_transcript_projection_with_context<'a>(
        &'a self,
        context: PersistenceCallContext,
        batch: TranscriptProjectionBatch,
    ) -> BoxFut<'a, TranscriptProjectionApplyReceipt> {
        let store = self.with_persistence_call_context(context);
        Box::pin(async move {
            context.ensure_not_expired()?;
            store.apply_transcript_projection(batch).await
        })
    }

    /// 带调用上下文的 [`Self::import_managed_messages`]：先检查截止时间再执行。
    fn import_managed_messages_with_context<'a>(
        &'a self,
        context: PersistenceCallContext,
        request: ManagedConversationImport,
    ) -> BoxFut<'a, TranscriptProjectionApplyReceipt> {
        let store = self.with_persistence_call_context(context);
        Box::pin(async move {
            context.ensure_not_expired()?;
            store.import_managed_messages(request).await
        })
    }

    /// 带调用上下文的 [`Self::update_managed_conversation`]：先检查截止时间再执行。
    fn update_managed_conversation_with_context<'a>(
        &'a self,
        context: PersistenceCallContext,
        request: ManagedConversationMetadataUpdate,
    ) -> BoxFut<'a, ManagedConversationMetadataUpdateReceipt> {
        let store = self.with_persistence_call_context(context);
        Box::pin(async move {
            context.ensure_not_expired()?;
            store.update_managed_conversation(request).await
        })
    }

    /// 带调用上下文的 [`Self::delete_managed_conversation`]：先检查截止时间再执行。
    fn delete_managed_conversation_with_context<'a>(
        &'a self,
        context: PersistenceCallContext,
        request: ManagedConversationDelete,
    ) -> BoxFut<'a, ManagedConversationDeleteReceipt> {
        let store = self.with_persistence_call_context(context);
        Box::pin(async move {
            context.ensure_not_expired()?;
            store.delete_managed_conversation(request).await
        })
    }

    /// 创建一个非托管会话；若同 id 的托管会话已存在则拒绝。
    fn create_conversation<'a>(&'a self, conv: NewConversation) -> BoxFut<'a, Conversation> {
        let scope = Self::conversation_scope(conv.conversation_id.clone());
        self.run_blocking(scope, move |store| {
            let _conversation = store.authority.scan_barrier.read().map_err(poison)?;
            if store
                .read_manifest(&conv.conversation_id)?
                .is_some_and(|record| record.projection.is_some())
            {
                return Err(Self::managed_conversation_error(&conv.conversation_id));
            }
            store.create_conversation_sync(conv)
        })
    }

    /// 读取会话元数据；已删除的托管会话视为不存在。
    fn get_conversation<'a>(
        &'a self,
        conversation_id: &'a str,
    ) -> BoxFut<'a, Option<Conversation>> {
        let conversation_id = conversation_id.to_string();
        self.run_blocking(
            Self::conversation_scope(conversation_id.clone()),
            move |store| {
                let _conversation = store.authority.scan_barrier.read().map_err(poison)?;
                Ok(store
                    .read_manifest(&conversation_id)?
                    .filter(|record| {
                        record.projection.as_ref().is_none_or(|projection| {
                            projection.lifecycle == ConversationProjectionLifecycle::Live
                        })
                    })
                    .map(|record| record.conversation))
            },
        )
    }

    /// 按过滤条件列出会话，按 `updated_at` 倒序并应用 offset/limit。
    fn list_conversations<'a>(
        &'a self,
        filter: ConversationFilter,
    ) -> BoxFut<'a, Vec<ConversationMeta>> {
        self.run_blocking(
            BlockingFileOperationScope::Collection("list".to_string()),
            move |store| {
                let _scan = store.authority.scan_barrier.write().map_err(poison)?;
                let mut metas: Vec<ConversationMeta> = store
                    .read_all_manifests()?
                    .into_iter()
                    .filter(|record| {
                        record.projection.as_ref().is_none_or(|projection| {
                            projection.lifecycle == ConversationProjectionLifecycle::Live
                        })
                    })
                    .filter(|r| {
                        filter
                            .user_id
                            .as_deref()
                            .is_none_or(|u| r.conversation.user_id == u)
                    })
                    .filter(|r| {
                        filter
                            .agent_type
                            .as_deref()
                            .is_none_or(|a| r.conversation.agent_type.as_deref() == Some(a))
                    })
                    .map(|r| {
                        let message_count = r
                            .message_log
                            .as_ref()
                            .map_or(r.messages.len(), |log| log.message_count);
                        ConversationMeta {
                            id: r.conversation.id,
                            conversation_id: r.conversation.conversation_id,
                            user_id: r.conversation.user_id,
                            title: r.conversation.title,
                            message_count,
                            created_at: r.conversation.created_at,
                            updated_at: r.conversation.updated_at,
                        }
                    })
                    .collect();
                // 等价于 ORDER BY updated_at DESC。
                metas.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
                // 先 OFFSET 再 LIMIT，避免直接按下标访问集合。
                let offset = filter.offset.unwrap_or(0);
                let limit = filter.limit.unwrap_or(usize::MAX);
                Ok(metas.into_iter().skip(offset).take(limit).collect())
            },
        )
    }

    /// 更新非托管会话的标题/摘要等元数据；会话不存在时静默成功。
    fn update_conversation<'a>(
        &'a self,
        conversation_id: &'a str,
        title: Option<&'a str>,
        summary: Option<&'a str>,
        compressed_before_id: Option<i64>,
    ) -> BoxFut<'a, ()> {
        let conversation_id = conversation_id.to_string();
        let title = title.map(str::to_string);
        let summary = summary.map(str::to_string);
        self.run_blocking(
            Self::conversation_scope(conversation_id.clone()),
            move |store| {
                let _conversation = store.authority.scan_barrier.read().map_err(poison)?;
                let mut record = match store.read_manifest(&conversation_id)? {
                    Some(r) => r,
                    None => return Ok(()), // 与 SQL UPDATE 命中 0 行时的行为一致。
                };
                if record.projection.is_some() {
                    return Err(Self::managed_conversation_error(&conversation_id));
                }
                if title.is_some() || summary.is_some() || compressed_before_id.is_some() {
                    if let Some(t) = title {
                        record.conversation.title = Some(t.clone());
                        if let Some(filter) = record.search_filter.as_mut() {
                            filter.insert_text(&t);
                        }
                    }
                    if let Some(s) = summary {
                        record.conversation.summary = Some(s);
                    }
                    if let Some(cbid) = compressed_before_id {
                        record.conversation.compressed_before_id = Some(cbid);
                    }
                    record.conversation.updated_at = now_rfc3339();
                    store.write_manifest(&record)?;
                }
                Ok(())
            },
        )
    }

    /// 删除非托管会话的清单与消息日志；托管会话需走投影删除接口。
    fn delete_conversation<'a>(&'a self, conversation_id: &'a str) -> BoxFut<'a, ()> {
        let conversation_id = conversation_id.to_string();
        self.run_blocking(
            Self::conversation_scope(conversation_id.clone()),
            move |store| {
                let _conversation = store.authority.scan_barrier.read().map_err(poison)?;
                let manifest = store.read_manifest(&conversation_id)?;
                if manifest
                    .as_ref()
                    .is_some_and(|record| record.projection.is_some())
                {
                    return Err(Self::managed_conversation_error(&conversation_id));
                }
                let path = store.conv_path(&conversation_id)?;
                match std::fs::remove_file(&path) {
                    Ok(()) => {}
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                    Err(e) => {
                        return Err(
                            MemoryError::IoError(format!("delete conversation: {e}")).into()
                        );
                    }
                }
                store
                    .authority
                    .conversations
                    .lock()
                    .map_err(poison)?
                    .remove(&conversation_id);
                store.remove_replaced_log(
                    &conversation_id,
                    manifest
                        .as_ref()
                        .and_then(|record| record.message_log.as_ref()),
                );
                Ok(())
            },
        )
    }

    /// 保存会话的完整消息列表：前缀匹配时只追加新后缀，否则原子替换为新 generation。
    fn save_messages<'a>(
        &'a self,
        conversation_id: &'a str,
        messages: &'a [StoredMessage],
    ) -> BoxFut<'a, ()> {
        let conversation_id = conversation_id.to_string();
        let messages = messages.to_vec();
        self.run_blocking(
            Self::conversation_scope(conversation_id.clone()),
            move |store| {
                let _conversation = store.authority.scan_barrier.read().map_err(poison)?;
                let mut record = store.read_manifest(&conversation_id)?.ok_or_else(|| {
                    MemoryError::NotFound(format!("conversation: {conversation_id}"))
                })?;
                if record.projection.is_some() {
                    return Err(Self::managed_conversation_error(&conversation_id));
                }
                let cache = match store
                    .authority
                    .conversations
                    .lock()
                    .map_err(poison)?
                    .remove(&conversation_id)
                {
                    Some(cache) if cache.log == record.message_log => cache,
                    _ => {
                        if let Some(log) = record.message_log.as_ref() {
                            record.messages = store.read_message_log(&conversation_id, log)?;
                        }
                        Self::cache_from_record(&record)?
                    }
                };
                let digests = messages
                    .iter()
                    .map(Self::semantic_digest)
                    .collect::<Result<Vec<_>>>()?;
                let append_suffix = record.message_log.is_some()
                    && Self::prefix_matches(&cache, &messages, &digests);
                record.conversation.updated_at = now_rfc3339();

                let updated_cache = if append_suffix {
                    let mut updated_cache = cache;
                    let incoming_suffix = messages
                        .iter()
                        .skip(updated_cache.messages.len())
                        .cloned()
                        .collect::<Vec<_>>();
                    let suffix = store.reserve_messages(&conversation_id, &incoming_suffix)?;
                    let mut cached_suffix = Vec::new();
                    for (assigned, digest) in suffix
                        .iter()
                        .zip(digests.iter().skip(updated_cache.messages.len()))
                    {
                        cached_suffix.push(Self::cached_message(assigned, *digest));
                    }
                    let bytes = Self::serialize_message_log(&suffix)?;
                    let mut log = record.message_log.clone().ok_or_else(|| {
                        MemoryError::SerializationError(
                            "append path is missing its message-log manifest".into(),
                        )
                    })?;
                    let path = store.prepare_log_for_append(&conversation_id, &log)?;
                    if !bytes.is_empty() {
                        echo_core::utils::fs::append_existing(
                            &path,
                            &bytes,
                            echo_core::utils::fs::FileDurability::SyncData,
                        )
                        .map_err(|error| {
                            MemoryError::IoError(format!(
                                "append conversation message log {}: {error}",
                                path.display()
                            ))
                        })?;
                    }
                    let added_bytes = u64::try_from(bytes.len()).map_err(|error| {
                        MemoryError::Unsupported(format!(
                            "message log length is unsupported: {error}"
                        ))
                    })?;
                    log.committed_bytes =
                        log.committed_bytes
                            .checked_add(added_bytes)
                            .ok_or_else(|| {
                                MemoryError::Unsupported(
                                    "conversation message log size exhausted".into(),
                                )
                            })?;
                    log.message_count = messages.len();
                    log.max_message_id = log
                        .max_message_id
                        .max(suffix.iter().filter_map(|message| message.id).max());
                    let had_search_filter = record.search_filter.is_some();
                    let mut search_filter = record.search_filter.take().unwrap_or_else(|| {
                        SearchFilter::from_conversation(&record.conversation, &messages)
                    });
                    if had_search_filter {
                        for message in &suffix {
                            search_filter.insert_message(message);
                        }
                    }
                    updated_cache.messages.extend(cached_suffix);
                    updated_cache.log = Some(log.clone());
                    record.message_log = Some(log);
                    record.search_filter = Some(search_filter);
                    record.messages.clear();
                    store.persist_current_meta()?;
                    store.write_manifest(&record)?;
                    updated_cache
                } else {
                    let assigned = store.reserve_messages(&conversation_id, &messages)?;
                    let generation = Self::next_generation(record.message_log.as_ref())?;
                    let bytes = Self::serialize_message_log(&assigned)?;
                    let path = store.message_log_path(&conversation_id, generation)?;
                    echo_core::utils::fs::atomic_write(&path, &bytes).map_err(|error| {
                        MemoryError::IoError(format!(
                            "replace conversation message log {}: {error}",
                            path.display()
                        ))
                    })?;
                    let committed_bytes = u64::try_from(bytes.len()).map_err(|error| {
                        MemoryError::Unsupported(format!(
                            "message log length is unsupported: {error}"
                        ))
                    })?;
                    let log = MessageLogMeta {
                        generation,
                        committed_bytes,
                        message_count: assigned.len(),
                        max_message_id: assigned.iter().filter_map(|message| message.id).max(),
                    };
                    let replaced = record.message_log.clone();
                    record.messages.clear();
                    record.message_log = Some(log.clone());
                    record.search_filter = Some(SearchFilter::from_conversation(
                        &record.conversation,
                        &assigned,
                    ));
                    store.persist_current_meta()?;
                    store.write_manifest(&record)?;
                    store.remove_replaced_log(&conversation_id, replaced.as_ref());
                    let cached_messages = assigned
                        .iter()
                        .zip(digests.iter())
                        .map(|(message, digest)| Self::cached_message(message, *digest))
                        .collect();
                    store
                        .authority
                        .conversations
                        .lock()
                        .map_err(poison)?
                        .insert(
                            conversation_id.clone(),
                            ConversationCache {
                                log: Some(log),
                                messages: cached_messages,
                            },
                        );
                    return Ok(());
                };

                store
                    .authority
                    .conversations
                    .lock()
                    .map_err(poison)?
                    .insert(conversation_id, updated_cache);
                Ok(())
            },
        )
    }

    /// 读取会话的全部已提交消息。
    fn get_messages<'a>(&'a self, conversation_id: &'a str) -> BoxFut<'a, Vec<StoredMessage>> {
        let conversation_id = conversation_id.to_string();
        self.run_blocking(
            Self::conversation_scope(conversation_id.clone()),
            move |store| {
                let _conversation = store.authority.scan_barrier.read().map_err(poison)?;
                let record = store.read_record(&conversation_id)?;
                if let Some(record) = record {
                    if record.projection.as_ref().is_some_and(|projection| {
                        projection.lifecycle == ConversationProjectionLifecycle::Deleted
                    }) {
                        return Ok(Vec::new());
                    }
                    store
                        .authority
                        .conversations
                        .lock()
                        .map_err(poison)?
                        .insert(conversation_id, Self::cache_from_record(&record)?);
                    Ok(record.messages)
                } else {
                    Ok(Vec::new())
                }
            },
        )
    }

    /// 返回会话已提交消息的数量（仅读取清单，不读消息正文）。
    fn count_messages<'a>(&'a self, conversation_id: &'a str) -> BoxFut<'a, usize> {
        let conversation_id = conversation_id.to_string();
        self.run_blocking(
            Self::conversation_scope(conversation_id.clone()),
            move |store| {
                let _conversation = store.authority.scan_barrier.read().map_err(poison)?;
                Ok(store
                    .read_manifest(&conversation_id)?
                    .filter(|record| {
                        record.projection.as_ref().is_none_or(|projection| {
                            projection.lifecycle == ConversationProjectionLifecycle::Live
                        })
                    })
                    .map(|record| {
                        record
                            .message_log
                            .as_ref()
                            .map_or(record.messages.len(), |log| log.message_count)
                    })
                    .unwrap_or(0))
            },
        )
    }

    /// 确保托管会话处于存活的投影 epoch：不存在则创建，已删除且墓碑匹配时重建。
    fn ensure_projection_epoch<'a>(
        &'a self,
        request: EnsureConversationProjectionRequest,
    ) -> BoxFut<'a, ConversationProjectionEpochReceipt> {
        let conversation_id = request.conversation.conversation_id.clone();
        self.run_blocking(
            Self::conversation_scope(conversation_id.clone()),
            move |store| {
                if conversation_id.trim().is_empty() {
                    return Err(MemoryError::SerializationError(
                        "managed conversation id must not be empty".to_string(),
                    )
                    .into());
                }
                let _conversation = store.authority.scan_barrier.read().map_err(poison)?;
                store.ensure_persistence_call_not_expired()?;
                let Some(mut record) = store.read_record(&conversation_id)? else {
                    let state = FileProjectionState::live();
                    if request.expected_tombstone_epoch.is_some() {
                        return Ok(ConversationProjectionEpochReceipt {
                            authority: state.authority(&conversation_id),
                            status: ConversationProjectionEpochStatus::EpochConflict,
                        });
                    }
                    store.create_conversation_sync_with_projection(
                        request.conversation,
                        Some(state.clone()),
                    )?;
                    return Ok(ConversationProjectionEpochReceipt {
                        authority: state.authority(&conversation_id),
                        status: ConversationProjectionEpochStatus::Created,
                    });
                };

                let Some(state) = record.projection.as_mut() else {
                    let state = FileProjectionState::live();
                    let authority = state.authority(&conversation_id);
                    if request.expected_tombstone_epoch.is_some() {
                        return Ok(ConversationProjectionEpochReceipt {
                            authority,
                            status: ConversationProjectionEpochStatus::EpochConflict,
                        });
                    }
                    record.projection = Some(state);
                    store.write_manifest(&record)?;
                    return Ok(ConversationProjectionEpochReceipt {
                        authority,
                        status: ConversationProjectionEpochStatus::AdoptedLegacy,
                    });
                };

                if state.lifecycle == ConversationProjectionLifecycle::Live {
                    return Ok(ConversationProjectionEpochReceipt {
                        authority: state.authority(&conversation_id),
                        status: if request.expected_tombstone_epoch.is_some() {
                            ConversationProjectionEpochStatus::EpochConflict
                        } else {
                            ConversationProjectionEpochStatus::Existing
                        },
                    });
                }
                if request.expected_tombstone_epoch != Some(state.epoch) {
                    return Ok(ConversationProjectionEpochReceipt {
                        authority: state.authority(&conversation_id),
                        status: if request.expected_tombstone_epoch.is_none() {
                            ConversationProjectionEpochStatus::Tombstoned
                        } else {
                            ConversationProjectionEpochStatus::EpochConflict
                        },
                    });
                }

                let next_epoch = state
                    .epoch
                    .checked_add(1)
                    .filter(|epoch| *epoch <= MAX_PROJECTION_EPOCH)
                    .ok_or_else(|| {
                        MemoryError::ProjectionEpochExhausted(conversation_id.clone())
                    })?;
                let next_revision = state.next_revision(&conversation_id)?;
                state.epoch = next_epoch;
                state.revision = next_revision;
                state.lifecycle = ConversationProjectionLifecycle::Live;
                state.applied_operations.clear();
                state.ordinals.clear();
                let now = now_rfc3339();
                record.conversation.user_id = request.conversation.user_id;
                record.conversation.agent_type = request.conversation.agent_type;
                record.conversation.title = request.conversation.title;
                record.conversation.summary = None;
                record.conversation.compressed_before_id = None;
                record.conversation.created_at = now.clone();
                record.conversation.updated_at = now;
                let authority = state.authority(&conversation_id);
                store.replace_record_messages(&mut record, &[])?;
                Ok(ConversationProjectionEpochReceipt {
                    authority,
                    status: ConversationProjectionEpochStatus::Recreated,
                })
            },
        )
    }

    /// 只读查询托管会话当前的投影权威状态；清单损坏时失败关闭。
    fn get_projection_authority<'a>(
        &'a self,
        conversation_id: &'a str,
    ) -> BoxFut<'a, Option<ConversationProjectionAuthority>> {
        let conversation_id = conversation_id.to_string();
        self.run_blocking(
            Self::conversation_scope(conversation_id.clone()),
            move |store| {
                if conversation_id.trim().is_empty() {
                    return Err(MemoryError::SerializationError(
                        "managed conversation id must not be empty".to_string(),
                    )
                    .into());
                }
                let _conversation = store.authority.scan_barrier.read().map_err(poison)?;
                store.ensure_persistence_call_not_expired()?;
                Ok(store
                    .read_manifest(&conversation_id)?
                    .and_then(|record| record.projection)
                    .map(|state| state.authority(&conversation_id)))
            },
        )
    }

    /// 以幂等、按 epoch 围栏的方式把一批转录投影应用到托管会话。
    fn apply_transcript_projection<'a>(
        &'a self,
        batch: TranscriptProjectionBatch,
    ) -> BoxFut<'a, TranscriptProjectionApplyReceipt> {
        let validation = batch.validate();
        let conversation_id = batch.conversation_id.clone();
        self.run_blocking(
            Self::conversation_scope(conversation_id.clone()),
            move |store| {
                validation?;
                let _conversation = store.authority.scan_barrier.read().map_err(poison)?;
                store.ensure_persistence_call_not_expired()?;
                let mut record = store.read_record(&conversation_id)?.ok_or_else(|| {
                    MemoryError::NotFound(format!("conversation: {conversation_id}"))
                })?;
                let state = record
                    .projection
                    .as_ref()
                    .ok_or_else(|| Self::managed_conversation_error(&conversation_id))?;
                let current_authority = state.authority(&conversation_id);
                if let Some(existing_digest) = state.applied_operations.get(&batch.operation_id) {
                    let status = if existing_digest == &batch.payload_digest {
                        TranscriptProjectionApplyStatus::AlreadyApplied
                    } else {
                        TranscriptProjectionApplyStatus::Conflict {
                            current_epoch: state.epoch,
                            current_revision: state.revision,
                            kind: TranscriptProjectionConflictKind::OperationIdentity,
                        }
                    };
                    return Ok(Self::apply_receipt(&batch, current_authority, status));
                }
                if state.epoch != batch.conversation_epoch
                    || state.lifecycle != ConversationProjectionLifecycle::Live
                {
                    return Ok(Self::apply_receipt(
                        &batch,
                        current_authority,
                        TranscriptProjectionApplyStatus::Fenced {
                            current_epoch: state.epoch,
                            lifecycle: state.lifecycle,
                        },
                    ));
                }

                let mut generation_frontier = 0_u64;
                if let Some(ordinals) = state.ordinals.get(&batch.generation_id) {
                    for ordinal in ordinals.keys() {
                        if *ordinal != generation_frontier {
                            return Err(MemoryError::SerializationError(format!(
                                "transcript projection generation is not contiguous: {}",
                                batch.generation_id
                            ))
                            .into());
                        }
                        generation_frontier =
                            generation_frontier.checked_add(1).ok_or_else(|| {
                                MemoryError::SerializationError(format!(
                                    "transcript projection frontier exhausted: {}",
                                    batch.generation_id
                                ))
                            })?;
                    }
                }
                let first_ordinal =
                    batch
                        .items
                        .first()
                        .map(|item| item.ordinal)
                        .ok_or_else(|| {
                            MemoryError::SerializationError(
                                "transcript projection batch must not be empty".to_string(),
                            )
                        })?;
                if first_ordinal != generation_frontier {
                    return Ok(Self::apply_receipt(
                        &batch,
                        current_authority,
                        TranscriptProjectionApplyStatus::Conflict {
                            current_epoch: state.epoch,
                            current_revision: state.revision,
                            kind: TranscriptProjectionConflictKind::OrdinalDigest,
                        },
                    ));
                }

                let mut new_messages = Vec::new();
                for item in &batch.items {
                    let existing = state
                        .ordinals
                        .get(&batch.generation_id)
                        .and_then(|ordinals| ordinals.get(&item.ordinal));
                    match existing {
                        Some(digest) if digest != &item.digest => {
                            return Ok(Self::apply_receipt(
                                &batch,
                                current_authority,
                                TranscriptProjectionApplyStatus::Conflict {
                                    current_epoch: state.epoch,
                                    current_revision: state.revision,
                                    kind: TranscriptProjectionConflictKind::OrdinalDigest,
                                },
                            ));
                        }
                        Some(_) => {}
                        None => new_messages.push(item.message.clone()),
                    }
                }

                let next_revision = state.next_revision(&conversation_id)?;
                let state = record
                    .projection
                    .as_mut()
                    .ok_or_else(|| Self::managed_conversation_error(&conversation_id))?;
                let generation = state
                    .ordinals
                    .entry(batch.generation_id.clone())
                    .or_default();
                for item in &batch.items {
                    generation.insert(item.ordinal, item.digest.clone());
                }
                state
                    .applied_operations
                    .insert(batch.operation_id.clone(), batch.payload_digest.clone());
                state.revision = next_revision;
                record.conversation.updated_at = now_rfc3339();
                let authority = state.authority(&conversation_id);
                if new_messages.is_empty() {
                    store.write_manifest(&record)?;
                } else {
                    let mut messages = record.messages.clone();
                    messages.extend(new_messages);
                    store.replace_record_messages(&mut record, &messages)?;
                }
                Ok(Self::apply_receipt(
                    &batch,
                    authority,
                    TranscriptProjectionApplyStatus::Applied,
                ))
            },
        )
    }

    /// 以 revision CAS 方式把外部消息导入托管会话。
    fn import_managed_messages<'a>(
        &'a self,
        request: ManagedConversationImport,
    ) -> BoxFut<'a, TranscriptProjectionApplyReceipt> {
        let validation = request.validate();
        let conversation_id = request.conversation_id.clone();
        self.run_blocking(
            Self::conversation_scope(conversation_id.clone()),
            move |store| {
                validation?;
                let _conversation = store.authority.scan_barrier.read().map_err(poison)?;
                store.ensure_persistence_call_not_expired()?;
                let mut record = store.read_record(&conversation_id)?.ok_or_else(|| {
                    MemoryError::NotFound(format!("conversation: {conversation_id}"))
                })?;
                let state = record
                    .projection
                    .as_ref()
                    .ok_or_else(|| Self::managed_conversation_error(&conversation_id))?;
                let current_authority = state.authority(&conversation_id);
                if let Some(existing_digest) = state.applied_operations.get(&request.operation_id) {
                    let status = if existing_digest == &request.payload_digest {
                        TranscriptProjectionApplyStatus::AlreadyApplied
                    } else {
                        TranscriptProjectionApplyStatus::Conflict {
                            current_epoch: state.epoch,
                            current_revision: state.revision,
                            kind: TranscriptProjectionConflictKind::OperationIdentity,
                        }
                    };
                    return Ok(Self::import_receipt(&request, current_authority, status));
                }
                if state.epoch != request.expected_epoch
                    || state.lifecycle != ConversationProjectionLifecycle::Live
                {
                    return Ok(Self::import_receipt(
                        &request,
                        current_authority,
                        TranscriptProjectionApplyStatus::Fenced {
                            current_epoch: state.epoch,
                            lifecycle: state.lifecycle,
                        },
                    ));
                }
                if state.revision != request.expected_revision {
                    return Ok(Self::import_receipt(
                        &request,
                        current_authority,
                        TranscriptProjectionApplyStatus::Conflict {
                            current_epoch: state.epoch,
                            current_revision: state.revision,
                            kind: TranscriptProjectionConflictKind::Revision,
                        },
                    ));
                }

                let next_epoch = state
                    .epoch
                    .checked_add(1)
                    .filter(|epoch| *epoch <= MAX_PROJECTION_EPOCH)
                    .ok_or_else(|| {
                        MemoryError::ProjectionEpochExhausted(conversation_id.clone())
                    })?;
                let next_revision = state.next_revision(&conversation_id)?;
                let state = record
                    .projection
                    .as_mut()
                    .ok_or_else(|| Self::managed_conversation_error(&conversation_id))?;
                state.epoch = next_epoch;
                state.applied_operations.clear();
                state.ordinals.clear();
                state
                    .applied_operations
                    .insert(request.operation_id.clone(), request.payload_digest.clone());
                state.revision = next_revision;
                record.conversation.updated_at = now_rfc3339();
                record.conversation.summary = None;
                record.conversation.compressed_before_id = None;
                let authority = state.authority(&conversation_id);
                store.replace_record_messages(&mut record, &request.messages)?;
                Ok(Self::import_receipt(
                    &request,
                    authority,
                    TranscriptProjectionApplyStatus::Applied,
                ))
            },
        )
    }

    /// 以幂等、按 epoch 围栏的方式更新托管会话的元数据。
    fn update_managed_conversation<'a>(
        &'a self,
        request: ManagedConversationMetadataUpdate,
    ) -> BoxFut<'a, ManagedConversationMetadataUpdateReceipt> {
        let validation = request.validate();
        let conversation_id = request.conversation_id.clone();
        self.run_blocking(
            Self::conversation_scope(conversation_id.clone()),
            move |store| {
                validation?;
                let _conversation = store.authority.scan_barrier.read().map_err(poison)?;
                store.ensure_persistence_call_not_expired()?;
                let mut record = store.read_record(&conversation_id)?.ok_or_else(|| {
                    MemoryError::NotFound(format!("conversation: {conversation_id}"))
                })?;
                let state = record
                    .projection
                    .as_ref()
                    .ok_or_else(|| Self::managed_conversation_error(&conversation_id))?;
                let current_authority = state.authority(&conversation_id);
                if let Some(existing_digest) = state.applied_operations.get(&request.operation_id) {
                    return Ok(ManagedConversationMetadataUpdateReceipt {
                        operation_id: request.operation_id,
                        payload_digest: request.payload_digest.clone(),
                        authority: current_authority,
                        status: if existing_digest == &request.payload_digest {
                            ManagedConversationMetadataUpdateStatus::AlreadyUpdated
                        } else {
                            ManagedConversationMetadataUpdateStatus::IdentityConflict
                        },
                    });
                }
                if state.epoch != request.expected_epoch
                    || state.lifecycle != ConversationProjectionLifecycle::Live
                {
                    return Ok(ManagedConversationMetadataUpdateReceipt {
                        operation_id: request.operation_id,
                        payload_digest: request.payload_digest,
                        authority: current_authority,
                        status: ManagedConversationMetadataUpdateStatus::EpochConflict,
                    });
                }
                if state.revision != request.expected_revision {
                    return Ok(ManagedConversationMetadataUpdateReceipt {
                        operation_id: request.operation_id,
                        payload_digest: request.payload_digest,
                        authority: current_authority,
                        status: ManagedConversationMetadataUpdateStatus::RevisionConflict,
                    });
                }
                if let Some(compressed_before_id) = request.compressed_before_id
                    && !record
                        .messages
                        .iter()
                        .any(|message| message.id == Some(compressed_before_id))
                {
                    return Err(MemoryError::NotFound(format!(
                        "conversation compression boundary: {compressed_before_id}"
                    ))
                    .into());
                }

                let next_revision = state.next_revision(&conversation_id)?;
                if let Some(title) = request.title.as_ref() {
                    record.conversation.title = Some(title.clone());
                    if let Some(filter) = record.search_filter.as_mut() {
                        filter.insert_text(title);
                    }
                }
                if let Some(summary) = request.summary.as_ref() {
                    record.conversation.summary = Some(summary.clone());
                }
                if let Some(compressed_before_id) = request.compressed_before_id {
                    record.conversation.compressed_before_id = Some(compressed_before_id);
                }
                record.conversation.updated_at = now_rfc3339();
                let state = record
                    .projection
                    .as_mut()
                    .ok_or_else(|| Self::managed_conversation_error(&conversation_id))?;
                state.revision = next_revision;
                state
                    .applied_operations
                    .insert(request.operation_id.clone(), request.payload_digest.clone());
                let authority = state.authority(&conversation_id);
                store.write_manifest(&record)?;
                Ok(ManagedConversationMetadataUpdateReceipt {
                    operation_id: request.operation_id,
                    payload_digest: request.payload_digest,
                    authority,
                    status: ManagedConversationMetadataUpdateStatus::Updated,
                })
            },
        )
    }

    /// 删除托管会话并留下墓碑与删除回执，以围栏迟到的写入。
    fn delete_managed_conversation<'a>(
        &'a self,
        request: ManagedConversationDelete,
    ) -> BoxFut<'a, ManagedConversationDeleteReceipt> {
        let validation = request.validate();
        let conversation_id = request.conversation_id.clone();
        self.run_blocking(
            Self::conversation_scope(conversation_id.clone()),
            move |store| {
                validation?;
                let _conversation = store.authority.scan_barrier.read().map_err(poison)?;
                store.ensure_persistence_call_not_expired()?;
                let mut record = store.read_record(&conversation_id)?.ok_or_else(|| {
                    MemoryError::NotFound(format!("conversation: {conversation_id}"))
                })?;
                let state = record
                    .projection
                    .as_ref()
                    .ok_or_else(|| Self::managed_conversation_error(&conversation_id))?;
                if let Some(existing) = state.delete_receipts.get(&request.operation_id) {
                    return Ok(ManagedConversationDeleteReceipt {
                        operation_id: request.operation_id,
                        payload_digest: existing.payload_digest.clone(),
                        deleted_epoch: existing.deleted_epoch,
                        retention_floor_epoch: state.retention_floor_epoch,
                        status: if existing.payload_digest == request.payload_digest {
                            ManagedConversationDeleteStatus::AlreadyDeleted
                        } else {
                            ManagedConversationDeleteStatus::IdentityConflict
                        },
                    });
                }
                if request.expected_epoch <= state.retention_floor_epoch {
                    return Ok(ManagedConversationDeleteReceipt {
                        operation_id: request.operation_id,
                        payload_digest: request.payload_digest,
                        deleted_epoch: request.expected_epoch,
                        retention_floor_epoch: state.retention_floor_epoch,
                        status: ManagedConversationDeleteStatus::ReceiptExpired,
                    });
                }
                if state.epoch != request.expected_epoch
                    || state.lifecycle != ConversationProjectionLifecycle::Live
                {
                    return Ok(ManagedConversationDeleteReceipt {
                        operation_id: request.operation_id,
                        payload_digest: request.payload_digest,
                        deleted_epoch: request.expected_epoch,
                        retention_floor_epoch: state.retention_floor_epoch,
                        status: ManagedConversationDeleteStatus::EpochConflict,
                    });
                }

                let next_revision = state.next_revision(&conversation_id)?;
                let state = record
                    .projection
                    .as_mut()
                    .ok_or_else(|| Self::managed_conversation_error(&conversation_id))?;
                state.revision = next_revision;
                state.lifecycle = ConversationProjectionLifecycle::Deleted;
                state.applied_operations.clear();
                state.ordinals.clear();
                state.delete_receipts.insert(
                    request.operation_id.clone(),
                    FileDeleteReceipt {
                        payload_digest: request.payload_digest.clone(),
                        deleted_epoch: request.expected_epoch,
                    },
                );
                state.retain_delete_receipts();
                let retention_floor_epoch = state.retention_floor_epoch;
                record.conversation.summary = None;
                record.conversation.compressed_before_id = None;
                record.conversation.updated_at = now_rfc3339();
                store.replace_record_messages(&mut record, &[])?;
                Ok(ManagedConversationDeleteReceipt {
                    operation_id: request.operation_id,
                    payload_digest: request.payload_digest,
                    deleted_epoch: request.expected_epoch,
                    retention_floor_epoch,
                    status: ManagedConversationDeleteStatus::Deleted,
                })
            },
        )
    }

    /// 在标题与消息内容中做大小写不敏感的子串搜索，先用过滤器筛选候选再精确校验。
    fn search_conversations<'a>(
        &'a self,
        query: &'a str,
        limit: usize,
    ) -> BoxFut<'a, Vec<ConversationMeta>> {
        let query = query.to_string();
        self.run_blocking(
            BlockingFileOperationScope::Collection("search".to_string()),
            move |store| {
                let _scan = store.authority.scan_barrier.write().map_err(poison)?;
                let needle = query.to_lowercase();
                let mut results = Vec::new();
                let records = store.read_all_manifests()?;
                #[cfg(test)]
                store.pause_search_after_snapshot()?;
                for mut record in records {
                    if record.projection.as_ref().is_some_and(|projection| {
                        projection.lifecycle == ConversationProjectionLifecycle::Deleted
                    }) {
                        continue;
                    }
                    let title_hit = record
                        .conversation
                        .title
                        .as_deref()
                        .is_some_and(|title| title.to_lowercase().contains(&needle));
                    let candidate = title_hit
                        || record
                            .search_filter
                            .as_ref()
                            .is_none_or(|filter| filter.might_contain(&query));
                    if !candidate {
                        continue;
                    }
                    if !title_hit {
                        if let Some(log) = record.message_log.as_ref() {
                            record.messages = store
                                .read_message_log(&record.conversation.conversation_id, log)?;
                        }
                        let message_hit = record.messages.iter().any(|message| {
                            message
                                .content
                                .as_deref()
                                .is_some_and(|content| content.to_lowercase().contains(&needle))
                        });
                        if !message_hit {
                            continue;
                        }
                    }
                    let message_count = record
                        .message_log
                        .as_ref()
                        .map_or(record.messages.len(), |log| log.message_count);
                    results.push(ConversationMeta {
                        id: record.conversation.id,
                        conversation_id: record.conversation.conversation_id,
                        user_id: record.conversation.user_id,
                        title: record.conversation.title,
                        message_count,
                        created_at: record.conversation.created_at,
                        updated_at: record.conversation.updated_at,
                    });
                }
                // 等价于 ORDER BY updated_at DESC，然后 LIMIT。
                results.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
                results.truncate(limit);
                Ok(results)
            },
        )
    }

    /// 获取已存在的非托管会话，不存在时创建。
    fn ensure_conversation<'a>(&'a self, conv: NewConversation) -> BoxFut<'a, Conversation> {
        let scope = Self::conversation_scope(conv.conversation_id.clone());
        self.run_blocking(scope, move |store| {
            let _conversation = store.authority.scan_barrier.read().map_err(poison)?;
            if let Some(existing) = store.read_manifest(&conv.conversation_id)? {
                if existing.projection.is_some() {
                    return Err(Self::managed_conversation_error(&conv.conversation_id));
                }
                return Ok(existing.conversation);
            }
            store.create_conversation_sync(conv)
        })
    }
}

/// 以 RFC 3339 格式返回当前本地时间。
fn now_rfc3339() -> String {
    echo_core::utils::time::now_local().to_rfc3339()
}

/// 校验 id，并把其精确的 UTF-8 字节编码为一个安全的路径片段（保留 `_meta`）。
fn safe_segment(id: &str) -> Result<String> {
    if id == "_meta" {
        return Err(MemoryError::Unsupported("conversation id is reserved: _meta".into()).into());
    }
    echo_core::utils::fs::encode_path_segment_identity(id)
        .map_err(|error| MemoryError::Unsupported(error.to_string()).into())
}

/// 判断文件名是否形如 `<safe_id>.messages.<generation>.jsonl` 的消息日志文件。
fn is_message_log_file_name(file_name: &str) -> bool {
    let Some(without_suffix) = file_name.strip_suffix(".jsonl") else {
        return false;
    };
    let Some((conversation_id, generation)) = without_suffix.rsplit_once(".messages.") else {
        return false;
    };
    conversation_id != "_meta"
        && echo_core::utils::fs::validate_path_segment(conversation_id).is_ok()
        && generation.parse::<u64>().is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use echo_core::memory::conversation::{
        ConversationProjectionCapability, ConversationProjectionEpochStatus,
        EnsureConversationProjectionRequest, ManagedConversationDelete,
        ManagedConversationDeleteStatus, ManagedConversationImport, PersistenceCallContext,
        TranscriptProjectionApplyStatus, TranscriptProjectionBatch,
        TranscriptProjectionConflictKind,
    };
    use std::time::Duration;

    /// 测试用的结果类型，错误统一装箱为 `dyn Error`。
    type TestResult<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;

    /// 生成一个唯一的临时根目录路径（按进程 id + uuid 区分）。
    fn tmp_base() -> PathBuf {
        std::env::temp_dir().join(format!(
            "echo-file-conv-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ))
    }

    /// 构造一个用于测试的新会话请求。
    fn new_conv(id: &str, title: Option<&str>) -> NewConversation {
        NewConversation {
            conversation_id: id.to_string(),
            user_id: "default".to_string(),
            agent_type: None,
            title: title.map(String::from),
        }
    }

    /// 构造一条会话 id 为 `c1`、无 id/时间戳的测试消息。
    fn stored(role: &str, content: &str) -> StoredMessage {
        StoredMessage {
            id: None,
            conversation_id: "c1".into(),
            role: role.into(),
            content: Some(content.into()),
            attachments_json: None,
            tool_calls_json: None,
            tool_result_json: None,
            created_at: String::new(),
        }
    }

    /// 构造一条带固定时间戳的投影测试消息。
    fn projection_message(conversation_id: &str, content: &str) -> StoredMessage {
        StoredMessage {
            id: None,
            conversation_id: conversation_id.to_string(),
            role: "user".to_string(),
            content: Some(content.to_string()),
            attachments_json: None,
            tool_calls_json: None,
            tool_result_json: None,
            created_at: "2026-09-16T00:00:00Z".to_string(),
        }
    }

    /// 构造一个转录投影批次（从 `first_ordinal` 开始依次编号）。
    fn projection_batch(
        conversation_id: &str,
        epoch: u64,
        generation_id: &str,
        first_ordinal: u64,
        contents: &[&str],
    ) -> Result<TranscriptProjectionBatch> {
        TranscriptProjectionBatch::prepare(
            conversation_id,
            epoch,
            generation_id,
            first_ordinal,
            contents
                .iter()
                .map(|content| projection_message(conversation_id, content))
                .collect(),
        )
    }

    /// 构造一个确保投影 epoch 的请求，可指定期望的墓碑 epoch。
    fn ensure_projection_request(
        conversation_id: &str,
        expected_tombstone_epoch: Option<u64>,
    ) -> EnsureConversationProjectionRequest {
        EnsureConversationProjectionRequest {
            conversation: new_conv(conversation_id, None),
            expected_tombstone_epoch,
        }
    }

    /// 构造一个托管消息导入请求。
    fn managed_import(
        conversation_id: &str,
        expected_epoch: u64,
        expected_revision: u64,
        contents: &[&str],
    ) -> Result<ManagedConversationImport> {
        ManagedConversationImport::prepare(
            conversation_id,
            expected_epoch,
            expected_revision,
            contents
                .iter()
                .map(|content| projection_message(conversation_id, content))
                .collect(),
        )
    }

    /// 构造一个托管会话删除请求。
    fn managed_delete(
        conversation_id: &str,
        expected_epoch: u64,
    ) -> Result<ManagedConversationDelete> {
        ManagedConversationDelete::prepare(conversation_id, expected_epoch)
    }

    /// 构造一个托管会话元数据更新请求。
    fn managed_metadata_update(
        conversation_id: &str,
        expected_epoch: u64,
        expected_revision: u64,
        title: &str,
        summary: &str,
        compressed_before_id: i64,
    ) -> Result<ManagedConversationMetadataUpdate> {
        ManagedConversationMetadataUpdate::prepare(
            conversation_id,
            expected_epoch,
            expected_revision,
            Some(title.to_string()),
            Some(summary.to_string()),
            Some(compressed_before_id),
        )
    }

    /// 判断结果是否为「托管会话必须走投影接口」错误。
    fn is_managed_projection_error<T>(result: echo_core::error::Result<T>) -> bool {
        matches!(
            result,
            Err(echo_core::error::ReactError::Memory(error))
                if matches!(
                    error.as_ref(),
                    MemoryError::ManagedConversationRequiresProjection(_)
                )
        )
    }

    /// 判断结果是否为截止时间已过错误。
    fn is_deadline_exceeded<T>(result: echo_core::error::Result<T>) -> bool {
        matches!(
            result,
            Err(echo_core::error::ReactError::Memory(error))
                if matches!(error.as_ref(), MemoryError::DeadlineExceeded(_))
        )
    }

    /// 验证：重建已删除的投影 epoch 时必须提供匹配的墓碑 epoch。
    #[tokio::test]
    async fn projection_epoch_recreation_requires_a_matching_tombstone() -> TestResult {
        let base = tmp_base();
        let store = FileConversationStore::new(&base)?;

        let absent = store
            .ensure_projection_epoch(ensure_projection_request("absent-epoch", Some(7)))
            .await?;
        assert_eq!(
            absent.status,
            ConversationProjectionEpochStatus::EpochConflict
        );
        assert!(store.get_conversation("absent-epoch").await?.is_none());
        assert_eq!(
            store
                .ensure_projection_epoch(ensure_projection_request("absent-epoch", None))
                .await?
                .status,
            ConversationProjectionEpochStatus::Created
        );
        assert_eq!(
            store
                .ensure_projection_epoch(ensure_projection_request("absent-epoch", Some(1)))
                .await?
                .status,
            ConversationProjectionEpochStatus::EpochConflict
        );
        assert_eq!(
            store
                .ensure_projection_epoch(ensure_projection_request("absent-epoch", None))
                .await?
                .status,
            ConversationProjectionEpochStatus::Existing
        );

        store
            .create_conversation(new_conv("legacy-epoch", Some("legacy")))
            .await?;
        store
            .save_messages(
                "legacy-epoch",
                &[projection_message("legacy-epoch", "before-conflict")],
            )
            .await?;
        let legacy = store
            .ensure_projection_epoch(ensure_projection_request("legacy-epoch", Some(1)))
            .await?;
        assert_eq!(
            legacy.status,
            ConversationProjectionEpochStatus::EpochConflict
        );
        assert_eq!(store.count_messages("legacy-epoch").await?, 1);
        assert_eq!(
            store
                .ensure_projection_epoch(ensure_projection_request("legacy-epoch", None))
                .await?
                .status,
            ConversationProjectionEpochStatus::AdoptedLegacy
        );

        std::fs::remove_dir_all(base)?;
        Ok(())
    }

    /// 验证：投影权威查询是只读的，且清单损坏时失败关闭。
    #[tokio::test]
    async fn projection_authority_query_is_read_only_and_fails_closed_on_corruption() -> TestResult
    {
        let base = tmp_base();
        let store = FileConversationStore::new(&base)?;

        assert!(
            store
                .get_projection_authority("authority-live")
                .await?
                .is_none()
        );
        assert!(store.get_conversation("authority-live").await?.is_none());
        let acquired = store
            .ensure_projection_epoch(ensure_projection_request("authority-live", None))
            .await?;
        assert_eq!(acquired.status, ConversationProjectionEpochStatus::Created);
        assert_eq!(
            store
                .get_projection_authority_with_context(
                    PersistenceCallContext {
                        absolute_deadline_unix_ms: i64::MAX,
                    },
                    "authority-live",
                )
                .await?,
            Some(acquired.authority.clone())
        );

        store
            .create_conversation(new_conv("authority-legacy", Some("legacy")))
            .await?;
        assert!(
            store
                .get_projection_authority("authority-legacy")
                .await?
                .is_none()
        );
        assert_eq!(
            store
                .ensure_projection_epoch(ensure_projection_request("authority-legacy", None))
                .await?
                .status,
            ConversationProjectionEpochStatus::AdoptedLegacy
        );

        let deleted = store
            .delete_managed_conversation(managed_delete(
                "authority-live",
                acquired.authority.epoch,
            )?)
            .await?;
        let deleted_authority = store
            .get_projection_authority("authority-live")
            .await?
            .ok_or_else(|| std::io::Error::other("deleted authority disappeared"))?;
        assert_eq!(deleted_authority.conversation_id, "authority-live");
        assert_eq!(deleted_authority.epoch, deleted.deleted_epoch);
        assert_eq!(
            deleted_authority.revision,
            acquired
                .authority
                .revision
                .checked_add(1)
                .ok_or_else(|| { std::io::Error::other("deleted authority revision overflow") })?
        );
        assert_eq!(
            deleted_authority.lifecycle,
            ConversationProjectionLifecycle::Deleted
        );
        assert_eq!(
            deleted_authority.delete_receipt_retention_floor_epoch,
            deleted.retention_floor_epoch
        );

        store
            .ensure_projection_epoch(ensure_projection_request("authority-corrupt", None))
            .await?;
        let mut record = manifest(&store, "authority-corrupt")?;
        record
            .projection
            .as_mut()
            .ok_or_else(|| std::io::Error::other("projection state was not persisted"))?
            .epoch = 0;
        store.write_manifest(&record)?;
        let corrupt_path = store.conv_path("authority-corrupt")?;
        let before = std::fs::read_to_string(&corrupt_path)?;
        let query = store.get_projection_authority("authority-corrupt").await;
        assert!(matches!(
            query,
            Err(echo_core::error::ReactError::Memory(error))
                if matches!(error.as_ref(), MemoryError::SerializationError(_))
        ));
        assert_eq!(std::fs::read_to_string(corrupt_path)?, before);

        let invalid = store.get_projection_authority("  ").await;
        assert!(matches!(
            invalid,
            Err(echo_core::error::ReactError::Memory(error))
                if matches!(error.as_ref(), MemoryError::SerializationError(_))
        ));

        std::fs::remove_dir_all(base)?;
        Ok(())
    }

    /// 验证：转录投影拒绝同一 generation 内序号不连续的批次。
    #[tokio::test]
    async fn transcript_projection_rejects_non_contiguous_generation_batches() -> TestResult {
        let base = tmp_base();
        let store = FileConversationStore::new(&base)?;
        store
            .ensure_projection_epoch(ensure_projection_request("frontier", None))
            .await?;

        let ordinal_ten = store
            .apply_transcript_projection(projection_batch(
                "frontier",
                1,
                "generation-a",
                10,
                &["ordinal-10"],
            )?)
            .await?;
        assert!(matches!(
            ordinal_ten.status,
            TranscriptProjectionApplyStatus::Conflict {
                kind: TranscriptProjectionConflictKind::OrdinalDigest,
                ..
            }
        ));
        assert_eq!(ordinal_ten.authority.revision, 1);
        assert_eq!(store.count_messages("frontier").await?, 0);

        let first = projection_batch("frontier", 1, "generation-a", 0, &["ordinal-0"])?;
        assert_eq!(
            store
                .apply_transcript_projection(first.clone())
                .await?
                .status,
            TranscriptProjectionApplyStatus::Applied
        );
        assert_eq!(
            store.apply_transcript_projection(first).await?.status,
            TranscriptProjectionApplyStatus::AlreadyApplied
        );

        let gap = store
            .apply_transcript_projection(projection_batch(
                "frontier",
                1,
                "generation-a",
                2,
                &["ordinal-2"],
            )?)
            .await?;
        assert!(matches!(
            gap.status,
            TranscriptProjectionApplyStatus::Conflict {
                kind: TranscriptProjectionConflictKind::OrdinalDigest,
                ..
            }
        ));
        assert_eq!(gap.authority.revision, 2);
        assert_eq!(store.count_messages("frontier").await?, 1);

        assert_eq!(
            store
                .apply_transcript_projection(projection_batch(
                    "frontier",
                    1,
                    "generation-a",
                    1,
                    &["ordinal-1"],
                )?)
                .await?
                .status,
            TranscriptProjectionApplyStatus::Applied
        );
        let late_zero = store
            .apply_transcript_projection(projection_batch(
                "frontier",
                1,
                "generation-a",
                0,
                &["late-ordinal-0"],
            )?)
            .await?;
        assert!(matches!(
            late_zero.status,
            TranscriptProjectionApplyStatus::Conflict {
                kind: TranscriptProjectionConflictKind::OrdinalDigest,
                ..
            }
        ));
        assert_eq!(late_zero.authority.revision, 3);
        let messages = store.get_messages("frontier").await?;
        assert_eq!(
            messages
                .iter()
                .filter_map(|message| message.content.as_deref())
                .collect::<Vec<_>>(),
            vec!["ordinal-0", "ordinal-1"]
        );

        std::fs::remove_dir_all(base)?;
        Ok(())
    }

    /// 验证：原子投影是幂等的，且能合并多个 generation。
    #[tokio::test]
    async fn atomic_projection_is_idempotent_and_merges_generations() -> TestResult {
        let base = tmp_base();
        let store = FileConversationStore::new(&base)?;
        assert_eq!(
            store.projection_capability(),
            ConversationProjectionCapability::AtomicV1
        );
        let acquired = store
            .ensure_projection_epoch(ensure_projection_request("atomic", None))
            .await?;
        assert_eq!(acquired.status, ConversationProjectionEpochStatus::Created);
        assert_eq!(acquired.authority.epoch, 1);

        let first = projection_batch("atomic", 1, "generation-a", 0, &["a0", "a1"])?;
        let applied = store.apply_transcript_projection(first.clone()).await?;
        assert_eq!(applied.status, TranscriptProjectionApplyStatus::Applied);
        let repeated = store.apply_transcript_projection(first).await?;
        assert_eq!(
            repeated.status,
            TranscriptProjectionApplyStatus::AlreadyApplied
        );

        let second = projection_batch("atomic", 1, "generation-b", 0, &["b0"])?;
        let merged = store.apply_transcript_projection(second).await?;
        assert_eq!(merged.status, TranscriptProjectionApplyStatus::Applied);
        let messages = store.get_messages("atomic").await?;
        assert_eq!(messages.len(), 3);
        assert_eq!(
            messages
                .iter()
                .filter_map(|message| message.content.as_deref())
                .collect::<Vec<_>>(),
            vec!["a0", "a1", "b0"]
        );

        let conflict = projection_batch("atomic", 1, "generation-a", 0, &["changed"])?;
        let receipt = store.apply_transcript_projection(conflict).await?;
        assert!(matches!(
            receipt.status,
            TranscriptProjectionApplyStatus::Conflict {
                kind: TranscriptProjectionConflictKind::OrdinalDigest,
                ..
            }
        ));
        assert_eq!(store.count_messages("atomic").await?, 3);
        std::fs::remove_dir_all(base)?;
        Ok(())
    }

    /// 验证：托管投影接口在调用上下文截止时间已过时拒绝执行。
    #[tokio::test]
    async fn managed_projection_context_rejects_expired_deadlines() -> TestResult {
        let base = tmp_base();
        let store = FileConversationStore::new(&base)?;
        assert_eq!(
            store.persistence_call_capability(),
            PersistenceCallCapability::AbsoluteDeadlineV1
        );
        let expired = PersistenceCallContext {
            absolute_deadline_unix_ms: 0,
        };

        assert!(is_deadline_exceeded(
            store
                .ensure_projection_epoch_with_context(
                    expired,
                    ensure_projection_request("expired", None),
                )
                .await
        ));
        assert!(is_deadline_exceeded(
            store
                .get_projection_authority_with_context(expired, "expired")
                .await
        ));
        assert!(is_deadline_exceeded(
            store
                .apply_transcript_projection_with_context(
                    expired,
                    projection_batch("expired", 1, "generation", 0, &["message"])?
                )
                .await
        ));
        assert!(is_deadline_exceeded(
            store
                .import_managed_messages_with_context(
                    expired,
                    managed_import("expired", 1, 0, &["message"])?
                )
                .await
        ));
        assert!(is_deadline_exceeded(
            store
                .update_managed_conversation_with_context(
                    expired,
                    managed_metadata_update("expired", 1, 0, "title", "summary", 1)?
                )
                .await
        ));
        assert!(is_deadline_exceeded(
            store
                .delete_managed_conversation_with_context(expired, managed_delete("expired", 1)?)
                .await
        ));
        assert!(store.get_conversation("expired").await?.is_none());

        let blocker_store = store.clone();
        let (entered_tx, entered_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let blocker = tokio::spawn(async move {
            blocker_store
                .run_blocking(
                    FileConversationStore::conversation_scope("queued-expiry"),
                    move |_| {
                        let _ignored = entered_tx.send(());
                        release_rx
                            .recv_timeout(Duration::from_secs(2))
                            .map_err(|error| {
                                MemoryError::IoError(format!(
                                    "release deadline queue blocker: {error}"
                                ))
                            })?;
                        Ok(())
                    },
                )
                .await
        });
        entered_rx.await?;
        let queued_context = PersistenceCallContext::with_timeout(Duration::from_millis(25))?;
        let (queued_result, release_result) = tokio::join!(
            store.ensure_projection_epoch_with_context(
                queued_context,
                ensure_projection_request("queued-expiry", None),
            ),
            async move {
                tokio::time::sleep(Duration::from_millis(75)).await;
                release_tx.send(())
            }
        );
        release_result?;
        blocker.await??;
        assert!(is_deadline_exceeded(queued_result));
        assert!(store.get_conversation("queued-expiry").await?.is_none());

        let authority = Arc::clone(&store.authority);
        let (locked_tx, locked_rx) = std::sync::mpsc::channel();
        let (unlock_tx, unlock_rx) = std::sync::mpsc::channel();
        let lock_blocker = std::thread::spawn(move || -> echo_core::error::Result<()> {
            let _scan = authority.scan_barrier.write().map_err(poison)?;
            locked_tx.send(()).map_err(|error| {
                MemoryError::IoError(format!("publish scan deadline blocker: {error}"))
            })?;
            unlock_rx
                .recv_timeout(Duration::from_secs(2))
                .map_err(|error| {
                    MemoryError::IoError(format!("release scan deadline blocker: {error}"))
                })?;
            Ok(())
        });
        locked_rx.recv_timeout(Duration::from_secs(2))?;
        let locked_context = PersistenceCallContext::with_timeout(Duration::from_millis(25))?;
        let (locked_result, unlock_result) = tokio::join!(
            store.ensure_projection_epoch_with_context(
                locked_context,
                ensure_projection_request("scan-expiry", None),
            ),
            async move {
                tokio::time::sleep(Duration::from_millis(75)).await;
                unlock_tx.send(())
            }
        );
        unlock_result?;
        lock_blocker
            .join()
            .map_err(|_| std::io::Error::other("scan deadline blocker panicked"))??;
        assert!(is_deadline_exceeded(locked_result));
        assert!(store.get_conversation("scan-expiry").await?.is_none());

        std::fs::remove_dir_all(base)?;
        Ok(())
    }

    /// 验证：托管导入使用 revision CAS，并阻止原始写接口修改托管会话。
    #[tokio::test]
    async fn managed_import_uses_revision_cas_and_fences_raw_mutators() -> TestResult {
        let base = tmp_base();
        let store = FileConversationStore::new(&base)?;
        store
            .create_conversation(new_conv("managed-import", None))
            .await?;
        store
            .save_messages(
                "managed-import",
                &[projection_message("managed-import", "legacy")],
            )
            .await?;
        let adopted = store
            .ensure_projection_epoch(ensure_projection_request("managed-import", None))
            .await?;
        assert_eq!(
            adopted.status,
            ConversationProjectionEpochStatus::AdoptedLegacy
        );

        assert!(is_managed_projection_error(
            store
                .save_messages(
                    "managed-import",
                    &[projection_message("managed-import", "raw-overwrite")],
                )
                .await
        ));
        assert!(is_managed_projection_error(
            store.delete_conversation("managed-import").await
        ));
        assert!(is_managed_projection_error(
            store
                .ensure_conversation(new_conv("managed-import", None))
                .await
        ));
        assert!(is_managed_projection_error(
            store
                .create_conversation(new_conv("managed-import", None))
                .await
        ));
        assert!(is_managed_projection_error(
            store
                .update_conversation("managed-import", Some("late"), Some("late"), Some(1))
                .await
        ));

        let old_batch = projection_batch(
            "managed-import",
            adopted.authority.epoch,
            "pre-import-generation",
            0,
            &["old-projection"],
        )?;
        let projected = store.apply_transcript_projection(old_batch.clone()).await?;
        assert_eq!(projected.status, TranscriptProjectionApplyStatus::Applied);
        let import = managed_import(
            "managed-import",
            adopted.authority.epoch,
            projected.authority.revision,
            &["restored"],
        )?;
        let imported = store.import_managed_messages(import.clone()).await?;
        assert_eq!(imported.status, TranscriptProjectionApplyStatus::Applied);
        let repeated = store.import_managed_messages(import.clone()).await?;
        assert_eq!(
            repeated.status,
            TranscriptProjectionApplyStatus::AlreadyApplied
        );
        let mut tampered_import = import;
        tampered_import.messages = vec![projection_message("managed-import", "tampered")];
        assert!(
            store
                .import_managed_messages(tampered_import)
                .await
                .is_err()
        );
        let late_old_projection = store.apply_transcript_projection(old_batch).await?;
        assert!(matches!(
            late_old_projection.status,
            TranscriptProjectionApplyStatus::Fenced {
                current_epoch: 2,
                ..
            }
        ));
        let stale = store
            .import_managed_messages(managed_import(
                "managed-import",
                imported.authority.epoch,
                adopted.authority.revision,
                &["stale"],
            )?)
            .await?;
        assert!(matches!(
            stale.status,
            TranscriptProjectionApplyStatus::Conflict {
                kind: TranscriptProjectionConflictKind::Revision,
                ..
            }
        ));
        let messages = store.get_messages("managed-import").await?;
        assert_eq!(
            messages
                .first()
                .and_then(|message| message.content.as_deref()),
            Some("restored")
        );
        std::fs::remove_dir_all(base)?;
        Ok(())
    }

    /// 验证：托管元数据更新是幂等的，且受 epoch 围栏保护。
    #[tokio::test]
    async fn managed_metadata_update_is_idempotent_and_epoch_fenced() -> TestResult {
        let base = tmp_base();
        let store = FileConversationStore::new(&base)?;
        let acquired = store
            .ensure_projection_epoch(ensure_projection_request("metadata", None))
            .await?;
        let applied = store
            .apply_transcript_projection(projection_batch(
                "metadata",
                acquired.authority.epoch,
                "generation-a",
                0,
                &["message"],
            )?)
            .await?;
        let message_id = store
            .get_messages("metadata")
            .await?
            .first()
            .and_then(|message| message.id)
            .ok_or_else(|| std::io::Error::other("managed message id was not assigned"))?;
        let update = managed_metadata_update(
            "metadata",
            applied.authority.epoch,
            applied.authority.revision,
            "title",
            "summary",
            message_id,
        )?;
        let updated = store.update_managed_conversation(update.clone()).await?;
        assert_eq!(
            updated.status,
            ManagedConversationMetadataUpdateStatus::Updated
        );
        assert_eq!(
            store
                .update_managed_conversation(update.clone())
                .await?
                .status,
            ManagedConversationMetadataUpdateStatus::AlreadyUpdated
        );
        let mut tampered = update.clone();
        tampered.summary = Some("tampered".to_string());
        assert!(store.update_managed_conversation(tampered).await.is_err());
        let stale = store
            .update_managed_conversation(managed_metadata_update(
                "metadata",
                updated.authority.epoch,
                applied.authority.revision,
                "stale",
                "stale",
                message_id,
            )?)
            .await?;
        assert_eq!(
            stale.status,
            ManagedConversationMetadataUpdateStatus::RevisionConflict
        );
        let conversation = store
            .get_conversation("metadata")
            .await?
            .ok_or_else(|| std::io::Error::other("managed conversation disappeared"))?;
        assert_eq!(conversation.title.as_deref(), Some("title"));
        assert_eq!(conversation.summary.as_deref(), Some("summary"));
        assert_eq!(conversation.compressed_before_id, Some(message_id));

        let imported = store
            .import_managed_messages(managed_import(
                "metadata",
                updated.authority.epoch,
                updated.authority.revision,
                &["replacement"],
            )?)
            .await?;
        let after_import = store
            .get_conversation("metadata")
            .await?
            .ok_or_else(|| std::io::Error::other("conversation disappeared after import"))?;
        assert!(after_import.summary.is_none());
        assert!(after_import.compressed_before_id.is_none());
        assert_eq!(
            store.update_managed_conversation(update).await?.status,
            ManagedConversationMetadataUpdateStatus::EpochConflict
        );
        let replacement_id = store
            .get_messages("metadata")
            .await?
            .first()
            .and_then(|message| message.id)
            .ok_or_else(|| std::io::Error::other("imported message id was not assigned"))?;
        let after_import = managed_metadata_update(
            "metadata",
            imported.authority.epoch,
            imported.authority.revision,
            "new-title",
            "new-summary",
            replacement_id,
        )?;
        let updated_again = store
            .update_managed_conversation(after_import.clone())
            .await?;
        let deleted = store
            .delete_managed_conversation(managed_delete("metadata", updated_again.authority.epoch)?)
            .await?;
        assert_eq!(deleted.status, ManagedConversationDeleteStatus::Deleted);
        let recreated = store
            .ensure_projection_epoch(ensure_projection_request(
                "metadata",
                Some(updated_again.authority.epoch),
            ))
            .await?;
        assert_eq!(
            store
                .update_managed_conversation(after_import)
                .await?
                .status,
            ManagedConversationMetadataUpdateStatus::EpochConflict
        );
        assert!(is_managed_projection_error(
            store
                .update_conversation("metadata", Some("late"), Some("late"), Some(1))
                .await
        ));
        let conversation = store
            .get_conversation("metadata")
            .await?
            .ok_or_else(|| std::io::Error::other("recreated conversation disappeared"))?;
        assert_eq!(recreated.authority.epoch, 3);
        assert!(conversation.title.is_none());
        assert!(conversation.summary.is_none());
        assert!(conversation.compressed_before_id.is_none());
        std::fs::remove_dir_all(base)?;
        Ok(())
    }

    /// 验证：托管删除回执能围栏迟到的写入，并在会话重建后依然保留。
    #[tokio::test]
    async fn managed_delete_receipt_fences_late_effects_and_survives_recreate() -> TestResult {
        let base = tmp_base();
        let store = FileConversationStore::new(&base)?;
        let acquired = store
            .ensure_projection_epoch(ensure_projection_request("delete-fence", None))
            .await?;
        let batch = projection_batch("delete-fence", 1, "generation-a", 0, &["before"])?;
        assert_eq!(
            store
                .apply_transcript_projection(batch.clone())
                .await?
                .status,
            TranscriptProjectionApplyStatus::Applied
        );
        let delete = managed_delete("delete-fence", acquired.authority.epoch)?;
        let deleted = store.delete_managed_conversation(delete.clone()).await?;
        assert_eq!(deleted.status, ManagedConversationDeleteStatus::Deleted);
        let repeated = store.delete_managed_conversation(delete.clone()).await?;
        assert_eq!(
            repeated.status,
            ManagedConversationDeleteStatus::AlreadyDeleted
        );
        let mut tampered_delete = delete.clone();
        tampered_delete.payload_digest = "different-delete-payload".to_string();
        assert!(
            store
                .delete_managed_conversation(tampered_delete)
                .await
                .is_err()
        );
        assert!(store.get_conversation("delete-fence").await?.is_none());
        assert!(
            store
                .ensure_projection_epoch(ensure_projection_request("delete-fence", None))
                .await?
                .status
                == ConversationProjectionEpochStatus::Tombstoned
        );

        let recreated = store
            .ensure_projection_epoch(ensure_projection_request("delete-fence", Some(1)))
            .await?;
        assert_eq!(
            recreated.status,
            ConversationProjectionEpochStatus::Recreated
        );
        assert_eq!(recreated.authority.epoch, 2);
        assert_eq!(
            store.delete_managed_conversation(delete).await?.status,
            ManagedConversationDeleteStatus::AlreadyDeleted
        );
        assert!(store.get_conversation("delete-fence").await?.is_some());

        let stale_delete = store
            .delete_managed_conversation(managed_delete("delete-fence", 3)?)
            .await?;
        assert_eq!(
            stale_delete.status,
            ManagedConversationDeleteStatus::EpochConflict
        );
        let late_apply = store.apply_transcript_projection(batch).await?;
        assert!(matches!(
            late_apply.status,
            TranscriptProjectionApplyStatus::Fenced {
                current_epoch: 2,
                ..
            }
        ));
        assert!(store.get_conversation("delete-fence").await?.is_some());
        std::fs::remove_dir_all(base)?;
        Ok(())
    }

    /// 验证：并发写入的多个 generation 合并时不会丢失更新。
    #[tokio::test]
    async fn concurrent_generations_merge_without_lost_updates() -> TestResult {
        let base = tmp_base();
        let store = FileConversationStore::new(&base)?;
        store
            .ensure_projection_epoch(ensure_projection_request("concurrent", None))
            .await?;
        let first_store = store.clone();
        let second_store = store.clone();
        let first = projection_batch("concurrent", 1, "generation-a", 0, &["a"])?;
        let second = projection_batch("concurrent", 1, "generation-b", 0, &["b"])?;
        let (first, second) = tokio::join!(
            first_store.apply_transcript_projection(first),
            second_store.apply_transcript_projection(second),
        );
        assert_eq!(first?.status, TranscriptProjectionApplyStatus::Applied);
        assert_eq!(second?.status, TranscriptProjectionApplyStatus::Applied);
        let mut contents = store
            .get_messages("concurrent")
            .await?
            .into_iter()
            .filter_map(|message| message.content)
            .collect::<Vec<_>>();
        contents.sort();
        assert_eq!(contents, vec!["a", "b"]);
        std::fs::remove_dir_all(base)?;
        Ok(())
    }

    /// 验证：托管清单提交失败时不会留下部分生效的副作用。
    #[tokio::test]
    async fn managed_manifest_publish_failures_leave_no_partial_effect() -> TestResult {
        let base = tmp_base();
        let store = FileConversationStore::new(&base)?;
        let acquired = store
            .ensure_projection_epoch(ensure_projection_request("fault", None))
            .await?;
        let batch = projection_batch("fault", acquired.authority.epoch, "generation-a", 0, &["a"])?;

        let blocked_meta = block_meta_writes(&store)?;
        assert!(
            store
                .apply_transcript_projection(batch.clone())
                .await
                .is_err()
        );
        unblock_meta_writes(&blocked_meta)?;
        assert_eq!(store.count_messages("fault").await?, 0);
        let after_failed_apply = store
            .ensure_projection_epoch(ensure_projection_request("fault", None))
            .await?;
        assert_eq!(after_failed_apply.authority, acquired.authority);
        let applied = store.apply_transcript_projection(batch).await?;
        assert_eq!(applied.status, TranscriptProjectionApplyStatus::Applied);

        let pre_import_message_id = store
            .get_messages("fault")
            .await?
            .first()
            .and_then(|message| message.id)
            .ok_or_else(|| std::io::Error::other("pre-import message id was not assigned"))?;
        let pre_import_metadata = store
            .update_managed_conversation(managed_metadata_update(
                "fault",
                applied.authority.epoch,
                applied.authority.revision,
                "pre-import-title",
                "pre-import-summary",
                pre_import_message_id,
            )?)
            .await?;
        let import = managed_import(
            "fault",
            pre_import_metadata.authority.epoch,
            pre_import_metadata.authority.revision,
            &["imported"],
        )?;
        let blocked_meta = block_meta_writes(&store)?;
        assert!(store.import_managed_messages(import.clone()).await.is_err());
        unblock_meta_writes(&blocked_meta)?;
        assert_eq!(
            store
                .get_messages("fault")
                .await?
                .first()
                .and_then(|message| message.content.as_deref()),
            Some("a")
        );
        let after_failed_import_metadata = store
            .get_conversation("fault")
            .await?
            .ok_or_else(|| std::io::Error::other("conversation disappeared after import fault"))?;
        assert_eq!(
            after_failed_import_metadata.summary.as_deref(),
            Some("pre-import-summary")
        );
        assert_eq!(
            after_failed_import_metadata.compressed_before_id,
            Some(pre_import_message_id)
        );
        let after_failed_import = store
            .ensure_projection_epoch(ensure_projection_request("fault", None))
            .await?;
        assert_eq!(after_failed_import.authority, pre_import_metadata.authority);
        let imported = store.import_managed_messages(import).await?;
        assert_eq!(imported.status, TranscriptProjectionApplyStatus::Applied);
        let after_import_metadata = store
            .get_conversation("fault")
            .await?
            .ok_or_else(|| std::io::Error::other("conversation disappeared after import"))?;
        assert!(after_import_metadata.summary.is_none());
        assert!(after_import_metadata.compressed_before_id.is_none());

        let imported_message_id = store
            .get_messages("fault")
            .await?
            .first()
            .and_then(|message| message.id)
            .ok_or_else(|| std::io::Error::other("imported message id was not assigned"))?;
        let metadata = managed_metadata_update(
            "fault",
            imported.authority.epoch,
            imported.authority.revision,
            "metadata-title",
            "metadata-summary",
            imported_message_id,
        )?;
        store.fail_next_manifest_write()?;
        assert!(
            store
                .update_managed_conversation(metadata.clone())
                .await
                .is_err()
        );
        let after_failed_metadata = store.get_conversation("fault").await?.ok_or_else(|| {
            std::io::Error::other("conversation disappeared after metadata fault")
        })?;
        assert_eq!(
            after_failed_metadata.title.as_deref(),
            Some("pre-import-title")
        );
        assert!(after_failed_metadata.summary.is_none());
        assert!(after_failed_metadata.compressed_before_id.is_none());
        let after_failed_metadata_authority = store
            .ensure_projection_epoch(ensure_projection_request("fault", None))
            .await?;
        assert_eq!(
            after_failed_metadata_authority.authority,
            imported.authority
        );
        let metadata_updated = store.update_managed_conversation(metadata).await?;
        assert_eq!(
            metadata_updated.status,
            ManagedConversationMetadataUpdateStatus::Updated
        );

        let delete = managed_delete("fault", metadata_updated.authority.epoch)?;
        let blocked_meta = block_meta_writes(&store)?;
        assert!(
            store
                .delete_managed_conversation(delete.clone())
                .await
                .is_err()
        );
        unblock_meta_writes(&blocked_meta)?;
        assert!(store.get_conversation("fault").await?.is_some());
        assert_eq!(
            store
                .get_messages("fault")
                .await?
                .first()
                .and_then(|message| message.content.as_deref()),
            Some("imported")
        );
        let after_failed_delete = store
            .ensure_projection_epoch(ensure_projection_request("fault", None))
            .await?;
        assert_eq!(after_failed_delete.authority, metadata_updated.authority);
        assert_eq!(
            store.delete_managed_conversation(delete).await?.status,
            ManagedConversationDeleteStatus::Deleted
        );
        std::fs::remove_dir_all(base)?;
        Ok(())
    }

    /// 验证：删除回执的保留下限可阻止已淘汰回执被重放。
    #[tokio::test]
    async fn delete_receipt_retention_floor_prevents_expired_replay() -> TestResult {
        let base = tmp_base();
        let store = FileConversationStore::new(&base)?;
        store
            .ensure_projection_epoch(ensure_projection_request("retention", None))
            .await?;
        let cycles = DELETE_RECEIPT_RETENTION.saturating_add(1);
        for offset in 0..cycles {
            let epoch = u64::try_from(offset)?.saturating_add(1);
            let receipt = store
                .delete_managed_conversation(managed_delete("retention", epoch)?)
                .await?;
            assert_eq!(receipt.status, ManagedConversationDeleteStatus::Deleted);
            if offset.saturating_add(1) < cycles {
                let recreated = store
                    .ensure_projection_epoch(ensure_projection_request("retention", Some(epoch)))
                    .await?;
                assert_eq!(
                    recreated.status,
                    ConversationProjectionEpochStatus::Recreated
                );
            }
        }

        let expired = store
            .delete_managed_conversation(managed_delete("retention", 1)?)
            .await?;
        assert_eq!(
            expired.status,
            ManagedConversationDeleteStatus::ReceiptExpired
        );
        assert!(expired.retention_floor_epoch >= 1);
        let last_epoch = u64::try_from(cycles)?;
        let live = store
            .ensure_projection_epoch(ensure_projection_request("retention", Some(last_epoch)))
            .await?;
        let mut tampered_expired = managed_delete("retention", 1)?;
        tampered_expired.expected_epoch = live.authority.epoch;
        assert!(
            store
                .delete_managed_conversation(tampered_expired)
                .await
                .is_err()
        );
        assert!(store.get_conversation("retention").await?.is_some());
        std::fs::remove_dir_all(base)?;
        Ok(())
    }

    /// 验证：epoch 已达上限的投影不能再被重建。
    #[tokio::test]
    async fn maximum_projection_epoch_cannot_be_recreated() -> TestResult {
        let base = tmp_base();
        let store = FileConversationStore::new(&base)?;
        store
            .ensure_projection_epoch(ensure_projection_request("epoch-max", None))
            .await?;
        let mut record = manifest(&store, "epoch-max")?;
        let state = record
            .projection
            .as_mut()
            .ok_or_else(|| std::io::Error::other("projection state was not persisted"))?;
        state.epoch = MAX_PROJECTION_EPOCH;
        state.lifecycle = ConversationProjectionLifecycle::Deleted;
        store.write_manifest(&record)?;

        let result = store
            .ensure_projection_epoch(ensure_projection_request(
                "epoch-max",
                Some(MAX_PROJECTION_EPOCH),
            ))
            .await;
        assert!(matches!(
            result,
            Err(echo_core::error::ReactError::Memory(error))
                if matches!(error.as_ref(), MemoryError::ProjectionEpochExhausted(_))
        ));
        std::fs::remove_dir_all(base)?;
        Ok(())
    }

    /// 验证：托管操作回执在重新打开存储后依然有效。
    #[tokio::test]
    async fn managed_receipts_survive_store_reopen() -> TestResult {
        let base = tmp_base();
        let batch = projection_batch("reopen", 1, "generation-a", 0, &["durable"])?;
        let delete = managed_delete("reopen", 1)?;
        {
            let store = FileConversationStore::new(&base)?;
            store
                .ensure_projection_epoch(ensure_projection_request("reopen", None))
                .await?;
            assert_eq!(
                store
                    .apply_transcript_projection(batch.clone())
                    .await?
                    .status,
                TranscriptProjectionApplyStatus::Applied
            );
        }
        {
            let reopened = FileConversationStore::new(&base)?;
            assert_eq!(
                reopened.apply_transcript_projection(batch).await?.status,
                TranscriptProjectionApplyStatus::AlreadyApplied
            );
            assert_eq!(
                reopened
                    .delete_managed_conversation(delete.clone())
                    .await?
                    .status,
                ManagedConversationDeleteStatus::Deleted
            );
        }
        let reopened = FileConversationStore::new(&base)?;
        assert_eq!(
            reopened.delete_managed_conversation(delete).await?.status,
            ManagedConversationDeleteStatus::AlreadyDeleted
        );
        assert_eq!(
            reopened
                .ensure_projection_epoch(ensure_projection_request("reopen", Some(1)))
                .await?
                .status,
            ConversationProjectionEpochStatus::Recreated
        );
        drop(reopened);
        std::fs::remove_dir_all(base)?;
        Ok(())
    }

    /// 读取指定会话的清单，不存在时返回错误。
    fn manifest(
        store: &FileConversationStore,
        conversation_id: &str,
    ) -> TestResult<ConversationRecord> {
        store.read_manifest(conversation_id)?.ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("missing conversation manifest: {conversation_id}"),
            )
            .into()
        })
    }

    /// 取出记录中的消息日志元数据，缺失时返回错误。
    fn log_meta(record: &ConversationRecord) -> TestResult<MessageLogMeta> {
        record
            .message_log
            .clone()
            .ok_or_else(|| std::io::Error::other("missing message-log metadata").into())
    }

    /// 把 `_meta.json` 替换为同名目录，使后续 meta 写入失败；返回该路径。
    fn block_meta_writes(store: &FileConversationStore) -> TestResult<PathBuf> {
        let path = FileConversationStore::meta_path(&store.base);
        match std::fs::remove_file(&path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
        std::fs::create_dir(&path)?;
        Ok(path)
    }

    /// 移除阻塞用的目录，恢复 meta 写入。
    fn unblock_meta_writes(path: &Path) -> std::io::Result<()> {
        std::fs::remove_dir(path)
    }

    /// 验证：文件操作在 tokio 运行时线程之外的阻塞线程上执行。
    #[tokio::test(flavor = "current_thread")]
    async fn file_operations_run_outside_the_tokio_runtime_thread() -> TestResult {
        let base = tmp_base();
        let store = FileConversationStore::new(&base)?;
        let runtime_thread = std::thread::current().id();
        let io_thread = store
            .run_blocking(
                FileConversationStore::conversation_scope("thread-check"),
                |_| Ok(std::thread::current().id()),
            )
            .await?;

        assert_ne!(io_thread, runtime_thread);
        std::fs::remove_dir_all(base)?;
        Ok(())
    }

    /// 验证：已受理的创建操作在调用方中止后仍会完成，且不会阻塞运行时。
    #[tokio::test(flavor = "current_thread")]
    async fn accepted_create_survives_caller_abort_without_stalling_runtime() -> TestResult {
        let base = tmp_base();
        let store = FileConversationStore::new(&base)?;
        let caller_store = store.clone();
        let (entered_tx, entered_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let caller = tokio::spawn(async move {
            caller_store
                .run_blocking(
                    FileConversationStore::conversation_scope("abort-create"),
                    move |store| {
                        let _conversation = store.authority.scan_barrier.read().map_err(poison)?;
                        let _ignored = entered_tx.send(());
                        release_rx
                            .recv_timeout(Duration::from_secs(2))
                            .map_err(|error| {
                                MemoryError::IoError(format!("release create: {error}"))
                            })?;
                        store.create_conversation_sync(new_conv("abort-create", Some("durable")))
                    },
                )
                .await
        });
        entered_rx.await?;
        tokio::time::timeout(Duration::from_millis(250), async {
            for _ in 0..64 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .map_err(|_| std::io::Error::other("runtime heartbeat stalled"))?;
        caller.abort();
        release_tx.send(())?;

        let created = tokio::time::timeout(
            Duration::from_secs(2),
            store.get_conversation("abort-create"),
        )
        .await
        .map_err(|_| std::io::Error::other("accepted create did not settle"))??
        .ok_or_else(|| std::io::Error::other("accepted create was cancelled"))?;
        assert_eq!(created.title.as_deref(), Some("durable"));
        std::fs::remove_dir_all(base)?;
        Ok(())
    }

    /// 验证：读取损坏记录与删除操作竞争时失败关闭。
    #[tokio::test(flavor = "current_thread")]
    async fn corrupt_read_and_delete_race_fails_closed() -> TestResult {
        let base = tmp_base();
        let store = FileConversationStore::new(&base)?;
        store
            .create_conversation(new_conv("corrupt-race", None))
            .await?;
        let manifest_path = store.conv_path("corrupt-race")?;
        std::fs::write(&manifest_path, b"{ invalid json")?;

        let blocker_store = store.clone();
        let (entered_tx, entered_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let blocker = tokio::spawn(async move {
            blocker_store
                .run_blocking(
                    FileConversationStore::conversation_scope("corrupt-race"),
                    move |_| {
                        let _ignored = entered_tx.send(());
                        release_rx
                            .recv_timeout(Duration::from_secs(2))
                            .map_err(|error| {
                                MemoryError::IoError(format!("release corrupt race: {error}"))
                            })?;
                        Ok(())
                    },
                )
                .await
        });
        entered_rx.await?;
        let read_store = store.clone();
        let read = tokio::spawn(async move { read_store.get_conversation("corrupt-race").await });
        tokio::task::yield_now().await;
        let delete_store = store.clone();
        let delete =
            tokio::spawn(async move { delete_store.delete_conversation("corrupt-race").await });
        tokio::task::yield_now().await;
        release_tx.send(())?;

        blocker.await??;
        assert!(read.await?.is_err());
        assert!(delete.await?.is_err());
        assert!(manifest_path.exists());
        std::fs::remove_dir_all(base)?;
        Ok(())
    }

    /// 验证：精确的 UTF-8 id 在大小写不敏感的文件系统上不会互相混淆。
    #[tokio::test]
    async fn exact_utf8_ids_do_not_alias_on_case_folding_filesystems() -> TestResult {
        let base = tmp_base();
        let store = FileConversationStore::new(&base)?;
        let ids = ["A", "a", "é", "e\u{301}"];
        let paths = ids
            .iter()
            .map(|id| store.conv_path(id))
            .collect::<Result<Vec<_>>>()?;
        let unique = paths.iter().collect::<HashSet<_>>();
        assert_eq!(unique.len(), paths.len());

        let (upper, lower, composed, decomposed) = tokio::join!(
            store.create_conversation(new_conv("A", Some("upper"))),
            store.create_conversation(new_conv("a", Some("lower"))),
            store.create_conversation(new_conv("é", Some("composed"))),
            store.create_conversation(new_conv("e\u{301}", Some("decomposed"))),
        );
        upper?;
        lower?;
        composed?;
        decomposed?;
        let upper_messages = [stored("user", "upper")];
        let lower_messages = [stored("user", "lower")];
        let composed_messages = [stored("user", "composed")];
        let decomposed_messages = [stored("user", "decomposed")];
        let (upper, lower, composed, decomposed) = tokio::join!(
            store.save_messages("A", &upper_messages),
            store.save_messages("a", &lower_messages),
            store.save_messages("é", &composed_messages),
            store.save_messages("e\u{301}", &decomposed_messages),
        );
        upper?;
        lower?;
        composed?;
        decomposed?;

        for (id, expected) in [
            ("A", "upper"),
            ("a", "lower"),
            ("é", "composed"),
            ("e\u{301}", "decomposed"),
        ] {
            let conversation = store
                .get_conversation(id)
                .await?
                .ok_or_else(|| std::io::Error::other("aliased conversation is missing"))?;
            assert_eq!(conversation.conversation_id, id);
            let messages = store.get_messages(id).await?;
            assert_eq!(
                messages
                    .first()
                    .and_then(|message| message.content.as_deref()),
                Some(expected)
            );
        }
        std::fs::remove_dir_all(base)?;
        Ok(())
    }

    /// 验证：不同会话的操作可在并行的阻塞槽位中执行。
    #[tokio::test]
    async fn different_conversations_run_in_parallel_blocking_slots() -> TestResult {
        let base = tmp_base();
        let store = FileConversationStore::new(&base)?;
        let first_store = store.clone();
        let second_store = store.clone();
        let (first_entered_tx, first_entered_rx) = tokio::sync::oneshot::channel();
        let (second_entered_tx, second_entered_rx) = tokio::sync::oneshot::channel();
        let (first_release_tx, first_release_rx) = std::sync::mpsc::channel();
        let (second_release_tx, second_release_rx) = std::sync::mpsc::channel();
        let first = tokio::spawn(async move {
            first_store
                .run_blocking(
                    FileConversationStore::conversation_scope("parallel-first"),
                    move |_| {
                        let _ignored = first_entered_tx.send(());
                        first_release_rx
                            .recv_timeout(Duration::from_secs(2))
                            .map_err(|error| {
                                MemoryError::IoError(format!("release first: {error}"))
                            })?;
                        Ok(())
                    },
                )
                .await
        });
        let second = tokio::spawn(async move {
            second_store
                .run_blocking(
                    FileConversationStore::conversation_scope("parallel-second"),
                    move |_| {
                        let _ignored = second_entered_tx.send(());
                        second_release_rx
                            .recv_timeout(Duration::from_secs(2))
                            .map_err(|error| {
                                MemoryError::IoError(format!("release second: {error}"))
                            })?;
                        Ok(())
                    },
                )
                .await
        });
        tokio::time::timeout(Duration::from_secs(2), async {
            first_entered_rx.await?;
            second_entered_rx.await
        })
        .await
        .map_err(|_| std::io::Error::other("distinct conversations were serialized"))??;
        first_release_tx.send(())?;
        second_release_tx.send(())?;
        first.await??;
        second.await??;
        std::fs::remove_dir_all(base)?;
        Ok(())
    }

    /// 验证：替换操作会等待仍在使用旧 generation 的搜索快照完成。
    #[tokio::test(flavor = "current_thread")]
    async fn replacement_waits_for_search_snapshot_using_old_generation() -> TestResult {
        let base = tmp_base();
        let store = FileConversationStore::new(&base)?;
        store.create_conversation(new_conv("c1", None)).await?;
        store.save_messages("c1", &[stored("user", "old")]).await?;
        let old_log = log_meta(&manifest(&store, "c1")?)?;
        let old_log_path = store.message_log_path("c1", old_log.generation)?;
        let (captured_tx, captured_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        store.install_search_snapshot_hook(captured_tx, release_rx)?;
        let scan_store = store.clone();
        let scan = tokio::spawn(async move { scan_store.search_conversations("old", 1).await });
        captured_rx.await?;
        let replacement_store = store.clone();
        let replacement = tokio::spawn(async move {
            replacement_store
                .save_messages("c1", &[stored("assistant", "replacement")])
                .await
        });
        tokio::task::yield_now().await;
        assert!(!replacement.is_finished());
        assert!(old_log_path.exists());
        release_tx.send(())?;
        let old_results = scan.await??;
        assert_eq!(
            old_results
                .first()
                .map(|conversation| conversation.conversation_id.as_str()),
            Some("c1")
        );
        replacement.await??;
        assert!(!old_log_path.exists());
        let found = store.search_conversations("replacement", 1).await?;
        assert_eq!(
            found
                .first()
                .map(|conversation| conversation.conversation_id.as_str()),
            Some("c1")
        );
        std::fs::remove_dir_all(base)?;
        Ok(())
    }

    /// 验证：独立构造的多个句柄共享权威状态，并分配唯一 id。
    #[tokio::test]
    async fn independent_constructors_share_authority_and_allocate_unique_ids() -> TestResult {
        let base = tmp_base();
        let first_base = base.clone();
        let second_base = base.clone();
        let (first, second) = tokio::join!(
            tokio::task::spawn_blocking(move || FileConversationStore::new(first_base)),
            tokio::task::spawn_blocking(move || FileConversationStore::new(second_base)),
        );
        let first = first??;
        let second = second??;
        assert!(Arc::ptr_eq(&first.authority, &second.authority));

        let (first_created, second_created) = tokio::join!(
            first.create_conversation(new_conv("first", None)),
            second.create_conversation(new_conv("second", None)),
        );
        let first_created = first_created?;
        let second_created = second_created?;
        assert_ne!(first_created.id, second_created.id);
        drop(first);
        drop(second);

        let reopened = FileConversationStore::new(&base)?;
        let listed = reopened
            .list_conversations(ConversationFilter::default())
            .await?;
        assert_eq!(listed.len(), 2);
        drop(reopened);
        std::fs::remove_dir_all(base)?;
        Ok(())
    }

    /// 验证：会话的增删改查与搜索基本流程。
    #[tokio::test]
    async fn conversation_crud_and_search() {
        let base = tmp_base();
        let store = FileConversationStore::new(&base).unwrap();

        store
            .create_conversation(new_conv("c1", Some("rust tokio help")))
            .await
            .unwrap();
        store
            .create_conversation(new_conv("c2", Some("python asyncio")))
            .await
            .unwrap();

        let list = store
            .list_conversations(ConversationFilter {
                limit: Some(10),
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(list.len(), 2);

        store
            .save_messages("c1", &[stored("user", "how do I use tokio")])
            .await
            .unwrap();
        let msgs = store.get_messages("c1").await.unwrap();
        assert_eq!(msgs.len(), 1);
        assert!(msgs[0].id.is_some());

        assert_eq!(store.count_messages("c1").await.unwrap(), 1);

        let found = store.search_conversations("tokio", 10).await.unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].conversation_id, "c1");

        let found = store.search_conversations("python", 10).await.unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].conversation_id, "c2");

        store
            .update_conversation("c1", Some("renamed"), None, None)
            .await
            .unwrap();
        let conv = store.get_conversation("c1").await.unwrap().unwrap();
        assert_eq!(conv.title.as_deref(), Some("renamed"));

        store.delete_conversation("c2").await.unwrap();
        assert!(store.get_conversation("c2").await.unwrap().is_none());

        let _ = std::fs::remove_dir_all(&base);
    }

    /// 验证：损坏的记录以错误形式暴露，而不是返回空结果。
    #[tokio::test]
    async fn corrupt_record_surfaces_as_error_not_empty()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let base = tmp_base();
        let store = FileConversationStore::new(&base)?;
        store
            .create_conversation(new_conv("c1", Some("title")))
            .await?;
        // 在磁盘上破坏该记录文件。
        let path = store.conv_path("c1")?;
        std::fs::write(&path, b"{ not valid json")?;

        let Err(err) = store.get_conversation("c1").await else {
            let failure: Box<dyn std::error::Error> = Box::new(std::io::Error::other(
                "corrupt conversation record unexpectedly loaded",
            ));
            return Err(failure);
        };
        assert!(
            matches!(err, echo_core::error::ReactError::Memory(_)),
            "expected a Memory error, got {err:?}"
        );
        std::fs::remove_dir_all(&base)?;
        Ok(())
    }

    /// 验证：包含路径穿越的会话 id 会被拒绝。
    #[tokio::test]
    async fn path_traversal_id_is_rejected() {
        let base = tmp_base();
        let store = FileConversationStore::new(&base).unwrap();
        let err = store.get_conversation("../escape").await.unwrap_err();
        assert!(matches!(err, echo_core::error::ReactError::Memory(_)));
        // 确认没有在根目录之外创建任何文件。
        assert!(!base.parent().unwrap().join("escape.json").exists());
        let _ = std::fs::remove_dir_all(&base);
    }

    /// 验证：消息经保存与读取后能完整往返。
    #[tokio::test]
    async fn round_trip_messages_via_project_and_restore() {
        use crate::memory::{project_messages, restore_messages};
        use echo_core::llm::types::{
            ContentPart, FunctionCall, ImageUrl, Message, MessageContent, Role, ToolCall,
        };

        let base = tmp_base();
        let store = FileConversationStore::new(&base).unwrap();
        store
            .create_conversation(new_conv("c1", Some("rt")))
            .await
            .unwrap();

        let mut multimodal = Message::user("placeholder".to_string());
        multimodal.content = MessageContent::Parts(vec![
            ContentPart::Text {
                text: "inspect ".to_string(),
            },
            ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: "data:image/png;base64,AA==".to_string(),
                    detail: Some("low".to_string()),
                },
            },
        ]);
        multimodal.reasoning_content = Some("reasoning trace".to_string());
        let original = vec![
            Message::system("be helpful".into()),
            multimodal,
            Message::assistant_with_tools(vec![ToolCall {
                id: "call-1".into(),
                call_type: "function".into(),
                function: FunctionCall {
                    name: "search".into(),
                    arguments: r#"{"q":"rust"}"#.to_string(),
                },
            }]),
            Message::tool_result("call-1".into(), "search".into(), "result".into()),
        ];
        let projected = project_messages("c1", &original).unwrap();
        store.save_messages("c1", &projected).await.unwrap();

        let loaded = store.get_messages("c1").await.unwrap();
        let restored = restore_messages(&loaded).unwrap();

        assert_eq!(restored.len(), original.len());
        assert_eq!(restored[0].role, Role::System);
        assert_eq!(restored[1].role, Role::User);
        assert!(matches!(&restored[1].content, MessageContent::Parts(_)));
        assert_eq!(
            restored[1].reasoning_content.as_deref(),
            Some("reasoning trace")
        );
        assert_eq!(restored[2].role, Role::Assistant);
        assert_eq!(restored[2].tool_calls.as_ref().unwrap().len(), 1);
        assert_eq!(restored[2].tool_calls.as_ref().unwrap()[0].id, "call-1");
        assert_eq!(
            restored[2].tool_calls.as_ref().unwrap()[0].function.name,
            "search"
        );
        assert_eq!(restored[3].role, Role::Tool);
        assert_eq!(restored[3].tool_call_id.as_deref(), Some("call-1"));

        let _ = std::fs::remove_dir_all(&base);
    }

    /// 验证：过期的 `_meta.json` 会根据已有记录进行协调。
    #[tokio::test]
    async fn stale_meta_is_reconciled_from_records() {
        let base = tmp_base();
        let first_id = {
            let store = FileConversationStore::new(&base).unwrap();
            let conversation = store
                .create_conversation(new_conv("first", None))
                .await
                .unwrap();
            store
                .save_messages("first", &[stored("user", "hello")])
                .await
                .unwrap();
            conversation.id
        };
        std::fs::write(
            base.join("conversations").join("_meta.json"),
            r#"{"next_id":0}"#,
        )
        .unwrap();

        let reopened = FileConversationStore::new(&base).unwrap();
        let second = reopened
            .create_conversation(new_conv("second", None))
            .await
            .unwrap();
        assert!(second.id > first_id.saturating_add(1));

        let _ = std::fs::remove_dir_all(&base);
    }

    /// 验证：调用方提供的消息 id 会推进运行中的 id 计数器。
    #[tokio::test]
    async fn supplied_message_ids_advance_the_live_counter() -> Result<()> {
        let base = tmp_base();
        let store = FileConversationStore::new(&base)?;
        store.create_conversation(new_conv("c1", None)).await?;

        let mut imported = stored("user", "imported");
        imported.id = Some(1_000);
        store.save_messages("c1", &[imported]).await?;
        store
            .save_messages("c1", &[stored("assistant", "next")])
            .await?;

        let saved = store.get_messages("c1").await?;
        assert_eq!(saved.first().and_then(|message| message.id), Some(1_001));
        let _ = std::fs::remove_dir_all(&base);
        Ok(())
    }

    /// 验证：重复保存完整消息列表时只追加新增的后缀。
    #[tokio::test]
    async fn repeated_full_projection_writes_only_the_new_suffix() -> TestResult {
        let base = tmp_base();
        let store = FileConversationStore::new(&base)?;
        store.create_conversation(new_conv("c1", None)).await?;

        let mut transcript = Vec::new();
        let mut previous_bytes = 0_u64;
        for turn in 0..64 {
            transcript.push(stored("user", &format!("第 {turn} 轮 🧪")));
            let mut projection = transcript.clone();
            for message in &mut projection {
                message.id = None;
                message.created_at = format!("projection-{turn}");
            }
            store.save_messages("c1", &projection).await?;

            let log = log_meta(&manifest(&store, "c1")?)?;
            assert_eq!(log.generation, 1);
            assert_eq!(log.message_count, turn + 1);
            let current_bytes = std::fs::metadata(store.message_log_path("c1", 1)?)?.len();
            assert_eq!(current_bytes, log.committed_bytes);
            let delta = current_bytes.saturating_sub(previous_bytes);
            assert!(
                delta > 0 && delta < 1_024,
                "unexpected append delta: {delta}"
            );
            previous_bytes = current_bytes;
        }

        let loaded = store.get_messages("c1").await?;
        assert_eq!(loaded.len(), 64);
        assert_eq!(
            loaded.first().map(|message| message.created_at.as_str()),
            Some("projection-0")
        );
        let ids = loaded
            .iter()
            .filter_map(|message| message.id)
            .collect::<Vec<_>>();
        assert_eq!(ids.len(), 64);
        assert!(ids.windows(2).all(|pair| {
            pair.first()
                .zip(pair.get(1))
                .is_some_and(|(left, right)| left < right)
        }));
        std::fs::remove_dir_all(base)?;
        Ok(())
    }

    /// 验证：创建提交前 meta 写入失败时，会话保持不可见。
    #[tokio::test]
    async fn meta_failure_before_create_publish_keeps_conversation_invisible() -> TestResult {
        let base = tmp_base();
        let store = FileConversationStore::new(&base)?;
        let meta_path = block_meta_writes(&store)?;

        assert!(
            store
                .create_conversation(new_conv("c1", Some("not published")))
                .await
                .is_err()
        );
        assert!(store.get_conversation("c1").await?.is_none());

        unblock_meta_writes(&meta_path)?;
        drop(store);
        let reopened = FileConversationStore::new(&base)?;
        assert!(reopened.get_conversation("c1").await?.is_none());
        assert!(!reopened.message_log_path("c1", 1)?.exists());
        std::fs::remove_dir_all(base)?;
        Ok(())
    }

    /// 验证：追加提交前 meta 写入失败时，新后缀保持未提交。
    #[tokio::test]
    async fn meta_failure_before_append_publish_keeps_suffix_uncommitted() -> TestResult {
        let base = tmp_base();
        let store = FileConversationStore::new(&base)?;
        store.create_conversation(new_conv("c1", None)).await?;
        let before = log_meta(&manifest(&store, "c1")?)?;
        let meta_path = block_meta_writes(&store)?;

        assert!(
            store
                .save_messages("c1", &[stored("user", "not committed")])
                .await
                .is_err()
        );
        assert_eq!(log_meta(&manifest(&store, "c1")?)?, before);
        assert!(store.get_messages("c1").await?.is_empty());

        unblock_meta_writes(&meta_path)?;
        store
            .save_messages("c1", &[stored("user", "committed after retry")])
            .await?;
        let loaded = store.get_messages("c1").await?;
        assert_eq!(
            loaded
                .first()
                .and_then(|message| message.content.as_deref()),
            Some("committed after retry")
        );
        std::fs::remove_dir_all(base)?;
        Ok(())
    }

    /// 验证：替换提交前 meta 写入失败时，旧 generation 保持不变。
    #[tokio::test]
    async fn meta_failure_before_replacement_publish_preserves_old_generation() -> TestResult {
        let base = tmp_base();
        let store = FileConversationStore::new(&base)?;
        store.create_conversation(new_conv("c1", None)).await?;
        store.save_messages("c1", &[stored("user", "old")]).await?;
        let before = log_meta(&manifest(&store, "c1")?)?;
        let uncommitted_generation = before
            .generation
            .checked_add(1)
            .ok_or_else(|| std::io::Error::other("test generation exhausted"))?;
        let uncommitted_path = store.message_log_path("c1", uncommitted_generation)?;
        let meta_path = block_meta_writes(&store)?;

        assert!(
            store
                .save_messages("c1", &[stored("assistant", "replacement")])
                .await
                .is_err()
        );
        assert_eq!(log_meta(&manifest(&store, "c1")?)?, before);
        assert_eq!(
            store
                .get_messages("c1")
                .await?
                .first()
                .and_then(|message| message.content.as_deref()),
            Some("old")
        );
        assert!(uncommitted_path.exists());

        unblock_meta_writes(&meta_path)?;
        drop(store);
        let reopened = FileConversationStore::new(&base)?;
        assert!(!uncommitted_path.exists());
        assert_eq!(
            reopened
                .get_messages("c1")
                .await?
                .first()
                .and_then(|message| message.content.as_deref()),
            Some("old")
        );
        std::fs::remove_dir_all(base)?;
        Ok(())
    }

    /// 验证：缩短、语义改写及显式时间戳变化都会触发 generation 替换。
    #[tokio::test]
    async fn shortening_semantic_rewrite_and_explicit_timestamp_change_replace_generation()
    -> TestResult {
        let base = tmp_base();
        let store = FileConversationStore::new(&base)?;
        store.create_conversation(new_conv("c1", None)).await?;
        store
            .save_messages(
                "c1",
                &[stored("user", "alpha"), stored("assistant", "beta")],
            )
            .await?;
        assert_eq!(log_meta(&manifest(&store, "c1")?)?.generation, 1);

        store
            .save_messages("c1", &[stored("user", "alpha")])
            .await?;
        assert_eq!(log_meta(&manifest(&store, "c1")?)?.generation, 2);
        let mut explicit = store
            .get_messages("c1")
            .await?
            .first()
            .cloned()
            .ok_or_else(|| std::io::Error::other("missing shortened message"))?;
        explicit.created_at = "explicitly-changed".to_string();
        store.save_messages("c1", &[explicit]).await?;
        assert_eq!(log_meta(&manifest(&store, "c1")?)?.generation, 3);

        store
            .save_messages("c1", &[stored("assistant", "same length, new semantics")])
            .await?;
        assert_eq!(log_meta(&manifest(&store, "c1")?)?.generation, 4);
        let loaded = store.get_messages("c1").await?;
        assert_eq!(loaded.len(), 1);
        assert_eq!(
            loaded
                .first()
                .and_then(|message| message.content.as_deref()),
            Some("same length, new semantics")
        );
        std::fs::remove_dir_all(base)?;
        Ok(())
    }

    /// 验证：未提交的追加尾部对读取不可见，并在下次追加时被修复。
    #[tokio::test]
    async fn uncommitted_append_tail_is_hidden_then_repaired() -> TestResult {
        let base = tmp_base();
        let store = FileConversationStore::new(&base)?;
        store.create_conversation(new_conv("c1", None)).await?;
        let first = stored("user", "committed");
        store
            .save_messages("c1", std::slice::from_ref(&first))
            .await?;
        let log = log_meta(&manifest(&store, "c1")?)?;
        let path = store.message_log_path("c1", log.generation)?;
        let tail = FileConversationStore::serialize_message_log(&[stored("assistant", "tail")])?;
        echo_core::utils::fs::append_existing(
            &path,
            &tail,
            echo_core::utils::fs::FileDurability::SyncData,
        )?;
        assert!(std::fs::metadata(&path)?.len() > log.committed_bytes);
        drop(store);

        let reopened = FileConversationStore::new(&base)?;
        assert_eq!(reopened.get_messages("c1").await?.len(), 1);
        reopened
            .save_messages("c1", &[first, stored("assistant", "committed next")])
            .await?;
        let repaired = log_meta(&manifest(&reopened, "c1")?)?;
        assert_eq!(std::fs::metadata(&path)?.len(), repaired.committed_bytes);
        assert_eq!(reopened.get_messages("c1").await?.len(), 2);
        std::fs::remove_dir_all(base)?;
        Ok(())
    }

    /// 验证：重新打开存储时会删除孤儿替换 generation 文件。
    #[tokio::test]
    async fn orphan_replacement_generation_is_removed_on_reopen() -> TestResult {
        let base = tmp_base();
        let store = FileConversationStore::new(&base)?;
        store.create_conversation(new_conv("c1", None)).await?;
        store.save_messages("c1", &[stored("user", "old")]).await?;
        let orphan = store.message_log_path("c1", 2)?;
        echo_core::utils::fs::atomic_write(
            &orphan,
            &FileConversationStore::serialize_message_log(&[stored("user", "uncommitted")])?,
        )?;
        let unrelated = base.join("conversations").join("keep.jsonl");
        std::fs::write(&unrelated, b"not a generation")?;
        drop(store);

        let reopened = FileConversationStore::new(&base)?;
        assert!(!orphan.exists());
        assert!(unrelated.exists());
        assert_eq!(
            reopened
                .get_messages("c1")
                .await?
                .first()
                .and_then(|message| message.content.as_deref()),
            Some("old")
        );
        std::fs::remove_dir_all(base)?;
        Ok(())
    }

    /// 验证：已提交日志被截断时失败关闭，且不做任何修改。
    #[tokio::test]
    async fn truncated_committed_log_fails_closed_without_mutation() -> TestResult {
        let base = tmp_base();
        let store = FileConversationStore::new(&base)?;
        store.create_conversation(new_conv("c1", None)).await?;
        store
            .save_messages("c1", &[stored("user", "durable")])
            .await?;
        let log = log_meta(&manifest(&store, "c1")?)?;
        let path = store.message_log_path("c1", log.generation)?;
        let truncated = log
            .committed_bytes
            .checked_sub(1)
            .ok_or_else(|| std::io::Error::other("message log was unexpectedly empty"))?;
        echo_core::utils::fs::truncate_existing(
            &path,
            truncated,
            echo_core::utils::fs::FileDurability::SyncData,
        )?;
        let before = std::fs::read(&path)?;
        assert!(store.get_messages("c1").await.is_err());
        assert_eq!(std::fs::read(&path)?, before);
        std::fs::remove_dir_all(base)?;
        Ok(())
    }

    /// 验证：重新打开后，过期的 meta 能从显式消息 id 中恢复计数器。
    #[tokio::test]
    async fn stale_meta_recovers_explicit_message_id_after_reopen() -> TestResult {
        let base = tmp_base();
        let store = FileConversationStore::new(&base)?;
        store.create_conversation(new_conv("c1", None)).await?;
        let mut imported = stored("user", "imported");
        imported.id = Some(10_000);
        store.save_messages("c1", &[imported]).await?;
        std::fs::write(
            base.join("conversations").join("_meta.json"),
            br#"{"next_id":0}"#,
        )?;
        drop(store);

        let reopened = FileConversationStore::new(&base)?;
        reopened
            .save_messages("c1", &[stored("assistant", "replacement")])
            .await?;
        assert_eq!(
            reopened
                .get_messages("c1")
                .await?
                .first()
                .and_then(|message| message.id),
            Some(10_001)
        );
        std::fs::remove_dir_all(base)?;
        Ok(())
    }

    /// 验证：旧版单文件记录可被读取，并在保存时迁移为追加式布局。
    #[tokio::test]
    async fn legacy_aggregate_record_is_read_then_migrated_on_save() -> TestResult {
        let base = tmp_base();
        let store = FileConversationStore::new(&base)?;
        store.create_conversation(new_conv("c1", None)).await?;
        let mut legacy = manifest(&store, "c1")?;
        let old_log = log_meta(&legacy)?;
        let old_log_path = store.message_log_path("c1", old_log.generation)?;
        let mut old_message = stored("user", "legacy");
        old_message.id = Some(50);
        legacy.messages = vec![old_message];
        legacy.message_log = None;
        legacy.search_filter = None;
        store.write_manifest(&legacy)?;
        std::fs::remove_file(old_log_path)?;
        drop(store);

        let reopened = FileConversationStore::new(&base)?;
        assert_eq!(
            reopened
                .get_messages("c1")
                .await?
                .first()
                .and_then(|message| message.content.as_deref()),
            Some("legacy")
        );
        reopened
            .save_messages(
                "c1",
                &[stored("user", "legacy"), stored("assistant", "new")],
            )
            .await?;
        let migrated = manifest(&reopened, "c1")?;
        assert!(migrated.messages.is_empty());
        assert!(migrated.message_log.is_some());
        assert_eq!(reopened.get_messages("c1").await?.len(), 2);
        std::fs::remove_dir_all(base)?;
        Ok(())
    }

    /// 验证：更新会话元数据不会重写消息日志。
    #[tokio::test]
    async fn conversation_metadata_update_does_not_rewrite_message_log() -> TestResult {
        let base = tmp_base();
        let store = FileConversationStore::new(&base)?;
        store
            .create_conversation(new_conv("c1", Some("before")))
            .await?;
        store
            .save_messages("c1", &[stored("user", "large body")])
            .await?;
        let before = log_meta(&manifest(&store, "c1")?)?;
        let path = store.message_log_path("c1", before.generation)?;
        let bytes = std::fs::read(&path)?;

        store
            .update_conversation("c1", Some("after"), Some("summary"), Some(1))
            .await?;
        let after = log_meta(&manifest(&store, "c1")?)?;
        assert_eq!(after, before);
        assert_eq!(std::fs::read(path)?, bytes);
        std::fs::remove_dir_all(base)?;
        Ok(())
    }

    /// 验证：Unicode 搜索仅把过滤器用于候选筛选，结果仍经精确校验。
    #[tokio::test]
    async fn unicode_search_uses_filter_only_for_candidates() -> TestResult {
        let base = tmp_base();
        let store = FileConversationStore::new(&base)?;
        for id in ["target", "corrupt-noncandidate", "saturated"] {
            store.create_conversation(new_conv(id, None)).await?;
        }
        store
            .save_messages("target", &[stored("user", "本地智能🧪助手")])
            .await?;
        store
            .save_messages("corrupt-noncandidate", &[stored("user", "plain ascii")])
            .await?;
        store
            .save_messages("saturated", &[stored("user", "healthy but absent")])
            .await?;

        let query = "智能🧪";
        let noncandidate_manifest = manifest(&store, "corrupt-noncandidate")?;
        assert!(
            !noncandidate_manifest
                .search_filter
                .as_ref()
                .is_some_and(|filter| filter.might_contain(query))
        );
        let noncandidate_log = log_meta(&noncandidate_manifest)?;
        let noncandidate_path =
            store.message_log_path("corrupt-noncandidate", noncandidate_log.generation)?;
        let noncandidate_bytes = std::fs::read(&noncandidate_path)?;
        echo_core::utils::fs::truncate_existing(
            &noncandidate_path,
            noncandidate_log.committed_bytes.saturating_sub(1),
            echo_core::utils::fs::FileDurability::SyncData,
        )?;

        let mut saturated = manifest(&store, "saturated")?;
        saturated.search_filter = Some(SearchFilter {
            bits: vec![u64::MAX; SEARCH_FILTER_WORDS],
        });
        store.write_manifest(&saturated)?;

        let found = store.search_conversations(query, 10).await?;
        assert_eq!(found.len(), 1);
        assert_eq!(
            found.first().map(|meta| meta.conversation_id.as_str()),
            Some("target")
        );
        echo_core::utils::fs::atomic_write(&noncandidate_path, &noncandidate_bytes)?;
        assert!(
            store
                .search_conversations("不存在的🔍", 10)
                .await?
                .is_empty()
        );
        std::fs::remove_dir_all(base)?;
        Ok(())
    }

    /// 验证：`safe_segment` 拒绝路径穿越并接受常见 id。
    #[test]
    fn safe_segment_rejects_traversal_and_accepts_typical_ids() {
        assert!(safe_segment("../x").is_err());
        assert!(safe_segment("a/b").is_err());
        assert!(safe_segment("a\\b").is_err());
        assert!(safe_segment("").is_err());
        assert!(safe_segment("_meta").is_err());
        assert!(safe_segment("conv-1709000000-abc123").is_ok());
        assert!(safe_segment("2026-07-28T10:00:00Z").is_ok());
    }
}
