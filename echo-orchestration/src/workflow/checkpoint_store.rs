/*!
检查点持久化存储

支持保存和恢复 Graph 执行状态，实现 LangGraph 风格的中断/检查点机制。

## 设计

- `CheckpointStore` trait：抽象存储接口
- `MemoryCheckpointStore`：内存存储（默认，不持久化）
- `FileCheckpointStore`：文件存储（支持持久化）

## 使用场景

- 长时间运行工作流的暂停/恢复
- 需要人工审批的节点中断
- 故障恢复与重试
*/

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{Mutex, RwLock};

use super::state::SharedState;
use echo_core::error::Result;

// ── Checkpoint ────────────────────────────────────────────────────────────────

/**
检查点——图执行状态的快照

包含恢复执行所需的全部信息：
- 执行路径与步数
- 共享状态快照
- 待执行节点信息

通过 `parent_checkpoint_id` 血缘链支持时间旅行调试，
并支持人类可读的 `label`/`tags` 以及 `branch` 分叉。
*/
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    /// 唯一标识符
    pub id: String,
    /// 图名称
    pub graph_name: String,
    /// 拥有该检查点的工作流执行的稳定标识。
    pub workflow_run_id: String,
    /// 创建该检查点时所用已编译图拓扑的哈希值。
    pub graph_revision: String,
    /// 工作流运行内单调递增的检查点代数。
    pub generation: u64,
    /// 由存储分配给胜出恢复尝试的唯一标识。
    pub resume_attempt_id: Option<String>,
    /// 当前节点
    pub current_node: String,
    /// 状态快照（JSON 序列化）
    pub state_snapshot: serde_json::Value,
    /// 执行路径
    pub path: Vec<String>,
    /// 步数
    pub step_count: usize,
    /// 创建时间
    #[serde(with = "echo_core::utils::time::local_rfc3339")]
    pub created_at: DateTime<Utc>,
    /// 审批通过后要执行的精确续接位置。
    pub continuation: WorkflowContinuation,
    /// 中断类型
    pub interrupt_type: InterruptType,
    /** 用于血缘追踪的父检查点 ID。
    会话中的第一个检查点为 `None`。 */
    #[serde(default)]
    pub parent_checkpoint_id: Option<String>,
    /// 人类可读的标签（例如 "before risky operation"、"branch:feature-x"）
    #[serde(default)]
    pub label: Option<String>,
    /// 用于过滤和检索的标签
    #[serde(default)]
    pub tags: Vec<String>,
    /// 若该检查点为分叉，则为分支名称
    #[serde(default)]
    pub branch: Option<String>,
}

impl Checkpoint {
    /// 由已序列化的状态快照构造检查点，其余字段取默认值。
    fn from_snapshot(
        id: String,
        graph_name: String,
        current_node: String,
        state_snapshot: serde_json::Value,
        path: Vec<String>,
        step_count: usize,
        interrupt_type: InterruptType,
    ) -> Self {
        Self {
            id,
            graph_name,
            workflow_run_id: uuid::Uuid::new_v4().to_string(),
            graph_revision: String::new(),
            generation: 0,
            resume_attempt_id: None,
            current_node,
            state_snapshot,
            path,
            step_count,
            created_at: Utc::now(),
            continuation: WorkflowContinuation::Node,
            interrupt_type,
            parent_checkpoint_id: None,
            label: None,
            tags: Vec::new(),
            branch: None,
        }
    }

    /// 创建检查点，并向上传播状态序列化失败。
    pub fn try_new(
        graph_name: String,
        current_node: String,
        state: &SharedState,
        path: Vec<String>,
        step_count: usize,
        interrupt_type: InterruptType,
    ) -> Result<Self> {
        let state_snapshot = state.to_json_value().map_err(|error| {
            echo_core::error::ReactError::Other(format!(
                "Failed to serialize workflow checkpoint state: {error}"
            ))
        })?;
        Ok(Self::from_snapshot(
            uuid::Uuid::new_v4().to_string(),
            graph_name,
            current_node,
            state_snapshot,
            path,
            step_count,
            interrupt_type,
        ))
    }

    /// 创建新的检查点（状态序列化失败时将错误信息写入快照）
    pub fn new(
        graph_name: String,
        current_node: String,
        state: &SharedState,
        path: Vec<String>,
        step_count: usize,
        interrupt_type: InterruptType,
    ) -> Self {
        let state_snapshot = state.to_json_value().unwrap_or_else(|error| {
            serde_json::json!({
                "__echo_checkpoint_state_error": error.to_string()
            })
        });
        Self::from_snapshot(
            uuid::Uuid::new_v4().to_string(),
            graph_name,
            current_node,
            state_snapshot,
            path,
            step_count,
            interrupt_type,
        )
    }

    /// 将该检查点绑定到已编译的图版本及续接位置。
    pub fn bind_execution(
        mut self,
        graph_revision: impl Into<String>,
        continuation: WorkflowContinuation,
    ) -> Self {
        self.graph_revision = graph_revision.into();
        self.continuation = continuation;
        self
    }

    /// 以下一检查点代数延续同一次工作流运行。
    pub fn continue_run(mut self, parent: &Checkpoint) -> Self {
        self.workflow_run_id = parent.workflow_run_id.clone();
        self.generation = parent.generation.saturating_add(1);
        self.parent_checkpoint_id = Some(parent.id.clone());
        self
    }

    /// 创建带有显式父检查点的新检查点，用于血缘追踪。
    #[allow(clippy::too_many_arguments)]
    pub fn with_parent(
        parent_id: impl Into<String>,
        graph_name: String,
        current_node: String,
        state: &SharedState,
        path: Vec<String>,
        step_count: usize,
        interrupt_type: InterruptType,
    ) -> Self {
        let mut cp = Self::new(
            graph_name,
            current_node,
            state,
            path,
            step_count,
            interrupt_type,
        );
        cp.parent_checkpoint_id = Some(parent_id.into());
        cp
    }

    /// 创建新的分支检查点（从已有检查点分叉）。
    #[allow(clippy::too_many_arguments)]
    pub fn branch(
        parent_id: impl Into<String>,
        branch_name: impl Into<String>,
        graph_name: String,
        current_node: String,
        state: &SharedState,
        path: Vec<String>,
        step_count: usize,
        interrupt_type: InterruptType,
    ) -> Self {
        let mut cp = Self::with_parent(
            parent_id,
            graph_name,
            current_node,
            state,
            path,
            step_count,
            interrupt_type,
        );
        cp.branch = Some(branch_name.into());
        cp
    }

    /// 为该检查点设置人类可读的标签。
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// 为该检查点设置标签。
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    /// 从检查点恢复 SharedState
    pub fn restore_state(&self) -> Result<SharedState> {
        SharedState::from_json(&self.state_snapshot).map_err(|e| {
            echo_core::error::ReactError::Other(format!("Failed to restore state: {}", e))
        })
    }
}

/// 检查点保存的类型化执行游标。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum WorkflowContinuation {
    /// 下一步执行 `current_node`。
    Node,
    /// 基于同一输入快照执行所有目标节点，然后从 `then` 继续。
    FanOut {
        /// 需要并行执行的目标节点
        targets: Vec<String>,
        /// 所有目标完成后继续执行的节点
        then: String,
    },
    /// 工作流已到达终止边。
    End,
}

/// 中断类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum InterruptType {
    /// 进入节点前暂停
    BeforeNode,
    /// 节点执行后暂停
    AfterNode,
}

/// 检查点信息摘要（用于列表展示）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointInfo {
    /// 检查点唯一标识符
    pub id: String,
    /// 图名称
    pub graph_name: String,
    /// 当前节点
    pub current_node: String,
    /// 步数
    pub step_count: usize,
    /// 创建时间
    #[serde(with = "echo_core::utils::time::local_rfc3339")]
    pub created_at: DateTime<Utc>,
    /// 中断类型
    pub interrupt_type: InterruptType,
    /// 父检查点 ID
    pub parent_checkpoint_id: Option<String>,
    /// 人类可读的标签
    pub label: Option<String>,
    /// 标签列表
    pub tags: Vec<String>,
    /// 分支名称
    pub branch: Option<String>,
}

impl From<&Checkpoint> for CheckpointInfo {
    /// 从完整检查点提取摘要信息。
    fn from(cp: &Checkpoint) -> Self {
        Self {
            id: cp.id.clone(),
            graph_name: cp.graph_name.clone(),
            current_node: cp.current_node.clone(),
            step_count: cp.step_count,
            created_at: cp.created_at,
            interrupt_type: cp.interrupt_type,
            parent_checkpoint_id: cp.parent_checkpoint_id.clone(),
            label: cp.label.clone(),
            tags: cp.tags.clone(),
            branch: cp.branch.clone(),
        }
    }
}

/// 列出检查点时使用的过滤参数
#[derive(Debug, Clone, Default)]
pub struct CheckpointFilter {
    /// 按图名称过滤
    pub graph_name: Option<String>,
    /// 按分支名称过滤
    pub branch: Option<String>,
    /// 按标签过滤
    pub tag: Option<String>,
    /// 限制结果数量
    pub limit: Option<usize>,
}

impl CheckpointFilter {
    /// 创建匹配指定图的过滤器。
    pub fn by_graph(graph_name: impl Into<String>) -> Self {
        Self {
            graph_name: Some(graph_name.into()),
            ..Default::default()
        }
    }
}

// ── CheckpointStore Trait ──────────────────────────────────────────────────────

/// 检查点存储接口
#[async_trait]
pub trait CheckpointStore: Send + Sync {
    /// 保存检查点
    async fn save(&self, checkpoint: &Checkpoint) -> Result<()>;

    /// 加载检查点
    async fn load(&self, id: &str) -> Result<Option<Checkpoint>>;

    /**
    原子地租用并返回一个检查点，供一次恢复尝试使用。

    租约必须通过 [`Self::ack_claim`] 或 [`Self::requeue_claim`] 结算。
    被领取的检查点对 `load`/`list` 仍然可见，因此崩溃不会使其变成不可见的空洞。
    */
    async fn claim(&self, id: &str) -> Result<Option<Checkpoint>>;

    /**
    确认一次已成功完成的恢复尝试。

    存储实现必须覆盖此操作。默认实现以失败关闭，
    使远程或自定义存储无法在未真正提交的情况下报告已结算。
    */
    async fn ack_claim(&self, _id: &str, _attempt_id: &str) -> Result<()> {
        Err(echo_core::error::ReactError::Other(
            "Checkpoint store does not support claim acknowledgement".to_string(),
        ))
    }

    /// 失败后将已租用的领取归还到待处理检查点集合。
    async fn requeue_claim(&self, _id: &str, _attempt_id: &str) -> Result<()> {
        Err(echo_core::error::ReactError::Other(
            "Checkpoint store does not support claim requeue".to_string(),
        ))
    }

    /// 为指定的恢复尝试续租其活跃领取。
    async fn renew_claim(&self, _id: &str, _attempt_id: &str) -> Result<()> {
        Err(echo_core::error::ReactError::Other(
            "Checkpoint store does not support claim renewal".to_string(),
        ))
    }

    /**
    执行所有者应续租活跃领取的时间间隔。

    `None` 表示该存储的领取不会过期。返回间隔的存储必须实现 [`Self::renew_claim`]。
    */
    fn claim_heartbeat_interval(&self) -> Option<Duration> {
        None
    }

    /** 领取、续租/归还与确认是否构成完整的结算契约。
    `Graph` 在获取领取前会检查此项。 */
    fn supports_claim_settlement(&self) -> bool {
        false
    }

    /**
    仅在检查点代数未变化时保存元数据。

    具备原子存储能力的实现应覆盖此方法。无法提供原子性的远程适配器
    必须拒绝更新，而不是退化为存在竞态的先加载后保存流程。
    */
    async fn save_if_generation(
        &self,
        checkpoint: &Checkpoint,
        expected_generation: u64,
    ) -> Result<bool> {
        let _ = (checkpoint, expected_generation);
        Ok(false)
    }

    /// 列出所有检查点信息
    async fn list(&self) -> Result<Vec<CheckpointInfo>>;

    /// 列出指定图的检查点
    async fn list_by_graph(&self, graph_name: &str) -> Result<Vec<CheckpointInfo>> {
        let all = self.list().await?;
        Ok(all
            .into_iter()
            .filter(|info| info.graph_name == graph_name)
            .collect())
    }

    /// 列出匹配过滤条件的检查点
    async fn list_filtered(&self, filter: &CheckpointFilter) -> Result<Vec<CheckpointInfo>> {
        let mut results = if let Some(ref graph_name) = filter.graph_name {
            self.list_by_graph(graph_name).await?
        } else {
            self.list().await?
        };

        if let Some(ref branch) = filter.branch {
            results.retain(|info| info.branch.as_deref() == Some(branch.as_str()));
        }
        if let Some(ref tag) = filter.tag {
            results.retain(|info| info.tags.contains(tag));
        }
        if let Some(limit) = filter.limit {
            results.truncate(limit);
        }

        Ok(results)
    }

    /// 删除检查点
    async fn delete(&self, id: &str) -> Result<()>;

    /// 清空所有检查点
    async fn clear(&self) -> Result<()>;
}

// ── 内存检查点存储 ────────────────────────────────────────────────────

/// 内存存储实现（默认，不持久化）
pub struct MemoryCheckpointStore {
    /// 串行化所有存储操作的互斥锁
    operation_lock: Mutex<()>,
    /// 待处理的检查点，按 ID 索引
    checkpoints: RwLock<HashMap<String, Checkpoint>>,
    /// 活跃领取，按（检查点 ID，恢复尝试 ID）索引
    claims: RwLock<HashMap<(String, String), Checkpoint>>,
}

impl MemoryCheckpointStore {
    /// 创建空的内存检查点存储。
    pub fn new() -> Self {
        Self {
            operation_lock: Mutex::new(()),
            checkpoints: RwLock::new(HashMap::new()),
            claims: RwLock::new(HashMap::new()),
        }
    }
}

impl Default for MemoryCheckpointStore {
    /// 等同于 [`MemoryCheckpointStore::new`]。
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl CheckpointStore for MemoryCheckpointStore {
    async fn save(&self, checkpoint: &Checkpoint) -> Result<()> {
        let _guard = self.operation_lock.lock().await;
        if self
            .claims
            .read()
            .await
            .keys()
            .any(|(checkpoint_id, _)| checkpoint_id == &checkpoint.id)
        {
            return Err(echo_core::error::ReactError::Other(format!(
                "Cannot save checkpoint '{}' while a resume claim is active",
                checkpoint.id
            )));
        }
        let mut checkpoints = self.checkpoints.write().await;
        checkpoints.insert(checkpoint.id.clone(), checkpoint.clone());
        Ok(())
    }

    async fn load(&self, id: &str) -> Result<Option<Checkpoint>> {
        let _guard = self.operation_lock.lock().await;
        let checkpoints = self.checkpoints.read().await;
        if let Some(checkpoint) = checkpoints.get(id).cloned() {
            return Ok(Some(checkpoint));
        }
        drop(checkpoints);
        let claims = self.claims.read().await;
        Ok(claims.iter().find_map(|((checkpoint_id, _), checkpoint)| {
            (checkpoint_id == id).then(|| checkpoint.clone())
        }))
    }

    async fn claim(&self, id: &str) -> Result<Option<Checkpoint>> {
        let _guard = self.operation_lock.lock().await;
        let mut checkpoints = self.checkpoints.write().await;
        let Some(mut checkpoint) = checkpoints.remove(id) else {
            return Ok(None);
        };
        let attempt_id = uuid::Uuid::new_v4().to_string();
        checkpoint.resume_attempt_id = Some(attempt_id.clone());
        drop(checkpoints);
        self.claims
            .write()
            .await
            .insert((id.to_string(), attempt_id), checkpoint.clone());
        Ok(Some(checkpoint))
    }

    async fn ack_claim(&self, id: &str, attempt_id: &str) -> Result<()> {
        let _guard = self.operation_lock.lock().await;
        let removed = self
            .claims
            .write()
            .await
            .remove(&(id.to_string(), attempt_id.to_string()));
        if removed.is_some() {
            Ok(())
        } else {
            Err(echo_core::error::ReactError::Other(format!(
                "Checkpoint claim '{id}' for attempt '{attempt_id}' is not active"
            )))
        }
    }

    async fn requeue_claim(&self, id: &str, attempt_id: &str) -> Result<()> {
        let _guard = self.operation_lock.lock().await;
        let key = (id.to_string(), attempt_id.to_string());
        let Some(mut checkpoint) = self.claims.write().await.remove(&key) else {
            return Err(echo_core::error::ReactError::Other(format!(
                "Checkpoint claim '{id}' for attempt '{attempt_id}' is not active"
            )));
        };
        checkpoint.resume_attempt_id = None;
        self.checkpoints
            .write()
            .await
            .insert(id.to_string(), checkpoint);
        Ok(())
    }

    async fn renew_claim(&self, id: &str, attempt_id: &str) -> Result<()> {
        let _guard = self.operation_lock.lock().await;
        if !self
            .claims
            .read()
            .await
            .contains_key(&(id.to_string(), attempt_id.to_string()))
        {
            return Err(echo_core::error::ReactError::Other(format!(
                "Checkpoint claim '{id}' for attempt '{attempt_id}' is not active"
            )));
        }
        Ok(())
    }

    fn supports_claim_settlement(&self) -> bool {
        true
    }

    async fn save_if_generation(
        &self,
        checkpoint: &Checkpoint,
        expected_generation: u64,
    ) -> Result<bool> {
        let _guard = self.operation_lock.lock().await;
        let mut checkpoints = self.checkpoints.write().await;
        let Some(current) = checkpoints.get(&checkpoint.id) else {
            return Ok(false);
        };
        if current.generation != expected_generation {
            return Ok(false);
        }
        checkpoints.insert(checkpoint.id.clone(), checkpoint.clone());
        Ok(true)
    }

    async fn list(&self) -> Result<Vec<CheckpointInfo>> {
        let _guard = self.operation_lock.lock().await;
        let checkpoints = self.checkpoints.read().await;
        let mut infos: Vec<CheckpointInfo> =
            checkpoints.values().map(CheckpointInfo::from).collect();
        drop(checkpoints);
        let claims = self.claims.read().await;
        infos.extend(claims.values().map(CheckpointInfo::from));
        Ok(infos)
    }

    async fn delete(&self, id: &str) -> Result<()> {
        let _guard = self.operation_lock.lock().await;
        self.checkpoints.write().await.remove(id);
        self.claims
            .write()
            .await
            .retain(|(checkpoint_id, _), _| checkpoint_id != id);
        Ok(())
    }

    async fn clear(&self) -> Result<()> {
        let _guard = self.operation_lock.lock().await;
        self.checkpoints.write().await.clear();
        self.claims.write().await.clear();
        Ok(())
    }
}

// ── 文件检查点存储 ──────────────────────────────────────────────────────

/// 持久化到磁盘的领取记录：检查点及其最近续租时间。
#[derive(Serialize, Deserialize)]
struct StoredCheckpointClaim {
    /// 被领取的检查点
    checkpoint: Checkpoint,
    /// 最近一次续租时间
    #[serde(with = "echo_core::utils::time::local_rfc3339")]
    renewed_at: DateTime<Utc>,
}

/// 文件存储实现（支持持久化）
pub struct FileCheckpointStore {
    /// 检查点文件所在目录
    base_path: PathBuf,
    /// 进程内串行化存储操作的互斥锁
    operation_lock: Arc<Mutex<()>>,
    /// 领取租约时长，超时未续租的领取视为过期
    claim_lease: Duration,
    /// 建议的领取续租心跳间隔
    claim_heartbeat_interval: Duration,
}

impl FileCheckpointStore {
    /// 以指定目录创建文件检查点存储，使用默认的租约与心跳时长。
    pub fn new<P: Into<PathBuf>>(base_path: P) -> Self {
        Self {
            base_path: base_path.into(),
            operation_lock: Arc::new(Mutex::new(())),
            claim_lease: Duration::from_secs(300),
            claim_heartbeat_interval: Duration::from_secs(60),
        }
    }

    /// 测试专用：自定义领取租约与心跳间隔（要求 0 < 心跳 < 租约）。
    #[cfg(test)]
    pub(crate) fn with_claim_timing(
        mut self,
        claim_lease: Duration,
        claim_heartbeat_interval: Duration,
    ) -> Result<Self> {
        if claim_lease.is_zero()
            || claim_heartbeat_interval.is_zero()
            || claim_heartbeat_interval >= claim_lease
        {
            return Err(echo_core::error::ReactError::Other(
                "Checkpoint claim timing requires 0 < heartbeat < lease".to_string(),
            ));
        }
        self.claim_lease = claim_lease;
        self.claim_heartbeat_interval = claim_heartbeat_interval;
        Ok(self)
    }

    /// 计算检查点文件路径。
    fn checkpoint_path(&self, id: &str) -> Result<PathBuf> {
        let file_name = format!("{id}.json");
        echo_core::utils::fs::join_path_segment(&self.base_path, &file_name).map_err(Into::into)
    }

    /// 确保存储目录存在，不存在则创建。
    fn ensure_dir_exists(&self) -> Result<()> {
        if !self.base_path.exists() {
            std::fs::create_dir_all(&self.base_path).map_err(|e| {
                echo_core::error::ReactError::Other(format!(
                    "Failed to create checkpoint dir: {}",
                    e
                ))
            })?;
        }
        Ok(())
    }
}

#[async_trait]
impl CheckpointStore for FileCheckpointStore {
    async fn save(&self, checkpoint: &Checkpoint) -> Result<()> {
        let _guard = self.operation_lock.lock().await;
        let _file_lock = self.acquire_file_lock().await?;
        if self.find_claim(&checkpoint.id).await?.is_some() {
            return Err(echo_core::error::ReactError::Other(format!(
                "Cannot save checkpoint '{}' while a resume claim is active",
                checkpoint.id
            )));
        }
        self.save_unlocked(checkpoint).await
    }

    async fn load(&self, id: &str) -> Result<Option<Checkpoint>> {
        let _guard = self.operation_lock.lock().await;
        let _file_lock = self.acquire_file_lock().await?;
        self.load_unlocked(id).await
    }

    async fn claim(&self, id: &str) -> Result<Option<Checkpoint>> {
        let _guard = self.operation_lock.lock().await;
        let _file_lock = self.acquire_file_lock().await?;
        let _ = self.recover_stale_claim_unlocked(id).await?;
        self.claim_unlocked(id).await
    }

    async fn ack_claim(&self, id: &str, attempt_id: &str) -> Result<()> {
        let _guard = self.operation_lock.lock().await;
        let _file_lock = self.acquire_file_lock().await?;
        let path = self.claim_path(id, attempt_id)?;
        if path.exists() {
            let (checkpoint, _) = Self::read_claim(&path).await?;
            if checkpoint.resume_attempt_id.as_deref() != Some(attempt_id) {
                return Err(echo_core::error::ReactError::Other(format!(
                    "Checkpoint claim '{id}' is owned by a different resume attempt"
                )));
            }
        }
        match tokio::fs::remove_file(path).await {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                Err(echo_core::error::ReactError::Other(format!(
                    "Checkpoint claim '{id}' for attempt '{attempt_id}' is not active"
                )))
            }
            Err(error) => Err(echo_core::error::ReactError::Other(format!(
                "Failed to acknowledge checkpoint claim: {error}"
            ))),
        }
    }

    async fn requeue_claim(&self, id: &str, attempt_id: &str) -> Result<()> {
        let _guard = self.operation_lock.lock().await;
        let _file_lock = self.acquire_file_lock().await?;
        let claim_path = self.claim_path(id, attempt_id)?;
        let path = self.checkpoint_path(id)?;
        if !claim_path.exists() {
            return Err(echo_core::error::ReactError::Other(format!(
                "Checkpoint claim '{id}' for attempt '{attempt_id}' is not active"
            )));
        }
        if path.exists() {
            return Err(echo_core::error::ReactError::Other(format!(
                "Cannot requeue checkpoint '{id}': pending checkpoint already exists"
            )));
        }
        let (mut checkpoint, _) = Self::read_claim(&claim_path).await?;
        if checkpoint.resume_attempt_id.as_deref() != Some(attempt_id) {
            return Err(echo_core::error::ReactError::Other(format!(
                "Checkpoint claim '{id}' is owned by a different resume attempt"
            )));
        }
        checkpoint.resume_attempt_id = None;
        Self::write_checkpoint(&claim_path, &checkpoint).await?;
        tokio::fs::rename(claim_path, &path)
            .await
            .map_err(|error| {
                echo_core::error::ReactError::Other(format!(
                    "Failed to requeue checkpoint claim: {error}"
                ))
            })?;
        Ok(())
    }

    async fn renew_claim(&self, id: &str, attempt_id: &str) -> Result<()> {
        let _guard = self.operation_lock.lock().await;
        let _file_lock = self.acquire_file_lock().await?;
        let claim_path = self.claim_path(id, attempt_id)?;
        if !claim_path.exists() {
            return Err(echo_core::error::ReactError::Other(format!(
                "Checkpoint claim '{id}' for attempt '{attempt_id}' is not active"
            )));
        }
        let (checkpoint, _) = Self::read_claim(&claim_path).await?;
        if checkpoint.resume_attempt_id.as_deref() != Some(attempt_id) {
            return Err(echo_core::error::ReactError::Other(format!(
                "Checkpoint claim '{id}' is owned by a different resume attempt"
            )));
        }
        Self::write_claim(&claim_path, &checkpoint, Utc::now()).await
    }

    fn claim_heartbeat_interval(&self) -> Option<Duration> {
        Some(self.claim_heartbeat_interval)
    }

    fn supports_claim_settlement(&self) -> bool {
        true
    }

    async fn save_if_generation(
        &self,
        checkpoint: &Checkpoint,
        expected_generation: u64,
    ) -> Result<bool> {
        let _guard = self.operation_lock.lock().await;
        let _file_lock = self.acquire_file_lock().await?;
        let Some(current) = self.load_unlocked(&checkpoint.id).await? else {
            return Ok(false);
        };
        if current.generation != expected_generation {
            return Ok(false);
        }
        let path = self.checkpoint_path(&checkpoint.id)?;
        if !path.exists() {
            return Ok(false);
        }
        self.save_unlocked(checkpoint).await?;
        Ok(true)
    }

    async fn list(&self) -> Result<Vec<CheckpointInfo>> {
        let _guard = self.operation_lock.lock().await;
        self.list_unlocked().await
    }

    async fn delete(&self, id: &str) -> Result<()> {
        let _guard = self.operation_lock.lock().await;
        let _file_lock = self.acquire_file_lock().await?;
        if !self.base_path.exists() {
            return Ok(());
        }
        let path = self.checkpoint_path(id)?;
        if path.exists() {
            tokio::fs::remove_file(path).await.map_err(|e| {
                echo_core::error::ReactError::Other(format!("Failed to delete checkpoint: {}", e))
            })?;
        }
        let mut entries = tokio::fs::read_dir(&self.base_path).await.map_err(|e| {
            echo_core::error::ReactError::Other(format!("Failed to read checkpoint dir: {}", e))
        })?;
        while let Some(entry) = entries.next_entry().await.map_err(|e| {
            echo_core::error::ReactError::Other(format!("Failed to read entry: {}", e))
        })? {
            let name = entry.file_name();
            if name
                .to_string_lossy()
                .starts_with(&format!("{id}.claimed-"))
            {
                tokio::fs::remove_file(entry.path()).await.map_err(|e| {
                    echo_core::error::ReactError::Other(format!(
                        "Failed to delete checkpoint claim: {e}"
                    ))
                })?;
            }
        }
        Ok(())
    }

    async fn clear(&self) -> Result<()> {
        let _guard = self.operation_lock.lock().await;
        let _file_lock = self.acquire_file_lock().await?;
        if !self.base_path.exists() {
            return Ok(());
        }
        let mut entries = tokio::fs::read_dir(&self.base_path).await.map_err(|e| {
            echo_core::error::ReactError::Other(format!("Failed to read checkpoint dir: {}", e))
        })?;
        while let Some(entry) = entries.next_entry().await.map_err(|e| {
            echo_core::error::ReactError::Other(format!("Failed to read entry: {}", e))
        })? {
            let path = entry.path();
            if path
                .extension()
                .is_some_and(|extension| extension == "json" || extension == "claim")
            {
                tokio::fs::remove_file(&path).await.map_err(|error| {
                    echo_core::error::ReactError::Other(format!(
                        "failed to remove checkpoint {}: {error}",
                        path.display()
                    ))
                })?;
            }
        }
        Ok(())
    }
}

impl FileCheckpointStore {
    /// 计算指定恢复尝试的领取文件路径。
    fn claim_path(&self, id: &str, attempt_id: &str) -> Result<PathBuf> {
        let claim_name = format!("{id}.claimed-{attempt_id}.claim");
        echo_core::utils::fs::join_path_segment(&self.base_path, &claim_name).map_err(Into::into)
    }

    /// 获取跨进程的独占文件锁，返回的文件句柄在释放时解锁。
    async fn acquire_file_lock(&self) -> Result<std::fs::File> {
        self.ensure_dir_exists()?;
        let path = echo_core::utils::fs::join_path_segment(
            &self.base_path,
            ".echo-checkpoint-store.lock",
        )?;
        tokio::task::spawn_blocking(move || {
            use fs2::FileExt;
            let file = std::fs::OpenOptions::new()
                .create(true)
                .truncate(false)
                .read(true)
                .write(true)
                .open(path)
                .map_err(|error| {
                    echo_core::error::ReactError::Other(format!(
                        "Failed to open checkpoint store lock: {error}"
                    ))
                })?;
            file.lock_exclusive().map_err(|error| {
                echo_core::error::ReactError::Other(format!(
                    "Failed to lock checkpoint store: {error}"
                ))
            })?;
            Ok(file)
        })
        .await
        .map_err(|error| {
            echo_core::error::ReactError::Other(format!("checkpoint lock task failed: {error}"))
        })?
    }

    /// 在已持有锁的前提下保存检查点。
    async fn save_unlocked(&self, checkpoint: &Checkpoint) -> Result<()> {
        self.ensure_dir_exists()?;
        let path = self.checkpoint_path(&checkpoint.id)?;
        Self::write_checkpoint(&path, checkpoint).await
    }

    /// 原子地将检查点写入指定路径。
    async fn write_checkpoint(path: &std::path::Path, checkpoint: &Checkpoint) -> Result<()> {
        let json = serde_json::to_string_pretty(checkpoint).map_err(|e| {
            echo_core::error::ReactError::Other(format!("Failed to serialize checkpoint: {}", e))
        })?;
        let path = path.to_path_buf();
        tokio::task::spawn_blocking(move || {
            echo_core::utils::fs::atomic_write(&path, json.as_bytes())
        })
        .await
        .map_err(|error| {
            echo_core::error::ReactError::Other(format!("checkpoint writer task failed: {error}"))
        })??;
        Ok(())
    }

    /// 原子地将领取记录（含续租时间）写入指定路径。
    async fn write_claim(
        path: &std::path::Path,
        checkpoint: &Checkpoint,
        renewed_at: DateTime<Utc>,
    ) -> Result<()> {
        let claim = StoredCheckpointClaim {
            checkpoint: checkpoint.clone(),
            renewed_at,
        };
        let json = serde_json::to_string_pretty(&claim).map_err(|error| {
            echo_core::error::ReactError::Other(format!(
                "Failed to serialize checkpoint claim: {error}"
            ))
        })?;
        let path = path.to_path_buf();
        tokio::task::spawn_blocking(move || {
            echo_core::utils::fs::atomic_write(&path, json.as_bytes())
        })
        .await
        .map_err(|error| {
            echo_core::error::ReactError::Other(format!(
                "checkpoint claim writer task failed: {error}"
            ))
        })??;
        Ok(())
    }

    /// 在已持有锁的前提下加载检查点，必要时回收过期领取。
    async fn load_unlocked(&self, id: &str) -> Result<Option<Checkpoint>> {
        let path = self.checkpoint_path(id)?;
        if path.exists() {
            return Self::read_checkpoint(&path).await.map(Some);
        }
        if self.recover_stale_claim_unlocked(id).await? {
            return Self::read_checkpoint(&path).await.map(Some);
        }
        if let Some((claim_path, _attempt_id)) = self.find_claim(id).await? {
            /* 已租用的检查点对恢复与诊断仍然可见；
            在恢复尚在进行时，不会被悄悄转换为"未找到"。 */
            return Self::read_claim(&claim_path)
                .await
                .map(|(checkpoint, _)| Some(checkpoint));
        }
        Ok(None)
    }

    /// 在已持有锁的前提下通过重命名原子地领取检查点。
    async fn claim_unlocked(&self, id: &str) -> Result<Option<Checkpoint>> {
        self.ensure_dir_exists()?;
        let path = self.checkpoint_path(id)?;
        if !path.exists() {
            return Ok(None);
        }
        let attempt_id = uuid::Uuid::new_v4().to_string();
        let claim_path = self.claim_path(id, &attempt_id)?;
        match tokio::fs::rename(&path, &claim_path).await {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => {
                return Err(echo_core::error::ReactError::Other(format!(
                    "Failed to claim checkpoint: {error}"
                )));
            }
        }
        let mut checkpoint = Self::read_checkpoint(&claim_path).await?;
        checkpoint.resume_attempt_id = Some(attempt_id);
        Self::write_claim(&claim_path, &checkpoint, Utc::now()).await?;
        Ok(Some(checkpoint))
    }

    /// 查找指定检查点的活跃领取文件，返回路径及恢复尝试 ID。
    async fn find_claim(&self, id: &str) -> Result<Option<(PathBuf, String)>> {
        if !self.base_path.exists() {
            return Ok(None);
        }
        let prefix = format!("{id}.claimed-");
        let mut entries = tokio::fs::read_dir(&self.base_path).await.map_err(|e| {
            echo_core::error::ReactError::Other(format!("Failed to read checkpoint dir: {}", e))
        })?;
        while let Some(entry) = entries.next_entry().await.map_err(|e| {
            echo_core::error::ReactError::Other(format!("Failed to read entry: {}", e))
        })? {
            let name = entry.file_name().to_string_lossy().into_owned();
            if let Some(attempt_id) = name
                .strip_prefix(&prefix)
                .and_then(|value| value.strip_suffix(".claim"))
            {
                return Ok(Some((entry.path(), attempt_id.to_string())));
            }
        }
        Ok(None)
    }

    /// 若领取已超过租约时长，则将其归还为待处理检查点；返回是否发生回收。
    async fn recover_stale_claim_unlocked(&self, id: &str) -> Result<bool> {
        let path = self.checkpoint_path(id)?;
        if path.exists() {
            return Ok(false);
        }
        let Some((claim_path, _attempt_id)) = self.find_claim(id).await? else {
            return Ok(false);
        };
        let (checkpoint, renewed_at) = Self::read_claim(&claim_path).await?;
        let stale = if let Some(renewed_at) = renewed_at {
            Utc::now()
                .signed_duration_since(renewed_at)
                .to_std()
                .is_ok_and(|age| age >= self.claim_lease)
        } else {
            let metadata = tokio::fs::metadata(&claim_path).await.map_err(|error| {
                echo_core::error::ReactError::Other(format!(
                    "Failed to inspect claimed checkpoint: {error}"
                ))
            })?;
            metadata
                .modified()
                .ok()
                .and_then(|modified| modified.elapsed().ok())
                .is_some_and(|age| age >= self.claim_lease)
        };
        if !stale {
            return Ok(false);
        }
        let mut checkpoint = checkpoint;
        checkpoint.resume_attempt_id = None;
        Self::write_checkpoint(&claim_path, &checkpoint).await?;
        match tokio::fs::rename(claim_path, &path).await {
            Ok(()) => Ok(true),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
            Err(error) => Err(echo_core::error::ReactError::Other(format!(
                "Failed to recover checkpoint claim: {error}"
            ))),
        }
    }

    /// 读取并解析检查点文件。
    async fn read_checkpoint(path: &std::path::Path) -> Result<Checkpoint> {
        let json = tokio::fs::read_to_string(path).await.map_err(|e| {
            echo_core::error::ReactError::Other(format!("Failed to read checkpoint: {}", e))
        })?;
        serde_json::from_str::<Checkpoint>(&json).map_err(|e| {
            echo_core::error::ReactError::Other(format!("Failed to parse checkpoint: {}", e))
        })
    }

    /// 读取领取文件，兼容旧格式（无续租时间）。
    async fn read_claim(path: &std::path::Path) -> Result<(Checkpoint, Option<DateTime<Utc>>)> {
        let json = tokio::fs::read_to_string(path).await.map_err(|error| {
            echo_core::error::ReactError::Other(format!("Failed to read checkpoint claim: {error}"))
        })?;
        if let Ok(claim) = serde_json::from_str::<StoredCheckpointClaim>(&json) {
            return Ok((claim.checkpoint, Some(claim.renewed_at)));
        }
        serde_json::from_str::<Checkpoint>(&json)
            .map(|checkpoint| (checkpoint, None))
            .map_err(|error| {
                echo_core::error::ReactError::Other(format!(
                    "Failed to parse checkpoint claim: {error}"
                ))
            })
    }

    /// 在已持有锁的前提下列出所有检查点（含已领取的）。
    async fn list_unlocked(&self) -> Result<Vec<CheckpointInfo>> {
        self.ensure_dir_exists()?;
        let mut entries = tokio::fs::read_dir(&self.base_path).await.map_err(|e| {
            echo_core::error::ReactError::Other(format!("Failed to read checkpoint dir: {}", e))
        })?;

        let mut infos = Vec::new();
        while let Some(entry) = entries.next_entry().await.map_err(|e| {
            echo_core::error::ReactError::Other(format!("Failed to read entry: {}", e))
        })? {
            let path = entry.path();
            if path
                .extension()
                .is_some_and(|extension| extension == "json" || extension == "claim")
            {
                let checkpoint = if path
                    .extension()
                    .is_some_and(|extension| extension == "claim")
                {
                    Self::read_claim(&path)
                        .await
                        .map(|(checkpoint, _)| checkpoint)
                } else {
                    Self::read_checkpoint(&path).await
                }
                .map_err(|error| {
                    echo_core::error::ReactError::Other(format!(
                        "corrupt checkpoint {}: {error}",
                        path.display()
                    ))
                })?;
                infos.push(CheckpointInfo::from(&checkpoint));
            }
        }

        // 按创建时间排序（最新优先）
        infos.sort_by_key(|info| std::cmp::Reverse(info.created_at));
        Ok(infos)
    }
}

// ── 单元测试 ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// 未实现领取结算的测试存储，用于验证默认实现以失败关闭。
    struct UnsupportedSettlementStore;

    #[async_trait::async_trait]
    impl CheckpointStore for UnsupportedSettlementStore {
        async fn save(&self, _checkpoint: &Checkpoint) -> Result<()> {
            Ok(())
        }

        async fn load(&self, _id: &str) -> Result<Option<Checkpoint>> {
            Ok(None)
        }

        async fn claim(&self, _id: &str) -> Result<Option<Checkpoint>> {
            Ok(None)
        }

        async fn list(&self) -> Result<Vec<CheckpointInfo>> {
            Ok(Vec::new())
        }

        async fn delete(&self, _id: &str) -> Result<()> {
            Ok(())
        }

        async fn clear(&self) -> Result<()> {
            Ok(())
        }
    }

    /// 默认的领取结算实现应以失败关闭。
    #[tokio::test]
    async fn default_claim_settlement_fails_closed() {
        let store = UnsupportedSettlementStore;
        assert!(!store.supports_claim_settlement());
        assert!(store.ack_claim("checkpoint", "attempt").await.is_err());
        assert!(store.requeue_claim("checkpoint", "attempt").await.is_err());
        assert!(store.renew_claim("checkpoint", "attempt").await.is_err());
    }

    #[test]
    fn test_checkpoint_create() {
        let state = SharedState::new();
        state.set("test_key", "test_value").unwrap();

        let cp = Checkpoint::new(
            "test_graph".to_string(),
            "node1".to_string(),
            &state,
            vec!["start".to_string(), "node1".to_string()],
            2,
            InterruptType::BeforeNode,
        );

        assert!(!cp.id.is_empty());
        assert_eq!(cp.graph_name, "test_graph");
        assert_eq!(cp.current_node, "node1");
        assert_eq!(cp.path.len(), 2);
        assert_eq!(cp.step_count, 2);
        assert_eq!(cp.interrupt_type, InterruptType::BeforeNode);
    }

    #[test]
    fn test_checkpoint_restore_state() {
        let state = SharedState::new();
        state.set("key", "value").unwrap();

        let cp = Checkpoint::new(
            "test".to_string(),
            "node".to_string(),
            &state,
            vec![],
            0,
            InterruptType::BeforeNode,
        );

        let restored = cp.restore_state().unwrap();
        assert_eq!(restored.get::<String>("key"), Some("value".to_string()));
    }

    #[tokio::test]
    async fn test_memory_store() {
        let store = MemoryCheckpointStore::new();

        let state = SharedState::new();
        state.set("x", 42).unwrap();

        let cp = Checkpoint::new(
            "graph".to_string(),
            "node".to_string(),
            &state,
            vec![],
            0,
            InterruptType::BeforeNode,
        );

        let id = cp.id.clone();

        // 保存
        store.save(&cp).await.unwrap();

        // 加载
        let loaded = store.load(&id).await.unwrap();
        assert!(loaded.is_some());
        let loaded = loaded.unwrap();
        assert_eq!(loaded.graph_name, "graph");

        // 列出
        let list = store.list().await.unwrap();
        assert_eq!(list.len(), 1);

        // 删除
        store.delete(&id).await.unwrap();
        let loaded = store.load(&id).await.unwrap();
        assert!(loaded.is_none());
    }

    #[tokio::test]
    async fn memory_claim_is_visible_and_can_be_requeued_or_acknowledged() {
        let store = MemoryCheckpointStore::new();
        let checkpoint = Checkpoint::new(
            "claim-lifecycle".to_string(),
            "node".to_string(),
            &SharedState::new(),
            Vec::new(),
            0,
            InterruptType::BeforeNode,
        );
        let id = checkpoint.id.clone();
        store.save(&checkpoint).await.unwrap();

        let claimed = store.claim(&id).await.unwrap().unwrap();
        let attempt_id = claimed.resume_attempt_id.clone().unwrap();
        assert_eq!(store.list().await.unwrap().len(), 1);
        assert!(store.save(&checkpoint).await.is_err());
        assert!(store.renew_claim(&id, "stale-attempt").await.is_err());
        assert!(store.ack_claim(&id, "stale-attempt").await.is_err());
        assert!(store.renew_claim(&id, &attempt_id).await.is_ok());

        store.requeue_claim(&id, &attempt_id).await.unwrap();
        let requeued = store.load(&id).await.unwrap().unwrap();
        assert!(requeued.resume_attempt_id.is_none());

        let claimed_again = store.claim(&id).await.unwrap().unwrap();
        let attempt_again = claimed_again.resume_attempt_id.clone().unwrap();
        store.ack_claim(&id, &attempt_again).await.unwrap();
        assert!(store.load(&id).await.unwrap().is_none());
        assert!(store.list().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_file_store() {
        // 使用系统临时目录下的路径
        let temp_path = std::env::temp_dir().join(format!("echo_test_{}", uuid::Uuid::new_v4()));
        let store = FileCheckpointStore::new(&temp_path);

        let state = SharedState::new();
        state.set("data", "test").unwrap();

        let cp = Checkpoint::new(
            "test_graph".to_string(),
            "node_a".to_string(),
            &state,
            vec!["start".to_string()],
            1,
            InterruptType::AfterNode,
        );

        let id = cp.id.clone();

        // 保存
        store.save(&cp).await.unwrap();

        // 加载
        let loaded = store.load(&id).await.unwrap();
        assert!(loaded.is_some());
        let loaded = loaded.unwrap();
        assert_eq!(loaded.graph_name, "test_graph");
        assert_eq!(loaded.current_node, "node_a");

        // 列出
        let list = store.list().await.unwrap();
        assert_eq!(list.len(), 1);

        // 清空
        store.clear().await.unwrap();
        let list = store.list().await.unwrap();
        assert!(list.is_empty());

        // 清理
        let _ = std::fs::remove_dir_all(&temp_path);
    }

    #[tokio::test]
    async fn file_claim_remains_discoverable_until_acknowledged() {
        let temp_path = std::env::temp_dir().join(format!("echo_claim_{}", uuid::Uuid::new_v4()));
        let store = FileCheckpointStore::new(&temp_path);
        let checkpoint = Checkpoint::new(
            "claim-file".to_string(),
            "node".to_string(),
            &SharedState::new(),
            Vec::new(),
            0,
            InterruptType::BeforeNode,
        );
        let id = checkpoint.id.clone();
        store.save(&checkpoint).await.unwrap();

        let claimed = store.claim(&id).await.unwrap().unwrap();
        let attempt_id = claimed.resume_attempt_id.clone().unwrap();
        assert!(store.load(&id).await.unwrap().is_some());
        assert_eq!(store.list().await.unwrap().len(), 1);
        assert!(store.save(&checkpoint).await.is_err());
        assert!(store.requeue_claim(&id, "stale-attempt").await.is_err());
        assert!(store.renew_claim(&id, "stale-attempt").await.is_err());
        assert!(store.renew_claim(&id, &attempt_id).await.is_ok());

        store.requeue_claim(&id, &attempt_id).await.unwrap();
        assert!(store.load(&id).await.unwrap().is_some());
        let claimed_again = store.claim(&id).await.unwrap().unwrap();
        let attempt_again = claimed_again.resume_attempt_id.clone().unwrap();
        store.ack_claim(&id, &attempt_again).await.unwrap();
        assert!(store.load(&id).await.unwrap().is_none());
        let _ = std::fs::remove_dir_all(&temp_path);
    }

    #[tokio::test]
    async fn renewed_file_claim_is_not_recovered_after_original_lease_age() -> Result<()> {
        let temp_path =
            std::env::temp_dir().join(format!("echo_claim_renew_{}", uuid::Uuid::new_v4()));
        let store = FileCheckpointStore::new(&temp_path)
            .with_claim_timing(Duration::from_secs(300), Duration::from_secs(60))?;
        let checkpoint = Checkpoint::new(
            "claim-renew".to_string(),
            "node".to_string(),
            &SharedState::new(),
            Vec::new(),
            0,
            InterruptType::BeforeNode,
        );
        let id = checkpoint.id.clone();
        store.save(&checkpoint).await?;
        let claimed = store.claim(&id).await?.ok_or_else(|| {
            echo_core::error::ReactError::Other("checkpoint was not claimed".to_string())
        })?;
        let attempt_id = claimed.resume_attempt_id.clone().ok_or_else(|| {
            echo_core::error::ReactError::Other("claim has no attempt identity".to_string())
        })?;

        /* 直接回拨原租约时间，而不是依赖 120ms 窗口内的调度/fsync 时序。
        续租必须替换掉已过期的时间。 */
        let expired_at = Utc::now()
            .checked_sub_signed(chrono::Duration::seconds(600))
            .ok_or_else(|| {
                echo_core::error::ReactError::Other("claim timestamp underflow".to_string())
            })?;
        FileCheckpointStore::write_claim(
            &store.claim_path(&id, &attempt_id)?,
            &claimed,
            expired_at,
        )
        .await?;
        store.renew_claim(&id, &attempt_id).await?;

        assert!(store.claim(&id).await?.is_none());
        assert!(store.load(&id).await?.is_some_and(|checkpoint| {
            checkpoint.resume_attempt_id.as_deref() == Some(attempt_id.as_str())
        }));
        assert!(store.ack_claim(&id, "stale-attempt").await.is_err());
        store.requeue_claim(&id, &attempt_id).await?;
        let _ = std::fs::remove_dir_all(&temp_path);
        Ok(())
    }

    #[tokio::test]
    async fn owner_cleared_claim_recovers_without_accepting_stale_attempt() -> Result<()> {
        let temp_path =
            std::env::temp_dir().join(format!("echo_claim_cut_{}", uuid::Uuid::new_v4()));
        let store = FileCheckpointStore::new(&temp_path)
            .with_claim_timing(Duration::from_millis(80), Duration::from_millis(20))?;
        let checkpoint = Checkpoint::new(
            "claim-cut".to_string(),
            "node".to_string(),
            &SharedState::new(),
            Vec::new(),
            0,
            InterruptType::BeforeNode,
        );
        let id = checkpoint.id.clone();
        store.save(&checkpoint).await?;
        let claimed = store.claim(&id).await?.ok_or_else(|| {
            echo_core::error::ReactError::Other("checkpoint was not claimed".to_string())
        })?;
        let attempt_id = claimed.resume_attempt_id.clone().ok_or_else(|| {
            echo_core::error::ReactError::Other("claim has no attempt identity".to_string())
        })?;
        let claim_path = store.claim_path(&id, &attempt_id)?;

        let (mut owner_cleared, _) = FileCheckpointStore::read_claim(&claim_path).await?;
        owner_cleared.resume_attempt_id = None;
        FileCheckpointStore::write_checkpoint(&claim_path, &owner_cleared).await?;
        assert!(store.renew_claim(&id, &attempt_id).await.is_err());
        assert!(store.ack_claim(&id, &attempt_id).await.is_err());

        tokio::time::sleep(Duration::from_millis(100)).await;
        let recovered = store.load(&id).await?.ok_or_else(|| {
            echo_core::error::ReactError::Other("checkpoint was not recovered".to_string())
        })?;
        assert!(recovered.resume_attempt_id.is_none());
        let replacement = store.claim(&id).await?.ok_or_else(|| {
            echo_core::error::ReactError::Other("checkpoint was not reclaimable".to_string())
        })?;
        assert_ne!(
            replacement.resume_attempt_id.as_deref(),
            Some(attempt_id.as_str())
        );
        let replacement_attempt = replacement.resume_attempt_id.as_deref().ok_or_else(|| {
            echo_core::error::ReactError::Other("replacement claim has no attempt".to_string())
        })?;
        store.ack_claim(&id, replacement_attempt).await?;
        let _ = std::fs::remove_dir_all(&temp_path);
        Ok(())
    }

    #[tokio::test]
    async fn file_store_instances_share_claim_and_generation_lock() {
        let temp_path =
            std::env::temp_dir().join(format!("echo_claim_lock_{}", uuid::Uuid::new_v4()));
        let first = FileCheckpointStore::new(&temp_path);
        let second = FileCheckpointStore::new(&temp_path);
        let checkpoint = Checkpoint::new(
            "claim-lock".to_string(),
            "node".to_string(),
            &SharedState::new(),
            Vec::new(),
            0,
            InterruptType::BeforeNode,
        );
        let id = checkpoint.id.clone();
        first.save(&checkpoint).await.unwrap();
        let claimed = second.claim(&id).await.unwrap().unwrap();
        let mut tagged = claimed.clone();
        tagged.label = Some("must-not-resurrect".to_string());
        assert!(
            !first
                .save_if_generation(&tagged, checkpoint.generation)
                .await
                .unwrap()
        );
        let attempt_id = claimed.resume_attempt_id.clone().unwrap();
        second.requeue_claim(&id, &attempt_id).await.unwrap();
        assert!(
            first
                .load(&id)
                .await
                .unwrap()
                .is_some_and(|checkpoint| checkpoint.resume_attempt_id.is_none())
        );
        let _ = std::fs::remove_dir_all(&temp_path);
    }
}
