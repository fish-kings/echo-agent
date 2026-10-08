/*!
 * 带修订版本的任务关系服务，由轻量级进度跟踪与可执行任务图共享。
 *
 * 框架负责补丁语义、结构校验、修订版本与乐观并发控制；
 * 应用通过窄接口适配器提供持久化与产品策略。
 */

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use async_trait::async_trait;
use echo_core::tools::ToolContext;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use tokio::sync::RwLock;

use super::{
    PlanValidator, RuntimePlanSnapshot, RuntimeTaskClaimOutcome, Task, TaskClaim, TaskExecution,
    TaskId, TaskSpec, TaskStatus,
};

/// 产品适配器驱动任务图的方式。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskGraphExecutionMode {
    /// 按顺序逐个执行任务。
    Sequential,
    /// 依赖满足的任务可并行执行（默认）。
    #[default]
    Parallel,
}

/// 与运行时 DAG 快照一同携带的通用计划级上下文。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskGraphContext {
    /// 计划目标。
    pub goal: String,
    /// 计划所依赖的假设。
    pub assumptions: Vec<String>,
    /// 已识别的风险。
    pub risks: Vec<String>,
    /// 任务图的执行方式。
    pub execution_mode: TaskGraphExecutionMode,
    /// 产品自定义的附加元数据。
    #[serde(default)]
    pub metadata: serde_json::Value,
}

/// 一个一致的任务图修订版本及其计划级上下文。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RevisionedTaskGraph {
    /// 运行时 DAG 快照（含修订号与任务列表）。
    pub snapshot: RuntimePlanSnapshot,
    /// 计划级上下文。
    pub context: TaskGraphContext,
}

/// 在添加产品默认值与元数据之前、与产品无关的任务输入。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaskDraft {
    /// 任务 ID。
    pub id: String,
    /// 任务标题。
    pub title: String,
    /// 任务描述。
    pub description: String,
    /// 所依赖的任务 ID 列表。
    pub depends_on: Vec<TaskId>,
    /// 最大重试次数。
    pub max_retries: u32,
    /// 产品自有的扩展数据。
    pub extension: serde_json::Value,
}

impl TaskDraft {
    /// 在框架应用产品策略之前附加一个类型化扩展。
    pub fn with_extension<T: Serialize>(
        mut self,
        extension: T,
    ) -> std::result::Result<Self, String> {
        self.extension = serde_json::to_value(extension)
            .map_err(|error| format!("failed to encode task draft extension: {error}"))?;
        Ok(self)
    }

    /// 将草稿扩展解码为调用方指定的类型。
    pub fn extension_as<T: DeserializeOwned>(&self) -> std::result::Result<T, String> {
        serde_json::from_value(self.extension.clone())
            .map_err(|error| format!("failed to decode task draft extension: {error}"))
    }
}

/// 解析后的 `task_create` 输入。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaskCreateInput {
    /// 待创建的任务草稿。
    pub tasks: Vec<TaskDraft>,
    /// 追加到已有任务图时所基于的修订号。
    pub base_revision: Option<u64>,
    /// 本次创建的原因。
    pub reason: Option<String>,
    /// 计划假设。
    pub assumptions: Vec<String>,
    /// 计划风险。
    pub risks: Vec<String>,
    /// 任务图执行方式。
    pub execution_mode: TaskGraphExecutionMode,
}

/// 对单个不可变任务规格的部分更新。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskSpecPatch {
    /// 新标题。
    pub title: Option<String>,
    /// 新描述。
    pub description: Option<String>,
    /// 新的依赖列表（整体替换）。
    pub depends_on: Option<Vec<TaskId>>,
    /// 新的最大重试次数。
    pub max_retries: Option<u32>,
    /** 产品自有的部分扩展。对象值会递归合并；非对象值直接替换现有扩展。 */
    pub extension: Option<serde_json::Value>,
}

impl TaskSpecPatch {
    /** 在该补丁上设置完整的类型化扩展。仅在需要部分对象合并时才直接使用原始的 `extension` 字段。 */
    pub fn with_extension<T: Serialize>(
        mut self,
        extension: T,
    ) -> std::result::Result<Self, String> {
        self.extension = Some(
            serde_json::to_value(extension)
                .map_err(|error| format!("failed to encode task patch extension: {error}"))?,
        );
        Ok(self)
    }

    /// 当补丁携带完整扩展时将其解码。
    pub fn extension_as<T: DeserializeOwned>(&self) -> std::result::Result<Option<T>, String> {
        self.extension
            .clone()
            .map(serde_json::from_value)
            .transpose()
            .map_err(|error| format!("failed to decode task patch extension: {error}"))
    }
}

/// 框架补丁引擎所应用的规范操作。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskPlanPatchOp {
    /// 插入一个新任务。
    Insert {
        /// 插入到该任务之后；为 `None` 时插入到开头。
        after_task_id: Option<TaskId>,
        /// 待插入的任务规格。
        task: TaskSpec,
    },
    /// 更新某个任务的规格。
    Update {
        /// 目标任务 ID。
        task_id: TaskId,
        /// 规格补丁。
        patch: TaskSpecPatch,
    },
    /// 跳过某个任务。
    Skip {
        /// 目标任务 ID。
        task_id: TaskId,
    },
    /// 按给定顺序重排任务。
    Reorder {
        /// 新的任务 ID 顺序。
        task_ids: Vec<TaskId>,
    },
    /// 手动设置任务状态。
    SetStatus {
        /// 目标任务 ID。
        task_id: TaskId,
        /// 新状态。
        status: TaskStatus,
    },
}

/// 规范的乐观并发补丁。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaskPlanPatch {
    /// 补丁所基于的修订号。
    pub base_revision: u64,
    /// 修改原因。
    pub reason: String,
    /// 要应用的操作列表。
    pub operations: Vec<TaskPlanPatchOp>,
}

/// 解析后的更新操作，此时 Insert 草稿尚未获得产品默认值。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskPlanPatchInputOp {
    /// 插入一个新任务草稿。
    Insert {
        /// 插入到该任务之后；为 `None` 时插入到开头。
        after_task_id: Option<TaskId>,
        /// 待插入的任务草稿。
        task: TaskDraft,
    },
    /// 更新某个任务的规格。
    Update {
        /// 目标任务 ID。
        task_id: TaskId,
        /// 规格补丁。
        patch: TaskSpecPatch,
    },
    /// 跳过某个任务。
    Skip {
        /// 目标任务 ID。
        task_id: TaskId,
    },
    /// 按给定顺序重排任务。
    Reorder {
        /// 新的任务 ID 顺序。
        task_ids: Vec<TaskId>,
    },
    /// 手动设置任务状态。
    SetStatus {
        /// 目标任务 ID。
        task_id: TaskId,
        /// 新状态。
        status: TaskStatus,
    },
}

/// 解析后的 `task_update` 输入。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaskUpdateInput {
    /// 更新所基于的修订号。
    pub base_revision: u64,
    /// 修改原因。
    pub reason: String,
    /// 待应用的输入操作。
    pub operations: Vec<TaskPlanPatchInputOp>,
}

/// 与一次已提交修订一同持久化的通用效果。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskPatchEffects {
    /// 新插入的任务 ID。
    pub inserted_task_ids: Vec<TaskId>,
    /// 规格被更新的任务 ID。
    pub updated_task_ids: Vec<TaskId>,
    /// 被跳过的任务 ID。
    pub skipped_task_ids: Vec<TaskId>,
    /// 执行状态被重置的任务 ID。
    pub reset_task_ids: Vec<TaskId>,
    /// 手动推进了进度的任务 ID。
    pub progressed_task_ids: Vec<TaskId>,
    /// 是否发生了重排。
    pub reordered: bool,
}

/// 交给持久化适配器的、已计算完成的候选提交。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskGraphCommit {
    /// 期望的当前修订号；为 `None` 表示期望任务图尚不存在。
    pub expected_revision: Option<u64>,
    /**
     * 准备关系补丁时观察到的精确运行时执行状态。
     *
     * 即使关系修订号未变，只要任一条目发生变化，持久化适配器也必须拒绝该提交。
     * 创建操作以及不携带这一更强前置条件的旧版/直接提交保持为 `None`。
     */
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected_executions: Option<BTreeMap<TaskId, TaskExecution>>,
    /// 提交后的新任务图。
    pub next: RevisionedTaskGraph,
    /// 提交原因。
    pub reason: String,
    /// 本次提交的效果。
    pub effects: TaskPatchEffects,
}

/// 修订任务图存储的错误。
#[derive(Debug, Clone, thiserror::Error, PartialEq, Eq)]
pub enum RevisionedTaskStoreError {
    /// 指定作用域下不存在任务图。
    #[error("task graph not found for scope {scope_id}")]
    NotFound {
        /// 作用域 ID。
        scope_id: String,
    },
    /// 修订号冲突。
    #[error("task graph revision conflict: expected {expected:?}, current {current:?}")]
    Conflict {
        /// 期望的修订号。
        expected: Option<u64>,
        /// 实际的当前修订号。
        current: Option<u64>,
    },
    /// 存储拒绝了该操作。
    #[error("{message}")]
    Rejected {
        /// 错误信息。
        message: String,
    },
    /// 后端故障。
    #[error("{message}")]
    Backend {
        /// 错误信息。
        message: String,
    },
}

/// 产品任务策略的错误。
#[derive(Debug, Clone, thiserror::Error, PartialEq, Eq)]
pub enum TaskPolicyError {
    /// 作用域不可用。
    #[error("{message}")]
    ScopeUnavailable {
        /// 错误信息。
        message: String,
    },
    /// 策略拒绝。
    #[error("{message}")]
    Rejected {
        /// 错误信息。
        message: String,
    },
    /// 后端故障。
    #[error("{message}")]
    Backend {
        /// 错误信息。
        message: String,
    },
}

/// 任务修订服务的错误。
#[derive(Debug, Clone, thiserror::Error, PartialEq, Eq)]
pub enum TaskRevisionError {
    /// 输入不合法。
    #[error("{message}")]
    InvalidInput {
        /// 错误信息。
        message: String,
    },
    /// 指定作用域下不存在任务图。
    #[error("task graph not found for scope {scope_id}")]
    GraphNotFound {
        /// 作用域 ID。
        scope_id: String,
    },
    /// 任务不存在。
    #[error("task not found: {task_id}")]
    TaskNotFound {
        /// 任务 ID。
        task_id: TaskId,
    },
    /// 计划修订号冲突。
    #[error("plan revision conflict: expected {expected:?}, current {current:?}")]
    RevisionConflict {
        /// 期望的修订号。
        expected: Option<u64>,
        /// 实际的当前修订号。
        current: Option<u64>,
    },
    /// 补丁不合法。
    #[error("{message}")]
    InvalidPatch {
        /// 错误信息。
        message: String,
    },
    /// 被产品策略拒绝。
    #[error("{message}")]
    PolicyRejected {
        /// 错误信息。
        message: String,
    },
    /// 被存储拒绝。
    #[error("{message}")]
    StoreRejected {
        /// 错误信息。
        message: String,
    },
    /// 后端故障。
    #[error("{message}")]
    Backend {
        /// 错误信息。
        message: String,
    },
}

impl From<RevisionedTaskStoreError> for TaskRevisionError {
    /// 将存储错误映射为修订服务错误。
    fn from(error: RevisionedTaskStoreError) -> Self {
        match error {
            RevisionedTaskStoreError::NotFound { scope_id } => Self::GraphNotFound { scope_id },
            RevisionedTaskStoreError::Conflict { expected, current } => {
                Self::RevisionConflict { expected, current }
            }
            RevisionedTaskStoreError::Rejected { message } => Self::StoreRejected { message },
            RevisionedTaskStoreError::Backend { message } => Self::Backend { message },
        }
    }
}

impl From<TaskPolicyError> for TaskRevisionError {
    /// 将策略错误映射为修订服务错误。
    fn from(error: TaskPolicyError) -> Self {
        match error {
            TaskPolicyError::ScopeUnavailable { message }
            | TaskPolicyError::Rejected { message } => Self::PolicyRejected { message },
            TaskPolicyError::Backend { message } => Self::Backend { message },
        }
    }
}

/// 修订任务图的持久化边界。
#[async_trait]
pub trait RevisionedTaskStore: Send + Sync {
    /// 加载指定作用域的任务图；不存在时返回 `None`。
    async fn load(
        &self,
        scope_id: &str,
    ) -> Result<Option<RevisionedTaskGraph>, RevisionedTaskStoreError>;

    /// 比较期望修订号（及执行状态）后原子提交新任务图。
    async fn compare_and_commit(
        &self,
        scope_id: &str,
        commit: TaskGraphCommit,
    ) -> Result<RevisionedTaskGraph, RevisionedTaskStoreError>;
}

/// 产品策略生成的默认值；策略不得借此改写通用任务字段。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedTaskPolicy {
    /// 产品为任务准备的扩展数据。
    pub extension: serde_json::Value,
}

/// 任务工具与修订服务所使用的产品策略边界。
#[async_trait]
pub trait TaskToolPolicy: Send + Sync {
    /// 返回任务输入 JSON Schema 中的扩展属性定义。
    fn task_input_schema_extensions(&self) -> serde_json::Map<String, serde_json::Value> {
        serde_json::Map::from_iter([(
            "extension".to_string(),
            serde_json::json!({
                "type": "object",
                "description": "Product-owned task data ignored by framework scheduling"
            }),
        )])
    }

    /// 返回任务输入中必填的扩展属性名。
    fn required_task_input_extensions(&self) -> Vec<String> {
        Vec::new()
    }

    /// 是否允许通过补丁手动更新任务进度状态。
    fn allow_manual_progress_updates(&self) -> bool {
        false
    }

    /// 根据工具上下文解析任务图作用域 ID。
    async fn resolve_scope(&self, context: &ToolContext) -> Result<String, TaskPolicyError>;

    /**
     * 为一次任务图创建尝试准备产品作用域。
     *
     * 返回 `Err` 的实现必须先清除其暂存的所有未发布副作用。
     * 只有当本方法返回 `Ok` 后才会调用 [`Self::abort_scope_preparation`]。
     */
    async fn ensure_scope(
        &self,
        scope_id: &str,
        input: &TaskCreateInput,
        context: &ToolContext,
    ) -> Result<(), TaskPolicyError>;

    /**
     * 当任务准备、校验或任务图提交失败时，回滚 [`Self::ensure_scope`] 暂存的产品资源。
     *
     * 实现必须幂等，且不得删除已发布的作用域。每个策略都必须显式实现该方法，
     * 以免带副作用的策略悄悄遗漏回滚处理。
     */
    async fn abort_scope_preparation(&self, scope_id: &str) -> Result<(), TaskPolicyError>;

    /// 为指定位置的任务草稿生成产品默认值。
    async fn prepare_task(
        &self,
        scope_id: &str,
        draft: &TaskDraft,
        position: usize,
    ) -> Result<PreparedTaskPolicy, TaskPolicyError>;

    /// 为新建任务图生成初始计划级上下文。
    async fn prepare_initial_context(
        &self,
        scope_id: &str,
        input: &TaskCreateInput,
    ) -> Result<TaskGraphContext, TaskPolicyError>;

    /// 在提交前最终确定某个任务的扩展数据。
    async fn finalize_task_extension(
        &self,
        scope_id: &str,
        task_id: &str,
        position: usize,
        extension: serde_json::Value,
    ) -> Result<serde_json::Value, TaskPolicyError>;

    /// 对候选任务列表执行产品级校验。
    async fn validate_candidate(
        &self,
        scope_id: &str,
        tasks: &[Task],
    ) -> Result<(), TaskPolicyError>;
}

/// 框架中每个 Agent 轻量级任务图的默认策略。
#[derive(Debug, Clone)]
pub struct DefaultTaskToolPolicy {
    /// 上下文中缺少 run/conversation ID 时使用的作用域 ID。
    default_scope_id: String,
}

impl Default for DefaultTaskToolPolicy {
    /// 使用随机生成的默认作用域 ID 创建策略。
    fn default() -> Self {
        Self::new(format!("agent-task-scope-{}", uuid::Uuid::new_v4()))
    }
}

impl DefaultTaskToolPolicy {
    /// 使用指定的默认作用域 ID 创建策略。
    pub fn new(default_scope_id: impl Into<String>) -> Self {
        Self {
            default_scope_id: default_scope_id.into(),
        }
    }
}

#[async_trait]
impl TaskToolPolicy for DefaultTaskToolPolicy {
    /// 默认策略允许手动更新进度。
    fn allow_manual_progress_updates(&self) -> bool {
        true
    }

    /// 依次使用 run ID、会话 ID 与默认作用域 ID。
    async fn resolve_scope(&self, context: &ToolContext) -> Result<String, TaskPolicyError> {
        Ok(context
            .run_id
            .clone()
            .or_else(|| context.conversation_id.clone())
            .unwrap_or_else(|| self.default_scope_id.clone()))
    }

    /// 默认策略无需准备作用域。
    async fn ensure_scope(
        &self,
        _scope_id: &str,
        _input: &TaskCreateInput,
        _context: &ToolContext,
    ) -> Result<(), TaskPolicyError> {
        Ok(())
    }

    /// 默认策略没有需要回滚的资源。
    async fn abort_scope_preparation(&self, _scope_id: &str) -> Result<(), TaskPolicyError> {
        Ok(())
    }

    /// 直接沿用草稿中的扩展数据。
    async fn prepare_task(
        &self,
        _scope_id: &str,
        draft: &TaskDraft,
        _position: usize,
    ) -> Result<PreparedTaskPolicy, TaskPolicyError> {
        Ok(PreparedTaskPolicy {
            extension: draft.extension.clone(),
        })
    }

    /// 以首个任务标题作为目标生成初始上下文。
    async fn prepare_initial_context(
        &self,
        _scope_id: &str,
        input: &TaskCreateInput,
    ) -> Result<TaskGraphContext, TaskPolicyError> {
        let goal = input
            .tasks
            .first()
            .map(|task| task.title.clone())
            .unwrap_or_else(|| "Task graph".to_string());
        Ok(TaskGraphContext {
            goal,
            assumptions: input.assumptions.clone(),
            risks: input.risks.clone(),
            execution_mode: input.execution_mode,
            metadata: serde_json::Value::Null,
        })
    }

    /// 原样返回扩展数据。
    async fn finalize_task_extension(
        &self,
        _scope_id: &str,
        _task_id: &str,
        _position: usize,
        extension: serde_json::Value,
    ) -> Result<serde_json::Value, TaskPolicyError> {
        Ok(extension)
    }

    /// 默认策略不做额外校验。
    async fn validate_candidate(
        &self,
        _scope_id: &str,
        _tasks: &[Task],
    ) -> Result<(), TaskPolicyError> {
        Ok(())
    }
}

/// 默认框架 Agent 使用的、按实例隔离的内存 CAS 存储。
#[derive(Debug, Default)]
pub struct InMemoryRevisionedTaskStore {
    /// 作用域 ID 到任务图的映射。
    graphs: RwLock<HashMap<String, RevisionedTaskGraph>>,
}

impl InMemoryRevisionedTaskStore {
    /// 创建空存储。
    pub fn new() -> Self {
        Self::default()
    }

    /**
     * 在不改变不可变任务图修订号的前提下，原子地认领一个待执行的运行时任务。
     * 供基于本存储的 [`RuntimeDagController`](super::RuntimeDagController) 适配器使用。
     */
    pub async fn claim_runtime_task(
        &self,
        scope_id: &str,
        task: &Task,
        expected_revision: u64,
    ) -> Result<RuntimeTaskClaimOutcome, RevisionedTaskStoreError> {
        let mut graphs = self.graphs.write().await;
        let Some(graph) = graphs.get_mut(scope_id) else {
            return Err(RevisionedTaskStoreError::NotFound {
                scope_id: scope_id.to_string(),
            });
        };
        super::runtime_service::claim_runtime_task(&mut graph.snapshot, task, expected_revision)
            .map_err(|error| RevisionedTaskStoreError::Rejected {
                message: error.to_string(),
            })
    }

    /// 原子地结算精确匹配的物理认领。
    pub async fn settle_runtime_claim(
        &self,
        scope_id: &str,
        task_id: &str,
        claim: &TaskClaim,
        status: TaskStatus,
    ) -> Result<super::RuntimeTaskSettlementOutcome, RevisionedTaskStoreError> {
        let mut graphs = self.graphs.write().await;
        let Some(graph) = graphs.get_mut(scope_id) else {
            return Err(RevisionedTaskStoreError::NotFound {
                scope_id: scope_id.to_string(),
            });
        };
        super::runtime_service::settle_runtime_claim(&mut graph.snapshot, task_id, claim, status)
            .map_err(|error| RevisionedTaskStoreError::Rejected {
                message: error.to_string(),
            })
    }

    /// 使用规范的运行时变更，原子地将一个精确认领重新入队。
    pub async fn requeue_runtime_claim(
        &self,
        scope_id: &str,
        task_id: &str,
        claim: &TaskClaim,
        failure_fingerprint: Option<String>,
        error: String,
        exhaustion: super::RuntimeRetryExhaustion,
    ) -> Result<super::RuntimeTaskRequeueOutcome, RevisionedTaskStoreError> {
        let mut graphs = self.graphs.write().await;
        let Some(graph) = graphs.get_mut(scope_id) else {
            return Err(RevisionedTaskStoreError::NotFound {
                scope_id: scope_id.to_string(),
            });
        };
        super::runtime_service::requeue_runtime_claim(
            &mut graph.snapshot,
            task_id,
            claim,
            failure_fingerprint,
            error,
            exhaustion,
        )
        .map_err(|error| RevisionedTaskStoreError::Rejected {
            message: error.to_string(),
        })
    }

    /// 原子地应用一个由框架持有的调度决议请求。
    pub async fn settle_runtime_resolution(
        &self,
        scope_id: &str,
        task_id: &str,
        claim: &TaskClaim,
        request: super::RuntimeTaskResolutionRequest,
    ) -> Result<super::RuntimeTaskResolution, RevisionedTaskStoreError> {
        let mut graphs = self.graphs.write().await;
        let Some(graph) = graphs.get_mut(scope_id) else {
            return Err(RevisionedTaskStoreError::NotFound {
                scope_id: scope_id.to_string(),
            });
        };
        super::runtime_service::settle_runtime_resolution(
            &mut graph.snapshot,
            task_id,
            claim,
            request,
        )
        .map_err(|error| RevisionedTaskStoreError::Rejected {
            message: error.to_string(),
        })
    }

    /** 使用规范的运行时变更，原子地重启一个精确匹配的未认领任务。产品的运行/后代策略不在本存储职责内。 */
    pub async fn retry_runtime_task(
        &self,
        scope_id: &str,
        expected_task: &Task,
        expected_revision: u64,
    ) -> Result<super::RuntimeTaskRetryOutcome, RevisionedTaskStoreError> {
        let mut graphs = self.graphs.write().await;
        let Some(graph) = graphs.get_mut(scope_id) else {
            return Err(RevisionedTaskStoreError::NotFound {
                scope_id: scope_id.to_string(),
            });
        };
        super::runtime_service::retry_runtime_task(
            &mut graph.snapshot,
            expected_task,
            expected_revision,
        )
        .map_err(|error| RevisionedTaskStoreError::Rejected {
            message: error.to_string(),
        })
    }

    /// 原子地恢复一个已暂停任务，不消耗重试预算。
    pub async fn resume_runtime_task(
        &self,
        scope_id: &str,
        expected_task: &Task,
        expected_revision: u64,
    ) -> Result<super::RuntimeTaskResumeOutcome, RevisionedTaskStoreError> {
        let mut graphs = self.graphs.write().await;
        let Some(graph) = graphs.get_mut(scope_id) else {
            return Err(RevisionedTaskStoreError::NotFound {
                scope_id: scope_id.to_string(),
            });
        };
        super::runtime_service::resume_runtime_task(
            &mut graph.snapshot,
            expected_task,
            expected_revision,
        )
        .map_err(|error| RevisionedTaskStoreError::Rejected {
            message: error.to_string(),
        })
    }

    /// 在精确的任务图修订号上原子地结算一次运行级中断。
    pub async fn settle_runtime_interruption(
        &self,
        scope_id: &str,
        expected_revision: u64,
        disposition: super::RuntimeInterruptionDisposition,
    ) -> Result<super::RuntimeInterruptionSettlementOutcome, RevisionedTaskStoreError> {
        let mut graphs = self.graphs.write().await;
        let Some(graph) = graphs.get_mut(scope_id) else {
            return Err(RevisionedTaskStoreError::NotFound {
                scope_id: scope_id.to_string(),
            });
        };
        super::runtime_service::settle_runtime_interruption(
            &mut graph.snapshot,
            expected_revision,
            disposition,
        )
        .map_err(|error| RevisionedTaskStoreError::Rejected {
            message: error.to_string(),
        })
    }

    /// 判断给定认领是否仍是该任务当前有效的认领。
    pub async fn runtime_claim_is_current(
        &self,
        scope_id: &str,
        task_id: &str,
        claim: &TaskClaim,
    ) -> Result<bool, RevisionedTaskStoreError> {
        let graphs = self.graphs.read().await;
        let Some(graph) = graphs.get(scope_id) else {
            return Err(RevisionedTaskStoreError::NotFound {
                scope_id: scope_id.to_string(),
            });
        };
        super::runtime_service::runtime_claim_is_current(&graph.snapshot, task_id, claim).map_err(
            |error| RevisionedTaskStoreError::Rejected {
                message: error.to_string(),
            },
        )
    }
}

#[async_trait]
impl RevisionedTaskStore for InMemoryRevisionedTaskStore {
    /// 从内存中读取任务图副本。
    async fn load(
        &self,
        scope_id: &str,
    ) -> Result<Option<RevisionedTaskGraph>, RevisionedTaskStoreError> {
        Ok(self.graphs.read().await.get(scope_id).cloned())
    }

    /// 校验修订号与执行状态后替换内存中的任务图。
    async fn compare_and_commit(
        &self,
        scope_id: &str,
        commit: TaskGraphCommit,
    ) -> Result<RevisionedTaskGraph, RevisionedTaskStoreError> {
        let mut graphs = self.graphs.write().await;
        let current = graphs.get(scope_id).map(|graph| graph.snapshot.revision);
        if current != commit.expected_revision {
            return Err(RevisionedTaskStoreError::Conflict {
                expected: commit.expected_revision,
                current,
            });
        }
        if let Some(current_graph) = graphs.get(scope_id) {
            let execution_drifted = if let Some(expected_executions) = &commit.expected_executions {
                let current_executions: BTreeMap<TaskId, TaskExecution> = current_graph
                    .snapshot
                    .tasks
                    .iter()
                    .map(|task| (task.spec.id.clone(), task.execution.clone()))
                    .collect();
                &current_executions != expected_executions
            } else {
                let intentionally_progressed = commit
                    .effects
                    .progressed_task_ids
                    .iter()
                    .chain(&commit.effects.skipped_task_ids)
                    .chain(&commit.effects.reset_task_ids)
                    .collect::<HashSet<_>>();
                current_graph.snapshot.tasks.iter().any(|current_task| {
                    if intentionally_progressed.contains(&current_task.spec.id) {
                        return false;
                    }
                    commit
                        .next
                        .snapshot
                        .tasks
                        .iter()
                        .find(|next_task| next_task.spec.id == current_task.spec.id)
                        .is_some_and(|next_task| next_task.execution != current_task.execution)
                })
            };
            if execution_drifted {
                return Err(RevisionedTaskStoreError::Conflict {
                    expected: commit.expected_revision,
                    current,
                });
            }
        }
        let expected_next = match commit.expected_revision {
            Some(revision) => revision.checked_add(1),
            None => Some(1),
        }
        .ok_or_else(|| RevisionedTaskStoreError::Rejected {
            message: "task graph revision overflow".to_string(),
        })?;
        if commit.next.snapshot.revision != expected_next {
            return Err(RevisionedTaskStoreError::Rejected {
                message: format!(
                    "invalid next task graph revision: expected {expected_next}, got {}",
                    commit.next.snapshot.revision
                ),
            });
        }
        graphs.insert(scope_id.to_string(), commit.next.clone());
        Ok(commit.next)
    }
}

/// 补丁操作应用后的结果。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaskPatchApplication {
    /// 应用补丁后的任务列表。
    pub tasks: Vec<Task>,
    /// 补丁产生的效果。
    pub effects: TaskPatchEffects,
}

/// 纯函数式、由框架持有的任务补丁语义。
#[derive(Debug, Clone, Copy, Default)]
pub struct TaskPatchEngine;

impl TaskPatchEngine {
    /// 将补丁操作依次应用到当前任务列表上，返回新任务列表与效果。
    pub fn apply_operations(
        current: &[Task],
        operations: Vec<TaskPlanPatchOp>,
        allow_manual_progress_updates: bool,
    ) -> Result<TaskPatchApplication, TaskRevisionError> {
        if operations.is_empty() {
            return Err(TaskRevisionError::InvalidPatch {
                message: "task_update requires at least one operation".to_string(),
            });
        }

        let mut tasks = current.to_vec();
        let mut effects = TaskPatchEffects::default();
        for operation in operations {
            match operation {
                TaskPlanPatchOp::Insert {
                    after_task_id,
                    task,
                } => {
                    if tasks.iter().any(|existing| existing.spec.id == task.id) {
                        return Err(TaskRevisionError::InvalidPatch {
                            message: format!("task '{}' already exists", task.id),
                        });
                    }
                    let position = match after_task_id {
                        Some(after) => tasks
                            .iter()
                            .position(|existing| existing.spec.id == after)
                            .map(|index| index.saturating_add(1))
                            .ok_or_else(|| TaskRevisionError::TaskNotFound {
                                task_id: after.clone(),
                            })?,
                        None => tasks.len(),
                    };
                    effects.inserted_task_ids.push(task.id.clone());
                    let execution = TaskExecution::pending(task.id.clone());
                    tasks.insert(
                        position,
                        Task {
                            spec: task,
                            execution,
                        },
                    );
                }
                TaskPlanPatchOp::Update { task_id, patch } => {
                    let task = tasks
                        .iter_mut()
                        .find(|task| task.spec.id == task_id)
                        .ok_or_else(|| TaskRevisionError::TaskNotFound {
                            task_id: task_id.clone(),
                        })?;
                    if !matches!(
                        task.execution.status,
                        TaskStatus::Pending | TaskStatus::Blocked(_)
                    ) {
                        return Err(TaskRevisionError::InvalidPatch {
                            message: format!(
                                "cannot modify task '{}' in {:?}",
                                task.spec.id, task.execution.status
                            ),
                        });
                    }
                    if matches!(task.execution.status, TaskStatus::Blocked(_)) {
                        task.execution.status = TaskStatus::Pending;
                        effects.reset_task_ids.push(task_id.clone());
                    }
                    apply_spec_patch(&mut task.spec, patch);
                    task.execution.claim = None;
                    effects.updated_task_ids.push(task_id);
                }
                TaskPlanPatchOp::Skip { task_id } => {
                    let task = tasks
                        .iter_mut()
                        .find(|task| task.spec.id == task_id)
                        .ok_or_else(|| TaskRevisionError::TaskNotFound {
                            task_id: task_id.clone(),
                        })?;
                    if !matches!(
                        task.execution.status,
                        TaskStatus::Pending | TaskStatus::Blocked(_)
                    ) {
                        return Err(TaskRevisionError::InvalidPatch {
                            message: format!(
                                "cannot skip task '{}' in {:?}",
                                task.spec.id, task.execution.status
                            ),
                        });
                    }
                    task.execution.status = TaskStatus::Skipped;
                    task.execution.claim = None;
                    effects.skipped_task_ids.push(task_id);
                }
                TaskPlanPatchOp::Reorder { task_ids } => {
                    let expected = tasks
                        .iter()
                        .map(|task| task.spec.id.clone())
                        .collect::<HashSet<_>>();
                    let actual = task_ids.iter().cloned().collect::<HashSet<_>>();
                    if expected != actual || task_ids.len() != tasks.len() {
                        return Err(TaskRevisionError::InvalidPatch {
                            message: "reorder must contain every task id exactly once".to_string(),
                        });
                    }
                    let positions = task_ids
                        .into_iter()
                        .enumerate()
                        .map(|(position, task_id)| (task_id, position))
                        .collect::<HashMap<_, _>>();
                    tasks.sort_by_key(|task| {
                        positions.get(&task.spec.id).copied().unwrap_or(usize::MAX)
                    });
                    effects.reordered = true;
                }
                TaskPlanPatchOp::SetStatus { task_id, status } => {
                    if !allow_manual_progress_updates {
                        return Err(TaskRevisionError::PolicyRejected {
                            message: "manual task status updates are disabled".to_string(),
                        });
                    }
                    if !matches!(
                        status,
                        TaskStatus::Pending
                            | TaskStatus::Running
                            | TaskStatus::Completed
                            | TaskStatus::Cancelled
                    ) {
                        return Err(TaskRevisionError::InvalidPatch {
                            message: "manual task status must be pending, in_progress, completed, or cancelled"
                                .to_string(),
                        });
                    }
                    let task = tasks
                        .iter_mut()
                        .find(|task| task.spec.id == task_id)
                        .ok_or_else(|| TaskRevisionError::TaskNotFound {
                            task_id: task_id.clone(),
                        })?;
                    if task.execution.status != status {
                        task.execution.status = task
                            .execution
                            .status
                            .transition_to(status)
                            .map_err(|message| TaskRevisionError::InvalidPatch { message })?;
                        task.execution.claim = None;
                        effects.progressed_task_ids.push(task_id);
                    }
                }
            }
        }
        Ok(TaskPatchApplication { tasks, effects })
    }
}

/// 将规格补丁应用到任务规格上。
fn apply_spec_patch(spec: &mut TaskSpec, patch: TaskSpecPatch) {
    if let Some(title) = patch.title {
        spec.title = title;
    }
    if let Some(description) = patch.description {
        spec.description = description;
    }
    if let Some(depends_on) = patch.depends_on {
        spec.depends_on = depends_on;
    }
    if let Some(max_retries) = patch.max_retries {
        spec.max_retries = max_retries;
    }
    if let Some(extension) = patch.extension {
        merge_extension(&mut spec.extension, extension);
    }
}

/// 递归合并扩展：对象按键合并，其他值直接替换。
fn merge_extension(current: &mut serde_json::Value, patch: serde_json::Value) {
    match (current, patch) {
        (serde_json::Value::Object(current), serde_json::Value::Object(patch)) => {
            for (key, value) in patch {
                if let Some(existing) = current.get_mut(&key) {
                    merge_extension(existing, value);
                } else {
                    current.insert(key, value);
                }
            }
        }
        (current, patch) => *current = patch,
    }
}

/// 任务创建的结果。
#[derive(Debug, Clone)]
pub struct TaskCreateOutcome {
    /// 提交后的任务图。
    pub graph: RevisionedTaskGraph,
    /// 本次创建的任务数量。
    pub created_count: usize,
    /// 是否为追加到已有任务图。
    pub appended: bool,
}

/// 框架中唯一负责任务创建、修订补丁与读取的服务。
pub struct TaskRevisionService {
    /// 持久化存储。
    store: Arc<dyn RevisionedTaskStore>,
    /// 产品策略。
    policy: Arc<dyn TaskToolPolicy>,
    /// 结构校验器。
    validator: PlanValidator,
}

impl TaskRevisionService {
    /// 使用存储与策略创建服务，校验器取默认值。
    pub fn new(store: Arc<dyn RevisionedTaskStore>, policy: Arc<dyn TaskToolPolicy>) -> Self {
        Self {
            store,
            policy,
            validator: PlanValidator::default(),
        }
    }

    /// 替换结构校验器。
    pub fn with_validator(mut self, validator: PlanValidator) -> Self {
        self.validator = validator;
        self
    }

    /// 返回策略提供的任务输入 Schema 扩展属性。
    pub fn task_input_schema_extensions(&self) -> serde_json::Map<String, serde_json::Value> {
        self.policy.task_input_schema_extensions()
    }

    /// 返回策略要求的必填扩展属性名。
    pub fn required_task_input_extensions(&self) -> Vec<String> {
        self.policy.required_task_input_extensions()
    }

    /// 策略是否允许手动更新进度。
    pub fn allow_manual_progress_updates(&self) -> bool {
        self.policy.allow_manual_progress_updates()
    }

    /// 通过策略解析作用域 ID。
    pub async fn resolve_scope(&self, context: &ToolContext) -> Result<String, TaskRevisionError> {
        self.policy.resolve_scope(context).await.map_err(Into::into)
    }

    /// 加载指定作用域的任务图。
    pub async fn load(
        &self,
        scope_id: &str,
    ) -> Result<Option<RevisionedTaskGraph>, TaskRevisionError> {
        self.store.load(scope_id).await.map_err(Into::into)
    }

    /// 处理 `task_create` 工具输入：准备作用域并创建或追加任务，失败时回滚作用域准备。
    pub async fn create_from_tool(
        &self,
        input: TaskCreateInput,
        context: &ToolContext,
    ) -> Result<TaskCreateOutcome, TaskRevisionError> {
        if input.tasks.is_empty() {
            return Err(TaskRevisionError::InvalidInput {
                message: "task_create requires at least one task".to_string(),
            });
        }
        let scope_id = self.resolve_scope(context).await?;
        self.policy
            .ensure_scope(&scope_id, &input, context)
            .await
            .map_err(TaskRevisionError::from)?;
        match self.create_after_scope_preparation(&scope_id, input).await {
            Ok(outcome) => Ok(outcome),
            Err(create_error) => match self.policy.abort_scope_preparation(&scope_id).await {
                Ok(()) => Err(create_error),
                Err(abort_error) => Err(TaskRevisionError::Backend {
                    message: format!(
                        "task creation failed: {create_error}; scope preparation rollback failed: {abort_error}"
                    ),
                }),
            },
        }
    }

    /// 在作用域准备完成后执行实际的任务创建或追加。
    async fn create_after_scope_preparation(
        &self,
        scope_id: &str,
        input: TaskCreateInput,
    ) -> Result<TaskCreateOutcome, TaskRevisionError> {
        let current = self.load(scope_id).await?;
        let start_position = current
            .as_ref()
            .map(|graph| graph.snapshot.tasks.len())
            .unwrap_or(0);
        let mut prepared_tasks = Vec::with_capacity(input.tasks.len());
        for (offset, draft) in input.tasks.iter().enumerate() {
            let position = start_position.saturating_add(offset);
            let prepared = self
                .policy
                .prepare_task(scope_id, draft, position)
                .await
                .map_err(TaskRevisionError::from)?;
            prepared_tasks.push(TaskSpec {
                id: draft.id.clone(),
                title: draft.title.clone(),
                description: draft.description.clone(),
                depends_on: draft.depends_on.clone(),
                max_retries: draft.max_retries,
                extension: prepared.extension,
            });
        }

        if let Some(graph) = current {
            let base_revision = input
                .base_revision
                .filter(|revision| *revision > 0)
                .ok_or_else(|| TaskRevisionError::InvalidInput {
                    message: "task_create requires base_revision when tasks already exist"
                        .to_string(),
                })?;
            let reason = input
                .reason
                .as_deref()
                .map(str::trim)
                .filter(|reason| !reason.is_empty())
                .unwrap_or("add tasks")
                .to_string();
            let operations = prepared_tasks
                .into_iter()
                .map(|task| TaskPlanPatchOp::Insert {
                    after_task_id: None,
                    task,
                })
                .collect();
            let graph = self
                .apply_patch_to_loaded(
                    scope_id,
                    graph,
                    TaskPlanPatch {
                        base_revision,
                        reason,
                        operations,
                    },
                )
                .await?;
            return Ok(TaskCreateOutcome {
                graph,
                created_count: input.tasks.len(),
                appended: true,
            });
        }

        let graph_context = self
            .policy
            .prepare_initial_context(scope_id, &input)
            .await
            .map_err(TaskRevisionError::from)?;
        let tasks = prepared_tasks
            .into_iter()
            .map(|spec| {
                let execution = TaskExecution::pending(spec.id.clone());
                Task { spec, execution }
            })
            .collect();
        let graph = self
            .create_prepared(
                scope_id,
                graph_context,
                tasks,
                "initial complete plan".to_string(),
            )
            .await?;
        Ok(TaskCreateOutcome {
            graph,
            created_count: input.tasks.len(),
            appended: false,
        })
    }

    /// 处理 `task_update` 工具输入：为插入草稿补全产品默认值后应用补丁。
    pub async fn update_from_tool(
        &self,
        input: TaskUpdateInput,
        context: &ToolContext,
    ) -> Result<RevisionedTaskGraph, TaskRevisionError> {
        let scope_id = self.resolve_scope(context).await?;
        let current =
            self.load(&scope_id)
                .await?
                .ok_or_else(|| TaskRevisionError::GraphNotFound {
                    scope_id: scope_id.clone(),
                })?;
        let mut operations = Vec::with_capacity(input.operations.len());
        let mut insert_offset = 0usize;
        for operation in input.operations {
            let operation = match operation {
                TaskPlanPatchInputOp::Insert {
                    after_task_id,
                    task,
                } => {
                    let position = current.snapshot.tasks.len().saturating_add(insert_offset);
                    insert_offset = insert_offset.saturating_add(1);
                    let prepared = self
                        .policy
                        .prepare_task(&scope_id, &task, position)
                        .await
                        .map_err(TaskRevisionError::from)?;
                    TaskPlanPatchOp::Insert {
                        after_task_id,
                        task: TaskSpec {
                            id: task.id,
                            title: task.title,
                            description: task.description,
                            depends_on: task.depends_on,
                            max_retries: task.max_retries,
                            extension: prepared.extension,
                        },
                    }
                }
                TaskPlanPatchInputOp::Update { task_id, patch } => {
                    TaskPlanPatchOp::Update { task_id, patch }
                }
                TaskPlanPatchInputOp::Skip { task_id } => TaskPlanPatchOp::Skip { task_id },
                TaskPlanPatchInputOp::Reorder { task_ids } => TaskPlanPatchOp::Reorder { task_ids },
                TaskPlanPatchInputOp::SetStatus { task_id, status } => {
                    TaskPlanPatchOp::SetStatus { task_id, status }
                }
            };
            operations.push(operation);
        }
        self.apply_patch_to_loaded(
            &scope_id,
            current,
            TaskPlanPatch {
                base_revision: input.base_revision,
                reason: input.reason,
                operations,
            },
        )
        .await
    }

    /// 以修订号 1 创建一个已准备好的任务图。
    pub async fn create_prepared(
        &self,
        scope_id: &str,
        context: TaskGraphContext,
        tasks: Vec<Task>,
        reason: String,
    ) -> Result<RevisionedTaskGraph, TaskRevisionError> {
        let tasks = self.finalize_and_validate(scope_id, tasks).await?;
        let graph = RevisionedTaskGraph {
            snapshot: RuntimePlanSnapshot { revision: 1, tasks },
            context,
        };
        self.store
            .compare_and_commit(
                scope_id,
                TaskGraphCommit {
                    expected_revision: None,
                    expected_executions: None,
                    next: graph,
                    reason,
                    effects: TaskPatchEffects::default(),
                },
            )
            .await
            .map_err(Into::into)
    }

    /// 加载当前任务图并应用补丁。
    pub async fn apply_patch(
        &self,
        scope_id: &str,
        patch: TaskPlanPatch,
    ) -> Result<RevisionedTaskGraph, TaskRevisionError> {
        let current =
            self.load(scope_id)
                .await?
                .ok_or_else(|| TaskRevisionError::GraphNotFound {
                    scope_id: scope_id.to_string(),
                })?;
        self.apply_patch_to_loaded(scope_id, current, patch).await
    }

    /// 在已加载的任务图上校验并应用补丁，然后以 CAS 方式提交。
    async fn apply_patch_to_loaded(
        &self,
        scope_id: &str,
        current: RevisionedTaskGraph,
        patch: TaskPlanPatch,
    ) -> Result<RevisionedTaskGraph, TaskRevisionError> {
        if patch.base_revision == 0 {
            return Err(TaskRevisionError::InvalidInput {
                message: "task_update requires base_revision".to_string(),
            });
        }
        if current.snapshot.revision != patch.base_revision {
            return Err(TaskRevisionError::RevisionConflict {
                expected: Some(patch.base_revision),
                current: Some(current.snapshot.revision),
            });
        }
        if patch.reason.trim().is_empty() {
            return Err(TaskRevisionError::InvalidPatch {
                message: "task_update requires a non-empty reason".to_string(),
            });
        }
        let application = TaskPatchEngine::apply_operations(
            &current.snapshot.tasks,
            patch.operations,
            self.policy.allow_manual_progress_updates(),
        )?;
        let expected_executions = current
            .snapshot
            .tasks
            .iter()
            .map(|task| (task.spec.id.clone(), task.execution.clone()))
            .collect();
        let tasks = self
            .finalize_and_validate(scope_id, application.tasks)
            .await?;
        let next_revision = current.snapshot.revision.checked_add(1).ok_or_else(|| {
            TaskRevisionError::InvalidPatch {
                message: "task graph revision overflow".to_string(),
            }
        })?;
        let next = RevisionedTaskGraph {
            snapshot: RuntimePlanSnapshot {
                revision: next_revision,
                tasks,
            },
            context: current.context,
        };
        self.store
            .compare_and_commit(
                scope_id,
                TaskGraphCommit {
                    expected_revision: Some(patch.base_revision),
                    expected_executions: Some(expected_executions),
                    next,
                    reason: patch.reason,
                    effects: application.effects,
                },
            )
            .await
            .map_err(Into::into)
    }

    /// 最终确定各任务扩展，并执行策略校验与结构校验。
    async fn finalize_and_validate(
        &self,
        scope_id: &str,
        mut tasks: Vec<Task>,
    ) -> Result<Vec<Task>, TaskRevisionError> {
        for (position, task) in tasks.iter_mut().enumerate() {
            let extension = std::mem::take(&mut task.spec.extension);
            task.spec.extension = self
                .policy
                .finalize_task_extension(scope_id, &task.spec.id, position, extension)
                .await
                .map_err(TaskRevisionError::from)?;
        }
        self.policy
            .validate_candidate(scope_id, &tasks)
            .await
            .map_err(TaskRevisionError::from)?;
        self.validator
            .validate_task_snapshot(&tasks)
            .map_err(|errors| TaskRevisionError::InvalidPatch {
                message: errors.join("; "),
            })?;
        Ok(tasks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn draft(id: &str, depends_on: &[&str]) -> TaskDraft {
        TaskDraft {
            id: id.to_string(),
            title: format!("Task {id}"),
            description: format!("Do {id}"),
            depends_on: depends_on.iter().map(|item| (*item).to_string()).collect(),
            max_retries: 3,
            extension: serde_json::json!({ "check": "accepted" }),
        }
    }

    fn create_input(tasks: Vec<TaskDraft>) -> TaskCreateInput {
        TaskCreateInput {
            tasks,
            base_revision: None,
            reason: None,
            assumptions: Vec::new(),
            risks: Vec::new(),
            execution_mode: TaskGraphExecutionMode::Parallel,
        }
    }

    fn service() -> TaskRevisionService {
        TaskRevisionService::new(
            Arc::new(InMemoryRevisionedTaskStore::new()),
            Arc::new(DefaultTaskToolPolicy::new("test-scope")),
        )
    }

    struct ClaimBeforePatchCommitStore {
        inner: Arc<InMemoryRevisionedTaskStore>,
    }

    #[async_trait]
    impl RevisionedTaskStore for ClaimBeforePatchCommitStore {
        async fn load(
            &self,
            scope_id: &str,
        ) -> Result<Option<RevisionedTaskGraph>, RevisionedTaskStoreError> {
            self.inner.load(scope_id).await
        }

        async fn compare_and_commit(
            &self,
            scope_id: &str,
            commit: TaskGraphCommit,
        ) -> Result<RevisionedTaskGraph, RevisionedTaskStoreError> {
            if commit.expected_revision.is_some() {
                let current = self.inner.load(scope_id).await?.ok_or_else(|| {
                    RevisionedTaskStoreError::Backend {
                        message: "interleaving store could not load the patch graph".to_string(),
                    }
                })?;
                let expected = commit.expected_executions.as_ref().ok_or_else(|| {
                    RevisionedTaskStoreError::Backend {
                        message: "canonical patch omitted its execution precondition".to_string(),
                    }
                })?;
                let current_executions: BTreeMap<TaskId, TaskExecution> = current
                    .snapshot
                    .tasks
                    .iter()
                    .map(|task| (task.spec.id.clone(), task.execution.clone()))
                    .collect();
                if expected != &current_executions {
                    return Err(RevisionedTaskStoreError::Backend {
                        message: "canonical patch did not preserve its loaded execution snapshot"
                            .to_string(),
                    });
                }
                let task = current.snapshot.tasks.first().ok_or_else(|| {
                    RevisionedTaskStoreError::Backend {
                        message: "interleaving store could not find the patch task".to_string(),
                    }
                })?;
                if matches!(
                    self.inner
                        .claim_runtime_task(scope_id, task, current.snapshot.revision)
                        .await?,
                    RuntimeTaskClaimOutcome::ReloadSnapshot
                ) {
                    return Err(RevisionedTaskStoreError::Backend {
                        message: "interleaving claim unexpectedly requested a reload".to_string(),
                    });
                }
            }
            self.inner.compare_and_commit(scope_id, commit).await
        }
    }

    fn commit_from_loaded(
        loaded: &RevisionedTaskGraph,
        operation: TaskPlanPatchOp,
    ) -> Result<TaskGraphCommit, String> {
        let application =
            TaskPatchEngine::apply_operations(&loaded.snapshot.tasks, vec![operation], true)
                .map_err(|error| error.to_string())?;
        let revision = loaded
            .snapshot
            .revision
            .checked_add(1)
            .ok_or_else(|| "test graph revision overflow".to_string())?;
        Ok(TaskGraphCommit {
            expected_revision: Some(loaded.snapshot.revision),
            expected_executions: Some(
                loaded
                    .snapshot
                    .tasks
                    .iter()
                    .map(|task| (task.spec.id.clone(), task.execution.clone()))
                    .collect(),
            ),
            next: RevisionedTaskGraph {
                snapshot: RuntimePlanSnapshot {
                    revision,
                    tasks: application.tasks,
                },
                context: loaded.context.clone(),
            },
            reason: "deterministic execution CAS test".to_string(),
            effects: application.effects,
        })
    }

    async fn assert_claim_conflicts_with_patch(
        scope_id: &str,
        operation: TaskPlanPatchOp,
    ) -> Result<(), String> {
        let store = Arc::new(InMemoryRevisionedTaskStore::new());
        let service = TaskRevisionService::new(
            store.clone(),
            Arc::new(DefaultTaskToolPolicy::new(scope_id)),
        );
        service
            .create_from_tool(create_input(vec![draft("a", &[])]), &ToolContext::default())
            .await
            .map_err(|error| error.to_string())?;
        let loaded = store
            .load(scope_id)
            .await
            .map_err(|error| error.to_string())?
            .ok_or_else(|| "claim conflict graph is missing".to_string())?;
        let commit = commit_from_loaded(&loaded, operation)?;
        let task = loaded
            .snapshot
            .tasks
            .first()
            .ok_or_else(|| "claim conflict task is missing".to_string())?;
        let claim = match store
            .claim_runtime_task(scope_id, task, loaded.snapshot.revision)
            .await
            .map_err(|error| error.to_string())?
        {
            RuntimeTaskClaimOutcome::Claimed(claim) => claim,
            RuntimeTaskClaimOutcome::ReloadSnapshot => {
                return Err("claim conflict task unexpectedly requested reload".to_string());
            }
        };
        let error = store
            .compare_and_commit(scope_id, commit)
            .await
            .err()
            .ok_or_else(|| "relation patch overwrote a concurrent claim".to_string())?;
        assert!(matches!(error, RevisionedTaskStoreError::Conflict { .. }));
        assert_eq!(
            store
                .settle_runtime_claim(scope_id, "a", &claim, TaskStatus::Completed)
                .await
                .map_err(|error| error.to_string())?,
            crate::tasks::RuntimeTaskSettlementOutcome::Settled
        );
        Ok(())
    }

    #[test]
    fn extension_patch_merges_without_losing_product_fields() {
        let mut spec = TaskSpec {
            id: "task-1".to_string(),
            title: "Task".to_string(),
            description: "Do work".to_string(),
            depends_on: Vec::new(),
            max_retries: 3,
            extension: serde_json::json!({
                "kind": "implementation",
                "review": { "required": true, "role": "reviewer" },
            }),
        };
        apply_spec_patch(
            &mut spec,
            TaskSpecPatch {
                extension: Some(serde_json::json!({
                    "review": { "role": "senior-reviewer" },
                })),
                ..TaskSpecPatch::default()
            },
        );

        assert_eq!(
            spec.extension.get("kind"),
            Some(&serde_json::json!("implementation"))
        );
        assert_eq!(
            spec.extension.pointer("/review/required"),
            Some(&serde_json::json!(true))
        );
        assert_eq!(
            spec.extension.pointer("/review/role"),
            Some(&serde_json::json!("senior-reviewer"))
        );
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    struct TypedDraftExtension {
        role: String,
        priority: u32,
    }

    #[test]
    fn draft_and_patch_extensions_round_trip_through_typed_api() -> Result<(), String> {
        let extension = TypedDraftExtension {
            role: "explorer".to_string(),
            priority: 1,
        };
        let draft = draft("task-1", &[]).with_extension(extension.clone())?;
        assert_eq!(draft.extension_as::<TypedDraftExtension>()?, extension);

        let patch = TaskSpecPatch::default().with_extension(extension.clone())?;
        assert_eq!(
            patch.extension_as::<TypedDraftExtension>()?,
            Some(extension)
        );
        Ok(())
    }

    struct PreparationTrackingPolicy {
        delegate: DefaultTaskToolPolicy,
        abort_count: Arc<std::sync::atomic::AtomicUsize>,
        abort_error: Option<String>,
        ensure_error: Option<String>,
        scope_staged: Arc<std::sync::atomic::AtomicBool>,
    }

    #[async_trait]
    impl TaskToolPolicy for PreparationTrackingPolicy {
        async fn resolve_scope(&self, context: &ToolContext) -> Result<String, TaskPolicyError> {
            self.delegate.resolve_scope(context).await
        }

        async fn ensure_scope(
            &self,
            scope_id: &str,
            input: &TaskCreateInput,
            context: &ToolContext,
        ) -> Result<(), TaskPolicyError> {
            if let Some(message) = &self.ensure_error {
                self.scope_staged
                    .store(true, std::sync::atomic::Ordering::SeqCst);
                self.scope_staged
                    .store(false, std::sync::atomic::Ordering::SeqCst);
                return Err(TaskPolicyError::Rejected {
                    message: message.clone(),
                });
            }
            self.delegate.ensure_scope(scope_id, input, context).await
        }

        async fn abort_scope_preparation(&self, _scope_id: &str) -> Result<(), TaskPolicyError> {
            self.abort_count
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            match &self.abort_error {
                Some(message) => Err(TaskPolicyError::Backend {
                    message: message.clone(),
                }),
                None => Ok(()),
            }
        }

        async fn prepare_task(
            &self,
            scope_id: &str,
            draft: &TaskDraft,
            position: usize,
        ) -> Result<PreparedTaskPolicy, TaskPolicyError> {
            self.delegate.prepare_task(scope_id, draft, position).await
        }

        async fn prepare_initial_context(
            &self,
            scope_id: &str,
            input: &TaskCreateInput,
        ) -> Result<TaskGraphContext, TaskPolicyError> {
            self.delegate.prepare_initial_context(scope_id, input).await
        }

        async fn finalize_task_extension(
            &self,
            scope_id: &str,
            task_id: &str,
            position: usize,
            extension: serde_json::Value,
        ) -> Result<serde_json::Value, TaskPolicyError> {
            self.delegate
                .finalize_task_extension(scope_id, task_id, position, extension)
                .await
        }

        async fn validate_candidate(
            &self,
            scope_id: &str,
            tasks: &[Task],
        ) -> Result<(), TaskPolicyError> {
            self.delegate.validate_candidate(scope_id, tasks).await
        }
    }

    fn preparation_tracking_service(
        ensure_error: Option<String>,
        abort_error: Option<String>,
    ) -> (
        TaskRevisionService,
        Arc<InMemoryRevisionedTaskStore>,
        Arc<std::sync::atomic::AtomicUsize>,
        Arc<std::sync::atomic::AtomicBool>,
    ) {
        let store = Arc::new(InMemoryRevisionedTaskStore::new());
        let abort_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let scope_staged = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let policy = PreparationTrackingPolicy {
            delegate: DefaultTaskToolPolicy::new("preparation-scope"),
            abort_count: abort_count.clone(),
            abort_error,
            ensure_error,
            scope_staged: scope_staged.clone(),
        };
        (
            TaskRevisionService::new(store.clone(), Arc::new(policy)),
            store,
            abort_count,
            scope_staged,
        )
    }

    #[tokio::test]
    async fn ensure_scope_failure_self_cleans_without_invoking_abort() -> Result<(), String> {
        let (service, store, abort_count, scope_staged) =
            preparation_tracking_service(Some("scope staging failed".to_string()), None);
        let error = service
            .create_from_tool(create_input(vec![draft("a", &[])]), &ToolContext::default())
            .await
            .err()
            .ok_or_else(|| "scope preparation unexpectedly succeeded".to_string())?;
        assert!(matches!(error, TaskRevisionError::PolicyRejected { .. }));
        assert!(!scope_staged.load(std::sync::atomic::Ordering::SeqCst));
        assert_eq!(abort_count.load(std::sync::atomic::Ordering::SeqCst), 0);
        assert!(
            store
                .load("preparation-scope")
                .await
                .map_err(|load_error| load_error.to_string())?
                .is_none()
        );
        Ok(())
    }

    #[tokio::test]
    async fn failed_creation_aborts_scope_preparation_without_publishing_graph()
    -> Result<(), String> {
        let (service, store, abort_count, _scope_staged) = preparation_tracking_service(None, None);
        let error = service
            .create_from_tool(
                create_input(vec![draft("a", &["missing"])]),
                &ToolContext::default(),
            )
            .await
            .err()
            .ok_or_else(|| "invalid graph unexpectedly committed".to_string())?;
        assert!(matches!(error, TaskRevisionError::InvalidPatch { .. }));
        assert_eq!(abort_count.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert!(
            store
                .load("preparation-scope")
                .await
                .map_err(|load_error| load_error.to_string())?
                .is_none()
        );
        Ok(())
    }

    #[tokio::test]
    async fn scope_preparation_rollback_failure_preserves_both_errors() -> Result<(), String> {
        let (service, store, abort_count, _scope_staged) =
            preparation_tracking_service(None, Some("staging cleanup refused".to_string()));
        let error = service
            .create_from_tool(
                create_input(vec![draft("a", &["missing"])]),
                &ToolContext::default(),
            )
            .await
            .err()
            .ok_or_else(|| "invalid graph unexpectedly committed".to_string())?;
        let TaskRevisionError::Backend { message } = error else {
            return Err("rollback failure did not surface as backend error".to_string());
        };
        assert!(message.contains("missing"));
        assert!(message.contains("staging cleanup refused"));
        assert_eq!(abort_count.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert!(
            store
                .load("preparation-scope")
                .await
                .map_err(|load_error| load_error.to_string())?
                .is_none()
        );
        Ok(())
    }

    #[tokio::test]
    async fn creates_and_patches_one_canonical_graph() -> Result<(), String> {
        let service = service();
        let context = ToolContext::default();
        let created = service
            .create_from_tool(create_input(vec![draft("a", &[])]), &context)
            .await
            .map_err(|error| error.to_string())?;
        assert_eq!(created.graph.snapshot.revision, 1);

        let updated = service
            .update_from_tool(
                TaskUpdateInput {
                    base_revision: 1,
                    reason: "add dependency".to_string(),
                    operations: vec![TaskPlanPatchInputOp::Insert {
                        after_task_id: Some("a".to_string()),
                        task: draft("b", &["a"]),
                    }],
                },
                &context,
            )
            .await
            .map_err(|error| error.to_string())?;
        assert_eq!(updated.snapshot.revision, 2);
        assert_eq!(updated.snapshot.tasks.len(), 2);
        Ok(())
    }

    #[tokio::test]
    async fn stale_patch_reports_conflict() -> Result<(), String> {
        let service = service();
        let context = ToolContext::default();
        service
            .create_from_tool(create_input(vec![draft("a", &[])]), &context)
            .await
            .map_err(|error| error.to_string())?;
        let error = service
            .apply_patch(
                "test-scope",
                TaskPlanPatch {
                    base_revision: 2,
                    reason: "stale".to_string(),
                    operations: vec![TaskPlanPatchOp::Skip {
                        task_id: "a".to_string(),
                    }],
                },
            )
            .await
            .err()
            .ok_or_else(|| "stale patch unexpectedly succeeded".to_string())?;
        assert!(matches!(error, TaskRevisionError::RevisionConflict { .. }));
        Ok(())
    }

    #[tokio::test]
    async fn default_policy_supports_manual_progress() -> Result<(), String> {
        let service = service();
        let context = ToolContext::default();
        service
            .create_from_tool(create_input(vec![draft("a", &[])]), &context)
            .await
            .map_err(|error| error.to_string())?;
        let running = service
            .apply_patch(
                "test-scope",
                TaskPlanPatch {
                    base_revision: 1,
                    reason: "start".to_string(),
                    operations: vec![TaskPlanPatchOp::SetStatus {
                        task_id: "a".to_string(),
                        status: TaskStatus::Running,
                    }],
                },
            )
            .await
            .map_err(|error| error.to_string())?;
        let completed = service
            .apply_patch(
                "test-scope",
                TaskPlanPatch {
                    base_revision: running.snapshot.revision,
                    reason: "finish".to_string(),
                    operations: vec![TaskPlanPatchOp::SetStatus {
                        task_id: "a".to_string(),
                        status: TaskStatus::Completed,
                    }],
                },
            )
            .await
            .map_err(|error| error.to_string())?;
        let task = completed
            .snapshot
            .tasks
            .first()
            .ok_or_else(|| "completed graph lost its task".to_string())?;
        assert_eq!(task.execution.status, TaskStatus::Completed);
        Ok(())
    }

    #[tokio::test]
    async fn in_memory_store_applies_explicit_retry_with_exact_snapshot_cas() -> Result<(), String>
    {
        let store = Arc::new(InMemoryRevisionedTaskStore::new());
        let service = TaskRevisionService::new(
            store.clone(),
            Arc::new(DefaultTaskToolPolicy::new("retry-scope")),
        );
        service
            .create_from_tool(create_input(vec![draft("a", &[])]), &ToolContext::default())
            .await
            .map_err(|error| error.to_string())?;
        let pending = store
            .load("retry-scope")
            .await
            .map_err(|error| error.to_string())?
            .ok_or_else(|| "pending retry graph is missing".to_string())?;
        let pending_task = pending
            .snapshot
            .tasks
            .first()
            .ok_or_else(|| "pending retry task is missing".to_string())?;
        let claim = match store
            .claim_runtime_task("retry-scope", pending_task, pending.snapshot.revision)
            .await
            .map_err(|error| error.to_string())?
        {
            RuntimeTaskClaimOutcome::Claimed(claim) => claim,
            RuntimeTaskClaimOutcome::ReloadSnapshot => {
                return Err("pending retry task requested reload".to_string());
            }
        };
        assert_eq!(
            store
                .settle_runtime_claim(
                    "retry-scope",
                    "a",
                    &claim,
                    TaskStatus::Blocked("acceptance pending".to_string()),
                )
                .await
                .map_err(|error| error.to_string())?,
            crate::tasks::RuntimeTaskSettlementOutcome::Settled
        );
        let blocked = store
            .load("retry-scope")
            .await
            .map_err(|error| error.to_string())?
            .ok_or_else(|| "retry graph is missing".to_string())?;
        let expected = blocked
            .snapshot
            .tasks
            .first()
            .cloned()
            .ok_or_else(|| "retry task is missing".to_string())?;

        assert_eq!(
            store
                .retry_runtime_task("retry-scope", &expected, blocked.snapshot.revision)
                .await
                .map_err(|error| error.to_string())?,
            crate::tasks::RuntimeTaskRetryOutcome::Retried { retry_count: 1 }
        );
        assert_eq!(
            store
                .retry_runtime_task("retry-scope", &expected, blocked.snapshot.revision)
                .await
                .map_err(|error| error.to_string())?,
            crate::tasks::RuntimeTaskRetryOutcome::Superseded
        );
        let retried = store
            .load("retry-scope")
            .await
            .map_err(|error| error.to_string())?
            .ok_or_else(|| "retried graph is missing".to_string())?;
        let task = retried
            .snapshot
            .tasks
            .first()
            .ok_or_else(|| "retried task is missing".to_string())?;
        assert_eq!(task.execution.status, TaskStatus::Pending);
        assert_eq!(task.execution.retry_count, 1);
        assert!(task.execution.claim.is_none());
        Ok(())
    }

    #[tokio::test]
    async fn idempotent_running_patch_preserves_live_physical_claim() -> Result<(), String> {
        let store = Arc::new(InMemoryRevisionedTaskStore::new());
        let service = TaskRevisionService::new(
            store.clone(),
            Arc::new(DefaultTaskToolPolicy::new("claim-preservation")),
        );
        service
            .create_from_tool(create_input(vec![draft("a", &[])]), &ToolContext::default())
            .await
            .map_err(|error| error.to_string())?;
        let pending = store
            .load("claim-preservation")
            .await
            .map_err(|error| error.to_string())?
            .ok_or_else(|| "claim preservation graph is missing".to_string())?;
        let task = pending
            .snapshot
            .tasks
            .first()
            .cloned()
            .ok_or_else(|| "claim preservation task is missing".to_string())?;
        let claim = match store
            .claim_runtime_task("claim-preservation", &task, pending.snapshot.revision)
            .await
            .map_err(|error| error.to_string())?
        {
            RuntimeTaskClaimOutcome::Claimed(claim) => claim,
            RuntimeTaskClaimOutcome::ReloadSnapshot => {
                return Err("claim preservation unexpectedly reloaded".to_string());
            }
        };

        let patched = service
            .apply_patch(
                "claim-preservation",
                TaskPlanPatch {
                    base_revision: pending.snapshot.revision,
                    reason: "idempotent running observation".to_string(),
                    operations: vec![TaskPlanPatchOp::SetStatus {
                        task_id: "a".to_string(),
                        status: TaskStatus::Running,
                    }],
                },
            )
            .await
            .map_err(|error| error.to_string())?;
        let current = patched
            .snapshot
            .tasks
            .first()
            .ok_or_else(|| "patched claim task is missing".to_string())?;
        assert_eq!(current.execution.claim.as_ref(), Some(&claim));
        assert_eq!(
            store
                .settle_runtime_claim("claim-preservation", "a", &claim, TaskStatus::Completed,)
                .await
                .map_err(|error| error.to_string())?,
            crate::tasks::RuntimeTaskSettlementOutcome::Settled
        );
        Ok(())
    }

    #[tokio::test]
    async fn relation_patch_loaded_before_claim_cannot_overwrite_live_claim() -> Result<(), String>
    {
        let store = Arc::new(InMemoryRevisionedTaskStore::new());
        let service = TaskRevisionService::new(
            store.clone(),
            Arc::new(DefaultTaskToolPolicy::new("claim-patch-cas")),
        );
        service
            .create_from_tool(create_input(vec![draft("a", &[])]), &ToolContext::default())
            .await
            .map_err(|error| error.to_string())?;

        let loaded = store
            .load("claim-patch-cas")
            .await
            .map_err(|error| error.to_string())?
            .ok_or_else(|| "claim patch graph is missing".to_string())?;
        let application = TaskPatchEngine::apply_operations(
            &loaded.snapshot.tasks,
            vec![TaskPlanPatchOp::Skip {
                task_id: "a".to_string(),
            }],
            true,
        )
        .map_err(|error| error.to_string())?;
        let next = RevisionedTaskGraph {
            snapshot: RuntimePlanSnapshot {
                revision: loaded.snapshot.revision.saturating_add(1),
                tasks: application.tasks,
            },
            context: loaded.context.clone(),
        };
        let task = loaded
            .snapshot
            .tasks
            .first()
            .ok_or_else(|| "claim patch task is missing".to_string())?;
        let claim = match store
            .claim_runtime_task("claim-patch-cas", task, loaded.snapshot.revision)
            .await
            .map_err(|error| error.to_string())?
        {
            RuntimeTaskClaimOutcome::Claimed(claim) => claim,
            RuntimeTaskClaimOutcome::ReloadSnapshot => {
                return Err("claim patch task unexpectedly requested reload".to_string());
            }
        };

        let error = store
            .compare_and_commit(
                "claim-patch-cas",
                TaskGraphCommit {
                    expected_revision: Some(loaded.snapshot.revision),
                    expected_executions: Some(
                        loaded
                            .snapshot
                            .tasks
                            .iter()
                            .map(|task| (task.spec.id.clone(), task.execution.clone()))
                            .collect(),
                    ),
                    next,
                    reason: "skip stale loaded task".to_string(),
                    effects: application.effects,
                },
            )
            .await
            .err()
            .ok_or_else(|| "stale relation patch overwrote a live claim".to_string())?;
        assert!(matches!(error, RevisionedTaskStoreError::Conflict { .. }));

        let current = store
            .load("claim-patch-cas")
            .await
            .map_err(|error| error.to_string())?
            .ok_or_else(|| "claimed graph disappeared".to_string())?;
        let current_task = current
            .snapshot
            .tasks
            .first()
            .ok_or_else(|| "claimed task disappeared".to_string())?;
        assert_eq!(current_task.execution.claim.as_ref(), Some(&claim));
        Ok(())
    }

    #[tokio::test]
    async fn canonical_patch_carries_loaded_execution_precondition() -> Result<(), String> {
        let inner = Arc::new(InMemoryRevisionedTaskStore::new());
        let service = TaskRevisionService::new(
            Arc::new(ClaimBeforePatchCommitStore {
                inner: inner.clone(),
            }),
            Arc::new(DefaultTaskToolPolicy::new("canonical-patch-cas")),
        );
        service
            .create_from_tool(create_input(vec![draft("a", &[])]), &ToolContext::default())
            .await
            .map_err(|error| error.to_string())?;

        let error = service
            .apply_patch(
                "canonical-patch-cas",
                TaskPlanPatch {
                    base_revision: 1,
                    reason: "exercise canonical execution precondition".to_string(),
                    operations: vec![TaskPlanPatchOp::Skip {
                        task_id: "a".to_string(),
                    }],
                },
            )
            .await
            .err()
            .ok_or_else(|| "canonical patch overwrote the interleaved claim".to_string())?;
        assert!(matches!(error, TaskRevisionError::RevisionConflict { .. }));

        let current = inner
            .load("canonical-patch-cas")
            .await
            .map_err(|error| error.to_string())?
            .ok_or_else(|| "canonical patch graph disappeared".to_string())?;
        let task = current
            .snapshot
            .tasks
            .first()
            .ok_or_else(|| "canonical patch task disappeared".to_string())?;
        assert_eq!(task.execution.status, TaskStatus::Running);
        assert!(task.execution.claim.is_some());
        Ok(())
    }

    #[tokio::test]
    async fn set_status_and_spec_update_conflict_with_concurrent_claim() -> Result<(), String> {
        assert_claim_conflicts_with_patch(
            "claim-set-status-cas",
            TaskPlanPatchOp::SetStatus {
                task_id: "a".to_string(),
                status: TaskStatus::Running,
            },
        )
        .await?;
        assert_claim_conflicts_with_patch(
            "claim-spec-update-cas",
            TaskPlanPatchOp::Update {
                task_id: "a".to_string(),
                patch: TaskSpecPatch {
                    title: Some("Updated after stale load".to_string()),
                    ..TaskSpecPatch::default()
                },
            },
        )
        .await
    }

    #[tokio::test]
    async fn relation_patch_conflicts_with_runtime_settlement_and_retry() -> Result<(), String> {
        let store = Arc::new(InMemoryRevisionedTaskStore::new());
        let service = TaskRevisionService::new(
            store.clone(),
            Arc::new(DefaultTaskToolPolicy::new("settlement-patch-cas")),
        );
        service
            .create_from_tool(create_input(vec![draft("a", &[])]), &ToolContext::default())
            .await
            .map_err(|error| error.to_string())?;
        let pending = store
            .load("settlement-patch-cas")
            .await
            .map_err(|error| error.to_string())?
            .ok_or_else(|| "settlement graph is missing".to_string())?;
        let stale_skip = commit_from_loaded(
            &pending,
            TaskPlanPatchOp::Skip {
                task_id: "a".to_string(),
            },
        )?;
        let pending_task = pending
            .snapshot
            .tasks
            .first()
            .ok_or_else(|| "settlement task is missing".to_string())?;
        let claim = match store
            .claim_runtime_task(
                "settlement-patch-cas",
                pending_task,
                pending.snapshot.revision,
            )
            .await
            .map_err(|error| error.to_string())?
        {
            RuntimeTaskClaimOutcome::Claimed(claim) => claim,
            RuntimeTaskClaimOutcome::ReloadSnapshot => {
                return Err("settlement task unexpectedly requested reload".to_string());
            }
        };
        assert_eq!(
            store
                .settle_runtime_claim("settlement-patch-cas", "a", &claim, TaskStatus::Completed,)
                .await
                .map_err(|error| error.to_string())?,
            crate::tasks::RuntimeTaskSettlementOutcome::Settled
        );
        let settlement_error = store
            .compare_and_commit("settlement-patch-cas", stale_skip)
            .await
            .err()
            .ok_or_else(|| "relation patch overwrote runtime settlement".to_string())?;
        assert!(matches!(
            settlement_error,
            RevisionedTaskStoreError::Conflict { .. }
        ));

        let retry_store = Arc::new(InMemoryRevisionedTaskStore::new());
        let retry_service = TaskRevisionService::new(
            retry_store.clone(),
            Arc::new(DefaultTaskToolPolicy::new("retry-patch-cas")),
        );
        retry_service
            .create_from_tool(create_input(vec![draft("a", &[])]), &ToolContext::default())
            .await
            .map_err(|error| error.to_string())?;
        {
            let mut graphs = retry_store.graphs.write().await;
            let graph = graphs
                .get_mut("retry-patch-cas")
                .ok_or_else(|| "retry graph is missing".to_string())?;
            let task = graph
                .snapshot
                .tasks
                .first_mut()
                .ok_or_else(|| "retry task is missing".to_string())?;
            task.execution.status = TaskStatus::Blocked("retry".to_string());
        }
        let blocked = retry_store
            .load("retry-patch-cas")
            .await
            .map_err(|error| error.to_string())?
            .ok_or_else(|| "blocked graph is missing".to_string())?;
        let stale_update = commit_from_loaded(
            &blocked,
            TaskPlanPatchOp::Update {
                task_id: "a".to_string(),
                patch: TaskSpecPatch {
                    title: Some("Updated after retry".to_string()),
                    ..TaskSpecPatch::default()
                },
            },
        )?;
        let blocked_task = blocked
            .snapshot
            .tasks
            .first()
            .ok_or_else(|| "blocked retry task is missing".to_string())?;
        assert_eq!(
            retry_store
                .retry_runtime_task("retry-patch-cas", blocked_task, blocked.snapshot.revision)
                .await
                .map_err(|error| error.to_string())?,
            crate::tasks::RuntimeTaskRetryOutcome::Retried { retry_count: 1 }
        );
        let retry_error = retry_store
            .compare_and_commit("retry-patch-cas", stale_update)
            .await
            .err()
            .ok_or_else(|| "relation patch overwrote runtime retry".to_string())?;
        assert!(matches!(
            retry_error,
            RevisionedTaskStoreError::Conflict { .. }
        ));
        Ok(())
    }

    #[tokio::test]
    async fn patch_cannot_reduce_retry_budget_below_consumed_retries() -> Result<(), String> {
        let store = Arc::new(InMemoryRevisionedTaskStore::new());
        let service = TaskRevisionService::new(
            store.clone(),
            Arc::new(DefaultTaskToolPolicy::new("retry-budget")),
        );
        service
            .create_from_tool(create_input(vec![draft("a", &[])]), &ToolContext::default())
            .await
            .map_err(|error| error.to_string())?;
        {
            let mut graphs = store.graphs.write().await;
            let graph = graphs
                .get_mut("retry-budget")
                .ok_or_else(|| "retry budget graph is missing".to_string())?;
            let task = graph
                .snapshot
                .tasks
                .first_mut()
                .ok_or_else(|| "retry budget task is missing".to_string())?;
            task.execution.status = TaskStatus::Blocked("review".to_string());
            task.execution.retry_count = 2;
        }

        let error = service
            .apply_patch(
                "retry-budget",
                TaskPlanPatch {
                    base_revision: 1,
                    reason: "invalid retry budget reduction".to_string(),
                    operations: vec![TaskPlanPatchOp::Update {
                        task_id: "a".to_string(),
                        patch: TaskSpecPatch {
                            max_retries: Some(1),
                            ..TaskSpecPatch::default()
                        },
                    }],
                },
            )
            .await
            .err()
            .ok_or_else(|| "invalid retry budget patch unexpectedly committed".to_string())?;
        assert!(
            error
                .to_string()
                .contains("retry_count 2 exceeds max_retries 1")
        );
        let current = store
            .load("retry-budget")
            .await
            .map_err(|load_error| load_error.to_string())?
            .ok_or_else(|| "retry budget graph disappeared".to_string())?;
        let task = current
            .snapshot
            .tasks
            .first()
            .ok_or_else(|| "retry budget task disappeared".to_string())?;
        assert_eq!(task.spec.max_retries, 3);
        assert_eq!(task.execution.retry_count, 2);
        assert_eq!(
            task.execution.status,
            TaskStatus::Blocked("review".to_string())
        );
        Ok(())
    }

    #[tokio::test]
    async fn runtime_claims_are_parallel_aba_safe_and_relation_commits_detect_drift()
    -> Result<(), String> {
        let store = Arc::new(InMemoryRevisionedTaskStore::new());
        let service = TaskRevisionService::new(
            store.clone(),
            Arc::new(DefaultTaskToolPolicy::new("runtime-scope")),
        );
        service
            .create_from_tool(
                create_input(vec![draft("a", &[]), draft("b", &[])]),
                &ToolContext::default(),
            )
            .await
            .map_err(|error| error.to_string())?;
        let stale = store
            .load("runtime-scope")
            .await
            .map_err(|error| error.to_string())?
            .ok_or_else(|| "runtime graph missing".to_string())?;
        let task_a = stale
            .snapshot
            .tasks
            .iter()
            .find(|task| task.spec.id == "a")
            .ok_or_else(|| "task a missing".to_string())?;
        let task_b = stale
            .snapshot
            .tasks
            .iter()
            .find(|task| task.spec.id == "b")
            .ok_or_else(|| "task b missing".to_string())?;
        let claim_a = match store
            .claim_runtime_task("runtime-scope", task_a, stale.snapshot.revision)
            .await
            .map_err(|error| error.to_string())?
        {
            RuntimeTaskClaimOutcome::Claimed(claim) => claim,
            RuntimeTaskClaimOutcome::ReloadSnapshot => {
                return Err("first runtime claim requested reload".to_string());
            }
        };
        assert!(matches!(
            store
                .claim_runtime_task("runtime-scope", task_b, stale.snapshot.revision)
                .await
                .map_err(|error| error.to_string())?,
            RuntimeTaskClaimOutcome::Claimed(_)
        ));
        assert_eq!(
            store
                .settle_runtime_claim("runtime-scope", "a", &claim_a, TaskStatus::Completed,)
                .await
                .map_err(|error| error.to_string())?,
            crate::tasks::RuntimeTaskSettlementOutcome::Settled
        );
        assert_eq!(
            store
                .settle_runtime_claim(
                    "runtime-scope",
                    "a",
                    &claim_a,
                    TaskStatus::Failed("late result".to_string()),
                )
                .await
                .map_err(|error| error.to_string())?,
            crate::tasks::RuntimeTaskSettlementOutcome::Superseded
        );

        let mut stale_next = stale.clone();
        stale_next.snapshot.revision = 2;
        let stale_task = stale_next
            .snapshot
            .tasks
            .iter_mut()
            .find(|task| task.spec.id == "a")
            .ok_or_else(|| "stale task a missing".to_string())?;
        stale_task.spec.title = "stale title update".to_string();
        let error = store
            .compare_and_commit(
                "runtime-scope",
                TaskGraphCommit {
                    expected_revision: Some(1),
                    expected_executions: None,
                    next: stale_next,
                    reason: "stale relation edit".to_string(),
                    effects: TaskPatchEffects {
                        updated_task_ids: vec!["a".to_string()],
                        ..TaskPatchEffects::default()
                    },
                },
            )
            .await
            .err()
            .ok_or_else(|| "stale relation commit overwrote runtime claim".to_string())?;
        assert!(matches!(error, RevisionedTaskStoreError::Conflict { .. }));
        Ok(())
    }
}
