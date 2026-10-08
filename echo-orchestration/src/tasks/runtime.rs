/*!
通用任务运行时原语。

这些类型刻意保持与产品无关。应用层自行负责具体的持久化、审批关卡、
UI 投影以及事件协议。
*/

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use echo_core::error::Result;
pub use echo_core::tools::NestedDelegationPolicy;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use tokio_util::sync::CancellationToken;

/// 运行时 DAG 原语使用的稳定任务标识符。
pub type TaskId = String;

/**
针对运行时中断请求的、与产品无关的处置方式。

框架负责如何结算 claim 与未完成的任务。应用只需决定当前的停止请求
是终态取消还是可恢复的暂停，并提供暂停原因。
*/
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum RuntimeInterruptionDisposition {
    /// 永久停止运行，并取消所有未完成的任务。
    #[default]
    Cancelled,
    /// 停止派发，同时保留尚未启动的任务以便显式恢复。
    Paused {
        /// 暂停原因
        reason: String,
    },
}

/// 任务规格与托管记录共享的通用生命周期状态。
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskStatus {
    /// 等待执行
    #[default]
    Pending,
    /// 正在执行
    Running,
    /// 被阻塞，附带阻塞原因
    Blocked(String),
    /// 已成功完成
    Completed,
    /// 执行失败，附带错误信息
    Failed(String),
    /// 已被跳过
    Skipped,
    /// 已被取消
    Cancelled,
    /// 执行超时
    TimedOut {
        /// 超时错误信息
        error: String,
    },
    /// 正在重试
    Retrying {
        /// 当前重试次数
        attempt: u32,
        /// 上一次失败的错误信息
        last_error: String,
    },
    /// 已暂停，附带暂停原因
    Paused(String),
}

impl TaskStatus {
    /// 该状态是否已无后续自动工作。
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            Self::Completed
                | Self::Failed(_)
                | Self::TimedOut { .. }
                | Self::Skipped
                | Self::Cancelled
        )
    }

    /// 该任务当前是否占用执行容量。
    pub fn is_running(&self) -> bool {
        matches!(self, Self::Running | Self::Retrying { .. })
    }

    /// 从当前状态迁移到 `target` 是否合法。
    pub fn can_transition_to(&self, target: &Self) -> bool {
        match self {
            Self::Pending => matches!(
                target,
                Self::Running
                    | Self::Cancelled
                    | Self::Blocked(_)
                    | Self::Skipped
                    | Self::Paused(_)
            ),
            Self::Running => matches!(
                target,
                Self::Completed
                    | Self::Cancelled
                    | Self::Failed(_)
                    | Self::TimedOut { .. }
                    | Self::Retrying { .. }
                    | Self::Blocked(_)
                    | Self::Skipped
                    | Self::Paused(_)
            ),
            Self::Retrying { .. } => matches!(
                target,
                Self::Pending
                    | Self::Completed
                    | Self::Cancelled
                    | Self::Failed(_)
                    | Self::TimedOut { .. }
                    | Self::Retrying { .. }
                    | Self::Paused(_)
            ),
            Self::Blocked(_) => matches!(target, Self::Pending | Self::Cancelled),
            Self::Paused(_) => matches!(target, Self::Pending | Self::Cancelled),
            Self::Completed
            | Self::Failed(_)
            | Self::TimedOut { .. }
            | Self::Skipped
            | Self::Cancelled => false,
        }
    }

    /// 校验并返回所请求的状态迁移。
    pub fn transition_to(&self, target: Self) -> std::result::Result<Self, String> {
        if !self.can_transition_to(&target) {
            return Err(format!(
                "Invalid task state transition: {self:?} -> {target:?}"
            ));
        }
        Ok(target)
    }
}

/// 用于运行时 DAG 调度的、不可变且与产品无关的任务规格。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskSpec {
    /// 任务 ID
    pub id: TaskId,
    /// 任务标题
    pub title: String,
    /// 任务描述
    pub description: String,
    /// 所依赖的任务 ID 列表
    pub depends_on: Vec<TaskId>,
    /// 最大重试次数
    pub max_retries: u32,
    /// 由产品持有、不参与框架调度的数据。
    #[serde(default)]
    pub extension: serde_json::Value,
}

impl TaskSpec {
    /**
    附加一个带类型的产品扩展，而无需在每个应用边界暴露序列化细节。
    框架以 JSON 形式存储扩展，使任务图哈希、持久化和通用工具保持与产品无关。
    */
    pub fn with_extension<T: Serialize>(
        mut self,
        extension: T,
    ) -> std::result::Result<Self, String> {
        self.extension = serde_json::to_value(extension).map_err(|error| {
            format!("failed to encode extension for task '{}': {error}", self.id)
        })?;
        Ok(self)
    }

    /// 将产品扩展解码为调用方指定的类型。
    pub fn extension_as<T: DeserializeOwned>(&self) -> std::result::Result<T, String> {
        serde_json::from_value(self.extension.clone())
            .map_err(|error| format!("failed to decode extension for task '{}': {error}", self.id))
    }

    /// 单个不可变任务规格的稳定 SHA-256 标识。
    pub fn stable_hash(&self) -> std::result::Result<String, String> {
        let bytes = serde_json::to_vec(self).map_err(|error| {
            format!(
                "failed to serialize task '{}' for hashing: {error}",
                self.id
            )
        })?;
        Ok(format!("{:x}", Sha256::digest(bytes)))
    }
}

/// 针对一次具体任务派发尝试的持久租约。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskClaim {
    /**
    本次物理租约的唯一标识，包括执行器崩溃后对同一逻辑修订版本
    与重试尝试的重新认领。
    */
    pub claim_id: String,
    /// 计划修订版本号
    pub revision: u64,
    /// 尝试次数
    pub attempt: u32,
    /// 任务规格哈希
    pub spec_hash: String,
}

impl TaskClaim {
    /// 创建新的 claim，自动生成唯一的 `claim_id`
    pub fn new(revision: u64, attempt: u32, spec_hash: String) -> Self {
        Self {
            claim_id: uuid::Uuid::new_v4().to_string(),
            revision,
            attempt,
            spec_hash,
        }
    }

    /**
    全局唯一的执行标识。运行命名空间可防止不同 TaskRun 中相同的
    计划任务 / 修订版本 / 尝试共享生命周期。
    */
    pub fn execution_id(&self, run_id: &str, task_id: &str) -> String {
        format!(
            "{run_id}:{task_id}:{}:{}:{}",
            self.revision, self.attempt, self.claim_id
        )
    }
}

/// 单个运行时任务规格的可变执行状态。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskExecution {
    /// 对应的任务 ID
    pub task_id: TaskId,
    /// 当前生命周期状态
    pub status: TaskStatus,
    /// 已重试次数
    pub retry_count: u32,
    /// 失败指纹，用于识别重复失败
    pub failure_fingerprint: Option<String>,
    /// 当前持有的派发租约
    #[serde(default)]
    pub claim: Option<TaskClaim>,
}

impl TaskExecution {
    /// 创建处于 Pending 状态的初始执行记录
    pub fn pending(task_id: impl Into<TaskId>) -> Self {
        Self {
            task_id: task_id.into(),
            status: TaskStatus::Pending,
            retry_count: 0,
            failure_fingerprint: None,
            claim: None,
        }
    }
}

/// 一个运行时 DAG 节点：不可变规格加上可变执行状态。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Task {
    /// 不可变的任务规格
    pub spec: TaskSpec,
    /// 可变的执行状态
    pub execution: TaskExecution,
}

/// 传递给任务 Subagent 的、与产品无关的上下文。
#[derive(Debug, Clone)]
pub struct TaskSubagentContext {
    /// 所属运行 ID
    run_id: String,
    /// 取消令牌
    cancel: CancellationToken,
    /// 嵌套委派策略
    delegation_policy: NestedDelegationPolicy,
    /// 因 Skipped 生命周期而被显式豁免（而非复用成功输出）的依赖。
    waived_dependency_ids: Vec<TaskId>,
    /// 绑定的任务 ID
    task_id: Option<TaskId>,
    /// 绑定的持久 claim
    claim: Option<TaskClaim>,
}

impl TaskSubagentContext {
    /// 创建仅用于测试的上下文
    #[cfg(test)]
    pub(crate) fn new(run_id: impl Into<String>) -> Self {
        Self {
            run_id: run_id.into(),
            cancel: CancellationToken::new(),
            delegation_policy: NestedDelegationPolicy::default(),
            waived_dependency_ids: Vec::new(),
            task_id: None,
            claim: None,
        }
    }

    /// 基于一个确切的持久 claim 构造绑定运行时的上下文。
    pub(crate) fn from_claim(
        run_id: impl Into<String>,
        task_id: impl Into<TaskId>,
        claim: TaskClaim,
        cancel: CancellationToken,
    ) -> std::result::Result<Self, String> {
        let run_id = run_id.into();
        let task_id = task_id.into();
        if run_id.trim().is_empty() {
            return Err("runtime task context requires run_id".to_string());
        }
        if task_id.trim().is_empty() {
            return Err("runtime task context requires task_id".to_string());
        }
        if claim.revision == 0 || claim.attempt == 0 {
            return Err(
                "runtime task context requires a non-zero claim revision and attempt".to_string(),
            );
        }
        Ok(Self {
            run_id,
            cancel,
            delegation_policy: NestedDelegationPolicy::default(),
            waived_dependency_ids: Vec::new(),
            task_id: Some(task_id),
            claim: Some(claim),
        })
    }

    /// 当该上下文绑定运行时时，返回确切的持久 claim。
    pub fn claim(&self) -> Option<&TaskClaim> {
        self.claim.as_ref()
    }

    /// 返回绑定的任务 ID
    pub fn task_id(&self) -> Option<&str> {
        self.task_id.as_deref()
    }

    /// 返回所属运行 ID
    pub fn run_id(&self) -> &str {
        &self.run_id
    }

    /// 返回取消令牌
    pub fn cancellation_token(&self) -> &CancellationToken {
        &self.cancel
    }

    /// 返回嵌套委派策略
    pub fn delegation_policy(&self) -> NestedDelegationPolicy {
        self.delegation_policy
    }

    /// 返回被豁免的依赖 ID 列表
    pub fn waived_dependency_ids(&self) -> &[TaskId] {
        &self.waived_dependency_ids
    }

    /**
    返回由绑定任务与 claim 派生出的稳定执行标识。
    调用方无法替换为其他任务 ID。
    */
    pub fn execution_id(&self) -> Option<String> {
        self.claim
            .as_ref()
            .zip(self.task_id.as_deref())
            .map(|(claim, task_id)| claim.execution_id(&self.run_id, task_id))
    }

    /**
    将确切的上下文转换为框架现有的运行时血缘载体，
    且不允许调用方替换由 claim 派生的标识。
    */
    pub fn runtime_context(&self) -> echo_core::tools::ExternalRunContext {
        let (execution_id, task_id, attempt, plan_revision) = self
            .claim
            .as_ref()
            .zip(self.task_id.as_deref())
            .map(|(claim, task_id)| {
                (
                    claim.execution_id(&self.run_id, task_id),
                    task_id.to_string(),
                    claim.attempt,
                    claim.revision,
                )
            })
            .unwrap_or_else(|| (String::new(), String::new(), 0, 0));
        echo_core::tools::ExternalRunContext {
            run_id: Some(self.run_id.clone()),
            execution_id: (!execution_id.is_empty()).then_some(execution_id.clone()),
            turn_id: (!execution_id.is_empty()).then_some(execution_id),
            cancel: Some(std::sync::Arc::new(self.cancel.clone())),
            subagent_lineage: Some(echo_core::tools::SubagentLineage {
                task_id: (!task_id.is_empty()).then_some(task_id),
                attempt: (attempt != 0).then_some(attempt),
                plan_revision: (plan_revision != 0).then_some(plan_revision),
                run_id: Some(self.run_id.clone()),
                ..echo_core::tools::SubagentLineage::default()
            }),
            ..echo_core::tools::ExternalRunContext::default()
        }
    }

    /// 设置嵌套委派策略
    pub(crate) fn with_delegation_policy(mut self, policy: NestedDelegationPolicy) -> Self {
        self.delegation_policy = policy;
        self
    }

    /// 设置被豁免的依赖 ID 列表
    pub(crate) fn with_waived_dependencies(mut self, dependency_ids: Vec<TaskId>) -> Self {
        self.waived_dependency_ids = dependency_ids;
        self
    }

    /// 构造子委派上下文；若策略不允许继续委派则返回 None
    pub fn child_delegation_context(&self) -> Option<Self> {
        self.delegation_policy
            .child_policy()
            .map(|delegation_policy| Self {
                run_id: self.run_id.clone(),
                cancel: self.cancel.child_token(),
                delegation_policy,
                waived_dependency_ids: self.waived_dependency_ids.clone(),
                task_id: self.task_id.clone(),
                claim: self.claim.clone(),
            })
    }
}

/// 由 Subagent 提出的有界后续任务。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SuggestedTask {
    /// 任务标题
    pub title: String,
    /// 任务描述
    pub description: String,
    /// 所依赖的任务 ID 列表
    #[serde(default)]
    pub dependencies: Vec<TaskId>,
    /// 需要该任务的理由
    pub why_needed: String,
    /// 风险说明
    pub risk: String,
    /// 由产品持有的扩展数据
    #[serde(default)]
    pub extension: serde_json::Value,
}

/// 在任务边界生成的精简单任务执行摘要。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskExecutionSummary {
    /// 所属运行 ID
    pub run_id: String,
    /// 任务 ID
    pub task_id: TaskId,
    /// 执行该任务的 Subagent 名称
    pub subagent_name: String,
    /// 已完成的工作
    pub completed_work: Vec<String>,
    /// 做出的决策
    pub decisions: Vec<String>,
    /// 遇到的失败
    pub failures: Vec<String>,
    /// 验证情况
    pub verification: Vec<String>,
    /// 对后续任务的影响
    pub next_implications: Vec<String>,
    /// 建议的后续任务
    #[serde(default)]
    pub suggested_tasks: Vec<SuggestedTask>,
    /// 由产品持有的证据、产物、涉及的资源或 UI 元数据。
    #[serde(default)]
    pub extension: serde_json::Value,
    /// 创建时间
    #[serde(with = "echo_core::utils::time::local_rfc3339")]
    pub created_at: DateTime<Utc>,
}

/**
与产品无关的 Subagent 契约。

框架定义契约；产品决定使用哪种具体的 Subagent 实现、工具、存储以及 UI 事件映射。
*/
#[async_trait]
pub trait TaskSubagent: Send + Sync {
    /// 执行任务，并基于依赖任务的摘要生成本任务的执行摘要
    async fn execute(
        &self,
        context: TaskSubagentContext,
        task: Task,
        dependency_summaries: Vec<TaskExecutionSummary>,
    ) -> Result<TaskExecutionSummary>;
}

/// 供任务运行时执行器使用的纯 DAG 簿记。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DagExecutionState {
    /// 已完成的任务
    pub completed: HashSet<TaskId>,
    /// 正在执行中的任务
    pub in_flight: HashSet<TaskId>,
    /// 失败或超时的任务
    pub failed: HashSet<TaskId>,
    /// 已跳过的任务
    pub skipped: HashSet<TaskId>,
    /// 已取消的任务
    pub cancelled: HashSet<TaskId>,
    /// 已暂停的任务
    pub paused: HashSet<TaskId>,
}

/**
已提交图快照中单个任务的派生依赖状态。

这是一种投影，而非持久化的任务生命周期。特别地，
当失败的祖先任务在更新的快照中被重试时，
[`BlockedByFailure`](Self::BlockedByFailure) 会自动消失。
应用仍可为产品自有的审核或输入策略持久化 [`TaskStatus::Blocked`]，
但依赖遍历不得把该投影编码为状态字符串。
*/
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DagDependencyState {
    /// 所有依赖均已成功完成或被显式豁免。
    Satisfied {
        /// 因被跳过（Skipped）而显式豁免的依赖。
        waived_dependency_ids: Vec<TaskId>,
    },
    /// 没有依赖失败，但至少有一个尚未完成。
    Waiting {
        /// 尚未解决的依赖 ID
        unresolved_dependency_ids: Vec<TaskId>,
    },
    /// 一个或多个传递祖先任务失败或超时。
    BlockedByFailure {
        /// 失败的祖先任务 ID
        failed_ancestor_ids: Vec<TaskId>,
    },
}

impl DagExecutionState {
    /**
    从快照构建状态。已完成的任务视为已解决；
    正在运行的任务视为外部执行中。
    */
    pub fn from_tasks(tasks: &[Task]) -> Self {
        let completed = tasks
            .iter()
            .filter(|task| task.execution.status == TaskStatus::Completed)
            .map(|task| task.spec.id.clone())
            .collect();
        let in_flight = tasks
            .iter()
            .filter(|task| task.execution.status.is_running())
            .map(|task| task.spec.id.clone())
            .collect();
        let failed = tasks
            .iter()
            .filter(|task| {
                matches!(
                    task.execution.status,
                    TaskStatus::Failed(_) | TaskStatus::TimedOut { .. }
                )
            })
            .map(|task| task.spec.id.clone())
            .collect();
        let skipped = tasks
            .iter()
            .filter(|task| task.execution.status == TaskStatus::Skipped)
            .map(|task| task.spec.id.clone())
            .collect();
        let cancelled = tasks
            .iter()
            .filter(|task| task.execution.status == TaskStatus::Cancelled)
            .map(|task| task.spec.id.clone())
            .collect();
        let paused = tasks
            .iter()
            .filter(|task| matches!(task.execution.status, TaskStatus::Paused(_)))
            .map(|task| task.spec.id.clone())
            .collect();

        Self {
            completed,
            in_flight,
            failed,
            skipped,
            cancelled,
            paused,
        }
    }

    /// 根据更新的任务快照刷新外部执行中的任务。
    pub fn refresh_in_flight(&mut self, tasks: &[Task]) -> DagRefresh {
        let mut refresh = DagRefresh::default();
        if self.in_flight.is_empty() {
            return refresh;
        }

        for task_id in self.in_flight.clone() {
            let Some(task) = tasks.iter().find(|task| task.spec.id == task_id) else {
                continue;
            };

            match &task.execution.status {
                TaskStatus::Completed => {
                    self.in_flight.remove(&task_id);
                    self.completed.insert(task_id.clone());
                    refresh.completed.push(task_id);
                }
                TaskStatus::Failed(_) | TaskStatus::TimedOut { .. } => {
                    self.in_flight.remove(&task_id);
                    self.failed.insert(task_id.clone());
                    refresh.failed.push(task_id);
                }
                TaskStatus::Skipped | TaskStatus::Cancelled => {
                    self.in_flight.remove(&task_id);
                    if task.execution.status == TaskStatus::Skipped {
                        self.skipped.insert(task_id.clone());
                    } else {
                        self.cancelled.insert(task_id.clone());
                    }
                    refresh.terminal_non_success.push(task_id);
                }
                TaskStatus::Pending
                | TaskStatus::Running
                | TaskStatus::Retrying { .. }
                | TaskStatus::Blocked(_)
                | TaskStatus::Paused(_) => {}
            }
        }

        refresh
    }

    /// 可在下一波次派发的任务 ID。
    pub fn ready_task_ids(&self, tasks: &[Task]) -> Vec<TaskId> {
        let dependencies = self.dependency_states(tasks);
        tasks
            .iter()
            .filter(|task| {
                !self.completed.contains(&task.spec.id)
                    && !self.in_flight.contains(&task.spec.id)
                    && !self.failed.contains(&task.spec.id)
                    && !self.skipped.contains(&task.spec.id)
                    && !self.cancelled.contains(&task.spec.id)
            })
            .filter(|task| {
                task.execution.status == TaskStatus::Pending
                    && matches!(
                        dependencies.get(&task.spec.id),
                        Some(DagDependencyState::Satisfied { .. })
                    )
            })
            .map(|task| task.spec.id.clone())
            .collect()
    }

    /// 在不修改快照的前提下推导每个任务的依赖状态。
    pub fn dependency_states(&self, tasks: &[Task]) -> HashMap<TaskId, DagDependencyState> {
        let mut failed_ancestors: HashMap<TaskId, HashSet<TaskId>> = self
            .failed
            .iter()
            .map(|task_id| (task_id.clone(), HashSet::from([task_id.clone()])))
            .collect();
        loop {
            let mut changed = false;
            for task in tasks {
                if self.completed.contains(&task.spec.id)
                    || self.skipped.contains(&task.spec.id)
                    || self.cancelled.contains(&task.spec.id)
                    || self.failed.contains(&task.spec.id)
                {
                    continue;
                }
                let mut inherited = failed_ancestors
                    .get(&task.spec.id)
                    .cloned()
                    .unwrap_or_default();
                for dependency in &task.spec.depends_on {
                    if let Some(ancestors) = failed_ancestors.get(dependency) {
                        inherited.extend(ancestors.iter().cloned());
                    }
                }
                if inherited.is_empty() {
                    continue;
                }
                let entry = failed_ancestors.entry(task.spec.id.clone()).or_default();
                let before = entry.len();
                entry.extend(inherited);
                if entry.len() != before {
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        tasks
            .iter()
            .map(|task| {
                let state = if !self.failed.contains(&task.spec.id) {
                    match failed_ancestors.get(&task.spec.id) {
                        Some(ancestors) if !ancestors.is_empty() => {
                            let mut failed_ancestor_ids: Vec<_> =
                                ancestors.iter().cloned().collect();
                            failed_ancestor_ids.sort();
                            DagDependencyState::BlockedByFailure {
                                failed_ancestor_ids,
                            }
                        }
                        _ => {
                            let mut unresolved_dependency_ids: Vec<_> = task
                                .spec
                                .depends_on
                                .iter()
                                .filter(|dependency| {
                                    !self.completed.contains(*dependency)
                                        && !self.skipped.contains(*dependency)
                                })
                                .cloned()
                                .collect();
                            unresolved_dependency_ids.sort();
                            if unresolved_dependency_ids.is_empty() {
                                let mut waived_dependency_ids: Vec<_> = task
                                    .spec
                                    .depends_on
                                    .iter()
                                    .filter(|dependency| self.skipped.contains(*dependency))
                                    .cloned()
                                    .collect();
                                waived_dependency_ids.sort();
                                DagDependencyState::Satisfied {
                                    waived_dependency_ids,
                                }
                            } else {
                                DagDependencyState::Waiting {
                                    unresolved_dependency_ids,
                                }
                            }
                        }
                    }
                } else {
                    let mut unresolved_dependency_ids: Vec<_> = task
                        .spec
                        .depends_on
                        .iter()
                        .filter(|dependency| {
                            !self.completed.contains(*dependency)
                                && !self.skipped.contains(*dependency)
                        })
                        .cloned()
                        .collect();
                    unresolved_dependency_ids.sort();
                    if unresolved_dependency_ids.is_empty() {
                        let mut waived_dependency_ids: Vec<_> = task
                            .spec
                            .depends_on
                            .iter()
                            .filter(|dependency| self.skipped.contains(*dependency))
                            .cloned()
                            .collect();
                        waived_dependency_ids.sort();
                        DagDependencyState::Satisfied {
                            waived_dependency_ids,
                        }
                    } else {
                        DagDependencyState::Waiting {
                            unresolved_dependency_ids,
                        }
                    }
                };
                (task.spec.id.clone(), state)
            })
            .collect()
    }

    /// 是否所有任务都已完成或被有意跳过。
    pub fn all_completed(&self, tasks: &[Task]) -> bool {
        self.completed.len().saturating_add(self.skipped.len()) == tasks.len()
    }

    /// 是否所有未完成的任务要么失败、要么被失败的依赖阻塞。
    pub fn all_unfinished_failed_or_blocked(&self, tasks: &[Task]) -> bool {
        let dependencies = self.dependency_states(tasks);
        tasks.iter().all(|task| {
            self.completed.contains(&task.spec.id)
                || self.skipped.contains(&task.spec.id)
                || self.cancelled.contains(&task.spec.id)
                || self.failed.contains(&task.spec.id)
                || matches!(
                    dependencies.get(&task.spec.id),
                    Some(DagDependencyState::BlockedByFailure { .. })
                )
                || matches!(task.execution.status, TaskStatus::Blocked(_))
        })
    }
}

/// 根据更新的快照刷新执行中任务的结果。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DagRefresh {
    /// 本次刷新中变为已完成的任务
    pub completed: Vec<TaskId>,
    /// 本次刷新中变为失败的任务
    pub failed: Vec<TaskId>,
    /// 本次刷新中以非成功终态结束（跳过或取消）的任务
    pub terminal_non_success: Vec<TaskId>,
}

#[cfg(test)]
mod tests {
    use super::{
        DagDependencyState, NestedDelegationPolicy, TaskClaim, TaskExecution, TaskExecutionSummary,
        TaskSpec, TaskStatus, TaskSubagentContext,
    };
    use crate::tasks::runtime::{DagExecutionState, Task};
    use serde::{Deserialize, Serialize};
    use tokio_util::sync::CancellationToken;

    #[test]
    fn terminal_statuses_are_classified() {
        assert!(TaskStatus::Completed.is_terminal());
        assert!(TaskStatus::Failed("boom".to_string()).is_terminal());
        assert!(!TaskStatus::Running.is_terminal());
        assert!(TaskStatus::Running.is_running());
    }

    #[test]
    fn delegation_policy_advances_depth() {
        let policy = NestedDelegationPolicy {
            can_spawn_subagents: true,
            delegate_depth: 1,
            max_delegate_depth: 2,
        };

        let child = policy.child_policy();
        assert!(child.is_some());
        let child = child.unwrap_or_default();
        assert_eq!(child.delegate_depth, 2);
        assert!(!child.can_delegate());
    }

    #[test]
    fn task_subagent_context_builds_child_delegation_context() {
        let context =
            TaskSubagentContext::new("run-1").with_delegation_policy(NestedDelegationPolicy {
                can_spawn_subagents: true,
                delegate_depth: 0,
                max_delegate_depth: 1,
            });

        let child = context.child_delegation_context();

        assert!(child.is_some());
        let child = child.unwrap_or_else(|| TaskSubagentContext::new("missing"));
        assert_eq!(child.run_id, "run-1");
        assert_eq!(child.delegation_policy.delegate_depth, 1);
        assert!(!child.delegation_policy.can_delegate());
    }

    #[test]
    fn task_claim_context_preserves_exact_lineage() -> Result<(), String> {
        let claim = TaskClaim::new(9, 4, "spec-hash".to_string());
        let context = TaskSubagentContext::from_claim(
            "run-team",
            "research-task",
            claim.clone(),
            CancellationToken::new(),
        )?;
        let runtime = context.runtime_context();
        let lineage = runtime.subagent_lineage.clone().unwrap_or_default();
        assert_eq!(lineage.task_id.as_deref(), Some("research-task"));
        assert_eq!(lineage.attempt, Some(4));
        assert_eq!(lineage.plan_revision, Some(9));
        assert_eq!(
            runtime.execution_id.as_deref(),
            Some(claim.execution_id("run-team", "research-task").as_str())
        );
        assert_eq!(context.execution_id(), runtime.execution_id);
        Ok(())
    }

    #[test]
    fn task_execution_summary_extension_round_trips_product_evidence() -> Result<(), String> {
        let summary = TaskExecutionSummary {
            run_id: "run-1".to_string(),
            task_id: "task-1".to_string(),
            subagent_name: "researcher".to_string(),
            completed_work: vec!["inspected runtime".to_string()],
            decisions: Vec::new(),
            failures: Vec::new(),
            verification: vec!["tests passed".to_string()],
            next_implications: Vec::new(),
            suggested_tasks: Vec::new(),
            extension: serde_json::json!({
                "touched_files": {
                    "read": ["src/lib.rs"],
                    "written": ["src/tasks.rs"]
                }
            }),
            created_at: chrono::Utc::now(),
        };

        let encoded = serde_json::to_value(&summary).map_err(|error| error.to_string())?;
        let decoded: TaskExecutionSummary =
            serde_json::from_value(encoded).map_err(|error| error.to_string())?;

        assert_eq!(decoded, summary);
        Ok(())
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    struct TypedTaskExtension {
        role: String,
        priority: u32,
    }

    #[test]
    fn task_spec_typed_extension_round_trips_without_manual_json() -> Result<(), String> {
        let extension = TypedTaskExtension {
            role: "explorer".to_string(),
            priority: 2,
        };
        let spec = TaskSpec {
            id: "task-1".to_string(),
            title: "Inspect".to_string(),
            description: "Inspect the repository".to_string(),
            depends_on: Vec::new(),
            max_retries: 1,
            extension: serde_json::Value::Null,
        }
        .with_extension(extension.clone())?;

        assert_eq!(spec.extension_as::<TypedTaskExtension>()?, extension);
        Ok(())
    }

    #[test]
    fn task_spec_typed_extension_error_identifies_task() {
        let spec = TaskSpec {
            id: "task-invalid".to_string(),
            title: String::new(),
            description: String::new(),
            depends_on: Vec::new(),
            max_retries: 0,
            extension: serde_json::json!({ "priority": "not-a-number" }),
        };
        let error = spec.extension_as::<TypedTaskExtension>().err();
        assert!(error.is_some_and(|message| message.contains("task-invalid")));
    }

    /// 构造测试用的运行时任务
    fn runtime_task(id: &str, status: TaskStatus, deps: &[&str]) -> Task {
        Task {
            spec: TaskSpec {
                id: id.to_string(),
                title: id.to_string(),
                description: format!("execute {id}"),
                depends_on: deps.iter().map(|dep| dep.to_string()).collect(),
                max_retries: 3,
                extension: serde_json::json!({ "role": "explorer" }),
            },
            execution: TaskExecution {
                task_id: id.to_string(),
                status,
                retry_count: 0,
                failure_fingerprint: None,
                claim: None,
            },
        }
    }

    #[test]
    fn task_spec_hash_changes_with_dispatch_contract() -> Result<(), String> {
        let original = runtime_task("task", TaskStatus::Pending, &[]).spec;
        let mut changed = original.clone();
        changed.description = "changed contract".to_string();

        assert_eq!(original.stable_hash()?, original.stable_hash()?);
        assert_ne!(original.stable_hash()?, changed.stable_hash()?);
        Ok(())
    }

    #[test]
    fn dag_state_initializes_from_snapshot() {
        let tasks = vec![
            runtime_task("done", TaskStatus::Completed, &[]),
            runtime_task("running", TaskStatus::Running, &[]),
            runtime_task("failed", TaskStatus::Failed("boom".to_string()), &[]),
        ];

        let state = DagExecutionState::from_tasks(&tasks);

        assert!(state.completed.contains("done"));
        assert!(state.in_flight.contains("running"));
        assert!(state.failed.contains("failed"));
    }

    #[test]
    fn dag_ready_frontier_requires_completed_dependencies() {
        let tasks = vec![
            runtime_task("a", TaskStatus::Completed, &[]),
            runtime_task("b", TaskStatus::Pending, &["a"]),
            runtime_task("c", TaskStatus::Pending, &["b"]),
        ];
        let state = DagExecutionState::from_tasks(&tasks);

        assert_eq!(state.ready_task_ids(&tasks), vec!["b".to_string()]);
        assert_eq!(
            state.dependency_states(&tasks).get("b"),
            Some(&DagDependencyState::Satisfied {
                waived_dependency_ids: Vec::new(),
            })
        );
    }

    #[test]
    fn skipped_dependency_is_resolved_for_ready_frontier() {
        let tasks = vec![
            runtime_task("a", TaskStatus::Skipped, &[]),
            runtime_task("b", TaskStatus::Pending, &["a"]),
        ];
        let state = DagExecutionState::from_tasks(&tasks);

        assert_eq!(state.ready_task_ids(&tasks), vec!["b".to_string()]);
        assert_eq!(
            state.dependency_states(&tasks).get("b"),
            Some(&DagDependencyState::Satisfied {
                waived_dependency_ids: vec!["a".to_string()],
            })
        );
    }

    #[test]
    fn dag_refresh_observes_external_in_flight_completion() {
        let original = vec![runtime_task("a", TaskStatus::Running, &[])];
        let updated = vec![runtime_task("a", TaskStatus::Completed, &[])];
        let mut state = DagExecutionState::from_tasks(&original);

        let refresh = state.refresh_in_flight(&updated);

        assert_eq!(refresh.completed, vec!["a".to_string()]);
        assert!(state.completed.contains("a"));
        assert!(!state.in_flight.contains("a"));
    }

    #[test]
    fn dag_detects_failed_downstream_blockers() {
        let tasks = vec![
            runtime_task("a", TaskStatus::Failed("boom".to_string()), &[]),
            runtime_task("b", TaskStatus::Pending, &["a"]),
        ];
        let state = DagExecutionState::from_tasks(&tasks);

        assert_eq!(
            state.dependency_states(&tasks).get("b"),
            Some(&DagDependencyState::BlockedByFailure {
                failed_ancestor_ids: vec!["a".to_string()],
            })
        );
        assert!(state.all_unfinished_failed_or_blocked(&tasks));
    }

    #[test]
    fn dag_failure_blocking_is_transitive() -> Result<(), String> {
        let mut tasks = vec![
            runtime_task("a", TaskStatus::Failed("boom".to_string()), &[]),
            runtime_task("b", TaskStatus::Pending, &["a"]),
            runtime_task("c", TaskStatus::Pending, &["b"]),
            runtime_task("d", TaskStatus::Pending, &["c"]),
        ];
        let state = DagExecutionState::from_tasks(&tasks);

        let dependencies = state.dependency_states(&tasks);
        for task_id in ["b", "c", "d"] {
            assert_eq!(
                dependencies.get(task_id),
                Some(&DagDependencyState::BlockedByFailure {
                    failed_ancestor_ids: vec!["a".to_string()],
                })
            );
        }
        assert!(state.all_unfinished_failed_or_blocked(&tasks));

        let upstream = tasks
            .iter_mut()
            .find(|task| task.spec.id == "a")
            .ok_or_else(|| "upstream fixture is missing".to_string())?;
        upstream.execution.status = TaskStatus::Pending;
        let retried_state = DagExecutionState::from_tasks(&tasks);
        assert!(
            retried_state
                .dependency_states(&tasks)
                .values()
                .all(|state| !matches!(state, DagDependencyState::BlockedByFailure { .. }))
        );
        Ok(())
    }
}
