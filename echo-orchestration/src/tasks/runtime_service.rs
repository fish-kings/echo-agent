/*!
统一的任务运行时服务与认领（claim）状态迁移。

[`RuntimeTaskService`] 是包裹规范私有 DAG 执行器的框架入口。持久化适配器
保留各自的原子事务与事件格式，但在持有该事务时调用本模块中的纯变更函数。
这样可以保证内存存储与产品文件存储之间的 claim/CAS/重试语义完全一致。
*/

use std::collections::HashSet;
use std::sync::Arc;

use echo_core::error::Result;
use tokio_util::sync::CancellationToken;

use super::runtime_executor::{
    AttemptSettlementObservation, RuntimeAttemptControlCleanupReceipt, RuntimeAttemptObservation,
    RuntimeDagExecutor,
};
use super::{
    RuntimeDagController, RuntimeDagOutcome, RuntimeInterruptionDisposition, RuntimePlanSnapshot,
    RuntimeRetryExhaustion, RuntimeTaskClaimOutcome, RuntimeTaskServiceConfig, Task, TaskClaim,
    TaskId, TaskStatus,
};
use super::{RuntimeTaskResolution, RuntimeTaskResolutionRequest};
use crate::planning::PlanValidator;

/// 对框架所有的运行时任务状态的变更被拒绝时返回的错误。
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RuntimeTaskMutationError {
    /// 任务规格无效，无法被认领。
    #[error("task specification cannot be claimed: {message}")]
    InvalidTaskSpec {
        /// 错误说明。
        message: String,
    },
    /// 非法的运行时任务状态迁移。
    #[error("invalid runtime task transition: {message}")]
    InvalidTransition {
        /// 错误说明。
        message: String,
    },
    /// 认领不能被结算为活跃状态。
    #[error("claim cannot be settled to active state {status:?}")]
    SettlementRequiresInactiveStatus {
        /// 请求结算到的状态。
        status: TaskStatus,
    },
    /// 任务仍持有运行时认领时不能被显式重试。
    #[error("task '{task_id}' cannot be explicitly retried while it owns a runtime claim")]
    ExplicitRetryRequiresUnclaimedTask {
        /// 任务 ID。
        task_id: String,
    },
    /// 任务当前状态不可显式重试。
    #[error("task '{task_id}' in state {status:?} cannot be explicitly retried")]
    ExplicitRetryRequiresRestartableStatus {
        /// 任务 ID。
        task_id: String,
        /// 任务当前状态。
        status: TaskStatus,
    },
    /// 只有 Paused 状态的任务才能恢复。
    #[error("task '{task_id}' in state {status:?} cannot be resumed; expected Paused")]
    ResumeRequiresPausedStatus {
        /// 任务 ID。
        task_id: String,
        /// 任务当前状态。
        status: TaskStatus,
    },
    /// 重试计数器溢出。
    #[error("task '{task_id}' retry counter overflowed")]
    RetryCounterOverflow {
        /// 任务 ID。
        task_id: String,
    },
    /// 尝试计数器溢出。
    #[error("task '{task_id}' attempt counter overflowed")]
    AttemptCounterOverflow {
        /// 任务 ID。
        task_id: String,
    },
    /// 已消耗的重试次数超过重试预算，违反不变量。
    #[error("task '{task_id}' retry_count {retry_count} exceeds max_retries {max_retries}")]
    RetryBudgetInvariant {
        /// 任务 ID。
        task_id: String,
        /// 已消耗的重试次数。
        retry_count: u32,
        /// 最大重试次数。
        max_retries: u32,
    },
    /// 任务持有的运行时认领无效。
    #[error("task '{task_id}' has an invalid runtime claim: {message}")]
    InvalidClaim {
        /// 任务 ID。
        task_id: String,
        /// 错误说明。
        message: String,
    },
}

/// 为某个精确的已认领尝试请求取消后立即得到的结果回执。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeTaskAttemptInterruptReceipt {
    /// 运行 ID。
    pub run_id: String,
    /// 任务 ID。
    pub task_id: TaskId,
    /// 该尝试的执行 ID。
    pub execution_id: String,
    /// 认领所属的图修订号。
    pub revision: u64,
    /// 尝试序号。
    pub attempt: u32,
    /// 本次请求是否真正发起了取消。
    pub requested: bool,
    /// 中断请求在进程内的处置结果。
    pub disposition: RuntimeTaskAttemptInterruptDisposition,
}

/// 将一条持久化的精确中断投影到进程内后得到的类型化结果。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeTaskAttemptInterruptDisposition {
    /// 尝试尚未预留，中断已排队等待预留时生效。
    QueuedBeforeReservation,
    /// 尝试已预留，已请求取消。
    ReservedRequested,
    /// 尝试正在运行，已请求取消。
    ActiveRequested,
    /// 尝试正在运行，且此前已请求过取消。
    ActiveAlreadyRequested,
    /// 尝试已结算，无需再取消。
    AlreadySettled {
        /// 结算后的状态描述。
        status: String,
    },
}

impl RuntimeTaskAttemptInterruptDisposition {
    /// 该处置是否会开始一次新的取消。
    fn starts_cancellation(&self) -> bool {
        matches!(
            self,
            Self::QueuedBeforeReservation | Self::ReservedRequested | Self::ActiveRequested
        )
    }
}

/// 请求中断某个精确运行时尝试失败时返回的错误。
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RuntimeTaskAttemptInterruptError {
    /// 认领已过期或已结算。
    #[error("runtime attempt claim is stale or settled: {run_id}/{task_id}/{execution_id}")]
    StaleOrSettledClaim {
        /// 运行 ID。
        run_id: String,
        /// 任务 ID。
        task_id: String,
        /// 执行 ID。
        execution_id: String,
    },
    /// 待处理中断的容量已满。
    #[error("runtime interrupt pending capacity exceeded (limit {limit})")]
    PendingCapacityExceeded {
        /// 容量上限。
        limit: usize,
    },
    /// 同一尝试的执行身份冲突。
    #[error(
        "runtime task '{task_id}' attempt {attempt} execution identity conflict: expected {expected_execution_id}, actual {actual_execution_id}"
    )]
    IdentityConflict {
        /// 任务 ID。
        task_id: String,
        /// 尝试序号。
        attempt: u32,
        /// 期望的执行 ID。
        expected_execution_id: String,
        /// 实际的执行 ID。
        actual_execution_id: String,
    },
    /// 持久化权威源不可用。
    #[error("runtime attempt durable authority unavailable: {message}")]
    DurableAuthority {
        /// 错误说明。
        message: String,
    },
    /// 进程内实时控制不可用。
    #[error("runtime attempt live control unavailable: {message}")]
    LiveControl {
        /// 错误说明。
        message: String,
    },
}

/// 控制器将一条持久化的精确命令投影到其进程内实时控制注册表时产生的拒绝错误。
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RuntimeTaskAttemptInterruptProjectionError {
    /// 待处理中断的容量已满。
    #[error("runtime interrupt pending capacity exceeded (limit {limit})")]
    PendingCapacityExceeded {
        /// 容量上限。
        limit: usize,
    },
    /// 同一尝试的执行身份冲突。
    #[error(
        "runtime task '{task_id}' attempt {attempt} execution identity conflict: expected {expected_execution_id}, actual {actual_execution_id}"
    )]
    IdentityConflict {
        /// 任务 ID。
        task_id: String,
        /// 尝试序号。
        attempt: u32,
        /// 期望的执行 ID。
        expected_execution_id: String,
        /// 实际的执行 ID。
        actual_execution_id: String,
    },
    /// 实时控制不可用。
    #[error("runtime attempt live control unavailable: {message}")]
    Unavailable {
        /// 错误说明。
        message: String,
    },
}

impl From<RuntimeTaskAttemptInterruptProjectionError> for RuntimeTaskAttemptInterruptError {
    /// 将投影错误转换为中断错误。
    fn from(error: RuntimeTaskAttemptInterruptProjectionError) -> Self {
        match error {
            RuntimeTaskAttemptInterruptProjectionError::PendingCapacityExceeded { limit } => {
                Self::PendingCapacityExceeded { limit }
            }
            RuntimeTaskAttemptInterruptProjectionError::IdentityConflict {
                task_id,
                attempt,
                expected_execution_id,
                actual_execution_id,
            } => Self::IdentityConflict {
                task_id,
                attempt,
                expected_execution_id,
                actual_execution_id,
            },
            RuntimeTaskAttemptInterruptProjectionError::Unavailable { message } => {
                Self::LiveControl { message }
            }
        }
    }
}

impl From<RuntimeTaskAttemptInterruptError> for echo_core::error::ReactError {
    /// 将中断错误包装为通用的 `ReactError::Other`。
    fn from(error: RuntimeTaskAttemptInterruptError) -> Self {
        Self::Other(error.to_string())
    }
}

/// 针对某个精确物理认领结算的类型化 CAS（比较并交换）结果。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeTaskSettlementOutcome {
    /// 精确认领仍为当前认领，其状态迁移已提交。
    Settled,
    /// 认领身份、尝试、规格或生命周期已被替换。
    Superseded,
}

/// 原子地处理一次重试请求的结果。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeTaskRequeueOutcome {
    /// 精确认领已回到 Pending，并消耗了一次重试。
    Requeued,
    /// 重试预算已耗尽；精确认领进入了请求的类型化失败状态。
    Exhausted,
    /// 物理认领已不是当前认领，状态未发生变化。
    Superseded,
}

/**
显式重启一个未被认领任务的结果。

它与 [`RuntimeTaskRequeueOutcome`] 不同：requeue 处理的是一个活跃的已认领尝试，
而显式重试是在运维人员或产品策略要求再试一次后，重启一个已持久化的失败、超时
或被显式阻塞的任务。暂停的工作使用 [`RuntimeTaskResumeOutcome`]。
*/
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeTaskRetryOutcome {
    /// 任务已回到 Pending，并消耗了一次重试。
    Retried {
        /// 重试后的已消耗重试次数。
        retry_count: u32,
    },
    /// 已无剩余重试预算；保留现有状态。
    Exhausted {
        /// 已消耗的重试次数。
        retry_count: u32,
        /// 最大重试次数。
        max_retries: u32,
    },
    /// 修订号或期望任务已变化，状态未发生变化。
    Superseded,
}

/// 在不消耗重试预算的情况下恢复一个精确暂停任务的结果。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeTaskResumeOutcome {
    /// 精确暂停任务已回到 Pending，且未增加重试计数。
    Resumed,
    /// 图修订号或期望任务已变化，状态未发生变化。
    Superseded,
}

/// 在安全点结算运行时中断后产生的持久化任务效果。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeInterruptionReceipt {
    /// 实际应用的取消或暂停策略。
    pub disposition: RuntimeInterruptionDisposition,
    /// 在结算期间生命周期发生变化的未完成任务。
    pub interrupted_task_ids: Vec<TaskId>,
    /// 被保留、既不重放也不覆盖的终态兄弟任务。
    pub retained_terminal_task_ids: Vec<TaskId>,
    /// 因可恢复暂停而保留的未启动任务。
    pub pending_task_ids: Vec<TaskId>,
}

/// 运行级中断结算的乐观并发结果。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeInterruptionSettlementOutcome {
    /// 精确修订已结算，并生成了类型化回执。
    Settled(RuntimeInterruptionReceipt),
    /// 结算前图修订号已变化；需重新加载后重试。
    ReloadSnapshot,
}

/**
带修订号的运行时任务图的公开框架入口。

该服务是规范 DAG 执行器的唯一构造边界。框架负责的机制包括：依赖投影、精确认领身份、
有界波次、重试记账、中断结算以及修订安全点。应用方注入持久化与分发实现，以及将停止
请求映射为 [`RuntimeInterruptionDisposition`] 的产品决策。UI 投影、审查策略、worktree
和产品运行状态均不在本服务范围内。
*/
pub struct RuntimeTaskService<C: RuntimeDagController> {
    /// 底层的规范 DAG 执行器。
    executor: RuntimeDagExecutor<C>,
}

impl<C: RuntimeDagController> RuntimeTaskService<C> {
    /// 使用给定控制器与配置创建服务。
    pub fn new(controller: Arc<C>, config: RuntimeTaskServiceConfig) -> Self {
        Self {
            executor: RuntimeDagExecutor::new(controller, config),
        }
    }

    /// 为执行器设置计划校验器。
    pub fn with_validator(mut self, validator: PlanValidator) -> Self {
        self.executor = self.executor.with_validator(validator);
        self
    }

    /// 先协调尝试控制状态，再执行指定运行的任务图。
    pub async fn execute(
        &self,
        run_id: &str,
        cancel: CancellationToken,
    ) -> Result<RuntimeDagOutcome> {
        self.reconcile_attempt_control(run_id).await?;
        self.executor.execute(run_id, cancel).await
    }

    /// 依据持久化的 TaskClaim 快照协调实时尝试投影；可在进程恢复时及命令重放前安全地重复执行。
    pub async fn reconcile_attempt_control(&self, run_id: &str) -> Result<()> {
        let snapshot = self.executor.controller.load_snapshot(run_id).await?;
        validate_runtime_snapshot_claims(&snapshot).map_err(|error| {
            echo_core::error::ReactError::Other(format!(
                "invalid runtime claim snapshot during control reconciliation: {error}"
            ))
        })?;
        let current_execution_ids = snapshot
            .tasks
            .iter()
            .filter_map(|task| {
                task.execution
                    .claim
                    .as_ref()
                    .map(|claim| claim.execution_id(run_id, &task.spec.id))
            })
            .collect::<HashSet<_>>();
        self.executor
            .reconcile_local_attempt_control(run_id, &current_execution_ids);
        self.executor
            .controller
            .reconcile_attempt_control(run_id, &current_execution_ids)
            .await
    }

    /**
    为某个精确的当前认领请求取消，不影响兄弟任务的子项。

    持久化认领检查是权威来源；进程内的令牌表只是实时取消的投影。
    */
    pub async fn request_attempt_interrupt(
        &self,
        run_id: &str,
        task_id: &str,
        claim: &TaskClaim,
    ) -> std::result::Result<RuntimeTaskAttemptInterruptReceipt, RuntimeTaskAttemptInterruptError>
    {
        let execution_id = claim.execution_id(run_id, task_id);
        let current = self
            .executor
            .controller
            .claim_is_current(run_id, task_id, claim)
            .await
            .map_err(|error| RuntimeTaskAttemptInterruptError::DurableAuthority {
                message: error.to_string(),
            })?;
        if !current {
            if let Err(error) = self.reconcile_attempt_control(run_id).await {
                self.observe_reconciliation_failure(run_id, task_id, claim, &error);
                tracing::warn!(
                    run_id,
                    task_id,
                    error = %error,
                    "stale runtime attempt control reconciliation requires retry"
                );
            }
            return Err(RuntimeTaskAttemptInterruptError::StaleOrSettledClaim {
                run_id: run_id.to_string(),
                task_id: task_id.to_string(),
                execution_id,
            });
        }
        let registry_disposition = self
            .executor
            .controller
            .request_live_interrupt(run_id, task_id, claim)
            .await
            .map_err(RuntimeTaskAttemptInterruptError::from)?;
        let observation = self.executor.observe_attempt(&execution_id);
        let (active_cancel, joined_settlement) = match observation {
            Some(RuntimeAttemptObservation::Active { cancel, settled_rx }) => {
                let controller_already_settled = matches!(
                    &registry_disposition,
                    Some(RuntimeTaskAttemptInterruptDisposition::AlreadySettled { .. })
                );
                if controller_already_settled {
                    (None, Some(settled_rx))
                } else {
                    (Some(cancel), None)
                }
            }
            Some(RuntimeAttemptObservation::JoinedAwaitingDurableSettlement { settled_rx }) => {
                (None, Some(settled_rx))
            }
            None => (None, None),
        };
        if let Some(settled_rx) = joined_settlement {
            if !self
                .executor
                .controller
                .claim_is_current(run_id, task_id, claim)
                .await
                .map_err(|error| RuntimeTaskAttemptInterruptError::DurableAuthority {
                    message: error.to_string(),
                })?
            {
                if let Err(error) = self.reconcile_attempt_control(run_id).await {
                    self.observe_reconciliation_failure(run_id, task_id, claim, &error);
                }
                return Err(RuntimeTaskAttemptInterruptError::StaleOrSettledClaim {
                    run_id: run_id.to_string(),
                    task_id: task_id.to_string(),
                    execution_id,
                });
            }
            return self
                .wait_for_attempt_settlement(run_id, task_id, claim, settled_rx)
                .await;
        }
        let disposition = match registry_disposition {
            Some(disposition) => disposition,
            None => match active_cancel.or_else(|| {
                self.executor
                    .attempt_cancellations
                    .lock()
                    .ok()
                    .and_then(|controls| controls.get(&execution_id).cloned())
            }) {
                Some(cancel) if cancel.is_cancelled() => {
                    RuntimeTaskAttemptInterruptDisposition::ActiveAlreadyRequested
                }
                Some(cancel) => {
                    cancel.cancel();
                    RuntimeTaskAttemptInterruptDisposition::ActiveRequested
                }
                None => {
                    let mut pending =
                        self.executor
                            .pending_attempt_interrupts
                            .lock()
                            .map_err(|_| RuntimeTaskAttemptInterruptError::LiveControl {
                                message: "runtime interrupt pending state is unavailable"
                                    .to_string(),
                            })?;
                    if pending.len() >= 256 && !pending.contains_key(&execution_id) {
                        return Err(RuntimeTaskAttemptInterruptError::PendingCapacityExceeded {
                            limit: 256,
                        });
                    }
                    pending.insert(execution_id.clone(), run_id.to_string());
                    RuntimeTaskAttemptInterruptDisposition::QueuedBeforeReservation
                }
            },
        };
        let requested = disposition.starts_cancellation();
        /*
        在第一次前置检查与写入实时令牌之间，认领可能已被替换。
        这里重新读取持久化权威源，只撤销刚写入的那条精确投影；
        绝不能让过期命令认领未来的尝试，也不能把竞态变成第二次任务迁移。
        */
        if !self
            .executor
            .controller
            .claim_is_current(run_id, task_id, claim)
            .await
            .map_err(|error| RuntimeTaskAttemptInterruptError::DurableAuthority {
                message: error.to_string(),
            })?
        {
            if let Err(error) = self.reconcile_attempt_control(run_id).await {
                self.observe_reconciliation_failure(run_id, task_id, claim, &error);
                tracing::warn!(
                    run_id,
                    task_id,
                    error = %error,
                    "raced stale runtime attempt control reconciliation requires retry"
                );
            }
            return Err(RuntimeTaskAttemptInterruptError::StaleOrSettledClaim {
                run_id: run_id.to_string(),
                task_id: task_id.to_string(),
                execution_id,
            });
        }
        if requested
            && disposition != RuntimeTaskAttemptInterruptDisposition::QueuedBeforeReservation
        {
            self.executor
                .schedule_attempt_abort(execution_id.clone(), false);
        }
        Ok(RuntimeTaskAttemptInterruptReceipt {
            run_id: run_id.to_string(),
            task_id: task_id.to_string(),
            execution_id,
            revision: claim.revision,
            attempt: claim.attempt,
            requested,
            disposition,
        })
    }

    /// 测试用：返回尚未生效的待处理尝试中断数量。
    #[cfg(test)]
    pub(crate) fn pending_attempt_interrupt_count(&self) -> usize {
        self.executor
            .pending_attempt_interrupts
            .lock()
            .map(|pending| pending.len())
            .unwrap_or_default()
    }

    /// 测试用：读取指定执行 ID 当前的结算观测值。
    #[cfg(test)]
    pub(crate) fn attempt_settlement_observation(
        &self,
        execution_id: &str,
    ) -> Option<AttemptSettlementObservation> {
        let settled_rx = match self.executor.observe_attempt(execution_id)? {
            RuntimeAttemptObservation::Active { settled_rx, .. }
            | RuntimeAttemptObservation::JoinedAwaitingDurableSettlement { settled_rx } => {
                settled_rx
            }
        };
        settled_rx.borrow().clone()
    }

    /// 测试用：获取指定执行 ID 的结算观测接收端。
    #[cfg(test)]
    pub(crate) fn attempt_settlement_receiver(
        &self,
        execution_id: &str,
    ) -> Option<tokio::sync::watch::Receiver<Option<AttemptSettlementObservation>>> {
        match self.executor.observe_attempt(execution_id)? {
            RuntimeAttemptObservation::Active { settled_rx, .. }
            | RuntimeAttemptObservation::JoinedAwaitingDurableSettlement { settled_rx } => {
                Some(settled_rx)
            }
        }
    }

    /// 测试用：判断指定执行 ID 是否仍有监督状态。
    #[cfg(test)]
    pub(crate) fn has_attempt_supervisor_state(&self, execution_id: &str) -> bool {
        self.executor.has_attempt_supervisor_state(execution_id)
    }

    /// 测试用：仅协调本地进程内的尝试控制状态。
    #[cfg(test)]
    pub(crate) fn reconcile_local_attempt_control(
        &self,
        run_id: &str,
        current_execution_ids: &HashSet<String>,
    ) {
        self.executor
            .reconcile_local_attempt_control(run_id, current_execution_ids);
    }

    /// 记录一次可重试的控制协调失败。
    fn observe_reconciliation_failure(
        &self,
        run_id: &str,
        task_id: &str,
        claim: &TaskClaim,
        error: &echo_core::error::ReactError,
    ) {
        self.executor.observe_attempt_control_cleanup(
            run_id,
            task_id,
            claim,
            RuntimeAttemptControlCleanupReceipt::RetryableFailure {
                error: error.to_string(),
            },
        );
    }

    /// 等待已加入的尝试完成持久化结算，并生成 `AlreadySettled` 回执。
    async fn wait_for_attempt_settlement(
        &self,
        run_id: &str,
        task_id: &str,
        claim: &TaskClaim,
        mut settled_rx: tokio::sync::watch::Receiver<Option<AttemptSettlementObservation>>,
    ) -> std::result::Result<RuntimeTaskAttemptInterruptReceipt, RuntimeTaskAttemptInterruptError>
    {
        let observation = loop {
            if let Some(observation) = settled_rx.borrow_and_update().clone() {
                break observation;
            }
            settled_rx.changed().await.map_err(|_| {
                RuntimeTaskAttemptInterruptError::LiveControl {
                    message: "joined attempt settlement observation closed".to_string(),
                }
            })?;
        };
        let status = match observation {
            AttemptSettlementObservation::Settled { status } => status,
            AttemptSettlementObservation::AuthorityUnknown { error } => {
                return Err(RuntimeTaskAttemptInterruptError::DurableAuthority { message: error });
            }
        };
        Ok(RuntimeTaskAttemptInterruptReceipt {
            run_id: run_id.to_string(),
            task_id: task_id.to_string(),
            execution_id: claim.execution_id(run_id, task_id),
            revision: claim.revision,
            attempt: claim.attempt,
            requested: false,
            disposition: RuntimeTaskAttemptInterruptDisposition::AlreadySettled { status },
        })
    }
}

/// 从已加载的关系修订中认领一个精确的 Pending 任务。
pub fn claim_runtime_task(
    snapshot: &mut RuntimePlanSnapshot,
    expected_task: &Task,
    expected_revision: u64,
) -> std::result::Result<RuntimeTaskClaimOutcome, RuntimeTaskMutationError> {
    if snapshot.revision != expected_revision {
        return Ok(RuntimeTaskClaimOutcome::ReloadSnapshot);
    }
    let Some(task) = snapshot
        .tasks
        .iter_mut()
        .find(|task| task.spec.id == expected_task.spec.id)
    else {
        return Ok(RuntimeTaskClaimOutcome::ReloadSnapshot);
    };
    if task.execution.retry_count > task.spec.max_retries {
        return Err(RuntimeTaskMutationError::RetryBudgetInvariant {
            task_id: task.spec.id.clone(),
            retry_count: task.execution.retry_count,
            max_retries: task.spec.max_retries,
        });
    }
    if task != expected_task
        || task.execution.task_id != task.spec.id
        || task.execution.status != TaskStatus::Pending
        || task.execution.claim.is_some()
    {
        return Ok(RuntimeTaskClaimOutcome::ReloadSnapshot);
    }
    let attempt = task.execution.retry_count.checked_add(1).ok_or_else(|| {
        RuntimeTaskMutationError::AttemptCounterOverflow {
            task_id: task.spec.id.clone(),
        }
    })?;
    let claim = TaskClaim::new(
        expected_revision,
        attempt,
        task.spec
            .stable_hash()
            .map_err(|message| RuntimeTaskMutationError::InvalidTaskSpec { message })?,
    );
    task.execution.status = TaskStatus::Running;
    task.execution.claim = Some(claim.clone());
    Ok(RuntimeTaskClaimOutcome::Claimed(claim))
}

/// 结算一个精确的物理认领并清除认领所有权。
pub fn settle_runtime_claim(
    snapshot: &mut RuntimePlanSnapshot,
    task_id: &str,
    claim: &TaskClaim,
    status: TaskStatus,
) -> std::result::Result<RuntimeTaskSettlementOutcome, RuntimeTaskMutationError> {
    if matches!(
        status,
        TaskStatus::Pending | TaskStatus::Running | TaskStatus::Retrying { .. }
    ) {
        return Err(RuntimeTaskMutationError::SettlementRequiresInactiveStatus { status });
    }
    let Some(task) = claimed_task_mut(snapshot, task_id, claim)? else {
        return Ok(RuntimeTaskSettlementOutcome::Superseded);
    };
    task.execution.status = task
        .execution
        .status
        .transition_to(status)
        .map_err(|message| RuntimeTaskMutationError::InvalidTransition { message })?;
    task.execution.claim = None;
    Ok(RuntimeTaskSettlementOutcome::Settled)
}

/**
将一个精确的失败尝试重新入队，且不暴露未认领的中间状态。

`retry_count` 统计已完成的重试决策次数，而认领的尝试身份始终为 `retry_count + 1`。
*/
pub fn requeue_runtime_claim(
    snapshot: &mut RuntimePlanSnapshot,
    task_id: &str,
    claim: &TaskClaim,
    failure_fingerprint: Option<String>,
    error: String,
    exhaustion: RuntimeRetryExhaustion,
) -> std::result::Result<RuntimeTaskRequeueOutcome, RuntimeTaskMutationError> {
    let Some(task) = claimed_task_mut(snapshot, task_id, claim)? else {
        return Ok(RuntimeTaskRequeueOutcome::Superseded);
    };
    if task.execution.retry_count >= task.spec.max_retries {
        task.execution.status = task
            .execution
            .status
            .transition_to(retry_exhaustion_status(exhaustion, error))
            .map_err(|message| RuntimeTaskMutationError::InvalidTransition { message })?;
        task.execution.failure_fingerprint = failure_fingerprint;
        task.execution.claim = None;
        return Ok(RuntimeTaskRequeueOutcome::Exhausted);
    }
    let retrying = task
        .execution
        .status
        .transition_to(TaskStatus::Retrying {
            attempt: claim.attempt,
            last_error: error,
        })
        .map_err(|message| RuntimeTaskMutationError::InvalidTransition { message })?;
    task.execution.status = retrying
        .transition_to(TaskStatus::Pending)
        .map_err(|message| RuntimeTaskMutationError::InvalidTransition { message })?;
    task.execution.retry_count = task.execution.retry_count.checked_add(1).ok_or_else(|| {
        RuntimeTaskMutationError::RetryCounterOverflow {
            task_id: task.spec.id.clone(),
        }
    })?;
    task.execution.failure_fingerprint = failure_fingerprint;
    task.execution.claim = None;
    Ok(RuntimeTaskRequeueOutcome::Requeued)
}

/**
将一次未提交的分发评估应用到精确的当前认领上。

持久化适配器应在持有原子事务时调用此函数，并且仅当类型化结果不是
[`RuntimeTaskResolution::Superseded`] 时才发布产品侧的副作用。
*/
pub fn settle_runtime_resolution(
    snapshot: &mut RuntimePlanSnapshot,
    task_id: &str,
    claim: &TaskClaim,
    request: RuntimeTaskResolutionRequest,
) -> std::result::Result<RuntimeTaskResolution, RuntimeTaskMutationError> {
    match request {
        RuntimeTaskResolutionRequest::Completed => settle_requested_status(
            snapshot,
            task_id,
            claim,
            TaskStatus::Completed,
            RuntimeTaskResolution::Completed,
        ),
        RuntimeTaskResolutionRequest::Requeue {
            failure_fingerprint,
            error,
            exhaustion,
        } => match requeue_runtime_claim(
            snapshot,
            task_id,
            claim,
            failure_fingerprint,
            error.clone(),
            exhaustion,
        )? {
            RuntimeTaskRequeueOutcome::Requeued => Ok(RuntimeTaskResolution::Pending),
            RuntimeTaskRequeueOutcome::Exhausted => {
                Ok(retry_exhaustion_resolution(exhaustion, error))
            }
            RuntimeTaskRequeueOutcome::Superseded => Ok(RuntimeTaskResolution::Superseded),
        },
        RuntimeTaskResolutionRequest::Skipped => settle_requested_status(
            snapshot,
            task_id,
            claim,
            TaskStatus::Skipped,
            RuntimeTaskResolution::Skipped,
        ),
        RuntimeTaskResolutionRequest::Failed { error } => settle_requested_status(
            snapshot,
            task_id,
            claim,
            TaskStatus::Failed(error.clone()),
            RuntimeTaskResolution::Failed { error },
        ),
        RuntimeTaskResolutionRequest::TimedOut { error } => settle_requested_status(
            snapshot,
            task_id,
            claim,
            TaskStatus::TimedOut {
                error: error.clone(),
            },
            RuntimeTaskResolution::TimedOut { error },
        ),
        RuntimeTaskResolutionRequest::Blocked { error, disposition } => settle_requested_status(
            snapshot,
            task_id,
            claim,
            TaskStatus::Blocked(error.clone()),
            RuntimeTaskResolution::Blocked { error, disposition },
        ),
        RuntimeTaskResolutionRequest::Cancelled => settle_requested_status(
            snapshot,
            task_id,
            claim,
            TaskStatus::Cancelled,
            RuntimeTaskResolution::Cancelled,
        ),
    }
}

/// 将重试耗尽策略映射为对应的任务终态。
fn retry_exhaustion_status(exhaustion: RuntimeRetryExhaustion, error: String) -> TaskStatus {
    match exhaustion {
        RuntimeRetryExhaustion::Failed => TaskStatus::Failed(error),
        RuntimeRetryExhaustion::TimedOut => TaskStatus::TimedOut { error },
    }
}

/// 将重试耗尽策略映射为对应的任务处理结果。
fn retry_exhaustion_resolution(
    exhaustion: RuntimeRetryExhaustion,
    error: String,
) -> RuntimeTaskResolution {
    match exhaustion {
        RuntimeRetryExhaustion::Failed => RuntimeTaskResolution::Failed { error },
        RuntimeRetryExhaustion::TimedOut => RuntimeTaskResolution::TimedOut { error },
    }
}

/// 按请求的状态结算认领，并在认领已被替换时返回 `Superseded`。
fn settle_requested_status(
    snapshot: &mut RuntimePlanSnapshot,
    task_id: &str,
    claim: &TaskClaim,
    status: TaskStatus,
    resolution: RuntimeTaskResolution,
) -> std::result::Result<RuntimeTaskResolution, RuntimeTaskMutationError> {
    match settle_runtime_claim(snapshot, task_id, claim, status)? {
        RuntimeTaskSettlementOutcome::Settled => Ok(resolution),
        RuntimeTaskSettlementOutcome::Superseded => Ok(RuntimeTaskResolution::Superseded),
    }
}

/**
从已加载的关系修订中显式重试一个精确的未认领任务。

整个期望任务（包括其失败指纹和重试计数）都参与乐观并发校验。运行重启、审查等产品
策略不属于本变更。读取新快照后，派生的依赖阻塞会从 [`super::DagExecutionState`] 中消失。
暂停的任务应使用 [`resume_runtime_task`]，不消耗重试预算。
*/
pub fn retry_runtime_task(
    snapshot: &mut RuntimePlanSnapshot,
    expected_task: &Task,
    expected_revision: u64,
) -> std::result::Result<RuntimeTaskRetryOutcome, RuntimeTaskMutationError> {
    if snapshot.revision != expected_revision {
        return Ok(RuntimeTaskRetryOutcome::Superseded);
    }
    let Some(task) = snapshot
        .tasks
        .iter_mut()
        .find(|task| task.spec.id == expected_task.spec.id)
    else {
        return Ok(RuntimeTaskRetryOutcome::Superseded);
    };
    if task != expected_task {
        return Ok(RuntimeTaskRetryOutcome::Superseded);
    }
    if task.execution.claim.is_some() {
        return Err(
            RuntimeTaskMutationError::ExplicitRetryRequiresUnclaimedTask {
                task_id: task.spec.id.clone(),
            },
        );
    }
    if !matches!(
        task.execution.status,
        TaskStatus::Failed(_) | TaskStatus::TimedOut { .. } | TaskStatus::Blocked(_)
    ) {
        return Err(
            RuntimeTaskMutationError::ExplicitRetryRequiresRestartableStatus {
                task_id: task.spec.id.clone(),
                status: task.execution.status.clone(),
            },
        );
    }
    if task.execution.retry_count >= task.spec.max_retries {
        return Ok(RuntimeTaskRetryOutcome::Exhausted {
            retry_count: task.execution.retry_count,
            max_retries: task.spec.max_retries,
        });
    }
    let retry_count = task.execution.retry_count.checked_add(1).ok_or_else(|| {
        RuntimeTaskMutationError::RetryCounterOverflow {
            task_id: task.spec.id.clone(),
        }
    })?;
    task.execution.status = TaskStatus::Pending;
    task.execution.retry_count = retry_count;
    Ok(RuntimeTaskRetryOutcome::Retried { retry_count })
}

/// 恢复一个精确的暂停任务，不增加其重试计数。
pub fn resume_runtime_task(
    snapshot: &mut RuntimePlanSnapshot,
    expected_task: &Task,
    expected_revision: u64,
) -> std::result::Result<RuntimeTaskResumeOutcome, RuntimeTaskMutationError> {
    if snapshot.revision != expected_revision {
        return Ok(RuntimeTaskResumeOutcome::Superseded);
    }
    let Some(task) = snapshot
        .tasks
        .iter_mut()
        .find(|task| task.spec.id == expected_task.spec.id)
    else {
        return Ok(RuntimeTaskResumeOutcome::Superseded);
    };
    if task != expected_task {
        return Ok(RuntimeTaskResumeOutcome::Superseded);
    }
    if task.execution.claim.is_some() {
        return Err(RuntimeTaskMutationError::InvalidClaim {
            task_id: task.spec.id.clone(),
            message: "paused task still owns a claim".to_string(),
        });
    }
    if !matches!(task.execution.status, TaskStatus::Paused(_)) {
        return Err(RuntimeTaskMutationError::ResumeRequiresPausedStatus {
            task_id: task.spec.id.clone(),
            status: task.execution.status.clone(),
        });
    }
    task.execution.status = task
        .execution
        .status
        .transition_to(TaskStatus::Pending)
        .map_err(|message| RuntimeTaskMutationError::InvalidTransition { message })?;
    Ok(RuntimeTaskResumeOutcome::Resumed)
}

/**
针对一个精确的图修订结算运行级的暂停或取消。

取消会把所有未完成任务标记为 Cancelled。暂停只把活跃的已认领任务转为 Paused，
未启动任务保持 Pending。所有认领在变更前都会先被校验，因此过期的修订号、尝试、
规格哈希或物理认领身份都会以失败关闭的方式拒绝，不会产生部分结算。
*/
pub fn settle_runtime_interruption(
    snapshot: &mut RuntimePlanSnapshot,
    expected_revision: u64,
    disposition: RuntimeInterruptionDisposition,
) -> std::result::Result<RuntimeInterruptionSettlementOutcome, RuntimeTaskMutationError> {
    if snapshot.revision != expected_revision {
        return Ok(RuntimeInterruptionSettlementOutcome::ReloadSnapshot);
    }
    validate_runtime_snapshot_claims(snapshot)?;

    let mut interrupted_task_ids = Vec::new();
    let mut retained_terminal_task_ids = Vec::new();
    let mut pending_task_ids = Vec::new();
    for task in &mut snapshot.tasks {
        if task.execution.status.is_terminal() {
            retained_terminal_task_ids.push(task.spec.id.clone());
            continue;
        }
        match &disposition {
            RuntimeInterruptionDisposition::Cancelled => {
                task.execution.status = task
                    .execution
                    .status
                    .transition_to(TaskStatus::Cancelled)
                    .map_err(|message| RuntimeTaskMutationError::InvalidTransition { message })?;
                task.execution.claim = None;
                interrupted_task_ids.push(task.spec.id.clone());
            }
            RuntimeInterruptionDisposition::Paused { reason } => {
                if matches!(
                    task.execution.status,
                    TaskStatus::Running | TaskStatus::Retrying { .. }
                ) {
                    task.execution.status = task
                        .execution
                        .status
                        .transition_to(TaskStatus::Paused(reason.clone()))
                        .map_err(|message| RuntimeTaskMutationError::InvalidTransition {
                            message,
                        })?;
                    task.execution.claim = None;
                    interrupted_task_ids.push(task.spec.id.clone());
                } else if task.execution.status == TaskStatus::Pending {
                    pending_task_ids.push(task.spec.id.clone());
                }
            }
        }
    }
    interrupted_task_ids.sort();
    retained_terminal_task_ids.sort();
    pending_task_ids.sort();
    Ok(RuntimeInterruptionSettlementOutcome::Settled(
        RuntimeInterruptionReceipt {
            disposition,
            interrupted_task_ids,
            retained_terminal_task_ids,
            pending_task_ids,
        },
    ))
}

/// 取消一个精确图修订中的所有未完成任务。
pub fn cancel_unfinished_runtime_tasks(
    snapshot: &mut RuntimePlanSnapshot,
    expected_revision: u64,
) -> std::result::Result<RuntimeInterruptionSettlementOutcome, RuntimeTaskMutationError> {
    settle_runtime_interruption(
        snapshot,
        expected_revision,
        RuntimeInterruptionDisposition::Cancelled,
    )
}

/// 判断给定认领是否仍是快照中该任务的当前认领。
pub fn runtime_claim_is_current(
    snapshot: &RuntimePlanSnapshot,
    task_id: &str,
    claim: &TaskClaim,
) -> std::result::Result<bool, RuntimeTaskMutationError> {
    let Some(task) = snapshot.tasks.iter().find(|task| task.spec.id == task_id) else {
        return Ok(false);
    };
    claim_matches_task(snapshot.revision, task, claim)
}

/// 仅当认领与任务匹配时，返回该任务的可变引用。
fn claimed_task_mut<'a>(
    snapshot: &'a mut RuntimePlanSnapshot,
    task_id: &str,
    claim: &TaskClaim,
) -> std::result::Result<Option<&'a mut Task>, RuntimeTaskMutationError> {
    let Some(position) = snapshot
        .tasks
        .iter()
        .position(|task| task.spec.id == task_id)
    else {
        return Ok(None);
    };
    let Some(task) = snapshot.tasks.get(position) else {
        return Ok(None);
    };
    if !claim_matches_task(snapshot.revision, task, claim)? {
        return Ok(None);
    }
    Ok(snapshot.tasks.get_mut(position))
}

/// 校验认领的修订号、状态、尝试序号与规格哈希是否与任务一致。
fn claim_matches_task(
    snapshot_revision: u64,
    task: &Task,
    claim: &TaskClaim,
) -> std::result::Result<bool, RuntimeTaskMutationError> {
    if claim.revision > snapshot_revision
        || !task.execution.status.is_running()
        || task.execution.claim.as_ref() != Some(claim)
    {
        return Ok(false);
    }
    let expected_attempt = task.execution.retry_count.checked_add(1).ok_or_else(|| {
        RuntimeTaskMutationError::AttemptCounterOverflow {
            task_id: task.spec.id.clone(),
        }
    })?;
    if claim.attempt != expected_attempt {
        return Ok(false);
    }
    let spec_hash = task
        .spec
        .stable_hash()
        .map_err(|message| RuntimeTaskMutationError::InvalidTaskSpec { message })?;
    Ok(claim.spec_hash == spec_hash)
}

/**
校验快照中每个任务的认领/状态/尝试/规格不变量。

执行器在每次加载并完成 DAG 结构校验后调用它，使格式错误的终态认领以失败关闭的方式
被拒绝，而不是误报成功。没有认领的 Running 任务仍被视为由外部管理的合法在途工作；
运行级中断仍可结算其生命周期。
*/
pub fn validate_runtime_snapshot_claims(
    snapshot: &RuntimePlanSnapshot,
) -> std::result::Result<(), RuntimeTaskMutationError> {
    for task in &snapshot.tasks {
        if let Some(claim) = task.execution.claim.as_ref()
            && !claim_matches_task(snapshot.revision, task, claim)?
        {
            return Err(RuntimeTaskMutationError::InvalidClaim {
                task_id: task.spec.id.clone(),
                message: "claim revision, attempt, specification, or status is stale".to_string(),
            });
        }
        if task.execution.claim.is_none()
            && matches!(task.execution.status, TaskStatus::Retrying { .. })
        {
            return Err(RuntimeTaskMutationError::InvalidClaim {
                task_id: task.spec.id.clone(),
                message: "Retrying task has no physical claim".to_string(),
            });
        }
    }
    Ok(())
}

/// 运行时任务服务与认领状态迁移的单元测试。
#[cfg(test)]
mod tests {
    use super::*;

    /// 投影错误转换后保留类型化的容量与身份信息。
    #[test]
    fn interrupt_projection_errors_preserve_typed_capacity_and_identity() {
        assert_eq!(
            RuntimeTaskAttemptInterruptError::from(
                RuntimeTaskAttemptInterruptProjectionError::PendingCapacityExceeded { limit: 9 }
            ),
            RuntimeTaskAttemptInterruptError::PendingCapacityExceeded { limit: 9 }
        );
        assert_eq!(
            RuntimeTaskAttemptInterruptError::from(
                RuntimeTaskAttemptInterruptProjectionError::IdentityConflict {
                    task_id: "task".to_string(),
                    attempt: 3,
                    expected_execution_id: "expected".to_string(),
                    actual_execution_id: "actual".to_string(),
                }
            ),
            RuntimeTaskAttemptInterruptError::IdentityConflict {
                task_id: "task".to_string(),
                attempt: 3,
                expected_execution_id: "expected".to_string(),
                actual_execution_id: "actual".to_string(),
            }
        );
    }
    use crate::tasks::{TaskExecution, TaskSpec};

    /// 构造一个用于测试的最小任务。
    fn task(id: &str) -> Task {
        Task {
            spec: TaskSpec {
                id: id.to_string(),
                title: id.to_string(),
                description: id.to_string(),
                depends_on: Vec::new(),
                max_retries: 2,
                extension: serde_json::Value::Null,
            },
            execution: TaskExecution::pending(id),
        }
    }

    /// 认领、结算与过期 CAS 共用同一迁移权威。
    #[test]
    fn claim_settle_and_stale_cas_share_one_transition_authority()
    -> std::result::Result<(), RuntimeTaskMutationError> {
        let expected = task("a");
        let mut snapshot = RuntimePlanSnapshot {
            revision: 3,
            tasks: vec![expected.clone()],
        };
        assert_eq!(
            claim_runtime_task(&mut snapshot, &expected, 2)?,
            RuntimeTaskClaimOutcome::ReloadSnapshot
        );
        let current = snapshot.tasks.first_mut().ok_or_else(|| {
            RuntimeTaskMutationError::InvalidTransition {
                message: "claim ABA fixture is missing".to_string(),
            }
        })?;
        current.execution.retry_count = 1;
        current.execution.failure_fingerprint = Some("newer-attempt".to_string());
        assert_eq!(
            claim_runtime_task(&mut snapshot, &expected, 3)?,
            RuntimeTaskClaimOutcome::ReloadSnapshot
        );
        snapshot.tasks = vec![expected.clone()];
        let claim = match claim_runtime_task(&mut snapshot, &expected, 3)? {
            RuntimeTaskClaimOutcome::Claimed(claim) => claim,
            RuntimeTaskClaimOutcome::ReloadSnapshot => {
                return Err(RuntimeTaskMutationError::InvalidTransition {
                    message: "claim unexpectedly reloaded".to_string(),
                });
            }
        };
        assert!(runtime_claim_is_current(&snapshot, "a", &claim)?);
        let stale = TaskClaim::new(3, claim.attempt, claim.spec_hash.clone());
        assert_eq!(
            settle_runtime_claim(&mut snapshot, "a", &stale, TaskStatus::Completed)?,
            RuntimeTaskSettlementOutcome::Superseded
        );
        assert_eq!(
            settle_runtime_claim(&mut snapshot, "a", &claim, TaskStatus::Completed)?,
            RuntimeTaskSettlementOutcome::Settled
        );
        let settled =
            snapshot
                .tasks
                .first()
                .ok_or_else(|| RuntimeTaskMutationError::InvalidTransition {
                    message: "settled task is missing".to_string(),
                })?;
        assert_eq!(settled.execution.status, TaskStatus::Completed);
        assert!(settled.execution.claim.is_none());
        Ok(())
    }

    /// 已消耗重试次数超过当前预算时拒绝认领。
    #[test]
    fn claim_rejects_consumed_retry_count_above_current_budget()
    -> std::result::Result<(), RuntimeTaskMutationError> {
        let mut invalid = task("retry-budget");
        invalid.spec.max_retries = 1;
        invalid.execution.retry_count = 2;
        let mut snapshot = RuntimePlanSnapshot {
            revision: 1,
            tasks: vec![invalid.clone()],
        };

        assert_eq!(
            claim_runtime_task(&mut snapshot, &invalid, 1),
            Err(RuntimeTaskMutationError::RetryBudgetInvariant {
                task_id: "retry-budget".to_string(),
                retry_count: 2,
                max_retries: 1,
            })
        );
        assert_eq!(snapshot.tasks, vec![invalid]);
        Ok(())
    }

    /// 结算到 Retrying 时被拒绝且不修改快照。
    #[test]
    fn settle_rejects_retrying_without_mutating_snapshot()
    -> std::result::Result<(), RuntimeTaskMutationError> {
        let expected = task("a");
        let mut snapshot = RuntimePlanSnapshot {
            revision: 7,
            tasks: vec![expected.clone()],
        };
        let claim = match claim_runtime_task(&mut snapshot, &expected, 7)? {
            RuntimeTaskClaimOutcome::Claimed(claim) => claim,
            RuntimeTaskClaimOutcome::ReloadSnapshot => {
                return Err(RuntimeTaskMutationError::InvalidTransition {
                    message: "claim unexpectedly reloaded".to_string(),
                });
            }
        };
        let before_revision = snapshot.revision;
        let before_tasks = snapshot.tasks.clone();

        let error = settle_runtime_claim(
            &mut snapshot,
            "a",
            &claim,
            TaskStatus::Retrying {
                attempt: claim.attempt,
                last_error: "transient".to_string(),
            },
        )
        .err()
        .ok_or_else(|| RuntimeTaskMutationError::InvalidTransition {
            message: "direct Retrying settlement unexpectedly succeeded".to_string(),
        })?;

        assert_eq!(
            error,
            RuntimeTaskMutationError::SettlementRequiresInactiveStatus {
                status: TaskStatus::Retrying {
                    attempt: claim.attempt,
                    last_error: "transient".to_string(),
                },
            }
        );
        assert_eq!(snapshot.revision, before_revision);
        assert_eq!(snapshot.tasks, before_tasks);
        assert!(runtime_claim_is_current(&snapshot, "a", &claim)?);
        Ok(())
    }

    /// 重新入队变更保持认领边界。
    #[test]
    fn requeue_mutation_preserves_claim_boundaries()
    -> std::result::Result<(), RuntimeTaskMutationError> {
        let expected = task("a");
        let mut snapshot = RuntimePlanSnapshot {
            revision: 1,
            tasks: vec![expected.clone()],
        };
        let claim = match claim_runtime_task(&mut snapshot, &expected, 1)? {
            RuntimeTaskClaimOutcome::Claimed(claim) => claim,
            RuntimeTaskClaimOutcome::ReloadSnapshot => {
                return Err(RuntimeTaskMutationError::InvalidTransition {
                    message: "claim unexpectedly reloaded".to_string(),
                });
            }
        };
        assert_eq!(
            requeue_runtime_claim(
                &mut snapshot,
                "a",
                &claim,
                Some("fingerprint".to_string()),
                "transient".to_string(),
                RuntimeRetryExhaustion::Failed,
            )?,
            RuntimeTaskRequeueOutcome::Requeued
        );
        let retried =
            snapshot
                .tasks
                .first()
                .ok_or_else(|| RuntimeTaskMutationError::InvalidTransition {
                    message: "retried task is missing".to_string(),
                })?;
        assert_eq!(retried.execution.status, TaskStatus::Pending);
        assert_eq!(retried.execution.retry_count, 1);
        Ok(())
    }

    /// 类型化处理请求走规范的认领结算路径。
    #[test]
    fn typed_resolution_request_uses_canonical_claim_settlement()
    -> std::result::Result<(), RuntimeTaskMutationError> {
        let completed = task("completed");
        let retrying = task("retrying");
        let timed_out = task("timed-out");
        let mut snapshot = RuntimePlanSnapshot {
            revision: 3,
            tasks: vec![completed.clone(), retrying.clone(), timed_out.clone()],
        };
        let completed_claim = match claim_runtime_task(&mut snapshot, &completed, 3)? {
            RuntimeTaskClaimOutcome::Claimed(claim) => claim,
            RuntimeTaskClaimOutcome::ReloadSnapshot => {
                return Err(RuntimeTaskMutationError::InvalidTransition {
                    message: "completed request claim unexpectedly reloaded".to_string(),
                });
            }
        };
        let retry_claim = match claim_runtime_task(&mut snapshot, &retrying, 3)? {
            RuntimeTaskClaimOutcome::Claimed(claim) => claim,
            RuntimeTaskClaimOutcome::ReloadSnapshot => {
                return Err(RuntimeTaskMutationError::InvalidTransition {
                    message: "retry request claim unexpectedly reloaded".to_string(),
                });
            }
        };
        let timeout_claim = match claim_runtime_task(&mut snapshot, &timed_out, 3)? {
            RuntimeTaskClaimOutcome::Claimed(claim) => claim,
            RuntimeTaskClaimOutcome::ReloadSnapshot => {
                return Err(RuntimeTaskMutationError::InvalidTransition {
                    message: "timeout request claim unexpectedly reloaded".to_string(),
                });
            }
        };

        assert_eq!(
            settle_runtime_resolution(
                &mut snapshot,
                "completed",
                &completed_claim,
                RuntimeTaskResolutionRequest::Completed,
            )?,
            RuntimeTaskResolution::Completed
        );
        assert_eq!(
            settle_runtime_resolution(
                &mut snapshot,
                "retrying",
                &retry_claim,
                RuntimeTaskResolutionRequest::Requeue {
                    failure_fingerprint: Some("transient".to_string()),
                    error: "retry later".to_string(),
                    exhaustion: RuntimeRetryExhaustion::Failed,
                },
            )?,
            RuntimeTaskResolution::Pending
        );
        assert_eq!(
            settle_runtime_resolution(
                &mut snapshot,
                "timed-out",
                &timeout_claim,
                RuntimeTaskResolutionRequest::TimedOut {
                    error: "provider deadline elapsed".to_string(),
                },
            )?,
            RuntimeTaskResolution::TimedOut {
                error: "provider deadline elapsed".to_string(),
            }
        );
        assert_eq!(
            settle_runtime_resolution(
                &mut snapshot,
                "completed",
                &completed_claim,
                RuntimeTaskResolutionRequest::Failed {
                    error: "late".to_string(),
                },
            )?,
            RuntimeTaskResolution::Superseded
        );
        let completed = snapshot
            .tasks
            .iter()
            .find(|task| task.spec.id == "completed")
            .ok_or_else(|| RuntimeTaskMutationError::InvalidTransition {
                message: "completed request task is missing".to_string(),
            })?;
        let retrying = snapshot
            .tasks
            .iter()
            .find(|task| task.spec.id == "retrying")
            .ok_or_else(|| RuntimeTaskMutationError::InvalidTransition {
                message: "retry request task is missing".to_string(),
            })?;
        let timed_out = snapshot
            .tasks
            .iter()
            .find(|task| task.spec.id == "timed-out")
            .ok_or_else(|| RuntimeTaskMutationError::InvalidTransition {
                message: "timeout request task is missing".to_string(),
            })?;
        assert_eq!(completed.execution.status, TaskStatus::Completed);
        assert!(completed.execution.claim.is_none());
        assert_eq!(retrying.execution.status, TaskStatus::Pending);
        assert_eq!(retrying.execution.retry_count, 1);
        assert!(retrying.execution.claim.is_none());
        assert_eq!(
            timed_out.execution.status,
            TaskStatus::TimedOut {
                error: "provider deadline elapsed".to_string(),
            }
        );
        assert!(timed_out.execution.claim.is_none());
        Ok(())
    }

    /// 重试耗尽时保留类型化的超时状态。
    #[test]
    fn retry_exhaustion_preserves_typed_timeout()
    -> std::result::Result<(), RuntimeTaskMutationError> {
        let mut expected = task("timeout");
        expected.spec.max_retries = 0;
        let mut snapshot = RuntimePlanSnapshot {
            revision: 1,
            tasks: vec![expected.clone()],
        };
        let claim = match claim_runtime_task(&mut snapshot, &expected, 1)? {
            RuntimeTaskClaimOutcome::Claimed(claim) => claim,
            RuntimeTaskClaimOutcome::ReloadSnapshot => {
                return Err(RuntimeTaskMutationError::InvalidTransition {
                    message: "timeout claim unexpectedly reloaded".to_string(),
                });
            }
        };

        assert_eq!(
            settle_runtime_resolution(
                &mut snapshot,
                "timeout",
                &claim,
                RuntimeTaskResolutionRequest::Requeue {
                    failure_fingerprint: Some("timeout-fingerprint".to_string()),
                    error: "provider deadline elapsed".to_string(),
                    exhaustion: RuntimeRetryExhaustion::TimedOut,
                },
            )?,
            RuntimeTaskResolution::TimedOut {
                error: "provider deadline elapsed".to_string(),
            }
        );
        let exhausted =
            snapshot
                .tasks
                .first()
                .ok_or_else(|| RuntimeTaskMutationError::InvalidTransition {
                    message: "timeout task is missing".to_string(),
                })?;
        assert_eq!(
            exhausted.execution.status,
            TaskStatus::TimedOut {
                error: "provider deadline elapsed".to_string(),
            }
        );
        assert_eq!(
            exhausted.execution.failure_fingerprint.as_deref(),
            Some("timeout-fingerprint")
        );
        assert!(exhausted.execution.claim.is_none());
        Ok(())
    }

    /// 重试预算为 0 时首次失败即耗尽。
    #[test]
    fn retry_budget_zero_exhausts_the_first_failed_claim()
    -> std::result::Result<(), RuntimeTaskMutationError> {
        let mut expected = task("a");
        expected.spec.max_retries = 0;
        let mut snapshot = RuntimePlanSnapshot {
            revision: 1,
            tasks: vec![expected.clone()],
        };
        let claim = match claim_runtime_task(&mut snapshot, &expected, 1)? {
            RuntimeTaskClaimOutcome::Claimed(claim) => claim,
            RuntimeTaskClaimOutcome::ReloadSnapshot => {
                return Err(RuntimeTaskMutationError::InvalidTransition {
                    message: "claim unexpectedly reloaded".to_string(),
                });
            }
        };

        assert_eq!(
            requeue_runtime_claim(
                &mut snapshot,
                "a",
                &claim,
                Some("fp-zero".to_string()),
                "no retry".to_string(),
                RuntimeRetryExhaustion::Failed,
            )?,
            RuntimeTaskRequeueOutcome::Exhausted
        );
        let exhausted =
            snapshot
                .tasks
                .first()
                .ok_or_else(|| RuntimeTaskMutationError::InvalidTransition {
                    message: "exhausted task is missing".to_string(),
                })?;
        assert_eq!(
            exhausted.execution.status,
            TaskStatus::Failed("no retry".to_string())
        );
        assert_eq!(exhausted.execution.retry_count, 0);
        assert_eq!(
            exhausted.execution.failure_fingerprint.as_deref(),
            Some("fp-zero")
        );
        assert!(exhausted.execution.claim.is_none());
        Ok(())
    }

    /// 在重试预算边界之前重新入队，到达边界后耗尽。
    #[test]
    fn retry_budget_requeues_until_boundary_then_exhausts()
    -> std::result::Result<(), RuntimeTaskMutationError> {
        let mut expected = task("a");
        expected.spec.max_retries = 2;
        let mut snapshot = RuntimePlanSnapshot {
            revision: 4,
            tasks: vec![expected.clone()],
        };

        for retry_count in 1..=2 {
            let claim = match claim_runtime_task(&mut snapshot, &expected, 4)? {
                RuntimeTaskClaimOutcome::Claimed(claim) => claim,
                RuntimeTaskClaimOutcome::ReloadSnapshot => {
                    return Err(RuntimeTaskMutationError::InvalidTransition {
                        message: "claim unexpectedly reloaded".to_string(),
                    });
                }
            };
            assert_eq!(
                requeue_runtime_claim(
                    &mut snapshot,
                    "a",
                    &claim,
                    None,
                    format!("retry {retry_count}"),
                    RuntimeRetryExhaustion::Failed,
                )?,
                RuntimeTaskRequeueOutcome::Requeued
            );
            let retried = snapshot.tasks.first().ok_or_else(|| {
                RuntimeTaskMutationError::InvalidTransition {
                    message: "retried task is missing".to_string(),
                }
            })?;
            assert_eq!(retried.execution.retry_count, retry_count);
            assert_eq!(retried.execution.status, TaskStatus::Pending);
            expected = retried.clone();
        }

        let final_claim = match claim_runtime_task(&mut snapshot, &expected, 4)? {
            RuntimeTaskClaimOutcome::Claimed(claim) => claim,
            RuntimeTaskClaimOutcome::ReloadSnapshot => {
                return Err(RuntimeTaskMutationError::InvalidTransition {
                    message: "boundary claim unexpectedly reloaded".to_string(),
                });
            }
        };
        assert_eq!(final_claim.attempt, 3);
        assert_eq!(
            requeue_runtime_claim(
                &mut snapshot,
                "a",
                &final_claim,
                Some("fp-final".to_string()),
                "exhausted".to_string(),
                RuntimeRetryExhaustion::Failed,
            )?,
            RuntimeTaskRequeueOutcome::Exhausted
        );
        assert_eq!(
            requeue_runtime_claim(
                &mut snapshot,
                "a",
                &final_claim,
                None,
                "late".to_string(),
                RuntimeRetryExhaustion::Failed,
            )?,
            RuntimeTaskRequeueOutcome::Superseded
        );
        let exhausted =
            snapshot
                .tasks
                .first()
                .ok_or_else(|| RuntimeTaskMutationError::InvalidTransition {
                    message: "boundary task is missing".to_string(),
                })?;
        assert_eq!(
            exhausted.execution.status,
            TaskStatus::Failed("exhausted".to_string())
        );
        assert_eq!(exhausted.execution.retry_count, 2);
        assert!(exhausted.execution.claim.is_none());
        Ok(())
    }

    /// 显式重试支持每种可重启状态并保留失败指纹。
    #[test]
    fn explicit_retry_supports_each_restartable_state_and_preserves_failure_fingerprint()
    -> std::result::Result<(), RuntimeTaskMutationError> {
        let restartable = vec![
            TaskStatus::Failed("failed".to_string()),
            TaskStatus::TimedOut {
                error: "timed out".to_string(),
            },
            TaskStatus::Blocked("blocked".to_string()),
        ];
        for status in restartable {
            let mut expected = task("restartable");
            expected.execution.status = status;
            expected.execution.retry_count = 1;
            expected.execution.failure_fingerprint = Some("stable-fingerprint".to_string());
            let mut snapshot = RuntimePlanSnapshot {
                revision: 8,
                tasks: vec![expected.clone()],
            };

            assert_eq!(
                retry_runtime_task(&mut snapshot, &expected, 8)?,
                RuntimeTaskRetryOutcome::Retried { retry_count: 2 }
            );
            let retried = snapshot.tasks.first().ok_or_else(|| {
                RuntimeTaskMutationError::InvalidTransition {
                    message: "explicitly retried task is missing".to_string(),
                }
            })?;
            assert_eq!(retried.execution.status, TaskStatus::Pending);
            assert_eq!(retried.execution.retry_count, 2);
            assert_eq!(
                retried.execution.failure_fingerprint.as_deref(),
                Some("stable-fingerprint")
            );
            assert!(retried.execution.claim.is_none());
        }
        Ok(())
    }

    /// 恢复暂停任务不消耗重试预算。
    #[test]
    fn paused_resume_does_not_consume_retry_budget()
    -> std::result::Result<(), RuntimeTaskMutationError> {
        let mut paused = task("paused");
        paused.execution.status = TaskStatus::Paused("user requested pause".to_string());
        paused.execution.retry_count = 1;
        paused.execution.failure_fingerprint = Some("prior-failure".to_string());
        let mut snapshot = RuntimePlanSnapshot {
            revision: 9,
            tasks: vec![paused.clone()],
        };

        assert_eq!(
            resume_runtime_task(&mut snapshot, &paused, 9)?,
            RuntimeTaskResumeOutcome::Resumed
        );
        let resumed =
            snapshot
                .tasks
                .first()
                .ok_or_else(|| RuntimeTaskMutationError::InvalidTransition {
                    message: "resumed task is missing".to_string(),
                })?;
        assert_eq!(resumed.execution.status, TaskStatus::Pending);
        assert_eq!(resumed.execution.retry_count, 1);
        assert_eq!(
            resumed.execution.failure_fingerprint.as_deref(),
            Some("prior-failure")
        );
        assert!(resumed.execution.claim.is_none());
        Ok(())
    }

    /// 暂停会结算认领，并保持未启动任务可恢复。
    #[test]
    fn pause_settles_claim_and_keeps_unstarted_tasks_resumable()
    -> std::result::Result<(), RuntimeTaskMutationError> {
        let active = task("active");
        let pending = task("pending");
        let mut snapshot = RuntimePlanSnapshot {
            revision: 4,
            tasks: vec![active.clone(), pending],
        };
        let claim = match claim_runtime_task(&mut snapshot, &active, 4)? {
            RuntimeTaskClaimOutcome::Claimed(claim) => claim,
            RuntimeTaskClaimOutcome::ReloadSnapshot => {
                return Err(RuntimeTaskMutationError::InvalidTransition {
                    message: "pause fixture claim unexpectedly reloaded".to_string(),
                });
            }
        };

        let receipt = match settle_runtime_interruption(
            &mut snapshot,
            4,
            RuntimeInterruptionDisposition::Paused {
                reason: "operator pause".to_string(),
            },
        )? {
            RuntimeInterruptionSettlementOutcome::Settled(receipt) => receipt,
            RuntimeInterruptionSettlementOutcome::ReloadSnapshot => {
                return Err(RuntimeTaskMutationError::InvalidTransition {
                    message: "pause settlement unexpectedly reloaded".to_string(),
                });
            }
        };
        assert_eq!(receipt.interrupted_task_ids, vec!["active".to_string()]);
        assert_eq!(receipt.pending_task_ids, vec!["pending".to_string()]);
        assert!(!runtime_claim_is_current(&snapshot, "active", &claim)?);
        let paused = snapshot
            .tasks
            .iter()
            .find(|task| task.spec.id == "active")
            .cloned()
            .ok_or_else(|| RuntimeTaskMutationError::InvalidTransition {
                message: "paused task is missing".to_string(),
            })?;
        assert_eq!(paused.execution.retry_count, 0);
        assert!(paused.execution.claim.is_none());
        assert_eq!(
            resume_runtime_task(&mut snapshot, &paused, 4)?,
            RuntimeTaskResumeOutcome::Resumed
        );
        let resumed = snapshot
            .tasks
            .iter()
            .find(|task| task.spec.id == "active")
            .ok_or_else(|| RuntimeTaskMutationError::InvalidTransition {
                message: "resumed active task is missing".to_string(),
            })?;
        assert_eq!(resumed.execution.status, TaskStatus::Pending);
        assert_eq!(resumed.execution.retry_count, 0);
        Ok(())
    }

    /// 取消保留终态兄弟任务并取消所有未完成任务。
    #[test]
    fn cancellation_retains_terminal_sibling_and_cancels_all_unfinished()
    -> std::result::Result<(), RuntimeTaskMutationError> {
        let mut completed = task("completed");
        completed.execution.status = TaskStatus::Completed;
        let active = task("active");
        let mut snapshot = RuntimePlanSnapshot {
            revision: 6,
            tasks: vec![completed, active.clone(), task("pending")],
        };
        let _claim = match claim_runtime_task(&mut snapshot, &active, 6)? {
            RuntimeTaskClaimOutcome::Claimed(claim) => claim,
            RuntimeTaskClaimOutcome::ReloadSnapshot => {
                return Err(RuntimeTaskMutationError::InvalidTransition {
                    message: "cancel fixture claim unexpectedly reloaded".to_string(),
                });
            }
        };
        let before = snapshot.tasks.clone();
        assert_eq!(
            cancel_unfinished_runtime_tasks(&mut snapshot, 5)?,
            RuntimeInterruptionSettlementOutcome::ReloadSnapshot
        );
        assert_eq!(snapshot.tasks, before);

        let receipt = match cancel_unfinished_runtime_tasks(&mut snapshot, 6)? {
            RuntimeInterruptionSettlementOutcome::Settled(receipt) => receipt,
            RuntimeInterruptionSettlementOutcome::ReloadSnapshot => {
                return Err(RuntimeTaskMutationError::InvalidTransition {
                    message: "cancel settlement unexpectedly reloaded".to_string(),
                });
            }
        };
        assert_eq!(
            receipt.interrupted_task_ids,
            vec!["active".to_string(), "pending".to_string()]
        );
        assert_eq!(
            receipt.retained_terminal_task_ids,
            vec!["completed".to_string()]
        );
        for task in &snapshot.tasks {
            if task.spec.id == "completed" {
                assert_eq!(task.execution.status, TaskStatus::Completed);
            } else {
                assert_eq!(task.execution.status, TaskStatus::Cancelled);
            }
            assert!(task.execution.claim.is_none());
        }
        Ok(())
    }

    /// 认领在更新的图修订下仍有效，但身份损坏时失败关闭。
    #[test]
    fn claims_survive_newer_graph_revision_but_corrupt_identity_fails_closed()
    -> std::result::Result<(), RuntimeTaskMutationError> {
        let expected = task("claimed");
        let mut snapshot = RuntimePlanSnapshot {
            revision: 2,
            tasks: vec![expected.clone()],
        };
        let claim = match claim_runtime_task(&mut snapshot, &expected, 2)? {
            RuntimeTaskClaimOutcome::Claimed(claim) => claim,
            RuntimeTaskClaimOutcome::ReloadSnapshot => {
                return Err(RuntimeTaskMutationError::InvalidTransition {
                    message: "stale fixture claim unexpectedly reloaded".to_string(),
                });
            }
        };
        snapshot.revision = 3;
        assert!(runtime_claim_is_current(&snapshot, "claimed", &claim)?);
        assert_eq!(
            settle_runtime_claim(&mut snapshot, "claimed", &claim, TaskStatus::Completed)?,
            RuntimeTaskSettlementOutcome::Settled
        );

        let expected = task("corrupt");
        let mut corrupt_snapshot = RuntimePlanSnapshot {
            revision: 2,
            tasks: vec![expected.clone()],
        };
        let corrupt_claim = match claim_runtime_task(&mut corrupt_snapshot, &expected, 2)? {
            RuntimeTaskClaimOutcome::Claimed(claim) => claim,
            RuntimeTaskClaimOutcome::ReloadSnapshot => {
                return Err(RuntimeTaskMutationError::InvalidTransition {
                    message: "corrupt fixture claim unexpectedly reloaded".to_string(),
                });
            }
        };
        let claimed = corrupt_snapshot.tasks.first_mut().ok_or_else(|| {
            RuntimeTaskMutationError::InvalidTransition {
                message: "claimed fixture is missing".to_string(),
            }
        })?;
        claimed.spec.title = "changed after claim".to_string();
        assert!(!runtime_claim_is_current(
            &corrupt_snapshot,
            "corrupt",
            &corrupt_claim
        )?);
        let before = corrupt_snapshot.tasks.clone();
        assert!(matches!(
            settle_runtime_interruption(
                &mut corrupt_snapshot,
                2,
                RuntimeInterruptionDisposition::Cancelled,
            ),
            Err(RuntimeTaskMutationError::InvalidClaim { .. })
        ));
        assert_eq!(corrupt_snapshot.tasks, before);
        Ok(())
    }

    /// 快照认领校验拒绝非活跃认领，并允许外部管理的 Running 任务。
    #[test]
    fn snapshot_claim_validation_rejects_inactive_claims_and_allows_external_running()
    -> std::result::Result<(), RuntimeTaskMutationError> {
        for status in [TaskStatus::Pending, TaskStatus::Completed] {
            let mut invalid = task("invalid-claim");
            invalid.execution.status = status;
            invalid.execution.claim = Some(TaskClaim::new(
                1,
                1,
                invalid
                    .spec
                    .stable_hash()
                    .map_err(|message| RuntimeTaskMutationError::InvalidTaskSpec { message })?,
            ));
            let snapshot = RuntimePlanSnapshot {
                revision: 1,
                tasks: vec![invalid],
            };
            assert!(matches!(
                validate_runtime_snapshot_claims(&snapshot),
                Err(RuntimeTaskMutationError::InvalidClaim { .. })
            ));
        }

        let mut retrying_without_claim = task("retrying-without-claim");
        retrying_without_claim.execution.status = TaskStatus::Retrying {
            attempt: 1,
            last_error: "transient".to_string(),
        };
        let invalid_retrying = RuntimePlanSnapshot {
            revision: 1,
            tasks: vec![retrying_without_claim],
        };
        assert!(matches!(
            validate_runtime_snapshot_claims(&invalid_retrying),
            Err(RuntimeTaskMutationError::InvalidClaim { .. })
        ));

        let mut external = task("external");
        external.execution.status = TaskStatus::Running;
        let mut snapshot = RuntimePlanSnapshot {
            revision: 2,
            tasks: vec![external],
        };
        validate_runtime_snapshot_claims(&snapshot)?;
        let receipt = match settle_runtime_interruption(
            &mut snapshot,
            2,
            RuntimeInterruptionDisposition::Paused {
                reason: "pause external work".to_string(),
            },
        )? {
            RuntimeInterruptionSettlementOutcome::Settled(receipt) => receipt,
            RuntimeInterruptionSettlementOutcome::ReloadSnapshot => {
                return Err(RuntimeTaskMutationError::InvalidTransition {
                    message: "external pause unexpectedly reloaded".to_string(),
                });
            }
        };
        assert_eq!(receipt.interrupted_task_ids, vec!["external".to_string()]);
        assert_eq!(
            snapshot.tasks.first().map(|task| &task.execution.status),
            Some(&TaskStatus::Paused("pause external work".to_string()))
        );
        Ok(())
    }

    /// 显式重试在预算为 0 和 N 的边界上表现精确。
    #[test]
    fn explicit_retry_budget_zero_and_n_boundary_are_exact()
    -> std::result::Result<(), RuntimeTaskMutationError> {
        let mut zero = task("zero");
        zero.spec.max_retries = 0;
        zero.execution.status = TaskStatus::Failed("failed".to_string());
        let mut zero_snapshot = RuntimePlanSnapshot {
            revision: 1,
            tasks: vec![zero.clone()],
        };
        assert_eq!(
            retry_runtime_task(&mut zero_snapshot, &zero, 1)?,
            RuntimeTaskRetryOutcome::Exhausted {
                retry_count: 0,
                max_retries: 0,
            }
        );
        assert_eq!(zero_snapshot.tasks, vec![zero]);

        let mut expected = task("bounded");
        expected.spec.max_retries = 2;
        expected.execution.status = TaskStatus::Failed("attempt one".to_string());
        let mut snapshot = RuntimePlanSnapshot {
            revision: 5,
            tasks: vec![expected.clone()],
        };
        assert_eq!(
            retry_runtime_task(&mut snapshot, &expected, 5)?,
            RuntimeTaskRetryOutcome::Retried { retry_count: 1 }
        );
        let current = snapshot.tasks.first_mut().ok_or_else(|| {
            RuntimeTaskMutationError::InvalidTransition {
                message: "first retry lost task".to_string(),
            }
        })?;
        current.execution.status = TaskStatus::TimedOut {
            error: "attempt two".to_string(),
        };
        current.execution.failure_fingerprint = Some("attempt-two".to_string());
        let second_expected = current.clone();
        assert_eq!(
            retry_runtime_task(&mut snapshot, &second_expected, 5)?,
            RuntimeTaskRetryOutcome::Retried { retry_count: 2 }
        );
        let current = snapshot.tasks.first_mut().ok_or_else(|| {
            RuntimeTaskMutationError::InvalidTransition {
                message: "second retry lost task".to_string(),
            }
        })?;
        current.execution.status = TaskStatus::Blocked("attempt three".to_string());
        let exhausted_expected = current.clone();
        let before_exhaustion = snapshot.tasks.clone();
        assert_eq!(
            retry_runtime_task(&mut snapshot, &exhausted_expected, 5)?,
            RuntimeTaskRetryOutcome::Exhausted {
                retry_count: 2,
                max_retries: 2,
            }
        );
        assert_eq!(snapshot.tasks, before_exhaustion);
        Ok(())
    }

    /// 显式重试拒绝过期、无效和已认领的任务且不产生变更。
    #[test]
    fn explicit_retry_rejects_stale_invalid_and_claimed_tasks_without_mutation()
    -> std::result::Result<(), RuntimeTaskMutationError> {
        let mut expected = task("retry");
        expected.execution.status = TaskStatus::Failed("failed".to_string());
        expected.execution.failure_fingerprint = Some("current".to_string());
        let mut snapshot = RuntimePlanSnapshot {
            revision: 11,
            tasks: vec![expected.clone()],
        };
        let before = snapshot.tasks.clone();
        assert_eq!(
            retry_runtime_task(&mut snapshot, &expected, 10)?,
            RuntimeTaskRetryOutcome::Superseded
        );
        let mut stale_fingerprint = expected.clone();
        stale_fingerprint.execution.failure_fingerprint = Some("stale".to_string());
        assert_eq!(
            retry_runtime_task(&mut snapshot, &stale_fingerprint, 11)?,
            RuntimeTaskRetryOutcome::Superseded
        );
        assert_eq!(snapshot.tasks, before);

        let invalid_statuses = vec![
            TaskStatus::Pending,
            TaskStatus::Running,
            TaskStatus::Completed,
            TaskStatus::Skipped,
            TaskStatus::Cancelled,
            TaskStatus::Paused("paused".to_string()),
            TaskStatus::Retrying {
                attempt: 1,
                last_error: "retrying".to_string(),
            },
        ];
        for status in invalid_statuses {
            let mut invalid = task("invalid");
            invalid.execution.status = status.clone();
            let mut invalid_snapshot = RuntimePlanSnapshot {
                revision: 12,
                tasks: vec![invalid.clone()],
            };
            let error = retry_runtime_task(&mut invalid_snapshot, &invalid, 12)
                .err()
                .ok_or_else(|| RuntimeTaskMutationError::InvalidTransition {
                    message: "invalid explicit retry unexpectedly succeeded".to_string(),
                })?;
            assert_eq!(
                error,
                RuntimeTaskMutationError::ExplicitRetryRequiresRestartableStatus {
                    task_id: "invalid".to_string(),
                    status,
                }
            );
            assert_eq!(invalid_snapshot.tasks, vec![invalid]);
        }

        let mut claimed = task("claimed");
        claimed.execution.status = TaskStatus::Blocked("blocked".to_string());
        claimed.execution.claim = Some(TaskClaim::new(12, 1, "spec".to_string()));
        let mut claimed_snapshot = RuntimePlanSnapshot {
            revision: 12,
            tasks: vec![claimed.clone()],
        };
        let error = retry_runtime_task(&mut claimed_snapshot, &claimed, 12)
            .err()
            .ok_or_else(|| RuntimeTaskMutationError::InvalidTransition {
                message: "claimed explicit retry unexpectedly succeeded".to_string(),
            })?;
        assert_eq!(
            error,
            RuntimeTaskMutationError::ExplicitRetryRequiresUnclaimedTask {
                task_id: "claimed".to_string(),
            }
        );
        assert_eq!(claimed_snapshot.tasks, vec![claimed]);
        Ok(())
    }
}
