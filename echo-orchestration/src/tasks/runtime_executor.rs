/*!
面向动态 Agent 计划的、由存储与策略驱动的 DAG 执行器。

框架自身负责依赖遍历、修订版本安全点、有界的 Subagent 波次调度、
取消、失败传播以及停滞检测。应用方则通过 [`RuntimeDagController`]
提供持久化、派发、复核与产品策略。
*/

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use echo_core::agent::ExecutionAdmission;
use echo_core::error::{ReactError, Result};
use serde::{Deserialize, Serialize};
use tokio::sync::{Semaphore, watch};
use tokio::task::{AbortHandle, Id, JoinSet};
use tokio_util::sync::CancellationToken;

use super::runtime::{
    DagDependencyState, DagExecutionState, NestedDelegationPolicy, RuntimeInterruptionDisposition,
    Task, TaskClaim, TaskId, TaskStatus, TaskSubagentContext,
};
use super::runtime_service::{
    RuntimeInterruptionSettlementOutcome, RuntimeTaskAttemptInterruptDisposition,
    RuntimeTaskAttemptInterruptProjectionError, RuntimeTaskSettlementOutcome,
    validate_runtime_snapshot_claims,
};
use crate::planning::PlanValidator;

/// 从运行时权威存储加载的、内部一致的某个计划修订版本。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimePlanSnapshot {
    /// 该快照对应的计划修订版本号，用于乐观并发控制。
    pub revision: u64,
    /// 该修订版本下的全部任务列表。
    pub tasks: Vec<Task>,
}

/// 当任务无法自动继续推进时，产品层做出的处置决定。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeStopDisposition {
    /// 判定为失败，终止该 DAG 的执行。
    Fail,
    /// 判定为暂停，可在之后恢复执行。
    Pause,
}

/// 应用方对一次已完成派发尝试的、尚未提交的评估结果。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeTaskResolutionRequest {
    /// 任务已成功完成。
    Completed,
    /// 请求重新排队等待重试。
    Requeue {
        /// 失败指纹，用于识别相同类型的失败以支持去重或统计。
        failure_fingerprint: Option<String>,
        /// 本次失败的错误描述。
        error: String,
        /// 若重试次数耗尽，最终应归为失败还是超时。
        exhaustion: RuntimeRetryExhaustion,
    },
    /// 任务被跳过，不再执行。
    Skipped,
    /// 任务最终失败。
    Failed {
        /// 失败原因描述。
        error: String,
    },
    /// 任务执行超时。
    TimedOut {
        /// 超时相关的错误描述。
        error: String,
    },
    /// 任务被阻塞，需要产品策略决定后续处置。
    Blocked {
        /// 阻塞原因描述。
        error: String,
        /// 阻塞后应采取的处置方式（失败或暂停）。
        disposition: RuntimeStopDisposition,
    },
    /// 任务被取消。
    Cancelled,
}

/// 对某个具体 claim 应用一次类型化解决请求后，已提交的最终结果。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeTaskResolution {
    /// 已完成。
    Completed,
    /// 已重新置为等待中（重试）。
    Pending,
    /// 已跳过。
    Skipped,
    /// 已失败。
    Failed {
        /// 失败原因描述。
        error: String,
    },
    /// 已超时。
    TimedOut {
        /// 超时相关的错误描述。
        error: String,
    },
    /// 已阻塞。
    Blocked {
        /// 阻塞原因描述。
        error: String,
        /// 阻塞后的处置方式。
        disposition: RuntimeStopDisposition,
    },
    /// 已取消。
    Cancelled,
    /// 该 claim 已被更新的一次尝试取代，本次结算不生效。
    Superseded,
}

/**
持久化 claim 结算完成后，进程本地控制投影给出的回执。
此时持久化任务状态已经是权威状态；可重试的清理失败会单独上报，
且不得触发第二次 CAS，也不得反转已提交的终态。
*/
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeAttemptControlCleanupReceipt {
    /// 本地控制投影已成功清退。
    Retired,
    /// 未找到对应的本地控制投影。
    NotFound,
    /// 该投影此前已被消费（清理过）。
    AlreadyConsumed,
    /// 清理失败，但可以重试；不影响已提交的持久化终态。
    RetryableFailure {
        /// 失败原因描述。
        error: String,
    },
}

/// 一次尝试控制清理动作的可观测记录，供诊断观察者使用。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeAttemptControlObservation {
    /// 所属的运行 ID。
    pub run_id: String,
    /// 所属的任务 ID。
    pub task_id: TaskId,
    /// 触发本次清理的 claim。
    pub claim: TaskClaim,
    /// 本次清理动作得到的回执。
    pub receipt: RuntimeAttemptControlCleanupReceipt,
}

/// 用于观察尝试控制清理动作的诊断回调类型。
pub type RuntimeAttemptControlObserver =
    Arc<dyn Fn(&RuntimeAttemptControlObservation) + Send + Sync>;

/**
当一次重新排队的 claim 用尽最后一次重试机会时使用的终态。

重试决策与终态分类保持分离，使应用方能够在框架完成结算后，
保留一个类型化的超时状态，而无需再修改规范化的任务状态。
*/
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeRetryExhaustion {
    /// 重试耗尽后归为失败。
    Failed,
    /// 重试耗尽后归为超时。
    TimedOut,
}

/// 从已加载的某个计划修订版本中原子性地认领一个任务后得到的结果。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeTaskClaimOutcome {
    /// 认领成功，得到对应的 claim。
    Claimed(TaskClaim),
    /**
    修订版本、状态或规格已发生变化。应重新加载快照；
    这是乐观并发控制的表现，而非任务失败。
    */
    ReloadSnapshot,
}

/// 一个无法达成正常结算结果的 claim 的终态结算方式。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeClaimAbandonment {
    /// 运行已停止，这个仍被持有的 claim 按类型化的停止策略处理。
    Interrupted {
        /// 具体的中断处置方式（取消或暂停）。
        disposition: RuntimeInterruptionDisposition,
    },
    /// 在达成正常结算结果之前，派发或结算基础设施出现了故障。
    Failed {
        /// 失败原因描述。
        error: String,
    },
}

/// 推进一次动态计划快照序列后得到的最终结果。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeDagOutcome {
    /// 全部任务已完成。
    Completed,
    /// 某个任务失败，导致整个 DAG 执行失败。
    Failed {
        /// 失败任务的 ID。
        failed_task_id: TaskId,
        /// 失败原因描述。
        error: String,
    },
    /// 执行被暂停，可在之后恢复。
    Paused {
        /// 触发暂停的任务 ID（若有）。
        task_id: Option<TaskId>,
        /// 暂停原因描述。
        reason: String,
    },
    /// DAG 陷入停滞（例如存在环或无法解除的阻塞）。
    Stalled {
        /// 停滞原因描述。
        reason: String,
    },
    /// 整个执行被取消。
    Cancelled,
}

/**
面向 [`super::RuntimeTaskService`] 的持久化、派发与产品策略适配器。

`resolve_dispatch` 负责具体应用层面的复核逻辑，但只返回一个尚未提交的
请求。`settle_resolution` 提供适配器的事务边界，必须调用
[`super::settle_runtime_resolution`] 来完成提交，而不能自行重新实现
claim、重试或终态迁移的逻辑。
*/
#[async_trait]
pub trait RuntimeDagController: Send + Sync + 'static {
    /// 一次任务派发成功后返回的输出类型。
    type DispatchOutput: Send + 'static;

    /// 加载当前运行所对应的最新计划快照。
    async fn load_snapshot(&self, run_id: &str) -> Result<RuntimePlanSnapshot>;

    /**
    原子性地将一个处于 Pending 状态、基于 `expected_revision` 的任务
    转换为一次已认领的 Running 尝试。
    */
    async fn claim_task(
        &self,
        run_id: &str,
        task: &Task,
        expected_revision: u64,
    ) -> Result<RuntimeTaskClaimOutcome>;

    /// 校验某个具体的物理 claim 是否仍然拥有其对应任务。
    async fn claim_is_current(
        &self,
        run_id: &str,
        task_id: &str,
        claim: &TaskClaim,
    ) -> Result<bool>;

    /**
    从就绪前沿中选出一个互不冲突的子集用于派发。

    默认实现会派发整个前沿。应用方可以出于产品特定的资源或文件所有权
    策略而推迟某些任务，但当 `ready_task_ids` 非空时，必须至少返回一个
    任务 ID。
    */
    fn select_ready_wave(&self, _tasks: &[Task], ready_task_ids: Vec<TaskId>) -> Vec<TaskId> {
        ready_task_ids
    }

    /// 派发一个任务并等待其执行结果。
    async fn dispatch_task(
        &self,
        context: TaskSubagentContext,
        task: Task,
    ) -> Result<Self::DispatchOutput>;

    /**
    在共享准入或信号量可能延迟派发之前，先注册由本次 claim 派生出的
    精确上下文。实现可以将该上下文绑定到进程本地的实时控制注册表；
    持久化 claim 仍然是权威来源，调用方会在触发该钩子之前先完成校验。
    */
    async fn reserve_attempt_control(&self, _context: &TaskSubagentContext) -> Result<()> {
        Ok(())
    }

    /**
    将一次持久化的精确中断请求投影到实时的预留/尝试注册表中。
    返回 `true` 表示控制器已接受该请求；运行时在返回前仍会重新校验
    持久化 claim 的状态。
    */
    async fn request_live_interrupt(
        &self,
        _run_id: &str,
        _task_id: &str,
        _claim: &TaskClaim,
    ) -> std::result::Result<
        Option<RuntimeTaskAttemptInterruptDisposition>,
        RuntimeTaskAttemptInterruptProjectionError,
    > {
        Ok(None)
    }

    /**
    在持久化 claim 已经达成终态或被取代之后，清退进程本地的控制投影。
    该清理动作仅用于诊断目的，绝不能修改已经提交的任务状态。
    */
    async fn retire_attempt_control(
        &self,
        _run_id: &str,
        _task: &Task,
        _claim: &TaskClaim,
    ) -> RuntimeAttemptControlCleanupReceipt {
        RuntimeAttemptControlCleanupReceipt::Retired
    }

    /**
    将进程本地的尝试投影与该运行的持久化快照中仍然有效的执行 ID
    进行核对与协调。
    */
    async fn reconcile_attempt_control(
        &self,
        _run_id: &str,
        _current_execution_ids: &HashSet<String>,
    ) -> Result<()> {
        Ok(())
    }

    /// 根据一次派发的结果，产出一个尚未提交的解决请求。
    async fn resolve_dispatch(
        &self,
        run_id: &str,
        claim: TaskClaim,
        task: Task,
        dispatch: Result<Self::DispatchOutput>,
    ) -> Result<RuntimeTaskResolutionRequest>;

    /// 原子性地应用一个由框架发起的解决请求，并返回其类型化的、已提交的回执。
    async fn settle_resolution(
        &self,
        run_id: &str,
        claim: &TaskClaim,
        task: &Task,
        request: RuntimeTaskResolutionRequest,
    ) -> Result<RuntimeTaskResolution>;

    /**
    结算一个无法达成正常解决结果的 claim。实现必须使用比较并交换（CAS）
    语义，以避免一个已被取代的 claim 覆盖更新的工作结果。
    */
    async fn abandon_claim(
        &self,
        run_id: &str,
        claim: &TaskClaim,
        task: &Task,
        abandonment: RuntimeClaimAbandonment,
    ) -> Result<RuntimeTaskSettlementOutcome>;

    /**
    在任务失败时，决定应将整个 DAG 判定为失败还是暂停。
    默认实现：当所有未完成任务都已失败或被阻塞时判定为失败，否则暂停。
    */
    async fn failed_task_disposition(
        &self,
        _run_id: &str,
        _task: &Task,
        all_unfinished_failed_or_blocked: bool,
    ) -> Result<RuntimeStopDisposition> {
        Ok(if all_unfinished_failed_or_blocked {
            RuntimeStopDisposition::Fail
        } else {
            RuntimeStopDisposition::Pause
        })
    }

    /**
    将取消令牌发出的停止请求映射为与产品无关的取消或可恢复的暂停。
    框架自身的任务结算结果始终是权威判断。
    */
    async fn interruption_disposition(
        &self,
        _run_id: &str,
    ) -> Result<RuntimeInterruptionDisposition> {
        Ok(RuntimeInterruptionDisposition::Cancelled)
    }

    /**
    使用 [`super::settle_runtime_interruption`]，在一个中断安全点原子性地
    结算所有未完成任务。若修订版本不匹配，必须返回 `ReloadSnapshot`；
    适配器不得重新实现 claim 或重试相关的语义。
    */
    async fn settle_interruption(
        &self,
        run_id: &str,
        expected_revision: u64,
        disposition: RuntimeInterruptionDisposition,
    ) -> Result<RuntimeInterruptionSettlementOutcome>;

    /// 通知应用方 DAG 已进入停滞状态（例如存在环或无法解除的阻塞）。
    async fn note_stalled(&self, _run_id: &str, _reason: &str) -> Result<()> {
        Ok(())
    }
}

/// [`super::RuntimeTaskService`] 接受的执行配置。
#[derive(Clone)]
pub struct RuntimeTaskServiceConfig {
    /// 同时运行的子 Agent 最大数量
    pub max_concurrent_subagents: usize,
    /// 轮询外部任务进度的时间间隔
    pub external_progress_poll_interval: Duration,
    /**
    允许可感知取消的子 Agent 完成持久化终态写入的最长时间，
    超时后仍未配合退出的分派将被中止。
    */
    pub cancellation_grace_period: Duration,
    /// 嵌套委派策略（是否允许派生子 Agent 及最大委派深度）
    pub delegation_policy: NestedDelegationPolicy,
    /// 可选的进程级准入控制，与子 Agent 分派器共享。
    pub shared_admission: Option<Arc<ExecutionAdmission>>,
    /// 可选的诊断观察者，用于观察 CAS 之后的实时控制清理。
    pub attempt_control_observer: Option<RuntimeAttemptControlObserver>,
}

impl std::fmt::Debug for RuntimeTaskServiceConfig {
    /// 输出调试信息；观察者回调只显示是否已设置。
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RuntimeTaskServiceConfig")
            .field("max_concurrent_subagents", &self.max_concurrent_subagents)
            .field(
                "external_progress_poll_interval",
                &self.external_progress_poll_interval,
            )
            .field("cancellation_grace_period", &self.cancellation_grace_period)
            .field("delegation_policy", &self.delegation_policy)
            .field("shared_admission", &self.shared_admission)
            .field(
                "has_attempt_control_observer",
                &self.attempt_control_observer.is_some(),
            )
            .finish()
    }
}

impl Default for RuntimeTaskServiceConfig {
    /// 默认配置：并发 4、轮询 250ms、取消宽限期 5s、最大委派深度 2。
    fn default() -> Self {
        Self {
            max_concurrent_subagents: 4,
            external_progress_poll_interval: Duration::from_millis(250),
            cancellation_grace_period: Duration::from_secs(5),
            delegation_policy: NestedDelegationPolicy {
                can_spawn_subagents: true,
                delegate_depth: 0,
                max_delegate_depth: 2,
            },
            shared_admission: None,
            attempt_control_observer: None,
        }
    }
}

/// 框架中面向可修订动态 Agent 计划的执行器。
pub(crate) struct RuntimeDagExecutor<C: RuntimeDagController> {
    /// 持久化状态与调度决策的控制器
    pub(crate) controller: Arc<C>,
    /// 执行配置
    pub(crate) config: RuntimeTaskServiceConfig,
    /// 计划结构校验器
    validator: PlanValidator,
    /// execution_id -> 该次尝试的取消令牌
    pub(crate) attempt_cancellations: Arc<Mutex<HashMap<String, CancellationToken>>>,
    /// execution_id -> 所属 run_id
    pub(crate) attempt_runs: Arc<Mutex<HashMap<String, String>>>,
    /// execution_id -> 分派任务的中止句柄
    pub(crate) attempt_abort_handles: Arc<Mutex<HashMap<String, AbortHandle>>>,
    /// tokio 任务 Id -> execution_id
    attempt_task_ids: Arc<Mutex<HashMap<Id, String>>>,
    /// execution_id -> run_id：已请求但尚未送达的尝试中断
    pub(crate) pending_attempt_interrupts: Arc<Mutex<HashMap<String, String>>>,
    /// execution_id -> 中止代次：已调度的延迟强制中止
    pub(crate) scheduled_attempt_aborts: Arc<Mutex<HashMap<String, u64>>>,
    /// execution_id -> 尝试结算投影
    attempt_settlements: Arc<Mutex<HashMap<String, AttemptSettlementProjection>>>,
}

/// 尝试结算所处阶段
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AttemptSettlementPhase {
    /// 尝试仍在运行
    Active,
    /// 本地句柄已 join，等待持久化结算（claim CAS）完成
    JoinedAwaitingDurableSettlement,
}

/// 单次尝试在本地的结算投影
struct AttemptSettlementProjection {
    /// 当前阶段
    phase: AttemptSettlementPhase,
    /// 结算结果广播发送端
    settled_tx: watch::Sender<Option<AttemptSettlementObservation>>,
    /// 该尝试的取消令牌
    cancel: CancellationToken,
}

/// 对外发布的尝试结算观察结果
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum AttemptSettlementObservation {
    /// 已完成持久化结算，附带最终状态
    Settled { status: String },
    /// 结算权威状态未知（例如持久化出错），附带错误信息
    AuthorityUnknown { error: String },
}

/// 外部对某次尝试的观察视图
pub(crate) enum RuntimeAttemptObservation {
    /// 尝试仍在运行，可取消并订阅结算结果
    Active {
        /// 该尝试的取消令牌
        cancel: CancellationToken,
        /// 结算结果订阅端
        settled_rx: watch::Receiver<Option<AttemptSettlementObservation>>,
    },
    /// 已 join，等待持久化结算
    JoinedAwaitingDurableSettlement {
        /// 结算结果订阅端
        settled_rx: watch::Receiver<Option<AttemptSettlementObservation>>,
    },
}

/// 中断边界的结算结果
enum InterruptionBoundary {
    /// 产生了最终的 DAG 执行结果
    Outcome(RuntimeDagOutcome),
    /// 需重新加载快照后继续
    ReloadSnapshot,
    /// 已无未完成的工作
    NoUnfinishedWork,
}

impl<C: RuntimeDagController> RuntimeDagExecutor<C> {
    /// 使用控制器与配置创建执行器，并使用默认的计划校验器。
    pub(crate) fn new(controller: Arc<C>, config: RuntimeTaskServiceConfig) -> Self {
        Self {
            controller,
            config,
            validator: PlanValidator::default(),
            attempt_cancellations: Arc::new(Mutex::new(HashMap::new())),
            attempt_runs: Arc::new(Mutex::new(HashMap::new())),
            attempt_abort_handles: Arc::new(Mutex::new(HashMap::new())),
            attempt_task_ids: Arc::new(Mutex::new(HashMap::new())),
            pending_attempt_interrupts: Arc::new(Mutex::new(HashMap::new())),
            scheduled_attempt_aborts: Arc::new(Mutex::new(HashMap::new())),
            attempt_settlements: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// 覆盖结构限制，同时保留规范的校验器。
    pub(crate) fn with_validator(mut self, validator: PlanValidator) -> Self {
        self.validator = validator;
        self
    }

    /**
    在取消宽限期结束后强制中止指定尝试；`rearm` 为 true 时重新计时。

    使用代次号保证只有最新一次调度会真正执行中止。
    */
    pub(crate) fn schedule_attempt_abort(&self, execution_id: String, rearm: bool) {
        let generation = {
            let Ok(mut scheduled) = self.scheduled_attempt_aborts.lock() else {
                return;
            };
            if scheduled.contains_key(&execution_id) && !rearm {
                return;
            }
            let generation = scheduled
                .get(&execution_id)
                .copied()
                .unwrap_or(0)
                .saturating_add(1);
            scheduled.insert(execution_id.clone(), generation);
            generation
        };
        let abort_handles = Arc::clone(&self.attempt_abort_handles);
        let scheduled_attempt_aborts = Arc::clone(&self.scheduled_attempt_aborts);
        let grace_period = self.config.cancellation_grace_period;
        tokio::spawn(async move {
            tokio::time::sleep(grace_period).await;
            let is_current = scheduled_attempt_aborts
                .lock()
                .ok()
                .and_then(|scheduled| scheduled.get(&execution_id).copied())
                == Some(generation);
            if !is_current {
                return;
            }
            let handle = abort_handles
                .lock()
                .ok()
                .and_then(|handles| handles.get(&execution_id).cloned());
            if let Some(handle) = handle.filter(|handle| !handle.is_finished()) {
                handle.abort();
            }
            if let Ok(mut scheduled) = scheduled_attempt_aborts.lock()
                && scheduled.get(&execution_id).copied() == Some(generation)
            {
                scheduled.remove(&execution_id);
            }
        });
    }

    /// 观察指定尝试的当前阶段，返回取消令牌与结算订阅端。
    pub(crate) fn observe_attempt(&self, execution_id: &str) -> Option<RuntimeAttemptObservation> {
        let settlements = self
            .attempt_settlements
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let projection = settlements.get(execution_id)?;
        let settled_rx = projection.settled_tx.subscribe();
        Some(match projection.phase {
            AttemptSettlementPhase::Active => RuntimeAttemptObservation::Active {
                cancel: projection.cancel.clone(),
                settled_rx,
            },
            AttemptSettlementPhase::JoinedAwaitingDurableSettlement => {
                RuntimeAttemptObservation::JoinedAwaitingDurableSettlement { settled_rx }
            }
        })
    }

    /// 将尝试标记为已 join、等待持久化结算。
    fn mark_attempt_joined(&self, execution_id: &str) {
        if let Ok(mut settlements) = self.attempt_settlements.lock()
            && let Some(projection) = settlements.get_mut(execution_id)
        {
            projection.phase = AttemptSettlementPhase::JoinedAwaitingDurableSettlement;
        }
    }

    /// 发布尝试的持久化结算状态。
    pub(crate) fn settle_attempt_projection(&self, execution_id: &str, status: impl Into<String>) {
        if let Ok(settlements) = self.attempt_settlements.lock()
            && let Some(projection) = settlements.get(execution_id)
        {
            projection
                .settled_tx
                .send_replace(Some(AttemptSettlementObservation::Settled {
                    status: status.into(),
                }));
        }
    }

    /**
    协调本地尝试控制状态：取消并回收不再属于当前快照的尝试。

    已 join 的陈旧尝试立即退役，仍在运行的则调度延迟中止。
    */
    pub(crate) fn reconcile_local_attempt_control(
        &self,
        run_id: &str,
        current_execution_ids: &HashSet<String>,
    ) {
        let stale_execution_ids = self
            .attempt_runs
            .lock()
            .map(|runs| {
                runs.iter()
                    .filter_map(|(execution_id, attempt_run_id)| {
                        (attempt_run_id == run_id && !current_execution_ids.contains(execution_id))
                            .then_some(execution_id.clone())
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        for execution_id in stale_execution_ids {
            let phase = self
                .attempt_settlements
                .lock()
                .ok()
                .and_then(|settlements| {
                    settlements
                        .get(&execution_id)
                        .map(|projection| projection.phase)
                });
            self.settle_attempt_projection(&execution_id, "superseded");
            if let Some(cancel) = self
                .attempt_cancellations
                .lock()
                .ok()
                .and_then(|controls| controls.get(&execution_id).cloned())
            {
                cancel.cancel();
            }
            if phase == Some(AttemptSettlementPhase::JoinedAwaitingDurableSettlement) {
                self.retire_local_attempt_control(&execution_id);
            } else {
                self.schedule_attempt_abort(execution_id, false);
            }
        }
        if let Ok(mut pending) = self.pending_attempt_interrupts.lock() {
            pending.retain(|execution_id, pending_run_id| {
                pending_run_id != run_id || current_execution_ids.contains(execution_id)
            });
        }
    }

    /// 测试辅助：判断执行器是否仍持有该尝试的任何监督状态。
    #[cfg(test)]
    pub(crate) fn has_attempt_supervisor_state(&self, execution_id: &str) -> bool {
        self.attempt_cancellations
            .lock()
            .is_ok_and(|controls| controls.contains_key(execution_id))
            || self
                .attempt_runs
                .lock()
                .is_ok_and(|runs| runs.contains_key(execution_id))
            || self
                .attempt_abort_handles
                .lock()
                .is_ok_and(|handles| handles.contains_key(execution_id))
            || self
                .pending_attempt_interrupts
                .lock()
                .is_ok_and(|pending| pending.contains_key(execution_id))
            || self
                .scheduled_attempt_aborts
                .lock()
                .is_ok_and(|scheduled| scheduled.contains_key(execution_id))
            || self
                .attempt_settlements
                .lock()
                .is_ok_and(|settlements| settlements.contains_key(execution_id))
    }

    /// 将尝试标记为结算权威状态未知。
    fn mark_attempt_authority_unknown(&self, execution_id: &str, error: impl Into<String>) {
        if let Ok(settlements) = self.attempt_settlements.lock()
            && let Some(projection) = settlements.get(execution_id)
        {
            projection.settled_tx.send_replace(Some(
                AttemptSettlementObservation::AuthorityUnknown {
                    error: error.into(),
                },
            ));
        }
    }

    /// 移除该尝试在本地的全部控制状态。
    fn retire_local_attempt_control(&self, execution_id: &str) {
        if let Ok(mut controls) = self.attempt_cancellations.lock() {
            controls.remove(execution_id);
        }
        if let Ok(mut runs) = self.attempt_runs.lock() {
            runs.remove(execution_id);
        }
        if let Ok(mut handles) = self.attempt_abort_handles.lock() {
            handles.remove(execution_id);
        }
        if let Ok(mut pending) = self.pending_attempt_interrupts.lock() {
            pending.remove(execution_id);
        }
        if let Ok(mut scheduled) = self.scheduled_attempt_aborts.lock() {
            scheduled.remove(execution_id);
        }
        if let Ok(mut settlements) = self.attempt_settlements.lock() {
            settlements.remove(execution_id);
        }
    }

    /// 若配置了观察者，则上报尝试控制清理回执。
    pub(crate) fn observe_attempt_control_cleanup(
        &self,
        run_id: &str,
        task_id: &str,
        claim: &TaskClaim,
        receipt: RuntimeAttemptControlCleanupReceipt,
    ) {
        if let Some(observer) = &self.config.attempt_control_observer {
            observer(&RuntimeAttemptControlObservation {
                run_id: run_id.to_string(),
                task_id: task_id.to_string(),
                claim: claim.clone(),
                receipt,
            });
        }
    }

    /**
    执行运行时 DAG

    循环加载快照、选择就绪波次、分派并结算任务，直到完成、失败、暂停或取消。
    */
    pub(crate) async fn execute(
        &self,
        run_id: &str,
        cancel: CancellationToken,
    ) -> Result<RuntimeDagOutcome> {
        let subagent_semaphore =
            Arc::new(Semaphore::new(self.config.max_concurrent_subagents.max(1)));
        let mut active_revision: Option<u64> = None;
        let mut failure_errors: HashMap<TaskId, String> = HashMap::new();

        loop {
            /*
            每个循环边界都是安全点：上一波次本地分派的
            所有句柄都已 join 并完成结算。
            */
            let snapshot = self.controller.load_snapshot(run_id).await?;
            if let Err(errors) = self.validator.validate_task_snapshot(&snapshot.tasks) {
                return Err(ReactError::Other(format!(
                    "invalid runtime plan snapshot: {}",
                    errors.join("; ")
                )));
            }
            validate_runtime_snapshot_claims(&snapshot).map_err(|error| {
                ReactError::Other(format!("invalid runtime claim snapshot: {error}"))
            })?;
            if active_revision != Some(snapshot.revision) {
                if let Some(previous) = active_revision {
                    tracing::info!(
                        run_id,
                        from_revision = previous,
                        to_revision = snapshot.revision,
                        "runtime DAG executor applied a plan revision at a safe point"
                    );
                }
                active_revision = Some(snapshot.revision);
            }

            let tasks = snapshot.tasks;
            let state = DagExecutionState::from_tasks(&tasks);

            if !state.cancelled.is_empty() {
                match self
                    .settle_interruption_boundary(
                        run_id,
                        snapshot.revision,
                        RuntimeInterruptionDisposition::Cancelled,
                        false,
                    )
                    .await?
                {
                    InterruptionBoundary::Outcome(outcome) => return Ok(outcome),
                    InterruptionBoundary::ReloadSnapshot => continue,
                    InterruptionBoundary::NoUnfinishedWork => {}
                }
            }

            if state.all_completed(&tasks) {
                return Ok(RuntimeDagOutcome::Completed);
            }

            let durable_pause = tasks
                .iter()
                .find(|task| matches!(task.execution.status, TaskStatus::Paused(_)))
                .map(|paused_task| RuntimeInterruptionDisposition::Paused {
                    reason: persisted_status_error(&paused_task.execution.status)
                        .unwrap_or_else(|| "runtime task paused".to_string()),
                });
            let requested_interruption = if cancel.is_cancelled() {
                Some(self.controller.interruption_disposition(run_id).await?)
            } else {
                None
            };
            let interruption =
                prioritize_interruption(durable_pause.clone(), requested_interruption);
            if let Some(disposition) = interruption {
                match self
                    .settle_interruption_boundary(
                        run_id,
                        snapshot.revision,
                        disposition,
                        durable_pause.is_none(),
                    )
                    .await?
                {
                    InterruptionBoundary::Outcome(outcome) => return Ok(outcome),
                    InterruptionBoundary::ReloadSnapshot => continue,
                    InterruptionBoundary::NoUnfinishedWork => {}
                }
            }

            if let Some(failed_task) = tasks
                .iter()
                .find(|task| state.failed.contains(&task.spec.id))
            {
                let dependency_states = state.dependency_states(&tasks);
                let derived_blocked: Vec<_> = dependency_states
                    .iter()
                    .filter_map(|(task_id, dependency_state)| {
                        matches!(
                            dependency_state,
                            DagDependencyState::BlockedByFailure { .. }
                        )
                        .then_some(task_id)
                    })
                    .collect();
                tracing::debug!(
                    run_id,
                    failed_task_id = %failed_task.spec.id,
                    derived_blocked_task_ids = ?derived_blocked,
                    "runtime DAG derived dependency blockers from the current snapshot"
                );

                let error = failure_errors
                    .remove(&failed_task.spec.id)
                    .or_else(|| persisted_status_error(&failed_task.execution.status))
                    .unwrap_or_else(|| format!("task '{}' failed", failed_task.spec.title));
                let disposition = self
                    .controller
                    .failed_task_disposition(
                        run_id,
                        failed_task,
                        state.all_unfinished_failed_or_blocked(&tasks),
                    )
                    .await?;
                return Ok(stop_outcome(
                    disposition,
                    failed_task.spec.id.clone(),
                    error,
                ));
            }

            let ready_task_ids = state.ready_task_ids(&tasks);
            if ready_task_ids.is_empty() {
                if !state.in_flight.is_empty() {
                    tokio::select! {
                        _ = cancel.cancelled() => {
                            let disposition = self.controller.interruption_disposition(run_id).await?;
                            match self
                                .settle_interruption_boundary(
                                    run_id,
                                    snapshot.revision,
                                    disposition,
                                    true,
                                )
                                .await?
                            {
                                InterruptionBoundary::Outcome(outcome) => return Ok(outcome),
                                InterruptionBoundary::ReloadSnapshot
                                | InterruptionBoundary::NoUnfinishedWork => {}
                            }
                        }
                        _ = tokio::time::sleep(self.config.external_progress_poll_interval) => {}
                    }
                    continue;
                }

                if let Some(blocked_task) = tasks
                    .iter()
                    .find(|task| matches!(task.execution.status, TaskStatus::Blocked(_)))
                {
                    let error = persisted_status_error(&blocked_task.execution.status)
                        .unwrap_or_else(|| format!("task '{}' blocked", blocked_task.spec.title));
                    let disposition = self
                        .controller
                        .failed_task_disposition(
                            run_id,
                            blocked_task,
                            state.all_unfinished_failed_or_blocked(&tasks),
                        )
                        .await?;
                    return Ok(stop_outcome(
                        disposition,
                        blocked_task.spec.id.clone(),
                        error,
                    ));
                }

                let reason = "DAG stalled with unfinished tasks (cycle or blocked)";
                self.controller.note_stalled(run_id, reason).await?;
                return Ok(RuntimeDagOutcome::Stalled {
                    reason: reason.to_string(),
                });
            }

            let selected_ids = self
                .controller
                .select_ready_wave(&tasks, ready_task_ids.clone());
            let selected_ids = validate_selected_wave(&ready_task_ids, selected_ids)?;
            let selected_set: HashSet<&str> = selected_ids.iter().map(String::as_str).collect();
            let dependency_states = state.dependency_states(&tasks);
            let selected_tasks: Vec<Task> = tasks
                .iter()
                .filter(|task| selected_set.contains(task.spec.id.as_str()))
                .cloned()
                .collect();

            tracing::info!(
                run_id,
                revision = snapshot.revision,
                ready_count = ready_task_ids.len(),
                selected_count = selected_tasks.len(),
                selected_tasks = ?selected_ids,
                completed_count = state.completed.len(),
                skipped_count = state.skipped.len(),
                total_count = tasks.len(),
                "runtime DAG executor dispatching wave"
            );

            let mut join_set = JoinSet::new();
            let mut outstanding_claims: HashMap<String, (Task, TaskClaim)> = HashMap::new();
            let mut wave_errors = Vec::new();
            for task in selected_tasks {
                if cancel.is_cancelled() {
                    break;
                }
                let controller = self.controller.clone();
                let semaphore = subagent_semaphore.clone();
                let shared_admission = self.config.shared_admission.clone();
                let claim = match self
                    .controller
                    .claim_task(run_id, &task, snapshot.revision)
                    .await
                {
                    Ok(RuntimeTaskClaimOutcome::Claimed(claim)) => claim,
                    Ok(RuntimeTaskClaimOutcome::ReloadSnapshot) => continue,
                    Err(error) => {
                        wave_errors.push(error.to_string());
                        break;
                    }
                };
                let claim_id = claim.claim_id.clone();
                /*
                每个 claim 持有一个子令牌：取消某一次具体尝试
                不得取消同一就绪波次中的兄弟任务。
                令牌会在等待共享准入之前发布。
                */
                let task_cancel = cancel.child_token();
                let execution_id = claim.execution_id(run_id, &task.spec.id);
                let pending_interrupt = self
                    .pending_attempt_interrupts
                    .lock()
                    .ok()
                    .is_some_and(|mut pending| pending.remove(&execution_id).is_some());
                if let Ok(mut controls) = self.attempt_cancellations.lock() {
                    controls.insert(execution_id.clone(), task_cancel.clone());
                }
                if let Ok(mut runs) = self.attempt_runs.lock() {
                    runs.insert(execution_id.clone(), run_id.to_string());
                }
                if let Ok(mut settlements) = self.attempt_settlements.lock() {
                    let (settled_tx, _) = watch::channel(None);
                    settlements.insert(
                        execution_id.clone(),
                        AttemptSettlementProjection {
                            phase: AttemptSettlementPhase::Active,
                            settled_tx,
                            cancel: task_cancel.clone(),
                        },
                    );
                }
                if pending_interrupt {
                    task_cancel.cancel();
                }
                let dispatch_run_id = run_id.to_string();
                let delegation_policy = self.config.delegation_policy;
                let waived_dependency_ids = match dependency_states.get(&task.spec.id) {
                    Some(DagDependencyState::Satisfied {
                        waived_dependency_ids,
                    }) => waived_dependency_ids.clone(),
                    _ => Vec::new(),
                };
                let context = match TaskSubagentContext::from_claim(
                    dispatch_run_id.clone(),
                    task.spec.id.clone(),
                    claim.clone(),
                    task_cancel.clone(),
                ) {
                    Ok(context) => context
                        .with_delegation_policy(delegation_policy)
                        .with_waived_dependencies(waived_dependency_ids),
                    Err(error) => {
                        let message = format!("invalid exact task context: {error}");
                        match self
                            .controller
                            .abandon_claim(
                                run_id,
                                &claim,
                                &task,
                                RuntimeClaimAbandonment::Failed {
                                    error: message.clone(),
                                },
                            )
                            .await
                        {
                            Ok(_) => {
                                self.cleanup_attempt_control(run_id, &task, &claim).await;
                            }
                            Err(settlement_error) => wave_errors.push(format!(
                                "invalid context abandonment outcome is unknown: {settlement_error}"
                            )),
                        }
                        wave_errors.push(message);
                        continue;
                    }
                };
                if let Err(error) = self.controller.reserve_attempt_control(&context).await {
                    let message = format!("exact attempt control reservation failed: {error}");
                    match self
                        .controller
                        .abandon_claim(
                            run_id,
                            &claim,
                            &task,
                            RuntimeClaimAbandonment::Failed {
                                error: message.clone(),
                            },
                        )
                        .await
                    {
                        Ok(_) => {
                            self.cleanup_attempt_control(run_id, &task, &claim).await;
                        }
                        Err(settlement_error) => {
                            wave_errors.push(format!(
                                "reservation failure abandonment outcome is unknown: {settlement_error}"
                            ));
                        }
                    }
                    wave_errors.push(message);
                    continue;
                }
                outstanding_claims.insert(claim_id.clone(), (task.clone(), claim.clone()));
                let abort_execution_id = execution_id.clone();
                let supervisor_cancel = task_cancel.clone();
                let abort_handle = join_set.spawn(async move {
                    let dispatch = if let Some(admission) = shared_admission {
                        let lease = tokio::select! {
                            biased;
                            _ = task_cancel.cancelled() => None,
                            lease = admission.issue_wait(format!("runtime:{dispatch_run_id}:{claim_id}")) => Some(
                                lease.map_err(|error| ReactError::Other(format!("shared execution admission rejected task: {error}")))
                            ),
                        };
                        match lease {
                            Some(Ok(lease)) => {
                                let result = controller.dispatch_task(context, task).await;
                                drop(lease);
                                result
                            }
                            Some(Err(error)) => Err(error),
                            None => controller.dispatch_task(context, task).await,
                        }
                    } else {
                        let permit = tokio::select! {
                            biased;
                            _ = task_cancel.cancelled() => None,
                            permit = semaphore.acquire_owned() => Some(permit),
                        };
                        match permit {
                            Some(Ok(permit)) => {
                                let result = controller.dispatch_task(context, task).await;
                                drop(permit);
                                result
                            }
                            Some(Err(error)) => Err(ReactError::Other(format!(
                                "Subagent semaphore closed: {error}"
                            ))),
                            None => controller.dispatch_task(context, task).await,
                        }
                    };
                    (claim_id, dispatch)
                });
                if let Ok(mut task_ids) = self.attempt_task_ids.lock() {
                    task_ids.insert(abort_handle.id(), abort_execution_id.clone());
                }
                if let Ok(mut handles) = self.attempt_abort_handles.lock() {
                    handles.insert(abort_execution_id.clone(), abort_handle);
                }
                if supervisor_cancel.is_cancelled() {
                    self.schedule_attempt_abort(abort_execution_id, true);
                }
            }

            let mut wave_results = Vec::new();
            let mut cancellation_observed = false;
            let cancellation_grace = tokio::time::sleep(Duration::ZERO);
            tokio::pin!(cancellation_grace);
            while !join_set.is_empty() {
                tokio::select! {
                    biased;
                    joined = join_set.join_next_with_id() => {
                        match joined {
                            Some(Ok((task_id, result))) => {
                                if let Some(execution_id) = self
                                    .attempt_task_ids
                                    .lock()
                                    .ok()
                                    .and_then(|mut task_ids| task_ids.remove(&task_id))
                                {
                                    self.mark_attempt_joined(&execution_id);
                                }
                                wave_results.push((
                                    result,
                                    !cancellation_observed && !cancel.is_cancelled(),
                                ));
                            }
                            Some(Err(error)) => {
                                let execution_id = self
                                    .attempt_task_ids
                                    .lock()
                                    .ok()
                                    .and_then(|mut task_ids| task_ids.remove(&error.id()));
                                if let Some(execution_id) = execution_id {
                                    self.mark_attempt_joined(&execution_id);
                                    let cancelled = self
                                        .attempt_cancellations
                                        .lock()
                                        .ok()
                                        .and_then(|controls| controls.get(&execution_id).cloned())
                                        .is_some_and(|token| token.is_cancelled());
                                    let claim_id = cancelled.then(|| {
                                        outstanding_claims
                                            .iter()
                                            .find(|(_, (task, claim))| {
                                                claim.execution_id(run_id, &task.spec.id)
                                                    == execution_id
                                            })
                                            .map(|(claim_id, _)| claim_id.clone())
                                    }).flatten();
                                    if error.is_cancelled() && let Some(claim_id) = claim_id {
                                        wave_results.push((
                                            (
                                                claim_id,
                                                Err(ReactError::Agent(Box::new(
                                                    echo_core::error::AgentError::Cancelled(
                                                        "exact attempt aborted after cancellation grace period".to_string(),
                                                    ),
                                                ))),
                                            ),
                                            false,
                                        ));
                                    } else {
                                        wave_errors.push(format!(
                                            "Subagent dispatch task '{execution_id}' failed to join: {error}"
                                        ));
                                    }
                                } else {
                                    wave_errors.push(format!(
                                        "Subagent dispatch task failed to join: {error}"
                                    ));
                                }
                            }
                            None => {}
                        }
                    }
                    _ = cancel.cancelled(), if !cancellation_observed => {
                        cancellation_observed = true;
                        cancellation_grace.as_mut().reset(
                            tokio::time::Instant::now() + self.config.cancellation_grace_period,
                        );
                    }
                    _ = &mut cancellation_grace, if cancellation_observed => {
                        join_set.abort_all();
                        while let Some(joined) = join_set.join_next_with_id().await {
                            match joined {
                                Ok((task_id, result)) => {
                                    if let Some(execution_id) = self
                                        .attempt_task_ids
                                        .lock()
                                        .ok()
                                        .and_then(|mut task_ids| task_ids.remove(&task_id))
                                    {
                                        self.mark_attempt_joined(&execution_id);
                                    }
                                    wave_results.push((result, false));
                                }
                                Err(error) if error.is_cancelled() => {
                                    if let Some(execution_id) = self
                                        .attempt_task_ids
                                        .lock()
                                        .ok()
                                        .and_then(|mut task_ids| task_ids.remove(&error.id()))
                                    {
                                        self.mark_attempt_joined(&execution_id);
                                    }
                                }
                                Err(error) => {
                                    if let Some(execution_id) = self
                                        .attempt_task_ids
                                        .lock()
                                        .ok()
                                        .and_then(|mut task_ids| task_ids.remove(&error.id()))
                                    {
                                        self.mark_attempt_joined(&execution_id);
                                    }
                                    wave_errors.push(format!(
                                        "Subagent dispatch task failed to join: {error}"
                                    ));
                                }
                            }
                        }
                        break;
                    }
                }
            }

            /*
            已 join 的尝试在其持久化 claim CAS
            被确认结算或被取代之前仍可寻址，之后才进行清理。
            */

            let mut interruption_policy_failed = false;
            let mut interruption_is_durable = false;
            let mut interruption = if cancellation_observed || cancel.is_cancelled() {
                match self.controller.interruption_disposition(run_id).await {
                    Ok(disposition) => Some(disposition),
                    Err(error) => {
                        interruption_policy_failed = true;
                        wave_errors.push(format!(
                            "failed to resolve runtime interruption disposition: {error}"
                        ));
                        Some(RuntimeInterruptionDisposition::Paused {
                            reason: format!(
                                "interruption policy unavailable; preserved for resume: {error}"
                            ),
                        })
                    }
                }
            } else {
                None
            };
            let mut pending_outcome = None;
            for ((claim_id, dispatch), completed_before_interruption) in wave_results {
                let Some((task, claim)) = outstanding_claims.remove(&claim_id) else {
                    wave_errors.push(format!("dispatch returned unknown claim '{claim_id}'"));
                    continue;
                };
                if interruption.is_some() && !completed_before_interruption && dispatch.is_err() {
                    let disposition = interruption
                        .clone()
                        .ok_or_else(|| ReactError::Other("interruption disappeared".to_string()))?;
                    match self
                        .settle_abandonment(
                            run_id,
                            &claim,
                            &task,
                            RuntimeClaimAbandonment::Interrupted { disposition },
                        )
                        .await
                    {
                        Ok(_) => {
                            self.cleanup_attempt_control(run_id, &task, &claim).await;
                        }
                        Err(error) => {
                            self.mark_attempt_authority_unknown(
                                &claim.execution_id(run_id, &task.spec.id),
                                error.to_string(),
                            );
                            wave_errors.push(error.to_string());
                        }
                    }
                    continue;
                }
                let request = match self
                    .controller
                    .resolve_dispatch(run_id, claim.clone(), task.clone(), dispatch)
                    .await
                {
                    Ok(resolution) => resolution,
                    Err(error) => {
                        let message = error.to_string();
                        match self
                            .controller
                            .claim_is_current(run_id, &task.spec.id, &claim)
                            .await
                        {
                            Ok(false) => {
                                self.settle_attempt_projection(
                                    &claim.execution_id(run_id, &task.spec.id),
                                    "superseded",
                                );
                                self.cleanup_attempt_control(run_id, &task, &claim).await;
                                continue;
                            }
                            Ok(true) => match self
                                .settle_abandonment(
                                    run_id,
                                    &claim,
                                    &task,
                                    RuntimeClaimAbandonment::Failed {
                                        error: message.clone(),
                                    },
                                )
                                .await
                            {
                                Ok(_) => {
                                    self.cleanup_attempt_control(run_id, &task, &claim).await;
                                }
                                Err(abandon_error) => {
                                    self.mark_attempt_authority_unknown(
                                        &claim.execution_id(run_id, &task.spec.id),
                                        abandon_error.to_string(),
                                    );
                                    wave_errors.push(abandon_error.to_string());
                                }
                            },
                            Err(lookup_error) => {
                                let lookup_message = format!(
                                    "runtime settlement outcome is unknown and claim lookup failed: {lookup_error}"
                                );
                                self.mark_attempt_authority_unknown(
                                    &claim.execution_id(run_id, &task.spec.id),
                                    lookup_message.clone(),
                                );
                                wave_errors.push(lookup_message);
                            }
                        }
                        wave_errors.push(message);
                        continue;
                    }
                };
                let resolution = match self
                    .settle_dispatch_request(run_id, &claim, &task, request)
                    .await
                {
                    Ok(resolution) => resolution,
                    Err(error) => {
                        let message = error.to_string();
                        match self
                            .controller
                            .claim_is_current(run_id, &task.spec.id, &claim)
                            .await
                        {
                            Ok(false) => {
                                self.settle_attempt_projection(
                                    &claim.execution_id(run_id, &task.spec.id),
                                    "superseded",
                                );
                                self.cleanup_attempt_control(run_id, &task, &claim).await;
                                continue;
                            }
                            Ok(true) => match self
                                .settle_abandonment(
                                    run_id,
                                    &claim,
                                    &task,
                                    RuntimeClaimAbandonment::Failed {
                                        error: message.clone(),
                                    },
                                )
                                .await
                            {
                                Ok(_) => {
                                    self.cleanup_attempt_control(run_id, &task, &claim).await;
                                }
                                Err(abandon_error) => {
                                    self.mark_attempt_authority_unknown(
                                        &claim.execution_id(run_id, &task.spec.id),
                                        abandon_error.to_string(),
                                    );
                                    wave_errors.push(abandon_error.to_string());
                                }
                            },
                            Err(lookup_error) => {
                                let lookup_message = format!(
                                    "runtime settlement outcome is unknown and claim lookup failed: {lookup_error}"
                                );
                                self.mark_attempt_authority_unknown(
                                    &claim.execution_id(run_id, &task.spec.id),
                                    lookup_message.clone(),
                                );
                                wave_errors.push(lookup_message);
                            }
                        }
                        wave_errors.push(message);
                        continue;
                    }
                };
                self.cleanup_attempt_control(run_id, &task, &claim).await;
                match resolution {
                    RuntimeTaskResolution::Completed
                    | RuntimeTaskResolution::Pending
                    | RuntimeTaskResolution::Skipped
                    | RuntimeTaskResolution::Superseded => {}
                    RuntimeTaskResolution::Failed { error }
                    | RuntimeTaskResolution::TimedOut { error } => {
                        failure_errors.entry(task.spec.id).or_insert(error);
                    }
                    RuntimeTaskResolution::Blocked { error, disposition } => {
                        if pending_outcome.is_none() {
                            pending_outcome = Some(stop_outcome(disposition, task.spec.id, error));
                        }
                    }
                    RuntimeTaskResolution::Cancelled => {
                        interruption = prioritize_interruption(
                            interruption,
                            Some(RuntimeInterruptionDisposition::Cancelled),
                        );
                        interruption_is_durable = true;
                    }
                }
            }

            if cancel.is_cancelled() && !interruption_policy_failed {
                match self.controller.interruption_disposition(run_id).await {
                    Ok(disposition) => {
                        interruption = prioritize_interruption(interruption, Some(disposition));
                    }
                    Err(error) => {
                        interruption_policy_failed = true;
                        wave_errors.push(format!(
                            "failed to resolve runtime interruption disposition at wave boundary: {error}"
                        ));
                        interruption = prioritize_interruption(
                            interruption,
                            Some(RuntimeInterruptionDisposition::Paused {
                                reason: format!(
                                    "interruption policy unavailable; preserved for resume: {error}"
                                ),
                            }),
                        );
                    }
                }
            }

            for (_claim_id, (task, claim)) in outstanding_claims {
                let abandonment = if let Some(disposition) = interruption.clone() {
                    RuntimeClaimAbandonment::Interrupted { disposition }
                } else {
                    RuntimeClaimAbandonment::Failed {
                        error: wave_errors
                            .first()
                            .cloned()
                            .unwrap_or_else(|| "dispatch ended without a result".to_string()),
                    }
                };
                match self
                    .settle_abandonment(run_id, &claim, &task, abandonment)
                    .await
                {
                    Ok(_) => {
                        self.cleanup_attempt_control(run_id, &task, &claim).await;
                    }
                    Err(error) => {
                        self.mark_attempt_authority_unknown(
                            &claim.execution_id(run_id, &task.spec.id),
                            error.to_string(),
                        );
                        wave_errors.push(error.to_string());
                    }
                }
            }

            if interruption_policy_failed {
                let cleanup_disposition = interruption.clone().unwrap_or_else(|| {
                    RuntimeInterruptionDisposition::Paused {
                        reason: "interruption policy unavailable; preserved for resume".to_string(),
                    }
                });
                match self
                    .controller
                    .settle_interruption(run_id, snapshot.revision, cleanup_disposition)
                    .await
                {
                    Ok(RuntimeInterruptionSettlementOutcome::Settled(_)) => {}
                    Ok(RuntimeInterruptionSettlementOutcome::ReloadSnapshot) => wave_errors.push(
                        "runtime interruption cleanup lost its expected revision".to_string(),
                    ),
                    Err(error) => wave_errors.push(format!(
                        "runtime interruption cleanup failed after policy error: {error}"
                    )),
                }
            }

            if !wave_errors.is_empty() {
                return Err(ReactError::Other(wave_errors.join("; ")));
            }

            /*
            先结算整个波次再响应停止请求，
            确保已完成的兄弟任务在恢复后不会被重放。
            */
            if let Some(disposition) = interruption {
                match self
                    .settle_interruption_boundary(
                        run_id,
                        snapshot.revision,
                        disposition,
                        !interruption_is_durable,
                    )
                    .await?
                {
                    InterruptionBoundary::Outcome(outcome) => return Ok(outcome),
                    InterruptionBoundary::ReloadSnapshot
                    | InterruptionBoundary::NoUnfinishedWork => {}
                }
            }
            if let Some(outcome) = pending_outcome {
                return Ok(outcome);
            }
        }
    }

    /// 结算中断边界：由控制器持久化中断处置，并转换为对应的执行结果。
    async fn settle_interruption_boundary(
        &self,
        run_id: &str,
        expected_revision: u64,
        disposition: RuntimeInterruptionDisposition,
        suppress_noop_outcome: bool,
    ) -> Result<InterruptionBoundary> {
        match self
            .controller
            .settle_interruption(run_id, expected_revision, disposition.clone())
            .await?
        {
            RuntimeInterruptionSettlementOutcome::ReloadSnapshot => {
                Ok(InterruptionBoundary::ReloadSnapshot)
            }
            RuntimeInterruptionSettlementOutcome::Settled(receipt) => {
                if receipt.disposition != disposition {
                    return Err(ReactError::Other(
                        "runtime interruption receipt changed the requested disposition"
                            .to_string(),
                    ));
                }
                if suppress_noop_outcome
                    && receipt.interrupted_task_ids.is_empty()
                    && receipt.pending_task_ids.is_empty()
                {
                    return Ok(InterruptionBoundary::NoUnfinishedWork);
                }
                Ok(InterruptionBoundary::Outcome(match disposition {
                    RuntimeInterruptionDisposition::Cancelled => RuntimeDagOutcome::Cancelled,
                    RuntimeInterruptionDisposition::Paused { reason } => {
                        RuntimeDagOutcome::Paused {
                            task_id: None,
                            reason,
                        }
                    }
                }))
            }
        }
    }

    /// 持久化任务结算请求，并校验回执与请求一致且 claim 已不再活跃。
    async fn settle_dispatch_request(
        &self,
        run_id: &str,
        claim: &TaskClaim,
        task: &Task,
        request: RuntimeTaskResolutionRequest,
    ) -> Result<RuntimeTaskResolution> {
        let resolution = self
            .controller
            .settle_resolution(run_id, claim, task, request.clone())
            .await?;
        if !resolution_matches_request(&request, &resolution) {
            return Err(ReactError::Other(format!(
                "runtime settlement receipt {resolution:?} does not match request {request:?}"
            )));
        }
        if self
            .controller
            .claim_is_current(run_id, &task.spec.id, claim)
            .await?
        {
            return Err(ReactError::Other(format!(
                "runtime settlement receipt {resolution:?} left claim '{}' active",
                claim.claim_id
            )));
        }
        self.settle_attempt_projection(
            &claim.execution_id(run_id, &task.spec.id),
            runtime_resolution_status(&resolution),
        );
        Ok(resolution)
    }

    /// 放弃 claim 并持久化放弃结果，校验 claim 已不再活跃。
    async fn settle_abandonment(
        &self,
        run_id: &str,
        claim: &TaskClaim,
        task: &Task,
        abandonment: RuntimeClaimAbandonment,
    ) -> Result<RuntimeTaskSettlementOutcome> {
        let intended_status = abandonment_status(&abandonment);
        let settlement = self
            .controller
            .abandon_claim(run_id, claim, task, abandonment)
            .await?;
        if self
            .controller
            .claim_is_current(run_id, &task.spec.id, claim)
            .await?
        {
            return Err(ReactError::Other(format!(
                "runtime claim abandonment {settlement:?} left claim '{}' active",
                claim.claim_id
            )));
        }
        let status = match settlement {
            RuntimeTaskSettlementOutcome::Settled => intended_status,
            RuntimeTaskSettlementOutcome::Superseded => "superseded",
        };
        self.settle_attempt_projection(&claim.execution_id(run_id, &task.spec.id), status);
        Ok(settlement)
    }

    /// 持久化结算后清理尝试控制状态，失败时记录告警以便重试。
    async fn cleanup_attempt_control(&self, run_id: &str, task: &Task, claim: &TaskClaim) {
        let execution_id = claim.execution_id(run_id, &task.spec.id);
        let receipt = self
            .controller
            .retire_attempt_control(run_id, task, claim)
            .await;
        self.observe_attempt_control_cleanup(run_id, &task.spec.id, claim, receipt.clone());
        self.retire_local_attempt_control(&execution_id);
        match receipt {
            RuntimeAttemptControlCleanupReceipt::RetryableFailure { ref error } => {
                tracing::warn!(
                    run_id,
                    task_id = %task.spec.id,
                    claim_id = %claim.claim_id,
                    error = %error,
                    "exact attempt control cleanup requires retry after durable settlement"
                );
            }
            RuntimeAttemptControlCleanupReceipt::Retired
            | RuntimeAttemptControlCleanupReceipt::NotFound
            | RuntimeAttemptControlCleanupReceipt::AlreadyConsumed => {}
        }
    }
}

/// 将任务结算结果映射为状态字符串。
fn runtime_resolution_status(resolution: &RuntimeTaskResolution) -> &'static str {
    match resolution {
        RuntimeTaskResolution::Completed => "completed",
        RuntimeTaskResolution::Pending => "pending",
        RuntimeTaskResolution::Skipped => "skipped",
        RuntimeTaskResolution::Failed { .. } => "failed",
        RuntimeTaskResolution::TimedOut { .. } => "timed_out",
        RuntimeTaskResolution::Blocked { .. } => "blocked",
        RuntimeTaskResolution::Cancelled => "cancelled",
        RuntimeTaskResolution::Superseded => "superseded",
    }
}

/// 将 claim 放弃原因映射为状态字符串。
fn abandonment_status(abandonment: &RuntimeClaimAbandonment) -> &'static str {
    match abandonment {
        RuntimeClaimAbandonment::Interrupted {
            disposition: RuntimeInterruptionDisposition::Cancelled,
        } => "cancelled",
        RuntimeClaimAbandonment::Interrupted {
            disposition: RuntimeInterruptionDisposition::Paused { .. },
        } => "paused",
        RuntimeClaimAbandonment::Failed { .. } => "failed",
    }
}

/// 判断结算回执是否与结算请求一致（Superseded 视为匹配）。
fn resolution_matches_request(
    request: &RuntimeTaskResolutionRequest,
    resolution: &RuntimeTaskResolution,
) -> bool {
    if resolution == &RuntimeTaskResolution::Superseded {
        return true;
    }
    match (request, resolution) {
        (RuntimeTaskResolutionRequest::Completed, RuntimeTaskResolution::Completed)
        | (RuntimeTaskResolutionRequest::Skipped, RuntimeTaskResolution::Skipped)
        | (RuntimeTaskResolutionRequest::Cancelled, RuntimeTaskResolution::Cancelled) => true,
        (RuntimeTaskResolutionRequest::Requeue { .. }, RuntimeTaskResolution::Pending) => true,
        (
            RuntimeTaskResolutionRequest::Requeue {
                error: requested,
                exhaustion: RuntimeRetryExhaustion::Failed,
                ..
            },
            RuntimeTaskResolution::Failed { error: settled },
        )
        | (
            RuntimeTaskResolutionRequest::Requeue {
                error: requested,
                exhaustion: RuntimeRetryExhaustion::TimedOut,
                ..
            },
            RuntimeTaskResolution::TimedOut { error: settled },
        )
        | (
            RuntimeTaskResolutionRequest::Failed { error: requested },
            RuntimeTaskResolution::Failed { error: settled },
        )
        | (
            RuntimeTaskResolutionRequest::TimedOut { error: requested },
            RuntimeTaskResolution::TimedOut { error: settled },
        ) => requested == settled,
        (
            RuntimeTaskResolutionRequest::Blocked {
                error: requested_error,
                disposition: requested_disposition,
            },
            RuntimeTaskResolution::Blocked {
                error: settled_error,
                disposition: settled_disposition,
            },
        ) => requested_error == settled_error && requested_disposition == settled_disposition,
        _ => false,
    }
}

/**
按明确的优先级合并持久化与请求中的中断信号：
Cancelled 优先于 Paused，且较早的 Paused 原因保持不变。
*/
fn prioritize_interruption(
    current: Option<RuntimeInterruptionDisposition>,
    incoming: Option<RuntimeInterruptionDisposition>,
) -> Option<RuntimeInterruptionDisposition> {
    match (current, incoming) {
        (Some(RuntimeInterruptionDisposition::Cancelled), _)
        | (_, Some(RuntimeInterruptionDisposition::Cancelled)) => {
            Some(RuntimeInterruptionDisposition::Cancelled)
        }
        (Some(paused @ RuntimeInterruptionDisposition::Paused { .. }), _) => Some(paused),
        (None, disposition) => disposition,
    }
}

/// 提取持久化任务状态中携带的错误信息。
fn persisted_status_error(status: &TaskStatus) -> Option<String> {
    match status {
        TaskStatus::Failed(error) | TaskStatus::Blocked(error) | TaskStatus::Paused(error) => {
            Some(error.clone())
        }
        TaskStatus::TimedOut { error } => Some(error.clone()),
        TaskStatus::Retrying { last_error, .. } => Some(last_error.clone()),
        TaskStatus::Pending
        | TaskStatus::Running
        | TaskStatus::Completed
        | TaskStatus::Skipped
        | TaskStatus::Cancelled => None,
    }
}

/// 校验控制器选择的波次：非空、只包含就绪任务且不重复。
fn validate_selected_wave(
    ready_task_ids: &[TaskId],
    selected_task_ids: Vec<TaskId>,
) -> Result<Vec<TaskId>> {
    if selected_task_ids.is_empty() {
        return Err(ReactError::Other(
            "runtime DAG controller selected an empty wave from a non-empty frontier".to_string(),
        ));
    }

    let ready: HashSet<&str> = ready_task_ids.iter().map(String::as_str).collect();
    let mut seen = HashSet::new();
    for selected in &selected_task_ids {
        if !ready.contains(selected.as_str()) {
            return Err(ReactError::Other(format!(
                "runtime DAG controller selected non-ready task '{selected}'"
            )));
        }
        if !seen.insert(selected.as_str()) {
            return Err(ReactError::Other(format!(
                "runtime DAG controller selected task '{selected}' more than once"
            )));
        }
    }
    Ok(selected_task_ids)
}

/// 根据停止处置将任务错误转换为失败或暂停结果。
fn stop_outcome(
    disposition: RuntimeStopDisposition,
    failed_task_id: TaskId,
    error: String,
) -> RuntimeDagOutcome {
    match disposition {
        RuntimeStopDisposition::Fail => RuntimeDagOutcome::Failed {
            failed_task_id,
            error,
        },
        RuntimeStopDisposition::Pause => RuntimeDagOutcome::Paused {
            task_id: Some(failed_task_id),
            reason: error,
        },
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;
    use crate::tasks::{RuntimeTaskAttemptInterruptError, TaskStatus};

    /// 脚本化的 RuntimeDagController 测试替身，通过字段注入各类行为。
    #[derive(Default)]
    struct ScriptedController {
        /// 当前计划快照
        snapshot: Mutex<Option<RuntimePlanSnapshot>>,
        /// 按分派顺序记录的任务 ID
        order: Mutex<Vec<TaskId>>,
        /// 分派时应返回失败的任务及其错误
        fail: Mutex<HashMap<TaskId, String>>,
        /// 分派成功后应解析为阻塞的任务及其错误与停止处置
        blocked: Mutex<HashMap<TaskId, (String, RuntimeStopDisposition)>>,
        /// 分派后等待取消信号再返回的任务
        wait_for_cancel: Mutex<HashSet<TaskId>>,
        /// 分派后永不返回（忽略取消）的任务
        ignore_cancel: Mutex<HashSet<TaskId>>,
        /// 该任务结算后向快照插入新任务并提升修订号
        insert_after: Mutex<Option<TaskId>>,
        /// 为 true 时下一次 claim 返回 ReloadSnapshot
        reload_claim_once: Mutex<bool>,
        /// interruption_disposition 返回的中断处置
        interruption: Mutex<RuntimeInterruptionDisposition>,
        /// interruption_disposition 返回的脚本化错误
        interruption_error: Mutex<Option<String>>,
        /// 分派时等待的屏障，用于让同一波次任务并发到达
        dispatch_barrier: Mutex<Option<Arc<tokio::sync::Barrier>>>,
        /// 分派完成后要触发的取消令牌
        cancel_after_dispatch: Mutex<HashMap<TaskId, CancellationToken>>,
        /// claim 成功后发出的通知
        claim_ready: Mutex<Option<Arc<tokio::sync::Notify>>>,
        /// 已预留尝试控制的 execution_id 列表
        reserved: Mutex<Vec<String>>,
        /// 已退役尝试控制的 execution_id 列表
        retired: Mutex<Vec<String>>,
        /// claim_is_current 被调用的次数
        current_check_count: Mutex<usize>,
        /// 第 N 次 claim_is_current 调用时返回一次错误
        current_check_error_at: Mutex<Option<usize>>,
        /// 第 N 次起 claim_is_current 返回 false（claim 陈旧）
        stale_after_current_check: Mutex<Option<usize>>,
        /// 陈旧检查时是否将任务直接标记为已完成（模拟被取代）
        supersede_on_stale_check: Mutex<bool>,
        /// 下一次结算返回的一次性错误
        settlement_error_once: Mutex<Option<String>>,
        /// 为 true 时结算已提交但返回错误（模拟响应丢失）
        settle_response_error_once: Mutex<bool>,
        /// 预留尝试控制时返回的错误
        reservation_error: Mutex<Option<String>>,
        /// 预留尝试控制前的延迟
        reservation_delay: Mutex<Option<Duration>>,
        /// 进入结算时发出的通知
        settlement_entered: Mutex<Option<Arc<tokio::sync::Notify>>>,
        /// 结算继续前需等待的通知
        settlement_release: Mutex<Option<Arc<tokio::sync::Notify>>>,
        /// 进入协调时发出的通知
        reconciliation_entered: Mutex<Option<Arc<tokio::sync::Notify>>>,
        /// 协调继续前需等待的通知
        reconciliation_release: Mutex<Option<Arc<tokio::sync::Notify>>>,
        /// 协调尝试控制时返回的错误
        reconciliation_error: Mutex<Option<String>>,
        /// 为 true 时下一次放弃 claim 返回错误（结果未知）
        abandonment_error_once: Mutex<bool>,
        /// 退役尝试控制时返回的回执（默认 Retired）
        cleanup_receipt: Mutex<Option<RuntimeAttemptControlCleanupReceipt>>,
    }

    impl ScriptedController {
        /// 用给定任务构造修订号为 1 的脚本化控制器。
        fn with_tasks(tasks: Vec<Task>) -> Self {
            Self {
                snapshot: Mutex::new(Some(RuntimePlanSnapshot { revision: 1, tasks })),
                ..Self::default()
            }
        }

        /// 返回当前快照中各任务的状态。
        fn statuses(&self) -> HashMap<TaskId, TaskStatus> {
            self.snapshot
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .as_ref()
                .map(|snapshot| {
                    snapshot
                        .tasks
                        .iter()
                        .map(|task| (task.spec.id.clone(), task.execution.status.clone()))
                        .collect()
                })
                .unwrap_or_default()
        }
    }

    #[async_trait]
    impl RuntimeDagController for ScriptedController {
        /// 分派结果为任务 ID。
        type DispatchOutput = TaskId;

        /// 返回当前快照的克隆。
        async fn load_snapshot(&self, _run_id: &str) -> Result<RuntimePlanSnapshot> {
            self.snapshot
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .clone()
                .ok_or_else(|| ReactError::Other("missing snapshot".to_string()))
        }

        /// 按脚本 claim 任务，可模拟一次 ReloadSnapshot。
        async fn claim_task(
            &self,
            _run_id: &str,
            task: &Task,
            expected_revision: u64,
        ) -> Result<RuntimeTaskClaimOutcome> {
            let mut reload_claim_once = self
                .reload_claim_once
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if *reload_claim_once {
                *reload_claim_once = false;
                return Ok(RuntimeTaskClaimOutcome::ReloadSnapshot);
            }
            drop(reload_claim_once);
            let mut snapshot = self
                .snapshot
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let snapshot = snapshot
                .as_mut()
                .ok_or_else(|| ReactError::Other("missing snapshot".to_string()))?;
            let result = super::super::runtime_service::claim_runtime_task(
                snapshot,
                task,
                expected_revision,
            )
            .map_err(|error| ReactError::Other(error.to_string()))?;
            if let Some(notify) = self
                .claim_ready
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .as_ref()
            {
                notify.notify_one();
            }
            Ok(result)
        }

        /// 检查 claim 是否仍为当前，可按脚本模拟查询错误或陈旧。
        async fn claim_is_current(
            &self,
            _run_id: &str,
            task_id: &str,
            claim: &TaskClaim,
        ) -> Result<bool> {
            let check_count = {
                let mut count = self
                    .current_check_count
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                *count = count.saturating_add(1);
                *count
            };
            let current_check_error = {
                let mut error_at = self
                    .current_check_error_at
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                if error_at.is_some_and(|error_at| check_count >= error_at) {
                    error_at.take()
                } else {
                    None
                }
            };
            if current_check_error.is_some() {
                return Err(ReactError::Other(
                    "scripted current-claim lookup unavailable".to_string(),
                ));
            }
            let stale = self
                .stale_after_current_check
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .is_some_and(|limit| check_count >= limit);
            if stale {
                if *self
                    .supersede_on_stale_check
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                {
                    let mut snapshot = self
                        .snapshot
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    if let Some(task) = snapshot.as_mut().and_then(|snapshot| {
                        snapshot
                            .tasks
                            .iter_mut()
                            .find(|task| task.spec.id == task_id)
                    }) {
                        task.execution.status = TaskStatus::Completed;
                        task.execution.claim = None;
                    }
                }
                return Ok(false);
            }
            let snapshot = self
                .snapshot
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let snapshot = snapshot
                .as_ref()
                .ok_or_else(|| ReactError::Other("missing snapshot".to_string()))?;
            super::super::runtime_service::runtime_claim_is_current(snapshot, task_id, claim)
                .map_err(|error| ReactError::Other(error.to_string()))
        }

        /// 按脚本预留尝试控制，可模拟延迟与错误。
        async fn reserve_attempt_control(&self, context: &TaskSubagentContext) -> Result<()> {
            let delay = *self
                .reservation_delay
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if let Some(delay) = delay {
                tokio::time::sleep(delay).await;
            }
            if let Some(error) = self
                .reservation_error
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .clone()
            {
                return Err(ReactError::Other(error));
            }
            let execution_id = context
                .execution_id()
                .ok_or_else(|| ReactError::Other("missing exact execution identity".to_string()))?;
            self.reserved
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .push(execution_id);
            Ok(())
        }

        /// 按脚本协调尝试控制，可通过通知暂停并模拟错误。
        async fn reconcile_attempt_control(
            &self,
            _run_id: &str,
            _current_execution_ids: &HashSet<String>,
        ) -> Result<()> {
            let entered = self
                .reconciliation_entered
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .clone();
            let release = self
                .reconciliation_release
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .clone();
            if let Some(entered) = entered {
                entered.notify_one();
            }
            if let Some(release) = release {
                release.notified().await;
            }
            if let Some(error) = self
                .reconciliation_error
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .clone()
            {
                return Err(ReactError::Other(error));
            }
            Ok(())
        }

        /// 记录退役的 execution_id 并返回脚本化回执。
        async fn retire_attempt_control(
            &self,
            run_id: &str,
            task: &Task,
            claim: &TaskClaim,
        ) -> RuntimeAttemptControlCleanupReceipt {
            self.retired
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .push(claim.execution_id(run_id, &task.spec.id));
            self.cleanup_receipt
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .clone()
                .unwrap_or(RuntimeAttemptControlCleanupReceipt::Retired)
        }

        /// 按脚本分派任务：可等待屏障、等待取消、永不返回或返回失败。
        async fn dispatch_task(
            &self,
            context: TaskSubagentContext,
            task: Task,
        ) -> Result<Self::DispatchOutput> {
            self.order
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .push(task.spec.id.clone());
            let barrier = self
                .dispatch_barrier
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .clone();
            if let Some(barrier) = barrier {
                barrier.wait().await;
            }
            let wait_for_cancel = self
                .wait_for_cancel
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .contains(&task.spec.id);
            if wait_for_cancel {
                context.cancellation_token().cancelled().await;
                return Err(ReactError::Agent(Box::new(
                    echo_core::error::AgentError::Cancelled("cancelled by test".to_string()),
                )));
            }
            let ignore_cancel = self
                .ignore_cancel
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .contains(&task.spec.id);
            if ignore_cancel {
                std::future::pending::<()>().await;
            }
            let failure = self
                .fail
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .get(&task.spec.id)
                .cloned();
            let task_id = task.spec.id.clone();
            let result = match failure {
                Some(error) => Err(ReactError::Other(error)),
                None => Ok(task_id.clone()),
            };
            if let Some(cancel) = self
                .cancel_after_dispatch
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .remove(&task_id)
            {
                cancel.cancel();
            }
            result
        }

        /// 将分派结果转换为结算请求。
        async fn resolve_dispatch(
            &self,
            _run_id: &str,
            _claim: TaskClaim,
            task: Task,
            dispatch: Result<Self::DispatchOutput>,
        ) -> Result<RuntimeTaskResolutionRequest> {
            let blocked = self
                .blocked
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .get(&task.spec.id)
                .cloned();
            Ok(match dispatch {
                Ok(_) if blocked.is_some() => {
                    let (error, disposition) = blocked.ok_or_else(|| {
                        ReactError::Other("scripted blocker disappeared".to_string())
                    })?;
                    RuntimeTaskResolutionRequest::Blocked { error, disposition }
                }
                Ok(_) => RuntimeTaskResolutionRequest::Completed,
                Err(ReactError::Agent(error))
                    if matches!(
                        error.as_ref(),
                        echo_core::error::AgentError::Cancelled(_)
                            | echo_core::error::AgentError::Interrupted
                    ) =>
                {
                    RuntimeTaskResolutionRequest::Cancelled
                }
                Err(error) => RuntimeTaskResolutionRequest::Failed {
                    error: error.to_string(),
                },
            })
        }

        /// 按脚本结算任务，可暂停、返回错误、插入新任务或模拟响应丢失。
        async fn settle_resolution(
            &self,
            _run_id: &str,
            claim: &TaskClaim,
            task: &Task,
            request: RuntimeTaskResolutionRequest,
        ) -> Result<RuntimeTaskResolution> {
            let settlement_entered = self
                .settlement_entered
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .clone();
            let settlement_release = self
                .settlement_release
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .clone();
            if let Some(entered) = settlement_entered {
                entered.notify_one();
            }
            if let Some(release) = settlement_release {
                release.notified().await;
            }
            if let Some(error) = self
                .settlement_error_once
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .take()
            {
                return Err(ReactError::Other(error));
            }
            let mut snapshot = self
                .snapshot
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let snapshot = snapshot
                .as_mut()
                .ok_or_else(|| ReactError::Other("missing snapshot".to_string()))?;
            let resolution = super::super::runtime_service::settle_runtime_resolution(
                snapshot,
                &task.spec.id,
                claim,
                request,
            )
            .map_err(|error| ReactError::Other(error.to_string()))?;
            let insert_after = self
                .insert_after
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .take();
            if resolution != RuntimeTaskResolution::Superseded
                && insert_after.as_deref() == Some(task.spec.id.as_str())
            {
                snapshot.revision = snapshot.revision.checked_add(1).ok_or_else(|| {
                    ReactError::Other("scripted snapshot revision overflowed".to_string())
                })?;
                snapshot.tasks.push(runtime_task(
                    "inserted",
                    TaskStatus::Pending,
                    &[task.spec.id.as_str()],
                ));
            }
            let mut response_error = self
                .settle_response_error_once
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if *response_error {
                *response_error = false;
                return Err(ReactError::Other(
                    "scripted lost settlement response".to_string(),
                ));
            }
            Ok(resolution)
        }

        /// 按放弃原因结算 claim，可模拟一次结果未知的错误。
        async fn abandon_claim(
            &self,
            _run_id: &str,
            claim: &TaskClaim,
            task: &Task,
            abandonment: RuntimeClaimAbandonment,
        ) -> Result<RuntimeTaskSettlementOutcome> {
            let mut abandonment_error = self
                .abandonment_error_once
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if *abandonment_error {
                *abandonment_error = false;
                return Err(ReactError::Other(
                    "scripted abandonment outcome unknown".to_string(),
                ));
            }
            drop(abandonment_error);
            let mut snapshot = self
                .snapshot
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let snapshot = snapshot
                .as_mut()
                .ok_or_else(|| ReactError::Other("missing snapshot".to_string()))?;
            let status = match abandonment {
                RuntimeClaimAbandonment::Interrupted { disposition } => match disposition {
                    RuntimeInterruptionDisposition::Cancelled => TaskStatus::Cancelled,
                    RuntimeInterruptionDisposition::Paused { reason } => TaskStatus::Paused(reason),
                },
                RuntimeClaimAbandonment::Failed { error } => TaskStatus::Failed(error),
            };
            super::super::runtime_service::settle_runtime_claim(
                snapshot,
                &task.spec.id,
                claim,
                status,
            )
            .map_err(|error| ReactError::Other(error.to_string()))
        }

        /// 返回脚本化的中断处置或错误。
        async fn interruption_disposition(
            &self,
            _run_id: &str,
        ) -> Result<RuntimeInterruptionDisposition> {
            if let Some(error) = self
                .interruption_error
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .clone()
            {
                return Err(ReactError::Other(error));
            }
            Ok(self
                .interruption
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .clone())
        }

        /// 在快照上结算中断处置。
        async fn settle_interruption(
            &self,
            _run_id: &str,
            expected_revision: u64,
            disposition: RuntimeInterruptionDisposition,
        ) -> Result<RuntimeInterruptionSettlementOutcome> {
            let mut snapshot = self
                .snapshot
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let snapshot = snapshot
                .as_mut()
                .ok_or_else(|| ReactError::Other("missing snapshot".to_string()))?;
            super::super::runtime_service::settle_runtime_interruption(
                snapshot,
                expected_revision,
                disposition,
            )
            .map_err(|error| ReactError::Other(error.to_string()))
        }
    }

    /// 构造测试用运行时任务。
    fn runtime_task(id: &str, status: TaskStatus, dependencies: &[&str]) -> Task {
        Task {
            spec: crate::tasks::TaskSpec {
                id: id.to_string(),
                title: id.to_string(),
                description: format!("execute {id}"),
                depends_on: dependencies
                    .iter()
                    .map(|dependency| dependency.to_string())
                    .collect(),
                max_retries: 1,
                extension: serde_json::Value::Null,
            },
            execution: crate::tasks::TaskExecution {
                task_id: id.to_string(),
                status,
                retry_count: 0,
                failure_fingerprint: None,
                claim: None,
            },
        }
    }

    /// 验证执行器遵循依赖顺序，并在安全点采纳新的修订。
    #[tokio::test]
    async fn executor_follows_dependencies_and_safe_point_revision() -> Result<()> {
        let controller = Arc::new(ScriptedController::with_tasks(vec![
            runtime_task("a", TaskStatus::Pending, &[]),
            runtime_task("b", TaskStatus::Pending, &["a"]),
        ]));
        *controller
            .insert_after
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some("a".to_string());
        let executor =
            RuntimeDagExecutor::new(controller.clone(), RuntimeTaskServiceConfig::default());

        let outcome = executor.execute("run", CancellationToken::new()).await?;

        assert_eq!(outcome, RuntimeDagOutcome::Completed);
        let order = controller
            .order
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();
        let a_position = order
            .iter()
            .position(|id| id == "a")
            .ok_or_else(|| ReactError::Other("task 'a' was not dispatched".to_string()))?;
        let b_position = order
            .iter()
            .position(|id| id == "b")
            .ok_or_else(|| ReactError::Other("task 'b' was not dispatched".to_string()))?;
        let inserted_position = order
            .iter()
            .position(|id| id == "inserted")
            .ok_or_else(|| ReactError::Other("revised task was not dispatched".to_string()))?;
        assert!(a_position < b_position);
        assert!(a_position < inserted_position);
        Ok(())
    }

    /// 验证精确尝试控制在分派前预留、结算后退役。
    #[tokio::test]
    async fn exact_control_is_reserved_before_dispatch_and_retired_after_settlement() -> Result<()>
    {
        let controller = Arc::new(ScriptedController::with_tasks(vec![runtime_task(
            "exact",
            TaskStatus::Pending,
            &[],
        )]));
        let service = super::super::RuntimeTaskService::new(
            controller.clone(),
            RuntimeTaskServiceConfig::default(),
        );

        assert_eq!(
            service
                .execute("exact-run", CancellationToken::new())
                .await?,
            RuntimeDagOutcome::Completed
        );
        let reserved = controller
            .reserved
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();
        let retired = controller
            .retired
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();
        assert_eq!(reserved.len(), 1);
        assert_eq!(retired, reserved);
        assert!(
            reserved
                .first()
                .is_some_and(|execution_id| execution_id.starts_with("exact-run:exact:1:1:"))
        );
        Ok(())
    }

    /// 验证结算响应丢失时重新加载已提交的权威状态，而不放弃 claim。
    #[tokio::test]
    async fn lost_settlement_response_reloads_committed_authority_without_abandonment() -> Result<()>
    {
        let controller = Arc::new(ScriptedController::with_tasks(vec![runtime_task(
            "committed",
            TaskStatus::Pending,
            &[],
        )]));
        *controller
            .settle_response_error_once
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = true;
        let service = super::super::RuntimeTaskService::new(
            controller.clone(),
            RuntimeTaskServiceConfig::default(),
        );

        assert_eq!(
            service
                .execute("lost-settlement", CancellationToken::new())
                .await?,
            RuntimeDagOutcome::Completed
        );
        assert_eq!(
            controller.statuses().get("committed"),
            Some(&TaskStatus::Completed)
        );
        Ok(())
    }

    /// 验证已 join 的尝试在持久化结算前仍可寻址。
    #[tokio::test]
    async fn joined_attempt_remains_addressable_until_durable_settlement() -> Result<()> {
        let controller = Arc::new(ScriptedController::with_tasks(vec![runtime_task(
            "joined",
            TaskStatus::Pending,
            &[],
        )]));
        let entered = Arc::new(tokio::sync::Notify::new());
        let release = Arc::new(tokio::sync::Notify::new());
        *controller
            .settlement_entered
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(entered.clone());
        *controller
            .settlement_release
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(release.clone());
        let service = Arc::new(super::super::RuntimeTaskService::new(
            controller.clone(),
            RuntimeTaskServiceConfig::default(),
        ));
        let execution = tokio::spawn({
            let service = Arc::clone(&service);
            async move {
                service
                    .execute("joined-settlement", CancellationToken::new())
                    .await
            }
        });
        tokio::time::timeout(Duration::from_secs(2), entered.notified())
            .await
            .map_err(|_| ReactError::Other("settlement barrier was not reached".to_string()))?;
        let claim = controller
            .snapshot
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .as_ref()
            .and_then(|snapshot| snapshot.tasks.first())
            .and_then(|task| task.execution.claim.clone())
            .ok_or_else(|| ReactError::Other("joined claim was not current".to_string()))?;
        let interrupt = tokio::spawn({
            let service = Arc::clone(&service);
            let claim = claim.clone();
            async move {
                service
                    .request_attempt_interrupt("joined-settlement", "joined", &claim)
                    .await
            }
        });
        tokio::task::yield_now().await;
        assert!(!interrupt.is_finished());
        release.notify_one();
        assert_eq!(
            tokio::time::timeout(Duration::from_secs(2), execution)
                .await
                .map_err(|_| ReactError::Other("joined settlement did not finish".to_string()))?
                .map_err(|error| ReactError::Other(format!("runtime task panicked: {error}")))??,
            RuntimeDagOutcome::Completed
        );
        let receipt = tokio::time::timeout(Duration::from_secs(2), interrupt)
            .await
            .map_err(|_| ReactError::Other("joined interrupt receipt did not settle".to_string()))?
            .map_err(|error| ReactError::Other(format!("interrupt task panicked: {error}")))?
            .map_err(ReactError::from)?;
        assert!(!receipt.requested);
        assert_eq!(
            receipt.disposition,
            RuntimeTaskAttemptInterruptDisposition::AlreadySettled {
                status: "completed".to_string(),
            }
        );
        assert_eq!(service.pending_attempt_interrupt_count(), 0);
        Ok(())
    }

    /// 验证已 join 的尝试报告权威未知，恢复后结算观察结果。
    #[tokio::test]
    async fn joined_attempt_reports_unknown_authority_and_recovery_settles_observation()
    -> Result<()> {
        let controller = Arc::new(ScriptedController::with_tasks(vec![runtime_task(
            "joined",
            TaskStatus::Pending,
            &[],
        )]));
        let entered = Arc::new(tokio::sync::Notify::new());
        let release = Arc::new(tokio::sync::Notify::new());
        *controller
            .settlement_entered
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(entered.clone());
        *controller
            .settlement_release
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(release.clone());
        *controller
            .settlement_error_once
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) =
            Some("scripted durable settlement unavailable".to_string());
        *controller
            .current_check_error_at
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(3);
        let service = Arc::new(super::super::RuntimeTaskService::new(
            controller.clone(),
            RuntimeTaskServiceConfig::default(),
        ));
        let execution = tokio::spawn({
            let service = Arc::clone(&service);
            async move {
                service
                    .execute("joined-unknown", CancellationToken::new())
                    .await
            }
        });
        tokio::time::timeout(Duration::from_secs(2), entered.notified())
            .await
            .map_err(|_| ReactError::Other("settlement barrier was not reached".to_string()))?;
        let claim = controller
            .snapshot
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .as_ref()
            .and_then(|snapshot| snapshot.tasks.first())
            .and_then(|task| task.execution.claim.clone())
            .ok_or_else(|| ReactError::Other("joined claim was not current".to_string()))?;
        let execution_id = claim.execution_id("joined-unknown", "joined");
        let interrupt = tokio::spawn({
            let service = Arc::clone(&service);
            let claim = claim.clone();
            async move {
                service
                    .request_attempt_interrupt("joined-unknown", "joined", &claim)
                    .await
            }
        });
        tokio::task::yield_now().await;
        assert!(!interrupt.is_finished());
        release.notify_one();

        let execution_error = tokio::time::timeout(Duration::from_secs(2), execution)
            .await
            .map_err(|_| ReactError::Other("unknown settlement did not finish".to_string()))?
            .map_err(|error| ReactError::Other(format!("runtime task panicked: {error}")))?
            .err()
            .ok_or_else(|| {
                ReactError::Other("unknown settlement unexpectedly succeeded".to_string())
            })?;
        assert!(execution_error.to_string().contains("claim lookup failed"));
        let interrupt_error = tokio::time::timeout(Duration::from_secs(2), interrupt)
            .await
            .map_err(|_| ReactError::Other("unknown interrupt remained pending".to_string()))?
            .map_err(|error| ReactError::Other(format!("interrupt task panicked: {error}")))?
            .err()
            .ok_or_else(|| {
                ReactError::Other("unknown interrupt unexpectedly succeeded".to_string())
            })?;
        assert!(matches!(
            interrupt_error,
            RuntimeTaskAttemptInterruptError::DurableAuthority { .. }
        ));
        assert!(matches!(
            service.attempt_settlement_observation(&execution_id),
            Some(AttemptSettlementObservation::AuthorityUnknown { .. })
        ));
        let mut recovered_settlement = service
            .attempt_settlement_receiver(&execution_id)
            .ok_or_else(|| ReactError::Other("recovery settlement watch missing".to_string()))?;
        let _ = recovered_settlement.borrow_and_update();
        let recovery_waiter = tokio::spawn(async move {
            recovered_settlement
                .changed()
                .await
                .map_err(|_| ReactError::Other("recovery settlement watch closed".to_string()))?;
            Ok::<_, ReactError>(recovered_settlement.borrow_and_update().clone())
        });

        let reconcile_entered = Arc::new(tokio::sync::Notify::new());
        let reconcile_release = Arc::new(tokio::sync::Notify::new());
        *controller
            .reconciliation_entered
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(reconcile_entered.clone());
        *controller
            .reconciliation_release
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(reconcile_release.clone());
        {
            let mut snapshot = controller
                .snapshot
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let task = snapshot
                .as_mut()
                .and_then(|snapshot| snapshot.tasks.first_mut())
                .ok_or_else(|| ReactError::Other("joined task disappeared".to_string()))?;
            task.execution.status = TaskStatus::Completed;
            task.execution.claim = None;
        }
        let reconciliation = tokio::spawn({
            let service = Arc::clone(&service);
            async move { service.reconcile_attempt_control("joined-unknown").await }
        });
        tokio::time::timeout(Duration::from_secs(2), reconcile_entered.notified())
            .await
            .map_err(|_| ReactError::Other("reconciliation hook was not reached".to_string()))?;
        assert_eq!(
            tokio::time::timeout(Duration::from_secs(2), recovery_waiter)
                .await
                .map_err(|_| {
                    ReactError::Other(
                        "durable recovery did not release the settlement waiter".to_string(),
                    )
                })?
                .map_err(|error| {
                    ReactError::Other(format!("recovery waiter task panicked: {error}"))
                })??,
            Some(AttemptSettlementObservation::Settled {
                status: "superseded".to_string(),
            })
        );
        assert!(!service.has_attempt_supervisor_state(&execution_id));
        service.reconcile_local_attempt_control("joined-unknown", &HashSet::new());
        assert!(!service.has_attempt_supervisor_state(&execution_id));
        reconcile_release.notify_one();
        tokio::time::timeout(Duration::from_secs(2), reconciliation)
            .await
            .map_err(|_| ReactError::Other("reconciliation remained blocked".to_string()))?
            .map_err(|error| {
                ReactError::Other(format!("reconciliation task panicked: {error}"))
            })??;
        Ok(())
    }

    /// 验证预留失败且放弃结果未知时保留控制投影。
    #[tokio::test]
    async fn unknown_reservation_abandonment_preserves_control_projection() -> Result<()> {
        let controller = Arc::new(ScriptedController::with_tasks(vec![runtime_task(
            "reserved",
            TaskStatus::Pending,
            &[],
        )]));
        *controller
            .reservation_error
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) =
            Some("reservation failed".to_string());
        *controller
            .abandonment_error_once
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = true;
        let service = super::super::RuntimeTaskService::new(
            controller.clone(),
            RuntimeTaskServiceConfig::default(),
        );

        let error = service
            .execute("unknown-reservation", CancellationToken::new())
            .await
            .err()
            .ok_or_else(|| ReactError::Other("reservation failure unexpectedly ran".to_string()))?;
        assert!(error.to_string().contains("outcome is unknown"));
        assert!(
            controller
                .retired
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .is_empty()
        );
        assert_eq!(
            controller.statuses().get("reserved"),
            Some(&TaskStatus::Running)
        );
        Ok(())
    }

    /// 验证清理失败会被观察到，但不会回滚已提交的终态。
    #[tokio::test]
    async fn cleanup_failure_is_observed_without_reversing_committed_terminal() -> Result<()> {
        let controller = Arc::new(ScriptedController::with_tasks(vec![runtime_task(
            "completed",
            TaskStatus::Pending,
            &[],
        )]));
        *controller
            .cleanup_receipt
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) =
            Some(RuntimeAttemptControlCleanupReceipt::RetryableFailure {
                error: "cleanup unavailable".to_string(),
            });
        let observed = Arc::new(Mutex::new(Vec::new()));
        let observed_receipts = Arc::clone(&observed);
        let service = super::super::RuntimeTaskService::new(
            controller.clone(),
            RuntimeTaskServiceConfig {
                attempt_control_observer: Some(Arc::new(move |observation| {
                    observed_receipts
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .push((observation.task_id.clone(), observation.receipt.clone()));
                })),
                ..RuntimeTaskServiceConfig::default()
            },
        );

        assert_eq!(
            service
                .execute("cleanup-observer", CancellationToken::new())
                .await?,
            RuntimeDagOutcome::Completed
        );
        assert_eq!(
            controller.statuses().get("completed"),
            Some(&TaskStatus::Completed)
        );
        assert!(
            observed
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .iter()
                .any(|(task_id, receipt)| {
                    task_id == "completed"
                        && matches!(
                            receipt,
                            RuntimeAttemptControlCleanupReceipt::RetryableFailure { .. }
                        )
                })
        );
        Ok(())
    }

    /// 验证执行器将已跳过的任务视为已解决。
    #[tokio::test]
    async fn executor_treats_skipped_tasks_as_resolved() -> Result<()> {
        let controller = Arc::new(ScriptedController::with_tasks(vec![runtime_task(
            "skipped",
            TaskStatus::Skipped,
            &[],
        )]));
        let executor = RuntimeDagExecutor::new(controller, RuntimeTaskServiceConfig::default());

        let outcome = executor.execute("run", CancellationToken::new()).await?;

        assert_eq!(outcome, RuntimeDagOutcome::Completed);
        Ok(())
    }

    /// 验证执行器拒绝附着在非活跃状态上的 claim。
    #[tokio::test]
    async fn executor_rejects_claims_attached_to_inactive_statuses() -> Result<()> {
        for status in [TaskStatus::Pending, TaskStatus::Completed] {
            let mut invalid = runtime_task("invalid", status, &[]);
            invalid.execution.claim = Some(TaskClaim::new(
                1,
                1,
                invalid.spec.stable_hash().map_err(ReactError::Other)?,
            ));
            let controller = Arc::new(ScriptedController::with_tasks(vec![invalid]));
            let service = super::super::RuntimeTaskService::new(
                controller,
                RuntimeTaskServiceConfig::default(),
            );

            let error = service
                .execute("invalid-claim", CancellationToken::new())
                .await
                .err()
                .ok_or_else(|| {
                    ReactError::Other("inactive claim snapshot unexpectedly executed".to_string())
                })?;
            assert!(error.to_string().contains("invalid runtime claim snapshot"));
        }
        Ok(())
    }

    /// 验证执行器将已取消的快照视为已中断。
    #[tokio::test]
    async fn executor_treats_cancelled_snapshot_as_interrupted() -> Result<()> {
        let controller = Arc::new(ScriptedController::with_tasks(vec![runtime_task(
            "cancelled",
            TaskStatus::Cancelled,
            &[],
        )]));
        let executor = RuntimeDagExecutor::new(controller, RuntimeTaskServiceConfig::default());

        let outcome = executor.execute("run", CancellationToken::new()).await?;

        assert_eq!(outcome, RuntimeDagOutcome::Cancelled);
        Ok(())
    }

    /// 验证恢复出的中断优先于保留的失败。
    #[tokio::test]
    async fn recovered_interruption_takes_precedence_over_retained_failure() -> Result<()> {
        let cancelled = Arc::new(ScriptedController::with_tasks(vec![
            runtime_task("failed", TaskStatus::Failed("old failure".to_string()), &[]),
            runtime_task("cancelled", TaskStatus::Cancelled, &[]),
            runtime_task("pending", TaskStatus::Pending, &[]),
        ]));
        let cancelled_service = super::super::RuntimeTaskService::new(
            cancelled.clone(),
            RuntimeTaskServiceConfig::default(),
        );
        *cancelled
            .interruption
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) =
            RuntimeInterruptionDisposition::Paused {
                reason: "weaker requested pause".to_string(),
            };
        let requested_pause = CancellationToken::new();
        requested_pause.cancel();
        assert_eq!(
            cancelled_service
                .execute("cancelled-run", requested_pause)
                .await?,
            RuntimeDagOutcome::Cancelled
        );
        let cancelled_statuses = cancelled.statuses();
        assert_eq!(
            cancelled_statuses.get("failed"),
            Some(&TaskStatus::Failed("old failure".to_string()))
        );
        assert_eq!(
            cancelled_statuses.get("pending"),
            Some(&TaskStatus::Cancelled)
        );

        let paused = Arc::new(ScriptedController::with_tasks(vec![
            runtime_task("failed", TaskStatus::Failed("old failure".to_string()), &[]),
            runtime_task(
                "paused",
                TaskStatus::Paused("resume later".to_string()),
                &[],
            ),
            runtime_task("pending", TaskStatus::Pending, &[]),
        ]));
        let paused_service = super::super::RuntimeTaskService::new(
            paused.clone(),
            RuntimeTaskServiceConfig::default(),
        );
        assert_eq!(
            paused_service
                .execute("paused-run", CancellationToken::new())
                .await?,
            RuntimeDagOutcome::Paused {
                task_id: None,
                reason: "resume later".to_string(),
            }
        );
        assert_eq!(paused.statuses().get("pending"), Some(&TaskStatus::Pending));
        Ok(())
    }

    /// 验证已完成的快照优先于迟到的取消请求。
    #[tokio::test]
    async fn completed_snapshot_wins_over_late_cancellation_request() -> Result<()> {
        let controller = Arc::new(ScriptedController::with_tasks(vec![
            runtime_task("completed", TaskStatus::Completed, &[]),
            runtime_task("skipped", TaskStatus::Skipped, &[]),
        ]));
        let service =
            super::super::RuntimeTaskService::new(controller, RuntimeTaskServiceConfig::default());
        let cancel = CancellationToken::new();
        cancel.cancel();

        assert_eq!(
            service.execute("completed-run", cancel).await?,
            RuntimeDagOutcome::Completed
        );
        Ok(())
    }

    /// 验证最后一次成功结算优先于同一边界上的取消。
    #[tokio::test]
    async fn last_success_settlement_wins_over_same_boundary_cancellation() -> Result<()> {
        let controller = Arc::new(ScriptedController::with_tasks(vec![runtime_task(
            "last",
            TaskStatus::Pending,
            &[],
        )]));
        let cancel = CancellationToken::new();
        controller
            .cancel_after_dispatch
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert("last".to_string(), cancel.clone());
        let service = super::super::RuntimeTaskService::new(
            controller.clone(),
            RuntimeTaskServiceConfig::default(),
        );

        assert_eq!(
            service.execute("last-success", cancel).await?,
            RuntimeDagOutcome::Completed
        );
        assert_eq!(
            controller.statuses().get("last"),
            Some(&TaskStatus::Completed)
        );
        Ok(())
    }

    /// 验证边界上迟到的取消覆盖阻塞结算。
    #[tokio::test]
    async fn late_cancellation_overrides_blocked_resolution_at_boundary() -> Result<()> {
        let controller = Arc::new(ScriptedController::with_tasks(vec![runtime_task(
            "blocked",
            TaskStatus::Pending,
            &[],
        )]));
        controller
            .blocked
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(
                "blocked".to_string(),
                ("await review".to_string(), RuntimeStopDisposition::Pause),
            );
        let cancel = CancellationToken::new();
        controller
            .cancel_after_dispatch
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert("blocked".to_string(), cancel.clone());
        let service = super::super::RuntimeTaskService::new(
            controller.clone(),
            RuntimeTaskServiceConfig::default(),
        );

        assert_eq!(
            service.execute("blocked-cancel", cancel).await?,
            RuntimeDagOutcome::Cancelled
        );
        assert_eq!(
            controller.statuses().get("blocked"),
            Some(&TaskStatus::Cancelled)
        );
        Ok(())
    }

    /// 验证波次开始前的取消直接结算，不进行分派。
    #[tokio::test]
    async fn service_settles_pre_wave_cancellation_without_dispatch() -> Result<()> {
        let controller = Arc::new(ScriptedController::with_tasks(vec![
            runtime_task("a", TaskStatus::Pending, &[]),
            runtime_task("b", TaskStatus::Pending, &[]),
        ]));
        let runtime_tasks = Arc::new(super::super::RuntimeTaskService::new(
            controller.clone(),
            RuntimeTaskServiceConfig::default(),
        ));
        let cancel = CancellationToken::new();
        cancel.cancel();

        let outcome = runtime_tasks.execute("run", cancel).await?;

        assert_eq!(outcome, RuntimeDagOutcome::Cancelled);
        assert!(
            controller
                .order
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .is_empty()
        );
        assert!(
            controller
                .statuses()
                .values()
                .all(|status| status == &TaskStatus::Cancelled)
        );
        Ok(())
    }

    /// 验证波次开始前的暂停直接结算，不重试也不分派。
    #[tokio::test]
    async fn service_settles_pre_wave_pause_without_retry_or_dispatch() -> Result<()> {
        let controller = Arc::new(ScriptedController::with_tasks(vec![runtime_task(
            "pending",
            TaskStatus::Pending,
            &[],
        )]));
        *controller
            .interruption
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) =
            RuntimeInterruptionDisposition::Paused {
                reason: "deterministic pause".to_string(),
            };
        let runtime_tasks = Arc::new(super::super::RuntimeTaskService::new(
            controller.clone(),
            RuntimeTaskServiceConfig::default(),
        ));
        let cancel = CancellationToken::new();
        cancel.cancel();

        let outcome = runtime_tasks.execute("run", cancel).await?;

        assert_eq!(
            outcome,
            RuntimeDagOutcome::Paused {
                task_id: None,
                reason: "deterministic pause".to_string(),
            }
        );
        assert_eq!(
            controller.statuses().get("pending"),
            Some(&TaskStatus::Pending)
        );
        Ok(())
    }

    /// 验证取消会排空当前波次并保留已完成的兄弟任务。
    #[tokio::test]
    async fn cancellation_drains_wave_and_preserves_completed_siblings() -> Result<()> {
        let controller = Arc::new(ScriptedController::with_tasks(vec![
            runtime_task("fast", TaskStatus::Pending, &[]),
            runtime_task("slow", TaskStatus::Pending, &[]),
        ]));
        controller
            .ignore_cancel
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert("slow".to_string());
        *controller
            .dispatch_barrier
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) =
            Some(Arc::new(tokio::sync::Barrier::new(2)));
        let cancel = CancellationToken::new();
        controller
            .cancel_after_dispatch
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert("fast".to_string(), cancel.clone());
        let runtime_tasks = super::super::RuntimeTaskService::new(
            controller.clone(),
            RuntimeTaskServiceConfig {
                cancellation_grace_period: Duration::from_millis(200),
                ..RuntimeTaskServiceConfig::default()
            },
        );

        let outcome = runtime_tasks.execute("run", cancel).await?;

        assert_eq!(outcome, RuntimeDagOutcome::Cancelled);
        let statuses = controller.statuses();
        assert_eq!(statuses.get("fast"), Some(&TaskStatus::Completed));
        assert_eq!(statuses.get("slow"), Some(&TaskStatus::Cancelled));
        Ok(())
    }

    /// 验证精确运行时中断只取消被 claim 任务的子令牌。
    #[tokio::test]
    async fn exact_runtime_interrupt_cancels_only_claimed_task_child() -> Result<()> {
        let controller = Arc::new(ScriptedController::with_tasks(vec![runtime_task(
            "slow",
            TaskStatus::Pending,
            &[],
        )]));
        let claim_ready = Arc::new(tokio::sync::Notify::new());
        *controller
            .claim_ready
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(claim_ready.clone());
        controller
            .wait_for_cancel
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert("slow".to_string());
        let runtime_tasks = Arc::new(super::super::RuntimeTaskService::new(
            controller.clone(),
            RuntimeTaskServiceConfig::default(),
        ));
        let run_cancel = CancellationToken::new();
        let execution = tokio::spawn({
            let runtime_tasks = Arc::clone(&runtime_tasks);
            let run_cancel = run_cancel.clone();
            async move { runtime_tasks.execute("exact-interrupt", run_cancel).await }
        });
        tokio::time::timeout(Duration::from_secs(2), claim_ready.notified())
            .await
            .map_err(|_| ReactError::Other("claim was not admitted".to_string()))?;
        let claim = controller
            .snapshot
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .as_ref()
            .and_then(|snapshot| snapshot.tasks.first())
            .and_then(|task| task.execution.claim.clone())
            .ok_or_else(|| ReactError::Other("claimed task was not persisted".to_string()))?;
        let receipt = runtime_tasks
            .request_attempt_interrupt("exact-interrupt", "slow", &claim)
            .await?;
        assert!(receipt.requested);
        let outcome = tokio::time::timeout(Duration::from_secs(2), execution)
            .await
            .map_err(|_| ReactError::Other("exact interrupt did not settle".to_string()))?
            .map_err(|error| ReactError::Other(format!("runtime task panicked: {error}")))??;
        assert_eq!(outcome, RuntimeDagOutcome::Cancelled);
        assert_eq!(
            controller.statuses().get("slow"),
            Some(&TaskStatus::Cancelled)
        );
        Ok(())
    }

    /// 验证在共享准入前的精确中断会以预先取消的状态分派。
    #[tokio::test]
    async fn exact_interrupt_before_shared_admission_runs_pre_cancelled_dispatch() -> Result<()> {
        let controller = Arc::new(ScriptedController::with_tasks(vec![runtime_task(
            "queued",
            TaskStatus::Pending,
            &[],
        )]));
        let claim_ready = Arc::new(tokio::sync::Notify::new());
        *controller
            .claim_ready
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(claim_ready.clone());
        controller
            .wait_for_cancel
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert("queued".to_string());
        let admission = Arc::new(ExecutionAdmission::with_capacity(1));
        let held = admission
            .issue("held")
            .map_err(|error| ReactError::Other(error.to_string()))?;
        let runtime_tasks = Arc::new(super::super::RuntimeTaskService::new(
            controller.clone(),
            RuntimeTaskServiceConfig {
                shared_admission: Some(admission),
                ..RuntimeTaskServiceConfig::default()
            },
        ));
        let execution = tokio::spawn({
            let runtime_tasks = Arc::clone(&runtime_tasks);
            async move {
                runtime_tasks
                    .execute("pre-admission-interrupt", CancellationToken::new())
                    .await
            }
        });
        tokio::time::timeout(Duration::from_secs(2), claim_ready.notified())
            .await
            .map_err(|_| ReactError::Other("claim was not admitted".to_string()))?;
        let claim = controller
            .snapshot
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .as_ref()
            .and_then(|snapshot| snapshot.tasks.first())
            .and_then(|task| task.execution.claim.clone())
            .ok_or_else(|| ReactError::Other("claimed task was not persisted".to_string()))?;
        assert!(
            runtime_tasks
                .request_attempt_interrupt("pre-admission-interrupt", "queued", &claim)
                .await?
                .requested
        );
        let outcome = tokio::time::timeout(Duration::from_secs(2), execution)
            .await
            .map_err(|_| ReactError::Other("pre-admission interrupt did not settle".to_string()))?
            .map_err(|error| ReactError::Other(format!("runtime task panicked: {error}")))??;
        assert_eq!(outcome, RuntimeDagOutcome::Cancelled);
        assert_eq!(
            controller
                .order
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .as_slice(),
            ["queued"]
        );
        drop(held);
        Ok(())
    }

    /// 验证在运行时预留前的精确中断返回类型化的排队回执。
    #[tokio::test]
    async fn exact_interrupt_before_runtime_reservation_returns_typed_queued_receipt() -> Result<()>
    {
        let controller = Arc::new(ScriptedController::with_tasks(vec![runtime_task(
            "queued",
            TaskStatus::Pending,
            &[],
        )]));
        let expected = controller
            .snapshot
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .as_ref()
            .and_then(|snapshot| snapshot.tasks.first())
            .cloned()
            .ok_or_else(|| ReactError::Other("queued task missing".to_string()))?;
        let claim = match controller
            .claim_task("queued-receipt", &expected, 1)
            .await?
        {
            RuntimeTaskClaimOutcome::Claimed(claim) => claim,
            RuntimeTaskClaimOutcome::ReloadSnapshot => {
                return Err(ReactError::Other("queued task was not claimed".to_string()));
            }
        };
        let service =
            super::super::RuntimeTaskService::new(controller, RuntimeTaskServiceConfig::default());

        let receipt = service
            .request_attempt_interrupt("queued-receipt", "queued", &claim)
            .await?;
        assert_eq!(
            receipt.disposition,
            RuntimeTaskAttemptInterruptDisposition::QueuedBeforeReservation
        );
        assert!(receipt.requested);
        Ok(())
    }

    /// 验证延迟预留时，为句柄就绪前的中断重新调度中止。
    #[tokio::test]
    async fn delayed_reservation_rearms_abort_for_pre_handle_interrupt() -> Result<()> {
        let controller = Arc::new(ScriptedController::with_tasks(vec![runtime_task(
            "delayed",
            TaskStatus::Pending,
            &[],
        )]));
        let claim_ready = Arc::new(tokio::sync::Notify::new());
        *controller
            .claim_ready
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(claim_ready.clone());
        *controller
            .reservation_delay
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(Duration::from_millis(80));
        controller
            .ignore_cancel
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert("delayed".to_string());
        let service = Arc::new(super::super::RuntimeTaskService::new(
            controller.clone(),
            RuntimeTaskServiceConfig {
                cancellation_grace_period: Duration::from_millis(20),
                ..RuntimeTaskServiceConfig::default()
            },
        ));
        let execution = tokio::spawn({
            let service = Arc::clone(&service);
            async move {
                service
                    .execute("delayed-reservation", CancellationToken::new())
                    .await
            }
        });
        tokio::time::timeout(Duration::from_secs(2), claim_ready.notified())
            .await
            .map_err(|_| ReactError::Other("delayed claim was not admitted".to_string()))?;
        let claim = controller
            .snapshot
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .as_ref()
            .and_then(|snapshot| snapshot.tasks.first())
            .and_then(|task| task.execution.claim.clone())
            .ok_or_else(|| ReactError::Other("delayed claim was not persisted".to_string()))?;
        let receipt = service
            .request_attempt_interrupt("delayed-reservation", "delayed", &claim)
            .await?;
        assert_eq!(
            receipt.disposition,
            RuntimeTaskAttemptInterruptDisposition::ActiveRequested
        );
        let outcome = tokio::time::timeout(Duration::from_secs(2), execution)
            .await
            .map_err(|_| ReactError::Other("delayed reservation escaped exact abort".to_string()))?
            .map_err(|error| ReactError::Other(format!("runtime task panicked: {error}")))??;
        assert_eq!(outcome, RuntimeDagOutcome::Cancelled);
        assert_eq!(
            controller.statuses().get("delayed"),
            Some(&TaskStatus::Cancelled)
        );
        Ok(())
    }

    /// 验证请求期间 claim 被取代时，精确中断会退役投影。
    #[tokio::test]
    async fn exact_interrupt_retires_projection_when_claim_supersedes_during_request() -> Result<()>
    {
        let controller = Arc::new(ScriptedController::with_tasks(vec![runtime_task(
            "slow",
            TaskStatus::Pending,
            &[],
        )]));
        let claim_ready = Arc::new(tokio::sync::Notify::new());
        *controller
            .claim_ready
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(claim_ready.clone());
        controller
            .wait_for_cancel
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert("slow".to_string());
        *controller
            .stale_after_current_check
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(2);
        *controller
            .supersede_on_stale_check
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = true;
        let runtime_tasks = Arc::new(super::super::RuntimeTaskService::new(
            controller.clone(),
            RuntimeTaskServiceConfig {
                cancellation_grace_period: Duration::from_millis(20),
                ..RuntimeTaskServiceConfig::default()
            },
        ));
        let execution = tokio::spawn({
            let runtime_tasks = Arc::clone(&runtime_tasks);
            async move {
                runtime_tasks
                    .execute("stale-interrupt", CancellationToken::new())
                    .await
            }
        });
        tokio::time::timeout(Duration::from_secs(2), claim_ready.notified())
            .await
            .map_err(|_| ReactError::Other("claim was not admitted".to_string()))?;
        let claim = controller
            .snapshot
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .as_ref()
            .and_then(|snapshot| snapshot.tasks.first())
            .and_then(|task| task.execution.claim.clone())
            .ok_or_else(|| ReactError::Other("claimed task was not persisted".to_string()))?;
        let error = runtime_tasks
            .request_attempt_interrupt("stale-interrupt", "slow", &claim)
            .await
            .err()
            .ok_or_else(|| {
                ReactError::Other("stale interrupt unexpectedly succeeded".to_string())
            })?;
        assert!(error.to_string().contains("stale or settled"));
        let outcome = tokio::time::timeout(Duration::from_secs(2), execution)
            .await
            .map_err(|_| ReactError::Other("stale interrupt did not settle".to_string()))?
            .map_err(|error| ReactError::Other(format!("runtime task panicked: {error}")))??;
        assert_eq!(outcome, RuntimeDagOutcome::Completed);
        assert_eq!(
            controller.statuses().get("slow"),
            Some(&TaskStatus::Completed)
        );
        Ok(())
    }

    /// 验证协调失败时陈旧 claim 的结果仍被保留。
    #[tokio::test]
    async fn stale_claim_outcome_survives_reconciliation_failure() -> Result<()> {
        let controller = Arc::new(ScriptedController::with_tasks(vec![runtime_task(
            "settled",
            TaskStatus::Completed,
            &[],
        )]));
        *controller
            .reconciliation_error
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) =
            Some("live cleanup unavailable".to_string());
        let observed = Arc::new(Mutex::new(Vec::new()));
        let observed_cleanup = Arc::clone(&observed);
        let service = super::super::RuntimeTaskService::new(
            controller,
            RuntimeTaskServiceConfig {
                attempt_control_observer: Some(Arc::new(move |observation| {
                    observed_cleanup
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .push(observation.clone());
                })),
                ..RuntimeTaskServiceConfig::default()
            },
        );
        let stale = TaskClaim::new(1, 1, "stale".to_string());

        assert!(matches!(
            service
                .request_attempt_interrupt("stale-cleanup", "settled", &stale)
                .await,
            Err(super::super::RuntimeTaskAttemptInterruptError::StaleOrSettledClaim { .. })
        ));
        assert!(
            observed
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .iter()
                .any(|observation| {
                    observation.run_id == "stale-cleanup"
                        && observation.task_id == "settled"
                        && matches!(
                            &observation.receipt,
                            RuntimeAttemptControlCleanupReceipt::RetryableFailure { .. }
                        )
                })
        );
        Ok(())
    }

    /// 验证精确中断在宽限期后中止不配合的尝试。
    #[tokio::test]
    async fn exact_interrupt_aborts_non_cooperative_attempt_after_grace() -> Result<()> {
        let controller = Arc::new(ScriptedController::with_tasks(vec![runtime_task(
            "stuck",
            TaskStatus::Pending,
            &[],
        )]));
        let claim_ready = Arc::new(tokio::sync::Notify::new());
        *controller
            .claim_ready
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(claim_ready.clone());
        controller
            .ignore_cancel
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert("stuck".to_string());
        let runtime_tasks = Arc::new(super::super::RuntimeTaskService::new(
            controller.clone(),
            RuntimeTaskServiceConfig {
                cancellation_grace_period: Duration::from_millis(20),
                ..RuntimeTaskServiceConfig::default()
            },
        ));
        let execution = tokio::spawn({
            let runtime_tasks = Arc::clone(&runtime_tasks);
            async move {
                runtime_tasks
                    .execute("forced-interrupt", CancellationToken::new())
                    .await
            }
        });
        tokio::time::timeout(Duration::from_secs(2), claim_ready.notified())
            .await
            .map_err(|_| ReactError::Other("claim was not admitted".to_string()))?;
        let claim = controller
            .snapshot
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .as_ref()
            .and_then(|snapshot| snapshot.tasks.first())
            .and_then(|task| task.execution.claim.clone())
            .ok_or_else(|| ReactError::Other("claimed task was not persisted".to_string()))?;
        let mut receipt = None;
        for _ in 0..100 {
            let candidate = runtime_tasks
                .request_attempt_interrupt("forced-interrupt", "stuck", &claim)
                .await?;
            if candidate.requested {
                receipt = Some(candidate);
                break;
            }
            tokio::task::yield_now().await;
        }
        assert!(receipt.is_some());
        let outcome = tokio::time::timeout(Duration::from_secs(2), execution)
            .await
            .map_err(|_| ReactError::Other("non-cooperative interrupt did not settle".to_string()))?
            .map_err(|error| ReactError::Other(format!("runtime task panicked: {error}")))??;
        assert_eq!(outcome, RuntimeDagOutcome::Cancelled);
        assert_eq!(
            controller.statuses().get("stuck"),
            Some(&TaskStatus::Cancelled)
        );
        Ok(())
    }

    /// 验证并发的精确中止各自结算自己的 claim。
    #[tokio::test]
    async fn concurrent_exact_aborts_settle_their_own_claims() -> Result<()> {
        let controller = Arc::new(ScriptedController::with_tasks(vec![
            runtime_task("left", TaskStatus::Pending, &[]),
            runtime_task("right", TaskStatus::Pending, &[]),
        ]));
        controller
            .ignore_cancel
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .extend(["left".to_string(), "right".to_string()]);
        let runtime_tasks = Arc::new(super::super::RuntimeTaskService::new(
            controller.clone(),
            RuntimeTaskServiceConfig {
                max_concurrent_subagents: 2,
                cancellation_grace_period: Duration::from_millis(20),
                ..RuntimeTaskServiceConfig::default()
            },
        ));
        let execution = tokio::spawn({
            let runtime_tasks = Arc::clone(&runtime_tasks);
            async move {
                runtime_tasks
                    .execute("concurrent-aborts", CancellationToken::new())
                    .await
            }
        });
        let claims = tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                let claims = controller
                    .snapshot
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .as_ref()
                    .map(|snapshot| {
                        snapshot
                            .tasks
                            .iter()
                            .filter_map(|task| {
                                task.execution
                                    .claim
                                    .clone()
                                    .map(|claim| (task.spec.id.clone(), claim))
                            })
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();
                if claims.len() == 2 {
                    break claims;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .map_err(|_| ReactError::Other("both exact claims were not admitted".to_string()))?;
        for (task_id, claim) in claims {
            assert!(
                runtime_tasks
                    .request_attempt_interrupt("concurrent-aborts", &task_id, &claim)
                    .await?
                    .requested
            );
        }
        let outcome = tokio::time::timeout(Duration::from_secs(2), execution)
            .await
            .map_err(|_| ReactError::Other("concurrent exact aborts did not settle".to_string()))?
            .map_err(|error| ReactError::Other(format!("runtime task panicked: {error}")))??;
        assert_eq!(outcome, RuntimeDagOutcome::Cancelled);
        let statuses = controller.statuses();
        assert_eq!(statuses.get("left"), Some(&TaskStatus::Cancelled));
        assert_eq!(statuses.get("right"), Some(&TaskStatus::Cancelled));
        Ok(())
    }

    /// 验证根取消与精确中止收敛，且不会错误归属到其他尝试。
    #[tokio::test]
    async fn root_and_exact_abort_converge_without_cross_attempt_attribution() -> Result<()> {
        let controller = Arc::new(ScriptedController::with_tasks(vec![
            runtime_task("exact", TaskStatus::Pending, &[]),
            runtime_task("root", TaskStatus::Pending, &[]),
        ]));
        controller
            .ignore_cancel
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .extend(["exact".to_string(), "root".to_string()]);
        let runtime_tasks = Arc::new(super::super::RuntimeTaskService::new(
            controller.clone(),
            RuntimeTaskServiceConfig {
                max_concurrent_subagents: 2,
                cancellation_grace_period: Duration::from_millis(20),
                ..RuntimeTaskServiceConfig::default()
            },
        ));
        let run_cancel = CancellationToken::new();
        let execution = tokio::spawn({
            let runtime_tasks = Arc::clone(&runtime_tasks);
            let run_cancel = run_cancel.clone();
            async move { runtime_tasks.execute("root-exact", run_cancel).await }
        });
        let exact_claim = tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                let claim = controller
                    .snapshot
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .as_ref()
                    .and_then(|snapshot| {
                        let all_claimed = snapshot
                            .tasks
                            .iter()
                            .all(|task| task.execution.claim.is_some());
                        all_claimed.then(|| {
                            snapshot
                                .tasks
                                .iter()
                                .find(|task| task.spec.id == "exact")
                                .and_then(|task| task.execution.claim.clone())
                        })
                    })
                    .flatten();
                if let Some(claim) = claim {
                    break claim;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .map_err(|_| ReactError::Other("root/exact claims were not admitted".to_string()))?;
        assert!(
            runtime_tasks
                .request_attempt_interrupt("root-exact", "exact", &exact_claim)
                .await?
                .requested
        );
        run_cancel.cancel();
        let outcome = tokio::time::timeout(Duration::from_secs(2), execution)
            .await
            .map_err(|_| ReactError::Other("root/exact abort did not settle".to_string()))?
            .map_err(|error| ReactError::Other(format!("runtime task panicked: {error}")))??;
        assert_eq!(outcome, RuntimeDagOutcome::Cancelled);
        let statuses = controller.statuses();
        assert_eq!(statuses.get("exact"), Some(&TaskStatus::Cancelled));
        assert_eq!(statuses.get("root"), Some(&TaskStatus::Cancelled));
        Ok(())
    }

    /// 验证波次中途暂停保留已完成的兄弟任务，并暂停被中断的 claim。
    #[tokio::test]
    async fn mid_wave_pause_preserves_completed_sibling_and_pauses_interrupted_claim() -> Result<()>
    {
        let controller = Arc::new(ScriptedController::with_tasks(vec![
            runtime_task("fast", TaskStatus::Pending, &[]),
            runtime_task("slow", TaskStatus::Pending, &[]),
        ]));
        controller
            .wait_for_cancel
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert("slow".to_string());
        *controller
            .dispatch_barrier
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) =
            Some(Arc::new(tokio::sync::Barrier::new(2)));
        *controller
            .interruption
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) =
            RuntimeInterruptionDisposition::Paused {
                reason: "pause at wave barrier".to_string(),
            };
        let cancel = CancellationToken::new();
        controller
            .cancel_after_dispatch
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert("fast".to_string(), cancel.clone());
        let runtime_tasks = super::super::RuntimeTaskService::new(
            controller.clone(),
            RuntimeTaskServiceConfig {
                cancellation_grace_period: Duration::from_millis(200),
                ..RuntimeTaskServiceConfig::default()
            },
        );

        let outcome = runtime_tasks.execute("run", cancel).await?;

        assert_eq!(
            outcome,
            RuntimeDagOutcome::Paused {
                task_id: None,
                reason: "pause at wave barrier".to_string(),
            }
        );
        let snapshot = controller
            .snapshot
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
            .ok_or_else(|| ReactError::Other("missing paused snapshot".to_string()))?;
        let fast = snapshot
            .tasks
            .iter()
            .find(|task| task.spec.id == "fast")
            .ok_or_else(|| ReactError::Other("fast task is missing".to_string()))?;
        let slow = snapshot
            .tasks
            .iter()
            .find(|task| task.spec.id == "slow")
            .ok_or_else(|| ReactError::Other("slow task is missing".to_string()))?;
        assert_eq!(fast.execution.status, TaskStatus::Completed);
        assert_eq!(
            slow.execution.status,
            TaskStatus::Paused("pause at wave barrier".to_string())
        );
        assert!(slow.execution.claim.is_none());
        assert_eq!(slow.execution.retry_count, 0);
        Ok(())
    }

    /// 验证中断策略出错时先清理所有 claim 再返回错误。
    #[tokio::test]
    async fn interruption_policy_error_cleans_every_claim_before_returning_error() -> Result<()> {
        let controller = Arc::new(ScriptedController::with_tasks(vec![
            runtime_task("fast", TaskStatus::Pending, &[]),
            runtime_task("slow", TaskStatus::Pending, &[]),
        ]));
        controller
            .wait_for_cancel
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert("slow".to_string());
        *controller
            .dispatch_barrier
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) =
            Some(Arc::new(tokio::sync::Barrier::new(2)));
        *controller
            .interruption_error
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) =
            Some("interruption authority unavailable".to_string());
        let cancel = CancellationToken::new();
        controller
            .cancel_after_dispatch
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert("fast".to_string(), cancel.clone());
        let service = super::super::RuntimeTaskService::new(
            controller.clone(),
            RuntimeTaskServiceConfig {
                cancellation_grace_period: Duration::from_millis(200),
                ..RuntimeTaskServiceConfig::default()
            },
        );

        let error = service
            .execute("policy-error", cancel)
            .await
            .err()
            .ok_or_else(|| ReactError::Other("policy error unexpectedly succeeded".to_string()))?;
        assert!(
            error
                .to_string()
                .contains("interruption authority unavailable")
        );
        let snapshot = controller
            .snapshot
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
            .ok_or_else(|| ReactError::Other("policy-error snapshot missing".to_string()))?;
        assert!(
            snapshot
                .tasks
                .iter()
                .all(|task| task.execution.claim.is_none())
        );
        assert!(snapshot.tasks.iter().all(|task| {
            matches!(
                task.execution.status,
                TaskStatus::Completed | TaskStatus::Paused(_) | TaskStatus::Pending
            )
        }));
        assert!(
            snapshot
                .tasks
                .iter()
                .any(|task| matches!(task.execution.status, TaskStatus::Paused(_)))
        );
        assert!(
            snapshot
                .tasks
                .iter()
                .all(|task| task.execution.status != TaskStatus::Cancelled)
        );
        Ok(())
    }

    /// 验证重新 claim 的物理尝试拥有不同的 execution_id。
    #[test]
    fn reclaimed_physical_attempt_has_a_distinct_execution_id() {
        let first = TaskClaim::new(7, 2, "same-spec".to_string());
        let second = TaskClaim::new(7, 2, "same-spec".to_string());

        assert_ne!(first.claim_id, second.claim_id);
        assert_ne!(
            first.execution_id("run", "task"),
            second.execution_id("run", "task")
        );
    }

    /// 验证取消会放弃不配合的分派 claim。
    #[tokio::test]
    async fn cancellation_abandons_a_non_cooperative_dispatch_claim() -> Result<()> {
        let controller = Arc::new(ScriptedController::with_tasks(vec![runtime_task(
            "stuck",
            TaskStatus::Pending,
            &[],
        )]));
        controller
            .ignore_cancel
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert("stuck".to_string());
        let executor = RuntimeDagExecutor::new(
            controller.clone(),
            RuntimeTaskServiceConfig {
                cancellation_grace_period: Duration::from_millis(10),
                ..RuntimeTaskServiceConfig::default()
            },
        );
        let cancel = CancellationToken::new();
        let run_cancel = cancel.clone();
        let run = tokio::spawn(async move { executor.execute("run", run_cancel).await });

        tokio::time::timeout(Duration::from_secs(1), async {
            while controller
                .order
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .is_empty()
            {
                tokio::task::yield_now().await;
            }
        })
        .await
        .map_err(|_| ReactError::Other("task was not dispatched".to_string()))?;
        cancel.cancel();

        let outcome = tokio::time::timeout(Duration::from_secs(1), run)
            .await
            .map_err(|_| ReactError::Other("executor cancellation timed out".to_string()))?
            .map_err(|error| ReactError::Other(format!("executor failed to join: {error}")))??;
        assert_eq!(outcome, RuntimeDagOutcome::Cancelled);
        assert_eq!(
            controller.statuses().get("stuck"),
            Some(&TaskStatus::Cancelled)
        );
        Ok(())
    }

    /// 验证 claim 冲突时重新加载，既不失败也不分派陈旧工作。
    #[tokio::test]
    async fn claim_conflict_reloads_without_failing_or_dispatching_stale_work() -> Result<()> {
        let controller = Arc::new(ScriptedController::with_tasks(vec![runtime_task(
            "claimable",
            TaskStatus::Pending,
            &[],
        )]));
        *controller
            .reload_claim_once
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = true;
        let executor =
            RuntimeDagExecutor::new(controller.clone(), RuntimeTaskServiceConfig::default());

        let outcome = executor.execute("run", CancellationToken::new()).await?;

        assert_eq!(outcome, RuntimeDagOutcome::Completed);
        assert_eq!(
            controller
                .order
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .as_slice(),
            ["claimable"]
        );
        Ok(())
    }

    /// 验证执行器保留持久化终态中的错误详情。
    #[tokio::test]
    async fn executor_preserves_persisted_terminal_error_details() -> Result<()> {
        let terminal_statuses = [
            TaskStatus::Failed("persisted failure".to_string()),
            TaskStatus::TimedOut {
                error: "persisted timeout".to_string(),
            },
            TaskStatus::Blocked("persisted blocker".to_string()),
        ];

        for status in terminal_statuses {
            let expected_error = persisted_status_error(&status)
                .ok_or_else(|| ReactError::Other("terminal status lost its detail".to_string()))?;
            let controller = Arc::new(ScriptedController::with_tasks(vec![runtime_task(
                "terminal",
                status,
                &[],
            )]));
            let executor = RuntimeDagExecutor::new(controller, RuntimeTaskServiceConfig::default());

            let outcome = executor.execute("run", CancellationToken::new()).await?;

            assert_eq!(
                outcome,
                RuntimeDagOutcome::Failed {
                    failed_task_id: "terminal".to_string(),
                    error: expected_error,
                }
            );
        }
        Ok(())
    }

    /// 验证执行器推导下游阻塞而不将其持久化。
    #[tokio::test]
    async fn executor_derives_downstream_block_without_persisting_it() -> Result<()> {
        let controller = Arc::new(ScriptedController::with_tasks(vec![
            runtime_task("a", TaskStatus::Pending, &[]),
            runtime_task("b", TaskStatus::Pending, &["a"]),
        ]));
        controller
            .fail
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert("a".to_string(), "boom".to_string());
        let executor =
            RuntimeDagExecutor::new(controller.clone(), RuntimeTaskServiceConfig::default());

        let outcome = executor.execute("run", CancellationToken::new()).await?;

        assert_eq!(
            outcome,
            RuntimeDagOutcome::Failed {
                failed_task_id: "a".to_string(),
                error: "boom".to_string(),
            }
        );
        assert_eq!(controller.statuses().get("b"), Some(&TaskStatus::Pending));
        let tasks = controller
            .snapshot
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .as_ref()
            .map(|snapshot| snapshot.tasks.clone())
            .ok_or_else(|| ReactError::Other("missing derived-block snapshot".to_string()))?;
        assert!(matches!(
            DagExecutionState::from_tasks(&tasks)
                .dependency_states(&tasks)
                .get("b"),
            Some(DagDependencyState::BlockedByFailure { .. })
        ));
        Ok(())
    }

    /// 验证执行器推导传递性的下游阻塞链而不持久化。
    #[tokio::test]
    async fn executor_derives_transitive_downstream_chain_without_persistence() -> Result<()> {
        let controller = Arc::new(ScriptedController::with_tasks(vec![
            runtime_task("a", TaskStatus::Pending, &[]),
            runtime_task("b", TaskStatus::Pending, &["a"]),
            runtime_task("c", TaskStatus::Pending, &["b"]),
        ]));
        controller
            .fail
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert("a".to_string(), "boom".to_string());
        let executor =
            RuntimeDagExecutor::new(controller.clone(), RuntimeTaskServiceConfig::default());

        let outcome = executor.execute("run", CancellationToken::new()).await?;
        assert!(matches!(outcome, RuntimeDagOutcome::Failed { .. }));
        let statuses = controller.statuses();
        for id in ["b", "c"] {
            assert_eq!(statuses.get(id), Some(&TaskStatus::Pending));
        }
        let tasks = controller
            .snapshot
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .as_ref()
            .map(|snapshot| snapshot.tasks.clone())
            .ok_or_else(|| ReactError::Other("missing transitive snapshot".to_string()))?;
        let dependencies = DagExecutionState::from_tasks(&tasks).dependency_states(&tasks);
        for id in ["b", "c"] {
            assert!(matches!(
                dependencies.get(id),
                Some(DagDependencyState::BlockedByFailure { .. })
            ));
        }
        Ok(())
    }
}
