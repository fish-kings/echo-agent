/*!
 * 面向单次执行尝试（attempt）、仅在进程内生效的子代理（subagent）派发控制。
 *
 * 持久化的命令身份与恢复逻辑由框架的使用方（consumer）负责维护；本注册表
 * 只负责把一次精确的执行尝试绑定到它的取消令牌（cancellation token），
 * 以及 agent 现有的实时引导（live-steering）安全点上。
 */

use echo_core::agent::{
    Agent, AgentSteerError, AgentSteerReceipt, AgentSteerState, CancellationToken,
};
use echo_core::llm::types::Message;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use std::sync::{Arc, Mutex};
use tokio::sync::watch;

use super::types::SubagentStatus;

const SETTLED_RETENTION: usize = 256;
const PENDING_INTERRUPT_CAPACITY: usize = 256;

/// 一个逻辑子代理任务（Subagent task）中某次执行尝试的稳定身份标识。
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
pub struct SubagentAttemptIdentity {
    /// 所属运行（run）的标识；未绑定到具体运行时为 `None`。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub run_id: Option<String>,
    /// 控制作用域标识，用于在同一 `run_id` 下隔离不同的运行时作用域。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub control_scope_id: Option<String>,
    /// 逻辑任务标识。
    pub task_id: String,
    /// 本次执行的唯一标识。
    pub execution_id: String,
    /// 尝试次数（从 1 开始计数）。
    pub attempt: u32,
}

/// 反序列化的中间字段结构：先解析所有字段，再统一走 `validate` 校验。
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SubagentAttemptIdentityFields {
    #[serde(default)]
    run_id: Option<String>,
    #[serde(default)]
    control_scope_id: Option<String>,
    task_id: String,
    execution_id: String,
    attempt: u32,
}

impl<'de> Deserialize<'de> for SubagentAttemptIdentity {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let fields = SubagentAttemptIdentityFields::deserialize(deserializer)?;
        let identity = Self {
            run_id: fields.run_id,
            control_scope_id: fields.control_scope_id,
            task_id: fields.task_id,
            execution_id: fields.execution_id,
            attempt: fields.attempt,
        };
        identity.validate().map_err(serde::de::Error::custom)?;
        Ok(identity)
    }
}

/**
 * 一次寻址到某个精确 Subagent 执行尝试的、具备幂等性的命令的持久化身份标识。
 *
 * 命令 id 与任务图（task-graph）修订版本号都是通用的生命周期事实，因此
 * 使用方可以直接持久化本结构体，而不必围绕 [`SubagentAttemptIdentity`]
 * 再定义一套平行的命令身份标识。
 */
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
pub struct SubagentCommandIdentity {
    /// 所属运行（run）的标识。
    pub run_id: String,
    /// 逻辑任务标识。
    pub task_id: String,
    /// 本次执行的唯一标识。
    pub execution_id: String,
    /// 任务图（task-graph）的修订版本号。
    pub plan_revision: u64,
    /// 尝试次数（从 1 开始计数）。
    pub attempt: u32,
    /// 命令标识，用于实现幂等性。
    pub command_id: String,
}

/// 反序列化的中间字段结构：先解析所有字段，再统一走 `validate` 校验。
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SubagentCommandIdentityFields {
    run_id: String,
    task_id: String,
    execution_id: String,
    plan_revision: u64,
    attempt: u32,
    command_id: String,
}

impl<'de> Deserialize<'de> for SubagentCommandIdentity {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let fields = SubagentCommandIdentityFields::deserialize(deserializer)?;
        let identity = Self {
            run_id: fields.run_id,
            task_id: fields.task_id,
            execution_id: fields.execution_id,
            plan_revision: fields.plan_revision,
            attempt: fields.attempt,
            command_id: fields.command_id,
        };
        identity.validate().map_err(serde::de::Error::custom)?;
        Ok(identity)
    }
}

impl SubagentCommandIdentity {
    /// 构造并校验一个新的命令身份标识。
    pub fn new(
        run_id: impl Into<String>,
        task_id: impl Into<String>,
        execution_id: impl Into<String>,
        plan_revision: u64,
        attempt: u32,
        command_id: impl Into<String>,
    ) -> Result<Self, SubagentControlError> {
        let identity = Self {
            run_id: run_id.into(),
            task_id: task_id.into(),
            execution_id: execution_id.into(),
            plan_revision,
            attempt,
            command_id: command_id.into(),
        };
        identity.validate()?;
        Ok(identity)
    }

    /// 校验各字段是否合法（非空、修订版本号非零等）。
    pub fn validate(&self) -> Result<(), SubagentControlError> {
        if self.run_id.trim().is_empty() {
            return Err(SubagentControlError::InvalidIdentity { field: "run_id" });
        }
        if self.plan_revision == 0 {
            return Err(SubagentControlError::InvalidIdentity {
                field: "plan_revision",
            });
        }
        if self.command_id.trim().is_empty() {
            return Err(SubagentControlError::InvalidIdentity {
                field: "command_id",
            });
        }
        self.attempt_identity().map(|_| ())
    }

    /// 返回该命令对应的精确执行尝试的实时控制身份标识。
    pub fn attempt_identity(&self) -> Result<SubagentAttemptIdentity, SubagentControlError> {
        SubagentAttemptIdentity::for_run(
            self.run_id.clone(),
            self.task_id.clone(),
            self.execution_id.clone(),
            self.attempt,
        )
    }
}

impl SubagentAttemptIdentity {
    /// 构造一个未绑定运行（run）和控制作用域的执行尝试身份标识。
    pub fn new(
        task_id: impl Into<String>,
        execution_id: impl Into<String>,
        attempt: u32,
    ) -> Result<Self, SubagentControlError> {
        let identity = Self {
            run_id: None,
            control_scope_id: None,
            task_id: task_id.into(),
            execution_id: execution_id.into(),
            attempt,
        };
        identity.validate()?;
        Ok(identity)
    }

    /**
     * 构造一个绑定到指定运行（run）的执行尝试身份标识，并将该运行 id
     * 同时作为控制作用域标识。
     */
    pub fn for_run(
        run_id: impl Into<String>,
        task_id: impl Into<String>,
        execution_id: impl Into<String>,
        attempt: u32,
    ) -> Result<Self, SubagentControlError> {
        let run_id = run_id.into();
        let identity = Self {
            control_scope_id: Some(run_id.clone()),
            run_id: Some(run_id),
            task_id: task_id.into(),
            execution_id: execution_id.into(),
            attempt,
        };
        identity.validate()?;
        Ok(identity)
    }

    /**
     * 构造一个绑定到指定运行时作用域（control scope）和运行（run）的
     * 执行尝试身份标识。
     */
    pub fn for_runtime_scope(
        control_scope_id: impl Into<String>,
        run_id: impl Into<String>,
        task_id: impl Into<String>,
        execution_id: impl Into<String>,
        attempt: u32,
    ) -> Result<Self, SubagentControlError> {
        let identity = Self {
            run_id: Some(run_id.into()),
            control_scope_id: Some(control_scope_id.into()),
            task_id: task_id.into(),
            execution_id: execution_id.into(),
            attempt,
        };
        identity.validate()?;
        Ok(identity)
    }

    /// 校验各字段是否合法（非空等）。
    fn validate(&self) -> Result<(), SubagentControlError> {
        if self
            .run_id
            .as_ref()
            .is_some_and(|run_id| run_id.trim().is_empty())
        {
            return Err(SubagentControlError::InvalidIdentity { field: "run_id" });
        }
        if self
            .control_scope_id
            .as_ref()
            .is_some_and(|scope_id| scope_id.trim().is_empty())
        {
            return Err(SubagentControlError::InvalidIdentity {
                field: "control_scope_id",
            });
        }
        if self.task_id.trim().is_empty() {
            return Err(SubagentControlError::InvalidIdentity { field: "task_id" });
        }
        if self.execution_id.trim().is_empty() {
            return Err(SubagentControlError::InvalidIdentity {
                field: "execution_id",
            });
        }
        if self.attempt == 0 {
            return Err(SubagentControlError::InvalidIdentity { field: "attempt" });
        }
        Ok(())
    }

    /// 返回用于按 (作用域, 运行, 任务, 尝试次数) 索引的复合键。
    fn task_attempt(&self) -> ScopedTaskAttemptKey {
        (
            self.control_scope_id.clone(),
            self.run_id.clone(),
            self.task_id.clone(),
            self.attempt,
        )
    }
}

/// 在中断请求发起前或正在结算过程中观察到的控制阶段。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubagentControlPhase {
    /// 已预留（reserve）但尚未绑定实时 Agent。
    Starting,
    /// 已绑定实时 Agent，正在运行。
    Running,
    /// 已收到中断请求，等待结算。
    InterruptRequested,
    /// 已结算（终态）。
    Settled,
}

/**
 * 一个 Subagent 命令的持久化生命周期阶段。
 *
 * 此阶段与描述“进程内执行器绑定”的 [`SubagentControlPhase`] 是不同的概念。
 * 它从命令被持久化写入开始，到所属回合（turn）结算为止，因此使用方无需
 * 再发明另一套生命周期枚举，就能对外暴露一份完整的命令回执。
 */
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SubagentCommandPhase {
    /// 命令已持久化写入。
    Persisted,
    /// 命令已被邮箱（mailbox）接受。
    MailboxAccepted,
    /// 命令已被消费（drain）进入模型上下文。
    Drained,
    /// 所属回合已结算。
    TurnSettled,
}

impl SubagentCommandPhase {
    /// 返回该阶段对应的线路（wire）字符串表示。
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Persisted => "persisted",
            Self::MailboxAccepted => "mailbox_accepted",
            Self::Drained => "drained",
            Self::TurnSettled => "turn_settled",
        }
    }

    /// 从线路（wire）字符串解析出对应的阶段；无法识别时返回 `None`。
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "persisted" => Some(Self::Persisted),
            "mailbox_accepted" => Some(Self::MailboxAccepted),
            "drained" => Some(Self::Drained),
            "turn_settled" => Some(Self::TurnSettled),
            _ => None,
        }
    }
}

/**
 * 活跃 Agent 邮箱（mailbox）接受的一条实时消息、绑定到精确执行尝试的信封。
 * 内嵌的框架回执（receipt）是生命周期状态的唯一权威来源。
 */
#[derive(Debug, Clone)]
pub struct SubagentMessageReceipt {
    /// 本次执行的唯一标识。
    pub execution_id: String,
    /// 尝试次数。
    pub attempt: u32,
    /// 框架持有的、生命周期状态权威的跟踪回执。
    receipt: AgentSteerReceipt,
}

impl SubagentMessageReceipt {
    /// 该条精确消息对应的、由框架持有的跟踪回执。
    pub fn receipt(&self) -> &AgentSteerReceipt {
        &self.receipt
    }

    /// 消费本次投递，返回其框架持有的跟踪回执。
    pub fn into_receipt(self) -> AgentSteerReceipt {
        self.receipt
    }

    /// 等待消息进入模型上下文，或所属回合结算为止。
    pub async fn wait_for_drained(&mut self) -> AgentSteerState {
        self.receipt.wait_for_drained().await
    }

    /// 等待所属回合到达其类型化的终态结果。
    pub async fn wait_for_turn_settled(&mut self) -> AgentSteerState {
        self.receipt.wait_for_turn_settled().await
    }
}

/// 为某次未来的精确执行尝试排队等待的指导信息（guidance）回执。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubagentGuidanceQueueReceipt {
    /// 逻辑任务标识。
    pub task_id: String,
    /// 目标尝试次数。
    pub attempt: u32,
    /// 当前排队中的指导信息条数。
    pub queued_count: usize,
}

/// 某次精确执行尝试的中断请求完成结算后返回的结果。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubagentInterruptOutcome {
    /// 本次执行的唯一标识。
    pub execution_id: String,
    /// 尝试次数。
    pub attempt: u32,
    /// 是否是本次调用实际发起了中断请求。
    pub requested: bool,
    /// 是否已经结算完成。
    pub settled: bool,
    /// 发起中断请求前观察到的控制阶段。
    pub previous_status: SubagentControlPhase,
    /// 结算后的终态子代理状态；仅在已结算时为 `Some`。
    pub terminal_status: Option<SubagentStatus>,
}

/// 对某次精确中断请求的即时确认回执。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubagentInterruptRequestReceipt {
    /// 目标执行尝试的身份标识。
    pub identity: SubagentAttemptIdentity,
    /// 请求处理结果的处置方式。
    pub disposition: SubagentInterruptRequestDisposition,
}

/// 中断请求在完全结算之前、仅在进程本地生效的处置方式。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubagentInterruptRequestDisposition {
    /// 执行尝试尚未准入（admit），请求被排队等待。
    QueuedBeforeAdmission,
    /// 执行尝试处于预留（reserve）阶段，请求已记录。
    ReservedRequested,
    /// 执行尝试处于活跃状态，本次请求触发了取消。
    ActiveRequested,
    /// 执行尝试处于活跃状态，但此前已经请求过中断。
    ActiveAlreadyRequested,
    /// 执行尝试已经结算，附带其终态状态。
    AlreadySettled(SubagentStatus),
}

/// 来自进程作用域 Subagent 控制平面的类型化拒绝错误。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SubagentControlError {
    /// 身份标识中的某个字段不合法。
    InvalidIdentity {
        /// 不合法的字段名。
        field: &'static str,
    },
    /// 指导信息（instruction）为空。
    EmptyInstruction,
    /// 控制状态当前不可用（例如锁被污染）。
    StateUnavailable,
    /// 该执行标识对应的执行已存在，不能重复注册。
    DuplicateExecution {
        /// 冲突的执行标识。
        execution_id: String,
    },
    /// 执行身份标识不匹配。
    ExecutionIdentityMismatch {
        /// 期望的执行标识。
        expected: String,
        /// 实际收到的执行标识。
        actual: String,
    },
    /// 执行尝试的身份标识字段不匹配。
    AttemptIdentityMismatch {
        /// 相关的执行标识。
        execution_id: String,
        /// 不匹配的字段名。
        field: &'static str,
        /// 期望值。
        expected: String,
        /// 实际值。
        actual: String,
    },
    /// 该任务的这次尝试已经开始过。
    AttemptAlreadyStarted {
        /// 逻辑任务标识。
        task_id: String,
        /// 尝试次数。
        attempt: u32,
    },
    /// 未知的执行标识。
    UnknownExecution {
        /// 未知的执行标识。
        execution_id: String,
    },
    /// 尝试次数不匹配。
    AttemptMismatch {
        /// 相关的执行标识。
        execution_id: String,
        /// 期望的尝试次数。
        expected: u32,
        /// 实际的尝试次数。
        actual: u32,
    },
    /// 该执行尝试正在结算一次中断请求。
    InterruptPending {
        /// 相关的执行标识。
        execution_id: String,
        /// 尝试次数。
        attempt: u32,
    },
    /// 待处理中断请求的数量超出了容量上限。
    PendingCapacityExceeded {
        /// 容量上限。
        limit: usize,
    },
    /// 同一个任务尝试上出现了执行身份冲突。
    IdentityConflict {
        /// 逻辑任务标识。
        task_id: String,
        /// 尝试次数。
        attempt: u32,
        /// 期望的执行标识。
        expected_execution_id: String,
        /// 实际的执行标识。
        actual_execution_id: String,
    },
    /// 该执行尝试已经结算完成。
    AttemptSettled {
        /// 相关的执行标识。
        execution_id: String,
        /// 尝试次数。
        attempt: u32,
        /// 结算后的终态状态。
        status: SubagentStatus,
    },
    /// 底层的实时引导（steer）操作被拒绝。
    Steer(AgentSteerError),
}

impl fmt::Display for SubagentControlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidIdentity { field } => {
                write!(f, "invalid Subagent identity field: {field}")
            }
            Self::EmptyInstruction => f.write_str("Subagent instruction must not be empty"),
            Self::StateUnavailable => f.write_str("Subagent control state is unavailable"),
            Self::DuplicateExecution { execution_id } => {
                write!(f, "Subagent execution already exists: {execution_id}")
            }
            Self::ExecutionIdentityMismatch { expected, actual } => write!(
                f,
                "Subagent execution identity mismatch: expected {expected}, actual {actual}"
            ),
            Self::AttemptIdentityMismatch {
                execution_id,
                field,
                expected,
                actual,
            } => write!(
                f,
                "Subagent execution {execution_id} identity field {field} mismatch: expected {expected}, actual {actual}"
            ),
            Self::AttemptAlreadyStarted { task_id, attempt } => {
                write!(
                    f,
                    "Subagent task {task_id} attempt {attempt} already started"
                )
            }
            Self::UnknownExecution { execution_id } => {
                write!(f, "unknown Subagent execution: {execution_id}")
            }
            Self::AttemptMismatch {
                execution_id,
                expected,
                actual,
            } => write!(
                f,
                "Subagent execution {execution_id} attempt mismatch: expected {expected}, actual {actual}"
            ),
            Self::InterruptPending {
                execution_id,
                attempt,
            } => write!(
                f,
                "Subagent execution {execution_id} attempt {attempt} is settling an interrupt"
            ),
            Self::PendingCapacityExceeded { limit } => {
                write!(
                    f,
                    "Subagent interrupt pending capacity exceeded (limit {limit})"
                )
            }
            Self::IdentityConflict {
                task_id,
                attempt,
                expected_execution_id,
                actual_execution_id,
            } => write!(
                f,
                "Subagent task {task_id} attempt {attempt} execution identity conflict: expected {expected_execution_id}, actual {actual_execution_id}"
            ),
            Self::AttemptSettled {
                execution_id,
                attempt,
                status,
            } => write!(
                f,
                "Subagent execution {execution_id} attempt {attempt} already settled as {}",
                status.as_str()
            ),
            Self::Steer(error) => write!(f, "Subagent steering rejected: {error}"),
        }
    }
}

impl std::error::Error for SubagentControlError {}

/// 精确执行尝试与其所属注册表之间的内部绑定句柄。
#[derive(Clone)]
pub(crate) struct SubagentAttemptBinding {
    registry: Arc<SubagentControlRegistry>,
    identity: SubagentAttemptIdentity,
}

impl SubagentAttemptBinding {
    /// 返回该绑定对应的执行尝试身份标识。
    pub(crate) fn identity(&self) -> &SubagentAttemptIdentity {
        &self.identity
    }

    /// 将实时 Agent 与所属回合绑定到该执行尝试上，返回引导租约（lease）。
    pub(crate) fn attach(
        &self,
        agent: Arc<dyn Agent>,
        turn_id: String,
    ) -> Result<SubagentSteeringLease, SubagentControlError> {
        self.registry.attach(&self.identity, agent, turn_id)?;
        Ok(SubagentSteeringLease {
            binding: self.clone(),
        })
    }
}

/**
 * 一次已被准入（admit）的执行尝试，携带其绑定句柄、待投递的排队指导信息
 * 以及取消令牌。
 */
pub(crate) struct SubagentAttemptAdmission {
    /// 该执行尝试与其所属注册表之间的绑定句柄。
    pub(crate) binding: SubagentAttemptBinding,
    /// 准入时消费掉的、此前为该尝试排队的指导信息。
    pub(crate) guidance: Vec<String>,
    /// 该执行尝试的取消令牌。
    pub(crate) cancel: CancellationToken,
    /// 是否已经结算完成（用于 Drop 时判断是否需要兜底结算）。
    settled: bool,
}

impl SubagentAttemptAdmission {
    /// 以给定终态状态结算该执行尝试。
    pub(crate) fn settle(mut self, status: SubagentStatus) {
        self.binding.registry.settle(&self.binding.identity, status);
        self.settled = true;
    }
}

impl Drop for SubagentAttemptAdmission {
    fn drop(&mut self) {
        if !self.settled {
            let status = if self.binding.registry.is_cancelled(&self.binding.identity) {
                SubagentStatus::Cancelled
            } else {
                SubagentStatus::Failed
            };
            self.binding.registry.settle(&self.binding.identity, status);
        }
    }
}

/// 实时引导（steering）租约；释放（drop）时自动从注册表上解绑。
pub(crate) struct SubagentSteeringLease {
    binding: SubagentAttemptBinding,
}

impl Drop for SubagentSteeringLease {
    fn drop(&mut self) {
        self.binding.registry.detach(&self.binding.identity);
    }
}

/// 一次正在进行中的执行尝试的进程内状态。
struct ActiveAttempt {
    identity: SubagentAttemptIdentity,
    cancel: CancellationToken,
    reserved: bool,
    phase: SubagentControlPhase,
    agent: Option<Arc<dyn Agent>>,
    turn_id: Option<String>,
    ready_tx: watch::Sender<bool>,
    settled_tx: watch::Sender<Option<SubagentStatus>>,
}

/// 已结算执行尝试的终态记录。
#[derive(Debug, Clone)]
struct SettledAttempt {
    identity: SubagentAttemptIdentity,
    status: SubagentStatus,
}

/// 用于索引 (控制作用域, 运行, 任务, 尝试次数) 的复合键类型。
type ScopedTaskAttemptKey = (Option<String>, Option<String>, String, u32);

/// 控制注册表的内部可变状态。
#[derive(Default)]
struct ControlState {
    active: HashMap<String, ActiveAttempt>,
    active_by_task_attempt: HashMap<ScopedTaskAttemptKey, String>,
    queued_guidance: HashMap<ScopedTaskAttemptKey, VecDeque<String>>,
    pending_interrupts: HashMap<String, SubagentAttemptIdentity>,
    settled: HashMap<String, SettledAttempt>,
    settled_by_task_attempt: HashMap<ScopedTaskAttemptKey, String>,
    settled_order: VecDeque<String>,
}

/**
 * 一次活跃执行尝试的只读摘要，数量有上限，供 `subagent_list` 等内建
 * 发现类工具使用。
 */
#[derive(Clone, Debug)]
pub struct ActiveAttemptSummary {
    /// 本次执行的唯一标识。
    pub execution_id: String,
    /// 逻辑任务标识。
    pub task_id: String,
    /// 尝试次数。
    pub attempt: u32,
    /// 当前控制阶段。
    pub phase: SubagentControlPhase,
}

/**
 * 进程作用域的执行控制器。持久化状态仍由使用方（consumer）负责管理。
 *
 * 按每个 `SubagentRegistry` 共享一份，使得同一个 `execution_id` 可以在
 * 基于该注册表构建的每一个执行器（executor）之间寻址
 * （参见 `SubagentRegistry::control_registry`）。
 */
#[derive(Default)]
pub struct SubagentControlRegistry {
    /// 内部可变状态（活跃/已结算执行尝试、排队指导信息等）。
    state: Mutex<ControlState>,
}

impl SubagentControlRegistry {
    /**
     * 在执行尝试等待共享执行容量之前，先为其预留（reserve）一个位置。
     * 之后调用 [`Self::admit`] 会消费这次预留并拿到同一个绑定句柄；
     * 不会再创建第二份身份标识或取消令牌。
     */
    pub(crate) fn reserve(
        self: &Arc<Self>,
        identity: SubagentAttemptIdentity,
        cancel: CancellationToken,
    ) -> Result<(), SubagentControlError> {
        identity.validate()?;
        let mut state = self
            .state
            .lock()
            .map_err(|_| SubagentControlError::StateUnavailable)?;
        if let Some(active) = state.active.get(&identity.execution_id) {
            Self::validate_identity(&active.identity, &identity)?;
            return Err(SubagentControlError::DuplicateExecution {
                execution_id: identity.execution_id,
            });
        }
        if let Some(settled) = state.settled.get(&identity.execution_id) {
            Self::validate_identity(&settled.identity, &identity)?;
            return Err(SubagentControlError::DuplicateExecution {
                execution_id: identity.execution_id,
            });
        }
        let task_attempt = identity.task_attempt();
        if state.active_by_task_attempt.contains_key(&task_attempt)
            || state.settled_by_task_attempt.contains_key(&task_attempt)
        {
            return Err(SubagentControlError::AttemptAlreadyStarted {
                task_id: identity.task_id,
                attempt: identity.attempt,
            });
        }
        if let Some(pending) = state.pending_interrupts.get(&identity.execution_id) {
            Self::validate_identity(pending, &identity)?;
        }
        if let Some(conflict) = state.pending_interrupts.values().find(|pending| {
            pending.control_scope_id == identity.control_scope_id
                && pending.run_id == identity.run_id
                && pending.task_id == identity.task_id
                && pending.attempt == identity.attempt
                && pending.execution_id != identity.execution_id
        }) {
            return Err(SubagentControlError::IdentityConflict {
                task_id: identity.task_id,
                attempt: identity.attempt,
                expected_execution_id: identity.execution_id,
                actual_execution_id: conflict.execution_id.clone(),
            });
        }
        let pending_interrupt = state.pending_interrupts.remove(&identity.execution_id);
        let (ready_tx, _) = watch::channel(false);
        let (settled_tx, _) = watch::channel(None);
        state
            .active_by_task_attempt
            .insert(task_attempt, identity.execution_id.clone());
        state.active.insert(
            identity.execution_id.clone(),
            ActiveAttempt {
                identity: identity.clone(),
                cancel: cancel.clone(),
                reserved: true,
                phase: SubagentControlPhase::Starting,
                agent: None,
                turn_id: None,
                ready_tx,
                settled_tx,
            },
        );
        drop(state);
        if pending_interrupt.is_some() {
            cancel.cancel();
        }
        Ok(())
    }

    /**
     * 将一次执行尝试准入（admit）为活跃状态，消费此前的预留（若存在），
     * 并返回其绑定句柄、待投递的排队指导信息与取消令牌。
     */
    pub(crate) fn admit(
        self: &Arc<Self>,
        identity: SubagentAttemptIdentity,
        cancel: CancellationToken,
    ) -> Result<SubagentAttemptAdmission, SubagentControlError> {
        identity.validate()?;
        let mut state = self
            .state
            .lock()
            .map_err(|_| SubagentControlError::StateUnavailable)?;
        if let Some(active) = state.active.get_mut(&identity.execution_id) {
            Self::validate_identity(&active.identity, &identity)?;
            if !active.reserved {
                return Err(SubagentControlError::DuplicateExecution {
                    execution_id: identity.execution_id,
                });
            }
            /* 该预留（reservation）持有子取消令牌的所有权。即使调用方为
             * 该身份提供了另一个令牌，也要使用预留时的令牌来完成派发。 */
            let cancel = active.cancel.clone();
            active.reserved = false;
            let guidance = state
                .queued_guidance
                .remove(&identity.task_attempt())
                .map(VecDeque::into_iter)
                .into_iter()
                .flatten()
                .collect();
            return Ok(SubagentAttemptAdmission {
                binding: SubagentAttemptBinding {
                    registry: Arc::clone(self),
                    identity,
                },
                guidance,
                cancel,
                settled: false,
            });
        }
        if let Some(settled) = state.settled.get(&identity.execution_id) {
            Self::validate_identity(&settled.identity, &identity)?;
            return Err(SubagentControlError::DuplicateExecution {
                execution_id: identity.execution_id,
            });
        }
        let task_attempt = identity.task_attempt();
        if state.active_by_task_attempt.contains_key(&task_attempt)
            || state.settled_by_task_attempt.contains_key(&task_attempt)
        {
            return Err(SubagentControlError::AttemptAlreadyStarted {
                task_id: identity.task_id,
                attempt: identity.attempt,
            });
        }
        if let Some(pending) = state.pending_interrupts.get(&identity.execution_id) {
            Self::validate_identity(pending, &identity)?;
        }
        let pending_interrupt = state.pending_interrupts.remove(&identity.execution_id);
        let guidance = state
            .queued_guidance
            .remove(&task_attempt)
            .map(VecDeque::into_iter)
            .into_iter()
            .flatten()
            .collect();
        let (ready_tx, _) = watch::channel(false);
        let (settled_tx, _) = watch::channel(None);
        state
            .active_by_task_attempt
            .insert(task_attempt, identity.execution_id.clone());
        state.active.insert(
            identity.execution_id.clone(),
            ActiveAttempt {
                identity: identity.clone(),
                cancel: cancel.clone(),
                reserved: false,
                phase: SubagentControlPhase::Starting,
                agent: None,
                turn_id: None,
                ready_tx,
                settled_tx,
            },
        );
        drop(state);
        if pending_interrupt.is_some() {
            cancel.cancel();
        }
        Ok(SubagentAttemptAdmission {
            binding: SubagentAttemptBinding {
                registry: Arc::clone(self),
                identity,
            },
            guidance,
            cancel,
            settled: false,
        })
    }

    /// 请求对某次精确执行尝试发起中断，不等待执行结算完成。
    pub(crate) fn request_interrupt(
        &self,
        identity: SubagentAttemptIdentity,
    ) -> Result<SubagentInterruptRequestReceipt, SubagentControlError> {
        identity.validate()?;
        let cancel: Option<CancellationToken> = {
            let mut state = self
                .state
                .lock()
                .map_err(|_| SubagentControlError::StateUnavailable)?;
            if let Some(settled) = state.settled.get(&identity.execution_id) {
                Self::validate_identity(&settled.identity, &identity)?;
                return Ok(SubagentInterruptRequestReceipt {
                    identity,
                    disposition: SubagentInterruptRequestDisposition::AlreadySettled(
                        settled.status,
                    ),
                });
            }
            if let Some(active) = state.active.get_mut(&identity.execution_id) {
                Self::validate_identity(&active.identity, &identity)?;
                let already_requested = active.phase == SubagentControlPhase::InterruptRequested;
                active.phase = SubagentControlPhase::InterruptRequested;
                let cancel = active.cancel.clone();
                let disposition = if active.reserved {
                    SubagentInterruptRequestDisposition::ReservedRequested
                } else if already_requested {
                    SubagentInterruptRequestDisposition::ActiveAlreadyRequested
                } else {
                    SubagentInterruptRequestDisposition::ActiveRequested
                };
                drop(state);
                if !already_requested {
                    cancel.cancel();
                }
                return Ok(SubagentInterruptRequestReceipt {
                    identity,
                    disposition,
                });
            }
            if let Some(pending) = state.pending_interrupts.get(&identity.execution_id) {
                Self::validate_identity(pending, &identity)?;
                return Ok(SubagentInterruptRequestReceipt {
                    identity,
                    disposition: SubagentInterruptRequestDisposition::QueuedBeforeAdmission,
                });
            }
            if let Some(conflict) = state.pending_interrupts.values().find(|pending| {
                pending.control_scope_id == identity.control_scope_id
                    && pending.run_id == identity.run_id
                    && pending.task_id == identity.task_id
                    && pending.attempt == identity.attempt
            }) {
                return Err(SubagentControlError::IdentityConflict {
                    task_id: identity.task_id,
                    attempt: identity.attempt,
                    expected_execution_id: identity.execution_id,
                    actual_execution_id: conflict.execution_id.clone(),
                });
            }
            if state.pending_interrupts.len() >= PENDING_INTERRUPT_CAPACITY {
                return Err(SubagentControlError::PendingCapacityExceeded {
                    limit: PENDING_INTERRUPT_CAPACITY,
                });
            }
            state
                .pending_interrupts
                .insert(identity.execution_id.clone(), identity.clone());
            None
        };
        if let Some(cancel) = cancel {
            cancel.cancel();
            return Ok(SubagentInterruptRequestReceipt {
                identity,
                disposition: SubagentInterruptRequestDisposition::ActiveRequested,
            });
        }
        Ok(SubagentInterruptRequestReceipt {
            identity,
            disposition: SubagentInterruptRequestDisposition::QueuedBeforeAdmission,
        })
    }

    /**
     * 在持久化的任务认领（claim）结算完成后，退役（retire）某个精确的
     * 待处理或已结算投影。活跃/预留中的投影会被取消，但仍会保留，
     * 直到权威派发结算完成为止。
     */
    pub(crate) fn retire_exact(
        &self,
        identity: &SubagentAttemptIdentity,
    ) -> Result<bool, SubagentControlError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| SubagentControlError::StateUnavailable)?;
        if let Some(current) = state.pending_interrupts.get(&identity.execution_id) {
            Self::validate_identity(current, identity)?;
            state.pending_interrupts.remove(&identity.execution_id);
            return Ok(true);
        }
        if let Some(active) = state.active.get_mut(&identity.execution_id) {
            Self::validate_identity(&active.identity, identity)?;
            active.phase = SubagentControlPhase::InterruptRequested;
            let cancel = active.cancel.clone();
            drop(state);
            cancel.cancel();
            return Err(SubagentControlError::InterruptPending {
                execution_id: identity.execution_id.clone(),
                attempt: identity.attempt,
            });
        }
        if let Some(settled) = state.settled.get(&identity.execution_id) {
            Self::validate_identity(&settled.identity, identity)?;
            let settled = state
                .settled
                .remove(&identity.execution_id)
                .ok_or(SubagentControlError::StateUnavailable)?;
            state
                .settled_by_task_attempt
                .remove(&settled.identity.task_attempt());
            state
                .settled_order
                .retain(|execution_id| execution_id != &identity.execution_id);
            return Ok(true);
        }
        Ok(false)
    }

    /**
     * 将进程本地的投影与仍归持久化 TaskClaim 所有的精确执行 ID 集合进行
     * 对账（reconcile）。过期的活跃/预留执行尝试会被取消，但仍会保留，
     * 直到其权威派发结算完成为止；过期的待处理与已结算投影可以立即移除。
     */
    pub(crate) fn reconcile_scope(
        &self,
        control_scope_id: &str,
        run_id: &str,
        current_execution_ids: &HashSet<String>,
    ) -> Result<(), SubagentControlError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| SubagentControlError::StateUnavailable)?;
        state.pending_interrupts.retain(|execution_id, identity| {
            identity.control_scope_id.as_deref() != Some(control_scope_id)
                || identity.run_id.as_deref() != Some(run_id)
                || current_execution_ids.contains(execution_id)
        });

        let stale_active = state
            .active
            .iter_mut()
            .filter_map(|(execution_id, active)| {
                if active.identity.control_scope_id.as_deref() != Some(control_scope_id)
                    || active.identity.run_id.as_deref() != Some(run_id)
                    || current_execution_ids.contains(execution_id)
                {
                    return None;
                }
                active.phase = SubagentControlPhase::InterruptRequested;
                Some(active.cancel.clone())
            })
            .collect::<Vec<_>>();

        let stale_settled = state
            .settled
            .keys()
            .filter(|execution_id| {
                state.settled.get(*execution_id).is_some_and(|settled| {
                    settled.identity.control_scope_id.as_deref() == Some(control_scope_id)
                        && settled.identity.run_id.as_deref() == Some(run_id)
                        && !current_execution_ids.contains(*execution_id)
                })
            })
            .cloned()
            .collect::<Vec<_>>();
        for execution_id in stale_settled {
            if let Some(settled) = state.settled.remove(&execution_id) {
                state
                    .settled_by_task_attempt
                    .remove(&settled.identity.task_attempt());
            }
            state
                .settled_order
                .retain(|retained| retained != &execution_id);
        }
        drop(state);
        for cancel in stale_active {
            cancel.cancel();
        }
        Ok(())
    }

    /// 为尚未开始的未来执行尝试排队一条指导信息（guidance）。
    pub(crate) fn queue_guidance(
        &self,
        task_id: &str,
        expected_next_attempt: u32,
        instruction: impl Into<String>,
    ) -> Result<SubagentGuidanceQueueReceipt, SubagentControlError> {
        let task_id = task_id.trim();
        if task_id.is_empty() {
            return Err(SubagentControlError::InvalidIdentity { field: "task_id" });
        }
        if expected_next_attempt == 0 {
            return Err(SubagentControlError::InvalidIdentity { field: "attempt" });
        }
        let instruction = instruction.into();
        if instruction.trim().is_empty() {
            return Err(SubagentControlError::EmptyInstruction);
        }
        let key = (None, None, task_id.to_string(), expected_next_attempt);
        let mut state = self
            .state
            .lock()
            .map_err(|_| SubagentControlError::StateUnavailable)?;
        if state.active_by_task_attempt.contains_key(&key)
            || state.settled_by_task_attempt.contains_key(&key)
        {
            return Err(SubagentControlError::AttemptAlreadyStarted {
                task_id: task_id.to_string(),
                attempt: expected_next_attempt,
            });
        }
        let guidance = state.queued_guidance.entry(key).or_default();
        guidance.push_back(instruction);
        Ok(SubagentGuidanceQueueReceipt {
            task_id: task_id.to_string(),
            attempt: expected_next_attempt,
            queued_count: guidance.len(),
        })
    }

    /**
     * 向指定的精确执行尝试投递一条实时消息，并返回可跟踪的回执；若该
     * 尝试尚未就绪（未绑定实时 Agent），则等待其就绪后再投递。
     */
    pub(crate) async fn send_message_tracked(
        &self,
        execution_id: &str,
        expected_attempt: u32,
        instruction: impl Into<String>,
    ) -> Result<SubagentMessageReceipt, SubagentControlError> {
        let instruction = instruction.into();
        if instruction.trim().is_empty() {
            return Err(SubagentControlError::EmptyInstruction);
        }
        loop {
            let mut ready_rx = {
                let state = self
                    .state
                    .lock()
                    .map_err(|_| SubagentControlError::StateUnavailable)?;
                let Some(active) = state.active.get(execution_id) else {
                    return Err(self.missing_execution_error(
                        &state,
                        execution_id,
                        expected_attempt,
                    ));
                };
                Self::validate_attempt(&active.identity, expected_attempt)?;
                if active.phase == SubagentControlPhase::InterruptRequested {
                    return Err(SubagentControlError::InterruptPending {
                        execution_id: execution_id.to_string(),
                        attempt: expected_attempt,
                    });
                }
                if let (Some(agent), Some(turn_id)) = (&active.agent, active.turn_id.as_deref()) {
                    let tracked = agent
                        .steer_input_tracked(Some(turn_id), Message::user(instruction.clone()))
                        .map_err(SubagentControlError::Steer)?;
                    return Ok(SubagentMessageReceipt {
                        execution_id: execution_id.to_string(),
                        attempt: expected_attempt,
                        receipt: tracked,
                    });
                }
                active.ready_tx.subscribe()
            };
            if ready_rx.changed().await.is_err() {
                /* 结算（settle）会丢弃就绪发送端。重新读取有界的终态记录，
                 * 并返回其精确结果。 */
                continue;
            }
        }
    }

    /**
     * 将一条仅排队投递的消息发送给 `execution_id` 下当前活跃的那次执行
     * 尝试，尝试次数在内部自动解析。
     *
     * 用于默认上行链路（父级 steer）和 `subagent_message`（同级 steer）
     * 场景：调用方只知道目标执行标识，但不知道其尝试次数。回执契约与
     * [`Self::send_message_tracked`] 保持一致。
     */
    pub(crate) async fn send_message_to_active(
        &self,
        execution_id: &str,
        instruction: impl Into<String>,
    ) -> Result<SubagentMessageReceipt, SubagentControlError> {
        let attempt = {
            let state = self
                .state
                .lock()
                .map_err(|_| SubagentControlError::StateUnavailable)?;
            let active = state.active.get(execution_id).ok_or_else(|| {
                SubagentControlError::UnknownExecution {
                    execution_id: execution_id.to_string(),
                }
            })?;
            active.identity.attempt
        };
        self.send_message_tracked(execution_id, attempt, instruction)
            .await
    }

    /**
     * 当前活跃执行尝试的快照，数量受 `limit` 限制（最多截断到 64 条），
     * 并按执行标识排序以保证输出稳定。
     */
    pub fn active_snapshot(&self, limit: usize) -> Vec<ActiveAttemptSummary> {
        let limit = limit.min(64);
        let Ok(state) = self.state.lock() else {
            return Vec::new();
        };
        let mut summaries: Vec<ActiveAttemptSummary> = state
            .active
            .values()
            .map(|active| ActiveAttemptSummary {
                execution_id: active.identity.execution_id.clone(),
                task_id: active.identity.task_id.clone(),
                attempt: active.identity.attempt,
                phase: active.phase,
            })
            .collect();
        summaries.sort_by(|left, right| left.execution_id.cmp(&right.execution_id));
        summaries.truncate(limit);
        summaries
    }

    /// 对某次精确执行尝试发起中断，并等待其结算完成后返回结果。
    pub(crate) async fn interrupt(
        &self,
        execution_id: &str,
        expected_attempt: u32,
    ) -> Result<SubagentInterruptOutcome, SubagentControlError> {
        let (previous_status, requested, cancel, mut settled_rx) = {
            let mut state = self
                .state
                .lock()
                .map_err(|_| SubagentControlError::StateUnavailable)?;
            if let Some(settled) = state.settled.get(execution_id) {
                Self::validate_attempt(&settled.identity, expected_attempt)?;
                return Ok(SubagentInterruptOutcome {
                    execution_id: execution_id.to_string(),
                    attempt: expected_attempt,
                    requested: false,
                    settled: true,
                    previous_status: SubagentControlPhase::Settled,
                    terminal_status: Some(settled.status),
                });
            }
            let active = state.active.get_mut(execution_id).ok_or_else(|| {
                SubagentControlError::UnknownExecution {
                    execution_id: execution_id.to_string(),
                }
            })?;
            Self::validate_attempt(&active.identity, expected_attempt)?;
            let previous_status = active.phase;
            let requested = active.phase != SubagentControlPhase::InterruptRequested;
            active.phase = SubagentControlPhase::InterruptRequested;
            (
                previous_status,
                requested,
                active.cancel.clone(),
                active.settled_tx.subscribe(),
            )
        };
        cancel.cancel();

        let terminal_status = loop {
            if let Some(status) = *settled_rx.borrow_and_update() {
                break status;
            }
            settled_rx
                .changed()
                .await
                .map_err(|_| SubagentControlError::StateUnavailable)?;
        };
        Ok(SubagentInterruptOutcome {
            execution_id: execution_id.to_string(),
            attempt: expected_attempt,
            requested,
            settled: true,
            previous_status,
            terminal_status: Some(terminal_status),
        })
    }

    /// 将实时 Agent 与回合 id 绑定到指定的活跃执行尝试上。
    fn attach(
        &self,
        identity: &SubagentAttemptIdentity,
        agent: Arc<dyn Agent>,
        turn_id: String,
    ) -> Result<(), SubagentControlError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| SubagentControlError::StateUnavailable)?;
        let active = state
            .active
            .get_mut(&identity.execution_id)
            .ok_or_else(|| SubagentControlError::UnknownExecution {
                execution_id: identity.execution_id.clone(),
            })?;
        Self::validate_identity(&active.identity, identity)?;
        active.agent = Some(agent);
        active.turn_id = Some(turn_id);
        active.ready_tx.send_replace(true);
        if active.phase == SubagentControlPhase::Starting {
            active.phase = SubagentControlPhase::Running;
        }
        Ok(())
    }

    /// 解除某个活跃执行尝试与实时 Agent 之间的绑定。
    fn detach(&self, identity: &SubagentAttemptIdentity) {
        let Ok(mut state) = self.state.lock() else {
            return;
        };
        let Some(active) = state.active.get_mut(&identity.execution_id) else {
            return;
        };
        if active.identity == *identity {
            active.agent = None;
            active.turn_id = None;
            active.ready_tx.send_replace(false);
            if active.phase == SubagentControlPhase::Running {
                active.phase = SubagentControlPhase::Starting;
            }
        }
    }

    /// 将一次活跃执行尝试以给定终态状态结算，并迁移进已结算集合。
    fn settle(&self, identity: &SubagentAttemptIdentity, status: SubagentStatus) {
        let Ok(mut state) = self.state.lock() else {
            return;
        };
        let Some(active) = state.active.remove(&identity.execution_id) else {
            return;
        };
        if active.identity != *identity {
            state.active.insert(identity.execution_id.clone(), active);
            return;
        }
        state.pending_interrupts.remove(&identity.execution_id);
        state
            .active_by_task_attempt
            .remove(&identity.task_attempt());
        active.settled_tx.send_replace(Some(status));
        state
            .settled_by_task_attempt
            .insert(identity.task_attempt(), identity.execution_id.clone());
        state.settled.insert(
            identity.execution_id.clone(),
            SettledAttempt {
                identity: identity.clone(),
                status,
            },
        );
        state.settled_order.push_back(identity.execution_id.clone());
        while state.settled_order.len() > SETTLED_RETENTION {
            let Some(old_execution_id) = state.settled_order.pop_front() else {
                break;
            };
            if let Some(old) = state.settled.remove(&old_execution_id) {
                state
                    .settled_by_task_attempt
                    .remove(&old.identity.task_attempt());
            }
        }
    }

    /// 判断指定执行尝试对应的取消令牌是否已被取消。
    fn is_cancelled(&self, identity: &SubagentAttemptIdentity) -> bool {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .active
            .get(&identity.execution_id)
            .is_some_and(|active| active.identity == *identity && active.cancel.is_cancelled())
    }

    /// 校验身份标识中的尝试次数是否与期望值一致。
    fn validate_attempt(
        identity: &SubagentAttemptIdentity,
        expected_attempt: u32,
    ) -> Result<(), SubagentControlError> {
        if identity.attempt != expected_attempt {
            return Err(SubagentControlError::AttemptMismatch {
                execution_id: identity.execution_id.clone(),
                expected: expected_attempt,
                actual: identity.attempt,
            });
        }
        Ok(())
    }

    /// 逐字段校验实际身份标识是否与期望的身份标识一致。
    fn validate_identity(
        expected: &SubagentAttemptIdentity,
        actual: &SubagentAttemptIdentity,
    ) -> Result<(), SubagentControlError> {
        Self::validate_attempt(expected, actual.attempt)?;
        Self::validate_identity_field(
            &expected.execution_id,
            "control_scope_id",
            expected.control_scope_id.as_deref(),
            actual.control_scope_id.as_deref(),
        )?;
        Self::validate_identity_field(
            &expected.execution_id,
            "run_id",
            expected.run_id.as_deref(),
            actual.run_id.as_deref(),
        )?;
        Self::validate_identity_field(
            &expected.execution_id,
            "task_id",
            Some(expected.task_id.as_str()),
            Some(actual.task_id.as_str()),
        )?;
        Self::validate_identity_field(
            &expected.execution_id,
            "execution_id",
            Some(expected.execution_id.as_str()),
            Some(actual.execution_id.as_str()),
        )
    }

    /// 比较单个身份字段的期望值与实际值，不一致时返回错误。
    fn validate_identity_field(
        execution_id: &str,
        field: &'static str,
        expected: Option<&str>,
        actual: Option<&str>,
    ) -> Result<(), SubagentControlError> {
        if expected != actual {
            return Err(SubagentControlError::AttemptIdentityMismatch {
                execution_id: execution_id.to_string(),
                field,
                expected: expected.unwrap_or("<none>").to_string(),
                actual: actual.unwrap_or("<none>").to_string(),
            });
        }
        Ok(())
    }

    /// 在找不到活跃执行时，结合已结算记录构造更精确的错误信息。
    fn missing_execution_error(
        &self,
        state: &ControlState,
        execution_id: &str,
        expected_attempt: u32,
    ) -> SubagentControlError {
        if let Some(settled) = state.settled.get(execution_id) {
            if settled.identity.attempt != expected_attempt {
                return SubagentControlError::AttemptMismatch {
                    execution_id: execution_id.to_string(),
                    expected: expected_attempt,
                    actual: settled.identity.attempt,
                };
            }
            return SubagentControlError::AttemptSettled {
                execution_id: execution_id.to_string(),
                attempt: expected_attempt,
                status: settled.status,
            };
        }
        SubagentControlError::UnknownExecution {
            execution_id: execution_id.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use echo_core::agent::AgentEvent;
    use echo_core::agent::AgentSteerTurnOutcome;
    use echo_core::error::Result as AgentResult;
    use futures::future::BoxFuture;
    use futures::stream::BoxStream;

    struct SteerTestAgent {
        active_turn: Mutex<Option<String>>,
        messages: Mutex<Vec<String>>,
        lifecycle: Mutex<Option<watch::Sender<AgentSteerState>>>,
    }

    impl SteerTestAgent {
        fn new(turn_id: &str) -> Self {
            Self {
                active_turn: Mutex::new(Some(turn_id.to_string())),
                messages: Mutex::new(Vec::new()),
                lifecycle: Mutex::new(None),
            }
        }

        fn mark_drained(&self) {
            if let Some(sender) = self
                .lifecycle
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .as_ref()
            {
                let _ = sender.send(AgentSteerState::Drained);
            }
        }

        fn settle(&self, outcome: AgentSteerTurnOutcome) {
            if let Some(sender) = self
                .lifecycle
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .as_ref()
            {
                let _ = sender.send(AgentSteerState::TurnSettled {
                    outcome,
                    drained: matches!(outcome, AgentSteerTurnOutcome::Completed),
                });
            }
        }
    }

    impl Agent for SteerTestAgent {
        fn name(&self) -> &str {
            "steer-test"
        }

        fn model_name(&self) -> &str {
            "test"
        }

        fn system_prompt(&self) -> &str {
            ""
        }

        fn execute<'a>(&'a self, _task: &'a str) -> BoxFuture<'a, AgentResult<String>> {
            Box::pin(async { Ok("done".to_string()) })
        }

        fn execute_stream<'a>(
            &'a self,
            _task: &'a str,
        ) -> BoxFuture<'a, AgentResult<BoxStream<'a, AgentResult<AgentEvent>>>> {
            Box::pin(async {
                Ok(Box::pin(futures::stream::once(async {
                    Ok(AgentEvent::FinalAnswer("done".to_string()))
                }))
                    as BoxStream<'a, AgentResult<AgentEvent>>)
            })
        }

        fn steer_input(
            &self,
            expected_turn_id: Option<&str>,
            message: Message,
        ) -> Result<String, AgentSteerError> {
            let active = self
                .active_turn
                .lock()
                .map_err(|_| AgentSteerError::StateUnavailable)?;
            let actual = active.as_deref().ok_or(AgentSteerError::NoActiveTurn)?;
            if let Some(expected) = expected_turn_id
                && expected != actual
            {
                return Err(AgentSteerError::TurnMismatch {
                    expected: expected.to_string(),
                    actual: actual.to_string(),
                });
            }
            let text = message
                .content
                .as_text()
                .filter(|text| !text.trim().is_empty())
                .ok_or(AgentSteerError::EmptyInput)?;
            self.messages
                .lock()
                .map_err(|_| AgentSteerError::StateUnavailable)?
                .push(text);
            Ok(actual.to_string())
        }

        fn steer_input_tracked(
            &self,
            expected_turn_id: Option<&str>,
            message: Message,
        ) -> Result<AgentSteerReceipt, AgentSteerError> {
            let turn_id = self.steer_input(expected_turn_id, message)?;
            let (sender, receiver) = watch::channel(AgentSteerState::Accepted);
            *self
                .lifecycle
                .lock()
                .map_err(|_| AgentSteerError::StateUnavailable)? = Some(sender);
            Ok(AgentSteerReceipt::new(
                uuid::Uuid::new_v4().to_string(),
                turn_id,
                receiver,
            ))
        }
    }

    #[test]
    fn queued_guidance_is_claimed_once_by_exact_attempt() -> Result<(), SubagentControlError> {
        let registry = Arc::new(SubagentControlRegistry::default());
        let first = registry.queue_guidance("task-1", 2, "inspect the current diff")?;
        let second = registry.queue_guidance("task-1", 2, "run the focused test")?;
        assert_eq!(first.queued_count, 1);
        assert_eq!(second.queued_count, 2);

        let identity = SubagentAttemptIdentity::new("task-1", "execution-2", 2)?;
        let admission = registry.admit(identity, CancellationToken::new())?;
        assert_eq!(
            admission.guidance,
            vec![
                "inspect the current diff".to_string(),
                "run the focused test".to_string()
            ]
        );
        assert!(matches!(
            registry.queue_guidance("task-1", 2, "late"),
            Err(SubagentControlError::AttemptAlreadyStarted { .. })
        ));
        admission.settle(SubagentStatus::Completed);
        Ok(())
    }

    #[test]
    fn invalid_and_empty_inputs_are_rejected() {
        assert!(matches!(
            SubagentAttemptIdentity::new("", "execution", 1),
            Err(SubagentControlError::InvalidIdentity { field: "task_id" })
        ));
        let registry = SubagentControlRegistry::default();
        assert_eq!(
            registry.queue_guidance("task", 1, "  "),
            Err(SubagentControlError::EmptyInstruction)
        );
    }

    #[test]
    fn attempt_identity_round_trips_as_a_framework_value() -> Result<(), String> {
        let identity = SubagentAttemptIdentity::new("task-1", "execution-1", 2)
            .map_err(|error| error.to_string())?;
        let encoded = serde_json::to_value(&identity).map_err(|error| error.to_string())?;
        let decoded: SubagentAttemptIdentity =
            serde_json::from_value(encoded).map_err(|error| error.to_string())?;
        assert_eq!(decoded, identity);
        Ok(())
    }

    #[test]
    fn attempt_identity_deserialization_rejects_invalid_values() {
        let invalid = serde_json::json!({
            "task_id": "",
            "execution_id": "execution-1",
            "attempt": 0,
        });
        assert!(serde_json::from_value::<SubagentAttemptIdentity>(invalid).is_err());
        let unknown = serde_json::json!({
            "task_id": "task-1",
            "execution_id": "execution-1",
            "attempt": 1,
            "extra": true,
        });
        assert!(serde_json::from_value::<SubagentAttemptIdentity>(unknown).is_err());
    }

    #[test]
    fn command_identity_round_trips_and_exposes_the_live_attempt() -> Result<(), String> {
        let identity =
            SubagentCommandIdentity::new("run-1", "task-1", "execution-1", 3, 2, "command-1")
                .map_err(|error| error.to_string())?;
        let encoded = serde_json::to_value(&identity).map_err(|error| error.to_string())?;
        let decoded: SubagentCommandIdentity =
            serde_json::from_value(encoded).map_err(|error| error.to_string())?;
        assert_eq!(decoded, identity);
        assert_eq!(
            decoded
                .attempt_identity()
                .map_err(|error| error.to_string())?,
            SubagentAttemptIdentity::for_run("run-1", "task-1", "execution-1", 2)
                .map_err(|error| error.to_string())?
        );
        assert!(SubagentCommandIdentity::new("run-1", "task-1", "execution-1", 1, 1, "").is_err());
        Ok(())
    }

    #[test]
    fn command_phase_wire_names_round_trip() {
        for phase in [
            SubagentCommandPhase::Persisted,
            SubagentCommandPhase::MailboxAccepted,
            SubagentCommandPhase::Drained,
            SubagentCommandPhase::TurnSettled,
        ] {
            assert_eq!(SubagentCommandPhase::parse(phase.as_str()), Some(phase));
        }
        assert_eq!(SubagentCommandPhase::parse("unknown"), None);
    }

    #[tokio::test]
    async fn live_message_is_exact_attempt_and_never_crosses_settlement() -> Result<(), String> {
        let registry = Arc::new(SubagentControlRegistry::default());
        let identity = SubagentAttemptIdentity::new("task", "execution-1", 1)
            .map_err(|error| error.to_string())?;
        let admission = registry
            .admit(identity.clone(), CancellationToken::new())
            .map_err(|error| error.to_string())?;
        let concrete = Arc::new(SteerTestAgent::new("turn-1"));
        let agent: Arc<dyn Agent> = concrete.clone();
        let steering = admission
            .binding
            .attach(agent, "turn-1".to_string())
            .map_err(|error| error.to_string())?;

        let mut delivered = registry
            .send_message_tracked("execution-1", 1, "first")
            .await
            .map_err(|error| error.to_string())?;
        assert_eq!(delivered.receipt().turn_id(), "turn-1");
        assert_eq!(delivered.receipt().state(), AgentSteerState::Accepted);
        concrete.mark_drained();
        assert_eq!(delivered.wait_for_drained().await, AgentSteerState::Drained);
        assert!(matches!(
            registry
                .send_message_tracked("execution-1", 2, "stale")
                .await,
            Err(SubagentControlError::AttemptMismatch { .. })
        ));
        let messages = concrete
            .messages
            .lock()
            .map_err(|_| "message state unavailable".to_string())?
            .clone();
        assert_eq!(messages, vec!["first".to_string()]);

        concrete.settle(AgentSteerTurnOutcome::Completed);
        assert_eq!(
            delivered.wait_for_turn_settled().await,
            AgentSteerState::TurnSettled {
                outcome: AgentSteerTurnOutcome::Completed,
                drained: true,
            }
        );
        drop(steering);
        admission.settle(SubagentStatus::Completed);
        assert!(matches!(
            registry
                .send_message_tracked("execution-1", 1, "late")
                .await,
            Err(SubagentControlError::AttemptSettled { .. })
        ));
        assert_eq!(
            concrete
                .messages
                .lock()
                .map_err(|_| "message state unavailable".to_string())?
                .as_slice(),
            ["first".to_string()]
        );
        Ok(())
    }

    #[tokio::test]
    async fn live_message_waits_for_existing_attempt_safe_point() -> Result<(), String> {
        let registry = Arc::new(SubagentControlRegistry::default());
        let identity = SubagentAttemptIdentity::new("task", "execution-starting", 1)
            .map_err(|error| error.to_string())?;
        let admission = registry
            .admit(identity, CancellationToken::new())
            .map_err(|error| error.to_string())?;

        let delivery = registry.send_message_tracked("execution-starting", 1, "wait for ready");
        futures::pin_mut!(delivery);
        assert!(matches!(
            futures::poll!(delivery.as_mut()),
            std::task::Poll::Pending
        ));

        let concrete = Arc::new(SteerTestAgent::new("turn-starting"));
        let agent: Arc<dyn Agent> = concrete.clone();
        let steering = admission
            .binding
            .attach(agent, "turn-starting".to_string())
            .map_err(|error| error.to_string())?;
        let mut delivered = delivery.await.map_err(|error| error.to_string())?;
        assert_eq!(delivered.receipt().turn_id(), "turn-starting");
        concrete.mark_drained();
        assert_eq!(delivered.wait_for_drained().await, AgentSteerState::Drained);
        assert_eq!(
            concrete
                .messages
                .lock()
                .map_err(|_| "message state unavailable".to_string())?
                .as_slice(),
            ["wait for ready".to_string()]
        );
        concrete.settle(AgentSteerTurnOutcome::Completed);
        drop(steering);
        admission.settle(SubagentStatus::Completed);
        Ok(())
    }

    #[tokio::test]
    async fn interrupt_waits_for_exact_attempt_settlement() -> Result<(), String> {
        let registry = Arc::new(SubagentControlRegistry::default());
        let cancel = CancellationToken::new();
        let identity = SubagentAttemptIdentity::new("task", "execution-1", 1)
            .map_err(|error| error.to_string())?;
        let admission = registry
            .admit(identity, cancel.clone())
            .map_err(|error| error.to_string())?;
        let interrupt_registry = registry.clone();
        let interrupt =
            tokio::spawn(async move { interrupt_registry.interrupt("execution-1", 1).await });

        cancel.cancelled().await;
        admission.settle(SubagentStatus::Cancelled);
        let outcome = interrupt
            .await
            .map_err(|error| error.to_string())?
            .map_err(|error| error.to_string())?;
        assert!(outcome.requested);
        assert!(outcome.settled);
        assert_eq!(outcome.previous_status, SubagentControlPhase::Starting);
        assert_eq!(outcome.terminal_status, Some(SubagentStatus::Cancelled));

        let replay = registry
            .interrupt("execution-1", 1)
            .await
            .map_err(|error| error.to_string())?;
        assert!(!replay.requested);
        assert!(replay.settled);
        assert_eq!(replay.previous_status, SubagentControlPhase::Settled);
        Ok(())
    }

    #[tokio::test]
    async fn pending_interrupt_is_consumed_by_exact_admission() -> Result<(), String> {
        let registry = Arc::new(SubagentControlRegistry::default());
        let identity = SubagentAttemptIdentity::new("task", "execution-pending", 2)
            .map_err(|error| error.to_string())?;
        let receipt = registry
            .request_interrupt(identity.clone())
            .map_err(|error| error.to_string())?;
        assert_eq!(
            receipt.disposition,
            SubagentInterruptRequestDisposition::QueuedBeforeAdmission
        );
        let cancel = CancellationToken::new();
        let admission = registry
            .admit(identity.clone(), cancel.clone())
            .map_err(|error| error.to_string())?;
        assert!(cancel.is_cancelled());
        admission.settle(SubagentStatus::Cancelled);
        assert!(
            registry
                .retire_exact(&identity)
                .is_ok_and(|retired| retired)
        );
        assert!(
            registry
                .retire_exact(&identity)
                .is_ok_and(|retired| !retired)
        );
        Ok(())
    }

    #[test]
    fn reserved_attempt_is_consumed_by_dispatch_admission() -> Result<(), String> {
        let registry = Arc::new(SubagentControlRegistry::default());
        let identity = SubagentAttemptIdentity::new("task-reserved", "execution-reserved", 1)
            .map_err(|error| error.to_string())?;
        let cancel = CancellationToken::new();
        registry
            .reserve(identity.clone(), cancel.clone())
            .map_err(|error| error.to_string())?;

        let admission = registry
            .admit(identity.clone(), CancellationToken::new())
            .map_err(|error| error.to_string())?;
        assert!(!admission.cancel.is_cancelled());
        admission.settle(SubagentStatus::Completed);
        assert!(matches!(
            registry
                .request_interrupt(identity)
                .map_err(|error| error.to_string())?
                .disposition,
            SubagentInterruptRequestDisposition::AlreadySettled(SubagentStatus::Completed)
        ));
        Ok(())
    }

    #[test]
    fn reserved_interrupt_drops_as_cancelled_not_failed() -> Result<(), String> {
        let registry = Arc::new(SubagentControlRegistry::default());
        let identity = SubagentAttemptIdentity::new("task-reserved", "execution-cancelled", 1)
            .map_err(|error| error.to_string())?;
        let cancel = CancellationToken::new();
        registry
            .reserve(identity.clone(), cancel.clone())
            .map_err(|error| error.to_string())?;

        let receipt = registry
            .request_interrupt(identity.clone())
            .map_err(|error| error.to_string())?;
        assert_eq!(
            receipt.disposition,
            SubagentInterruptRequestDisposition::ReservedRequested
        );
        assert!(cancel.is_cancelled());
        let admission = registry
            .admit(identity.clone(), CancellationToken::new())
            .map_err(|error| error.to_string())?;
        drop(admission);
        assert!(matches!(
            registry
                .request_interrupt(identity)
                .map_err(|error| error.to_string())?
                .disposition,
            SubagentInterruptRequestDisposition::AlreadySettled(SubagentStatus::Cancelled)
        ));
        Ok(())
    }

    #[test]
    fn exact_attempt_control_rejects_cross_scope_identity() -> Result<(), String> {
        let registry = Arc::new(SubagentControlRegistry::default());
        let expected =
            SubagentAttemptIdentity::for_runtime_scope("scope-a", "run", "task", "execution", 1)
                .map_err(|error| error.to_string())?;
        let wrong_scope =
            SubagentAttemptIdentity::for_runtime_scope("scope-b", "run", "task", "execution", 1)
                .map_err(|error| error.to_string())?;
        let cancel = CancellationToken::new();
        registry
            .reserve(expected.clone(), cancel.clone())
            .map_err(|error| error.to_string())?;

        for result in [
            registry.reserve(wrong_scope.clone(), CancellationToken::new()),
            registry
                .admit(wrong_scope.clone(), CancellationToken::new())
                .map(|_| ()),
            registry.request_interrupt(wrong_scope.clone()).map(|_| ()),
            registry.retire_exact(&wrong_scope).map(|_| ()),
        ] {
            assert!(matches!(
                result,
                Err(SubagentControlError::AttemptIdentityMismatch {
                    field: "control_scope_id",
                    ..
                })
            ));
        }
        assert!(!cancel.is_cancelled());
        let admission = registry
            .admit(expected.clone(), CancellationToken::new())
            .map_err(|error| error.to_string())?;
        admission.settle(SubagentStatus::Completed);
        assert!(matches!(
            registry.request_interrupt(wrong_scope),
            Err(SubagentControlError::AttemptIdentityMismatch {
                field: "control_scope_id",
                ..
            })
        ));
        assert!(matches!(
            registry
                .request_interrupt(expected)
                .map_err(|error| error.to_string())?
                .disposition,
            SubagentInterruptRequestDisposition::AlreadySettled(SubagentStatus::Completed)
        ));
        Ok(())
    }

    #[test]
    fn reconciliation_isolates_runs_within_one_control_scope() -> Result<(), String> {
        let registry = Arc::new(SubagentControlRegistry::default());
        let run_a = SubagentAttemptIdentity::for_runtime_scope(
            "shared-scope",
            "run-a",
            "task",
            "execution-a",
            1,
        )
        .map_err(|error| error.to_string())?;
        let run_b = SubagentAttemptIdentity::for_runtime_scope(
            "shared-scope",
            "run-b",
            "task",
            "execution-b",
            1,
        )
        .map_err(|error| error.to_string())?;
        let cancel_a = CancellationToken::new();
        let cancel_b = CancellationToken::new();
        registry
            .reserve(run_a, cancel_a.clone())
            .map_err(|error| error.to_string())?;
        registry
            .reserve(run_b, cancel_b.clone())
            .map_err(|error| error.to_string())?;

        registry
            .reconcile_scope("shared-scope", "run-a", &HashSet::new())
            .map_err(|error| error.to_string())?;
        assert!(cancel_a.is_cancelled());
        assert!(!cancel_b.is_cancelled());
        Ok(())
    }

    #[test]
    fn reconciliation_retires_stale_pending_and_settled_and_cancels_active() -> Result<(), String> {
        let registry = Arc::new(SubagentControlRegistry::default());
        let pending = SubagentAttemptIdentity::for_run("run-a", "pending", "execution-pending", 1)
            .map_err(|error| error.to_string())?;
        registry
            .request_interrupt(pending.clone())
            .map_err(|error| error.to_string())?;

        let active = SubagentAttemptIdentity::for_run("run-a", "active", "execution-active", 1)
            .map_err(|error| error.to_string())?;
        let active_cancel = CancellationToken::new();
        registry
            .reserve(active.clone(), active_cancel.clone())
            .map_err(|error| error.to_string())?;

        let settled = SubagentAttemptIdentity::for_run("run-a", "settled", "execution-settled", 1)
            .map_err(|error| error.to_string())?;
        registry
            .admit(settled.clone(), CancellationToken::new())
            .map_err(|error| error.to_string())?
            .settle(SubagentStatus::Completed);

        let other_run =
            SubagentAttemptIdentity::for_run("run-b", "active", "execution-other-run", 1)
                .map_err(|error| error.to_string())?;
        let other_cancel = CancellationToken::new();
        registry
            .reserve(other_run.clone(), other_cancel.clone())
            .map_err(|error| error.to_string())?;
        let direct = SubagentAttemptIdentity::new("active", "execution-direct", 1)
            .map_err(|error| error.to_string())?;
        let direct_cancel = CancellationToken::new();
        registry
            .reserve(direct.clone(), direct_cancel.clone())
            .map_err(|error| error.to_string())?;
        let parent_scope = SubagentAttemptIdentity::for_runtime_scope(
            "parent-scope",
            "run-a",
            "active",
            "execution-parent-scope",
            1,
        )
        .map_err(|error| error.to_string())?;
        let parent_cancel = CancellationToken::new();
        registry
            .reserve(parent_scope.clone(), parent_cancel.clone())
            .map_err(|error| error.to_string())?;

        registry
            .reconcile_scope("run-a", "run-a", &HashSet::new())
            .map_err(|error| error.to_string())?;
        assert!(active_cancel.is_cancelled());
        assert!(!other_cancel.is_cancelled());
        assert!(!direct_cancel.is_cancelled());
        assert!(!parent_cancel.is_cancelled());
        assert_eq!(
            registry
                .request_interrupt(pending)
                .map_err(|error| error.to_string())?
                .disposition,
            SubagentInterruptRequestDisposition::QueuedBeforeAdmission
        );
        assert_eq!(
            registry
                .request_interrupt(settled)
                .map_err(|error| error.to_string())?
                .disposition,
            SubagentInterruptRequestDisposition::QueuedBeforeAdmission
        );
        let admission = registry
            .admit(active.clone(), CancellationToken::new())
            .map_err(|error| error.to_string())?;
        assert!(admission.cancel.is_cancelled());
        drop(admission);
        assert!(matches!(
            registry
                .request_interrupt(active)
                .map_err(|error| error.to_string())?
                .disposition,
            SubagentInterruptRequestDisposition::AlreadySettled(SubagentStatus::Cancelled)
        ));
        Ok(())
    }
}
