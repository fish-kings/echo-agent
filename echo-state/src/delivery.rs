/*!
与具体产品无关的持久化投递账本（delivery ledger）原语。

该账本对有序的投递生命周期建模，而不关心应用层的路由（route）、Agent、
工作区或负载（payload）具体代表什么。调用方提供自己的路由和负载类型，
框架则负责生命周期身份、尝试顺序、终态以及一个有界的可重放投影。
*/

use crate::journal::{
    ApplyBatchReceipt, ApplyReceipt, CheckpointStore, CheckpointedApplyError, CheckpointedReducer,
    EventJournal, EventReducer, JournalBatchAppendResult, JournalBatchLookup, JournalRecord,
    PreparedJournalBatch, RecoveryReceipt,
};
use chrono::{DateTime, Utc};
use echo_core::error::{ReactError, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::fmt;
use std::sync::Arc;

/// 框架投递事件契约的版本号。
pub const DELIVERY_LEDGER_SCHEMA_VERSION: u16 = 1;

/**
类型化投递账本所接受的路由值。

框架只需要一个稳定、可序列化并带有少量校验能力的路由。产品代码可以为
自己的地址类型实现此契约，而无需把该地址转换成框架自有的字符串类型。
*/
pub trait DeliveryRoute:
    Clone + PartialEq + Serialize + for<'de> Deserialize<'de> + Send + Sync + fmt::Debug + 'static
{
    /// 在路由进入持久化账本之前，拒绝不可用的路由。
    fn validate(&self) -> Result<()>;
}

impl DeliveryRoute for String {
    /// 字符串路由仅要求去除空白后非空。
    fn validate(&self) -> Result<()> {
        if self.trim().is_empty() {
            return Err(ReactError::Other("delivery route is empty".to_string()));
        }
        Ok(())
    }
}

/// 类型化投递账本所保留的负载值。
pub trait DeliveryPayload:
    Clone + PartialEq + Serialize + for<'de> Deserialize<'de> + Send + Sync + fmt::Debug + 'static
{
}

impl<T> DeliveryPayload for T where
    T: Clone
        + PartialEq
        + Serialize
        + for<'de> Deserialize<'de>
        + Send
        + Sync
        + fmt::Debug
        + 'static
{
}

/// 一次投递尝试或一条消息的最终结果。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeliveryOutcome {
    /// 已成功完成。
    Completed,
    /// 已失败。
    Failed,
    /// 已被取消。
    Cancelled,
    /// 已被丢弃。
    Dropped,
    /// 结果未知。
    OutcomeUnknown,
}

impl DeliveryOutcome {
    /// 用于日志和线上客户端的稳定 snake_case 拼写形式。
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
            Self::Dropped => "dropped",
            Self::OutcomeUnknown => "outcome_unknown",
        }
    }
}

/// 单条投递记录的持久化阶段。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeliveryPhase {
    /// 已持久化入队，尚未被认领。
    Persisted,
    /// 已被某次尝试认领。
    Claimed,
    /// 副作用（effect）已开始执行。
    EffectStarted,
    /// 输入已被邮箱（mailbox）接收。
    MailboxAccepted,
    /// 输入已被上下文消费（drained）。
    Drained,
    /// 已延后，等待下一次可尝试时间。
    Deferred,
    /// 本轮已结算（终态）。
    TurnSettled,
}

impl DeliveryPhase {
    /// 用于日志和线上客户端的稳定 snake_case 拼写形式。
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Persisted => "persisted",
            Self::Claimed => "claimed",
            Self::EffectStarted => "effect_started",
            Self::MailboxAccepted => "mailbox_accepted",
            Self::Drained => "drained",
            Self::Deferred => "deferred",
            Self::TurnSettled => "turn_settled",
        }
    }
}

/// 与具体产品无关的类型化消息信封。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeliveryEnvelope<Route = String, Payload = Value> {
    /// 由应用选定的稳定幂等身份。
    pub message_id: String,
    /// 调用方拥有的路由身份。
    pub route: Route,
    /// 调用方拥有的负载，保留原值而不经过 JSON 往返转换。
    pub payload: Payload,
    /// 用于关联追踪和调用方策略的稳定元数据。
    #[serde(default)]
    pub metadata: BTreeMap<String, String>,
    /// 可选的关联 ID，用于串联同一业务链路的消息。
    pub correlation_id: Option<String>,
    /// 可选的因果 ID，指向触发本消息的上游消息。
    pub causation_id: Option<String>,
}

impl<Route, Payload> DeliveryEnvelope<Route, Payload>
where
    Route: DeliveryRoute,
{
    /// 构造一个类型化信封，不复制路由或负载值。
    pub fn new(message_id: impl Into<String>, route: Route, payload: Payload) -> Self {
        Self {
            message_id: message_id.into(),
            route,
            payload,
            metadata: BTreeMap::new(),
            correlation_id: None,
            causation_id: None,
        }
    }

    /// 校验身份与路由，但不应用任何产品层策略。
    pub fn validate(&self) -> Result<()> {
        if self.message_id.trim().is_empty() {
            return Err(ReactError::Other(
                "delivery message_id is empty".to_string(),
            ));
        }
        self.route.validate()
    }
}

/// 单个信封的持久化生命周期事实。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "snake_case", tag = "kind")]
pub enum DeliveryEvent<Route = String, Payload = Value> {
    /// 信封已持久化入队。
    Persisted {
        /// 被持久化的信封。
        envelope: DeliveryEnvelope<Route, Payload>,
        /// 持久化时间。
        persisted_at: DateTime<Utc>,
    },
    /// 某次尝试认领了该消息。
    Claimed {
        /// 消息 ID。
        message_id: String,
        /// 本次尝试的唯一 ID。
        attempt_id: String,
        /// 尝试序号（从 1 开始递增）。
        attempt: u32,
        /// 认领时间。
        claimed_at: DateTime<Utc>,
    },
    /// 副作用已针对某个 turn 开始执行。
    EffectStarted {
        /// 消息 ID。
        message_id: String,
        /// 尝试 ID。
        attempt_id: String,
        /// 执行该副作用的 turn ID。
        turn_id: String,
        /// 开始时间。
        started_at: DateTime<Utc>,
    },
    /// 邮箱已接收该副作用的输入。
    MailboxAccepted {
        /// 消息 ID。
        message_id: String,
        /// 尝试 ID。
        attempt_id: String,
        /// 所属 turn ID。
        turn_id: String,
        /// 接收时间。
        accepted_at: DateTime<Utc>,
    },
    /// 输入已被上下文消费。
    Drained {
        /// 消息 ID。
        message_id: String,
        /// 尝试 ID。
        attempt_id: String,
        /// 所属 turn ID。
        turn_id: String,
        /// 消费时间。
        drained_at: DateTime<Utc>,
    },
    /// 尝试被延后。
    Deferred {
        /// 消息 ID。
        message_id: String,
        /// 尝试 ID。
        attempt_id: String,
        /// 延后原因。
        reason: String,
        /// 延后发生时间。
        deferred_at: DateTime<Utc>,
        /// 下一次可尝试时间。
        next_attempt_at: DateTime<Utc>,
    },
    /// 本轮已结算。
    TurnSettled {
        /// 消息 ID。
        message_id: String,
        /// 尝试 ID。
        attempt_id: String,
        /// 所属 turn ID（若有）。
        turn_id: Option<String>,
        /// 结算结果。
        outcome: DeliveryOutcome,
        /// 输入是否已被消费（`None` 表示保持原值）。
        drained: Option<bool>,
        /// 结算原因。
        reason: Option<String>,
        /// 是否可重试（为真时重新入队）。
        retryable: bool,
        /// 可重试时的下一次尝试时间。
        next_attempt_at: Option<DateTime<Utc>>,
        /// 回复消息 ID（若有）。
        reply_message_id: Option<String>,
        /// 结算时间。
        settled_at: DateTime<Utc>,
    },
}

/// 热投影中所保留终态记录的逻辑上限。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeliveryLedgerConfig {
    /// 最多保留的终态记录条数。
    pub terminal_retention: usize,
    /// 终态记录最多占用的字节数。
    pub terminal_retention_bytes: usize,
}

impl Default for DeliveryLedgerConfig {
    /// 默认保留 256 条、256 KiB 的终态记录。
    fn default() -> Self {
        Self {
            terminal_retention: 256,
            terminal_retention_bytes: 256 * 1024,
        }
    }
}

/// 一条投影后的投递记录。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeliveryRecord<Route = String, Payload = Value> {
    /// 消息 ID。
    pub message_id: String,
    /// 路由。
    pub route: Route,
    /// 负载。
    pub payload: Payload,
    /// 元数据。
    #[serde(default)]
    pub metadata: BTreeMap<String, String>,
    /// 关联 ID。
    pub correlation_id: Option<String>,
    /// 因果 ID。
    pub causation_id: Option<String>,
    /// 持久化时间。
    pub persisted_at: DateTime<Utc>,
    /// 当前生命周期阶段。
    pub phase: DeliveryPhase,
    /// 结算结果（若已结算）。
    pub outcome: Option<DeliveryOutcome>,
    /// 输入是否已被消费。
    pub drained: bool,
    /// 最近一次延后或结算的原因。
    pub reason: Option<String>,
    /// 当前尝试 ID。
    pub attempt_id: Option<String>,
    /// 当前尝试序号。
    pub attempt: u32,
    /// 认领时间。
    pub claimed_at: Option<DateTime<Utc>>,
    /// 副作用开始时间。
    pub effect_started_at: Option<DateTime<Utc>>,
    /// 邮箱接收时间。
    pub mailbox_accepted_at: Option<DateTime<Utc>>,
    /// 消费时间。
    pub drained_at: Option<DateTime<Utc>>,
    /// 结算时间。
    pub turn_settled_at: Option<DateTime<Utc>>,
    /// 当前 turn ID。
    pub turn_id: Option<String>,
    /// 回复消息 ID。
    pub reply_message_id: Option<String>,
    /// 下一次可尝试时间。
    pub next_attempt_at: Option<DateTime<Utc>>,
    /// 是否已处于终态。
    pub terminal: bool,
    /// 该记录计入保留配额的字节数（信封序列化后大小）。
    pub retained_bytes: usize,
}

impl<Route, Payload> DeliveryRecord<Route, Payload>
where
    Route: DeliveryRoute,
    Payload: DeliveryPayload,
{
    /// 由信封构造初始的 `Persisted` 记录，并计算其保留字节数。
    fn new(
        envelope: DeliveryEnvelope<Route, Payload>,
        persisted_at: DateTime<Utc>,
    ) -> std::result::Result<Self, String> {
        let retained_bytes = serde_json::to_vec(&envelope)
            .map_err(|error| format!("delivery envelope serialization failed: {error}"))?
            .len();
        Ok(Self {
            message_id: envelope.message_id,
            route: envelope.route,
            payload: envelope.payload,
            metadata: envelope.metadata,
            correlation_id: envelope.correlation_id,
            causation_id: envelope.causation_id,
            persisted_at,
            phase: DeliveryPhase::Persisted,
            outcome: None,
            drained: false,
            reason: None,
            attempt_id: None,
            attempt: 0,
            claimed_at: None,
            effect_started_at: None,
            mailbox_accepted_at: None,
            drained_at: None,
            turn_settled_at: None,
            turn_id: None,
            reply_message_id: None,
            next_attempt_at: None,
            terminal: false,
            retained_bytes,
        })
    }

    /// 判断记录是否对应指定消息的指定尝试。
    fn matches_attempt(&self, message_id: &str, attempt_id: &str) -> bool {
        self.message_id == message_id && self.attempt_id.as_deref() == Some(attempt_id)
    }

    /// 当调用方需要事件形态时，重建事件信封。
    pub fn envelope(&self) -> DeliveryEnvelope<Route, Payload> {
        DeliveryEnvelope {
            message_id: self.message_id.clone(),
            route: self.route.clone(),
            payload: self.payload.clone(),
            metadata: self.metadata.clone(),
            correlation_id: self.correlation_id.clone(),
            causation_id: self.causation_id.clone(),
        }
    }
}

/// 从 FIFO 前沿中选出的一次确切投递尝试。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeliveryClaim<Route = String, Payload = Value> {
    /// 从 FIFO 前沿中选出的、由调用方拥有的负载。
    pub payload: Payload,
    /// 消息 ID。
    pub message_id: String,
    /// 路由。
    pub route: Route,
    /// 尝试 ID。
    pub attempt_id: String,
    /// 尝试序号。
    pub attempt: u32,
    /// 认领时间。
    pub claimed_at: DateTime<Utc>,
}

/**
一次认领及其在物理提交前已校验过的生命周期事件。

具有自定义持久化能力的宿主可以通过 [`DeliveryLedger::apply_prepared_with`]
追加 `event`，然后把 `claim` 返回给自己的调用方。框架已经选定了 FIFO 条目
并分配了确切的尝试身份。
*/
#[derive(Debug, Clone, PartialEq)]
pub struct DeliveryClaimDraft<Route = String, Payload = Value> {
    /// 认领信息。
    pub claim: DeliveryClaim<Route, Payload>,
    /// 对应的已校验认领事件。
    pub event: DeliveryEvent<Route, Payload>,
}

/// 对副作用已开始的尝试进行对账所需的持久化状态。
#[derive(Debug, Clone, PartialEq)]
pub struct DeliveryInFlight<Route = String, Payload = Value> {
    /// 认领信息。
    pub claim: DeliveryClaim<Route, Payload>,
    /// 当前阶段。
    pub phase: DeliveryPhase,
    /// 副作用是否已开始。
    pub effect_started: bool,
    /// 所属 turn ID。
    pub turn_id: String,
}

/// 一次投递尝试的终态事实。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeliverySettlement {
    /// 所属 turn ID。
    pub turn_id: Option<String>,
    /// 结算结果。
    pub outcome: DeliveryOutcome,
    /// 输入是否已被消费。
    pub drained: Option<bool>,
    /// 结算原因。
    pub reason: Option<String>,
    /// 是否可重试。
    pub retryable: bool,
    /// 下一次可尝试时间（仅可重试时有效）。
    pub next_attempt_at: Option<DateTime<Utc>>,
    /// 回复消息 ID。
    pub reply_message_id: Option<String>,
}

impl DeliverySettlement {
    /// 构造一个不会重新入队的终态结算。
    pub fn terminal(
        turn_id: Option<String>,
        outcome: DeliveryOutcome,
        drained: Option<bool>,
        reason: Option<String>,
        reply_message_id: Option<String>,
    ) -> Self {
        Self {
            turn_id,
            outcome,
            drained,
            reason,
            retryable: false,
            next_attempt_at: None,
            reply_message_id,
        }
    }

    /// 构造一个可重试的结算，并附带下一次可尝试时间。
    pub fn retry(
        turn_id: Option<String>,
        outcome: DeliveryOutcome,
        drained: Option<bool>,
        reason: Option<String>,
        next_attempt_at: DateTime<Utc>,
    ) -> Self {
        Self {
            turn_id,
            outcome,
            drained,
            reason,
            retryable: true,
            next_attempt_at: Some(next_attempt_at),
            reply_message_id: None,
        }
    }
}

/**
针对一次确切投递认领的单个生命周期转换。

应用可以把该值传入自己的物理日志权威，而无需再定义一个镜像框架生命周期的
枚举。该转换只描述框架层事实；路由策略、唤醒行为和对外回执仍属于应用关注点。
*/
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum DeliveryTransition {
    /// 副作用开始。
    EffectStarted {
        /// 执行副作用的 turn ID。
        turn_id: String,
    },
    /// 邮箱已接收。
    MailboxAccepted {
        /// 所属 turn ID。
        turn_id: String,
    },
    /// 输入已消费。
    Drained {
        /// 所属 turn ID。
        turn_id: String,
    },
    /// 延后。
    Deferred {
        /// 延后原因。
        reason: String,
        /// 下一次可尝试时间。
        next_attempt_at: DateTime<Utc>,
    },
    /// 结算。
    Settled {
        /// 结算事实。
        settlement: DeliverySettlement,
    },
}

impl DeliveryTransition {
    /// 构造“副作用开始”转换。
    pub fn effect_started(turn_id: impl Into<String>) -> Self {
        Self::EffectStarted {
            turn_id: turn_id.into(),
        }
    }

    /// 构造“邮箱已接收”转换。
    pub fn mailbox_accepted(turn_id: impl Into<String>) -> Self {
        Self::MailboxAccepted {
            turn_id: turn_id.into(),
        }
    }

    /// 构造“输入已消费”转换。
    pub fn drained(turn_id: impl Into<String>) -> Self {
        Self::Drained {
            turn_id: turn_id.into(),
        }
    }

    /// 构造“延后”转换。
    pub fn deferred(reason: impl Into<String>, next_attempt_at: DateTime<Utc>) -> Self {
        Self::Deferred {
            reason: reason.into(),
            next_attempt_at,
        }
    }

    /// 构造“结算”转换。
    pub fn settled(settlement: DeliverySettlement) -> Self {
        Self::Settled { settlement }
    }
}

/// 投递账本可重建的 FIFO 与生命周期投影。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeliveryLedgerProjection<Route = String, Payload = Value> {
    /// 所有保留记录的 FIFO 顺序。
    order: VecDeque<String>,
    /// 非终态记录的 FIFO 前沿。
    frontier: VecDeque<String>,
    /// 按消息 ID 索引的记录。
    entries: HashMap<String, DeliveryRecord<Route, Payload>>,
    /// 终态记录累计占用的字节数。
    terminal_retained_bytes: usize,
    /// 事件折叠失败时记录的首个错误；一旦设置，投影即视为无效。
    invalid: Option<String>,
}

/// 判断候选记录当前是否可被认领：副作用未开始且已到可尝试时间。
fn claim_is_eligible<Route, Payload>(candidate: &DeliveryRecord<Route, Payload>) -> bool {
    candidate.effect_started_at.is_none()
        && !matches!(
            candidate.phase,
            DeliveryPhase::EffectStarted | DeliveryPhase::MailboxAccepted | DeliveryPhase::Drained
        )
        && candidate
            .next_attempt_at
            .is_none_or(|deadline| deadline <= Utc::now())
}

impl<Route, Payload> Default for DeliveryLedgerProjection<Route, Payload> {
    /// 创建空投影。
    fn default() -> Self {
        Self {
            order: VecDeque::new(),
            frontier: VecDeque::new(),
            entries: HashMap::new(),
            terminal_retained_bytes: 0,
            invalid: None,
        }
    }
}

impl<Route, Payload> EventReducer for DeliveryLedgerProjection<Route, Payload>
where
    Route: DeliveryRoute,
    Payload: DeliveryPayload,
{
    /// 投影折叠的事件类型。
    type Event = DeliveryEvent<Route, Payload>;

    /// 折叠事件；首次失败后记录错误并停止继续折叠。
    fn apply(&mut self, event: &Self::Event) {
        if self.invalid.is_none()
            && let Err(error) = self.apply_checked(event)
        {
            self.invalid = Some(error);
        }
    }

    /// 折叠日志记录中的事件。
    fn apply_record(&mut self, record: &JournalRecord<Self::Event>) {
        self.apply(&record.event);
    }
}

impl<Route, Payload> DeliveryLedgerProjection<Route, Payload>
where
    Route: DeliveryRoute,
    Payload: DeliveryPayload,
{
    /// 从保留的记录重新构建投影。
    pub fn from_records(
        records: impl IntoIterator<Item = DeliveryRecord<Route, Payload>>,
        config: DeliveryLedgerConfig,
    ) -> Result<Self> {
        let mut projection = Self::default();
        for record in records {
            record.route.validate()?;
            let message_id = record.message_id.clone();
            if projection.entries.contains_key(&message_id) {
                return Err(ReactError::Other(format!(
                    "duplicate delivery checkpoint record {message_id}"
                )));
            }
            projection.order.push_back(message_id.clone());
            if !record.terminal {
                projection.frontier.push_back(message_id.clone());
            } else {
                projection.terminal_retained_bytes = projection
                    .terminal_retained_bytes
                    .saturating_add(record.retained_bytes);
            }
            projection.entries.insert(message_id, record);
        }
        projection.validate(config)?;
        Ok(projection)
    }

    /// 按配置的条数和字节上限，从最早的终态记录开始淘汰。
    fn enforce_retention(&mut self, config: DeliveryLedgerConfig) {
        loop {
            let terminal_count = self.entries.values().filter(|entry| entry.terminal).count();
            if terminal_count <= config.terminal_retention
                && self.terminal_retained_bytes <= config.terminal_retention_bytes
            {
                return;
            }
            let Some(message_id) = self
                .order
                .iter()
                .find(|id| self.entries.get(*id).is_some_and(|entry| entry.terminal))
                .cloned()
            else {
                return;
            };
            self.order.retain(|id| id != &message_id);
            if let Some(entry) = self.entries.remove(&message_id) {
                self.terminal_retained_bytes = self
                    .terminal_retained_bytes
                    .saturating_sub(entry.retained_bytes);
            }
        }
    }

    /// 获取指定消息记录的可变引用，不存在时返回错误。
    fn entry_mut(
        &mut self,
        message_id: &str,
    ) -> std::result::Result<&mut DeliveryRecord<Route, Payload>, String> {
        self.entries
            .get_mut(message_id)
            .ok_or_else(|| format!("delivery message {message_id} is unknown"))
    }

    /// 校验事件是否针对记录当前的尝试，否则视为过期认领。
    fn check_attempt(
        entry: &DeliveryRecord<Route, Payload>,
        message_id: &str,
        attempt_id: &str,
    ) -> std::result::Result<(), String> {
        if !entry.matches_attempt(message_id, attempt_id) {
            return Err(format!("stale delivery claim for message {message_id}"));
        }
        Ok(())
    }

    /// 校验并折叠单个事件，违反生命周期约束时返回错误描述。
    fn apply_checked(
        &mut self,
        event: &DeliveryEvent<Route, Payload>,
    ) -> std::result::Result<(), String> {
        match event {
            DeliveryEvent::Persisted {
                envelope,
                persisted_at,
            } => {
                envelope.validate().map_err(|error| error.to_string())?;
                if self.entries.contains_key(&envelope.message_id) {
                    return Err(format!(
                        "duplicate delivery message {}",
                        envelope.message_id
                    ));
                }
                let record = DeliveryRecord::new(envelope.clone(), *persisted_at)?;
                let message_id = envelope.message_id.clone();
                self.order.push_back(message_id.clone());
                self.frontier.push_back(message_id.clone());
                self.entries.insert(message_id, record);
            }
            DeliveryEvent::Claimed {
                message_id,
                attempt_id,
                attempt,
                claimed_at,
            } => {
                if attempt_id.trim().is_empty() || *attempt == 0 {
                    return Err(format!("invalid claim identity for message {message_id}"));
                }
                let Some(position) = self.frontier.iter().position(|id| id == message_id) else {
                    return Err(format!(
                        "message {message_id} is missing from the delivery frontier"
                    ));
                };
                let predecessors_started = self.frontier.iter().take(position).all(|id| {
                    self.entries.get(id).is_some_and(|entry| {
                        entry.effect_started_at.is_some()
                            && matches!(
                                entry.phase,
                                DeliveryPhase::EffectStarted
                                    | DeliveryPhase::MailboxAccepted
                                    | DeliveryPhase::Drained
                            )
                    })
                });
                if !predecessors_started {
                    return Err(format!(
                        "message {message_id} cannot pass an unstarted FIFO predecessor"
                    ));
                }
                let entry = self.entry_mut(message_id)?;
                if entry.terminal
                    || !matches!(
                        entry.phase,
                        DeliveryPhase::Persisted | DeliveryPhase::Deferred
                    ) && !(entry.phase == DeliveryPhase::Claimed
                        && entry.effect_started_at.is_none())
                {
                    return Err(format!(
                        "message {message_id} cannot be claimed from {:?}",
                        entry.phase
                    ));
                }
                if *attempt <= entry.attempt {
                    return Err(format!(
                        "delivery attempt {} is not newer than {}",
                        attempt, entry.attempt
                    ));
                }
                entry.phase = DeliveryPhase::Claimed;
                entry.outcome = None;
                entry.drained = false;
                entry.reason = None;
                entry.attempt_id = Some(attempt_id.clone());
                entry.attempt = *attempt;
                entry.claimed_at = Some(*claimed_at);
                entry.effect_started_at = None;
                entry.mailbox_accepted_at = None;
                entry.drained_at = None;
                entry.turn_settled_at = None;
                entry.turn_id = None;
                entry.reply_message_id = None;
                entry.next_attempt_at = None;
            }
            DeliveryEvent::EffectStarted {
                message_id,
                attempt_id,
                turn_id,
                started_at,
            } => {
                if turn_id.trim().is_empty() {
                    return Err(format!(
                        "effect for message {message_id} has no turn identity"
                    ));
                }
                let entry = self.entry_mut(message_id)?;
                Self::check_attempt(entry, message_id, attempt_id)?;
                if entry.phase != DeliveryPhase::Claimed || entry.effect_started_at.is_some() {
                    return Err(format!(
                        "message {message_id} cannot start an effect from {:?}",
                        entry.phase
                    ));
                }
                entry.phase = DeliveryPhase::EffectStarted;
                entry.effect_started_at = Some(*started_at);
                entry.turn_id = Some(turn_id.clone());
            }
            DeliveryEvent::MailboxAccepted {
                message_id,
                attempt_id,
                turn_id,
                accepted_at,
            } => {
                let entry = self.entry_mut(message_id)?;
                Self::check_attempt(entry, message_id, attempt_id)?;
                if entry.phase != DeliveryPhase::EffectStarted
                    || entry.turn_id.as_deref() != Some(turn_id)
                {
                    return Err(format!(
                        "message {message_id} mailbox acceptance is out of order"
                    ));
                }
                entry.phase = DeliveryPhase::MailboxAccepted;
                entry.mailbox_accepted_at = Some(*accepted_at);
            }
            DeliveryEvent::Drained {
                message_id,
                attempt_id,
                turn_id,
                drained_at,
            } => {
                let entry = self.entry_mut(message_id)?;
                Self::check_attempt(entry, message_id, attempt_id)?;
                if entry.phase != DeliveryPhase::MailboxAccepted
                    || entry.turn_id.as_deref() != Some(turn_id)
                {
                    return Err(format!("message {message_id} drain is out of order"));
                }
                entry.phase = DeliveryPhase::Drained;
                entry.drained = true;
                entry.drained_at = Some(*drained_at);
            }
            DeliveryEvent::Deferred {
                message_id,
                attempt_id,
                reason,
                deferred_at: _deferred_at,
                next_attempt_at,
            } => {
                let entry = self.entry_mut(message_id)?;
                Self::check_attempt(entry, message_id, attempt_id)?;
                if !matches!(
                    entry.phase,
                    DeliveryPhase::Claimed | DeliveryPhase::EffectStarted
                ) || entry.mailbox_accepted_at.is_some()
                    || entry.drained
                {
                    return Err(format!(
                        "message {message_id} cannot defer from {:?}",
                        entry.phase
                    ));
                }
                entry.phase = DeliveryPhase::Deferred;
                entry.reason = Some(reason.clone());
                entry.next_attempt_at = Some(*next_attempt_at);
                entry.effect_started_at = None;
                entry.mailbox_accepted_at = None;
                entry.drained_at = None;
                entry.turn_id = None;
            }
            DeliveryEvent::TurnSettled {
                message_id,
                attempt_id,
                turn_id,
                outcome,
                drained,
                reason,
                retryable,
                next_attempt_at,
                reply_message_id,
                settled_at,
            } => {
                let retained_bytes = {
                    let entry = self.entry_mut(message_id)?;
                    Self::check_attempt(entry, message_id, attempt_id)?;
                    if entry.terminal
                        || !matches!(
                            entry.phase,
                            DeliveryPhase::Claimed
                                | DeliveryPhase::EffectStarted
                                | DeliveryPhase::MailboxAccepted
                                | DeliveryPhase::Drained
                        )
                    {
                        return Err(format!(
                            "message {message_id} cannot settle from {:?}",
                            entry.phase
                        ));
                    }
                    if let Some(turn_id) = turn_id
                        && entry.turn_id.as_deref() != Some(turn_id)
                    {
                        return Err(format!(
                            "message {message_id} settlement turn identity is stale"
                        ));
                    }
                    entry.outcome = Some(*outcome);
                    entry.drained = drained.unwrap_or(entry.drained);
                    entry.reason = reason.clone();
                    entry.turn_settled_at = Some(*settled_at);
                    entry.reply_message_id = reply_message_id.clone();
                    entry.next_attempt_at = *next_attempt_at;
                    if *retryable {
                        if next_attempt_at.is_none() {
                            return Err(format!(
                                "retryable message {message_id} has no next attempt time"
                            ));
                        }
                        entry.phase = DeliveryPhase::Deferred;
                        entry.effect_started_at = None;
                        entry.mailbox_accepted_at = None;
                        entry.drained_at = None;
                        entry.turn_id = None;
                        return Ok(());
                    }
                    entry.phase = DeliveryPhase::TurnSettled;
                    entry.terminal = true;
                    entry.retained_bytes
                };
                if !self.frontier.iter().any(|id| id == message_id) {
                    return Err(format!(
                        "message {message_id} is missing from delivery frontier"
                    ));
                }
                self.frontier.retain(|id| id != message_id);
                self.terminal_retained_bytes =
                    self.terminal_retained_bytes.saturating_add(retained_bytes);
            }
        }
        Ok(())
    }

    /// 校验投影不变式以及逻辑保留上限。
    pub fn validate(&self, config: DeliveryLedgerConfig) -> Result<()> {
        if let Some(error) = &self.invalid {
            return Err(ReactError::Other(error.clone()));
        }
        if self.entries.len() != self.order.len() {
            return Err(ReactError::Other(
                "delivery order and entry counts differ".to_string(),
            ));
        }
        let mut ordered = HashSet::new();
        for message_id in &self.order {
            if !ordered.insert(message_id) {
                return Err(ReactError::Other(format!(
                    "delivery order duplicates {message_id}"
                )));
            }
            let entry = self.entries.get(message_id).ok_or_else(|| {
                ReactError::Other(format!("delivery order references unknown {message_id}"))
            })?;
            if entry.message_id != *message_id {
                return Err(ReactError::Other(format!(
                    "delivery order identity disagrees for {message_id}"
                )));
            }
            entry.route.validate()?;
            validate_record_state(entry)?;
        }
        if ordered.len() != self.entries.len() {
            return Err(ReactError::Other(
                "delivery entries contain an id absent from order".to_string(),
            ));
        }

        let mut seen = HashSet::new();
        for message_id in &self.frontier {
            if !seen.insert(message_id) {
                return Err(ReactError::Other(format!(
                    "delivery frontier duplicates {message_id}"
                )));
            }
            let entry = self.entries.get(message_id).ok_or_else(|| {
                ReactError::Other(format!("delivery frontier references unknown {message_id}"))
            })?;
            if entry.terminal {
                return Err(ReactError::Other(format!(
                    "terminal delivery {message_id} remains in frontier"
                )));
            }
        }
        let expected_frontier = self
            .order
            .iter()
            .filter(|message_id| {
                self.entries
                    .get(*message_id)
                    .is_some_and(|entry| !entry.terminal)
            })
            .collect::<Vec<_>>();
        if expected_frontier != self.frontier.iter().collect::<Vec<_>>() {
            return Err(ReactError::Other(
                "delivery frontier order does not match non-terminal order".to_string(),
            ));
        }
        let expected_bytes = self
            .entries
            .values()
            .filter(|entry| entry.terminal)
            .map(|entry| entry.retained_bytes)
            .fold(0_usize, usize::saturating_add);
        if expected_bytes != self.terminal_retained_bytes {
            return Err(ReactError::Other(
                "delivery terminal byte accounting diverged".to_string(),
            ));
        }
        let terminal_count = self.entries.values().filter(|entry| entry.terminal).count();
        if terminal_count > config.terminal_retention
            || self.terminal_retained_bytes > config.terminal_retention_bytes
        {
            return Err(ReactError::Other(
                "delivery terminal retention exceeds configured bounds".to_string(),
            ));
        }
        Ok(())
    }

    /// 按顺序返回当前 FIFO 前沿。
    pub fn frontier(&self) -> impl Iterator<Item = &DeliveryRecord<Route, Payload>> {
        self.frontier.iter().filter_map(|id| self.entries.get(id))
    }

    /// 若该身份的记录仍被保留，则返回之。
    pub fn record(&self, message_id: &str) -> Option<&DeliveryRecord<Route, Payload>> {
        self.entries.get(message_id)
    }

    /// 按 FIFO 顺序返回所有保留的记录。
    pub fn records(&self) -> impl Iterator<Item = &DeliveryRecord<Route, Payload>> {
        self.order.iter().filter_map(|id| self.entries.get(id))
    }
}

/// 校验单条记录各字段与其阶段之间的一致性。
fn validate_record_state<Route, Payload>(entry: &DeliveryRecord<Route, Payload>) -> Result<()>
where
    Route: DeliveryRoute,
    Payload: DeliveryPayload,
{
    let invalid = |detail: &str| {
        ReactError::Other(format!(
            "delivery record {} is invalid: {detail}",
            entry.message_id
        ))
    };
    let has_claim = entry.attempt_id.is_some() && entry.claimed_at.is_some();
    if entry.attempt_id.is_some() != entry.claimed_at.is_some() {
        return Err(invalid("attempt identity and claimed_at must be paired"));
    }
    if entry.attempt_id.as_deref().is_some_and(str::is_empty) {
        return Err(invalid("attempt identity is empty"));
    }
    if entry.phase != DeliveryPhase::Persisted && !has_claim {
        return Err(invalid("non-persisted phase has no claim identity"));
    }
    if entry.phase == DeliveryPhase::Persisted && entry.attempt != 0 {
        return Err(invalid("persisted phase has a non-zero attempt"));
    }
    if entry.phase != DeliveryPhase::Persisted && entry.attempt == 0 {
        return Err(invalid("active phase has a zero attempt"));
    }

    match entry.phase {
        DeliveryPhase::Persisted => {
            if entry.terminal
                || entry.outcome.is_some()
                || entry.drained
                || entry.reason.is_some()
                || entry.next_attempt_at.is_some()
                || entry.effect_started_at.is_some()
                || entry.mailbox_accepted_at.is_some()
                || entry.drained_at.is_some()
                || entry.turn_settled_at.is_some()
                || entry.turn_id.is_some()
                || entry.reply_message_id.is_some()
            {
                return Err(invalid("persisted phase contains lifecycle state"));
            }
        }
        DeliveryPhase::Claimed => {
            if entry.terminal
                || entry.outcome.is_some()
                || entry.drained
                || entry.reason.is_some()
                || entry.next_attempt_at.is_some()
                || entry.effect_started_at.is_some()
                || entry.mailbox_accepted_at.is_some()
                || entry.drained_at.is_some()
                || entry.turn_settled_at.is_some()
                || entry.turn_id.is_some()
                || entry.reply_message_id.is_some()
            {
                return Err(invalid("claimed phase contains a later lifecycle state"));
            }
        }
        DeliveryPhase::EffectStarted => {
            if entry.terminal
                || entry.outcome.is_some()
                || entry.drained
                || entry.reason.is_some()
                || entry.next_attempt_at.is_some()
                || entry.effect_started_at.is_none()
                || entry.mailbox_accepted_at.is_some()
                || entry.drained_at.is_some()
                || entry.turn_settled_at.is_some()
                || entry.turn_id.as_deref().is_none_or(str::is_empty)
                || entry.reply_message_id.is_some()
            {
                return Err(invalid(
                    "effect-started phase has inconsistent lifecycle state",
                ));
            }
        }
        DeliveryPhase::MailboxAccepted => {
            if entry.terminal
                || entry.outcome.is_some()
                || entry.drained
                || entry.reason.is_some()
                || entry.next_attempt_at.is_some()
                || entry.effect_started_at.is_none()
                || entry.mailbox_accepted_at.is_none()
                || entry.drained_at.is_some()
                || entry.turn_settled_at.is_some()
                || entry.turn_id.as_deref().is_none_or(str::is_empty)
                || entry.reply_message_id.is_some()
            {
                return Err(invalid(
                    "mailbox-accepted phase has inconsistent lifecycle state",
                ));
            }
        }
        DeliveryPhase::Drained => {
            if entry.terminal
                || entry.outcome.is_some()
                || !entry.drained
                || entry.reason.is_some()
                || entry.next_attempt_at.is_some()
                || entry.effect_started_at.is_none()
                || entry.mailbox_accepted_at.is_none()
                || entry.drained_at.is_none()
                || entry.turn_settled_at.is_some()
                || entry.turn_id.as_deref().is_none_or(str::is_empty)
                || entry.reply_message_id.is_some()
            {
                return Err(invalid("drained phase has inconsistent lifecycle state"));
            }
        }
        DeliveryPhase::Deferred => {
            if entry.terminal
                || entry.next_attempt_at.is_none()
                || entry.effect_started_at.is_some()
                || entry.mailbox_accepted_at.is_some()
                || entry.drained_at.is_some()
                || entry.turn_id.is_some()
            {
                return Err(invalid("deferred phase has inconsistent retry state"));
            }
        }
        DeliveryPhase::TurnSettled => {
            if !entry.terminal
                || entry.outcome.is_none()
                || entry.turn_settled_at.is_none()
                || entry.next_attempt_at.is_some()
            {
                return Err(invalid("settled phase has inconsistent terminal state"));
            }
        }
    }
    if entry.phase == DeliveryPhase::TurnSettled && entry.terminal != (entry.outcome.is_some()) {
        return Err(invalid("terminal flag and outcome disagree"));
    }
    Ok(())
}

/**
基于日志（journal）的通用投递账本。

# 示例

框架使用方在整个生命周期中都保留自己的领域路由与负载类型。负载只需实现
`PartialEq`，因此像浮点字段这样的普通值依然可用。

```
use echo_state::delivery::{
    DeliveryEnvelope, DeliveryLedger, DeliveryLedgerConfig, DeliveryLedgerProjection,
    DeliveryOutcome, DeliveryRoute, DeliverySettlement, DeliveryEvent,
};
use echo_state::journal::{CheckpointStore, MemoryCheckpointStore, MemoryEventJournal};
use echo_core::error::{ReactError, Result};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Route(String);

impl DeliveryRoute for Route {
    fn validate(&self) -> Result<()> {
        if self.0.trim().is_empty() {
            return Err(ReactError::Other("route is empty".to_string()));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Payload {
    text: String,
    score: f64,
}

# fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
type Event = DeliveryEvent<Route, Payload>;
type Journal = MemoryEventJournal<Event>;
type Projection = DeliveryLedgerProjection<Route, Payload>;
type Ledger = DeliveryLedger<Journal, Route, Payload>;

let journal = Arc::new(Journal::new());
let checkpoints: Arc<dyn CheckpointStore<Projection>> =
    Arc::new(MemoryCheckpointStore::new());
let ledger = Ledger::new(
    journal,
    checkpoints,
    DeliveryLedgerConfig::default(),
    16,
);
ledger.enqueue(DeliveryEnvelope::new(
    "message-1",
    Route("conversation-1".to_string()),
    Payload { text: "hello".to_string(), score: 0.95 },
))?;
let claim = ledger
    .claim_next()?
    .ok_or_else(|| ReactError::Other("delivery frontier is empty".to_string()))?;
ledger.begin_effect(&claim, "turn-1")?;
ledger.accept_mailbox(&claim, "turn-1")?;
ledger.mark_drained(&claim, "turn-1")?;
ledger.settle(
    &claim,
    DeliverySettlement::terminal(
        Some("turn-1".to_string()),
        DeliveryOutcome::Completed,
        Some(true),
        None,
        None,
    ),
)?;
assert!(ledger.with_projection(|projection| projection.frontier().next().is_none()));
# Ok(())
# }
```
*/
pub struct DeliveryLedger<J, Route = String, Payload = Value> {
    /// 带检查点的投影归约器。
    reducer: CheckpointedReducer<J, DeliveryLedgerProjection<Route, Payload>>,
    /// 底层事件日志。
    journal: Arc<J>,
    /// 保留配置。
    config: DeliveryLedgerConfig,
    /// 串行化预检与提交操作的互斥锁。
    operation: std::sync::Mutex<()>,
}

/// 投递账本在日志提交之前或之后返回的错误。
#[derive(Debug)]
pub enum DeliveryLedgerError<Route = String, Payload = Value> {
    /// 事件违反当前投影，未被持久化。
    InvalidEvent {
        /// 被拒绝的事件。
        event: Box<DeliveryEvent<Route, Payload>>,
        /// 拒绝原因。
        error: String,
    },
    /// 预备批次违反当前投影，未被持久化。
    InvalidBatch {
        /// 被拒绝的批次。
        batch: Box<PreparedJournalBatch<DeliveryEvent<Route, Payload>>>,
        /// 拒绝原因。
        error: String,
    },
    /// 底层日志/检查点权威拒绝了追加操作。
    Apply(Box<CheckpointedApplyError<DeliveryEvent<Route, Payload>>>),
    /// 已提交的回执不满足投影不变式。
    CommittedInvariant {
        /// 已提交批次的 ID。
        batch_id: String,
        /// 违反的不变式描述。
        error: String,
    },
}

impl<Route, Payload> fmt::Display for DeliveryLedgerError<Route, Payload>
where
    Route: DeliveryRoute,
    Payload: DeliveryPayload,
{
    /// 输出错误的可读描述。
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidEvent { error, .. } => {
                write!(formatter, "invalid delivery event: {error}")
            }
            Self::InvalidBatch { batch, error } => {
                write!(
                    formatter,
                    "invalid delivery batch {}: {error}",
                    batch.batch_id()
                )
            }
            Self::Apply(error) => error.fmt(formatter),
            Self::CommittedInvariant { batch_id, error } => {
                write!(
                    formatter,
                    "committed delivery batch {batch_id} violated invariant: {error}"
                )
            }
        }
    }
}

impl<Route, Payload> std::error::Error for DeliveryLedgerError<Route, Payload>
where
    Route: DeliveryRoute,
    Payload: DeliveryPayload,
{
}

impl<Route, Payload> From<CheckpointedApplyError<DeliveryEvent<Route, Payload>>>
    for DeliveryLedgerError<Route, Payload>
{
    /// 把检查点应用错误包装为 `Apply` 变体。
    fn from(error: CheckpointedApplyError<DeliveryEvent<Route, Payload>>) -> Self {
        Self::Apply(Box::new(error))
    }
}

impl<J, Route, Payload> fmt::Debug for DeliveryLedger<J, Route, Payload> {
    /// 仅输出配置，省略内部归约器等字段。
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DeliveryLedger")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl<J, Route, Payload> DeliveryLedger<J, Route, Payload>
where
    Route: DeliveryRoute,
    Payload: DeliveryPayload,
    J: EventJournal<DeliveryEvent<Route, Payload>>,
{
    /// 基于已有的框架日志和检查点存储创建账本。
    pub fn new(
        journal: Arc<J>,
        checkpoints: Arc<dyn CheckpointStore<DeliveryLedgerProjection<Route, Payload>>>,
        config: DeliveryLedgerConfig,
        checkpoint_every: u64,
    ) -> Self {
        Self {
            reducer: CheckpointedReducer::new(Arc::clone(&journal), checkpoints, checkpoint_every),
            journal,
            config,
            operation: std::sync::Mutex::new(()),
        }
    }

    /// 从检查点和日志尾部恢复投影。
    pub fn recover(&self) -> Result<RecoveryReceipt> {
        let _operation = self
            .operation
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let receipt = self.reducer.recover()?;
        self.reducer
            .with_state_mut(|projection| projection.enforce_retention(self.config));
        self.validate()?;
        Ok(receipt)
    }

    /// 以当前时间入队一个类型化信封。
    pub fn enqueue(
        &self,
        envelope: DeliveryEnvelope<Route, Payload>,
    ) -> std::result::Result<ApplyReceipt, DeliveryLedgerError<Route, Payload>> {
        self.apply(self.prepare_enqueue(envelope)?)
    }

    /// 构建并预检一个入队事件，但不进行物理提交。
    pub fn prepare_enqueue(
        &self,
        envelope: DeliveryEnvelope<Route, Payload>,
    ) -> std::result::Result<DeliveryEvent<Route, Payload>, DeliveryLedgerError<Route, Payload>>
    {
        self.prepare_event(DeliveryEvent::Persisted {
            envelope,
            persisted_at: Utc::now(),
        })
    }

    /// 认领下一个可认领的 FIFO 投递，并为其分配尝试身份。
    pub fn claim_next(
        &self,
    ) -> std::result::Result<
        Option<DeliveryClaim<Route, Payload>>,
        DeliveryLedgerError<Route, Payload>,
    > {
        let Some(draft) = self.prepare_claim_next()? else {
            return Ok(None);
        };
        self.apply(draft.event)?;
        Ok(Some(draft.claim))
    }

    /**
    在所有副作用已开始的前驱之后，认领最早一个可认领的投递。

    这样既保持了 FIFO 的副作用准入顺序，又允许相互独立的副作用并发结算。
    需要严格同一时刻只有一个在途投递的使用方，应继续使用 [`Self::claim_next`]。
    */
    pub fn claim_next_available(
        &self,
    ) -> std::result::Result<
        Option<DeliveryClaim<Route, Payload>>,
        DeliveryLedgerError<Route, Payload>,
    > {
        let Some(draft) = self.prepare_claim_next_available()? else {
            return Ok(None);
        };
        self.apply(draft.event)?;
        Ok(Some(draft.claim))
    }

    /// 预备最早一个可并发准入的 FIFO 认领。
    pub fn prepare_claim_next_available(
        &self,
    ) -> std::result::Result<
        Option<DeliveryClaimDraft<Route, Payload>>,
        DeliveryLedgerError<Route, Payload>,
    > {
        let candidate = self.with_projection(|projection| {
            projection
                .frontier()
                /* 只有第一个未开始的记录可以被准入。被延后的前驱在到期前
                仍然占据 FIFO 前沿。 */
                .find(|candidate| candidate.effect_started_at.is_none())
                .filter(|candidate| claim_is_eligible(candidate))
                .cloned()
        });
        let Some(candidate) = candidate else {
            return Ok(None);
        };
        self.prepare_claim(candidate)
    }

    /// 选出并预检下一个可认领的 FIFO 认领，但不提交。
    pub fn prepare_claim_next(
        &self,
    ) -> std::result::Result<
        Option<DeliveryClaimDraft<Route, Payload>>,
        DeliveryLedgerError<Route, Payload>,
    > {
        let candidate = self.with_projection(|projection| projection.frontier().next().cloned());
        let Some(candidate) = candidate else {
            return Ok(None);
        };
        if !claim_is_eligible(&candidate) {
            return Ok(None);
        }
        self.prepare_claim(candidate)
    }

    /// 为候选记录分配新的尝试身份，并预检对应的认领事件。
    fn prepare_claim(
        &self,
        candidate: DeliveryRecord<Route, Payload>,
    ) -> std::result::Result<
        Option<DeliveryClaimDraft<Route, Payload>>,
        DeliveryLedgerError<Route, Payload>,
    > {
        let attempt = candidate.attempt.saturating_add(1);
        let attempt_id = uuid::Uuid::new_v4().to_string();
        let claimed_at = Utc::now();
        let event = self.prepare_event(DeliveryEvent::Claimed {
            message_id: candidate.message_id.clone(),
            attempt_id: attempt_id.clone(),
            attempt,
            claimed_at,
        })?;
        Ok(Some(DeliveryClaimDraft {
            claim: DeliveryClaim {
                payload: candidate.payload,
                message_id: candidate.message_id,
                route: candidate.route,
                attempt_id,
                attempt,
                claimed_at,
            },
            event,
        }))
    }

    /// 记录已认领的副作用已针对某个具体 turn 开始执行。
    pub fn begin_effect(
        &self,
        claim: &DeliveryClaim<Route, Payload>,
        turn_id: impl Into<String>,
    ) -> std::result::Result<ApplyReceipt, DeliveryLedgerError<Route, Payload>> {
        self.apply(self.prepare_begin_effect(claim, turn_id)?)
    }

    /// 为一次确切认领构建并预检“副作用开始”事件。
    pub fn prepare_begin_effect(
        &self,
        claim: &DeliveryClaim<Route, Payload>,
        turn_id: impl Into<String>,
    ) -> std::result::Result<DeliveryEvent<Route, Payload>, DeliveryLedgerError<Route, Payload>>
    {
        self.prepare_transition(claim, DeliveryTransition::effect_started(turn_id))
    }

    /// 记录副作用的输入已被邮箱接收。
    pub fn accept_mailbox(
        &self,
        claim: &DeliveryClaim<Route, Payload>,
        turn_id: impl Into<String>,
    ) -> std::result::Result<ApplyReceipt, DeliveryLedgerError<Route, Payload>> {
        self.apply(self.prepare_accept_mailbox(claim, turn_id)?)
    }

    /// 为一次确切认领构建并预检“邮箱已接收”事件。
    pub fn prepare_accept_mailbox(
        &self,
        claim: &DeliveryClaim<Route, Payload>,
        turn_id: impl Into<String>,
    ) -> std::result::Result<DeliveryEvent<Route, Payload>, DeliveryLedgerError<Route, Payload>>
    {
        self.prepare_transition(claim, DeliveryTransition::mailbox_accepted(turn_id))
    }

    /// 记录副作用的输入已被上下文消费。
    pub fn mark_drained(
        &self,
        claim: &DeliveryClaim<Route, Payload>,
        turn_id: impl Into<String>,
    ) -> std::result::Result<ApplyReceipt, DeliveryLedgerError<Route, Payload>> {
        self.apply(self.prepare_mark_drained(claim, turn_id)?)
    }

    /// 为一次确切认领构建并预检“已消费”事件。
    pub fn prepare_mark_drained(
        &self,
        claim: &DeliveryClaim<Route, Payload>,
        turn_id: impl Into<String>,
    ) -> std::result::Result<DeliveryEvent<Route, Payload>, DeliveryLedgerError<Route, Payload>>
    {
        self.prepare_transition(claim, DeliveryTransition::drained(turn_id))
    }

    /// 将尚未被接收的副作用延后到给定的重试时间。
    pub fn defer(
        &self,
        claim: &DeliveryClaim<Route, Payload>,
        reason: impl Into<String>,
        next_attempt_at: DateTime<Utc>,
    ) -> std::result::Result<ApplyReceipt, DeliveryLedgerError<Route, Payload>> {
        self.apply(self.prepare_defer(claim, reason, next_attempt_at)?)
    }

    /// 为一次确切认领构建并预检“延后”事件。
    pub fn prepare_defer(
        &self,
        claim: &DeliveryClaim<Route, Payload>,
        reason: impl Into<String>,
        next_attempt_at: DateTime<Utc>,
    ) -> std::result::Result<DeliveryEvent<Route, Payload>, DeliveryLedgerError<Route, Payload>>
    {
        self.prepare_transition(claim, DeliveryTransition::deferred(reason, next_attempt_at))
    }

    /// 结算一次已认领的投递，可选择重新入队以便重试。
    pub fn settle(
        &self,
        claim: &DeliveryClaim<Route, Payload>,
        settlement: DeliverySettlement,
    ) -> std::result::Result<ApplyReceipt, DeliveryLedgerError<Route, Payload>> {
        self.apply(self.prepare_settle(claim, settlement)?)
    }

    /// 为一次确切认领构建并预检终态事实。
    pub fn prepare_settle(
        &self,
        claim: &DeliveryClaim<Route, Payload>,
        settlement: DeliverySettlement,
    ) -> std::result::Result<DeliveryEvent<Route, Payload>, DeliveryLedgerError<Route, Payload>>
    {
        self.prepare_transition(claim, DeliveryTransition::settled(settlement))
    }

    /// 通过日志权威应用单个生命周期转换。
    pub fn transition(
        &self,
        claim: &DeliveryClaim<Route, Payload>,
        transition: DeliveryTransition,
    ) -> std::result::Result<ApplyReceipt, DeliveryLedgerError<Route, Payload>> {
        self.apply(self.prepare_transition(claim, transition)?)
    }

    /// 为一次确切认领构建并预检任意生命周期转换。
    pub fn prepare_transition(
        &self,
        claim: &DeliveryClaim<Route, Payload>,
        transition: DeliveryTransition,
    ) -> std::result::Result<DeliveryEvent<Route, Payload>, DeliveryLedgerError<Route, Payload>>
    {
        let event = match transition {
            DeliveryTransition::EffectStarted { turn_id } => DeliveryEvent::EffectStarted {
                message_id: claim.message_id.clone(),
                attempt_id: claim.attempt_id.clone(),
                turn_id,
                started_at: Utc::now(),
            },
            DeliveryTransition::MailboxAccepted { turn_id } => DeliveryEvent::MailboxAccepted {
                message_id: claim.message_id.clone(),
                attempt_id: claim.attempt_id.clone(),
                turn_id,
                accepted_at: Utc::now(),
            },
            DeliveryTransition::Drained { turn_id } => DeliveryEvent::Drained {
                message_id: claim.message_id.clone(),
                attempt_id: claim.attempt_id.clone(),
                turn_id,
                drained_at: Utc::now(),
            },
            DeliveryTransition::Deferred {
                reason,
                next_attempt_at,
            } => DeliveryEvent::Deferred {
                message_id: claim.message_id.clone(),
                attempt_id: claim.attempt_id.clone(),
                reason,
                deferred_at: Utc::now(),
                next_attempt_at,
            },
            DeliveryTransition::Settled { settlement } => DeliveryEvent::TurnSettled {
                message_id: claim.message_id.clone(),
                attempt_id: claim.attempt_id.clone(),
                turn_id: settlement.turn_id,
                outcome: settlement.outcome,
                drained: settlement.drained,
                reason: settlement.reason,
                retryable: settlement.retryable,
                next_attempt_at: settlement.next_attempt_at,
                reply_message_id: settlement.reply_message_id,
                settled_at: Utc::now(),
            },
        };
        self.prepare_event(event)
    }

    /// 在投影副本上试应用事件并校验，通过后返回该事件。
    fn prepare_event(
        &self,
        event: DeliveryEvent<Route, Payload>,
    ) -> std::result::Result<DeliveryEvent<Route, Payload>, DeliveryLedgerError<Route, Payload>>
    {
        let mut candidate = self.with_projection(Clone::clone);
        candidate.apply(&event);
        candidate.enforce_retention(self.config);
        candidate
            .validate(self.config)
            .map_err(|error| DeliveryLedgerError::InvalidEvent {
                event: Box::new(event.clone()),
                error: error.to_string(),
            })?;
        Ok(event)
    }

    /// 通过日志权威应用单个生命周期事件。
    pub fn apply(
        &self,
        event: DeliveryEvent<Route, Payload>,
    ) -> std::result::Result<ApplyReceipt, DeliveryLedgerError<Route, Payload>> {
        let batch = PreparedJournalBatch::new(vec![event]).map_err(|error| {
            DeliveryLedgerError::Apply(Box::new(CheckpointedApplyError::Prepare(error)))
        })?;
        let receipt = self.apply_prepared(batch)?;
        let sequence = receipt.first_sequence;
        if receipt.record_count != 1 || receipt.last_sequence != sequence {
            return Err(DeliveryLedgerError::CommittedInvariant {
                batch_id: receipt.batch_id,
                error: "single delivery event produced a non-single receipt".to_string(),
            });
        }
        Ok(ApplyReceipt {
            batch_id: receipt.batch_id,
            sequence,
            journal: receipt.journal,
            commit: receipt.commit,
            checkpoint: receipt.checkpoint,
        })
    }

    /// 应用一个已预校验的批次，同时保留其稳定身份，供调用方重试与对账。
    pub fn apply_prepared(
        &self,
        batch: PreparedJournalBatch<DeliveryEvent<Route, Payload>>,
    ) -> std::result::Result<ApplyBatchReceipt, DeliveryLedgerError<Route, Payload>> {
        self.apply_prepared_with(batch, |batch| self.journal.append_batch(batch))
    }

    /**
    使用调用方拥有的物理提交权威来应用预备批次。

    回调可以提供自定义的重试、重新打开或持久化处理，但必须为该确切的预备身份
    返回框架日志回执。投影预检、折叠、检查点、保留以及提交后校验仍由本账本负责。
    */
    pub fn apply_prepared_with<F>(
        &self,
        batch: PreparedJournalBatch<DeliveryEvent<Route, Payload>>,
        append: F,
    ) -> std::result::Result<ApplyBatchReceipt, DeliveryLedgerError<Route, Payload>>
    where
        F: FnOnce(
            PreparedJournalBatch<DeliveryEvent<Route, Payload>>,
        ) -> JournalBatchAppendResult<DeliveryEvent<Route, Payload>>,
    {
        let _operation = self
            .operation
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let preflight = self.reducer.with_state(|projection| {
            let mut candidate = projection.clone();
            for event in batch.events() {
                candidate.apply(event.as_ref());
            }
            candidate.enforce_retention(self.config);
            candidate.validate(self.config)
        });
        if let Err(error) = preflight {
            return Err(DeliveryLedgerError::InvalidBatch {
                batch: Box::new(batch),
                error: error.to_string(),
            });
        }
        let expected = batch.clone();
        let appended = append(batch).map_err(|error| {
            DeliveryLedgerError::Apply(Box::new(CheckpointedApplyError::Journal(error)))
        })?;
        if !expected.matches_receipt(&appended).map_err(|error| {
            DeliveryLedgerError::CommittedInvariant {
                batch_id: expected.batch_id().to_string(),
                error: error.to_string(),
            }
        })? {
            return Err(DeliveryLedgerError::CommittedInvariant {
                batch_id: expected.batch_id().to_string(),
                error: "physical journal receipt does not match the prepared delivery batch"
                    .to_string(),
            });
        }
        let receipt = self.reducer.apply_committed(&expected, appended)?;
        self.reducer
            .with_state_mut(|projection| projection.enforce_retention(self.config));
        if let Err(error) = self.validate() {
            return Err(DeliveryLedgerError::CommittedInvariant {
                batch_id: receipt.batch_id.clone(),
                error: error.to_string(),
            });
        }
        Ok(receipt)
    }

    /// 在不写入、不触发持久化的前提下，按预备身份进行对账查询。
    pub fn lookup_batch(
        &self,
        batch: &PreparedJournalBatch<DeliveryEvent<Route, Payload>>,
    ) -> Result<JournalBatchLookup<DeliveryEvent<Route, Payload>>> {
        self.journal.lookup_batch(batch)
    }

    /// 持久化当前投影状态，使调用方在应用逻辑保留后可以安全地裁剪旧的物理日志段。
    pub fn checkpoint(&self) -> Result<()> {
        self.reducer.checkpoint()
    }

    /// 读取当前投影，而不暴露归约器内部细节。
    pub fn with_projection<T>(
        &self,
        operation: impl FnOnce(&DeliveryLedgerProjection<Route, Payload>) -> T,
    ) -> T {
        self.reducer.with_state(operation)
    }

    /// 按配置的保留上限校验当前投影。
    pub fn validate(&self) -> Result<()> {
        self.reducer
            .with_state(|projection| projection.validate(self.config))
    }
}

/// 投递账本的单元测试。
#[cfg(test)]
mod tests {
    use super::*;
    use crate::journal::{MemoryCheckpointStore, MemoryEventJournal};

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    /// 测试用路由类型。
    struct TestRoute(String);

    impl DeliveryRoute for TestRoute {
        /// 空路由视为无效。
        fn validate(&self) -> Result<()> {
            if self.0.trim().is_empty() {
                return Err(ReactError::Other("test route is empty".to_string()));
            }
            Ok(())
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    /// 测试用负载类型。
    struct TestPayload {
        /// 负载文本。
        text: String,
    }

    /// 构造测试用 JSON 负载信封。
    fn envelope(id: &str) -> DeliveryEnvelope {
        DeliveryEnvelope {
            message_id: id.to_string(),
            route: "opaque-route".to_string(),
            payload: serde_json::json!({"text": id}),
            metadata: BTreeMap::new(),
            correlation_id: Some("correlation".to_string()),
            causation_id: Some("causation".to_string()),
        }
    }

    /// 构造基于内存日志与检查点的测试账本。
    fn ledger() -> DeliveryLedger<MemoryEventJournal<DeliveryEvent>> {
        DeliveryLedger::new(
            Arc::new(MemoryEventJournal::new()),
            Arc::new(MemoryCheckpointStore::new()),
            DeliveryLedgerConfig::default(),
            2,
        )
    }

    #[test]
    /// 类型化账本应原样保留路由与负载，不经过 JSON 映射。
    fn typed_ledger_preserves_route_and_payload_without_json_mapping() -> Result<()> {
        /// 测试用类型化投递事件。
        type TypedEvent = DeliveryEvent<TestRoute, TestPayload>;
        /// 测试用类型化内存事件日志。
        type TypedJournal = MemoryEventJournal<TypedEvent>;
        /// 测试用类型化投递账本。
        type TypedLedger = DeliveryLedger<TypedJournal, TestRoute, TestPayload>;

        let journal = Arc::new(MemoryEventJournal::<TypedEvent>::new());
        let checkpoints = Arc::new(MemoryCheckpointStore::new());
        let ledger = TypedLedger::new(
            Arc::clone(&journal),
            checkpoints
                as Arc<dyn CheckpointStore<DeliveryLedgerProjection<TestRoute, TestPayload>>>,
            DeliveryLedgerConfig::default(),
            0,
        );
        let envelope = DeliveryEnvelope::new(
            "typed-message",
            TestRoute("conversation".to_string()),
            TestPayload {
                text: "hello".to_string(),
            },
        );
        ledger
            .apply(TypedEvent::Persisted {
                envelope,
                persisted_at: Utc::now(),
            })
            .map_err(|error| ReactError::Other(error.to_string()))?;

        ledger.with_projection(|projection| {
            let record = projection.record("typed-message");
            assert_eq!(
                record.map(|record| &record.route),
                Some(&TestRoute("conversation".to_string()))
            );
            assert_eq!(
                record.map(|record| &record.payload),
                Some(&TestPayload {
                    text: "hello".to_string()
                })
            );
        });
        Ok(())
    }

    #[test]
    /// 生命周期保持 FIFO 顺序，并拒绝过期认领。
    fn lifecycle_keeps_fifo_and_rejects_stale_claim() -> Result<()> {
        let ledger = ledger();
        let now = Utc::now();
        ledger
            .apply(DeliveryEvent::Persisted {
                envelope: envelope("first"),
                persisted_at: now,
            })
            .map_err(|error| ReactError::Other(error.to_string()))?;
        ledger
            .apply(DeliveryEvent::Persisted {
                envelope: envelope("second"),
                persisted_at: now,
            })
            .map_err(|error| ReactError::Other(error.to_string()))?;
        assert!(
            ledger
                .apply(DeliveryEvent::Claimed {
                    message_id: "second".to_string(),
                    attempt_id: "attempt-2".to_string(),
                    attempt: 1,
                    claimed_at: now,
                })
                .is_err()
        );
        ledger
            .apply(DeliveryEvent::Claimed {
                message_id: "first".to_string(),
                attempt_id: "attempt-1".to_string(),
                attempt: 1,
                claimed_at: now,
            })
            .map_err(|error| ReactError::Other(error.to_string()))?;
        assert!(
            ledger
                .apply(DeliveryEvent::EffectStarted {
                    message_id: "first".to_string(),
                    attempt_id: "stale".to_string(),
                    turn_id: "turn-1".to_string(),
                    started_at: now,
                })
                .is_err()
        );
        let first = ledger.with_projection(|projection| {
            projection
                .frontier()
                .next()
                .map(|entry| entry.message_id.clone())
        });
        assert_eq!(first.as_deref(), Some("first"));
        Ok(())
    }

    #[test]
    /// 已完成的投递离开前沿，重试结算后可重新认领。
    fn completed_delivery_leaves_frontier_and_reopens_after_retry() -> Result<()> {
        let ledger = ledger();
        let now = Utc::now();
        ledger
            .apply(DeliveryEvent::Persisted {
                envelope: envelope("message"),
                persisted_at: now,
            })
            .map_err(|error| ReactError::Other(error.to_string()))?;
        ledger
            .apply(DeliveryEvent::Claimed {
                message_id: "message".to_string(),
                attempt_id: "attempt-1".to_string(),
                attempt: 1,
                claimed_at: now,
            })
            .map_err(|error| ReactError::Other(error.to_string()))?;
        ledger
            .apply(DeliveryEvent::TurnSettled {
                message_id: "message".to_string(),
                attempt_id: "attempt-1".to_string(),
                turn_id: None,
                outcome: DeliveryOutcome::Failed,
                drained: Some(false),
                reason: Some("retry".to_string()),
                retryable: true,
                next_attempt_at: Some(now),
                reply_message_id: None,
                settled_at: now,
            })
            .map_err(|error| ReactError::Other(error.to_string()))?;
        assert_eq!(
            ledger.with_projection(|projection| projection.frontier().count()),
            1
        );
        ledger
            .apply(DeliveryEvent::Claimed {
                message_id: "message".to_string(),
                attempt_id: "attempt-2".to_string(),
                attempt: 2,
                claimed_at: now,
            })
            .map_err(|error| ReactError::Other(error.to_string()))?;
        ledger
            .apply(DeliveryEvent::TurnSettled {
                message_id: "message".to_string(),
                attempt_id: "attempt-2".to_string(),
                turn_id: None,
                outcome: DeliveryOutcome::Completed,
                drained: Some(true),
                reason: None,
                retryable: false,
                next_attempt_at: None,
                reply_message_id: Some("reply".to_string()),
                settled_at: now,
            })
            .map_err(|error| ReactError::Other(error.to_string()))?;
        assert_eq!(
            ledger.with_projection(|projection| projection.frontier().count()),
            0
        );
        Ok(())
    }

    #[test]
    /// 类型化拒绝可以在邮箱接收前延后副作用准入。
    fn typed_rejection_can_defer_effect_admission_before_mailbox_acceptance() -> Result<()> {
        let ledger = ledger();
        let now = Utc::now();
        let apply = |event| {
            ledger
                .apply(event)
                .map(|_| ())
                .map_err(|error| ReactError::Other(error.to_string()))
        };
        apply(DeliveryEvent::Persisted {
            envelope: envelope("rejected-effect"),
            persisted_at: now,
        })?;
        apply(DeliveryEvent::Claimed {
            message_id: "rejected-effect".to_string(),
            attempt_id: "attempt-1".to_string(),
            attempt: 1,
            claimed_at: now,
        })?;
        apply(DeliveryEvent::EffectStarted {
            message_id: "rejected-effect".to_string(),
            attempt_id: "attempt-1".to_string(),
            turn_id: "candidate-turn".to_string(),
            started_at: now,
        })?;
        apply(DeliveryEvent::Deferred {
            message_id: "rejected-effect".to_string(),
            attempt_id: "attempt-1".to_string(),
            reason: "typed steer rejection".to_string(),
            deferred_at: now,
            next_attempt_at: now,
        })?;
        let record =
            ledger.with_projection(|projection| projection.record("rejected-effect").cloned());
        let record =
            record.ok_or_else(|| ReactError::Other("deferred record missing".to_string()))?;
        assert_eq!(record.phase, DeliveryPhase::Deferred);
        assert!(record.effect_started_at.is_none());
        assert!(record.turn_id.is_none());
        apply(DeliveryEvent::Claimed {
            message_id: "rejected-effect".to_string(),
            attempt_id: "attempt-2".to_string(),
            attempt: 2,
            claimed_at: now,
        })?;
        let record = ledger
            .with_projection(|projection| projection.record("rejected-effect").cloned())
            .ok_or_else(|| ReactError::Other("reclaimed record missing".to_string()))?;
        assert_eq!(record.phase, DeliveryPhase::Claimed);
        assert_eq!(record.attempt_id.as_deref(), Some("attempt-2"));
        assert_eq!(record.attempt, 2);
        Ok(())
    }

    #[test]
    /// 邮箱接收之后不允许再延后。
    fn typed_rejection_cannot_defer_after_mailbox_acceptance() -> Result<()> {
        let ledger = ledger();
        let now = Utc::now();
        let apply = |event| {
            ledger
                .apply(event)
                .map(|_| ())
                .map_err(|error| ReactError::Other(error.to_string()))
        };
        apply(DeliveryEvent::Persisted {
            envelope: envelope("accepted-effect"),
            persisted_at: now,
        })?;
        apply(DeliveryEvent::Claimed {
            message_id: "accepted-effect".to_string(),
            attempt_id: "attempt-1".to_string(),
            attempt: 1,
            claimed_at: now,
        })?;
        apply(DeliveryEvent::EffectStarted {
            message_id: "accepted-effect".to_string(),
            attempt_id: "attempt-1".to_string(),
            turn_id: "accepted-turn".to_string(),
            started_at: now,
        })?;
        apply(DeliveryEvent::MailboxAccepted {
            message_id: "accepted-effect".to_string(),
            attempt_id: "attempt-1".to_string(),
            turn_id: "accepted-turn".to_string(),
            accepted_at: now,
        })?;
        let deferred = ledger.apply(DeliveryEvent::Deferred {
            message_id: "accepted-effect".to_string(),
            attempt_id: "attempt-1".to_string(),
            reason: "late typed steer rejection".to_string(),
            deferred_at: now,
            next_attempt_at: now,
        });
        assert!(deferred.is_err());
        let record = ledger
            .with_projection(|projection| projection.record("accepted-effect").cloned())
            .ok_or_else(|| ReactError::Other("accepted record missing".to_string()))?;
        assert_eq!(record.phase, DeliveryPhase::MailboxAccepted);
        assert!(record.mailbox_accepted_at.is_some());
        assert_eq!(record.turn_id.as_deref(), Some("accepted-turn"));
        Ok(())
    }

    #[test]
    /// 副作用、邮箱、消费与结算全程使用同一尝试身份。
    fn effect_mailbox_drain_and_settlement_keep_one_attempt_identity() -> Result<()> {
        let ledger = ledger();
        let now = Utc::now();
        let apply = |event| {
            ledger
                .apply(event)
                .map(|_| ())
                .map_err(|error| ReactError::Other(error.to_string()))
        };
        apply(DeliveryEvent::Persisted {
            envelope: envelope("lifecycle"),
            persisted_at: now,
        })?;
        apply(DeliveryEvent::Claimed {
            message_id: "lifecycle".to_string(),
            attempt_id: "attempt-1".to_string(),
            attempt: 1,
            claimed_at: now,
        })?;
        apply(DeliveryEvent::EffectStarted {
            message_id: "lifecycle".to_string(),
            attempt_id: "attempt-1".to_string(),
            turn_id: "turn-1".to_string(),
            started_at: now,
        })?;
        apply(DeliveryEvent::MailboxAccepted {
            message_id: "lifecycle".to_string(),
            attempt_id: "attempt-1".to_string(),
            turn_id: "turn-1".to_string(),
            accepted_at: now,
        })?;
        apply(DeliveryEvent::Drained {
            message_id: "lifecycle".to_string(),
            attempt_id: "attempt-1".to_string(),
            turn_id: "turn-1".to_string(),
            drained_at: now,
        })?;
        apply(DeliveryEvent::TurnSettled {
            message_id: "lifecycle".to_string(),
            attempt_id: "attempt-1".to_string(),
            turn_id: Some("turn-1".to_string()),
            outcome: DeliveryOutcome::Completed,
            drained: Some(true),
            reason: None,
            retryable: false,
            next_attempt_at: None,
            reply_message_id: Some("reply-1".to_string()),
            settled_at: now,
        })?;
        let record = ledger.with_projection(|projection| projection.record("lifecycle").cloned());
        let record =
            record.ok_or_else(|| ReactError::Other("settled record disappeared".to_string()))?;
        assert_eq!(record.phase, DeliveryPhase::TurnSettled);
        assert!(record.terminal);
        assert_eq!(record.attempt_id.as_deref(), Some("attempt-1"));
        assert_eq!(record.turn_id.as_deref(), Some("turn-1"));
        assert_eq!(record.reply_message_id.as_deref(), Some("reply-1"));
        Ok(())
    }

    #[test]
    /// 恢复时可从日志重放出投递投影。
    fn recovery_replays_delivery_projection_from_journal() -> Result<()> {
        let journal = Arc::new(MemoryEventJournal::new());
        let checkpoints = Arc::new(MemoryCheckpointStore::new());
        let ledger = DeliveryLedger::new(
            Arc::clone(&journal),
            Arc::clone(&checkpoints) as Arc<dyn CheckpointStore<DeliveryLedgerProjection>>,
            DeliveryLedgerConfig::default(),
            1,
        );
        let now = Utc::now();
        ledger
            .apply(DeliveryEvent::Persisted {
                envelope: envelope("recover"),
                persisted_at: now,
            })
            .map_err(|error| ReactError::Other(error.to_string()))?;
        let recovered =
            DeliveryLedger::new(journal, checkpoints, DeliveryLedgerConfig::default(), 1);
        let receipt = recovered.recover()?;
        assert_eq!(receipt.last_applied_sequence, 1);
        assert!(recovered.with_projection(|projection| projection.record("recover").is_some()));
        Ok(())
    }

    #[test]
    /// 终态保留同时受条数与字节数限制。
    fn terminal_retention_is_bounded_by_count_and_bytes() -> Result<()> {
        let config = DeliveryLedgerConfig {
            terminal_retention: 1,
            terminal_retention_bytes: 1024,
        };
        let ledger = DeliveryLedger::new(
            Arc::new(MemoryEventJournal::new()),
            Arc::new(MemoryCheckpointStore::new()),
            config,
            0,
        );
        let now = Utc::now();
        for id in ["first", "second"] {
            ledger
                .apply(DeliveryEvent::Persisted {
                    envelope: envelope(id),
                    persisted_at: now,
                })
                .map_err(|error| ReactError::Other(error.to_string()))?;
            ledger
                .apply(DeliveryEvent::Claimed {
                    message_id: id.to_string(),
                    attempt_id: format!("{id}-attempt"),
                    attempt: 1,
                    claimed_at: now,
                })
                .map_err(|error| ReactError::Other(error.to_string()))?;
            ledger
                .apply(DeliveryEvent::TurnSettled {
                    message_id: id.to_string(),
                    attempt_id: format!("{id}-attempt"),
                    turn_id: None,
                    outcome: DeliveryOutcome::Completed,
                    drained: Some(false),
                    reason: None,
                    retryable: false,
                    next_attempt_at: None,
                    reply_message_id: None,
                    settled_at: now,
                })
                .map_err(|error| ReactError::Other(error.to_string()))?;
        }
        let records = ledger.with_projection(|projection| {
            projection
                .records()
                .map(|record| record.message_id.clone())
                .collect::<Vec<_>>()
        });
        assert_eq!(records, vec!["second"]);
        Ok(())
    }

    #[test]
    /// 预备批次保持单一身份以便对账。
    fn prepared_batch_keeps_one_identity_for_reconciliation() -> Result<()> {
        let journal = Arc::new(MemoryEventJournal::new());
        let checkpoints = Arc::new(MemoryCheckpointStore::new());
        let ledger = DeliveryLedger::new(
            Arc::clone(&journal),
            checkpoints,
            DeliveryLedgerConfig::default(),
            2,
        );
        let now = Utc::now();
        let prepared = PreparedJournalBatch::new(vec![DeliveryEvent::Persisted {
            envelope: envelope("prepared"),
            persisted_at: now,
        }])
        .map_err(|error| ReactError::Other(error.to_string()))?;
        let batch_id = prepared.batch_id().to_string();
        let receipt = ledger
            .apply_prepared(prepared)
            .map_err(|error| ReactError::Other(error.to_string()))?;
        assert_eq!(receipt.batch_id, batch_id);
        assert_eq!(receipt.record_count, 1);
        assert!(ledger.validate().is_ok());
        Ok(())
    }

    #[test]
    /// 并发认领只跳过副作用已开始的前驱。
    fn concurrent_claim_skips_only_predecessors_whose_effect_started() -> Result<()> {
        let ledger = ledger();
        for id in ["first", "second"] {
            ledger
                .enqueue(envelope(id))
                .map_err(|error| ReactError::Other(error.to_string()))?;
        }
        let first = ledger
            .claim_next()
            .map_err(|error| ReactError::Other(error.to_string()))?
            .ok_or_else(|| ReactError::Other("first claim missing".to_string()))?;
        let pre_effect = ledger
            .prepare_claim_next_available()
            .map_err(|error| ReactError::Other(error.to_string()))?
            .ok_or_else(|| ReactError::Other("pre-effect claim missing".to_string()))?;
        assert_eq!(pre_effect.claim.message_id, "first");
        ledger
            .begin_effect(&first, "first-turn")
            .map_err(|error| ReactError::Other(error.to_string()))?;

        let second = ledger
            .claim_next_available()
            .map_err(|error| ReactError::Other(error.to_string()))?
            .ok_or_else(|| ReactError::Other("second concurrent claim missing".to_string()))?;
        assert_eq!(second.message_id, "second");
        assert!(ledger.validate().is_ok());
        Ok(())
    }

    #[test]
    /// 并发认领不会越过被延后的前驱。
    fn concurrent_claim_does_not_bypass_a_deferred_predecessor() -> Result<()> {
        let ledger = ledger();
        for id in ["first", "second"] {
            ledger
                .enqueue(envelope(id))
                .map_err(|error| ReactError::Other(error.to_string()))?;
        }
        let first = ledger
            .claim_next()
            .map_err(|error| ReactError::Other(error.to_string()))?
            .ok_or_else(|| ReactError::Other("first claim missing".to_string()))?;
        ledger
            .defer(
                &first,
                "retry later",
                Utc::now() + chrono::Duration::hours(1),
            )
            .map_err(|error| ReactError::Other(error.to_string()))?;

        assert!(
            ledger
                .claim_next_available()
                .map_err(|error| ReactError::Other(error.to_string()))?
                .is_none()
        );
        assert!(ledger.validate().is_ok());
        Ok(())
    }
}
