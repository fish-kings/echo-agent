//! Subagent 事件系统 —— Subagent 操作的生命周期通知

use echo_core::agent::{EventEnvelope, EventId, EventIdentity, StreamId, ToolInvocation};
use echo_core::tools::{SubagentLineage, ToolResult};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock, Weak};
use tokio::sync::broadcast;
use tracing::info;

use super::types::{ExecutionMode, ObservedIsolation, SubagentOutcome, SubagentStatus};

/// 事件总线默认的信道容量。
const DEFAULT_CHANNEL_CAPACITY: usize = 128;

/// Subagent 系统发出的生命周期事件。
#[derive(Debug, Clone, Serialize, Deserialize)]
// 将 ToolResult 装箱会改变每个消费者所依赖的公共事件契约。
#[allow(clippy::large_enum_variant)]
pub enum SubagentEvent {
    /// 某个 Subagent 已注册。
    Registered {
        /// 已注册的 Subagent 名称。
        name: String,
    },
    /// 某个 Subagent 已注销。
    Unregistered {
        /// 已注销的 Subagent 名称。
        name: String,
    },
    /**
     * 一个正在运行的 Subagent 通过默认上行链路(uplink)接收器发送了一条
     * 父级/兄弟消息。仅用于可观测性;实际的投递结果记录在 `status` 中
     * (例如 `parent_steered`、`event_emitted`、`delivered_to_sibling`)。
     */
    UplinkReceived {
        /// 发起调度的父 Agent 名称。
        parent: String,
        /// 发送消息的 Subagent 名称。
        agent: String,
        /// `parent` 或 `sibling`。
        direction: String,
        /// 报告给发送方的投递结果。
        status: String,
        /// 有长度上限的消息预览(前 200 个字符)。
        summary: String,
        /// 发送方本次尝试的 execution id。
        execution_id: Option<String>,
        /// 发送方本次尝试的 run id。
        run_id: Option<String>,
    },
    /// 调度已开始。
    DispatchStarted {
        /// 发起本次调度的父 Agent 名称。
        parent: String,
        /// 被调度的 Subagent 名称。
        agent: String,
        /// 执行模式(例如 `ExecutionMode::Parallel`)。
        mode: ExecutionMode,
        /// 本次调度所携带的任务描述。
        task: String,
        /**
         * 来自调用方 `ExternalRunContext` 的不透明稳定 execution id。
         * 基于任务的调度会从精确的认领(claim)血统中派生该值;调用方
         * 不得解析或重建该 id。`None` 表示尚未接入的旧版调用方;桥接层
         * 会退回到临时 id 分配。前端应将其作为规范的 `subagent_run_id`
         * 使用。
         */
        execution_id: Option<String>,
        /**
         * 来自调用方 `ExternalRunContext` 的父级 run id。`None` 表示旧版
         * 调用方。
         */
        run_id: Option<String>,
        /**
         * 来自调用方 `ExternalRunContext` 的会话 id。即使是没有正式
         * run id 的临时调度,该字段也会被保留。
         */
        conversation_id: Option<String>,
        /**
         * 触发本次运行的消息 id(聊天场景下的 `message_key`)。用于让
         * 前端把 Subagent 流固定到正确的聊天消息块上。`None` 表示非
         * 聊天路径(例如定时任务等)。
         */
        message_id: Option<String>,
        /**
         * 当本次调度通过 `dispatch_background`(非阻塞)方式启动时为
         * true;UI 会据此展示后台卡片,并在完成时向父级聊天注入一条
         * 完成提示。
         */
        background: bool,
    },
    /// 在设置完成、模型开始执行之前建立的隔离边界。
    DispatchIsolationObserved {
        /// 发起本次调度的父 Agent 名称。
        parent: String,
        /// 被调度的 Subagent 名称。
        agent: String,
        /// 本次调度观测到的隔离边界。
        isolation: ObservedIsolation,
        /// 稳定的 execution id(参见 [`Self::DispatchStarted::execution_id`])。
        execution_id: Option<String>,
        /// 父级 run id(参见 [`Self::DispatchStarted::run_id`])。
        run_id: Option<String>,
    },
    /// 调度成功完成。
    DispatchCompleted {
        /// 发起本次调度的父 Agent 名称。
        parent: String,
        /// 完成该任务的 Subagent 名称。
        agent: String,
        /// 本次调度耗时(毫秒)。
        duration_ms: u64,
        /// 消耗的总 token 数(输入 + 输出),如果可获得的话。
        tokens_used: Option<u64>,
        /// 执行的 ReAct 迭代次数。
        iterations: Option<u64>,
        /// Subagent 产出的最终输出文本。
        output: String,
        /// 提供给父级/应用消费的结构化终态结果。
        outcome: SubagentOutcome,
        /// 稳定的 execution id(参见 [`Self::DispatchStarted::execution_id`])。
        execution_id: Option<String>,
        /// 父级 run id(参见 [`Self::DispatchStarted::run_id`])。
        run_id: Option<String>,
    },
    /// 调度失败。
    DispatchFailed {
        /// 发起本次调度的父 Agent 名称。
        parent: String,
        /// 失败的 Subagent 名称。
        agent: String,
        /// 描述失败原因的错误信息。
        error: String,
        /// 失败或超时的终态状态。
        status: SubagentStatus,
        /// 结构化的终态结果,包含剩余未完成的工作。
        outcome: SubagentOutcome,
        /// 稳定的 execution id(参见 [`Self::DispatchStarted::execution_id`])。
        execution_id: Option<String>,
        /// 父级 run id(参见 [`Self::DispatchStarted::run_id`])。
        run_id: Option<String>,
    },
    /// 调度已被取消。
    DispatchCancelled {
        /// 取消本次调度的父 Agent 名称。
        parent: String,
        /// 被取消调度的 Subagent 名称。
        agent: String,
        /// 结构化的已取消终态结果。
        outcome: SubagentOutcome,
        /// 稳定的 execution id(参见 [`Self::DispatchStarted::execution_id`])。
        execution_id: Option<String>,
        /// 父级 run id(参见 [`Self::DispatchStarted::run_id`])。
        run_id: Option<String>,
    },
    /// Subagent 开始推理。
    DispatchThinkingStarted {
        /// 发起本次调度的父 Agent 名称。
        parent: String,
        /// 正在推理的 Subagent 名称。
        agent: String,
        /// 稳定的 execution id(参见 [`Self::DispatchStarted::execution_id`])。
        execution_id: Option<String>,
        /// 父级 run id(参见 [`Self::DispatchStarted::run_id`])。
        run_id: Option<String>,
    },
    /// Subagent 推理过程中产生了增量内容。
    DispatchThinkingDelta {
        /// 发起本次调度的父 Agent 名称。
        parent: String,
        /// 正在推理的 Subagent 名称。
        agent: String,
        /// 增量的推理文本。
        content: String,
        /// 稳定的 execution id(参见 [`Self::DispatchStarted::execution_id`])。
        execution_id: Option<String>,
        /// 父级 run id(参见 [`Self::DispatchStarted::run_id`])。
        run_id: Option<String>,
    },
    /// Subagent 推理结束。
    DispatchThinkingEnded {
        /// 发起本次调度的父 Agent 名称。
        parent: String,
        /// 完成推理的 Subagent 名称。
        agent: String,
        /// 消耗的 prompt token 数量。
        prompt_tokens: usize,
        /// 消耗的 completion token 数量。
        completion_tokens: usize,
        /// 稳定的 execution id(参见 [`Self::DispatchStarted::execution_id`])。
        execution_id: Option<String>,
        /// 父级 run id(参见 [`Self::DispatchStarted::run_id`])。
        run_id: Option<String>,
    },
    /// Subagent 输出了最终答案文本。
    DispatchTokenDelta {
        /// 发起本次调度的父 Agent 名称。
        parent: String,
        /// 正在产出结果的 Subagent 名称。
        agent: String,
        /// 增量的最终答案文本。
        content: String,
        /// 稳定的 execution id(参见 [`Self::DispatchStarted::execution_id`])。
        execution_id: Option<String>,
        /// 父级 run id(参见 [`Self::DispatchStarted::run_id`])。
        run_id: Option<String>,
    },
    /**
     * Subagent 底层模型调用报告的 LLM 用量(携带完整的缓存诊断细分数据)。
     * 每次模型调用都会发出一次,便于前端在不窥探旧版 `subagent://trace`
     * 通道的情况下渲染 token / 缓存命中等指标。
     */
    DispatchLlmUsage {
        /// 发起本次调度的父 Agent 名称。
        parent: String,
        /// 发起本次模型调用的 Subagent 名称。
        agent: String,
        /// 模型名称(与具体供应商相关)。
        model: String,
        /// 本次调用的 prompt(输入)token 数。
        prompt_tokens: usize,
        /// 本次调用的 completion(输出)token 数。
        completion_tokens: usize,
        /// 供应商上报的总 token 数(输入 + 输出)。
        total_tokens: usize,
        /// 由前缀缓存命中(cache hit)提供服务的 prompt token 数。
        cached_prompt_tokens: usize,
        /// 写入缓存(cache write)的 prompt token 数。
        cache_creation_prompt_tokens: usize,
        /// 供应商是否针对本次调用实际返回了用量报告。
        usage_reported: bool,
        /// 稳定的 execution id(参见 [`Self::DispatchStarted::execution_id`])。
        execution_id: Option<String>,
        /// 父级 run id(参见 [`Self::DispatchStarted::run_id`])。
        run_id: Option<String>,
    },
    /// Subagent 发起了一次工具调用。
    DispatchToolStarted {
        /// 发起本次调度的父 Agent 名称。
        parent: String,
        /// 正在调用工具的 Subagent 名称。
        agent: String,
        /// 模型发出的稳定工具调用标识。
        call_id: String,
        /// 经过所有运行时改写后的规范请求/生效调用。
        invocation: ToolInvocation,
        /// 稳定的 execution id(参见 [`Self::DispatchStarted::execution_id`])。
        execution_id: Option<String>,
        /// 父级 run id(参见 [`Self::DispatchStarted::run_id`])。
        run_id: Option<String>,
    },
    /// Subagent 完成了一次工具调用。
    DispatchToolCompleted {
        /// 发起本次调度的父 Agent 名称。
        parent: String,
        /// 调用工具的 Subagent 名称。
        agent: String,
        /// 与 [`Self::DispatchToolStarted`] 对应的稳定工具调用标识。
        call_id: String,
        /// 与调用事件一致的生效工具名称。
        name: String,
        /// ReAct 运行器发出的规范化富终态结果。
        result: ToolResult,
        /// 稳定的 execution id(参见 [`Self::DispatchStarted::execution_id`])。
        execution_id: Option<String>,
        /// 父级 run id(参见 [`Self::DispatchStarted::run_id`])。
        run_id: Option<String>,
    },
}

/**
 * 哈希进每个执行信封(execution envelope)中的完整调度尝试身份。
 *
 * 传输层寻址信息保留在 [`EventEnvelope`] 中。这些 Subagent 专属字段用于
 * 保存任务与血统(lineage)事实,而无需解析 execution id,也不会给框架
 * 引入应用层私有的工作区元数据。
 */
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubagentInvocationIdentity {
    /// 发起调度的父 Agent 名称。
    pub parent_agent: String,
    /// 被调度的 Subagent 名称。
    pub agent_name: String,
    /// 父级调度尝试的 execution id(如果存在)。
    pub parent_execution_id: Option<String>,
    /// 从根 Agent 到当前 Subagent 的调度路径。
    pub agent_path: Option<String>,
    /// 关联的任务 id(如果由任务系统驱动)。
    pub task_id: Option<String>,
    /// 当前尝试的序号,从 1 开始计数。
    pub attempt: Option<u32>,
    /// 生成本次调度时所依据的计划版本号。
    pub plan_revision: Option<u64>,
}

impl SubagentInvocationIdentity {
    /**
     * 根据调用方提供的 lineage(血统)信息构建一个 Subagent 调用身份。
     *
     * 若同时提供了 `lineage`,会校验其携带的各字段与本次显式传入的
     * `parent_agent`/`agent_name`,以及 `transport_identity` 中的
     * `execution_id`/`run_id` 是否一致;一旦发生冲突则返回错误。
     */
    pub fn from_lineage(
        parent_agent: impl Into<String>,
        agent_name: impl Into<String>,
        transport_identity: &EventIdentity,
        lineage: Option<&SubagentLineage>,
    ) -> crate::error::Result<Self> {
        let parent_agent = parent_agent.into();
        let agent_name = agent_name.into();
        if parent_agent.trim().is_empty() || agent_name.trim().is_empty() {
            return Err(crate::error::ReactError::Other(
                "Subagent invocation agent identity must not be empty".to_string(),
            ));
        }
        if let Some(lineage) = lineage {
            for (field, actual, expected) in [
                (
                    "agent_name",
                    lineage.agent_name.as_deref(),
                    Some(agent_name.as_str()),
                ),
                (
                    "parent_agent",
                    lineage.parent_agent.as_deref(),
                    Some(parent_agent.as_str()),
                ),
                (
                    "execution_id",
                    lineage.execution_id.as_deref(),
                    transport_identity
                        .execution_id
                        .as_ref()
                        .map(|value| value.as_str()),
                ),
                (
                    "run_id",
                    lineage.run_id.as_deref(),
                    transport_identity
                        .run_id
                        .as_ref()
                        .map(|value| value.as_str()),
                ),
            ] {
                if actual.is_some() && actual != expected {
                    return Err(crate::error::ReactError::Other(format!(
                        "Subagent lineage {field} conflicts with transport identity"
                    )));
                }
            }
        }
        Ok(Self {
            parent_execution_id: lineage.and_then(|value| value.parent_execution_id.clone()),
            agent_path: lineage.and_then(|value| value.agent_path.clone()),
            task_id: lineage.and_then(|value| value.task_id.clone()),
            attempt: lineage.and_then(|value| value.attempt),
            plan_revision: lineage.and_then(|value| value.plan_revision),
            parent_agent,
            agent_name,
        })
    }
}

/// 由通用框架 [`EventEnvelope`] 承载的、经过哈希处理的负载。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubagentEventPayload {
    /// 本次事件所属调度尝试的调用身份。
    pub invocation: SubagentInvocationIdentity,
    /// 实际的 Subagent 生命周期事件。
    pub event: SubagentEvent,
}

/// 一次 Subagent 调度尝试所使用的、权威且带版本号的事件传输类型。
pub type SubagentEventEnvelope = EventEnvelope<SubagentEventPayload>;

/**
 * 明确指出一次信封重放无法从请求的序号开始提供连续后缀,因为有限保留
 * 窗口已经向前推进。
 */
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubagentEventGap {
    /// 出现缺口的事件流标识。
    pub stream_id: StreamId,
    /// 调用方请求的起始序号(即"在此序号之后")。
    pub requested_after: u64,
    /**
     * 缺口之后第一个仍被保留的序号。`None` 表示该已知事件流虽然已经
     * 推进,但其可重放的后缀已不再被保留。
     */
    pub available_from: Option<u64>,
    /// 该事件流当前已知的最新序号。
    pub latest_sequence: u64,
}

/// 针对某一具体调度尝试事件流的有限重放结果。
#[derive(Debug)]
pub struct SubagentEventReplay {
    /// 重放得到的事件序列。
    pub events: Vec<Arc<SubagentEventEnvelope>>,
    /// 若重放存在不连续之处,则给出对应的缺口描述。
    pub gap: Option<SubagentEventGap>,
    /**
     * 最新保留的终态事件会被独立返回,这样即使高频率的增量事件已经从
     * 历史记录中被淘汰,消费方依然可以据此核对最终输出。
     */
    pub terminal: Option<Arc<SubagentEventEnvelope>>,
}

/// 单次发布者(publisher)所维护的内部可变状态。
#[derive(Default)]
struct SubagentPublisherState {
    /// 当前已分配到的最新序号。
    sequence: u64,
    /// 本次调度起始事件的 event id。
    dispatch_started: Option<EventId>,
    /**
     * 在活跃发布者的整个生命周期内保留的不可变路由锚点,即便有限重放
     * 窗口已经淘汰了该事件流。
     */
    dispatch_anchor: Option<Arc<SubagentEventEnvelope>>,
    /// 正在进行中的工具调用:call_id 到其起始事件 id 的映射。
    tool_started: HashMap<String, EventId>,
    /// 是否已经发出过终态事件。
    terminal_emitted: bool,
}

/// 归属于单次调度尝试的发布者所共享的内部状态。
struct SubagentEventPublisherInner {
    /// 关联的事件总线。
    bus: SubagentEventBus,
    /// 本次调度尝试的传输层身份。
    transport_identity: EventIdentity,
    /// 本次调度尝试的调用身份,支持在委派时被重新定向(retarget)。
    invocation_identity: RwLock<SubagentInvocationIdentity>,
    /// 本次发布者的可变状态。
    state: Mutex<SubagentPublisherState>,
}

/// 可克隆的生产者,独占地负责一次调度尝试的事件排序。
#[derive(Clone)]
pub struct SubagentEventPublisher {
    /// 共享的内部状态。
    inner: Arc<SubagentEventPublisherInner>,
}

impl SubagentEventPublisher {
    /// 创建一个新的发布者,并在构造前校验传输层身份的合法性。
    fn new(
        bus: SubagentEventBus,
        transport_identity: EventIdentity,
        invocation_identity: SubagentInvocationIdentity,
    ) -> crate::error::Result<Self> {
        transport_identity.validate()?;
        Ok(Self {
            inner: Arc::new(SubagentEventPublisherInner {
                bus,
                transport_identity,
                invocation_identity: RwLock::new(invocation_identity),
                state: Mutex::new(SubagentPublisherState::default()),
            }),
        })
    }

    /**
     * 当框架钩子将同一个逻辑调度尝试委派给另一个已注册的 Subagent 后,
     * 更新实际的执行段(execution segment)。
     */
    pub(super) fn retarget(
        &self,
        invocation_identity: SubagentInvocationIdentity,
    ) -> crate::error::Result<()> {
        let mut current = self
            .inner
            .invocation_identity
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if current.parent_agent != invocation_identity.parent_agent
            || current.parent_execution_id != invocation_identity.parent_execution_id
            || current.task_id != invocation_identity.task_id
            || current.attempt != invocation_identity.attempt
            || current.plan_revision != invocation_identity.plan_revision
        {
            return Err(crate::error::ReactError::Other(
                "Subagent publisher retarget changed stable attempt identity".to_string(),
            ));
        }
        *current = invocation_identity;
        Ok(())
    }

    /// 根据 lineage 信息构造调用身份并对当前发布者执行重新定向。
    pub(super) fn retarget_from_lineage(
        &self,
        parent_agent: impl Into<String>,
        agent_name: impl Into<String>,
        lineage: Option<&SubagentLineage>,
    ) -> crate::error::Result<()> {
        let identity = SubagentInvocationIdentity::from_lineage(
            parent_agent,
            agent_name,
            &self.inner.transport_identity,
            lineage,
        )?;
        self.retarget(identity)
    }

    /// 在本次调度尝试的单调递增序列中发出一个事件。
    pub fn emit(&self, event: SubagentEvent) -> crate::error::Result<Arc<SubagentEventEnvelope>> {
        let invocation_identity = self
            .inner
            .invocation_identity
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();
        let mut state = self
            .inner
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if state.terminal_emitted {
            return Err(crate::error::ReactError::Other(
                "Subagent event emitted after terminal settlement".to_string(),
            ));
        }
        if !is_execution_event(&event) {
            return Err(crate::error::ReactError::Other(
                "registry events cannot be emitted through a Subagent attempt publisher"
                    .to_string(),
            ));
        }
        if let SubagentEvent::DispatchStarted {
            conversation_id,
            message_id,
            ..
        } = &event
        {
            let expected_conversation = self
                .inner
                .transport_identity
                .conversation_id
                .as_ref()
                .map(|value| value.as_str());
            let expected_message = self
                .inner
                .transport_identity
                .message_id
                .as_ref()
                .map(|value| value.as_str());
            if conversation_id.as_deref() != expected_conversation
                || message_id.as_deref() != expected_message
            {
                return Err(crate::error::ReactError::Other(
                    "Subagent dispatch start conversation or message identity changed".to_string(),
                ));
            }
        }
        match &event {
            SubagentEvent::DispatchStarted { .. } if state.dispatch_started.is_some() => {
                return Err(crate::error::ReactError::Other(
                    "Subagent dispatch start was emitted more than once".to_string(),
                ));
            }
            SubagentEvent::DispatchStarted { .. } => {}
            _ if state.dispatch_started.is_none() => {
                return Err(crate::error::ReactError::Other(
                    "Subagent execution event was emitted before dispatch start".to_string(),
                ));
            }
            _ => {}
        }
        if let SubagentEvent::DispatchToolStarted { call_id, .. } = &event
            && state.tool_started.contains_key(call_id)
        {
            return Err(crate::error::ReactError::Other(format!(
                "duplicate in-flight Subagent tool call: {call_id}"
            )));
        }
        let route = event_route(&event).ok_or_else(|| {
            crate::error::ReactError::Other(
                "Subagent execution event has no routing identity".to_string(),
            )
        })?;
        validate_event_route(route, &self.inner.transport_identity, &invocation_identity)?;
        let sequence = state.sequence.checked_add(1).ok_or_else(|| {
            crate::error::ReactError::Other(
                "Subagent event sequence exhausted before terminal event".to_string(),
            )
        })?;
        let parent_event_id = match &event {
            SubagentEvent::DispatchStarted { .. } if state.dispatch_started.is_none() => {
                self.inner.transport_identity.parent_event_id.clone()
            }
            SubagentEvent::DispatchToolCompleted { call_id, .. } => {
                Some(state.tool_started.get(call_id).cloned().ok_or_else(|| {
                    crate::error::ReactError::Other(format!(
                        "orphan Subagent tool completion: {call_id}"
                    ))
                })?)
            }
            _ => state
                .dispatch_started
                .clone()
                .or_else(|| self.inner.transport_identity.parent_event_id.clone()),
        };
        let envelope = Arc::new(EventEnvelope::new(
            &self.inner.transport_identity,
            sequence,
            parent_event_id,
            SubagentEventPayload {
                invocation: invocation_identity,
                event: event.clone(),
            },
        )?);
        state.sequence = sequence;
        match &event {
            SubagentEvent::DispatchStarted { .. } if state.dispatch_started.is_none() => {
                state.dispatch_started = Some(envelope.event_id.clone());
                state.dispatch_anchor = Some(Arc::clone(&envelope));
            }
            SubagentEvent::DispatchToolStarted { call_id, .. } => {
                state
                    .tool_started
                    .insert(call_id.clone(), envelope.event_id.clone());
            }
            SubagentEvent::DispatchToolCompleted { call_id, .. } => {
                state.tool_started.remove(call_id);
            }
            _ => {}
        }
        if is_terminal_event(&event) {
            state.terminal_emitted = true;
        }
        drop(state);

        self.inner.bus.emit_envelope(Arc::clone(&envelope));
        if is_terminal_event(&event) {
            self.inner.bus.retire_publisher(self);
        }
        Ok(envelope)
    }

    /// 返回本次调度尝试的事件流标识。
    pub fn stream_id(&self) -> &StreamId {
        &self.inner.transport_identity.stream_id
    }

    /// 返回本次调度起始事件的 event id(如果已经发出过)。
    pub fn dispatch_started_event_id(&self) -> Option<EventId> {
        self.inner
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .dispatch_started
            .clone()
    }
}

/// 同步事件监听器 trait。
pub trait SubagentEventListener: Send + Sync {
    /**
     * 处理一个 subagent 生命周期事件。
     *
     * # 参数
     * * `event` - 待处理的事件。
     */
    fn on_event(&self, event: &SubagentEvent);
}

/**
 * 日志监听器 —— 发出 tracing 事件。
 *
 * 实现 `SubagentEventListener`,通过 `tracing::info!` 记录事件日志。
 */
pub struct LoggingSubagentListener;

impl SubagentEventListener for LoggingSubagentListener {
    fn on_event(&self, event: &SubagentEvent) {
        match event {
            SubagentEvent::Registered { name } => {
                info!(subagent = %name, "subagent_registered");
            }
            SubagentEvent::DispatchStarted {
                parent,
                agent,
                mode,
                ..
            } => {
                info!(
                    parent = %parent,
                    agent = %agent,
                    mode = %mode,
                    "subagent_dispatch_started"
                );
            }
            SubagentEvent::DispatchCompleted {
                parent,
                agent,
                duration_ms,
                ..
            } => {
                info!(
                    parent = %parent,
                    agent = %agent,
                    duration_ms = duration_ms,
                    "subagent_dispatch_completed"
                );
            }
            SubagentEvent::DispatchIsolationObserved {
                parent,
                agent,
                isolation,
                ..
            } => {
                info!(
                    parent = %parent,
                    agent = %agent,
                    isolation = isolation.as_str(),
                    "subagent_dispatch_isolation_observed"
                );
            }
            SubagentEvent::DispatchFailed {
                parent,
                agent,
                error,
                ..
            } => {
                info!(
                    parent = %parent,
                    agent = %agent,
                    error = %error,
                    "subagent_dispatch_failed"
                );
            }
            _ => {}
        }
    }
}

/**
 * Subagent 生命周期事件所使用的异步事件总线。
 *
 * 为了兼容旧代码,依旧保留原始(raw)订阅方式。执行侧的生产者应当创建
 * 一个 [`SubagentEventPublisher`];需要身份信息、顺序保证或恢复能力的
 * 消费者应当订阅信封(envelope)。
 */
pub struct SubagentEventBus {
    /// 原始事件的广播发送端,用于兼容旧订阅方式。
    raw_tx: broadcast::Sender<Arc<SubagentEvent>>,
    /// 信封事件的广播发送端。
    envelope_tx: broadcast::Sender<Arc<SubagentEventEnvelope>>,
    /// 已注册的同步监听器列表。
    sync_listeners: Arc<RwLock<Vec<Arc<dyn SubagentEventListener>>>>,
    /// 有限重放窗口所维护的状态。
    replay: Arc<Mutex<SubagentReplayState>>,
    /// 按 execution id 索引的活跃发布者集合。
    active_publishers: Arc<Mutex<HashMap<String, Weak<SubagentEventPublisherInner>>>>,
    /// 按事件流标识索引的活跃发布者集合。
    active_streams: Arc<Mutex<HashMap<StreamId, Weak<SubagentEventPublisherInner>>>>,
}

/// 有限重放窗口的内部状态:历史事件、重放边界与已保留的终态事件。
struct SubagentReplayState {
    /// 每个队列所允许保留的最大条目数。
    capacity: usize,
    /// 按到达顺序保留的近期事件历史。
    history: VecDeque<Arc<SubagentEventEnvelope>>,
    /// 可作为重放边界的事件(排除高频增量事件)。
    boundaries: VecDeque<Arc<SubagentEventEnvelope>>,
    /// 每个事件流最新保留的终态事件。
    terminals: VecDeque<Arc<SubagentEventEnvelope>>,
}

impl SubagentReplayState {
    /// 创建一个指定容量的重放状态。
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            history: VecDeque::with_capacity(capacity),
            boundaries: VecDeque::with_capacity(capacity),
            terminals: VecDeque::with_capacity(capacity),
        }
    }

    /// 将一个事件信封纳入有限重放窗口,并按容量淘汰最旧的条目。
    fn retain(&mut self, envelope: Arc<SubagentEventEnvelope>) {
        self.history.push_back(Arc::clone(&envelope));
        while self.history.len() > self.capacity {
            self.history.pop_front();
        }
        if is_replay_boundary(&envelope.payload.event) {
            self.boundaries.push_back(Arc::clone(&envelope));
            while self.boundaries.len() > self.capacity {
                self.boundaries.pop_front();
            }
        }
        if is_terminal_event(&envelope.payload.event) {
            self.terminals
                .retain(|candidate| candidate.stream_id != envelope.stream_id);
            self.terminals.push_back(envelope);
            while self.terminals.len() > self.capacity {
                self.terminals.pop_front();
            }
        }
    }
}

impl SubagentEventBus {
    /// 使用默认容量创建一个新的事件总线。
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_CHANNEL_CAPACITY)
    }

    /**
     * 使用指定的信道容量创建一个新的事件总线。
     *
     * # 参数
     * * `capacity` - 在丢弃旧事件之前,允许缓冲的最大事件数。
     */
    pub fn with_capacity(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        let (raw_tx, _) = broadcast::channel(capacity);
        let (envelope_tx, _) = broadcast::channel(capacity);
        Self {
            raw_tx,
            envelope_tx,
            sync_listeners: Arc::new(RwLock::new(Vec::new())),
            replay: Arc::new(Mutex::new(SubagentReplayState::new(capacity))),
            active_publishers: Arc::new(Mutex::new(HashMap::new())),
            active_streams: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// 注册一个同步监听器(在事件发出时立即被调用)。
    pub fn register(&mut self, listener: Box<dyn SubagentEventListener>) {
        self.sync_listeners
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(Arc::from(listener));
    }

    /// 订阅异步事件流。
    pub fn subscribe(&self) -> broadcast::Receiver<Arc<SubagentEvent>> {
        self.raw_tx.subscribe()
    }

    /// 订阅权威的执行事件信封。
    pub fn subscribe_envelopes(&self) -> broadcast::Receiver<Arc<SubagentEventEnvelope>> {
        self.envelope_tx.subscribe()
    }

    /// 为某一具体调度尝试创建唯一的排序权威(发布者)。
    pub fn publisher(
        &self,
        transport_identity: EventIdentity,
        invocation_identity: SubagentInvocationIdentity,
    ) -> crate::error::Result<SubagentEventPublisher> {
        if invocation_identity.parent_agent.trim().is_empty()
            || invocation_identity.agent_name.trim().is_empty()
        {
            return Err(crate::error::ReactError::Other(
                "Subagent publisher agent identity must not be empty".to_string(),
            ));
        }
        if invocation_identity.attempt == Some(0) {
            return Err(crate::error::ReactError::Other(
                "Subagent publisher attempt must be one-based".to_string(),
            ));
        }
        let execution_id = transport_identity
            .execution_id
            .as_ref()
            .map(|value| value.as_str().to_string());
        let stream_id = transport_identity.stream_id.clone();
        let retained_stream = self.replay_after(&stream_id, 0);
        if !retained_stream.events.is_empty() || retained_stream.terminal.is_some() {
            return Err(crate::error::ReactError::Other(format!(
                "Subagent event stream '{stream_id}' is still retained"
            )));
        }
        if let Some(execution_id) = execution_id.as_deref()
            && self.replay_for_execution(execution_id, 0).is_some()
        {
            return Err(crate::error::ReactError::Other(format!(
                "Subagent execution '{execution_id}' is still retained"
            )));
        }
        let publisher =
            SubagentEventPublisher::new(self.clone(), transport_identity, invocation_identity)?;
        let mut streams = self
            .active_streams
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        streams.retain(|_, candidate| candidate.strong_count() > 0);
        if streams.get(&stream_id).and_then(Weak::upgrade).is_some() {
            return Err(crate::error::ReactError::Other(format!(
                "Subagent event publisher already exists for stream '{stream_id}'"
            )));
        }
        let mut publishers = self
            .active_publishers
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        publishers.retain(|_, candidate| candidate.strong_count() > 0);
        if let Some(execution_id) = execution_id.as_deref()
            && publishers
                .get(execution_id)
                .and_then(Weak::upgrade)
                .is_some()
        {
            return Err(crate::error::ReactError::Other(format!(
                "Subagent event publisher already exists for execution '{execution_id}'"
            )));
        }
        streams.insert(stream_id, Arc::downgrade(&publisher.inner));
        if let Some(execution_id) = execution_id {
            publishers.insert(execution_id, Arc::downgrade(&publisher.inner));
        }
        Ok(publisher)
    }

    /**
     * 发出一个非执行类,或仅用于兼容目的的原始事件。
     *
     * 调度生命周期的生产者必须使用 [`Self::publisher`],这样信封身份与
     * 排序才不会被意外绕过。
     */
    pub fn emit(&self, event: SubagentEvent) {
        if let Err(error) = self.try_emit(event) {
            tracing::warn!(%error, "rejected raw Subagent execution event");
        }
    }

    /**
     * 发出一个仅限于注册表(registry)场景的原始事件;若传入的是必须经过
     * 尝试级发布者(attempt publisher)的执行事件,则会拒绝。
     */
    pub fn try_emit(&self, event: SubagentEvent) -> crate::error::Result<()> {
        if is_execution_event(&event) {
            return Err(crate::error::ReactError::Other(
                "Subagent execution events require an attempt publisher".to_string(),
            ));
        }
        self.emit_raw(Arc::new(event));
        Ok(())
    }

    /// 将一个原始事件分发给同步监听器,并广播到原始事件信道。
    fn emit_raw(&self, event: Arc<SubagentEvent>) {
        let listeners = self
            .sync_listeners
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();
        for listener in &listeners {
            listener.on_event(event.as_ref());
        }
        let _ = self.raw_tx.send(event);
    }

    /// 将一个事件信封纳入重放窗口,并广播到原始事件与信封两个信道。
    fn emit_envelope(&self, envelope: Arc<SubagentEventEnvelope>) {
        self.replay
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .retain(Arc::clone(&envelope));
        self.emit_raw(Arc::new(envelope.payload.event.clone()));
        let _ = self.envelope_tx.send(envelope);
    }

    /// 根据 execution id 查找对应的活跃发布者(若其已被回收则一并清理索引)。
    pub(crate) fn publisher_for_execution(
        &self,
        execution_id: &str,
    ) -> Option<SubagentEventPublisher> {
        let mut publishers = self
            .active_publishers
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let publisher = publishers
            .get(execution_id)
            .and_then(Weak::upgrade)
            .map(|inner| SubagentEventPublisher { inner });
        if publisher.is_none() {
            publishers.remove(execution_id);
        }
        publisher
    }

    /// 在一次调度尝试终态结算后,将对应发布者从活跃索引中移除。
    fn retire_publisher(&self, publisher: &SubagentEventPublisher) {
        let stream_id = &publisher.inner.transport_identity.stream_id;
        let mut streams = self
            .active_streams
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let remove_stream = streams
            .get(stream_id)
            .and_then(Weak::upgrade)
            .is_none_or(|inner| Arc::ptr_eq(&inner, &publisher.inner));
        if remove_stream {
            streams.remove(stream_id);
        }
        let Some(execution_id) = publisher.inner.transport_identity.execution_id.as_ref() else {
            return;
        };
        let mut publishers = self
            .active_publishers
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let should_remove = publishers
            .get(execution_id.as_str())
            .and_then(Weak::upgrade)
            .is_none_or(|inner| Arc::ptr_eq(&inner, &publisher.inner));
        if should_remove {
            publishers.remove(execution_id.as_str());
        }
    }

    /// 查询指定 execution 下某个工具调用起始事件的 event id。
    pub(crate) fn parent_event_id_for_tool(
        &self,
        execution_id: &str,
        call_id: &str,
    ) -> Option<EventId> {
        self.publisher_for_execution(execution_id)
            .and_then(|publisher| {
                publisher
                    .inner
                    .state
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .tool_started
                    .get(call_id)
                    .cloned()
            })
    }

    /// 重放某个具体事件流中仍被保留的后缀部分。
    pub fn replay_after(&self, stream_id: &StreamId, after_sequence: u64) -> SubagentEventReplay {
        let (mut events, terminal) = {
            let replay = self
                .replay
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let events = replay
                .history
                .iter()
                .chain(replay.boundaries.iter())
                .chain(replay.terminals.iter())
                .filter(|event| event.stream_id == *stream_id && event.sequence > after_sequence)
                .cloned()
                .collect::<Vec<_>>();
            let terminal = replay
                .terminals
                .iter()
                .rev()
                .find(|event| event.stream_id == *stream_id)
                .cloned();
            (events, terminal)
        };
        events.sort_by_key(|event| event.sequence);
        events.dedup_by_key(|event| event.sequence);
        let retained_latest = events
            .iter()
            .map(|event| event.sequence)
            .max()
            .unwrap_or(after_sequence);
        let active_latest = self.active_latest_for_stream(stream_id);
        let latest_sequence = active_latest
            .map(|sequence| sequence.max(retained_latest))
            .unwrap_or(retained_latest);
        let mut contiguous_through = after_sequence;
        let mut gap = None;
        for event in &events {
            let Some(expected) = contiguous_through.checked_add(1) else {
                break;
            };
            if event.sequence > expected {
                gap = Some(SubagentEventGap {
                    stream_id: stream_id.clone(),
                    requested_after: contiguous_through,
                    available_from: Some(event.sequence),
                    latest_sequence,
                });
                break;
            }
            if event.sequence == expected {
                contiguous_through = event.sequence;
            }
        }
        if gap.is_none() && contiguous_through < latest_sequence {
            gap = Some(SubagentEventGap {
                stream_id: stream_id.clone(),
                requested_after: contiguous_through,
                available_from: None,
                latest_sequence,
            });
        }
        SubagentEventReplay {
            events,
            gap,
            terminal,
        }
    }

    /**
     * 在无需预先知道其随机生成的传输层事件流标识的情况下,解析并重放
     * 一次被保留的调度尝试。
     */
    pub fn replay_for_execution(
        &self,
        execution_id: &str,
        after_sequence: u64,
    ) -> Option<SubagentEventReplay> {
        let retained_stream_id = {
            let replay = self
                .replay
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            replay
                .history
                .iter()
                .chain(replay.boundaries.iter())
                .chain(replay.terminals.iter())
                .filter(|event| {
                    event
                        .execution_id
                        .as_ref()
                        .is_some_and(|candidate| candidate.as_str() == execution_id)
                })
                .max_by_key(|event| event.timestamp)
                .map(|event| event.stream_id.clone())
        };
        let stream_id = retained_stream_id.or_else(|| {
            self.publisher_for_execution(execution_id)
                .map(|publisher| publisher.stream_id().clone())
        })?;
        Some(self.replay_after(&stream_id, after_sequence))
    }

    /**
     * 枚举所有仍保留着事件、边界或终态记录的事件流。进程级消费者会在
     * 广播出现滞后(lag)之后使用此方法,以恢复那些在收到滞后信号之前
     * 未曾观测到的短生命周期调度尝试。
     */
    pub fn retained_stream_ids(&self) -> Vec<StreamId> {
        let replay = self
            .replay
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut stream_ids = replay
            .history
            .iter()
            .chain(replay.boundaries.iter())
            .chain(replay.terminals.iter())
            .map(|event| event.stream_id.clone())
            .collect::<Vec<_>>();
        stream_ids.sort_by(|left, right| left.as_str().cmp(right.as_str()));
        stream_ids.dedup();
        stream_ids
    }

    /**
     * 枚举活跃的调度尝试事件流,包括那些其完整可重放后缀已经被其他
     * 生产者淘汰出保留窗口的事件流。
     */
    pub fn active_stream_ids(&self) -> Vec<StreamId> {
        let mut streams = self
            .active_streams
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        streams.retain(|_, candidate| candidate.strong_count() > 0);
        let mut stream_ids = streams.keys().cloned().collect::<Vec<_>>();
        stream_ids.sort_by(|left, right| left.as_str().cmp(right.as_str()));
        stream_ids
    }

    /**
     * 返回某个活跃事件流不可变的调度起始信封。仅当重放报告一个完全被
     * 淘汰的缺口时,消费者才应将其用作路由身份;不得将其误认为一条
     * 新重放出来的事件。
     */
    pub fn active_stream_anchor(&self, stream_id: &StreamId) -> Option<Arc<SubagentEventEnvelope>> {
        let mut streams = self
            .active_streams
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let publisher = streams.get(stream_id).and_then(Weak::upgrade);
        if publisher.is_none() {
            streams.remove(stream_id);
        }
        publisher.and_then(|publisher| {
            publisher
                .state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .dispatch_anchor
                .clone()
        })
    }

    /// 返回某个活跃事件流当前已分配到的最新序号(若该发布者仍存活)。
    fn active_latest_for_stream(&self, stream_id: &StreamId) -> Option<u64> {
        let mut streams = self
            .active_streams
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let publisher = streams.get(stream_id).and_then(Weak::upgrade);
        if publisher.is_none() {
            streams.remove(stream_id);
        }
        publisher.map(|publisher| {
            publisher
                .state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .sequence
        })
    }

    /// 获取当前订阅异步事件流的活跃订阅者数量。
    pub fn subscriber_count(&self) -> usize {
        self.raw_tx.receiver_count()
    }

    /// 获取当前订阅事件信封流的活跃订阅者数量。
    pub fn envelope_subscriber_count(&self) -> usize {
        self.envelope_tx.receiver_count()
    }
}

impl Clone for SubagentEventBus {
    fn clone(&self) -> Self {
        Self {
            raw_tx: self.raw_tx.clone(),
            envelope_tx: self.envelope_tx.clone(),
            sync_listeners: Arc::clone(&self.sync_listeners),
            replay: Arc::clone(&self.replay),
            active_publishers: Arc::clone(&self.active_publishers),
            active_streams: Arc::clone(&self.active_streams),
        }
    }
}

/// 判断一个事件是否属于调度尝试的终态事件(完成/失败/取消)。
fn is_terminal_event(event: &SubagentEvent) -> bool {
    matches!(
        event,
        SubagentEvent::DispatchCompleted { .. }
            | SubagentEvent::DispatchFailed { .. }
            | SubagentEvent::DispatchCancelled { .. }
    )
}

/// 判断一个事件是否属于必须经由尝试级发布者发出的执行事件(而非注册表事件)。
fn is_execution_event(event: &SubagentEvent) -> bool {
    !matches!(
        event,
        SubagentEvent::Registered { .. } | SubagentEvent::Unregistered { .. }
    )
}

/// 判断一个事件是否可以作为重放边界保留(排除高频的增量事件)。
fn is_replay_boundary(event: &SubagentEvent) -> bool {
    !matches!(
        event,
        SubagentEvent::DispatchThinkingDelta { .. } | SubagentEvent::DispatchTokenDelta { .. }
    )
}

impl Default for SubagentEventBus {
    fn default() -> Self {
        Self::new()
    }
}

/// 从一个事件中提取出的路由标识信息。
struct SubagentEventRoute<'a> {
    /// 父 Agent 名称。
    parent: &'a str,
    /// Subagent 名称。
    agent: &'a str,
    /// execution id(如果存在)。
    execution_id: Option<&'a str>,
    /// run id(如果存在)。
    run_id: Option<&'a str>,
}

/// 从一个 Subagent 事件中提取路由标识;注册表事件(无路由信息)返回 `None`。
fn event_route(event: &SubagentEvent) -> Option<SubagentEventRoute<'_>> {
    let (parent, agent, execution_id, run_id) = match event {
        SubagentEvent::UplinkReceived {
            parent,
            agent,
            execution_id,
            run_id,
            ..
        }
        | SubagentEvent::DispatchStarted {
            parent,
            agent,
            execution_id,
            run_id,
            ..
        }
        | SubagentEvent::DispatchIsolationObserved {
            parent,
            agent,
            execution_id,
            run_id,
            ..
        }
        | SubagentEvent::DispatchCompleted {
            parent,
            agent,
            execution_id,
            run_id,
            ..
        }
        | SubagentEvent::DispatchFailed {
            parent,
            agent,
            execution_id,
            run_id,
            ..
        }
        | SubagentEvent::DispatchCancelled {
            parent,
            agent,
            execution_id,
            run_id,
            ..
        }
        | SubagentEvent::DispatchThinkingStarted {
            parent,
            agent,
            execution_id,
            run_id,
        }
        | SubagentEvent::DispatchThinkingDelta {
            parent,
            agent,
            execution_id,
            run_id,
            ..
        }
        | SubagentEvent::DispatchThinkingEnded {
            parent,
            agent,
            execution_id,
            run_id,
            ..
        }
        | SubagentEvent::DispatchTokenDelta {
            parent,
            agent,
            execution_id,
            run_id,
            ..
        }
        | SubagentEvent::DispatchLlmUsage {
            parent,
            agent,
            execution_id,
            run_id,
            ..
        }
        | SubagentEvent::DispatchToolStarted {
            parent,
            agent,
            execution_id,
            run_id,
            ..
        }
        | SubagentEvent::DispatchToolCompleted {
            parent,
            agent,
            execution_id,
            run_id,
            ..
        } => (parent, agent, execution_id, run_id),
        SubagentEvent::Registered { .. } | SubagentEvent::Unregistered { .. } => return None,
    };
    Some(SubagentEventRoute {
        parent,
        agent,
        execution_id: execution_id.as_deref(),
        run_id: run_id.as_deref(),
    })
}

/// 校验事件携带的路由标识与传输层身份、调用身份是否一致。
fn validate_event_route(
    route: SubagentEventRoute<'_>,
    transport: &EventIdentity,
    invocation: &SubagentInvocationIdentity,
) -> crate::error::Result<()> {
    let expected_execution_id = transport.execution_id.as_ref().map(|value| value.as_str());
    let expected_run_id = transport.run_id.as_ref().map(|value| value.as_str());
    if route.parent != invocation.parent_agent {
        return Err(crate::error::ReactError::Other(format!(
            "Subagent event parent changed: expected '{}', got '{}'",
            invocation.parent_agent, route.parent
        )));
    }
    if route.agent != invocation.agent_name {
        return Err(crate::error::ReactError::Other(format!(
            "Subagent event agent changed: expected '{}', got '{}'",
            invocation.agent_name, route.agent
        )));
    }
    if route.execution_id != expected_execution_id {
        return Err(crate::error::ReactError::Other(
            "Subagent event execution identity changed".to_string(),
        ));
    }
    if route.run_id != expected_run_id {
        return Err(crate::error::ReactError::Other(
            "Subagent event run identity changed".to_string(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use echo_core::agent::ToolInvocationRewrite;
    use echo_core::tools::{ToolFailure, ToolFailureCategory, ToolResultKind};
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicBool, Ordering};

    #[test]
    fn tool_events_round_trip_without_losing_invocation_or_result_fields()
    -> Result<(), Box<dyn std::error::Error>> {
        let invocation = ToolInvocation {
            requested_name: "requested_shell".to_string(),
            requested_args: serde_json::json!({"command": "echo requested"}),
            name: "shell".to_string(),
            args: serde_json::json!({"command": "echo effective"}),
            rewrites: vec![ToolInvocationRewrite::Approval],
        };
        let started = SubagentEvent::DispatchToolStarted {
            parent: "root".to_string(),
            agent: "explorer".to_string(),
            call_id: "call-1".to_string(),
            invocation: invocation.clone(),
            execution_id: Some("task-1:1".to_string()),
            run_id: Some("run-1".to_string()),
        };
        let started = serde_json::from_value::<SubagentEvent>(serde_json::to_value(started)?)?;
        let SubagentEvent::DispatchToolStarted {
            invocation: decoded_invocation,
            ..
        } = started
        else {
            return Err(std::io::Error::other("tool-start event changed variant").into());
        };
        assert_eq!(decoded_invocation, invocation);

        let result = ToolResult {
            kind: ToolResultKind::Json,
            success: false,
            output: "preview".to_string(),
            error: Some("partial failure".to_string()),
            failure: Some(ToolFailure::new(ToolFailureCategory::PartialSideEffect)),
            data: Some(serde_json::json!({"partial": true})),
            truncated: true,
            mime_type: Some("application/json".to_string()),
            artifact: Some(echo_core::tools::artifact::ToolOutputArtifactRef {
                path: std::path::PathBuf::from("/tmp/tool.log"),
                artifact_bytes: 12,
                payload_bytes: 12,
                sha256: "a".repeat(64),
                retention: "test".to_string(),
            }),
            metadata: HashMap::new(),
            effects: Vec::new(),
            model_content: Vec::new(),
        };
        let completed = SubagentEvent::DispatchToolCompleted {
            parent: "root".to_string(),
            agent: "explorer".to_string(),
            call_id: "call-1".to_string(),
            name: "shell".to_string(),
            result,
            execution_id: Some("task-1:1".to_string()),
            run_id: Some("run-1".to_string()),
        };
        let completed = serde_json::from_value::<SubagentEvent>(serde_json::to_value(completed)?)?;
        let SubagentEvent::DispatchToolCompleted {
            name,
            result: decoded_result,
            ..
        } = completed
        else {
            return Err(std::io::Error::other("tool-result event changed variant").into());
        };
        assert_eq!(name, "shell");
        assert_eq!(decoded_result.kind, ToolResultKind::Json);
        assert!(!decoded_result.success);
        assert_eq!(decoded_result.output, "preview");
        assert_eq!(decoded_result.error.as_deref(), Some("partial failure"));
        assert_eq!(
            decoded_result.data,
            Some(serde_json::json!({"partial": true}))
        );
        assert_eq!(
            decoded_result.mime_type.as_deref(),
            Some("application/json")
        );
        assert!(decoded_result.truncated);
        assert_eq!(
            decoded_result
                .artifact
                .as_ref()
                .map(|artifact| artifact.path.as_path()),
            Some(std::path::Path::new("/tmp/tool.log"))
        );
        assert_eq!(
            decoded_result.failure.map(|failure| failure.category),
            Some(ToolFailureCategory::PartialSideEffect)
        );
        Ok(())
    }

    #[test]
    fn test_event_bus_emit() {
        let bus = SubagentEventBus::new();
        bus.emit(SubagentEvent::Registered {
            name: "test".into(),
        });
    }

    #[tokio::test]
    async fn test_event_bus_subscribe() -> Result<(), Box<dyn std::error::Error>> {
        let bus = SubagentEventBus::new();
        let mut rx = bus.subscribe();

        bus.emit(SubagentEvent::Registered {
            name: "test".into(),
        });

        let event = rx.try_recv()?;
        if let SubagentEvent::Registered { name } = event.as_ref() {
            assert_eq!(name, "test");
            Ok(())
        } else {
            Err(std::io::Error::other("wrong event type").into())
        }
    }

    fn publisher(bus: &SubagentEventBus) -> crate::error::Result<SubagentEventPublisher> {
        publisher_for(bus, "subagent-stream-1", "execution-1")
    }

    fn publisher_for(
        bus: &SubagentEventBus,
        stream_id: &str,
        execution_id: &str,
    ) -> crate::error::Result<SubagentEventPublisher> {
        let transport = EventIdentity::new(stream_id, format!("turn-{execution_id}"))?
            .with_conversation_id("conversation-1")?
            .with_run_id("run-1")?
            .with_message_id("message-1")?
            .with_execution_id(execution_id)?;
        bus.publisher(
            transport,
            SubagentInvocationIdentity {
                parent_agent: "root".to_string(),
                agent_name: "explorer".to_string(),
                parent_execution_id: None,
                agent_path: Some("root/explorer".to_string()),
                task_id: Some("task-1".to_string()),
                attempt: Some(1),
                plan_revision: Some(2),
            },
        )
    }

    fn started_event() -> SubagentEvent {
        started_event_for("execution-1")
    }

    fn started_event_for(execution_id: &str) -> SubagentEvent {
        SubagentEvent::DispatchStarted {
            parent: "root".to_string(),
            agent: "explorer".to_string(),
            mode: ExecutionMode::Sync,
            task: "inspect".to_string(),
            execution_id: Some(execution_id.to_string()),
            run_id: Some("run-1".to_string()),
            conversation_id: Some("conversation-1".to_string()),
            message_id: Some("message-1".to_string()),
            background: false,
        }
    }

    fn completed_event_for(execution_id: &str) -> SubagentEvent {
        SubagentEvent::DispatchCompleted {
            parent: "root".to_string(),
            agent: "explorer".to_string(),
            duration_ms: 1,
            tokens_used: None,
            iterations: Some(1),
            output: "done".to_string(),
            outcome: SubagentOutcome::terminal(SubagentStatus::Completed, "done", Vec::new()),
            execution_id: Some(execution_id.to_string()),
            run_id: Some("run-1".to_string()),
        }
    }

    #[tokio::test]
    async fn publisher_sequences_envelopes_and_preserves_raw_compatibility()
    -> Result<(), Box<dyn std::error::Error>> {
        let bus = SubagentEventBus::with_capacity(16);
        let mut raw = bus.subscribe();
        let mut envelopes = bus.subscribe_envelopes();
        let publisher = publisher(&bus)?;

        let started = publisher.emit(started_event())?;
        publisher.emit(SubagentEvent::DispatchThinkingStarted {
            parent: "root".to_string(),
            agent: "explorer".to_string(),
            execution_id: Some("execution-1".to_string()),
            run_id: Some("run-1".to_string()),
        })?;
        let tool_started = publisher.emit(SubagentEvent::DispatchToolStarted {
            parent: "root".to_string(),
            agent: "explorer".to_string(),
            call_id: "call-1".to_string(),
            invocation: ToolInvocation {
                requested_name: "shell".to_string(),
                requested_args: serde_json::json!({}),
                name: "shell".to_string(),
                args: serde_json::json!({}),
                rewrites: Vec::new(),
            },
            execution_id: Some("execution-1".to_string()),
            run_id: Some("run-1".to_string()),
        })?;
        assert_eq!(
            bus.parent_event_id_for_tool("execution-1", "call-1"),
            Some(tool_started.event_id.clone())
        );
        let tool_completed = publisher.emit(SubagentEvent::DispatchToolCompleted {
            parent: "root".to_string(),
            agent: "explorer".to_string(),
            call_id: "call-1".to_string(),
            name: "shell".to_string(),
            result: ToolResult::success("done"),
            execution_id: Some("execution-1".to_string()),
            run_id: Some("run-1".to_string()),
        })?;
        assert!(
            bus.parent_event_id_for_tool("execution-1", "call-1")
                .is_none()
        );
        publisher.emit(SubagentEvent::DispatchCompleted {
            parent: "root".to_string(),
            agent: "explorer".to_string(),
            duration_ms: 1,
            tokens_used: Some(2),
            iterations: Some(1),
            output: "done".to_string(),
            outcome: SubagentOutcome::terminal(SubagentStatus::Completed, "done", Vec::new()),
            execution_id: Some("execution-1".to_string()),
            run_id: Some("run-1".to_string()),
        })?;

        let mut captured = Vec::new();
        for _ in 0..5 {
            captured.push(envelopes.recv().await?);
        }
        assert_eq!(captured.first().map(|event| event.sequence), Some(1));
        assert_eq!(captured.last().map(|event| event.sequence), Some(5));
        assert_eq!(
            tool_completed.parent_event_id,
            Some(tool_started.event_id.clone())
        );
        assert_eq!(
            captured
                .get(1)
                .and_then(|event| event.parent_event_id.as_ref()),
            Some(&started.event_id)
        );
        assert!(matches!(
            raw.recv().await?.as_ref(),
            SubagentEvent::DispatchStarted { .. }
        ));
        Ok(())
    }

    #[test]
    fn publisher_rejects_raw_bypass_lifecycle_drift_and_orphan_tools()
    -> Result<(), Box<dyn std::error::Error>> {
        let bus = SubagentEventBus::with_capacity(16);
        let mut raw = bus.subscribe();
        assert!(bus.try_emit(started_event()).is_err());
        assert!(matches!(
            raw.try_recv(),
            Err(broadcast::error::TryRecvError::Empty)
        ));

        let publisher = publisher(&bus)?;
        let mut wrong_conversation = started_event();
        if let SubagentEvent::DispatchStarted {
            conversation_id, ..
        } = &mut wrong_conversation
        {
            *conversation_id = Some("conversation-other".to_string());
        }
        assert!(publisher.emit(wrong_conversation).is_err());
        let mut wrong_message = started_event();
        if let SubagentEvent::DispatchStarted { message_id, .. } = &mut wrong_message {
            *message_id = Some("message-other".to_string());
        }
        assert!(publisher.emit(wrong_message).is_err());
        assert!(
            publisher
                .emit(SubagentEvent::DispatchTokenDelta {
                    parent: "root".to_string(),
                    agent: "explorer".to_string(),
                    content: "before-start".to_string(),
                    execution_id: Some("execution-1".to_string()),
                    run_id: Some("run-1".to_string()),
                })
                .is_err()
        );
        assert_eq!(publisher.emit(started_event())?.sequence, 1);
        assert!(publisher.emit(started_event()).is_err());
        assert!(
            publisher
                .emit(SubagentEvent::DispatchToolCompleted {
                    parent: "root".to_string(),
                    agent: "explorer".to_string(),
                    call_id: "orphan".to_string(),
                    name: "shell".to_string(),
                    result: ToolResult::success("done"),
                    execution_id: Some("execution-1".to_string()),
                    run_id: Some("run-1".to_string()),
                })
                .is_err()
        );
        let tool_started = SubagentEvent::DispatchToolStarted {
            parent: "root".to_string(),
            agent: "explorer".to_string(),
            call_id: "call-1".to_string(),
            invocation: ToolInvocation {
                requested_name: "shell".to_string(),
                requested_args: serde_json::json!({}),
                name: "shell".to_string(),
                args: serde_json::json!({}),
                rewrites: Vec::new(),
            },
            execution_id: Some("execution-1".to_string()),
            run_id: Some("run-1".to_string()),
        };
        assert_eq!(publisher.emit(tool_started.clone())?.sequence, 2);
        assert!(publisher.emit(tool_started).is_err());
        for (parent, agent) in [("other-parent", "explorer"), ("root", "other-agent")] {
            assert!(
                publisher
                    .emit(SubagentEvent::DispatchTokenDelta {
                        parent: parent.to_string(),
                        agent: agent.to_string(),
                        content: "wrong-route".to_string(),
                        execution_id: Some("execution-1".to_string()),
                        run_id: Some("run-1".to_string()),
                    })
                    .is_err()
            );
        }
        assert!(
            publisher
                .emit(SubagentEvent::DispatchTokenDelta {
                    parent: "root".to_string(),
                    agent: "explorer".to_string(),
                    content: "wrong-run".to_string(),
                    execution_id: Some("execution-1".to_string()),
                    run_id: Some("run-other".to_string()),
                })
                .is_err()
        );
        assert_eq!(
            publisher
                .emit(SubagentEvent::DispatchToolCompleted {
                    parent: "root".to_string(),
                    agent: "explorer".to_string(),
                    call_id: "call-1".to_string(),
                    name: "shell".to_string(),
                    result: ToolResult::success("done"),
                    execution_id: Some("execution-1".to_string()),
                    run_id: Some("run-1".to_string()),
                })?
                .sequence,
            3
        );
        Ok(())
    }

    #[test]
    fn invocation_identity_rejects_conflicting_caller_lineage()
    -> Result<(), Box<dyn std::error::Error>> {
        let transport = EventIdentity::new("lineage-stream", "lineage-turn")?
            .with_run_id("run-1")?
            .with_execution_id("execution-1")?;
        for lineage in [
            SubagentLineage {
                agent_name: Some("other".to_string()),
                ..SubagentLineage::default()
            },
            SubagentLineage {
                parent_agent: Some("other-parent".to_string()),
                ..SubagentLineage::default()
            },
            SubagentLineage {
                execution_id: Some("execution-other".to_string()),
                ..SubagentLineage::default()
            },
            SubagentLineage {
                run_id: Some("run-other".to_string()),
                ..SubagentLineage::default()
            },
        ] {
            assert!(
                SubagentInvocationIdentity::from_lineage(
                    "root",
                    "explorer",
                    &transport,
                    Some(&lineage),
                )
                .is_err()
            );
        }
        Ok(())
    }

    #[tokio::test]
    async fn bounded_replay_reports_gap_and_retains_terminal()
    -> Result<(), Box<dyn std::error::Error>> {
        let bus = SubagentEventBus::with_capacity(2);
        let mut receiver = bus.subscribe_envelopes();
        let publisher = publisher(&bus)?;
        publisher.emit(started_event())?;
        for content in ["a", "b", "c"] {
            publisher.emit(SubagentEvent::DispatchTokenDelta {
                parent: "root".to_string(),
                agent: "explorer".to_string(),
                content: content.to_string(),
                execution_id: Some("execution-1".to_string()),
                run_id: Some("run-1".to_string()),
            })?;
        }
        let terminal = publisher.emit(SubagentEvent::DispatchCompleted {
            parent: "root".to_string(),
            agent: "explorer".to_string(),
            duration_ms: 1,
            tokens_used: None,
            iterations: Some(1),
            output: "abc".to_string(),
            outcome: SubagentOutcome::terminal(SubagentStatus::Completed, "abc", Vec::new()),
            execution_id: Some("execution-1".to_string()),
            run_id: Some("run-1".to_string()),
        })?;

        let replay = bus.replay_after(publisher.stream_id(), 0);
        assert_eq!(replay.events.len(), 3);
        assert_eq!(replay.gap.as_ref().map(|gap| gap.requested_after), Some(1));
        assert_eq!(
            replay.gap.as_ref().and_then(|gap| gap.available_from),
            Some(4)
        );
        assert_eq!(replay.gap.as_ref().map(|gap| gap.latest_sequence), Some(5));
        assert_eq!(
            replay
                .terminal
                .as_ref()
                .map(|event| event.event_id.as_str()),
            Some(terminal.event_id.as_str())
        );
        assert!(matches!(
            receiver.recv().await,
            Err(broadcast::error::RecvError::Lagged(3))
        ));
        assert!(
            publisher
                .emit(SubagentEvent::DispatchTokenDelta {
                    parent: "root".to_string(),
                    agent: "explorer".to_string(),
                    content: "late".to_string(),
                    execution_id: Some("execution-1".to_string()),
                    run_id: Some("run-1".to_string()),
                })
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn replay_finds_terminal_by_execution_when_dispatch_start_was_evicted()
    -> Result<(), Box<dyn std::error::Error>> {
        let bus = SubagentEventBus::with_capacity(2);
        let first = publisher_for(&bus, "stream-first", "execution-first")?;
        first.emit(started_event_for("execution-first"))?;
        first.emit(completed_event_for("execution-first"))?;
        let second = publisher_for(&bus, "stream-second", "execution-second")?;
        second.emit(started_event_for("execution-second"))?;
        second.emit(completed_event_for("execution-second"))?;

        let replay = bus
            .replay_for_execution("execution-first", 0)
            .ok_or_else(|| std::io::Error::other("retained terminal was not discoverable"))?;
        assert_eq!(replay.events.len(), 1);
        assert_eq!(replay.gap.as_ref().map(|gap| gap.requested_after), Some(0));
        assert_eq!(
            replay.gap.as_ref().and_then(|gap| gap.available_from),
            Some(2)
        );
        assert!(replay.terminal.is_some());
        assert!(bus.publisher_for_execution("execution-first").is_none());
        assert!(
            publisher_for(&bus, "stream-reused", "execution-first")
                .err()
                .is_some()
        );
        Ok(())
    }

    #[test]
    fn active_stream_watermark_reports_fully_evicted_suffix()
    -> Result<(), Box<dyn std::error::Error>> {
        let bus = SubagentEventBus::with_capacity(1);
        let first = publisher_for(&bus, "stream-active-first", "execution-active-first")?;
        first.emit(started_event_for("execution-active-first"))?;
        let second = publisher_for(&bus, "stream-active-second", "execution-active-second")?;
        second.emit(started_event_for("execution-active-second"))?;

        let replay = bus
            .replay_for_execution("execution-active-first", 0)
            .ok_or_else(|| std::io::Error::other("active stream was not discoverable"))?;
        assert!(replay.events.is_empty());
        assert_eq!(replay.gap.as_ref().map(|gap| gap.latest_sequence), Some(1));
        assert_eq!(replay.gap.as_ref().and_then(|gap| gap.available_from), None);
        Ok(())
    }

    #[test]
    fn active_stream_watermark_reports_missing_tail_after_retained_prefix()
    -> Result<(), Box<dyn std::error::Error>> {
        let bus = SubagentEventBus::with_capacity(2);
        let first = publisher_for(&bus, "stream-tail-first", "execution-tail-first")?;
        first.emit(started_event_for("execution-tail-first"))?;
        first.emit(SubagentEvent::DispatchTokenDelta {
            parent: "root".to_string(),
            agent: "explorer".to_string(),
            content: "lost-tail".to_string(),
            execution_id: Some("execution-tail-first".to_string()),
            run_id: Some("run-1".to_string()),
        })?;
        let second = publisher_for(&bus, "stream-tail-second", "execution-tail-second")?;
        second.emit(started_event_for("execution-tail-second"))?;
        second.emit(SubagentEvent::DispatchTokenDelta {
            parent: "root".to_string(),
            agent: "explorer".to_string(),
            content: "other-stream".to_string(),
            execution_id: Some("execution-tail-second".to_string()),
            run_id: Some("run-1".to_string()),
        })?;

        let replay = bus.replay_after(first.stream_id(), 0);
        assert_eq!(
            replay
                .events
                .iter()
                .map(|event| event.sequence)
                .collect::<Vec<_>>(),
            vec![1]
        );
        assert_eq!(replay.gap.as_ref().map(|gap| gap.requested_after), Some(1));
        assert_eq!(replay.gap.as_ref().map(|gap| gap.latest_sequence), Some(2));
        assert_eq!(replay.gap.as_ref().and_then(|gap| gap.available_from), None);
        Ok(())
    }

    #[test]
    fn legacy_stream_without_execution_id_keeps_active_gap_watermark()
    -> Result<(), Box<dyn std::error::Error>> {
        let bus = SubagentEventBus::with_capacity(1);
        let legacy = bus.publisher(
            EventIdentity::new("stream-legacy", "turn-legacy")?,
            SubagentInvocationIdentity {
                parent_agent: "root".to_string(),
                agent_name: "legacy".to_string(),
                parent_execution_id: None,
                agent_path: Some("root/legacy".to_string()),
                task_id: None,
                attempt: None,
                plan_revision: None,
            },
        )?;
        legacy.emit(SubagentEvent::DispatchStarted {
            parent: "root".to_string(),
            agent: "legacy".to_string(),
            mode: ExecutionMode::Sync,
            task: "legacy".to_string(),
            execution_id: None,
            run_id: None,
            conversation_id: None,
            message_id: None,
            background: false,
        })?;
        let other = publisher_for(&bus, "stream-with-execution", "execution-other")?;
        other.emit(started_event_for("execution-other"))?;

        let replay = bus.replay_after(legacy.stream_id(), 0);
        assert!(replay.events.is_empty());
        assert_eq!(replay.gap.as_ref().map(|gap| gap.latest_sequence), Some(1));
        assert_eq!(replay.gap.as_ref().and_then(|gap| gap.available_from), None);
        Ok(())
    }

    #[test]
    fn publisher_index_rejects_duplicates_and_cleans_dropped_entries()
    -> Result<(), Box<dyn std::error::Error>> {
        let bus = SubagentEventBus::new();
        let first = publisher_for(&bus, "stream-duplicate", "execution-duplicate")?;
        assert!(
            publisher_for(&bus, "stream-conflict", "execution-duplicate")
                .err()
                .is_some()
        );
        drop(first);

        for index in 0..32 {
            let publisher = publisher_for(
                &bus,
                &format!("stream-drop-{index}"),
                &format!("execution-drop-{index}"),
            )?;
            drop(publisher);
        }
        let final_publisher = publisher_for(&bus, "stream-final", "execution-final")?;
        assert_eq!(
            bus.active_publishers
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .len(),
            1
        );
        drop(final_publisher);
        assert!(bus.publisher_for_execution("execution-final").is_none());
        let _ = bus.replay_after(&StreamId::new("stream-final")?, 0);
        assert!(
            bus.active_publishers
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .is_empty()
        );
        assert!(
            bus.active_streams
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .is_empty()
        );
        Ok(())
    }

    struct NoopListener;

    impl SubagentEventListener for NoopListener {
        fn on_event(&self, _event: &SubagentEvent) {}
    }

    struct ReentrantListener {
        bus: SubagentEventBus,
        called: Arc<AtomicBool>,
    }

    impl SubagentEventListener for ReentrantListener {
        fn on_event(&self, _event: &SubagentEvent) {
            let mut bus = self.bus.clone();
            bus.register(Box::new(NoopListener));
            self.called.store(true, Ordering::SeqCst);
        }
    }

    #[test]
    fn sync_listener_may_register_another_listener_without_deadlock() {
        let mut bus = SubagentEventBus::new();
        let called = Arc::new(AtomicBool::new(false));
        bus.register(Box::new(ReentrantListener {
            bus: bus.clone(),
            called: Arc::clone(&called),
        }));
        bus.emit(SubagentEvent::Registered {
            name: "test".to_string(),
        });
        assert!(called.load(Ordering::SeqCst));
    }

    #[test]
    fn retained_stream_ids_include_distinct_unobserved_attempts()
    -> Result<(), Box<dyn std::error::Error>> {
        let bus = SubagentEventBus::with_capacity(4);
        for (stream_id, execution_id) in [
            ("retained-stream-b", "retained-execution-b"),
            ("retained-stream-a", "retained-execution-a"),
        ] {
            let publisher = publisher_for(&bus, stream_id, execution_id)?;
            publisher.emit(SubagentEvent::DispatchStarted {
                parent: "root".to_string(),
                agent: "explorer".to_string(),
                mode: ExecutionMode::Fork,
                task: "inspect".to_string(),
                execution_id: Some(execution_id.to_string()),
                run_id: Some("run-1".to_string()),
                conversation_id: Some("conversation-1".to_string()),
                message_id: Some("message-1".to_string()),
                background: true,
            })?;
        }
        assert_eq!(
            bus.retained_stream_ids()
                .into_iter()
                .map(|stream_id| stream_id.as_str().to_string())
                .collect::<Vec<_>>(),
            vec![
                "retained-stream-a".to_string(),
                "retained-stream-b".to_string(),
            ]
        );
        Ok(())
    }

    #[test]
    fn active_stream_identity_survives_complete_replay_eviction()
    -> Result<(), Box<dyn std::error::Error>> {
        let bus = SubagentEventBus::with_capacity(1);
        let active = publisher_for(&bus, "active-stream", "active-execution")?;
        active.emit(started_event_for("active-execution"))?;
        let noisy = publisher_for(&bus, "noisy-stream", "noisy-execution")?;
        noisy.emit(started_event_for("noisy-execution"))?;

        let active_ids = bus
            .active_stream_ids()
            .into_iter()
            .map(|stream_id| stream_id.as_str().to_string())
            .collect::<Vec<_>>();
        assert!(active_ids.contains(&"active-stream".to_string()));
        let anchor = bus
            .active_stream_anchor(active.stream_id())
            .ok_or_else(|| std::io::Error::other("active stream anchor was not retained"))?;
        assert_eq!(anchor.sequence, 1);
        assert_eq!(anchor.stream_id.as_str(), "active-stream");
        assert_eq!(
            anchor.execution_id.as_ref().map(|value| value.as_str()),
            Some("active-execution")
        );
        Ok(())
    }

    #[test]
    fn test_logging_listener() {
        let listener = LoggingSubagentListener;
        listener.on_event(&SubagentEvent::Registered {
            name: "test".into(),
        });
    }
}
