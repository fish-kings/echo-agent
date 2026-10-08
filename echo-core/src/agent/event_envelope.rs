/*!
框架事件传输契约：事件信封（`EventEnvelope`）、强类型身份标识以及事件流包装与轨迹校验。

本模块为原始 `AgentEvent` 流补充稳定身份、单调序号、父子关联、内容哈希与终止语义，
并提供不依赖重新运行模型即可校验已捕获事件轨迹的工具函数。
*/

use super::AgentEvent;
#[cfg(test)]
use super::ToolInvocation;
use crate::error::Result;
use chrono::{DateTime, Utc};
use futures::stream::{BoxStream, StreamExt};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fmt;

/// 框架事件传输契约当前的 schema 版本号。
pub const AGENT_EVENT_SCHEMA_VERSION: u16 = 5;

/// 生成经过校验、在线路上透明序列化的字符串身份类型的宏。
macro_rules! identity_id {
    ($name:ident, $label:literal) => {
        #[doc = concat!("经过校验、线路透明的 `", stringify!($name), "` 值。")]
        /**

各身份类型刻意设计为互不可替换：

```compile_fail
use echo_agent::agent::{RunId, TurnId};
let run_id = RunId::new("run-1")?;
let _turn_id: TurnId = run_id;
# Ok::<(), echo_agent::Error>(())
```
        */
        #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name(String);

        impl $name {
            /// 创建身份值；若去除空白后为空则返回错误。
            pub fn new(value: impl Into<String>) -> Result<Self> {
                let value = value.into();
                if value.trim().is_empty() {
                    return Err(crate::error::ReactError::Other(format!(
                        "{} must not be empty",
                        $label
                    )));
                }
                Ok(Self(value))
            }

            /// 以字符串切片形式返回内部值。
            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl fmt::Display for $name {
            /// 直接输出内部字符串值。
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str(&self.0)
            }
        }

        impl AsRef<str> for $name {
            /// 以 `&str` 形式借用内部值。
            fn as_ref(&self) -> &str {
                self.as_str()
            }
        }
    };
}

identity_id!(StreamId, "event stream_id");
identity_id!(ConversationId, "conversation_id");
identity_id!(RunId, "run_id");
identity_id!(TurnId, "turn_id");
identity_id!(MessageId, "message_id");
identity_id!(ExecutionId, "execution_id");
identity_id!(EventId, "event_id");

/// 复制到每个已发出事件上的稳定调用身份。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EventIdentity {
    /// 某一次具体事件流调用的唯一身份。
    pub stream_id: StreamId,
    /// 所属会话身份（可选）。
    pub conversation_id: Option<ConversationId>,
    /// 正式任务运行（run）身份；普通聊天轮次保持为 `None`。
    pub run_id: Option<RunId>,
    /// 当前轮次（turn）身份。
    pub turn_id: TurnId,
    /// 当消息身份与轮次身份不同时，触发本次调用的消息身份。
    pub message_id: Option<MessageId>,
    /// 执行（execution）身份（可选）。
    pub execution_id: Option<ExecutionId>,
    /// 子调用（例如被委派的子 agent）的父事件。
    pub parent_event_id: Option<EventId>,
}

/**
包裹事件负载的带版本传输契约。

`AgentEvent` 仍是默认负载类型，因此现有使用者可以继续不带类型参数地引用
`EventEnvelope`。其他框架事件族通过提供各自可序列化的负载，复用相同的身份、
排序、哈希与时间戳规则。
*/
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventEnvelope<T = AgentEvent> {
    /// 事件传输契约的 schema 版本号。
    pub schema_version: u16,
    /// 由身份与序号确定性生成的事件 ID。
    pub event_id: EventId,
    /// 占据该逻辑槽位的父链接与负载的完整性摘要。
    pub content_hash: String,
    /// 在单个被包装的调用/执行流内单调递增，从 1 开始。
    pub sequence: u64,
    /// 事件流身份。
    pub stream_id: StreamId,
    /// 会话身份（可选）。
    pub conversation_id: Option<ConversationId>,
    /// 正式运行身份（可选）。
    pub run_id: Option<RunId>,
    /// 轮次身份。
    pub turn_id: TurnId,
    /// 触发消息身份（可选）。
    pub message_id: Option<MessageId>,
    /// 执行身份（可选）。
    pub execution_id: Option<ExecutionId>,
    /// 父事件 ID（例如工具结果指向对应的工具调用事件）。
    pub parent_event_id: Option<EventId>,
    /// 事件创建时间（UTC）。
    pub timestamp: DateTime<Utc>,
    /// 事件负载。
    pub payload: T,
}

impl<T> EventEnvelope<T>
where
    T: Serialize,
{
    /// 在事件流尚不存在、调用方又必须上报失败时，构造单个事件。
    pub fn new(
        identity: &EventIdentity,
        sequence: u64,
        parent_event_id: Option<EventId>,
        payload: T,
    ) -> Result<Self> {
        identity.validate()?;
        if sequence == 0 {
            return Err(crate::error::ReactError::Other(
                "event sequence must start at one".to_string(),
            ));
        }
        let event_id = stable_event_id(identity, sequence);
        let content_hash = event_content_hash(parent_event_id.as_ref(), &payload)?;
        Ok(Self {
            schema_version: AGENT_EVENT_SCHEMA_VERSION,
            event_id,
            content_hash,
            sequence,
            stream_id: identity.stream_id.clone(),
            conversation_id: identity.conversation_id.clone(),
            run_id: identity.run_id.clone(),
            turn_id: identity.turn_id.clone(),
            message_id: identity.message_id.clone(),
            execution_id: identity.execution_id.clone(),
            parent_event_id,
            timestamp: Utc::now(),
            payload,
        })
    }
}

/// 基于父事件链接与规范化 JSON 负载计算 `sha256:` 前缀的内容哈希。
fn event_content_hash<T>(parent_event_id: Option<&EventId>, payload: &T) -> Result<String>
where
    T: Serialize,
{
    let mut hasher = Sha256::new();
    match parent_event_id {
        Some(value) => {
            hasher.update([1]);
            hasher.update(value.as_str().as_bytes());
        }
        None => hasher.update([0]),
    }
    let encoded = crate::utils::canonical_json::canonical_json_bytes(payload).map_err(|error| {
        crate::error::ReactError::Other(format!("failed to encode event payload: {error}"))
    })?;
    hasher.update(encoded);
    Ok(format!("sha256:{:x}", hasher.finalize()))
}

/// 由 schema 版本、身份字段与序号确定性派生事件 ID（`evt_` 前缀）。
fn stable_event_id(identity: &EventIdentity, sequence: u64) -> EventId {
    let mut hasher = Sha256::new();
    hasher.update(AGENT_EVENT_SCHEMA_VERSION.to_be_bytes());
    hasher.update(identity.stream_id.as_str().as_bytes());
    for part in [
        identity
            .conversation_id
            .as_ref()
            .map(ConversationId::as_str),
        identity.run_id.as_ref().map(RunId::as_str),
        Some(identity.turn_id.as_str()),
        identity.message_id.as_ref().map(MessageId::as_str),
        identity.execution_id.as_ref().map(ExecutionId::as_str),
    ] {
        match part {
            Some(value) => {
                hasher.update([1]);
                hasher.update(u64::try_from(value.len()).unwrap_or(u64::MAX).to_be_bytes());
                hasher.update(value.as_bytes());
            }
            None => hasher.update([0]),
        }
    }
    hasher.update(sequence.to_be_bytes());
    EventId(format!("evt_{:x}", hasher.finalize()))
}

impl EventIdentity {
    /// 以流 ID 与轮次 ID 创建最小身份，其余可选字段为空。
    pub fn new(stream_id: impl Into<String>, turn_id: impl Into<String>) -> Result<Self> {
        let identity = Self {
            stream_id: StreamId::new(stream_id)?,
            conversation_id: None,
            run_id: None,
            turn_id: TurnId::new(turn_id)?,
            message_id: None,
            execution_id: None,
            parent_event_id: None,
        };
        identity.validate()?;
        Ok(identity)
    }

    /// 校验必填的流 ID 与轮次 ID 非空。
    pub fn validate(&self) -> Result<()> {
        StreamId::new(self.stream_id.as_str())?;
        TurnId::new(self.turn_id.as_str())?;
        Ok(())
    }

    /// 为正式运行构造身份，其首个轮次与执行复用同一外部分配的值。
    pub fn for_run(run_id: impl Into<String>) -> Result<Self> {
        let run_id = run_id.into();
        Ok(Self {
            stream_id: StreamId(uuid::Uuid::new_v4().to_string()),
            conversation_id: None,
            run_id: Some(RunId::new(run_id.clone())?),
            turn_id: TurnId::new(run_id.clone())?,
            message_id: None,
            execution_id: Some(ExecutionId::new(run_id)?),
            parent_event_id: None,
        })
    }

    /// 为交互式聊天轮次构造身份，同时分别保留可选的正式运行身份与触发消息身份。
    pub fn for_chat(
        conversation_id: Option<String>,
        turn_id: impl Into<String>,
        message_id: impl Into<String>,
        run_id: Option<String>,
    ) -> Result<Self> {
        Ok(Self {
            stream_id: StreamId(uuid::Uuid::new_v4().to_string()),
            conversation_id: conversation_id.map(ConversationId::new).transpose()?,
            run_id: run_id.map(RunId::new).transpose()?,
            turn_id: TurnId::new(turn_id)?,
            message_id: Some(MessageId::new(message_id)?),
            execution_id: None,
            parent_event_id: None,
        })
    }

    /// 设置执行身份并返回自身。
    pub fn with_execution_id(mut self, execution_id: impl Into<String>) -> Result<Self> {
        self.execution_id = Some(ExecutionId::new(execution_id)?);
        Ok(self)
    }

    /// 设置会话身份并返回自身。
    pub fn with_conversation_id(mut self, conversation_id: impl Into<String>) -> Result<Self> {
        self.conversation_id = Some(ConversationId::new(conversation_id)?);
        Ok(self)
    }

    /// 设置正式运行身份并返回自身。
    pub fn with_run_id(mut self, run_id: impl Into<String>) -> Result<Self> {
        self.run_id = Some(RunId::new(run_id)?);
        Ok(self)
    }

    /// 设置触发消息身份并返回自身。
    pub fn with_message_id(mut self, message_id: impl Into<String>) -> Result<Self> {
        self.message_id = Some(MessageId::new(message_id)?);
        Ok(self)
    }

    /// 设置父事件 ID 并返回自身。
    pub fn with_parent_event_id(mut self, parent_event_id: impl Into<String>) -> Result<Self> {
        self.parent_event_id = Some(EventId::new(parent_event_id)?);
        Ok(self)
    }

    /// 从单次值作用域的 agent 调用派生传输身份。
    pub fn from_invocation(invocation: &super::AgentInvocationContext) -> Result<Self> {
        Self::from_runtime_context(invocation.runtime.as_ref())
    }

    /**
    直接从外部运行时上下文派生传输身份。

    调度协调器会在子 Agent 调用创建之前使用此方法，使开始事件、内部流事件与
    终止事件共享同一个流身份。
    */
    pub fn from_runtime_context(
        runtime: Option<&crate::tools::ExternalRunContext>,
    ) -> Result<Self> {
        let run_id = runtime.and_then(|value| value.run_id.clone());
        let execution_id = runtime.and_then(|value| value.execution_id.clone());
        let turn_id = runtime
            .and_then(|value| value.turn_id.clone())
            .or_else(|| execution_id.clone())
            .or_else(|| run_id.clone())
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        Ok(Self {
            stream_id: StreamId(uuid::Uuid::new_v4().to_string()),
            conversation_id: runtime
                .and_then(|value| value.conversation_id.clone())
                .map(ConversationId::new)
                .transpose()?,
            run_id: run_id.map(RunId::new).transpose()?,
            turn_id: TurnId::new(turn_id)?,
            message_id: runtime
                .and_then(|value| value.message_id.clone())
                .map(MessageId::new)
                .transpose()?,
            execution_id: execution_id.map(ExecutionId::new).transpose()?,
            parent_event_id: runtime
                .and_then(|value| value.subagent_lineage.as_ref())
                .and_then(|value| value.parent_event_id.clone())
                .map(EventId::new)
                .transpose()?,
        })
    }
}

/**
为原始事件流包装稳定身份、顺序、关联关系与终止语义。

工具结果/进度事件会指向其对应的 `ToolCall` 事件。原始流错误以及未产生终止负载
就结束的流，都会被规范化为恰好一个终止性的 `AgentEvent::Error` 信封。
*/
pub fn envelope_event_stream<'a>(
    stream: BoxStream<'a, Result<AgentEvent>>,
    identity: EventIdentity,
) -> BoxStream<'a, Result<EventEnvelope>> {
    envelope_event_stream_after(stream, identity, 0)
}

/**
从最后一个已持久化的序号之后继续进行信封编号。

结合确定性的 `event_id` 生成，持久化适配器可以幂等地拒绝重复的副作用完成事件。
*/
pub fn envelope_event_stream_after<'a>(
    mut stream: BoxStream<'a, Result<AgentEvent>>,
    identity: EventIdentity,
    last_persisted_sequence: u64,
) -> BoxStream<'a, Result<EventEnvelope>> {
    let wrapped = async_stream::stream! {
        if let Err(error) = identity.validate() {
            yield Err(error);
            return;
        }
        let mut sequence = last_persisted_sequence;
        let mut terminal_emitted = false;
        let mut tool_calls = HashMap::<String, EventId>::new();

        while let Some(item) = stream.next().await {
            let payload = match item {
                Ok(event) => event,
                Err(error) => AgentEvent::Error {
                    source: "agent_stream".to_string(),
                    message: error.to_string(),
                    failure: crate::error::AgentFailure::from(&error),
                },
            };

            let Some(next_sequence) = sequence.checked_add(1) else {
                yield Err(crate::error::ReactError::Other(
                    "event sequence exhausted".to_string(),
                ));
                return;
            };
            sequence = next_sequence;
            let tool_call_id = match &payload {
                AgentEvent::ToolCall { call_id, .. } => Some(call_id.clone()),
                _ => None,
            };
            let parent_event_id = match &payload {
                AgentEvent::ToolResult { call_id, .. }
                | AgentEvent::ToolStream { call_id, .. } => tool_calls.get(call_id).cloned(),
                _ => identity.parent_event_id.clone(),
            };
            let is_terminal = payload.is_terminal();
            let envelope = match EventEnvelope::new(&identity, sequence, parent_event_id, payload) {
                Ok(envelope) => envelope,
                Err(error) => {
                    yield Err(error);
                    return;
                }
            };

            if let Some(call_id) = tool_call_id {
                tool_calls.insert(call_id, envelope.event_id.clone());
            }
            if let AgentEvent::ToolResult { call_id, .. } = &envelope.payload {
                tool_calls.remove(call_id);
            }

            yield Ok(envelope);
            if is_terminal {
                terminal_emitted = true;
                break;
            }
        }

        if !terminal_emitted {
            let Some(next_sequence) = sequence.checked_add(1) else {
                yield Err(crate::error::ReactError::Other(
                    "event sequence exhausted before terminal event".to_string(),
                ));
                return;
            };
            sequence = next_sequence;
            yield EventEnvelope::new(
                &identity,
                sequence,
                identity.parent_event_id.clone(),
                AgentEvent::Error {
                    source: "agent_stream".to_string(),
                    message: "agent stream ended without a terminal event".to_string(),
                    failure: crate::error::AgentFailure::message(
                        "agent_stream",
                        "agent stream ended without a terminal event",
                    ),
                },
            );
        }
    };
    Box::pin(wrapped)
}

/// 校验与负载无关的信封身份、顺序、哈希及父链接不变量，不解释特定事件族的终止语义。
pub fn validate_envelope_trajectory<T>(events: &[EventEnvelope<T>]) -> Vec<String>
where
    T: Serialize,
{
    let mut violations = Vec::new();
    let Some(first) = events.first() else {
        return vec!["trajectory is empty".to_string()];
    };
    let mut seen_ids = std::collections::HashSet::<EventId>::new();
    let mut seen_event_ids = std::collections::HashSet::<EventId>::new();

    for (index, event) in events.iter().enumerate() {
        let expected_sequence = u64::try_from(index)
            .ok()
            .and_then(|offset| first.sequence.checked_add(offset));
        if event.schema_version != AGENT_EVENT_SCHEMA_VERSION {
            violations.push(format!(
                "unsupported schema version at sequence {}: {}",
                event.sequence, event.schema_version
            ));
        }
        if expected_sequence != Some(event.sequence) {
            violations.push(format!(
                "non-contiguous sequence: expected {}, got {}",
                expected_sequence
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "sequence overflow".to_string()),
                event.sequence
            ));
        }
        if event.stream_id != first.stream_id
            || event.conversation_id != first.conversation_id
            || event.run_id != first.run_id
            || event.turn_id != first.turn_id
            || event.message_id != first.message_id
            || event.execution_id != first.execution_id
        {
            violations.push(format!("identity changed at sequence {}", event.sequence));
        }
        if !seen_event_ids.insert(event.event_id.clone()) {
            violations.push(format!("duplicate event id: {}", event.event_id));
        }
        let identity = EventIdentity {
            stream_id: event.stream_id.clone(),
            conversation_id: event.conversation_id.clone(),
            run_id: event.run_id.clone(),
            turn_id: event.turn_id.clone(),
            message_id: event.message_id.clone(),
            execution_id: event.execution_id.clone(),
            parent_event_id: first.parent_event_id.clone(),
        };
        if event.event_id != stable_event_id(&identity, event.sequence) {
            violations.push(format!("invalid event id at sequence {}", event.sequence));
        }
        match event_content_hash(event.parent_event_id.as_ref(), &event.payload) {
            Ok(expected) if event.content_hash != expected => violations.push(format!(
                "invalid content hash at sequence {}",
                event.sequence
            )),
            Err(error) => violations.push(format!(
                "could not validate content hash at sequence {}: {error}",
                event.sequence
            )),
            Ok(_) => {}
        }
        if let Some(parent_id) = event.parent_event_id.as_ref()
            && first.parent_event_id.as_ref() != Some(parent_id)
            && !seen_ids.contains(parent_id)
        {
            violations.push(format!(
                "parent event {} was not emitted before sequence {}",
                parent_id, event.sequence
            ));
        }
        seen_ids.insert(event.event_id.clone());
    }

    violations
}

/// 在不重新运行模型的情况下校验已捕获的 Agent 事件轨迹。
pub fn validate_event_trajectory(events: &[EventEnvelope]) -> Vec<String> {
    let mut violations = validate_envelope_trajectory(events);
    if events.is_empty() {
        return violations;
    }
    let mut tool_calls = HashMap::<String, EventId>::new();
    let mut terminal_count = 0_usize;

    for (index, event) in events.iter().enumerate() {
        match &event.payload {
            AgentEvent::ToolCall { call_id, .. }
                if tool_calls
                    .insert(call_id.clone(), event.event_id.clone())
                    .is_some() =>
            {
                violations.push(format!("duplicate in-flight tool call: {call_id}"));
            }
            AgentEvent::ToolCall { .. } => {}
            AgentEvent::ToolResult { call_id, .. } => match tool_calls.remove(call_id) {
                Some(parent_id) if event.parent_event_id.as_ref() == Some(&parent_id) => {}
                Some(parent_id) => violations.push(format!(
                    "tool completion {call_id} has wrong parent; expected {parent_id}"
                )),
                None => violations.push(format!("orphan tool completion: {call_id}")),
            },
            AgentEvent::ToolStream { call_id, .. } => {
                if let Some(parent_id) = tool_calls.get(call_id) {
                    if event.parent_event_id.as_ref() != Some(parent_id) {
                        violations.push(format!("tool stream {call_id} has wrong parent"));
                    }
                } else {
                    violations.push(format!("orphan tool stream: {call_id}"));
                }
            }
            _ => {}
        }
        if event.payload.is_terminal() {
            terminal_count = terminal_count.saturating_add(1);
            if index.saturating_add(1) != events.len() {
                violations.push(format!(
                    "terminal event at sequence {} is not last",
                    event.sequence
                ));
            }
        }
    }

    if terminal_count != 1 {
        violations.push(format!(
            "trajectory must contain exactly one terminal event, got {terminal_count}"
        ));
    }
    let mut unfinished = tool_calls.into_keys().collect::<Vec<_>>();
    unfinished.sort();
    for call_id in unfinished {
        violations.push(format!("tool call without completion: {call_id}"));
    }
    violations
}

/// 事件信封与标识类型的单元测试。
#[cfg(test)]
mod tests {
    use super::*;
    use futures::stream;

    /// 测试用的自定义事件负载。
    #[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
    struct TestPayload {
        label: String,
    }

    /// 构造测试用的固定事件身份。
    fn identity() -> EventIdentity {
        EventIdentity {
            stream_id: StreamId("stream-1".to_string()),
            conversation_id: Some(ConversationId("conversation-1".to_string())),
            run_id: None,
            turn_id: TurnId("turn-1".to_string()),
            message_id: Some(MessageId("message-1".to_string())),
            execution_id: Some(ExecutionId("execution-1".to_string())),
            parent_event_id: None,
        }
    }

    /// 验证事件按序编号，且工具结果关联到对应的工具调用事件。
    #[tokio::test]
    async fn sequences_events_and_correlates_tool_lifecycle() -> Result<()> {
        let raw = stream::iter(vec![
            Ok(AgentEvent::ToolCall {
                call_id: "call-1".to_string(),
                invocation: ToolInvocation {
                    requested_name: "shell".to_string(),
                    requested_args: serde_json::json!({}),
                    name: "shell".to_string(),
                    args: serde_json::json!({}),
                    rewrites: Vec::new(),
                },
            }),
            Ok(AgentEvent::ToolResult {
                call_id: "call-1".to_string(),
                name: "shell".to_string(),
                result: crate::tools::ToolResult::success("done"),
            }),
            Ok(AgentEvent::FinalAnswer("ok".to_string())),
        ]);
        let events = envelope_event_stream(Box::pin(raw), identity())
            .collect::<Vec<_>>()
            .await;
        let events = events.into_iter().collect::<Result<Vec<_>>>()?;

        assert_eq!(events.len(), 3);
        assert_eq!(events.first().map(|event| event.sequence), Some(1));
        assert_eq!(events.get(1).map(|event| event.sequence), Some(2));
        assert_eq!(events.get(2).map(|event| event.sequence), Some(3));
        assert_eq!(
            events
                .get(1)
                .and_then(|event| event.parent_event_id.as_ref()),
            events.first().map(|event| &event.event_id)
        );
        assert!(
            events
                .last()
                .is_some_and(|event| event.payload.is_terminal())
        );
        assert!(events.iter().all(|event| event.run_id.is_none()));
        Ok(())
    }

    /// 验证内容哈希与工具元数据的插入顺序无关。
    #[test]
    fn content_hash_is_independent_of_tool_metadata_insertion_order() -> Result<()> {
        let mut forward = crate::tools::ToolResult::success("done");
        forward
            .metadata
            .insert("alpha".to_string(), "1".to_string());
        forward
            .metadata
            .insert("omega".to_string(), "2".to_string());

        let mut reverse = crate::tools::ToolResult::success("done");
        reverse
            .metadata
            .insert("omega".to_string(), "2".to_string());
        reverse
            .metadata
            .insert("alpha".to_string(), "1".to_string());

        let forward = AgentEvent::ToolResult {
            call_id: "call-1".to_string(),
            name: "shell".to_string(),
            result: forward,
        };
        let reverse = AgentEvent::ToolResult {
            call_id: "call-1".to_string(),
            name: "shell".to_string(),
            result: reverse,
        };

        assert_eq!(
            event_content_hash(None, &forward)?,
            event_content_hash(None, &reverse)?
        );
        Ok(())
    }

    /// 验证序列化往返后的轨迹能按规范化嵌套元数据重新计算出相同哈希。
    #[test]
    fn serialized_trajectory_recomputes_canonical_nested_metadata_hashes() -> Result<()> {
        let tool_call = EventEnvelope::new(
            &identity(),
            1,
            None,
            AgentEvent::ToolCall {
                call_id: "call-1".to_string(),
                invocation: ToolInvocation {
                    requested_name: "shell".to_string(),
                    requested_args: serde_json::json!({"command": "true"}),
                    name: "shell".to_string(),
                    args: serde_json::json!({"command": "true"}),
                    rewrites: Vec::new(),
                },
            },
        )?;
        let mut result = crate::tools::ToolResult::success("done");
        result.metadata.insert("omega".to_string(), "2".to_string());
        result.metadata.insert("alpha".to_string(), "1".to_string());
        let tool_result = EventEnvelope::new(
            &identity(),
            2,
            Some(tool_call.event_id.clone()),
            AgentEvent::ToolResult {
                call_id: "call-1".to_string(),
                name: "shell".to_string(),
                result,
            },
        )?;
        let expected_result_hash = tool_result.content_hash.clone();
        let terminal = EventEnvelope::new(
            &identity(),
            3,
            None,
            AgentEvent::FinalAnswer("done".to_string()),
        )?;
        let encoded = serde_json::to_vec(&vec![tool_call, tool_result, terminal])
            .map_err(|error| crate::error::ReactError::Other(error.to_string()))?;
        let decoded: Vec<EventEnvelope> = serde_json::from_slice(&encoded)
            .map_err(|error| crate::error::ReactError::Other(error.to_string()))?;

        assert!(validate_event_trajectory(&decoded).is_empty());
        assert_eq!(
            decoded.get(1).map(|event| event.content_hash.as_str()),
            Some(expected_result_hash.as_str())
        );
        Ok(())
    }

    /// 验证缺失终止事件、重复终止事件与流错误均被规范化为单一终止事件。
    #[tokio::test]
    async fn normalizes_missing_and_duplicate_terminals() -> Result<()> {
        let missing = stream::iter(vec![Ok(AgentEvent::Token("partial".to_string()))]);
        let missing_events = envelope_event_stream(Box::pin(missing), identity())
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect::<Result<Vec<_>>>()?;
        assert_eq!(missing_events.len(), 2);
        assert!(
            missing_events
                .last()
                .is_some_and(|event| matches!(event.payload, AgentEvent::Error { .. }))
        );

        let duplicate = stream::iter(vec![
            Ok(AgentEvent::FinalAnswer("first".to_string())),
            Ok(AgentEvent::Cancelled),
        ]);
        let duplicate_events = envelope_event_stream(Box::pin(duplicate), identity())
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect::<Result<Vec<_>>>()?;
        assert_eq!(duplicate_events.len(), 1);
        assert!(matches!(
            duplicate_events.first().map(|event| &event.payload),
            Some(AgentEvent::FinalAnswer(answer)) if answer == "first"
        ));

        let failed = stream::iter(vec![Err(crate::error::ReactError::Other(
            "provider disconnected".to_string(),
        ))]);
        let failed_events = envelope_event_stream(Box::pin(failed), identity())
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect::<Result<Vec<_>>>()?;
        assert_eq!(failed_events.len(), 1);
        assert!(matches!(
            failed_events.first().map(|event| &event.payload),
            Some(AgentEvent::Error {
                source, message, ..
            })
                if source == "agent_stream" && message.contains("provider disconnected")
        ));
        Ok(())
    }

    /// 验证从已持久化序号之后恢复编号，且事件 ID 稳定。
    #[tokio::test]
    async fn resumes_after_persisted_sequence_with_stable_ids() -> Result<()> {
        let first = EventEnvelope::new(
            &identity(),
            8,
            None,
            AgentEvent::FinalAnswer("done".to_string()),
        )?;
        let repeated = EventEnvelope::new(
            &identity(),
            8,
            None,
            AgentEvent::FinalAnswer("done".to_string()),
        )?;
        assert_eq!(first.event_id, repeated.event_id);

        let resumed = envelope_event_stream_after(
            Box::pin(stream::iter(vec![Ok(AgentEvent::FinalAnswer(
                "resumed".to_string(),
            ))])),
            identity(),
            8,
        )
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>>>()?;
        assert_eq!(resumed.first().map(|event| event.sequence), Some(9));
        Ok(())
    }

    /// 验证业务身份相同但流不同的事件 ID 不会冲突。
    #[test]
    fn independent_streams_with_same_business_identity_do_not_collide() -> Result<()> {
        let first_identity = identity();
        let mut second_identity = first_identity.clone();
        second_identity.stream_id = StreamId("stream-2".to_string());
        let first = EventEnvelope::new(
            &first_identity,
            1,
            None,
            AgentEvent::FinalAnswer("done".to_string()),
        )?;
        let second = EventEnvelope::new(
            &second_identity,
            1,
            None,
            AgentEvent::FinalAnswer("done".to_string()),
        )?;
        assert_ne!(first.event_id, second.event_id);
        Ok(())
    }

    /// 验证信封序列化包含 schema 版本与负载类型标签。
    #[test]
    fn serializes_versioned_contract() -> Result<()> {
        let envelope = EventEnvelope::new(
            &identity(),
            1,
            None,
            AgentEvent::FinalAnswer("done".to_string()),
        )?;
        let value = serde_json::to_value(&envelope)
            .map_err(|error| crate::error::ReactError::Other(error.to_string()))?;
        assert_eq!(
            value.get("schema_version").and_then(|value| value.as_u64()),
            Some(u64::from(AGENT_EVENT_SCHEMA_VERSION))
        );
        assert_eq!(
            value
                .get("payload")
                .and_then(|payload| payload.get("type"))
                .and_then(|value| value.as_str()),
            Some("final_answer")
        );
        Ok(())
    }

    /// 验证成功、取消、错误及带人工确认/预算前缀的规范轨迹均能通过校验。
    #[tokio::test]
    async fn validates_canonical_success_cancel_error_and_hitl_budget_prefixes() -> Result<()> {
        for raw in [
            vec![
                Ok(AgentEvent::BudgetDecision {
                    decision: super::super::BudgetDecision::FinalOnly,
                    reason: "token_budget".to_string(),
                    iteration: 2,
                    reported_model_tokens: 100,
                    usage_complete: true,
                }),
                Ok(AgentEvent::FinalAnswer("完成 🧪".to_string())),
            ],
            vec![
                Ok(AgentEvent::SafetyNotice {
                    action: "write file".to_string(),
                    reason: "requested".to_string(),
                    risk: "local mutation".to_string(),
                    permission: "confirm".to_string(),
                }),
                Ok(AgentEvent::Cancelled),
            ],
            vec![Ok(AgentEvent::error_message("model", "stream interrupted"))],
        ] {
            let events = envelope_event_stream(Box::pin(stream::iter(raw)), identity())
                .collect::<Vec<_>>()
                .await
                .into_iter()
                .collect::<Result<Vec<_>>>()?;
            assert!(validate_event_trajectory(&events).is_empty());
        }
        Ok(())
    }

    /// 验证轨迹校验器能报告序号、身份、父链接与终止事件的偏差。
    #[test]
    fn trajectory_validator_reports_sequence_identity_parent_and_terminal_drift() -> Result<()> {
        let mut first = EventEnvelope::new(
            &identity(),
            1,
            Some(EventId("missing-parent".to_string())),
            AgentEvent::FinalAnswer("early".to_string()),
        )?;
        first.schema_version = AGENT_EVENT_SCHEMA_VERSION.saturating_add(1);
        let mut second = EventEnvelope::new(&identity(), 3, None, AgentEvent::Cancelled)?;
        second.turn_id = TurnId("different-turn".to_string());
        second.event_id = first.event_id.clone();
        let violations = validate_event_trajectory(&[first, second]);
        for expected in [
            "schema version",
            "not last",
            "non-contiguous",
            "identity changed",
            "duplicate event id",
            "exactly one terminal",
        ] {
            assert!(violations.iter().any(|value| value.contains(expected)));
        }
        Ok(())
    }

    /// 验证空白身份与零序号会被拒绝。
    #[test]
    fn rejects_blank_identity_and_zero_sequence() {
        let mut blank = identity();
        blank.turn_id = TurnId(String::new());
        assert!(EventEnvelope::new(&blank, 1, None, AgentEvent::Cancelled).is_err());
        assert!(EventEnvelope::new(&identity(), 0, None, AgentEvent::Cancelled).is_err());
    }

    /// 验证泛型信封复用身份、哈希与序号校验规则。
    #[test]
    fn generic_envelope_reuses_identity_hash_and_sequence_validation() -> Result<()> {
        let first = EventEnvelope::new(
            &identity(),
            4,
            None,
            TestPayload {
                label: "开始".to_string(),
            },
        )?;
        let second = EventEnvelope::new(
            &identity(),
            5,
            Some(first.event_id.clone()),
            TestPayload {
                label: "完成".to_string(),
            },
        )?;
        let encoded = serde_json::to_vec(&vec![first, second])
            .map_err(|error| crate::error::ReactError::Other(error.to_string()))?;
        let decoded: Vec<EventEnvelope<TestPayload>> = serde_json::from_slice(&encoded)
            .map_err(|error| crate::error::ReactError::Other(error.to_string()))?;

        assert!(validate_envelope_trajectory(&decoded).is_empty());
        assert_eq!(decoded.get(1).map(|event| event.sequence), Some(5));
        Ok(())
    }

    /// 验证从运行时上下文派生的身份保留嵌套父事件。
    #[test]
    fn runtime_identity_preserves_nested_parent_event() -> Result<()> {
        let runtime = crate::tools::ExternalRunContext {
            turn_id: Some("child-turn".to_string()),
            execution_id: Some("child-execution".to_string()),
            subagent_lineage: Some(crate::tools::SubagentLineage {
                parent_event_id: Some("evt_parent_tool".to_string()),
                ..crate::tools::SubagentLineage::default()
            }),
            ..crate::tools::ExternalRunContext::default()
        };
        let identity = EventIdentity::from_runtime_context(Some(&runtime))?;
        assert_eq!(
            identity.parent_event_id.as_ref().map(EventId::as_str),
            Some("evt_parent_tool")
        );
        Ok(())
    }

    /// 验证轨迹校验会拒绝被篡改的事件 ID 与内容哈希。
    #[test]
    fn trajectory_rejects_tampered_slot_and_content() -> Result<()> {
        let mut slot = EventEnvelope::new(
            &identity(),
            1,
            None,
            AgentEvent::FinalAnswer("original".to_string()),
        )?;
        slot.event_id = EventId("evt_tampered".to_string());
        slot.content_hash = "sha256:tampered".to_string();
        let violations = validate_event_trajectory(&[slot]);
        assert!(
            violations
                .iter()
                .any(|value| value.contains("invalid event id"))
        );
        assert!(
            violations
                .iter()
                .any(|value| value.contains("invalid content hash"))
        );
        Ok(())
    }

    /// 验证恢复序号已耗尽时返回错误。
    #[tokio::test]
    async fn rejects_exhausted_resume_sequence() {
        let events = envelope_event_stream_after(
            Box::pin(stream::iter(vec![Ok(AgentEvent::Cancelled)])),
            identity(),
            u64::MAX,
        )
        .collect::<Vec<_>>()
        .await;
        assert_eq!(events.len(), 1);
        assert!(events.first().is_some_and(Result::is_err));
    }
}
