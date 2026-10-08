use crate::agent::{AgentEvent, EventEnvelope};
use crate::error::{ReactError, Result};
use crate::runtime::{EventSink, SinkControl};
use agent_client_protocol::schema::v1::{
    ContentBlock, ContentChunk, SessionId, SessionNotification, SessionUpdate, TextContent,
    ToolCall, ToolCallStatus, ToolCallUpdate, ToolCallUpdateFields,
};
use agent_client_protocol::{Client, ConnectionTo};
use async_trait::async_trait;
use std::sync::Mutex;

/// 投影器（projector）的内部可变状态。
#[derive(Default)]
struct ProjectionState {
    /// 当前是否处于“思考中”阶段。
    thinking: bool,
    /// 本轮是否已经发出过 Agent 正文文本。
    emitted_agent_text: bool,
    /// 本轮已发送的 update 数量。
    updates_sent: usize,
    /// 本轮已发送 update 的累计字符数。
    total_update_chars: usize,
}

/**
一次运行（run）已提交事件的标准 ACP 视图。

投影器（projector）永远不拥有事件本身：共享的运行账本（run ledger）
会先提交每一个被接受的 [`EventEnvelope`]，而这个类型只是对同一批
事实渲染出受限的 `session/update` 投影（对应设计文档 §11.1 —— 该
投影不允许引入新的事件序列或终态）。
*/
pub struct AcpEventProjector {
    /// 本次会话的 ACP session id。
    session_id: SessionId,
    /// 用于向客户端发送通知的连接句柄。
    connection: ConnectionTo<Client>,
    /// 单条 update 允许的最大字符数。
    max_update_chars: usize,
    /// 单轮（turn）允许发送的最大 update 数量。
    max_updates_per_turn: usize,
    /// 单轮累计允许发送的最大字符总数。
    max_total_update_chars: usize,
    /// 投影器的内部可变状态。
    state: Mutex<ProjectionState>,
}

impl AcpEventProjector {
    /// 创建一个新的 [`AcpEventProjector`]。
    pub fn new(
        session_id: SessionId,
        connection: ConnectionTo<Client>,
        max_update_chars: usize,
        max_updates_per_turn: usize,
        max_total_update_chars: usize,
    ) -> Self {
        Self {
            session_id,
            connection,
            max_update_chars,
            max_updates_per_turn,
            max_total_update_chars,
            state: Mutex::new(ProjectionState::default()),
        }
    }

    /**
    渲染并发送一个已提交 envelope 对应的标准 update。
    越界（bounds violation）或发送失败都会被上报为投递失败；
    这类失败永远不会变成第二个 Agent 执行终态。
    */
    pub async fn emit(&self, envelope: &EventEnvelope) -> Result<()> {
        for notification in self.project(envelope)? {
            self.reserve(&notification)?;
            self.connection
                .send_notification(notification)
                .map_err(|error| {
                    ReactError::Other(format!("failed to send ACP session/update: {error}"))
                })?;
        }
        Ok(())
    }

    /// 将一个事件信封（envelope）渲染为若干条 ACP session 通知。
    fn project(&self, envelope: &EventEnvelope) -> Result<Vec<SessionNotification>> {
        let updates = self.updates(&envelope.payload)?;
        Ok(updates
            .into_iter()
            .map(|update| SessionNotification::new(self.session_id.clone(), update))
            .collect())
    }

    /// 根据单个 Agent 事件计算出应当发出的 ACP update 列表。
    fn updates(&self, payload: &AgentEvent) -> Result<Vec<SessionUpdate>> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| ReactError::Other("ACP projector state lock poisoned".to_string()))?;
        let updates = match payload {
            AgentEvent::ThinkStart => {
                state.thinking = true;
                Vec::new()
            }
            AgentEvent::ThinkEnd { .. } => {
                state.thinking = false;
                Vec::new()
            }
            AgentEvent::Token(text) => {
                validate_update_size(text, self.max_update_chars)?;
                let chunk = ContentChunk::new(ContentBlock::Text(TextContent::new(text.clone())));
                if state.thinking {
                    vec![SessionUpdate::AgentThoughtChunk(chunk)]
                } else {
                    state.emitted_agent_text = true;
                    vec![SessionUpdate::AgentMessageChunk(chunk)]
                }
            }
            AgentEvent::FinalAnswer(answer) if !state.emitted_agent_text => {
                validate_update_size(answer, self.max_update_chars)?;
                state.emitted_agent_text = true;
                vec![SessionUpdate::AgentMessageChunk(ContentChunk::new(
                    ContentBlock::Text(TextContent::new(answer.clone())),
                ))]
            }
            AgentEvent::ToolCall {
                call_id,
                invocation,
            } => {
                validate_update_size(&invocation.name, self.max_update_chars)?;
                let raw_input = encode_bounded(&invocation.args, self.max_update_chars)?;
                vec![SessionUpdate::ToolCall(
                    ToolCall::new(call_id.clone(), invocation.name.clone())
                        .status(ToolCallStatus::Pending)
                        .raw_input(raw_input),
                )]
            }
            AgentEvent::ToolStream { call_id, event, .. } => {
                let output = encode_bounded(event, self.max_update_chars)?;
                vec![SessionUpdate::ToolCallUpdate(ToolCallUpdate::new(
                    call_id.clone(),
                    ToolCallUpdateFields::new()
                        .status(ToolCallStatus::InProgress)
                        .raw_output(output),
                ))]
            }
            AgentEvent::ToolResult {
                call_id, result, ..
            } => {
                let failed = !result.success;
                let output = encode_bounded(result, self.max_update_chars)?;
                vec![SessionUpdate::ToolCallUpdate(ToolCallUpdate::new(
                    call_id.clone(),
                    ToolCallUpdateFields::new()
                        .status(if failed {
                            ToolCallStatus::Failed
                        } else {
                            ToolCallStatus::Completed
                        })
                        .raw_output(output),
                ))]
            }
            _ => Vec::new(),
        };
        Ok(updates)
    }

    /// 为一条 notification 登记预算（字符数与条数），超出配置的上限时返回错误。
    fn reserve(&self, notification: &SessionNotification) -> Result<()> {
        let serialized = serde_json::to_string(notification).map_err(|error| {
            ReactError::Other(format!("failed to encode ACP session/update: {error}"))
        })?;
        let update_chars = serialized.chars().count();
        if update_chars > self.max_update_chars {
            return Err(ReactError::Other(format!(
                "ACP update exceeds the configured {} character limit",
                self.max_update_chars
            )));
        }
        let mut state = self
            .state
            .lock()
            .map_err(|_| ReactError::Other("ACP projector state lock poisoned".to_string()))?;
        let updates_sent = state
            .updates_sent
            .checked_add(1)
            .ok_or_else(|| ReactError::Other("ACP update count overflow".to_string()))?;
        let total_update_chars = state
            .total_update_chars
            .checked_add(update_chars)
            .ok_or_else(|| ReactError::Other("ACP update size overflow".to_string()))?;
        if updates_sent > self.max_updates_per_turn
            || total_update_chars > self.max_total_update_chars
        {
            return Err(ReactError::Other(
                "ACP Turn exceeded its cumulative update budget".to_string(),
            ));
        }
        state.updates_sent = updates_sent;
        state.total_update_chars = total_update_chars;
        Ok(())
    }
}

#[async_trait]
impl EventSink for AcpEventProjector {
    /// 将事件投影为 ACP 通知发出，并始终继续接收后续事件。
    async fn on_event(&self, envelope: EventEnvelope) -> Result<SinkControl> {
        self.emit(&envelope).await?;
        Ok(SinkControl::Continue)
    }
}

/// 校验文本长度是否超过允许的最大字符数。
fn validate_update_size(text: &str, max_chars: usize) -> Result<()> {
    if text.chars().count() > max_chars {
        return Err(ReactError::Other(format!(
            "ACP update exceeds the configured {max_chars} character limit"
        )));
    }
    Ok(())
}

/// 将值序列化为 JSON，并在超出大小限制时返回错误。
fn encode_bounded<T: serde::Serialize>(value: &T, max_chars: usize) -> Result<serde_json::Value> {
    let encoded = serde_json::to_string(value)
        .map_err(|error| ReactError::Other(format!("failed to encode ACP tool update: {error}")))?;
    validate_update_size(&encoded, max_chars)?;
    serde_json::from_str(&encoded).map_err(|error| {
        ReactError::Other(format!("failed to materialize ACP tool update: {error}"))
    })
}
