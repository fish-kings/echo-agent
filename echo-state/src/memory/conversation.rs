/*!
对话持久化——具体实现与辅助函数

trait 定义与数据类型位于 [`echo_core::memory::conversation`]。
本模块提供 `project_messages`/`project_message` 辅助函数。
*/

use echo_core::error::{MemoryError, Result};
use echo_core::llm::types::{Message, MessageContent, Role, ToolCall};
pub use echo_core::memory::conversation::StoredMessage;
use sha2::{Digest, Sha256};

/// 当前消息投影格式的版本号。
const MESSAGE_PROJECTION_VERSION: u8 = 1;

/**
消息投影的元数据结构，用于在 `attachments_json` 中携带无法用普通字段
表达的结构化信息，并支持后续版本演进。
*/
#[derive(serde::Serialize, serde::Deserialize)]
struct MessageProjectionMeta {
    /// 投影格式版本号，用于兼容性校验。
    #[serde(rename = "_echo_message_version")]
    version: u8,
    /// 结构化内容（仅当消息内容为 `Parts` 或 `Empty` 时存在）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    content: Option<MessageContent>,
    /// 模型的推理/思考内容（如有）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
    /// 消息关联的名称（如工具名）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    /// 关联的工具调用 ID（如有）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
    /// 该消息所属的会话记录生成 ID（用于追加排序）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    transcript_generation_id: Option<String>,
    /// 该消息在所属生成内的序号。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    transcript_ordinal: Option<u64>,
}

/// 单条产品级会话记录消息的类型化追加标识。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranscriptProjectionMeta {
    /// 消息所属的生成（generation）ID。
    pub generation_id: String,
    /// 消息在该生成内的序号。
    pub ordinal: u64,
}

/**
为单条用户可见的会话记录行计算稳定的生成游标指纹。

计算时排除后端 ID、时间戳以及生成/序号元数据，使得当前活跃游标能够在
投影批次准备前后匹配到同一条逻辑消息。
*/
pub fn transcript_projection_message_digest(message: &StoredMessage) -> Result<String> {
    let normalized = serde_json::to_vec(&(
        &message.role,
        &message.content,
        normalized_transcript_attachments(message)?,
        &message.tool_calls_json,
        &message.tool_result_json,
    ))?;
    Ok(format!("{:x}", Sha256::digest(normalized)))
}

/// 将运行时 Message 列表投影为可持久化的会话记录。
pub fn project_messages(conversation_id: &str, messages: &[Message]) -> Result<Vec<StoredMessage>> {
    messages
        .iter()
        .map(|message| project_message(conversation_id, message))
        .collect()
}

/// 将单条运行时 Message 投影为一条会话记录。
pub fn project_message(conversation_id: &str, message: &Message) -> Result<StoredMessage> {
    let tool_calls_json = message
        .tool_calls
        .as_ref()
        .map(serde_json::to_string)
        .transpose()?;

    let tool_result_json = if message.role == Role::Tool {
        Some(
            serde_json::json!({
                "tool_call_id": message.tool_call_id,
                "name": message.name,
            })
            .to_string(),
        )
    } else {
        None
    };

    /*
    `content` 是可供搜索的文本投影。每条规范化的运行时消息都会携带该
    标记（包括纯文本消息），使应用层投影能够可靠地区分由 Agent 生成的
    会话记录与仅用于 UI 引导的记录。
    */
    let structured_content = matches!(
        &message.content,
        MessageContent::Parts(_) | MessageContent::Empty
    )
    .then(|| message.content.clone());
    let attachments_json = Some(serde_json::to_string(&MessageProjectionMeta {
        version: MESSAGE_PROJECTION_VERSION,
        content: structured_content,
        reasoning_content: message.reasoning_content.clone(),
        name: message.name.clone(),
        tool_call_id: message.tool_call_id.clone(),
        transcript_generation_id: None,
        transcript_ordinal: None,
    })?);

    Ok(StoredMessage {
        id: None,
        conversation_id: conversation_id.to_string(),
        role: message.role.as_str().to_string(),
        content: message.text_content(),
        attachments_json,
        tool_calls_json,
        tool_result_json,
        created_at: echo_core::utils::time::now_local().to_rfc3339(),
    })
}

/// 为一条规范化投影消息附加其生成内局部序号。
pub fn set_transcript_projection_meta(
    stored: &mut StoredMessage,
    generation_id: &str,
    ordinal: u64,
) -> Result<()> {
    let mut projection = restore_projection_meta(stored)?.ok_or_else(|| {
        MemoryError::SerializationError(
            "transcript projection metadata requires a canonical message projection".to_string(),
        )
    })?;
    projection.transcript_generation_id = Some(generation_id.to_string());
    projection.transcript_ordinal = Some(ordinal);
    stored.attachments_json = Some(serde_json::to_string(&projection)?);
    Ok(())
}

/// 从一条规范化投影消息中读取其生成内局部序号。
pub fn transcript_projection_meta(
    stored: &StoredMessage,
) -> Result<Option<TranscriptProjectionMeta>> {
    let Some(projection) = restore_projection_meta(stored)? else {
        return Ok(None);
    };
    Ok(
        match (
            projection.transcript_generation_id,
            projection.transcript_ordinal,
        ) {
            (Some(generation_id), Some(ordinal)) => Some(TranscriptProjectionMeta {
                generation_id,
                ordinal,
            }),
            (None, None) => None,
            _ => {
                return Err(MemoryError::SerializationError(
                    "transcript projection generation and ordinal must be present together"
                        .to_string(),
                )
                .into());
            }
        },
    )
}

/// 返回去除了生成内追加坐标信息（generation/ordinal）的附件元数据。
pub fn normalized_transcript_attachments(stored: &StoredMessage) -> Result<Option<String>> {
    let Some(mut projection) = restore_projection_meta(stored)? else {
        return Ok(stored.attachments_json.clone());
    };
    projection.transcript_generation_id = None;
    projection.transcript_ordinal = None;
    Ok(Some(serde_json::to_string(&projection)?))
}

/** 将一条持久化的会话记录还原为运行时 [`Message`]。

这是 [`project_message`] 的逆操作：解析 role 以及
`tool_calls_json` / `tool_result_json` 投影，还原为结构化的
`Message`。对四种规范角色（system/user/assistant/tool）以及携带
工具调用的 assistant 消息可实现无损往返。

# 错误

以下情况会返回错误：
- 存在 `tool_calls_json` 但无法反序列化为 `Vec<ToolCall>`（数据损坏——
  会显式报错而不是静默丢弃）。
- `tool`角色消息存在 `tool_result_json` 但无法反序列化为
  `{ tool_call_id, name }`。
- role 不是 `system`/`user`/`assistant`/`tool` 之一。未知角色会报错，
  而不是静默降级为 `user`，使调用方能够检测到 schema 漂移，而不是
  悄悄丢失角色信息。
*/
pub fn restore_message(stored: &StoredMessage) -> Result<Message> {
    let text = stored.content.clone().unwrap_or_default();
    let projection = restore_projection_meta(stored)?;
    let mut message = match stored.role.as_str() {
        "system" => Message::system(text),
        "assistant" => {
            let calls = stored
                .tool_calls_json
                .as_deref()
                .map(serde_json::from_str::<Vec<ToolCall>>)
                .transpose()?;
            match calls {
                Some(calls) => {
                    let mut message = Message::assistant_with_tools(calls);
                    if !text.is_empty() {
                        message.content = MessageContent::Text(text);
                    }
                    message
                }
                None => Message::assistant(text),
            }
        }
        "tool" => {
            /// `tool_result_json` 中携带的工具结果元数据。
            #[derive(serde::Deserialize)]
            struct ToolResultMeta {
                /// 关联的工具调用 ID。
                tool_call_id: Option<String>,
                /// 工具名称。
                name: Option<String>,
            }
            let meta = stored
                .tool_result_json
                .as_deref()
                .map(serde_json::from_str::<ToolResultMeta>)
                .transpose()?;
            /*
            如果投影携带了调用 ID/名称，则直接使用；否则回退到一个稳定的
            占位符，保证工具消息依旧格式良好（下游代码依据 role==Tool
            判断，而不是依据该 ID）。
            */
            let tool_call_id = meta
                .as_ref()
                .and_then(|value| value.tool_call_id.clone())
                .unwrap_or_else(|| "unknown_tool_call".to_string());
            let name = meta.and_then(|value| value.name).unwrap_or_default();
            Message::tool_result(tool_call_id, name, text)
        }
        "user" => Message::user(text),
        other => {
            return Err(MemoryError::SerializationError(format!(
                "cannot restore message: unknown role '{other}'"
            ))
            .into());
        }
    };
    if let Some(projection) = projection {
        if let Some(content) = projection.content {
            message.content = content;
        }
        message.reasoning_content = projection.reasoning_content;
        message.name = projection.name;
        message.tool_call_id = projection.tool_call_id;
    }
    Ok(message)
}

/**
尝试从 `attachments_json` 中解析出投影元数据；若不是本模块写入的
规范化投影负载，则返回 `None`。
*/
fn restore_projection_meta(stored: &StoredMessage) -> Result<Option<MessageProjectionMeta>> {
    let Some(raw) = stored.attachments_json.as_deref() else {
        return Ok(None);
    };
    /*
    应用层也会使用 attachments_json 存放 UI 元数据。这里只认领携带
    我们显式标记的负载；无关或历史遗留的负载会有意留给应用层投影处理。
    */
    let Ok(value) = serde_json::from_str::<serde_json::Value>(raw) else {
        return Ok(None);
    };
    if value.get("_echo_message_version").is_none() {
        return Ok(None);
    }
    let projection: MessageProjectionMeta = serde_json::from_value(value)?;
    if projection.version != MESSAGE_PROJECTION_VERSION {
        return Err(MemoryError::SerializationError(format!(
            "cannot restore message projection version {}",
            projection.version
        ))
        .into());
    }
    Ok(Some(projection))
}

/**
将一批持久化的会话记录还原为运行时 [`Message`] 列表。

是 [`restore_message`] 的批量形式；遇到第一条无法还原的记录即失败
（与 `project_messages` 的快速失败语义一致）。
*/
pub fn restore_messages(stored: &[StoredMessage]) -> Result<Vec<Message>> {
    stored.iter().map(restore_message).collect()
}

/// 对话投影与还原的单元测试。
#[cfg(test)]
mod tests {
    use super::*;

    /// 验证空内容及消息身份字段（name、tool_call_id）可经投影无损往返。
    #[test]
    fn projection_round_trips_empty_content_and_message_identity() -> Result<()> {
        let mut original = Message::assistant(String::new());
        original.content = MessageContent::Empty;
        original.name = Some("assistant-name".to_string());
        original.tool_call_id = Some("carried-call-id".to_string());

        let stored = project_message("conversation", &original)?;
        let restored = restore_message(&stored)?;

        assert!(matches!(restored.content, MessageContent::Empty));
        assert_eq!(restored.name.as_deref(), Some("assistant-name"));
        assert_eq!(restored.tool_call_id.as_deref(), Some("carried-call-id"));
        Ok(())
    }

    /// 验证纯文本消息的投影同样携带规范化标记。
    #[test]
    fn plain_text_projection_carries_canonical_marker() -> Result<()> {
        let stored = project_message("conversation", &Message::user("hello".to_string()))?;
        let projection = stored.attachments_json.as_deref().ok_or_else(|| {
            MemoryError::SerializationError("missing canonical projection marker".to_string())
        })?;

        assert!(projection.contains("_echo_message_version"));
        assert_eq!(
            restore_message(&stored)?.text_content().as_deref(),
            Some("hello")
        );
        Ok(())
    }
}
