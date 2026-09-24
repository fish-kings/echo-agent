//! A2A 协议类型定义
//!
//! 根据 Google A2A 协议规范定义的 Agent Card、Task 等类型。

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── JSON-RPC / A2A 常量 ─────────────────────────────────────────────────

/// JSON-RPC 协议版本
pub const JSONRPC_VERSION: &str = "2.0";

/// A2A 方法名
pub const METHOD_SEND: &str = "tasks/send";
/// 订阅任务更新。
pub const METHOD_SEND_SUBSCRIBE: &str = "tasks/sendSubscribe";
/// 获取任务状态。
pub const METHOD_GET: &str = "tasks/get";
/// 取消一个正在运行的任务。
pub const METHOD_CANCEL: &str = "tasks/cancel";

/// A2A 错误码
pub const ERROR_CODE_PARSE: i64 = -32700;
/// 方法未找到。
pub const ERROR_CODE_METHOD_NOT_FOUND: i64 = -32601;
/// 参数无效。
pub const ERROR_CODE_INVALID_PARAMS: i64 = -32602;
/// 任务执行失败。
pub const ERROR_CODE_TASK_FAILED: i64 = -32000;
/// 任务未找到。
pub const ERROR_CODE_TASK_NOT_FOUND: i64 = -32001;
/// 任务已处于终态。
pub const ERROR_CODE_TERMINAL_STATE: i64 = -32002;
/// 无效的状态转换。
pub const ERROR_CODE_INVALID_TRANSITION: i64 = -32003;

// ── Agent Card ───────────────────────────────────────────────────────────────

/**
Agent Card——描述一个 Agent 的能力与接口（A2A 规范的核心类型）

通过 `/.well-known/agent.json` 端点发布，供其他 Agent 发现与调用。
*/
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentCard {
    /// Agent 名称
    pub name: String,
    /// Agent 描述
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Agent 服务的 URL 端点
    pub url: String,
    /// Agent 版本
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    /// Agent 提供方信息
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<AgentProvider>,
    /// Agent 技能列表
    #[serde(default)]
    pub skills: Vec<AgentSkill>,
    /// 支持的输入内容类型
    #[serde(default = "default_content_types")]
    pub default_input_modes: Vec<String>,
    /// 支持的输出内容类型
    #[serde(default = "default_content_types")]
    pub default_output_modes: Vec<String>,
    /// 认证方式
    #[serde(skip_serializing_if = "Option::is_none")]
    pub authentication: Option<AgentAuthentication>,
    /// 额外的能力标志
    #[serde(default)]
    pub capabilities: AgentCapabilities,
}

/// 默认的内容类型列表。
fn default_content_types() -> Vec<String> {
    vec!["text/plain".to_string()]
}

impl AgentCard {
    /// 创建一个 AgentCard 构建器
    pub fn builder(name: impl Into<String>, url: impl Into<String>) -> AgentCardBuilder {
        AgentCardBuilder {
            name: name.into(),
            url: url.into(),
            description: None,
            version: None,
            provider: None,
            skills: Vec::new(),
            default_input_modes: default_content_types(),
            default_output_modes: default_content_types(),
            authentication: None,
            capabilities: AgentCapabilities::default(),
        }
    }

    /// 从一个已有的 Agent trait 对象自动生成 Agent Card
    pub fn from_agent(agent: &dyn crate::agent::Agent, url: impl Into<String>) -> Self {
        let skills: Vec<AgentSkill> = agent
            .tool_definitions()
            .into_iter()
            .map(|td| AgentSkill::new(&td.function.name, &td.function.description))
            .collect();

        AgentCard {
            name: agent.name().to_string(),
            description: Some(agent.system_prompt().to_string()),
            url: url.into(),
            version: Some("1.0.0".to_string()),
            provider: None,
            skills,
            default_input_modes: default_content_types(),
            default_output_modes: default_content_types(),
            authentication: None,
            capabilities: AgentCapabilities::default(),
        }
    }
}

/// Agent Card 构建器
pub struct AgentCardBuilder {
    /// Agent 名称。
    name: String,
    /// Agent 服务的 URL 端点。
    url: String,
    /// Agent 描述。
    description: Option<String>,
    /// Agent 版本。
    version: Option<String>,
    /// Agent 提供方信息。
    provider: Option<AgentProvider>,
    /// Agent 技能列表。
    skills: Vec<AgentSkill>,
    /// 支持的输入内容类型。
    default_input_modes: Vec<String>,
    /// 支持的输出内容类型。
    default_output_modes: Vec<String>,
    /// 认证方式。
    authentication: Option<AgentAuthentication>,
    /// 额外的能力标志。
    capabilities: AgentCapabilities,
}

impl AgentCardBuilder {
    /// 设置该 agent 的描述。
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// 设置该 agent 的版本。
    pub fn version(mut self, version: impl Into<String>) -> Self {
        self.version = Some(version.into());
        self
    }

    /// 设置该 agent 的提供方信息。
    pub fn provider(mut self, provider: AgentProvider) -> Self {
        self.provider = Some(provider);
        self
    }

    /// 为该 agent 添加一个技能。
    pub fn skill(mut self, skill: AgentSkill) -> Self {
        self.skills.push(skill);
        self
    }

    /// 为该 agent 添加多个技能。
    pub fn skills(mut self, skills: Vec<AgentSkill>) -> Self {
        self.skills.extend(skills);
        self
    }

    /// 设置该 agent 支持的默认输入模式（内容类型）。
    pub fn input_modes(mut self, modes: Vec<impl Into<String>>) -> Self {
        self.default_input_modes = modes.into_iter().map(|m| m.into()).collect();
        self
    }

    /// 设置该 agent 支持的默认输出模式（内容类型）。
    pub fn output_modes(mut self, modes: Vec<impl Into<String>>) -> Self {
        self.default_output_modes = modes.into_iter().map(|m| m.into()).collect();
        self
    }

    /// 设置该 agent 的认证配置。
    pub fn authentication(mut self, auth: AgentAuthentication) -> Self {
        self.authentication = Some(auth);
        self
    }

    /// 为该 agent 启用流式能力。
    pub fn streaming(mut self) -> Self {
        self.capabilities.streaming = true;
        self
    }

    /// 为该 agent 启用推送通知能力。
    pub fn push_notifications(mut self) -> Self {
        self.capabilities.push_notifications = true;
        self
    }

    /// 使用已配置的字段构建 AgentCard。
    pub fn build(self) -> AgentCard {
        AgentCard {
            name: self.name,
            description: self.description,
            url: self.url,
            version: self.version,
            provider: self.provider,
            skills: self.skills,
            default_input_modes: self.default_input_modes,
            default_output_modes: self.default_output_modes,
            authentication: self.authentication,
            capabilities: self.capabilities,
        }
    }
}

/// Agent 提供方信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentProvider {
    /// 组织/公司名称
    pub organization: String,
    /// 联系 URL
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
}

impl AgentProvider {
    /// 使用给定的组织名称创建一个新的 AgentProvider。
    pub fn new(organization: impl Into<String>) -> Self {
        Self {
            organization: organization.into(),
            url: None,
        }
    }

    /// 设置该 agent 提供方的 URL。
    pub fn with_url(mut self, url: impl Into<String>) -> Self {
        self.url = Some(url.into());
        self
    }
}

/// Agent 技能描述
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentSkill {
    /// 技能 ID
    pub id: String,
    /// 技能名称
    pub name: String,
    /// 技能描述
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// 输入示例
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub examples: Vec<String>,
    /// 支持的输入类型
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub input_modes: Vec<String>,
    /// 支持的输出类型
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub output_modes: Vec<String>,
    /// 自定义标签
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
}

impl AgentSkill {
    /// 使用给定的名称与描述创建一个新的 AgentSkill。
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        let name_str: String = name.into();
        Self {
            id: name_str.clone(),
            name: name_str,
            description: Some(description.into()),
            examples: Vec::new(),
            input_modes: Vec::new(),
            output_modes: Vec::new(),
            tags: Vec::new(),
        }
    }

    /// 为该技能添加示例。
    pub fn with_examples(mut self, examples: Vec<impl Into<String>>) -> Self {
        self.examples = examples.into_iter().map(|e| e.into()).collect();
        self
    }

    /// 为该技能添加标签。
    pub fn with_tags(mut self, tags: Vec<impl Into<String>>) -> Self {
        self.tags = tags.into_iter().map(|t| t.into()).collect();
        self
    }
}

/// Agent 认证配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentAuthentication {
    /// 认证方式列表
    pub schemes: Vec<AuthenticationScheme>,
}

/// 认证方式
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AuthenticationScheme {
    /// 方式类型："apiKey"、"bearer"、"oauth2" 等。
    pub scheme: String,
    /// 附加配置
    #[serde(flatten)]
    pub config: HashMap<String, serde_json::Value>,
}

/// Agent 能力标志
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentCapabilities {
    /// 是否支持流式输出
    #[serde(default)]
    pub streaming: bool,
    /// 是否支持推送通知
    #[serde(default)]
    pub push_notifications: bool,
    /// 是否支持会话状态
    #[serde(default)]
    pub state_transition_history: bool,
}

// ── A2A Task 状态机 ───────────────────────────────────────────────────
//
//  submitted → working → [input-required] → completed / failed
//                       ↑___________________↓
//
//  终态：completed、failed、canceled

/**
任务生命周期状态（A2A 规范状态机）

```text
submitted → working → completed
                    → failed
                    → input-required ⇄ working

任意非终态 → canceled
```
*/
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TaskState {
    /// 任务已提交，等待处理
    Submitted,
    /// Agent 正在执行
    Working,
    /// Agent 需要更多输入才能继续
    InputRequired,
    /// 任务成功完成
    Completed,
    /// 任务执行失败
    Failed,
    /// 任务已被取消
    Canceled,
}

impl TaskState {
    /// 该状态是否为终态（completed / failed / canceled）
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Canceled)
    }

    /// 检查某次状态转换是否有效
    pub fn can_transition_to(self, next: Self) -> bool {
        if self.is_terminal() {
            return false;
        }
        matches!(
            (self, next),
            (Self::Submitted, Self::Working)
                | (Self::Submitted, Self::Canceled)
                | (Self::Working, Self::Completed)
                | (Self::Working, Self::Failed)
                | (Self::Working, Self::InputRequired)
                | (Self::Working, Self::Canceled)
                | (Self::InputRequired, Self::Working)
                | (Self::InputRequired, Self::Canceled)
        )
    }
}

/// 将该状态格式化为 A2A 协议使用的 kebab-case 字符串。
impl std::fmt::Display for TaskState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Submitted => write!(f, "submitted"),
            Self::Working => write!(f, "working"),
            Self::InputRequired => write!(f, "input-required"),
            Self::Completed => write!(f, "completed"),
            Self::Failed => write!(f, "failed"),
            Self::Canceled => write!(f, "canceled"),
        }
    }
}

// ── A2A Task 类型 ───────────────────────────────────────────────────────────

/// A2A 任务请求
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct A2ATaskRequest {
    /// JSON-RPC 版本
    pub jsonrpc: String,
    /// 请求 ID
    pub id: String,
    /// 方法名
    pub method: String,
    /// 参数
    pub params: A2ATaskParams,
}

/// A2A 任务参数
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct A2ATaskParams {
    /// 任务 ID（可选；新任务不携带该字段）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// 会话 ID
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    /// 消息内容
    pub message: A2AMessage,
}

/// A2A 消息
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct A2AMessage {
    /// 角色："user" 或 "agent"
    pub role: String,
    /// 消息各部分（parts）的列表
    pub parts: Vec<A2APart>,
}

/// 消息内容的一个部分
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum A2APart {
    /// 文本内容
    #[serde(rename = "text")]
    Text {
        /// 文本内容
        text: String,
    },
    /// 文件内容
    #[serde(rename = "file")]
    File {
        /// 文件的 MIME 类型。
        #[serde(rename = "mimeType")]
        mime_type: String,
        /// Base64 编码的文件数据。
        data: String,
    },
}

impl A2AMessage {
    /// 创建一条用户文本消息
    pub fn user_text(text: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            parts: vec![A2APart::Text { text: text.into() }],
        }
    }

    /// 创建一条 Agent 文本消息
    pub fn agent_text(text: impl Into<String>) -> Self {
        Self {
            role: "agent".to_string(),
            parts: vec![A2APart::Text { text: text.into() }],
        }
    }

    /// 获取全部文本内容
    pub fn text_content(&self) -> String {
        self.parts
            .iter()
            .filter_map(|p| match p {
                A2APart::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// A2A 任务响应
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct A2ATaskResponse {
    /// JSON-RPC 版本
    pub jsonrpc: String,
    /// 请求 ID（解析失败时为 None）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// 任务结果
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<A2ATask>,
    /// 错误信息
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<A2AError>,
}

/// A2A 任务
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct A2ATask {
    /// 任务 ID
    pub id: String,
    /// 会话 ID
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    /// 任务状态
    pub status: A2ATaskStatus,
    /// 消息历史
    #[serde(default)]
    pub history: Vec<A2AMessage>,
    /// Agent 产生的 artifact
    #[serde(default)]
    pub artifacts: Vec<A2AArtifact>,
}

/// 任务状态
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct A2ATaskStatus {
    /// 状态枚举
    pub state: TaskState,
    /// 状态消息（可能包含 Agent 的回复或错误描述）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<A2AMessage>,
    /// 状态变化的时间戳（ISO 8601）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<String>,
}

impl A2ATaskStatus {
    /// 使用给定状态与当前时间戳创建一个新的任务状态。
    pub fn new(state: TaskState) -> Self {
        Self {
            state,
            message: None,
            timestamp: Some(crate::utils::time::now_local().to_rfc3339()),
        }
    }

    /// 使用给定状态与消息创建一个新的任务状态。
    pub fn with_message(state: TaskState, message: A2AMessage) -> Self {
        Self {
            state,
            message: Some(message),
            timestamp: Some(crate::utils::time::now_local().to_rfc3339()),
        }
    }
}

/// Agent 产物（artifact，输出内容）
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct A2AArtifact {
    /// Artifact 名称
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Artifact 在列表中的索引（在流式追加时用于标识同一个 artifact）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index: Option<usize>,
    /// Artifact 的内容部分
    pub parts: Vec<A2APart>,
    /// 是否追加到具有相同索引的已有 artifact（流式场景）
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub append: bool,
}

/// A2A 错误
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct A2AError {
    /// 错误码
    pub code: i32,
    /// 错误信息
    pub message: String,
}

// ── A2A 流式事件类型 ─────────────────────────────────────────────────

// 用于 tasks/sendSubscribe 的 SSE 流式响应。
// 每个事件是一行 JSON，格式为：`data: <json>\n\n`

/// 流式响应中的事件类型
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum A2AStreamEvent {
    /// 任务状态变化事件
    #[serde(rename = "status")]
    StatusUpdate(TaskStatusUpdateEvent),
    /// Artifact 更新事件（流式输出）
    #[serde(rename = "artifact")]
    ArtifactUpdate(TaskArtifactUpdateEvent),
}

/// 任务状态变化事件
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TaskStatusUpdateEvent {
    /// 任务 ID
    pub task_id: String,
    /// 新状态
    pub status: A2ATaskStatus,
    /// 是否为该任务的最后一个事件
    #[serde(rename = "final", default)]
    pub is_final: bool,
}

/// Artifact 更新事件（增量输出）
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TaskArtifactUpdateEvent {
    /// 任务 ID
    pub task_id: String,
    /// 已更新的 artifact
    pub artifact: A2AArtifact,
    /// 是否为该任务的最后一个事件
    #[serde(rename = "final", default)]
    pub is_final: bool,
}

/// 流式 JSON-RPC 响应包装器（用于承载 SSE 的 `data:` 行）
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct A2AStreamResponse {
    /// JSON-RPC 版本
    pub jsonrpc: String,
    /// 请求 ID
    pub id: String,
    /// 事件结果
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<A2AStreamEvent>,
    /// 错误信息
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<A2AError>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_card_builder() {
        let card = AgentCard::builder("test-agent", "http://localhost:8080")
            .description("Test agent")
            .version("1.0.0")
            .skill(AgentSkill::new("calc", "Math calculation"))
            .streaming()
            .build();

        assert_eq!(card.name, "test-agent");
        assert_eq!(card.description.as_deref(), Some("Test agent"));
        assert_eq!(card.skills.len(), 1);
        assert!(card.capabilities.streaming);
    }

    #[test]
    fn test_agent_card_serialization() {
        let card = AgentCard::builder("test", "http://localhost")
            .skill(AgentSkill::new("echo", "Echo"))
            .build();

        let json = serde_json::to_string_pretty(&card).unwrap();
        assert!(json.contains("\"name\":"));
        assert!(json.contains("\"skills\":"));

        let parsed: AgentCard = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.name, "test");
    }

    #[test]
    fn test_a2a_message() {
        let msg = A2AMessage::user_text("Hello");
        assert_eq!(msg.role, "user");
        assert_eq!(msg.text_content(), "Hello");
    }

    #[test]
    fn test_agent_skill() {
        let skill = AgentSkill::new("translate", "Translate")
            .with_tags(vec!["nlp", "translation"])
            .with_examples(vec!["Translate 'hello' to Chinese"]);

        assert_eq!(skill.id, "translate");
        assert_eq!(skill.tags.len(), 2);
        assert_eq!(skill.examples.len(), 1);
    }

    // ── TaskState 状态机测试 ─────────────────────────

    #[test]
    fn test_task_state_terminal() {
        assert!(!TaskState::Submitted.is_terminal());
        assert!(!TaskState::Working.is_terminal());
        assert!(!TaskState::InputRequired.is_terminal());
        assert!(TaskState::Completed.is_terminal());
        assert!(TaskState::Failed.is_terminal());
        assert!(TaskState::Canceled.is_terminal());
    }

    #[test]
    fn test_task_state_transitions() {
        assert!(TaskState::Submitted.can_transition_to(TaskState::Working));
        assert!(TaskState::Submitted.can_transition_to(TaskState::Canceled));
        assert!(!TaskState::Submitted.can_transition_to(TaskState::Completed));

        assert!(TaskState::Working.can_transition_to(TaskState::Completed));
        assert!(TaskState::Working.can_transition_to(TaskState::Failed));
        assert!(TaskState::Working.can_transition_to(TaskState::InputRequired));
        assert!(TaskState::Working.can_transition_to(TaskState::Canceled));
        assert!(!TaskState::Working.can_transition_to(TaskState::Submitted));

        // input-required ⇄ working 循环
        assert!(TaskState::InputRequired.can_transition_to(TaskState::Working));
        assert!(TaskState::InputRequired.can_transition_to(TaskState::Canceled));
        assert!(!TaskState::InputRequired.can_transition_to(TaskState::Completed));

        // 终态不能再转换
        assert!(!TaskState::Completed.can_transition_to(TaskState::Working));
        assert!(!TaskState::Failed.can_transition_to(TaskState::Working));
        assert!(!TaskState::Canceled.can_transition_to(TaskState::Working));
    }

    #[test]
    fn test_task_state_serde_kebab_case() {
        let json = serde_json::to_string(&TaskState::InputRequired).unwrap();
        assert_eq!(json, "\"input-required\"");

        let parsed: TaskState = serde_json::from_str("\"input-required\"").unwrap();
        assert_eq!(parsed, TaskState::InputRequired);

        let parsed: TaskState = serde_json::from_str("\"working\"").unwrap();
        assert_eq!(parsed, TaskState::Working);
    }

    #[test]
    fn test_task_status_with_timestamp() {
        let status = A2ATaskStatus::new(TaskState::Working);
        assert_eq!(status.state, TaskState::Working);
        assert!(status.timestamp.is_some());
        assert!(status.message.is_none());

        let status =
            A2ATaskStatus::with_message(TaskState::Completed, A2AMessage::agent_text("done"));
        assert_eq!(status.state, TaskState::Completed);
        assert!(status.message.is_some());
    }

    #[test]
    fn test_artifact_with_streaming_fields() {
        let artifact = A2AArtifact {
            name: Some("output".to_string()),
            index: Some(0),
            parts: vec![A2APart::Text {
                text: "chunk".into(),
            }],
            append: true,
        };
        let json = serde_json::to_string(&artifact).unwrap();
        assert!(json.contains("\"index\":0"));
        assert!(json.contains("\"append\":true"));

        let non_append = A2AArtifact {
            name: None,
            index: None,
            parts: vec![A2APart::Text {
                text: "full".into(),
            }],
            append: false,
        };
        let json = serde_json::to_string(&non_append).unwrap();
        assert!(!json.contains("index"));
        assert!(!json.contains("append"));
    }

    #[test]
    fn test_stream_event_serialization() {
        let event = A2AStreamEvent::StatusUpdate(TaskStatusUpdateEvent {
            task_id: "t1".into(),
            status: A2ATaskStatus::new(TaskState::Working),
            is_final: false,
        });
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("\"type\":\"status\""));
        assert!(json.contains("\"working\""));

        let event = A2AStreamEvent::ArtifactUpdate(TaskArtifactUpdateEvent {
            task_id: "t1".into(),
            artifact: A2AArtifact {
                name: None,
                index: Some(0),
                parts: vec![A2APart::Text { text: "hi".into() }],
                append: true,
            },
            is_final: false,
        });
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("\"type\":\"artifact\""));
    }
}
