//! 审计日志核心 trait 与事件类型

use crate::error::Result;
use crate::guard::GuardDirection;
use crate::tools::permission::ToolPermission;
use crate::utils::retention::ContentRetentionPolicy;
use chrono::{DateTime, Utc};
use futures::future::BoxFuture;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// 审计事件
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEvent {
    /// 事件被记录时的时间。
    #[serde(with = "crate::utils::time::local_rfc3339")]
    pub timestamp: DateTime<Utc>,
    /// 可选的会话标识符。
    pub session_id: Option<String>,
    /// 与该事件关联的 agent 名称。
    pub agent_name: String,
    /// 描述事件种类的结构化负载。
    pub event_type: AuditEventType,
    /// 可选的分布式追踪标识符。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trace_id: Option<String>,
}

impl AuditEvent {
    /// 构造一个以当前 UTC 时间为时间戳的新事件。
    pub fn now(session_id: Option<String>, agent_name: String, event_type: AuditEventType) -> Self {
        Self {
            timestamp: Utc::now(),
            session_id,
            agent_name,
            event_type,
            trace_id: None,
        }
    }

    /// 按照内容保留策略，就地脱敏/清理该事件中携带的文本与 JSON 内容。
    pub fn apply_retention(&mut self, retention: &ContentRetentionPolicy) {
        match &mut self.event_type {
            AuditEventType::UserInput { content } | AuditEventType::FinalAnswer { content } => {
                *content = retention.sanitize_text(content);
            }
            AuditEventType::LlmCall { .. } => {}
            AuditEventType::ToolCall { input, output, .. } => {
                retention.sanitize_json(input);
                *output = retention.sanitize_text(output);
            }
            AuditEventType::GuardBlock { reason, .. } => {
                *reason = retention.sanitize_text(reason);
            }
            AuditEventType::PermissionDenied { reason, .. } => {
                *reason = retention.sanitize_text(reason);
            }
            AuditEventType::ApprovalRequested { .. } => {}
            AuditEventType::ApprovalCompleted { reason, .. } => {
                *reason = reason.as_deref().map(|text| retention.sanitize_text(text));
            }
        }
    }
}

/// 审计事件类型
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AuditEventType {
    /// agent 开始处理前捕获的原始用户输入。
    UserInput {
        /// 用户提供的内容。
        content: String,
    },
    /// 一次模型调用，附带可选的 token 计数。
    LlmCall {
        /// 该请求所使用的模型名称。
        model: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        /// 若 provider 有上报，则为提示词的 token 数。
        prompt_tokens: Option<u64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        /// 若 provider 有上报，则为补全内容的 token 数。
        completion_tokens: Option<u64>,
    },
    /// 一次工具调用及其观测到的结果。
    ToolCall {
        /// 由运行时提供的规范化工具调用标识（若存在）。
        #[serde(default, skip_serializing_if = "Option::is_none")]
        call_id: Option<String>,
        /// 工具标识符。
        tool: String,
        /// 序列化后的工具输入。
        input: Value,
        /// 工具输出文本。
        output: String,
        /// 工具是否执行成功。
        success: bool,
        /// 执行耗时（毫秒）。
        duration_ms: u64,
    },
    /// 某个 guard 拒绝了输入或输出。
    GuardBlock {
        /// Guard 名称。
        guard: String,
        /// 该 guard 作用于输入还是输出。
        direction: GuardDirection,
        /// 人类可读的原因。
        reason: String,
    },
    /// 权限策略拒绝了一次工具调用。
    PermissionDenied {
        /// 工具标识符。
        tool: String,
        /// 触发拒绝的权限项。
        required: Vec<ToolPermission>,
        /// 人类可读的原因。
        reason: String,
    },
    /// 返回给调用方的最终答案。
    FinalAnswer {
        /// 最终的响应内容。
        content: String,
    },
    /// 审批请求已发起
    ApprovalRequested {
        /// 工具名称
        tool: String,
        /// 参数哈希（用于缓存匹配）
        args_hash: String,
        /// 风险等级
        risk_level: String,
    },
    /// 审批结果已返回
    ApprovalCompleted {
        /// 工具名称
        tool: String,
        /// 决策结果
        decision: String,
        /// 审批范围
        scope: String,
        /// 拒绝原因（如有）
        reason: Option<String>,
        /// 从发起请求到得出决策的耗时（毫秒）
        duration_ms: u64,
    },
}

/// 审计查询过滤器
#[derive(Debug, Default, Clone)]
pub struct AuditFilter {
    /// 仅限定于某个会话的结果。
    pub session_id: Option<String>,
    /// 仅限定于某个 agent 名称的结果。
    pub agent_name: Option<String>,
    /// 时间戳下界（含）。
    pub from: Option<DateTime<Utc>>,
    /// 时间戳上界（含）。
    pub to: Option<DateTime<Utc>>,
    /// 返回结果的最大数量。
    pub limit: Option<usize>,
}

/// 审计日志记录器 trait
pub trait AuditLogger: Send + Sync {
    /// 持久化一条审计事件。
    fn log<'a>(&'a self, event: AuditEvent) -> BoxFuture<'a, Result<()>>;
    /// 按过滤条件查询已存储的事件。
    fn query<'a>(&'a self, filter: AuditFilter) -> BoxFuture<'a, Result<Vec<AuditEvent>>>;
}
