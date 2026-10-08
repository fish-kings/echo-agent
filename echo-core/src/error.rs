/*!
统一错误类型。

所有公开 API 均返回 [`Result<T>`]；底层错误通过 `From` 自动转换为 [`ReactError`]。
*/

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// 跨 Agent 事件与适配器边界传递的稳定失败领域分类。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentFailureCategory {
    /// LLM 调用失败。
    Llm,
    /// 工具执行失败。
    Tool,
    /// 解析失败。
    Parse,
    /// Agent 执行失败。
    Agent,
    /// 配置错误。
    Config,
    /// MCP 相关失败。
    Mcp,
    /// 记忆系统失败。
    Memory,
    /// 沙箱失败。
    Sandbox,
    /// 运行时状态失败。
    RuntimeState,
    /// 渠道 / IM 集成失败。
    Channel,
    /// I/O 失败。
    Io,
    /// 其他失败。
    Other,
}

/// 与展示文本无关的终止生命周期类别。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentTerminalKind {
    /// 执行失败。
    Failed,
    /// 被取消。
    Cancelled,
    /// 超时。
    TimedOut,
    /// 权限被拒绝。
    PermissionDenied,
}

/// 面向 Agent 事件消费者的可序列化非工具失败契约。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentFailure {
    /// 失败所属领域。
    pub category: AgentFailureCategory,
    /// 终止生命周期类别。
    pub terminal_kind: AgentTerminalKind,
    /// 是否可重试。
    pub retryable: bool,
    /// 稳定的机器可读错误码。
    pub code: String,
    /// 相关的 HTTP 状态码（如有）。
    pub http_status: Option<u16>,
    /// 人类可读的错误消息。
    pub message: String,
}

impl From<&ReactError> for AgentFailure {
    /// 将框架错误映射为稳定的失败契约（领域、终止类别、可重试性、错误码）。
    fn from(error: &ReactError) -> Self {
        let (category, terminal_kind, retryable, code, http_status) = match error {
            ReactError::Llm(inner) => match inner.as_ref() {
                LlmError::NetworkError(_) => (
                    AgentFailureCategory::Llm,
                    AgentTerminalKind::Failed,
                    true,
                    "llm_network",
                    None,
                ),
                LlmError::ApiError { status, .. } => (
                    AgentFailureCategory::Llm,
                    AgentTerminalKind::Failed,
                    *status == 408 || *status == 429 || *status >= 500,
                    "llm_api",
                    Some(*status),
                ),
                LlmError::InvalidResponse(_) => (
                    AgentFailureCategory::Llm,
                    AgentTerminalKind::Failed,
                    false,
                    "llm_invalid_response",
                    None,
                ),
                LlmError::EmptyResponse => (
                    AgentFailureCategory::Llm,
                    AgentTerminalKind::Failed,
                    true,
                    "llm_empty_response",
                    None,
                ),
                LlmError::SerializationError(_) => (
                    AgentFailureCategory::Llm,
                    AgentTerminalKind::Failed,
                    false,
                    "llm_serialization",
                    None,
                ),
            },
            ReactError::Tool(_) => (
                AgentFailureCategory::Tool,
                AgentTerminalKind::Failed,
                false,
                "tool",
                None,
            ),
            ReactError::Parse(_) => (
                AgentFailureCategory::Parse,
                AgentTerminalKind::Failed,
                false,
                "parse",
                None,
            ),
            ReactError::Agent(inner) => match inner.as_ref() {
                AgentError::Interrupted | AgentError::Cancelled(_) => (
                    AgentFailureCategory::Agent,
                    AgentTerminalKind::Cancelled,
                    false,
                    "agent_cancelled",
                    None,
                ),
                AgentError::Timeout(_) => (
                    AgentFailureCategory::Agent,
                    AgentTerminalKind::TimedOut,
                    true,
                    "agent_timeout",
                    None,
                ),
                AgentError::PermissionDenied(_) => (
                    AgentFailureCategory::Agent,
                    AgentTerminalKind::PermissionDenied,
                    false,
                    "agent_permission_denied",
                    None,
                ),
                _ => (
                    AgentFailureCategory::Agent,
                    AgentTerminalKind::Failed,
                    false,
                    "agent",
                    None,
                ),
            },
            ReactError::Config(_) => (
                AgentFailureCategory::Config,
                AgentTerminalKind::Failed,
                false,
                "config",
                None,
            ),
            #[cfg(feature = "mcp")]
            ReactError::Mcp(_) => (
                AgentFailureCategory::Mcp,
                AgentTerminalKind::Failed,
                true,
                "mcp",
                None,
            ),
            ReactError::Memory(inner) => match inner.as_ref() {
                MemoryError::DeadlineExceeded(_) => (
                    AgentFailureCategory::Memory,
                    AgentTerminalKind::TimedOut,
                    true,
                    "memory_deadline",
                    None,
                ),
                MemoryError::TransientNoCommit(_) => (
                    AgentFailureCategory::Memory,
                    AgentTerminalKind::Failed,
                    true,
                    "memory_transient_no_commit",
                    None,
                ),
                _ => (
                    AgentFailureCategory::Memory,
                    AgentTerminalKind::Failed,
                    false,
                    "memory",
                    None,
                ),
            },
            ReactError::Sandbox(inner) => match inner.as_ref() {
                SandboxError::Timeout(_) => (
                    AgentFailureCategory::Sandbox,
                    AgentTerminalKind::TimedOut,
                    true,
                    "sandbox_timeout",
                    None,
                ),
                SandboxError::Cancelled(_) => (
                    AgentFailureCategory::Sandbox,
                    AgentTerminalKind::Cancelled,
                    false,
                    "sandbox_cancelled",
                    None,
                ),
                SandboxError::PermissionDenied(_) => (
                    AgentFailureCategory::Sandbox,
                    AgentTerminalKind::PermissionDenied,
                    false,
                    "sandbox_permission_denied",
                    None,
                ),
                _ => (
                    AgentFailureCategory::Sandbox,
                    AgentTerminalKind::Failed,
                    false,
                    "sandbox",
                    None,
                ),
            },
            ReactError::RuntimeState(inner) => match inner.as_ref() {
                RuntimeStateError::TranscriptProjectionDeferred { .. } => (
                    AgentFailureCategory::RuntimeState,
                    AgentTerminalKind::Failed,
                    true,
                    "transcript_projection_deferred",
                    None,
                ),
                RuntimeStateError::DeadlineExceeded(_) => (
                    AgentFailureCategory::RuntimeState,
                    AgentTerminalKind::TimedOut,
                    true,
                    "runtime_state_deadline",
                    None,
                ),
                RuntimeStateError::TranscriptProjectionBlocked { .. } => (
                    AgentFailureCategory::RuntimeState,
                    AgentTerminalKind::Failed,
                    false,
                    "transcript_projection_blocked",
                    None,
                ),
                RuntimeStateError::Io(_) => (
                    AgentFailureCategory::RuntimeState,
                    AgentTerminalKind::Failed,
                    true,
                    "runtime_state_io",
                    None,
                ),
                RuntimeStateError::TransientNoCommit(_) => (
                    AgentFailureCategory::RuntimeState,
                    AgentTerminalKind::Failed,
                    true,
                    "runtime_state_transient_no_commit",
                    None,
                ),
                _ => (
                    AgentFailureCategory::RuntimeState,
                    AgentTerminalKind::Failed,
                    false,
                    "runtime_state",
                    None,
                ),
            },
            #[cfg(feature = "channels")]
            ReactError::Channel(_) => (
                AgentFailureCategory::Channel,
                AgentTerminalKind::Failed,
                true,
                "channel",
                None,
            ),
            ReactError::Io(_) => (
                AgentFailureCategory::Io,
                AgentTerminalKind::Failed,
                true,
                "io",
                None,
            ),
            ReactError::Other(_) => (
                AgentFailureCategory::Other,
                AgentTerminalKind::Failed,
                false,
                "other",
                None,
            ),
        };
        Self {
            category,
            terminal_kind,
            retryable,
            code: code.to_string(),
            http_status,
            message: error.to_string(),
        }
    }
}

impl AgentFailure {
    /// 以来源作为错误码，构造一个"其他/失败/不可重试"的消息型失败。
    pub fn message(source: &str, message: impl Into<String>) -> Self {
        Self {
            category: AgentFailureCategory::Other,
            terminal_kind: AgentTerminalKind::Failed,
            retryable: false,
            code: source.to_string(),
            http_status: None,
            message: message.into(),
        }
    }
}

/**
框架顶层错误，聚合所有子系统错误。

# Box 包装约定

所有子错误均使用 `Box` 包装以保持枚举体积较小（Rust 枚举大小由其最大变体决定），
从而避免在每个调用点膨胀 `Result<T>` 返回类型。`From` 实现为手工提供，
调用方无需显式装箱即可使用 `?`。
*/
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ReactError {
    /// LLM 相关错误
    #[error("LLM Error: {0}")]
    Llm(Box<LlmError>),
    /// 工具执行错误
    #[error("Tool Error: {0}")]
    Tool(Box<ToolError>),
    /// 解析错误
    #[error("Parse Error: {0}")]
    Parse(Box<ParseError>),
    /// Agent 执行错误
    #[error("Agent Error: {0}")]
    Agent(Box<AgentError>),
    /// 配置错误
    #[error("Config Error: {0}")]
    Config(Box<ConfigError>),
    /// MCP 相关错误
    #[cfg(feature = "mcp")]
    #[error("MCP Error: {0}")]
    Mcp(Box<McpError>),
    /// 记忆系统错误
    #[error("Memory Error: {0}")]
    Memory(Box<MemoryError>),
    /// 沙箱错误
    #[error("Sandbox Error: {0}")]
    Sandbox(Box<SandboxError>),
    /// 运行时状态错误
    #[error("Runtime State Error: {0}")]
    RuntimeState(Box<RuntimeStateError>),
    /// 渠道 / IM 集成错误
    #[cfg(feature = "channels")]
    #[error("Channel Error: {0}")]
    Channel(Box<ChannelError>),
    /// IO 错误
    #[error("IO Error: {0}")]
    Io(#[from] std::io::Error),
    /// 其他错误
    #[error("{0}")]
    Other(String),
}

/// 记忆系统错误
#[derive(Debug, Error)]
pub enum MemoryError {
    /// I/O 错误
    #[error("IO error: {0}")]
    IoError(String),
    /// 序列化错误
    #[error("Serialization error: {0}")]
    SerializationError(String),
    /// 记忆条目不存在
    #[error("Memory '{0}' not found")]
    NotFound(String),
    /// 不支持的操作
    #[error("Unsupported operation: {0}")]
    Unsupported(String),
    /// 旧式变更试图覆盖受管的会话记录权威数据。
    #[error("Managed conversation requires projection operation: {0}")]
    ManagedConversationRequiresProjection(String),
    /// 受管会话的投影纪元（epoch）无法再递增，否则会溢出。
    #[error("Conversation projection epoch exhausted: {0}")]
    ProjectionEpochExhausted(String),
    /// 已知发生在任何持久化提交之前的可重试失败。
    #[error("Transient no-commit failure: {0}")]
    TransientNoCommit(String),
    /// 持久化调用耗尽了共享的绝对截止时间。
    #[error("Persistence deadline exceeded: {0}")]
    DeadlineExceeded(String),
    /// 已审阅的变更提案与当前记忆状态不再匹配。
    #[error("Stale proposal: {0}")]
    StaleProposal(String),
}

impl From<std::io::Error> for MemoryError {
    /// 将标准 I/O 错误转换为记忆系统 I/O 错误。
    fn from(err: std::io::Error) -> Self {
        MemoryError::IoError(err.to_string())
    }
}

/// LLM 相关错误
#[derive(Debug, Error)]
pub enum LlmError {
    /// 网络错误
    #[error("Network error: {0}")]
    NetworkError(String),
    /// API 错误（状态码与消息）
    #[error("API error (status {status}): {message}")]
    ApiError {
        /// HTTP 状态码
        status: u16,
        /// 错误消息
        message: String,
    },
    /// 无效响应
    #[error("Invalid response: {0}")]
    InvalidResponse(String),
    /// 空响应
    #[error("Empty response from LLM")]
    EmptyResponse,
    /// 序列化错误
    #[error("Serialization error: {0}")]
    SerializationError(String),
}

/// 工具执行错误
#[derive(Debug, Error)]
pub enum ToolError {
    /// 工具不存在
    #[error("Tool '{0}' not found")]
    NotFound(String),
    /// 缺少参数
    #[error("Missing parameter: {0}")]
    MissingParameter(String),
    /// 参数无效
    #[error("Invalid parameter '{name}': {message}")]
    InvalidParameter {
        /// 参数名
        name: String,
        /// 错误消息
        message: String,
    },
    /// 工具执行失败
    #[error("Tool '{tool}' execution failed: {message}")]
    ExecutionFailed {
        /// 工具名称
        tool: String,
        /// 错误消息
        message: String,
    },
    /// 执行超时
    #[error("Tool '{0}' execution timed out")]
    Timeout(String),
    /// 执行被所属运行取消。
    #[error("Tool '{0}' execution cancelled")]
    Cancelled(String),
    /// 路径无效（检测到路径穿越攻击）
    #[error("Invalid path: {path} ({reason})")]
    InvalidPath {
        /// 被拒绝的路径
        path: String,
        /// 拒绝原因
        reason: String,
    },
    /// 访问被拒绝（超出允许的目录范围）
    #[error("Access denied: {path} ({reason})")]
    AccessDenied {
        /// 被拒绝的路径
        path: String,
        /// 拒绝原因
        reason: String,
    },
    /// 文件过大
    #[error("File too large: {size} bytes (max: {max} bytes)")]
    FileTooLarge {
        /// 文件大小（字节）
        size: u64,
        /// 允许的最大文件大小（字节）
        max: u64,
    },
}

/// 解析错误
#[derive(Debug, Error)]
pub enum ParseError {
    /// Thought 格式无效
    #[error("Invalid Thought: {0}")]
    InvalidThought(String),
    /// Action 格式无效
    #[error("Invalid Action: {0}")]
    InvalidAction(String),
    /// Action Input 无效
    #[error("Invalid Action Input: {0}")]
    InvalidActionInput(String),
    /// JSON 解析错误
    #[error("JSON parse error: {0}")]
    JsonError(#[from] serde_json::Error),
    /// 意外的格式
    #[error("Unexpected format: {0}")]
    UnexpectedFormat(String),
}

/// Agent 执行错误
#[derive(Debug, Error)]
pub enum AgentError {
    /// 超过最大迭代次数
    #[error("Max iterations exceeded: {0}")]
    MaxIterationsExceeded(usize),
    /// 没有可用工具
    #[error("No tools available")]
    NoToolsAvailable,
    /// 初始化失败
    #[error("Initialization failed: {0}")]
    InitializationFailed(String),
    /// 执行被中断
    #[error("Execution interrupted")]
    Interrupted,
    /// 通过 CancellationToken 取消了执行
    #[error("Cancelled: {0}")]
    Cancelled(String),
    /// LLM 无响应
    #[error("No response from LLM (model: {model}, agent: {agent})")]
    NoResponse {
        /// 所使用的模型名称
        model: String,
        /// Agent 名称
        agent: String,
    },
    /// 超过 token 上限
    #[error("Token limit exceeded")]
    TokenLimitExceeded,
    /// 权限被拒绝
    #[error("Permission denied: {0}")]
    PermissionDenied(String),
    /// 钩子执行错误
    #[error("Hook error: {0}")]
    HookError(String),
    /// 子 agent 执行错误
    #[error("Subagent error: {0}")]
    SubagentError(String),
    /// 执行超时
    #[error("Timeout: {0}")]
    Timeout(String),
    /// 超出上下文限制（例如委派深度、记忆上限等）
    #[error("Context limit exceeded: {0}")]
    ContextLimitExceeded(String),
}

/// MCP 相关错误
#[cfg(feature = "mcp")]
#[derive(Debug, Error)]
pub enum McpError {
    /// 连接失败
    #[error("Connection failed: {0}")]
    ConnectionFailed(String),
    /// 初始化失败
    #[error("Initialization failed: {0}")]
    InitializationFailed(String),
    /// 协议错误
    #[error("Protocol error: {0}")]
    ProtocolError(String),
    /// 工具调用失败，并附带 JSON-RPC 错误码。
    #[error("Tool call failed ({code}): {message}")]
    ToolCallFailed {
        /// JSON-RPC 错误码
        code: i32,
        /// 错误消息
        message: String,
    },
    /// 传输通道已关闭
    #[error("MCP transport closed unexpectedly")]
    TransportClosed,
}

/// 沙箱错误
#[derive(Debug, Error)]
pub enum SandboxError {
    /// 沙箱不可用（未安装 Docker、没有 K8s 集群等）
    #[error("Sandbox unavailable: {0}")]
    Unavailable(String),
    /// 沙箱启动失败
    #[error("Sandbox start failed: {0}")]
    StartFailed(String),
    /// 执行超时
    #[error("Sandbox timeout: {0}")]
    Timeout(String),
    /// 执行被所属运行取消。
    #[error("Sandbox execution cancelled: {0}")]
    Cancelled(String),
    /// 超出资源限制
    #[error("Resource exceeded: {0}")]
    ResourceExceeded(String),
    /// 权限被拒绝
    #[error("Permission denied: {0}")]
    PermissionDenied(String),
    /// IO 错误
    #[error("IO error: {0}")]
    IoError(String),
}

/// 渠道 / IM 集成错误
#[cfg(feature = "channels")]
#[derive(Debug, Error)]
pub enum ChannelError {
    /// 网络错误
    #[error("Network error: {0}")]
    NetworkError(String),
    /// API 错误（状态码与消息）
    #[error("API error (status {status}): {message}")]
    ApiError {
        /// HTTP 状态码
        status: u16,
        /// 错误消息
        message: String,
    },
    /// 认证错误
    #[error("Auth error: {0}")]
    AuthError(String),
    /// 连接错误
    #[error("Connection error: {0}")]
    ConnectionError(String),
    /// 发送错误
    #[error("Send error: {0}")]
    SendError(String),
    /// 输出属于已退役的渠道会话代次。
    #[error("Stale channel delivery generation: {incarnation_id}")]
    StaleDelivery {
        /// 已不再拥有渠道投递权的不透明化身（incarnation）标识。
        incarnation_id: String,
    },
    /// 配置无效
    #[error("Invalid config: {0}")]
    InvalidConfig(String),
    /// 其他错误
    #[error("Channel error: {0}")]
    Other(String),
}

/// 配置错误
#[derive(Debug, Error)]
pub enum ConfigError {
    /// 环境变量解析错误
    #[error("Failed to parse environment variable: {0}")]
    EnvParseError(String),
    /// 缺少配置项
    #[error("Model '{0}' missing required config: {1}")]
    MissingConfig(String, String),
    /// 环境变量格式无效
    #[error("Invalid environment variable format: {0}")]
    EnvFormatError(String),
    /// 配置不匹配
    #[error("Model '{0}' mismatched config error: {1}")]
    UnMatchConfigError(String, String),
    /// 未找到模型配置
    #[error("No configuration found for model: {0}")]
    NotFindModelError(String),
    /// 配置文件错误
    #[error("Config file error: {0}")]
    ConfigFileError(String),
}

// ── From 实现（Box 包装 + 自定义转换） ────────────────────────────────────

impl From<LlmError> for ReactError {
    /// 将 `LlmError` 转换为 `ReactError`。
    fn from(err: LlmError) -> Self {
        ReactError::Llm(Box::new(err))
    }
}

impl From<ToolError> for ReactError {
    /// 将 `ToolError` 转换为 `ReactError`。
    fn from(err: ToolError) -> Self {
        ReactError::Tool(Box::new(err))
    }
}

impl From<ParseError> for ReactError {
    /// 将 `ParseError` 转换为 `ReactError`。
    fn from(err: ParseError) -> Self {
        ReactError::Parse(Box::new(err))
    }
}

impl From<AgentError> for ReactError {
    /// 将 `AgentError` 转换为 `ReactError`。
    fn from(err: AgentError) -> Self {
        ReactError::Agent(Box::new(err))
    }
}

impl From<ConfigError> for ReactError {
    /// 将 `ConfigError` 转换为 `ReactError`。
    fn from(err: ConfigError) -> Self {
        ReactError::Config(Box::new(err))
    }
}

#[cfg(feature = "mcp")]
impl From<McpError> for ReactError {
    /// 将 `McpError` 转换为 `ReactError`。
    fn from(err: McpError) -> Self {
        ReactError::Mcp(Box::new(err))
    }
}

impl From<MemoryError> for ReactError {
    /// 将 `MemoryError` 转换为 `ReactError`。
    fn from(err: MemoryError) -> Self {
        ReactError::Memory(Box::new(err))
    }
}

impl From<SandboxError> for ReactError {
    /// 将 `SandboxError` 转换为 `ReactError`。
    fn from(err: SandboxError) -> Self {
        ReactError::Sandbox(Box::new(err))
    }
}

#[cfg(feature = "channels")]
impl From<ChannelError> for ReactError {
    /// 将 `ChannelError` 转换为 `ReactError`。
    fn from(err: ChannelError) -> Self {
        ReactError::Channel(Box::new(err))
    }
}

impl From<serde_json::Error> for ReactError {
    /// 将 `serde_json::Error` 转换为 `ReactError`。
    fn from(err: serde_json::Error) -> Self {
        ReactError::Parse(Box::new(ParseError::JsonError(err)))
    }
}

#[cfg(feature = "reqwest")]
impl From<reqwest::Error> for ReactError {
    /// 将 `reqwest::Error` 转换为 `ReactError`。
    fn from(err: reqwest::Error) -> Self {
        if err.is_timeout() {
            ReactError::Llm(Box::new(LlmError::NetworkError(
                "Request timeout".to_string(),
            )))
        } else if err.is_connect() {
            ReactError::Llm(Box::new(LlmError::NetworkError(format!(
                "Connection failed: {}",
                err
            ))))
        } else {
            ReactError::Llm(Box::new(LlmError::NetworkError(err.to_string())))
        }
    }
}

/// 便捷的 Result 类型别名
pub type Result<T> = std::result::Result<T, ReactError>;

#[cfg(test)]
mod agent_failure_tests {
    use super::*;

    /// 验证失败契约保留终止类别、可重试性与 HTTP 状态，并可序列化往返。
    #[test]
    fn preserves_terminal_and_retry_facts() -> std::result::Result<(), serde_json::Error> {
        let cases = [
            (
                ReactError::Llm(Box::new(LlmError::ApiError {
                    status: 429,
                    message: "rate limited".to_string(),
                })),
                AgentFailureCategory::Llm,
                AgentTerminalKind::Failed,
                true,
                Some(429),
            ),
            (
                ReactError::Agent(Box::new(AgentError::Cancelled("stop".to_string()))),
                AgentFailureCategory::Agent,
                AgentTerminalKind::Cancelled,
                false,
                None,
            ),
            (
                ReactError::Agent(Box::new(AgentError::Timeout("late".to_string()))),
                AgentFailureCategory::Agent,
                AgentTerminalKind::TimedOut,
                true,
                None,
            ),
            (
                ReactError::Agent(Box::new(AgentError::PermissionDenied("no".to_string()))),
                AgentFailureCategory::Agent,
                AgentTerminalKind::PermissionDenied,
                false,
                None,
            ),
            (
                ReactError::Parse(Box::new(ParseError::UnexpectedFormat("bad".to_string()))),
                AgentFailureCategory::Parse,
                AgentTerminalKind::Failed,
                false,
                None,
            ),
            (
                ReactError::Io(std::io::Error::other("disk")),
                AgentFailureCategory::Io,
                AgentTerminalKind::Failed,
                true,
                None,
            ),
            (
                ReactError::Memory(Box::new(MemoryError::DeadlineExceeded("late".to_string()))),
                AgentFailureCategory::Memory,
                AgentTerminalKind::TimedOut,
                true,
                None,
            ),
            (
                ReactError::Memory(Box::new(MemoryError::TransientNoCommit(
                    "retry".to_string(),
                ))),
                AgentFailureCategory::Memory,
                AgentTerminalKind::Failed,
                true,
                None,
            ),
            (
                ReactError::RuntimeState(Box::new(
                    RuntimeStateError::TranscriptProjectionDeferred {
                        operation_id: Some("operation".to_string()),
                        reason: "backend unavailable".to_string(),
                    },
                )),
                AgentFailureCategory::RuntimeState,
                AgentTerminalKind::Failed,
                true,
                None,
            ),
            (
                ReactError::RuntimeState(Box::new(RuntimeStateError::TransientNoCommit(
                    "retry".to_string(),
                ))),
                AgentFailureCategory::RuntimeState,
                AgentTerminalKind::Failed,
                true,
                None,
            ),
        ];
        for (error, category, terminal, retryable, status) in cases {
            let failure = AgentFailure::from(&error);
            assert_eq!(failure.category, category);
            assert_eq!(failure.terminal_kind, terminal);
            assert_eq!(failure.retryable, retryable);
            assert_eq!(failure.http_status, status);
            let json = serde_json::to_string(&failure)?;
            let decoded: AgentFailure = serde_json::from_str(&json)?;
            assert_eq!(decoded, failure);
        }
        Ok(())
    }
}

/// 运行时状态错误
#[derive(Debug, Error)]
pub enum RuntimeStateError {
    /// I/O 错误
    #[error("IO error: {0}")]
    Io(String),
    /// 序列化错误
    #[error("Serialization error: {0}")]
    SerializationError(String),
    /// 状态不存在
    #[error("State not found: {0}")]
    NotFound(String),
    /// 后端未实现所需的原子运行时状态契约。
    #[error("Unsupported operation: {0}")]
    Unsupported(String),
    /// 旧式变更试图覆盖受修订版本管理的状态。
    #[error("Managed runtime state requires compare-and-save: {0}")]
    ManagedStateRequiresCas(String),
    /// 持久化修订版本无法再递增，否则会溢出。
    #[error("Runtime state revision exhausted: {0}")]
    RevisionExhausted(String),
    /// 持久化调用超出了共享的绝对截止时间。
    #[error("Runtime state deadline exceeded: {0}")]
    DeadlineExceeded(String),
    /// 持久化的会话记录副作用在新的准入之前仍可重试。
    #[error("Transcript projection deferred ({operation_id:?}): {reason}")]
    TranscriptProjectionDeferred {
        /// 相关操作 ID（如有）
        operation_id: Option<String>,
        /// 延迟原因
        reason: String,
    },
    /// 观察到持久化失败，必须替换原有的终止结果。
    #[error("Transcript projection blocked ({status}): {reason}")]
    TranscriptProjectionBlocked {
        /// 阻塞状态
        status: String,
        /// 阻塞原因
        reason: String,
    },
    /// 已知未产生任何持久化提交的可重试运行时状态失败。
    #[error("Transient runtime-state no-commit failure: {0}")]
    TransientNoCommit(String),
}

impl From<RuntimeStateError> for ReactError {
    /// 将 `RuntimeStateError` 转换为 `ReactError`。
    fn from(err: RuntimeStateError) -> Self {
        ReactError::RuntimeState(Box::new(err))
    }
}
