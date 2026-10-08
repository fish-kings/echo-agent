//! 工具系统的核心 trait 与类型定义

/// 持久化、有界的工具输出 artifact。
pub mod artifact;
/// 命令 cell 契约：支持长轮询等待的后台 shell 命令。
pub mod cell;
/// 供框架消费者共享的带版本号的工具可见性控制。
pub mod control;
/// 返回集合类工具共享的游标分页契约。
pub mod pagination;
/// 工具权限模式与权限决策。
pub mod permission;
/// Skill 系统的核心 trait 与类型。
pub mod skill;

use crate::error::{ReactError, Result, SandboxError, ToolError};
use crate::sandbox::SandboxExecutor;
use futures::future::BoxFuture;
use futures::stream::{self, Stream};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;

/// 工具执行失败的稳定分类，被工具执行、trace、checkpoint 和 UI 共用。
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum ToolFailureCategory {
    /// 参数无效，需要先修正参数再重试。
    InvalidArguments,
    /// 工具当前不可用（例如依赖服务尚未就绪）。
    Unavailable,
    /// 执行超时。
    Timeout,
    /// 执行被取消。
    Cancelled,
    /// 瞬时性错误，通常可直接重试。
    Transient,
    /// 永久性失败，重试无意义。
    Permanent,
    /// 失败前可能已经产生了部分副作用。
    PartialSideEffect,
}

impl ToolFailureCategory {
    /// 返回该分类的稳定字符串标识，用于序列化和日志。
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::InvalidArguments => "invalid_arguments",
            Self::Unavailable => "unavailable",
            Self::Timeout => "timeout",
            Self::Cancelled => "cancelled",
            Self::Transient => "transient",
            Self::Permanent => "permanent",
            Self::PartialSideEffect => "partial_side_effect",
        }
    }
}

/// 同一个逻辑工具操作要继续执行之前，必须先发生的动作。
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum ToolRecoveryAction {
    /// 需要先修正参数，再重新调用。
    CorrectArguments,
    /// 可以直接重试。
    Retry,
    /// 需要先恢复到已知状态，再重试。
    RestoreThenRetry,
    /// 需要先校验当前实际状态，再决定是否重试。
    VerifyThenRetry,
    /// 应当停止，不要再重试。
    Stop,
}

impl ToolRecoveryAction {
    /// 返回该恢复动作的稳定字符串标识，用于序列化和日志。
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CorrectArguments => "correct_arguments",
            Self::Retry => "retry",
            Self::RestoreThenRetry => "restore_then_retry",
            Self::VerifyThenRetry => "verify_then_retry",
            Self::Stop => "stop",
        }
    }
}

/// 一次未成功的尝试是否可能已经改变了外部状态。
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum ToolSideEffect {
    /// 确定没有产生副作用。
    None,
    /// 可能已产生副作用，但无法确定。
    Possible,
    /// 已确认产生了副作用。
    Confirmed,
}

/// 一次未成功工具结果的结构化恢复信息。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolFailure {
    /// 失败分类。
    pub category: ToolFailureCategory,
    /// 建议采取的恢复动作。
    pub recovery: ToolRecoveryAction,
    /// 是否可能已产生副作用。
    pub side_effect: ToolSideEffect,
    /// 建议的重试等待时间（毫秒）。
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub retry_after_ms: Option<u64>,
    /// 幂等键；存在时即使有副作用也可安全地自动重试。
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub idempotency_key: Option<String>,
    /// 重试前用于校验操作是否已生效的后置条件描述。
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub postcondition: Option<String>,
}

impl ToolFailure {
    /// 根据失败分类构造一个带默认恢复策略的 `ToolFailure`。
    pub fn new(category: ToolFailureCategory) -> Self {
        let recovery = match category {
            ToolFailureCategory::InvalidArguments => ToolRecoveryAction::CorrectArguments,
            ToolFailureCategory::Unavailable => ToolRecoveryAction::RestoreThenRetry,
            ToolFailureCategory::Timeout | ToolFailureCategory::PartialSideEffect => {
                ToolRecoveryAction::VerifyThenRetry
            }
            ToolFailureCategory::Transient => ToolRecoveryAction::Retry,
            ToolFailureCategory::Cancelled | ToolFailureCategory::Permanent => {
                ToolRecoveryAction::Stop
            }
        };
        let side_effect = if category == ToolFailureCategory::PartialSideEffect {
            ToolSideEffect::Possible
        } else {
            ToolSideEffect::None
        };
        Self {
            category,
            recovery,
            side_effect,
            retry_after_ms: None,
            idempotency_key: None,
            postcondition: None,
        }
    }

    /// 对可重试的失败分类，把恢复动作强制改为 `Retry`。
    pub fn retryable(mut self) -> Self {
        if matches!(
            self.category,
            ToolFailureCategory::Unavailable
                | ToolFailureCategory::Timeout
                | ToolFailureCategory::Transient
        ) {
            self.recovery = ToolRecoveryAction::Retry;
        }
        self
    }

    /// 设置建议的重试等待时间。
    pub fn with_retry_after(mut self, retry_after_ms: u64) -> Self {
        self.retry_after_ms = Some(retry_after_ms);
        self
    }

    /// 设置副作用状态；若存在副作用且尚无幂等键，恢复动作会被调整为先校验再重试。
    pub fn with_side_effect(mut self, side_effect: ToolSideEffect) -> Self {
        self.side_effect = side_effect;
        if side_effect != ToolSideEffect::None && self.idempotency_key.is_none() {
            self.recovery = ToolRecoveryAction::VerifyThenRetry;
        }
        self
    }

    /// 设置幂等键，使带副作用的失败也能安全地自动重试。
    pub fn with_idempotency_key(mut self, key: impl Into<String>) -> Self {
        self.idempotency_key = Some(key.into());
        self
    }

    /// 设置后置条件描述，供重试前校验操作是否已生效。
    pub fn with_postcondition(mut self, postcondition: impl Into<String>) -> Self {
        self.postcondition = Some(postcondition.into());
        self
    }

    /// 判定标准刻意比"也许能恢复"更严格：只有确定安全时才允许自动重放。
    pub fn allows_automatic_retry(&self) -> bool {
        matches!(
            self.category,
            ToolFailureCategory::Unavailable
                | ToolFailureCategory::Timeout
                | ToolFailureCategory::Transient
        ) && self.recovery == ToolRecoveryAction::Retry
            && (self.side_effect == ToolSideEffect::None || self.idempotency_key.is_some())
    }

    /// 保守地对框架执行错误进行分类（不确定时倾向于更谨慎的分类）。
    pub fn from_error(error: &ReactError, may_have_side_effects: bool) -> Self {
        let possible_side_effect = |mut failure: Self| {
            if may_have_side_effects {
                if matches!(
                    failure.category,
                    ToolFailureCategory::Unavailable
                        | ToolFailureCategory::Transient
                        | ToolFailureCategory::Permanent
                ) {
                    failure.category = ToolFailureCategory::PartialSideEffect;
                }
                return failure.with_side_effect(ToolSideEffect::Possible);
            }
            failure
        };
        match error {
            ReactError::Tool(tool_error) => match tool_error.as_ref() {
                ToolError::MissingParameter(_)
                | ToolError::InvalidParameter { .. }
                | ToolError::InvalidPath { .. }
                | ToolError::AccessDenied { .. }
                | ToolError::FileTooLarge { .. } => {
                    Self::new(ToolFailureCategory::InvalidArguments)
                }
                ToolError::Timeout(_) => {
                    possible_side_effect(Self::new(ToolFailureCategory::Timeout).retryable())
                }
                ToolError::Cancelled(_) => {
                    possible_side_effect(Self::new(ToolFailureCategory::Cancelled))
                }
                ToolError::NotFound(_) | ToolError::ExecutionFailed { .. } => {
                    possible_side_effect(Self::new(ToolFailureCategory::Permanent))
                }
            },
            ReactError::Sandbox(sandbox_error) => match sandbox_error.as_ref() {
                SandboxError::Unavailable(_) | SandboxError::StartFailed(_) => {
                    Self::new(ToolFailureCategory::Unavailable)
                }
                SandboxError::Timeout(_) => {
                    possible_side_effect(Self::new(ToolFailureCategory::Timeout).retryable())
                }
                SandboxError::Cancelled(_) => {
                    possible_side_effect(Self::new(ToolFailureCategory::Cancelled))
                }
                SandboxError::ResourceExceeded(_)
                | SandboxError::PermissionDenied(_)
                | SandboxError::IoError(_) => {
                    possible_side_effect(Self::new(ToolFailureCategory::Permanent))
                }
            },
            ReactError::Io(io_error) => match io_error.kind() {
                std::io::ErrorKind::TimedOut => {
                    possible_side_effect(Self::new(ToolFailureCategory::Timeout).retryable())
                }
                std::io::ErrorKind::Interrupted
                | std::io::ErrorKind::WouldBlock
                | std::io::ErrorKind::ConnectionAborted
                | std::io::ErrorKind::ConnectionReset
                | std::io::ErrorKind::BrokenPipe
                | std::io::ErrorKind::NotConnected => {
                    possible_side_effect(Self::new(ToolFailureCategory::Transient).retryable())
                }
                std::io::ErrorKind::InvalidInput | std::io::ErrorKind::InvalidData => {
                    Self::new(ToolFailureCategory::InvalidArguments)
                }
                _ => possible_side_effect(Self::new(ToolFailureCategory::Permanent)),
            },
            #[cfg(feature = "mcp")]
            ReactError::Mcp(mcp_error) => match mcp_error.as_ref() {
                crate::error::McpError::ConnectionFailed(_)
                | crate::error::McpError::InitializationFailed(_)
                | crate::error::McpError::TransportClosed => {
                    let failure = Self::new(ToolFailureCategory::Unavailable);
                    if may_have_side_effects {
                        possible_side_effect(failure)
                    } else {
                        failure.retryable()
                    }
                }
                crate::error::McpError::ToolCallFailed { code: -32602, .. } => {
                    Self::new(ToolFailureCategory::InvalidArguments)
                }
                crate::error::McpError::ToolCallFailed {
                    code: -32603 | -32099..=-32000,
                    ..
                } => possible_side_effect(Self::new(ToolFailureCategory::Transient).retryable()),
                crate::error::McpError::ProtocolError(_)
                | crate::error::McpError::ToolCallFailed { .. } => {
                    possible_side_effect(Self::new(ToolFailureCategory::Permanent))
                }
            },
            _ => possible_side_effect(Self::new(ToolFailureCategory::Permanent)),
        }
    }
}

/**
对工具产生的结果种类进行分类。

这为 [`ToolResult`] 提供了额外信息，使下游消费者（CLI 渲染、trace 分析、
评测打分等）可以按类型正确处理结果，而不必解析原始的 output 字符串。
*/
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind_type", rename_all = "snake_case")]
pub enum ToolResultKind {
    /// 纯文本输出。
    #[default]
    Text,
    /// 结构化 JSON 数据（也可在 `ToolResult.data` 中获取）。
    Json,
    /// 图片（MIME 类型记录在 `ToolResult.mime_type` 中）。
    Image {
        /// 图片的 MIME 类型，例如 `image/png`。
        mime_type: String,
    },
    /// 带列标题和行数据的表格数据。
    Table {
        /// 列标题。
        columns: Vec<String>,
        /// 每一行的数据。
        rows: Vec<Vec<String>>,
    },
    /// 统一格式的 diff（例如来自 `apply_patch`）。
    Diff {
        /// 统一 diff 文本。
        unified_diff: String,
    },
    /// 指向某个文件的引用（路径记录在 metadata 中）。
    FileReference {
        /// 文件路径。
        path: String,
    },
    /// 命令执行输出，附带退出码。
    CommandOutput {
        /// 进程退出码；未知时为 `None`。
        exit_code: Option<i32>,
    },
    /// 必须成为规范激活投影的 skill 提示块。
    SkillActivation {
        /// skill 名称。
        name: String,
    },
    /// 带错误码的结构化错误。
    StructuredError {
        /// 错误码。
        error_code: String,
    },
}

/**
工具产生的、对模型 provider 可见的富内容。

该 payload 故意不参与序列化的工具事件和日志。ReAct 运行时会在对应的文本
工具结果之后，把它投影进模型对话中，使二进制数据能够到达多模态模型，同时
不会把体积巨大的 data URL 泄漏进 UI 事件、trace 或审计输出。
*/
#[derive(Clone, PartialEq, Eq)]
pub enum ToolResultContent {
    /// 以 HTTPS URL 或 Base64 data URL 形式表示的图片数据。
    ImageUrl {
        /// 图片的 URL（HTTPS 链接或 Base64 data URL）。
        url: String,
        /// 可选的细节等级提示（provider 相关，如 `"low"` / `"high"`）。
        detail: Option<String>,
    },
}

/// 由工具实现产生的、权威的执行事实。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "effect", rename_all = "snake_case")]
pub enum ToolEffect {
    /// 读取了某个文件。
    FileRead {
        /// 被读取文件的路径。
        path: String,
    },
    /// 编辑了某个文件。
    FileEdit {
        /// 被编辑文件的路径。
        path: String,
    },
    /// 执行了一次测试命令。
    TestRun {
        /// 执行的命令。
        command: String,
        /// 测试是否通过。
        passed: bool,
        /// 来自结构化测试报告的精确失败数；未知时为空。
        #[serde(default, skip_serializing_if = "Option::is_none")]
        failure_count: Option<usize>,
    },
    /// 派发并运行了一个 subagent。
    SubagentRun {
        /// subagent 的角色名。
        agent_name: String,
        /// 交给 subagent 的任务描述。
        task: String,
        /// subagent 的执行结果概述。
        outcome: String,
    },
}

/**
针对"生产者在工具调用返回之后才最终确定"的效果（例如后台 subagent 派发）
的、绑定到单次调用范围的延续回调。
*/
pub type ToolEffectSinkFn = Arc<dyn Fn(ToolEffect) -> BoxFuture<'static, ()> + Send + Sync>;

impl std::fmt::Debug for ToolResultContent {
    /// 自定义调试输出：URL 只显示字符数，避免打印可能很长的内联数据。
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ImageUrl { url, detail } => formatter
                .debug_struct("ImageUrl")
                .field(
                    "url",
                    &format_args!("<omitted:{} chars>", url.chars().count()),
                )
                .field("detail", detail)
                .finish(),
        }
    }
}

/// 工具执行结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    /// 结果的种类（使下游可以按类型感知地处理）。
    #[serde(default)]
    pub kind: ToolResultKind,
    /// 工具是否执行成功。
    pub success: bool,
    /// 返回给调用方的文本输出。
    pub output: String,
    /// `success` 为 false 时的错误信息。
    pub error: Option<String>,
    /// `success` 为 false 时的结构化失败与恢复信息。
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub failure: Option<ToolFailure>,
    /**
    可选的结构化数据（JSON）。存在时调用方可直接渲染它，
    而无需解析 `output` 字符串。
    */
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub data: Option<serde_json::Value>,
    /// 输出是否因超出长度限制而被截断。
    #[serde(default)]
    pub truncated: bool,
    /// 输出内容的 MIME 类型（已知时），例如 `text/html`、`image/png`。
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub mime_type: Option<String>,
    /**
    当结果从内联输出溢出（spilled）时，保存完整输出的 artifact。

    这是一个带类型的终态事实。消费者不得从开放式的 `metadata` map 中
    重新构造 artifact 的身份或完整性。
    */
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub artifact: Option<artifact::ToolOutputArtifactRef>,
    /// 任意键值对形式的 metadata（例如来源 URL、文件路径、耗时、token 数）。
    #[serde(default)]
    pub metadata: HashMap<String, String>,
    /// 由所属工具实现产生的、已确认的执行事实。
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub effects: Vec<ToolEffect>,
    /// 仅投影进实时模型对话中的富内容。
    #[serde(skip)]
    pub model_content: Vec<ToolResultContent>,
}

impl ToolResult {
    /// 构造一个成功的纯文本结果。
    pub fn success(output: impl Into<String>) -> Self {
        Self {
            kind: ToolResultKind::Text,
            success: true,
            output: output.into(),
            error: None,
            failure: None,
            data: None,
            truncated: false,
            mime_type: None,
            artifact: None,
            metadata: HashMap::new(),
            effects: Vec::new(),
            model_content: Vec::new(),
        }
    }

    /**
    构造一个带结构化 JSON 数据的成功结果。

    `output` 字段会被设置为该数据的紧凑 JSON 表示，
    这样纯文本消费者依然能看到有意义的内容。
    */
    pub fn success_json(data: serde_json::Value) -> Self {
        let output = serde_json::to_string(&data).unwrap_or_default();
        Self {
            kind: ToolResultKind::Json,
            success: true,
            output,
            error: None,
            failure: None,
            data: Some(data),
            truncated: false,
            mime_type: None,
            artifact: None,
            metadata: HashMap::new(),
            effects: Vec::new(),
            model_content: Vec::new(),
        }
    }

    /// 构造一个带指定 [`ToolResultKind`] 的成功结果。
    pub fn success_with_kind(kind: ToolResultKind, output: impl Into<String>) -> Self {
        Self {
            kind,
            success: true,
            output: output.into(),
            error: None,
            failure: None,
            data: None,
            truncated: false,
            mime_type: None,
            artifact: None,
            metadata: HashMap::new(),
            effects: Vec::new(),
            model_content: Vec::new(),
        }
    }

    /// 构造一个带错误信息的失败结果。
    pub fn error(error: impl Into<String>) -> Self {
        Self {
            kind: ToolResultKind::StructuredError {
                error_code: "tool_error".into(),
            },
            success: false,
            output: String::new(),
            error: Some(error.into()),
            failure: Some(ToolFailure::new(ToolFailureCategory::Permanent)),
            data: None,
            truncated: false,
            mime_type: None,
            artifact: None,
            metadata: HashMap::new(),
            effects: Vec::new(),
            model_content: Vec::new(),
        }
    }

    /// 覆盖已有结果上的文本输出。
    pub fn with_output(mut self, output: impl Into<String>) -> Self {
        self.output = output.into();
        self
    }

    /// 覆盖已有结果上的错误文本。
    pub fn with_error(mut self, error: impl Into<String>) -> Self {
        self.success = false;
        self.error = Some(error.into());
        if self.failure.is_none() {
            self.failure = Some(ToolFailure::new(ToolFailureCategory::Permanent));
        }
        self
    }

    /// 构造一个带明确失败分类的失败结果。
    pub fn failure(category: ToolFailureCategory, error: impl Into<String>) -> Self {
        let mut result = Self::error(error);
        result.failure = Some(ToolFailure::new(category));
        result
    }

    /// 构造一个需要修正参数的失败结果。
    pub fn invalid_arguments(error: impl Into<String>) -> Self {
        Self::failure(ToolFailureCategory::InvalidArguments, error)
    }

    /// 替换未成功结果上的结构化恢复信息。
    pub fn with_failure(mut self, failure: ToolFailure) -> Self {
        self.success = false;
        self.failure = Some(failure);
        self
    }

    /// 追加一条已确认的执行事实。
    pub fn with_effect(mut self, effect: ToolEffect) -> Self {
        self.effects.push(effect);
        self
    }

    /// 给已有结果附加结构化数据。
    pub fn with_data(mut self, data: serde_json::Value) -> Self {
        self.data = Some(data);
        self
    }

    /// 标记该输出是否被截断。
    pub fn with_truncated(mut self, truncated: bool) -> Self {
        self.truncated = truncated;
        self
    }

    /// 设置输出内容的 MIME 类型。
    pub fn with_mime_type(mut self, mime_type: impl Into<String>) -> Self {
        self.mime_type = Some(mime_type.into());
        self
    }

    /// 给该结果附加完整的类型化 artifact 描述符。
    pub fn with_artifact(mut self, artifact: artifact::ToolOutputArtifactRef) -> Self {
        self.artifact = Some(artifact);
        self
    }

    /// 插入一条键值对形式的 metadata。
    pub fn with_meta(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// 批量插入 metadata 条目。
    pub fn with_metadata(mut self, meta: HashMap<String, String>) -> Self {
        self.metadata = meta;
        self
    }

    /// 附加应对模型可见、但不进入序列化工具事件和日志的富内容。
    pub fn with_model_content(mut self, content: ToolResultContent) -> Self {
        self.model_content.push(content);
        self
    }
}

/**
工具增量输出所属的逻辑通道。

像 shell 这样的工具会把进程管道映射到这些通道上；其他工具可以用
[`ToolOutputChannel::Log`] 来表示非结构化的进度文本。
*/
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ToolOutputChannel {
    /// 主进程/命令的 stdout。
    Stdout,
    /// 进程 stderr 或等价的错误流。
    Stderr,
    /// 工具自定义的日志/诊断流。
    Log,
}

/**
流式工具输出事件

当一个工具实现了 [`Tool::execute_stream_with_context`] 和
[`Tool::supports_streaming`] 时，它会在执行期间产生一系列这样的事件，
从而支持实时进度上报，以及向 UI / 调用方增量交付输出。
*/
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "event_type", rename_all = "snake_case")]
// Complete 保持内联，以维持公开的 stream 事件 API 不变。
#[allow(clippy::large_enum_variant)]
pub enum ToolStreamEvent {
    /// 带可选百分比（0-100）的进度通知。
    Progress {
        /// 人类可读的进度描述。
        message: String,
        /// 可选的完成百分比（0-100）。
        percent: Option<u8>,
    },
    /// 某个命名通道上的增量输出片段。
    Output {
        /// 该片段所属的逻辑流。
        channel: ToolOutputChannel,
        /// 输出片段（UTF-8 文本；可能是不完整的一行）。
        chunk: String,
    },
    /**
    携带最终 [`ToolResult`] 的终止事件。
    该事件发出后 stream 结束。
    */
    Complete(ToolResult),
}

/// 工具执行配置：超时、重试、并发
#[derive(Debug, Clone)]
pub struct ToolExecutionConfig {
    /// 单次尝试的最大执行时间。
    pub timeout_ms: u64,
    /// 失败的执行是否应自动重试。
    pub retry_on_fail: bool,
    /// 最大重试次数。
    pub max_retries: u32,
    /// 重试之间的延迟。
    pub retry_delay_ms: u64,
    /// 由工具管理器共享的可选并发上限（写入/执行类工具）。
    pub max_concurrency: Option<usize>,
    /// 只读工具的可选并发上限（更高的限制，默认 32）。
    pub max_read_concurrency: Option<usize>,
}

impl Default for ToolExecutionConfig {
    /// 默认配置：30 秒超时等。
    fn default() -> Self {
        Self {
            timeout_ms: 30_000,
            retry_on_fail: true,
            max_retries: 2,
            retry_delay_ms: 200,
            max_concurrency: None,
            max_read_concurrency: Some(32),
        }
    }
}

/// 工具参数类型（来自 LLM 的简单键值对）。
pub type ToolParameters = HashMap<String, serde_json::Value>;

/// 从原始 JSON 中提取出的、带类型安全保证的参数值。
#[derive(Debug, Clone, PartialEq)]
pub enum ParamValue {
    /// 字符串值。
    String(String),
    /// 数值。
    Number(f64),
    /// 布尔值。
    Bool(bool),
    /// 数组值。
    Array(Vec<ParamValue>),
    /// 对象值。
    Object(HashMap<String, ParamValue>),
    /// 空值。
    Null,
}

impl ParamValue {
    /// 从一个 JSON 值中提取。
    pub fn from_json(v: &serde_json::Value) -> Self {
        match v {
            serde_json::Value::String(s) => Self::String(s.clone()),
            serde_json::Value::Number(n) => Self::Number(n.as_f64().unwrap_or(0.0)),
            serde_json::Value::Bool(b) => Self::Bool(*b),
            serde_json::Value::Array(arr) => Self::Array(arr.iter().map(Self::from_json).collect()),
            serde_json::Value::Object(obj) => Self::Object(
                obj.iter()
                    .map(|(k, v)| (k.clone(), Self::from_json(v)))
                    .collect(),
            ),
            serde_json::Value::Null => Self::Null,
        }
    }

    /// 若为 String 变体，返回其字符串值。
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// 若为 Number 变体，返回其数值。
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Number(n) => Some(*n),
            _ => None,
        }
    }

    /// 若为 Bool 变体，返回其布尔值。
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// 用于错误信息的类型名称。
    pub fn type_name(&self) -> &str {
        match self {
            Self::String(_) => "string",
            Self::Number(_) => "number",
            Self::Bool(_) => "bool",
            Self::Array(_) => "array",
            Self::Object(_) => "object",
            Self::Null => "null",
        }
    }
}

/// 带校验支持的、类型安全的工具调用参数。
#[derive(Debug, Clone)]
pub struct ToolCallParams {
    /// 来自 LLM 的原始 JSON。
    pub raw: serde_json::Value,
    /// 解析后的类型安全值。
    parsed: HashMap<String, ParamValue>,
}

impl ToolCallParams {
    /// 从一个原始 JSON 值创建。
    pub fn from_value(value: &serde_json::Value) -> Self {
        let parsed = if let serde_json::Value::Object(map) = value {
            map.iter()
                .map(|(k, v)| (k.clone(), ParamValue::from_json(v)))
                .collect()
        } else {
            HashMap::new()
        };
        Self {
            raw: value.clone(),
            parsed,
        }
    }

    /// 从一个 ToolParameters map 创建。
    pub fn from_params(params: &ToolParameters) -> Self {
        let mut map = serde_json::Map::new();
        for (k, v) in params {
            map.insert(k.clone(), v.clone());
        }
        Self::from_value(&serde_json::Value::Object(map))
    }

    /// 获取一个字符串参数。
    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.parsed.get(key).and_then(|v| v.as_str())
    }

    /// 获取一个数值参数。
    pub fn get_number(&self, key: &str) -> Option<f64> {
        self.parsed.get(key).and_then(|v| v.as_f64())
    }

    /// 获取一个布尔参数。
    pub fn get_bool(&self, key: &str) -> Option<bool> {
        self.parsed.get(key).and_then(|v| v.as_bool())
    }

    /// 按 key 获取一个参数值。
    pub fn get(&self, key: &str) -> Option<&ParamValue> {
        self.parsed.get(key)
    }

    /**
    校验某个必需参数是否存在且类型符合预期。

    返回 `Ok(())` 或 `Err(error_message)`。
    */
    pub fn validate_required(
        &self,
        key: &str,
        expected_type: &str,
    ) -> std::result::Result<(), String> {
        match self.parsed.get(key) {
            None => Err(format!("Missing required parameter: {key}")),
            Some(v) if v.type_name() != expected_type => Err(format!(
                "Parameter '{key}': expected {expected_type}, got {}",
                v.type_name()
            )),
            _ => Ok(()),
        }
    }

    /// 检查某个参数是否存在。
    pub fn has(&self, key: &str) -> bool {
        self.parsed.contains_key(key)
    }

    /// 参数数量。
    pub fn len(&self) -> usize {
        self.parsed.len()
    }

    /// 是否没有任何参数。
    pub fn is_empty(&self) -> bool {
        self.parsed.is_empty()
    }
}

/// 工具风险等级
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum ToolRiskLevel {
    /// 只读操作，无副作用（例如搜索、读文件）
    ReadOnly,
    /// 标准操作，有限的副作用（例如写文件、调用 API）
    #[default]
    Standard,
    /// 危险操作，不可逆的副作用（例如执行 shell 命令、删除数据、SQL 写操作）
    Dangerous,
}

/// 工具是否可以改变当前模型调用之外的状态。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ToolAccess {
    /// 该工具仅观察状态。
    ReadOnly,
    /// 该工具可能创建、更新或删除状态。
    #[default]
    Mutating,
}

/**
用于组合和过滤工具集合的稳定能力元数据。

Capabilities 描述一个工具能做什么。审批策略仍是一个独立的运行时关注点，
因此应用可以在不维护基于名称的白名单的情况下，为同一个工具包暴露不同的交互策略。
*/
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCapabilities {
    /// 该工具的访问类型（只读 / 可变更）。
    pub access: ToolAccess,
    /// 该工具的风险等级。
    pub risk: ToolRiskLevel,
    /// 调用该工具所需的权限列表。
    pub permissions: Vec<permission::ToolPermission>,
}

impl ToolCapabilities {
    /// 构造一组只读能力元数据。
    pub fn read_only(permissions: Vec<permission::ToolPermission>) -> Self {
        Self {
            access: ToolAccess::ReadOnly,
            risk: ToolRiskLevel::ReadOnly,
            permissions,
        }
    }

    /// 构造一组带指定风险等级的可变更能力元数据。
    pub fn mutating(risk: ToolRiskLevel, permissions: Vec<permission::ToolPermission>) -> Self {
        Self {
            access: ToolAccess::Mutating,
            risk,
            permissions,
        }
    }

    /// 该能力是否为只读。
    pub fn is_read_only(&self) -> bool {
        self.access == ToolAccess::ReadOnly
    }
}

/// 一个具体的工具及其被 [`ToolPack`] 使用的能力事实。
pub struct ToolPackEntry {
    /// 具体工具实例。
    pub tool: Box<dyn Tool>,
    /// 该工具的能力元数据。
    pub capabilities: ToolCapabilities,
}

impl ToolPackEntry {
    /// 由工具自身的 `capabilities()` 推导构造一个条目。
    pub fn new(tool: Box<dyn Tool>) -> Self {
        let capabilities = tool.capabilities();
        Self { tool, capabilities }
    }

    /// 构造一个只读条目（强制标记为只读能力）。
    pub fn read_only(tool: Box<dyn Tool>) -> Self {
        let permissions = tool.permissions();
        Self {
            tool,
            capabilities: ToolCapabilities::read_only(permissions),
        }
    }
}

/// 一组可复用的、具名的工具组合。
pub trait ToolPack: Send + Sync {
    /// 该工具包的名称。
    fn name(&self) -> &str;
    /// 该工具包包含的工具条目列表。
    fn tools(&self) -> Vec<ToolPackEntry>;

    /// 把工具包中所有工具注册到给定的注册器。
    fn install(&self, registrar: &mut dyn ToolRegistrar) {
        for entry in self.tools() {
            registrar.register(entry.tool);
        }
    }

    /// 只把工具包中只读的工具注册到给定的注册器。
    fn install_read_only(&self, registrar: &mut dyn ToolRegistrar) {
        for entry in self
            .tools()
            .into_iter()
            .filter(|entry| entry.capabilities.is_read_only())
        {
            registrar.register(entry.tool);
        }
    }
}

/// 在进程执行之前应用命令策略所得到的结果。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommandPolicyDecision {
    /// 安全，可直接执行。
    Safe,
    /// 需要审批，原因见附带字符串。
    RequiresApproval(String),
    /// 危险，原因见附带字符串。
    Dangerous(String),
}

/// 供执行本地命令的工具遵循的策略契约。
pub trait CommandPolicy: Send + Sync {
    /// 对给定命令做出策略判定。
    fn evaluate(&self, command: &str) -> CommandPolicyDecision;
}

/**
可注册工具的类型所实现的 trait。

把工具注册这一行为，与任何具体的工具管理器实现解耦。
*/
pub trait ToolRegistrar {
    /// 注册一个工具。
    fn register(&mut self, tool: Box<dyn Tool>);
}

/**
供 `#[derive(Tool)]` 使用的辅助 trait —— 提供一个带类型的 `run` 方法，
派生宏生成的 `execute()` 会委托给它。

使用者用自己工具的业务逻辑覆盖 `run()`；派生宏会自动处理 JSON Schema
生成、参数反序列化，以及 `Tool` trait 的样板代码。
*/
pub trait ToolRunner<P = ToolParameters>: Tool + Sized {
    /// 用已反序列化的类型化参数执行该工具。
    fn run(&self, params: P) -> impl std::future::Future<Output = Result<ToolResult>> + Send;
}

/// 工具接口 trait
pub trait Tool: Send + Sync {
    /// 暴露给模型的稳定工具标识符。
    fn name(&self) -> &str;
    /// 人类可读的工具描述。
    fn description(&self) -> &str;
    /// 描述所接受参数的 JSON Schema。
    fn parameters(&self) -> serde_json::Value;
    /// 对于 schema 依赖运行时元数据的工具，单调递增的版本号。
    fn schema_revision(&self) -> u64 {
        0
    }

    /**
    消费该工具结果所需要的模型输入模态。

    当存在明确的模型能力契约时，运行时会从模型可见的 schema 中
    省略掉不兼容的工具。
    */
    fn required_input_modalities(&self) -> &'static [crate::llm::ModelInputModality] {
        &[]
    }

    /**
    供运行时定义的工具使用的、拥有所有权的模态投影；这类工具所需的模态
    是通过某条连接协商得到的。静态工具实现继续使用
    [`Self::required_input_modalities`]。
    */
    #[doc(hidden)]
    fn required_input_modalities_owned(&self) -> Vec<crate::llm::ModelInputModality> {
        self.required_input_modalities().to_vec()
    }

    /**
    用未类型化的 JSON 参数执行该工具。

    **默认实现**会委托给 [`Self::execute_with_context`]，并传入一个空的
    [`ToolContext`] —— 因此 override 了 `execute_with_context` 的工具无需
    同时实现 `execute`。只实现了 `execute` 的旧式工具依然可以正常工作，
    因为默认的 `execute_with_context` 会委托回 `execute`（不会无限递归：
    一个具体的 `impl Tool` 必须至少覆盖这两者之一）。
    */
    fn execute<'a>(&'a self, parameters: ToolParameters) -> BoxFuture<'a, Result<ToolResult>> {
        /*
        不能直接委托给 execute_with_context，因为它的 ctx 借用与 'a
        （self 的生命周期）绑定，而一个 default() 临时值活不了那么久。
        因此把一个由该 future 拥有的默认 ctx 内联在这里。
        */
        Box::pin(async move {
            let ctx = ToolContext::default();
            self.execute_with_context(parameters, &ctx).await
        })
    }

    /**
    用运行时 [`ToolContext`] 执行该工具。

    **默认实现**忽略 `ctx`，委托给 [`Self::execute`] —— 因此已有的
    `impl Tool` 代码块无需任何改动即可继续工作。关心 `working_dir` 的工具
    （ShellTool、文件类工具、git、worktree_tool）会覆盖此方法，
    在构造 `SandboxCommand` 或解析文件路径时遵循
    `ctx.working_dir` / `ctx.resolve_path`。

    框架的 `ToolManager::execute_tool_with_context` 总是通过该方法路由，
    因此每个 agent 各自的 `working_dir` 能够自然地传递给工具，
    而无需让（共享、无状态的）ToolManager 持有任何会话状态 ——
    从而避免跨会话的 cwd 污染。
    */
    fn execute_with_context<'a>(
        &'a self,
        parameters: ToolParameters,
        _ctx: &'a ToolContext,
    ) -> BoxFuture<'a, Result<ToolResult>> {
        self.execute(parameters)
    }

    /**
    可选的沙箱注入（P2：run_code 真沙箱）。

    持有沙箱的工具（`ShellTool`/`RunCodeTool`）会覆盖此方法，
    在 agent 建立时（经由 `set_sandbox_manager` → `ToolManager::apply_sandbox`）
    接收执行器。若工具接受了该沙箱返回 `true`；不使用沙箱的工具
    保持默认的 `false`。
    */
    fn set_sandbox(&self, _sandbox: Arc<dyn SandboxExecutor>) -> bool {
        false
    }

    /// 注入一个由应用层拥有的、持久化脚本运行时解析器。
    fn set_script_execution_profile_resolver(
        &self,
        _resolver: Arc<dyn ScriptExecutionProfileResolver>,
    ) -> bool {
        false
    }

    /**
    该工具是否豁免并行批处理超时。

    内部自行运行多步 ReAct 的长耗时工具（subagent 派发类，如 `agent_tool` /
    `delegate_readonly`，以及网络调研类）应覆盖此方法返回 `true`。
    此类工具会从并发批次的总超时中被单独分离出来（见 `react_loop.rs`），
    转而使用自己的单次执行超时（例如 `SubagentExecutor` 中 subagent 的
    默认 600 秒），因为它们的延迟天然远高于典型的文件/shell 工具，
    否则会主导批次预算，过早取消同批次的其他工具。

    默认值：`false`（与其他所有工具一样受批处理超时约束）。
    */
    fn exempt_from_batch_timeout(&self) -> bool {
        false
    }

    /**
    对该工具的调用是否可以与同一次模型响应中的其他调用并发执行。

    输入依赖某个版本号、游标，或另一次调用结果的有状态工具应返回 `false`。
    */
    fn allows_parallel_batch_execution(&self) -> bool {
        true
    }

    /**
    该工具是否自行强制执行截止时间。

    选择退出外层批处理超时的工具，按设计就是长耗时的，通常自己持有更
    合适的截止时间（例如 Subagent 派发超时）。默认保持这一既有行为，
    同时把该决策显式暴露给 `ToolManager`，否则它还会额外施加普通的
    单工具超时。
    */
    fn manages_own_timeout(&self) -> bool {
        self.exempt_from_batch_timeout()
    }

    /**
    以完整的 [`ToolContext`] 流式执行该工具。

    优先使用它而不是 [`Self::execute_stream`]。默认实现调用
    [`Self::execute_with_context`]，并产出单个
    [`ToolStreamEvent::Complete`]。当工具能够实时发出
    [`ToolStreamEvent::Progress`] / [`ToolStreamEvent::Output`] 时，
    应与 [`Self::supports_streaming`] 一并覆盖此方法。

    返回的 stream 必须以 [`ToolStreamEvent::Complete`] 结束。
    */
    fn execute_stream_with_context<'a>(
        &'a self,
        params: ToolParameters,
        ctx: &ToolContext,
    ) -> BoxFuture<'a, Result<Pin<Box<dyn Stream<Item = ToolStreamEvent> + Send + 'a>>>> {
        let ctx = ctx.clone();
        Box::pin(async move {
            let result = self.execute_with_context(params, &ctx).await?;
            let stream: Pin<Box<dyn Stream<Item = ToolStreamEvent> + Send + 'a>> = Box::pin(
                stream::once(async move { ToolStreamEvent::Complete(result) }),
            );
            Ok(stream)
        })
    }

    /**
    以空的 [`ToolContext`] 流式执行该工具。

    兼容性入口 —— 委托给 [`Self::execute_stream_with_context`]。
    新的调用点应改为传入一个真实的 context。
    */
    fn execute_stream<'a>(
        &'a self,
        params: ToolParameters,
    ) -> BoxFuture<'a, Result<Pin<Box<dyn Stream<Item = ToolStreamEvent> + Send + 'a>>>> {
        Box::pin(async move {
            let ctx = ToolContext::default();
            self.execute_stream_with_context(params, &ctx).await
        })
    }

    /**
    该工具是否支持流式执行。

    只有当 [`Self::execute_stream_with_context`] 会发出有实际意义的中间
    事件（`Progress` / `Output`）时才返回 `true`。默认是 `false`，
    以避免不支持流式的工具被误路由到 stream 路径。
    */
    fn supports_streaming(&self) -> bool {
        false
    }

    /// 在执行前校验参数。
    fn validate_parameters<'a>(&'a self, _params: &'a ToolParameters) -> BoxFuture<'a, Result<()>> {
        Box::pin(async { Ok(()) })
    }

    /// 调用该工具所需的权限。
    fn permissions(&self) -> Vec<permission::ToolPermission> {
        vec![]
    }

    /// 该工具的风险等级。危险工具需要显式审批。
    fn risk_level(&self) -> ToolRiskLevel {
        ToolRiskLevel::Standard
    }

    /// 供工具包和过滤器使用的能力元数据。
    fn capabilities(&self) -> ToolCapabilities {
        match self.risk_level() {
            ToolRiskLevel::ReadOnly => ToolCapabilities::read_only(self.permissions()),
            risk => ToolCapabilities::mutating(risk, self.permissions()),
        }
    }

    /// 人类可读的能力声明（例如 "Reads files"、"Executes shell commands"）。
    fn capability_description(&self) -> &str {
        match self.risk_level() {
            ToolRiskLevel::ReadOnly => "Read-only: no side effects",
            ToolRiskLevel::Standard => "Standard: limited side effects",
            ToolRiskLevel::Dangerous => "Dangerous: irreversible side effects",
        }
    }
}

/// run 级 trace 事件 sink：接收序列化为 `serde_json::Value` 的应用层事件。
pub type TraceSinkFn = std::sync::Arc<dyn Fn(serde_json::Value) + Send + Sync>;

/**
在一次 Agent 调用的生命周期内保留的、不透明的所有权令牌。

应用层可以用它包裹一个 permit、租约、临时目录持有者，或任何其他
`Send + Sync + 'static` 资源，其析构函数负责释放所有权。框架只在
一次调用及其 subagent、工具运行期间克隆并保留这个令牌。包裹的值
无法通过该 API 被读取或向下转型，从而把应用层策略挡在框架之外。

Debug 输出只包含 Rust 类型名，绝不会格式化包裹的值。
*/
#[derive(Clone)]
pub struct InvocationResourceGuard {
    /// 被包裹资源的类型名（仅用于 Debug 输出）。
    resource_type: &'static str,
    /// 被保留的不透明资源。
    _resource: std::sync::Arc<dyn std::any::Any + Send + Sync>,
    /// 可选的、不可变的类型化身份描述符。
    identity: Option<std::sync::Arc<dyn std::any::Any + Send + Sync>>,
}

impl InvocationResourceGuard {
    /// 保留 `resource`，直到该 guard 的最后一个克隆被丢弃。
    pub fn new<T>(resource: T) -> Self
    where
        T: Send + Sync + 'static,
    {
        Self {
            resource_type: std::any::type_name::<T>(),
            _resource: std::sync::Arc::new(resource),
            identity: None,
        }
    }

    /**
    保留 `resource`，并附带一个不可变的、类型化的身份描述符。

    该描述符只能通过 [`Self::matches_identity`] 进行比较。资源本身和它的
    身份都无法通过公开 API 被读取或向下转型。
    */
    pub fn new_identified<T, I>(resource: T, identity: I) -> Self
    where
        T: Send + Sync + 'static,
        I: Send + Sync + 'static,
    {
        Self {
            resource_type: std::any::type_name::<T>(),
            _resource: std::sync::Arc::new(resource),
            identity: Some(std::sync::Arc::new(identity)),
        }
    }

    /**
    该 guard 是否恰好保留了类型 `T`。

    这个判定支持在一次调用携带多个独立资源时按类型过滤。它绝不会
    返回或向下转型出包裹的值。如果该 guard 是用 `Arc<T>` 构造的，
    应查询 `retains::<Arc<T>>()`，因为其真实的包裹类型是 `Arc<T>`。
    */
    pub fn retains<T>(&self) -> bool
    where
        T: Send + Sync + 'static,
    {
        self._resource.is::<T>()
    }

    /**
    可选身份是否恰好为类型 `I` 且等于 `expected`。

    缺失身份或类型不匹配的描述符都会返回 `false`。该方法绝不会
    暴露所存储的描述符本身。
    */
    pub fn matches_identity<I>(&self, expected: &I) -> bool
    where
        I: PartialEq + Send + Sync + 'static,
    {
        self.identity
            .as_ref()
            .and_then(|identity| identity.downcast_ref::<I>())
            .is_some_and(|identity| identity == expected)
    }
}

impl std::fmt::Debug for InvocationResourceGuard {
    /// 脱敏的调试输出：只显示资源类型和是否带身份，不泄露具体值。
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InvocationResourceGuard")
            .field("resource_type", &self.resource_type)
            .field("has_identity", &self.identity.is_some())
            .finish_non_exhaustive()
    }
}

/**
Agent 与 subagent 的嵌套委派策略。

之所以放在 `echo_core` 中，是因为 [`ToolContext`] 是 `agent_tool` 使用的跨 spawn
通道；更高层的 task/runtime crate 可以重新导出该类型。
*/
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct NestedDelegationPolicy {
    /// 该 subagent 角色是否可以派生子 subagent。
    pub can_spawn_subagents: bool,
    /// 该 subagent 当前的委派深度。
    pub delegate_depth: u8,
    /// 允许的最大委派深度。
    pub max_delegate_depth: u8,
}

impl Default for NestedDelegationPolicy {
    /// 默认策略：禁止派生子 agent。
    fn default() -> Self {
        Self {
            can_spawn_subagents: false,
            delegate_depth: 0,
            max_delegate_depth: 2,
        }
    }
}

impl NestedDelegationPolicy {
    /// 在此策略下是否可以创建子 subagent。
    pub fn can_delegate(&self) -> bool {
        self.can_spawn_subagents && self.delegate_depth < self.max_delegate_depth
    }

    /// 若允许委派，返回要传给子 subagent 的策略。
    pub fn child_policy(&self) -> Option<Self> {
        if !self.can_delegate() {
            return None;
        }

        Some(Self {
            can_spawn_subagents: self.can_spawn_subagents,
            delegate_depth: self.delegate_depth.saturating_add(1),
            max_delegate_depth: self.max_delegate_depth,
        })
    }
}

/**
调用方解析好的、用于执行一次持久化脚本的运行时。

这是一种执行机制，而不是模型可见的参数。应用层可以准备一个确定性的
语言环境，再通过 [`ToolContext`] 注入解析好的可执行文件。工具不得在其
JSON schema 中暴露该可执行文件或环境变量 map。
*/
#[derive(Clone, PartialEq, Eq)]
pub struct ScriptExecutionProfile {
    /// 记录在工具 metadata 中的、由内容派生的稳定身份标识。
    pub id: String,
    /// 该 profile 可执行的规范化语言名，例如 `python`。
    pub language: String,
    /// 解析后的可执行文件路径，或后端可见的可执行文件名。
    pub program: std::path::PathBuf,
    /// 插入在相对脚本路径之前的参数。
    pub args_prefix: Vec<String>,
    /// 仅提供给沙箱化子进程的环境变量。
    pub env: HashMap<String, String>,
    /// 沙箱必须被允许读取的运行时文件。
    pub read_only_paths: Vec<std::path::PathBuf>,
}

impl std::fmt::Debug for ScriptExecutionProfile {
    /// 自定义调试输出：环境变量只显示键名，不泄露其值。
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScriptExecutionProfile")
            .field("id", &self.id)
            .field("language", &self.language)
            .field("program", &self.program)
            .field("args_prefix", &self.args_prefix)
            .field("environment_keys", &self.env.keys().collect::<Vec<_>>())
            .field("read_only_paths", &self.read_only_paths)
            .finish()
    }
}

impl ScriptExecutionProfile {
    /// 以身份标识、语言名与可执行程序创建 profile，其余字段为空。
    pub fn new(
        id: impl Into<String>,
        language: impl Into<String>,
        program: impl Into<std::path::PathBuf>,
    ) -> Self {
        Self {
            id: id.into(),
            language: language.into(),
            program: program.into(),
            args_prefix: Vec::new(),
            env: HashMap::new(),
            read_only_paths: Vec::new(),
        }
    }

    /// 追加一个插入在脚本路径之前的参数。
    pub fn with_arg_prefix(mut self, argument: impl Into<String>) -> Self {
        self.args_prefix.push(argument.into());
        self
    }

    /// 为沙箱化子进程追加一个环境变量。
    pub fn with_env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.env.insert(key.into(), value.into());
        self
    }

    /// 追加一个沙箱必须允许读取的运行时路径。
    pub fn with_read_only_path(mut self, path: impl Into<std::path::PathBuf>) -> Self {
        self.read_only_paths.push(path.into());
        self
    }
}

/**
为持久化脚本解析一个由应用层拥有的运行时。

实现可以惰性地准备一个受锁定的环境。框架拥有调用和沙箱化的所有权；
应用层拥有解释器和包策略的所有权。返回 `None` 会保留内置的语言可执行文件。
*/
pub trait ScriptExecutionProfileResolver: Send + Sync {
    /// 为给定语言解析出一个脚本执行 profile。
    fn resolve<'a>(
        &'a self,
        language: &'a str,
    ) -> BoxFuture<'a, Result<Option<Arc<ScriptExecutionProfile>>>>;
}

/**
一次已派发 Subagent 尝试的身份与谱系信息。

通过 [`ExternalRunContext`] 以值的形式携带，使得在 Subagent 内部执行的工具
能够知道自己是谁、是谁派发了自己，以及自己在派发树中的位置 ——
而无需持有 agent 句柄。所有字段均为 `Option`；`subagent_lineage: None`
的上下文属于主 agent。
*/
#[derive(Clone, Default, Debug, PartialEq, Eq)]
pub struct SubagentLineage {
    /// 该 Subagent 的角色名（自身身份）。
    pub agent_name: Option<String>,
    /// 本次尝试自身的 execution id（可自我寻址）。
    pub execution_id: Option<String>,
    /// 若在一次正式 run 内被派发，则为对应的应用层 run id。
    pub run_id: Option<String>,
    /// 派发方父 agent 的名称。
    pub parent_agent: Option<String>,
    /**
    当父级本身也是被派发的 Subagent 时，父级当前尝试的 execution id。
    使默认 uplink 能够引导父级的活跃 turn，而不只是发出一个事件。
    */
    pub parent_execution_id: Option<String>,
    /**
    当父级本身也被 envelope 观测时，创建此子派发的父级工具启动事件的
    稳定框架事件 id。
    */
    pub parent_event_id: Option<String>,
    /**
    规范的派发树路径（`root/<child>/<grandchild>`）。在同一逻辑树的多次
    尝试之间保持稳定；在派发时分配。
    */
    pub agent_path: Option<String>,
    /// 当由带修订版本的任务运行时派发时，本次尝试执行的任务图任务 id。
    pub task_id: Option<String>,
    /// 该任务内的尝试序号（从 1 开始）。
    pub attempt: Option<u32>,
    /// 本次尝试所属的任务图计划修订版本。
    pub plan_revision: Option<u64>,
}

/// 一条 uplink 消息的去向。
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SubagentUplinkTarget {
    /// 派发方父级（一个 agent，或一个应用层 run driver）。
    Parent {
        /// 意图：信息性报告 vs. 阻塞式澄清请求。
        kind: SubagentUplinkKind,
        /// 消息正文（已由调用方工具限定长度）。
        text: String,
    },
    /**
    同一派发图中的一个兄弟节点。投递方式仅为入队：该消息会在兄弟节点
    下一个模型边界处引导其活跃 turn，绝不会另起一个新 turn。
    */
    Sibling {
        /// 目标兄弟节点的地址。
        to: SubagentPeerAddress,
        /// 消息正文。
        text: String,
    },
}

/// 如何寻址一个兄弟 peer。
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SubagentPeerAddress {
    /// 一次当前活跃的尝试 —— 可通过共享控制面（实时引导）投递。
    ByExecutionId(String),
    /**
    同一派发图中尚无活跃尝试的任务。框架的默认 sink 无法解析出下一次
    尝试的编号，因此投递需要一个拥有尝试调度权的应用层 sink（例如
    排队等待下一次尝试）。
    */
    ByTaskId(String),
}

/// 一条面向父级的 uplink 消息的意图。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SubagentUplinkKind {
    /// 非阻塞的信息性报告；发送方的尝试继续运行。
    Report,
    /**
    阻塞式的澄清请求。发送方并不等待 —— 由宿主决定（暂停该 run、
    通过引导回答，或忽略），与此同时该尝试继续尽力而为地工作。
    */
    Escalate,
}

/// 一条从运行中的 Subagent 发往其父级或兄弟节点的消息。
#[derive(Clone, Debug)]
pub struct SubagentUplinkMessage {
    /// 从派发运行时上下文中捕获到的发送方身份。
    pub from: SubagentLineage,
    /// 消息的去向及内容。
    pub target: SubagentUplinkTarget,
}

/**
一条 uplink 消息的有界确认回执。按设计是不可失败的：
投递问题通过 `accepted`/`status` 上报，使调用方工具能够把它们转化为
模型可见的结果，而不是错误。
*/
#[derive(Clone, Debug)]
pub struct SubagentUplinkReceipt {
    /// 该消息是否被接受。
    pub accepted: bool,
    /**
    机器可读的处置结果，例如 `event_emitted`、`parent_steered`、
    `delivered_to_sibling`、`no_parent`、`sibling_not_active`、`rejected`。
    */
    pub status: String,
    /// 人类可读的详情说明。
    pub detail: String,
}

/**
由派发方安装进 Subagent 的 [`ExternalRunContext`] 中的 uplink 通道。
沿用与 `TraceSinkFn` 相同的注入模式：框架提供一个默认 sink（事件总线 +
control-registry 引导），应用层可以在派发前安装自己的实现（日志、暂停策略、
界面通知）。
*/
pub type SubagentUplinkFn = std::sync::Arc<
    dyn Fn(SubagentUplinkMessage) -> futures::future::BoxFuture<'static, SubagentUplinkReceipt>
        + Send
        + Sync,
>;

/**
应用层注入的 run 级上下文（跨 spawn 安全，值传递）。

由应用层在每次 run 启动时构造，经 `AgentInvocationContext` 随单次调用
传入，最终由 pipeline 填入 [`ToolContext`]。

为什么需要这个：`tokio::task_local!` 不会跨 `tokio::spawn` 继承。subagent 在框架
层的 `tokio::spawn`（subagent_executor.rs）里执行，task_local 全部丢失。而
`ExternalRunContext` 是值传递（Clone），天然跨 spawn 安全，是传递 run_id /
cancel / trace_sink 的正确通路。

`trace_sink` 用 `serde_json::Value`（而非应用层的具体事件类型）以避免框架依赖
应用层类型——应用层在填入时把自己的事件序列化成 Value。
*/
#[derive(Clone, Default)]
pub struct ExternalRunContext {
    /// 当前会话标识，跨主 agent/subagent 保持稳定。
    pub conversation_id: Option<String>,
    /// 当前应用层正式 run 标识。普通对话 turn 尚未创建任务 run 时为 None。
    pub run_id: Option<String>,
    /// 当前用户输入/agent turn 的稳定标识。
    pub turn_id: Option<String>,
    /**
    当前 run 内的一次具体执行标识。

    `None` 表示只有 run 级上下文。设置后，subagent / tool trace 应使用它
    作为前端可见执行记录的稳定 id，而不是再临时分配一套 dispatch id。
    */
    pub execution_id: Option<String>,
    /**
    隔离资源的稳定标识，可跨同一逻辑任务的多次执行尝试复用。

    与 `execution_id` 不同，此字段不表示一次具体执行，也不用于事件关联。
    Worktree / workspace 等隔离资源可优先使用它来避免重试时重复创建。
    */
    pub isolation_id: Option<String>,
    /**
    触发本次 run 的消息 id（chat 场景下 = message_key，用于把 subagent
    执行流钉到聊天流里对应的消息区块上）。`None` = 非 chat 路径（cron 等）。
    */
    pub message_id: Option<String>,
    /// 当前 run 的取消令牌。
    pub cancel: Option<std::sync::Arc<tokio_util::sync::CancellationToken>>,
    /// Subagent trace 事件回传通道。
    pub trace_sink: Option<TraceSinkFn>,
    /// 要传播进 `agent_tool` 等工具的嵌套委派策略。
    pub delegation_policy: Option<NestedDelegationPolicy>,
    /// 在主 Agent、subagent 与工具 spawn 之间保留的不透明资源。
    pub resource_guards: Vec<InvocationResourceGuard>,
    /**
    该上下文所属的、已派发 Subagent 尝试的身份/谱系信息。
    主 agent 调用时为 `None`。
    */
    pub subagent_lineage: Option<SubagentLineage>,
    /**
    由派发方安装的 Subagent→父级 / Subagent→兄弟节点 消息通道。
    `None` 表示未接入任何 uplink；消息类工具必须上报不可用，
    而不是静默丢弃消息。
    */
    pub uplink: Option<SubagentUplinkFn>,
}

impl std::fmt::Debug for ExternalRunContext {
    /// 自定义调试输出：回调、令牌等不可打印字段以占位符代替。
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExternalRunContext")
            .field("conversation_id", &self.conversation_id)
            .field("run_id", &self.run_id)
            .field("turn_id", &self.turn_id)
            .field("execution_id", &self.execution_id)
            .field("isolation_id", &self.isolation_id)
            .field("message_id", &self.message_id)
            .field(
                "cancel",
                &self.cancel.as_ref().map(|_| "<CancellationToken>"),
            )
            .field("trace_sink", &self.trace_sink.as_ref().map(|_| "<sink>"))
            .field("delegation_policy", &self.delegation_policy)
            .field("resource_guard_count", &self.resource_guards.len())
            .field("resource_guards", &self.resource_guards)
            .field("subagent_lineage", &self.subagent_lineage)
            .field("uplink", &self.uplink.as_ref().map(|_| "<uplink>"))
            .finish()
    }
}

/**
运行时上下文，工具执行时由 ExecuteStage 注入。

所有字段均为 `Option`，`None` = 回退默认行为（向后兼容老工具）。
工具 override `Tool::execute_with_context` 时可读取这些字段。

`cancel` / `trace_sink` 由应用层经 [`ExternalRunContext`] 注入 subagent，
再由 pipeline 填入此处——这是一条跨 `tokio::spawn` 安全的值传递通路（替代会跨
spawn 断裂的 task_local）。
*/
#[derive(Clone, Default)]
pub struct ToolContext {
    /**
    会话绑定的默认工作目录（通常是 worktree 路径）。
    None = 回退进程 `current_dir`。
    */
    pub working_dir: Option<std::path::PathBuf>,
    /// 会话标识（透传给 trace/audit）。
    pub conversation_id: Option<String>,
    /// 当前 run 标识（透传给 trace/audit）。
    pub run_id: Option<String>,
    /// 当前用户输入/agent turn 标识。
    pub turn_id: Option<String>,
    /**
    触发当前工具调用的消息标识。Chat 场景用于把工具派生的执行流绑定到
    正确的消息区块；非 Chat 路径为 `None`。
    */
    pub message_id: Option<String>,
    /// 当前 run 内的一次具体执行标识。
    pub execution_id: Option<String>,
    /// 该逻辑工具调用及其所有重试尝试共用的稳定身份标识。
    pub call_id: Option<String>,
    /**
    由调用方绑定到这一次具体 trace run 和工具调用的可选效果观察者。
    共享工具不得跨调用保留它。
    */
    pub effect_sink: Option<ToolEffectSinkFn>,
    /**
    触发当前 turn 的用户消息。委派类工具会转发这个带类型的值，
    以保持二进制附件完整。
    */
    pub active_message: Option<crate::llm::types::Message>,
    /**
    应用层为完整工具输出 artifact 选定的可选根路径与保留策略。
    流式工具应使用它，而不是在内存中无限保留输出。
    */
    pub output_artifacts: Option<artifact::ToolOutputArtifactConfig>,
    /// 与 `tool_search` 共享的、限定于本次调用的 schema 可见性。
    pub tool_visibility: Option<std::sync::Arc<ToolVisibilityState>>,
    /**
    应用层为一次持久化脚本执行解析出的运行时。

    该值刻意不出现在模型可见的工具参数中。`run_code` 只有在使用了
    `script_path` 且其规范化语言与本次工具调用匹配时，才会应用它。
    */
    pub script_execution_profile: Option<std::sync::Arc<ScriptExecutionProfile>>,
    /// 跨 spawn 安全的取消令牌（值传递，非 task_local）。
    pub cancel: Option<std::sync::Arc<tokio_util::sync::CancellationToken>>,
    /// 跨 spawn 安全的 trace 回传（值传递）。
    pub trace_sink: Option<TraceSinkFn>,
    /// 供 LLM 可调用的派发类工具使用的嵌套委派策略。
    pub delegation_policy: Option<NestedDelegationPolicy>,
    /**
    不透明的调用资源。当工作跨越内部 spawn 或阻塞边界时，
    工具可以克隆并保留这些 guard。
    */
    pub resource_guards: Vec<InvocationResourceGuard>,
    /**
    调用该工具的 Subagent 尝试的身份/谱系信息。
    主 agent 调用该工具时为 `None`。
    */
    pub subagent_lineage: Option<SubagentLineage>,
    /**
    Subagent→父级 / Subagent→兄弟节点 消息通道。
    本次调用未接入 uplink 时为 `None`。
    */
    pub uplink: Option<SubagentUplinkFn>,
}

impl std::fmt::Debug for ToolContext {
    /// 自定义调试输出：回调、令牌等不可打印字段以占位符代替。
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolContext")
            .field("working_dir", &self.working_dir)
            .field("conversation_id", &self.conversation_id)
            .field("run_id", &self.run_id)
            .field("turn_id", &self.turn_id)
            .field("message_id", &self.message_id)
            .field("execution_id", &self.execution_id)
            .field("call_id", &self.call_id)
            .field("effect_sink", &self.effect_sink.as_ref().map(|_| "<sink>"))
            .field("has_active_message", &self.active_message.is_some())
            .field("output_artifacts", &self.output_artifacts)
            .field(
                "visible_tool_count",
                &self
                    .tool_visibility
                    .as_ref()
                    .map(|visibility| visibility.visible_names().len()),
            )
            .field(
                "script_execution_profile",
                &self
                    .script_execution_profile
                    .as_ref()
                    .map(|profile| profile.id.as_str()),
            )
            .field(
                "cancel",
                &self.cancel.as_ref().map(|_| "<CancellationToken>"),
            )
            .field("trace_sink", &self.trace_sink.as_ref().map(|_| "<sink>"))
            .field("delegation_policy", &self.delegation_policy)
            .field("resource_guard_count", &self.resource_guards.len())
            .field("resource_guards", &self.resource_guards)
            .field("subagent_lineage", &self.subagent_lineage)
            .field("uplink", &self.uplink.as_ref().map(|_| "<uplink>"))
            .finish()
    }
}

/// 每次调用可变的工具可见性状态，不重复维护一份完整的工具注册表。
#[derive(Debug)]
pub struct ToolVisibilityState {
    /// 该次调用可激活的完整工具名目录。
    available: std::collections::HashSet<String>,
    /// 当前策略允许的工具名集合。
    eligible: std::sync::RwLock<std::collections::HashSet<String>>,
    /// 当前已对模型可见的工具名集合。
    visible: std::sync::RwLock<std::collections::HashSet<String>>,
}

impl ToolVisibilityState {
    /**
    用一份 eligible 策略集合和初始可见集合构造状态
    （`available` 与 `eligible` 相同）。
    */
    pub fn new(
        eligible: std::collections::HashSet<String>,
        initial: std::collections::HashSet<String>,
    ) -> Self {
        Self::with_available(eligible.clone(), eligible, initial)
    }

    /**
    用比当前 eligible 策略集合更大的可激活目录构建可见性状态。
    产品策略和 skill 之后可以通过 [`Self::extend_eligibility_and_activate`]
    从 `available` 中提升更多名称。
    */
    pub fn with_available(
        available: std::collections::HashSet<String>,
        eligible: std::collections::HashSet<String>,
        initial: std::collections::HashSet<String>,
    ) -> Self {
        let eligible = eligible
            .into_iter()
            .filter(|name| available.contains(name))
            .collect();
        let state = Self {
            available,
            eligible: std::sync::RwLock::new(eligible),
            visible: std::sync::RwLock::new(std::collections::HashSet::new()),
        };
        state.activate(initial);
        state
    }

    /// 给定名称当前是否属于策略允许（eligible）的集合。
    pub fn is_eligible(&self, name: &str) -> bool {
        self.eligible
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .contains(name)
    }

    /// 给定名称当前是否对模型可见。
    pub fn is_visible(&self, name: &str) -> bool {
        self.visible
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .contains(name)
    }

    /// 激活给定名称中所有 eligible 的工具，返回新激活（此前不可见）的名称列表。
    pub fn activate<I>(&self, names: I) -> Vec<String>
    where
        I: IntoIterator<Item = String>,
    {
        let eligible = self
            .eligible
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut visible = self
            .visible
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut activated = Vec::new();
        for name in names {
            if eligible.contains(&name) && visible.insert(name.clone()) {
                activated.push(name);
            }
        }
        activated.sort();
        activated
    }

    /// 扩展本次调用的策略，并一并暴露被提升的 schema。
    pub fn extend_eligibility_and_activate<I>(&self, names: I) -> Vec<String>
    where
        I: IntoIterator<Item = String>,
    {
        let promoted = {
            let mut eligible = self
                .eligible
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            names
                .into_iter()
                .filter(|name| self.available.contains(name))
                .inspect(|name| {
                    eligible.insert(name.clone());
                })
                .collect::<Vec<_>>()
        };
        self.activate(promoted)
    }

    /// 返回排序后的、完整可激活目录中的所有工具名。
    pub fn available_names(&self) -> Vec<String> {
        let mut names = self.available.iter().cloned().collect::<Vec<_>>();
        names.sort();
        names
    }

    /// 返回排序后的、当前对模型可见的所有工具名。
    pub fn visible_names(&self) -> Vec<String> {
        let mut names = self
            .visible
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .iter()
            .cloned()
            .collect::<Vec<_>>();
        names.sort();
        names
    }
}

impl ToolContext {
    /// 解析路径：有绑定且为相对路径则 join；绝对路径或无绑定则原样返回。
    pub fn resolve_path<'a>(
        &self,
        path: &'a std::path::Path,
    ) -> std::borrow::Cow<'a, std::path::Path> {
        match &self.working_dir {
            Some(base) if !path.is_absolute() => std::borrow::Cow::Owned(base.join(path)),
            _ => std::borrow::Cow::Borrowed(path),
        }
    }
}

/// `ToolContext` 与调用资源守卫相关的单元测试。
#[cfg(test)]
mod tool_context_tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// 测试资源：被丢弃时递增计数器，用于验证守卫释放资源。
    struct DropCounter {
        /// 共享的丢弃次数计数器。
        drops: std::sync::Arc<AtomicUsize>,
        /// 不应出现在调试输出中的敏感值。
        _secret: &'static str,
    }

    /// 另一种资源类型，用于验证类型不匹配时取不到资源。
    struct OtherResource;

    /// 测试用资源身份：包含敏感字段与代次。
    #[derive(PartialEq, Eq)]
    struct GuardIdentity {
        /// 不应被泄露的敏感值。
        secret: &'static str,
        /// 身份代次。
        generation: u64,
    }

    /// 另一种身份类型，用于验证身份类型必须精确匹配。
    #[derive(PartialEq, Eq)]
    struct OtherIdentity {
        /// 身份代次。
        generation: u64,
    }

    impl Drop for DropCounter {
        /// 丢弃时将计数器加一。
        fn drop(&mut self) {
            self.drops.fetch_add(1, Ordering::SeqCst);
        }
    }

    /// 验证调用资源守卫不透明、可克隆且调试输出已脱敏。
    #[test]
    fn invocation_resource_guard_is_opaque_cloneable_and_redacted() {
        let drops = std::sync::Arc::new(AtomicUsize::new(0));
        let guard = InvocationResourceGuard::new(DropCounter {
            drops: std::sync::Arc::clone(&drops),
            _secret: "must-not-appear",
        });
        let retained = guard.clone();
        let guards = [guard.clone(), InvocationResourceGuard::new(OtherResource)];

        let debug = format!("{guard:?}");
        assert!(debug.contains("DropCounter"));
        assert!(!debug.contains("must-not-appear"));
        assert!(guard.retains::<DropCounter>());
        assert!(!guard.retains::<OtherResource>());
        assert_eq!(
            guards
                .iter()
                .filter(|candidate| candidate.retains::<DropCounter>())
                .count(),
            1
        );
        assert!(!format!("{guards:?}").contains("must-not-appear"));
        drop(guards);
        drop(guard);
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        drop(retained);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    /// 验证带身份的资源守卫只匹配精确的类型与值，且不泄露身份内容。
    #[test]
    fn identified_resource_guard_matches_exact_type_and_value_without_disclosure() {
        let drops = std::sync::Arc::new(AtomicUsize::new(0));
        let guard = InvocationResourceGuard::new_identified(
            DropCounter {
                drops: std::sync::Arc::clone(&drops),
                _secret: "resource-secret",
            },
            GuardIdentity {
                secret: "identity-secret",
                generation: 7,
            },
        );
        let retained = guard.clone();

        assert!(guard.retains::<DropCounter>());
        assert!(guard.matches_identity(&GuardIdentity {
            secret: "identity-secret",
            generation: 7,
        }));
        assert!(!guard.matches_identity(&GuardIdentity {
            secret: "identity-secret",
            generation: 8,
        }));
        assert!(!guard.matches_identity(&OtherIdentity { generation: 7 }));
        assert!(
            !InvocationResourceGuard::new(OtherResource).matches_identity(&GuardIdentity {
                secret: "identity-secret",
                generation: 7,
            })
        );
        assert!(retained.matches_identity(&GuardIdentity {
            secret: "identity-secret",
            generation: 7,
        }));

        let debug = format!("{guard:?}");
        assert!(debug.contains("has_identity: true"));
        assert!(!debug.contains("resource-secret"));
        assert!(!debug.contains("identity-secret"));
        assert!(!debug.contains("generation"));
        drop(guard);
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        drop(retained);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    /// 验证 `ToolContext` 默认值中所有可选字段均为 None。
    #[test]
    fn test_default_is_all_none() {
        let ctx = ToolContext::default();
        assert!(ctx.working_dir.is_none());
        assert!(ctx.conversation_id.is_none());
        assert!(ctx.run_id.is_none());
        assert!(ctx.message_id.is_none());
        assert!(ctx.delegation_policy.is_none());
        assert!(ctx.resource_guards.is_empty());
    }

    /// 验证没有精确计数的效果保持“未知”，与计数为零区分开。
    #[test]
    fn test_effect_without_exact_count_keeps_unknown_distinct_from_zero()
    -> std::result::Result<(), serde_json::Error> {
        let effect: ToolEffect = serde_json::from_value(serde_json::json!({
            "effect": "test_run",
            "command": "cargo test",
            "passed": false
        }))?;
        assert!(matches!(
            effect,
            ToolEffect::TestRun {
                failure_count: None,
                ..
            }
        ));
        Ok(())
    }

    /// 验证永久失败与部分失败不能被标记为可自动重试。
    #[test]
    fn permanent_and_partial_failures_cannot_be_made_automatically_retryable() {
        let permanent = ToolFailure::new(ToolFailureCategory::Permanent).retryable();
        let partial = ToolFailure::new(ToolFailureCategory::PartialSideEffect)
            .with_idempotency_key("call-1")
            .retryable();

        assert!(!permanent.allows_automatic_retry());
        assert!(!partial.allows_automatic_retry());
    }

    /*
    ── stage4 P4.1: cache_user_id 单一来源 ────────────────────────────
    原先的外部 cache_user_id 通道（ExternalRunContext.cache_user_id +
    ToolContext.cache_user_id）已经是死代码 —— 从没有工具读取过
    ToolContext 的该字段。现已移除，改用单一的配置来源。以下编译期
    断言用于守护这次移除：两个结构体都必须能在没有该字段的情况下构造。
    */

    /// 验证 `ExternalRunContext` 无需缓存用户 ID 即可构造。
    #[test]
    fn external_run_context_constructs_without_cache_user_id() {
        let _ctx = ExternalRunContext {
            conversation_id: Some("conv-1".to_string()),
            run_id: Some("run-1".to_string()),
            turn_id: Some("turn-1".to_string()),
            execution_id: None,
            isolation_id: None,
            message_id: None,
            cancel: None,
            trace_sink: None,
            delegation_policy: None,
            resource_guards: Vec::new(),
            subagent_lineage: None,
            uplink: None,
            // 不再有 cache_user_id 字段——若它仍然存在，这里会编译失败。
        };
    }

    /// 验证 `ToolContext` 无需缓存用户 ID 即可构造。
    #[test]
    fn tool_context_constructs_without_cache_user_id() {
        let _ctx = ToolContext {
            working_dir: None,
            conversation_id: None,
            run_id: None,
            turn_id: None,
            message_id: None,
            execution_id: None,
            call_id: None,
            effect_sink: None,
            output_artifacts: None,
            tool_visibility: None,
            script_execution_profile: None,
            cancel: None,
            trace_sink: None,
            delegation_policy: None,
            active_message: None,
            resource_guards: Vec::new(),
            subagent_lineage: None,
            uplink: None,
            // 不再有 cache_user_id 字段——若它仍然存在，这里会编译失败。
        };
    }

    /// 验证 `new` 正确设置各字段。
    #[test]
    fn test_new_sets_fields() {
        let ctx = ToolContext {
            working_dir: Some(PathBuf::from("/tmp/wt")),
            conversation_id: Some("conv-1".into()),
            run_id: Some("run-1".into()),
            ..Default::default()
        };
        assert_eq!(
            ctx.working_dir.as_deref(),
            Some(std::path::Path::new("/tmp/wt"))
        );
        assert_eq!(ctx.conversation_id.as_deref(), Some("conv-1"));
        assert_eq!(ctx.run_id.as_deref(), Some("run-1"));
    }

    /// 验证相对路径会拼接到工作目录下。
    #[test]
    fn test_resolve_path_relative_joins() {
        let ctx = ToolContext {
            working_dir: Some(PathBuf::from("/repo/wt")),
            ..Default::default()
        };
        let resolved = ctx.resolve_path(std::path::Path::new("src/main.rs"));
        assert_eq!(
            resolved.as_ref(),
            std::path::Path::new("/repo/wt/src/main.rs")
        );
    }

    /// 验证绝对路径不会被拼接。
    #[test]
    fn test_resolve_path_absolute_not_joined() {
        let ctx = ToolContext {
            working_dir: Some(PathBuf::from("/repo/wt")),
            ..Default::default()
        };
        let resolved = ctx.resolve_path(std::path::Path::new("/etc/hosts"));
        assert_eq!(resolved.as_ref(), std::path::Path::new("/etc/hosts"));
    }

    /// 验证未设置工作目录时路径原样返回。
    #[test]
    fn test_resolve_path_no_working_dir_passthrough() {
        let ctx = ToolContext::default();
        let resolved = ctx.resolve_path(std::path::Path::new("a/b"));
        assert_eq!(resolved.as_ref(), std::path::Path::new("a/b"));
    }

    /// 验证可见性状态能把策略允许但尚未可见的工具提升为可见。
    #[test]
    fn visibility_can_promote_available_policy_names() {
        let state = ToolVisibilityState::with_available(
            ["tool_search", "git_status", "git_commit"]
                .into_iter()
                .map(str::to_string)
                .collect(),
            ["tool_search", "git_status"]
                .into_iter()
                .map(str::to_string)
                .collect(),
            ["tool_search", "git_commit"]
                .into_iter()
                .map(str::to_string)
                .collect(),
        );

        assert!(state.is_visible("tool_search"));
        assert!(!state.is_visible("git_commit"));
        assert_eq!(
            state.extend_eligibility_and_activate(["git_commit".to_string()]),
            vec!["git_commit"]
        );
        assert!(state.is_eligible("git_commit"));
        assert!(state.is_visible("git_commit"));
    }

    /// 验证被取消的变更类操作会报告可能的副作用。
    #[test]
    fn cancelled_mutation_reports_possible_side_effect() {
        let error = ReactError::from(ToolError::Cancelled("git_branch".to_string()));
        let failure = ToolFailure::from_error(&error, true);

        assert_eq!(failure.category, ToolFailureCategory::Cancelled);
        assert_eq!(failure.side_effect, ToolSideEffect::Possible);
        assert_eq!(failure.recovery, ToolRecoveryAction::VerifyThenRetry);
        assert!(!failure.allows_automatic_retry());
    }
}

/// `Tool::exempt_from_batch_timeout` 默认值与覆盖行为的测试。
#[cfg(test)]
mod exempt_from_batch_timeout_tests {
    use super::*;

    /**
    一个不覆盖 `exempt_from_batch_timeout` 的工具。用于验证该 trait 的默认值
    是 `false`（与所有普通工具一样受批处理超时约束）。防止默认值被意外改动。
    */
    struct OrdinaryTool;

    impl Tool for OrdinaryTool {
        /// 返回测试工具名称。
        fn name(&self) -> &str {
            "ordinary"
        }
        /// 返回测试工具描述。
        fn description(&self) -> &str {
            "test tool"
        }
        /// 返回测试工具的参数 schema。
        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({})
        }
    }

    /**
    一个覆盖 `exempt_from_batch_timeout -> true` 的工具，模拟长耗时的
    派发类工具（agent_tool / delegate_readonly）。
    */
    struct LongRunningTool;

    impl Tool for LongRunningTool {
        /// 返回测试工具名称。
        fn name(&self) -> &str {
            "long_running"
        }
        /// 返回测试工具描述。
        fn description(&self) -> &str {
            "long-running dispatch tool"
        }
        /// 返回测试工具的参数 schema。
        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({})
        }
        /// 声明豁免批处理超时。
        fn exempt_from_batch_timeout(&self) -> bool {
            true
        }
    }

    /// 验证普通工具默认不豁免批处理超时。
    #[test]
    fn ordinary_tool_is_not_exempt_by_default() {
        let t = OrdinaryTool;
        assert!(
            !t.exempt_from_batch_timeout(),
            "default must be false — ordinary tools are subject to the batch timeout"
        );
    }

    /// 验证长耗时工具可以选择豁免批处理超时。
    #[test]
    fn long_running_tool_can_opt_into_exempt() {
        let t = LongRunningTool;
        assert!(
            t.exempt_from_batch_timeout(),
            "override -> true must be honored so long-running tools bypass the batch timeout"
        );
    }

    /**
    trait 对象派发：该豁免标记必须能通过 `dyn Tool` 被观察到
    （`react_loop` 正是通过 `ToolManager::get_tool` 以这种方式读取它的）。
    */
    #[test]
    fn exemption_visible_through_dyn_trait_object() {
        let tools: Vec<Box<dyn Tool>> = vec![Box::new(OrdinaryTool), Box::new(LongRunningTool)];
        let ordinary = tools
            .iter()
            .find(|t| t.name() == "ordinary")
            .expect("ordinary tool present");
        let long_running = tools
            .iter()
            .find(|t| t.name() == "long_running")
            .expect("long_running tool present");
        assert!(!ordinary.exempt_from_batch_timeout());
        assert!(long_running.exempt_from_batch_timeout());
        assert!(long_running.manages_own_timeout());
    }
}

/// `Tool::execute_with_context` 默认委托行为的测试。
#[cfg(test)]
mod execute_with_context_tests {
    use super::*;

    /**
    一个"旧式"的工具，只实现了 `execute`。用于验证该 trait 默认的
    `execute_with_context` 无论传入什么 context，都会委托给 `execute`。
    */
    struct LegacyTool;

    impl Tool for LegacyTool {
        /// 返回测试工具名称。
        fn name(&self) -> &str {
            "legacy"
        }
        /// 返回测试工具描述。
        fn description(&self) -> &str {
            "old-style tool, no context awareness"
        }
        /// 返回测试工具的参数 schema。
        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({"type": "object", "properties": {}})
        }
        /// 回显参数 `x` 的值。
        fn execute<'a>(&'a self, params: ToolParameters) -> BoxFuture<'a, Result<ToolResult>> {
            Box::pin(async move {
                let msg = format!(
                    "echo: {}",
                    params.get("x").and_then(|v| v.as_str()).unwrap_or("")
                );
                Ok(ToolResult::success(msg))
            })
        }
    }

    /// 验证默认 `execute_with_context` 忽略上下文并委托给 `execute`。
    #[tokio::test]
    async fn test_default_delegates_to_execute_ignoring_ctx() {
        let tool = LegacyTool;
        // 一个非默认的 ctx，默认实现必须忽略它。
        let ctx = ToolContext {
            working_dir: Some(std::path::PathBuf::from("/wt")),
            conversation_id: Some("c".into()),
            run_id: Some("r".into()),
            ..Default::default()
        };
        let mut params = ToolParameters::new();
        params.insert("x".into(), serde_json::json!("hello"));
        let result = tool.execute_with_context(params, &ctx).await.unwrap();
        assert!(result.success, "expected success");
        assert_eq!(result.output, "echo: hello");
    }
}

/// `ToolStreamEvent` 序列化契约的测试。
#[cfg(test)]
mod tool_stream_event_contract_tests {
    use super::*;

    /// 验证各输出通道的 `ToolStreamEvent` 序列化往返一致。
    #[test]
    fn tool_stream_event_serde_round_trip_output_channels() {
        let events = vec![
            ToolStreamEvent::Progress {
                message: "running".into(),
                percent: Some(40),
            },
            ToolStreamEvent::Output {
                channel: ToolOutputChannel::Stdout,
                chunk: "hello 中文 🦀".into(),
            },
            ToolStreamEvent::Output {
                channel: ToolOutputChannel::Stderr,
                chunk: "warn\n".into(),
            },
            ToolStreamEvent::Output {
                channel: ToolOutputChannel::Log,
                chunk: "note".into(),
            },
            ToolStreamEvent::Complete(ToolResult::success("done").with_meta("exit_code", "0")),
        ];

        for event in events {
            let json = serde_json::to_string(&event).expect("serialize");
            let back: ToolStreamEvent = serde_json::from_str(&json).expect("deserialize");
            let again = serde_json::to_string(&back).expect("re-serialize");
            assert_eq!(json, again, "serde round-trip must be stable");
        }
    }

    /// 验证 JSON 中使用 `output` 字段而非 `partial_output`。
    #[test]
    fn tool_stream_event_json_uses_output_not_partial_output() {
        let event = ToolStreamEvent::Output {
            channel: ToolOutputChannel::Stdout,
            chunk: "x".into(),
        };
        let json = serde_json::to_value(&event).expect("to_value");
        assert_eq!(json["event_type"], "output");
        assert!(json.get("chunk").is_some() || json.get("channel").is_some());
        assert!(
            json.get("event_type").and_then(|v| v.as_str()) != Some("partial_output"),
            "PartialOutput variant must not exist"
        );
    }
}
