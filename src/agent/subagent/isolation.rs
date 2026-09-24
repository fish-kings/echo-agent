//! 面向 Fork 方式派发的子 Agent 的、与具体产品无关的隔离边界。

use std::path::PathBuf;
use std::sync::Arc;

use super::types::{ObservedIsolation, SubagentArtifact, SubagentEvidence};

/// 向隔离策略提供方发起的隔离请求。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IsolationRequest {
    /// 隔离方式的种类标识，由应用方自行定义与解释。
    pub kind: String,
    /// 该隔离环境的可读标签，通常用于日志与展示。
    pub label: String,
}

/// 一次隔离环境结束（finalize）后产出的结果。
#[derive(Debug, Default)]
pub struct IsolationOutcome {
    /// 对本次隔离执行结果的概要描述。
    pub summary: String,
    /// 本次执行产生的产物（artifact）列表。
    pub artifacts: Vec<SubagentArtifact>,
    /// 本次执行留下的证据（evidence）列表。
    pub evidence: Vec<SubagentEvidence>,
}

/// 隔离环境创建或收尾失败时返回的错误。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IsolationError {
    /// 错误的可读描述信息。
    pub message: String,
}

impl std::fmt::Display for IsolationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for IsolationError {}

impl IsolationError {
    /// 使用给定的错误信息构造一个 `IsolationError`。
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

/// 一次已建立的隔离环境的句柄。
pub struct IsolationHandle {
    /// 该隔离环境对应的文件系统路径。
    pub path: PathBuf,
    /// 已观测到的隔离方式描述。
    pub observed: ObservedIsolation,
    /// 用于结束并收尾该隔离环境、产出最终结果的回调。
    pub finalize: Box<dyn FnOnce() -> Result<IsolationOutcome, IsolationError> + Send>,
}

impl std::fmt::Debug for IsolationHandle {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("IsolationHandle")
            .field("path", &self.path)
            .field("observed", &self.observed)
            .finish_non_exhaustive()
    }
}

/// 由应用方提供的隔离策略。框架既不解释 `kind` 的含义，
/// 也不拥有返回句柄背后的生命周期管理策略。
pub trait IsolationProvider: Send + Sync {
    /// 根据给定的请求建立一个隔离环境，并返回其句柄。
    fn isolate(&self, request: &IsolationRequest) -> Result<IsolationHandle, IsolationError>;
}

/// 可在多处共享的隔离策略提供方引用。
pub type SharedIsolationProvider = Arc<dyn IsolationProvider>;
