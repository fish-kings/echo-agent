//! 护栏（Guardrails）与安全子系统
//!
//! 集中管理护栏检查、审计日志、熔断器等
//! 与安全相关的组件。

use crate::guard::GuardManager;
use echo_core::circuit_breaker::CircuitBreaker;
use std::sync::Arc;

/// 护栏（Guardrails）与安全子系统
///
/// 汇聚了输入/输出护栏、审计日志记录以及 LLM 熔断器。
pub(crate) struct GuardSubsystem {
    /// 可选的护栏管理器，负责执行输入/输出内容检查。
    pub(crate) guard_manager: Option<GuardManager>,
    /// 可选的审计日志记录器，用于记录工具调用与决策事件。
    pub(crate) audit_logger: Option<Arc<dyn crate::audit::AuditLogger>>,
    /// 可选的 LLM 调用熔断器，用于在异常情况下快速失败。
    pub(crate) circuit_breaker: Option<Arc<CircuitBreaker>>,
}
