/*!
人机协作（Human-in-the-loop）审批子系统

集中管理人机协作审批流程所需的 Provider（提供方）、
权限服务以及待处理的刷新规则。
*/

#[cfg(feature = "human-loop")]
use crate::human_loop::{HumanLoopProvider, PermissionService};
#[cfg(feature = "human-loop")]
use echo_core::tools::permission::PermissionRule;
#[cfg(feature = "human-loop")]
use std::sync::Arc;

/**
人机协作（Human-in-the-loop）审批子系统

汇聚了审批 Provider、统一权限服务以及待处理的权限规则。
仅在启用 `human-loop` feature 时才拥有实际字段。
*/
pub(crate) struct ApprovalSubsystem {
    /// 审批提供方，负责向外部（如人类审核者）发起审批请求。
    #[cfg(feature = "human-loop")]
    pub(crate) approval_provider: Arc<dyn HumanLoopProvider>,
    /// 统一权限服务，可选，用于集中判定工具调用是否需要审批。
    #[cfg(feature = "human-loop")]
    pub(crate) permission_service: Option<Arc<PermissionService>>,
    /// 尚未提交生效、等待刷新的权限规则列表。
    #[cfg(feature = "human-loop")]
    pub(crate) pending_permission_rules: std::sync::Mutex<Vec<PermissionRule>>,
}
