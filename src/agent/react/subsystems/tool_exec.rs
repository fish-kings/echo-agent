//! 工具执行子系统
//!
//! 集中管理工具注册/执行、Skill、Hook、MCP、Subagent、Sandbox、
//! 干预回调等与工具调用直接相关的组件。

use crate::agent::InterventionCallback;
#[cfg(feature = "subagent")]
use crate::agent::subagent::SubagentRegistry;
#[cfg(feature = "mcp")]
use crate::mcp::McpManager;
use crate::skills::SkillRegistry;
use crate::skills::hooks::HookRegistry;
use crate::skills::registry::SharedRegistry;
use crate::tools::ToolManager;
use std::sync::Arc;

/// 工具执行子系统
///
/// 汇聚了与工具调用直接相关的所有组件：工具注册表、Skill/Hook
/// 系统、MCP 连接管理、Subagent 调度、沙箱环境以及干预回调。
pub(crate) struct ToolExecutionSubsystem {
    /// 工具注册表（使用 Arc 以便与 StreamRunner 共享）。
    pub(crate) tool_manager: Arc<ToolManager>,
    /// 该 Agent 的任务工具所使用的任务修订版本权威（task revision authority）。
    /// 保留此字段是为了让宿主侧接口（SDK 任务 RPC、EKO TUI/GUI）
    /// 与会话内 `task_*` 工具共享同一个服务，而不是另外构造出一个
    /// 分叉、不一致的存储。
    pub(crate) task_revision_service: Arc<echo_orchestration::tasks::TaskRevisionService>,
    /// 子 Agent 注册表。
    #[cfg(feature = "subagent")]
    pub(crate) subagent_registry: Arc<SubagentRegistry>,
    /// 共享的子 Agent 执行器（带 hook 配置）—— 被 `delegate_task()`
    /// 和 `delegate_to_agent()` 复用，而不是各自创建一次性的执行器。
    #[cfg(feature = "subagent")]
    pub(crate) subagent_executor: Arc<crate::agent::subagent::SubagentExecutor>,
    /// Skill 注册表。
    pub(crate) skill_registry: SkillRegistry,
    /// 可选的渐进式披露 Skill 注册表。
    pub(crate) progressive_skill_registry: Option<SharedRegistry>,
    /// Hook 注册表。
    pub(crate) hook_registry: Arc<tokio::sync::RwLock<HookRegistry>>,
    /// MCP 连接管理器。
    #[cfg(feature = "mcp")]
    pub(crate) mcp_manager: McpManager,
    /// 可选的沙箱执行器，用于隔离运行代码类工具。
    pub(crate) sandbox_manager: Option<Arc<dyn crate::sandbox::SandboxExecutor>>,
    /// 能够在工具调用、LLM 推理与最终回答之前影响 Agent 行为的
    /// 干预回调列表。
    pub(crate) intervention_callbacks: Vec<Arc<dyn InterventionCallback>>,
    /// Agent 级别默认禁用的工具名集合。该集合内的工具会对 LLM 隐藏
    /// （从发送给模型的工具列表中过滤掉）。由应用层填充，用于设置
    /// 持久生效的、Agent 范围的策略。仅针对单次调用的排除项应放在
    /// `AgentInvocationContext::disabled_tools` 中。当前值会被克隆进
    /// 每次运行的快照中，该次运行不会再读取此字段的后续变化。
    ///
    /// 与 `ToolManager::unregister`（会修改共享注册表，从而影响
    /// 池化 Agent 上其他正在进行的轮次）不同，这是一个独立的运行时
    /// 标志 —— 工具本身依然保持注册且对其他 Agent 可用；只是发给
    /// LLM 的工具列表被过滤了。
    pub(crate) tool_visibility: echo_core::agent::ToolVisibilityPolicy,
}

impl ToolExecutionSubsystem {
    /// 为后续运行设置 Agent 级别的默认禁用工具集合。
    ///
    /// 已有的快照是不可变的，不受此调用影响。
    pub(crate) fn set_disabled_tools(&self, names: Option<std::collections::HashSet<String>>) {
        self.tool_visibility.set_disabled(names);
    }

    /// 返回当前 Agent 级别默认禁用的工具名集合。
    pub(crate) fn disabled_tool_names(&self) -> std::collections::HashSet<String> {
        self.tool_visibility.disabled_names()
    }
}
