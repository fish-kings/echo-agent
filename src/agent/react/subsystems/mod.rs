//! ReactAgent 子系统模块
//!
//! 按职责将 ReactAgent 约 25 个字段拆分为 4 个子系统：
//!
//! | 模块 | 子系统 | 职责 |
//! |--------|-----------|----------------|
//! | `tool_exec` | `ToolExecutionSubsystem` | 工具注册/执行、Skill、Hook、MCP、Subagent、Sandbox |
//! | `guard` | `GuardSubsystem` | 护栏（Guardrails）、权限策略、审计日志、熔断器 |
//! | `memory` | `MemorySubsystem` | 上下文管理、长期记忆、快照、Checkpoint、会话持久化 |
//! | `approval` | `ApprovalSubsystem` | 人机协作（Human-in-the-loop）审批 |

#[cfg(feature = "human-loop")]
pub(crate) mod approval;
pub(crate) mod guard;
pub(crate) mod memory;
pub(crate) mod tool_exec;
