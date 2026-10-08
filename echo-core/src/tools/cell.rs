/*!
命令单元（command cell）契约 —— 支持长轮询等待的后台 shell 命令。

定义 [`CommandCellRegistry`] trait 及其数据类型。该契约位于 `echo_core`
（纯数据 + trait，不含运行时），这样只依赖 `echo_core` 的 `echo_tools`
就能从 `ShellTool` 中启动 cell，而具体实现（类似
[`crate::sandbox::SandboxExecutor`] 的运行时行为）则位于 `echo_orchestration`
等更上层的 crate 中。

# Cell 语义（Codex 风格的后台命令）

- `launch` 异步地准入并启动一条后台命令，在有界的跟踪容量可用后
  返回一个带类型的回执。
- `wait` 对 cell 进行长轮询：当 cell 进入终止阶段、出现新输出，或调用方的
  让出预算耗尽时返回。它是**可安全重试的**：终止状态可以被重复读取（多个
  等待者 —— 例如主 Agent 与一个确定性的观察者 —— 都能观察到同一个最终
  状态），输出则通过调用方在下次调用时回传的字节游标来消费。
- `stop` 终止底层进程；`list` 对所有 cell 做快照。

[`CommandCellSnapshot::total_output_bytes`] 是指向完整（逻辑）输出流的
单调递增字节偏移；调用方应把返回的 [`CommandCellDelta::next_cursor`]
视为不透明值并原样回传。

Cell 是进程作用域的运行时对象。产品可以持久化生命周期事件，并在重启后
将孤立的 cell 标记为已中断，但本 trait 不承诺跨进程的进程重新接管或
等待的连续性。
*/

use futures::future::BoxFuture;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

use super::artifact::{
    ToolOutputArtifactConfig, ToolOutputArtifactIdentity, ToolOutputArtifactRef,
};

/**
跨越工具边界传递的应用/运行时归属元数据。

通用的 cell 运行时将其视为不透明的关联数据。产品适配器可以用它把 cell
生命周期事件投影到自己的运行存储中，而无需让框架了解应用层的 TaskRun 模型。
*/
#[derive(Debug, Clone, Default)]
pub struct CommandCellOwner {
    /// 所属会话 ID。
    pub conversation_id: Option<String>,
    /// 所属运行 ID。
    pub run_id: Option<String>,
    /// 所属轮次（Turn）ID。
    pub turn_id: Option<String>,
    /// 所属消息 ID。
    pub message_id: Option<String>,
    /// 所属执行 ID。
    pub execution_id: Option<String>,
    /// 触发该 cell 的工具调用 ID。
    pub call_id: Option<String>,
}

/**
后台命令 cell 的启动请求。

调用方（例如 `ShellTool`）负责在调用 [`CommandCellRegistry::launch`]
之前完成安全校验。
*/
#[derive(Debug, Clone, Default)]
pub struct CommandCellRequest {
    /// 要执行的 shell 命令行（通过 `sh -c`）。
    pub command: String,
    /// 命令的可选工作目录。
    pub working_dir: Option<String>,
    /**
    可选的 cell 生命周期（秒）。`None` 表示使用注册表的默认值；
    0 是无效值，因为每个被接受的 cell 都必须有有限的截止时间。
    */
    pub timeout_secs: Option<u64>,
    /**
    前台 shell 配有沙箱执行器，因此 cell 运行时不得悄悄地
    将此次启动降级为直接在宿主机上执行。
    */
    pub require_sandbox: bool,
    /**
    由注册表调用方提供的可选生命周期取消令牌。不应使用普通的前台 Turn
    令牌，因为 cell 的生命周期可能长于该 Turn；具有暂停/恢复语义的产品
    应在运行被最终取消时显式停止 cell。
    */
    pub cancel: Option<Arc<CancellationToken>>,
    /// 不透明的应用关联元数据。
    pub owner: CommandCellOwner,
    /// 从工具上下文继承的完整输出溢出（spill）策略。
    pub output_artifacts: Option<ToolOutputArtifactConfig>,
    /// 与 `output_artifacts` 配对的稳定 artifact 身份。
    pub artifact_identity: Option<ToolOutputArtifactIdentity>,
}

/// 命令 cell 的生命周期阶段。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CommandCellPhase {
    /// 已校验并发布，但尚不允许执行。
    Prepared,
    /// 启动已被接受，cell 正在等待执行容量。
    Queued,
    /// 进程或沙箱执行正在运行。
    Running,
    /// 进程以成功状态退出。
    Succeeded,
    /// 进程以非零状态退出或超时。
    Failed,
    /// 已请求取消，进程已被终止。
    Cancelled,
    /// 进程根本无法启动。
    LaunchFailed,
}

/**
命令 cell 进入终止阶段的带类型原因。

[`CommandCellPhase`] 仍是粗粒度的渲染状态。该值让超时、取消、进程退出
与运行时故障可以区分开来，而无需消费者去对错误文本分类。
*/
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CommandCellTerminalCause {
    /// 进程或沙箱返回了退出状态。
    Exited,
    /// 启动时设定的墙钟截止时间已到。
    TimedOut,
    /// 显式的 cell 取消或所属方取消在终止竞争中胜出。
    Cancelled,
    /// 进程/沙箱无法启动或未能准入。
    LaunchFailed,
    /// 等待进程终止状态时失败。
    WaitFailed,
    /// stdout/stderr 读取器或沙箱流在完整排空之前失败。
    OutputDrainFailed,
}

impl CommandCellTerminalCause {
    /// 返回稳定的 snake_case 名称。
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Exited => "exited",
            Self::TimedOut => "timed_out",
            Self::Cancelled => "cancelled",
            Self::LaunchFailed => "launch_failed",
            Self::WaitFailed => "wait_failed",
            Self::OutputDrainFailed => "output_drain_failed",
        }
    }
}

/// 可选的完整输出 artifact 写入器的状态。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CommandCellArtifactStatus {
    /// 该 cell 未提供 artifact 策略/身份。
    NotRequested,
    /// 输出仍在累积或写入中。
    Writing,
    /// 写入完成，但未达到配置的溢出阈值。
    BelowThreshold,
    /// 可通过 `output_artifact` 获取持久化的 artifact。
    Available,
    /// 写入或最终落盘 artifact 失败。
    Failed,
}

impl CommandCellArtifactStatus {
    /// 返回稳定的 snake_case 名称。
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::NotRequested => "not_requested",
            Self::Writing => "writing",
            Self::BelowThreshold => "below_threshold",
            Self::Available => "available",
            Self::Failed => "failed",
        }
    }
}

impl CommandCellPhase {
    /// 返回稳定的 snake_case 名称。
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Prepared => "prepared",
            Self::Queued => "queued",
            Self::Running => "running",
            Self::Succeeded => "succeeded",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
            Self::LaunchFailed => "launch_failed",
        }
    }

    /// 该阶段是否为终止阶段（cell 不会再改变状态）。
    pub const fn is_terminal(self) -> bool {
        !matches!(self, Self::Prepared | Self::Queued | Self::Running)
    }
}

/// 一轮长轮询返回的原因。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CommandCellWaitReason {
    /// 出现了新的输出。
    Output,
    /// cell 已进入终止阶段。
    Terminal,
    /// 调用方的让出时间预算已耗尽。
    YieldElapsed,
}

/// cell 当前状态的非阻塞快照。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandCellSnapshot {
    /// cell 的唯一标识。
    pub cell_id: String,
    /// 命令的 UTF-8 安全预览（前 N 个字符）。
    pub name: String,
    /// 当前生命周期阶段。
    pub phase: CommandCellPhase,
    /// 带类型的终止类别。cell 运行期间为 `None`。
    pub terminal_cause: Option<CommandCellTerminalCause>,
    /**
    启动/等待/排空失败时的可选诊断信息。消费者必须使用
    `terminal_cause` 进行控制流判断，而不是解析这段文本。
    */
    pub terminal_message: Option<String>,
    /**
    退出码，在进入终止阶段后可用（例如超时和启动失败时为 `None`）。
    */
    pub exit_code: Option<i32>,
    /**
    完整输出流的单调递增字节计数（包括已从内存保留缓冲区中丢弃的字节）。
    */
    pub total_output_bytes: u64,
    /**
    内存保留缓冲区是否已达上限（只保留尾部；较早的输出字节已被丢弃）。
    */
    pub output_truncated: bool,
    /// 可选的完整输出 artifact 写入器的带类型状态。
    pub artifact_status: CommandCellArtifactStatus,
    /**
    可选的 artifact 失败诊断信息。消费者必须使用
    `artifact_status` 进行控制流判断，而不是解析这段文本。
    */
    pub artifact_message: Option<String>,
    /**
    当 cell 已结束且输出超过配置的溢出阈值时，保存完整输出的 artifact。
    */
    pub output_artifact: Option<ToolOutputArtifactRef>,
}

/// 一轮 `wait` 长轮询的结果。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandCellDelta {
    /// 本轮返回时 cell 的状态快照。
    pub snapshot: CommandCellSnapshot,
    /// 本轮返回的原因。
    pub wait_reason: CommandCellWaitReason,
    /// 自调用方游标以来的增量输出（UTF-8 有损转换，受字节上限约束）。
    pub new_output: String,
    /// 下次调用 `wait` 时需要回传的字节游标。
    pub next_cursor: u64,
    /**
    `new_output` 是否被省略：要么触及了单次响应的字节上限，
    要么游标之前的输出已从保留缓冲区中丢弃。
    */
    pub output_elided: bool,
}

/// 已准备好的 cell 被接受执行后返回的回执。
#[must_use]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommandCellLaunchReceipt {
    /// 被接受的 cell 的唯一标识。
    pub cell_id: String,
    /// 被接受执行的时间。
    pub accepted_at: chrono::DateTime<chrono::Utc>,
    /// cell 的墙钟截止时间。
    pub deadline: chrono::DateTime<chrono::Utc>,
}

/// cell 准入、查找与关闭过程中的稳定错误类型。
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CommandCellError {
    /// 请求校验失败。
    #[error("invalid command cell request: {message}")]
    Validation {
        /// 校验失败的说明。
        message: String,
    },
    /// 具有相同标识的 cell 已存在。
    #[error("command cell identity already exists: {cell_id}")]
    DuplicateIdentity {
        /// 冲突的 cell 标识。
        cell_id: String,
    },
    /// 等待容量时超过了截止时间。
    #[error("command cell capacity deadline elapsed")]
    CapacityDeadline,
    /// 准入过程被取消。
    #[error("command cell admission was cancelled")]
    Cancelled,
    /// 注册表正在关闭。
    #[error("command cell registry is shutting down")]
    Shutdown,
    /// 找不到指定的 cell。
    #[error("command cell '{cell_id}' was not found")]
    NotFound {
        /// 未找到的 cell 标识。
        cell_id: String,
    },
    /// 运行时故障。
    #[error("command cell runtime failed: {message}")]
    Runtime {
        /// 故障说明。
        message: String,
    },
}

/// 在多轮 wait 之间保持某个 cell 不被清理的租约。
#[must_use]
pub struct CommandCellObservationLease {
    /// 被保留的 cell 标识。
    cell_id: String,
    /// 租约释放时调用的回调；释放后为 `None`。
    release: Option<Box<dyn FnOnce() + Send + 'static>>,
}

impl CommandCellObservationLease {
    /// 创建一个租约，`release` 会在租约被 drop 时调用一次。
    pub fn new(cell_id: impl Into<String>, release: impl FnOnce() + Send + 'static) -> Self {
        Self {
            cell_id: cell_id.into(),
            release: Some(Box::new(release)),
        }
    }

    /// 返回被保留的 cell 标识。
    pub fn cell_id(&self) -> &str {
        &self.cell_id
    }
}

impl fmt::Debug for CommandCellObservationLease {
    /// 仅输出 cell 标识。
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CommandCellObservationLease")
            .field("cell_id", &self.cell_id)
            .finish_non_exhaustive()
    }
}

impl Drop for CommandCellObservationLease {
    /// drop 时调用释放回调。
    fn drop(&mut self) {
        if let Some(release) = self.release.take() {
            release();
        }
    }
}

/**
后台命令 cell 的注册表。

实现必须是 `Send + Sync`，并支持同一个 cell 上有多个并发等待者。
`launch` 可能需要等待有界容量；`wait` 会在调用方的让出预算内返回。
*/
pub trait CommandCellRegistry: Send + Sync {
    /// 异步注册并启动一个后台命令 cell。
    fn launch(
        &self,
        request: CommandCellRequest,
    ) -> BoxFuture<'_, std::result::Result<CommandCellLaunchReceipt, CommandCellError>>;

    /**
    对 cell 进行长轮询：当其进入终止状态、`cursor` 之后出现新输出，
    或 `yield_ms` 耗尽时返回（以先发生者为准）。
    可安全重试：只要 cell 仍处于配置的进程内保留期内，使用返回的
    `next_cursor` 再次调用都是有效的。活跃的等待租约会保护该 cell
    不被保留清理，直到本轮返回其增量结果。
    `yield_ms = 0` 表示非阻塞轮询。
    */
    fn wait(
        &self,
        cell_id: &str,
        cursor: u64,
        yield_ms: u64,
    ) -> BoxFuture<'_, std::result::Result<CommandCellDelta, CommandCellError>>;

    /// 在多轮 wait 之间保留某个 cell。
    fn observe(
        &self,
        cell_id: &str,
    ) -> std::result::Result<CommandCellObservationLease, CommandCellError>;

    /**
    请求终止某个 cell 的进程。返回该 cell 是否存在（取消本身是异步的）。
    */
    fn stop(&self, cell_id: &str) -> bool;

    /// 对所有被跟踪的 cell（运行中与已终止）做快照。
    fn list(&self) -> BoxFuture<'_, Vec<CommandCellSnapshot>>;

    /// 关闭准入、取消活跃的 cell，并等待自有任务结束。
    fn shutdown(&self) -> BoxFuture<'_, std::result::Result<(), CommandCellError>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phase_as_str_and_terminality() {
        assert_eq!(CommandCellPhase::Prepared.as_str(), "prepared");
        assert_eq!(CommandCellPhase::Queued.as_str(), "queued");
        assert_eq!(CommandCellPhase::Running.as_str(), "running");
        assert_eq!(CommandCellPhase::Succeeded.as_str(), "succeeded");
        assert_eq!(CommandCellPhase::Failed.as_str(), "failed");
        assert_eq!(CommandCellPhase::Cancelled.as_str(), "cancelled");
        assert_eq!(CommandCellPhase::LaunchFailed.as_str(), "launch_failed");

        assert!(!CommandCellPhase::Prepared.is_terminal());
        assert!(!CommandCellPhase::Queued.is_terminal());
        assert!(!CommandCellPhase::Running.is_terminal());
        assert!(CommandCellPhase::Succeeded.is_terminal());
        assert!(CommandCellPhase::Failed.is_terminal());
        assert!(CommandCellPhase::Cancelled.is_terminal());
        assert!(CommandCellPhase::LaunchFailed.is_terminal());
    }

    #[test]
    fn terminal_cause_and_artifact_status_have_stable_names() {
        assert_eq!(CommandCellTerminalCause::Exited.as_str(), "exited");
        assert_eq!(CommandCellTerminalCause::TimedOut.as_str(), "timed_out");
        assert_eq!(
            CommandCellTerminalCause::OutputDrainFailed.as_str(),
            "output_drain_failed"
        );
        assert_eq!(
            CommandCellArtifactStatus::BelowThreshold.as_str(),
            "below_threshold"
        );
        assert_eq!(CommandCellArtifactStatus::Failed.as_str(), "failed");
    }
}
