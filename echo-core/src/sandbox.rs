/*!
沙箱执行器核心 trait 与类型

定义 [`SandboxExecutor`] trait 及其参数/结果类型。
该 trait 由 `echo-execution` crate 中的 `LocalSandbox`、`DockerSandbox`
和 `K8sSandbox` 实现。
*/

use futures::future::BoxFuture;
use futures::stream::{self, Stream};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;
use tokio_util::sync::CancellationToken;

use crate::error::{ReactError, Result, SandboxError};

/**
沙箱执行器统一接口。

全部三层（Local / Docker / K8s）都实现此 trait。
*/
pub trait SandboxExecutor: Send + Sync {
    /// 执行器名称
    fn name(&self) -> &str;

    /// 当前隔离级别
    fn isolation_level(&self) -> IsolationLevel;

    /// 检查该执行器是否可用
    fn is_available(&self) -> BoxFuture<'_, bool>;

    /**
    检查一个可用的后端是否满足调用方要求的最低隔离级别。
    组合式执行器应覆盖此方法，以检查其实际将使用的后端。
    */
    fn is_available_at(&self, minimum: IsolationLevel) -> BoxFuture<'_, bool> {
        Box::pin(async move { self.isolation_level() >= minimum && self.is_available().await })
    }

    /// 执行一条命令
    fn execute(&self, command: SandboxCommand) -> BoxFuture<'_, Result<ExecutionResult>>;

    /**
    以事件流形式执行一条命令。

    不支持实时管道的执行器会显式降级为发出一个缓冲的
    [`SandboxStreamEvent::Complete`] 事件。调用方可以在将输出视为
    实时之前先检查 [`Self::supports_streaming`]。支持实时的后端会在
    流开始后执行或必要清理失败时发出 [`SandboxStreamEvent::Failed`]。
    */
    fn execute_stream<'a>(
        &'a self,
        command: SandboxCommand,
    ) -> BoxFuture<'a, Result<Pin<Box<dyn Stream<Item = SandboxStreamEvent> + Send + 'a>>>> {
        Box::pin(async move {
            let result = self.execute(command).await?;
            let events: Pin<Box<dyn Stream<Item = SandboxStreamEvent> + Send + 'a>> = Box::pin(
                stream::once(async move { SandboxStreamEvent::Complete(result) }),
            );
            Ok(events)
        })
    }

    /// [`Self::execute_stream`] 是否会在完成前就发出输出。
    fn supports_streaming(&self) -> bool {
        false
    }

    /// 在给定资源限制下执行一条命令
    fn execute_with_limits(
        &self,
        command: SandboxCommand,
        limits: ResourceLimits,
    ) -> BoxFuture<'_, Result<ExecutionResult>> {
        let _ = limits;
        self.execute(command)
    }

    /**
    在给定资源限制及可选的所属运行取消令牌下执行。

    当取消需要后端特定的清理逻辑时，具体后端应覆盖此方法。默认实现
    会直接丢弃执行 future 并报告一个带类型的取消结果，而不是悄悄
    当作超时处理，因此只适用于“丢弃 future 就能释放所有已持有资源”
    的情形。进程型、容器型以及编排型后端必须覆盖此方法，并在返回
    最终结果前等待各自的清理完成。
    */
    fn execute_with_limits_and_cancel(
        &self,
        command: SandboxCommand,
        limits: ResourceLimits,
        cancel: Option<Arc<CancellationToken>>,
    ) -> BoxFuture<'_, Result<ExecutionResult>> {
        Box::pin(async move {
            let Some(cancel) = cancel else {
                return self.execute_with_limits(command, limits).await;
            };
            tokio::select! {
                _ = cancel.cancelled() => Err(ReactError::Sandbox(Box::new(
                    SandboxError::Cancelled("owning run was cancelled".to_string()),
                ))),
                result = self.execute_with_limits(command, limits) => result,
            }
        })
    }

    /**
    执行命令，并报告实际运行所用后端的隔离级别。
    组合式执行器必须覆盖此方法，而不是报告其配置的最大隔离级别；
    叶子执行器可以使用默认实现。
    */
    fn execute_with_isolation_receipt(
        &self,
        command: SandboxCommand,
        limits: ResourceLimits,
        cancel: Option<Arc<CancellationToken>>,
    ) -> BoxFuture<'_, Result<(ExecutionResult, IsolationLevel)>> {
        Box::pin(async move {
            let result = self
                .execute_with_limits_and_cancel(command, limits, cancel)
                .await?;
            Ok((result, self.isolation_level()))
        })
    }

    /// 清理沙箱资源（容器、临时文件等）
    fn cleanup(&self) -> BoxFuture<'_, Result<()>> {
        Box::pin(async { Ok(()) })
    }
}

/// 沙箱输出所属的管道通道。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SandboxOutputChannel {
    /// 标准输出管道。
    Stdout,
    /// 标准错误管道。
    Stderr,
}

/// 实时沙箱流发出的带类型终止失败。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SandboxStreamFailure {
    /// 所属运行在后端清理完成后取消了执行。
    Cancelled {
        /// 取消原因的描述信息。
        message: String,
    },
    /// 执行、输出排空或必要的清理失败。
    IoError {
        /// 失败原因的描述信息。
        message: String,
    },
}

impl SandboxStreamFailure {
    /// 返回该失败携带的描述信息。
    pub fn message(&self) -> &str {
        match self {
            Self::Cancelled { message } | Self::IoError { message } => message,
        }
    }

    /// 该失败是否由取消引起。
    pub const fn is_cancelled(&self) -> bool {
        matches!(self, Self::Cancelled { .. })
    }
}

/// 沙箱执行的增量事件。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "event_type", rename_all = "snake_case")]
pub enum SandboxStreamEvent {
    /// 一段增量输出。
    Output {
        /// 该输出片段所属的管道通道。
        channel: SandboxOutputChannel,
        /// 输出文本片段。
        chunk: String,
    },
    /// 终止失败。发出此事件后流即结束。
    Failed {
        /// 带类型的失败详情。
        failure: SandboxStreamFailure,
    },
    /// 执行完成，携带最终的执行结果。
    Complete(ExecutionResult),
}

/// 隔离级别
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum IsolationLevel {
    /// 无隔离（直接在宿主机上执行）
    None = 0,
    /// 进程级隔离（资源限制、路径限制）
    Process = 1,
    /// 操作系统沙箱隔离（bubblewrap / sandbox-exec / seccomp）
    OsSandbox = 2,
    /// 容器隔离（Docker / Podman）
    Container = 3,
    /// 编排工作负载隔离（例如临时的 K8s Pod）
    Orchestrated = 4,
}

impl std::fmt::Display for IsolationLevel {
    /// 以 kebab-case 形式输出隔离级别名称。
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IsolationLevel::None => write!(f, "none"),
            IsolationLevel::Process => write!(f, "process"),
            IsolationLevel::OsSandbox => write!(f, "os-sandbox"),
            IsolationLevel::Container => write!(f, "container"),
            IsolationLevel::Orchestrated => write!(f, "orchestrated"),
        }
    }
}

/// 沙箱执行命令
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandboxCommand {
    /// 命令类型
    pub kind: CommandKind,
    /// 调用方对该命令要求的最低隔离级别。
    #[serde(default)]
    pub minimum_isolation: Option<IsolationLevel>,
    /// 工作目录
    pub working_dir: Option<PathBuf>,
    /// 环境变量
    pub env: HashMap<String, String>,
    /// 执行超时时间
    pub timeout: Duration,
    /// 标准输入
    pub stdin: Option<String>,
}

/// 命令类型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CommandKind {
    /// Shell 命令（通过 sh -c / cmd /C 执行）
    Shell(String),
    /// 直接执行程序
    Program {
        /// 要执行的程序路径或名称。
        program: String,
        /// 传给程序的参数列表。
        args: Vec<String>,
    },
    /// 执行代码片段（需要对应的语言运行时）
    Code {
        /// 代码所用的语言。
        language: String,
        /// 要执行的代码内容。
        code: String,
    },
}

impl SandboxCommand {
    /// 创建一条 shell 命令
    pub fn shell(cmd: impl Into<String>) -> Self {
        Self {
            kind: CommandKind::Shell(cmd.into()),
            minimum_isolation: None,
            working_dir: None,
            env: HashMap::new(),
            timeout: Duration::from_secs(30),
            stdin: None,
        }
    }

    /// 创建一条直接执行程序的命令
    pub fn program(program: impl Into<String>, args: Vec<String>) -> Self {
        Self {
            kind: CommandKind::Program {
                program: program.into(),
                args,
            },
            minimum_isolation: None,
            working_dir: None,
            env: HashMap::new(),
            timeout: Duration::from_secs(30),
            stdin: None,
        }
    }

    /// 创建一条执行代码片段的命令
    pub fn code(language: impl Into<String>, code: impl Into<String>) -> Self {
        Self {
            kind: CommandKind::Code {
                language: language.into(),
                code: code.into(),
            },
            minimum_isolation: None,
            working_dir: None,
            env: HashMap::new(),
            timeout: Duration::from_secs(30),
            stdin: None,
        }
    }

    /// 设置工作目录
    pub fn with_working_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.working_dir = Some(dir.into());
        self
    }

    /// 设置超时时间
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// 要求至少达到该隔离级别，无论管理器策略如何。
    pub fn with_minimum_isolation(mut self, isolation: IsolationLevel) -> Self {
        self.minimum_isolation = Some(isolation);
        self
    }

    /// 添加一个环境变量
    pub fn with_env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.env.insert(key.into(), value.into());
        self
    }

    /// 设置标准输入
    pub fn with_stdin(mut self, stdin: impl Into<String>) -> Self {
        self.stdin = Some(stdin.into());
        self
    }
}

/// 执行结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionResult {
    /// 退出码（0 表示成功）
    pub exit_code: i32,
    /// 标准输出
    pub stdout: String,
    /// 标准错误
    pub stderr: String,
    /// 执行耗时
    pub duration: Duration,
    /// 实际使用的沙箱类型
    pub sandbox_type: String,
    /// 执行是否因超时而被终止
    pub timed_out: bool,
    /// 执行是否因所属运行被取消而终止。
    #[serde(default)]
    pub cancelled: bool,
    /// 保留的输出是否超出了配置的内存上限。
    #[serde(default)]
    pub output_truncated: bool,
    /// 截断前观测到的 stdout 总字节数。
    #[serde(default)]
    pub stdout_bytes: u64,
    /// 截断前观测到的 stderr 总字节数。
    #[serde(default)]
    pub stderr_bytes: u64,
}

impl ExecutionResult {
    /// 执行是否成功
    pub fn success(&self) -> bool {
        self.exit_code == 0 && !self.timed_out && !self.cancelled
    }

    /// 合并后的 stdout + stderr 输出
    pub fn combined_output(&self) -> String {
        if self.stderr.is_empty() {
            self.stdout.clone()
        } else if self.stdout.is_empty() {
            self.stderr.clone()
        } else {
            format!("{}\n{}", self.stdout, self.stderr)
        }
    }

    /// 将保留的 stdout + stderr 限制在一个共享的、UTF-8 安全的字节预算内。
    pub fn enforce_output_limit(&mut self, max_output_bytes: u64) {
        let max_output_bytes = usize::try_from(max_output_bytes).unwrap_or(usize::MAX);
        let original_stdout_bytes = self.stdout.len();
        let original_stderr_bytes = self.stderr.len();

        self.stdout = retain_utf8_prefix(&self.stdout, max_output_bytes);
        let remaining = max_output_bytes.saturating_sub(self.stdout.len());
        self.stderr = retain_utf8_prefix(&self.stderr, remaining);

        self.output_truncated |=
            self.stdout.len() < original_stdout_bytes || self.stderr.len() < original_stderr_bytes;
    }
}

/// 截取不超过 `max_bytes` 字节的最长 UTF-8 安全前缀。
fn retain_utf8_prefix(value: &str, max_bytes: usize) -> String {
    let mut retained_bytes = 0_usize;
    value
        .chars()
        .take_while(|character| {
            let next = retained_bytes.saturating_add(character.len_utf8());
            if next > max_bytes {
                return false;
            }
            retained_bytes = next;
            true
        })
        .collect()
}

/// 资源限制
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    /// 最长执行时间（秒，按墙钟超时处理）
    pub cpu_time_secs: Option<u64>,
    /// 最大内存（字节）
    pub memory_bytes: Option<u64>,
    /// 最大输出大小（字节）
    pub max_output_bytes: Option<u64>,
    /// 最大进程数
    pub max_processes: Option<u32>,
    /// 是否允许联网
    pub network: bool,
    /// 允许挂载的路径（只读）
    pub read_only_paths: Vec<PathBuf>,
    /// 允许挂载的路径（读写）
    pub writable_paths: Vec<PathBuf>,
}

impl Default for ResourceLimits {
    /// 默认限制：30 秒、256 MB 内存、1 MB 输出、64 个进程、禁止联网。
    fn default() -> Self {
        Self {
            cpu_time_secs: Some(30),
            memory_bytes: Some(256 * 1024 * 1024), // 256 MB
            max_output_bytes: Some(1024 * 1024),   // 1 MB
            max_processes: Some(64),
            network: false,
            read_only_paths: vec![],
            writable_paths: vec![],
        }
    }
}

impl ResourceLimits {
    /// 不设任何限制（仅限可信环境）
    pub fn unrestricted() -> Self {
        Self {
            cpu_time_secs: None,
            memory_bytes: None,
            max_output_bytes: None,
            max_processes: None,
            network: true,
            read_only_paths: vec![],
            writable_paths: vec![],
        }
    }

    /// 严格限制（适用于不可信代码）
    pub fn strict() -> Self {
        Self {
            cpu_time_secs: Some(10),
            memory_bytes: Some(64 * 1024 * 1024), // 64 MB
            max_output_bytes: Some(256 * 1024),   // 256 KB
            max_processes: Some(8),
            network: false,
            read_only_paths: vec![],
            writable_paths: vec![],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 永不完成执行的测试用执行器，用于验证取消逻辑。
    struct PendingExecutor;

    impl SandboxExecutor for PendingExecutor {
        fn name(&self) -> &str {
            "pending"
        }

        fn isolation_level(&self) -> IsolationLevel {
            IsolationLevel::OsSandbox
        }

        fn is_available(&self) -> BoxFuture<'_, bool> {
            Box::pin(async { true })
        }

        fn execute(&self, _command: SandboxCommand) -> BoxFuture<'_, Result<ExecutionResult>> {
            Box::pin(std::future::pending())
        }
    }

    /// 构造一个测试用的执行结果。
    fn execution_result(stdout: &str, stderr: &str) -> ExecutionResult {
        ExecutionResult {
            exit_code: 0,
            stdout: stdout.to_string(),
            stderr: stderr.to_string(),
            duration: Duration::ZERO,
            sandbox_type: "test".to_string(),
            timed_out: false,
            cancelled: false,
            output_truncated: false,
            stdout_bytes: u64::try_from(stdout.len()).unwrap_or(u64::MAX),
            stderr_bytes: u64::try_from(stderr.len()).unwrap_or(u64::MAX),
        }
    }

    #[test]
    fn output_limit_is_shared_and_utf8_safe() {
        let mut result = execution_result("中文", "abc");
        result.enforce_output_limit(7);
        assert_eq!(result.stdout, "中文");
        assert_eq!(result.stderr, "a");
        assert!(result.output_truncated);
    }

    #[test]
    fn cancelled_execution_is_not_successful() {
        let mut result = execution_result("", "");
        result.cancelled = true;
        assert!(!result.success());
    }

    #[test]
    fn stream_failure_round_trips_as_typed_terminal() -> Result<()> {
        let event = SandboxStreamEvent::Failed {
            failure: SandboxStreamFailure::IoError {
                message: "cleanup failed".to_string(),
            },
        };
        let encoded = serde_json::to_string(&event).map_err(|error| {
            ReactError::Other(format!("failed to encode sandbox stream failure: {error}"))
        })?;
        let decoded: SandboxStreamEvent = serde_json::from_str(&encoded).map_err(|error| {
            ReactError::Other(format!("failed to decode sandbox stream failure: {error}"))
        })?;
        assert!(matches!(
            decoded,
            SandboxStreamEvent::Failed {
                failure: SandboxStreamFailure::IoError { message }
            } if message == "cleanup failed"
        ));
        Ok(())
    }

    #[tokio::test]
    async fn default_controlled_execution_reports_cancellation() -> Result<()> {
        let cancel = Arc::new(CancellationToken::new());
        cancel.cancel();
        let result = PendingExecutor
            .execute_with_limits_and_cancel(
                SandboxCommand::shell("ignored"),
                ResourceLimits::default(),
                Some(cancel),
            )
            .await;
        match result {
            Err(ReactError::Sandbox(error)) if matches!(*error, SandboxError::Cancelled(_)) => {
                Ok(())
            }
            other => Err(ReactError::Other(format!(
                "expected sandbox cancellation, got {other:?}"
            ))),
        }
    }
}
