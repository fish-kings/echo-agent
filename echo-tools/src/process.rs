use std::ffi::OsStr;
use std::path::Path;
use std::process::{ExitStatus, Stdio};
use std::time::Duration;

use echo_core::error::{Result, ToolError};
use echo_core::tools::ToolContext;
use tokio::io::AsyncReadExt;
use tokio::process::Command;

/// 每次从子进程管道读取的最大字节数。
const READ_CHUNK_BYTES: usize = 16 * 1024;

/// 受限执行的子进程输出：退出状态、（被截断保留的）标准输出/标准错误，以及是否发生过截断。
pub(crate) struct BoundedProcessOutput {
    /// 子进程的退出状态。
    pub status: ExitStatus,
    /// 保留下来的标准输出内容（可能因超出总字节上限而被截断）。
    pub stdout: Vec<u8>,
    /// 保留下来的标准错误内容（可能因超出总字节上限而被截断）。
    pub stderr: Vec<u8>,
    /// 标记 `stdout`/`stderr` 是否因超过 `max_output_bytes` 而被截断。
    pub truncated: bool,
}

/// 启动并执行一个子进程，带超时控制、取消信号响应，并对输出总字节数做上限截断。
///
/// 会持续读取子进程的标准输出与标准错误，直到进程退出或以下情况先发生：
/// 超时、外部取消信号触发。进程退出后清理整个进程组，避免残留子进程。
pub(crate) async fn run_bounded_command<I, S>(
    tool_name: &str,
    program: &str,
    args: I,
    working_dir: &Path,
    context: &ToolContext,
    timeout: Duration,
    max_output_bytes: usize,
) -> Result<BoundedProcessOutput>
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    let mut command = Command::new(program);
    command
        .args(args)
        .current_dir(working_dir)
        .env("GIT_TERMINAL_PROMPT", "0")
        .env("GCM_INTERACTIVE", "Never")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    #[cfg(unix)]
    command.process_group(0);

    let mut child = command
        .spawn()
        .map_err(|error| ToolError::ExecutionFailed {
            tool: tool_name.to_string(),
            message: format!("Unable to start {program}: {error}"),
        })?;
    let mut stdout = child.stdout.take();
    let mut stderr = child.stderr.take();
    let mut stdout_buffer = [0_u8; READ_CHUNK_BYTES];
    let mut stderr_buffer = [0_u8; READ_CHUNK_BYTES];
    let mut retained_stdout = Vec::new();
    let mut retained_stderr = Vec::new();
    let mut truncated = false;
    let mut status = None;
    let deadline = tokio::time::sleep(timeout);
    tokio::pin!(deadline);

    loop {
        if stdout.is_none() && stderr.is_none() && status.is_some() {
            break;
        }

        tokio::select! {
            _ = async {
                if let Some(cancel) = &context.cancel {
                    cancel.cancelled().await;
                } else {
                    std::future::pending::<()>().await;
                }
            } => {
                cleanup_process_tree(&mut child).await;
                return Err(ToolError::Cancelled(tool_name.to_string()).into());
            }
            _ = &mut deadline => {
                cleanup_process_tree(&mut child).await;
                return Err(ToolError::Timeout(tool_name.to_string()).into());
            }
            read = async {
                match stdout.as_mut() {
                    Some(pipe) => pipe.read(&mut stdout_buffer).await,
                    None => Ok(0),
                }
            }, if stdout.is_some() => {
                match read {
                    Ok(0) => stdout = None,
                    Ok(count) => retain_output(
                        stdout_buffer.get(..count).unwrap_or_default(),
                        &mut retained_stdout,
                        retained_stderr.len(),
                        max_output_bytes,
                        &mut truncated,
                    ),
                    Err(error) => {
                        cleanup_process_tree(&mut child).await;
                        return Err(ToolError::ExecutionFailed {
                            tool: tool_name.to_string(),
                            message: format!("Failed to read {program} stdout: {error}"),
                        }.into());
                    }
                }
            }
            read = async {
                match stderr.as_mut() {
                    Some(pipe) => pipe.read(&mut stderr_buffer).await,
                    None => Ok(0),
                }
            }, if stderr.is_some() => {
                match read {
                    Ok(0) => stderr = None,
                    Ok(count) => retain_output(
                        stderr_buffer.get(..count).unwrap_or_default(),
                        &mut retained_stderr,
                        retained_stdout.len(),
                        max_output_bytes,
                        &mut truncated,
                    ),
                    Err(error) => {
                        cleanup_process_tree(&mut child).await;
                        return Err(ToolError::ExecutionFailed {
                            tool: tool_name.to_string(),
                            message: format!("Failed to read {program} stderr: {error}"),
                        }.into());
                    }
                }
            }
            waited = child.wait(), if status.is_none() => {
                match waited {
                    Ok(exit_status) => status = Some(exit_status),
                    Err(error) => {
                        cleanup_process_tree(&mut child).await;
                        return Err(ToolError::ExecutionFailed {
                            tool: tool_name.to_string(),
                            message: format!("Failed to wait for {program}: {error}"),
                        }.into());
                    }
                }
            }
        }
    }

    status
        .map(|status| BoundedProcessOutput {
            status,
            stdout: retained_stdout,
            stderr: retained_stderr,
            truncated,
        })
        .ok_or_else(|| {
            ToolError::ExecutionFailed {
                tool: tool_name.to_string(),
                message: format!("{program} exited without a status"),
            }
            .into()
        })
}

/// 将新读取到的字节追加到 `destination` 中，但不超过 `max_output_bytes` 与 `other_len`（另一路输出已占用的字节数）之和的总预算；
/// 超出部分会被丢弃，并将 `truncated` 置为 `true`。
fn retain_output(
    bytes: &[u8],
    destination: &mut Vec<u8>,
    other_len: usize,
    max_output_bytes: usize,
    truncated: &mut bool,
) {
    let retained = destination.len().saturating_add(other_len);
    let count = max_output_bytes.saturating_sub(retained).min(bytes.len());
    if let Some(prefix) = bytes.get(..count) {
        destination.extend_from_slice(prefix);
    }
    *truncated |= count < bytes.len();
}

/// 终止子进程及其整个进程组（在 Unix 上），并等待其退出以完成回收（reap）。
async fn cleanup_process_tree(child: &mut tokio::process::Child) {
    #[cfg(unix)]
    if let Some(pid) = child.id().and_then(|pid| i32::try_from(pid).ok()) {
        // SAFETY: `kill` 接收的是上面创建的进程组 id 的相反数（负数）；
        // 此处没有指针或借用内存跨越 FFI 边界。
        let _ = unsafe { libc::kill(-pid, libc::SIGKILL) };
    }
    let _ = child.kill().await;
    let _ = child.wait().await;
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use echo_core::agent::CancellationToken;
    use std::sync::Arc;

    /// 取消令牌触发后，正在运行的子进程（及其进程组）应被回收（reap），且返回 `Cancelled` 错误。
    #[tokio::test]
    async fn cancellation_reaps_a_running_process_group() {
        let cancel = Arc::new(CancellationToken::new());
        let context = ToolContext {
            cancel: Some(cancel.clone()),
            ..ToolContext::default()
        };
        let trigger = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(25)).await;
            cancel.cancel();
        });

        let error = run_bounded_command(
            "test_process",
            "sh",
            ["-c", "sleep 30 & wait"],
            Path::new("."),
            &context,
            Duration::from_secs(5),
            1024,
        )
        .await
        .err()
        .ok_or_else(|| "cancelled process unexpectedly succeeded".to_string());
        let _ = trigger.await;

        assert!(matches!(
            error,
            Ok(echo_core::error::ReactError::Tool(tool_error))
                if matches!(*tool_error, ToolError::Cancelled(_))
        ));
    }

    /// 子进程输出应被完全读取（避免阻塞进程退出），但保留在内存中的部分应受 `max_output_bytes` 限制并标记为已截断。
    #[tokio::test]
    async fn output_is_drained_but_retention_is_bounded() -> Result<()> {
        let output = run_bounded_command(
            "test_process",
            "sh",
            ["-c", "yes x | head -c 65536"],
            Path::new("."),
            &ToolContext::default(),
            Duration::from_secs(5),
            1024,
        )
        .await?;

        assert!(output.status.success());
        assert_eq!(
            output.stdout.len().saturating_add(output.stderr.len()),
            1024
        );
        assert!(output.truncated);
        Ok(())
    }
}
