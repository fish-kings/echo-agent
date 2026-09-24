//! 沙箱化的内联代码或持久化脚本执行。
//!
//! 允许子代理执行任意的 Python/R/JS/... 代码片段，或执行已保存在
//! `ctx.working_dir` 中、可供审查的脚本。持久化脚本由对应语言的运行时
//! 直接执行，而不是被读回并包装成内联代码再执行。
//!
//! `run_code` 是一个由代理控制的任意代码执行原语，因此它始终要求配置
//! OS 级别或更强的 [`echo_core::sandbox::SandboxExecutor`]。交互式终端是
//! 独立的、由用户控制的能力，不受此处的限制约束。
//!
//! 设计上参照 `ShellTool`（`shell.rs`）：持有 `Option<Arc<dyn SandboxExecutor>>`
//! （echo-core 中定义的 trait，不依赖 echo-execution），并重写
//! `execute_with_context` 以遵循 `ctx.working_dir`。

use echo_core::error::{ReactError, Result, ToolError};
use echo_core::sandbox::{
    ExecutionResult, IsolationLevel, ResourceLimits, SandboxCommand, SandboxExecutor,
};
use echo_core::tools::permission::ToolPermission;
use echo_core::tools::{
    ScriptExecutionProfile, ScriptExecutionProfileResolver, Tool, ToolContext, ToolFailure,
    ToolFailureCategory, ToolParameters, ToolResult, ToolResultKind, ToolRiskLevel, ToolSideEffect,
};
use futures::future::BoxFuture;
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// [`RunCodeTool`] 支持的语言列表。内联代码使用现有的基于参数的 `Code`
/// 后端；持久化脚本则直接调用对应的解释器程序。
const SUPPORTED_LANGUAGES: &[&str] = &[
    "python",
    "python3",
    "r",
    "javascript",
    "js",
    "node",
    "ruby",
    "perl",
    "php",
    "bash",
    "sh",
];
/// 默认保留的标准输出+标准错误字节预算上限。
const DEFAULT_MAX_OUTPUT_BYTES: u64 = 1024 * 1024;

/// 校验语言是否在支持集合内（大小写不敏感）。
///
/// 决策 D-10b-case-1：LLM 可能会给出 `Python`/`PYTHON`/`R` 等形式，因此
/// 匹配前先统一归一化为小写。
///
/// 返回归一化后的小写语言名，或返回错误。
fn validate_language(language: &str) -> Result<String> {
    let normalized = language.to_lowercase();
    if SUPPORTED_LANGUAGES.iter().any(|&l| l == normalized) {
        Ok(normalized)
    } else {
        // ToolError::InvalidParameter 是一个结构体变体 {name, message}
        // （echo-core/src/error.rs:118）。这里显式包装为 ReactError，
        // 以避免调用处的 Into 转换产生歧义。
        Err(ReactError::from(ToolError::InvalidParameter {
            name: "language".to_string(),
            message: format!(
                "Unsupported language '{language}'. Supported: {:?}",
                SUPPORTED_LANGUAGES
            ),
        }))
    }
}

/// 返回给定语言对应的脚本解释器可执行文件名，用于直接执行持久化脚本。
fn script_program(language: &str) -> Result<&'static str> {
    match language {
        "python" | "python3" => Ok("python3"),
        "r" => Ok("Rscript"),
        "javascript" | "js" | "node" => Ok("node"),
        "ruby" => Ok("ruby"),
        "perl" => Ok("perl"),
        "php" => Ok("php"),
        "bash" => Ok("bash"),
        "sh" => Ok("sh"),
        _ => Err(ReactError::from(ToolError::InvalidParameter {
            name: "language".to_string(),
            message: format!("No script runtime is configured for '{language}'"),
        })),
    }
}

/// 根据语言和（可选的）脚本执行画像构建持久化脚本的沙箱命令。
///
/// 若提供了 `profile`，会校验其语言与目标语言一致，并使用画像中指定的
/// 解释器程序、前置参数与环境变量；否则退回使用默认的解释器程序。
fn script_command(
    language: &str,
    relative_script: &Path,
    profile: Option<&Arc<ScriptExecutionProfile>>,
) -> Result<(SandboxCommand, Option<Arc<ScriptExecutionProfile>>)> {
    let script_argument = relative_script.to_string_lossy().to_string();
    let Some(profile) = profile else {
        return Ok((
            SandboxCommand::program(script_program(language)?, vec![script_argument]),
            None,
        ));
    };

    let profile_language = validate_language(profile.language.trim())?;
    if profile_language != language {
        return Err(ReactError::from(ToolError::InvalidParameter {
            name: "script_execution_profile".to_string(),
            message: format!(
                "Script profile '{}' is for language '{}', not '{language}'",
                profile.id, profile_language
            ),
        }));
    }
    if profile.id.trim().is_empty() || profile.program.as_os_str().is_empty() {
        return Err(ReactError::from(ToolError::InvalidParameter {
            name: "script_execution_profile".to_string(),
            message: "Script profile must have a non-empty id and program".to_string(),
        }));
    }

    let mut args = profile.args_prefix.clone();
    args.push(script_argument);
    let mut command = SandboxCommand::program(profile.program.to_string_lossy(), args);
    for (key, value) in &profile.env {
        command = command.with_env(key, value);
    }
    Ok((command, Some(profile.clone())))
}

/// 校验脚本路径必须是相对路径，且规范化后仍停留在执行工作目录之内
/// （防止通过 `..`、绝对路径或跨目录符号链接逃逸出工作目录）。
///
/// 返回校验通过的相对路径。
fn validate_script_path(script_path: &str, working_dir: Option<&Path>) -> Result<PathBuf> {
    let trimmed = script_path.trim();
    if trimmed.is_empty() {
        return Err(ReactError::from(ToolError::InvalidParameter {
            name: "script_path".to_string(),
            message: "Script path cannot be empty".to_string(),
        }));
    }

    let relative = Path::new(trimmed);
    if relative.is_absolute()
        || relative.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
    {
        return Err(ReactError::from(ToolError::InvalidParameter {
            name: "script_path".to_string(),
            message: "Script path must stay inside the execution working directory".to_string(),
        }));
    }

    let base = match working_dir {
        Some(path) => path.to_path_buf(),
        None => std::env::current_dir().map_err(|error| ToolError::ExecutionFailed {
            tool: "run_code".to_string(),
            message: format!("Cannot resolve the current working directory: {error}"),
        })?,
    };
    let canonical_base = base
        .canonicalize()
        .map_err(|error| ToolError::ExecutionFailed {
            tool: "run_code".to_string(),
            message: format!("Cannot access the execution working directory: {error}"),
        })?;
    let resolved = base.join(relative);
    let canonical_script = resolved
        .canonicalize()
        .map_err(|error| ToolError::ExecutionFailed {
            tool: "run_code".to_string(),
            message: format!("Cannot access script '{trimmed}': {error}"),
        })?;
    if !canonical_script.starts_with(&canonical_base) || !canonical_script.is_file() {
        return Err(ReactError::from(ToolError::InvalidParameter {
            name: "script_path".to_string(),
            message: "Script must be a file inside the execution working directory".to_string(),
        }));
    }

    Ok(relative.to_path_buf())
}

/// 沙箱化的内联代码/持久化脚本执行工具。
pub struct RunCodeTool {
    /// 使用内部可变性，使得 `Tool::set_sandbox`（一个 `&self` 方法）能够
    /// 在构造之后（通过 `set_sandbox_manager` → `ToolManager::apply_sandbox`）
    /// 注入执行器，而无需 `&mut self`。
    sandbox: Mutex<Option<Arc<dyn SandboxExecutor>>>,
    /// 应用侧持有的、用于持久化脚本的可选惰性运行时解析器。
    script_profile_resolver: Mutex<Option<Arc<dyn ScriptExecutionProfileResolver>>>,
    /// 单次调用的超时时间（秒），默认 60，最大限制为 300（与 ShellTool 一致）。
    timeout_secs: u64,
    /// 共享的标准输出+标准错误保留字节预算。
    max_output_bytes: u64,
}

impl Default for RunCodeTool {
    fn default() -> Self {
        Self {
            sandbox: Mutex::new(None),
            script_profile_resolver: Mutex::new(None),
            timeout_secs: 60,
            max_output_bytes: DEFAULT_MAX_OUTPUT_BYTES,
        }
    }
}

impl RunCodeTool {
    /// 使用默认配置创建一个新的 `RunCodeTool`。
    pub fn new() -> Self {
        Self::default()
    }

    /// 注入一个 OS 级别或更强的沙箱执行器。
    pub fn with_sandbox(self, sandbox: Arc<dyn SandboxExecutor>) -> Self {
        *self
            .sandbox
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(sandbox);
        self
    }

    /// 设置单次调用超时时间（默认 60 秒，最大限制为 300 秒，与 ShellTool 一致）。
    pub fn with_timeout_secs(mut self, secs: u64) -> Self {
        self.timeout_secs = secs;
        self
    }

    /// 覆盖默认的标准输出+标准错误保留字节预算。
    pub fn with_max_output_bytes(mut self, max_output_bytes: u64) -> Self {
        self.max_output_bytes = max_output_bytes;
        self
    }
}

impl Tool for RunCodeTool {
    /// 返回工具名称 `run_code`。
    fn name(&self) -> &str {
        "run_code"
    }

    /// 返回面向 LLM 的工具描述文本。
    fn description(&self) -> &str {
        "Execute Python/R/JavaScript/etc. code in the task working directory. For real analysis, save the script first and run it via `script_path` (a reviewed file); for quick exploration use `code`. Provide exactly one of the two. Returns stdout/stderr/exit code."
    }

    /// 返回工具的 JSON Schema 参数定义。
    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "language": {
                    "type": "string",
                    "enum": ["python", "r", "javascript", "ruby", "perl", "php", "bash"],
                    "description": "Code language (case-insensitive). Defaults to python."
                },
                "code": {
                    "type": "string",
                    "description": "Inline code snippet for quick execution; mutually exclusive with `script_path`."
                },
                "script_path": {
                    "type": "string",
                    "description": "Relative path of a saved, reviewed script inside `working_dir`; mutually exclusive with `code`."
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (optional, default 60, max 300)."
                }
            },
            "required": ["language"],
            "oneOf": [
                {"required": ["code"], "not": {"required": ["script_path"]}},
                {"required": ["script_path"], "not": {"required": ["code"]}}
            ]
        })
    }

    /// 返回该工具所需的权限：执行权限。
    fn permissions(&self) -> Vec<ToolPermission> {
        vec![ToolPermission::Execute]
    }

    /// 返回该工具的风险等级：危险（可执行任意代码）。
    fn risk_level(&self) -> ToolRiskLevel {
        ToolRiskLevel::Dangerous
    }

    /// P2：在代理初始化阶段接收沙箱执行器（通过 ToolManager::apply_sandbox 注入）。
    fn set_sandbox(&self, sandbox: Arc<dyn SandboxExecutor>) -> bool {
        *self
            .sandbox
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(sandbox);
        true
    }

    /// 注入用于解析持久化脚本运行时画像的解析器。
    fn set_script_execution_profile_resolver(
        &self,
        resolver: Arc<dyn ScriptExecutionProfileResolver>,
    ) -> bool {
        *self
            .script_profile_resolver
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(resolver);
        true
    }

    /// 校验参数、构建沙箱命令并在沙箱中执行代码/脚本，返回执行结果。
    fn execute_with_context<'a>(
        &'a self,
        parameters: ToolParameters,
        ctx: &'a ToolContext,
    ) -> BoxFuture<'a, Result<ToolResult>> {
        Box::pin(async move {
            let sandbox = self
                .sandbox
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .clone();
            let script_profile_resolver = self
                .script_profile_resolver
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .clone();
            // 1. 解析语言参数，并在语言未知时立即熔断（用户评审补丁 #1）。
            let raw_lang = parameters
                .get("language")
                .and_then(|v| v.as_str())
                .ok_or_else(|| {
                    ReactError::from(ToolError::MissingParameter("language".to_string()))
                })?;
            let language = validate_language(raw_lang)?;

            let code = parameters
                .get("code")
                .and_then(|value| value.as_str())
                .filter(|value| !value.trim().is_empty());
            let script_path = parameters
                .get("script_path")
                .and_then(|value| value.as_str())
                .filter(|value| !value.trim().is_empty());
            let resolved_script_profile = if script_path.is_some() {
                if let Some(profile) = &ctx.script_execution_profile {
                    Some(profile.clone())
                } else if let Some(resolver) = script_profile_resolver {
                    resolver.resolve(&language).await?
                } else {
                    None
                }
            } else {
                None
            };
            let (mut sandbox_cmd, execution_mode, persisted_script, script_profile) =
                match (code, script_path) {
                    (Some(code), None) => {
                        (SandboxCommand::code(&language, code), "inline", None, None)
                    }
                    (None, Some(script_path)) => {
                        let relative =
                            validate_script_path(script_path, ctx.working_dir.as_deref())?;
                        let (command, profile) =
                            script_command(&language, &relative, resolved_script_profile.as_ref())?;
                        (command, "script", Some(relative), profile)
                    }
                    (Some(_), Some(_)) => {
                        return Err(ReactError::from(ToolError::InvalidParameter {
                            name: "code/script_path".to_string(),
                            message: "Provide exactly one of code or script_path".to_string(),
                        }));
                    }
                    (None, None) => {
                        return Err(ReactError::from(ToolError::InvalidParameter {
                            name: "code/script_path".to_string(),
                            message: "Provide exactly one of code or script_path".to_string(),
                        }));
                    }
                };

            let timeout_secs = parameters
                .get("timeout")
                .and_then(|v| v.as_u64())
                .unwrap_or(self.timeout_secs)
                .clamp(1, 300);
            let timeout_duration = Duration::from_secs(timeout_secs);

            // 2. 绑定子代理的工作目录，并设置最低隔离级别。
            //    `with_timeout` 接收一个 Duration（echo-core/src/sandbox.rs:148）。
            sandbox_cmd = sandbox_cmd.with_minimum_isolation(IsolationLevel::OsSandbox);
            if let Some(dir) = &ctx.working_dir {
                sandbox_cmd = sandbox_cmd.with_working_dir(dir.clone());
            }
            sandbox_cmd = sandbox_cmd.with_timeout(timeout_duration);

            let Some(sandbox) = sandbox else {
                return Ok(sandbox_unavailable_result(
                    "run_code requires a configured SandboxExecutor; unsandboxed execution is disabled",
                ));
            };
            if ctx
                .cancel
                .as_ref()
                .is_some_and(|cancel| cancel.is_cancelled())
            {
                return Ok(cancelled_result(
                    "run_code was cancelled before sandbox startup",
                ));
            }
            if !sandbox.is_available_at(IsolationLevel::OsSandbox).await {
                return Ok(sandbox_unavailable_result(format!(
                    "run_code sandbox '{}' has no available OS-isolated backend",
                    sandbox.name()
                )));
            }

            let mut limits = ResourceLimits {
                cpu_time_secs: Some(timeout_secs),
                max_output_bytes: Some(self.max_output_bytes),
                network: false,
                ..Default::default()
            };
            if let Some(working_dir) = &ctx.working_dir {
                limits.writable_paths.push(working_dir.clone());
            }
            if let Some(profile) = &script_profile {
                limits
                    .read_only_paths
                    .extend(profile.read_only_paths.iter().cloned());
            }

            match sandbox
                .execute_with_isolation_receipt(sandbox_cmd, limits, ctx.cancel.clone())
                .await
            {
                Ok((result, actual_isolation)) => {
                    let mut tool_result = tool_result_from_execution(
                        result,
                        ctx.working_dir.as_ref(),
                        actual_isolation,
                        self.max_output_bytes,
                    )
                    .with_meta("execution_mode", execution_mode);
                    if let Some(script) = persisted_script {
                        tool_result = tool_result
                            .with_meta("script_path", script.to_string_lossy().to_string());
                    }
                    if let Some(profile) = script_profile {
                        tool_result = tool_result.with_meta("script_profile", profile.id.clone());
                    }
                    Ok(tool_result)
                }
                Err(error) => Ok(sandbox_error_result(error)),
            }
        })
    }
}

/// 构建表示「沙箱不可用」的错误结果。
fn sandbox_unavailable_result(message: impl Into<String>) -> ToolResult {
    ToolResult::error(message).with_failure(
        ToolFailure::new(ToolFailureCategory::Unavailable)
            .with_postcondition("configure an OS-level SandboxExecutor before retrying run_code"),
    )
}

/// 构建表示「执行被取消」的错误结果，并提示调用方需要核实沙箱进程状态。
fn cancelled_result(message: impl Into<String>) -> ToolResult {
    let mut failure = ToolFailure::new(ToolFailureCategory::Cancelled);
    failure.side_effect = ToolSideEffect::Possible;
    failure.postcondition = Some(
        "verify the sandbox process stopped and inspect working-directory outputs".to_string(),
    );
    ToolResult::error(message).with_failure(failure)
}

/// 将沙箱执行过程中产生的错误转换为工具错误结果。
fn sandbox_error_result(error: ReactError) -> ToolResult {
    let mut failure = ToolFailure::from_error(&error, true);
    if failure.category == ToolFailureCategory::Cancelled {
        failure.side_effect = ToolSideEffect::Possible;
        failure.postcondition = Some(
            "verify the sandbox process stopped and inspect working-directory outputs".to_string(),
        );
    }
    ToolResult::error(format!("Sandbox execution failed: {error}")).with_failure(failure)
}

/// 将沙箱的原始 `ExecutionResult` 转换为携带完整元数据的 `ToolResult`
/// （包括退出码、耗时、隔离级别、是否超时/被取消/被截断等信息）。
fn tool_result_from_execution(
    mut execution: ExecutionResult,
    working_dir: Option<&std::path::PathBuf>,
    isolation_level: IsolationLevel,
    max_output_bytes: u64,
) -> ToolResult {
    execution.enforce_output_limit(max_output_bytes);
    let stdout_bytes = if execution.stdout_bytes == 0 {
        u64::try_from(execution.stdout.len()).unwrap_or(u64::MAX)
    } else {
        execution.stdout_bytes
    };
    let stderr_bytes = if execution.stderr_bytes == 0 {
        u64::try_from(execution.stderr.len()).unwrap_or(u64::MAX)
    } else {
        execution.stderr_bytes
    };
    let combined_output = execution.combined_output();

    let mut result = if execution.success() {
        ToolResult::success_with_kind(
            ToolResultKind::CommandOutput {
                exit_code: Some(execution.exit_code),
            },
            combined_output,
        )
    } else {
        let failure = if execution.cancelled {
            let mut failure = ToolFailure::new(ToolFailureCategory::Cancelled);
            failure.side_effect = ToolSideEffect::Possible;
            failure.postcondition = Some(
                "verify the sandbox process stopped and inspect working-directory outputs"
                    .to_string(),
            );
            failure
        } else if execution.timed_out {
            ToolFailure::new(ToolFailureCategory::Timeout)
                .with_side_effect(ToolSideEffect::Possible)
                .with_postcondition(
                    "verify the sandbox process stopped and inspect working-directory outputs before retrying",
                )
        } else {
            let mut failure = ToolFailure::new(ToolFailureCategory::Permanent);
            failure.side_effect = ToolSideEffect::Possible;
            failure.postcondition = Some(
                "inspect stdout, stderr, and working-directory outputs before continuing"
                    .to_string(),
            );
            failure
        };
        let message = if execution.cancelled {
            "Code execution cancelled".to_string()
        } else if execution.timed_out {
            "Code execution timed out".to_string()
        } else {
            format!(
                "Code execution failed with exit code {}",
                execution.exit_code
            )
        };
        ToolResult::error(message)
            .with_failure(failure)
            .with_output(combined_output)
    };

    result.kind = ToolResultKind::CommandOutput {
        exit_code: Some(execution.exit_code),
    };
    result.truncated = execution.output_truncated;
    result
        .with_meta("duration_ms", execution.duration.as_millis().to_string())
        .with_meta("exit_code", execution.exit_code.to_string())
        .with_meta("sandbox_type", execution.sandbox_type)
        .with_meta("isolation_level", isolation_level.to_string())
        .with_meta(
            "working_dir",
            working_dir
                .map(|dir| dir.display().to_string())
                .unwrap_or_default(),
        )
        .with_meta("timed_out", execution.timed_out.to_string())
        .with_meta("cancelled", execution.cancelled.to_string())
        .with_meta("output_truncated", execution.output_truncated.to_string())
        .with_meta("stdout_bytes", stdout_bytes.to_string())
        .with_meta("stderr_bytes", stderr_bytes.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use echo_core::agent::CancellationToken;
    use echo_core::sandbox::CommandKind;
    use std::collections::HashMap;

    /// 测试用：记录一次被拦截的沙箱调用，便于断言实际传入的命令与限制。
    #[derive(Clone)]
    struct RecordedCall {
        /// 实际传给沙箱的命令。
        command: SandboxCommand,
        /// 实际传给沙箱的资源限制。
        limits: ResourceLimits,
    }

    /// 测试用沙箱：不真正执行命令，只记录调用并返回预设结果。
    struct RecordingSandbox {
        /// 该沙箱声明的隔离级别。
        isolation: IsolationLevel,
        /// 若设置，表示实际生效的隔离级别与声明的不同（用于测试隔离级别上报逻辑）。
        actual_isolation: Option<IsolationLevel>,
        /// 该沙箱是否可用。
        available: bool,
        /// 调用执行方法时返回的预设结果。
        result: ExecutionResult,
        /// 最近一次被记录下来的调用。
        seen: std::sync::Mutex<Option<RecordedCall>>,
    }

    /// 测试用脚本执行画像解析器：只对固定语言返回固定画像，并统计调用次数。
    struct FixedProfileResolver {
        /// 固定返回的脚本执行画像。
        profile: Arc<ScriptExecutionProfile>,
        /// 记录 `resolve` 被调用的次数。
        calls: std::sync::atomic::AtomicUsize,
    }

    impl ScriptExecutionProfileResolver for FixedProfileResolver {
        /// 若请求的语言与固定画像的语言一致则返回该画像，否则返回 `None`。
        fn resolve<'a>(
            &'a self,
            language: &'a str,
        ) -> BoxFuture<'a, Result<Option<Arc<ScriptExecutionProfile>>>> {
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let profile = (language == self.profile.language).then(|| self.profile.clone());
            Box::pin(async move { Ok(profile) })
        }
    }

    impl SandboxExecutor for RecordingSandbox {
        /// 返回固定的沙箱名称 "recording"。
        fn name(&self) -> &str {
            "recording"
        }

        /// 返回该沙箱声明的隔离级别。
        fn isolation_level(&self) -> IsolationLevel {
            self.isolation
        }

        /// 返回预设的可用性。
        fn is_available(&self) -> BoxFuture<'_, bool> {
            let available = self.available;
            Box::pin(async move { available })
        }

        /// 忽略传入命令，直接返回预设的执行结果（不记录调用）。
        fn execute(
            &self,
            _command: SandboxCommand,
        ) -> BoxFuture<'_, std::result::Result<ExecutionResult, ReactError>> {
            let result = self.result.clone();
            Box::pin(async move { Ok(result) })
        }

        /// 记录本次调用的命令与限制，然后返回预设的执行结果。
        fn execute_with_limits_and_cancel(
            &self,
            command: SandboxCommand,
            limits: ResourceLimits,
            _cancel: Option<Arc<CancellationToken>>,
        ) -> BoxFuture<'_, std::result::Result<ExecutionResult, ReactError>> {
            *self
                .seen
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()) =
                Some(RecordedCall { command, limits });
            let result = self.result.clone();
            Box::pin(async move { Ok(result) })
        }

        /// 记录调用后返回预设结果，并附带 `actual_isolation`（若未设置则回退为
        /// `isolation`），用于测试实际隔离级别的上报逻辑。
        fn execute_with_isolation_receipt(
            &self,
            command: SandboxCommand,
            limits: ResourceLimits,
            cancel: Option<Arc<CancellationToken>>,
        ) -> BoxFuture<'_, std::result::Result<(ExecutionResult, IsolationLevel), ReactError>>
        {
            Box::pin(async move {
                let result = self
                    .execute_with_limits_and_cancel(command, limits, cancel)
                    .await?;
                Ok((result, self.actual_isolation.unwrap_or(self.isolation)))
            })
        }
    }

    /// 构建一个成功的（退出码 0）测试用执行结果，标准输出为给定字符串。
    fn success_execution(stdout: &str) -> ExecutionResult {
        ExecutionResult {
            exit_code: 0,
            stdout: stdout.to_string(),
            stderr: String::new(),
            duration: Duration::from_millis(1),
            sandbox_type: "recording".to_string(),
            timed_out: false,
            cancelled: false,
            output_truncated: false,
            stdout_bytes: u64::try_from(stdout.len()).unwrap_or(u64::MAX),
            stderr_bytes: 0,
        }
    }

    /// 构建一个测试用 `RecordingSandbox`，`actual_isolation` 默认为 `None`。
    fn recording_sandbox(
        isolation: IsolationLevel,
        available: bool,
        result: ExecutionResult,
    ) -> Arc<RecordingSandbox> {
        Arc::new(RecordingSandbox {
            isolation,
            actual_isolation: None,
            available,
            result,
            seen: std::sync::Mutex::new(None),
        })
    }

    /// 构建内联代码模式的工具调用参数（`language` + `code`）。
    fn code_params(language: &str, code: &str) -> HashMap<String, serde_json::Value> {
        [
            ("language".to_string(), serde_json::json!(language)),
            ("code".to_string(), serde_json::json!(code)),
        ]
        .into_iter()
        .collect()
    }

    /// 构建持久化脚本模式的工具调用参数（`language` + `script_path`）。
    fn script_params(language: &str, script_path: &str) -> HashMap<String, serde_json::Value> {
        [
            ("language".to_string(), serde_json::json!(language)),
            ("script_path".to_string(), serde_json::json!(script_path)),
        ]
        .into_iter()
        .collect()
    }

    #[test]
    fn validate_language_lowercases_and_accepts_known() -> Result<()> {
        assert_eq!(validate_language("Python")?, "python");
        assert_eq!(validate_language("PYTHON")?, "python");
        assert_eq!(validate_language("R")?, "r");
        assert_eq!(validate_language("JavaScript")?, "javascript");
        Ok(())
    }

    #[test]
    fn validate_language_rejects_unknown() {
        // 熔断机制（用户评审补丁 #1）：未知语言在工具层就失败，不会到达沙箱。
        assert!(validate_language("haskell").is_err());
        assert!(validate_language("").is_err());
    }

    #[test]
    fn supported_languages_includes_r() {
        // Sprint 10b 的重点特性：R 是一等公民语言。
        assert!(SUPPORTED_LANGUAGES.contains(&"r"));
        assert!(SUPPORTED_LANGUAGES.contains(&"python"));
    }

    #[tokio::test]
    async fn run_code_without_sandbox_fails_closed() -> Result<()> {
        let result = RunCodeTool::new()
            .execute_with_context(
                code_params("python", "print('unsafe')"),
                &ToolContext::default(),
            )
            .await?;
        assert!(!result.success);
        assert_eq!(
            result.failure.as_ref().map(|failure| failure.category),
            Some(ToolFailureCategory::Unavailable)
        );
        assert!(
            result
                .error
                .as_deref()
                .is_some_and(|error| error.contains("unsandboxed"))
        );
        Ok(())
    }

    #[tokio::test]
    async fn run_code_rejects_process_only_executor() -> Result<()> {
        let sandbox = recording_sandbox(
            IsolationLevel::Process,
            true,
            success_execution("should not run"),
        );
        let result = RunCodeTool::new()
            .with_sandbox(sandbox.clone())
            .execute_with_context(code_params("python", "print(1)"), &ToolContext::default())
            .await?;
        assert_eq!(
            result.failure.as_ref().map(|failure| failure.category),
            Some(ToolFailureCategory::Unavailable)
        );
        assert!(
            sandbox
                .seen
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .is_none()
        );
        Ok(())
    }

    #[tokio::test]
    async fn run_code_binds_working_dir_and_resource_limits() -> Result<()> {
        let sandbox = recording_sandbox(IsolationLevel::OsSandbox, true, success_execution("ok"));
        let captured: Arc<RecordingSandbox> = sandbox.clone();
        let tool = RunCodeTool::new().with_sandbox(sandbox);
        let working_dir = std::path::PathBuf::from("/tmp/eko-data-subagent-xyz");
        let ctx = ToolContext {
            working_dir: Some(working_dir.clone()),
            ..ToolContext::default()
        };
        let result = tool
            .execute_with_context(code_params("Python", "print(1+1)"), &ctx)
            .await?;
        assert!(result.success);

        let seen = captured
            .seen
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
            .ok_or_else(|| ReactError::Other("sandbox call was not recorded".to_string()))?;
        assert_eq!(seen.command.working_dir, Some(working_dir.clone()));
        assert_eq!(seen.command.timeout, Duration::from_secs(60));
        assert_eq!(seen.limits.cpu_time_secs, Some(60));
        assert_eq!(seen.limits.max_output_bytes, Some(DEFAULT_MAX_OUTPUT_BYTES));
        assert!(!seen.limits.network);
        assert!(seen.limits.writable_paths.contains(&working_dir));
        match seen.command.kind {
            CommandKind::Code { language, code } => {
                assert_eq!(language, "python");
                assert_eq!(code, "print(1+1)");
            }
            _ => {
                return Err(ReactError::Other(
                    "expected code sandbox command".to_string(),
                ));
            }
        }
        Ok(())
    }

    #[tokio::test]
    async fn run_code_reports_the_selected_isolation_not_a_composite_maximum() -> Result<()> {
        let sandbox = Arc::new(RecordingSandbox {
            isolation: IsolationLevel::Container,
            actual_isolation: Some(IsolationLevel::OsSandbox),
            available: true,
            result: success_execution("selected local OS sandbox"),
            seen: std::sync::Mutex::new(None),
        });
        let result = RunCodeTool::new()
            .with_sandbox(sandbox)
            .execute_with_context(code_params("python", "print(1)"), &ToolContext::default())
            .await?;

        assert!(result.success);
        assert_eq!(
            result.metadata.get("isolation_level").map(String::as_str),
            Some("os-sandbox")
        );
        Ok(())
    }

    #[tokio::test]
    async fn run_code_executes_persisted_script_directly() -> Result<()> {
        let workspace = tempfile::tempdir()?;
        std::fs::write(workspace.path().join("analysis.py"), "print('reviewed')\n")?;
        let sandbox = recording_sandbox(IsolationLevel::OsSandbox, true, success_execution("ok"));
        let captured = sandbox.clone();
        let ctx = ToolContext {
            working_dir: Some(workspace.path().to_path_buf()),
            ..ToolContext::default()
        };

        let result = RunCodeTool::new()
            .with_sandbox(sandbox)
            .execute_with_context(script_params("python", "analysis.py"), &ctx)
            .await?;
        assert!(result.success);
        assert_eq!(
            result.metadata.get("execution_mode").map(String::as_str),
            Some("script")
        );
        assert_eq!(
            result.metadata.get("script_path").map(String::as_str),
            Some("analysis.py")
        );

        let seen = captured
            .seen
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
            .ok_or_else(|| ReactError::Other("sandbox call was not recorded".to_string()))?;
        match seen.command.kind {
            CommandKind::Program { program, args } => {
                assert_eq!(program, "python3");
                assert_eq!(args, vec!["analysis.py".to_string()]);
            }
            _ => {
                return Err(ReactError::Other(
                    "expected direct program sandbox command".to_string(),
                ));
            }
        }
        Ok(())
    }

    #[tokio::test]
    async fn run_code_uses_caller_resolved_script_profile() -> Result<()> {
        let workspace = tempfile::tempdir()?;
        let runtime = tempfile::tempdir()?;
        std::fs::write(workspace.path().join("分析.py"), "print('reviewed')\n")?;
        let python = runtime.path().join("bin/python");
        let profile = ScriptExecutionProfile::new("analytics:sha256", "python", &python)
            .with_arg_prefix("-I")
            .with_env("PYTHONUTF8", "1")
            .with_read_only_path(runtime.path());
        let sandbox = recording_sandbox(IsolationLevel::OsSandbox, true, success_execution("ok"));
        let captured = sandbox.clone();
        let ctx = ToolContext {
            working_dir: Some(workspace.path().to_path_buf()),
            script_execution_profile: Some(Arc::new(profile)),
            ..ToolContext::default()
        };

        let result = RunCodeTool::new()
            .with_sandbox(sandbox)
            .execute_with_context(script_params("Python", "分析.py"), &ctx)
            .await?;
        assert!(result.success);
        assert_eq!(
            result.metadata.get("script_profile").map(String::as_str),
            Some("analytics:sha256")
        );

        let seen = captured
            .seen
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
            .ok_or_else(|| ReactError::Other("sandbox call was not recorded".to_string()))?;
        assert_eq!(
            seen.command.env.get("PYTHONUTF8").map(String::as_str),
            Some("1")
        );
        assert!(
            seen.limits
                .read_only_paths
                .contains(&runtime.path().to_path_buf())
        );
        match seen.command.kind {
            CommandKind::Program { program, args } => {
                assert_eq!(program, python.to_string_lossy());
                assert_eq!(args, vec!["-I".to_string(), "分析.py".to_string()]);
            }
            _ => {
                return Err(ReactError::Other(
                    "expected resolved program sandbox command".to_string(),
                ));
            }
        }
        Ok(())
    }

    #[tokio::test]
    async fn run_code_rejects_mismatched_script_profile() -> Result<()> {
        let workspace = tempfile::tempdir()?;
        std::fs::write(workspace.path().join("analysis.py"), "print(1)\n")?;
        let profile = ScriptExecutionProfile::new("r-profile", "r", "/usr/bin/Rscript");
        let ctx = ToolContext {
            working_dir: Some(workspace.path().to_path_buf()),
            script_execution_profile: Some(Arc::new(profile)),
            ..ToolContext::default()
        };

        let result = RunCodeTool::new()
            .execute_with_context(script_params("python", "analysis.py"), &ctx)
            .await;
        assert!(result.is_err());
        Ok(())
    }

    #[tokio::test]
    async fn run_code_lazily_resolves_persisted_script_profile() -> Result<()> {
        let workspace = tempfile::tempdir()?;
        std::fs::write(workspace.path().join("analysis.py"), "print(1)\n")?;
        let runtime = tempfile::tempdir()?;
        let profile = Arc::new(ScriptExecutionProfile::new(
            "lazy-profile",
            "python",
            runtime.path().join("python"),
        ));
        let resolver = Arc::new(FixedProfileResolver {
            profile,
            calls: std::sync::atomic::AtomicUsize::new(0),
        });
        let sandbox = recording_sandbox(IsolationLevel::OsSandbox, true, success_execution("ok"));
        let captured = sandbox.clone();
        let tool = RunCodeTool::new().with_sandbox(sandbox);
        assert!(tool.set_script_execution_profile_resolver(resolver.clone()));
        let ctx = ToolContext {
            working_dir: Some(workspace.path().to_path_buf()),
            ..ToolContext::default()
        };

        let result = tool
            .execute_with_context(script_params("python", "analysis.py"), &ctx)
            .await?;
        assert!(result.success);
        assert_eq!(resolver.calls.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert_eq!(
            result.metadata.get("script_profile").map(String::as_str),
            Some("lazy-profile")
        );
        let seen = captured
            .seen
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
            .ok_or_else(|| ReactError::Other("sandbox call was not recorded".to_string()))?;
        match seen.command.kind {
            CommandKind::Program { program, .. } => {
                assert_eq!(program, runtime.path().join("python").to_string_lossy());
            }
            _ => {
                return Err(ReactError::Other(
                    "expected lazily resolved program".to_string(),
                ));
            }
        }
        Ok(())
    }

    #[tokio::test]
    async fn run_code_rejects_ambiguous_or_escaping_script_inputs() -> Result<()> {
        let workspace = tempfile::tempdir()?;
        std::fs::write(workspace.path().join("analysis.py"), "print(1)\n")?;
        let ctx = ToolContext {
            working_dir: Some(workspace.path().to_path_buf()),
            ..ToolContext::default()
        };
        let tool = RunCodeTool::new();

        let mut ambiguous = script_params("python", "analysis.py");
        ambiguous.insert("code".to_string(), serde_json::json!("print(2)"));
        assert!(tool.execute_with_context(ambiguous, &ctx).await.is_err());
        assert!(
            tool.execute_with_context(script_params("python", "../analysis.py"), &ctx)
                .await
                .is_err()
        );
        assert!(
            tool.execute_with_context(script_params("python", "missing.py"), &ctx)
                .await
                .is_err()
        );
        Ok(())
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn run_code_rejects_script_symlink_outside_working_dir() -> Result<()> {
        use std::os::unix::fs::symlink;

        let workspace = tempfile::tempdir()?;
        let outside = tempfile::NamedTempFile::new()?;
        symlink(outside.path(), workspace.path().join("analysis.py"))?;
        let ctx = ToolContext {
            working_dir: Some(workspace.path().to_path_buf()),
            ..ToolContext::default()
        };

        assert!(
            RunCodeTool::new()
                .execute_with_context(script_params("python", "analysis.py"), &ctx)
                .await
                .is_err()
        );
        Ok(())
    }

    #[tokio::test]
    async fn run_code_preserves_timeout_and_cancellation_categories() -> Result<()> {
        let mut timed_out = success_execution("");
        timed_out.exit_code = -1;
        timed_out.timed_out = true;
        let timeout_result = RunCodeTool::new()
            .with_sandbox(recording_sandbox(
                IsolationLevel::OsSandbox,
                true,
                timed_out,
            ))
            .execute_with_context(code_params("r", "print(1)"), &ToolContext::default())
            .await?;
        assert_eq!(
            timeout_result
                .failure
                .as_ref()
                .map(|failure| failure.category),
            Some(ToolFailureCategory::Timeout)
        );

        let mut cancelled = success_execution("");
        cancelled.exit_code = -1;
        cancelled.cancelled = true;
        let cancelled_result = RunCodeTool::new()
            .with_sandbox(recording_sandbox(
                IsolationLevel::OsSandbox,
                true,
                cancelled,
            ))
            .execute_with_context(code_params("python", "print(1)"), &ToolContext::default())
            .await?;
        assert_eq!(
            cancelled_result
                .failure
                .as_ref()
                .map(|failure| failure.category),
            Some(ToolFailureCategory::Cancelled)
        );
        Ok(())
    }

    #[tokio::test]
    async fn run_code_applies_shared_utf8_output_limit() -> Result<()> {
        let result = RunCodeTool::new()
            .with_max_output_bytes(5)
            .with_sandbox(recording_sandbox(
                IsolationLevel::OsSandbox,
                true,
                success_execution("中文abc"),
            ))
            .execute_with_context(
                code_params("python", "print('中文abc')"),
                &ToolContext::default(),
            )
            .await?;
        assert!(result.success);
        assert_eq!(result.output, "中");
        assert!(result.truncated);
        assert_eq!(
            result.metadata.get("output_truncated").map(String::as_str),
            Some("true")
        );
        Ok(())
    }
}
