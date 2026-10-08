/*!
SKILL.md 内容的内联 shell 执行与变量替换。

技能被激活时，其 Markdown 正文中可能包含：

- **内联命令**：`` !`pwd` `` -> 替换为该命令的 stdout
- **块命令**：`` ```! \n git status \n ``` `` -> 替换为 stdout
- **变量**：`${SKILL_DIR}`、`${SESSION_ID}`、`${ARGUMENTS}`、`${1}`、`${2}` 等

这与 Claude Code 的 `promptShellExecution.ts` 行为一致：
命令在**激活时**执行，其输出被内联替换，从而使技能指令具备动态性。

## 安全

- 来自 MCP 的技能**绝不**执行内联命令（不受信的远程内容）
- 本地技能通过 SandboxManager 至少使用 Process 隔离级别
- 每条命令都遵循配置的超时时间（默认每条 10 秒）
- 超时后通过 `Child::kill()` 终止子进程
- 命令以技能所在目录作为工作目录运行
*/

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::sync::LazyLock;
use std::time::Duration;

use regex::Regex;
use tracing::warn;

use crate::sandbox::{SandboxCommand, SandboxExecutor};
use crate::skills::minimal_env;

/// 单条内联命令的默认超时时间。
const DEFAULT_CMD_TIMEOUT: Duration = Duration::from_secs(10);

/// 预编译的块命令语法正则：```! ... ```
static BLOCK_RE: LazyLock<Option<Regex>> =
    LazyLock::new(|| Regex::new(r"```!\s*\n?([\s\S]*?)\n?```").ok());

/// 预编译的内联命令语法正则：!`...`
static INLINE_RE: LazyLock<Option<Regex>> =
    LazyLock::new(|| Regex::new(r"(?:^|\s)!`([^`]+)`").ok());

/// 预编译的块命令标记检测正则。
static BLOCK_MARKER_RE: LazyLock<Option<Regex>> = LazyLock::new(|| Regex::new(r"```!\s*\n?").ok());

/// 预编译的内联命令标记检测正则。
static INLINE_MARKER_RE: LazyLock<Option<Regex>> =
    LazyLock::new(|| Regex::new(r"(?:^|\s)!`[^`]*`").ok());

/// 预编译的 SOH 分隔占位符正则。
static PLACEHOLDER_RE: LazyLock<Option<Regex>> =
    LazyLock::new(|| Regex::new(r"\x01CMD_OUT_\d+\x01").ok());

/// 技能来源（影响安全策略）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkillSource {
    /// 本地文件系统技能（受信）。
    Local,
    /// 远程/MCP 技能（不受信 —— 不执行 shell）。
    Mcp,
}

/// 变量替换与命令执行所需的上下文。
#[derive(Clone)]
pub struct PromptContext {
    /// 技能目录的绝对路径。
    pub skill_dir: String,
    /// 会话标识（如有）。
    pub session_id: String,
    /// 用户提供的参数（按位置）。
    pub arguments: Vec<String>,
    /// 由程序化描述符或宿主策略提供的首选 shell。
    pub shell: Option<String>,
    /// 单条命令的超时时间。
    pub timeout: Duration,
    /// 技能来源（本地或 MCP）。
    pub source: SkillSource,
    /**
    可选的沙箱管理器，用于执行命令。
    为 Some 时，命令将经由沙箱执行。
    */
    pub sandbox: Option<Arc<dyn SandboxExecutor>>,
}

impl Default for PromptContext {
    /// 构造默认上下文：本地来源、默认超时、无沙箱。
    fn default() -> Self {
        Self {
            skill_dir: String::new(),
            session_id: String::new(),
            arguments: Vec::new(),
            shell: None,
            timeout: DEFAULT_CMD_TIMEOUT,
            source: SkillSource::Local,
            sandbox: None,
        }
    }
}

/**
处理技能的 Markdown 正文：先替换变量，再执行内联命令。

# 安全

为防止用户提供的参数注入新命令：
1. 首先从原始内容中提取所有命令区域。
2. 变量替换只作用于非命令文本（占位符保护命令区域免受替换和注入）。
3. 替换完成后，扫描结果中是否出现新的命令标记。若发现，
   则拒绝替换，并返回不执行任何命令的内容。
4. 执行原始命令，并用其输出替换占位符。
*/
pub async fn process_skill_content(content: &str, ctx: &PromptContext) -> String {
    // MCP 技能：完全禁止执行内联命令
    if ctx.source == SkillSource::Mcp {
        return substitute_variables(content, ctx);
    }

    // 第 1 步：从原始内容（替换之前）中提取所有命令区域
    let regions = extract_command_regions(content);

    // 第 2 步：构建安全版本，用唯一占位符替代命令区域
    let mut safe = String::with_capacity(content.len());
    let mut pos = 0;
    for (i, region) in regions.iter().enumerate() {
        safe.push_str(content.get(pos..region.start).unwrap_or_default());
        safe.push_str(&format!("\x01CMD_OUT_{}\x01", i)); // SOH 分隔的占位符
        pos = region.end;
    }
    safe.push_str(content.get(pos..).unwrap_or_default());

    // 第 3 步：仅在安全（非命令）内容中替换变量
    let substituted = substitute_variables(&safe, ctx);

    // 第 4 步：检测变量展开后引入的注入命令标记
    if has_command_markers(&substituted) {
        warn!(
            "Possible command injection detected after variable substitution — \
               command execution disabled for this skill activation"
        );
        // 返回移除占位符后的替换结果（不执行命令）
        return strip_placeholders(&substituted);
    }

    // 第 5 步：执行原始命令并收集输出
    let mut cmd_outputs: HashMap<usize, String> = HashMap::new();
    for (i, region) in regions.iter().enumerate() {
        let output = run_command(&region.command_text, ctx).await;
        cmd_outputs.insert(i, output);
    }

    // 第 6 步：用命令输出替换占位符
    let mut result = substituted;
    for i in (0..regions.len()).rev() {
        let placeholder = format!("\x01CMD_OUT_{}\x01", i);
        if let Some(output) = cmd_outputs.remove(&i) {
            let Some(region) = regions.get(i) else {
                continue;
            };
            if region.is_block {
                // 块命令：输出替换整个 ```! ... ``` 块
                result = result.replace(&placeholder, &output);
            } else {
                // 内联命令：保留原始匹配中的前导空白
                let matched = content.get(region.start..region.end).unwrap_or_default();
                let leading_ws: String =
                    matched.chars().take_while(|c| c.is_whitespace()).collect();
                result = result.replace(&placeholder, &format!("{}{}", leading_ws, output.trim()));
            }
        }
    }

    result
}

/// 技能内容中检测到的命令区域。
struct CmdRegion {
    /// 区域在原始内容中的起始字节偏移。
    start: usize,
    /// 区域在原始内容中的结束字节偏移（不含）。
    end: usize,
    /// 待执行的命令文本。
    command_text: String,
    /// 是否为块命令（否则为内联命令）。
    is_block: bool,
}

/// 按顺序从内容中提取所有命令区域（块命令与内联命令）。
fn extract_command_regions(content: &str) -> Vec<CmdRegion> {
    let mut regions: Vec<CmdRegion> = Vec::new();

    if let Some(block_re) = BLOCK_RE.as_ref() {
        for cap in block_re.captures_iter(content) {
            if let Some(matched) = cap.get(0) {
                regions.push(CmdRegion {
                    start: matched.start(),
                    end: matched.end(),
                    command_text: cap
                        .get(1)
                        .map(|m| m.as_str().trim())
                        .unwrap_or("")
                        .to_string(),
                    is_block: true,
                });
            }
        }
    }
    if let Some(inline_re) = INLINE_RE.as_ref() {
        for cap in inline_re.captures_iter(content) {
            if let Some(matched) = cap.get(0) {
                regions.push(CmdRegion {
                    start: matched.start(),
                    end: matched.end(),
                    command_text: cap
                        .get(1)
                        .map(|m| m.as_str().trim())
                        .unwrap_or("")
                        .to_string(),
                    is_block: false,
                });
            }
        }
    }

    regions.sort_by_key(|r| r.start);
    regions
}

/// 检查内容中是否包含块命令或内联命令标记。
fn has_command_markers(content: &str) -> bool {
    BLOCK_MARKER_RE
        .as_ref()
        .is_some_and(|regex| regex.is_match(content))
        || INLINE_MARKER_RE
            .as_ref()
            .is_some_and(|regex| regex.is_match(content))
}

/// 移除内容中的占位符标记。
fn strip_placeholders(content: &str) -> String {
    PLACEHOLDER_RE.as_ref().map_or_else(
        || content.to_string(),
        |regex| regex.replace_all(content, "").to_string(),
    )
}

/**
替换技能内容中的模板变量。

支持的变量：
- `${SKILL_DIR}` -- 技能目录的绝对路径
- `${SESSION_ID}` -- 当前会话标识
- `${ARGUMENTS}` -- 以空格连接的全部参数
- `${1}`、`${2}`、... -- 位置参数
*/
fn substitute_variables(content: &str, ctx: &PromptContext) -> String {
    let mut result = content.to_string();

    result = result.replace("${SKILL_DIR}", &ctx.skill_dir);
    result = result.replace("${CLAUDE_SKILL_DIR}", &ctx.skill_dir);

    result = result.replace("${SESSION_ID}", &ctx.session_id);
    result = result.replace("${CLAUDE_SESSION_ID}", &ctx.session_id);

    let args_joined = ctx.arguments.join(" ");
    result = result.replace("${ARGUMENTS}", &args_joined);

    // 位置参数：${1}、${2}、...
    for (i, arg) in ctx.arguments.iter().enumerate() {
        let placeholder = format!("${{{}}}", i + 1);
        result = result.replace(&placeholder, arg);
    }

    result
}

/// 执行块命令：` ```! \n command \n ``` ` -> stdout
#[cfg(test)]
async fn execute_block_commands(content: &str, ctx: &PromptContext) -> String {
    // 模式：```! 后跟可选换行，然后是命令，最后是 ```
    let mut result = content.to_string();
    let matches: Vec<_> = BLOCK_RE
        .as_ref()
        .map(|regex| regex.captures_iter(content).collect())
        .unwrap_or_default();

    // 逆序处理，保证字节偏移仍然有效
    for cap in matches.into_iter().rev() {
        let full_match = match cap.get(0) {
            Some(m) => m,
            None => continue,
        };
        let command = cap.get(1).map(|m| m.as_str().trim()).unwrap_or("");

        if command.is_empty() {
            continue;
        }

        let output = run_command(command, ctx).await;

        result.replace_range(full_match.range(), &output);
    }

    result
}

/**
执行内联命令：`` !`command` `` -> stdout

该模式要求 `!`` 之前为空白或行首，以避免误匹配
Markdown 图片语法（`![alt](url)`）。
*/
#[cfg(test)]
async fn execute_inline_commands(content: &str, ctx: &PromptContext) -> String {
    let mut result = content.to_string();
    let matches: Vec<_> = INLINE_RE
        .as_ref()
        .map(|regex| regex.captures_iter(content).collect())
        .unwrap_or_default();

    for cap in matches.into_iter().rev() {
        let full_match = match cap.get(0) {
            Some(m) => m,
            None => continue,
        };
        let command = cap.get(1).map(|m| m.as_str().trim()).unwrap_or("");

        if command.is_empty() {
            continue;
        }

        let output = run_command(command, ctx).await;

        // 保留匹配中的前导空白
        let matched_str = full_match.as_str();
        let leading_ws: String = matched_str
            .chars()
            .take_while(|c| c.is_whitespace())
            .collect();
        let replacement = format!("{}{}", leading_ws, output.trim());

        result.replace_range(full_match.range(), &replacement);
    }

    result
}

/**
执行 shell 命令并返回其 stdout（或错误信息）。

沙箱路由：
- 若配置了 SandboxManager，命令经由沙箱执行，且至少使用 Process 隔离级别。
- 若无沙箱，则回退为直接派生进程，使用最小化环境变量，
  并具备正确的超时与终止行为。
*/
async fn run_command(command: &str, ctx: &PromptContext) -> String {
    // 沙箱执行路径
    if let Some(ref manager) = ctx.sandbox {
        return run_command_sandboxed(command, ctx, manager.as_ref()).await;
    }

    // 回退：以最小化环境直接执行进程，并正确终止
    run_command_direct(command, ctx).await
}

/// 通过 SandboxManager 执行命令。
async fn run_command_sandboxed(
    command: &str,
    ctx: &PromptContext,
    manager: &dyn SandboxExecutor,
) -> String {
    // 使用最小化环境
    let env = minimal_env(&ctx.skill_dir, &ctx.session_id, HashMap::new());

    let mut sandbox_cmd = SandboxCommand::shell(command).with_timeout(ctx.timeout);

    if !ctx.skill_dir.is_empty() && Path::new(&ctx.skill_dir).exists() {
        sandbox_cmd = sandbox_cmd.with_working_dir(&ctx.skill_dir);
    }

    for (k, v) in env {
        sandbox_cmd = sandbox_cmd.with_env(k, v);
    }

    match manager.execute(sandbox_cmd).await {
        Ok(result) => {
            if result.success() {
                result.stdout.trim().to_string()
            } else {
                let stderr = result.stderr.trim();
                if !stderr.is_empty() {
                    warn!(
                        command = command,
                        exit_code = result.exit_code,
                        stderr = %stderr,
                        "Inline skill command failed (sandboxed)"
                    );
                    format!("[error: {}]", stderr)
                } else {
                    format!("[error: exit code {}]", result.exit_code)
                }
            }
        }
        Err(e) => {
            warn!(command = command, error = %e, "Inline skill command sandbox error");
            format!("[sandbox error: {}]", e)
        }
    }
}

/**
使用最小化环境直接执行命令。

# ⚠️ 警告：Fallback 路径的限制

此函数是 `run_command` 的 fallback 路径，仅在无 `SandboxManager` 时使用。
它有以下限制：

direct fallback 会使用 `kill_on_drop(true)` 配合 `timeout(...)` 尽力终止超时子进程，
但生产环境仍建议优先配置 `SandboxManager`（通过 `PromptContext::sandbox` 字段），
以获得更强的隔离、资源限制和统一清理。

**适用场景**：仅建议用于简单 demo、测试、或无沙箱依赖的开发环境。

**推荐做法**：在 `PromptContext` 中设置 `sandbox = Some(Arc::new(SandboxManager::auto_detect()))`
以启用沙箱执行路径，确保超时后进程被正确终止。
*/
async fn run_command_direct(command: &str, ctx: &PromptContext) -> String {
    /* 安全闸门：技能的 shell 块是来自 SKILL.md 文件的任意代码。
    在无沙箱（默认）运行时，使用与 ShellTool 相同的严格策略进行校验，
    防止本地技能借助 `!` 块执行 `rm -rf` 等破坏性命令。
    需要此类命令的技能应改为经由沙箱执行。 */
    if let Err(reason) = check_skill_command_safety(command) {
        warn!(command = command, reason = %reason, "Skill shell command rejected by safety policy");
        return format!("[error: command rejected by safety policy: {reason}]");
    }

    let shell_cmd = build_shell_command(command, ctx);

    let mut cmd = tokio::process::Command::new(&shell_cmd.program);
    for arg in &shell_cmd.args {
        cmd.arg(arg);
    }
    cmd.kill_on_drop(true);

    if !ctx.skill_dir.is_empty() && Path::new(&ctx.skill_dir).exists() {
        cmd.current_dir(&ctx.skill_dir);
    }

    // 使用最小化环境，而非继承全部环境变量
    set_minimal_cmd_env(&mut cmd, ctx);

    // 使用 output() 捕获 stdout/stderr
    match tokio::time::timeout(ctx.timeout, cmd.output()).await {
        Ok(Ok(output)) => {
            let stdout = String::from_utf8_lossy(&output.stdout).to_string();
            let stderr = String::from_utf8_lossy(&output.stderr).to_string();

            if !output.status.success() {
                warn!(
                    command = command,
                    exit_code = output.status.code(),
                    stderr = %stderr.trim(),
                    "Inline skill command failed"
                );
                if !stderr.is_empty() {
                    format!("[error: {}]", stderr.trim())
                } else {
                    format!("[error: exit code {}]", output.status.code().unwrap_or(-1))
                }
            } else {
                stdout.trim().to_string()
            }
        }
        Ok(Err(e)) => {
            warn!(command = command, error = %e, "Inline skill command execution error");
            format!("[error: {}]", e)
        }
        Err(_) => {
            warn!(
                command = command,
                timeout_secs = ctx.timeout.as_secs(),
                "Inline skill command timed out"
            );
            format!("[timeout after {}s]", ctx.timeout.as_secs())
        }
    }
}

/// 待派生的 shell 程序及其参数。
struct ShellCommand {
    /// 可执行程序名或路径。
    program: String,
    /// 传给程序的参数列表。
    args: Vec<String>,
}

/// 根据上下文中的 shell 偏好与当前平台构建 shell 调用命令。
fn build_shell_command(command: &str, ctx: &PromptContext) -> ShellCommand {
    let shell_pref = ctx.shell.as_deref().unwrap_or("bash");

    if shell_pref == "powershell" {
        let program = if which_exists("pwsh") {
            "pwsh"
        } else if cfg!(target_os = "windows") {
            "powershell"
        } else {
            "sh"
        };

        if program == "powershell" || program == "pwsh" {
            ShellCommand {
                program: program.to_string(),
                args: vec![
                    "-NoProfile".to_string(),
                    "-NonInteractive".to_string(),
                    "-Command".to_string(),
                    command.to_string(),
                ],
            }
        } else {
            ShellCommand {
                program: "sh".to_string(),
                args: vec!["-c".to_string(), command.to_string()],
            }
        }
    } else if cfg!(target_os = "windows") {
        // Windows：优先尝试 Git Bash
        if let Some(bash) = find_git_bash_path() {
            ShellCommand {
                program: bash,
                args: vec!["-c".to_string(), command.to_string()],
            }
        } else {
            ShellCommand {
                program: "cmd".to_string(),
                args: vec!["/C".to_string(), command.to_string()],
            }
        }
    } else {
        ShellCommand {
            program: "bash".to_string(),
            args: vec!["-c".to_string(), command.to_string()],
        }
    }
}

/// 通过运行 `<cmd> --version` 判断命令是否可用。
fn which_exists(cmd: &str) -> bool {
    std::process::Command::new(cmd)
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

/// 在 Windows 上查找 Git Bash 的路径，找不到时回退为 PATH 中的 `bash`。
#[cfg(target_os = "windows")]
pub fn find_git_bash_path() -> Option<String> {
    let candidates = [
        std::env::var("ProgramFiles")
            .ok()
            .map(|p| format!("{}\\Git\\bin\\bash.exe", p)),
        Some(r"C:\Program Files\Git\bin\bash.exe".to_string()),
    ];
    for candidate in candidates.into_iter().flatten() {
        if std::path::Path::new(&candidate).exists() {
            return Some(candidate);
        }
    }
    if which_exists("bash") {
        return Some("bash".to_string());
    }
    None
}

/// 非 Windows 平台不需要 Git Bash，始终返回 `None`。
#[cfg(not(target_os = "windows"))]
pub fn find_git_bash_path() -> Option<String> {
    None
}

/**
为子进程设置最小化环境变量。
仅传递清理后的 PATH、SKILL_DIR 与 SESSION_ID。
*/
fn set_minimal_cmd_env(cmd: &mut tokio::process::Command, ctx: &PromptContext) {
    cmd.env_clear();
    if let Ok(path) = std::env::var("PATH") {
        cmd.env("PATH", path);
    }
    if !ctx.skill_dir.is_empty() {
        cmd.env("SKILL_DIR", &ctx.skill_dir);
    }
    if !ctx.session_id.is_empty() {
        cmd.env("SESSION_ID", &ctx.session_id);
    }
}

/**
检查技能的 shell 命令能否在沙箱之外运行。

技能的 `!` 块是来自 SKILL.md 文件（本地文件，或将来远程获取）的任意代码。
未配置沙箱时，此函数使用执行层的保守闸门校验命令，
防止技能借助内联 shell 块执行破坏性命令（`rm -rf`、任意元字符等）。
需要更丰富语法的技能必须使用沙箱。

命令安全时返回 `Ok(())`，否则返回描述拒绝原因的 `Err(reason)`。
*/
fn check_skill_command_safety(command: &str) -> std::result::Result<(), String> {
    const SHELL_META: &[char] = &['|', ';', '&', '$', '`', '>', '<', '(', ')', '\n', '\r'];
    if command
        .chars()
        .any(|character| SHELL_META.contains(&character))
    {
        return Err(
            "Skill command contains shell metacharacters; sandbox execution required.".to_string(),
        );
    }
    let base = command.split_whitespace().next().unwrap_or("");
    let base = base.rsplit(['/', '\\']).next().unwrap_or(base);
    if matches!(
        base,
        "rm" | "rmdir" | "mkfs" | "dd" | "shutdown" | "reboot" | "halt" | "poweroff"
    ) {
        return Err(format!(
            "Skill command '{base}' is destructive; sandbox execution required"
        ));
    }
    Ok(())
}

// -- 测试 --

#[cfg(test)]
mod tests {
    use super::*;

    /// 构造测试用的本地技能上下文。
    fn test_ctx() -> PromptContext {
        PromptContext {
            skill_dir: "/tmp/test-skill".into(),
            session_id: "session-abc123".into(),
            arguments: vec!["arg1".into(), "arg2".into()],
            shell: None,
            timeout: Duration::from_secs(5),
            source: SkillSource::Local,
            sandbox: None,
        }
    }

    /// 验证 SKILL_DIR、SESSION_ID、ARGUMENTS 与位置参数变量均被正确替换。
    #[test]
    fn test_substitute_variables() {
        let ctx = test_ctx();
        let content = "Dir: ${SKILL_DIR}\nSession: ${SESSION_ID}\nArgs: ${ARGUMENTS}\nFirst: ${1}";
        let result = substitute_variables(content, &ctx);
        assert_eq!(
            result,
            "Dir: /tmp/test-skill\nSession: session-abc123\nArgs: arg1 arg2\nFirst: arg1"
        );
    }

    /// 验证兼容 Claude Code 的 CLAUDE_SKILL_DIR 与 CLAUDE_SESSION_ID 变量被正确替换。
    #[test]
    fn test_substitute_claude_compat_vars() {
        let ctx = test_ctx();
        let content = "${CLAUDE_SKILL_DIR}/scripts/run.py ${CLAUDE_SESSION_ID}";
        let result = substitute_variables(content, &ctx);
        assert_eq!(result, "/tmp/test-skill/scripts/run.py session-abc123");
    }

    /// 验证无参数时 ARGUMENTS 替换为空、未提供的位置参数保持原样。
    #[test]
    fn test_substitute_no_args() {
        let ctx = PromptContext {
            arguments: vec![],
            ..test_ctx()
        };
        let content = "No args: ${ARGUMENTS} and ${1}";
        let result = substitute_variables(content, &ctx);
        assert_eq!(result, "No args:  and ${1}");
    }

    /// 验证块命令被执行且其输出替换整个块。
    #[tokio::test]
    async fn test_block_command_execution() {
        if cfg!(target_os = "windows") {
            return;
        }
        let ctx = test_ctx();
        let content = "Before\n```!\necho hello-world\n```\nAfter";
        let result = execute_block_commands(content, &ctx).await;
        assert!(result.contains("hello-world"), "got: {}", result);
        assert!(result.contains("Before"));
        assert!(result.contains("After"));
        assert!(!result.contains("```!"));
    }

    /// 验证内联命令被执行且其输出替换命令语法。
    #[tokio::test]
    async fn test_inline_command_execution() {
        if cfg!(target_os = "windows") {
            return;
        }
        let ctx = test_ctx();
        let content = "Current dir is !`echo /test/path`";
        let result = execute_inline_commands(content, &ctx).await;
        assert!(result.contains("/test/path"), "got: {}", result);
        assert!(!result.contains("!`"));
    }

    /// 验证 MCP 来源的技能不会执行内联命令。
    #[tokio::test]
    async fn test_mcp_source_skips_execution() {
        let ctx = PromptContext {
            source: SkillSource::Mcp,
            ..test_ctx()
        };
        let content = "Run !`echo dangerous` here";
        let result = process_skill_content(content, &ctx).await;
        assert!(
            result.contains("!`echo dangerous`"),
            "MCP skill should not execute commands: {}",
            result
        );
    }

    /// 验证完整处理流程同时完成变量替换与命令执行。
    #[tokio::test]
    async fn test_full_processing() {
        if cfg!(target_os = "windows") {
            return;
        }
        let ctx = test_ctx();
        let content = "Skill at ${SKILL_DIR}\nVersion: !`echo 1.0.0`";
        let result = process_skill_content(content, &ctx).await;
        assert!(result.contains("/tmp/test-skill"));
        assert!(result.contains("1.0.0"));
    }

    /// 验证内联命令不会继承未列入白名单的环境变量。
    #[tokio::test]
    async fn test_inline_command_does_not_inherit_unlisted_env() {
        if cfg!(target_os = "windows") {
            return;
        }

        let key = "ECHO_EXECUTION_TEST_SECRET";
        let secret = "super-secret-value";
        // 仅用于测试的进程环境变量修改，用于验证 env_clear() 行为。
        unsafe {
            std::env::set_var(key, secret);
        }

        let ctx = test_ctx();
        let content = format!("secret=!`printf %s \"${key}\"`");
        let result = execute_inline_commands(&content, &ctx).await;

        assert!(
            !result.contains(secret),
            "unexpected leaked env in: {}",
            result
        );

        unsafe {
            std::env::remove_var(key);
        }
    }

    // ── 注入防护测试 ──────────────────────────────────────────

    /// 验证参数中的内联命令语法不会被执行。
    #[tokio::test]
    async fn test_injection_via_arguments_inline() {
        if cfg!(target_os = "windows") {
            return;
        }
        // 包含内联命令语法的用户参数不应被执行
        let ctx = PromptContext {
            arguments: vec!["!`echo injected`".into()],
            ..test_ctx()
        };
        let content = "Args: ${ARGUMENTS}";
        let result = process_skill_content(content, &ctx).await;
        // 完整的命令语法必须保持字面量（不被执行、不被输出替换）
        assert!(
            result.contains("!`echo injected`"),
            "Injected inline command should remain as literal text, got: {}",
            result
        );
    }

    /// 验证参数中的块命令语法不会被执行。
    #[tokio::test]
    async fn test_injection_via_arguments_block() {
        if cfg!(target_os = "windows") {
            return;
        }
        // 包含块命令语法的用户参数不应被执行
        let ctx = PromptContext {
            arguments: vec!["```!\necho injected\n```".into()],
            ..test_ctx()
        };
        let content = "Args: ${ARGUMENTS}";
        let result = process_skill_content(content, &ctx).await;
        // 块命令标记必须保持为字面文本
        assert!(
            result.contains("```!"),
            "Injected block command marker should remain literal, got: {}",
            result
        );
    }

    /// 验证位置参数中注入的命令不会被执行。
    #[tokio::test]
    async fn test_injection_via_positional_arg() {
        if cfg!(target_os = "windows") {
            return;
        }
        let ctx = PromptContext {
            arguments: vec!["!`touch /tmp/evil`".into()],
            ..test_ctx()
        };
        let content = "Arg1: ${1}";
        let result = process_skill_content(content, &ctx).await;
        assert!(
            result.contains("!`touch /tmp/evil`"),
            "Injected command should remain literal, got: {}",
            result
        );
    }

    /// 验证变量在文本区域中仍会展开，且原始命令仍会执行。
    #[tokio::test]
    async fn test_variables_still_expand_in_text_regions() {
        if cfg!(target_os = "windows") {
            return;
        }
        let ctx = PromptContext {
            arguments: vec!["world".into()],
            ..test_ctx()
        };
        // 命令位于原始内容中，变量位于文本区域
        let content = "Hello ${1}! Status: !`echo ok`";
        let result = process_skill_content(content, &ctx).await;
        assert!(
            result.contains("Hello world!"),
            "Variable should expand: {}",
            result
        );
        assert!(
            result.contains("ok"),
            "Command should still execute: {}",
            result
        );
    }

    /// 验证不含命令语法的安全参数可正常替换。
    #[tokio::test]
    async fn test_injection_safe_argument_no_command() {
        // 安全参数应正常工作
        let ctx = PromptContext {
            arguments: vec!["safe-value".into()],
            ..test_ctx()
        };
        let content = "Arg: ${1}";
        let result = process_skill_content(content, &ctx).await;
        assert_eq!(result, "Arg: safe-value");
    }
}
