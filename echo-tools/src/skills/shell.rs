use std::sync::Arc;

use echo_core::sandbox::SandboxExecutor;
use echo_core::tools::Tool;
use echo_core::tools::skill::Skill;

use crate::shell::ShellTool;

/// 受控的本地命令执行技能。
pub struct ShellSkill {
    /// 是否使用宽松的命令策略。
    permissive: bool,
}

impl ShellSkill {
    /// 创建使用默认（严格）命令策略的 shell 技能。
    pub fn new() -> Self {
        Self { permissive: false }
    }

    /// 创建使用宽松命令策略的 shell 技能。
    pub fn permissive() -> Self {
        Self { permissive: true }
    }
}

impl Default for ShellSkill {
    /// 等同于 [`ShellSkill::new`]。
    fn default() -> Self {
        Self::new()
    }
}

impl Skill for ShellSkill {
    /// 返回技能名称 `shell`。
    fn name(&self) -> &str {
        "shell"
    }

    /// 返回技能描述。
    fn description(&self) -> &str {
        "Controlled local command execution"
    }

    /// 返回按策略构建的 shell 工具。
    fn tools(&self) -> Vec<Box<dyn Tool>> {
        let tool = if self.permissive {
            ShellTool::new_permissive()
        } else {
            ShellTool::new()
        };
        vec![Box::new(tool)]
    }

    /// 返回 shell 工具；若提供沙箱则让其在沙箱中执行。
    fn tools_with_sandbox(&self, sandbox: Option<Arc<dyn SandboxExecutor>>) -> Vec<Box<dyn Tool>> {
        let mut tool = if self.permissive {
            ShellTool::new_permissive()
        } else {
            ShellTool::new()
        };
        if let Some(sandbox) = sandbox {
            tool = tool.with_sandbox(sandbox);
        }
        vec![Box::new(tool)]
    }

    /// 返回注入系统提示词的 shell 能力说明。
    fn system_prompt_injection(&self) -> Option<String> {
        Some("\n\n## Shell Capability\nUse the `shell` tool for local commands. The installed command policy decides whether a command is allowed, needs approval, or is denied.".to_string())
    }
}
