/*!
技能（Skill）系统核心 trait 与类型

技能是更高层级的能力单元，将相关工具与可选的系统提示词注入打包在一起。
本模块定义了核心 [`Skill`] trait 以及共享工具函数。
*/

use super::Tool;
use crate::sandbox::SandboxExecutor;
use std::collections::HashMap;
use std::path::{Component, Path};
use std::sync::Arc;

/**
Agent 技能 —— 将相关工具与可选系统提示词注入打包在一起的高层能力单元。

从 `SKILL.md` 加载的基于文件的技能，请参见
`echo_execution::skills::external::SkillDescriptor` 与
`echo_execution::skills::registry::SkillRegistry`。

# Skill 与 Tool 对比

| 维度 | Tool | Skill |
|-----------|------|-------|
| 粒度 | 单个原子操作 | 领域能力（多工具 + 提示词） |
| 注册方式 | `agent.add_tool(box)` | `agent.add_skill(box)` |
| 提示词 | 无 | 可选的指导注入 |
| 语义 | “做一件事” | “我精通一个领域” |
*/
pub trait Skill: Send + Sync {
    /// 技能唯一标识（小写，例如 `"calculator"`）。
    fn name(&self) -> &str;

    /// 人类可读的描述。
    fn description(&self) -> &str;

    /**
    该技能提供的工具。

    每次调用都应返回全新的 `Box<dyn Tool>` 实例。
    */
    fn tools(&self) -> Vec<Box<dyn Tool>>;

    /**
    该技能提供的工具，可选择接入具备沙箱能力的执行器。

    大多数技能可忽略沙箱并回退到 `tools()`。执行代码或 shell 命令的技能
    可覆盖此方法以挂载执行器。
    */
    fn tools_with_sandbox(&self, _sandbox: Option<Arc<dyn SandboxExecutor>>) -> Vec<Box<dyn Tool>> {
        self.tools()
    }

    /// 安装该技能时追加到 Agent 系统提示词中的可选文本。
    fn system_prompt_injection(&self) -> Option<String> {
        None
    }
}

/// 已安装的代码型技能的元数据快照。
#[derive(Debug, Clone)]
pub struct SkillInfo {
    /// 技能名称
    pub name: String,
    /// 技能描述
    pub description: String,
    /// 技能提供的工具名称列表
    pub tool_names: Vec<String>,
    /// 是否包含系统提示词注入
    pub has_prompt_injection: bool,
}

// -- 共享安全工具函数 --

/**
检查 `sub` 是否为 `base` 下的安全子路径。

使用 `Path::components` 检测 `ParentDir`（".."）组件，
比字符串级别的 `contains("..")` 检查更健壮。
若存在任一 `ParentDir` 组件、`sub` 为绝对路径，
或任一侧无法规范化为位于 `base` 之下的路径，则返回 `false`。
*/
pub fn is_path_safe(base: &Path, sub: &Path) -> bool {
    // 拒绝相对路径中的任何 ParentDir 组件
    for component in sub.components() {
        if matches!(component, Component::ParentDir) {
            return false;
        }
    }

    if sub.is_absolute() {
        return false;
    }

    // 校验解析后的路径仍位于 base 之下
    if let (Ok(canonical_base), Ok(canonical_sub)) =
        (base.canonicalize(), base.join(sub).canonicalize())
    {
        canonical_sub.starts_with(&canonical_base)
    } else {
        false
    }
}

/**
返回子进程执行所需的最小环境变量集合。

仅包含（清理后的）PATH 与显式传入的变量。

调用方必须先配合 `Command::env_clear()` 再添加返回的条目；
本函数本身只负责构造白名单映射。
*/
pub fn minimal_env(
    skill_dir: &str,
    session_id: &str,
    extra: HashMap<String, String>,
) -> HashMap<String, String> {
    let mut env = HashMap::new();

    // 仅传递清理后的 PATH（不继承 PYTHONPATH 等变量）
    if let Ok(path) = std::env::var("PATH") {
        env.insert("PATH".to_string(), path);
    }

    /* 转发安全、非机密的功能性变量，使技能脚本在清空环境（env_clear）后仍能正常运行（P1-12）：
    LANG/LC_ALL 保证正确的 UTF-8 处理（否则 C locale 会破坏非 ASCII 字符），
    TMPDIR 供写临时文件的工具使用，TZ 保证时间戳正确，HOME 用于 ~/.config 等。 */
    for var in ["LANG", "LC_ALL", "TMPDIR", "TZ", "HOME"] {
        if let Ok(val) = std::env::var(var) {
            env.insert(var.to_string(), val);
        }
    }

    // 技能专属变量
    env.insert("SKILL_DIR".to_string(), skill_dir.to_string());
    env.insert("SESSION_ID".to_string(), session_id.to_string());

    // 调用方提供的额外变量
    for (k, v) in extra {
        env.insert(k, v);
    }

    env
}

/**
返回 hook 执行所需的最小环境变量（仅技能目录 + 会话）。

与 [`minimal_env`] 一样，调用方应在应用该白名单前清空继承的环境变量。
*/
pub fn minimal_hook_env(skill_dir: &str, session_id: &str) -> HashMap<String, String> {
    let mut env = HashMap::new();
    if let Ok(path) = std::env::var("PATH") {
        env.insert("PATH".to_string(), path);
    }
    // 与 minimal_env 相同的安全功能性变量（P1-12）。
    for var in ["LANG", "LC_ALL", "TMPDIR", "TZ"] {
        if let Ok(val) = std::env::var(var) {
            env.insert(var.to_string(), val);
        }
    }
    env.insert("SKILL_DIR".to_string(), skill_dir.to_string());
    env.insert("SESSION_ID".to_string(), session_id.to_string());
    env
}

/**
返回带额外生命周期上下文的 hook 最小环境变量。

在 [`minimal_hook_env`] 基础上增加 `HOOK_EVENT` 与 `CWD` 变量，
供需要知道触发来源的生命周期 hook 使用。
*/
pub fn minimal_hook_env_with_context(
    skill_dir: &str,
    session_id: &str,
    hook_event: &str,
    cwd: &str,
) -> HashMap<String, String> {
    let mut env = minimal_hook_env(skill_dir, session_id);
    env.insert("HOOK_EVENT".to_string(), hook_event.to_string());
    env.insert("CWD".to_string(), cwd.to_string());
    env
}

// 单元测试：skill 工具的数据结构与辅助函数。
#[cfg(test)]
mod tests {
    use super::*;

    /// 验证 `SkillInfo` 结构体各字段按预期构造。
    #[test]
    fn test_skill_info_structure() {
        let info = SkillInfo {
            name: "calculator".to_string(),
            description: "Performs calculations".to_string(),
            tool_names: vec!["add".to_string(), "subtract".to_string()],
            has_prompt_injection: true,
        };

        assert_eq!(info.name, "calculator");
        assert_eq!(info.description, "Performs calculations");
        assert_eq!(info.tool_names.len(), 2);
        assert!(info.has_prompt_injection);
    }

    /// 验证路径安全检查能拒绝越界路径、接受正常路径。
    #[test]
    fn test_is_path_safe_basic() {
        let base =
            std::env::temp_dir().join(format!("echo-core-skill-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(base.join("scripts")).unwrap();
        std::fs::write(base.join("scripts/run.py"), "print('ok')").unwrap();
        std::fs::write(base.join("README.md"), "docs").unwrap();

        assert!(is_path_safe(&base, Path::new("scripts/run.py")));
        assert!(is_path_safe(&base, Path::new("README.md")));
        assert!(!is_path_safe(&base, Path::new("../secret.txt")));
        assert!(!is_path_safe(&base, Path::new("foo/../../escape")));
        assert!(!is_path_safe(&base, Path::new("/etc/passwd")));
        assert!(!is_path_safe(&base, Path::new("missing.py")));

        let _ = std::fs::remove_dir_all(&base);
    }

    /// 验证最小化环境变量集合包含预期的键。
    #[test]
    fn test_minimal_env_contains_expected_keys() {
        let env = minimal_env("/tmp/skill", "sess-1", HashMap::new());
        assert!(env.contains_key("PATH") || env.is_empty()); // PATH 可能不存在
        assert_eq!(env.get("SKILL_DIR").unwrap(), "/tmp/skill");
        assert_eq!(env.get("SESSION_ID").unwrap(), "sess-1");
        assert!(
            env.contains_key("HOME"),
            "HOME must be present for local desktop agent scripts"
        );
    }
}
