//! Skill 系统 —— 对齐 agentskills.io（核心部分从 echo_core + echo_execution 重新导出）
//!
//! 支持两种 Skill：
//!
//! | 类型 | 注册方式 | 渐进式披露 |
//! |------|-------------|----------------------|
//! | **代码型** | `agent.add_skill(Box::new(MySkill))` | 即时（工具与提示词立刻注入） |
//! | **文件型** | `agent.discover_skills(path)` | 三层式（目录 → 激活 → 资源） |
//!
//! 文件型 Skill 遵循 [agentskills.io 规范](https://agentskills.io/specification)。

/// Echo Agent 框架提供的内置 Skill（从 echo_execution 重新导出）。
pub mod builtin {
    #[cfg(feature = "files")]
    pub use echo_tools::skills::FileSystemSkill;
    #[cfg(feature = "shell")]
    pub use echo_tools::skills::ShellSkill;
}

/// 文件型 Skill 的依赖声明与本地可用性探测。
pub mod dependency_probe {
    pub use echo_execution::skills::dependency_probe::*;
}

/// 文件型外部 Skill 加载（从 echo_execution 重新导出）。
pub mod external {
    pub use echo_execution::skills::external::*;
}

/// Skill 钩子（hook）系统（从 echo_execution 重新导出）。
pub mod hooks {
    pub use echo_execution::skills::hooks::*;
}

/// Skill 注册表（从 echo_execution 重新导出）。
pub mod registry {
    pub use echo_execution::skills::registry::*;
}

// 核心 Skill trait + SkillInfo（定义于 echo_core::tools::skill）
pub use echo_core::tools::skill::{
    Skill, SkillInfo, is_path_safe, minimal_env, minimal_hook_env, minimal_hook_env_with_context,
};

// 执行层类型（SkillRegistry 等）
pub use echo_execution::skills::{SkillActivationHandle, SkillRegistry};
