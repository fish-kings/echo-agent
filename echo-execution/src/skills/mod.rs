/*! Skill 系统 —— 与 agentskills.io 规范对齐

两种类型的 skill：

| 类型 | 注册方式 | 渐进式加载 |
|------|-------------|----------------------|
| **代码型** | `agent.add_skill(Box::new(MySkill))` | 即时加载（工具与提示词立刻注入） |
| **文件型** | `agent.discover_skills(path)` | 三层加载（目录 -> 激活 -> 资源） |

文件型 skill 遵循 [agentskills.io 规范](https://agentskills.io/specification)。
*/

pub mod dependency_probe;
pub mod external;
pub mod hooks;
pub mod registry;

// -- 便捷重导出 --

pub use echo_core::error::{ReactError, Result, ToolError};
pub use echo_core::tools::skill::{
    Skill, SkillInfo, is_path_safe, minimal_env, minimal_hook_env, minimal_hook_env_with_context,
};

// -- 重导出 --

pub use registry::{SkillActivationHandle, SkillRegistry};
