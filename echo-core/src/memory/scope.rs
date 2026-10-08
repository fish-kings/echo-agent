/*!
记忆作用域 —— 为不同的记忆生命周期分层。

# 作用域

- **User**：跨项目的用户偏好与模式。
- **Project**：特定项目的约定与规则。
- **Repo**：特定仓库的模式（嵌套于项目之内）。
- **Task**：当前任务状态（在任务完成前始终有效）。
- **Session**：当前会话上下文（在 agent 重启前始终有效）。
- **Run**：单次执行运行（每次运行结束后清空）。
*/

use serde::{Deserialize, Serialize};

/// 记忆的生命周期层级。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryScope {
    /// 跨项目的用户偏好（永久持久化）。
    User,
    /// 特定项目的约定（随项目持久化）。
    Project,
    /// 特定仓库的模式（嵌套于项目之内）。
    Repo,
    /// 当前任务状态（任务完成后清空）。
    Task,
    /// 当前会话上下文（agent 重启后清空）。
    Session,
    /// 单次执行运行（每次运行结束后清空）。
    Run,
}

impl MemoryScope {
    /// 返回人类可读的名称。
    pub fn name(&self) -> &str {
        match self {
            Self::User => "user",
            Self::Project => "project",
            Self::Repo => "repo",
            Self::Task => "task",
            Self::Session => "session",
            Self::Run => "run",
        }
    }

    /// 优先级顺序（数值越小，生命周期越长）。
    pub fn priority(&self) -> u8 {
        match self {
            Self::User => 0,
            Self::Project => 1,
            Self::Repo => 2,
            Self::Task => 3,
            Self::Session => 4,
            Self::Run => 5,
        }
    }

    /// 该作用域是否在 agent 重启后依然持久化。
    pub fn is_persistent(&self) -> bool {
        matches!(self, Self::User | Self::Project | Self::Repo)
    }

    /// 按优先级顺序返回所有作用域。
    pub fn all() -> [Self; 6] {
        [
            Self::User,
            Self::Project,
            Self::Repo,
            Self::Task,
            Self::Session,
            Self::Run,
        ]
    }
}

impl std::str::FromStr for MemoryScope {
    type Err = String;

    /// 从字符串解析出对应的记忆作用域（大小写不敏感，支持部分别名）。
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_lowercase().as_str() {
            "user" => Ok(Self::User),
            "project" | "proj" => Ok(Self::Project),
            "repo" => Ok(Self::Repo),
            "task" => Ok(Self::Task),
            "session" | "sess" => Ok(Self::Session),
            "run" => Ok(Self::Run),
            _ => Err(format!("unknown memory scope: {value}")),
        }
    }
}
