/*!
插件安装作用域 —— 插件存放位置以及可使用范围。

| 作用域 | 路径 | 使用场景 |
|-------|------|----------|
| `User` | `<plugin_data_root>/plugins/` | 个人插件 |
| `Project` | `.echo-agent/plugins/`（项目根目录） | 通过版本控制与团队共享 |
| `Local` | `.echo-agent/plugins.local/`（项目根目录） | 项目私有，被 gitignore 忽略 |
*/

use std::path::{Path, PathBuf};

/// 插件的安装位置及可访问范围。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PluginScope {
    /** 对该用户的所有项目可用。
    存放于 `~/.echo-agent/plugins/`。 */
    User,
    /** 通过版本控制与团队共享。
    存放于 `<project-root>/.echo-agent/plugins/`。 */
    Project,
    /** 项目专属，不提交到版本控制。
    存放于 `<project-root>/.echo-agent/plugins.local/`。 */
    Local,
}

impl PluginScope {
    /**
    解析该作用域对应的文件系统路径。

    - `User`：由应用提供的用户插件目录
    - `Project`：`<project_root>/.echo-agent/plugins/`
    - `Local`：`<project_root>/.echo-agent/plugins.local/`

    Project/Local 作用域有意始终使用 `.echo-agent`，不受用户作用域覆盖的影响：
    它们是相对于工作区的约定目录，而非用户数据目录；提交到版本控制的团队共享插件
    必须落在与任何开发者品牌设置无关的稳定路径上。
    */
    pub fn resolve_dir(&self, user_plugins_dir: &Path, project_root: Option<&Path>) -> PathBuf {
        match self {
            Self::User => user_plugins_dir.to_path_buf(),
            Self::Project => {
                let root = project_root.unwrap_or_else(|| Path::new("."));
                root.join(".echo-agent").join("plugins")
            }
            Self::Local => {
                let root = project_root.unwrap_or_else(|| Path::new("."));
                root.join(".echo-agent").join("plugins.local")
            }
        }
    }

    /// 按优先级顺序返回所有作用域（user → project → local）。
    pub fn all() -> &'static [PluginScope] {
        &[PluginScope::User, PluginScope::Project, PluginScope::Local]
    }
}

impl std::str::FromStr for PluginScope {
    /// 解析失败时返回的错误信息。
    type Err = String;

    /// 从字符串解析作用域（支持全称与首字母缩写，大小写不敏感）
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_lowercase().as_str() {
            "user" | "u" => Ok(Self::User),
            "project" | "p" => Ok(Self::Project),
            "local" | "l" => Ok(Self::Local),
            _ => Err(format!("unknown plugin scope: {value}")),
        }
    }
}

impl std::fmt::Display for PluginScope {
    /// 输出作用域的小写名称
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::User => write!(f, "user"),
            Self::Project => write!(f, "project"),
            Self::Local => write!(f, "local"),
        }
    }
}

/// 插件的安装来源 —— 本地目录或 git URL。
#[derive(Debug, Clone)]
pub enum InstallSource {
    /// 从本地目录路径安装。
    Local(PathBuf),
    /// 从 git 仓库 URL 安装，可选指定子目录。
    Git {
        /// git 仓库地址
        url: String,
        /// 仓库内的可选子目录
        subdir: Option<String>,
    },
}

impl InstallSource {
    /// 解析安装来源字符串 —— 区分 git URL 与本地路径。
    pub fn parse(s: &str) -> Self {
        if s.starts_with("http://")
            || s.starts_with("https://")
            || s.starts_with("ssh://")
            || s.starts_with("git://")
            || s.starts_with("git@")
            || s.ends_with(".git")
        {
            Self::Git {
                url: s.to_string(),
                subdir: None,
            }
        } else {
            Self::Local(PathBuf::from(s))
        }
    }

    /// 是否为 git 来源。
    pub fn is_git(&self) -> bool {
        matches!(self, Self::Git { .. })
    }
}

impl std::fmt::Display for InstallSource {
    /// 输出本地路径，或 `url[:subdir]` 形式的 git 来源
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Local(p) => write!(f, "{}", p.display()),
            Self::Git { url, subdir } => {
                write!(f, "{url}")?;
                if let Some(sub) = subdir {
                    write!(f, ":{sub}")?;
                }
                Ok(())
            }
        }
    }
}

/// 插件作用域与安装来源的单元测试。
#[cfg(test)]
mod tests {
    use super::*;

    /// 验证 User 作用域的路径解析。
    #[test]
    fn test_scope_resolve_user() {
        let dir = PluginScope::User.resolve_dir(Path::new("/data/plugins"), None);
        assert_eq!(dir, PathBuf::from("/data/plugins"));
    }

    /// 验证 Project 作用域的路径解析。
    #[test]
    fn test_scope_resolve_project() {
        let dir = PluginScope::Project.resolve_dir(
            Path::new("/data/plugins"),
            Some(Path::new("/home/user/my-project")),
        );
        assert_eq!(
            dir,
            PathBuf::from("/home/user/my-project/.echo-agent/plugins")
        );
    }

    /// 验证 Local 作用域的路径解析。
    #[test]
    fn test_scope_resolve_local() {
        let dir = PluginScope::Local.resolve_dir(
            Path::new("/data/plugins"),
            Some(Path::new("/home/user/my-project")),
        );
        assert_eq!(
            dir,
            PathBuf::from("/home/user/my-project/.echo-agent/plugins.local")
        );
    }

    /// 验证本地安装来源的解析。
    #[test]
    fn test_install_source_parse_local() {
        let src = InstallSource::parse("/home/user/my-plugin");
        assert!(!src.is_git());
    }

    /// 验证 Git 安装来源的解析。
    #[test]
    fn test_install_source_parse_git() {
        let src = InstallSource::parse("https://github.com/echo/plugin.git");
        assert!(src.is_git());
    }

    /// 验证作用域的字符串解析。
    #[test]
    fn test_scope_from_str() {
        assert_eq!("user".parse(), Ok(PluginScope::User));
        assert_eq!("project".parse(), Ok(PluginScope::Project));
        assert_eq!("local".parse(), Ok(PluginScope::Local));
        assert!("invalid".parse::<PluginScope>().is_err());
    }
}
