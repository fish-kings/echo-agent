//! 从 `.lsp.yaml` 文件加载 LSP 配置。

use echo_core::lsp::LspServerConfig;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// 顶层 LSP 配置。
#[derive(Debug, Clone, Default)]
pub struct LspConfig {
    /// 语言名称到语言服务器配置的映射。
    pub servers: HashMap<String, LspServerConfig>,
}

impl LspConfig {
    /**
    发现本机上已安装的语言服务器。

    发现策略刻意保守：只有当项目包含对应语言、且其可执行文件存在于 `PATH`
    中时才会配置该服务器。不会下载或安装任何东西。
    */
    pub fn discover(project_root: &Path) -> Self {
        let search_paths = std::env::var_os("PATH")
            .map(|value| std::env::split_paths(&value).collect::<Vec<_>>())
            .unwrap_or_default();
        Self::discover_with_search_paths(project_root, &search_paths)
    }

    /// 与 [`Self::discover`] 相同，但使用显式给定的搜索路径代替 `PATH`。
    pub fn discover_with_search_paths(project_root: &Path, search_paths: &[PathBuf]) -> Self {
        let mut config = Self::default();
        for definition in language_definitions() {
            if !project_uses_language(project_root, definition) {
                continue;
            }
            let executable = definition
                .executables
                .iter()
                .find(|candidate| command_available(candidate, search_paths));
            let Some(executable) = executable else {
                continue;
            };
            config.servers.insert(
                definition.language.to_string(),
                LspServerConfig {
                    language: definition.language.to_string(),
                    command: executable.command.to_string(),
                    args: executable
                        .args
                        .iter()
                        .map(|value| value.to_string())
                        .collect(),
                    extensions: definition
                        .extensions
                        .iter()
                        .map(|value| value.to_string())
                        .collect(),
                    env: HashMap::new(),
                    initialization_options: None,
                    max_restarts: 3,
                },
            );
        }
        config
    }

    /// 从 `.lsp.yaml` 文件加载配置。
    pub fn from_file(path: &Path) -> Result<Self, String> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read LSP config file {}: {e}", path.display()))?;
        Self::from_yaml(&content)
    }

    /// 从 YAML 字符串解析配置。
    pub fn from_yaml(yaml: &str) -> Result<Self, String> {
        let file: echo_core::lsp::LspConfigFile = serde_yaml_ng::from_str(yaml)
            .map_err(|e| format!("Failed to parse LSP config YAML: {e}"))?;
        Ok(Self {
            servers: file.languages,
        })
    }

    /// 获取指定语言的配置。
    pub fn get(&self, language: &str) -> Option<&LspServerConfig> {
        self.servers.get(language)
    }

    /// 根据文件扩展名获取配置。
    pub fn get_for_extension(&self, ext: &str) -> Option<(&str, &LspServerConfig)> {
        let ext = if ext.starts_with('.') {
            ext
        } else {
            &format!(".{ext}")
        };
        for (lang, config) in &self.servers {
            if config.extensions.iter().any(|e| e == ext) {
                return Some((lang.as_str(), config));
            }
        }
        None
    }

    /// 将另一份配置合并进来（`other` 中的配置优先）。
    pub fn merge(&mut self, other: LspConfig) {
        self.servers.extend(other.servers);
    }
}

/// 候选语言服务器可执行文件的定义。
struct ExecutableDefinition {
    /// 可执行命令名或路径
    command: &'static str,
    /// 启动语言服务器时传入的参数
    args: &'static [&'static str],
    /// 探测可用性时使用的参数（`None` 表示不探测）
    probe_args: Option<&'static [&'static str]>,
}

/// 内置语言定义：用于判断项目是否使用某语言以及可选的服务器。
struct LanguageDefinition {
    /// 语言名称
    language: &'static str,
    /// 表明项目使用该语言的标记文件（如 `Cargo.toml`）
    markers: &'static [&'static str],
    /// 该语言的文件扩展名（带前导 `.`）
    extensions: &'static [&'static str],
    /// 按优先级排列的候选可执行文件
    executables: &'static [ExecutableDefinition],
}

/// 返回内置的语言定义表。
fn language_definitions() -> &'static [LanguageDefinition] {
    &[
        LanguageDefinition {
            language: "rust",
            markers: &["Cargo.toml"],
            extensions: &[".rs"],
            executables: &[ExecutableDefinition {
                command: "rust-analyzer",
                args: &[],
                probe_args: Some(&["--version"]),
            }],
        },
        LanguageDefinition {
            language: "python",
            markers: &["pyproject.toml", "requirements.txt", "setup.py", "Pipfile"],
            extensions: &[".py", ".pyi"],
            executables: &[
                ExecutableDefinition {
                    command: "basedpyright-langserver",
                    args: &["--stdio"],
                    probe_args: Some(&["--version"]),
                },
                ExecutableDefinition {
                    command: "pyright-langserver",
                    args: &["--stdio"],
                    probe_args: Some(&["--version"]),
                },
                ExecutableDefinition {
                    command: "pylsp",
                    args: &[],
                    probe_args: Some(&["--version"]),
                },
            ],
        },
        LanguageDefinition {
            language: "typescript",
            markers: &["package.json", "tsconfig.json", "jsconfig.json"],
            extensions: &[".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"],
            executables: &[ExecutableDefinition {
                command: "typescript-language-server",
                args: &["--stdio"],
                probe_args: Some(&["--version"]),
            }],
        },
        LanguageDefinition {
            language: "go",
            markers: &["go.mod", "go.work"],
            extensions: &[".go"],
            executables: &[ExecutableDefinition {
                command: "gopls",
                args: &[],
                probe_args: Some(&["version"]),
            }],
        },
        LanguageDefinition {
            language: "java",
            markers: &["pom.xml", "build.gradle", "build.gradle.kts"],
            extensions: &[".java"],
            executables: &[ExecutableDefinition {
                command: "jdtls",
                args: &[],
                probe_args: None,
            }],
        },
        LanguageDefinition {
            language: "c_cpp",
            markers: &["CMakeLists.txt", "compile_commands.json", "meson.build"],
            extensions: &[".c", ".h", ".cc", ".cpp", ".cxx", ".hpp"],
            executables: &[ExecutableDefinition {
                command: "clangd",
                args: &[],
                probe_args: Some(&["--version"]),
            }],
        },
    ]
}

/// 判断项目是否使用某语言：先查标记文件，再有限深度地扫描文件扩展名。
fn project_uses_language(root: &Path, definition: &LanguageDefinition) -> bool {
    if definition
        .markers
        .iter()
        .any(|marker| root.join(marker).is_file())
    {
        return true;
    }
    let mut stack = vec![(root.to_path_buf(), 0usize)];
    let mut visited = 0usize;
    while let Some((directory, depth)) = stack.pop() {
        if depth > 4 || visited >= 10_000 {
            continue;
        }
        let entries = match std::fs::read_dir(directory) {
            Ok(entries) => entries,
            Err(_) => continue,
        };
        for entry in entries.flatten() {
            visited = visited.saturating_add(1);
            let path = entry.path();
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if path.is_dir() {
                if !matches!(
                    name.as_ref(),
                    ".git"
                        | "target"
                        | "node_modules"
                        | "vendor"
                        | ".venv"
                        | "venv"
                        | "dist"
                        | "build"
                ) {
                    stack.push((path, depth.saturating_add(1)));
                }
                continue;
            }
            let extension = path
                .extension()
                .and_then(|value| value.to_str())
                .map(|value| format!(".{value}"));
            if extension.as_ref().is_some_and(|extension| {
                definition.extensions.iter().any(|value| value == extension)
            }) {
                return true;
            }
        }
    }
    false
}

/// 判断候选可执行文件是否存在且（如需）探测成功。
fn command_available(definition: &ExecutableDefinition, search_paths: &[PathBuf]) -> bool {
    let Some(command_path) = find_command(definition.command, search_paths) else {
        return false;
    };
    definition
        .probe_args
        .is_none_or(|args| command_probe_succeeds(&command_path, args))
}

/// 在搜索路径中查找命令；含路径分隔的命令直接检查文件是否存在。
fn find_command(command: &str, search_paths: &[PathBuf]) -> Option<PathBuf> {
    let command_path = Path::new(command);
    if command_path.components().count() > 1 {
        return command_path.is_file().then(|| command_path.to_path_buf());
    }
    search_paths.iter().find_map(|directory| {
        let candidate = directory.join(command);
        if candidate.is_file() {
            return Some(candidate);
        }
        if cfg!(windows) {
            return ["exe", "cmd", "bat"]
                .iter()
                .map(|extension| directory.join(format!("{command}.{extension}")))
                .find(|path| path.is_file());
        }
        None
    })
}

/// 以探测参数运行命令，在 2 秒内成功退出则视为可用，超时则终止进程。
fn command_probe_succeeds(command: &Path, args: &[&str]) -> bool {
    let mut child = match Command::new(command)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
    {
        Ok(child) => child,
        Err(_) => return false,
    };
    let deadline = Instant::now() + Duration::from_secs(2);
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return status.success(),
            Ok(None) if Instant::now() < deadline => {
                std::thread::sleep(Duration::from_millis(20));
            }
            Ok(None) | Err(_) => {
                let _ = child.kill();
                let _ = child.wait();
                return false;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// 验证 LSP 配置解析。
    #[test]
    fn test_parse_config() -> Result<(), String> {
        let yaml = r#"
languages:
  python:
    language: python
    command: pyright-langserver
    args: ["--stdio"]
    extensions: [".py", ".pyi"]
  typescript:
    language: typescript
    command: typescript-language-server
    args: ["--stdio"]
    extensions: [".ts", ".tsx", ".js", ".jsx"]
  rust:
    language: rust
    command: rust-analyzer
    args: []
    extensions: [".rs"]
"#;
        let config = LspConfig::from_yaml(yaml)?;
        assert_eq!(config.servers.len(), 3);
        assert_eq!(
            config
                .get("python")
                .ok_or_else(|| "python config missing".to_string())?
                .command,
            "pyright-langserver"
        );
        assert_eq!(
            config
                .get_for_extension(".rs")
                .ok_or_else(|| "Rust extension missing".to_string())?
                .0,
            "rust"
        );
        assert_eq!(
            config
                .get_for_extension("py")
                .ok_or_else(|| "Python extension missing".to_string())?
                .0,
            "python"
        );
        Ok(())
    }

    /// 验证能为项目语言发现已安装的语言服务器。
    #[test]
    fn discovers_installed_server_for_project_language() -> Result<(), Box<dyn std::error::Error>> {
        let project = tempfile::tempdir()?;
        let bin = tempfile::tempdir()?;
        fs::write(
            project.path().join("Cargo.toml"),
            "[package]\nname='demo'\n",
        )?;
        let executable = bin.path().join("rust-analyzer");
        fs::write(&executable, "#!/bin/sh\nexit 0\n")?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut permissions = fs::metadata(&executable)?.permissions();
            permissions.set_mode(0o755);
            fs::set_permissions(&executable, permissions)?;
        }
        let config =
            LspConfig::discover_with_search_paths(project.path(), &[bin.path().to_path_buf()]);
        let rust = config.get("rust").ok_or("rust server was not discovered")?;
        assert_eq!(rust.command, "rust-analyzer");
        Ok(())
    }

    /// 验证探测失败的可执行文件会被拒绝。
    #[test]
    fn rejects_executable_that_fails_probe() -> Result<(), Box<dyn std::error::Error>> {
        let project = tempfile::tempdir()?;
        let bin = tempfile::tempdir()?;
        fs::write(
            project.path().join("Cargo.toml"),
            "[package]\nname='demo'\n",
        )?;
        let executable = bin.path().join("rust-analyzer");
        fs::write(&executable, "#!/bin/sh\nexit 1\n")?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut permissions = fs::metadata(&executable)?.permissions();
            permissions.set_mode(0o755);
            fs::set_permissions(&executable, permissions)?;
        }

        let config =
            LspConfig::discover_with_search_paths(project.path(), &[bin.path().to_path_buf()]);
        assert!(config.get("rust").is_none());
        Ok(())
    }
}
