//! 在明确的仓库边界内发现项目级指令文件。

use std::path::{Path, PathBuf};

/// echo-agent 原生指令文件（相对路径, 类型标识），按优先级排列。
const NATIVE_FILES: &[(&str, &str)] = &[
    (".echo-agent/AGENT.md", "echo_agent"),
    (".echo-agent/RULES.md", "echo_agent"),
    (".echo-agent/rules.md", "echo_agent"),
];
/// 与其他工具兼容的指令文件（相对路径, 类型标识），按优先级排列。
const COMPATIBLE_FILES: &[(&str, &str)] = &[
    ("AGENTS.override.md", "agents_override"),
    ("AGENTS.md", "agents"),
    ("CLAUDE.md", "claude"),
];
/// 仅包含跨工具 AGENTS 规范的指令文件（相对路径, 类型标识）。
const AGENTS_FILES: &[(&str, &str)] = &[
    ("AGENTS.override.md", "agents_override"),
    ("AGENTS.md", "agents"),
];

/// 解析时要考虑的指令文件集合。
#[derive(Debug, Clone, Copy, Default)]
enum InstructionFileSet {
    /// 原生文件与所有兼容文件。
    #[default]
    All,
    /// 仅 `AGENTS.override.md` / `AGENTS.md`。
    AgentsOnly,
}

/// 解析链中包含的一个指令文件。
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct InstructionSource {
    /// 指令文件的规范化路径。
    pub path: PathBuf,
    /// 指令文件类型标识（例如 "agents"、"claude"、"echo_agent"）。
    pub kind: String,
    /// 从根到叶的顺序。数值越大，越靠近工作目录。
    pub precedence: usize,
}

/// 完整解析后的项目指令及诊断信息。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ResolvedInstructions {
    /// 按从根到叶顺序拼接、带来源标注的指令内容。
    pub content: String,
    /// 参与拼接的指令文件来源列表。
    pub sources: Vec<InstructionSource>,
    /// 识别出的项目根目录（未找到时为 `None`）。
    pub project_root: Option<PathBuf>,
}

impl ResolvedInstructions {
    /// 是否没有解析到任何指令文件。
    pub fn is_empty(&self) -> bool {
        self.sources.is_empty()
    }

    /// 渲染带来源标注的解析链，用于诊断；内容为空时返回 `None`。
    pub fn annotated(&self) -> Option<String> {
        (!self.content.is_empty()).then(|| self.content.clone())
    }
}

/// 为单个工作目录解析原生及兼容的指令文件。
#[derive(Debug, Clone)]
pub struct InstructionResolver {
    /// 要解析指令的工作目录。
    working_dir: PathBuf,
    /// 显式指定的项目根目录；未指定时向上查找 git 根目录。
    explicit_project_root: Option<PathBuf>,
    /// 要考虑的指令文件集合。
    file_set: InstructionFileSet,
}

impl InstructionResolver {
    /// 为指定工作目录创建解析器，默认考虑全部指令文件。
    pub fn new(working_dir: impl Into<PathBuf>) -> Self {
        Self {
            working_dir: working_dir.into(),
            explicit_project_root: None,
            file_set: InstructionFileSet::All,
        }
    }

    /// 显式指定项目根目录（仅当工作目录位于其下时生效）。
    pub fn project_root(mut self, root: impl Into<PathBuf>) -> Self {
        self.explicit_project_root = Some(root.into());
        self
    }

    /**
    仅解析跨工具的 `AGENTS.override.md` / `AGENTS.md` 链。

    这会排除 echo-agent 原生的 `.echo-agent/` 目录下的文件以及 `CLAUDE.md`。
    适用于拥有独立的产品专属指令命名空间、但仍希望使用标准的从根到叶
    AGENTS 语义与仓库边界保护的使用方。
    */
    pub fn agents_files_only(mut self) -> Self {
        self.file_set = InstructionFileSet::AgentsOnly;
        self
    }

    /// 从项目根到工作目录逐级解析指令文件，每个目录最多取一个文件。
    pub fn resolve(&self) -> ResolvedInstructions {
        let working_dir = canonical_or_original(&self.working_dir);
        let project_root = self
            .explicit_project_root
            .as_ref()
            .map(|root| canonical_or_original(root))
            .filter(|root| working_dir.starts_with(root))
            .or_else(|| find_git_root(&working_dir));
        let scan_root = project_root.clone().unwrap_or_else(|| working_dir.clone());
        let directories = path_from_root_to_working_dir(&scan_root, &working_dir);

        let mut sources = Vec::new();
        let mut blocks = Vec::new();
        for directory in directories {
            let Some((path, kind, content)) =
                load_one_directory(&directory, &scan_root, self.file_set)
            else {
                continue;
            };
            let precedence = sources.len();
            blocks.push(format!(
                "<!-- PROJECT INSTRUCTIONS: {} -->\n{}\n<!-- END PROJECT INSTRUCTIONS -->",
                path.display(),
                content.trim()
            ));
            sources.push(InstructionSource {
                path,
                kind: kind.to_string(),
                precedence,
            });
        }

        ResolvedInstructions {
            content: blocks.join("\n\n"),
            sources,
            project_root,
        }
    }
}

/// 尝试规范化路径，失败时返回原路径。
fn canonical_or_original(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
}

/// 向上查找包含 `.git`（目录或文件）的最近祖先目录。
fn find_git_root(working_dir: &Path) -> Option<PathBuf> {
    working_dir
        .ancestors()
        .find(|directory| directory.join(".git").exists())
        .map(Path::to_path_buf)
}

/// 列出从根目录到工作目录（含两端）的目录路径，按从根到叶排序。
fn path_from_root_to_working_dir(root: &Path, working_dir: &Path) -> Vec<PathBuf> {
    if !working_dir.starts_with(root) {
        return vec![working_dir.to_path_buf()];
    }
    let mut directories: Vec<PathBuf> = working_dir
        .ancestors()
        .take_while(|directory| directory.starts_with(root))
        .map(Path::to_path_buf)
        .collect();
    directories.reverse();
    directories
}

/// 按优先级在单个目录中加载第一个存在、位于项目根内且非空的指令文件。
fn load_one_directory(
    directory: &Path,
    project_root: &Path,
    file_set: InstructionFileSet,
) -> Option<(PathBuf, &'static str, String)> {
    let candidates: Box<dyn Iterator<Item = &(&str, &str)> + '_> = match file_set {
        InstructionFileSet::All => Box::new(NATIVE_FILES.iter().chain(COMPATIBLE_FILES.iter())),
        InstructionFileSet::AgentsOnly => Box::new(AGENTS_FILES.iter()),
    };
    for (relative, kind) in candidates {
        let candidate = directory.join(relative);
        let canonical = match candidate.canonicalize() {
            Ok(path) => path,
            Err(_) => continue,
        };
        if !canonical.starts_with(project_root) || !canonical.is_file() {
            continue;
        }
        match std::fs::read_to_string(&canonical) {
            Ok(content) if !content.trim().is_empty() => return Some((canonical, kind, content)),
            _ => continue,
        }
    }
    None
}

/**
向后兼容的便捷 API，返回优先级最高的来源路径以及完整的解析链内容。
*/
pub fn load_project_rules(working_dir: &Path) -> Option<(PathBuf, String)> {
    let resolved = InstructionResolver::new(working_dir).resolve();
    let path = resolved.sources.last()?.path.clone();
    Some((path, resolved.content))
}

/// 返回工作目录对应的、带来源标注的项目指令文本。
pub fn rules_injection(working_dir: &Path) -> Option<String> {
    InstructionResolver::new(working_dir).resolve().annotated()
}

/// 同 [`rules_injection`]，但可显式指定项目根目录。
pub fn rules_injection_with_root(
    working_dir: &Path,
    project_root: Option<&Path>,
) -> Option<String> {
    let resolver = InstructionResolver::new(working_dir);
    let resolver = match project_root {
        Some(root) => resolver.project_root(root),
        None => resolver,
    };
    resolver.resolve().annotated()
}

/// 将项目指令前置到已有提示词之前；没有指令时原样返回。
pub fn inject_rules(existing_prompt: &str, working_dir: &Path) -> String {
    match rules_injection(working_dir) {
        Some(rules) => format!("{}\n\n{}", rules, existing_prompt),
        None => existing_prompt.to_string(),
    }
}

/// 同 [`inject_rules`]，但可显式指定项目根目录。
pub fn inject_rules_with_root(
    existing_prompt: &str,
    working_dir: &Path,
    project_root: Option<&Path>,
) -> String {
    match rules_injection_with_root(working_dir, project_root) {
        Some(rules) => format!("{}\n\n{}", rules, existing_prompt),
        None => existing_prompt.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// 测试结束时自动删除的临时目录。
    struct TestDir(PathBuf);

    impl TestDir {
        /// 在系统临时目录下创建一个唯一的测试目录。
        fn new() -> std::io::Result<Self> {
            let path = std::env::temp_dir()
                .join(format!("echo-agent-instructions-{}", uuid::Uuid::new_v4()));
            fs::create_dir_all(&path)?;
            Ok(Self(path))
        }

        /// 返回测试目录路径。
        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TestDir {
        /// 删除测试目录（忽略错误）。
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    /// 写入文件，必要时先创建父目录。
    fn write(path: &Path, content: &str) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, content)
    }

    #[test]
    fn resolves_root_to_leaf_and_one_file_per_directory() -> Result<(), Box<dyn std::error::Error>>
    {
        let temp = TestDir::new()?;
        let root = temp.path().join("repo");
        let child = root.join("src/module");
        fs::create_dir_all(root.join(".git"))?;
        write(&root.join("AGENTS.md"), "root agents")?;
        write(&root.join(".echo-agent/AGENT.md"), "root native")?;
        write(&root.join("src/AGENTS.md"), "src agents")?;
        write(&root.join("src/CLAUDE.md"), "ignored same level")?;
        write(&child.join("AGENTS.override.md"), "leaf override")?;

        let resolved = InstructionResolver::new(&child).resolve();
        let canonical_root = canonical_or_original(&root);
        assert_eq!(
            resolved.project_root.as_deref(),
            Some(canonical_root.as_path())
        );
        assert_eq!(resolved.sources.len(), 3);
        assert_eq!(
            resolved.sources.first().map(|source| source.kind.as_str()),
            Some("echo_agent")
        );
        assert_eq!(
            resolved.sources.get(1).map(|source| source.kind.as_str()),
            Some("agents")
        );
        assert_eq!(
            resolved.sources.get(2).map(|source| source.kind.as_str()),
            Some("agents_override")
        );
        assert!(resolved.content.contains("root native"));
        assert!(!resolved.content.contains("root agents"));
        assert!(!resolved.content.contains("ignored same level"));
        assert!(resolved.content.find("root native") < resolved.content.find("leaf override"));
        Ok(())
    }

    #[test]
    fn explicit_root_blocks_parent_instructions() -> Result<(), Box<dyn std::error::Error>> {
        let temp = TestDir::new()?;
        let parent = temp.path().join("parent");
        let root = parent.join("repo");
        let child = root.join("src");
        fs::create_dir_all(&child)?;
        write(&parent.join("AGENTS.md"), "outside")?;
        write(&root.join("AGENTS.md"), "inside")?;

        let resolved = InstructionResolver::new(&child)
            .project_root(&root)
            .resolve();
        assert!(resolved.content.contains("inside"));
        assert!(!resolved.content.contains("outside"));
        Ok(())
    }

    #[test]
    fn no_git_root_checks_only_working_directory() -> Result<(), Box<dyn std::error::Error>> {
        let temp = TestDir::new()?;
        let parent = temp.path().join("parent");
        let child = parent.join("child");
        fs::create_dir_all(&child)?;
        write(&parent.join("AGENTS.md"), "outside")?;
        write(&child.join("CLAUDE.md"), "local")?;

        let resolved = InstructionResolver::new(&child).resolve();
        assert_eq!(resolved.project_root, None);
        assert_eq!(resolved.sources.len(), 1);
        assert!(resolved.content.contains("local"));
        assert!(!resolved.content.contains("outside"));
        Ok(())
    }

    #[test]
    fn empty_and_invalid_utf8_files_are_skipped() -> Result<(), Box<dyn std::error::Error>> {
        let temp = TestDir::new()?;
        let root = temp.path().join("repo");
        fs::create_dir_all(root.join(".git"))?;
        write(&root.join(".echo-agent/AGENT.md"), "  \n")?;
        fs::write(root.join("AGENTS.override.md"), [0xff, 0xfe])?;
        write(&root.join("AGENTS.md"), "valid fallback")?;

        let resolved = InstructionResolver::new(&root).resolve();
        assert_eq!(resolved.sources.len(), 1);
        assert!(resolved.content.contains("valid fallback"));
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn symlink_outside_project_root_is_skipped() -> Result<(), Box<dyn std::error::Error>> {
        use std::os::unix::fs::symlink;

        let temp = TestDir::new()?;
        let root = temp.path().join("repo");
        let outside = temp.path().join("outside.md");
        fs::create_dir_all(root.join(".git"))?;
        write(&outside, "outside")?;
        symlink(&outside, root.join("AGENTS.override.md"))?;
        write(&root.join("AGENTS.md"), "inside")?;

        let resolved = InstructionResolver::new(&root).resolve();
        assert_eq!(resolved.sources.len(), 1);
        assert!(resolved.content.contains("inside"));
        assert!(!resolved.content.contains("outside"));
        Ok(())
    }

    #[test]
    fn git_file_marks_worktree_root() -> Result<(), Box<dyn std::error::Error>> {
        let temp = TestDir::new()?;
        let root = temp.path().join("worktree");
        let child = root.join("src");
        fs::create_dir_all(&child)?;
        write(&root.join(".git"), "gitdir: ../git/worktrees/test")?;
        write(&root.join("AGENTS.md"), "worktree rules")?;

        let resolved = InstructionResolver::new(&child).resolve();
        let canonical_root = canonical_or_original(&root);
        assert_eq!(
            resolved.project_root.as_deref(),
            Some(canonical_root.as_path())
        );
        assert!(resolved.content.contains("worktree rules"));
        Ok(())
    }

    #[test]
    fn agents_only_excludes_native_and_claude_files() -> Result<(), Box<dyn std::error::Error>> {
        let temp = TestDir::new()?;
        let root = temp.path().join("repo");
        let child = root.join("src");
        fs::create_dir_all(root.join(".git"))?;
        fs::create_dir_all(&child)?;
        write(&root.join(".echo-agent/AGENT.md"), "native")?;
        write(&root.join("CLAUDE.md"), "claude")?;
        write(&root.join("AGENTS.md"), "root agents")?;
        write(&child.join("AGENTS.override.md"), "leaf override")?;

        let resolved = InstructionResolver::new(&child)
            .project_root(&root)
            .agents_files_only()
            .resolve();

        assert_eq!(resolved.sources.len(), 2);
        assert!(resolved.content.contains("root agents"));
        assert!(resolved.content.contains("leaf override"));
        assert!(!resolved.content.contains("native"));
        assert!(!resolved.content.contains("claude"));
        Ok(())
    }
}
