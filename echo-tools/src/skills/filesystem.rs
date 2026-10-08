use std::path::PathBuf;

use echo_core::tools::Tool;
use echo_core::tools::skill::Skill;

use crate::files::apply_patch::ApplyPatchTool;
use crate::files::artifact::ReadArtifactTool;
use crate::files::diff::DiffTool;
use crate::files::files::{ListDirTool, ReadFileTool};
use crate::files::glob::GlobTool;
use crate::files::grep::GrepTool;

/// 文件系统技能：提供唯一、规范的事务性变更入口（以及配套的只读工具）。
pub struct FileSystemSkill {
    /// 所有工具路径解析的限制范围；为 `None` 时不做限制。
    base_dir: Option<PathBuf>,
}

impl FileSystemSkill {
    /// 创建一个不限定基准目录的 `FileSystemSkill`。
    pub fn new() -> Self {
        Self { base_dir: None }
    }

    /// 创建一个限定了基准目录的 `FileSystemSkill`，所有工具的操作都会被限制在该目录内。
    pub fn with_base_dir(base: impl Into<PathBuf>) -> Self {
        Self {
            base_dir: Some(base.into()),
        }
    }
}

impl Default for FileSystemSkill {
    /// 等价于 [`FileSystemSkill::new`]，创建不限定基准目录的实例。
    fn default() -> Self {
        Self::new()
    }
}

impl Skill for FileSystemSkill {
    /// 返回该技能的标识名称：`"filesystem"`。
    fn name(&self) -> &str {
        "filesystem"
    }

    /// 返回该技能的功能描述。
    fn description(&self) -> &str {
        "Local filesystem capability with transactional patch editing, file reading, search, diff, and directory listing"
    }

    /**
    构造该技能提供的全部工具实例（读取、补丁写入、目录列表、搜索、diff 等），
    若设置了 `base_dir`，则每个工具都会以该目录作为操作限制范围。
    */
    fn tools(&self) -> Vec<Box<dyn Tool>> {
        let base = self.base_dir.clone();
        vec![
            Box::new(match &base {
                Some(base) => ReadFileTool::with_base_dir(base),
                None => ReadFileTool::new(),
            }),
            Box::new(ReadArtifactTool),
            Box::new(match &base {
                Some(base) => ApplyPatchTool::with_base_dir(base),
                None => ApplyPatchTool::new(),
            }),
            Box::new(match &base {
                Some(base) => ListDirTool::with_base_dir(base),
                None => ListDirTool::new(),
            }),
            Box::new(match &base {
                Some(base) => GrepTool::with_base_dir(base),
                None => GrepTool::new(),
            }),
            Box::new(match &base {
                Some(base) => GlobTool::with_base_dir(base),
                None => GlobTool::new(),
            }),
            Box::new(match &base {
                Some(base) => DiffTool::with_base_dir(base),
                None => DiffTool::new(),
            }),
        ]
    }

    /// 生成注入系统提示词的说明文本，告知模型可用的文件系统工具及其路径限制范围。
    fn system_prompt_injection(&self) -> Option<String> {
        let restriction = self.base_dir.as_ref().map_or_else(
            || " (no path restriction, exercise caution when operating)".to_string(),
            |base| format!(" (operations restricted to '{}' directory)", base.display()),
        );
        Some(format!(
            "\n\n## Filesystem Capability{restriction}\nUse `apply_patch` for mutations and `read_file`, `read_artifact`, `diff`, `list_dir`, `grep`, and `glob` for inspection."
        ))
    }
}
