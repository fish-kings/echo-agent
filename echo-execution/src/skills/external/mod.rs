//! 文件型（外部）skill 的发现、激活、校验与工具封装。

pub mod activate_tool;
pub mod loader;
pub mod prompt_exec;
pub mod resource_tool;
pub mod run_script_tool;
/// skill 描述、文档与资源等公共类型。
pub mod types;
pub mod validate;

pub use activate_tool::ActivateSkillTool;
pub use loader::{DiscoveryScope, SkillDiscoveryDiagnostic, SkillLoadPolicy, SkillLoader};
pub use prompt_exec::{PromptContext, SkillSource, find_git_bash_path, process_skill_content};
pub use resource_tool::ReadSkillResourceTool;
pub use run_script_tool::RunSkillScriptTool;
pub use types::{
    SkillContent, SkillDescriptor, SkillDocument, SkillResourceEntry, SkillResourceKind,
    SkillSandboxPolicy, is_skill_control_tool, skill_allows_tool, tool_matcher,
};
pub use validate::{
    OFFICIAL_FRONTMATTER_FIELDS, SkillValidationReport, validate_skill_dir, validate_skill_markdown,
};
