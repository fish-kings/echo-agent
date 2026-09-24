//! ReadSkillResourceTool —— 按需加载资源文件（第三层）。
//!
//! 当技能的指令内容引用了随附文件（脚本、参考文档、素材）时，
//! LLM 会调用此工具将其内容加载到上下文中。
//!
//! 安全性：拒绝路径穿越（`..`），且只允许从已激活技能的目录中读取文件。

use std::sync::Arc;

use futures::future::BoxFuture;
use serde_json::json;
use tokio::sync::RwLock;

use crate::skills::registry::SkillRegistry;
use echo_core::error::{Result, ToolError};
use echo_core::tools::{Tool, ToolParameters, ToolResult};

/// 资源文件的默认最大大小限制（1 MB）。
const DEFAULT_MAX_RESOURCE_BYTES: usize = 1024 * 1024;

/// 用于从已激活技能目录中读取资源文件的工具。
///
/// 仅允许读取已通过 `activate_skill` 激活的技能的资源。
/// 出于安全考虑会拒绝路径穿越尝试。
pub struct ReadSkillResourceTool {
    /// 技能注册表，用于查询技能是否已激活及其描述符信息。
    registry: Arc<RwLock<SkillRegistry>>,
    /// 允许读取的资源文件最大字节数。
    max_resource_bytes: usize,
}

impl ReadSkillResourceTool {
    /// 使用默认的最大资源文件大小构造一个新的 `ReadSkillResourceTool`。
    pub fn new(registry: Arc<RwLock<SkillRegistry>>) -> Self {
        Self {
            registry,
            max_resource_bytes: DEFAULT_MAX_RESOURCE_BYTES,
        }
    }

    /// 设置允许的资源文件最大字节数。
    pub fn with_max_bytes(mut self, bytes: usize) -> Self {
        self.max_resource_bytes = bytes;
        self
    }
}

impl Tool for ReadSkillResourceTool {
    /// 返回该工具在工具调用协议中使用的名称：`"read_skill_resource"`。
    fn name(&self) -> &str {
        "read_skill_resource"
    }

    /// 返回该工具的自然语言描述，用于告知模型何时调用它。
    fn description(&self) -> &str {
        "Read a resource file from an activated skill's directory. \
         Use this when a skill's instructions reference a file \
         (e.g., scripts/extract.py, references/guide.md). \
         The skill must be activated first via activate_skill."
    }

    /// 返回该工具调用参数的 JSON Schema，包含技能名称和资源文件相对路径。
    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "skill_name": {
                    "type": "string",
                    "description": "Name of the activated skill"
                },
                "path": {
                    "type": "string",
                    "description": "Relative path to the resource file within the skill directory \
                                    (e.g., 'references/guide.md', 'scripts/run.py', 'checklist.md')"
                }
            },
            "required": ["skill_name", "path"]
        })
    }

    /// 执行资源读取：校验技能已激活、路径安全且未逃逸技能目录、文件大小未超限，
    /// 随后读取文件内容并包装为 `<skill_resource>` 标签文本返回。
    fn execute(&self, parameters: ToolParameters) -> BoxFuture<'_, Result<ToolResult>> {
        Box::pin(async move {
            let skill_name = parameters
                .get("skill_name")
                .and_then(|v| v.as_str())
                .ok_or_else(|| ToolError::MissingParameter("skill_name".to_string()))?
                .to_string();

            let rel_path = parameters
                .get("path")
                .and_then(|v| v.as_str())
                .ok_or_else(|| ToolError::MissingParameter("path".to_string()))?
                .to_string();

            // 使用 Path::components 进行路径安全检查（能有效应对编码后的路径穿越攻击）
            let rel_path_obj = std::path::Path::new(&rel_path);
            if !is_path_traversal_safe(rel_path_obj) {
                return Ok(ToolResult::error(
                    "Path traversal ('..') is not allowed in resource paths",
                ));
            }

            let registry = self.registry.read().await;

            if !registry.is_activated(&skill_name) {
                return Ok(ToolResult::error(format!(
                    "Skill '{}' has not been activated. Call activate_skill first.",
                    skill_name
                )));
            }

            let descriptor = match registry.get_descriptor(&skill_name) {
                Some(d) => d,
                None => {
                    return Ok(ToolResult::error(format!(
                        "Skill '{}' not found in catalog",
                        skill_name
                    )));
                }
            };

            if !descriptor.permits_tool(self.name()) {
                return Ok(ToolResult::error(format!(
                    "Skill '{}' does not permit tool '{}'; allowed-tools: {}",
                    skill_name,
                    self.name(),
                    descriptor.allowed_tools.join(", ")
                )));
            }

            let skill_dir = match descriptor.location.parent() {
                Some(d) => d,
                None => {
                    return Ok(ToolResult::error(format!(
                        "Cannot determine skill directory for '{}'",
                        skill_name
                    )));
                }
            };

            // 先规范化技能目录路径（该目录必须存在）
            let canonical_skill =
                skill_dir
                    .canonicalize()
                    .map_err(|_| ToolError::ExecutionFailed {
                        tool: self.name().to_string(),
                        message: format!("Cannot resolve skill directory for '{}'", skill_name),
                    })?;

            // 在规范化资源路径之前先检查文件是否存在
            let resource_path = skill_dir.join(&rel_path);
            if !resource_path.exists() {
                return Ok(ToolResult::error(format!(
                    "Resource file not found: {} (in skill '{}')",
                    rel_path, skill_name
                )));
            }

            // 验证解析后的路径仍然位于技能目录之下
            let canonical_resource =
                resource_path
                    .canonicalize()
                    .map_err(|_| ToolError::ExecutionFailed {
                        tool: self.name().to_string(),
                        message: "Cannot resolve resource path".to_string(),
                    })?;
            if !canonical_resource.starts_with(&canonical_skill) {
                return Ok(ToolResult::error(
                    "Resolved path escapes the skill directory",
                ));
            }

            let metadata = match tokio::fs::metadata(&resource_path).await {
                Ok(m) => m,
                Err(e) => {
                    return Ok(ToolResult::error(format!(
                        "Cannot stat resource '{}': {}",
                        rel_path, e
                    )));
                }
            };

            if metadata.len() > self.max_resource_bytes as u64 {
                return Ok(ToolResult::error(format!(
                    "Resource file '{}' exceeds maximum size ({} bytes, limit: {} bytes)",
                    rel_path,
                    metadata.len(),
                    self.max_resource_bytes
                )));
            }

            match tokio::fs::read_to_string(&resource_path).await {
                Ok(content) => {
                    let header = format!(
                        "<skill_resource skill=\"{}\" path=\"{}\">\n",
                        skill_name, rel_path
                    );
                    let footer = "\n</skill_resource>";
                    Ok(ToolResult::success(format!(
                        "{}{}{}",
                        header, content, footer
                    )))
                }
                Err(e) => Ok(ToolResult::error(format!(
                    "Failed to read '{}' in skill '{}': {}",
                    rel_path, skill_name, e
                ))),
            }
        })
    }
}

/// 检查路径中是否包含上级目录穿越（`..`）组件。
///
/// 使用 `Path::components` 而非字符串匹配，以应对编码后或非常规形式的路径穿越尝试。
fn is_path_traversal_safe(path: &std::path::Path) -> bool {
    use std::path::Component;
    path.components()
        .all(|c| !matches!(c, Component::ParentDir))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skills::external::types::SkillDescriptor;
    use crate::skills::registry::SkillRegistry;
    use std::collections::HashMap;
    use std::sync::Arc;

    /// 验证即使技能通过 `allowed_tools` 限制了可用工具集，
    /// `read_skill_resource` 工具依然可用并能正确读取该技能目录下的资源内容。
    #[tokio::test]
    async fn read_skill_resource_remains_available_to_restricted_skill() {
        let root =
            std::env::temp_dir().join(format!("echo-skill-resource-test-{}", std::process::id()));
        let skill_dir = root.join("locked-skill");
        tokio::fs::create_dir_all(skill_dir.join("references"))
            .await
            .unwrap();
        tokio::fs::write(skill_dir.join("SKILL.md"), "body")
            .await
            .unwrap();
        tokio::fs::write(skill_dir.join("references/guide.md"), "hello")
            .await
            .unwrap();

        let mut primary = SkillRegistry::new();
        primary.register_descriptor(SkillDescriptor {
            source: None,
            name: "locked-skill".into(),
            description: "desc".into(),
            location: skill_dir.join("SKILL.md"),
            license: None,
            compatibility: None,
            metadata: HashMap::new(),
            allowed_tools: vec!["run_skill_script".into()],
            shell: None,
            paths: vec![],
            triggers: vec![],
            hooks: None,
            sandbox: None,
            depends_on: vec![],
        });
        let mut registry = primary.activation_view();
        registry.register_descriptor(SkillDescriptor {
            source: None,
            name: "locked-skill".into(),
            description: "desc".into(),
            location: skill_dir.join("SKILL.md"),
            license: None,
            compatibility: None,
            metadata: HashMap::new(),
            allowed_tools: vec!["run_skill_script".into()],
            shell: None,
            paths: vec![],
            triggers: vec![],
            hooks: None,
            sandbox: None,
            depends_on: vec![],
        });
        primary.mark_activated("locked-skill");

        let tool = ReadSkillResourceTool::new(Arc::new(RwLock::new(registry)));
        let result = tool
            .execute(
                [
                    ("skill_name".to_string(), json!("locked-skill")),
                    ("path".to_string(), json!("references/guide.md")),
                ]
                .into(),
            )
            .await
            .unwrap();

        assert!(result.success);
        assert!(result.output.contains("hello"));

        let _ = tokio::fs::remove_dir_all(root).await;
    }
}
