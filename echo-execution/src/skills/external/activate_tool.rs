//! ActivateSkillTool —— 由模型驱动的技能激活（第二层）。
//!
//! 当 LLM 判断某个任务匹配某个技能的描述时，会调用此工具
//! 将该技能的完整指令加载到对话上下文中。
//! 该工具返回一个带 XML 标签的结构化文本块，其中包含：
//!
//! 1. 技能的 Markdown 指令内容（SKILL.md 正文）
//! 2. 随附资源文件（脚本、参考资料、素材）的清单
//! 3. 用于解析相对引用的技能目录路径

use std::sync::Arc;

use futures::future::BoxFuture;
use serde_json::json;
use tokio::sync::RwLock;

use crate::skills::registry::SkillRegistry;
use echo_core::error::{Result, ToolError};
use echo_core::tools::{Tool, ToolContext, ToolParameters, ToolResult, ToolResultKind};

/// 根据技能声明的 `allowed_tools` 匹配规则，在工具可见性状态中放开对应工具的可见性，
/// 返回被实际激活（放开可见）的工具名称列表。若上下文中没有工具可见性状态，则返回空列表。
fn activate_allowed_tools(ctx: &ToolContext, allowed_tools: &[String]) -> Vec<String> {
    let Some(visibility) = ctx.tool_visibility.as_ref() else {
        return Vec::new();
    };
    let matches = visibility
        .available_names()
        .into_iter()
        .filter(|tool_name| {
            allowed_tools
                .iter()
                .any(|matcher| super::types::tool_matcher(matcher, tool_name))
        })
        .collect::<Vec<_>>();
    visibility.extend_eligibility_and_activate(matches)
}

/// 用于模型驱动技能激活的工具。
///
/// 在发现基于文件的技能时会自动注册。
/// 当某个任务匹配技能目录中的描述时，LLM 会调用此工具。
pub struct ActivateSkillTool {
    /// 技能注册表，用于查找技能描述符并执行激活逻辑。
    registry: Arc<RwLock<SkillRegistry>>,
    /// 缓存的可用技能名称列表，用于生成参数描述文本。
    available_names: Vec<String>,
}

impl ActivateSkillTool {
    /// 使用给定的技能注册表和可用技能名称列表构造一个新的 `ActivateSkillTool`。
    pub fn new(registry: Arc<RwLock<SkillRegistry>>, available_names: Vec<String>) -> Self {
        Self {
            registry,
            available_names,
        }
    }
}

impl Tool for ActivateSkillTool {
    /// 返回该工具在工具调用协议中使用的名称：`"activate_skill"`。
    fn name(&self) -> &str {
        "activate_skill"
    }

    /// 返回该工具的自然语言描述，用于告知模型何时调用它。
    fn description(&self) -> &str {
        "Activate a skill to load its full instructions and available resources. \
         Call this when a task matches one of the available skills listed in the system prompt. \
         For skills with `paths` constraints, also provide `context_path` for the touched file."
    }

    /// 返回该工具调用参数的 JSON Schema，包含技能名称、可选参数和上下文路径。
    fn parameters(&self) -> serde_json::Value {
        let names_desc = if self.available_names.is_empty() {
            "(no skills available)".to_string()
        } else {
            format!("One of: {}", self.available_names.join(", "))
        };

        json!({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": format!("The skill name to activate. {}", names_desc)
                },
                "arguments": {
                    "type": "string",
                    "description": "Optional arguments to pass to the skill (space-separated). \
                                    Available inside skill content as ${ARGUMENTS}, ${1}, ${2}, etc."
                },
                "context_path": {
                    "type": "string",
                    "description": "Optional touched file path for conditional activation. \
                                    Required when the target skill declares `paths` constraints."
                }
            },
            "required": ["name"]
        })
    }

    /// 执行技能激活：校验 `context_path` 是否满足技能声明的路径约束，
    /// 激活技能并将其内容渲染为提示词文本块，同时放开该技能允许使用的工具的可见性。
    fn execute_with_context<'a>(
        &'a self,
        parameters: ToolParameters,
        ctx: &'a ToolContext,
    ) -> BoxFuture<'a, Result<ToolResult>> {
        Box::pin(async move {
            let name = parameters
                .get("name")
                .and_then(|v| v.as_str())
                .ok_or_else(|| ToolError::MissingParameter("name".to_string()))?
                .to_string();

            let args: Vec<String> = parameters
                .get("arguments")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .split_whitespace()
                .filter(|s| !s.is_empty())
                .map(String::from)
                .collect();

            let context_path = parameters
                .get("context_path")
                .and_then(|v| v.as_str())
                .map(str::to_string);

            let registry = self.registry.read().await;

            if let Some(descriptor) = registry.get_descriptor(&name)
                && !descriptor.paths.is_empty()
            {
                let Some(path) = context_path.as_deref() else {
                    return Ok(ToolResult::error(format!(
                        "Skill '{}' requires a matching context_path because it declares activation paths: {}",
                        name,
                        descriptor.paths.join(", ")
                    )));
                };

                if !descriptor.matches_context_path(path) {
                    return Ok(ToolResult::error(format!(
                        "Skill '{}' cannot be activated for context_path '{}'; expected one of: {}",
                        name,
                        path,
                        descriptor.paths.join(", ")
                    )));
                }
            }

            let allowed_tools = registry
                .get_descriptor(&name)
                .map(|descriptor| descriptor.allowed_tools.clone())
                .unwrap_or_default();

            match registry
                .activate_with_args(
                    &name,
                    &args,
                    crate::skills::external::prompt_exec::SkillSource::Local,
                )
                .await
            {
                Ok(content) => {
                    let block = content.to_prompt_block();
                    let activated_tools = activate_allowed_tools(ctx, &allowed_tools);
                    let mut result = ToolResult::success_with_kind(
                        ToolResultKind::SkillActivation { name: name.clone() },
                        block,
                    );
                    result
                        .metadata
                        .insert("activated_tools".to_string(), activated_tools.join(","));
                    Ok(result)
                }
                Err(e) => Ok(ToolResult::error(format!(
                    "Failed to activate skill '{}': {}",
                    name, e
                ))),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skills::external::SkillLoader;

    /// 验证激活技能时，匹配 `allowed_tools` 规则的工具会被放开可见，其余工具保持隐藏。
    #[tokio::test]
    async fn activation_promotes_matching_tool_schemas() -> Result<()> {
        let root = std::env::temp_dir().join(format!(
            "echo-skill-activation-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        let skill_dir = root.join("git-skill");
        std::fs::create_dir_all(&skill_dir)?;
        std::fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: git-skill\ndescription: Inspect git state\nallowed-tools: git_*\n---\nUse git tools.\n",
        )?;
        let mut loader = SkillLoader::new();
        let descriptors = loader.discover_directory(&root).await?;
        let mut registry = SkillRegistry::new();
        for descriptor in descriptors {
            registry.register_descriptor(descriptor);
        }
        let registry = Arc::new(RwLock::new(registry));
        let tool = ActivateSkillTool::new(registry, vec!["git-skill".to_string()]);
        let visibility = Arc::new(echo_core::tools::ToolVisibilityState::with_available(
            ["activate_skill", "git_status", "shell"]
                .into_iter()
                .map(str::to_string)
                .collect(),
            ["activate_skill"].into_iter().map(str::to_string).collect(),
            ["activate_skill"].into_iter().map(str::to_string).collect(),
        ));
        let ctx = ToolContext {
            tool_visibility: Some(Arc::clone(&visibility)),
            ..Default::default()
        };

        let result = tool
            .execute_with_context(
                ToolParameters::from([(
                    "name".to_string(),
                    serde_json::Value::String("git-skill".to_string()),
                )]),
                &ctx,
            )
            .await?;

        assert!(result.success);
        assert!(visibility.is_visible("git_status"));
        assert!(!visibility.is_visible("shell"));
        assert_eq!(
            result.metadata.get("activated_tools").map(String::as_str),
            Some("git_status")
        );
        assert_eq!(
            result.kind,
            ToolResultKind::SkillActivation {
                name: "git-skill".to_string()
            }
        );
        std::fs::remove_dir_all(root)?;
        Ok(())
    }
}
