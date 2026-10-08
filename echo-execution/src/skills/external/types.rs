use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::path::PathBuf;

use glob;
use serde::{Deserialize, Serialize};

use echo_core::sandbox::IsolationLevel;

// -- 面向编程注册技能描述符的沙箱策略 --

/**
针对以编程方式注册的技能描述符的单技能沙箱隔离策略。
官方 `SKILL.md` 文件格式没有 sandbox 字段；
需要此策略的宿主可通过运行时 API 附加该策略。
*/
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SkillSandboxPolicy {
    /**
    所需的隔离级别。`None` 表示不强制（默认行为）。

    可选值：`none`、`process`、`os-sandbox`、`container`、`orchestrated`
    */
    pub isolation: Option<IsolationLevel>,

    /**
    沙箱内是否允许网络访问。
    当指定了 isolation 时，默认值为 `false`（拒绝）。
    */
    pub network: Option<bool>,

    /// 执行超时时间（秒）。`None` 表示不额外设置超时。
    #[serde(alias = "timeout")]
    pub timeout_secs: Option<u64>,

    /// 沙箱允许访问的文件路径。
    #[serde(default)]
    pub allowed_paths: Vec<PathBuf>,

    /// 沙箱禁止访问的文件路径。
    #[serde(default)]
    pub denied_paths: Vec<PathBuf>,
}

impl SkillSandboxPolicy {
    /// 该策略是否实际上会约束执行。
    pub fn is_constraining(&self) -> bool {
        self.isolation.is_some()
            || self.network.is_some()
            || self.timeout_secs.is_some()
            || !self.allowed_paths.is_empty()
            || !self.denied_paths.is_empty()
    }

    /// 是否允许网络访问。设置了 isolation 时默认为 `false`。
    pub fn network_allowed(&self) -> bool {
        self.network.unwrap_or(false)
    }
}

// -- 第一层：SkillDescriptor（目录元数据，每个技能约 50-100 tokens）--

/**
在发现阶段加载的轻量级技能元数据（第一层）。

与 [agentskills.io 规范](https://agentskills.io/specification) 对齐。
仅 `name` 和 `description` 为必填字段，其余字段均为可选。

会话启动时，agent 会基于所有已发现的描述符构建一份**目录**，
并注入到系统提示词中。每个描述符大约消耗 50-100 tokens，
因此即使有几十个技能，基础上下文占用依然很小。
*/
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillDescriptor {
    /**
    唯一的技能名称（kebab-case：小写字母、连字符，长度 1-64）。
    根据规范必须与其所在的父目录名一致。
    */
    pub name: String,

    /**
    人类可读的描述（最多 1024 字符）。
    应同时说明该技能**做什么**以及**何时使用**。
    */
    pub description: String,

    /// `SKILL.md` 文件的绝对路径。
    #[serde(skip)]
    pub location: PathBuf,

    /// SPDX 许可证标识符，或指向随附许可证文件的引用。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub license: Option<String>,

    /// 环境要求（目标产品、系统依赖包等）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compatibility: Option<String>,

    /// 任意键值对元数据（作者、版本、标签等）。
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, String>,

    /**
    用于分组卸载的来源标记（例如 `"plugin:my-plugin"`）。

    由插件集成方在从插件目录加载技能时设置，以便
    `SkillRegistry::unregister_by_source` 能在禁用/卸载时精确移除
    该插件的技能。用户加载的 / 内置技能（永远不会被分组卸载）
    该字段为 `None`。序列化时跳过此字段，以避免对前端消费者造成破坏性变更。
    */
    #[serde(skip, default)]
    pub source: Option<String>,

    /// 该技能可使用的预先批准的工具列表（在 SKILL.md 中以空格分隔）。
    #[serde(
        default,
        skip_serializing_if = "Vec::is_empty",
        alias = "allowed-tools"
    )]
    pub allowed_tools: Vec<String>,

    /// 内联命令使用的首选 shell：`"bash"`（默认）或 `"powershell"`。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shell: Option<String>,

    /// 用于条件激活的文件路径模式。
    /// 设置后，只有当用户涉及的文件匹配其中某个 glob 模式时
    /// （例如 `["*.py", "tests/**"]`），该技能才会被展示/激活。
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub paths: Vec<String>,

    /**
    用于技能路由的显式触发关键词/短语。
    当用户查询匹配任一触发词时，该技能即被激活。
    作为 `description` 字段的补充，以实现精确路由。
    */
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub triggers: Vec<String>,

    /// 用于拦截 agent 生命周期事件的钩子。
    #[serde(skip)]
    pub hooks: Option<crate::skills::hooks::HooksDefinition>,

    /**
    单技能沙箱隔离策略。
    设置后，该技能上下文内的工具执行将按此策略受到约束
    （隔离级别、网络、路径、超时）。
    */
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sandbox: Option<SkillSandboxPolicy>,

    /**
    必须先于该技能被激活的其他技能。
    框架会在 `activate_skill` 过程中自动激活这些依赖项。
    循环依赖会在发现阶段被检测出来并记录为警告。
    */
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub depends_on: Vec<String>,
}

impl SkillDescriptor {
    /**
    生成一行用于注入系统提示词的目录条目。

    格式：`- <name>: <description>`（可附带路径约束和触发词）
    */
    pub fn catalog_line(&self) -> String {
        let mut line = format!("- {}: {}", self.name, self.description);

        let mut annotations = Vec::new();
        if !self.paths.is_empty() {
            annotations.push(format!("activates for: {}", self.paths.join(", ")));
        }
        if !self.triggers.is_empty() {
            annotations.push(format!("triggers: {}", self.triggers.join(", ")));
        }
        if !self.depends_on.is_empty() {
            annotations.push(format!("depends: {}", self.depends_on.join(", ")));
        }

        if !annotations.is_empty() {
            line.push_str(&format!(" [{}]", annotations.join("; ")));
        }

        line
    }

    /**
    根据 agentskills.io 规范校验名称。
    返回警告列表（空列表表示有效）。
    */
    pub fn validate_name(&self) -> Vec<String> {
        let mut warnings = Vec::new();
        let name = &self.name;

        if name.is_empty() || name.len() > 64 {
            warnings.push(format!(
                "name '{}' length {} outside 1-64 range",
                name,
                name.len()
            ));
        }
        if name.starts_with('-') || name.ends_with('-') {
            warnings.push(format!("name '{}' must not start or end with hyphen", name));
        }
        if name.contains("--") {
            warnings.push(format!(
                "name '{}' must not contain consecutive hyphens",
                name
            ));
        }
        for ch in name.chars() {
            if !ch.is_ascii_lowercase() && !ch.is_ascii_digit() && ch != '-' {
                warnings.push(format!(
                    "name '{}' contains invalid character '{}' (only lowercase, digits, hyphens allowed)",
                    name, ch
                ));
                break;
            }
        }
        warnings
    }

    /**
    校验 `paths` 字段中的 glob 模式。
    对无效或可疑的模式返回警告列表。
    */
    pub fn validate_paths(&self) -> Vec<String> {
        let mut warnings = Vec::new();
        for pattern in &self.paths {
            // 拒绝模式中的路径穿越
            if pattern.contains("..") {
                warnings.push(format!(
                    "path pattern '{}' contains '..' which is not allowed",
                    pattern
                ));
                continue;
            }

            // 检查 glob 特殊字符是否配对
            let open_brace = pattern.matches('{').count();
            let close_brace = pattern.matches('}').count();
            if open_brace != close_brace {
                warnings.push(format!("path pattern '{}' has unbalanced braces", pattern));
            }

            let open_bracket = pattern.matches('[').count();
            let close_bracket = pattern.matches(']').count();
            if open_bracket != close_bracket {
                warnings.push(format!(
                    "path pattern '{}' has unbalanced brackets",
                    pattern
                ));
            }

            // 对过于宽泛的模式发出警告
            if pattern == "**" || pattern == "*" {
                warnings.push(format!(
                    "path pattern '{}' is overly broad and may match unintended files",
                    pattern
                ));
            }

            // 验证该模式能否被 glob crate 正确编译
            if let Err(e) = glob::Pattern::new(pattern) {
                warnings.push(format!(
                    "path pattern '{}' is not valid glob syntax: {}",
                    pattern, e
                ));
            }
        }
        warnings
    }

    /**
    检查涉及的文件路径是否满足该技能的条件激活规则。

    没有设置 `paths` 的技能始终视为匹配。
    */
    pub fn matches_context_path(&self, context_path: &str) -> bool {
        if self.paths.is_empty() {
            return true;
        }

        let normalized = context_path.replace('\\', "/");
        let trimmed = normalized.trim_start_matches("./");
        let file_name = Path::new(trimmed)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(trimmed);

        self.paths.iter().any(|pattern| {
            glob::Pattern::new(pattern).ok().is_some_and(|glob| {
                glob.matches(trimmed) || glob.matches(&normalized) || glob.matches(file_name)
            })
        })
    }

    /**
    检查该描述符是否允许使用指定名称的工具。

    `allowed_tools` 为空表示不受限制。匹配语义与 hook/tool 匹配保持一致，
    因此精确名称、glob 模式以及 `Bash` -> `Bash(git:*)` 前缀变体的行为都一致。
    */
    pub fn permits_tool(&self, tool_name: &str) -> bool {
        if self.allowed_tools.is_empty() {
            return true;
        }

        skill_allows_tool(&self.allowed_tools, tool_name)
    }
}

/**
即使某个技能缩小了其领域工具的可见范围，框架控制类工具依然保持可用。
如果没有这些工具，受限技能可能会导致无法完成收尾、
无法进一步发现、无法加载资源，或无法向用户提出明确问题。
*/
pub fn is_skill_control_tool(tool_name: &str) -> bool {
    matches!(
        tool_name,
        "final_answer"
            | "tool_search"
            | "activate_skill"
            | "read_skill_resource"
            | "run_skill_script"
            | "human_in_loop"
    )
}

/// 判断给定工具名是否被 `allowed_tools` 放行；框架控制类工具始终放行。
pub fn skill_allows_tool<'a, I>(allowed_tools: I, tool_name: &str) -> bool
where
    I: IntoIterator<Item = &'a String>,
{
    is_skill_control_tool(tool_name)
        || allowed_tools
            .into_iter()
            .any(|matcher| tool_matcher(matcher, tool_name))
}

/**
将工具名与匹配器模式进行匹配。

支持：
- 精确匹配：`"Read"` 匹配 `"Read"`
- 通配符：`"*"` 匹配所有工具
- Glob 模式：`"Bash(*)"` 匹配 `"Bash(git:status)"`
- 前缀匹配：`"Bash"` 匹配 `"Bash(git:status)"`
*/
pub fn tool_matcher(matcher: &str, tool_name: &str) -> bool {
    if matcher == "*" || matcher == tool_name {
        return true;
    }
    if let Ok(pattern) = glob::Pattern::new(matcher)
        && pattern.matches(tool_name)
    {
        return true;
    }
    tool_name.starts_with(matcher) && tool_name.as_bytes().get(matcher.len()).copied() == Some(b'(')
}

/**
一份已解析的 `SKILL.md` 文档，尚未进行运行时资源发现。

descriptor 是 frontmatter 在目录层面的投影，
而 `instructions` 则是结束分隔符之后的 Markdown 正文。
*/
#[derive(Debug, Clone)]
pub struct SkillDocument {
    descriptor: SkillDescriptor,
    instructions: String,
    source: String,
}

impl SkillDocument {
    /// 构造一个新的已解析技能文档。
    pub(crate) fn new(descriptor: SkillDescriptor, instructions: String, source: String) -> Self {
        Self {
            descriptor,
            instructions,
            source,
        }
    }

    /// 返回该文档的目录描述符。
    pub fn descriptor(&self) -> &SkillDescriptor {
        &self.descriptor
    }

    /// 返回该技能的 Markdown 正文说明。
    pub fn instructions(&self) -> &str {
        &self.instructions
    }

    /// 返回该文档的来源标识。
    pub fn source(&self) -> &str {
        &self.source
    }

    /// 消费该文档并取出其目录描述符。
    pub fn into_descriptor(self) -> SkillDescriptor {
        self.descriptor
    }

    /// 附加运行时来源信息，不改变已解析的 frontmatter 内容。
    pub fn set_registration_source(&mut self, source: impl Into<String>) {
        self.descriptor.source = Some(source.into());
    }

    /**
    使用新的标准 `allowed-tools` 值重新渲染该已解析文档。
    演化（evolution）与宿主工具使用这一规范化写入器，
    而不是另外再实现一个 YAML 解析器或旧字段归一化逻辑。
    */
    pub fn render_with_allowed_tools(
        &self,
        allowed_tools: &[String],
    ) -> std::result::Result<String, String> {
        #[derive(Serialize)]
        struct OfficialFrontmatter {
            /// 技能名称。
            name: String,
            /// 技能描述。
            description: String,
            /// SPDX 许可证标识符（可选）。
            #[serde(skip_serializing_if = "Option::is_none")]
            license: Option<String>,
            /// 环境兼容性要求（可选）。
            #[serde(skip_serializing_if = "Option::is_none")]
            compatibility: Option<String>,
            /// 任意键值对元数据。
            #[serde(skip_serializing_if = "BTreeMap::is_empty")]
            metadata: BTreeMap<String, String>,
            /// 以空格分隔的允许工具列表字符串。
            #[serde(rename = "allowed-tools", skip_serializing_if = "Option::is_none")]
            allowed_tools: Option<String>,
        }

        let frontmatter = OfficialFrontmatter {
            name: self.descriptor.name.clone(),
            description: self.descriptor.description.clone(),
            license: self.descriptor.license.clone(),
            compatibility: self.descriptor.compatibility.clone(),
            metadata: self.descriptor.metadata.clone().into_iter().collect(),
            allowed_tools: (!allowed_tools.is_empty()).then(|| allowed_tools.join(" ")),
        };
        let yaml = serde_yaml_ng::to_string(&frontmatter).map_err(|error| error.to_string())?;
        Ok(format!("---\n{yaml}---\n{}", self.instructions))
    }
}

// -- 第二层：SkillContent（完整说明内容，在激活时加载）--

/**
技能被激活时返回的完整技能内容（第二层）。

包含 `SKILL.md` 的 Markdown 正文（已剥离 frontmatter），
以及在 `scripts/`、`references/`、`assets/` 目录中发现的
随附资源文件清单。
*/
#[derive(Debug, Clone)]
pub struct SkillContent {
    /// 该技能的目录描述符。
    pub descriptor: SkillDescriptor,

    /**
    `SKILL.md` 的 Markdown 正文（frontmatter 之后的全部内容）。
    这是该技能的主要说明文本。
    */
    pub instructions: String,

    /// 在技能目录中发现的随附资源文件。
    pub resources: Vec<SkillResourceEntry>,
}

impl SkillContent {
    /**
    将技能内容格式化为用于注入 LLM 的结构化文本块。

    使用类 XML 标签，以便 agent 框架在上下文压缩期间
    能够识别出技能内容。
    */
    pub fn to_prompt_block(&self) -> String {
        let skill_dir = self
            .descriptor
            .location
            .parent()
            .map(|p| p.display().to_string())
            .unwrap_or_default();

        let mut block = format!(
            "<skill_content name=\"{}\">\n{}\n\nSkill directory: {}\n\
             Relative paths in this skill are relative to the skill directory.",
            self.descriptor.name,
            self.instructions.trim(),
            skill_dir,
        );

        if !self.descriptor.allowed_tools.is_empty() {
            block.push_str(&format!(
                "\n\n<allowed_tools>\nThis skill declares the following preferred/allowed tools: {}\n\
                 Runtime enforcement currently applies to the built-in skill tools such as \
                 read_skill_resource and run_skill_script.\n</allowed_tools>",
                self.descriptor.allowed_tools.join(", ")
            ));
        }

        if !self.resources.is_empty() {
            block.push_str("\n\n<skill_resources>");
            for res in &self.resources {
                block.push_str(&format!(
                    "\n  <file kind=\"{}\">{}</file>",
                    res.kind, res.relative_path
                ));
            }
            block.push_str("\n</skill_resources>");
        }

        block.push_str("\n</skill_content>");
        block
    }
}

// -- 第三层：资源条目 --

/// 在技能目录中发现的一个随附资源文件。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillResourceEntry {
    /// 相对于技能目录的路径（例如 `references/guide.md`）。
    pub relative_path: String,

    /// 资源的类型，根据其所在父目录名推断得出。
    pub kind: SkillResourceKind,
}

/// 随附资源文件的分类。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SkillResourceKind {
    /// 脚本文件（位于 `scripts/` 目录）。
    Script,
    /// 参考文档（位于 `references/` 目录）。
    Reference,
    /// 资源文件，如图片、模板等（位于 `assets/` 目录）。
    Asset,
    /// 其他未分类的文件。
    Other,
}

impl std::fmt::Display for SkillResourceKind {
    /// 将资源类型格式化为小写字符串（如 `"script"`）。
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Script => write!(f, "script"),
            Self::Reference => write!(f, "reference"),
            Self::Asset => write!(f, "asset"),
            Self::Other => write!(f, "other"),
        }
    }
}

// -- SKILL.md Frontmatter（原始反序列化目标）--

/**
从 `SKILL.md` 文件反序列化得到的原始 YAML frontmatter。

仅接受官方 agentskills.io 规范中的字段（`deny_unknown_fields`），
且 `metadata` 将字符串键映射为字符串值，与规范中
string-to-string 的约定保持一致。echo-agent 不使用任何私有
frontmatter 扩展字段：路由完全由 description 驱动，
单技能钩子也不属于文件格式的一部分。
*/
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RawFrontmatter {
    /// 技能名称。
    pub name: String,
    /// 技能描述。
    pub description: String,

    /// SPDX 许可证标识符（可选）。
    #[serde(default = "default_optional_string")]
    pub license: Option<String>,
    /// 环境兼容性要求（可选）。
    #[serde(default = "default_optional_string")]
    pub compatibility: Option<String>,
    /// 任意键值对元数据（可选）。
    #[serde(default = "default_optional_metadata")]
    pub metadata: Option<HashMap<String, String>>,
    /**
    官方格式：以空格分隔的单个字符串。YAML 序列形式以及
    旧版下划线拼写会被 serde 直接拒绝，而不是被静默归一化。
    */
    #[serde(default = "default_optional_string", rename = "allowed-tools")]
    pub allowed_tools: Option<String>,
}

/// serde 默认值辅助函数：返回空的可选字符串。
fn default_optional_string() -> Option<String> {
    None
}

/// serde 默认值辅助函数：返回空的可选元数据映射。
fn default_optional_metadata() -> Option<HashMap<String, String>> {
    None
}

impl RawFrontmatter {
    /**
    将已校验的 frontmatter 转换为目录元数据。

    [`SkillDescriptor`] 上的运行时扩展字段（`triggers`、
    `paths` 等）在标准格式中没有对应的文件来源；
    这些字段仅对以编程方式构造的描述符可用。
    */
    pub fn into_descriptor(self, location: PathBuf) -> SkillDescriptor {
        SkillDescriptor {
            source: None,
            name: self.name,
            description: self.description,
            location,
            license: self.license,
            compatibility: self.compatibility,
            metadata: self.metadata.unwrap_or_default(),
            allowed_tools: self
                .allowed_tools
                .map(|value| value.split_whitespace().map(str::to_string).collect())
                .unwrap_or_default(),
            shell: None,
            paths: Vec::new(),
            triggers: Vec::new(),
            // 单技能钩子在官方格式中没有对应的文件来源。
            hooks: None,
            sandbox: None,
            depends_on: Vec::new(),
        }
    }
}

// -- 测试 --

#[cfg(test)]
mod tests {
    use super::*;

    /// 验证合法的技能名称不会产生警告。
    #[test]
    fn test_descriptor_validate_name_valid() {
        let d = SkillDescriptor {
            source: None,
            name: "code-review".into(),
            description: "Review code".into(),
            location: PathBuf::new(),
            license: None,
            compatibility: None,
            metadata: HashMap::new(),
            allowed_tools: vec![],
            shell: None,
            paths: vec![],
            triggers: vec![],
            hooks: None,
            sandbox: None,
            depends_on: vec![],
        };
        assert!(d.validate_name().is_empty());
    }

    /// 验证各种不合法的技能名称都会产生警告。
    #[test]
    fn test_descriptor_validate_name_invalid() {
        let cases = vec![
            ("Code-Review", "uppercase"),
            ("-code", "starts with hyphen"),
            ("code-", "ends with hyphen"),
            ("code--review", "consecutive hyphens"),
            ("code_review", "underscore"),
        ];
        for (name, reason) in cases {
            let d = SkillDescriptor {
                source: None,
                name: name.into(),
                description: "test".into(),
                location: PathBuf::new(),
                license: None,
                compatibility: None,
                metadata: HashMap::new(),
                allowed_tools: vec![],
                shell: None,
                paths: vec![],
                triggers: vec![],
                hooks: None,
                sandbox: None,
                depends_on: vec![],
            };
            assert!(
                !d.validate_name().is_empty(),
                "expected warnings for {}: {}",
                name,
                reason
            );
        }
    }

    /// 验证目录行的格式化输出符合预期。
    #[test]
    fn test_descriptor_catalog_line() {
        let d = SkillDescriptor {
            source: None,
            name: "pdf-processing".into(),
            description: "Extract PDF text, fill forms.".into(),
            location: PathBuf::new(),
            license: None,
            compatibility: None,
            metadata: HashMap::new(),
            allowed_tools: vec![],
            shell: None,
            paths: vec![],
            triggers: vec![],
            hooks: None,
            sandbox: None,
            depends_on: vec![],
        };
        assert_eq!(
            d.catalog_line(),
            "- pdf-processing: Extract PDF text, fill forms."
        );
    }

    /// 验证技能内容能正确格式化为提示词注入块。
    #[test]
    fn test_skill_content_prompt_block() {
        let content = SkillContent {
            descriptor: SkillDescriptor {
                source: None,
                name: "test-skill".into(),
                description: "A test".into(),
                location: PathBuf::from("/home/user/skills/test-skill/SKILL.md"),
                license: None,
                compatibility: None,
                metadata: HashMap::new(),
                allowed_tools: vec![],
                shell: None,
                paths: vec![],
                triggers: vec![],
                hooks: None,
                sandbox: None,
                depends_on: vec![],
            },
            instructions: "# Instructions\n\nDo the thing.".into(),
            resources: vec![
                SkillResourceEntry {
                    relative_path: "scripts/run.py".into(),
                    kind: SkillResourceKind::Script,
                },
                SkillResourceEntry {
                    relative_path: "references/guide.md".into(),
                    kind: SkillResourceKind::Reference,
                },
            ],
        };
        let block = content.to_prompt_block();
        assert!(block.contains("<skill_content name=\"test-skill\">"));
        assert!(block.contains("# Instructions"));
        assert!(block.contains("<skill_resources>"));
        assert!(block.contains("scripts/run.py"));
        assert!(block.contains("</skill_content>"));
    }

    /// 验证资源类型的 Display 输出符合预期字符串。
    #[test]
    fn test_resource_kind_display() {
        assert_eq!(SkillResourceKind::Script.to_string(), "script");
        assert_eq!(SkillResourceKind::Reference.to_string(), "reference");
        assert_eq!(SkillResourceKind::Asset.to_string(), "asset");
        assert_eq!(SkillResourceKind::Other.to_string(), "other");
    }

    /// 验证原始 frontmatter 能正确转换为目录描述符。
    #[test]
    fn test_raw_frontmatter_into_descriptor() {
        let raw = RawFrontmatter {
            name: "my-skill".into(),
            description: "Does things".into(),
            license: Some("MIT".into()),
            compatibility: None,
            metadata: Some(HashMap::from([
                ("version".to_string(), "1.0.0".to_string()),
                ("author".to_string(), "team".to_string()),
                ("tags".to_string(), "code, review".to_string()),
            ])),
            allowed_tools: Some("Bash(git:*) Read".into()),
        };

        let desc = raw.into_descriptor(PathBuf::from("/skills/my-skill/SKILL.md"));
        assert_eq!(desc.name, "my-skill");
        assert_eq!(desc.license, Some("MIT".into()));
        assert_eq!(desc.allowed_tools, vec!["Bash(git:*)", "Read"]);
        assert_eq!(
            desc.metadata.get("version").map(String::as_str),
            Some("1.0.0")
        );
        assert_eq!(
            desc.metadata.get("author").map(String::as_str),
            Some("team")
        );
        assert_eq!(
            desc.metadata.get("tags").map(String::as_str),
            Some("code, review")
        );
    }

    /// 验证标准布局的 frontmatter 解析后，运行时扩展字段均为空。
    #[test]
    fn test_standard_metadata_keeps_extension_fields_empty() -> std::result::Result<(), String> {
        let content = r#"---
name: routed-skill
description: Standard layout; routing is description-driven.
metadata:
  category: automation
  sample-count: "7"
allowed-tools: shell read_file apply_patch
---
Body"#;
        let document = SkillDocument::parse(content).map_err(|error| error.to_string())?;
        let desc = document.descriptor();
        assert_eq!(
            desc.metadata.get("category").map(String::as_str),
            Some("automation")
        );
        assert_eq!(
            desc.metadata.get("sample-count").map(String::as_str),
            Some("7")
        );
        assert_eq!(
            desc.allowed_tools,
            vec!["shell", "read_file", "apply_patch"]
        );
        assert!(desc.triggers.is_empty());
        assert!(desc.paths.is_empty());
        assert!(desc.depends_on.is_empty());
        assert!(desc.shell.is_none());
        assert!(desc.sandbox.is_none());
        assert!(desc.hooks.is_none());
        Ok::<(), String>(())
    }

    /// 验证旧版顶层扩展字段会被拒绝解析。
    #[test]
    fn test_legacy_top_level_extension_fields_are_rejected() {
        /*
        旧版标准化之前的文件会在 frontmatter 顶层声明 echo-agent 语义；
        解析必须失败（fail closed），以便目录工具能发现未迁移的文件，
        而不是静默丢弃路由/沙箱意图。
        */
        for legacy_field in [
            "triggers:\n  - 周报\n",
            "shell: bash\n",
            "paths:\n  - \"*.pdf\"\n",
            "sandbox:\n  isolation: process\n",
            "depends_on:\n  - web-search\n",
        ] {
            let content =
                format!("---\nname: legacy\ndescription: Legacy layout.\n{legacy_field}---\nBody");
            assert!(
                SkillDocument::parse(&content).is_err(),
                "legacy top-level field must be rejected: {legacy_field}"
            );
        }
    }

    /// 验证 frontmatter 中出现 hooks 字段时解析会失败。
    #[test]
    fn test_frontmatter_hooks_are_rejected() {
        /*
        单技能钩子不属于官方文件格式的一部分；
        frontmatter 中的 hooks 块是未知的官方字段，必须解析失败。
        */
        let content = "---\nname: hooked\ndescription: Hooked.\nhooks:\n  Stop:\n    - hooks:\n        - type: command\n          command: echo done\n---\nBody";
        assert!(SkillDocument::parse(content).is_err());
    }

    /// 验证非标量（嵌套）的 metadata 会被拒绝解析。
    #[test]
    fn test_non_scalar_metadata_is_rejected() {
        /*
        规范将 metadata 定义为字符串键到字符串值的映射；
        嵌套映射（包括任何供应商命名空间）会导致解析失败，
        而不是被静默地重新整形。
        */
        for nested in [
            "metadata:\n  echo-agent:\n    triggers: [周报]\n",
            "metadata:\n  vendor:\n    nested: true\n",
        ] {
            let content = format!("---\nname: nested\ndescription: Nested.\n{nested}---\nBody");
            assert!(
                SkillDocument::parse(&content).is_err(),
                "non-scalar metadata must be rejected: {nested}"
            );
        }
    }

    /// 验证 allowed-tools 字段必须为字符串形式，而非列表。
    #[test]
    fn test_allowed_tools_must_be_string() {
        let content = "---\nname: list-tools\ndescription: Invalid list form.\nallowed-tools: [Read, Write]\n---\nBody";
        assert!(SkillDocument::parse(content).is_err());
        let content = "---\nname: underscore-tools\ndescription: Invalid alias form.\nallowed_tools: Read Write\n---\nBody";
        assert!(SkillDocument::parse(content).is_err());
    }

    /// 测试辅助函数：构造一个带指定 `paths` 的技能描述符。
    fn make_desc_with_paths(paths: Vec<&str>) -> SkillDescriptor {
        SkillDescriptor {
            source: None,
            name: "test".into(),
            description: "test".into(),
            location: PathBuf::new(),
            license: None,
            compatibility: None,
            metadata: HashMap::new(),
            allowed_tools: vec![],
            shell: None,
            paths: paths.into_iter().map(String::from).collect(),
            triggers: vec![],
            hooks: None,
            sandbox: None,
            depends_on: vec![],
        }
    }

    /// 验证合法的 glob 路径模式不会产生警告。
    #[test]
    fn test_validate_paths_good() {
        let d = make_desc_with_paths(vec!["*.py", "src/**/*.rs", "tests/**"]);
        let warnings = d.validate_paths();
        // 不应出现路径穿越或语法错误
        assert!(!warnings.iter().any(|w| w.contains("'..'")));
        assert!(!warnings.iter().any(|w| w.contains("not valid glob")));
    }

    /// 验证包含路径穿越（`..`）的模式会产生警告。
    #[test]
    fn test_validate_paths_traversal() {
        let d = make_desc_with_paths(vec!["../secret.txt"]);
        let warnings = d.validate_paths();
        assert!(warnings.iter().any(|w| w.contains("'..'")));
    }

    /// 验证括号不配对的 glob 模式会产生警告。
    #[test]
    fn test_validate_paths_unbalanced_braces() {
        let d = make_desc_with_paths(vec!["*.py", "{foo,bar"]);
        let warnings = d.validate_paths();
        assert!(warnings.iter().any(|w| w.contains("unbalanced")));
    }

    /// 验证过于宽泛的模式（如 `**`）会产生警告。
    #[test]
    fn test_validate_paths_overly_broad() {
        let d = make_desc_with_paths(vec!["**"]);
        let warnings = d.validate_paths();
        assert!(warnings.iter().any(|w| w.contains("overly broad")));
    }

    /// 验证路径匹配逻辑能正确识别命中与未命中的情况。
    #[test]
    fn test_matches_context_path() {
        let d = make_desc_with_paths(vec!["*.py", "tests/**"]);
        assert!(d.matches_context_path("main.py"));
        assert!(d.matches_context_path("./tests/unit/test_demo.rs"));
        assert!(!d.matches_context_path("src/main.rs"));
    }

    /// 验证工具许可判断逻辑，包括控制类工具始终放行。
    #[test]
    fn test_permits_tool() {
        let mut d = make_desc_with_paths(vec![]);
        d.allowed_tools = vec!["read_skill_resource".into(), "Bash(*)".into()];
        assert!(d.permits_tool("read_skill_resource"));
        assert!(d.permits_tool("Bash(git:status)"));
        assert!(d.permits_tool("run_skill_script"));
        assert!(d.permits_tool("final_answer"));
        assert!(!d.permits_tool("write_file"));
    }
}
