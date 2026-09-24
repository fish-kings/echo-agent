//! 技能加载器 —— 多作用域发现，并遵循 agentskills.io 规范进行解析。
//!
//! 支持 [agentskills.io](https://agentskills.io/specification) 的目录约定:
//! 仅接受官方 frontmatter 字段;路由完全由 description 驱动,
//! 单技能文件不定义私有钩子扩展。
//!
//! # 发现作用域
//!
//! | 作用域 | 扫描路径 |
//! |-------|--------------|
//! | 项目级 | `./skills/`、`./.agents/skills/` |
//! | 用户级 | `~/.agents/skills/` |
//! | 自定义 | 任意用户指定路径 |
//!
//! 当名称冲突时,项目级技能会覆盖用户级技能。

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use futures::future::BoxFuture;
use serde_yaml_ng::Value;
use tracing::{debug, info, warn};

use echo_core::error::{ReactError, Result};

use super::types::{RawFrontmatter, SkillDescriptor, SkillDocument};
const SKILL_FILE: &str = "SKILL.md";
const MAX_SCAN_DEPTH: usize = 4;

/// 扫描过程中需要跳过的目录名。
const SKIP_DIRS: &[&str] = &[
    ".git",
    "node_modules",
    "target",
    "__pycache__",
    ".venv",
    "dist",
    "build",
];

// -- DiscoveryScope --

/// 扫描技能的位置范围。
#[derive(Debug, Clone)]
pub enum DiscoveryScope {
    /// 项目级: `<root>/skills/` 与 `<root>/.agents/skills/`
    Project(PathBuf),
    /// 用户级: `~/.agents/skills/`
    User,
    /// 自定义路径(按原样扫描)
    Custom(PathBuf),
}

/// 由产品方提供的权威判定,用于决定已发现的技能是否可以被加载。
///
/// 框架本身刻意不了解产品的生命周期文件。调用方可以基于自己的
/// 启用/禁用状态或审核状态实现该 trait,而加载器本身仍是一个
/// 可复用的发现原语。
pub trait SkillLoadPolicy: Send + Sync {
    /// 当该描述符可以进入运行时目录时返回 `true`。
    ///
    /// 该判定是异步的,这样嵌入方应用或来源 SDK 可以作为实时的
    /// 策略权威,而不会阻塞 Agent 运行时。
    fn allows<'a>(&'a self, descriptor: &'a SkillDescriptor) -> BoxFuture<'a, bool>;
}

// -- SkillLoader --

/// 支持多作用域发现、并遵循 agentskills.io 规范解析的技能加载器。
///
/// # 解析行为
///
/// - **当前格式**: YAML frontmatter(`name`、`description` 为必填),
///   Markdown 正文即为说明文本。
/// - **严格文件校验**: 未知字段、类型错误、名称不匹配以及描述为空
///   都会在注册前拒绝该文档。
pub struct SkillLoader {
    /// 已发现的描述符,以技能名称为键。
    descriptors: HashMap<String, SkillDescriptor>,
    /// 已发现的完整已解析文档,以技能名称为键。
    documents: HashMap<String, SkillDocument>,
    /// 已准备好的身份文档列表(文件路径与来源内容),以技能名称为键。
    prepared_identity_documents: HashMap<String, Vec<(PathBuf, String)>>,
    /// 发现过程中产生的诊断信息(如解析失败、被拒绝的私有扩展等)。
    discovery_diagnostics: Vec<SkillDiscoveryDiagnostic>,
    /// 解析完成后、注册之前所咨询的可选权威策略。
    policy: Option<Arc<dyn SkillLoadPolicy>>,
    /// 在解析 SKILL.md 元数据与正文之前应用的可选插件变量上下文。
    plugin_variables: Option<echo_core::plugin::PluginVariables>,
}

impl SkillLoader {
    /// 构造一个空的技能加载器。
    pub fn new() -> Self {
        Self {
            descriptors: HashMap::new(),
            documents: HashMap::new(),
            prepared_identity_documents: HashMap::new(),
            discovery_diagnostics: Vec::new(),
            policy: None,
            plugin_variables: None,
        }
    }

    /// 安装一个由产品方拥有的技能加载策略。
    pub fn with_policy(mut self, policy: Arc<dyn SkillLoadPolicy>) -> Self {
        self.policy = Some(policy);
        self
    }

    /// 在发现过程中,对技能元数据与钩子应用插件变量。
    pub fn with_plugin_variables(mut self, variables: echo_core::plugin::PluginVariables) -> Self {
        self.plugin_variables = Some(variables);
        self
    }

    /// 从多个作用域发现技能。
    ///
    /// 返回所有成功解析的 `SkillDescriptor`。名称冲突按顺序解决:
    /// 更靠前的作用域优先。当某个技能被遮蔽时会记录警告日志。
    pub async fn discover(&mut self, scopes: &[DiscoveryScope]) -> Result<Vec<SkillDescriptor>> {
        let mut results = Vec::new();

        for scope in scopes {
            let dirs = scope_to_dirs(scope);
            for dir in dirs {
                if !tokio::fs::metadata(&dir)
                    .await
                    .is_ok_and(|metadata| metadata.is_dir())
                {
                    debug!(
                        "Skill directory does not exist, skipping: {}",
                        dir.display()
                    );
                    continue;
                }
                let found = self.scan_directory(&dir, 0, true).await?;
                for (document, identity_documents) in found {
                    let desc = document.descriptor().clone();
                    if let Some(policy) = &self.policy
                        && !policy.allows(&desc).await
                    {
                        info!(
                            skill = %desc.name,
                            path = %desc.location.display(),
                            "Skill excluded by load policy"
                        );
                        continue;
                    }
                    if let Some(existing) = self.descriptors.get(&desc.name) {
                        warn!(
                            "Skill '{}' at '{}' shadowed by existing at '{}'",
                            desc.name,
                            desc.location.display(),
                            existing.location.display()
                        );
                    } else {
                        let name = desc.name.clone();
                        self.descriptors.insert(name.clone(), desc.clone());
                        self.documents.insert(name.clone(), document);
                        self.prepared_identity_documents
                            .insert(name, identity_documents);
                        results.push(desc);
                    }
                }
            }
        }

        info!("Skill discovery complete: {} skills found", results.len());

        // 校验依赖关系并进行拓扑排序
        validate_and_sort_dependencies(&results);

        Ok(results)
    }

    /// 仅从直接子目录中发现 Agent Plugin 技能。
    ///
    /// Agent Plugins 1.0 将组件固定在 `skills/<name>/SKILL.md`,
    /// 不会递归搜索分类目录。
    pub async fn discover_agent_plugin_skills(
        &mut self,
        dir: impl Into<PathBuf>,
    ) -> Result<Vec<SkillDescriptor>> {
        let dir = dir.into();
        if !tokio::fs::metadata(&dir)
            .await
            .is_ok_and(|metadata| metadata.is_dir())
        {
            return Ok(Vec::new());
        }
        let found = self.scan_directory(&dir, 0, false).await?;
        let mut results = Vec::new();
        for (document, identity_documents) in found {
            let desc = document.descriptor().clone();
            if let Some(policy) = &self.policy
                && !policy.allows(&desc).await
            {
                info!(
                    skill = %desc.name,
                    path = %desc.location.display(),
                    "Skill excluded by load policy"
                );
                continue;
            }
            if let Some(existing) = self.descriptors.get(&desc.name) {
                warn!(
                    "Skill '{}' at '{}' shadowed by existing at '{}'",
                    desc.name,
                    desc.location.display(),
                    existing.location.display()
                );
                continue;
            }
            let name = desc.name.clone();
            self.descriptors.insert(name.clone(), desc.clone());
            self.documents.insert(name.clone(), document);
            self.prepared_identity_documents
                .insert(name, identity_documents);
            results.push(desc);
        }
        validate_and_sort_dependencies(&results);
        Ok(results)
    }

    /// 在一个显式指定的目录下发现技能。
    pub async fn discover_directory(
        &mut self,
        dir: impl Into<PathBuf>,
    ) -> Result<Vec<SkillDescriptor>> {
        self.discover(&[DiscoveryScope::Custom(dir.into())]).await
    }

    /// 扫描单个目录,查找 SKILL.md 文件。
    async fn scan_directory(
        &mut self,
        dir: &Path,
        depth: usize,
        recursive: bool,
    ) -> Result<Vec<(SkillDocument, Vec<(PathBuf, String)>)>> {
        if depth > MAX_SCAN_DEPTH {
            return Ok(vec![]);
        }

        let mut found = Vec::new();

        let mut entries = tokio::fs::read_dir(dir).await.map_err(|e| {
            ReactError::Other(format!("Cannot read directory '{}': {}", dir.display(), e))
        })?;

        while let Some(entry) = entries
            .next_entry()
            .await
            .map_err(|e| ReactError::Other(format!("Error reading directory entry: {}", e)))?
        {
            let path = entry.path();
            if !entry.file_type().await.is_ok_and(|kind| kind.is_dir()) {
                continue;
            }

            let dir_name = match path.file_name().and_then(|n| n.to_str()) {
                Some(n) => n.to_string(),
                None => continue,
            };

            if SKIP_DIRS.contains(&dir_name.as_str()) {
                continue;
            }

            let skill_file = path.join(SKILL_FILE);
            if tokio::fs::metadata(&skill_file)
                .await
                .is_ok_and(|metadata| metadata.is_file())
            {
                let private_hook_file = path.join("hooks.json");
                if tokio::fs::metadata(&private_hook_file)
                    .await
                    .is_ok_and(|metadata| metadata.is_file())
                {
                    self.discovery_diagnostics.push(SkillDiscoveryDiagnostic {
                        path: private_hook_file.clone(),
                        message: "hooks.json is not part of the official Agent Skills file format"
                            .to_string(),
                        is_error: true,
                    });
                    warn!(
                        "Skipping skill '{}' because hooks.json is a private extension",
                        path.display()
                    );
                    continue;
                }
                match parse_skill_file_with_variables(
                    &skill_file,
                    &dir_name,
                    self.plugin_variables.as_ref(),
                )
                .await
                {
                    Ok(document) => {
                        let source = document.source().to_string();
                        info!(
                            "Discovered skill '{}' at {}",
                            document.descriptor().name,
                            skill_file.display()
                        );
                        found.push((document, vec![(skill_file.clone(), source)]));
                    }
                    Err(e) => {
                        self.discovery_diagnostics.push(SkillDiscoveryDiagnostic {
                            path: skill_file.clone(),
                            message: e.to_string(),
                            is_error: true,
                        });
                        warn!(
                            "Failed to parse '{}', skipping: {}",
                            skill_file.display(),
                            e
                        );
                    }
                }
            } else if recursive {
                // 该直接子目录下没有 SKILL.md:递归进入以支持嵌套分类
                // 布局(skills/<category>/<name>/SKILL.md)。
                // 如果没有这一步,只能发现扁平的 skills/<name>/SKILL.md。
                // MAX_SCAN_DEPTH 用于限制递归深度,避免无限循环/过大的目录树。
                // 使用 Box::pin 是因为 async fn 不能直接递归调用自身(Rust 的限制)。
                let nested = Box::pin(self.scan_directory(&path, depth + 1, true)).await?;
                found.extend(nested);
            }
        }

        Ok(found)
    }

    /// 按名称获取一个描述符。
    pub fn get_descriptor(&self, name: &str) -> Option<&SkillDescriptor> {
        self.descriptors.get(name)
    }

    /// 列出所有已发现的描述符。
    pub fn list_descriptors(&self) -> Vec<&SkillDescriptor> {
        let mut descs: Vec<&SkillDescriptor> = self.descriptors.values().collect();
        descs.sort_by_key(|d| &d.name);
        descs
    }

    /// 消费该加载器并返回所有描述符。
    pub fn into_descriptors(self) -> Vec<SkillDescriptor> {
        let mut descs: Vec<SkillDescriptor> = self.descriptors.into_values().collect();
        descs.sort_by(|a, b| a.name.cmp(&b.name));
        descs
    }

    /// 已发现的技能数量。
    pub fn skill_count(&self) -> usize {
        self.descriptors.len()
    }

    /// 按名称获取已解析的完整技能文档。
    pub fn get_document(&self, name: &str) -> Option<&SkillDocument> {
        self.documents.get(name)
    }

    /// 按名称获取该技能已准备好的身份文档列表(文件路径与来源内容)。
    pub fn get_prepared_identity_documents(&self, name: &str) -> Option<&[(PathBuf, String)]> {
        self.prepared_identity_documents
            .get(name)
            .map(Vec::as_slice)
    }

    /// 返回发现过程中产生的所有诊断信息。
    pub fn discovery_diagnostics(&self) -> &[SkillDiscoveryDiagnostic] {
        &self.discovery_diagnostics
    }
}

/// 一条技能发现诊断信息,记录解析失败或被拒绝的文件及其原因。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkillDiscoveryDiagnostic {
    /// 触发该诊断信息的文件路径。
    pub path: PathBuf,
    /// 诊断信息的描述文本。
    pub message: String,
    /// 是否为错误级别(而非普通警告)。
    pub is_error: bool,
}

impl Default for SkillLoader {
    /// 构造一个默认(空)的技能加载器,等价于 [`SkillLoader::new`]。
    fn default() -> Self {
        Self::new()
    }
}

// -- 解析 --

/// 将单个 SKILL.md 文件解析为目录元数据及其已准备好的文档。
///
/// 强制执行 agentskills.io 校验中与文件格式相关的部分:
/// - 名称必须与所在的技能目录名一致
/// - 名称必须满足小写 kebab-case 约束
/// - 描述必须存在且非空
/// - 未知字段和非法 YAML 会被拒绝
///
async fn parse_skill_file_with_variables(
    path: &Path,
    parent_dir_name: &str,
    variables: Option<&echo_core::plugin::PluginVariables>,
) -> Result<SkillDocument> {
    let content = tokio::fs::read_to_string(path)
        .await
        .map_err(|e| ReactError::Other(format!("Failed to read '{}': {}", path.display(), e)))?;
    let content = match variables {
        Some(variables) => variables.substitute(&content),
        None => content,
    };

    let location = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    let document = SkillDocument::parse_at(&content, location)?;
    let descriptor = document.descriptor();

    if descriptor.name != parent_dir_name {
        return Err(ReactError::Other(format!(
            "Skill name '{}' must match directory '{}' per agentskills.io",
            descriptor.name, parent_dir_name
        )));
    }

    Ok(document)
}

impl SkillDocument {
    /// 解析并校验一份内存中的 `SKILL.md` 文档。
    pub fn parse(content: &str) -> Result<Self> {
        Self::parse_at(content, PathBuf::new())
    }

    /// 解析并校验一份带有来源位置的 `SKILL.md` 文档。
    pub fn parse_at(content: &str, location: impl Into<PathBuf>) -> Result<Self> {
        let (raw, instructions) = parse_document(content)?;
        let descriptor = raw.into_descriptor(location.into());
        validate_official_descriptor(&descriptor)?;
        Ok(Self::new(descriptor, instructions, content.to_string()))
    }
}

/// 校验所有基于文件的入口共用的官方 agentskills.io 限制。
/// 将这些检查集中在此处,可以防止运行时发现、SkillsHub、
/// 安装流程以及独立校验器各自接受不同的格式。
fn validate_official_descriptor(descriptor: &SkillDescriptor) -> Result<()> {
    let name_errors = descriptor.validate_name();
    if !name_errors.is_empty() {
        return Err(ReactError::Other(format!(
            "Skill '{}' violates agentskills.io name rules: {}",
            descriptor.name,
            name_errors.join("; ")
        )));
    }
    if descriptor.description.trim().is_empty() {
        return Err(ReactError::Other(
            "SKILL.md description is empty (required per spec)".to_string(),
        ));
    }
    if descriptor.description.chars().count() > 1024 {
        return Err(ReactError::Other(
            "SKILL.md description exceeds the 1024-character limit".to_string(),
        ));
    }
    if descriptor
        .compatibility
        .as_ref()
        .is_some_and(|value| value.chars().count() > 500)
    {
        return Err(ReactError::Other(
            "SKILL.md compatibility exceeds the 500-character limit".to_string(),
        ));
    }
    Ok(())
}

/// 将 SKILL.md 原始文本拆分为原始 frontmatter 与 Markdown 正文两部分。
fn parse_document(content: &str) -> Result<(RawFrontmatter, String)> {
    if !content.starts_with("---") {
        return Err(ReactError::Other(
            "SKILL.md must begin with YAML frontmatter (---)".to_string(),
        ));
    }
    let trimmed = content;

    // 跳过起始的 --- 以及其后的换行符
    let after_open = trimmed
        .get(3..)
        .unwrap_or("")
        .trim_start_matches('\r')
        .trim_start_matches('\n');

    // 查找结束的 ---,它必须独占一行。
    // 这可以防止 Markdown 的水平分隔线(例如文档中间出现的 `---`)
    // 被误认为是 frontmatter 的结束符。
    // 结束的 --- 必须出现在某一行的行首。
    let close_idx = after_open
        .find("\n---")
        .ok_or_else(|| ReactError::Other("SKILL.md frontmatter missing closing ---".to_string()))?;

    // 校验结束的 --- 确实位于行首(而非行中间)
    let yaml_str = after_open.get(..close_idx).unwrap_or_default();

    // 确保结束的 --- 所在行没有多余内容
    // (--- 之后应当只跟随空白字符、\r 或 \n)
    let after_close_start = after_open
        .get(close_idx.saturating_add(4)..)
        .unwrap_or_default(); // 跳过 "\n---"
    // "---" 之后第一个非空白字符应当是 Markdown 正文或文件结尾
    // 如果 "---" 同一行还有其他文本,说明它不是一个合法的分隔符
    let close_line_end = after_close_start
        .find('\n')
        .unwrap_or(after_close_start.len());
    let close_line_remainder = after_close_start.get(..close_line_end).unwrap_or_default();
    if !close_line_remainder.trim().is_empty() {
        return Err(ReactError::Other(
            "SKILL.md frontmatter closing --- has trailing content on same line".to_string(),
        ));
    }

    validate_official_frontmatter_types(yaml_str)?;

    let raw = serde_yaml_ng::from_str(yaml_str)
        .map_err(|e| ReactError::Other(format!("SKILL.md YAML parse error: {}", e)))?;
    let instructions = after_close_start
        .get(close_line_end..)
        .unwrap_or_default()
        .trim_start_matches('\r')
        .trim_start_matches('\n')
        .to_string();
    Ok((raw, instructions))
}

/// 在 serde 的 `Option<T>` 默认值机制抹去"字段缺省"与"显式非法值"
/// 之间的区别之前,拒绝那些存在但为 null 或其他非字符串类型的值。
fn validate_official_frontmatter_types(yaml: &str) -> Result<()> {
    let value: Value = serde_yaml_ng::from_str(yaml)
        .map_err(|error| ReactError::Other(format!("SKILL.md YAML parse error: {error}")))?;
    let Some(mapping) = value.as_mapping() else {
        return Err(ReactError::Other(
            "SKILL.md frontmatter must be a YAML mapping".to_string(),
        ));
    };

    for field in [
        "name",
        "description",
        "license",
        "compatibility",
        "allowed-tools",
    ] {
        let key = Value::String(field.to_string());
        if let Some(value) = mapping.get(&key)
            && !matches!(value, Value::String(_))
        {
            return Err(ReactError::Other(format!(
                "SKILL.md frontmatter field '{field}' must be a string"
            )));
        }
    }

    if let Some(metadata) = mapping.get(Value::String("metadata".to_string())) {
        let Some(metadata) = metadata.as_mapping() else {
            return Err(ReactError::Other(
                "SKILL.md frontmatter field 'metadata' must be a mapping".to_string(),
            ));
        };
        for (key, value) in metadata {
            if key.as_str().is_none() || !matches!(value, Value::String(_)) {
                return Err(ReactError::Other(
                    "SKILL.md metadata keys and values must be strings".to_string(),
                ));
            }
        }
    }
    Ok(())
}

// -- 依赖校验 --

/// 校验技能依赖关系,并对发现的问题记录警告日志。
///
/// 在所有已发现的技能之间执行基于 DFS 的环检测。
/// 循环依赖和缺失依赖都会被记录为警告,但不会阻止技能加载
/// (它们会在激活阶段被处理)。
fn validate_and_sort_dependencies(skills: &[SkillDescriptor]) {
    let name_set: HashSet<&str> = skills.iter().map(|s| s.name.as_str()).collect();

    // 检查是否存在缺失的依赖
    for skill in skills {
        for dep in &skill.depends_on {
            if !name_set.contains(dep.as_str()) {
                warn!(
                    "Skill '{}' depends on '{}' which is not available",
                    skill.name, dep
                );
            }
        }
    }

    // 通过 DFS 检测循环依赖
    let skill_deps: HashMap<&str, &[String]> = skills
        .iter()
        .map(|s| (s.name.as_str(), s.depends_on.as_slice()))
        .collect();

    let mut visited: HashSet<&str> = HashSet::new();
    let mut temp_visited: HashSet<&str> = HashSet::new();

    for skill in skills {
        detect_cycle(
            skill.name.as_str(),
            &skill_deps,
            &mut visited,
            &mut temp_visited,
        );
    }
}

/// 基于 DFS 的环检测。如果检测到环,记录警告日志并跳过该成环的边。
fn detect_cycle<'a>(
    name: &'a str,
    deps: &HashMap<&'a str, &'a [String]>,
    visited: &mut HashSet<&'a str>,
    temp_visited: &mut HashSet<&'a str>,
) {
    if visited.contains(name) {
        return;
    }
    if temp_visited.contains(name) {
        warn!("Circular dependency detected involving skill: {}", name);
        return;
    }

    temp_visited.insert(name);

    if let Some(skill_deps) = deps.get(name) {
        for dep in *skill_deps {
            detect_cycle(dep.as_str(), deps, visited, temp_visited);
        }
    }

    temp_visited.remove(name);
    visited.insert(name);
}

// -- 作用域解析 --

/// 将一个 `DiscoveryScope` 解析为要扫描的具体目录路径列表。
fn scope_to_dirs(scope: &DiscoveryScope) -> Vec<PathBuf> {
    match scope {
        DiscoveryScope::Project(root) => {
            vec![root.join("skills"), root.join(".agents").join("skills")]
        }
        DiscoveryScope::User => {
            if let Some(home) = dirs::home_dir() {
                vec![home.join(".agents").join("skills")]
            } else {
                warn!("Cannot determine home directory for user-level skill discovery");
                vec![]
            }
        }
        DiscoveryScope::Custom(path) => {
            vec![path.clone()]
        }
    }
}

// -- 测试 --

#[cfg(test)]
mod tests {
    use super::*;

    /// 验证标准的 SKILL.md 文档能被正确解析,包括名称、许可证及正文说明。
    #[test]
    fn test_skill_document_standard() -> std::result::Result<(), String> {
        let content = r#"---
name: pdf-processing
description: Extract PDF text, fill forms, merge files. Use when handling PDFs.
license: Apache-2.0
metadata:
  author: example-org
  version: "1.0"
---

# PDF Processing

Instructions here.
"#;
        let document = SkillDocument::parse(content).map_err(|error| error.to_string())?;
        assert_eq!(document.descriptor().name, "pdf-processing");
        assert_eq!(document.descriptor().license, Some("Apache-2.0".into()));
        assert_eq!(
            document.instructions().trim(),
            "# PDF Processing\n\nInstructions here."
        );
        Ok(())
    }

    /// 验证已移除的旧版 echo-agent 专有字段(version、author、tags 等)
    /// 会导致解析失败。
    #[test]
    fn test_skill_document_rejects_removed_echo_agent_fields() {
        let content = r#"---
name: code_review
version: "1.0.0"
description: "Code review skill"
author: "team"
tags: [code, review]
instructions: |
  Review the code carefully.
resources:
  - name: checklist
    path: checklist.md
    description: "Review checklist"
---
"#;
        assert!(SkillDocument::parse(content).is_err());
    }

    /// 验证空的 description 字段会导致解析失败。
    #[test]
    fn test_skill_document_rejects_empty_description() {
        let content = "---\nname: test\ndescription: \"\"\n---\n";
        assert!(SkillDocument::parse(content).is_err());
    }

    /// 验证官方规定的长度限制及字段类型约束(description/compatibility 长度、
    /// license/compatibility/allowed-tools/metadata 字段的类型)均被强制执行。
    #[test]
    fn test_skill_document_enforces_official_limits_and_types() {
        let long_description = "x".repeat(1025);
        let content = format!("---\nname: test\ndescription: {long_description}\n---\nBody");
        assert!(SkillDocument::parse(&content).is_err());

        let long_compatibility = "x".repeat(501);
        let content = format!(
            "---\nname: test\ndescription: Test\ncompatibility: {long_compatibility}\n---\nBody"
        );
        assert!(SkillDocument::parse(&content).is_err());

        for field in ["license", "compatibility", "allowed-tools"] {
            let content = format!("---\nname: test\ndescription: Test\n{field}: null\n---\nBody");
            assert!(SkillDocument::parse(&content).is_err());
        }
        assert!(
            SkillDocument::parse("---\nname: test\ndescription: Test\nmetadata: null\n---\nBody")
                .is_err()
        );
    }

    /// 验证缺少 frontmatter 的文档会导致解析失败。
    #[test]
    fn test_skill_document_rejects_missing_frontmatter() {
        let content = "# Just markdown";
        assert!(SkillDocument::parse(content).is_err());
    }

    /// 验证 frontmatter 未正确关闭时解析会失败。
    #[test]
    fn test_skill_document_rejects_unclosed_frontmatter() {
        let content = "---\nname: test\ndescription: Test\n";
        assert!(SkillDocument::parse(content).is_err());
    }

    /// 验证 frontmatter 之后的 Markdown 正文会被正确用作说明文本。
    #[test]
    fn test_skill_document_uses_markdown_body_as_instructions() -> std::result::Result<(), String> {
        let content = "---\nname: test\ndescription: Test\n---\n\n# Instructions\n\nDo stuff.";
        let document = SkillDocument::parse(content).map_err(|error| error.to_string())?;
        assert_eq!(document.instructions(), "# Instructions\n\nDo stuff.");
        Ok(())
    }

    /// 验证 scan_directory 递归:嵌套布局 skills/<category>/<name>/SKILL.md
    /// 必须能被发现(B2 修复前只能找到扁平 skills/<name>/SKILL.md)。
    #[tokio::test]
    async fn scan_directory_finds_nested_category_skills()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        // 用 std 临时目录避免为单个测试增加 dev-dependency。用进程 id + 原子计数
        // 保证唯一,测试结束清理。
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let uid = COUNTER.fetch_add(1, Ordering::SeqCst);
        let root = std::env::temp_dir().join(format!(
            "echo_b2_test_{}_{}_{}",
            std::process::id(),
            uid,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        // cleanup guard
        struct Guard(std::path::PathBuf);
        impl Drop for Guard {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
        let _guard = Guard(root.clone());

        // 扁平技能(直接 root/<name>/SKILL.md)
        let flat = root.join("coding");
        std::fs::create_dir_all(&flat)?;
        std::fs::write(
            flat.join("SKILL.md"),
            "---\nname: coding\ndescription: flat skill\n---\nbody",
        )?;

        // 嵌套技能(root/<category>/<name>/SKILL.md)——B2 修复前扫不到
        let nested = root.join("methodology").join("brainstorming");
        std::fs::create_dir_all(&nested)?;
        std::fs::write(
            nested.join("SKILL.md"),
            "---\nname: brainstorming\ndescription: nested skill\n---\nbody",
        )?;
        // 第二个嵌套,确保多个都能扫到
        let nested2 = root.join("methodology").join("writing-plans");
        std::fs::create_dir_all(&nested2)?;
        std::fs::write(
            nested2.join("SKILL.md"),
            "---\nname: writing-plans\ndescription: nested skill 2\n---\nbody",
        )?;

        let mut loader = SkillLoader::new();
        let descs = loader.discover_directory(root.clone()).await?;

        let names: Vec<String> = descs.iter().map(|d| d.name.clone()).collect();
        assert!(
            names.contains(&"coding".to_string()),
            "flat skill should be found: {:?}",
            names
        );
        assert!(
            names.contains(&"brainstorming".to_string()),
            "nested methodology skill must be found (B2 recursion): {:?}",
            names
        );
        assert!(
            names.contains(&"writing-plans".to_string()),
            "second nested skill must be found: {:?}",
            names
        );
        assert_eq!(descs.len(), 3, "expected 3 skills (1 flat + 2 nested)");
        Ok(())
    }

    /// 验证 `SkillLoadPolicy` 能在发现阶段排除掉被拒绝的技能描述符。
    #[tokio::test]
    async fn load_policy_excludes_disallowed_descriptor()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        struct DenyBlocked;
        impl SkillLoadPolicy for DenyBlocked {
            fn allows<'a>(&'a self, descriptor: &'a SkillDescriptor) -> BoxFuture<'a, bool> {
                Box::pin(async move { descriptor.name != "blocked" })
            }
        }

        let root = std::env::temp_dir().join(format!("echo_skill_policy_{}", uuid::Uuid::new_v4()));
        struct Guard(std::path::PathBuf);
        impl Drop for Guard {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
        let _guard = Guard(root.clone());
        for name in ["allowed", "blocked"] {
            let dir = root.join(name);
            std::fs::create_dir_all(&dir)?;
            std::fs::write(
                dir.join("SKILL.md"),
                format!("---\nname: {name}\ndescription: {name} skill\n---\nbody"),
            )?;
        }

        let mut loader = SkillLoader::new().with_policy(Arc::new(DenyBlocked));
        let descriptors = loader.discover_directory(&root).await?;
        assert_eq!(descriptors.len(), 1);
        assert_eq!(
            descriptors.first().map(|value| value.name.as_str()),
            Some("allowed")
        );
        Ok(())
    }

    #[tokio::test]
    async fn agent_plugin_discovery_uses_only_immediate_children()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let root =
            std::env::temp_dir().join(format!("echo_agent_plugin_skills_{}", uuid::Uuid::new_v4()));
        struct Guard(std::path::PathBuf);
        impl Drop for Guard {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
        let _guard = Guard(root.clone());
        let direct = root.join("direct");
        let nested = root.join("category/nested");
        std::fs::create_dir_all(&direct)?;
        std::fs::create_dir_all(&nested)?;
        std::fs::write(
            direct.join("SKILL.md"),
            "---\nname: direct\ndescription: direct skill\n---\nbody",
        )?;
        std::fs::write(
            nested.join("SKILL.md"),
            "---\nname: nested\ndescription: nested skill\n---\nbody",
        )?;

        let mut loader = SkillLoader::new();
        let descriptors = loader.discover_agent_plugin_skills(&root).await?;
        assert_eq!(descriptors.len(), 1);
        assert_eq!(
            descriptors
                .first()
                .map(|descriptor| descriptor.name.as_str()),
            Some("direct")
        );
        Ok(())
    }

    #[tokio::test]
    async fn plugin_variables_apply_to_frontmatter_and_body()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let root = std::env::temp_dir().join(format!(
            "echo_plugin_skill_variables_{}",
            uuid::Uuid::new_v4()
        ));
        struct Guard(std::path::PathBuf);
        impl Drop for Guard {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
        let _guard = Guard(root.clone());
        let skill_dir = root.join("configured-skill");
        std::fs::create_dir_all(&skill_dir)?;
        std::fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: configured-skill\ndescription: Uses ${user_config.endpoint}\n---\nBody ${user_config.endpoint}\n",
        )?;
        let variables = echo_core::plugin::PluginVariables::new(
            root.clone(),
            root.join("plugin-data/configured-plugin"),
            root.join("project"),
        )
        .with_user_config(std::collections::HashMap::from([(
            "endpoint".to_string(),
            "http://localhost:9100".to_string(),
        )]));
        let mut loader = SkillLoader::new().with_plugin_variables(variables);
        let descriptors = loader.discover_directory(&root).await?;
        let descriptor = descriptors
            .first()
            .ok_or("plugin skill was not discovered")?;
        assert_eq!(descriptor.description, "Uses http://localhost:9100");
        assert!(descriptor.hooks.is_none());
        Ok(())
    }

    #[tokio::test]
    async fn standard_frontmatter_parses_without_private_extensions()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let root =
            std::env::temp_dir().join(format!("echo_standard_skill_{}", uuid::Uuid::new_v4()));
        struct Guard(std::path::PathBuf);
        impl Drop for Guard {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
        let _guard = Guard(root.clone());
        let skill_dir = root.join("routed-skill");
        std::fs::create_dir_all(&skill_dir)?;
        std::fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: routed-skill\ndescription: Official layout with routing keywords.\nmetadata:\n  category: demo\nallowed-tools: shell read_file\n---\nBody",
        )?;
        let mut loader = SkillLoader::new();
        let descriptors = loader.discover_directory(&root).await?;
        let descriptor = descriptors
            .first()
            .ok_or("standard skill was not discovered")?;
        assert!(descriptor.triggers.is_empty());
        assert_eq!(
            descriptor.allowed_tools,
            vec!["shell".to_string(), "read_file".to_string()]
        );
        assert!(descriptor.hooks.is_none());
        Ok(())
    }

    #[tokio::test]
    async fn private_hook_sidecar_is_rejected_during_discovery()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let root = std::env::temp_dir().join(format!("echo_private_hook_{}", uuid::Uuid::new_v4()));
        struct Guard(std::path::PathBuf);
        impl Drop for Guard {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
        let _guard = Guard(root.clone());
        let skill_dir = root.join("private-hook-skill");
        std::fs::create_dir_all(&skill_dir)?;
        std::fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: private-hook-skill\ndescription: Invalid private extension.\n---\nBody",
        )?;
        std::fs::write(skill_dir.join("hooks.json"), "{}\n")?;

        let mut loader = SkillLoader::new();
        let descriptors = loader.discover_directory(&root).await?;
        assert!(descriptors.is_empty());
        assert!(
            loader
                .discovery_diagnostics()
                .iter()
                .any(|diagnostic| diagnostic.is_error && diagnostic.path.ends_with("hooks.json"))
        );
        Ok(())
    }

    #[test]
    fn test_scope_to_dirs_project() {
        let dirs = scope_to_dirs(&DiscoveryScope::Project(PathBuf::from("/my/project")));
        assert_eq!(dirs.len(), 2);
        assert_eq!(dirs.first(), Some(&PathBuf::from("/my/project/skills")));
        assert_eq!(
            dirs.get(1),
            Some(&PathBuf::from("/my/project/.agents/skills"))
        );
    }

    #[test]
    fn test_scope_to_dirs_custom() {
        let dirs = scope_to_dirs(&DiscoveryScope::Custom(PathBuf::from("/custom/path")));
        assert_eq!(dirs, vec![PathBuf::from("/custom/path")]);
    }

    #[test]
    fn test_allowed_tools_string() -> std::result::Result<(), String> {
        let content = "---\nname: test\ndescription: Test\nallowed-tools: Bash(git:*) Read\n---\n";
        let document = SkillDocument::parse_at(content, PathBuf::from("/test/SKILL.md"))
            .map_err(|error| error.to_string())?;
        assert_eq!(
            document.descriptor().allowed_tools,
            vec!["Bash(git:*)", "Read"]
        );
        Ok(())
    }
}
