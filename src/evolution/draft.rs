/*!
 * Skill 草稿生成 —— 根据检测到的候选项创建 SKILL.md 文件。
 *
 * 接收一个 [`SkillCandidate`]（由 [`super::candidate::SkillCandidateDetector`]
 * 产出），并在 `<consumer-root>/skills/_drafts/<name>/SKILL.md` 处
 * 生成一份草稿 SKILL.md 文件。该草稿只是一份*提案* —— 需要人工
 * 审核后才能晋升为 Active 状态。
 *
 * # 基于模板的生成方式
 *
 * Phase 3 使用确定性的模板（不调用 LLM）。这样可以保持系统
 * 快速、免费且可复现。未来的迭代可以通过可选的、由评测驱动的
 * 改进流水线加入 LLM 辅助的精修能力。
 */

use std::path::PathBuf;
use std::sync::Arc;

use echo_state::memory::typed_store::TypedMemoryStore;

use super::candidate::{CANDIDATE_NAMESPACE, SkillCandidate};
use super::curator::SkillLifecycle;
use super::skill_mutation::{
    SkillApprovalArtifact, SkillFileMutation, SkillMutationAuthority, SkillMutationKind,
    SkillMutationOutcome, SkillMutationPreview, SkillMutationRequest,
};
use crate::error::Result;
use echo_core::error::ReactError;

// ── Constants ──────────────────────────────────────────────────────────

/// 草稿 SKILL.md 文件的保存位置，相对于 consumer root。
const DRAFTS_DIR: &str = "skills/_drafts";

// ── DraftResult ────────────────────────────────────────────────────────

/// 生成 skill 草稿的结果。
#[derive(Debug, Clone)]
pub struct DraftResult {
    /// skill 的名称。
    pub name: String,
    /// 生成的 SKILL.md 文件路径。
    pub skill_md_path: PathBuf,
    /// 本次是新建（true）还是更新（false）。
    pub created: bool,
}

/// 一次经过审核的确切草稿请求，及其与摘要（digest）绑定的预览。
pub struct SkillDraftPreview {
    /// 草稿生成的结果
    pub result: DraftResult,
    /// 提交给权威的变更请求
    pub request: SkillMutationRequest,
    /// 该请求对应的预览（含操作摘要）
    pub preview: SkillMutationPreview,
}

// ── SkillDraftGenerator ────────────────────────────────────────────────

/**
 * 根据 skill 候选项生成草稿 SKILL.md 文件。
 *
 * 该生成器会把基于模板生成的 SKILL.md 文件写入 `_drafts`
 * 目录，并通过 [`Curator`] 将候选项晋升为 `Draft` 生命周期状态。
 */
pub struct SkillDraftGenerator {
    /// 由 consumer 提供的 evolution 存储根目录。
    storage_root: PathBuf,
    /// 用于审核并执行文件/生命周期变更的权威
    authority: Arc<SkillMutationAuthority>,
}

impl SkillDraftGenerator {
    /// 创建一个新的生成器。
    pub fn new(storage_root: PathBuf, authority: Arc<SkillMutationAuthority>) -> Self {
        Self {
            storage_root,
            authority,
        }
    }

    /**
     * 旧版的按名称查找候选项的入口。它会读取候选项，但由于
     * 没有经过审核的批准工件（approval artifact），会失败关闭。
     */
    #[deprecated(
        note = "load the candidate, call preview_generate_from_candidate, then generate_from_preview with approval"
    )]
    pub async fn generate(
        &self,
        name: &str,
        typed_store: &TypedMemoryStore,
    ) -> Result<DraftResult> {
        // 1. 从 Store 中读取候选项。
        let entry = typed_store
            .get_typed(CANDIDATE_NAMESPACE, name)
            .await?
            .ok_or_else(|| {
                ReactError::Other(format!("Skill candidate '{}' not found in store", name))
            })?;

        let candidate: SkillCandidate = serde_json::from_str(&entry.content).map_err(|e| {
            ReactError::Other(format!("Failed to parse candidate '{}': {}", name, e))
        })?;

        Err(ReactError::Other(format!(
            "draft mutation for {:?} requires preview_generate_from_candidate followed by generate_from_preview with approval",
            candidate.name
        )))
    }

    /// 旧版的直接入口；没有批准工件时会失败关闭。
    #[deprecated(note = "use preview_generate_from_candidate followed by generate_from_preview")]
    pub async fn generate_from_candidate(&self, candidate: &SkillCandidate) -> Result<DraftResult> {
        Err(ReactError::Other(format!(
            "draft mutation for {:?} requires preview_generate_from_candidate followed by generate_from_preview with approval",
            candidate.name
        )))
    }

    /// 构建由宿主审核的确切文件/生命周期变更。
    pub async fn preview_generate_from_candidate(
        &self,
        candidate: &SkillCandidate,
        request_id: impl Into<String>,
    ) -> Result<SkillDraftPreview> {
        let name = &candidate.name;
        let drafts_root = self.storage_root.join(DRAFTS_DIR);
        let dir = echo_core::utils::fs::join_path_segment(&drafts_root, name).map_err(|error| {
            ReactError::Other(format!("Unsafe skill candidate name {name:?}: {error}"))
        })?;
        let skill_md_path = dir.join("SKILL.md");

        let content = render_skill_md(candidate);
        let previous = match std::fs::read(&skill_md_path) {
            Ok(bytes) => Some(bytes),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(error) => return Err(error.into()),
        };
        let created = previous.is_none();
        let file_mutation =
            SkillFileMutation::new(&skill_md_path, previous, Some(content.into_bytes()))?;
        let canonical_skill_path = file_mutation.path.clone();
        let canonical_drafts_root = SkillFileMutation::canonical_path(&drafts_root)?;
        if !canonical_skill_path.starts_with(&canonical_drafts_root) {
            return Err(ReactError::Other(format!(
                "skill draft path escapes the drafts root: {}",
                canonical_skill_path.display()
            )));
        }
        let curator_before = self.authority.curator().load_state()?;
        let mut curator_after = curator_before.clone();
        let meta = curator_after.skills.get_mut(name).ok_or_else(|| {
            ReactError::Other(format!("candidate '{name}' is not registered with Curator"))
        })?;
        if !matches!(
            meta.lifecycle,
            SkillLifecycle::Candidate | SkillLifecycle::Draft
        ) {
            return Err(ReactError::Other(format!(
                "candidate '{name}' is not in Candidate lifecycle state"
            )));
        }
        meta.lifecycle = SkillLifecycle::Draft;
        meta.path = Some(canonical_skill_path.clone());
        meta.last_modified_at = chrono::Utc::now();
        let request = SkillMutationRequest {
            request_id: request_id.into(),
            entity_key: name.clone(),
            kind: SkillMutationKind::Draft,
            reason: format!(
                "draft SKILL.md for candidate '{}' from {} observations",
                name, candidate.sample_count
            ),
            files: vec![file_mutation],
            curator_before,
            curator_after,
            rollback_of: None,
        };
        let preview = self.authority.preview(&request)?;
        Ok(SkillDraftPreview {
            result: DraftResult {
                name: name.clone(),
                skill_md_path: canonical_skill_path,
                created,
            },
            request,
            preview,
        })
    }

    /// 使用一次性的、与摘要绑定的批准工件来应用一份确切的预览。
    pub async fn generate_from_preview(
        &self,
        preview: SkillDraftPreview,
        approval: SkillApprovalArtifact,
    ) -> Result<DraftResult> {
        match self.authority.apply(preview.request, approval).await? {
            SkillMutationOutcome::Applied(_) | SkillMutationOutcome::AlreadyApplied(_) => {
                Ok(preview.result)
            }
            outcome => Err(ReactError::Other(format!(
                "draft mutation did not apply: {outcome:?}"
            ))),
        }
    }
}

// ── 模板渲染 ──────────────────────────────────────────────────────────

/// 使用确定性模板，根据候选项渲染出一个 SKILL.md 文件。
fn render_skill_md(candidate: &SkillCandidate) -> String {
    let SkillCandidate {
        name,
        description,
        /* 触发模式（trigger patterns）保留在 curator 状态中：标准的
         * SKILL.md 格式没有用于路由触发的字段。
         */
        trigger_patterns: _,
        tool_sequence,
        sample_count,
        confidence,
        topic,
        source_type,
        created_at,
    } = candidate;

    /* 官方 agentskills.io 格式：一个以空格分隔的纯字符串。
     * 路由由 description 驱动；触发模式保留在 curator 状态中，
     * 因为标准格式没有对应的字段。
     */
    let tools_inline = (!tool_sequence.is_empty()).then(|| {
        tool_sequence
            .iter()
            .map(|t| yaml_escape(t))
            .collect::<Vec<_>>()
            .join(" ")
    });

    let workflow_steps = if tool_sequence.is_empty() {
        "1. Analyze the user's request\n2. Apply the relevant tools\n3. Verify the result"
            .to_string()
    } else {
        tool_sequence
            .iter()
            .enumerate()
            .map(|(i, t)| {
                format!(
                    "{}. Use `{}` to accomplish the task step",
                    i + 1,
                    yaml_escape(t)
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    };

    let source_label = match source_type {
        echo_core::memory::types::MemoryType::WorkflowPattern => "workflow",
        echo_core::memory::types::MemoryType::DebuggingLesson => "debugging",
        _ => "usage",
    };

    // 转义将写入 YAML frontmatter 的值，以防止注入。
    let safe_name = yaml_escape(name);
    let safe_description = yaml_escape(description);
    let safe_topic = yaml_escape(topic);

    format!(
        r#"---
name: {safe_name}
description: >-
    {safe_description}
{allowed_tools_line}
metadata:
    author: echo-agent
    source: auto-candidate
    confidence: "{confidence:.2}"
    sample_count: "{sample_count}"
    lifecycle: draft
    topic: "{safe_topic}"
    created_at: "{created_at}"
---

## {safe_name}

Auto-generated skill from {sample_count} observed {source_label} patterns on topic `{safe_topic}`.

### Workflow

{workflow_steps}

### Common Patterns

This skill was proposed based on {sample_count} repeated observations.
Confidence: {confidence:.0}%.

### Safety

- Always verify the result before presenting to the user.
- Do not apply destructive operations without confirmation.
"#,
        safe_name = safe_name,
        safe_description = safe_description,
        allowed_tools_line = tools_inline
            .map(|tools| format!("allowed-tools: {tools}"))
            .unwrap_or_default(),
        sample_count = sample_count,
        confidence = confidence,
        safe_topic = safe_topic,
        source_label = source_label,
        created_at = crate::utils::time::to_local(*created_at).to_rfc3339(),
        workflow_steps = workflow_steps,
    )
}

/**
 * 转义一个字符串值，以便安全地写入 YAML 的双引号或无引号上下文中。
 *
 * 替换掉会破坏 YAML 解析或注入额外 frontmatter 字段的字符
 * （`: `、`"`、`\`、换行符、`---`）。
 * 转义一个值，以便将其写入**双引号**包裹的 YAML 字符串中。
 *
 * 该 YAML 模板会用 `"..."` 包裹取值，因此只需转义 `\`、`"`、
 * 换行符、`\r` 以及 `---`（frontmatter 分隔符）即可。根据
 * YAML 1.2 规范 §7.3.1，双引号内的其他字符（`:`、`#`、`[`、
 * `]`、`{`、`}`）都是无害的。
 */
fn yaml_escape(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', " ")
        .replace('\r', "")
        .replace("---", "-\\-\\-")
}

// ── 测试 ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evolution::{Curator, CuratorConfig, JsonlChangeLog};
    use chrono::Utc;
    use echo_core::memory::types::MemoryType;

    /// 构造一个用于测试的技能候选项样例。
    fn sample_candidate() -> SkillCandidate {
        SkillCandidate {
            name: "cargo-build".to_string(),
            description: "Auto-detected workflow pattern for 'cargo-build'.".to_string(),
            trigger_patterns: vec![
                "cargo-build".to_string(),
                "build".to_string(),
                "compile".to_string(),
            ],
            tool_sequence: vec!["Bash(*)".to_string(), "Read".to_string()],
            sample_count: 5,
            confidence: 0.85,
            topic: "cargo-build".to_string(),
            source_type: MemoryType::WorkflowPattern,
            created_at: Utc::now(),
        }
    }

    /// 在给定根目录下创建一个带变更授权的草稿生成器。
    fn make_generator(root: PathBuf) -> Result<SkillDraftGenerator> {
        let curator = Curator::new(CuratorConfig::default(), root.join("curator_state.json"));
        let authority = Arc::new(SkillMutationAuthority::open(
            curator,
            Arc::new(JsonlChangeLog::new(root.join("draft-changes.jsonl"))?),
        )?);
        Ok(SkillDraftGenerator::new(root, authority))
    }

    /// 注册候选项、生成预览并附带审批工件，走完一次经批准的草稿生成流程。
    async fn generate_approved(
        generator: &SkillDraftGenerator,
        candidate: &SkillCandidate,
    ) -> Result<DraftResult> {
        generator
            .authority
            .curator()
            .register_candidate(&candidate.name)?;
        let preview = generator
            .preview_generate_from_candidate(candidate, uuid::Uuid::new_v4().to_string())
            .await?;
        let approval = SkillApprovalArtifact::new(
            uuid::Uuid::new_v4().to_string(),
            &preview.preview.operation_digest,
            "test-reviewer",
            Utc::now(),
        );
        generator.generate_from_preview(preview, approval).await
    }

    /// 验证草稿生成会创建 SKILL.md 文件并写入草稿生命周期。
    #[tokio::test]
    async fn test_draft_generation_creates_file() {
        let dir = tempfile::tempdir().expect("tempdir").keep();
        let generator = make_generator(dir.clone()).unwrap();

        let candidate = sample_candidate();
        let result = generate_approved(&generator, &candidate).await.unwrap();

        assert_eq!(result.name, "cargo-build");
        assert!(result.created);
        assert!(result.skill_md_path.exists());

        let content = std::fs::read_to_string(&result.skill_md_path).unwrap();
        assert!(content.contains("name: cargo-build"));
        assert!(content.contains("lifecycle: draft"));
    }

    /// 验证包含路径逃逸的候选项名称会被拒绝。
    #[tokio::test]
    async fn draft_generation_rejects_path_escape_name() {
        let dir = tempfile::tempdir().expect("tempdir").keep();
        let outside = dir.parent().map(|parent| parent.join("escaped-skill"));
        let generator = make_generator(dir).unwrap();
        let mut candidate = sample_candidate();
        candidate.name = "../escaped-skill".to_string();
        assert!(
            generator
                .preview_generate_from_candidate(&candidate, "escape-test")
                .await
                .is_err()
        );
        assert!(outside.is_none_or(|path| !path.exists()));
    }

    /// 验证候选项目录为符号链接时草稿生成会被拒绝。
    #[cfg(unix)]
    #[tokio::test]
    async fn draft_generation_rejects_symlink_candidate_directory() -> Result<()> {
        use std::os::unix::fs::symlink;

        let root = tempfile::tempdir()?;
        let outside = tempfile::tempdir()?;
        let drafts = root.path().join(DRAFTS_DIR);
        std::fs::create_dir_all(&drafts)?;
        symlink(outside.path(), drafts.join("cargo-build"))?;
        let generator = make_generator(root.path().to_path_buf())?;

        assert!(
            generator
                .preview_generate_from_candidate(&sample_candidate(), "symlink-test")
                .await
                .is_err()
        );
        assert!(!outside.path().join("SKILL.md").exists());
        Ok(())
    }

    /// 验证生成的草稿 YAML frontmatter 合法。
    #[tokio::test]
    async fn test_draft_yaml_frontmatter_valid() {
        let dir = tempfile::tempdir().expect("tempdir").keep();
        let generator = make_generator(dir).unwrap();

        let candidate = sample_candidate();
        let result = generate_approved(&generator, &candidate).await.unwrap();

        let content = std::fs::read_to_string(&result.skill_md_path).unwrap();

        // 提取 YAML frontmatter。
        let yaml = if let Some(rest) = content.strip_prefix("---") {
            let end = rest.find("---").unwrap_or(rest.len());
            &rest[..end]
        } else {
            panic!("Missing frontmatter delimiter");
        };

        // 应当是合法的 YAML。
        let parsed: serde_yaml_ng::Value =
            serde_yaml_ng::from_str(yaml).expect("YAML should parse");
        assert_eq!(parsed["name"].as_str(), Some("cargo-build"));
        assert_eq!(parsed["metadata"]["lifecycle"].as_str(), Some("draft"));
        // 仅标准布局：没有路由命名空间，allowed-tools 为字符串。
        assert!(parsed.get("triggers").is_none());
        assert!(parsed["metadata"].get("echo-agent").is_none());
        assert!(parsed["allowed-tools"].as_str().is_some());
    }

    /// 验证对同一候选项重复生成草稿时为幂等更新。
    #[tokio::test]
    async fn test_draft_idempotent_update() {
        let dir = tempfile::tempdir().expect("tempdir").keep();
        let generator = make_generator(dir).unwrap();

        let candidate = sample_candidate();

        // 第一次生成：新建。
        let result1 = generate_approved(&generator, &candidate).await.unwrap();
        assert!(result1.created);

        // 第二次生成：更新，而非新建。
        let mut candidate2 = candidate.clone();
        candidate2.sample_count = 7;
        let preview = generator
            .preview_generate_from_candidate(&candidate2, uuid::Uuid::new_v4().to_string())
            .await
            .unwrap();
        let approval = SkillApprovalArtifact::new(
            uuid::Uuid::new_v4().to_string(),
            &preview.preview.operation_digest,
            "test-reviewer",
            Utc::now(),
        );
        let result2 = generator
            .generate_from_preview(preview, approval)
            .await
            .unwrap();
        assert!(!result2.created);

        // 内容应当反映更新后的样本数量。
        let content = std::fs::read_to_string(&result2.skill_md_path).unwrap();
        assert!(content.contains("7 observed"));
    }

    /// 验证 `render_skill_md` 渲染出的 SKILL.md 内容结构。
    #[test]
    fn test_render_skill_md_content() {
        let candidate = sample_candidate();
        let md = render_skill_md(&candidate);

        // 基础结构检查。
        assert!(md.starts_with("---"));
        assert!(md.contains("name: cargo-build"));
        assert!(md.contains("lifecycle: draft"));
        assert!(md.contains("## cargo-build"));
        assert!(md.contains("### Workflow"));
        assert!(md.contains("### Safety"));
        assert!(md.contains("5 observed"));
        assert!(md.contains("confidence: \"0.85\""));
    }
}
