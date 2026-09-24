//! 基于失败模式生成技能补丁。
//!
//! 分析技能遥测数据以识别常见的失败模式，并生成补丁提案，
//! 通过更完善的错误处理来改进技能说明。

use chrono::{DateTime, Utc};
use sha2::{Digest, Sha256};
use std::sync::Arc;

use echo_core::memory::store::Store;
use echo_state::skill_telemetry::SkillTelemetryStore;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::evolution::skill_mutation::{
    SkillApprovalArtifact, SkillFileMutation, SkillMutationAuthority, SkillMutationKind,
    SkillMutationOutcome, SkillMutationPreview, SkillMutationRequest,
};

// 重新导出 SkillDescriptor，供本模块使用。
pub use echo_execution::skills::external::SkillDescriptor;

/// 应用到某个技能的补丁类型。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PatchType {
    /// 针对某个特定失败模式，添加错误处理说明。
    ErrorHandling {
        /// 要处理的失败模式。
        failure_pattern: String,
        /// 建议的错误处理说明。
        handling_instructions: String,
    },
    /// 在执行前添加前置条件检查。
    PrerequisiteCheck {
        /// 前置条件的描述。
        check_description: String,
        /// 如何验证该前置条件。
        verification_steps: Vec<String>,
    },
    /// 当主要方案失败时，添加兜底（fallback）策略。
    FallbackStrategy {
        /// 何时使用该兜底策略。
        trigger_condition: String,
        /// 兜底说明。
        fallback_instructions: String,
    },
    /// 为已有说明补充更多细节，加以改进。
    InstructionEnhancement {
        /// 要改进的说明部分。
        target_section: String,
        /// 改进细节。
        enhancement: String,
    },
}

impl PatchType {
    /// 该补丁类型的可读标签（用于 markdown 标题）。
    pub fn label(&self) -> &'static str {
        match self {
            PatchType::ErrorHandling { .. } => "Error Handling",
            PatchType::PrerequisiteCheck { .. } => "Prerequisite Check",
            PatchType::FallbackStrategy { .. } => "Fallback Strategy",
            PatchType::InstructionEnhancement { .. } => "Instruction Enhancement",
        }
    }
}

/// 针对某个技能提出的一个补丁提案。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillPatch {
    /// 技能名称。
    pub skill_name: String,
    /// 补丁类型。
    pub patch_type: PatchType,
    /// 该补丁的理由说明。
    pub rationale: String,
    /// 基于失败频率得出的置信度分数（0.0-1.0）。
    pub confidence: f64,
    /// 用于排序补丁的优先级分数（0-10）。
    pub priority: u8,
    /// 该补丁提出的时间。
    #[serde(with = "crate::utils::time::local_rfc3339")]
    pub proposed_at: DateTime<Utc>,
    /// 本次提案所审查的 SKILL.md 精确内容的 SHA-256 哈希。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_hash: Option<String>,
}

impl SkillPatch {
    /// 将该提案绑定到当前的精确源版本。
    pub async fn bind_to_source(mut self, path: &std::path::Path) -> Result<Self> {
        let content = tokio::fs::read(path).await?;
        self.source_hash = Some(sha256_hex(&content));
        Ok(self)
    }
    /// 生成该补丁的可读摘要。
    pub fn summary(&self) -> String {
        match &self.patch_type {
            PatchType::ErrorHandling {
                failure_pattern,
                handling_instructions,
            } => {
                format!(
                    "Add error handling for '{}' (confidence: {:.0}%):\n  {}",
                    failure_pattern,
                    self.confidence * 100.0,
                    handling_instructions
                )
            }
            PatchType::PrerequisiteCheck {
                check_description,
                verification_steps,
            } => {
                format!(
                    "Add prerequisite check: {} (priority: {})\n  Steps:\n    - {}",
                    check_description,
                    self.priority,
                    verification_steps.join("\n    - ")
                )
            }
            PatchType::FallbackStrategy {
                trigger_condition,
                fallback_instructions,
            } => {
                format!(
                    "Add fallback when: {} (confidence: {:.0}%)\n  Fallback: {}",
                    trigger_condition,
                    self.confidence * 100.0,
                    fallback_instructions
                )
            }
            PatchType::InstructionEnhancement {
                target_section,
                enhancement,
            } => {
                format!(
                    "Enhance section '{}' (priority: {}):\n  {}",
                    target_section, self.priority, enhancement
                )
            }
        }
    }

    /**
    将该补丁渲染为一个适合追加到 SKILL.md 的 markdown 小节。

    该小节会被放在 frontmatter 与已有正文内容之后。它包含一个带
    时间戳的标题、理由说明，以及从补丁类型推导出的结构化说明。
    */
    pub fn to_markdown(&self) -> String {
        let ts = self.proposed_at.format("%Y-%m-%d").to_string();
        let mut lines = vec![
            format!(
                "\n---\n\n## Auto-patch: {} ({})\n",
                self.patch_type.label(),
                ts
            ),
            format!("**Rationale:** {}\n", self.rationale),
            format!("**Confidence:** {:.0}%\n", self.confidence * 100.0),
        ];

        match &self.patch_type {
            PatchType::ErrorHandling {
                failure_pattern,
                handling_instructions,
            } => {
                lines.push(format!("**Failure pattern:** `{failure_pattern}`\n"));
                lines.push(format!("**Instructions:**\n{handling_instructions}\n"));
            }
            PatchType::PrerequisiteCheck {
                check_description,
                verification_steps,
            } => {
                lines.push(format!("**Check:** {check_description}\n"));
                lines.push("**Steps:**\n".into());
                for step in verification_steps {
                    lines.push(format!("- {step}"));
                }
                lines.push(String::new());
            }
            PatchType::FallbackStrategy {
                trigger_condition,
                fallback_instructions,
            } => {
                lines.push(format!("**Trigger:** {trigger_condition}\n"));
                lines.push(format!("**Fallback:**\n{fallback_instructions}\n"));
            }
            PatchType::InstructionEnhancement {
                target_section,
                enhancement,
            } => {
                lines.push(format!("**Section:** {target_section}\n"));
                lines.push(format!("**Enhancement:**\n{enhancement}\n"));
            }
        }

        lines.join("\n")
    }
}

/// 基于遥测分析生成技能补丁。
pub struct SkillPatcher {
    /// 技能遥测数据存储。
    telemetry_store: SkillTelemetryStore,
    /// 可选的、已绑定的技能变更权威。
    authority: Option<Arc<SkillMutationAuthority>>,
}

/// 一次已生成预览、待审批的技能补丁变更。
pub struct SkillPatchPreview {
    /// 提交给变更权威的原始请求。
    pub request: SkillMutationRequest,
    /// 变更权威返回的预览结果。
    pub preview: SkillMutationPreview,
}

impl SkillPatcher {
    /// 创建一个新的技能补丁生成器。
    pub fn new(store: Arc<dyn Store>) -> Self {
        let telemetry_store = SkillTelemetryStore::new(store);
        Self {
            telemetry_store,
            authority: None,
        }
    }

    /// 绑定一个技能变更权威，使该补丁生成器可以执行预览/应用流程。
    pub fn with_authority(mut self, authority: Arc<SkillMutationAuthority>) -> Self {
        self.authority = Some(authority);
        self
    }

    /// 分析某个技能的失败模式，并生成补丁提案。
    pub async fn analyze_and_propose(&self, skill_name: &str) -> Result<Vec<SkillPatch>> {
        let telemetry = match self.telemetry_store.get_telemetry(skill_name).await? {
            Some(t) => t,
            None => return Ok(Vec::new()),
        };

        let mut patches = Vec::new();

        // 分析常见失败
        for failure in &telemetry.common_failures {
            if let Some(patch) = self.generate_patch_for_failure(skill_name, failure) {
                patches.push(patch);
            }
        }

        // 先按优先级（降序）、再按置信度（降序）排序
        patches.sort_by(|a, b| {
            b.priority.cmp(&a.priority).then(
                b.confidence
                    .partial_cmp(&a.confidence)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
        });

        Ok(patches)
    }

    /// 为所有存在失败模式的技能生成补丁。
    pub async fn analyze_all_skills(&self) -> Result<Vec<SkillPatch>> {
        let all_telemetry = self.telemetry_store.list_all().await?;
        let mut all_patches = Vec::new();

        for telemetry in all_telemetry {
            if !telemetry.common_failures.is_empty() {
                for failure in &telemetry.common_failures {
                    if let Some(patch) =
                        self.generate_patch_for_failure(&telemetry.skill_name, failure)
                    {
                        all_patches.push(patch);
                    }
                }
            }
        }

        // 先按优先级、再按置信度排序
        all_patches.sort_by(|a, b| {
            b.priority.cmp(&a.priority).then(
                b.confidence
                    .partial_cmp(&a.confidence)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
        });

        Ok(all_patches)
    }

    /**
    将一个补丁应用到某个技能的 SKILL.md 文件。

    会把补丁说明作为一个 markdown 小节追加到正文中（位于已有 frontmatter
    之后）。**不会**修改 YAML frontmatter。这个旧版直接入口会失败关闭
    （fail closed）；请改用绑定权威上的 `preview_patch` 与 `apply_preview`。

    # 参数
    * `patch` — 要应用的补丁（来自 `analyze_and_propose`）。
    * `descriptor` — 技能描述符（提供 `.location`，即 SKILL.md 路径）。
    */
    pub async fn apply_patch(
        &self,
        patch: &SkillPatch,
        descriptor: &SkillDescriptor,
    ) -> Result<()> {
        let _ = (patch, descriptor);
        Err(crate::error::ReactError::Other(
            "skill patch requires preview_patch followed by apply_preview with approval".into(),
        ))
    }

    /**
    为一次补丁变更生成预览：校验源哈希、拼装新内容，
    并向绑定的变更权威提交一份待审批的变更请求。
    */
    pub async fn preview_patch(
        &self,
        request_id: impl Into<String>,
        patch: &SkillPatch,
        descriptor: &SkillDescriptor,
    ) -> Result<SkillPatchPreview> {
        let authority = self.authority.as_ref().ok_or_else(|| {
            crate::error::ReactError::Other(
                "skill patch requires a Curator-bound mutation authority".into(),
            )
        })?;
        let path = &descriptor.location;
        let lock_path = path.with_extension("md.patch.lock");
        let lock_file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&lock_path)?;
        fs2::FileExt::try_lock_exclusive(&lock_file).map_err(|error| {
            crate::error::ReactError::Other(format!(
                "another patch is already updating {}: {error}",
                path.display()
            ))
        })?;
        let content = tokio::fs::read_to_string(path).await.map_err(|e| {
            crate::error::ReactError::Other(format!(
                "Failed to read SKILL.md at {}: {e}",
                path.display()
            ))
        })?;
        let current_hash = sha256_hex(content.as_bytes());
        let expected_hash = patch.source_hash.as_deref().ok_or_else(|| {
            crate::error::ReactError::Other(
                "patch proposal is not bound to a source revision; call bind_to_source first"
                    .to_string(),
            )
        })?;
        if expected_hash != current_hash {
            return Err(crate::error::ReactError::Other(format!(
                "SKILL.md changed after proposal review (expected {expected_hash}, found {current_hash})"
            )));
        }

        // 将 frontmatter（--- ... ---）与正文拆分开。若没有 frontmatter，
        // 则将全部内容都当作正文处理。
        let (frontmatter, body) = if let Some(rest) = content.strip_prefix("---\n") {
            if let Some(end) = rest.find("\n---\n") {
                let fm_end = end + 8; // 4（前缀）+ 内容 + 4（\n---\n）
                let (fm, rest) = content.split_at(fm_end);
                (fm.to_string(), rest.to_string())
            } else {
                (String::new(), content.clone())
            }
        } else {
            (String::new(), content.clone())
        };

        // 将补丁 markdown 小节追加到正文中。
        let patch_md = patch.to_markdown();
        let new_body = format!("{}\n{}", body.trim_end(), patch_md);
        let new_content = if frontmatter.is_empty() {
            new_body
        } else {
            format!("{frontmatter}\n{new_body}\n")
        };

        let state = authority.curator().load_state()?;
        let request = SkillMutationRequest {
            request_id: request_id.into(),
            entity_key: patch.skill_name.clone(),
            kind: SkillMutationKind::Patch,
            reason: format!(
                "Applied patch: {} (confidence: {:.0}%)",
                patch.patch_type.label(),
                patch.confidence * 100.0
            ),
            files: vec![SkillFileMutation::new(
                path,
                Some(content.into_bytes()),
                Some(new_content.into_bytes()),
            )?],
            curator_before: state.clone(),
            curator_after: state,
            rollback_of: None,
        };
        let preview = authority.preview(&request)?;
        Ok(SkillPatchPreview { request, preview })
    }

    /// 使用给定的审批凭证，应用一次已生成预览的补丁变更。
    pub async fn apply_preview(
        &self,
        preview: SkillPatchPreview,
        approval: SkillApprovalArtifact,
    ) -> Result<()> {
        let authority = self.authority.as_ref().ok_or_else(|| {
            crate::error::ReactError::Other(
                "skill patch requires a Curator-bound mutation authority".into(),
            )
        })?;
        match authority.apply(preview.request, approval).await? {
            SkillMutationOutcome::Applied(_) | SkillMutationOutcome::AlreadyApplied(_) => Ok(()),
            outcome => Err(crate::error::ReactError::Other(format!(
                "skill patch did not apply: {outcome:?}"
            ))),
        }
    }

    /// 针对某个特定失败模式生成补丁提案。
    fn generate_patch_for_failure(
        &self,
        skill_name: &str,
        failure: &echo_state::skill_telemetry::FailurePattern,
    ) -> Option<SkillPatch> {
        let snippet_lower = failure.error_snippet.to_lowercase();

        // 根据失败频率计算置信度
        let confidence = if failure.count >= 10 {
            0.95
        } else if failure.count >= 5 {
            0.85
        } else if failure.count >= 3 {
            0.75
        } else {
            0.60
        };

        // 根据错误模式分析确定补丁类型
        let (patch_type, priority) = if snippet_lower.contains("permission denied")
            || snippet_lower.contains("access denied")
        {
            (
                PatchType::PrerequisiteCheck {
                    check_description: "Verify user has required permissions".to_string(),
                    verification_steps: vec![
                        "Check if target path is writable".to_string(),
                        "Verify file/directory ownership".to_string(),
                        "Confirm user has necessary privileges".to_string(),
                    ],
                },
                9,
            )
        } else if snippet_lower.contains("not found")
            || snippet_lower.contains("no such file")
            || snippet_lower.contains("does not exist")
        {
            (
                PatchType::PrerequisiteCheck {
                    check_description: "Verify required files/resources exist".to_string(),
                    verification_steps: vec![
                        "Check if target file or directory exists".to_string(),
                        "Verify path is correct and accessible".to_string(),
                        "Confirm dependencies are installed".to_string(),
                    ],
                },
                8,
            )
        } else if snippet_lower.contains("timeout") || snippet_lower.contains("timed out") {
            (
                PatchType::FallbackStrategy {
                    trigger_condition: "Operation times out or takes too long".to_string(),
                    fallback_instructions: "Consider breaking the operation into smaller chunks, \
                        increasing timeout limits, or using a more efficient approach. \
                        For network operations, implement retry logic with exponential backoff."
                        .to_string(),
                },
                7,
            )
        } else if snippet_lower.contains("connection refused")
            || snippet_lower.contains("network")
            || snippet_lower.contains("unreachable")
        {
            (
                PatchType::FallbackStrategy {
                    trigger_condition: "Network connectivity issues".to_string(),
                    fallback_instructions:
                        "Check network connectivity before attempting network operations. \
                        Implement retry logic with delays. Provide clear error messages about \
                        network requirements."
                            .to_string(),
                },
                7,
            )
        } else if snippet_lower.contains("syntax error")
            || snippet_lower.contains("parse error")
            || snippet_lower.contains("invalid")
        {
            (
                PatchType::ErrorHandling {
                    failure_pattern: failure.error_snippet.clone(),
                    handling_instructions: "Validate input format before processing. \
                        Provide clear examples of expected input format. \
                        Add detailed error messages that explain what went wrong and how to fix it."
                        .to_string(),
                },
                6,
            )
        } else if snippet_lower.contains("memory")
            || snippet_lower.contains("oom")
            || snippet_lower.contains("out of memory")
        {
            (
                PatchType::FallbackStrategy {
                    trigger_condition: "Memory exhaustion or resource limits".to_string(),
                    fallback_instructions: "Process data in smaller batches. \
                        Use streaming where possible. \
                        Monitor memory usage and provide warnings before hitting limits."
                        .to_string(),
                },
                8,
            )
        } else {
            // 针对无法识别的模式使用通用错误处理
            (
                PatchType::ErrorHandling {
                    failure_pattern: failure.error_snippet.clone(),
                    handling_instructions: format!(
                        "Add error handling for: '{}'. \
                        Check common causes, provide troubleshooting steps, \
                        and suggest alternative approaches.",
                        failure.error_snippet
                    ),
                },
                5,
            )
        };

        let rationale = format!(
            "This failure has occurred {} time{}. Recent occurrence: {}",
            failure.count,
            if failure.count == 1 { "" } else { "s" },
            failure.error_snippet
        );

        Some(SkillPatch {
            skill_name: skill_name.to_string(),
            patch_type,
            rationale,
            confidence,
            priority,
            proposed_at: Utc::now(),
            source_hash: None,
        })
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evolution::audit::JsonlChangeLog;
    use crate::evolution::curator::{Curator, CuratorConfig};
    use echo_state::memory::store::InMemoryStore;
    use echo_state::skill_telemetry::SkillExecutionRecord;
    use std::collections::HashMap;

    fn make_record(skill_name: &str, success: bool, error: Option<&str>) -> SkillExecutionRecord {
        SkillExecutionRecord {
            skill_name: skill_name.to_string(),
            session_id: "test-session".to_string(),
            activated_at: Utc::now().timestamp_millis() as u64,
            duration_ms: 1000,
            tools_used: vec!["Bash".to_string()],
            tool_calls_count: 1,
            success,
            error_message: error.map(|s| s.to_string()),
        }
    }

    #[tokio::test]
    async fn approved_patch_uses_canonical_authority() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("SKILL.md");
        std::fs::write(&path, "---\nname: test\ndescription: test\n---\nbody\n")?;
        let descriptor = SkillDescriptor {
            name: "test".into(),
            description: "test".into(),
            location: path.clone(),
            license: None,
            compatibility: None,
            metadata: HashMap::new(),
            source: None,
            allowed_tools: Vec::new(),
            shell: None,
            paths: Vec::new(),
            triggers: Vec::new(),
            hooks: None,
            sandbox: None,
            depends_on: Vec::new(),
        };
        let patch = SkillPatch {
            skill_name: "test".into(),
            patch_type: PatchType::InstructionEnhancement {
                target_section: "body".into(),
                enhancement: "add recovery guidance".into(),
            },
            rationale: "test".into(),
            confidence: 0.9,
            priority: 8,
            proposed_at: Utc::now(),
            source_hash: Some(sha256_hex(&std::fs::read(&path)?)),
        };
        let curator = Curator::new(CuratorConfig::default(), dir.path().join("curator.json"));
        let log = Arc::new(JsonlChangeLog::new(dir.path().join("changes.jsonl"))?);
        let authority = Arc::new(SkillMutationAuthority::open(curator, log)?);
        let patcher = SkillPatcher::new(Arc::new(InMemoryStore::new())).with_authority(authority);
        let preview = patcher
            .preview_patch("patch-request", &patch, &descriptor)
            .await?;
        let approval = SkillApprovalArtifact::new(
            "patch-approval",
            &preview.preview.operation_digest,
            "reviewer",
            Utc::now(),
        );
        patcher.apply_preview(preview, approval).await?;
        assert!(std::fs::read_to_string(path)?.contains("add recovery guidance"));
        Ok(())
    }

    #[tokio::test]
    async fn test_patch_for_permission_denied() {
        let store = Arc::new(InMemoryStore::new()) as Arc<dyn Store>;
        let patcher = SkillPatcher::new(store.clone());

        // 记录权限拒绝错误
        let telemetry_store = SkillTelemetryStore::new(store.clone());
        for _ in 0..5 {
            telemetry_store
                .record_execution(&make_record(
                    "test-skill",
                    false,
                    Some("Permission denied: cannot write to /etc/config"),
                ))
                .await
                .unwrap();
        }

        let patches = patcher.analyze_and_propose("test-skill").await.unwrap();
        assert!(!patches.is_empty());

        let patch = &patches[0];
        assert!(matches!(
            patch.patch_type,
            PatchType::PrerequisiteCheck { .. }
        ));
        assert!(patch.priority >= 8);
        assert!(patch.confidence >= 0.8);
    }

    #[tokio::test]
    async fn test_patch_for_timeout() {
        let store = Arc::new(InMemoryStore::new()) as Arc<dyn Store>;
        let patcher = SkillPatcher::new(store.clone());

        let telemetry_store = SkillTelemetryStore::new(store.clone());
        for _ in 0..3 {
            telemetry_store
                .record_execution(&make_record(
                    "test-skill",
                    false,
                    Some("Operation timed out after 30 seconds"),
                ))
                .await
                .unwrap();
        }

        let patches = patcher.analyze_and_propose("test-skill").await.unwrap();
        assert!(!patches.is_empty());

        let patch = &patches[0];
        assert!(matches!(
            patch.patch_type,
            PatchType::FallbackStrategy { .. }
        ));
    }

    #[tokio::test]
    async fn test_patch_for_network_error() {
        let store = Arc::new(InMemoryStore::new()) as Arc<dyn Store>;
        let patcher = SkillPatcher::new(store.clone());

        let telemetry_store = SkillTelemetryStore::new(store.clone());
        telemetry_store
            .record_execution(&make_record(
                "test-skill",
                false,
                Some("Connection refused: network unreachable"),
            ))
            .await
            .unwrap();

        let patches = patcher.analyze_and_propose("test-skill").await.unwrap();
        assert!(!patches.is_empty());

        let patch = &patches[0];
        assert!(matches!(
            patch.patch_type,
            PatchType::FallbackStrategy { .. }
        ));
    }

    #[tokio::test]
    async fn test_no_patches_for_successful_skill() {
        let store = Arc::new(InMemoryStore::new()) as Arc<dyn Store>;
        let patcher = SkillPatcher::new(store.clone());

        let telemetry_store = SkillTelemetryStore::new(store.clone());
        for _ in 0..10 {
            telemetry_store
                .record_execution(&make_record("test-skill", true, None))
                .await
                .unwrap();
        }

        let patches = patcher.analyze_and_propose("test-skill").await.unwrap();
        assert!(patches.is_empty());
    }

    #[tokio::test]
    async fn test_patches_sorted_by_priority() {
        let store = Arc::new(InMemoryStore::new()) as Arc<dyn Store>;
        let patcher = SkillPatcher::new(store.clone());

        let telemetry_store = SkillTelemetryStore::new(store.clone());

        // 低优先级错误
        telemetry_store
            .record_execution(&make_record(
                "test-skill",
                false,
                Some("Unknown error occurred"),
            ))
            .await
            .unwrap();

        // 高优先级错误
        for _ in 0..5 {
            telemetry_store
                .record_execution(&make_record("test-skill", false, Some("Permission denied")))
                .await
                .unwrap();
        }

        let patches = patcher.analyze_and_propose("test-skill").await.unwrap();
        assert!(patches.len() >= 2);

        // 第一个补丁的优先级应更高或相等
        assert!(patches[0].priority >= patches[1].priority);
    }

    #[test]
    fn test_patch_summary_error_handling() {
        let patch = SkillPatch {
            skill_name: "test".to_string(),
            patch_type: PatchType::ErrorHandling {
                failure_pattern: "file not found".to_string(),
                handling_instructions: "Check file exists first".to_string(),
            },
            rationale: "Common failure".to_string(),
            confidence: 0.85,
            priority: 7,
            proposed_at: Utc::now(),
            source_hash: None,
        };

        let summary = patch.summary();
        assert!(summary.contains("file not found"));
        assert!(summary.contains("85%"));
        assert!(summary.contains("Check file exists first"));
    }

    #[test]
    fn test_patch_summary_prerequisite() {
        let patch = SkillPatch {
            skill_name: "test".to_string(),
            patch_type: PatchType::PrerequisiteCheck {
                check_description: "Verify permissions".to_string(),
                verification_steps: vec![
                    "Check ownership".to_string(),
                    "Check write access".to_string(),
                ],
            },
            rationale: "Permission errors".to_string(),
            confidence: 0.9,
            priority: 9,
            proposed_at: Utc::now(),
            source_hash: None,
        };

        let summary = patch.summary();
        assert!(summary.contains("Verify permissions"));
        assert!(summary.contains("Check ownership"));
        assert!(summary.contains("Check write access"));
    }
}
