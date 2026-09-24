//! 技能相似度检测与合并。
//!
//! 基于触发词、路径、工具以及描述文本检测重叠的技能，
//! 然后生成或执行合并以减少冗余。

use chrono::{DateTime, Utc};
use std::collections::HashSet;
use std::sync::Arc;

use echo_core::memory::store::Store;
use echo_state::skill_telemetry::{SkillTelemetry, SkillTelemetryStore};
use serde::{Deserialize, Serialize};

use super::audit::{ChangeEntryBuilder, ChangeLog, ChangeType, EntityType};
use super::curator::SkillLifecycle;
use super::skill_mutation::{
    SkillApprovalArtifact, SkillFileMutation, SkillMutationAuthority, SkillMutationKind,
    SkillMutationOutcome, SkillMutationPreview, SkillMutationRequest,
};
use crate::error::Result;

// 从 echo-execution 重新导出 SkillDescriptor 供本模块使用。
// 实际类型定义在 echo_execution::skills::external::types 中。
pub use echo_execution::skills::external::SkillDescriptor;

/// 超过该阈值即生成合并提案。
const MERGE_THRESHOLD: f64 = 0.75;

/// 存储合并提案所使用的命名空间。
const MERGE_NAMESPACE: &[&str] = &["agent", "evolution", "merges"];

/// 合并两个相似技能的提案。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillMergeProposal {
    /// 第一个技能的名称。
    pub skill_a: String,
    /// 第二个技能的名称。
    pub skill_b: String,
    /// 综合相似度分数（0.0 到 1.0）。
    pub similarity_score: f64,
    /// 按维度拆分的相似度明细。
    pub breakdown: SimilarityBreakdown,
    /// 应当保留的技能（遥测数据更多的那个）。
    pub primary_skill: String,
    /// 应当被弃用的技能。
    pub deprecated_skill: String,
    /// 该提案的创建时间。
    #[serde(with = "crate::utils::time::local_rfc3339")]
    pub created_at: DateTime<Utc>,
}

/// 各维度相似度的详细拆分。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityBreakdown {
    /// 触发关键词的重叠度（Jaccard）。
    pub trigger_overlap: f64,
    /// 文件路径模式的重叠度（Jaccard）。
    pub path_overlap: f64,
    /// 允许工具的重叠度（Jaccard）。
    pub tool_overlap: f64,
    /// 描述文本的相似度（词级别 Jaccard）。
    pub description_similarity: f64,
}

/// 检测相似技能并生成合并提案。
pub struct SkillSimilarityDetector {
    store: Arc<dyn Store>,
    telemetry_store: SkillTelemetryStore,
}

impl SkillSimilarityDetector {
    /// 创建一个新的检测器。
    pub fn new(store: Arc<dyn Store>) -> Self {
        let telemetry_store = SkillTelemetryStore::new(store.clone());
        Self {
            store,
            telemetry_store,
        }
    }

    /// 分析两个技能并计算它们的相似度分数。
    pub fn compute_similarity(
        desc_a: &SkillDescriptor,
        desc_b: &SkillDescriptor,
    ) -> SimilarityBreakdown {
        let trigger_overlap = jaccard_overlap(&desc_a.triggers, &desc_b.triggers);
        let path_overlap = jaccard_overlap(&desc_a.paths, &desc_b.paths);
        let tool_overlap = jaccard_overlap(&desc_a.allowed_tools, &desc_b.allowed_tools);
        let description_similarity = word_similarity(&desc_a.description, &desc_b.description);

        SimilarityBreakdown {
            trigger_overlap,
            path_overlap,
            tool_overlap,
            description_similarity,
        }
    }

    /// 根据明细计算综合相似度分数。
    /// 权重：triggers 0.4，paths 0.2，tools 0.2，description 0.2
    pub fn overall_score(breakdown: &SimilarityBreakdown) -> f64 {
        breakdown.trigger_overlap * 0.4
            + breakdown.path_overlap * 0.2
            + breakdown.tool_overlap * 0.2
            + breakdown.description_similarity * 0.2
    }

    /// 扫描所有已注册的技能，为相似的技能对生成合并提案。
    pub async fn scan_and_propose(
        &self,
        descriptors: &[SkillDescriptor],
        change_log: &dyn ChangeLog,
    ) -> Result<Vec<SkillMergeProposal>> {
        let mut proposals = Vec::new();
        let mut seen_pairs = HashSet::new();

        for (i, desc_a) in descriptors.iter().enumerate() {
            for desc_b in descriptors.iter().skip(i + 1) {
                // 规范化配对键以避免重复（A-B 与 B-A 视为同一对）。
                let (first, second) = if desc_a.name < desc_b.name {
                    (&desc_a.name, &desc_b.name)
                } else {
                    (&desc_b.name, &desc_a.name)
                };
                let pair_key = format!("{}__{}", first, second);
                if seen_pairs.contains(&pair_key) {
                    continue;
                }
                seen_pairs.insert(pair_key);

                let breakdown = Self::compute_similarity(desc_a, desc_b);
                let score = Self::overall_score(&breakdown);

                if score >= MERGE_THRESHOLD {
                    let (primary, deprecated) = self.determine_primary(desc_a, desc_b).await;

                    let proposal = SkillMergeProposal {
                        skill_a: desc_a.name.clone(),
                        skill_b: desc_b.name.clone(),
                        similarity_score: score,
                        breakdown,
                        primary_skill: primary,
                        deprecated_skill: deprecated,
                        created_at: Utc::now(),
                    };

                    // 存储该提案。
                    let store_key = format!("{}__{}", proposal.skill_a, proposal.skill_b);
                    let value = serde_json::to_value(&proposal).map_err(|e| {
                        echo_core::error::ReactError::Other(format!(
                            "Failed to serialize merge proposal: {}",
                            e
                        ))
                    })?;
                    self.store.put(MERGE_NAMESPACE, &store_key, value).await?;

                    // 记录到审计日志中。
                    let entry =
                        ChangeEntryBuilder::new(EntityType::Skill, &store_key, ChangeType::Create)
                            .reason(format!(
                                "Merge proposal: {} <-> {} (score: {:.2})",
                                proposal.skill_a, proposal.skill_b, score
                            ))
                            .trigger("skill_similarity_detector".to_string())
                            .build(change_log);
                    change_log.record(entry)?;

                    proposals.push(proposal);
                }
            }
        }

        Ok(proposals)
    }

    /// 根据遥测数据决定哪个技能应作为主技能。
    /// 激活次数更多或使用时间更近的一方会成为主技能。
    async fn determine_primary(
        &self,
        desc_a: &SkillDescriptor,
        desc_b: &SkillDescriptor,
    ) -> (String, String) {
        let telem_a = self
            .telemetry_store
            .get_telemetry(&desc_a.name)
            .await
            .ok()
            .flatten();
        let telem_b = self
            .telemetry_store
            .get_telemetry(&desc_b.name)
            .await
            .ok()
            .flatten();

        // 归一化打分：50% activation_count（≤1.0）+ 50% recency
        // （通过近似的 epoch→now 窗口把 last_used 归一化到 [0, 1]）。
        // 此前 `last_used / 1e12` ≈ 1.77，会盖过 ≤1.0 的 activation_count。
        fn usage_score(t: &SkillTelemetry) -> f64 {
            let recency = (t.last_used as f64 / 1.7e12 - 1.0).clamp(0.0, 1.0);
            t.activation_count as f64 * 0.5 + recency * 0.5
        }
        let score_a = telem_a.as_ref().map(usage_score).unwrap_or(0.0);
        let score_b = telem_b.as_ref().map(usage_score).unwrap_or(0.0);

        if score_a >= score_b {
            (desc_a.name.clone(), desc_b.name.clone())
        } else {
            (desc_b.name.clone(), desc_a.name.clone())
        }
    }
}

/// 通过更新描述符并把技能标记为已弃用来执行技能合并。
pub struct SkillMerger {
    /// 技能变更权限方（负责预览、审批与应用变更）。
    authority: Arc<SkillMutationAuthority>,
}

/// 一次技能合并的预览：待应用的变更请求、预览结果以及合并后的描述符。
pub struct SkillMergePreview {
    /// 待提交给权限方的变更请求。
    pub request: SkillMutationRequest,
    /// 该变更请求的预览结果（含操作摘要）。
    pub preview: SkillMutationPreview,
    /// 合并后的技能描述符（尚未落盘，仅在预览通过后才会生效）。
    pub merged_descriptor: SkillDescriptor,
}

impl SkillMerger {
    /// 创建一个新的合并器。
    pub fn new(authority: Arc<SkillMutationAuthority>) -> Self {
        Self { authority }
    }

    /** 旧版直接合并入口。由于没有经过审阅的审批凭证，该方法总是失败
    （fail closed）；请改用 `preview_merge` + `execute_preview`。

    `primary_descriptor` 会被原地更新，合并后的 triggers、paths 和 tools
    会写入其中。被弃用的技能会通过 Curator 标记。
    */
    pub async fn execute_merge(
        &self,
        proposal: &SkillMergeProposal,
        primary_descriptor: &mut SkillDescriptor,
        deprecated_descriptor: Option<&SkillDescriptor>,
    ) -> Result<()> {
        let _ = (proposal, primary_descriptor, deprecated_descriptor);
        Err(echo_core::error::ReactError::Other(
            "skill merge requires preview_merge followed by execute_preview with approval".into(),
        ))
    }

    pub async fn preview_merge(
        &self,
        request_id: impl Into<String>,
        proposal: &SkillMergeProposal,
        primary_descriptor: &SkillDescriptor,
        deprecated_descriptor: Option<&SkillDescriptor>,
    ) -> Result<SkillMergePreview> {
        let mut merged = primary_descriptor.clone();
        // 把被弃用技能的 triggers 合并进主技能。
        if let Some(dep_desc) = deprecated_descriptor {
            let mut merged_triggers: HashSet<String> = merged.triggers.iter().cloned().collect();
            for trigger in &dep_desc.triggers {
                merged_triggers.insert(trigger.clone());
            }
            merged.triggers = merged_triggers.into_iter().collect();

            // 合并 paths。
            let mut merged_paths: HashSet<String> = merged.paths.iter().cloned().collect();
            for path in &dep_desc.paths {
                merged_paths.insert(path.clone());
            }
            merged.paths = merged_paths.into_iter().collect();

            // 合并 allowed_tools。
            let mut merged_tools: HashSet<String> = merged.allowed_tools.iter().cloned().collect();
            for tool in &dep_desc.allowed_tools {
                merged_tools.insert(tool.clone());
            }
            merged.allowed_tools = merged_tools.into_iter().collect();
        }

        merged.triggers.sort();
        merged.paths.sort();
        merged.allowed_tools.sort();

        let original = tokio::fs::read(&merged.location).await?;
        let original_text = String::from_utf8(original.clone()).map_err(|error| {
            echo_core::error::ReactError::Other(format!("SKILL.md is not UTF-8: {error}"))
        })?;
        let updated = update_skill_frontmatter(&original_text, &merged)?;
        let curator_before = self.authority.curator().load_state()?;
        let mut curator_after = curator_before.clone();
        let meta = curator_after
            .skills
            .get_mut(&proposal.deprecated_skill)
            .ok_or_else(|| {
                echo_core::error::ReactError::Other(format!(
                    "skill {:?} is not tracked by Curator",
                    proposal.deprecated_skill
                ))
            })?;
        if !matches!(
            meta.lifecycle,
            SkillLifecycle::Active | SkillLifecycle::Stale
        ) {
            return Err(echo_core::error::ReactError::Other(format!(
                "skill {:?} is not in a lifecycle state that can be deprecated",
                proposal.deprecated_skill
            )));
        }
        meta.lifecycle = SkillLifecycle::Deprecated;
        meta.superseded_by = Some(proposal.primary_skill.clone());
        if let Some(path) = &meta.path {
            meta.path = Some(SkillFileMutation::canonical_path(path)?);
        }
        meta.last_modified_at = Utc::now();
        let merge_key = format!("{}__{}", proposal.skill_a, proposal.skill_b);
        let file_mutation =
            SkillFileMutation::new(&merged.location, Some(original), Some(updated.into_bytes()))?;
        merged.location = file_mutation.path.clone();
        let request = SkillMutationRequest {
            request_id: request_id.into(),
            entity_key: merge_key,
            kind: SkillMutationKind::Merge,
            reason: format!(
                "Merged {} into {}",
                proposal.deprecated_skill, proposal.primary_skill
            ),
            files: vec![file_mutation],
            curator_before,
            curator_after,
            rollback_of: None,
        };
        let preview = self.authority.preview(&request)?;
        Ok(SkillMergePreview {
            request,
            preview,
            merged_descriptor: merged,
        })
    }

    pub async fn execute_preview(
        &self,
        preview: SkillMergePreview,
        approval: SkillApprovalArtifact,
        primary_descriptor: &mut SkillDescriptor,
    ) -> Result<crate::evolution::SkillMutationReceipt> {
        match self.authority.apply(preview.request, approval).await? {
            SkillMutationOutcome::Applied(receipt)
            | SkillMutationOutcome::AlreadyApplied(receipt) => {
                *primary_descriptor = preview.merged_descriptor;
                Ok(receipt)
            }
            outcome => Err(echo_core::error::ReactError::Other(format!(
                "skill merge did not apply: {outcome:?}"
            ))),
        }
    }
}

fn update_skill_frontmatter(content: &str, descriptor: &SkillDescriptor) -> Result<String> {
    let document = echo_execution::skills::external::SkillDocument::parse(content)
        .map_err(|error| echo_core::error::ReactError::Other(error.to_string()))?;
    document
        .render_with_allowed_tools(&descriptor.allowed_tools)
        .map_err(echo_core::error::ReactError::Other)
}

// ── 相似度辅助函数 ────────────────────────────────────────────────────

/// 计算两个字符串列表之间的 Jaccard 相似度。
fn jaccard_overlap(list_a: &[String], list_b: &[String]) -> f64 {
    if list_a.is_empty() && list_b.is_empty() {
        return 1.0;
    }
    if list_a.is_empty() || list_b.is_empty() {
        return 0.0;
    }

    let set_a: HashSet<&str> = list_a.iter().map(|s| s.as_str()).collect();
    let set_b: HashSet<&str> = list_b.iter().map(|s| s.as_str()).collect();

    let intersection = set_a.intersection(&set_b).count();
    let union = set_a.union(&set_b).count();

    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}

/// 计算两段文本之间的词级别 Jaccard 相似度。
fn word_similarity(text_a: &str, text_b: &str) -> f64 {
    let words_a: HashSet<&str> = text_a.split_whitespace().collect();
    let words_b: HashSet<&str> = text_b.split_whitespace().collect();

    if words_a.is_empty() && words_b.is_empty() {
        return 1.0;
    }
    if words_a.is_empty() || words_b.is_empty() {
        return 0.0;
    }

    let intersection = words_a.intersection(&words_b).count();
    let union = words_a.union(&words_b).count();

    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evolution::{JsonlChangeLog, SkillRollbackPreviewOutcome, SkillRollbackTarget};
    use echo_state::memory::store::InMemoryStore;
    use std::collections::HashMap;

    fn make_descriptor(
        name: &str,
        triggers: Vec<&str>,
        paths: Vec<&str>,
        tools: Vec<&str>,
        description: &str,
    ) -> SkillDescriptor {
        SkillDescriptor {
            source: None,
            name: name.to_string(),
            description: description.to_string(),
            triggers: triggers.into_iter().map(|s| s.to_string()).collect(),
            paths: paths.into_iter().map(|s| s.to_string()).collect(),
            allowed_tools: tools.into_iter().map(|s| s.to_string()).collect(),
            license: None,
            compatibility: None,
            metadata: HashMap::new(),
            shell: None,
            hooks: None,
            sandbox: None,
            depends_on: vec![],
            location: std::path::PathBuf::new(),
        }
    }

    /// 用于测试的空操作（no-op）ChangeLog。
    struct NullChangeLog;
    impl ChangeLog for NullChangeLog {
        fn record(&self, _entry: super::super::audit::ChangeEntry) -> Result<()> {
            Ok(())
        }
        fn record_idempotent(
            &self,
            _entry: super::super::audit::ChangeEntry,
        ) -> Result<super::super::audit::ChangeRecordOutcome> {
            Ok(super::super::audit::ChangeRecordOutcome::AlreadyRecorded)
        }
        fn query(
            &self,
            _filter: &super::super::audit::ChangeFilter,
        ) -> Result<Vec<super::super::audit::ChangeEntry>> {
            Ok(Vec::new())
        }
        fn latest_for(
            &self,
            _entity_type: EntityType,
            _entity_key: &str,
        ) -> Result<Option<super::super::audit::ChangeEntry>> {
            Ok(None)
        }
        fn len(&self) -> usize {
            0
        }
    }

    #[test]
    fn test_jaccard_overlap_identical() {
        let list_a = vec!["git".to_string(), "commit".to_string(), "push".to_string()];
        let list_b = vec!["git".to_string(), "commit".to_string(), "push".to_string()];
        let overlap = jaccard_overlap(&list_a, &list_b);
        assert!((overlap - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_jaccard_overlap_disjoint() {
        let list_a = vec!["git".to_string(), "commit".to_string()];
        let list_b = vec!["docker".to_string(), "build".to_string()];
        let overlap = jaccard_overlap(&list_a, &list_b);
        assert!((overlap - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_jaccard_overlap_partial() {
        let list_a = vec!["git".to_string(), "commit".to_string(), "push".to_string()];
        let list_b = vec!["git".to_string(), "commit".to_string(), "pull".to_string()];
        // 交集：{git, commit} = 2
        // 并集：{git, commit, push, pull} = 4
        let overlap = jaccard_overlap(&list_a, &list_b);
        assert!((overlap - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_jaccard_overlap_both_empty() {
        let list_a: Vec<String> = vec![];
        let list_b: Vec<String> = vec![];
        let overlap = jaccard_overlap(&list_a, &list_b);
        assert!((overlap - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_word_similarity() {
        let text_a = "Git workflow for version control";
        let text_b = "Git workflow management";
        let similarity = word_similarity(text_a, text_b);
        // 交集：{Git, workflow} = 2
        // 并集：{Git, workflow, for, version, control, management} = 6
        assert!((similarity - 2.0 / 6.0).abs() < 1e-9);
    }

    #[test]
    fn test_compute_similarity_high() {
        let desc_a = make_descriptor(
            "git-workflow",
            vec!["git", "commit", "push"],
            vec!["*.rs", "*.toml"],
            vec!["Bash", "Read", "Write"],
            "Git version control workflow",
        );
        let desc_b = make_descriptor(
            "git-ops",
            vec!["git", "commit", "push", "pull"],
            vec!["*.rs", "*.toml", "*.md"],
            vec!["Bash", "Read", "Write"],
            "Git operations workflow",
        );
        let breakdown = SkillSimilarityDetector::compute_similarity(&desc_a, &desc_b);
        let score = SkillSimilarityDetector::overall_score(&breakdown);
        assert!(score > 0.6, "Expected high similarity, got {}", score);
    }

    #[test]
    fn test_compute_similarity_low() {
        let desc_a = make_descriptor(
            "git-workflow",
            vec!["git", "commit"],
            vec!["*.rs"],
            vec!["Bash"],
            "Git version control",
        );
        let desc_b = make_descriptor(
            "docker-build",
            vec!["docker", "build"],
            vec!["Dockerfile"],
            vec!["Bash"],
            "Docker container building",
        );
        let breakdown = SkillSimilarityDetector::compute_similarity(&desc_a, &desc_b);
        let score = SkillSimilarityDetector::overall_score(&breakdown);
        assert!(score < 0.3, "Expected low similarity, got {}", score);
    }

    #[tokio::test]
    async fn test_scan_and_propose() {
        let store = Arc::new(InMemoryStore::new()) as Arc<dyn Store>;
        let detector = SkillSimilarityDetector::new(store);

        let descriptors = vec![
            make_descriptor(
                "git-workflow",
                vec!["git", "commit", "push"],
                vec!["*.rs"],
                vec!["Bash", "Read"],
                "Git workflow for commits",
            ),
            make_descriptor(
                "git-ops",
                vec!["git", "commit", "push"],
                vec!["*.rs"],
                vec!["Bash", "Read"],
                "Git operations for commits",
            ),
            make_descriptor(
                "docker-build",
                vec!["docker", "build"],
                vec!["Dockerfile"],
                vec!["Bash"],
                "Docker container building",
            ),
        ];

        let change_log = NullChangeLog;
        let proposals = detector
            .scan_and_propose(&descriptors, &change_log)
            .await
            .unwrap();

        // 应当找到 git-workflow <-> git-ops 这一对。
        assert!(!proposals.is_empty());
        let proposal = &proposals[0];
        assert!(
            (proposal.skill_a == "git-workflow" && proposal.skill_b == "git-ops")
                || (proposal.skill_a == "git-ops" && proposal.skill_b == "git-workflow")
        );
        assert!(proposal.similarity_score >= MERGE_THRESHOLD);
    }

    fn make_merger(_store: Arc<dyn Store>) -> SkillMerger {
        use crate::evolution::curator::{Curator, CuratorConfig};
        let config = CuratorConfig::default();
        let state_path = std::env::temp_dir().join(format!(
            "echo-agent-test-curator-state-{}.json",
            uuid::Uuid::new_v4()
        ));
        let change_path = state_path.with_extension("changes.jsonl");
        let curator = Curator::new(config, state_path);
        let authority = Arc::new(
            SkillMutationAuthority::open(
                curator,
                Arc::new(JsonlChangeLog::new(change_path).unwrap()),
            )
            .unwrap(),
        );
        SkillMerger::new(authority)
    }

    #[tokio::test]
    async fn test_execute_merge() -> std::result::Result<(), String> {
        let store = Arc::new(InMemoryStore::new()) as Arc<dyn Store>;
        let merger = make_merger(store);
        let root = std::env::temp_dir().join(format!("echo-agent-merge-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).map_err(|error| error.to_string())?;
        let primary_path = root.join("primary-SKILL.md");
        std::fs::write(
            &primary_path,
            "---\nname: git-workflow\ndescription: Git workflow\n---\nBody\n",
        )
        .map_err(|error| error.to_string())?;

        let mut primary = make_descriptor(
            "git-workflow",
            vec!["git", "commit"],
            vec!["*.rs"],
            vec!["Bash"],
            "Git workflow",
        );
        let deprecated = make_descriptor(
            "git-ops",
            vec!["git", "push", "pull"],
            vec!["*.toml"],
            vec!["Read", "Write"],
            "Git operations",
        );
        primary.location = primary_path.clone();

        merger
            .authority
            .curator()
            .touch_skill("git-ops", true)
            .map_err(|error| error.to_string())?;

        let proposal = SkillMergeProposal {
            skill_a: "git-workflow".to_string(),
            skill_b: "git-ops".to_string(),
            similarity_score: 0.85,
            breakdown: SimilarityBreakdown {
                trigger_overlap: 0.5,
                path_overlap: 0.0,
                tool_overlap: 0.0,
                description_similarity: 0.5,
            },
            primary_skill: "git-workflow".to_string(),
            deprecated_skill: "git-ops".to_string(),
            created_at: Utc::now(),
        };

        let preview = merger
            .preview_merge(
                uuid::Uuid::new_v4().to_string(),
                &proposal,
                &primary,
                Some(&deprecated),
            )
            .await
            .map_err(|error| error.to_string())?;
        let approval = SkillApprovalArtifact::new(
            uuid::Uuid::new_v4().to_string(),
            &preview.preview.operation_digest,
            "test-reviewer",
            Utc::now(),
        );
        let receipt = merger
            .execute_preview(preview, approval, &mut primary)
            .await
            .map_err(|error| error.to_string())?;

        // 主技能现在应当拥有合并后的 triggers。
        assert!(primary.triggers.contains(&"git".to_string()));
        assert!(primary.triggers.contains(&"commit".to_string()));
        assert!(primary.triggers.contains(&"push".to_string()));
        assert!(primary.triggers.contains(&"pull".to_string()));

        // 主技能应当拥有合并后的 paths。
        assert!(primary.paths.contains(&"*.rs".to_string()));
        assert!(primary.paths.contains(&"*.toml".to_string()));

        // 主技能应当拥有合并后的 tools。
        assert!(primary.allowed_tools.contains(&"Bash".to_string()));
        assert!(primary.allowed_tools.contains(&"Read".to_string()));
        assert!(primary.allowed_tools.contains(&"Write".to_string()));
        // 持久化文件是标准格式：合并后的工具集合以空格分隔字符串的形式写入，
        // 而路由用的 triggers 没有标准字段，只保留在内存中的描述符里。
        let persisted =
            std::fs::read_to_string(&primary_path).map_err(|error| error.to_string())?;
        assert!(persisted.contains("allowed-tools: Bash Read Write"));
        assert!(!persisted.contains("\ntriggers:"));
        let authority = merger.authority.clone();
        let target = SkillRollbackTarget::BatchId(receipt.batch_id);
        let SkillRollbackPreviewOutcome::Ready(rollback_preview) = authority
            .preview_rollback("merge-rollback", &target)
            .await
            .map_err(|error| error.to_string())?
        else {
            return Err("merge rollback preview was not ready".into());
        };
        let rollback_approval = SkillApprovalArtifact::new(
            "merge-rollback-approval",
            &rollback_preview.operation_digest,
            "test-reviewer",
            Utc::now(),
        );
        authority
            .rollback("merge-rollback", target, rollback_approval)
            .await
            .map_err(|error| error.to_string())?;
        let restored = std::fs::read_to_string(&primary_path).map_err(|error| error.to_string())?;
        assert_eq!(
            restored,
            "---\nname: git-workflow\ndescription: Git workflow\n---\nBody\n"
        );
        assert_eq!(
            merger
                .authority
                .curator()
                .load_state()
                .map_err(|error| error.to_string())?
                .skills
                .get("git-ops")
                .map(|meta| meta.lifecycle),
            Some(SkillLifecycle::Active)
        );
        std::fs::remove_dir_all(root).map_err(|error| error.to_string())?;
        Ok::<(), String>(())
    }
}
