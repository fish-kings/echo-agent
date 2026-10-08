/*!
自我进化系统 —— 类型化记忆、变更审计、安全机制与技能创建。

本模块为 Agent 随时间持续演化自身能力提供基础设施，具体包括：

- **技能生命周期**：[`SkillMutationAuthority`] 持久化地应用经过批准的
  Curator + 精确的 SKILL.md 状态转换（Candidate → Draft → Active → Stale → Deprecated → Archived）
- **类型化记忆**：为每条记忆提供结构化元数据（类型、置信度、稳定性、风险）
- **变更审计**：仅追加式证据记录，用于持久化、按代际隔离的
  记忆变更以及技能所有者发起的回滚
- **安全机制**：密钥扫描、不可信输入隔离、注入检测
- **记忆复核**：陈旧度评分、冲突检测、合并与归档
- **技能创建**：从观察到的模式中检测候选项、生成草稿 SKILL.md

# 相关模块

本模块与 `trace` 紧密协作以获取执行证据。离线评估与
提示词优化仍作为独立的可选能力存在。

# 安全性

分层记忆变更携带持久化的审计身份；技能与规则的
变更覆盖范围分别单独追踪。
高风险变更（规则晋升、技能合并）需要人工复核。
来自不可信来源的内容永远不会被自动晋升。
*/

/// 演化变更审计日志。
pub mod audit;
/// 后台复盘审查。
pub mod background_review;
/// 演化候选项。
pub mod candidate;
/// 演化内容整理（curator）。
pub mod curator;
/// 演化草稿。
pub mod draft;
/// 离线“做梦”式经验整合。
pub mod dreaming;
/// 演化健康度检查。
pub mod health;
/// 演化分层。
pub mod layer;
/// 演化结果合并。
pub mod merge;
/// 演化变更（mutation）内部实现。
mod mutation;
/// 演化补丁。
pub mod patch;
/// 演化经验召回。
pub mod recall;
/// 演化结果审查。
pub mod review;
/// 演化与运行时的集成。
pub mod runtime_integration;
/// 演化内容安全检查。
pub mod security;
/// 技能变更。
pub mod skill_mutation;
/// 演化触发条件。
pub mod triggers;

pub use audit::{
    ChangeEntry, ChangeEntryBuilder, ChangeFilter, ChangeLog, ChangeRecordOutcome, ChangeType,
    EntityType, JsonlChangeLog,
};
pub use background_review::{
    BackgroundReviewConfig, BackgroundReviewer, ReviewCandidate, ReviewCandidateKind, ReviewOutcome,
};
pub use candidate::{CandidateReport, SkillCandidate, SkillCandidateDetector};
pub use curator::{Curator, CuratorConfig, CuratorState, CuratorStatus, SkillLifecycle, SkillMeta};
pub use draft::{DraftResult, SkillDraftGenerator, SkillDraftPreview};
pub use dreaming::{Dreaming, DreamingAction, DreamingConfig, DreamingDecision, DreamingReport};
pub use health::{HealthBreakdown, HealthStatus, SkillHealthMonitor, SkillHealthReport};
pub use layer::{
    EvolutionObserver, HotEntryMeta, LayerChangeResult, MemoryFile, MemoryLayer,
    MemoryLayerManager, MemoryRollbackConflict, MemoryRollbackHistoryUnavailable,
    MemoryRollbackOutcome, MemoryRollbackPreview, MemoryRollbackPreviewOutcome,
    MemoryRollbackReceipt, MemoryRollbackTarget, is_stale_memory_proposal_error,
};
pub use merge::{
    SimilarityBreakdown, SkillMergePreview, SkillMergeProposal, SkillMerger,
    SkillSimilarityDetector,
};
pub use patch::{PatchType, SkillPatch, SkillPatchPreview, SkillPatcher};
pub use recall::MemoryRecaller;
pub use review::{
    AppliedMemoryMerge, ConflictDetector, ConflictGroup, MemoryConflictMember,
    MemoryConflictProposal, MemoryMergeSnapshot, MemoryMerger, MemoryReviewer, MergeResult,
    ReviewChange, ReviewConfig, ReviewReport, StalenessReport, StalenessScorer,
};
pub use runtime_integration::{HookEvolutionObserver, MemoryRuntimeIntegrationBuilder};
pub use security::{
    EvolutionSecurityGuard, InputTrustLevel, PromptInjectionDetector, ScanResult, SecretScanner,
    SecurityConfig, SecurityVerdict,
};
pub use skill_mutation::{
    SkillApprovalArtifact, SkillFileMutation, SkillMutationAuthority, SkillMutationKind,
    SkillMutationObserver, SkillMutationOutcome, SkillMutationPreview, SkillMutationReceipt,
    SkillMutationRequest, SkillRollbackLineage, SkillRollbackPreviewOutcome, SkillRollbackSupport,
    SkillRollbackTarget, SkillUsageHandle, SkillUsageOutcome,
};
pub use triggers::{
    ExplicitSaveRecord, MemoryTriggerDisposition, MemoryTriggerSink, ToolFailureRecord,
    ToolSequenceRecord, ToolSuccessRecord, TriggerContext, TriggerDetector, TriggerEvidence,
    TriggerMatch,
};
