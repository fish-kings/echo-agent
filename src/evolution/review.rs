/*! 记忆的复核分析与显式冲突解决。
 *
 * 这是演化（evolution）系统的第二阶段。第一阶段为记忆提供了类型化元数据、
 * 分层存储（热/温/冷），以及审计日志，但记忆在创建之后从未被重新评估过。
 * 本模块新增了：
 *
 * - [`StalenessScorer`] —— 根据年龄、使用度、不稳定性、矛盾冲突和来源弱点
 *   等因素计算完整的陈旧度分数（取代 `MemoryMeta::base_staleness`）。
 * - [`ConflictDetector`] —— 将拥有相同主题 + 类型、但内容哈希不同的记忆
 *   分组。
 * - [`MemoryMerger`] —— 显式的变更原语，将一个经用户批准的冲突组合并为
 *   一条主条目，并将其余条目标记为被取代。
 * - [`MemoryReviewer`] —— 仅做分析的编排器：扫描 → 评分 → 提出建议。
 *
 * # 状态阈值
 *
 * 公式中各因子的上限（年龄因子 ≤ 0.8，矛盾因子 ≤ 0.5）意味着理论上的
 * 最大陈旧度只有约 0.83——这是一种有意为之的保守设计，确保没有任何单一
 * 因子在被拉满时就能单独判定一条记忆“该被淘汰”。下面的阈值就是按照这个
 * 实际动态范围校准的，而不是完整的 `[0, 1]` 区间：
 *
 * | 陈旧度       | 状态                         |
 * |------------|------------------------------|
 * | `< 0.35`   | 活跃（Active）                |
 * | `0.35–0.50`| 活跃（标记待复核）             |
 * | `0.50–0.65`| 待取代候选                    |
 * | `≥ 0.65`   | 待归档候选                    |
 *
 * 复核器本身从不修改记忆。确定性的定期维护由 [`super::dreaming::Dreaming`]
 * 负责；语义层面的冲突解决则需要调用方显式调用 [`MemoryMerger`]。
 */

use chrono::{DateTime, Duration, Utc};
use echo_core::memory::types::{MemoryMeta, MemoryStatus, MemoryType};
use echo_core::utils::hash::fnv1a_64;
use echo_state::memory::typed_store::{MemoryFilter, TypedMemoryEntry, TypedMemoryStore};
use std::collections::HashMap;

#[cfg(test)]
use super::audit::{ChangeLog, EntityType};
use super::layer::MemoryLayerManager;
use crate::error::Result;

// ── 常量 ──────────────────────────────────────────────────────────

/** 陈旧度低于该阈值时，条目维持 Active 状态。
 *
 * 该阈值是按照公式的实际动态范围（最大约 0.83）校准的，而不是完整的 `[0,1]`。
 */
const STALENESS_ACTIVE_MAX: f32 = 0.35;
/// 陈旧度高于该阈值时，条目成为待取代候选。
const STALENESS_SUPERSEDED_MIN: f32 = 0.50;
/// 陈旧度高于该阈值时，条目被归档。
const STALENESS_ARCHIVE_MIN: f32 = 0.65;

// ── StalenessScorer ────────────────────────────────────────────────────

/** 为一条类型化记忆计算陈旧度分数（0.0 = 新鲜，1.0 = 非常陈旧）。
 *
 * `staleness = age·0.35 + low_usage·0.20 + instability·0.20 + contradiction·0.20 + source_weakness·0.05`
 *
 * 各因子权重经过调优，使得没有单一因子能够主导结果：一条较旧但经常被
 * 修订、来自可信来源且置信度高的条目，依然可能停留在归档阈值之下；而
 * 一条新鲜但存在矛盾、由自动抽取且置信度较低的条目，依然可能被标记。
 */
pub struct StalenessScorer;

impl StalenessScorer {
    /// 创建一个新的评分器。
    pub fn new() -> Self {
        Self
    }

    /** 对单条记忆条目评分。
     *
     * 当 [`ConflictDetector`] 发现该条目位于某个冲突组中时，
     * `has_contradiction` 应为 `true`。复核器会在单次评分之前先
     * 完成冲突检测。
     */
    pub fn score(
        &self,
        entry: &TypedMemoryEntry,
        now: DateTime<Utc>,
        has_contradiction: bool,
    ) -> StalenessReport {
        let age_factor = age_factor(entry, now);
        let usage_factor = low_usage_factor(&entry.meta);
        let instability_factor = 1.0 - entry.meta.stability;
        let contradiction_factor = if has_contradiction { 0.5 } else { 0.0 };
        let source_factor = 1.0 - entry.meta.source.default_confidence();

        let staleness = (age_factor * 0.35
            + usage_factor * 0.20
            + instability_factor * 0.20
            + contradiction_factor * 0.20
            + source_factor * 0.05)
            .clamp(0.0, 1.0);

        let recommended_status = recommended_status(staleness, entry.meta.status);

        StalenessReport {
            key: entry.key.clone(),
            staleness,
            age_factor,
            usage_factor,
            instability_factor,
            contradiction_factor,
            source_factor,
            recommended_status,
        }
    }
}

impl Default for StalenessScorer {
    /// 等价于 [`StalenessScorer::new`]。
    fn default() -> Self {
        Self::new()
    }
}

/// 单条记忆的陈旧度评分明细。
#[derive(Debug, Clone)]
pub struct StalenessReport {
    /// 被评分的记忆键。
    pub key: String,
    /// 综合陈旧度（0.0–1.0）。
    pub staleness: f32,
    /// 距离最近一次被检索（recall）的时间，若没有则回退为创建时间。
    pub age_factor: f32,
    /// 低检索次数因子。
    pub usage_factor: f32,
    /// `1.0 - stability`。
    pub instability_factor: f32,
    /// 0.0（无冲突）或 0.5（处于某个冲突组中）。
    pub contradiction_factor: f32,
    /// `1.0 - source.default_confidence()`。
    pub source_factor: f32,
    /// 复核器建议将该条目转移到的状态。
    pub recommended_status: MemoryStatus,
}

/** 根据条目最近一次检索活动计算年龄因子。
 *
 * `<7天 → 0.0`，`7–30天 → 0.2`，`30–90天 → 0.5`，`>90天 → 0.8`。
 */
fn age_factor(entry: &TypedMemoryEntry, now: DateTime<Utc>) -> f32 {
    /* 仅更新元数据的检索操作也会改变 StoreItem::updated_at，如果用这个
     * 字段判断，会让遥测写入看起来像语义编辑。因此这里保留一个专门的
     * 检索时间戳字段，否则回退到创建时间。 */
    let secs = entry.meta.last_recalled_at.unwrap_or(entry.raw.created_at);
    let Some(then) = DateTime::<Utc>::from_timestamp(secs as i64, 0) else {
        // 时间戳无法解析——按未知年龄（中性值）处理。
        return 0.2;
    };
    let age = now.signed_duration_since(then);
    if age < Duration::days(7) {
        0.0
    } else if age < Duration::days(30) {
        0.2
    } else if age < Duration::days(90) {
        0.5
    } else {
        0.8
    }
}

/** 基于真实检索遥测数据计算的低使用度因子。
 *
 * `recall_count` 由共享的检索路径统一递增，与 Dreaming 消费的是同一份
 * 信号。修订次数衡量的是编辑次数，而不是有用程度。
 */
fn low_usage_factor(meta: &MemoryMeta) -> f32 {
    1.0 - (meta.recall_count as f32 / 3.0).min(1.0)
}

/** 将陈旧度分数映射为建议状态。
 *
 * 已经归档/被取代的条目保持其终态——复核器绝不会重新激活此前某次
 * 复核已经淘汰的条目。
 */
fn recommended_status(staleness: f32, current: MemoryStatus) -> MemoryStatus {
    match current {
        // 终态是粘性的，不会被改变。
        MemoryStatus::Archived | MemoryStatus::Superseded => current,
        MemoryStatus::Draft | MemoryStatus::Active => {
            if staleness >= STALENESS_ARCHIVE_MIN {
                MemoryStatus::Archived
            } else if staleness >= STALENESS_SUPERSEDED_MIN {
                MemoryStatus::Superseded
            } else {
                MemoryStatus::Active
            }
        }
    }
}

// ── ConflictDetector ───────────────────────────────────────────────────

/** 将拥有相同主题 + 类型、但内容哈希不同的记忆分组。
 *
 * 当两条记忆位于同一个 `(topic, memory_type)` 分桶中，但去除首尾空白后的
 * 内容哈希不同时，就称它们“冲突”——也就是说它们对同一主题给出了不同的
 * 说法。完全相同的内容（真正的重复）*不会*被本检测器视为冲突；去重逻辑
 * 由别处（`memory_promoter.rs`）负责。
 */
pub struct ConflictDetector;

impl ConflictDetector {
    /// 创建一个新的检测器。
    pub fn new() -> Self {
        Self
    }

    /** 在一组条目中检测冲突组。
     *
     * 仅当同一个 `(topic, memory_type)` 分桶下存在 **2 条或更多**内容
     * 哈希*不同*的条目时，才会返回一个冲突组。全部内容相同的分组（纯
     * 重复）会被丢弃——它们属于去重候选，而不是冲突。
     */
    pub fn detect(&self, entries: &[TypedMemoryEntry]) -> Vec<ConflictGroup> {
        // 按 (topic, memory_type) 分桶 → (哈希, 条目下标) 列表。
        let mut buckets: HashMap<(String, MemoryType), Vec<(u64, usize)>> = HashMap::new();
        for (idx, entry) in entries.iter().enumerate() {
            let hash = content_hash(&entry.content);
            buckets
                .entry((entry.meta.topic.clone(), entry.meta.memory_type))
                .or_default()
                .push((hash, idx));
        }

        let mut groups = Vec::new();
        for ((topic, memory_type), members) in buckets {
            if members.len() < 2 {
                continue;
            }
            // 至少需要两个不同的哈希才能称之为冲突。
            let distinct_hashes: std::collections::HashSet<u64> =
                members.iter().map(|(h, _)| *h).collect();
            if distinct_hashes.len() < 2 {
                continue;
            }
            let group_entries = members
                .into_iter()
                .filter_map(|(_, idx)| entries.get(idx).cloned())
                .collect();
            groups.push(ConflictGroup {
                topic,
                memory_type,
                entries: group_entries,
            });
        }
        groups
    }
}

impl Default for ConflictDetector {
    /// 等价于 [`ConflictDetector::new`]。
    fn default() -> Self {
        Self::new()
    }
}

/// 在同一 `(topic, memory_type)` 上发生冲突的一组（≥2 条）记忆。
#[derive(Debug, Clone)]
pub struct ConflictGroup {
    /// 共同的主题。
    pub topic: String,
    /// 共同的记忆类型。
    pub memory_type: MemoryType,
    /// 冲突的条目（≥2 条，且至少存在两种不同内容）。
    pub entries: Vec<TypedMemoryEntry>,
}

/// 语义记忆冲突提案中的一个成员。
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MemoryConflictMember {
    /// 该成员的记忆键。
    pub key: String,
    /// 该成员的记忆内容。
    pub content: String,
    /// 该成员的置信度。
    pub confidence: f32,
    /// 该成员当前的状态。
    pub status: MemoryStatus,
    /// 该成员被检索的次数。
    pub recall_count: u32,
    /// 该成员最近一次更新的时间戳。
    pub updated_at: u64,
}

/// 仅做分析、描述某个需要显式选择的冲突的提案。
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MemoryConflictProposal {
    /// 冲突所在的主题。
    pub topic: String,
    /// 冲突所在的记忆类型。
    pub memory_type: MemoryType,
    /// 建议作为主条目的键。
    pub recommended_primary_key: String,
    /// 参与该冲突的所有成员。
    pub members: Vec<MemoryConflictMember>,
}

/// 用于撤销一次已获批准的合并操作的、合并前精确状态快照。
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MemoryMergeSnapshot {
    /// 该条目的记忆键。
    pub key: String,
    /// 合并前的记忆内容。
    pub content: String,
    /// 合并前的元数据。
    pub meta: MemoryMeta,
}

/// 应用一次冲突提案后的结果。
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AppliedMemoryMerge {
    /// 合并后存活下来的主条目键。
    pub primary_key: String,
    /// 被主条目取代的所有条目键。
    pub superseded_keys: Vec<String>,
    /// 合并前所有条目的快照，用于回滚。
    pub before: Vec<MemoryMergeSnapshot>,
    /// 可用于之后回滚的持久化操作批次标识。
    #[serde(default)]
    pub batch_id: Option<String>,
}

impl MemoryConflictProposal {
    /// 从一个冲突组构建一份冲突提案；空组返回 `None`。
    pub(crate) fn from_group(group: &ConflictGroup) -> Option<Self> {
        let ordered = ordered_conflict_entries(group);
        let recommended_primary_key = ordered.first()?.key.clone();
        let members = ordered
            .into_iter()
            .map(|entry| MemoryConflictMember {
                key: entry.key,
                content: entry.content,
                confidence: entry.meta.confidence,
                status: entry.meta.status,
                recall_count: entry.meta.recall_count,
                updated_at: entry.raw.updated_at,
            })
            .collect();
        Some(Self {
            topic: group.topic.clone(),
            memory_type: group.memory_type,
            recommended_primary_key,
            members,
        })
    }
}

/// 按置信度（降序）、更新时间（降序）、键（升序）对冲突组内的条目排序。
pub(super) fn ordered_conflict_entries(group: &ConflictGroup) -> Vec<TypedMemoryEntry> {
    let mut ordered = group.entries.clone();
    ordered.sort_by(|a, b| {
        b.meta
            .confidence
            .partial_cmp(&a.meta.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                b.raw
                    .updated_at
                    .cmp(&a.raw.updated_at)
                    .then_with(|| a.key.cmp(&b.key))
            })
    });
    ordered
}

/** 用于冲突比较的内容哈希——对去除首尾空白后的内容做 FNV-1a。
 *
 * 与 `memory_promoter.rs` 中现有的去重哈希保持一致，使两套系统对“内容
 * 相同”的判定标准一致。
 */
fn content_hash(content: &str) -> u64 {
    fnv1a_64(content.trim().as_bytes())
}

// ── MemoryMerger ───────────────────────────────────────────────────────

/** 将一个 [`ConflictGroup`] 合并为单一的主条目，并取代其余条目。
 *
 * 合并策略：
 * - `confidence` 最高的条目获胜（相同则按 `updated_at` 更新时间较新者
 *   优先，仍相同则按键排序以保证确定性）。
 * - 主条目的内容保持原样不变；来源信息记录在审计数据中。
 * - 次要条目会被改写为 `status = Superseded`，`superseded_by = <主条目键>`，
 *   其 `revision_count` 会被并入主条目。
 * - 管理器会先准备好完整的分组，再对任何成员做投影，然后在业务审计中
 *   以稳定身份记录每个成员。
 */
pub struct MemoryMerger<'a> {
    /// 承载合并变更的持久化状态管理器。
    manager: &'a MemoryLayerManager,
}

/// 合并一个冲突组后的结果。
#[derive(Debug, Clone)]
pub struct MergeResult {
    /// 存活下来的主条目的键。
    pub primary_key: String,
    /// 被主条目取代的所有条目的键。
    pub superseded_keys: Vec<String>,
    /// 供之后回滚使用的持久化操作批次标识。
    pub batch_id: Option<String>,
}

impl<'a> MemoryMerger<'a> {
    /// 将合并策略绑定到权威的持久化变更所有者上。
    pub fn new(manager: &'a MemoryLayerManager) -> Self {
        Self { manager }
    }

    /** 合并单个冲突组。
     *
     * 当分组条目少于 2 条（无需合并）时，返回 `Ok(MergeResult)`，
     * 其 `superseded_keys` 为空。
     */
    pub async fn merge_group(&self, group: &ConflictGroup) -> Result<MergeResult> {
        self.manager.commit_merge_group(group).await
    }
}

// ── MemoryReviewer（编排器） ──────────────────────────────────────

/// 编排一次仅做分析的复核流程：扫描 → 评分 → 提出建议。
pub struct MemoryReviewer {
    /// 用于计算陈旧度的评分器。
    scorer: StalenessScorer,
    /// 用于发现语义冲突的检测器。
    conflict_detector: ConflictDetector,
}

/// 一次复核流程的可调参数。
#[derive(Debug, Clone)]
pub struct ReviewConfig {
    /// 是否在会话结束时运行一次复核。默认：`false`。
    pub review_on_session_end: bool,
    /// 每次复核返回的冲突提案数量上限。默认：`10`。
    pub max_conflicts_per_review: usize,
    /**
     * 单个提案允许包含的最大成员数。更大的分组会被上报，但不会出现在
     * 可操作的输出中，以此控制 JSONL 与提示词的膨胀。
     */
    pub max_conflict_members: usize,
    /// 是否在复核期间运行技能候选检测。默认：`true`。
    pub detect_skill_candidates: bool,
    /// 是否为新候选自动生成 SKILL.md 草稿。默认：`false`。
    pub auto_generate_drafts: bool,
}

impl Default for ReviewConfig {
    /// 返回默认的复核配置。
    fn default() -> Self {
        Self {
            review_on_session_end: false,
            max_conflicts_per_review: 10,
            max_conflict_members: 16,
            detect_skill_candidates: true,
            auto_generate_drafts: false,
        }
    }
}

/// 复核流程中产生的单条提案。
#[derive(Debug, Clone)]
pub enum ReviewChange {
    /// 一条记忆的陈旧度越过了复核阈值。
    StalenessSuggested {
        /// 该记忆的键。
        key: String,
        /// 建议转移到的状态。
        recommended_status: MemoryStatus,
        /// 该记忆的陈旧度分数。
        staleness: f32,
    },
    /// 一个语义冲突需要显式选择主条目。
    ConflictProposed {
        /// 冲突所在的主题。
        topic: String,
        /// 建议作为主条目的键。
        recommended_primary_key: String,
        /// 参与冲突的所有成员键。
        member_keys: Vec<String>,
    },
    /// 根据观察到的模式提出了一个新的技能候选。
    CandidateProposed {
        /// 候选技能的名称。
        name: String,
        /// 支撑该候选的样本数量。
        sample_count: usize,
    },
    /// 为某个候选生成了一份 SKILL.md 草稿。
    DraftGenerated {
        /// 候选技能的名称。
        name: String,
        /// 生成的草稿文件路径。
        path: String,
    },
}

/// 一次复核流程的汇总结果。
#[derive(Debug, Clone, Default)]
pub struct ReviewReport {
    /// 在温层（warm layer）中扫描到的条目总数。
    pub total_scanned: usize,
    /// 陈旧度越过标记阈值（≥ 0.35）的条目数量。
    pub stale_count: usize,
    /// 发现的冲突组数量。
    pub conflict_groups: usize,
    /// 可解释的陈旧度建议；复核器本身不会改变任何状态。
    pub staleness_suggestions: Vec<StalenessReport>,
    /// 语义冲突提案，数量受 `max_conflicts_per_review` 限制。
    pub conflict_proposals: Vec<MemoryConflictProposal>,
    /// 本次复核中提出的技能候选数量。
    pub candidates_proposed: usize,
    /// 本次复核中生成的 SKILL.md 草稿数量。
    pub drafts_generated: usize,
    /// 按确定性顺序排列的各条提案。
    pub changes: Vec<ReviewChange>,
}

impl MemoryReviewer {
    /// 使用默认的评分器与冲突检测器创建一个新的复核器。
    pub fn new() -> Self {
        Self {
            scorer: StalenessScorer::new(),
            conflict_detector: ConflictDetector::new(),
        }
    }

    /** 分析温层（warm layer）而不修改它。
     *
     * 确定性的生命周期维护属于 `Dreaming` 的职责；语义层面的冲突解决
     * 则属于显式调用 `MemoryMerger` 的调用方的职责。
     */
    pub async fn review(
        &self,
        typed_store: &TypedMemoryStore,
        config: &ReviewConfig,
    ) -> Result<ReviewReport> {
        let now = Utc::now();
        let mut report = ReviewReport::default();

        // ── 1. 扫描温层 ──
        let entries = typed_store
            .list_typed(
                crate::evolution::layer::WARM_NAMESPACE,
                &MemoryFilter::new(),
            )
            .await?;
        report.total_scanned = entries.len();
        if entries.is_empty() {
            return Ok(report);
        }

        // ── 2. 冲突检测 ──
        let reviewable_entries: Vec<_> = entries
            .iter()
            .filter(|entry| entry.meta.status != MemoryStatus::Superseded)
            .cloned()
            .collect();
        let conflict_groups = self.conflict_detector.detect(&reviewable_entries);
        /* 收集所有参与任意冲突组的键的集合，以便评分阶段能够
         * 标记它们。 */
        let mut conflicted_keys: std::collections::HashSet<String> =
            std::collections::HashSet::new();
        for group in &conflict_groups {
            for entry in &group.entries {
                conflicted_keys.insert(entry.key.clone());
            }
        }
        report.conflict_groups = conflict_groups.len();

        // ── 3. 为每条条目评分 ──
        let mut scored: Vec<StalenessReport> = Vec::with_capacity(entries.len());
        for entry in &entries {
            let has_contradiction = conflicted_keys.contains(&entry.key);
            scored.push(self.scorer.score(entry, now, has_contradiction));
        }
        report.staleness_suggestions = scored
            .into_iter()
            .filter(|entry| entry.staleness >= STALENESS_ACTIVE_MAX)
            .collect();
        report.stale_count = report.staleness_suggestions.len();
        for suggestion in &report.staleness_suggestions {
            report.changes.push(ReviewChange::StalenessSuggested {
                key: suggestion.key.clone(),
                recommended_status: suggestion.recommended_status,
                staleness: suggestion.staleness,
            });
        }

        report.conflict_proposals = conflict_groups
            .iter()
            .filter(|group| group.entries.len() <= config.max_conflict_members)
            .take(config.max_conflicts_per_review)
            .filter_map(MemoryConflictProposal::from_group)
            .collect();
        for proposal in &report.conflict_proposals {
            report.changes.push(ReviewChange::ConflictProposed {
                topic: proposal.topic.clone(),
                recommended_primary_key: proposal.recommended_primary_key.clone(),
                member_keys: proposal
                    .members
                    .iter()
                    .map(|member| member.key.clone())
                    .collect(),
            });
        }

        Ok(report)
    }
}

impl Default for MemoryReviewer {
    /// 等价于 `Self::new()`。
    fn default() -> Self {
        Self::new()
    }
}

// ── 测试 ──────────────────────────────────────────────────────────────

/// 记忆复核流程的单元测试。
#[cfg(test)]
mod tests {
    use super::*;
    use echo_core::memory::store::StoreItem;
    use echo_core::memory::types::MemorySource;
    use echo_state::memory::store::InMemoryStore;
    use std::sync::Arc;

    /// 用于测试的空操作 ChangeLog。
    struct NullChangeLog;
    impl ChangeLog for NullChangeLog {
        /// 空操作：直接丢弃条目。
        fn record(&self, _entry: super::super::audit::ChangeEntry) -> Result<()> {
            Ok(())
        }
        /// 空操作：总是视为已记录。
        fn record_idempotent(
            &self,
            _entry: super::super::audit::ChangeEntry,
        ) -> Result<super::super::audit::ChangeRecordOutcome> {
            Ok(super::super::audit::ChangeRecordOutcome::AlreadyRecorded)
        }
        /// 空操作：总是返回空列表。
        fn query(
            &self,
            _filter: &super::super::audit::ChangeFilter,
        ) -> Result<Vec<super::super::audit::ChangeEntry>> {
            Ok(Vec::new())
        }
        /// 空操作：总是返回 `None`。
        fn latest_for(
            &self,
            _entity_type: EntityType,
            _entity_key: &str,
        ) -> Result<Option<super::super::audit::ChangeEntry>> {
            Ok(None)
        }
        /// 空操作：总是返回 `0`。
        fn len(&self) -> usize {
            0
        }
    }

    /// 构造一条用于测试的类型化记忆条目。
    fn make_entry(key: &str, content: &str, meta: MemoryMeta, updated_at: u64) -> TypedMemoryEntry {
        let mut raw = StoreItem::new(
            vec!["agent".to_string(), "typed_memories".to_string()],
            key.to_string(),
            serde_json::Value::Null,
        );
        raw.updated_at = updated_at;
        raw.created_at = updated_at;
        TypedMemoryEntry {
            key: key.to_string(),
            content: content.to_string(),
            meta,
            raw,
        }
    }

    /// 返回当前 Unix 时间戳（秒）。
    fn now_secs() -> u64 {
        Utc::now().timestamp().max(0) as u64
    }

    /// 返回若干天前的 Unix 时间戳（秒）。
    fn days_ago_secs(days: i64) -> u64 {
        ((Utc::now() - Duration::days(days)).timestamp()).max(0) as u64
    }

    // ── StalenessScorer 测试 ──

    /// 验证新鲜且高质量的条目保持激活状态。
    #[test]
    fn test_fresh_high_quality_entry_is_active() {
        let scorer = StalenessScorer::new();
        let meta = MemoryMeta::new(
            MemoryType::UserPreference,
            MemorySource::ExplicitSave,
            "style",
        )
        .with_confidence(0.95)
        .with_stability(0.90);
        let entry = make_entry("pref", "User prefers concise output", meta, now_secs());

        let report = scorer.score(&entry, Utc::now(), false);
        assert!(
            report.staleness < STALENESS_ACTIVE_MAX,
            "fresh high-quality entry should be Active, got staleness={:.3}",
            report.staleness
        );
        assert_eq!(report.recommended_status, MemoryStatus::Active);
        // 新鲜条目 ⇒ 年龄因子为 0；ExplicitSave 来源 ⇒ 来源因子为 0。
        assert_eq!(report.age_factor, 0.0);
        assert_eq!(report.source_factor, 0.0);
    }

    /// 验证陈旧且低质量的条目成为归档候选。
    #[test]
    fn test_old_low_quality_entry_is_archive_candidate() {
        let scorer = StalenessScorer::new();
        let meta = MemoryMeta::new(MemoryType::ProjectFact, MemorySource::L3Promotion, "old")
            .with_confidence(0.40)
            .with_stability(0.20);
        // 120 天前创建，从未被修订过。
        let entry = make_entry("old_fact", "Stale project fact", meta, days_ago_secs(120));

        let report = scorer.score(&entry, Utc::now(), false);
        assert!(
            report.staleness >= STALENESS_ARCHIVE_MIN,
            "old low-quality entry should be an archive candidate, got staleness={:.3}",
            report.staleness
        );
        assert_eq!(report.recommended_status, MemoryStatus::Archived);
        assert_eq!(report.age_factor, 0.8);
    }

    /// 验证矛盾信号会提高陈旧度。
    #[test]
    fn test_contradiction_bumps_staleness() {
        let scorer = StalenessScorer::new();
        let meta = MemoryMeta::new(
            MemoryType::ProjectFact,
            MemorySource::AutoExtracted,
            "build",
        );
        let entry = make_entry("fact", "Build uses cargo", meta, now_secs());

        let no_conflict = scorer.score(&entry, Utc::now(), false);
        let with_conflict = scorer.score(&entry, Utc::now(), true);
        assert!(
            with_conflict.staleness > no_conflict.staleness,
            "contradiction factor should raise staleness"
        );
        assert_eq!(no_conflict.contradiction_factor, 0.0);
        assert_eq!(with_conflict.contradiction_factor, 0.5);
    }

    /// 验证终态的推荐状态保持不变。
    #[test]
    fn test_recommended_status_is_sticky_for_terminal() {
        // 已归档/已被取代的条目绝不应因评分而被重新激活。
        let archived = recommended_status(0.99, MemoryStatus::Archived);
        assert_eq!(archived, MemoryStatus::Archived);
        let superseded = recommended_status(0.10, MemoryStatus::Superseded);
        assert_eq!(superseded, MemoryStatus::Superseded);
    }

    /// 验证召回次数会降低低使用率因子。
    #[test]
    fn test_recall_count_lowers_usage_factor() {
        let scorer = StalenessScorer::new();
        let meta_low = MemoryMeta::new(MemoryType::ProjectFact, MemorySource::ExplicitSave, "t");
        let mut meta_high = meta_low.clone();
        meta_high.recall_count = 5;

        let entry_low = make_entry("a", "x", meta_low, now_secs());
        let entry_high = make_entry("b", "x", meta_high, now_secs());
        let r_low = scorer.score(&entry_low, Utc::now(), false);
        let r_high = scorer.score(&entry_high, Utc::now(), false);
        assert!(
            r_high.usage_factor < r_low.usage_factor,
            "more recalls ⇒ lower usage factor"
        );
        assert!(r_low.usage_factor > 0.0);
        assert_eq!(r_high.usage_factor, 0.0);
    }

    /// 验证复核默认配置为手动触发且仅产出提案。
    #[test]
    fn review_defaults_are_manual_and_proposal_only() {
        let config = ReviewConfig::default();
        assert!(!config.review_on_session_end);
        assert_eq!(config.max_conflicts_per_review, 10);
        assert_eq!(config.max_conflict_members, 16);
    }

    // ── ConflictDetector 测试 ──

    /// 验证能检测出同主题同类型但内容不同的冲突。
    #[test]
    fn test_detects_same_topic_type_different_content() {
        let detector = ConflictDetector::new();
        let meta = MemoryMeta::new(
            MemoryType::ProjectFact,
            MemorySource::AutoExtracted,
            "build",
        );
        let entries = vec![
            make_entry("a", "Build uses cargo", meta.clone(), now_secs()),
            make_entry("b", "Build uses make", meta.clone(), now_secs()),
        ];
        let groups = detector.detect(&entries);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].entries.len(), 2);
        assert_eq!(groups[0].topic, "build");
        assert_eq!(groups[0].memory_type, MemoryType::ProjectFact);
    }

    /// 验证不同主题之间不会产生冲突。
    #[test]
    fn test_no_conflict_for_different_topics() {
        let detector = ConflictDetector::new();
        let entries = vec![
            make_entry(
                "a",
                "x",
                MemoryMeta::new(
                    MemoryType::ProjectFact,
                    MemorySource::AutoExtracted,
                    "build",
                ),
                now_secs(),
            ),
            make_entry(
                "b",
                "x",
                MemoryMeta::new(
                    MemoryType::ProjectFact,
                    MemorySource::AutoExtracted,
                    "deploy",
                ),
                now_secs(),
            ),
        ];
        assert!(detector.detect(&entries).is_empty());
    }

    /// 验证内容相同的条目不会产生冲突。
    #[test]
    fn test_no_conflict_for_identical_content() {
        // 两条内容完全相同的条目属于去重候选，而不是冲突。
        let detector = ConflictDetector::new();
        let meta = MemoryMeta::new(
            MemoryType::ProjectFact,
            MemorySource::AutoExtracted,
            "build",
        );
        let entries = vec![
            make_entry("a", "Build uses cargo", meta.clone(), now_secs()),
            make_entry("b", "Build uses cargo", meta, now_secs()),
        ];
        assert!(detector.detect(&entries).is_empty());
    }

    // ── MemoryMerger 测试 ──

    /// 验证合并时保留置信度最高的条目作为主条目。
    #[tokio::test]
    async fn test_merge_keeps_highest_confidence_as_primary() -> Result<()> {
        let store = Arc::new(InMemoryStore::new());
        let typed = TypedMemoryStore::new(store.clone());
        let dir = tempfile::tempdir()?;
        let manager = MemoryLayerManager::new(dir.keep(), store, Box::new(NullChangeLog));

        let meta_high =
            MemoryMeta::new(MemoryType::ProjectFact, MemorySource::ExplicitSave, "build")
                .with_confidence(0.95);
        let meta_low = MemoryMeta::new(
            MemoryType::ProjectFact,
            MemorySource::AutoExtracted,
            "build",
        )
        .with_confidence(0.50);
        typed
            .put_typed(
                crate::evolution::layer::WARM_NAMESPACE,
                "high",
                "Build uses cargo 1.80",
                meta_high,
            )
            .await
            .unwrap();
        typed
            .put_typed(
                crate::evolution::layer::WARM_NAMESPACE,
                "low",
                "Build uses cargo 1.70",
                meta_low,
            )
            .await
            .unwrap();

        let group = ConflictDetector::new()
            .detect(
                &typed
                    .list_typed(
                        crate::evolution::layer::WARM_NAMESPACE,
                        &MemoryFilter::new(),
                    )
                    .await
                    .unwrap(),
            )
            .into_iter()
            .next()
            .expect("should detect one group");

        let result = MemoryMerger::new(&manager)
            .merge_group(&group)
            .await
            .unwrap();
        assert_eq!(result.primary_key, "high");
        assert_eq!(result.superseded_keys, vec!["low".to_string()]);

        // 主条目的内容保持原样不变；来源信息属于审计元数据的范畴。
        let primary = typed
            .get_typed(crate::evolution::layer::WARM_NAMESPACE, "high")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(primary.content, "Build uses cargo 1.80");

        // 次要条目应变为 Superseded 状态，且 superseded_by 指向主条目。
        let secondary = typed
            .get_typed(crate::evolution::layer::WARM_NAMESPACE, "low")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(secondary.meta.status, MemoryStatus::Superseded);
        assert_eq!(secondary.meta.superseded_by.as_deref(), Some("high"));
        Ok(())
    }

    /// 验证单条目合并是空操作。
    #[tokio::test]
    async fn test_merge_single_entry_is_noop() -> Result<()> {
        let store = Arc::new(InMemoryStore::new());
        let dir = tempfile::tempdir()?;
        let manager = MemoryLayerManager::new(dir.keep(), store, Box::new(NullChangeLog));

        let group = ConflictGroup {
            topic: "build".to_string(),
            memory_type: MemoryType::ProjectFact,
            entries: vec![make_entry(
                "solo",
                "only one",
                MemoryMeta::new(MemoryType::ProjectFact, MemorySource::ExplicitSave, "build"),
                now_secs(),
            )],
        };
        let result = MemoryMerger::new(&manager)
            .merge_group(&group)
            .await
            .unwrap();
        assert!(result.superseded_keys.is_empty());
        Ok(())
    }

    // ── MemoryReviewer 端到端测试 ──

    /// 验证复核会报告陈旧与冲突，但不修改数据。
    #[tokio::test]
    async fn test_review_reports_stale_and_conflicts_without_mutation() {
        let mem_store = Arc::new(InMemoryStore::new());
        let store: Arc<dyn echo_core::memory::store::Store> = mem_store.clone();
        let typed = TypedMemoryStore::new(store.clone());

        // (1) 一条非常陈旧的条目 → 应该被报告为归档候选。
        let stale_meta = MemoryMeta::new(MemoryType::ProjectFact, MemorySource::L3Promotion, "old")
            .with_confidence(0.30)
            .with_stability(0.20);
        typed
            .put_typed(
                crate::evolution::layer::WARM_NAMESPACE,
                "stale",
                "Long-forgotten fact",
                stale_meta,
            )
            .await
            .unwrap();
        /* 将底层条目的时间戳强制改为很久以前。
         * 这里使用 `put_raw`，这样 Store 就不会用当前时间覆盖 updated_at。 */
        {
            let item = store
                .get(crate::evolution::layer::WARM_NAMESPACE, "stale")
                .await
                .unwrap()
                .unwrap();
            let mut past = item;
            past.created_at = days_ago_secs(200);
            past.updated_at = days_ago_secs(200);
            mem_store.put_raw(past).await;
        }

        // (2) 同一主题下的两条冲突条目 → 应该产生一份提案。
        let c1 = MemoryMeta::new(MemoryType::ProjectFact, MemorySource::ExplicitSave, "build")
            .with_confidence(0.95);
        let c2 = MemoryMeta::new(
            MemoryType::ProjectFact,
            MemorySource::AutoExtracted,
            "build",
        )
        .with_confidence(0.55);
        typed
            .put_typed(
                crate::evolution::layer::WARM_NAMESPACE,
                "build_a",
                "Build uses cargo 1.80",
                c1,
            )
            .await
            .unwrap();
        typed
            .put_typed(
                crate::evolution::layer::WARM_NAMESPACE,
                "build_b",
                "Build uses cargo 1.70",
                c2,
            )
            .await
            .unwrap();

        // (3) 一条新鲜、高质量的条目 → 应该保持不变。
        let good = MemoryMeta::new(
            MemoryType::UserPreference,
            MemorySource::ExplicitSave,
            "style",
        )
        .with_confidence(0.95)
        .with_stability(0.90);
        typed
            .put_typed(
                crate::evolution::layer::WARM_NAMESPACE,
                "good",
                "User prefers concise output",
                good,
            )
            .await
            .unwrap();

        let config = ReviewConfig::default();
        let report = MemoryReviewer::new().review(&typed, &config).await.unwrap();

        assert_eq!(report.total_scanned, 4);
        assert!(
            report
                .staleness_suggestions
                .iter()
                .any(|s| s.key == "stale")
        );
        assert_eq!(report.conflict_proposals.len(), 1);
        assert_eq!(
            report.conflict_proposals[0].recommended_primary_key,
            "build_a"
        );

        // 分析过程不能归档或合并任何内容。
        let stale = typed
            .get_typed(crate::evolution::layer::WARM_NAMESPACE, "stale")
            .await
            .unwrap()
            .expect("stale entry remains present");
        assert_eq!(stale.meta.status, MemoryStatus::Active);
        let build_b = typed
            .get_typed(crate::evolution::layer::WARM_NAMESPACE, "build_b")
            .await
            .unwrap()
            .expect("conflict member remains present");
        assert_eq!(build_b.meta.status, MemoryStatus::Active);
        assert_eq!(build_b.content, "Build uses cargo 1.70");
    }

    /// 验证复核会省略成员数量超限的冲突提案。
    #[tokio::test]
    async fn review_omits_oversized_conflict_proposals() -> crate::error::Result<()> {
        let store = Arc::new(InMemoryStore::new());
        let typed = TypedMemoryStore::new(store);
        let meta = MemoryMeta::new(
            MemoryType::ProjectFact,
            MemorySource::AutoExtracted,
            "oversized",
        );
        for index in 0..17 {
            typed
                .put_typed(
                    crate::evolution::layer::WARM_NAMESPACE,
                    &format!("member_{index}"),
                    &format!("value_{index}"),
                    meta.clone(),
                )
                .await?;
        }
        let report = MemoryReviewer::new()
            .review(&typed, &ReviewConfig::default())
            .await?;

        assert_eq!(report.conflict_groups, 1);
        assert!(report.conflict_proposals.is_empty());
        Ok(())
    }
}
