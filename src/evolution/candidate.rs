/*!
 * 技能候选检测 —— 从类型化记忆中发现可复用的模式。
 *
 * 扫描 `TypedMemoryStore` 中的 `WorkflowPattern`(工作流模式)与
 * `DebuggingLesson`(调试经验)条目,当观测数量足够多时提出 [`SkillCandidate`]
 * (技能候选)。这是技能创建流水线的第一步:
 *
 * ```text
 * observations → SkillCandidateDetector → SkillCandidate → SkillDraftGenerator → SKILL.md
 * ```
 *
 * # 检测逻辑
 *
 * - 按 `(topic, memory_type)` 对 `WorkflowPattern` / `DebuggingLesson` 记忆分组。
 * - 当一个分组的条目数 ≥ `min_observations`(默认 3)时,提出一个候选。
 * - 会检查 `["agent", "skill_candidates"]` 中已有的候选以避免重复;被强化的
 *   候选会更新其样本数量。
 * - 每个新候选都会向 `Curator` 生命周期系统注册。
 */

use chrono::{DateTime, Utc};
use echo_core::error::ReactError;
use echo_core::memory::store::StoreCompareAndPutOutcome;
use echo_core::memory::types::{MemorySource, MemoryType, TypedMemoryValue};
use echo_core::utils::fs::FileDurability;
use echo_state::journal::file::FileEventJournal;
use echo_state::journal::{EventJournal, JournalDurabilityStatus, PreparedJournalBatch};
use echo_state::memory::typed_store::{MemoryFilter, TypedMemoryEntry, TypedMemoryStore};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use super::audit::{ChangeEntryBuilder, ChangeFilter, ChangeLog, ChangeType, EntityType};
use super::curator::Curator;
use super::layer::EvolutionObserver;
use crate::error::Result;

// ── 常量 ──────────────────────────────────────────────────────────

/// Store 中技能候选提案所使用的命名空间。
pub const CANDIDATE_NAMESPACE: &[&str] = &["agent", "skill_candidates"];
/// Store 中候选权属绑定标记所使用的命名空间。
const CANDIDATE_AUTHORITY_NAMESPACE: &[&str] = &["agent", "skill_candidate_authority"];
/// 候选权属绑定标记在 Store 中的键名。
const CANDIDATE_AUTHORITY_KEY: &str = "binding";
/// 候选权属绑定标记在 ChangeLog 审计流水中的键名。
const CANDIDATE_AUTHORITY_AUDIT_KEY: &str = "__skill_candidate_authority_binding__";

/// 一次候选写入操作的完整描述,用于恢复日志中的"准备-结算"两阶段记录。
#[derive(Clone, Debug, Serialize, Deserialize)]
struct CandidateOperation {
    /// 本次操作的唯一标识。
    id: String,
    /// 本次操作所属的权属(authority)标识。
    authority_id: String,
    /// 候选在 Store 中的键。
    key: String,
    /// 变更前的值(创建操作为 `None`)。
    before: Option<serde_json::Value>,
    /// 变更后的值。
    after: serde_json::Value,
    /// 是否需要向 Curator 注册该候选。
    register_candidate: bool,
    /// 关联的审计变更条目。
    audit: super::audit::ChangeEntry,
}

/// 候选恢复日志中记录的事件类型。
#[derive(Clone, Debug, Serialize, Deserialize)]
enum CandidateOperationEvent {
    /// 日志被绑定到某个权属标识(仅允许发生一次)。
    Bound { authority_id: String },
    /// 一次操作已准备就绪,尚未结算。
    Prepared(Box<CandidateOperation>),
    /// 指定 id 的操作已结算完成。
    Settled { id: String },
}

/// 候选操作的持久化恢复日志,用于在崩溃后重放未结算的写入。
struct CandidateOperationJournal {
    /// 底层的文件事件日志。
    journal: Arc<FileEventJournal<CandidateOperationEvent>>,
    /// 同一日志路径下所有实例共享的串行化锁,保证并发调用互斥。
    serial: Arc<tokio::sync::Mutex<()>>,
}

/// 从日志重放得到的历史状态:权属绑定与操作列表(含是否已结算)。
struct CandidateJournalHistory {
    /// 日志当前绑定的权属标识(尚未绑定时为 `None`)。
    authority_id: Option<String>,
    /// 按重放顺序排列的操作及其结算状态。
    operations: Vec<(CandidateOperation, bool)>,
}

impl CandidateOperationJournal {
    /// 打开(或创建)指定路径下的候选操作日志,并绑定该路径对应的串行化锁。
    fn open(path: &Path) -> Result<Self> {
        let journal = Arc::new(FileEventJournal::open(path, FileDurability::SyncData)?);
        static SERIALS: OnceLock<Mutex<HashMap<PathBuf, Weak<tokio::sync::Mutex<()>>>>> =
            OnceLock::new();
        let mut serials = SERIALS
            .get_or_init(|| Mutex::new(HashMap::new()))
            .lock()
            .map_err(|error| {
                ReactError::Other(format!(
                    "candidate operation lock registry is poisoned: {error}"
                ))
            })?;
        let serial = match serials.get(journal.path()).and_then(Weak::upgrade) {
            Some(existing) => existing,
            None => {
                let created = Arc::new(tokio::sync::Mutex::new(()));
                serials.insert(journal.path().to_path_buf(), Arc::downgrade(&created));
                created
            }
        };
        Ok(Self { journal, serial })
    }

    /// 返回该日志路径关联的串行化锁的克隆句柄。
    fn serial(&self) -> Arc<tokio::sync::Mutex<()>> {
        Arc::clone(&self.serial)
    }

    /// 追加一条事件并要求其被确认落盘,否则返回错误。
    fn append_confirmed(&self, event: CandidateOperationEvent) -> Result<()> {
        let batch = PreparedJournalBatch::new(vec![event])
            .map_err(|error| ReactError::Other(error.to_string()))?;
        let receipt = self
            .journal
            .append_batch(batch)
            .map_err(|error| ReactError::Other(error.to_string()))?;
        match receipt.durability() {
            JournalDurabilityStatus::Confirmed => Ok(()),
            JournalDurabilityStatus::Degraded { .. } => self.journal.sync_data(),
            JournalDurabilityStatus::Unconfirmed => Err(ReactError::Other(
                "candidate operation journal append has unconfirmed durability".into(),
            )),
        }
    }

    /// 校验并记录一次"准备"事件,标志该操作即将被投影到 Store。
    fn prepare(&self, operation: CandidateOperation) -> Result<()> {
        Self::validate_operation(&operation)?;
        self.append_confirmed(CandidateOperationEvent::Prepared(Box::new(operation)))
    }

    /**
     * 将日志绑定到指定的权属标识;若日志为空则完成首次绑定,
     * 若已绑定到相同标识则视为幂等成功,绑定到不同标识则报错。
     */
    fn bind(&self, authority_id: &str) -> Result<()> {
        if authority_id.is_empty() {
            return Err(ReactError::Other(
                "candidate authority identity must not be empty".into(),
            ));
        }
        let history = self.history()?;
        match history.authority_id {
            Some(existing) if existing == authority_id => Ok(()),
            Some(existing) => Err(ReactError::Other(format!(
                "candidate journal is bound to authority {existing}, not {authority_id}"
            ))),
            None if history.operations.is_empty() => {
                self.append_confirmed(CandidateOperationEvent::Bound {
                    authority_id: authority_id.to_string(),
                })
            }
            None => Err(ReactError::Other(
                "candidate journal has operations without an authority binding".into(),
            )),
        }
    }

    /**
     * 校验一次候选操作的审计身份与变更形状是否自洽(创建操作必须无
     * `before` 且类型为 `Create`,更新操作必须有 `before` 且类型为 `Update`)。
     */
    fn validate_operation(operation: &CandidateOperation) -> Result<()> {
        let shape_is_valid = matches!(
            (
                operation.register_candidate,
                operation.before.is_none(),
                operation.audit.change_type,
            ),
            (true, true, ChangeType::Create) | (false, false, ChangeType::Update)
        );
        if operation.id.is_empty()
            || operation.authority_id.is_empty()
            || operation.key.is_empty()
            || operation.id != operation.audit.change_id
            || operation.key != operation.audit.entity_key
            || operation.audit.entity_type != EntityType::Skill
            || !shape_is_valid
        {
            return Err(ReactError::Other(format!(
                "candidate operation {} has an invalid audit identity or mutation shape",
                operation.id
            )));
        }
        Ok(())
    }

    /// 记录一条"结算"事件,标志指定 id 的操作已经完成投影。
    fn settle(&self, id: &str) -> Result<()> {
        self.append_confirmed(CandidateOperationEvent::Settled { id: id.to_owned() })
    }

    /// 重放整条日志,重建权属绑定与所有操作(含是否已结算)的历史视图。
    fn history(&self) -> Result<CandidateJournalHistory> {
        self.journal.sync_data()?;
        let mut authority_id = None;
        let mut pending = HashMap::<String, usize>::new();
        let mut known = HashSet::<String>::new();
        let mut history = Vec::<(CandidateOperation, bool)>::new();
        let mut sequence = 0_u64;
        loop {
            let records = self.journal.replay_after(sequence, 512)?;
            if records.is_empty() {
                break;
            }
            for record in &records {
                match record.event.as_ref() {
                    CandidateOperationEvent::Bound {
                        authority_id: observed,
                    } => {
                        if observed.is_empty() || authority_id.is_some() || !history.is_empty() {
                            return Err(ReactError::Other(
                                "candidate journal has an invalid authority binding".into(),
                            ));
                        }
                        authority_id = Some(observed.clone());
                    }
                    CandidateOperationEvent::Prepared(operation) => {
                        Self::validate_operation(operation)?;
                        if authority_id.as_ref() != Some(&operation.authority_id) {
                            return Err(ReactError::Other(format!(
                                "candidate operation {} does not match the journal authority",
                                operation.id
                            )));
                        }
                        if !known.insert(operation.id.clone()) {
                            return Err(ReactError::Other(format!(
                                "candidate operation {} has an invalid or duplicate prepare fact",
                                operation.id
                            )));
                        }
                        let index = history.len();
                        pending.insert(operation.id.clone(), index);
                        history.push((operation.as_ref().clone(), false));
                    }
                    CandidateOperationEvent::Settled { id } => {
                        let Some(index) = pending.remove(id) else {
                            return Err(ReactError::Other(format!(
                                "candidate operation {id} has no pending prepare fact"
                            )));
                        };
                        let Some((_, settled)) = history.get_mut(index) else {
                            return Err(ReactError::Other(format!(
                                "candidate operation {id} history index is invalid"
                            )));
                        };
                        *settled = true;
                    }
                }
                sequence = record.sequence;
            }
        }
        Ok(CandidateJournalHistory {
            authority_id,
            operations: history,
        })
    }
}

// ── SkillCandidate ─────────────────────────────────────────────────────

/// 一个提出的技能候选,由重复出现的记忆观测推导而来。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillCandidate {
    /// 自动生成的技能名称(经过清洗的主题)。
    pub name: String,
    /// 该模式的可读描述。
    pub description: String,
    /// 应当触发该技能的关键词。
    pub trigger_patterns: Vec<String>,
    /// 该模式中常用的工具。
    pub tool_sequence: Vec<String>,
    /// 促成该候选的记忆观测数量。
    pub sample_count: usize,
    /// 由观测质量推导出的置信度(0.0–1.0)。
    pub confidence: f32,
    /// 源记忆共享的主题。
    pub topic: String,
    /// 源观测的记忆类型。
    pub source_type: MemoryType,
    /// 该候选的创建时间。
    #[serde(with = "crate::utils::time::local_rfc3339")]
    pub created_at: DateTime<Utc>,
}

impl SkillCandidate {
    /// 从一组共享同一主题的记忆条目构建候选。
    fn from_group(topic: &str, source_type: MemoryType, entries: &[TypedMemoryEntry]) -> Self {
        let name = sanitize_name(topic);
        let sample_count = entries.len();

        // 置信度:所有条目置信度的平均值,并根据观测数量给予加成。
        let avg_confidence =
            entries.iter().map(|e| e.meta.confidence).sum::<f32>() / sample_count.max(1) as f32;
        let observation_bonus = (sample_count as f32 / 10.0).min(0.15);
        let confidence = (avg_confidence + observation_bonus).min(1.0);

        // 描述:根据来源类型和主题合成简要说明。
        let description = format!(
            "Auto-detected {} pattern for '{}'. Based on {} observations.",
            match source_type {
                MemoryType::WorkflowPattern => "workflow",
                MemoryType::DebuggingLesson => "debugging",
                _ => "usage",
            },
            topic,
            sample_count
        );

        // 触发模式:主题词 + 内容关键词。
        let mut trigger_patterns = vec![topic.to_string()];
        // 从内容中提取简短关键词(前 3 个长度 > 3 字符的唯一单词)。
        let mut seen = std::collections::HashSet::new();
        for entry in entries {
            for word in entry.content.split_whitespace() {
                let w = word.to_lowercase();
                if w.len() > 3 && seen.insert(w.clone()) && trigger_patterns.len() < 5 {
                    trigger_patterns.push(w);
                }
            }
        }

        // 工具序列:从内容中形如 "tool 'X'" 的模式提取工具名称。
        let tool_sequence = extract_tool_names(entries);

        Self {
            name,
            description,
            trigger_patterns,
            tool_sequence,
            sample_count,
            confidence,
            topic: topic.to_string(),
            source_type,
            created_at: Utc::now(),
        }
    }
}

// ── CandidateReport ────────────────────────────────────────────────────

/// 一次候选检测过程的结果。
#[derive(Debug, Clone, Default)]
pub struct CandidateReport {
    /// 本次过程新提出的候选。
    pub new_candidates: Vec<SkillCandidate>,
    /// 被强化(观测数增加)的已有候选。
    pub reinforced: Vec<String>,
    /// 本次检查的分组总数。
    pub groups_scanned: usize,
}

// ── SkillCandidateDetector ─────────────────────────────────────────────

/**
 * 从累积的记忆观测中检测可复用的技能候选。
 *
 * 不依赖 LLM,纯模式匹配 —— 快速且零成本。检测到的候选会存入
 * `["agent", "skill_candidates"]`,并向 [`Curator`] 生命周期系统注册。
 */
pub struct SkillCandidateDetector {
    /// 提出候选所需的最少观测数量。默认: 3。
    pub min_observations: usize,
    /// 每次扫描最多提出的候选数量。默认: 5。
    pub max_candidates_per_scan: usize,
    /// 负责技能生命周期持久化的 Curator。
    curator: Curator,
    /// 可选的演化事件观察者,用于对外发布检测事件。
    observer: Option<Arc<dyn EvolutionObserver>>,
}

impl SkillCandidateDetector {
    /// 创建一个带有显式生命周期持久化的检测器。
    pub fn new(curator: Curator) -> Self {
        Self {
            min_observations: 3,
            max_candidates_per_scan: 5,
            curator,
            observer: None,
        }
    }

    /// 创建一个使用自定义阈值的检测器。
    pub fn with_thresholds(
        curator: Curator,
        min_observations: usize,
        max_candidates_per_scan: usize,
    ) -> Self {
        Self {
            min_observations,
            max_candidates_per_scan,
            curator,
            observer: None,
        }
    }

    /// 使用调用方提供的 curator 状态文件。
    pub fn with_curator(mut self, curator: Curator) -> Self {
        self.curator = curator;
        self
    }

    /// 将新持久化的候选发布给演化事件观察者。
    pub fn with_evolution_observer(mut self, observer: Arc<dyn EvolutionObserver>) -> Self {
        self.observer = Some(observer);
        self
    }

    /**
     * 对类型化记忆存储执行一次检测过程。
     *
     * 扫描 `WorkflowPattern` 与 `DebuggingLesson` 条目,按 `(topic,
     * memory_type)` 分组,并为超过观测阈值的分组提出候选。
     *
     * 新候选会被持久化到 Store 并向 `Curator` 注册;已有候选若观测数增加,
     * 则会被更新。
     */
    pub async fn detect(
        &self,
        typed_store: &TypedMemoryStore,
        change_log: &dyn ChangeLog,
    ) -> Result<CandidateReport> {
        let journal =
            CandidateOperationJournal::open(&self.curator.candidate_operation_journal_path())?;
        let serial = journal.serial();
        let _serial_guard = serial.lock().await;
        let mut authority_id = self
            .validate_candidate_authorities(typed_store, change_log, &journal)
            .await?;
        self.reconcile_candidate_operations(typed_store, change_log, &journal)
            .await?;
        let mut report = CandidateReport::default();

        // 1. 查询 WorkflowPattern 条目。
        let wf_filter = MemoryFilter::new()
            .with_type(MemoryType::WorkflowPattern)
            .with_source(MemorySource::RepeatedWorkflow);
        let workflow_entries = typed_store
            .list_typed(crate::evolution::layer::WARM_NAMESPACE, &wf_filter)
            .await?;

        // 2. 查询 DebuggingLesson 条目(不限来源)。
        let dl_filter = MemoryFilter::new().with_type(MemoryType::DebuggingLesson);
        let debugging_entries = typed_store
            .list_typed(crate::evolution::layer::WARM_NAMESPACE, &dl_filter)
            .await?;

        // 3. 按 (topic, memory_type) 分组。
        let mut groups: HashMap<(String, MemoryType), Vec<TypedMemoryEntry>> = HashMap::new();
        for entry in &workflow_entries {
            groups
                .entry((entry.meta.topic.clone(), MemoryType::WorkflowPattern))
                .or_default()
                .push(entry.clone());
        }
        for entry in &debugging_entries {
            groups
                .entry((entry.meta.topic.clone(), MemoryType::DebuggingLesson))
                .or_default()
                .push(entry.clone());
        }
        report.groups_scanned = groups.len();

        // 4. 加载已有候选以检查重复。
        let existing_filter = MemoryFilter::new();
        let existing_candidates = typed_store
            .list_typed(CANDIDATE_NAMESPACE, &existing_filter)
            .await?;
        let existing_names: std::collections::HashSet<String> =
            existing_candidates.iter().map(|e| e.key.clone()).collect();

        // 5. 对每个超过阈值的分组,提出新候选或强化已有候选。
        let mut ordered_groups: Vec<_> = groups.iter().collect();
        ordered_groups.sort_by(|((topic_a, type_a), _), ((topic_b, type_b), _)| {
            topic_a
                .cmp(topic_b)
                .then_with(|| format!("{type_a:?}").cmp(&format!("{type_b:?}")))
        });
        let mut proposed = 0usize;
        for ((topic, source_type), entries) in ordered_groups {
            if entries.len() < self.min_observations {
                continue;
            }
            if proposed >= self.max_candidates_per_scan {
                break;
            }

            let candidate = SkillCandidate::from_group(topic, *source_type, entries);
            let key = candidate.name.clone();

            if existing_names.contains(&key) {
                // 强化已有候选 —— 更新样本数量。
                if let Some(existing) = typed_store.get_typed(CANDIDATE_NAMESPACE, &key).await? {
                    let previous = parse_candidate(&existing)?;
                    let old_count = previous.sample_count;
                    if candidate.sample_count > old_count {
                        let operation_authority = self
                            .ensure_candidate_authority(
                                typed_store,
                                change_log,
                                &journal,
                                &mut authority_id,
                            )
                            .await?;
                        // 使用新的样本数量与置信度更新。
                        let updated = SkillCandidate {
                            created_at: previous.created_at,
                            ..candidate.clone()
                        };
                        let content = serde_json::to_string(&updated)?;
                        let before = Some(existing.raw.value.clone());
                        let after =
                            TypedMemoryValue::new(&content, existing.meta.clone()).to_value()?;
                        let id = uuid::Uuid::new_v4().to_string();
                        let audit = ChangeEntryBuilder::new(
                            EntityType::Skill,
                            &key,
                            ChangeType::Update,
                        )
                        .before(serde_json::to_value(&previous)?)
                        .after(serde_json::to_value(&updated)?)
                        .reason(format!(
                            "skill candidate reinforced from {old_count} to {} observations on topic '{}'",
                            updated.sample_count, updated.topic
                        ))
                        .trigger("skill_candidate_detector".to_string())
                        .build_with(id.clone(), Utc::now());
                        self.commit_candidate_operation(
                            typed_store,
                            change_log,
                            &journal,
                            CandidateOperation {
                                id,
                                authority_id: operation_authority,
                                key: key.clone(),
                                before,
                                after,
                                register_candidate: false,
                                audit,
                            },
                        )
                        .await?;
                        report.reinforced.push(key);
                    }
                }
            } else {
                let operation_authority = self
                    .ensure_candidate_authority(
                        typed_store,
                        change_log,
                        &journal,
                        &mut authority_id,
                    )
                    .await?;
                // 新候选 —— 通过同一权属所有者完成持久化、注册与审计。
                let content = serde_json::to_string(&candidate)?;
                let after = TypedMemoryValue::new(&content, default_candidate_meta()).to_value()?;
                let id = uuid::Uuid::new_v4().to_string();
                let entry = ChangeEntryBuilder::new(EntityType::Skill, &key, ChangeType::Create)
                    .after(serde_json::to_value(&candidate)?)
                    .reason(format!(
                        "skill candidate proposed from {} observations on topic '{}'",
                        candidate.sample_count, candidate.topic
                    ))
                    .trigger("skill_candidate_detector".to_string())
                    .build_with(id.clone(), Utc::now());
                self.commit_candidate_operation(
                    typed_store,
                    change_log,
                    &journal,
                    CandidateOperation {
                        id,
                        authority_id: operation_authority,
                        key: key.clone(),
                        before: None,
                        after,
                        register_candidate: true,
                        audit: entry,
                    },
                )
                .await?;

                if let Some(observer) = &self.observer {
                    observer.on_skill_candidate_detected(&key).await;
                }
                report.new_candidates.push(candidate);
                proposed += 1;
            }
        }

        Ok(report)
    }

    /// 校验日志、Store 与 ChangeLog 三者记录的权属标识是否一致,
    /// 返回当前已确立的权属标识(若尚未确立则为 `None`)。
    async fn validate_candidate_authorities(
        &self,
        typed_store: &TypedMemoryStore,
        change_log: &dyn ChangeLog,
        journal: &CandidateOperationJournal,
    ) -> Result<Option<String>> {
        let journal_history = journal.history()?;
        let store_authority = self.store_authority_id(typed_store).await?;
        let audit_authority = Self::audit_authority_id(change_log)?;
        match journal_history.authority_id {
            Some(journal_authority) => {
                if store_authority.as_ref() != Some(&journal_authority)
                    || audit_authority.as_ref() != Some(&journal_authority)
                {
                    return Err(ReactError::Other(
                        "candidate journal authority does not match Store and ChangeLog markers"
                            .into(),
                    ));
                }
                Ok(Some(journal_authority))
            }
            None => match (store_authority, audit_authority) {
                (None, None) => Ok(None),
                (Some(store), Some(audit)) if store == audit => Ok(Some(store)),
                _ => Err(ReactError::Other(
                    "candidate authority markers are incomplete or disagree".into(),
                )),
            },
        }
    }

    /**
     * 确保权属标识已经建立:若已存在则直接绑定日志并返回;否则生成新的
     * 标识,依次在 Store、ChangeLog、日志上完成绑定,并回写到 `authority_id`。
     */
    async fn ensure_candidate_authority(
        &self,
        typed_store: &TypedMemoryStore,
        change_log: &dyn ChangeLog,
        journal: &CandidateOperationJournal,
        authority_id: &mut Option<String>,
    ) -> Result<String> {
        if let Some(existing) = authority_id.as_ref() {
            journal.bind(existing)?;
            return Ok(existing.clone());
        }

        let created = uuid::Uuid::new_v4().to_string();
        let marker_meta = echo_core::memory::types::MemoryMeta::new(
            MemoryType::SkillCandidate,
            MemorySource::AutoExtracted,
            "skill_candidate_authority",
        );
        let outcome = typed_store
            .compare_and_put_typed(
                CANDIDATE_AUTHORITY_NAMESPACE,
                CANDIDATE_AUTHORITY_KEY,
                None,
                &created,
                marker_meta,
            )
            .await?;
        if outcome != StoreCompareAndPutOutcome::Applied {
            return Err(ReactError::Other(
                "candidate Store authority marker changed during first binding".into(),
            ));
        }
        let marker = ChangeEntryBuilder::new(
            EntityType::Skill,
            CANDIDATE_AUTHORITY_AUDIT_KEY,
            ChangeType::Create,
        )
        .after(serde_json::json!({ "authority_id": created.clone() }))
        .reason("bind candidate journal to Store and ChangeLog authorities")
        .trigger("skill_candidate_detector_authority".to_string())
        .build_with(format!("candidate_authority_{created}"), Utc::now());
        change_log.record_idempotent(marker)?;
        if self.store_authority_id(typed_store).await?.as_ref() != Some(&created)
            || Self::audit_authority_id(change_log)?.as_ref() != Some(&created)
        {
            return Err(ReactError::Other(
                "candidate authority markers were not durably observable after binding".into(),
            ));
        }
        journal.bind(&created)?;
        *authority_id = Some(created.clone());
        Ok(created)
    }

    /// 读取 Store 中记录的权属标识。
    async fn store_authority_id(&self, typed_store: &TypedMemoryStore) -> Result<Option<String>> {
        let marker = typed_store
            .get_typed(CANDIDATE_AUTHORITY_NAMESPACE, CANDIDATE_AUTHORITY_KEY)
            .await?;
        marker
            .map(|entry| {
                if entry.content.is_empty() {
                    Err(ReactError::Other(
                        "candidate Store authority marker is empty".into(),
                    ))
                } else {
                    Ok(entry.content)
                }
            })
            .transpose()
    }

    /// 读取 ChangeLog 中记录的权属标识。
    fn audit_authority_id(change_log: &dyn ChangeLog) -> Result<Option<String>> {
        let markers = change_log
            .query(&ChangeFilter::new().with_key_prefix(CANDIDATE_AUTHORITY_AUDIT_KEY))?
            .into_iter()
            .filter(|entry| {
                entry.entity_type == EntityType::Skill
                    && entry.entity_key == CANDIDATE_AUTHORITY_AUDIT_KEY
            })
            .collect::<Vec<_>>();
        match markers.as_slice() {
            [] => Ok(None),
            [marker] if marker.change_type == ChangeType::Create => marker
                .after
                .as_ref()
                .and_then(|value| value.get("authority_id"))
                .and_then(serde_json::Value::as_str)
                .filter(|value| !value.is_empty())
                .map(str::to_string)
                .ok_or_else(|| {
                    ReactError::Other("candidate ChangeLog authority marker is invalid".into())
                })
                .map(Some),
            _ => Err(ReactError::Other(
                "candidate ChangeLog has duplicate or invalid authority markers".into(),
            )),
        }
    }

    /**
     * 提交一次候选操作的完整流程:比对当前值、写入日志、投影到 Store、
     * 视需要向 Curator 注册、写审计日志,最后标记日志结算。
     */
    async fn commit_candidate_operation(
        &self,
        typed_store: &TypedMemoryStore,
        change_log: &dyn ChangeLog,
        journal: &CandidateOperationJournal,
        operation: CandidateOperation,
    ) -> Result<()> {
        let current = typed_store
            .inner()
            .get(CANDIDATE_NAMESPACE, &operation.key)
            .await?
            .map(|item| item.value);
        if current != operation.before {
            return Err(ReactError::Other(format!(
                "candidate operation {} is stale for {}",
                operation.id, operation.key
            )));
        }
        journal.prepare(operation.clone())?;
        self.project_candidate_operation(typed_store, &operation, std::slice::from_ref(&operation))
            .await?;
        if operation.register_candidate {
            self.curator
                .register_candidate_with_authority(&operation.key, &operation.authority_id)?;
        }
        change_log.record_idempotent(operation.audit.clone())?;
        journal.settle(&operation.id)
    }

    /**
     * 在每次检测开始前对恢复日志做一致性重放:对每个键取最新一次操作重新
     * 投影,并补齐尚未结算的 Curator 注册与审计记录,使崩溃恢复具备幂等性。
     */
    async fn reconcile_candidate_operations(
        &self,
        typed_store: &TypedMemoryStore,
        change_log: &dyn ChangeLog,
        journal: &CandidateOperationJournal,
    ) -> Result<()> {
        let history = journal.history()?.operations;
        let mut by_key = BTreeMap::<String, Vec<CandidateOperation>>::new();
        for (operation, _) in &history {
            by_key
                .entry(operation.key.clone())
                .or_default()
                .push(operation.clone());
        }
        for operations in by_key.values() {
            let Some(latest) = operations.last() else {
                return Err(ReactError::Other(
                    "candidate operation history has an empty key series".into(),
                ));
            };
            self.project_candidate_operation(typed_store, latest, operations)
                .await?;
        }
        for (operation, settled) in history {
            if operation.register_candidate {
                self.curator
                    .register_candidate_with_authority(&operation.key, &operation.authority_id)?;
            }
            change_log.record_idempotent(operation.audit.clone())?;
            if !settled {
                journal.settle(&operation.id)?;
            }
        }
        Ok(())
    }

    /**
     * 将一次候选操作的结果原子地投影到 Store。若当前值已等于目标值则跳过;
     * 若当前值与历史中已知的某次操作前/后值都不匹配,说明发生了外部并发
     * 修改,返回错误以保证写入闭环("fail closed")。
     */
    async fn project_candidate_operation(
        &self,
        typed_store: &TypedMemoryStore,
        operation: &CandidateOperation,
        history: &[CandidateOperation],
    ) -> Result<()> {
        let current = typed_store
            .inner()
            .get(CANDIDATE_NAMESPACE, &operation.key)
            .await?
            .map(|item| item.value);
        if current.as_ref() == Some(&operation.after) {
            return Ok(());
        }
        let known = history
            .iter()
            .any(|prior| current == prior.before || current.as_ref() == Some(&prior.after));
        if !known {
            return Err(ReactError::Other(format!(
                "candidate operation {} conflicts with externally changed payload {}",
                operation.id, operation.key
            )));
        }
        let projected = TypedMemoryValue::from_value(&operation.after).map_err(|error| {
            ReactError::Other(format!(
                "candidate operation {} has an invalid typed payload: {error}",
                operation.id
            ))
        })?;
        let outcome = typed_store
            .compare_and_put_typed(
                CANDIDATE_NAMESPACE,
                &operation.key,
                current,
                &projected.content,
                projected.meta,
            )
            .await?;
        if outcome == StoreCompareAndPutOutcome::Applied {
            Ok(())
        } else {
            Err(ReactError::Other(format!(
                "candidate operation {} lost its atomic Store comparison for {}",
                operation.id, operation.key
            )))
        }
    }
}

// ── 辅助函数 ────────────────────────────────────────────────────────────

/**
 * 将主题字符串清洗为合法的技能名称。
 *
 * 转小写、将非字母数字字符替换为 `-`、折叠连续的 `-`、去除首尾的 `-`,
 * 并截断到 64 字符。
 */
fn sanitize_name(topic: &str) -> String {
    let mut name: String = topic
        .to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { '-' })
        .collect();
    // 折叠连续的 `-`。
    let mut prev_dash = false;
    name = name
        .chars()
        .filter(|&c| {
            if c == '-' {
                if prev_dash {
                    false
                } else {
                    prev_dash = true;
                    true
                }
            } else {
                prev_dash = false;
                true
            }
        })
        .collect();
    // 去除首尾的 `-`。
    let name = name.trim_matches('-');
    // 兜底:若清洗后结果为空,使用基于稳定哈希的名称。
    if name.is_empty() {
        let hash = echo_core::utils::hash::fnv1a_64(topic.as_bytes());
        return format!("candidate-{:x}", hash);
    }
    // 截断长度。
    name.chars().take(64).collect()
}

/**
 * 从记忆内容中提取工具名称。
 *
 * 在内容中查找形如 `tool 'name'`(空格+引号)、`tool:name`、`Tool:name`
 * 或 `tool "name"` 的模式。
 */
fn extract_tool_names(entries: &[TypedMemoryEntry]) -> Vec<String> {
    let mut tools: Vec<String> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for entry in entries {
        let content = &entry.content;
        let lower = content.to_lowercase();
        // 扫描所有 "tool" 出现的位置,提取紧随其后的名称。
        for (idx, _) in lower.match_indices("tool") {
            let after = lower.get(idx.saturating_add(4)..).unwrap_or_default();
            let name = extract_tool_name_after_keyword(after);
            if !name.is_empty() && seen.insert(name.to_string()) && tools.len() < 8 {
                tools.push(name.to_string());
            }
        }
    }
    tools
}

/**
 * 给定紧跟在 "tool" 之后的文本,提取其中的工具名称。
 *
 * 处理以下形式: `:cargo`、`: 'cargo'`、` 'cargo'`、` "cargo"`、`: cargo` 等。
 */
fn extract_tool_name_after_keyword(after: &str) -> String {
    let s = after.trim_start();
    // 去除开头的冒号及可选空白: "tool: cargo" 或 "tool:cargo"
    let s = s.strip_prefix(':').unwrap_or(s).trim_start();
    // 去除两侧引号: "tool 'cargo'" 或 "tool \"cargo\""
    if let Some(rest) = s.strip_prefix('\'')
        && let Some(end) = rest.find('\'')
    {
        return rest.get(..end).unwrap_or_default().to_string();
    }
    if let Some(rest) = s.strip_prefix('"')
        && let Some(end) = rest.find('"')
    {
        return rest.get(..end).unwrap_or_default().to_string();
    }
    // 无引号形式:取第一个单词(字母数字加连字符/下划线)。
    s.split(|c: char| !c.is_alphanumeric() && c != '-' && c != '_')
        .next()
        .unwrap_or("")
        .to_string()
}

/**
 * 从已有候选条目中解析出 SkillCandidate。
 *
 * 内容以候选对象的 JSON 序列化字符串形式存储,因此直接用 `from_str`
 * 反序列化。
 */
fn parse_candidate(entry: &TypedMemoryEntry) -> Result<SkillCandidate> {
    serde_json::from_str::<SkillCandidate>(&entry.content).map_err(|error| {
        ReactError::Other(format!(
            "candidate payload {} is invalid: {error}",
            entry.key
        ))
    })
}

/// 为技能候选创建默认的 MemoryMeta。
fn default_candidate_meta() -> echo_core::memory::types::MemoryMeta {
    echo_core::memory::types::MemoryMeta::new(
        MemoryType::SkillCandidate,
        MemorySource::AutoExtracted,
        "skill_candidate",
    )
    .with_confidence(0.75)
    .with_stability(0.60)
}

// ── 测试 ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evolution::audit::{ChangeFilter, JsonlChangeLog};
    use crate::evolution::curator::CuratorConfig;
    use echo_core::memory::store::{Store, StoreItem};
    use echo_core::memory::types::MemorySource;
    use echo_state::memory::store::InMemoryStore;
    use futures::future::BoxFuture;
    use serde_json::Value;
    use std::sync::Arc;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, Ordering};

    /// 用于测试的观察者:记录所有被检测到的候选技能名称。
    struct CandidateObserver {
        names: Arc<Mutex<Vec<String>>>,
    }

    impl EvolutionObserver for CandidateObserver {
        fn on_skill_candidate_detected<'a>(&'a self, skill_name: &'a str) -> BoxFuture<'a, ()> {
            Box::pin(async move {
                self.names
                    .lock()
                    .unwrap_or_else(|error| error.into_inner())
                    .push(skill_name.to_string());
            })
        }
    }

    /// 用于候选单元测试的轻量级内存 ChangeLog。
    #[derive(Default)]
    struct NullChangeLog {
        entries: Mutex<Vec<super::super::audit::ChangeEntry>>,
    }
    impl ChangeLog for NullChangeLog {
        fn record(&self, entry: super::super::audit::ChangeEntry) -> Result<()> {
            self.record_idempotent(entry).map(|_| ())
        }
        fn record_idempotent(
            &self,
            entry: super::super::audit::ChangeEntry,
        ) -> Result<super::super::audit::ChangeRecordOutcome> {
            let mut entries = self
                .entries
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            if let Some(existing) = entries
                .iter()
                .find(|existing| existing.change_id == entry.change_id)
            {
                return if existing == &entry {
                    Ok(super::super::audit::ChangeRecordOutcome::AlreadyRecorded)
                } else {
                    Err(ReactError::Other(
                        "test ChangeLog identity collision".into(),
                    ))
                };
            }
            entries.push(entry);
            Ok(super::super::audit::ChangeRecordOutcome::Appended)
        }
        fn query(
            &self,
            filter: &super::super::audit::ChangeFilter,
        ) -> Result<Vec<super::super::audit::ChangeEntry>> {
            Ok(self
                .entries
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .iter()
                .rev()
                .filter(|entry| filter.matches(entry))
                .cloned()
                .collect())
        }
        fn latest_for(
            &self,
            entity_type: EntityType,
            entity_key: &str,
        ) -> Result<Option<super::super::audit::ChangeEntry>> {
            Ok(self
                .entries
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .iter()
                .rev()
                .find(|entry| entry.entity_type == entity_type && entry.entity_key == entity_key)
                .cloned())
        }
        fn len(&self) -> usize {
            self.entries
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .len()
        }
    }

    /// 会在指定变更类型的第一次记录时注入失败的 ChangeLog,用于测试崩溃恢复。
    struct FailChangeTypeLog {
        inner: JsonlChangeLog,
        change_type: ChangeType,
        failed: AtomicBool,
    }

    impl FailChangeTypeLog {
        fn new(inner: JsonlChangeLog, change_type: ChangeType) -> Self {
            Self {
                inner,
                change_type,
                failed: AtomicBool::new(false),
            }
        }
    }

    impl ChangeLog for FailChangeTypeLog {
        fn record(&self, entry: super::super::audit::ChangeEntry) -> Result<()> {
            self.record_idempotent(entry).map(|_| ())
        }

        fn record_idempotent(
            &self,
            entry: super::super::audit::ChangeEntry,
        ) -> Result<super::super::audit::ChangeRecordOutcome> {
            if entry.change_type == self.change_type
                && entry.entity_key != CANDIDATE_AUTHORITY_AUDIT_KEY
                && !self.failed.swap(true, Ordering::AcqRel)
            {
                return Err(ReactError::Other(format!(
                    "injected {:?} candidate audit failure",
                    self.change_type
                )));
            }
            self.inner.record_idempotent(entry)
        }

        fn query(
            &self,
            filter: &super::super::audit::ChangeFilter,
        ) -> Result<Vec<super::super::audit::ChangeEntry>> {
            self.inner.query(filter)
        }

        fn latest_for(
            &self,
            entity_type: EntityType,
            entity_key: &str,
        ) -> Result<Option<super::super::audit::ChangeEntry>> {
            self.inner.latest_for(entity_type, entity_key)
        }

        fn len(&self) -> usize {
            self.inner.len()
        }
    }

    /// 允许在 `compare_and_put` 之前注入一次外部写入的 Store,用于模拟并发竞争。
    struct InterleavingStore {
        inner: Arc<InMemoryStore>,
        injected: Mutex<Option<Value>>,
    }

    impl InterleavingStore {
        fn new(inner: Arc<InMemoryStore>) -> Self {
            Self {
                inner,
                injected: Mutex::new(None),
            }
        }

        /// 在下一次 `compare_and_put` 调用前,先向底层 Store 写入指定的外部值。
        fn inject_before_next_compare(&self, value: Value) {
            *self
                .injected
                .lock()
                .unwrap_or_else(|error| error.into_inner()) = Some(value);
        }
    }

    impl Store for InterleavingStore {
        fn put<'a>(
            &'a self,
            namespace: &'a [&'a str],
            key: &'a str,
            value: Value,
        ) -> BoxFuture<'a, Result<()>> {
            self.inner.put(namespace, key, value)
        }

        fn compare_and_put<'a>(
            &'a self,
            namespace: &'a [&'a str],
            key: &'a str,
            expected: Option<Value>,
            value: Value,
        ) -> BoxFuture<'a, Result<StoreCompareAndPutOutcome>> {
            Box::pin(async move {
                let injected = self
                    .injected
                    .lock()
                    .unwrap_or_else(|error| error.into_inner())
                    .take();
                if let Some(external) = injected {
                    self.inner.put(namespace, key, external).await?;
                }
                self.inner
                    .compare_and_put(namespace, key, expected, value)
                    .await
            })
        }

        fn get<'a>(
            &'a self,
            namespace: &'a [&'a str],
            key: &'a str,
        ) -> BoxFuture<'a, Result<Option<StoreItem>>> {
            self.inner.get(namespace, key)
        }

        fn search<'a>(
            &'a self,
            namespace: &'a [&'a str],
            query: &'a str,
            limit: usize,
        ) -> BoxFuture<'a, Result<Vec<StoreItem>>> {
            self.inner.search(namespace, query, limit)
        }

        fn delete<'a>(
            &'a self,
            namespace: &'a [&'a str],
            key: &'a str,
        ) -> BoxFuture<'a, Result<bool>> {
            self.inner.delete(namespace, key)
        }

        fn list_namespaces<'a>(
            &'a self,
            prefix: Option<&'a [&'a str]>,
        ) -> BoxFuture<'a, Result<Vec<Vec<String>>>> {
            self.inner.list_namespaces(prefix)
        }

        fn list<'a>(&'a self, namespace: &'a [&'a str]) -> BoxFuture<'a, Result<Vec<StoreItem>>> {
            self.inner.list(namespace)
        }
    }

    /// 不支持原子 `compare_and_put` 的 Store,用于测试该能力缺失时的拒绝行为。
    struct UnsupportedCasStore {
        inner: Arc<InMemoryStore>,
    }

    impl Store for UnsupportedCasStore {
        fn put<'a>(
            &'a self,
            namespace: &'a [&'a str],
            key: &'a str,
            value: Value,
        ) -> BoxFuture<'a, Result<()>> {
            self.inner.put(namespace, key, value)
        }

        fn get<'a>(
            &'a self,
            namespace: &'a [&'a str],
            key: &'a str,
        ) -> BoxFuture<'a, Result<Option<StoreItem>>> {
            self.inner.get(namespace, key)
        }

        fn search<'a>(
            &'a self,
            namespace: &'a [&'a str],
            query: &'a str,
            limit: usize,
        ) -> BoxFuture<'a, Result<Vec<StoreItem>>> {
            self.inner.search(namespace, query, limit)
        }

        fn delete<'a>(
            &'a self,
            namespace: &'a [&'a str],
            key: &'a str,
        ) -> BoxFuture<'a, Result<bool>> {
            self.inner.delete(namespace, key)
        }

        fn list_namespaces<'a>(
            &'a self,
            prefix: Option<&'a [&'a str]>,
        ) -> BoxFuture<'a, Result<Vec<Vec<String>>>> {
            self.inner.list_namespaces(prefix)
        }

        fn list<'a>(&'a self, namespace: &'a [&'a str]) -> BoxFuture<'a, Result<Vec<StoreItem>>> {
            self.inner.list(namespace)
        }
    }

    /// 构造一个用于测试的类型化记忆条目。
    fn make_entry(
        key: &str,
        content: &str,
        meta: echo_core::memory::types::MemoryMeta,
    ) -> TypedMemoryEntry {
        let raw = StoreItem::new(
            vec!["agent".to_string(), "typed_memories".to_string()],
            key.to_string(),
            serde_json::Value::Null,
        );
        TypedMemoryEntry {
            key: key.to_string(),
            content: content.to_string(),
            meta,
            raw,
        }
    }

    /// 构造一条 WorkflowPattern 类型的测试元数据。
    fn wf_meta(topic: &str) -> echo_core::memory::types::MemoryMeta {
        echo_core::memory::types::MemoryMeta::new(
            MemoryType::WorkflowPattern,
            MemorySource::RepeatedWorkflow,
            topic,
        )
        .with_confidence(0.75)
    }

    /// 构造一条 DebuggingLesson 类型的测试元数据。
    fn dl_meta(topic: &str) -> echo_core::memory::types::MemoryMeta {
        echo_core::memory::types::MemoryMeta::new(
            MemoryType::DebuggingLesson,
            MemorySource::ErrorResolution,
            topic,
        )
        .with_confidence(0.80)
    }

    /// 构造一个使用临时状态文件的测试用检测器。
    fn test_detector() -> SkillCandidateDetector {
        let path = std::env::temp_dir().join(format!(
            "echo-agent-candidate-{}.json",
            uuid::Uuid::new_v4()
        ));
        SkillCandidateDetector::new(Curator::new(CuratorConfig::default(), path))
    }

    /// 向 typed store 批量写入 `[start, end)` 范围内编号的构建模式观测。
    async fn put_workflow_observations(
        typed: &TypedMemoryStore,
        start: usize,
        end: usize,
    ) -> Result<()> {
        for index in start..end {
            typed
                .put_typed(
                    crate::evolution::layer::WARM_NAMESPACE,
                    &format!("wf_{index}"),
                    &format!("Build pattern observation {index}"),
                    wf_meta("build"),
                )
                .await?;
        }
        Ok(())
    }

    #[tokio::test]
    async fn test_candidate_from_repeated_workflow() {
        let store = Arc::new(InMemoryStore::new());
        let typed = TypedMemoryStore::new(store);
        let log = NullChangeLog::default();

        // 插入 3 条主题相同的 WorkflowPattern 条目。
        for i in 0..3 {
            typed
                .put_typed(
                    crate::evolution::layer::WARM_NAMESPACE,
                    &format!("wf_{}", i),
                    &format!(
                        "Repeated workflow pattern: tool 'cargo' used {} times across sessions",
                        i + 1
                    ),
                    wf_meta("cargo-build"),
                )
                .await
                .unwrap();
        }

        let observed_names = Arc::new(Mutex::new(Vec::new()));
        let detector = test_detector().with_evolution_observer(Arc::new(CandidateObserver {
            names: observed_names.clone(),
        }));
        let report = detector.detect(&typed, &log).await.unwrap();

        assert_eq!(report.new_candidates.len(), 1);
        assert_eq!(report.new_candidates[0].topic, "cargo-build");
        assert!(report.new_candidates[0].sample_count >= 3);
        assert_eq!(
            report.new_candidates[0].source_type,
            MemoryType::WorkflowPattern
        );
        assert_eq!(
            *observed_names
                .lock()
                .unwrap_or_else(|error| error.into_inner()),
            vec!["cargo-build".to_string()]
        );
    }

    #[tokio::test]
    async fn test_no_candidate_below_threshold() {
        let store = Arc::new(InMemoryStore::new());
        let typed = TypedMemoryStore::new(store);
        let log = NullChangeLog::default();

        // 仅插入 2 条条目 —— 低于阈值。
        for i in 0..2 {
            typed
                .put_typed(
                    crate::evolution::layer::WARM_NAMESPACE,
                    &format!("wf_{}", i),
                    "Some workflow",
                    wf_meta("build"),
                )
                .await
                .unwrap();
        }

        let detector = test_detector();
        let report = detector.detect(&typed, &log).await.unwrap();

        assert!(report.new_candidates.is_empty());
    }

    #[tokio::test]
    async fn test_no_duplicate_candidates() {
        let store = Arc::new(InMemoryStore::new());
        let typed = TypedMemoryStore::new(store);
        let log = NullChangeLog::default();

        // 插入 3 条条目。
        for i in 0..3 {
            typed
                .put_typed(
                    crate::evolution::layer::WARM_NAMESPACE,
                    &format!("wf_{}", i),
                    "Build pattern",
                    wf_meta("build"),
                )
                .await
                .unwrap();
        }

        let detector = test_detector();
        // 第一次检测:应当创建候选。
        let report1 = detector.detect(&typed, &log).await.unwrap();
        assert_eq!(report1.new_candidates.len(), 1);

        // 第二次检测:不应产生重复候选。
        let report2 = detector.detect(&typed, &log).await.unwrap();
        assert!(report2.new_candidates.is_empty());
    }

    #[tokio::test]
    async fn reinforcement_records_one_audit_and_no_growth_records_nothing() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let curator_path = directory.path().join("curator.json");
        let typed = TypedMemoryStore::new(Arc::new(InMemoryStore::new()));
        let log_path = directory.path().join("changes.jsonl");
        let log = JsonlChangeLog::new(log_path.clone())?;
        let detector = SkillCandidateDetector::new(Curator::new(
            CuratorConfig::default(),
            curator_path.clone(),
        ));

        for index in 0..3 {
            typed
                .put_typed(
                    crate::evolution::layer::WARM_NAMESPACE,
                    &format!("wf_{index}"),
                    "Build pattern",
                    wf_meta("build"),
                )
                .await?;
        }
        let created = detector.detect(&typed, &log).await?;
        assert_eq!(created.new_candidates.len(), 1);
        assert_eq!(log.len(), 2);

        typed
            .put_typed(
                crate::evolution::layer::WARM_NAMESPACE,
                "wf_3",
                "Build pattern with one more observation",
                wf_meta("build"),
            )
            .await?;
        let reinforced = detector.detect(&typed, &log).await?;
        assert_eq!(reinforced.reinforced, vec!["build".to_string()]);

        let changes = log.query(
            &ChangeFilter::new()
                .with_entity_type(EntityType::Skill)
                .with_key_prefix("build"),
        )?;
        assert_eq!(changes.len(), 2);
        assert_eq!(changes[0].change_type, ChangeType::Update);
        assert_eq!(
            changes[0]
                .before
                .as_ref()
                .and_then(|value| value.get("sample_count"))
                .and_then(serde_json::Value::as_u64),
            Some(3)
        );
        assert_eq!(
            changes[0]
                .after
                .as_ref()
                .and_then(|value| value.get("sample_count"))
                .and_then(serde_json::Value::as_u64),
            Some(4)
        );

        let candidate_before = typed
            .inner()
            .get(CANDIDATE_NAMESPACE, "build")
            .await?
            .ok_or_else(|| ReactError::Other("candidate projection is missing".into()))?;
        let mut journal_path = curator_path.as_os_str().to_os_string();
        journal_path.push(".candidate-operations.jsonl");
        let journal_path = PathBuf::from(journal_path);
        let journal_len = std::fs::metadata(&journal_path)?.len();
        let audit_len = std::fs::metadata(&log_path)?.len();

        let no_growth = detector.detect(&typed, &log).await?;
        assert!(no_growth.reinforced.is_empty());
        assert_eq!(log.len(), 3);
        let candidate_after = typed
            .inner()
            .get(CANDIDATE_NAMESPACE, "build")
            .await?
            .ok_or_else(|| ReactError::Other("candidate projection is missing".into()))?;
        assert_eq!(candidate_after.value, candidate_before.value);
        assert_eq!(candidate_after.updated_at, candidate_before.updated_at);
        assert_eq!(std::fs::metadata(journal_path)?.len(), journal_len);
        assert_eq!(std::fs::metadata(log_path)?.len(), audit_len);
        Ok(())
    }

    #[tokio::test]
    async fn create_audit_failure_reconciles_once_without_publishing_observer() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let curator_path = directory.path().join("curator.json");
        let log_path = directory.path().join("changes.jsonl");
        let typed = TypedMemoryStore::new(Arc::new(InMemoryStore::new()));
        put_workflow_observations(&typed, 0, 3).await?;

        let observed_names = Arc::new(Mutex::new(Vec::new()));
        let detector = SkillCandidateDetector::new(Curator::new(
            CuratorConfig::default(),
            curator_path.clone(),
        ))
        .with_evolution_observer(Arc::new(CandidateObserver {
            names: Arc::clone(&observed_names),
        }));
        let failing =
            FailChangeTypeLog::new(JsonlChangeLog::new(log_path.clone())?, ChangeType::Create);
        assert!(detector.detect(&typed, &failing).await.is_err());
        assert!(
            observed_names
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .is_empty()
        );
        assert_eq!(failing.len(), 1);
        drop(failing);
        drop(detector);

        let recovered_log = JsonlChangeLog::new(log_path)?;
        let recovered_curator = Curator::new(CuratorConfig::default(), curator_path);
        let recovered = SkillCandidateDetector::new(recovered_curator.clone())
            .detect(&typed, &recovered_log)
            .await?;
        assert!(recovered.new_candidates.is_empty());
        assert!(recovered.reinforced.is_empty());
        assert_eq!(recovered_log.len(), 2);
        assert!(recovered_curator.skill("build")?.is_some());

        let repeated = SkillCandidateDetector::new(recovered_curator)
            .detect(&typed, &recovered_log)
            .await?;
        assert!(repeated.new_candidates.is_empty());
        assert!(repeated.reinforced.is_empty());
        assert_eq!(recovered_log.len(), 2);
        Ok(())
    }

    #[tokio::test]
    async fn reinforcement_audit_failure_reconciles_once_after_restart() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let curator_path = directory.path().join("curator.json");
        let log_path = directory.path().join("changes.jsonl");
        let typed = TypedMemoryStore::new(Arc::new(InMemoryStore::new()));
        put_workflow_observations(&typed, 0, 3).await?;

        let created_log = JsonlChangeLog::new(log_path.clone())?;
        SkillCandidateDetector::new(Curator::new(CuratorConfig::default(), curator_path.clone()))
            .detect(&typed, &created_log)
            .await?;
        assert_eq!(created_log.len(), 2);
        drop(created_log);

        put_workflow_observations(&typed, 3, 4).await?;
        let failing =
            FailChangeTypeLog::new(JsonlChangeLog::new(log_path.clone())?, ChangeType::Update);
        let detector = SkillCandidateDetector::new(Curator::new(
            CuratorConfig::default(),
            curator_path.clone(),
        ));
        assert!(detector.detect(&typed, &failing).await.is_err());
        assert_eq!(failing.len(), 2);
        let projected = typed
            .get_typed(CANDIDATE_NAMESPACE, "build")
            .await?
            .ok_or_else(|| ReactError::Other("candidate projection is missing".into()))?;
        assert_eq!(parse_candidate(&projected)?.sample_count, 4);
        drop(failing);
        drop(detector);

        let recovered_log = JsonlChangeLog::new(log_path)?;
        let recovered =
            SkillCandidateDetector::new(Curator::new(CuratorConfig::default(), curator_path));
        let report = recovered.detect(&typed, &recovered_log).await?;
        assert!(report.reinforced.is_empty());
        assert_eq!(recovered_log.len(), 3);
        recovered.detect(&typed, &recovered_log).await?;
        assert_eq!(recovered_log.len(), 3);
        Ok(())
    }

    #[tokio::test]
    async fn pending_reinforcement_refuses_unknown_external_candidate_payload() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let curator_path = directory.path().join("curator.json");
        let log_path = directory.path().join("changes.jsonl");
        let typed = TypedMemoryStore::new(Arc::new(InMemoryStore::new()));
        put_workflow_observations(&typed, 0, 3).await?;

        let created_log = JsonlChangeLog::new(log_path.clone())?;
        SkillCandidateDetector::new(Curator::new(CuratorConfig::default(), curator_path.clone()))
            .detect(&typed, &created_log)
            .await?;
        drop(created_log);

        put_workflow_observations(&typed, 3, 4).await?;
        let failing =
            FailChangeTypeLog::new(JsonlChangeLog::new(log_path.clone())?, ChangeType::Update);
        let detector = SkillCandidateDetector::new(Curator::new(
            CuratorConfig::default(),
            curator_path.clone(),
        ));
        assert!(detector.detect(&typed, &failing).await.is_err());
        drop(failing);
        drop(detector);

        let current = typed
            .get_typed(CANDIDATE_NAMESPACE, "build")
            .await?
            .ok_or_else(|| ReactError::Other("candidate projection is missing".into()))?;
        let mut external = parse_candidate(&current)?;
        external.sample_count = 99;
        let external_content = serde_json::to_string(&external)?;
        typed
            .put_typed(
                CANDIDATE_NAMESPACE,
                "build",
                &external_content,
                current.meta,
            )
            .await?;

        let recovered_log = JsonlChangeLog::new(log_path)?;
        let recovered =
            SkillCandidateDetector::new(Curator::new(CuratorConfig::default(), curator_path));
        assert!(recovered.detect(&typed, &recovered_log).await.is_err());
        assert_eq!(recovered_log.len(), 2);
        let preserved = typed
            .get_typed(CANDIDATE_NAMESPACE, "build")
            .await?
            .ok_or_else(|| ReactError::Other("external candidate is missing".into()))?;
        assert_eq!(parse_candidate(&preserved)?.sample_count, 99);
        Ok(())
    }

    #[tokio::test]
    async fn journal_binding_rejects_another_store_or_change_log() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let curator_path = directory.path().join("curator.json");
        let log_a = JsonlChangeLog::new(directory.path().join("changes-a.jsonl"))?;
        let log_b = JsonlChangeLog::new(directory.path().join("changes-b.jsonl"))?;
        let typed_a = TypedMemoryStore::new(Arc::new(InMemoryStore::new()));
        let typed_b = TypedMemoryStore::new(Arc::new(InMemoryStore::new()));
        put_workflow_observations(&typed_a, 0, 3).await?;
        put_workflow_observations(&typed_b, 0, 3).await?;
        let detector =
            SkillCandidateDetector::new(Curator::new(CuratorConfig::default(), curator_path));

        detector.detect(&typed_a, &log_a).await?;
        assert!(detector.detect(&typed_b, &log_b).await.is_err());
        assert!(detector.detect(&typed_a, &log_b).await.is_err());
        assert!(
            typed_b
                .get_typed(CANDIDATE_NAMESPACE, "build")
                .await?
                .is_none()
        );
        assert_eq!(log_b.len(), 0);
        Ok(())
    }

    #[tokio::test]
    async fn same_stem_curators_run_sequentially_without_private_path_aliasing() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let json_path = directory.path().join("state.json");
        let toml_path = directory.path().join("state.toml");
        let json_curator = Curator::new(CuratorConfig::default(), json_path.clone());
        let toml_curator = Curator::new(CuratorConfig::default(), toml_path.clone());
        let json_typed = TypedMemoryStore::new(Arc::new(InMemoryStore::new()));
        let toml_typed = TypedMemoryStore::new(Arc::new(InMemoryStore::new()));
        let json_log = JsonlChangeLog::new(directory.path().join("json-changes.jsonl"))?;
        let toml_log = JsonlChangeLog::new(directory.path().join("toml-changes.jsonl"))?;
        put_workflow_observations(&json_typed, 0, 3).await?;
        put_workflow_observations(&toml_typed, 0, 3).await?;

        SkillCandidateDetector::new(json_curator.clone())
            .detect(&json_typed, &json_log)
            .await?;
        SkillCandidateDetector::new(toml_curator.clone())
            .detect(&toml_typed, &toml_log)
            .await?;

        assert_ne!(
            json_curator.candidate_authority("build")?,
            toml_curator.candidate_authority("build")?
        );
        assert_ne!(
            json_curator.candidate_operation_journal_path(),
            toml_curator.candidate_operation_journal_path()
        );
        assert!(json_curator.candidate_operation_journal_path().exists());
        assert!(toml_curator.candidate_operation_journal_path().exists());
        let mut json_lock = json_path.as_os_str().to_os_string();
        json_lock.push(".lock");
        let mut toml_lock = toml_path.as_os_str().to_os_string();
        toml_lock.push(".lock");
        assert_ne!(json_lock, toml_lock);
        assert!(PathBuf::from(json_lock).exists());
        assert!(PathBuf::from(toml_lock).exists());
        Ok(())
    }

    #[tokio::test]
    async fn same_stem_curators_run_concurrently_without_private_path_aliasing() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let json_path = directory.path().join("parallel-state.json");
        let toml_path = directory.path().join("parallel-state.toml");
        let json_curator = Curator::new(CuratorConfig::default(), json_path);
        let toml_curator = Curator::new(CuratorConfig::default(), toml_path);
        let json_typed = TypedMemoryStore::new(Arc::new(InMemoryStore::new()));
        let toml_typed = TypedMemoryStore::new(Arc::new(InMemoryStore::new()));
        let json_log = JsonlChangeLog::new(directory.path().join("parallel-json.jsonl"))?;
        let toml_log = JsonlChangeLog::new(directory.path().join("parallel-toml.jsonl"))?;
        put_workflow_observations(&json_typed, 0, 3).await?;
        put_workflow_observations(&toml_typed, 0, 3).await?;
        let json_detector = SkillCandidateDetector::new(json_curator.clone());
        let toml_detector = SkillCandidateDetector::new(toml_curator.clone());

        let (json_result, toml_result) = tokio::join!(
            json_detector.detect(&json_typed, &json_log),
            toml_detector.detect(&toml_typed, &toml_log),
        );
        json_result?;
        toml_result?;
        assert!(json_curator.skill("build")?.is_some());
        assert!(toml_curator.skill("build")?.is_some());
        assert_ne!(
            json_curator.candidate_authority("build")?,
            toml_curator.candidate_authority("build")?
        );
        assert_ne!(
            json_curator.candidate_operation_journal_path(),
            toml_curator.candidate_operation_journal_path()
        );
        Ok(())
    }

    #[tokio::test]
    async fn store_without_atomic_compare_and_put_is_rejected_before_candidate_mutation()
    -> Result<()> {
        let directory = tempfile::tempdir()?;
        let inner = Arc::new(InMemoryStore::new());
        let typed = TypedMemoryStore::new(Arc::new(UnsupportedCasStore { inner }));
        let log = JsonlChangeLog::new(directory.path().join("changes.jsonl"))?;
        put_workflow_observations(&typed, 0, 3).await?;
        let detector = SkillCandidateDetector::new(Curator::new(
            CuratorConfig::default(),
            directory.path().join("curator.json"),
        ));

        assert!(detector.detect(&typed, &log).await.is_err());
        assert!(
            typed
                .get_typed(CANDIDATE_NAMESPACE, "build")
                .await?
                .is_none()
        );
        assert_eq!(log.len(), 0);
        Ok(())
    }

    #[tokio::test]
    async fn external_write_between_read_and_compare_put_fails_closed() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let inner = Arc::new(InMemoryStore::new());
        let interleaving = Arc::new(InterleavingStore::new(inner));
        let typed = TypedMemoryStore::new(interleaving.clone());
        let log = JsonlChangeLog::new(directory.path().join("changes.jsonl"))?;
        let detector = SkillCandidateDetector::new(Curator::new(
            CuratorConfig::default(),
            directory.path().join("curator.json"),
        ));
        put_workflow_observations(&typed, 0, 3).await?;
        detector.detect(&typed, &log).await?;

        let current = typed
            .get_typed(CANDIDATE_NAMESPACE, "build")
            .await?
            .ok_or_else(|| ReactError::Other("candidate projection is missing".into()))?;
        let mut external = parse_candidate(&current)?;
        external.sample_count = 99;
        let external_content = serde_json::to_string(&external)?;
        let external_value = TypedMemoryValue::new(&external_content, current.meta).to_value()?;
        interleaving.inject_before_next_compare(external_value);
        put_workflow_observations(&typed, 3, 4).await?;

        assert!(detector.detect(&typed, &log).await.is_err());
        let preserved = typed
            .get_typed(CANDIDATE_NAMESPACE, "build")
            .await?
            .ok_or_else(|| ReactError::Other("external candidate is missing".into()))?;
        assert_eq!(parse_candidate(&preserved)?.sample_count, 99);
        assert_eq!(
            log.query(
                &ChangeFilter::new()
                    .with_entity_type(EntityType::Skill)
                    .with_key_prefix("build"),
            )?
            .len(),
            1
        );
        Ok(())
    }

    #[tokio::test]
    async fn reconciliation_restores_missing_curator_lineage() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let curator_path = directory.path().join("curator.json");
        let curator = Curator::new(CuratorConfig::default(), curator_path);
        let typed = TypedMemoryStore::new(Arc::new(InMemoryStore::new()));
        let log = JsonlChangeLog::new(directory.path().join("changes.jsonl"))?;
        put_workflow_observations(&typed, 0, 3).await?;
        SkillCandidateDetector::new(curator.clone())
            .detect(&typed, &log)
            .await?;
        let authority = curator
            .candidate_authority("build")?
            .ok_or_else(|| ReactError::Other("candidate lineage is missing".into()))?;
        let mut state = curator.load_state()?;
        state.skills.remove("build");
        curator.save_state(&state)?;

        SkillCandidateDetector::new(curator.clone())
            .detect(&typed, &log)
            .await?;
        let restored = curator
            .skill("build")?
            .ok_or_else(|| ReactError::Other("candidate lifecycle was not restored".into()))?;
        assert_eq!(
            restored.lifecycle,
            super::super::curator::SkillLifecycle::Candidate
        );
        assert_eq!(
            curator.candidate_authority("build")?.as_deref(),
            Some(authority.as_str())
        );
        Ok(())
    }

    #[tokio::test]
    async fn matching_lineage_accepts_legitimate_draft_and_active_transitions() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let curator = Curator::new(
            CuratorConfig::default(),
            directory.path().join("curator.json"),
        );
        let typed = TypedMemoryStore::new(Arc::new(InMemoryStore::new()));
        let log = JsonlChangeLog::new(directory.path().join("changes.jsonl"))?;
        put_workflow_observations(&typed, 0, 3).await?;
        let detector = SkillCandidateDetector::new(curator.clone());
        detector.detect(&typed, &log).await?;
        assert!(curator.promote_to_draft("build")?);

        detector.detect(&typed, &log).await?;
        let meta = curator
            .skill("build")?
            .ok_or_else(|| ReactError::Other("draft lifecycle is missing".into()))?;
        assert_eq!(meta.lifecycle, super::super::curator::SkillLifecycle::Draft);
        assert!(curator.candidate_authority("build")?.is_some());
        assert!(curator.promote_to_active("build")?);
        detector.detect(&typed, &log).await?;
        let active = curator
            .skill("build")?
            .ok_or_else(|| ReactError::Other("active lifecycle is missing".into()))?;
        assert_eq!(
            active.lifecycle,
            super::super::curator::SkillLifecycle::Active
        );
        Ok(())
    }

    #[tokio::test]
    async fn unbound_same_name_curator_skill_conflicts_with_candidate_registration() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let curator = Curator::new(
            CuratorConfig::default(),
            directory.path().join("curator.json"),
        );
        curator.register_candidate("build")?;
        let typed = TypedMemoryStore::new(Arc::new(InMemoryStore::new()));
        let log = JsonlChangeLog::new(directory.path().join("changes.jsonl"))?;
        put_workflow_observations(&typed, 0, 3).await?;

        assert!(
            SkillCandidateDetector::new(curator.clone())
                .detect(&typed, &log)
                .await
                .is_err()
        );
        assert!(curator.skill("build")?.is_some());
        assert!(curator.candidate_authority("build")?.is_none());
        assert_eq!(log.len(), 1);
        Ok(())
    }

    #[test]
    fn test_name_sanitization() {
        assert_eq!(sanitize_name("Cargo Build"), "cargo-build");
        assert_eq!(sanitize_name("test/deploy:ci"), "test-deploy-ci");
        assert_eq!(sanitize_name("  leading  spaces  "), "leading-spaces");
        assert_eq!(sanitize_name("---dashes---"), "dashes");
        assert_eq!(sanitize_name("a/b/c/d/e"), "a-b-c-d-e");
    }

    #[tokio::test]
    async fn test_candidate_with_debugging_lesson() {
        let store = Arc::new(InMemoryStore::new());
        let typed = TypedMemoryStore::new(store);
        let log = NullChangeLog::default();

        // 插入 3 条 DebuggingLesson 条目。
        for i in 0..3 {
            typed
                .put_typed(
                    crate::evolution::layer::WARM_NAMESPACE,
                    &format!("dl_{}", i),
                    &format!(
                        "Lesson: always run cargo check before cargo build (attempt {})",
                        i + 1
                    ),
                    dl_meta("cargo-check"),
                )
                .await
                .unwrap();
        }

        let detector = test_detector();
        let report = detector.detect(&typed, &log).await.unwrap();

        assert_eq!(report.new_candidates.len(), 1);
        assert_eq!(
            report.new_candidates[0].source_type,
            MemoryType::DebuggingLesson
        );
        assert_eq!(report.new_candidates[0].topic, "cargo-check");
    }

    #[test]
    fn test_extract_tool_names() {
        let entries = vec![
            make_entry("a", "Used tool:cargo to build project", wf_meta("t")),
            make_entry("b", "Used tool:rustfmt for formatting", wf_meta("t")),
        ];
        let tools = extract_tool_names(&entries);
        assert!(tools.contains(&"cargo".to_string()));
    }
}
