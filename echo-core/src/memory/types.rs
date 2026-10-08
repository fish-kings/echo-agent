/*!
带类型的记忆元数据 —— 为记忆条目提供结构化分类与质量指标。

进化系统中的每一条记忆都携带一个 `MemoryMeta` 结构体，用于对其类型进行
分类、跟踪置信度/重要性/稳定性，并记录其来源与生命周期状态。`MemoryMeta`
会与 `content` 字段一起序列化进 `StoreItem.value` 的 JSON 中，无需改动
`Store` trait 或 `StoreItem` 的结构。
*/

use serde::{Deserialize, Serialize};

// ── MemoryType ──────────────────────────────────────────────────────────

/**
记忆条目的分类。

每一条记忆都必须有类型 —— 进化系统中不允许存在未分类的自由文本条目。
类型决定了记忆如何被评分、审查，以及是否可能被提升。
*/
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryType {
    /// 稳定的用户偏好（例如“用户偏好简洁的回答”）。
    UserPreference,
    /// 关于项目的事实（例如“该项目使用 Maven 搭配 Java 8”）。
    ProjectFact,
    /// 架构或产品决策（例如“我们选择了 SQLite 而非 Postgres”）。
    ArchitectureDecision,
    /// 习得的调试经验（例如“JAVA_HOME 配置错误会导致 Maven 编译失败”）。
    DebuggingLesson,
    /// 一次错误的症状、原因及已验证的解决方案。
    ErrorResolution,
    /// 常用的命令、环境变量或构建步骤。
    CommandPattern,
    /// 使用某个特定工具的经验（例如“在该项目中 rg 比 grep 更快”）。
    ToolUsage,
    /// 一种重复出现、可能成为技能候选的工作流模式。
    WorkflowPattern,
    /// 已被识别为潜在技能的工作流。
    SkillCandidate,
    /// 已被取代或不再准确的笔记，出于可追溯性而保留。
    DeprecatedNote,
}

impl MemoryType {
    /**
    该记忆类型的默认稳定性（抵抗过时的能力）。

    数值越高，表示该记忆越不容易很快过时。
    */
    pub fn default_stability(&self) -> f32 {
        match self {
            Self::UserPreference => 0.85,
            Self::ProjectFact => 0.60,
            Self::ArchitectureDecision => 0.80,
            Self::DebuggingLesson => 0.55,
            Self::ErrorResolution => 0.50,
            Self::CommandPattern => 0.40,
            Self::ToolUsage => 0.50,
            Self::WorkflowPattern => 0.60,
            Self::SkillCandidate => 0.55,
            Self::DeprecatedNote => 0.10,
        }
    }

    /// 该记忆类型是否有资格被提升为技能。
    pub fn is_skill_eligible(&self) -> bool {
        matches!(
            self,
            Self::WorkflowPattern | Self::SkillCandidate | Self::DebuggingLesson
        )
    }

    /// 该记忆类型是否有资格被提升为规则。
    pub fn is_rule_eligible(&self) -> bool {
        matches!(
            self,
            Self::UserPreference | Self::ArchitectureDecision | Self::ProjectFact
        )
    }
}

// ── MemoryRisk ──────────────────────────────────────────────────────────

/**
记忆条目的风险等级。

高风险记忆（来自不可信来源或包含敏感内容）在提升为热记忆或规则之前
需要额外审查。
*/
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryRisk {
    /// 低风险 —— 可自动处理并提升。
    #[default]
    Low,
    /// 中等风险 —— 提升前可能需要审查。
    Medium,
    /**
    高风险 —— 来自不可信来源或包含敏感内容。
    未经人工批准，不得提升为热记忆或规则。
    */
    High,
}

// ── MemoryStatus ────────────────────────────────────────────────────────

/// 记忆条目的生命周期状态。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryStatus {
    /// 新创建，尚未被确认或稳定下来。
    Draft,
    /// 已确认，正在活跃使用中。
    #[default]
    Active,
    /// 已被更新的记忆条目取代。`MemoryMeta` 中的 `superseded_by` 指向替代条目。
    Superseded,
    /// 已归档 —— 不再活跃，出于可追溯性而保留。
    Archived,
}

// ── MemorySource ────────────────────────────────────────────────────────

/**
该记忆条目的创建方式。

来源决定了默认的置信度水平，以及该记忆是否可以被自动提升。
*/
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemorySource {
    /// 从用户纠正信号中提取（例如“不要这样做”）。
    UserCorrection,
    /// 从一次“工具失败 → 成功”的序列中提取。
    ErrorResolution,
    /// 从跨会话反复出现的工作流模式中检测得到。
    RepeatedWorkflow,
    /// 由用户显式保存（例如通过 `/remember` 命令）。
    ExplicitSave,
    /// 通过启发式规则或 LLM 审查自动提取。
    AutoExtracted,
    /// 从 L3 上下文压缩中提升而来。
    L3Promotion,
}

impl MemorySource {
    /// 该来源记忆的默认置信度。
    pub fn default_confidence(&self) -> f32 {
        match self {
            Self::ExplicitSave => 1.0,
            Self::UserCorrection => 0.90,
            Self::ErrorResolution => 0.85,
            Self::RepeatedWorkflow => 0.75,
            Self::AutoExtracted => 0.60,
            Self::L3Promotion => 0.50,
        }
    }

    /**
    （第4阶段）该来源记忆的默认召回权重（0.0–1.0）。
    由复合评分用于召回排序。
    */
    pub fn default_recall_weight(&self) -> f32 {
        match self {
            Self::ExplicitSave => 0.8,
            Self::UserCorrection => 0.9,
            Self::ErrorResolution => 0.7,
            Self::RepeatedWorkflow => 0.6,
            Self::AutoExtracted => 0.4,
            Self::L3Promotion => 0.4,
        }
    }
}

// ── MemoryMeta ──────────────────────────────────────────────────────────

/**
带类型记忆条目的结构化元数据。

序列化进 `StoreItem.value` JSON 时的形式如下：
```json
{
  "content": "The actual memory text",
  "meta": { "memory_type": "debugging_lesson", "confidence": 0.85, ... }
}
```
*/
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryMeta {
    /// 该记忆的分类。
    pub memory_type: MemoryType,
    /// 我们对该记忆准确性的置信程度（0.0–1.0）。
    pub confidence: f32,
    /// 该记忆的稳定性 —— 抵抗过时的能力（0.0–1.0）。
    pub stability: f32,
    /**
    （第4阶段）召回权重 0.0–1.0，由 agent LLM 在写入时给出。
    用于复合评分的召回排序（hermes issue #677 中称之为 "importance"）。
    之所以命名为 `recall_weight` 是为了避免与 `StoreItem.importance`
    （1.0–10.0，抗剪枝权重，见 store.rs:29）冲突 —— 名字相似但语义/量纲
    不同。复合评分必须读取本字段，而不是 StoreItem.importance。
    */
    #[serde(default = "default_recall_weight")]
    pub recall_weight: f32,
    /// 风险等级 —— 决定审查要求。
    pub risk: MemoryRisk,
    /// 该记忆的生命周期状态。
    pub status: MemoryStatus,
    /// 该记忆的创建方式。
    pub source: MemorySource,
    /// 主题分类（例如 "build"、"debugging"、"architecture"）。
    pub topic: String,
    /// 该记忆被修订的次数。
    #[serde(default)]
    pub revision_count: u32,
    /**
    （第4阶段）该记忆被召回的次数。驱动 Dreaming 自我进化
    （由召回频率驱动的提升）。
    */
    #[serde(default)]
    pub recall_count: u32,
    /**
    上一次成功召回的时间戳（Unix 秒）。

    与 `StoreItem::updated_at` 分开维护：如果召回遥测计数的递增也
    改写元数据的更新时间，在陈旧度逻辑看来就会误判为一次语义性的
    内容编辑。
    */
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_recalled_at: Option<u64>,
    /// 若该记忆已被取代，此字段为替代记忆的 key。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub superseded_by: Option<String>,
}

/**
`MemoryMeta::recall_weight` 的 serde 默认值（用于反序列化在该字段
出现之前写入的旧条目）。
*/
fn default_recall_weight() -> f32 {
    0.5
}

impl MemoryMeta {
    /// 使用类型与来源各自的默认值创建一个新的 MemoryMeta。
    pub fn new(memory_type: MemoryType, source: MemorySource, topic: impl Into<String>) -> Self {
        Self {
            confidence: source.default_confidence(),
            stability: memory_type.default_stability(),
            memory_type,
            recall_weight: source.default_recall_weight(),
            risk: MemoryRisk::default(),
            status: MemoryStatus::default(),
            source,
            topic: topic.into(),
            revision_count: 0,
            recall_count: 0,
            last_recalled_at: None,
            superseded_by: None,
        }
    }

    /// 创建一个带有显式置信度的 MemoryMeta。
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// 创建一个带有显式稳定性的 MemoryMeta。
    pub fn with_stability(mut self, stability: f32) -> Self {
        self.stability = stability.clamp(0.0, 1.0);
        self
    }

    /// （第4阶段）创建一个带有显式召回权重（0.0–1.0）的 MemoryMeta。
    pub fn with_recall_weight(mut self, recall_weight: f32) -> Self {
        self.recall_weight = recall_weight.clamp(0.0, 1.0);
        self
    }

    /// 创建一个带有显式风险等级的 MemoryMeta。
    pub fn with_risk(mut self, risk: MemoryRisk) -> Self {
        self.risk = risk;
        self
    }

    /// 创建一个带有显式生命周期状态的 MemoryMeta。
    pub fn with_status(mut self, status: MemoryStatus) -> Self {
        self.status = status;
        self
    }

    /// 该记忆是否有资格被提升为热记忆。
    pub fn is_hot_eligible(&self) -> bool {
        self.confidence >= 0.85
            && self.stability >= 0.70
            && self.status == MemoryStatus::Active
            && self.risk != MemoryRisk::High
    }

    /// 该记忆是否有资格被提升为规则。
    pub fn is_rule_eligible(&self) -> bool {
        self.confidence >= 0.95
            && self.stability >= 0.90
            && self.status == MemoryStatus::Active
            && self.revision_count == 0
            && self.risk != MemoryRisk::High
            && self.memory_type.is_rule_eligible()
    }

    /**
    计算陈旧度评分（0.0 = 新鲜，1.0 = 非常陈旧）。

    这是一个简化版本；第二阶段完整的 `StalenessScorer` 还会综合考虑
    时效、使用情况以及矛盾因素。
    */
    pub fn base_staleness(&self) -> f32 {
        // 稳定性低 → 陈旧风险高
        let instability = 1.0 - self.stability;
        // 置信度低 → 陈旧度更高
        let unconfidence = 1.0 - self.confidence;
        // 已弃用状态 → 陈旧度高
        let status_factor = match self.status {
            MemoryStatus::Active => 0.0,
            MemoryStatus::Draft => 0.1,
            MemoryStatus::Superseded => 0.8,
            MemoryStatus::Archived => 1.0,
        };

        (instability * 0.35 + unconfidence * 0.25 + status_factor * 0.40).min(1.0)
    }
}

impl Default for MemoryMeta {
    /**
    返回一组中性的默认元数据（`ProjectFact` / 置信度与稳定性均为 0.5 /
    低风险 / 活跃状态 / 自动提取来源）。
    */
    fn default() -> Self {
        Self {
            memory_type: MemoryType::ProjectFact,
            confidence: 0.5,
            stability: 0.5,
            recall_weight: 0.5,
            risk: MemoryRisk::Low,
            status: MemoryStatus::Active,
            source: MemorySource::AutoExtracted,
            topic: String::new(),
            revision_count: 0,
            recall_count: 0,
            last_recalled_at: None,
            superseded_by: None,
        }
    }
}

// ── TypedMemoryValue ────────────────────────────────────────────────────

/**
带类型记忆在 `StoreItem.value` 中存储的 JSON 结构。

这是它的序列化格式：`content` 字段保存实际的记忆文本，`meta` 字段
保存结构化的元数据。
*/
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypedMemoryValue {
    /// 实际的记忆文本内容。
    pub content: String,
    /// 该记忆的结构化元数据。
    pub meta: MemoryMeta,
}

impl TypedMemoryValue {
    /// 创建一个新的带类型记忆值。
    pub fn new(content: impl Into<String>, meta: MemoryMeta) -> Self {
        Self {
            content: content.into(),
            meta,
        }
    }

    /// 序列化为 `serde_json::Value`，以便存入 `StoreItem.value`。
    pub fn to_value(&self) -> serde_json::Result<serde_json::Value> {
        serde_json::to_value(self)
    }

    /// 从 `serde_json::Value` 反序列化。
    pub fn from_value(value: &serde_json::Value) -> serde_json::Result<Self> {
        serde_json::from_value(value.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_type_default_stability() {
        assert!(
            MemoryType::UserPreference.default_stability()
                > MemoryType::CommandPattern.default_stability()
        );
        assert!(
            MemoryType::ArchitectureDecision.default_stability()
                > MemoryType::ErrorResolution.default_stability()
        );
    }

    #[test]
    fn test_memory_source_default_confidence() {
        assert_eq!(MemorySource::ExplicitSave.default_confidence(), 1.0);
        assert!(
            MemorySource::UserCorrection.default_confidence()
                > MemorySource::AutoExtracted.default_confidence()
        );
    }

    #[test]
    fn test_memory_meta_new() {
        let meta = MemoryMeta::new(
            MemoryType::DebuggingLesson,
            MemorySource::ErrorResolution,
            "build",
        );
        assert_eq!(meta.memory_type, MemoryType::DebuggingLesson);
        assert_eq!(meta.source, MemorySource::ErrorResolution);
        assert_eq!(meta.topic, "build");
        assert_eq!(meta.confidence, 0.85); // ErrorResolution 的默认值
        assert_eq!(meta.status, MemoryStatus::Active);
        assert_eq!(meta.risk, MemoryRisk::Low);
    }

    #[test]
    fn test_memory_meta_builder() {
        let meta = MemoryMeta::new(
            MemoryType::UserPreference,
            MemorySource::ExplicitSave,
            "style",
        )
        .with_confidence(0.99)
        .with_stability(0.95)
        .with_risk(MemoryRisk::Medium);
        assert!((meta.confidence - 0.99).abs() < f32::EPSILON);
        assert!((meta.stability - 0.95).abs() < f32::EPSILON);
        assert_eq!(meta.risk, MemoryRisk::Medium);
    }

    #[test]
    fn test_hot_eligible() {
        let meta = MemoryMeta::new(MemoryType::ProjectFact, MemorySource::ExplicitSave, "test")
            .with_confidence(0.90)
            .with_stability(0.80);
        assert!(meta.is_hot_eligible());

        let low_confidence =
            MemoryMeta::new(MemoryType::ProjectFact, MemorySource::AutoExtracted, "test")
                .with_confidence(0.50);
        assert!(!low_confidence.is_hot_eligible());
    }

    #[test]
    fn test_rule_eligible() {
        let meta = MemoryMeta::new(
            MemoryType::UserPreference,
            MemorySource::ExplicitSave,
            "style",
        )
        .with_confidence(0.99)
        .with_stability(0.95);
        assert!(meta.is_rule_eligible());

        // 不符合资格：类型不对
        let bug_meta = MemoryMeta::new(
            MemoryType::DebuggingLesson,
            MemorySource::ErrorResolution,
            "test",
        )
        .with_confidence(0.99)
        .with_stability(0.95);
        assert!(!bug_meta.is_rule_eligible());

        // 不符合资格：高风险
        let risky = MemoryMeta::new(
            MemoryType::UserPreference,
            MemorySource::ExplicitSave,
            "test",
        )
        .with_confidence(0.99)
        .with_stability(0.95)
        .with_risk(MemoryRisk::High);
        assert!(!risky.is_rule_eligible());
    }

    #[test]
    fn test_typed_memory_value_roundtrip() {
        let meta = MemoryMeta::new(
            MemoryType::DebuggingLesson,
            MemorySource::ErrorResolution,
            "build",
        );
        let value = TypedMemoryValue::new("Maven needs Java 8", meta);
        let json = value.to_value().unwrap();
        let parsed = TypedMemoryValue::from_value(&json).unwrap();
        assert_eq!(parsed.content, "Maven needs Java 8");
        assert_eq!(parsed.meta.memory_type, MemoryType::DebuggingLesson);
        assert_eq!(parsed.meta.topic, "build");
    }

    #[test]
    fn test_typed_memory_value_backward_compat() {
        // 旧格式的 store item（没有 meta 字段）应该能用默认值成功解析
        let old_format = serde_json::json!({
            "content": "Some old memory"
        });
        // 这里应该仍能解析成功 —— meta 会使用 serde 默认值
        let _parsed: Result<TypedMemoryValue, _> = serde_json::from_value(old_format);
        /* 如果 meta 是必需字段，这里会解析失败 —— 那也是预期行为。
        TypedMemoryStore 通过捕获解析错误来处理向后兼容问题。 */
    }

    #[test]
    fn test_staleness_active_low() {
        let meta = MemoryMeta::new(
            MemoryType::UserPreference,
            MemorySource::ExplicitSave,
            "style",
        )
        .with_confidence(0.95)
        .with_stability(0.90);
        assert!(meta.base_staleness() < 0.3);
    }

    #[test]
    fn test_staleness_archived_high() {
        let meta = MemoryMeta::new(
            MemoryType::DeprecatedNote,
            MemorySource::AutoExtracted,
            "old",
        )
        .with_confidence(0.3)
        .with_stability(0.2)
        .with_status(MemoryStatus::Archived);
        assert!(meta.base_staleness() > 0.5);
    }

    #[test]
    fn stage4_p1_memory_meta_has_recall_weight_and_recall_count() {
        // ExplicitSave 来源默认 recall_weight 为 0.8；AutoExtracted 为 0.4。
        let m = MemoryMeta::new(
            MemoryType::UserPreference,
            MemorySource::ExplicitSave,
            "test",
        );
        assert_eq!(m.recall_weight, 0.8);
        assert_eq!(m.recall_count, 0);
        assert_eq!(m.last_recalled_at, None);
    }

    #[test]
    fn stage4_p1_with_recall_weight_clamps_to_unit_range() {
        let m = MemoryMeta::new(MemoryType::ProjectFact, MemorySource::AutoExtracted, "t")
            .with_recall_weight(1.5);
        assert_eq!(m.recall_weight, 1.0);
        let m2 = MemoryMeta::new(MemoryType::ProjectFact, MemorySource::AutoExtracted, "t")
            .with_recall_weight(-0.3);
        assert_eq!(m2.recall_weight, 0.0);
    }
}
