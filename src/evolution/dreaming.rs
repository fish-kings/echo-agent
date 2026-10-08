/*!
（stage4 F1）Dreaming —— 由召回频率驱动的自我演化。

用 OpenClaw-Dreaming 风格的定时批处理，取代旧的“每 N 次写入触发一次全量
审查”模型（割裂点 6/9）：扫描统一记忆存储，将高召回记忆（包括 Archived，
先复活）提升到热层（MEMORY.md → 系统提示词稳定前缀），并将陈旧的低召回
记忆批量降级为 Archived。陈旧度只是召回衰减权重，而非“死刑” ——
Archived 记忆仍可被召回（带衰减），并可在未来的 Dreaming 过程中被复活。

设计依据：`stage4-memory-evolution-compression-decision.md`（D14）、
OpenClaw Dreaming（cron + 召回统计驱动的提升）、hermes 综合得分召回
（`recall_count` 由 `MemoryRecaller` 以 fire-and-forget 方式递增）。
*/

use crate::evolution::layer::MemoryLayerManager;
use echo_core::memory::types::MemoryStatus;
use echo_core::utils::time::now_secs;
use echo_state::memory::typed_store::MemoryFilter;
use std::sync::Arc;

// ── 配置 ─────────────────────────────────────────────────────────────

/// 单次 Dreaming 过程的配置。
#[derive(Debug, Clone)]
pub struct DreamingConfig {
    /**
    记忆被考虑提升到热层所需的最小 `recall_count`。
    召回次数低于该值的记忆不会被提升（即使很久以前就存在）。
    */
    pub recall_count_threshold: u32,
    /**
    单次过程最多提升到热层的记忆数量上限（避免一次运行就塞满
    MEMORY.md / 系统提示词稳定前缀）。
    */
    pub max_promoted_per_run: usize,
    /**
    高召回记忆还必须在该时间窗口（天）内被使用过，才能被提升或复活。
    这可防止累计召回次数让一条过时事实永久具备热层资格。
    */
    pub promotion_recency_days: u32,
    /**
    存在时间超过该天数且低于 `low_recall_threshold` 的记忆会被降级为
    Archived（陈旧衰减，而非删除）。
    */
    pub stale_age_days: u32,
    /// `recall_count` 低于该值时，陈旧的 Active 记忆会被降级为 Archived。
    pub low_recall_threshold: u32,
}

/// 默认配置：召回阈值 5、单次最多提升 20 条、近期窗口 30 天、陈旧 30 天、低召回阈值 1。
impl Default for DreamingConfig {
    fn default() -> Self {
        Self {
            recall_count_threshold: 5,
            max_promoted_per_run: 20,
            promotion_recency_days: 30,
            stale_age_days: 30,
            low_recall_threshold: 1,
        }
    }
}

// ── 报告 ─────────────────────────────────────────────────────────────

/// 单次 Dreaming 过程的汇总。
#[derive(Debug, Clone, Default)]
pub struct DreamingReport {
    /// 在统一命名空间中扫描的记忆总数。
    pub scanned: usize,
    /// 被提升到热层（MEMORY.md）的记忆数。
    pub promoted: usize,
    /// 从 Archived 复活为 Active 的记忆数（G2 —— 提升的前置条件）。
    pub revived: usize,
    /// 被降级为 Archived 的陈旧低召回 Active 记忆数。
    pub demoted: usize,
    /// 本次过程应用的可解释的确定性变更。
    pub decisions: Vec<DreamingDecision>,
}

/// Dreaming 应用的一项确定性维护动作。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DreamingAction {
    /// 从 Archived 复活为 Active。
    Revived,
    /// 提升到热层（MEMORY.md）。
    PromotedToHot,
    /// 降级为 Archived。
    Archived,
}

/// Dreaming 对单条记忆做出的一项决策记录，用于解释变更原因。
#[derive(Debug, Clone)]
pub struct DreamingDecision {
    /// 记忆的键。
    pub key: String,
    /// 应用的动作。
    pub action: DreamingAction,
    /// 决策时该记忆的召回次数。
    pub recall_count: u32,
    /// 距最近一次召回（或创建）的天数。
    pub inactive_days: f64,
    /// 决策原因说明。
    pub reason: String,
}

// ── Dreaming ───────────────────────────────────────────────────────────

/**
Dreaming 自我演化过程的驱动器。

使用共享的 `MemoryLayerManager` 构造（与 agent 召回所用的是同一存储，
因此复活/降级会落在 agent 读取的统一 `["agent","memories"]` 命名空间中）。
按 cron/每日计划调用 [`Dreaming::run`]。
*/
pub struct Dreaming {
    /// Dreaming 配置。
    config: DreamingConfig,
    /// 共享的记忆分层管理器。
    layer_manager: Arc<MemoryLayerManager>,
}

impl Dreaming {
    /// 创建新的 Dreaming 驱动器。
    pub fn new(layer_manager: Arc<MemoryLayerManager>, config: DreamingConfig) -> Self {
        Self {
            config,
            layer_manager,
        }
    }

    /**
    在统一的 `["agent","memories"]` 命名空间上执行一次 Dreaming 过程。

    按记忆尽力而为：某条记忆的复活/提升/降级出错只会记录日志，不会中止
    整个过程。返回汇总报告。
    */
    pub async fn run(&self) -> crate::error::Result<DreamingReport> {
        let now = now_secs();
        let entries = self
            .layer_manager
            .list_warm_memories(&MemoryFilter::new())
            .await?;
        let mut report = DreamingReport {
            scanned: entries.len(),
            ..Default::default()
        };

        for e in &entries {
            // Superseded 记忆是墓碑 —— 永不复活/提升/降级。
            if e.meta.status == MemoryStatus::Superseded {
                continue;
            }
            let inactive_days = age_days(e.meta.last_recalled_at.unwrap_or(e.raw.created_at), now);

            /* 1. 将高召回记忆（包括 Archived）提升到热层。G2：先将
               Archived→Active 复活，因为 `is_hot_eligible()` 要求
               `status == Active`（否则无论置信度/稳定性如何，所有 Archived
               记忆都无法通过该门槛）。 */
            if e.meta.recall_count >= self.config.recall_count_threshold
                && inactive_days <= self.config.promotion_recency_days as f64
                && report.promoted < self.config.max_promoted_per_run
            {
                if e.meta.status == MemoryStatus::Archived {
                    match self.layer_manager.revive_archived(&e.key).await {
                        Ok(true) => {
                            report.revived += 1;
                            report.decisions.push(DreamingDecision {
                                key: e.key.clone(),
                                action: DreamingAction::Revived,
                                recall_count: e.meta.recall_count,
                                inactive_days,
                                reason: "archived memory was recalled frequently and recently"
                                    .to_string(),
                            });
                        }
                        Ok(false) => {}
                        Err(error) => {
                            tracing::warn!(key = %e.key, %error, "Dreaming failed to revive memory");
                        }
                    }
                }
                match self.layer_manager.consider_promotion(&e.key).await {
                    Ok(Some(_)) => {
                        report.promoted += 1;
                        report.decisions.push(DreamingDecision {
                            key: e.key.clone(),
                            action: DreamingAction::PromotedToHot,
                            recall_count: e.meta.recall_count,
                            inactive_days,
                            reason: "memory met deterministic hot-layer eligibility".to_string(),
                        });
                    }
                    Ok(None) => {}
                    Err(error) => {
                        tracing::warn!(key = %e.key, %error, "Dreaming failed to promote memory");
                    }
                }
            }

            /* 2. 将陈旧的低召回 Active 记忆批量降级为 Archived
               （取代旧审查中的陈旧降级；Archived 仍可带衰减地被召回，
               并可在未来的过程中被复活）。 */
            if e.meta.status == MemoryStatus::Active
                && e.meta.recall_count < self.config.low_recall_threshold
                && inactive_days > self.config.stale_age_days as f64
            {
                match self
                    .layer_manager
                    .demote(&e.key, "dreaming: stale + low recall")
                    .await
                {
                    Ok(_) => {
                        report.demoted += 1;
                        report.decisions.push(DreamingDecision {
                            key: e.key.clone(),
                            action: DreamingAction::Archived,
                            recall_count: e.meta.recall_count,
                            inactive_days,
                            reason: "memory exceeded the inactivity window with low recall"
                                .to_string(),
                        });
                    }
                    Err(error) => {
                        tracing::warn!(key = %e.key, %error, "Dreaming failed to archive memory");
                    }
                }
            }
        }

        tracing::info!(
            scanned = report.scanned,
            promoted = report.promoted,
            revived = report.revived,
            demoted = report.demoted,
            "Dreaming pass complete"
        );
        Ok(report)
    }
}

/// 计算从 `created_at` 到 `now`（秒级时间戳）经过的天数。
fn age_days(created_at: u64, now: u64) -> f64 {
    now.saturating_sub(created_at) as f64 / 86400.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evolution::audit::NullChangeLog;
    use echo_core::memory::store::StoreItem;
    use echo_core::memory::types::{
        MemoryMeta, MemorySource, MemoryStatus, MemoryType, TypedMemoryValue,
    };
    use echo_state::memory::store::InMemoryStore;

    /// 创建基于内存存储与临时目录的记忆分层管理器。
    fn make_manager() -> (Arc<MemoryLayerManager>, Arc<InMemoryStore>) {
        let store = Arc::new(InMemoryStore::new());
        let dir = tempfile::tempdir().expect("tempdir").keep();
        let lm = Arc::new(MemoryLayerManager::new(
            dir,
            store.clone(),
            Box::new(NullChangeLog),
        ));
        (lm, store)
    }

    /// 高召回的 Archived 记忆会被复活（G2）并提升到热层。
    #[tokio::test]
    async fn dreaming_revives_and_promotes_high_recall_archived() {
        let (lm, store) = make_manager();
        /* 高置信度 UserPreference，Archived，recall_count=10。通过 put_raw 插入，
        以绕过 `write_memory` 的自动提升到热层（否则条目会在我们于温层将其
        标记为 Archived 之前就被移到热层）。 */
        let mut meta = MemoryMeta::new(
            MemoryType::UserPreference,
            MemorySource::ExplicitSave,
            "user",
        )
        .with_confidence(0.9)
        .with_recall_weight(0.9);
        meta.status = MemoryStatus::Archived;
        meta.recall_count = 10;
        let value = TypedMemoryValue::new("user prefers Rust over Python", meta)
            .to_value()
            .expect("to_value");
        store
            .put_raw(StoreItem::new(
                vec!["agent".to_string(), "memories".to_string()],
                "k1".to_string(),
                value,
            ))
            .await;

        let dreaming = Dreaming::new(lm.clone(), DreamingConfig::default());
        let report = dreaming.run().await.expect("dreaming run");

        assert_eq!(report.scanned, 1, "one memory scanned");
        assert_eq!(
            report.revived, 1,
            "Archived high-recall memory revived to Active"
        );
        assert_eq!(
            report.promoted, 1,
            "hot-eligible revived memory promoted to hot"
        );
        assert_eq!(report.decisions.len(), 2);
        // 被提升的记忆已从温层移除（现位于 MEMORY.md）。
        let warm = lm
            .list_warm_memories(&MemoryFilter::new())
            .await
            .expect("list");
        assert!(
            !warm.iter().any(|e| e.key == "k1"),
            "promoted memory should be gone from warm, got: {:?}",
            warm
        );
    }

    /**
    陈旧的低召回 Active 记忆会被降级为 Archived（陈旧衰减，而非删除）。
    使用 `put_raw` 回拨 `created_at`。
    */
    #[tokio::test]
    async fn dreaming_demotes_stale_low_recall_active() {
        let (lm, store) = make_manager();
        // 低置信度 ProjectFact，recall_count=0，Active。回拨 created_at。
        let meta = MemoryMeta::new(
            MemoryType::ProjectFact,
            MemorySource::AutoExtracted,
            "project",
        )
        .with_confidence(0.4);
        let value = TypedMemoryValue::new("maybe uses yarn", meta)
            .to_value()
            .expect("to_value");
        let mut item = StoreItem::new(
            vec!["agent".to_string(), "memories".to_string()],
            "k_stale".to_string(),
            value,
        );
        item.created_at = now_secs().saturating_sub(40 * 86400); // 40 天前
        store.put_raw(item).await;

        let dreaming = Dreaming::new(lm.clone(), DreamingConfig::default());
        let report = dreaming.run().await.expect("dreaming run");

        assert_eq!(report.scanned, 1);
        assert_eq!(report.demoted, 1, "stale low-recall Active memory demoted");
        // 被降级的记忆原地保留在温层，但状态已变为 Archived。
        let warm = lm
            .list_warm_memories(&MemoryFilter::new())
            .await
            .expect("list");
        let e = warm
            .iter()
            .find(|e| e.key == "k_stale")
            .expect("still in warm");
        assert_eq!(
            e.meta.status,
            MemoryStatus::Archived,
            "demoted to Archived in place"
        );
    }

    /// 新鲜（不陈旧）的低召回 Active 记忆保持不变。
    #[tokio::test]
    async fn dreaming_leaves_fresh_low_recall_active_alone() {
        let (lm, _store) = make_manager();
        let meta = MemoryMeta::new(MemoryType::ProjectFact, MemorySource::AutoExtracted, "p")
            .with_confidence(0.4);
        lm.write_memory("k_fresh", "a fresh low-recall fact", meta)
            .await
            .expect("write");

        let dreaming = Dreaming::new(lm.clone(), DreamingConfig::default());
        let report = dreaming.run().await.expect("dreaming run");

        assert_eq!(report.scanned, 1);
        assert_eq!(report.promoted, 0, "low-recall not promoted");
        assert_eq!(report.demoted, 0, "fresh memory not demoted (not stale)");
    }

    /// 累计召回次数高但长期未使用的记忆不会被复活或提升。
    #[tokio::test]
    async fn dreaming_does_not_promote_old_lifetime_recall_count() -> crate::error::Result<()> {
        let (lm, store) = make_manager();
        let mut meta = MemoryMeta::new(
            MemoryType::UserPreference,
            MemorySource::ExplicitSave,
            "user",
        )
        .with_confidence(0.9)
        .with_recall_weight(0.9);
        meta.status = MemoryStatus::Archived;
        meta.recall_count = 10;
        let value = TypedMemoryValue::new("old preference", meta).to_value()?;
        let mut item = StoreItem::new(
            vec!["agent".to_string(), "memories".to_string()],
            "old_recall".to_string(),
            value,
        );
        item.created_at = now_secs().saturating_sub(90 * 86400);
        store.put_raw(item).await;

        let report = Dreaming::new(lm, DreamingConfig::default()).run().await?;

        assert_eq!(report.revived, 0);
        assert_eq!(report.promoted, 0);
        assert!(report.decisions.is_empty());
        Ok(())
    }
}
