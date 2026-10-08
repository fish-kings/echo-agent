/*!
（stage4 D1）统一的记忆召回入口。

自动召回路径（`ReactAgent::recall_long_term_memories`）与
工具召回路径（`LayeredRecallTool` → `MemoryLayerManager::search_layered`）
都委托给 [`MemoryRecaller`]，从而在统一的 `["agent","memories"]`
命名空间上，一致地返回按复合分数排序的结果
（割裂点 3/9 —— 此前两条路径读取的是不同的命名空间，
排序方式也不一致）。
*/

use echo_core::memory::store::{Store, StoreItem};
use echo_core::memory::types::MemoryStatus;
use echo_state::memory::typed_store::{TypedMemoryEntry, TypedMemoryStore};
use std::sync::Arc;

use crate::memory::SearchQuery;

/// 在统一记忆命名空间上进行统一的复合分数召回。
pub struct MemoryRecaller {
    /// 底层的长期记忆存储后端。
    store: Arc<dyn Store>,
}

impl MemoryRecaller {
    /// 基于给定的 store 创建一个新的 [`MemoryRecaller`]。
    pub fn new(store: Arc<dyn Store>) -> Self {
        Self { store }
    }

    /**
    通过复合分数为 `query` 召回 `top_k` 条记忆。

    `S = 0.5·sim + 0.3·decay(age, 30d) + 0.2·recall_weight`。
    已被取代（Superseded）的记忆会被过滤掉（割裂点 5）；
    `recall_count` 的自增采用 fire-and-forget（触发后不等待）方式，
    供 Dreaming 自我进化（stage 2）消费。
    */
    pub async fn recall(&self, query: &str, top_k: usize) -> crate::error::Result<Vec<StoreItem>> {
        let ns = crate::evolution::layer::WARM_NAMESPACE; // ["agent","memories"]

        /*
        1. 通过混合检索获取候选项；混合检索出错时回退到关键词
           检索（不对错误文本做字符串匹配判断）。
        */
        let candidates = match self
            .store
            .search_with(ns, SearchQuery::hybrid(query, top_k * 3))
            .await
        {
            Ok(items) => items,
            Err(_) => self.store.search(ns, query, top_k * 3).await?,
        };

        // 2. 按复合分数重新排序 + 状态过滤（丢弃 Superseded）。
        let mut scored: Vec<(f64, TypedMemoryEntry)> = candidates
            .into_iter()
            .filter_map(|item| {
                let entry = TypedMemoryEntry::from_store_item(item);
                if entry.meta.status == MemoryStatus::Superseded {
                    return None;
                }
                let sim = entry.raw.score.unwrap_or(0.0) as f64;
                let age = age_days_from_storeitem(&entry.raw);
                let s = composite_score(sim, age, entry.meta.recall_weight as f64);
                Some((s, entry))
            })
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);

        // 3. recall_count +1（fire-and-forget；由 Dreaming 在 stage 2 消费）。
        let typed_for_count = TypedMemoryStore::new(self.store.clone());
        let keys: Vec<String> = scored.iter().map(|(_, e)| e.raw.key.clone()).collect();
        tokio::spawn(async move {
            for key in keys {
                if let Err(error) =
                    incr_recall_count(&typed_for_count, &["agent", "memories"], &key).await
                {
                    tracing::debug!(%key, %error, "failed to update memory recall telemetry");
                }
            }
        });

        Ok(scored.into_iter().map(|(_, e)| e.raw).collect())
    }
}

/// 复合召回分数：`S = 0.5·sim + 0.3·decay(age, 30d) + 0.2·recall_weight`。
fn composite_score(sim: f64, age_days: f64, recall_weight: f64) -> f64 {
    0.5 * sim + 0.3 * 0.5_f64.powf(age_days / 30.0) + 0.2 * recall_weight
}

/// 根据 `StoreItem::created_at`（Unix 秒）计算距今的天数。
fn age_days_from_storeitem(item: &StoreItem) -> f64 {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    now.saturating_sub(item.created_at) as f64 / 86400.0
}

/**
自增 `recall_count`（先取后写；`update_meta` 接受的是完整的
`MemoryMeta`，而不是一个闭包）。在召回过程中以 fire-and-forget
方式调用；由于这只是一个诊断用计数器，丢失个别自增是可以接受的。
*/
async fn incr_recall_count(
    typed: &TypedMemoryStore,
    ns: &[&str],
    key: &str,
) -> crate::error::Result<()> {
    if let Some(entry) = typed.get_typed(ns, key).await? {
        let mut meta = entry.meta;
        meta.recall_count = meta.recall_count.saturating_add(1);
        meta.last_recalled_at = Some(crate::utils::time::now_secs());
        typed.update_meta(ns, key, meta).await?;
    }
    Ok(())
}
