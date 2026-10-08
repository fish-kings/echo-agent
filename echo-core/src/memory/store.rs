/*!
长期记忆 Store trait 与数据类型

数据以 `namespace / key / value` 三元组组织，其中 namespace 为 `&[&str]` 切片
（例如 `&["alice", "memories"]`），天然支持多用户/多 Agent 隔离。

具体实现（`InMemoryStore`、`FileStore`）位于 `echo_state`。
*/

use crate::error::{MemoryError, Result};
use futures::future::BoxFuture;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Store 中的单条记录
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoreItem {
    /// 命名空间（例如 `["user_123", "memories"]`）
    pub namespace: Vec<String>,
    /// 记录的唯一键
    pub key: String,
    /// 任意 JSON 值
    pub value: Value,
    /// 创建时间（Unix 秒）
    pub created_at: u64,
    /// 最后更新时间（Unix 秒）
    pub updated_at: u64,
    /// 搜索相关性得分（仅当由 `search` 返回时为非 None）
    #[serde(default)]
    pub score: Option<f32>,
    /// 重要性评分（1.0–10.0，默认 5.0）。越高越不容易被清理。
    #[serde(default = "default_importance")]
    pub importance: f32,
    /// 最后访问时间戳（Unix 秒），用于衰减/清理决策。
    #[serde(default)]
    pub last_accessed: Option<u64>,
    /** 可选的过期时间（Unix 秒）。超过该时间的记录会成为清理候选。
    `None` 表示永不过期。 */
    #[serde(default)]
    pub expires_at: Option<u64>,
}

/// serde 反序列化时 `importance` 字段的默认值
fn default_importance() -> f32 {
    5.0
}

impl StoreItem {
    /// 以当前时间戳创建新的 StoreItem
    pub fn new(namespace: Vec<String>, key: String, value: Value) -> Self {
        let now = crate::utils::time::now_secs();
        Self {
            namespace,
            key,
            value,
            created_at: now,
            updated_at: now,
            score: None,
            importance: 5.0,
            last_accessed: None,
            expires_at: None,
        }
    }

    /// 创建指定重要性的 StoreItem（重要性会被限制在 1.0–10.0）。
    pub fn with_importance(
        namespace: Vec<String>,
        key: String,
        value: Value,
        importance: f32,
    ) -> Self {
        let mut item = Self::new(namespace, key, value);
        item.importance = importance.clamp(1.0, 10.0);
        item
    }

    /// 创建带 TTL（从现在起的存活秒数）的 StoreItem。
    pub fn with_ttl(namespace: Vec<String>, key: String, value: Value, ttl_secs: u64) -> Self {
        let mut item = Self::new(namespace, key, value);
        item.expires_at = Some(item.created_at.saturating_add(ttl_secs));
        item
    }

    /// 将该记录标记为已访问（更新 `last_accessed`）。
    pub fn touch(&mut self) {
        self.last_accessed = Some(crate::utils::time::now_secs());
    }
}

/// 清理/去重操作的结果。
#[derive(Debug, Clone, Default)]
pub struct PruneResult {
    /// 被移除的记录数。
    pub removed: u64,
    /// 被检查的记录数。
    pub checked: u64,
}

/// 搜索模式
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SearchMode {
    /// 仅关键词搜索
    Keyword,
    /// 仅语义搜索
    Semantic,
    /** 关键词 + 语义混合搜索，向量权重可配置。
    `weight` 为向量搜索的贡献度（0.0 = 纯关键词，1.0 = 纯语义，默认 0.5）。 */
    Hybrid {
        /** 向量/语义搜索在最终得分中的权重（0.0–1.0）。
        关键词权重为 `1.0 - vector_weight`。 */
        vector_weight: f32,
    },
}

/// 默认 RRF 常数 k（标准值为 60）
pub const RRF_K: f64 = 60.0;

/**
基于两个排序结果集计算 Reciprocal Rank Fusion（RRF）得分。

`vector_rank` 与 `keyword_rank` 为各自结果列表中从 0 开始的位置
（不在某列表中的条目使用 `usize::MAX`）。
`vector_weight` 控制两者的相对贡献。
*/
pub fn rrf_score(vector_rank: usize, keyword_rank: usize, vector_weight: f32) -> f32 {
    let vw = vector_weight as f64;
    let kw = 1.0 - vw;
    let vr = if vector_rank == usize::MAX {
        0.0
    } else {
        vw / (RRF_K + vector_rank as f64)
    };
    let kr = if keyword_rank == usize::MAX {
        0.0
    } else {
        kw / (RRF_K + keyword_rank as f64)
    };
    (vr + kr) as f32
}

/// 统一搜索请求
#[derive(Debug, Clone, Copy)]
pub struct SearchQuery<'a> {
    /// 查询文本
    pub text: &'a str,
    /// 最多返回的结果数
    pub limit: usize,
    /// 搜索模式
    pub mode: SearchMode,
}

/// 精确值条件写入（compare-and-put）的结果。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoreCompareAndPutOutcome {
    /// 当前值与 `expected` 匹配，替换已提交。
    Applied,
    /// 当前值与 `expected` 不匹配，未发生任何修改。
    Mismatch,
}

impl<'a> SearchQuery<'a> {
    /// 构造关键词搜索请求。
    pub fn keyword(text: &'a str, limit: usize) -> Self {
        Self {
            text,
            limit,
            mode: SearchMode::Keyword,
        }
    }

    /// 构造语义搜索请求。
    pub fn semantic(text: &'a str, limit: usize) -> Self {
        Self {
            text,
            limit,
            mode: SearchMode::Semantic,
        }
    }

    /// 构造混合搜索请求（向量权重 0.5）。
    pub fn hybrid(text: &'a str, limit: usize) -> Self {
        Self {
            text,
            limit,
            mode: SearchMode::Hybrid { vector_weight: 0.5 },
        }
    }

    /// 使用自定义向量权重的混合搜索（0.0 = 纯关键词，1.0 = 纯语义）。
    pub fn hybrid_weighted(text: &'a str, limit: usize, vector_weight: f32) -> Self {
        Self {
            text,
            limit,
            mode: SearchMode::Hybrid { vector_weight },
        }
    }
}

// ── Store trait ───────────────────────────────────────────────────────────────

/// 长期记忆的统一存储接口
pub trait Store: Send + Sync {
    /// 写入或更新一条记录（upsert）
    fn put<'a>(
        &'a self,
        namespace: &'a [&'a str],
        key: &'a str,
        value: Value,
    ) -> BoxFuture<'a, Result<()>>;

    /**
    当当前 JSON 值精确匹配时，原子地替换该值。

    `expected = None` 表示该键必须不存在。无法在同一权威下串行化比较与写入的实现
    必须返回 `Unsupported`；调用方不得用 `get` + `put` 模拟该操作。
    */
    fn compare_and_put<'a>(
        &'a self,
        namespace: &'a [&'a str],
        key: &'a str,
        expected: Option<Value>,
        value: Value,
    ) -> BoxFuture<'a, Result<StoreCompareAndPutOutcome>> {
        let _ = (namespace, key, expected, value);
        Box::pin(async move {
            Err(MemoryError::Unsupported(
                "atomic compare-and-put is not supported by this Store".to_string(),
            )
            .into())
        })
    }

    /// 按键精确获取
    fn get<'a>(
        &'a self,
        namespace: &'a [&'a str],
        key: &'a str,
    ) -> BoxFuture<'a, Result<Option<StoreItem>>>;

    /// 关键词搜索，最多返回 `limit` 条（按相关性排序）
    fn search<'a>(
        &'a self,
        namespace: &'a [&'a str],
        query: &'a str,
        limit: usize,
    ) -> BoxFuture<'a, Result<Vec<StoreItem>>>;

    /**
    统一搜索入口。

    默认仅支持关键词搜索；语义/混合搜索需由具体实现显式覆盖。
    */
    fn search_with<'a>(
        &'a self,
        namespace: &'a [&'a str],
        query: SearchQuery<'a>,
    ) -> BoxFuture<'a, Result<Vec<StoreItem>>> {
        Box::pin(async move {
            match query.mode {
                SearchMode::Keyword => self.search(namespace, query.text, query.limit).await,
                SearchMode::Semantic => Err(MemoryError::Unsupported(
                    "semantic search is not supported by this Store".to_string(),
                )
                .into()),
                SearchMode::Hybrid { .. } => Err(MemoryError::Unsupported(
                    "hybrid search is not supported by this Store".to_string(),
                )
                .into()),
            }
        })
    }

    /// 删除指定键，返回其是否存在并已被删除
    fn delete<'a>(&'a self, namespace: &'a [&'a str], key: &'a str) -> BoxFuture<'a, Result<bool>>;

    /// 列出匹配给定 `prefix` 的所有命名空间
    fn list_namespaces<'a>(
        &'a self,
        prefix: Option<&'a [&'a str]>,
    ) -> BoxFuture<'a, Result<Vec<Vec<String>>>>;

    /// 列出命名空间下的所有条目（无关键词过滤、无分页限制）。
    fn list<'a>(&'a self, namespace: &'a [&'a str]) -> BoxFuture<'a, Result<Vec<StoreItem>>>;

    /**
    从命名空间中移除已过期的记录。

    返回被移除的记录数。默认实现为空操作。
    */
    fn prune_expired<'a>(&'a self, namespace: &'a [&'a str]) -> BoxFuture<'a, Result<u64>> {
        let _ = namespace;
        Box::pin(async move { Ok(0) })
    }

    /**
    按内容哈希对命名空间中的记录去重。

    按内容指纹分组，仅保留最新的一条。
    返回被移除的记录数。默认实现为空操作。
    */
    fn dedup_by_content<'a>(&'a self, namespace: &'a [&'a str]) -> BoxFuture<'a, Result<u64>> {
        let _ = namespace;
        Box::pin(async move { Ok(0) })
    }
}
