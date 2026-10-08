/*!
类型化记忆存储——为 `Store` 包装一层结构化元数据支持。

提供类型化的读写操作，会在 `StoreItem.value` 中随内容一并自动序列化/反序列化
`MemoryMeta`。向后兼容：未类型化的条目（不含 `meta` 字段的原始 JSON）会以默认元数据读取。
*/

use echo_core::error::Result;
use echo_core::memory::store::{Store, StoreCompareAndPutOutcome, StoreItem};
use echo_core::memory::types::{
    MemoryMeta, MemoryRisk, MemorySource, MemoryStatus, MemoryType, TypedMemoryValue,
};
use futures::future::BoxFuture;
use std::sync::Arc;

// ── MemoryFilter ────────────────────────────────────────────────────────

/// 检索类型化记忆时使用的过滤条件。
#[derive(Debug, Clone, Default)]
pub struct MemoryFilter {
    /// 按记忆类型过滤。
    pub memory_type: Option<MemoryType>,
    /// 按状态过滤。
    pub status: Option<MemoryStatus>,
    /// 最低置信度阈值。
    pub min_confidence: Option<f32>,
    /// 按主题过滤（精确匹配）。
    pub topic: Option<String>,
    /// 按来源过滤。
    pub source: Option<MemorySource>,
    /// 允许的最高风险等级（含）。
    pub max_risk: Option<MemoryRisk>,
}

impl MemoryFilter {
    /// 创建空过滤器（匹配所有条目）。
    pub fn new() -> Self {
        Self::default()
    }

    /// 设置按记忆类型过滤。
    pub fn with_type(mut self, memory_type: MemoryType) -> Self {
        self.memory_type = Some(memory_type);
        self
    }

    /// 设置按状态过滤。
    pub fn with_status(mut self, status: MemoryStatus) -> Self {
        self.status = Some(status);
        self
    }

    /// 设置最低置信度过滤。
    pub fn with_min_confidence(mut self, confidence: f32) -> Self {
        self.min_confidence = Some(confidence);
        self
    }

    /// 设置按主题过滤。
    pub fn with_topic(mut self, topic: impl Into<String>) -> Self {
        self.topic = Some(topic.into());
        self
    }

    /// 设置按来源过滤。
    pub fn with_source(mut self, source: MemorySource) -> Self {
        self.source = Some(source);
        self
    }

    /// 设置最高风险等级过滤。
    pub fn with_max_risk(mut self, risk: MemoryRisk) -> Self {
        self.max_risk = Some(risk);
        self
    }

    /// 判断记忆条目是否满足本过滤器的全部条件。
    pub fn matches(&self, entry: &TypedMemoryEntry) -> bool {
        if let Some(ref mt) = self.memory_type
            && entry.meta.memory_type != *mt
        {
            return false;
        }
        if let Some(ref st) = self.status
            && entry.meta.status != *st
        {
            return false;
        }
        if let Some(min_c) = self.min_confidence
            && entry.meta.confidence < min_c
        {
            return false;
        }
        if let Some(ref topic) = self.topic
            && entry.meta.topic != *topic
        {
            return false;
        }
        if let Some(ref src) = self.source
            && entry.meta.source != *src
        {
            return false;
        }
        if let Some(max_r) = self.max_risk
            && risk_order(&entry.meta.risk) > risk_order(&max_r)
        {
            return false;
        }
        true
    }
}

// ── TypedMemoryEntry ────────────────────────────────────────────────────

/// 类型化记忆条目，包含键、内容、元数据以及原始 StoreItem。
#[derive(Debug, Clone)]
pub struct TypedMemoryEntry {
    /// 命名空间内的键。
    pub key: String,
    /// 记忆内容文本。
    pub content: String,
    /// 结构化元数据。
    pub meta: MemoryMeta,
    /// 原始 StoreItem（用于访问时间戳、重要度等信息）。
    pub raw: StoreItem,
}

impl TypedMemoryEntry {
    /** 从 StoreItem 创建 TypedMemoryEntry，并解析其中的类型化值。

    如果 StoreItem 的值不是类型化格式（没有 `meta` 字段），
    则使用默认元数据创建条目，并将整个值视为内容。 */
    pub fn from_store_item(item: StoreItem) -> Self {
        let key = item.key.clone();

        // 尝试解析为 TypedMemoryValue
        if let Ok(typed) = TypedMemoryValue::from_value(&item.value) {
            Self {
                key,
                content: typed.content,
                meta: typed.meta,
                raw: item,
            }
        } else {
            /* 向后兼容：视为未类型化条目，
            尝试从值中提取字符串内容 */
            let content = extract_content_fallback(&item.value);
            Self {
                key,
                content,
                meta: MemoryMeta::default(),
                raw: item,
            }
        }
    }
}

/// 从非类型化的 StoreItem 值中提取人类可读的内容字符串。
fn extract_content_fallback(value: &serde_json::Value) -> String {
    // 如果是简单字符串
    if let Some(s) = value.as_str() {
        return s.to_string();
    }
    // 如果含有 "content" 字段
    if let Some(content) = value.get("content").and_then(|v| v.as_str()) {
        return content.to_string();
    }
    // 如果含有 "review" 字段（来自 BackgroundReviewer）
    if let Some(review) = value.get("review").and_then(|v| v.as_str()) {
        return review.to_string();
    }
    // 回退为 JSON 字符串
    value.to_string()
}

/// 用于风险比较的数值序（Low=0 < Medium=1 < High=2）。
fn risk_order(risk: &MemoryRisk) -> u8 {
    match risk {
        MemoryRisk::Low => 0,
        MemoryRisk::Medium => 1,
        MemoryRisk::High => 2,
    }
}

// ── TypedMemoryStore ────────────────────────────────────────────────────

/**
对任意 `Store` 的类型化包装，增加结构化元数据支持。

类型化记忆以 JSON 形式存储在 `StoreItem.value` 中，格式如下：
```json
{ "content": "...", "meta": { "memory_type": "...", ... } }
```

底层 `Store` 保持不变——`TypedMemoryStore` 只是其上的一层轻量序列化/反序列化层。
*/
#[derive(Clone)]
pub struct TypedMemoryStore {
    /// 被包装的底层存储。
    inner: Arc<dyn Store>,
}

impl TypedMemoryStore {
    /// 创建包装给定 Store 的类型化存储。
    pub fn new(store: Arc<dyn Store>) -> Self {
        Self { inner: store }
    }

    /// 写入一条类型化记忆条目。
    pub fn put_typed<'a>(
        &'a self,
        namespace: &'a [&'a str],
        key: &'a str,
        content: &'a str,
        meta: MemoryMeta,
    ) -> BoxFuture<'a, Result<()>> {
        let value = TypedMemoryValue::new(content, meta);
        let json_value = match value.to_value() {
            Ok(v) => v,
            Err(e) => {
                return Box::pin(async move {
                    Err(echo_core::error::MemoryError::SerializationError(e.to_string()).into())
                });
            }
        };
        self.inner.put(namespace, key, json_value)
    }

    /// 当原始当前值与期望值匹配时，原子地写入类型化值。
    pub fn compare_and_put_typed<'a>(
        &'a self,
        namespace: &'a [&'a str],
        key: &'a str,
        expected: Option<serde_json::Value>,
        content: &'a str,
        meta: MemoryMeta,
    ) -> BoxFuture<'a, Result<StoreCompareAndPutOutcome>> {
        let value = TypedMemoryValue::new(content, meta);
        let json_value = match value.to_value() {
            Ok(value) => value,
            Err(error) => {
                return Box::pin(async move {
                    Err(echo_core::error::MemoryError::SerializationError(error.to_string()).into())
                });
            }
        };
        self.inner
            .compare_and_put(namespace, key, expected, json_value)
    }

    /// 按键读取类型化记忆条目。
    pub async fn get_typed(
        &self,
        namespace: &[&str],
        key: &str,
    ) -> Result<Option<TypedMemoryEntry>> {
        match self.inner.get(namespace, key).await? {
            Some(item) => Ok(Some(TypedMemoryEntry::from_store_item(item))),
            None => Ok(None),
        }
    }

    /// 使用关键字查询及可选过滤器检索类型化记忆。
    pub async fn search_typed(
        &self,
        namespace: &[&str],
        query: &str,
        limit: usize,
        filter: &MemoryFilter,
    ) -> Result<Vec<TypedMemoryEntry>> {
        let items = self.inner.search(namespace, query, limit * 3).await?;
        let mut entries: Vec<TypedMemoryEntry> = items
            .into_iter()
            .map(TypedMemoryEntry::from_store_item)
            .filter(|e| filter.matches(e))
            .collect();

        // 按置信度降序排序，其次按新近程度
        entries.sort_by(|a, b| {
            b.meta
                .confidence
                .partial_cmp(&a.meta.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        entries.truncate(limit);
        Ok(entries)
    }

    /// 列出命名空间内的全部类型化记忆，可选过滤。
    pub async fn list_typed(
        &self,
        namespace: &[&str],
        filter: &MemoryFilter,
    ) -> Result<Vec<TypedMemoryEntry>> {
        let items = self.inner.list(namespace).await?;
        Ok(items
            .into_iter()
            .map(TypedMemoryEntry::from_store_item)
            .filter(|e| filter.matches(e))
            .collect())
    }

    /// 删除一条类型化记忆条目。
    pub async fn delete_typed(&self, namespace: &[&str], key: &str) -> Result<bool> {
        self.inner.delete(namespace, key).await
    }

    /** 更新已有类型化记忆条目的元数据。

    若条目不存在则返回 `Ok(false)`。 */
    pub async fn update_meta(
        &self,
        namespace: &[&str],
        key: &str,
        meta: MemoryMeta,
    ) -> Result<bool> {
        match self.inner.get(namespace, key).await? {
            Some(item) => {
                // 解析已有内容
                let content = if let Ok(typed) = TypedMemoryValue::from_value(&item.value) {
                    typed.content
                } else {
                    extract_content_fallback(&item.value)
                };

                let new_value = TypedMemoryValue::new(content, meta);
                let json_value = new_value.to_value().map_err(|e| {
                    echo_core::error::MemoryError::SerializationError(e.to_string())
                })?;
                self.inner.put(namespace, key, json_value).await?;
                Ok(true)
            }
            None => Ok(false),
        }
    }

    /// 获取底层 Store 的引用。
    pub fn inner(&self) -> &Arc<dyn Store> {
        &self.inner
    }
}

// ── 测试 ───────────────────────────────────────────────────────────────

/// 类型化记忆存储的单元测试。
#[cfg(test)]
mod tests {
    use super::*;
    use echo_core::memory::types::{MemoryRisk, MemorySource, MemoryStatus, MemoryType};

    /// 创建用于测试的内存存储。
    fn make_state_store() -> Arc<dyn Store> {
        Arc::new(crate::memory::store::InMemoryStore::new())
    }

    /// 测试类型化条目的写入与读取。
    #[tokio::test]
    async fn test_put_and_get_typed() {
        let store = make_state_store();
        let typed = TypedMemoryStore::new(store);

        let meta = MemoryMeta::new(
            MemoryType::DebuggingLesson,
            MemorySource::ErrorResolution,
            "build",
        );
        typed
            .put_typed(&["test", "memories"], "m1", "Maven needs Java 8", meta)
            .await
            .unwrap();

        let entry = typed
            .get_typed(&["test", "memories"], "m1")
            .await
            .unwrap()
            .expect("entry should exist");

        assert_eq!(entry.content, "Maven needs Java 8");
        assert_eq!(entry.meta.memory_type, MemoryType::DebuggingLesson);
        assert_eq!(entry.meta.source, MemorySource::ErrorResolution);
        assert_eq!(entry.meta.topic, "build");
    }

    /// 测试带过滤器的检索。
    #[tokio::test]
    async fn test_search_with_filter() {
        let store = make_state_store();
        let typed = TypedMemoryStore::new(store);

        // 插入两条不同类型的记忆
        let meta1 = MemoryMeta::new(
            MemoryType::DebuggingLesson,
            MemorySource::ErrorResolution,
            "build",
        );
        typed
            .put_typed(&["test", "memories"], "m1", "Maven compile error", meta1)
            .await
            .unwrap();

        let meta2 = MemoryMeta::new(
            MemoryType::UserPreference,
            MemorySource::ExplicitSave,
            "style",
        );
        typed
            .put_typed(
                &["test", "memories"],
                "m2",
                "User prefers concise answers",
                meta2,
            )
            .await
            .unwrap();

        // 按类型过滤
        let filter = MemoryFilter::new().with_type(MemoryType::UserPreference);
        let results = typed
            .search_typed(&["test", "memories"], "answers", 10, &filter)
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].meta.memory_type, MemoryType::UserPreference);
    }

    /// 测试带过滤器的列举。
    #[tokio::test]
    async fn test_list_with_filter() {
        let store = make_state_store();
        let typed = TypedMemoryStore::new(store);

        let meta_active = MemoryMeta::new(
            MemoryType::ProjectFact,
            MemorySource::AutoExtracted,
            "project",
        )
        .with_confidence(0.9);
        typed
            .put_typed(&["test", "memories"], "m1", "Uses npm", meta_active)
            .await
            .unwrap();

        let meta_draft = MemoryMeta::new(
            MemoryType::ProjectFact,
            MemorySource::AutoExtracted,
            "project",
        )
        .with_confidence(0.3)
        .with_status(MemoryStatus::Draft);
        typed
            .put_typed(&["test", "memories"], "m2", "Maybe uses yarn", meta_draft)
            .await
            .unwrap();

        // 按 min_confidence 过滤
        let filter = MemoryFilter::new().with_min_confidence(0.8);
        let results = typed
            .list_typed(&["test", "memories"], &filter)
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].content, "Uses npm");
    }

    /// 测试读取未类型化条目的向后兼容性。
    #[tokio::test]
    async fn test_backward_compat_read() {
        let store = make_state_store();

        // 写入一条原始（未类型化）条目
        let raw_value = serde_json::json!({
            "review": "Some review content",
            "run_id": "test-123"
        });
        store.put(&["test", "raw"], "r1", raw_value).await.unwrap();

        // 通过 TypedMemoryStore 读取
        let typed = TypedMemoryStore::new(store);
        let entry = typed
            .get_typed(&["test", "raw"], "r1")
            .await
            .unwrap()
            .expect("entry should exist");

        // 应以默认元数据解析
        assert!(entry.content.contains("Some review content"));
        assert_eq!(entry.meta.memory_type, MemoryType::ProjectFact); // 默认值
    }

    /// 测试更新元数据且内容保持不变。
    #[tokio::test]
    async fn test_update_meta() {
        let store = make_state_store();
        let typed = TypedMemoryStore::new(store);

        let meta = MemoryMeta::new(
            MemoryType::DebuggingLesson,
            MemorySource::ErrorResolution,
            "build",
        );
        typed
            .put_typed(&["test", "memories"], "m1", "Maven needs Java 8", meta)
            .await
            .unwrap();

        // 将状态更新为 Superseded
        let new_meta = MemoryMeta::new(
            MemoryType::DebuggingLesson,
            MemorySource::ErrorResolution,
            "build",
        )
        .with_status(MemoryStatus::Superseded)
        .with_confidence(0.5);
        let updated = typed
            .update_meta(&["test", "memories"], "m1", new_meta)
            .await
            .unwrap();
        assert!(updated);

        let entry = typed
            .get_typed(&["test", "memories"], "m1")
            .await
            .unwrap()
            .expect("entry should exist");
        assert_eq!(entry.meta.status, MemoryStatus::Superseded);
        assert_eq!(entry.content, "Maven needs Java 8"); // 内容被保留
    }

    /// 测试删除类型化条目。
    #[tokio::test]
    async fn test_delete_typed() {
        let store = make_state_store();
        let typed = TypedMemoryStore::new(store);

        let meta = MemoryMeta::new(MemoryType::ProjectFact, MemorySource::AutoExtracted, "test");
        typed
            .put_typed(&["test", "memories"], "m1", "Some fact", meta)
            .await
            .unwrap();

        let deleted = typed
            .delete_typed(&["test", "memories"], "m1")
            .await
            .unwrap();
        assert!(deleted);

        let entry = typed.get_typed(&["test", "memories"], "m1").await.unwrap();
        assert!(entry.is_none());
    }

    /// 测试过滤器的各项匹配条件。
    #[test]
    fn test_memory_filter_matches() {
        let meta = MemoryMeta::new(
            MemoryType::DebuggingLesson,
            MemorySource::ErrorResolution,
            "build",
        )
        .with_confidence(0.85)
        .with_status(MemoryStatus::Active);

        let entry = TypedMemoryEntry {
            key: "m1".to_string(),
            content: "test".to_string(),
            meta,
            raw: echo_core::memory::store::StoreItem::new(
                vec!["test".to_string()],
                "m1".to_string(),
                serde_json::json!({}),
            ),
        };

        // 应匹配类型过滤
        assert!(
            MemoryFilter::new()
                .with_type(MemoryType::DebuggingLesson)
                .matches(&entry)
        );
        // 不应匹配不同类型
        assert!(
            !MemoryFilter::new()
                .with_type(MemoryType::UserPreference)
                .matches(&entry)
        );
        // 应匹配 min_confidence
        assert!(MemoryFilter::new().with_min_confidence(0.8).matches(&entry));
        assert!(!MemoryFilter::new().with_min_confidence(0.9).matches(&entry));
        // 应匹配主题
        assert!(MemoryFilter::new().with_topic("build").matches(&entry));
        assert!(!MemoryFilter::new().with_topic("style").matches(&entry));
    }

    /// 测试风险等级的数值排序。
    #[test]
    fn test_risk_ordering() {
        assert!(risk_order(&MemoryRisk::Low) < risk_order(&MemoryRisk::Medium));
        assert!(risk_order(&MemoryRisk::Medium) < risk_order(&MemoryRisk::High));
        assert!(risk_order(&MemoryRisk::Low) < risk_order(&MemoryRisk::High));
    }
}
