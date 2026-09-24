/*! Agent 记忆工具：remember / recall / forget

使用与 LangGraph 对齐的 [`Store`] API 实现持久化的长期记忆。

| 工具        | Store 操作                                                          |
|------------|----------------------------------------------------------------------|
| `remember` | 当安装了分层记忆时使用 `MemoryLayerManager::write_memory()`           |
| `recall`   | `store.search(namespace, query, limit)`                              |
| `forget`   | `store.delete(namespace, key)`                                       |
*/

use futures::future::BoxFuture;

use crate::error::ToolError;
use crate::evolution::{MemoryLayer, MemoryLayerManager};
use crate::memory::{SearchQuery, Store, StoreItem};
use crate::tools::{Tool, ToolParameters, ToolResult};
use echo_core::memory::types::{MemoryMeta, MemorySource, MemoryType};
use echo_core::tools::pagination::PageRequest;
use serde_json::{Value, json};
use std::sync::Arc;
use tracing::debug;

// ── LegacyStoreRememberTool ─────────────────────────────────────────────────

/** 旧版基于 Store 的 remember 工具。

新的运行时路径应安装 [`LayeredRememberTool`]，使显式保存都经过
`MemoryLayerManager::write_memory`。这个旧版本仅为那些未安装分层管理器、
仍启用旧版 Store 记忆工具的 agent 保留。
*/
pub struct LegacyStoreRememberTool {
    /// 底层持久化存储。
    pub store: Arc<dyn Store>,
    /// 存储命名空间，例如 `["alice", "memories"]`
    pub namespace: Vec<String>,
}

impl LegacyStoreRememberTool {
    /// 使用给定的存储与命名空间创建工具实例。
    pub fn new(store: Arc<dyn Store>, namespace: Vec<String>) -> Self {
        Self { store, namespace }
    }

    /// 将命名空间转换为 `&str` 切片，供 Store API 使用。
    fn ns_refs(&self) -> Vec<&str> {
        self.namespace.iter().map(String::as_str).collect()
    }
}

impl Tool for LegacyStoreRememberTool {
    /// 返回工具名称 `"remember"`。
    fn name(&self) -> &str {
        "remember"
    }

    /// 返回展示给模型的工具描述。
    fn description(&self) -> &str {
        "Store information worth long-term retention into persistent memory (cross-session). \
         Suitable for recording user preferences, important conclusions, to-do items, key facts, etc."
    }

    /// 返回该工具的 JSON Schema 参数定义。
    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The specific content to remember; please describe concisely and completely"
                },
                "tags": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "List of tags for categorization and retrieval (optional), e.g. [\"preferences\", \"programming\"]"
                },
                "importance": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "description": "Importance level (1-10), default 5; higher values are prioritized in recall"
                }
            },
            "required": ["content"]
        })
    }

    /// 解析参数，将内容写入 Store 并返回确认消息。
    fn execute(
        &self,
        parameters: ToolParameters,
    ) -> BoxFuture<'_, crate::error::Result<ToolResult>> {
        Box::pin(async move {
            let content = parameters
                .get("content")
                .and_then(|v| v.as_str())
                .ok_or_else(|| ToolError::MissingParameter("content".to_string()))?;

            let tags: Vec<String> = parameters
                .get("tags")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|t| t.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default();

            let importance = parameters
                .get("importance")
                .and_then(|v| v.as_u64())
                .map(|n| n.clamp(1, 10))
                .unwrap_or(5);

            let key = uuid::Uuid::new_v4().to_string();
            let value = json!({
                "content": content,
                "importance": importance,
                "tags": tags,
            });

            debug!(key = %key, importance = importance, "💡 remember tool writing to Store");

            let ns: Vec<&str> = self.ns_refs();
            self.store.put(&ns, &key, value).await?;

            let tag_str = if tags.is_empty() {
                String::new()
            } else {
                format!("(tags: {})", tags.join(", "))
            };

            Ok(ToolResult::success(format!(
                "✅ Remembered (ID: {}, importance: {}): \"{}\"{tag_str}",
                key.get(..8).unwrap_or(&key),
                importance,
                content,
            )))
        })
    }
}

// ── LayeredRememberTool ────────────────────────────────────────────────────

/** 通过 evolution 分层管理器存储记忆。

保持公开的 `remember` 工具契约不变，同时将写入路由经过类型化记忆、
安全检查、审计日志、层级晋升以及共享的复核计数器。
*/
pub struct LayeredRememberTool {
    /// 负责分层记忆读写、晋升与审计的管理器。
    pub layer_manager: Arc<MemoryLayerManager>,
}

impl LayeredRememberTool {
    /// 使用给定的分层记忆管理器创建工具实例。
    pub fn new(layer_manager: Arc<MemoryLayerManager>) -> Self {
        Self { layer_manager }
    }
}

impl Tool for LayeredRememberTool {
    /// 返回工具名称 `"remember"`。
    fn name(&self) -> &str {
        "remember"
    }

    /// 返回展示给模型的工具描述。
    fn description(&self) -> &str {
        "Store information worth long-term retention into persistent typed memory. \
         Suitable for user preferences, project facts, decisions, debugging lessons, and durable workflow patterns."
    }

    /// 返回该工具的 JSON Schema 参数定义。
    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The specific content to remember; please describe concisely and completely"
                },
                "tags": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "List of tags for categorization and retrieval (optional), e.g. [\"preferences\", \"programming\"]"
                },
                "importance": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "description": "Importance level (1-10), default 5; higher values are prioritized in recall"
                }
            },
            "required": ["content"]
        })
    }

    /// 解析参数，推断记忆类型后经分层管理器写入类型化记忆。
    fn execute(
        &self,
        parameters: ToolParameters,
    ) -> BoxFuture<'_, crate::error::Result<ToolResult>> {
        Box::pin(async move {
            let content = parameters
                .get("content")
                .and_then(|v| v.as_str())
                .ok_or_else(|| ToolError::MissingParameter("content".to_string()))?;

            let tags: Vec<String> = parameters
                .get("tags")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|t| t.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default();

            let importance = parameters
                .get("importance")
                .and_then(|v| v.as_u64())
                .map(|n| n.clamp(1, 10))
                .unwrap_or(5);

            let memory_type = classify_memory_type(content, &tags);
            let topic = tags
                .first()
                .cloned()
                .unwrap_or_else(|| memory_type_topic(memory_type).to_string());
            let confidence = (0.55 + importance as f32 * 0.045).clamp(0.0, 1.0);
            let meta = MemoryMeta::new(memory_type, MemorySource::ExplicitSave, topic)
                .with_confidence(confidence);
            let key = uuid::Uuid::new_v4().to_string();

            self.layer_manager.write_memory(&key, content, meta).await?;

            let tag_str = if tags.is_empty() {
                String::new()
            } else {
                format!(" (tags: {})", tags.join(", "))
            };
            Ok(ToolResult::success(format!(
                "✅ Remembered in typed memory (ID: {}, importance: {}): \"{}\"{}",
                key.get(..8).unwrap_or(&key),
                importance,
                content,
                tag_str
            )))
        })
    }
}

// ── RecallTool ───────────────────────────────────────────────────────────────

/** 从持久化 Store 中检索相关的历史记忆

内部调用 `store.search(namespace, query, limit)`
*/
pub struct RecallTool {
    /// 底层持久化存储。
    pub store: Arc<dyn Store>,
    /// 存储命名空间。
    pub namespace: Vec<String>,
}

impl RecallTool {
    /// 使用给定的存储与命名空间创建工具实例。
    pub fn new(store: Arc<dyn Store>, namespace: Vec<String>) -> Self {
        Self { store, namespace }
    }

    /// 将命名空间转换为 `&str` 切片，供 Store API 使用。
    fn ns_refs(&self) -> Vec<&str> {
        self.namespace.iter().map(String::as_str).collect()
    }
}

impl Tool for RecallTool {
    /// 返回工具名称 `"recall"`。
    fn name(&self) -> &str {
        "recall"
    }

    /// 返回展示给模型的工具描述。
    fn description(&self) -> &str {
        "Search persistent memory by keyword or natural-language fragment."
    }

    /// 返回该工具的 JSON Schema 参数定义。
    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search keywords or description, e.g. \"user preferences\" or \"project name mentioned last time\""
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                    "description": "Maximum number of results to return (default 5)"
                }
            },
            "required": ["query"]
        })
    }

    /// 解析参数，按关键词在 Store 中检索并格式化结果列表。
    fn execute(
        &self,
        parameters: ToolParameters,
    ) -> BoxFuture<'_, crate::error::Result<ToolResult>> {
        Box::pin(async move {
            let query = parameters
                .get("query")
                .and_then(|v| v.as_str())
                .ok_or_else(|| ToolError::MissingParameter("query".to_string()))?;

            let limit = parameters
                .get("limit")
                .and_then(|v| v.as_u64())
                .map(|n| n.clamp(1, 20) as usize)
                .unwrap_or(5);

            debug!(query = %query, limit = limit, "🔍 recall tool querying Store");

            let ns: Vec<&str> = self.ns_refs();
            let items = self.store.search(&ns, query, limit).await?;

            if items.is_empty() {
                return Ok(ToolResult::success(format!(
                    "No memories found matching \"{}\".",
                    query
                )));
            }

            let mut lines = vec![format!("Found {} matching memories:", items.len())];
            for (i, item) in items.iter().enumerate() {
                lines.push(format!(
                    "{}. [ID:{}] {}",
                    i + 1,
                    item.key.get(..8).unwrap_or(&item.key),
                    format_store_item(item),
                ));
            }

            Ok(ToolResult::success(lines.join("\n")))
        })
    }
}

// ── ForgetTool ───────────────────────────────────────────────────────────────

/** 按 ID（key）删除一条记忆，或清空某命名空间下的全部记忆

内部调用 `store.delete(namespace, key)`
*/
pub struct ForgetTool {
    /// 底层持久化存储。
    pub store: Arc<dyn Store>,
    /// 存储命名空间。
    pub namespace: Vec<String>,
}

impl ForgetTool {
    /// 使用给定的存储与命名空间创建工具实例。
    pub fn new(store: Arc<dyn Store>, namespace: Vec<String>) -> Self {
        Self { store, namespace }
    }

    /// 将命名空间转换为 `&str` 切片，供 Store API 使用。
    fn ns_refs(&self) -> Vec<&str> {
        self.namespace.iter().map(String::as_str).collect()
    }
}

impl Tool for ForgetTool {
    /// 返回工具名称 `"forget"`。
    fn name(&self) -> &str {
        "forget"
    }

    /// 返回展示给模型的工具描述。
    fn description(&self) -> &str {
        "Delete a memory entry by its ID. The ID can be obtained from recall tool results (first 8 chars is sufficient)."
    }

    /// 返回该工具的 JSON Schema 参数定义。
    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "Memory ID to delete (first 8 chars prefix from recall results)"
                }
            },
            "required": ["id"]
        })
    }

    /// 解析参数，按 ID 前缀或完整 key 在 Store 中删除对应记忆。
    fn execute(
        &self,
        parameters: ToolParameters,
    ) -> BoxFuture<'_, crate::error::Result<ToolResult>> {
        Box::pin(async move {
            let id_prefix = parameters
                .get("id")
                .and_then(|v| v.as_str())
                .ok_or_else(|| ToolError::MissingParameter("id".to_string()))?;

            let ns: Vec<&str> = self.ns_refs();

            // 先尝试精确匹配；若失败，则在所有 key 中按前缀查找
            let full_key = self.store.get(&ns, id_prefix).await?.map(|item| item.key);

            // 尝试直接删除（用户可能传入的是完整 key）
            let deleted = if let Some(key) = &full_key {
                self.store.delete(&ns, key).await?
            } else {
                // 假定用户传入的是完整 key（UUID 格式）
                self.store.delete(&ns, id_prefix).await?
            };

            if deleted {
                Ok(ToolResult::success(format!(
                    "🗑️ Deleted memory ID: {}",
                    id_prefix
                )))
            } else {
                Ok(ToolResult::success(format!(
                    "No memory entry found with ID \"{}\", nothing to delete.\nTip: use the recall tool to find the correct ID.",
                    id_prefix
                )))
            }
        })
    }
}

// ── SearchMemoryTool ─────────────────────────────────────────────────────────

/** 在持久化 Store 中执行混合检索（关键词 + 向量嵌入）

与仅支持关键词检索的 `RecallTool` 不同，本工具优先使用
[`Store::search_with`] 的 `Hybrid` 模式。当连接到支持 embedding 的 Store 时，
会同时利用向量相似度与关键词匹配；当底层 Store 不支持混合检索时，
会回退为关键词检索。
*/
pub struct SearchMemoryTool {
    /// 底层持久化存储。
    pub store: Arc<dyn Store>,
    /// 存储命名空间。
    pub namespace: Vec<String>,
}

impl SearchMemoryTool {
    /// 使用给定的存储与命名空间创建工具实例。
    pub fn new(store: Arc<dyn Store>, namespace: Vec<String>) -> Self {
        Self { store, namespace }
    }

    /// 将命名空间转换为 `&str` 切片，供 Store API 使用。
    fn ns_refs(&self) -> Vec<&str> {
        self.namespace.iter().map(String::as_str).collect()
    }
}

impl Tool for SearchMemoryTool {
    /// 返回工具名称 `"search_memory"`。
    fn name(&self) -> &str {
        "search_memory"
    }

    /// 返回展示给模型的工具描述。
    fn description(&self) -> &str {
        "Hybrid search persistent memory, with keyword fallback."
    }

    /// 返回该工具的 JSON Schema 参数定义。
    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language query describing the memory content you want to find"
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                    "description": "Results per page (default 5)"
                },
                "cursor": {
                    "type": "string",
                    "description": "Cursor from the previous page"
                }
            },
            "required": ["query"]
        })
    }

    /// 解析参数，优先执行混合检索，必要时回退为关键词检索并分页返回。
    fn execute(
        &self,
        parameters: ToolParameters,
    ) -> BoxFuture<'_, crate::error::Result<ToolResult>> {
        Box::pin(async move {
            let query = parameters
                .get("query")
                .and_then(|v| v.as_str())
                .ok_or_else(|| ToolError::MissingParameter("query".to_string()))?;

            let page_request = match PageRequest::from_parameters(&parameters, 5, 20) {
                Ok(request) => request,
                Err(error) => return Ok(ToolResult::invalid_arguments(error.to_string())),
            };

            debug!(query = %query, limit = page_request.limit, "🔎 search_memory hybrid search on Store");

            let ns: Vec<&str> = self.ns_refs();
            let items = match self
                .store
                .search_with(&ns, SearchQuery::hybrid(query, 20))
                .await
            {
                Ok(items) => items,
                Err(err) if format!("{err}").contains("hybrid search") => {
                    self.store.search(&ns, query, 20).await?
                }
                Err(err) => return Err(err),
            };

            let records = items
                .iter()
                .map(|item| {
                    format!(
                        "[ID:{}] {}",
                        memory_key_preview(&item.key),
                        format_store_item(item)
                    )
                })
                .collect();
            let query_identity = serde_json::json!({
                "query": query,
                "namespace": self.namespace,
                "mode": "hybrid",
            });
            paginated_memory_result(&page_request, records, &query_identity, query)
        })
    }
}

// ── LayeredRecallTool / LayeredSearchMemoryTool ────────────────────────────

/// 跨热层（hot）与温层（warm）检索类型化长期记忆的工具。
pub struct LayeredRecallTool {
    /// 负责分层记忆读写与检索的管理器。
    pub layer_manager: Arc<MemoryLayerManager>,
}

impl LayeredRecallTool {
    /// 使用给定的分层记忆管理器创建工具实例。
    pub fn new(layer_manager: Arc<MemoryLayerManager>) -> Self {
        Self { layer_manager }
    }
}

impl Tool for LayeredRecallTool {
    /// 返回工具名称 `"recall"`。
    fn name(&self) -> &str {
        "recall"
    }

    /// 返回展示给模型的工具描述。
    fn description(&self) -> &str {
        "Search typed long-term memory across hot and warm layers."
    }

    /// 返回该工具的 JSON Schema 参数定义。
    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search keywords or description, e.g. \"user preferences\" or \"project convention\""
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                    "description": "Maximum number of results to return (default 5)"
                }
            },
            "required": ["query"]
        })
    }

    /// 解析参数，跨分层调用管理器检索并格式化结果列表。
    fn execute(
        &self,
        parameters: ToolParameters,
    ) -> BoxFuture<'_, crate::error::Result<ToolResult>> {
        Box::pin(async move {
            let query = parameters
                .get("query")
                .and_then(|v| v.as_str())
                .ok_or_else(|| ToolError::MissingParameter("query".to_string()))?;
            let limit = parameters
                .get("limit")
                .and_then(|v| v.as_u64())
                .map(|n| n.clamp(1, 20) as usize)
                .unwrap_or(5);

            let items = self.layer_manager.search_layered(query, limit).await?;
            if items.is_empty() {
                return Ok(ToolResult::success(format!(
                    "No memories found matching \"{}\".",
                    query
                )));
            }

            let mut lines = vec![format!("Found {} matching typed memories:", items.len())];
            for (i, (layer, item)) in items.iter().enumerate() {
                lines.push(format!(
                    "{}. [{}:{}] {}",
                    i + 1,
                    format_memory_layer(*layer),
                    item.key.get(..8).unwrap_or(&item.key),
                    format_typed_memory(item),
                ));
            }
            Ok(ToolResult::success(lines.join("\n")))
        })
    }
}

/// 跨热层与温层进行分页检索类型化长期记忆的工具。
pub struct LayeredSearchMemoryTool {
    /// 负责分层记忆读写与检索的管理器。
    pub layer_manager: Arc<MemoryLayerManager>,
}

impl LayeredSearchMemoryTool {
    /// 使用给定的分层记忆管理器创建工具实例。
    pub fn new(layer_manager: Arc<MemoryLayerManager>) -> Self {
        Self { layer_manager }
    }
}

impl Tool for LayeredSearchMemoryTool {
    /// 返回工具名称 `"search_memory"`。
    fn name(&self) -> &str {
        "search_memory"
    }

    /// 返回展示给模型的工具描述。
    fn description(&self) -> &str {
        "Search typed long-term memory across hot and warm layers."
    }

    /// 返回该工具的 JSON Schema 参数定义。
    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language query describing the memory content you want to find"
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                    "description": "Results per page (default 5)"
                },
                "cursor": {
                    "type": "string",
                    "description": "Cursor from the previous page"
                }
            },
            "required": ["query"]
        })
    }

    /// 解析参数，跨分层检索并分页返回类型化记忆。
    fn execute(
        &self,
        parameters: ToolParameters,
    ) -> BoxFuture<'_, crate::error::Result<ToolResult>> {
        Box::pin(async move {
            let query = parameters
                .get("query")
                .and_then(|v| v.as_str())
                .ok_or_else(|| ToolError::MissingParameter("query".to_string()))?;
            let page_request = match PageRequest::from_parameters(&parameters, 5, 20) {
                Ok(request) => request,
                Err(error) => return Ok(ToolResult::invalid_arguments(error.to_string())),
            };
            let items = self.layer_manager.search_layered(query, 20).await?;
            let records = items
                .iter()
                .map(|(layer, item)| {
                    format!(
                        "[{}:{}] {}",
                        format_memory_layer(*layer),
                        memory_key_preview(&item.key),
                        format_typed_memory(item)
                    )
                })
                .collect();
            let query_identity = serde_json::json!({
                "query": query,
                "mode": "layered_hybrid",
            });
            paginated_memory_result(&page_request, records, &query_identity, query)
        })
    }
}

/// 将分页后的记忆记录格式化为工具结果，并附带分页信息。
fn paginated_memory_result(
    request: &PageRequest,
    records: Vec<String>,
    query_identity: &Value,
    query: &str,
) -> crate::error::Result<ToolResult> {
    let (page, page_info) = match request.paginate(records, query_identity) {
        Ok(page) => page,
        Err(error) => return Ok(ToolResult::invalid_arguments(error.to_string())),
    };
    let output = if page.is_empty() {
        format!("No memories found matching \"{query}\".")
    } else {
        format!(
            "Found {} matching memories on this page ({} total):\n{}",
            page_info.returned,
            page_info.total.unwrap_or(0),
            page.join("\n")
        )
    };
    let mut result = ToolResult::success(output);
    page_info.apply_to(&mut result);
    Ok(result)
}

/// 截取记忆 key 的前 8 个字符，用于展示预览。
fn memory_key_preview(key: &str) -> String {
    key.chars().take(8).collect()
}

// ── LayeredForgetTool ──────────────────────────────────────────────────────

/** 通过 evolution 分层管理器删除一条类型化长期记忆（C4 修复）。

经由 [`MemoryLayerManager::delete_memory`] 路由，从而使用统一的命名空间
`["agent", "memories"]`——与 [`LayeredRememberTool`] 的写入位置一致。
当安装了分层管理器时，本工具替代旧版 `ForgetTool`；若不替代，`forget`
会从按 agent 划分的命名空间删除，永远无法删除 `remember` 实际存入的内容。

接受完整 key（UUID）或 recall 结果中展示的 8 字符前缀。如果前缀存在歧义
（匹配到多条），则不会执行删除，而是列出所有匹配项供用户消歧。
*/
pub struct LayeredForgetTool {
    /// 负责分层记忆读写与删除的管理器。
    pub layer_manager: Arc<MemoryLayerManager>,
}

impl LayeredForgetTool {
    /// 使用给定的分层记忆管理器创建工具实例。
    pub fn new(layer_manager: Arc<MemoryLayerManager>) -> Self {
        Self { layer_manager }
    }
}

/** 将用户提供的 id（完整 key 或短前缀）解析为至多一个具体的记忆 key。

未匹配到任何记忆时返回 `Ok(None)`；当前缀存在歧义（匹配多条）时返回
`Err(InvalidParameter)`，交由调用方提示用户消歧。
*/
async fn resolve_forget_key(
    layer_manager: &Arc<MemoryLayerManager>,
    id: &str,
) -> crate::error::Result<Option<String>> {
    // 快速路径：精确匹配。
    if layer_manager.locate(id).await?.is_some() {
        return Ok(Some(id.to_string()));
    }
    // 慢速路径：将 `id` 视为前缀，扫描热、温两层。
    let mut matches = Vec::new();
    for entry in layer_manager.list_hot()? {
        if entry.key.starts_with(id) {
            matches.push(entry.key);
        }
    }
    let warm = layer_manager
        .list_warm_memories(&echo_state::memory::typed_store::MemoryFilter::new())
        .await?;
    for entry in warm {
        if entry.key.starts_with(id) {
            matches.push(entry.key);
        }
    }
    matches.sort();
    matches.dedup();
    match matches.len() {
        0 => Ok(None),
        1 => Ok(matches.into_iter().next()),
        _ => Err(ToolError::InvalidParameter {
            name: "id".to_string(),
            message: format!(
                "ID prefix '{id}' matches {} memories; provide more characters. Matches: {}",
                matches.len(),
                matches
                    .iter()
                    .map(|k| k.chars().take(8).collect::<String>())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        }
        .into()),
    }
}

impl Tool for LayeredForgetTool {
    /// 返回工具名称 `"forget"`。
    fn name(&self) -> &str {
        "forget"
    }

    /// 返回展示给模型的工具描述。
    fn description(&self) -> &str {
        "Delete a typed long-term memory by its ID (full key or the 8-char prefix shown in recall results). \
         If the prefix matches multiple memories, no deletion happens and the matches are listed for disambiguation."
    }

    /// 返回该工具的 JSON Schema 参数定义。
    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "Memory ID to delete (full key or first 8 chars from recall results)"
                }
            },
            "required": ["id"]
        })
    }

    /// 解析参数，消解 ID 前缀为具体 key 后经分层管理器删除对应记忆。
    fn execute(
        &self,
        parameters: ToolParameters,
    ) -> BoxFuture<'_, crate::error::Result<ToolResult>> {
        let layer_manager = self.layer_manager.clone();
        Box::pin(async move {
            let id = parameters
                .get("id")
                .and_then(|v| v.as_str())
                .ok_or_else(|| ToolError::MissingParameter("id".to_string()))?;

            // 解析出完整 key（精确匹配或无歧义的前缀匹配）。
            let key = match resolve_forget_key(&layer_manager, id).await? {
                Some(k) => k,
                None => {
                    return Ok(ToolResult::success(format!(
                        "No memory entry found with ID \"{id}\", nothing to delete.\n\
                         Tip: use the recall tool to find the correct ID."
                    )));
                }
            };

            let deleted = layer_manager.delete_memory(&key).await?;
            if deleted {
                Ok(ToolResult::success(format!(
                    "🗑️ Deleted memory ID: {}",
                    key.chars().take(8).collect::<String>()
                )))
            } else {
                Ok(ToolResult::success(format!(
                    "No memory entry found with ID \"{id}\", nothing to delete."
                )))
            }
        })
    }
}

// ── 辅助函数 ────────────────────────────────────────────────────────────────

/// 根据内容与标签中的关键词，粗略推断该记忆应归入的类型。
fn classify_memory_type(content: &str, tags: &[String]) -> MemoryType {
    let lower = format!(
        "{} {}",
        content.to_lowercase(),
        tags.join(" ").to_lowercase()
    );
    if lower.contains("prefer")
        || lower.contains("always")
        || lower.contains("never")
        || lower.contains("style")
        || lower.contains("user")
    {
        MemoryType::UserPreference
    } else if lower.contains("decision")
        || lower.contains("decided")
        || lower.contains("architecture")
        || lower.contains("choose")
    {
        MemoryType::ArchitectureDecision
    } else if lower.contains("error")
        || lower.contains("bug")
        || lower.contains("fix")
        || lower.contains("failed")
    {
        MemoryType::DebuggingLesson
    } else if lower.contains("command") || lower.contains("cargo") || lower.contains("npm") {
        MemoryType::CommandPattern
    } else if lower.contains("workflow") || lower.contains("process") {
        MemoryType::WorkflowPattern
    } else {
        MemoryType::ProjectFact
    }
}

/// 返回记忆类型对应的默认主题标签字符串。
fn memory_type_topic(memory_type: MemoryType) -> &'static str {
    match memory_type {
        MemoryType::UserPreference => "user",
        MemoryType::ProjectFact => "project",
        MemoryType::ArchitectureDecision => "architecture",
        MemoryType::DebuggingLesson | MemoryType::ErrorResolution => "debugging",
        MemoryType::CommandPattern => "commands",
        MemoryType::ToolUsage => "tools",
        MemoryType::WorkflowPattern | MemoryType::SkillCandidate => "workflow",
        MemoryType::DeprecatedNote => "deprecated",
    }
}

/// 将记忆层枚举格式化为展示用的字符串（hot/warm/cold）。
fn format_memory_layer(layer: MemoryLayer) -> &'static str {
    match layer {
        MemoryLayer::Hot => "hot",
        MemoryLayer::Warm => "warm",
        MemoryLayer::Cold => "cold",
    }
}

/// 将类型化记忆条目格式化为包含类型、置信度与主题的展示字符串。
fn format_typed_memory(item: &echo_state::memory::typed_store::TypedMemoryEntry) -> String {
    format!(
        "{} (type: {:?}, confidence: {:.0}%, topic: {})",
        item.content,
        item.meta.memory_type,
        item.meta.confidence * 100.0,
        item.meta.topic
    )
}

/// 将 Store 中的原始记忆条目格式化为包含内容、重要度与标签的展示字符串。
fn format_store_item(item: &StoreItem) -> String {
    match &item.value {
        Value::Object(map) => {
            let content = map
                .get("content")
                .and_then(|v| v.as_str())
                .unwrap_or("(no content)");
            let importance = map.get("importance").and_then(|v| v.as_u64());
            let tags = map
                .get("tags")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|t| t.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                })
                .filter(|s| !s.is_empty());

            let mut parts = vec![content.to_string()];
            if let Some(imp) = importance {
                parts.push(format!("[★{}]", imp));
            }
            if let Some(t) = tags {
                parts.push(format!("[{}]", t));
            }
            parts.join(" ")
        }
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evolution::EvolutionObserver;
    use echo_state::memory::store::InMemoryStore;
    use futures::future::BoxFuture;
    use std::sync::Mutex;

    #[derive(Default)]
    struct RecordingObserver {
        changes: Mutex<Vec<(String, String, String)>>,
    }

    impl EvolutionObserver for RecordingObserver {
        fn on_memory_layer_change<'a>(
            &'a self,
            key: &'a str,
            from_layer: &'a str,
            to_layer: &'a str,
        ) -> BoxFuture<'a, ()> {
            Box::pin(async move {
                if let Ok(mut changes) = self.changes.lock() {
                    changes.push((
                        key.to_string(),
                        from_layer.to_string(),
                        to_layer.to_string(),
                    ));
                }
            })
        }
    }

    #[tokio::test]
    async fn layered_forget_tool_emits_hot_delete_notification()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?.keep();
        let observer = Arc::new(RecordingObserver::default());
        let manager = Arc::new(
            MemoryLayerManager::new(
                dir,
                Arc::new(InMemoryStore::new()),
                Box::new(crate::evolution::audit::NullChangeLog),
            )
            .with_evolution_observer(observer.clone()),
        );
        let meta = MemoryMeta::new(
            MemoryType::UserPreference,
            MemorySource::ExplicitSave,
            "tool-test",
        )
        .with_confidence(0.95)
        .with_stability(0.90);
        manager
            .write_memory("tool_forget", "Forget through the tool", meta)
            .await?;
        observer
            .changes
            .lock()
            .map_err(|_| "observer lock is poisoned")?
            .clear();

        let tool = LayeredForgetTool::new(manager);
        let parameters = [("id".to_string(), json!("tool_forget"))]
            .into_iter()
            .collect();
        let result = tool.execute(parameters).await?;

        assert!(result.success);
        assert_eq!(
            observer
                .changes
                .lock()
                .map_err(|_| "observer lock is poisoned")?
                .as_slice(),
            &[(
                "tool_forget".to_string(),
                "hot".to_string(),
                "deleted".to_string(),
            )]
        );
        Ok(())
    }
}
