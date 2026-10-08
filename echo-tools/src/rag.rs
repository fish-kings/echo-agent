/*!
 * RAG（检索增强生成）工具
 *
 * 提供文档分块、向量索引和语义检索能力：
 * - rag_index：对文档分块并构建向量索引
 * - rag_search：在已索引文档上进行语义搜索
 * - rag_chunk_document：文档分块预览（不建立索引）
 */

use futures::future::BoxFuture;
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::RwLock;

use echo_core::error::{Result, ToolError};
use echo_core::tools::permission::ToolPermission;
use echo_core::tools::{Tool, ToolParameters, ToolResult};

// ── 共享向量存储 ───────────────────────────────────────────────────

/// 文档分块
#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct DocumentChunk {
    /// 分块唯一标识
    id: String,
    /// 分块文本内容
    content: String,
    /// 分块的向量表示
    embedding: Vec<f32>,
    /// 文档来源标识（可选）
    source: Option<String>,
    /// 分块序号（在文档中的位置）
    chunk_index: usize,
    /// 分块总数
    total_chunks: usize,
}

/// 内存中保留的最大分块数（防止无限增长）
const MAX_CHUNKS: usize = 10_000;

/// 内存向量存储
struct VectorStore {
    /// 已存储的分块
    chunks: Vec<DocumentChunk>,
}

impl VectorStore {
    /// 创建空的向量存储
    fn new() -> Self {
        Self { chunks: Vec::new() }
    }

    /// 添加分块；超出上限时淘汰最旧的分块，返回被淘汰的数量
    fn add_chunks(&mut self, mut chunks: Vec<DocumentChunk>) -> usize {
        self.chunks.append(&mut chunks);
        // 超出上限时淘汰最旧的分块
        if self.chunks.len() > MAX_CHUNKS {
            let excess = self.chunks.len() - MAX_CHUNKS;
            self.chunks.drain(0..excess);
            excess
        } else {
            0
        }
    }

    /// 按余弦相似度检索最相近的 top_k 个分块
    fn search(&self, query_embedding: &[f32], top_k: usize) -> Vec<(f32, DocumentChunk)> {
        let mut scored: Vec<(f32, DocumentChunk)> = self
            .chunks
            .iter()
            .map(|chunk| {
                let sim = cosine_similarity(query_embedding, &chunk.embedding);
                (sim, chunk.clone())
            })
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        scored
    }
}

/// 实例级向量存储，由配置好的一对索引/搜索工具共享。
#[derive(Clone)]
pub struct RagStore {
    /// 受读写锁保护的底层向量存储
    inner: Arc<RwLock<VectorStore>>,
}

impl RagStore {
    /// 创建空的 RAG 存储
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(VectorStore::new())),
        }
    }
}

impl Default for RagStore {
    /// 等价于 `new()`
    fn default() -> Self {
        Self::new()
    }
}

/// 余弦相似度
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

/// 感知段落的文本分块
fn chunk_text(text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
    if chunk_size == 0 || overlap >= chunk_size {
        return Vec::new();
    }
    let characters: Vec<char> = text.chars().collect();
    let step = chunk_size.saturating_sub(overlap);
    let mut chunks = Vec::new();
    let mut start = 0usize;
    while start < characters.len() {
        let end = start.saturating_add(chunk_size).min(characters.len());
        let chunk: String = characters
            .get(start..end)
            .unwrap_or_default()
            .iter()
            .collect();
        if !chunk.trim().is_empty() {
            chunks.push(chunk);
        }
        if end == characters.len() {
            break;
        }
        start = start.saturating_add(step);
    }
    chunks
}

// ── 嵌入辅助函数 ────────────────────────────────────────────────────────────────

/// 为一批文本生成向量
async fn generate_embeddings(
    embedder: &dyn echo_core::memory::Embedder,
    texts: &[String],
) -> Result<Vec<Vec<f32>>> {
    let mut embeddings = Vec::with_capacity(texts.len());

    for text in texts {
        let vec = embedder
            .embed(text)
            .await
            .map_err(|e| ToolError::ExecutionFailed {
                tool: "rag".to_string(),
                message: format!("Embedding generation failed: {}", e),
            })?;
        embeddings.push(vec);
    }

    Ok(embeddings)
}

// ── rag_index 工具 ──────────────────────────────────────────────────────────

/// 对文档分块并建立向量索引的 rag_index 工具
pub struct RagIndexTool {
    /// 用于生成向量的嵌入器
    embedder: Arc<dyn echo_core::memory::Embedder>,
    /// 共享的向量存储
    store: RagStore,
}

impl RagIndexTool {
    /// 使用嵌入器和共享存储创建工具
    pub fn new(embedder: Arc<dyn echo_core::memory::Embedder>, store: RagStore) -> Self {
        Self { embedder, store }
    }
}

impl Tool for RagIndexTool {
    /// 工具名称
    fn name(&self) -> &str {
        "rag_index"
    }

    /// 工具所需权限
    fn permissions(&self) -> Vec<ToolPermission> {
        vec![ToolPermission::Read, ToolPermission::Write]
    }

    /// 工具描述
    fn description(&self) -> &str {
        "Chunk documents and build a vector index for semantic retrieval. \
         Requires EMBEDDING_API_KEY env var (or compatible OPENAI_API_KEY / EMBEDDING_APIKEY). \
         Default chunk size 1000 chars, overlap 100 chars."
    }

    /// 参数 JSON Schema
    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Document text content to index"
                },
                "source": {
                    "type": "string",
                    "description": "Document source identifier (e.g. filename, URL) for result traceability"
                },
                "chunk_size": {
                    "type": "integer",
                    "description": "Chunk size in characters (default 1000)"
                },
                "overlap": {
                    "type": "integer",
                    "description": "Overlap characters between chunks (default 100)"
                }
            },
            "required": ["content"]
        })
    }

    /// 执行工具
    fn execute(&self, parameters: ToolParameters) -> BoxFuture<'_, Result<ToolResult>> {
        Box::pin(async move {
            let content = parameters
                .get("content")
                .and_then(|v| v.as_str())
                .ok_or_else(|| ToolError::MissingParameter("content".to_string()))?;

            let source = parameters
                .get("source")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());

            let chunk_size = parameters
                .get("chunk_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(1000) as usize;

            let overlap = parameters
                .get("overlap")
                .and_then(|v| v.as_u64())
                .unwrap_or(100) as usize;

            if chunk_size == 0 || overlap >= chunk_size {
                return Err(ToolError::InvalidParameter {
                    name: "overlap".to_string(),
                    message:
                        "chunk_size must be positive and overlap must be smaller than chunk_size"
                            .to_string(),
                }
                .into());
            }
            let texts = chunk_text(content, chunk_size, overlap);
            let total_chunks = texts.len();

            if texts.is_empty() {
                return Ok(ToolResult::success(
                    "Document content is empty, index skipped".to_string(),
                ));
            }

            // 生成向量
            let embeddings = match generate_embeddings(self.embedder.as_ref(), &texts).await {
                Ok(e) => e,
                Err(e) => {
                    return Ok(ToolResult::error(format!(
                        "Embedding generation failed: {}",
                        e
                    )));
                }
            };

            // 构建分块并存储
            let chunks: Vec<DocumentChunk> = texts
                .into_iter()
                .zip(embeddings)
                .enumerate()
                .map(|(i, (text, embedding))| DocumentChunk {
                    id: uuid::Uuid::new_v4().to_string(),
                    content: text,
                    embedding,
                    source: source.clone(),
                    chunk_index: i,
                    total_chunks,
                })
                .collect();

            let evicted = self.store.inner.write().await.add_chunks(chunks);

            let mut result = ToolResult::success(format!(
                "Successfully indexed {} document chunks{}",
                total_chunks,
                source
                    .map(|s| format!(" (source: {})", s))
                    .unwrap_or_default()
            ));
            result
                .metadata
                .insert("evicted_chunks".to_string(), evicted.to_string());
            Ok(result)
        })
    }
}

// ── rag_search 工具 ─────────────────────────────────────────────────────────

/// 在已索引文档上做语义检索的 rag_search 工具
pub struct RagSearchTool {
    /// 用于生成向量的嵌入器
    embedder: Arc<dyn echo_core::memory::Embedder>,
    /// 共享的向量存储
    store: RagStore,
}

impl RagSearchTool {
    /// 使用嵌入器和共享存储创建工具
    pub fn new(embedder: Arc<dyn echo_core::memory::Embedder>, store: RagStore) -> Self {
        Self { embedder, store }
    }
}

impl Tool for RagSearchTool {
    /// 工具名称
    fn name(&self) -> &str {
        "rag_search"
    }

    /// 工具所需权限
    fn permissions(&self) -> Vec<ToolPermission> {
        vec![ToolPermission::Read]
    }

    /// 工具描述
    fn description(&self) -> &str {
        "Semantic search over indexed documents, returning top_k most relevant chunks with similarity scores. \
         Requires prior rag_index to build the index."
    }

    /// 参数 JSON Schema
    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return (default 5)"
                }
            },
            "required": ["query"]
        })
    }

    /// 执行工具
    fn execute(&self, parameters: ToolParameters) -> BoxFuture<'_, Result<ToolResult>> {
        Box::pin(async move {
            let query = parameters
                .get("query")
                .and_then(|v| v.as_str())
                .ok_or_else(|| ToolError::MissingParameter("query".to_string()))?;

            let top_k = parameters
                .get("top_k")
                .and_then(|v| v.as_u64())
                .unwrap_or(5) as usize;

            // 生成查询向量
            let query_embedding =
                match generate_embeddings(self.embedder.as_ref(), &[query.to_string()]).await {
                    Ok(e) => match e.into_iter().next() {
                        Some(embedding) => embedding,
                        None => {
                            return Ok(ToolResult::error(
                                "Embedding generation returned empty result".to_string(),
                            ));
                        }
                    },
                    Err(e) => {
                        return Ok(ToolResult::error(format!(
                            "Embedding generation failed: {}",
                            e
                        )));
                    }
                };

            // 检索
            let results = self
                .store
                .inner
                .read()
                .await
                .search(&query_embedding, top_k);

            if results.is_empty() {
                return Ok(ToolResult::success(
                    "No relevant documents found. Please use rag_index to index documents first."
                        .to_string(),
                ));
            }

            let items: Vec<Value> = results
                .iter()
                .enumerate()
                .map(|(rank, (score, chunk))| {
                    serde_json::json!({
                        "rank": rank + 1,
                        "similarity_pct": format!("{:.1}", score * 100.0),
                        "chunk_index": chunk.chunk_index,
                        "total_chunks": chunk.total_chunks,
                        "source": chunk.source,
                        "content": chunk.content,
                    })
                })
                .collect();

            let result = serde_json::json!({
                "query": query,
                "total_results": results.len(),
                "results": items,
            });
            Ok(ToolResult::success_json(result))
        })
    }
}

// ── rag_chunk_document 工具 ─────────────────────────────────────────────────

/// 预览文档分块结果（不建立索引）的 rag_chunk_document 工具
pub struct RagChunkDocumentTool;

impl Tool for RagChunkDocumentTool {
    /// 工具名称
    fn name(&self) -> &str {
        "rag_chunk_document"
    }

    /// 工具所需权限
    fn permissions(&self) -> Vec<ToolPermission> {
        vec![ToolPermission::Read]
    }

    /// 工具描述
    fn description(&self) -> &str {
        "Preview document chunking results (does not build an index). Use to inspect chunking strategy and tune parameters."
    }

    /// 参数 JSON Schema
    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Text content to chunk"
                },
                "chunk_size": {
                    "type": "integer",
                    "description": "Chunk size in characters (default 1000)"
                },
                "overlap": {
                    "type": "integer",
                    "description": "Overlap characters between chunks (default 100)"
                }
            },
            "required": ["content"]
        })
    }

    /// 执行工具
    fn execute(&self, parameters: ToolParameters) -> BoxFuture<'_, Result<ToolResult>> {
        Box::pin(async move {
            let content = parameters
                .get("content")
                .and_then(|v| v.as_str())
                .ok_or_else(|| ToolError::MissingParameter("content".to_string()))?;

            let chunk_size = parameters
                .get("chunk_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(1000) as usize;

            let overlap = parameters
                .get("overlap")
                .and_then(|v| v.as_u64())
                .unwrap_or(100) as usize;

            let chunks = chunk_text(content, chunk_size, overlap);

            if chunks.is_empty() {
                return Ok(ToolResult::success("Document content is empty".to_string()));
            }

            let items: Vec<Value> = chunks
                .iter()
                .enumerate()
                .map(|(i, chunk)| {
                    let preview: String = chunk.chars().take(200).collect();
                    serde_json::json!({
                        "index": i + 1,
                        "char_count": chunk.len(),
                        "preview": preview,
                        "truncated": chunk.len() > 200,
                    })
                })
                .collect();

            let result = serde_json::json!({
                "chunk_size": chunk_size,
                "overlap": overlap,
                "total_chunks": chunks.len(),
                "chunks": items,
            });
            Ok(ToolResult::success_json(result))
        })
    }
}
