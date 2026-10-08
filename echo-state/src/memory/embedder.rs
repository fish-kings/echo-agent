/*!
HTTP 向量嵌入客户端（兼容 OpenAI 接口）

[`Embedder`] trait 定义于 [`echo_core::memory::embedder`]。
本模块提供其 [`HttpEmbedder`] 实现。
*/

use echo_core::error::{MemoryError, Result};
pub use echo_core::memory::embedder::Embedder;
use futures::future::BoxFuture;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tracing::debug;

// ── HTTP 向量嵌入客户端 ────────────────────────────────────────────────────

/// 向嵌入服务发起请求的请求体。
#[derive(Serialize)]
struct EmbedRequest<'a> {
    /// 使用的嵌入模型名称。
    model: &'a str,
    /// 待生成向量嵌入的输入文本。
    input: &'a str,
}

/// 嵌入服务返回的响应体。
#[derive(Deserialize)]
struct EmbedResponse {
    /// 返回的嵌入结果列表。
    data: Vec<EmbedData>,
}

/// 单条嵌入结果。
#[derive(Deserialize)]
struct EmbedData {
    /// 生成的向量嵌入数值。
    embedding: Vec<f32>,
}

/**
兼容 OpenAI 接口的 HTTP 向量嵌入客户端

支持 OpenAI、通义千问（DashScope）以及其他兼容 `/v1/embeddings` 接口的服务。

# 环境变量

| 用途 | 优先级 1 | 优先级 2 | 优先级 3 |
|---------|------------|------------|-------------|
| 完整端点 URL | `EMBEDDING_BASEURL` | — | — |
| 基础 URL（自动拼接 `/v1/embeddings`） | `EMBEDDING_API_URL` | — | `https://api.openai.com` |
| API 密钥 | `EMBEDDING_APIKEY` | `EMBEDDING_API_KEY` | `OPENAI_API_KEY` |
| 模型名称 | `EMBEDDING_MODEL` | — | `text-embedding-3-small` |
*/
pub struct HttpEmbedder {
    /// 底层 HTTP 客户端。
    client: reqwest::Client,
    /// 嵌入服务的完整请求 URL。
    url: String,
    /// 鉴权用的 API key。
    api_key: String,
    /// 使用的嵌入模型名称。
    model: String,
    /// 请求超时时间（默认 30 秒）
    timeout: Duration,
}

impl HttpEmbedder {
    /// 默认请求超时时间（30 秒）。
    const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);

    /// 基础 URL 构造函数：自动拼接 `/v1/embeddings`
    pub fn new(
        api_url: impl Into<String>,
        api_key: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        let base = api_url.into();
        let base = base.trim_end_matches('/').to_string();
        Self {
            client: reqwest::ClientBuilder::new()
                .timeout(Self::DEFAULT_TIMEOUT)
                .build()
                .unwrap_or_default(),
            url: format!("{base}/v1/embeddings"),
            api_key: api_key.into(),
            model: model.into(),
            timeout: Self::DEFAULT_TIMEOUT,
        }
    }

    /// 完整端点 URL 构造函数：直接使用传入的 URL
    pub fn with_endpoint(
        url: impl Into<String>,
        api_key: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        Self {
            client: reqwest::ClientBuilder::new()
                .timeout(Self::DEFAULT_TIMEOUT)
                .build()
                .unwrap_or_default(),
            url: url.into(),
            api_key: api_key.into(),
            model: model.into(),
            timeout: Self::DEFAULT_TIMEOUT,
        }
    }

    /// 设置请求超时时间
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self.client = reqwest::ClientBuilder::new()
            .timeout(timeout)
            .build()
            .unwrap_or_default();
        self
    }

    /// 从环境变量构造
    pub fn from_env() -> Self {
        let (url, is_full_url) = if let Ok(u) = std::env::var("EMBEDDING_BASEURL") {
            (u, true)
        } else {
            let base = std::env::var("EMBEDDING_API_URL")
                .unwrap_or_else(|_| "https://api.openai.com".to_string());
            (base, false)
        };

        let api_key = std::env::var("EMBEDDING_APIKEY")
            .or_else(|_| std::env::var("EMBEDDING_API_KEY"))
            .or_else(|_| std::env::var("OPENAI_API_KEY"))
            .unwrap_or_default();

        let model = std::env::var("EMBEDDING_MODEL")
            .unwrap_or_else(|_| "text-embedding-3-small".to_string());

        if is_full_url {
            Self::with_endpoint(url, api_key, model)
        } else {
            Self::new(url, api_key, model)
        }
        .with_timeout(
            std::env::var("EMBEDDING_TIMEOUT")
                .ok()
                .and_then(|s| s.parse::<u64>().ok())
                .map(Duration::from_secs)
                .unwrap_or(Self::DEFAULT_TIMEOUT),
        )
    }
}

impl Embedder for HttpEmbedder {
    /// 调用远程接口，为给定文本计算向量嵌入。
    fn embed<'a>(&'a self, text: &'a str) -> BoxFuture<'a, Result<Vec<f32>>> {
        Box::pin(async move {
            debug!(model = %self.model, chars = text.len(), "Computing text embedding");
            let req = EmbedRequest {
                model: &self.model,
                input: text,
            };
            let resp = self
                .client
                .post(&self.url)
                .bearer_auth(&self.api_key)
                .json(&req)
                .send()
                .await
                .map_err(|e| {
                    echo_core::error::ReactError::from(MemoryError::IoError(e.to_string()))
                })?;

            let status = resp.status();
            if !status.is_success() {
                let body = resp.text().await.unwrap_or_default();
                return Err(
                    MemoryError::IoError(format!("Embedding API error {status}: {body}")).into(),
                );
            }

            let body: EmbedResponse = resp.json().await.map_err(|e| {
                echo_core::error::ReactError::from(MemoryError::SerializationError(format!(
                    "Failed to parse embedding response: {e}"
                )))
            })?;

            body.data
                .into_iter()
                .next()
                .map(|d| d.embedding)
                .ok_or_else(|| {
                    MemoryError::IoError("Embedding API returned empty result".to_string()).into()
                })
        })
    }
}
