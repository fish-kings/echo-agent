//! 图片抓取工具：从 URL 下载图片并转换为 base64 编码。

use echo_core::error::{Result, ToolError};
use echo_core::tools::permission::ToolPermission;
use echo_core::tools::{Tool, ToolParameters, ToolResult};
use futures::future::BoxFuture;
use serde_json::Value;
use std::time::Duration;

/// 图片抓取工具
///
/// 从 URL 下载图片并转换为适合 LLM 视觉处理的 base64 格式。
pub struct ImageFetchTool {
    /// 下载请求的超时时间（秒）。
    timeout_secs: u64,
}

impl ImageFetchTool {
    /// 创建一个新的图片抓取工具（默认超时 30 秒）。
    pub fn new() -> Result<Self> {
        Ok(Self { timeout_secs: 30 })
    }

    /// 设置自定义超时时间（秒）。
    pub fn with_timeout(mut self, timeout_secs: u64) -> Self {
        self.timeout_secs = timeout_secs;
        self
    }
}

impl Default for ImageFetchTool {
    /// 创建默认配置（30 秒超时）的图片抓取工具。
    fn default() -> Self {
        Self { timeout_secs: 30 }
    }
}

impl Tool for ImageFetchTool {
    /// 返回该工具在工具注册表中的名称：`"image_fetch"`。
    fn name(&self) -> &str {
        "image_fetch"
    }

    /// 声明该工具所需的权限：需要访问网络。
    fn permissions(&self) -> Vec<ToolPermission> {
        vec![ToolPermission::Network]
    }

    /// 返回展示给模型/调用方的工具功能描述。
    fn description(&self) -> &str {
        "Downloads an image from a URL and converts it to base64 encoding, suitable for LLM multimodal input. \
         Parameters: url - image URL (required), max_size_mb - maximum file size in MB (optional, default 10MB)"
    }

    /// 返回该工具的 JSON Schema 参数定义。
    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Image URL (supports http:// or https://)"
                },
                "max_size_mb": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "description": "Maximum file size limit (MB, default 10)"
                }
            },
            "required": ["url"]
        })
    }

    /// 执行图片下载：校验 URL、以 SSRF 安全的方式发起请求、按大小上限截断，并将结果编码为 base64 数据 URI。
    fn execute_with_context<'a>(
        &'a self,
        parameters: ToolParameters,
        context: &'a echo_core::tools::ToolContext,
    ) -> BoxFuture<'a, Result<ToolResult>> {
        Box::pin(async move {
            let url = parameters
                .get("url")
                .and_then(|v| v.as_str())
                .ok_or_else(|| ToolError::MissingParameter("url".to_string()))?;

            if url.trim().is_empty() {
                return Ok(ToolResult::error("URL cannot be empty"));
            }

            if !url.starts_with("http://") && !url.starts_with("https://") {
                return Ok(ToolResult::error("URL must start with http:// or https://"));
            }

            let max_size_mb = parameters
                .get("max_size_mb")
                .and_then(|v| v.as_u64())
                .unwrap_or(10);

            tracing::info!("ImageFetch: url='{}', max_size_mb={}", url, max_size_mb);

            // 将 MB 换算为字节数
            let max_bytes = usize::try_from(max_size_mb)
                .ok()
                .and_then(|value| value.checked_mul(1024 * 1024))
                .ok_or_else(|| ToolError::InvalidParameter {
                    name: "max_size_mb".to_string(),
                    message: "image size limit is not supported on this platform".to_string(),
                })?;

            // 检查 URL 是否指向图片文件
            let is_image = url.to_lowercase().ends_with(".png")
                || url.to_lowercase().ends_with(".jpg")
                || url.to_lowercase().ends_with(".jpeg")
                || url.to_lowercase().ends_with(".gif")
                || url.to_lowercase().ends_with(".webp")
                || url.to_lowercase().ends_with(".bmp")
                || url.to_lowercase().ends_with(".svg");

            if !is_image {
                // 尝试通过 HEAD 请求检查（SSRF 安全：解析域名并锁定 IP）
                match crate::security::local_http_request(
                    url,
                    Duration::from_secs(self.timeout_secs),
                    5,
                    reqwest::Method::HEAD,
                )
                .await
                {
                    Ok(response) => {
                        if let Some(ct) = response.headers().get("content-type")
                            && let Ok(content_type) = ct.to_str()
                            && !content_type.starts_with("image/")
                        {
                            return Ok(ToolResult::error(format!(
                                "URL does not point to an image file, Content-Type: {}",
                                content_type
                            )));
                        }
                    }
                    Err(_) => {
                        // HEAD 请求失败（网络错误、被 SSRF 拦截等）。
                        // 继续走 GET 请求，GET 同样会做 SSRF 校验。
                    }
                }
            }

            // 下载图片（SSRF 安全：解析域名、校验并在锁定的 IP 上建立连接，
            // 从而关闭 DNS 重绑定造成的 TOCTOU 时间窗口）
            let response =
                crate::security::local_http_get(url, Duration::from_secs(self.timeout_secs), 5)
                    .await
                    .map_err(|e| {
                        echo_core::error::ReactError::Tool(Box::new(ToolError::ExecutionFailed {
                            tool: "image_fetch".into(),
                            message: format!("Failed to download image: {}", e),
                        }))
                    })?;

            if !response.status().is_success() {
                return Ok(ToolResult::error(format!(
                    "HTTP error: {}",
                    response.status()
                )));
            }

            // 在消费掉响应体之前先取出 Content-Type
            let content_type = response
                .headers()
                .get("content-type")
                .and_then(|v| v.to_str().ok())
                .unwrap_or("image/jpeg")
                .to_string();

            // 提取 MIME 子类型
            let mime_subtype = content_type
                .split_once('/')
                .map(|(_, subtype)| subtype)
                .unwrap_or("png")
                .split(';')
                .next()
                .unwrap_or("png")
                .trim();

            let bytes = crate::http_body::read_bounded_body(
                response,
                max_bytes,
                "image_fetch",
                Some(context),
            )
            .await?;

            // 编码为 base64
            use base64::Engine;
            let base64_data = base64::engine::general_purpose::STANDARD.encode(&bytes);

            let data_uri = format!("data:image/{};base64,{}", mime_subtype, base64_data);

            let output = format!(
                "URL: {}\nContent-Type: {}\nSize: {} bytes\nBase64 length: {} chars\n\nData URI: {}",
                url,
                content_type,
                bytes.len(),
                base64_data.len(),
                data_uri
            );

            Ok(ToolResult::success(output))
        })
    }
}
