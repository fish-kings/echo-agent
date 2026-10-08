/*!
 * 飞书 HTTP API
 *
 * 1. 获取 WebSocket 连接地址（长连接模式）
 * 2. 获取 tenant_access_token（用于发送消息）
 * 3. 发送消息 API
 */

use crate::redaction::{request_error, text_with_secrets, url as redact_url};
use echo_core::error::{ChannelError, ReactError, Result};
use reqwest::Client;
use serde::Deserialize;
use serde_json::json;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use tokio::sync::Mutex;
use tracing::{debug, info, warn};

/// 飞书 API 基础地址
pub const FEISHU_API_BASE: &str = "https://open.feishu.cn/open-apis";

/// Lark（国际版）API 基础地址
pub const LARK_API_BASE: &str = "https://open.larksuite.com/open-apis";

/// 飞书 WebSocket 端点基础地址（不含 /open-apis）
pub const FEISHU_WS_BASE: &str = "https://open.feishu.cn";

/// Lark（国际版）WebSocket 端点基础地址（不含 /open-apis）
pub const LARK_WS_BASE: &str = "https://open.larksuite.com";

/// 获取 WebSocket 端点的路径
pub const WS_ENDPOINT_PATH: &str = "/callback/ws/endpoint";

// ── HTTP 客户端 ───────────────────────────────────────────────────────────────

/// 创建 reqwest HTTP 客户端（30 秒超时）
pub fn http_client() -> Client {
    Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .unwrap_or_else(|_| Client::new())
}

// ── WebSocket 端点 API ────────────────────────────────────────────────────

/// 端点接口的响应结构
#[derive(Debug, Deserialize)]
pub struct EndpointResponse {
    /// 业务错误码，0 表示成功。
    pub code: i32,
    /// 错误或提示信息。
    pub msg: String,
    /// 端点数据（成功时存在）。
    #[serde(default)]
    pub data: Option<EndpointData>,
}

/// 端点数据
#[derive(Debug, Deserialize)]
pub struct EndpointData {
    /// WebSocket 连接地址（包含 device_id 与 service_id）
    #[serde(rename = "URL")]
    pub url: String,

    /// 客户端配置（可选）
    #[serde(rename = "ClientConfig")]
    pub client_config: Option<ClientConfig>,
}

/// 服务端下发的客户端配置
#[derive(Debug, Deserialize)]
pub struct ClientConfig {
    /// 重连次数
    #[serde(rename = "ReconnectCount")]
    pub reconnect_count: Option<i32>,

    /// 重连间隔（秒）
    #[serde(rename = "ReconnectInterval")]
    pub reconnect_interval: Option<i32>,

    /// 首次重连的随机抖动（秒）
    #[serde(rename = "ReconnectNonce")]
    pub reconnect_nonce: Option<i32>,

    /// Ping 间隔（秒）
    #[serde(rename = "PingInterval")]
    pub ping_interval: Option<i32>,
}

/// 获取 WebSocket 连接地址，返回 (连接 URL, service_id, 客户端配置)
pub async fn get_ws_endpoint(
    client: &Client,
    domain: &str,
    app_id: &str,
    app_secret: &str,
) -> Result<(String, String, Option<ClientConfig>)> {
    let url = format!("{}{}", domain, WS_ENDPOINT_PATH);

    info!(
        "Feishu: getting WebSocket endpoint from {}",
        redact_url(&url)
    );

    let body = json!({
        "AppID": app_id,
        "AppSecret": app_secret,
    });

    let res = client
        .post(&url)
        .header("Content-Type", "application/json")
        .header("locale", "zh")
        .json(&body)
        .send()
        .await
        .map_err(|e| {
            ChannelError::NetworkError(format!(
                "Failed to request endpoint: {}",
                request_error(e, [app_secret])
            ))
        })?;

    let status = res.status();
    if !status.is_success() {
        let text = res.text().await.unwrap_or_default();
        return Err(ReactError::Channel(Box::new(ChannelError::AuthError(
            format!(
                "Endpoint request failed (status {}): {}",
                status,
                text_with_secrets(&text, [app_secret])
            ),
        ))));
    }

    let resp: EndpointResponse = res.json().await.map_err(|e| {
        ChannelError::NetworkError(format!(
            "Failed to parse endpoint response: {}",
            request_error(e, [app_secret])
        ))
    })?;

    // 检查错误码
    match resp.code {
        0 => {}
        1 => {
            return Err(ReactError::Channel(Box::new(ChannelError::AuthError(
                "System busy, please retry later".to_string(),
            ))));
        }
        514 => {
            return Err(ReactError::Channel(Box::new(ChannelError::AuthError(
                "Authentication failed: invalid app_id or app_secret".to_string(),
            ))));
        }
        1000040350 => {
            return Err(ReactError::Channel(Box::new(ChannelError::AuthError(
                "Exceeded connection limit (max 50 connections per app)".to_string(),
            ))));
        }
        code => {
            return Err(ReactError::Channel(Box::new(ChannelError::AuthError(
                format!(
                    "Endpoint error (code {}): {}",
                    code,
                    text_with_secrets(&resp.msg, [app_secret])
                ),
            ))));
        }
    }

    let data = resp.data.ok_or_else(|| {
        ReactError::Channel(Box::new(ChannelError::AuthError(
            "Endpoint response missing data".to_string(),
        )))
    })?;

    if data.url.is_empty() {
        return Err(ReactError::Channel(Box::new(ChannelError::AuthError(
            "Endpoint URL is empty".to_string(),
        ))));
    }

    // 解析 URL 以获取 device_id 与 service_id
    let parsed_url = url::Url::parse(&data.url)
        .map_err(|e| ChannelError::ConnectionError(format!("Invalid endpoint URL: {}", e)))?;

    let device_id = parsed_url
        .query_pairs()
        .find(|(k, _)| k == "device_id")
        .map(|(_, v)| v.to_string())
        .unwrap_or_default();

    let service_id = parsed_url
        .query_pairs()
        .find(|(k, _)| k == "service_id")
        .map(|(_, v)| v.to_string())
        .unwrap_or_default();

    info!(
        "Feishu: got WebSocket endpoint (device_id present={}, service_id={})",
        !device_id.is_empty(),
        service_id
    );

    Ok((data.url, service_id, data.client_config))
}

// ── Token 管理 ────────────────────────────────────────────────────────────

/// tenant_access_token 管理器
pub struct TokenManager {
    /// 应用 ID。
    app_id: String,
    /// 应用密钥。
    app_secret: String,
    /// API 基础地址。
    domain: String,
    /// 缓存的 token。
    token: Arc<Mutex<Option<String>>>,
    /// 需要刷新的时间点（Unix 秒）。
    expires_at: Arc<AtomicU64>,
    /// 防止并发重复刷新
    refresh_lock: Arc<Mutex<()>>,
    /// HTTP 客户端。
    http: Client,
}

impl TokenManager {
    /// 创建 token 管理器。
    pub fn new(app_id: String, app_secret: String, domain: String) -> Self {
        Self {
            app_id,
            app_secret,
            domain,
            token: Arc::new(Mutex::new(None)),
            expires_at: Arc::new(AtomicU64::new(0)),
            refresh_lock: Arc::new(Mutex::new(())),
            http: http_client(),
        }
    }

    /// 判断缓存的 token 是否已到刷新时间。
    async fn is_expired(&self) -> bool {
        let now = chrono::Utc::now().timestamp() as u64;
        self.expires_at.load(Ordering::Relaxed) <= now
    }

    /// 返回未过期的缓存 token。
    async fn get_cached(&self) -> Option<String> {
        if self.is_expired().await {
            return None;
        }
        self.token.lock().await.clone()
    }

    /// 获取有效的 tenant_access_token，必要时刷新。
    pub async fn get_token(&self) -> Result<String> {
        if let Some(token) = self.get_cached().await {
            return Ok(token);
        }
        // 加锁以防止并发重复刷新
        let _lock = self.refresh_lock.lock().await;
        // 获取锁后再次检查；其他协程可能已完成刷新
        if let Some(token) = self.get_cached().await {
            return Ok(token);
        }
        self.refresh_token().await
    }

    /// 调用接口刷新 token，并提前 5 分钟设定刷新时间点。
    async fn refresh_token(&self) -> Result<String> {
        info!("Feishu: refreshing tenant_access_token");

        let body = json!({
            "app_id": self.app_id,
            "app_secret": self.app_secret,
        });

        let url = format!("{}{}", self.domain, "/auth/v3/tenant_access_token/internal");

        let res = self
            .http
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                ChannelError::NetworkError(format!(
                    "Token request failed: {}",
                    request_error(e, [&self.app_secret])
                ))
            })?;

        if !res.status().is_success() {
            return Err(ReactError::Channel(Box::new(ChannelError::AuthError(
                format!("Token request failed with status {}", res.status()),
            ))));
        }

        let json: serde_json::Value = res.json().await.map_err(|e| {
            ChannelError::NetworkError(format!(
                "Token response parse error: {}",
                request_error(e, [&self.app_secret])
            ))
        })?;

        let code = json["code"].as_i64().unwrap_or(-1);
        if code != 0 {
            let msg = json["msg"].as_str().unwrap_or("unknown error").to_string();
            return Err(ReactError::Channel(Box::new(ChannelError::AuthError(
                format!(
                    "Token API error (code {}): {}",
                    code,
                    text_with_secrets(&msg, [&self.app_secret])
                ),
            ))));
        }

        let token = json["tenant_access_token"]
            .as_str()
            .ok_or_else(|| {
                ReactError::Channel(Box::new(ChannelError::AuthError(
                    "Token response missing tenant_access_token".to_string(),
                )))
            })?
            .to_string();

        let expires_in = json["expire"].as_u64().unwrap_or(7200);
        let now = u64::try_from(chrono::Utc::now().timestamp()).unwrap_or_default();
        let refresh_at = now.saturating_add(expires_in.saturating_sub(300));
        self.expires_at.store(refresh_at, Ordering::Relaxed);

        let mut tok = self.token.lock().await;
        *tok = Some(token.clone());

        info!("Feishu: token refreshed, expires in {}s", expires_in);
        Ok(token)
    }
}

// ── 发送消息 ──────────────────────────────────────────────────────────────

/// 发送飞书消息，返回消息 ID
pub async fn send_message(
    client: &Client,
    domain: &str,
    token: &str,
    receive_id: &str,
    receive_id_type: &str,
    msg_type: &str,
    content: &str,
) -> Result<String> {
    let url = format!(
        "{}{}?receive_id_type={}",
        domain, "/im/v1/messages", receive_id_type
    );

    let body = json!({
        "receive_id": receive_id,
        "msg_type": msg_type,
        "content": content,
    });

    debug!(
        "Feishu: sending message to {} (type={})",
        receive_id, msg_type
    );

    let res = client
        .post(&url)
        .header("Authorization", format!("Bearer {}", token))
        .header("Content-Type", "application/json")
        .json(&body)
        .send()
        .await
        .map_err(|e| {
            ChannelError::SendError(format!(
                "Send request failed: {}",
                request_error(e, [token])
            ))
        })?;

    let status = res.status();
    if !status.is_success() {
        let text = res.text().await.unwrap_or_default();
        let text = text_with_secrets(&text, [token]);
        warn!("Feishu: send message failed (status {}): {}", status, text);
        return Err(ReactError::Channel(Box::new(ChannelError::SendError(
            format!("Send message failed (status {}): {}", status, text),
        ))));
    }

    let json: serde_json::Value = res.json().await.map_err(|e| {
        ChannelError::SendError(format!(
            "Send response parse error: {}",
            request_error(e, [token])
        ))
    })?;

    let code = json["code"].as_i64().unwrap_or(-1);
    if code != 0 {
        let msg = json["msg"].as_str().unwrap_or("unknown error").to_string();
        return Err(ReactError::Channel(Box::new(ChannelError::SendError(
            format!(
                "Send API error (code {}): {}",
                code,
                text_with_secrets(&msg, [token])
            ),
        ))));
    }

    // 返回消息 ID
    let message_id = json["data"]["message_id"]
        .as_str()
        .unwrap_or("")
        .to_string();

    debug!("Feishu: message sent, message_id={}", message_id);
    Ok(message_id)
}

/// 发送文本消息
pub async fn send_text_message(
    client: &Client,
    domain: &str,
    token: &str,
    receive_id: &str,
    receive_id_type: &str,
    text: &str,
) -> Result<String> {
    let content = json!({ "text": text }).to_string();
    send_message(
        client,
        domain,
        token,
        receive_id,
        receive_id_type,
        "text",
        &content,
    )
    .await
}

/// 发送卡片消息
pub async fn send_card_message(
    client: &Client,
    domain: &str,
    token: &str,
    receive_id: &str,
    receive_id_type: &str,
    card_content: &str,
) -> Result<String> {
    send_message(
        client,
        domain,
        token,
        receive_id,
        receive_id_type,
        "interactive",
        card_content,
    )
    .await
}

/// 回复消息（在话题中回复）
pub async fn reply_message(
    client: &Client,
    domain: &str,
    token: &str,
    message_id: &str,
    msg_type: &str,
    content: &str,
) -> Result<String> {
    let url = format!("{}/im/v1/messages/{}/reply", domain, message_id);

    let body = json!({
        "msg_type": msg_type,
        "content": content,
    });

    debug!("Feishu: replying to message {}", message_id);

    let res = client
        .post(&url)
        .header("Authorization", format!("Bearer {}", token))
        .header("Content-Type", "application/json")
        .json(&body)
        .send()
        .await
        .map_err(|e| {
            ChannelError::SendError(format!(
                "Reply request failed: {}",
                request_error(e, [token])
            ))
        })?;

    let status = res.status();
    if !status.is_success() {
        let text = res.text().await.unwrap_or_default();
        return Err(ReactError::Channel(Box::new(ChannelError::SendError(
            format!(
                "Reply failed (status {}): {}",
                status,
                text_with_secrets(&text, [token])
            ),
        ))));
    }

    let json: serde_json::Value = res.json().await.map_err(|e| {
        ChannelError::SendError(format!(
            "Reply response parse error: {}",
            request_error(e, [token])
        ))
    })?;

    let code = json["code"].as_i64().unwrap_or(-1);
    if code != 0 {
        let msg = json["msg"].as_str().unwrap_or("unknown error").to_string();
        return Err(ReactError::Channel(Box::new(ChannelError::SendError(
            format!(
                "Reply API error (code {}): {}",
                code,
                text_with_secrets(&msg, [token])
            ),
        ))));
    }

    let reply_id = json["data"]["message_id"]
        .as_str()
        .unwrap_or("")
        .to_string();

    Ok(reply_id)
}

/// 更新卡片消息
pub async fn patch_card_message(
    client: &Client,
    domain: &str,
    token: &str,
    message_id: &str,
    card_content: &str,
) -> Result<()> {
    let url = format!("{}{}{}", domain, "/im/v1/messages/", message_id);

    let body = json!({
        "content": card_content,
    });

    let res = client
        .patch(&url)
        .header("Authorization", format!("Bearer {}", token))
        .header("Content-Type", "application/json")
        .json(&body)
        .send()
        .await
        .map_err(|e| {
            ChannelError::SendError(format!(
                "Patch request failed: {}",
                request_error(e, [token])
            ))
        })?;

    let status = res.status();
    if !status.is_success() {
        let text = res.text().await.unwrap_or_default();
        return Err(ReactError::Channel(Box::new(ChannelError::SendError(
            format!(
                "Patch failed (status {}): {}",
                status,
                text_with_secrets(&text, [token])
            ),
        ))));
    }

    let json: serde_json::Value = res.json().await.map_err(|e| {
        ChannelError::SendError(format!(
            "Patch response parse error: {}",
            request_error(e, [token])
        ))
    })?;

    let code = json["code"].as_i64().unwrap_or(-1);
    if code != 0 {
        let msg = json["msg"].as_str().unwrap_or("unknown error").to_string();
        return Err(ReactError::Channel(Box::new(ChannelError::SendError(
            format!(
                "Patch API error (code {}): {}",
                code,
                text_with_secrets(&msg, [token])
            ),
        ))));
    }

    Ok(())
}

/// 为消息添加表情回应
pub async fn add_reaction(
    client: &Client,
    domain: &str,
    token: &str,
    message_id: &str,
    emoji_type: &str,
) -> Result<()> {
    let url = format!(
        "{}{}{}{}",
        domain, "/im/v1/messages/", message_id, "/reactions"
    );

    let body = json!({
        "reaction_type": {
            "emoji_type": emoji_type
        }
    });

    let res = client
        .post(&url)
        .header("Authorization", format!("Bearer {}", token))
        .header("Content-Type", "application/json")
        .json(&body)
        .send()
        .await
        .map_err(|e| {
            ChannelError::SendError(format!(
                "Reaction request failed: {}",
                request_error(e, [token])
            ))
        })?;

    if !res.status().is_success() {
        let text = res.text().await.unwrap_or_default();
        warn!(
            "Feishu: add reaction failed: {}",
            text_with_secrets(&text, [token])
        );
        // 不返回错误；表情回应失败不影响主流程
    }

    Ok(())
}
