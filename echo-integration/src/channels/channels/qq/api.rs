/*!
QQ 机器人 HTTP API —— 获取 Token + 发送消息

官方 API 文档：
- 获取 Token：POST <https://bots.qq.com/app/getAppAccessToken>
- 网关（Gateway）：GET <https://api.sgroup.qq.com/v2/gateway>
- 私聊消息：POST `https://api.sgroup.qq.com/v2/users/{openid}/messages`
- 群消息：POST `https://api.sgroup.qq.com/v2/groups/{guild_id}/messages`
*/

use super::super::super::types::ChatType;
use crate::redaction::{json as redact_json, json_with_secrets, request_error, text_with_secrets};
use echo_core::error::{ChannelError, ReactError, Result};
use reqwest::Client;
use serde_json::json;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::Mutex;
use tracing::{debug, info, warn};

/// QQ 机器人开放平台 API 基础地址
const QQ_API_BASE: &str = "https://api.sgroup.qq.com";
/// 获取 AppAccessToken 的接口地址
const QQ_TOKEN_URL: &str = "https://bots.qq.com/app/getAppAccessToken";

/// 构造一个共享使用的 `reqwest::Client`；若自定义构建失败则退回默认客户端。
pub fn reqwest_client() -> Client {
    Client::builder().build().unwrap_or_else(|_| Client::new())
}

// ── Token 管理 ────────────────────────────────────────────────────────────

/// QQ Token 管理器 —— 获取并缓存 access_token
pub(super) struct TokenManager {
    /// QQ 机器人应用 ID。
    app_id: String,
    /// QQ 机器人应用密钥。
    client_secret: String,
    /// 缓存的 access_token。
    token: Arc<Mutex<Option<String>>>,
    /// token 的过期时间（Unix 秒），用于判断是否需要刷新。
    expires_at: Arc<AtomicU64>,
    /// 防止并发重复刷新
    refresh_lock: Arc<Mutex<()>>,
    /// 用于请求 QQ 开放平台接口的 HTTP 客户端。
    http: reqwest::Client,
}

impl TokenManager {
    /// 使用给定的 `app_id` 与 `client_secret` 创建一个新的 Token 管理器。
    pub fn new(app_id: String, client_secret: String) -> Self {
        Self {
            app_id,
            client_secret,
            token: Arc::new(Mutex::new(None)),
            expires_at: Arc::new(AtomicU64::new(0)),
            refresh_lock: Arc::new(Mutex::new(())),
            http: reqwest_client(),
        }
    }

    /// 判断当前缓存的 token 是否已过期。
    async fn is_expired(&self) -> bool {
        let now = chrono::Utc::now().timestamp() as u64;
        self.expires_at.load(Ordering::Relaxed) <= now
    }

    /// 若缓存的 token 未过期，返回其克隆；否则返回 `None`。
    async fn get_cached(&self) -> Option<String> {
        if self.is_expired().await {
            return None;
        }
        self.token.lock().await.clone()
    }

    /// 获取 access_token（自动缓存 / 刷新）
    pub async fn get_token(&self) -> Result<String> {
        if let Some(token) = self.get_cached().await {
            return Ok(token);
        }
        // 加锁以防止并发重复刷新
        let _lock = self.refresh_lock.lock().await;
        // 获取锁后再次检查缓存
        if let Some(token) = self.get_cached().await {
            return Ok(token);
        }
        self.refresh_token().await
    }

    /// 向 QQ 开放平台请求新的 access_token，并更新本地缓存与过期时间。
    async fn refresh_token(&self) -> Result<String> {
        info!("QQ Bot: refreshing access_token");

        let body = json!({
            "appId": self.app_id,
            "clientSecret": self.client_secret,
        });

        let res = self
            .http
            .post(QQ_TOKEN_URL)
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                ChannelError::NetworkError(format!(
                    "QQ token request failed: {}",
                    request_error(e, [&self.client_secret])
                ))
            })?;

        if !res.status().is_success() {
            return Err(ReactError::Channel(Box::new(ChannelError::AuthError(
                format!("QQ token request failed with status {}", res.status()),
            ))));
        }

        let json: serde_json::Value = res.json().await.map_err(|e| {
            ChannelError::NetworkError(format!(
                "QQ token response parse error: {}",
                request_error(e, [&self.client_secret])
            ))
        })?;

        let redacted_response =
            text_with_secrets(&redact_json(&json).to_string(), [&self.client_secret]);
        debug!("QQ Bot: token response = {}", redacted_response);

        // QQ API 返回的字段可能是 access_token（snake_case）也可能是 accessToken（camelCase）
        let token = json
            .get("accessToken")
            .or_else(|| json.get("access_token"))
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                let response =
                    text_with_secrets(&redact_json(&json).to_string(), [&self.client_secret]);
                let err_msg = format!("QQ token response missing accessToken, got: {response}");
                ReactError::Channel(Box::new(ChannelError::AuthError(err_msg)))
            })?
            .to_string();

        /*
        QQ Token 有效期通常为 7200 秒；提前 5 分钟刷新
        注意：expires_in 可能是字符串 "7200"，也可能是数字 7200
        */
        let expires_in = json
            .get("expiresIn")
            .or_else(|| json.get("expires_in"))
            .and_then(|v| {
                v.as_u64()
                    .or_else(|| v.as_str().and_then(|s| s.parse::<u64>().ok()))
            })
            .unwrap_or(7200);
        let now = u64::try_from(chrono::Utc::now().timestamp()).unwrap_or_default();
        let refresh_at = now.saturating_add(expires_in.saturating_sub(300));
        self.expires_at.store(refresh_at, Ordering::Relaxed);

        let mut tok = self.token.lock().await;
        *tok = Some(token.clone());

        info!("QQ Bot: token refreshed, expires in {}s", expires_in);
        Ok(token)
    }
}

// ── 网关 URL ───────────────────────────────────────────────────────────────

/// 获取 WebSocket Gateway 地址
pub async fn get_gateway_url(client: &reqwest::Client, token: &str) -> Result<String> {
    // QQ 机器人 API v2 的 gateway 端点：依次尝试多个路径
    let endpoints = [
        format!("{}/gateway", QQ_API_BASE),        // 无版本前缀
        format!("{}/gateway/bot", QQ_API_BASE),    // bot 后缀
        format!("{}/v2/gateway", QQ_API_BASE),     // v2 前缀
        format!("{}/v2/gateway/bot", QQ_API_BASE), // v2 + bot
    ];

    for endpoint in &endpoints {
        debug!("QQ Bot: trying gateway endpoint: {}", endpoint);

        let res = client
            .get(endpoint)
            .header("Authorization", format!("QQBot {}", token))
            .send()
            .await
            .map_err(|e| {
                ChannelError::NetworkError(format!(
                    "QQ gateway request failed: {}",
                    request_error(e, [token])
                ))
            })?;

        let status = res.status();
        if status.is_success() {
            let json: serde_json::Value = res.json().await.map_err(|e| {
                ChannelError::NetworkError(format!(
                    "QQ gateway response parse error: {}",
                    request_error(e, [token])
                ))
            })?;

            debug!("QQ Bot: gateway response received");

            let url = json
                .get("url")
                .or_else(|| json.get("gateway_url"))
                .and_then(|v| v.as_str())
                .ok_or_else(|| {
                    ReactError::Channel(Box::new(ChannelError::NetworkError(format!(
                        "QQ gateway response missing url, got: {:?}",
                        json_with_secrets(&json, &[token.to_string()])
                    ))))
                })?
                .to_string();

            info!("QQ Bot: gateway URL obtained from {}", endpoint);
            return Ok(url);
        }

        let error_body = res.text().await.unwrap_or_default();
        debug!(
            "QQ Gateway: endpoint {} returned status {}, body: {}",
            endpoint,
            status,
            text_with_secrets(&error_body, [token])
        );
    }

    Err(ReactError::Channel(Box::new(ChannelError::ApiError {
        status: 404,
        message: "QQ gateway: all endpoints failed".to_string(),
    })))
}

// ── 发送消息 ──────────────────────────────────────────────────────────────

/// 发送一条 QQ 消息
pub async fn send_qq_message(
    client: &reqwest::Client,
    token: &str,
    to: &str,
    chat_type: &ChatType,
    text: &str,
    reply_to: Option<&str>,
) -> Result<()> {
    let auth = format!("QQBot {}", token);

    let body = if let Some(msg_id) = reply_to {
        json!({
            "content": text,
            "msg_type": 0,
            "msg_id": msg_id
        })
    } else {
        json!({
            "content": text,
            "msg_type": 0
        })
    };

    let url = match chat_type {
        ChatType::Direct => {
            format!("{}/v2/users/{}/messages", QQ_API_BASE, to)
        }
        ChatType::Group => {
            format!("{}/v2/groups/{}/messages", QQ_API_BASE, to)
        }
    };

    let res = client
        .post(&url)
        .header("Authorization", &auth)
        .header("Content-Type", "application/json")
        .json(&body)
        .send()
        .await
        .map_err(|e| {
            ChannelError::SendError(format!("QQ send failed: {}", request_error(e, [token])))
        })?;

    let status = res.status();
    if !status.is_success() {
        let error_text = res.text().await.unwrap_or_default();
        let error_text = text_with_secrets(&error_text, [token]);
        warn!("QQ message send failed (status {}): {}", status, error_text);
        return Err(ReactError::Channel(Box::new(ChannelError::SendError(
            format!("QQ message send failed (status {}): {}", status, error_text),
        ))));
    }

    debug!("QQ message sent to {} ({:?})", to, chat_type);
    Ok(())
}
