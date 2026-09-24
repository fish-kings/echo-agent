//! 飞书 Webhook 服务器
//!
//! HTTP 事件推送模式；需要公网 IP。
//!
//! 用于接收飞书推送的事件消息。

use super::super::super::types::*;
use axum::response::IntoResponse;
use axum::{Router, body::Bytes, extract::State, routing::post};
use dashmap::DashMap;
use hmac::{Hmac, Mac};
use serde_json::json;
use sha2::Sha256;
use std::sync::Arc;
use std::time::{Duration, Instant};
use subtle::ConstantTimeEq;
use tokio::net::TcpListener;
use tokio::sync::Mutex;
use tracing::{debug, info, warn};

/// 飞书事件类型
const FEISHU_IM_MESSAGE_RECEIVE: &str = "im.message.receive_v1";

/// 已处理事件去重缓存的 TTL
const DEDUP_TTL_SECS: u64 = 300;

type HmacSha256 = Hmac<Sha256>;

/// 校验飞书 Webhook 签名（使用常数时间比较以防止时序攻击）。
///
/// 飞书签名算法：
/// 1. 拼接 `timestamp + "\n" + nonce + "\n" + body`
/// 2. 使用 signing_key 计算 HMAC-SHA256
/// 3. Base64 编码得到签名
/// 4. 将 `X-Lark-Signature` 请求头与计算值进行比较
fn verify_feishu_signature(
    signing_key: &str,
    timestamp: &str,
    nonce: &str,
    body: &[u8],
    expected_signature: &str,
) -> bool {
    let Ok(mut mac) = HmacSha256::new_from_slice(signing_key.as_bytes()) else {
        return false;
    };
    mac.update(timestamp.as_bytes());
    mac.update(b"\n");
    mac.update(nonce.as_bytes());
    mac.update(b"\n");
    mac.update(body);
    let computed = mac.finalize().into_bytes();

    // 对计算结果进行 Base64 编码
    let computed_b64 = base64::Engine::encode(&base64::engine::general_purpose::STANDARD, computed);

    // 常数时间比较以防止时序攻击
    if computed_b64.len() != expected_signature.len() {
        return false;
    }
    computed_b64
        .as_bytes()
        .ct_eq(expected_signature.as_bytes())
        .into()
}

/// Webhook 状态
struct WebhookState {
    /// 消息处理器。
    handler: Arc<dyn MessageHandler>,
    /// 飞书事件订阅校验 token。
    verification_token: Option<String>,
    /// 签名密钥（用于 HMAC 签名验证）
    signing_key: Option<String>,
    /// 已处理事件去重缓存（message_id -> 处理时间）
    processed_events: DashMap<String, Instant>,
    /// 按 message_id 序列化并发事件处理的互斥锁集合。
    event_locks: DashMap<String, Arc<Mutex<()>>>,
}

/// 构造一个指定状态码、空 body 的 HTTP 响应。
fn empty_response(status: axum::http::StatusCode) -> axum::response::Response {
    let mut response = axum::response::Response::new(axum::body::Body::empty());
    *response.status_mut() = status;
    response
}

/// 处理飞书事件
async fn handle_event(
    State(state): State<Arc<WebhookState>>,
    headers: axum::http::HeaderMap,
    raw_body: Bytes,
) -> axum::response::Response {
    // 签名校验以原始 HTTP payload 为准，需在 JSON 规范化之前进行。
    if let Some(ref signing_key) = state.signing_key {
        let timestamp = headers
            .get("X-Lark-Request-Timestamp")
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default();
        let nonce = headers
            .get("X-Lark-Request-Nonce")
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default();
        let signature = headers
            .get("X-Lark-Signature")
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default();
        if !verify_feishu_signature(signing_key, timestamp, nonce, &raw_body, signature) {
            warn!("Feishu webhook: signature verification failed");
            return empty_response(axum::http::StatusCode::UNAUTHORIZED);
        }
    }
    let body: serde_json::Value = match serde_json::from_slice(&raw_body) {
        Ok(body) => body,
        Err(error) => {
            warn!(%error, "Feishu webhook: invalid JSON body");
            return empty_response(axum::http::StatusCode::BAD_REQUEST);
        }
    };
    // 1. challenge 验证
    if let Some(challenge) = body.get("challenge").and_then(|v| v.as_str()) {
        info!("Feishu webhook: responding to challenge verification");
        return axum::Json(json!({ "challenge": challenge })).into_response();
    }

    // 2. 校验 verification_token
    if let Some(ref expected_token) = state.verification_token {
        let actual_token = body.get("header").and_then(|h| h["token"].as_str());
        if actual_token != Some(expected_token.as_str()) {
            warn!("Feishu webhook: verification_token mismatch");
            return empty_response(axum::http::StatusCode::UNAUTHORIZED);
        }
    }

    // 4. 校验事件类型
    let event_type = body["header"]["event_type"].as_str();
    if event_type != Some(FEISHU_IM_MESSAGE_RECEIVE) {
        debug!("Feishu webhook: ignoring event type: {:?}", event_type);
        return axum::Json(json!({})).into_response();
    }

    // 5. 解析消息
    let event = &body["event"];
    let message = &event["message"];

    if message.is_null() {
        return axum::Json(json!({})).into_response();
    }

    let message_id = message["message_id"].as_str().unwrap_or("").to_string();

    // 6. 对同一事件 ID 的重复投递做串行化处理。
    let event_lock = if message_id.is_empty() {
        None
    } else {
        Some(
            state
                .event_locks
                .entry(message_id.clone())
                .or_insert_with(|| Arc::new(Mutex::new(())))
                .clone(),
        )
    };
    let _event_guard = if let Some(lock) = &event_lock {
        Some(lock.lock().await)
    } else {
        None
    };
    if !message_id.is_empty() {
        if state.processed_events.contains_key(&message_id) {
            debug!(
                "Feishu webhook: duplicate event, message_id={}, skipping",
                message_id
            );
            return axum::Json(json!({})).into_response();
        }
        let ttl = Duration::from_secs(DEDUP_TTL_SECS);
        state.processed_events.retain(|_, v| v.elapsed() < ttl);
    }

    let message_type = message["message_type"].as_str().unwrap_or("").to_string();
    let chat_id = match require_conversation_id(message["chat_id"].as_str()) {
        Ok(chat_id) => chat_id,
        Err(error) => {
            state.event_locks.remove(&message_id);
            warn!(%error, "Feishu webhook: invalid conversation identity");
            return empty_response(axum::http::StatusCode::BAD_REQUEST);
        }
    };
    let chat_type_str = message["chat_type"].as_str().unwrap_or("p2p").to_string();

    // 仅处理文本消息
    if message_type != "text" {
        debug!(
            "Feishu webhook: ignoring non-text message: {}",
            message_type
        );
        return axum::Json(json!({})).into_response();
    }

    // 解析文本内容
    let content_str = message["content"].as_str().unwrap_or("{}").to_string();
    let content: serde_json::Value = serde_json::from_str(&content_str).unwrap_or_default();
    let text = content["text"].as_str().unwrap_or("").to_string();

    if text.is_empty() {
        return axum::Json(json!({})).into_response();
    }

    let sender_id = match super::sender_scope(
        event["sender"]["sender_id"]["open_id"]
            .as_str()
            .filter(|value| !value.is_empty()),
        event["sender"]["sender_id"]["user_id"]
            .as_str()
            .filter(|value| !value.is_empty()),
    ) {
        Ok(sender_id) => sender_id,
        Err(error) => {
            state.event_locks.remove(&message_id);
            warn!(%error, "Feishu webhook: invalid sender identity");
            return empty_response(axum::http::StatusCode::BAD_REQUEST);
        }
    };

    // HTTP 响应成功即代表 handler 处理与消息投递均已完成。
    let handler = state.handler.clone();
    let chat_type = if chat_type_str == "group" {
        ChatType::Group
    } else {
        ChatType::Direct
    };
    let inbound = InboundMessage::new(
        "feishu",
        sender_id,
        chat_id,
        chat_type,
        text,
        message_id.clone(),
    );
    let delivered = match handler.handle(inbound).await {
        Ok(outbound) => handler.reply(outbound).await,
        Err(error) => Err(error),
    };
    if let Err(error) = delivered {
        state.event_locks.remove(&message_id);
        warn!(
            error = %crate::redaction::text_with_secrets(
                &error.to_string(),
                state
                    .verification_token
                    .as_deref()
                    .into_iter()
                    .chain(state.signing_key.as_deref()),
            ),
            "Feishu webhook: processing failed; returning retryable status"
        );
        return empty_response(axum::http::StatusCode::INTERNAL_SERVER_ERROR);
    }
    if !message_id.is_empty() {
        state
            .processed_events
            .insert(message_id.clone(), Instant::now());
        state.event_locks.remove(&message_id);
    }

    axum::Json(json!({})).into_response()
}

/// 启动 Webhook 服务器
pub(super) async fn run_webhook_server(
    bind_addr: String,
    webhook_path: String,
    handler: Arc<dyn MessageHandler>,
    verification_token: Option<String>,
    signing_key: Option<String>,
) -> Result<(), echo_core::error::ChannelError> {
    let state = Arc::new(WebhookState {
        handler,
        verification_token,
        signing_key,
        processed_events: DashMap::new(),
        event_locks: DashMap::new(),
    });

    let app = Router::new()
        .route(&webhook_path, post(handle_event))
        .layer(axum::extract::DefaultBodyLimit::max(2 * 1024 * 1024))
        .with_state(state);

    info!(
        "Feishu webhook server listening on {}{}",
        bind_addr, webhook_path
    );

    let listener = TcpListener::bind(&bind_addr).await.map_err(|e| {
        echo_core::error::ChannelError::ConnectionError(format!(
            "Failed to bind webhook server: {}",
            e
        ))
    })?;

    axum::serve(listener, app).await.map_err(|e| {
        echo_core::error::ChannelError::ConnectionError(format!("Webhook server error: {}", e))
    })?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct CountingHandler {
        calls: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl MessageHandler for CountingHandler {
        async fn handle(&self, msg: InboundMessage) -> echo_core::error::Result<OutboundMessage> {
            self.calls.fetch_add(1, Ordering::AcqRel);
            Ok(OutboundMessage::new(
                &msg.channel_id,
                msg.reply_target(),
                msg.chat_type,
                "ok",
            ))
        }

        async fn reply(&self, _msg: OutboundMessage) -> echo_core::error::Result<()> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn missing_sender_never_reaches_feishu_webhook_handler() -> Result<(), String> {
        let calls = Arc::new(AtomicUsize::new(0));
        let state = Arc::new(WebhookState {
            handler: Arc::new(CountingHandler {
                calls: calls.clone(),
            }),
            verification_token: None,
            signing_key: None,
            processed_events: DashMap::new(),
            event_locks: DashMap::new(),
        });
        let body = serde_json::to_vec(&json!({
            "header": {"event_type": FEISHU_IM_MESSAGE_RECEIVE},
            "event": {
                "sender": {"sender_id": {}},
                "message": {
                    "message_id": "m1",
                    "message_type": "text",
                    "chat_id": "group-1",
                    "chat_type": "group",
                    "content": "{\"text\":\"hello\"}"
                }
            }
        }))
        .map_err(|error| error.to_string())?;

        let response = handle_event(
            State(state.clone()),
            axum::http::HeaderMap::new(),
            Bytes::from(body),
        )
        .await;
        if response.status() != axum::http::StatusCode::BAD_REQUEST {
            return Err(format!(
                "Feishu webhook returned {} for missing sender",
                response.status()
            ));
        }
        if calls.load(Ordering::Acquire) != 0 || !state.event_locks.is_empty() {
            return Err("Feishu webhook retained or forwarded an invalid event".to_string());
        }
        Ok(())
    }

    #[tokio::test]
    async fn user_id_fallback_reaches_feishu_webhook_handler() -> Result<(), String> {
        let calls = Arc::new(AtomicUsize::new(0));
        let state = Arc::new(WebhookState {
            handler: Arc::new(CountingHandler {
                calls: calls.clone(),
            }),
            verification_token: None,
            signing_key: None,
            processed_events: DashMap::new(),
            event_locks: DashMap::new(),
        });
        let body = serde_json::to_vec(&json!({
            "header": {"event_type": FEISHU_IM_MESSAGE_RECEIVE},
            "event": {
                "sender": {"sender_id": {"user_id": "user-1"}},
                "message": {
                    "message_id": "m1",
                    "message_type": "text",
                    "chat_id": "group-1",
                    "chat_type": "group",
                    "content": "{\"text\":\"hello\"}"
                }
            }
        }))
        .map_err(|error| error.to_string())?;

        let response = handle_event(
            State(state.clone()),
            axum::http::HeaderMap::new(),
            Bytes::from(body),
        )
        .await;
        if response.status() != axum::http::StatusCode::OK {
            return Err(format!(
                "Feishu webhook rejected user_id fallback with {}",
                response.status()
            ));
        }
        if calls.load(Ordering::Acquire) != 1
            || !state.event_locks.is_empty()
            || !state.processed_events.contains_key("m1")
        {
            return Err("Feishu webhook did not complete the user_id event".to_string());
        }
        Ok(())
    }
}
