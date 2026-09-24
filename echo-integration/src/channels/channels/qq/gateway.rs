//! QQ 机器人 WebSocket Gateway —— 接收消息。
//!
//! 连接到官方 QQ WebSocket Gateway，解析事件并转发给 MessageHandler。
//!
//! QQ 机器人 WebSocket 协议：
//! 1. 连接建立后，接收 HELLO 事件（包含 heartbeat_interval）
//! 2. 发送 IDENTIFY（包含 token、intents 等）
//! 3. 周期性发送 HEARTBEAT
//! 4. 接收消息事件（C2C_MESSAGE_CREATE、GROUP_AT_MESSAGE_CREATE 等）

use super::super::super::types::*;
use crate::redaction::text_with_secrets;
use echo_core::error::ChannelError;
use futures::SinkExt;
use futures::StreamExt;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::Message;
use tracing::{debug, info, warn};

/// QQ 机器人 WebSocket 操作码（Discord Gateway 协议）
const OP_DISPATCH: u32 = 0; // 事件消息
const OP_HEARTBEAT: u32 = 1; // 心跳请求（客户端发送）
const OP_IDENTIFY: u32 = 2; // 鉴权（客户端发送）
const OP_RECONNECT: u32 = 7; // 重连请求（服务端发送）
const OP_HELLO: u32 = 10; // Hello（服务端发送，包含心跳间隔）
const OP_HEARTBEAT_ACK: u32 = 11; // 心跳确认（服务端发送）

/// QQ 机器人 intents（事件订阅位掩码）
/// 参考：https://bot.q.qq.com/wiki/develop/api-v2/dev-prepare/interface-framework/event-emit.html
/// 重要：GROUP_AND_C2C_EVENT（1<<25）包含 QQ 私聊与群 @ 消息事件
const INTENT_GUILDS: u32 = 1 << 0; // 频道事件（默认权限）
const INTENT_GUILD_MEMBERS: u32 = 1 << 1; // 频道成员事件（默认权限）
const INTENT_GROUP_AND_C2C_EVENT: u32 = 1 << 25; // QQ 私聊 + 群聊事件（关键！包含 C2C_MESSAGE_CREATE）

/// 连接到 QQ Gateway，完成鉴权、心跳维护并持续处理接收到的事件消息，直至连接关闭或出错。
pub(super) async fn connect_to_gateway(
    wss_url: String,
    handler: Arc<dyn MessageHandler>,
    token: String,
) -> std::result::Result<(), ChannelError> {
    let (ws_stream, _) = connect_async(wss_url.clone()).await.map_err(|e| {
        ChannelError::ConnectionError(format!(
            "WebSocket connect failed: {}",
            text_with_secrets(&e.to_string(), [token.as_str()])
        ))
    })?;

    info!("QQ Gateway: WebSocket connected");

    // 将 WebSocket 拆分为发送端和接收端
    let (ws_sender, mut ws_receiver) = ws_stream.split();

    // 共享状态
    let last_seq = Arc::new(Mutex::new(0u64));
    let heartbeat_interval = Arc::new(Mutex::new(Duration::from_secs(30)));
    let identified = Arc::new(Mutex::new(false));

    // 启动心跳任务
    let last_seq_clone = last_seq.clone();
    let heartbeat_interval_clone = heartbeat_interval.clone();
    let identified_clone = identified.clone();
    let ws_sender_clone = Arc::new(Mutex::new(ws_sender));
    let ws_sender_for_heartbeat = ws_sender_clone.clone();
    let token_for_heartbeat = token.clone();

    let heartbeat_task = tokio::spawn(async move {
        // 等待 IDENTIFY 完成
        while !*identified_clone.lock().await {
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        // 启动心跳循环
        let interval = *heartbeat_interval_clone.lock().await;
        let mut timer = tokio::time::interval(interval);

        loop {
            timer.tick().await;

            let seq = *last_seq_clone.lock().await;
            let heartbeat = serde_json::json!({
                "op": OP_HEARTBEAT,
                "d": seq
            });

            let mut sender = ws_sender_for_heartbeat.lock().await;
            if let Err(e) = sender.send(Message::Text(heartbeat.to_string())).await {
                warn!(
                    "QQ Gateway: heartbeat failed: {}",
                    text_with_secrets(&e.to_string(), [token_for_heartbeat.as_str()])
                );
                break;
            }
            debug!("QQ Gateway: heartbeat sent (seq={})", seq);
        }
    });

    // 处理消息循环
    loop {
        match ws_receiver.next().await {
            Some(Ok(Message::Text(text))) => {
                let redacted = text_with_secrets(&text, [token.as_str()]);
                debug!(
                    "QQ Gateway: received message: {}",
                    redacted.chars().take(200).collect::<String>()
                );

                let payload: serde_json::Value = serde_json::from_str(&text).map_err(|e| {
                    ChannelError::Other(format!("QQ Gateway: failed to parse event: {}", e))
                })?;

                let op = payload["op"].as_u64().unwrap_or(0);
                {
                    let mut seq = last_seq.lock().await;
                    *seq = payload["s"].as_u64().unwrap_or(*seq);
                }

                match op as u32 {
                    OP_HELLO => {
                        // HELLO：获取心跳间隔并发送 IDENTIFY
                        let interval_ms =
                            payload["d"]["heartbeat_interval"].as_u64().unwrap_or(30000);

                        info!(
                            "QQ Gateway: HELLO received, heartbeat_interval={}ms",
                            interval_ms
                        );

                        {
                            let mut interval = heartbeat_interval.lock().await;
                            *interval = Duration::from_millis(interval_ms);
                        }

                        // 发送 IDENTIFY
                        let identify = serde_json::json!({
                            "op": OP_IDENTIFY,
                            "d": {
                                "token": format!("QQBot {}", token),
                                "intents": INTENT_GUILDS | INTENT_GUILD_MEMBERS | INTENT_GROUP_AND_C2C_EVENT,
                                "shard": [0, 1],
                            }
                        });

                        {
                            let mut sender = ws_sender_clone.lock().await;
                            sender
                                .send(Message::Text(identify.to_string()))
                                .await
                                .map_err(|e| {
                                    ChannelError::ConnectionError(format!(
                                        "Failed to send IDENTIFY: {}",
                                        text_with_secrets(&e.to_string(), [token.as_str()])
                                    ))
                                })?;
                        }

                        info!("QQ Gateway: IDENTIFY sent");
                        {
                            let mut id = identified.lock().await;
                            *id = true;
                        }
                    }
                    OP_HEARTBEAT_ACK => {
                        debug!("QQ Gateway: heartbeat ACK received");
                    }
                    OP_RECONNECT => {
                        warn!("QQ Gateway: RECONNECT requested");
                        heartbeat_task.abort();
                        return Err(ChannelError::ConnectionError(
                            "Server requested reconnect".to_string(),
                        ));
                    }
                    OP_DISPATCH => {
                        // 事件消息（op=0 表示分发事件）
                        if *identified.lock().await
                            && let Err(e) =
                                handle_gateway_event(handler.clone(), &payload, &token).await
                        {
                            warn!(
                                "QQ Gateway: failed to handle event: {}",
                                text_with_secrets(&e.to_string(), [token.as_str()])
                            );
                        }
                    }
                    _ => {
                        debug!("QQ Gateway: unknown op code: {}", op);
                    }
                }
            }
            Some(Ok(Message::Ping(payload))) => {
                debug!("QQ Gateway: received PING, sending PONG");
                let mut sender = ws_sender_clone.lock().await;
                if let Err(e) = sender.send(Message::Pong(payload)).await {
                    warn!(
                        "QQ Gateway: failed to send PONG: {}",
                        text_with_secrets(&e.to_string(), [token.as_str()])
                    );
                }
            }
            Some(Ok(Message::Close(_))) => {
                info!("QQ Gateway: received close frame");
                heartbeat_task.abort();
                return Ok(());
            }
            Some(Ok(_)) => {}
            Some(Err(e)) => {
                warn!(
                    "QQ Gateway: WebSocket read error: {}",
                    text_with_secrets(&e.to_string(), [token.as_str()])
                );
                heartbeat_task.abort();
                return Err(ChannelError::ConnectionError(format!(
                    "WebSocket read error: {}",
                    text_with_secrets(&e.to_string(), [token.as_str()])
                )));
            }
            None => {
                info!("QQ Gateway: WebSocket stream ended");
                heartbeat_task.abort();
                return Ok(());
            }
        }
    }
}

/// 解析并处理 Gateway 事件（op=0）
async fn handle_gateway_event(
    handler: Arc<dyn MessageHandler>,
    payload: &serde_json::Value,
    access_token: &str,
) -> std::result::Result<(), ChannelError> {
    let event_type = payload["t"].as_str().unwrap_or("");

    debug!("QQ Gateway: event_type={}", event_type);

    match event_type {
        "READY" => {
            info!("QQ Gateway: READY event received, session established");
        }
        "RESUMED" => {
            info!("QQ Gateway: RESUMED event received");
        }
        "C2C_MESSAGE_CREATE" | "C2C_MESSAGE_CREATE_WITH_INTENT" => {
            handle_c2c_message(handler, payload, access_token).await?;
        }
        "GROUP_AT_MESSAGE_CREATE" | "AT_MESSAGE_CREATE" => {
            handle_group_at_message(handler, payload, access_token).await?;
        }
        _ => {
            // 忽略其他事件
        }
    }

    Ok(())
}

/// 处理私聊消息
async fn handle_c2c_message(
    handler: Arc<dyn MessageHandler>,
    payload: &serde_json::Value,
    access_token: &str,
) -> std::result::Result<(), ChannelError> {
    let data = &payload["d"];

    let sender_id = require_sender_id(data["author"]["id"].as_str())?;
    let text = data["content"].as_str().unwrap_or("").to_string();
    let message_id = data["id"].as_str().unwrap_or("").to_string();

    if text.is_empty() {
        return Ok(());
    }

    let inbound = InboundMessage::new(
        "qqbot",
        sender_id,
        sender_id,
        ChatType::Direct,
        &text,
        &message_id,
    );

    dispatch_to_handler(handler, inbound, access_token).await
}

/// 处理群 @ 消息
async fn handle_group_at_message(
    handler: Arc<dyn MessageHandler>,
    payload: &serde_json::Value,
    access_token: &str,
) -> std::result::Result<(), ChannelError> {
    let data = &payload["d"];

    let sender_id = require_sender_id(data["author"]["id"].as_str())?;
    // 在 QQ 机器人 v2 API 中群聊 ID 字段是 group_openid；同时兼容旧格式的 group.id
    let group_id = require_conversation_id(
        data["group_openid"]
            .as_str()
            .or_else(|| data["group"]["id"].as_str()),
    )?;
    let text = data["content"].as_str().unwrap_or("").to_string();
    let message_id = data["id"].as_str().unwrap_or("").to_string();

    if text.is_empty() {
        return Ok(());
    }

    let inbound = InboundMessage::new(
        "qqbot",
        sender_id,
        group_id,
        ChatType::Group,
        &text,
        &message_id,
    );

    dispatch_to_handler(handler, inbound, access_token).await
}

/// 统一分发给 Handler 并发送回复
async fn dispatch_to_handler(
    handler: Arc<dyn MessageHandler>,
    inbound: InboundMessage,
    access_token: &str,
) -> std::result::Result<(), ChannelError> {
    match handler.handle(inbound).await {
        Ok(outbound) => {
            // 使用 chars() 安全地截断 UTF-8 字符串
            let redacted = text_with_secrets(&outbound.text, [access_token]);
            let mut characters = redacted.chars();
            let text_preview: String = characters.by_ref().take(50).collect();
            let truncated = characters.next().is_some();
            info!(
                "Handler returned outbound: to={}, text={}",
                outbound.to,
                if truncated {
                    format!("{}...", text_preview)
                } else {
                    text_preview
                }
            );

            // 将回复发送给 QQ
            if let Err(e) = handler.reply(outbound).await {
                warn!(
                    "Failed to send reply: {}",
                    text_with_secrets(&e.to_string(), [access_token])
                );
            }

            Ok(())
        }
        Err(e) => {
            let error = text_with_secrets(&e.to_string(), [access_token]);
            warn!("Handler error: {}", error);
            Err(ChannelError::Other(format!("Handler error: {error}")))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use serde_json::json;
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
    async fn missing_or_unknown_sender_never_reaches_qq_handler() -> Result<(), String> {
        let calls = Arc::new(AtomicUsize::new(0));
        let handler: Arc<dyn MessageHandler> = Arc::new(CountingHandler {
            calls: calls.clone(),
        });
        let missing = json!({"d": {"content": "hello", "id": "m1"}});
        let unknown = json!({
            "d": {
                "author": {"id": "unknown"},
                "group_openid": "group-1",
                "content": "hello",
                "id": "m2"
            }
        });

        let direct = handle_c2c_message(handler.clone(), &missing, "test-token").await;
        let group = handle_group_at_message(handler, &unknown, "test-token").await;
        if !matches!(direct, Err(ChannelError::Other(ref message)) if message.contains("sender_id"))
            || !matches!(group, Err(ChannelError::Other(ref message)) if message.contains("sender_id"))
        {
            return Err("QQ did not return typed sender identity errors".to_string());
        }
        if calls.load(Ordering::Acquire) != 0 {
            return Err("QQ forwarded a message without a stable sender".to_string());
        }
        Ok(())
    }
}
