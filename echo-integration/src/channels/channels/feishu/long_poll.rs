/*!
飞书 WebSocket 长连接

实现飞书长连接协议（PBBP2），无需公网 IP。

流程：
1. 通过 HTTP 获取 WebSocket URL（包含 device_id 和 service_id）
2. 连接 WebSocket，接收二进制 Protobuf 帧
3. 处理控制帧（ping/pong）和数据帧（事件）
4. 大消息会被自动分片，需要重新组装
5. 处理完事件后返回响应帧
*/

use super::super::super::types::*;
use super::api::{ClientConfig, get_ws_endpoint, http_client};
use super::proto::*;
use crate::redaction::{text as redact_text, text_with_secrets, url as redact_url};
use dashmap::DashMap;
use echo_core::error::{ChannelError, Result};
use futures::SinkExt;
use futures::StreamExt;
use futures::stream::{SplitSink, SplitStream};
use rand::Rng;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::Message;
use tracing::{debug, error, info, warn};

/// 已处理事件缓存的 TTL（秒）——5 分钟内的重复事件会被丢弃
const DEDUP_TTL_SECS: u64 = 300;

/// 默认 ping 间隔（秒）
const DEFAULT_PING_INTERVAL: u64 = 120;

/// 默认重连间隔（秒）
const DEFAULT_RECONNECT_INTERVAL: u64 = 120;

/// 默认首次重连随机抖动上限（秒）
const DEFAULT_RECONNECT_NONCE: u64 = 30;

/// 默认重连次数（-1 表示不限）
const DEFAULT_RECONNECT_COUNT: i32 = -1;

/// 指数退避的最大延迟（秒）
const MAX_BACKOFF_SECS: u64 = 300;

/// 稳定连接阈值（秒）——连接持续超过该时间即视为稳定，并重置退避
const STABLE_THRESHOLD_SECS: u64 = 60;
/// 单条消息允许的最大分片数量
const MAX_FRAGMENT_COUNT: usize = 256;
/// 单条消息所有分片累计允许的最大字节数
const MAX_FRAGMENT_BYTES: usize = 2 * 1024 * 1024;

// 类型别名，用于简化复杂的类型签名
/// 底层 WebSocket 连接流
type WsConn =
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;
/// WebSocket 写半部
type WsSink = SplitSink<WsConn, Message>;
/// WebSocket 读半部
type WsRead = SplitStream<WsConn>;
/// 可在多个任务间共享的写半部
type SharedSink = Arc<Mutex<WsSink>>;
/// 按序号存放的分片列表（未到达的分片为 None）
type FragmentParts = Vec<Option<Vec<u8>>>;
/// 分片缓存条目：（创建时间，分片列表）
type FragmentCacheEntry = (Instant, FragmentParts);
/// 分片缓存：msg_id -> 分片缓存条目
type FragmentCache = Arc<DashMap<String, FragmentCacheEntry>>;

// ── WebSocket 客户端 ──────────────────────────────────────────────────────────

/// WebSocket 长连接客户端配置
pub struct WsClientConfig {
    /// 飞书应用 App ID
    pub app_id: String,
    /// 飞书应用 App Secret
    pub app_secret: String,
    /// 开放平台域名
    pub domain: String,
    /// 连接断开后是否自动重连
    pub auto_reconnect: bool,
    /// 心跳 ping 间隔
    pub ping_interval: Duration,
    /// 重连间隔
    pub reconnect_interval: Duration,
    /// 首次重连的随机抖动上限
    pub reconnect_nonce: Duration,
    /// 最大重连次数（-1 表示不限）
    pub reconnect_count: i32,
}

impl WsClientConfig {
    /// 使用默认的心跳与重连参数创建配置
    pub fn new(app_id: String, app_secret: String, domain: String) -> Self {
        Self {
            app_id,
            app_secret,
            domain,
            auto_reconnect: true,
            ping_interval: Duration::from_secs(DEFAULT_PING_INTERVAL),
            reconnect_interval: Duration::from_secs(DEFAULT_RECONNECT_INTERVAL),
            reconnect_nonce: Duration::from_secs(DEFAULT_RECONNECT_NONCE),
            reconnect_count: DEFAULT_RECONNECT_COUNT,
        }
    }

    /// 根据服务端下发的配置更新本地参数
    pub fn update_from_server(&mut self, config: &ClientConfig) {
        if let Some(v) = config.ping_interval {
            self.ping_interval = Duration::from_secs(v as u64);
        }
        if let Some(v) = config.reconnect_interval {
            self.reconnect_interval = Duration::from_secs(v as u64);
        }
        if let Some(v) = config.reconnect_nonce {
            self.reconnect_nonce = Duration::from_secs(v as u64);
        }
        if let Some(v) = config.reconnect_count {
            self.reconnect_count = v;
        }
    }
}

/// WebSocket 长连接客户端
pub struct WsClient {
    /// 客户端配置
    config: WsClientConfig,
    /// 用于获取连接地址的 HTTP 客户端
    http: reqwest::Client,
    /// 连接 ID（从 URL 中解析）
    conn_id: String,
    /// 服务 ID（发送帧时使用）
    service_id: i32,
    /// 消息分片缓存（msg_id -> (创建时间, 分片列表)）
    fragment_cache: FragmentCache,
    /// 已处理事件的去重缓存（事件 message_id -> 处理时间）
    processed_events: Arc<DashMap<String, Instant>>,
    /// 按事件 message_id 划分的处理锁，保证同一事件串行处理
    event_locks: Arc<DashMap<String, Arc<Mutex<()>>>>,
    /// 运行标志
    running: Arc<Mutex<bool>>,
}

impl WsClient {
    /// 根据配置创建客户端
    pub fn new(config: WsClientConfig) -> Self {
        Self {
            config,
            http: http_client(),
            conn_id: String::new(),
            service_id: 0,
            fragment_cache: Arc::new(DashMap::new()),
            processed_events: Arc::new(DashMap::new()),
            event_locks: Arc::new(DashMap::new()),
            running: Arc::new(Mutex::new(false)),
        }
    }

    /// 启动客户端（阻塞运行直至停止）
    pub async fn run(&mut self, handler: Arc<dyn MessageHandler>) -> Result<()> {
        *self.running.lock().await = true;

        let mut backoff_secs: u64 = 1;
        let mut attempts_since_stable: u32 = 0;

        loop {
            let connected_at = Instant::now();
            let connect_result = self.connect(handler.clone()).await;

            match connect_result {
                Ok(()) => {
                    info!("Feishu WebSocket: connection closed normally");
                }
                Err(e) => {
                    warn!(
                        "Feishu WebSocket: connection error: {}",
                        redact_text(&e.to_string())
                    );
                }
            }

            if !*self.running.lock().await {
                info!("Feishu WebSocket: stopping, no reconnect");
                break;
            }

            if !self.config.auto_reconnect {
                info!("Feishu WebSocket: auto_reconnect disabled, stopping");
                break;
            }

            // 判断连接是否稳定，决定是否重置退避
            let stable = connected_at.elapsed().as_secs() >= STABLE_THRESHOLD_SECS;
            if stable {
                backoff_secs = 1;
                attempts_since_stable = 0;
            } else {
                attempts_since_stable += 1;
            }

            // 重连逻辑
            if self.config.reconnect_count >= 0 {
                let max_attempts = self.config.reconnect_count as u32;
                if attempts_since_stable > max_attempts {
                    error!(
                        "Feishu WebSocket: failed to reconnect after {} attempts",
                        self.config.reconnect_count
                    );
                    break;
                }
            }

            // 指数退避等待
            self.do_backoff_sleep(backoff_secs).await;

            // 更新退避延迟（指数增长，上限为 MAX_BACKOFF_SECS）
            backoff_secs = (backoff_secs * 2).min(MAX_BACKOFF_SECS);
        }

        Ok(())
    }

    /// 指数退避等待（首次尝试时使用随机抖动）
    async fn do_backoff_sleep(&self, delay_secs: u64) {
        if delay_secs <= 1 {
            let jitter = rand_jitter(self.config.reconnect_nonce);
            if jitter > Duration::ZERO {
                tokio::time::sleep(jitter).await;
            }
        } else {
            info!(
                "Feishu WebSocket: backing off for {}s before reconnect",
                delay_secs
            );
            tokio::time::sleep(Duration::from_secs(delay_secs)).await;
        }
    }

    /// 停止客户端
    pub async fn stop(&mut self) {
        *self.running.lock().await = false;
    }

    /// 建立连接并处理消息
    async fn connect(&mut self, handler: Arc<dyn MessageHandler>) -> Result<()> {
        // 1. 获取 WebSocket URL
        let (ws_url, service_id_str, client_config) = get_ws_endpoint(
            &self.http,
            &self.config.domain,
            &self.config.app_id,
            &self.config.app_secret,
        )
        .await?;

        if let Some(config) = client_config {
            self.config.update_from_server(&config);
        }

        self.service_id = service_id_str.parse().unwrap_or(0);

        let parsed_url = url::Url::parse(&ws_url).map_err(|error| {
            ChannelError::InvalidConfig(format!("Invalid Feishu WebSocket URL: {error}"))
        })?;
        self.conn_id = parsed_url
            .query_pairs()
            .find(|(k, _)| k == "device_id")
            .map(|(_, v)| v.to_string())
            .unwrap_or_default();

        info!("Feishu WebSocket: connecting to {}", redact_url(&ws_url));

        // 2. 建立 WebSocket 连接，拆分读写两半
        let (ws_stream, _) = connect_async(&ws_url).await.map_err(|e| {
            ChannelError::ConnectionError(format!(
                "WebSocket connect failed: {}",
                redact_text(&e.to_string())
            ))
        })?;

        info!(
            "Feishu WebSocket: connected (conn_id present={}, service_id={})",
            !self.conn_id.is_empty(),
            self.service_id
        );

        let (write, read) = ws_stream.split();
        let ws_sink: SharedSink = Arc::new(Mutex::new(write));

        // 3. 启动心跳任务（实际发送 ping 帧）
        let ping_interval = self.config.ping_interval;
        let service_id = self.service_id;
        let running = self.running.clone();
        let sink_for_ping = ws_sink.clone();

        let ping_task = tokio::spawn(async move {
            loop {
                tokio::time::sleep(ping_interval).await;
                if !*running.lock().await {
                    break;
                }
                let ping_bytes = ProtoFrame::ping(service_id).encode_to_vec();
                let mut sink = sink_for_ping.lock().await;
                if let Err(e) = sink.send(Message::Binary(ping_bytes)).await {
                    warn!(
                        "Feishu WebSocket: ping failed: {}",
                        redact_text(&e.to_string())
                    );
                    break;
                }
                debug!("Feishu WebSocket: ping sent");
            }
        });

        // 4. 消息处理循环
        let result = self.message_loop(read, ws_sink, handler).await;
        ping_task.abort();
        result
    }

    /// 消息处理循环
    async fn message_loop(
        &mut self,
        mut stream: WsRead,
        sink: SharedSink,
        handler: Arc<dyn MessageHandler>,
    ) -> Result<()> {
        loop {
            match stream.next().await {
                Some(Ok(Message::Binary(bytes))) => {
                    self.handle_binary_frame(&sink, bytes, handler.clone())
                        .await?;
                }
                Some(Ok(Message::Ping(_))) => {
                    debug!("Feishu WebSocket: received raw ping");
                }
                Some(Ok(Message::Pong(_))) => {
                    debug!("Feishu WebSocket: received raw pong");
                }
                Some(Ok(Message::Close(_))) => {
                    info!("Feishu WebSocket: server closed connection");
                    return Ok(());
                }
                Some(Ok(Message::Text(text))) => {
                    warn!(
                        "Feishu WebSocket: unexpected text message: {}",
                        text_with_secrets(&text, [&self.config.app_secret])
                    );
                }
                Some(Ok(Message::Frame(_))) => {}
                Some(Err(e)) => {
                    warn!(
                        "Feishu WebSocket: read error: {}",
                        redact_text(&e.to_string())
                    );
                    return Err(ChannelError::ConnectionError(format!(
                        "WebSocket error: {}",
                        redact_text(&e.to_string())
                    ))
                    .into());
                }
                None => {
                    info!("Feishu WebSocket: stream ended");
                    return Ok(());
                }
            }
        }
    }

    /// 处理二进制 Protobuf 帧
    async fn handle_binary_frame(
        &mut self,
        sink: &SharedSink,
        bytes: Vec<u8>,
        handler: Arc<dyn MessageHandler>,
    ) -> Result<()> {
        let frame = ProtoFrame::decode_from_slice(&bytes).ok_or_else(|| {
            ChannelError::ConnectionError("Failed to decode Protobuf frame".to_string())
        })?;

        match frame.method {
            FRAME_TYPE_CONTROL => {
                self.handle_control_frame(&frame);
            }
            FRAME_TYPE_DATA => {
                self.handle_data_frame(sink, frame, handler).await?;
            }
            _ => {
                warn!("Feishu WebSocket: unknown frame method: {}", frame.method);
            }
        }

        Ok(())
    }

    /// 处理控制帧（pong）
    fn handle_control_frame(&mut self, frame: &ProtoFrame) {
        let msg_type = frame.get_header(HEADER_TYPE).unwrap_or("");

        match msg_type {
            MESSAGE_TYPE_PONG => {
                debug!("Feishu WebSocket: received pong");
                if let Some(payload) = &frame.payload
                    && let Ok(config) = serde_json::from_slice::<ClientConfig>(payload)
                {
                    self.config.update_from_server(&config);
                    debug!("Feishu WebSocket: updated config from pong");
                }
            }
            MESSAGE_TYPE_PING => {
                debug!("Feishu WebSocket: received ping from server");
            }
            _ => {
                debug!("Feishu WebSocket: unknown control frame type: {}", msg_type);
            }
        }
    }

    /// 处理数据帧（事件）
    async fn handle_data_frame(
        &mut self,
        sink: &SharedSink,
        frame: ProtoFrame,
        handler: Arc<dyn MessageHandler>,
    ) -> Result<()> {
        let msg_type = frame.get_header(HEADER_TYPE).unwrap_or("");
        let msg_id = frame.get_header(HEADER_MESSAGE_ID).unwrap_or("");
        let sum = frame.get_header_int(HEADER_SUM);
        let seq = frame.get_header_int(HEADER_SEQ);

        // 分片处理
        let payload = if sum > 1 {
            match self.combine_fragments(msg_id, sum, seq, frame.payload.clone()) {
                Some(p) => p,
                None => return Ok(()),
            }
        } else {
            frame.payload.clone()
        };

        let payload = match payload {
            Some(p) => p,
            None => return Ok(()),
        };

        let payload_str = String::from_utf8_lossy(&payload);

        info!(
            "[V3] Feishu WebSocket: received data frame, msg_id={}, type={}",
            msg_id, msg_type
        );

        // 先处理再确认：成功应答必须意味着可恢复的工作已经完成。
        if msg_type == MESSAGE_TYPE_EVENT {
            // 从 payload 中提取飞书事件的 message_id 用于去重
            let event_message_id = Self::extract_event_message_id(&payload_str);

            let event_lock = event_message_id.as_ref().map(|event_mid| {
                self.event_locks
                    .entry(event_mid.clone())
                    .or_insert_with(|| Arc::new(Mutex::new(())))
                    .clone()
            });
            let _event_guard = if let Some(lock) = &event_lock {
                Some(lock.lock().await)
            } else {
                None
            };

            // 去重：检查该事件是否已处理过
            if let Some(ref event_mid) = event_message_id
                && self.processed_events.contains_key(event_mid)
            {
                info!(
                    "[V3] Feishu WebSocket: duplicate event detected, message_id={}, skipping",
                    event_mid
                );
                return Ok(());
            }

            // 顺带清理过期的去重缓存条目
            self.cleanup_processed_events();

            if let Err(error) =
                Self::process_event_async(payload_str.to_string(), handler, &self.config.app_secret)
                    .await
            {
                if let Some(event_mid) = &event_message_id {
                    self.event_locks.remove(event_mid);
                }
                return Err(error);
            }
            if let Some(event_mid) = event_message_id {
                self.processed_events
                    .insert(event_mid.clone(), Instant::now());
                self.event_locks.remove(&event_mid);
            }
        }

        self.send_response_frame(sink, frame, true).await?;

        Ok(())
    }

    /// 从事件 payload 中提取飞书消息的 message_id（用于去重）
    fn extract_event_message_id(payload: &str) -> Option<String> {
        let v: serde_json::Value = serde_json::from_str(payload).ok()?;
        v["event"]["message"]["message_id"]
            .as_str()
            .or_else(|| v["header"]["event_id"].as_str())
            .map(|s| s.to_string())
    }

    /// 清理过期的去重缓存条目
    fn cleanup_processed_events(&self) {
        let ttl = Duration::from_secs(DEDUP_TTL_SECS);
        self.processed_events.retain(|_, v| v.elapsed() < ttl);
    }

    /// 重新组装分片消息（带 TTL，防止内存泄漏）
    fn combine_fragments(
        &self,
        msg_id: &str,
        sum: i32,
        seq: i32,
        payload: Option<Vec<u8>>,
    ) -> Option<Option<Vec<u8>>> {
        let count = usize::try_from(sum).ok()?;
        let index = usize::try_from(seq).ok()?;
        if msg_id.is_empty()
            || count == 0
            || count > MAX_FRAGMENT_COUNT
            || index >= count
            || payload
                .as_ref()
                .is_some_and(|part| part.len() > MAX_FRAGMENT_BYTES)
        {
            return None;
        }
        let cache_key = msg_id.to_string();

        // 清理超时未完成的分片（5 分钟未收齐即视为丢失）
        let ttl = Duration::from_secs(DEDUP_TTL_SECS);
        self.fragment_cache
            .retain(|_, (created, _)| created.elapsed() < ttl);

        /*
        message id 理应稳定，但格式错误或重试的帧可能以不同的分片数量复用同一 id。
        当声明的分片形状变化时替换缓存条目，否则旧的、更短的向量会让新的序号越界。
        */
        let shape_changed = self
            .fragment_cache
            .get(&cache_key)
            .is_some_and(|entry| entry.value().1.len() != count);
        if shape_changed {
            self.fragment_cache.remove(&cache_key);
        }

        // 初始化缓存
        if !self.fragment_cache.contains_key(&cache_key) {
            let fragments: Vec<Option<Vec<u8>>> = vec![None; count];
            self.fragment_cache
                .insert(cache_key.clone(), (Instant::now(), fragments));
        }

        // 更新分片
        if let Some(mut entry) = self.fragment_cache.get_mut(&cache_key) {
            let (_, fragments) = entry.value_mut();
            let current_bytes = fragments
                .iter()
                .filter_map(Option::as_ref)
                .fold(0usize, |total, part| total.saturating_add(part.len()));
            let incoming_bytes = payload.as_ref().map_or(0, Vec::len);
            if current_bytes.saturating_add(incoming_bytes) > MAX_FRAGMENT_BYTES {
                drop(entry);
                self.fragment_cache.remove(&cache_key);
                return None;
            }
            let Some(fragment) = fragments.get_mut(index) else {
                /*
                形状检查之后条目可能被并发替换。此时将该帧视为不完整，
                并丢弃这个异常的缓存条目，而不是 panic。
                */
                drop(entry);
                self.fragment_cache.remove(&cache_key);
                return None;
            };
            *fragment = payload;

            let all_received = fragments.iter().all(|f| f.is_some());
            if all_received {
                let combined: Vec<u8> = fragments
                    .iter()
                    .filter_map(Option::as_ref)
                    .flatten()
                    .copied()
                    .collect();
                drop(entry);
                self.fragment_cache.remove(&cache_key);
                return Some(Some(combined));
            }
        }

        None
    }

    /// 异步处理事件
    async fn process_event_async(
        payload: String,
        handler: Arc<dyn MessageHandler>,
        app_secret: &str,
    ) -> Result<()> {
        let event: serde_json::Value = serde_json::from_str(&payload).map_err(|e| {
            ChannelError::ConnectionError(format!(
                "Failed to parse event JSON: {}",
                redact_text(&e.to_string())
            ))
        })?;

        let event_type = event["header"]["event_type"].as_str().unwrap_or("");

        debug!(
            "Feishu WebSocket: async processing event type: {}",
            event_type
        );

        if event_type == "im.message.receive_v1" {
            Self::process_im_message(event, handler, app_secret).await?;
        }

        Ok(())
    }

    /// 处理 IM 消息事件
    async fn process_im_message(
        event: serde_json::Value,
        handler: Arc<dyn MessageHandler>,
        app_secret: &str,
    ) -> Result<()> {
        let message = &event["event"]["message"];
        let sender = &event["event"]["sender"];

        let sender_id = super::sender_scope(
            sender["sender_id"]["open_id"]
                .as_str()
                .filter(|value| !value.is_empty()),
            sender["sender_id"]["user_id"]
                .as_str()
                .filter(|value| !value.is_empty()),
        )?;

        let chat_id = require_conversation_id(message["chat_id"].as_str())?;
        let chat_type_str = message["chat_type"].as_str().unwrap_or("p2p");
        let message_id = message["message_id"].as_str().unwrap_or("").to_string();

        let chat_type = if chat_type_str == "group" {
            ChatType::Group
        } else {
            ChatType::Direct
        };

        let msg_type = message["message_type"].as_str().unwrap_or("text");
        let content_str = message["content"].as_str().unwrap_or("{}");
        let text = Self::parse_message_content_static(msg_type, content_str);

        if text.is_empty() {
            debug!("Feishu WebSocket: empty text, ignoring");
            return Ok(());
        }

        let redacted = text_with_secrets(&text, [app_secret]);
        info!(
            "[V3] Feishu WebSocket: processing message from {} in {}: {}",
            sender_id,
            chat_id,
            redacted.chars().take(100).collect::<String>()
        );

        let inbound =
            InboundMessage::new("feishu", sender_id, chat_id, chat_type, text, message_id);

        match handler.handle(inbound).await {
            Ok(outbound) => {
                if let Err(e) = handler.reply(outbound).await {
                    warn!(
                        "Feishu WebSocket: failed to send reply: {}",
                        text_with_secrets(&e.to_string(), [app_secret])
                    );
                }
            }
            Err(e) => {
                warn!(
                    "Feishu WebSocket: handler error: {}",
                    text_with_secrets(&e.to_string(), [app_secret])
                );
            }
        }

        Ok(())
    }

    /// 按消息类型把消息内容解析为纯文本
    fn parse_message_content_static(msg_type: &str, content_str: &str) -> String {
        let content: serde_json::Value = serde_json::from_str(content_str).unwrap_or_default();

        match msg_type {
            "text" => content["text"].as_str().unwrap_or("").to_string(),
            "post" => Self::parse_rich_text_static(&content),
            "interactive" => String::new(),
            "image" => "[image]".to_string(),
            "file" => "[file]".to_string(),
            _ => content["text"].as_str().unwrap_or("").to_string(),
        }
    }

    /// 把富文本（post）内容解析为纯文本，段落之间以空行分隔
    fn parse_rich_text_static(content: &serde_json::Value) -> String {
        let paragraphs: Vec<String> = content["content"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|para| {
                        if let Some(elements) = para.as_array() {
                            let text_parts: Vec<String> = elements
                                .iter()
                                .filter_map(|elem| {
                                    let tag = elem["tag"].as_str().unwrap_or("");
                                    match tag {
                                        "text" | "at" => elem["text"].as_str().map(String::from),
                                        "img" => Some("[image]".to_string()),
                                        "file" | "media" => Some("[file]".to_string()),
                                        _ => None,
                                    }
                                })
                                .collect();
                            if text_parts.is_empty() {
                                None
                            } else {
                                Some(text_parts.join(" "))
                            }
                        } else {
                            None
                        }
                    })
                    .collect()
            })
            .unwrap_or_default();

        paragraphs.join("\n\n")
    }

    /// 发送响应帧
    async fn send_response_frame(
        &mut self,
        sink: &SharedSink,
        original_frame: ProtoFrame,
        success: bool,
    ) -> Result<()> {
        let msg_id = original_frame
            .get_header(HEADER_MESSAGE_ID)
            .unwrap_or("")
            .to_string();

        let mut response_frame = original_frame.clone();
        let code = if success { 0 } else { 500 };
        let response_json = serde_json::json!({ "code": code });
        response_frame.payload = Some(response_json.to_string().into_bytes());
        response_frame.payload_encoding = Some("json".to_string());

        let frame_bytes = response_frame.encode_to_vec();

        let mut ws = sink.lock().await;
        ws.send(Message::Binary(frame_bytes)).await.map_err(|e| {
            ChannelError::SendError(format!(
                "Failed to send response: {}",
                redact_text(&e.to_string())
            ))
        })?;

        info!(
            "[V3] Feishu WebSocket: response sent immediately for msg_id={}",
            msg_id
        );
        Ok(())
    }
}

/// 生成 [0, max) 范围内的随机抖动
fn rand_jitter(max: Duration) -> Duration {
    let max_ms = max.as_millis() as u64;
    if max_ms == 0 {
        return Duration::ZERO;
    }
    let jitter_ms = rand::rng().random_range(0..max_ms);
    Duration::from_millis(jitter_ms)
}

/// 单元测试
#[cfg(test)]
mod tests {
    use super::{WsClient, WsClientConfig};
    use crate::channels::{ChatType, InboundMessage, MessageHandler, OutboundMessage};
    use async_trait::async_trait;
    use echo_core::error::{ChannelError, ReactError};
    use serde_json::json;

    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// 构造测试用客户端
    fn client() -> WsClient {
        WsClient::new(WsClientConfig::new(
            "app-id".to_string(),
            "app-secret".to_string(),
            "wss://example.test".to_string(),
        ))
    }

    /// 分片数量变化时替换缓存形状而不会 panic
    #[test]
    fn fragment_count_change_replaces_cached_shape_without_panicking() {
        let client = client();

        assert_eq!(
            client.combine_fragments("same-id", 2, 0, Some(b"old".to_vec())),
            None
        );

        /*
        同一个 id 现在声明为三个分片。过去这里会用 seq=2 访问旧的
        两元素向量并 panic。
        */
        assert_eq!(
            client.combine_fragments("same-id", 3, 2, Some(b"c".to_vec())),
            None
        );
        assert_eq!(
            client.combine_fragments("same-id", 3, 0, Some(b"a".to_vec())),
            None
        );
        assert_eq!(
            client.combine_fragments("same-id", 3, 1, Some(b"b".to_vec())),
            Some(Some(b"abc".to_vec()))
        );
    }

    /// 统计 handle 调用次数的测试处理器
    struct CountingHandler {
        /// handle 被调用的次数
        calls: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl MessageHandler for CountingHandler {
        /// 测试实现：处理消息
        async fn handle(&self, msg: InboundMessage) -> echo_core::error::Result<OutboundMessage> {
            self.calls.fetch_add(1, Ordering::AcqRel);
            Ok(OutboundMessage::new(
                &msg.channel_id,
                msg.reply_target(),
                ChatType::Group,
                "ok",
            ))
        }

        /// 测试实现：忽略回复
        async fn reply(&self, _msg: OutboundMessage) -> echo_core::error::Result<()> {
            Ok(())
        }
    }

    /// 缺少发送者的消息绝不会到达飞书长连接处理器
    #[tokio::test]
    async fn missing_sender_never_reaches_feishu_long_poll_handler() -> Result<(), String> {
        let calls = Arc::new(AtomicUsize::new(0));
        let handler: Arc<dyn MessageHandler> = Arc::new(CountingHandler {
            calls: calls.clone(),
        });
        let event = json!({
            "event": {
                "sender": {"sender_id": {}},
                "message": {
                    "chat_id": "group-1",
                    "chat_type": "group",
                    "message_id": "m1",
                    "message_type": "text",
                    "content": "{\"text\":\"hello\"}"
                }
            }
        });

        let result = WsClient::process_im_message(event, handler, "test-app-secret").await;
        if !matches!(result, Err(ReactError::Channel(ref error)) if matches!(error.as_ref(), ChannelError::Other(message) if message.contains("sender_id")))
        {
            return Err("Feishu long poll did not return a typed sender error".to_string());
        }
        if calls.load(Ordering::Acquire) != 0 {
            return Err("Feishu long poll forwarded a message without a sender".to_string());
        }
        Ok(())
    }
}
