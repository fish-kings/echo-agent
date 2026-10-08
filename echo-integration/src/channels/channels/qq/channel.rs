//! QQ 机器人的 ChannelPlugin 实现

use super::super::super::types::*;
use super::super::{DeliveryRequest, DeliverySender};
use super::api::*;
use async_trait::async_trait;
use echo_core::error::ChannelError;
use echo_core::error::ReactError;
use echo_core::error::Result;
use std::fmt;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tracing::{info, warn};

// ── 配置 ────────────────────────────────────────────────────────────────────

/// QQ 机器人渠道配置
#[derive(Clone)]
pub struct QqConfig {
    /// QQ 机器人应用 ID
    pub app_id: String,
    /// QQ 机器人应用密钥
    pub client_secret: String,
}

impl fmt::Debug for QqConfig {
    /// 输出配置信息，`client_secret` 以占位符脱敏。
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("QqConfig")
            .field("app_id", &self.app_id)
            .field("client_secret", &crate::redaction::REDACTED)
            .finish()
    }
}

impl QqConfig {
    /// 使用给定的 `app_id` 与 `client_secret` 创建 QQ 机器人配置。
    pub fn new(app_id: impl Into<String>, client_secret: impl Into<String>) -> Self {
        Self {
            app_id: app_id.into(),
            client_secret: client_secret.into(),
        }
    }
}

// ── 渠道实现 ───────────────────────────────────────────────────────────────────

/// QQ 机器人 IM 渠道实现
pub struct QqChannel {
    /// 渠道配置（app_id、client_secret）。
    config: QqConfig,
    /// 共享的 HTTP 客户端（复用连接池）
    http: reqwest::Client,
    /// access_token 管理器，渠道启动后才会创建。
    token_manager: Option<Arc<TokenManager>>,
    /// 出站消息投递队列的发送端，渠道启动后才会创建。
    send_tx: Option<DeliverySender>,
    /// 后台消息发送任务的句柄。
    send_handle: Option<JoinHandle<()>>,
    /// Gateway 连接循环任务的句柄。
    gateway_handle: Option<JoinHandle<()>>,
}

impl QqChannel {
    /// 使用给定配置创建一个新的 QQ 渠道实例；`app_id`/`client_secret` 缺失时返回错误。
    pub fn new(config: QqConfig) -> Result<Self> {
        if config.app_id.is_empty() || config.client_secret.is_empty() {
            return Err(ReactError::Channel(Box::new(ChannelError::InvalidConfig(
                "QQ Bot requires app_id and client_secret".to_string(),
            ))));
        }

        Ok(Self {
            config,
            http: reqwest_client(),
            token_manager: None,
            send_tx: None,
            send_handle: None,
            gateway_handle: None,
        })
    }
}

#[async_trait]
impl ChannelPlugin for QqChannel {
    /// 返回渠道唯一标识 `"qqbot"`。
    fn id(&self) -> &str {
        "qqbot"
    }

    /// 返回渠道的展示名称 `"QQ Bot"`。
    fn label(&self) -> &str {
        "QQ Bot"
    }

    /// 返回该渠道支持的能力（会话类型、媒体、话题等）。
    fn capabilities(&self) -> &ChannelCapabilities {
        /// 渠道能力声明（惰性初始化的全局单例）
        static CAPS: std::sync::OnceLock<ChannelCapabilities> = std::sync::OnceLock::new();
        CAPS.get_or_init(|| ChannelCapabilities {
            chat_types: &[ChatType::Direct, ChatType::Group],
            supports_media: false,
            supports_threads: false,
        })
    }

    /// 启动 QQ 渠道：初始化 token 管理器、启动后台发送任务与 Gateway 连接循环。
    async fn start(&mut self, handler: Arc<dyn MessageHandler>) -> Result<()> {
        if self.gateway_handle.is_some() || self.send_handle.is_some() {
            return Err(ReactError::Channel(Box::new(ChannelError::Other(
                "QQ Bot channel is already started".to_string(),
            ))));
        }
        info!("Starting QQ Bot channel...");

        // 1. 初始化 token 管理器
        let token_manager = Arc::new(TokenManager::new(
            self.config.app_id.clone(),
            self.config.client_secret.clone(),
        ));
        self.token_manager = Some(token_manager.clone());

        // 2. 启动后台消息发送任务
        let (send_tx, mut send_rx) = mpsc::channel::<DeliveryRequest>(256);
        let token_manager_clone = token_manager.clone();

        // 3. 创建包装 handler —— 调用内部 handler 后自动发送回复
        let send_tx_clone = send_tx.clone();
        self.send_tx = Some(send_tx);
        let wrapper = Arc::new(QqMessageHandler {
            inner: handler,
            send_tx: send_tx_clone,
        });

        // 4. 启动消息发送任务
        let http_for_send = self.http.clone();
        let token_manager_clone2 = token_manager.clone();
        let send_handle = tokio::spawn(async move {
            while let Some(request) = send_rx.recv().await {
                let DeliveryRequest {
                    message,
                    receipt,
                    delivery_permit,
                } = request;
                let result = async {
                    let token = token_manager_clone.get_token().await?;
                    send_qq_message(
                        &http_for_send,
                        &token,
                        &message.to,
                        &message.chat_type,
                        &message.text,
                        message.reply_to.as_deref(),
                    )
                    .await
                }
                .await;
                drop(delivery_permit);
                let _ = receipt.send(result);
            }
        });
        self.send_handle = Some(send_handle);

        // 5. 启动 Gateway 连接循环（带指数退避重连）
        let http_for_gw = self.http.clone();
        let gateway_handle = tokio::spawn(async move {
            let mut reconnect_delay: u64 = 1;
            /// 重连退避的最大延迟（秒）
            const MAX_DELAY: u64 = 60;
            /// 若连接时长超过此阈值则视为稳定，重置退避延迟
            const STABLE_THRESHOLD_SECS: u64 = 60;

            loop {
                let token = match token_manager_clone2.get_token().await {
                    Ok(t) => t,
                    Err(e) => {
                        warn!(
                            "QQ Gateway: failed to get token: {}",
                            crate::redaction::text(&e.to_string())
                        );
                        tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                        continue;
                    }
                };

                let ws_url = match get_gateway_url(&http_for_gw, &token).await {
                    Ok(u) => u,
                    Err(e) => {
                        warn!(
                            "QQ Gateway: failed to get gateway URL: {}",
                            crate::redaction::text(&e.to_string())
                        );
                        tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                        continue;
                    }
                };

                info!("QQ Gateway: connecting...");
                let connected_at = std::time::Instant::now();

                match super::gateway::connect_to_gateway(ws_url, wrapper.clone(), token.clone())
                    .await
                {
                    Ok(()) => {
                        warn!(
                            "QQ Gateway: connection closed, reconnecting in {}s...",
                            reconnect_delay
                        );
                    }
                    Err(e) => {
                        warn!(
                            "QQ Gateway: connection error: {:?}, reconnecting in {}s...",
                            crate::redaction::text(&e.to_string()),
                            reconnect_delay
                        );
                    }
                }

                // 若连接稳定运行超过阈值，重置退避延迟
                if connected_at.elapsed().as_secs() >= STABLE_THRESHOLD_SECS {
                    reconnect_delay = 1;
                } else {
                    reconnect_delay = (reconnect_delay * 2).min(MAX_DELAY);
                }

                tokio::time::sleep(std::time::Duration::from_secs(reconnect_delay)).await;
            }
        });

        self.gateway_handle = Some(gateway_handle);

        info!("QQ Bot channel started");
        Ok(())
    }

    /// 停止 QQ 渠道：终止 Gateway 连接任务与发送任务，并清空运行时状态。
    async fn stop(&mut self) -> Result<()> {
        info!("Stopping QQ Bot channel...");

        if let Some(handle) = self.gateway_handle.take() {
            handle.abort();
            let _ = handle.await;
        }

        self.send_tx = None;
        if let Some(handle) = self.send_handle.take() {
            handle.abort();
            let _ = handle.await;
        }
        self.token_manager = None;
        info!("QQ Bot channel stopped");
        Ok(())
    }

    /// 直接发送一条出站消息（不经过后台发送队列）。
    async fn send(&self, msg: OutboundMessage) -> Result<()> {
        let _delivery_permit = msg.begin_delivery()?;
        let token_manager = self.token_manager.as_ref().ok_or_else(|| {
            ReactError::Channel(Box::new(ChannelError::SendError(
                "QQ Bot channel not started".to_string(),
            )))
        })?;
        let token = token_manager.get_token().await?;
        send_qq_message(
            &self.http,
            &token,
            &msg.to,
            &msg.chat_type,
            &msg.text,
            msg.reply_to.as_deref(),
        )
        .await
    }

    /// 健康检查：确认 Gateway 任务仍在运行，并尝试获取一次 access_token。
    async fn health_check(&self) -> Result<()> {
        if self
            .gateway_handle
            .as_ref()
            .is_some_and(|h| h.is_finished())
        {
            return Err(ReactError::Channel(Box::new(
                ChannelError::ConnectionError("QQ Bot gateway task has terminated".to_string()),
            )));
        }
        if let Some(ref tm) = self.token_manager {
            tm.get_token().await?;
        }
        Ok(())
    }
}

/// 包装器：调用内部 handler 后自动经 send_tx 发送回复
struct QqMessageHandler {
    /// 被包装的内部消息处理器。
    inner: Arc<dyn MessageHandler>,
    /// 用于投递出站消息的发送端。
    send_tx: DeliverySender,
}

#[async_trait]
impl MessageHandler for QqMessageHandler {
    /**
    消费 inner 的流式分段,逐 chunk 经 send_tx 投递(真流式);返回空 text 占位
    (gateway 会再调 reply,reply 对空 text no-op 防双发,见 super::reply_with_empty_guard)。
    */
    async fn handle(&self, msg: InboundMessage) -> Result<OutboundMessage> {
        super::super::dispatch_stream_to_send_tx(&self.inner, &self.send_tx, msg).await
    }

    /// 对空 text 视为流式占位不做发送，非空 text 则正常经 send_tx 投递。
    async fn reply(&self, msg: OutboundMessage) -> Result<()> {
        super::super::reply_with_empty_guard(&self.send_tx, msg).await
    }
}

#[cfg(test)]
mod config_tests {
    use super::*;
    use crate::channels::types::ChannelDeliveryFence;

    /// 验证 Debug 输出不会暴露 client secret。
    #[test]
    fn debug_does_not_expose_client_secret() {
        let config = QqConfig::new("app-id", "qq-client-secret");
        let debug = format!("{config:?}");
        assert!(debug.contains("app-id"));
        assert!(
            !debug.contains("qq-client-secret"),
            "debug leaked client secret: {debug}"
        );
    }

    /// 验证直接发送会在访问通道状态前拒绝已退役的代次。
    #[tokio::test]
    async fn direct_send_rejects_retired_generation_before_channel_state() -> Result<()> {
        let channel = QqChannel::new(QqConfig::new("app-id", "client-secret"))?;
        let fence = ChannelDeliveryFence::new("retired-qq".to_string());
        fence.retire();
        let result = channel
            .send(
                OutboundMessage::new("qqbot", "recipient", ChatType::Direct, "stale")
                    .with_delivery_fence(fence),
            )
            .await;
        if !matches!(
            result,
            Err(ReactError::Channel(ref error))
                if matches!(
                    error.as_ref(),
                    ChannelError::StaleDelivery { incarnation_id }
                        if incarnation_id == "retired-qq"
                )
        ) {
            return Err(ReactError::Other(
                "QQ direct send did not reject the retired generation first".to_string(),
            ));
        }
        Ok(())
    }
}
