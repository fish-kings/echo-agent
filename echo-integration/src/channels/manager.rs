//! ChannelManager —— 管理多个 IM 渠道插件的生命周期

use super::types::{ChannelPlugin, MessageHandler};
use echo_core::error::Result;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tracing::{info, warn};

/// 携带渠道标识的单个渠道生命周期操作结果。
pub struct ChannelLifecycleResult {
    /// 渠道 ID。
    pub channel_id: String,
    /// 该渠道生命周期操作的结果。
    pub result: Result<()>,
}

/**
管理多个 IM 渠道的启动、关闭与查询。

支持：
- 注册多个 ChannelPlugin（QQ 机器人、飞书等）
- 统一启动 / 停止
- 按 ID 查询或发送
- 通过 [`Self::stop_all`] 显式等待关闭完成
*/
pub struct ChannelManager {
    /// 已注册的渠道插件，按渠道 ID 索引。
    channels: HashMap<String, Box<dyn ChannelPlugin>>,
    /// 各渠道当前持有的消息处理器，按渠道 ID 索引。
    handlers: HashMap<String, Arc<dyn MessageHandler>>,
    /// 传输层已成功停止的渠道 ID 集合，避免重复调用插件 stop。
    stopped_transports: HashSet<String>,
}

impl Default for ChannelManager {
    /// 等价于 [`ChannelManager::new`]。
    fn default() -> Self {
        Self::new()
    }
}

impl ChannelManager {
    /// 创建空的渠道管理器。
    pub fn new() -> Self {
        Self {
            channels: HashMap::new(),
            handlers: HashMap::new(),
            stopped_transports: HashSet::new(),
        }
    }

    /// 注册渠道插件；ID 重复时返回错误
    pub fn register(&mut self, plugin: Box<dyn ChannelPlugin>) -> Result<()> {
        let id = plugin.id().to_string();
        if self.channels.contains_key(&id) {
            return Err(echo_core::error::ReactError::Channel(Box::new(
                echo_core::error::ChannelError::Other(format!(
                    "Channel '{id}' is already registered"
                )),
            )));
        }
        info!("Registering channel: {}", id);
        self.channels.insert(id, plugin);
        Ok(())
    }

    /// 获取已注册渠道数量
    pub fn len(&self) -> usize {
        self.channels.len()
    }

    /// 是否没有注册任何渠道。
    pub fn is_empty(&self) -> bool {
        self.channels.is_empty()
    }

    /**
    启动所有已注册渠道。

    为每个渠道创建处理器并启动。返回每个渠道的 [`ChannelLifecycleResult`]，
    调用方可逐一检查各渠道结果；单个渠道失败不影响其他渠道。
    */
    pub async fn start_all(
        &mut self,
        handler_factory: impl Fn(&str) -> Arc<dyn MessageHandler> + Sync,
    ) -> Vec<ChannelLifecycleResult> {
        info!("Starting all channels (count: {})", self.channels.len());

        let mut results = Vec::with_capacity(self.channels.len());

        for (id, plugin) in self.channels.iter_mut() {
            if self.handlers.contains_key(id) {
                results.push(ChannelLifecycleResult {
                    channel_id: id.clone(),
                    result: Err(echo_core::error::ReactError::Channel(Box::new(
                        echo_core::error::ChannelError::Other(format!(
                            "Channel '{id}' still owns a handler; stop it before restarting"
                        )),
                    ))),
                });
                continue;
            }
            self.stopped_transports.remove(id);
            let handler = handler_factory(id);
            self.handlers.insert(id.clone(), Arc::clone(&handler));
            match plugin.start(Arc::clone(&handler)).await {
                Ok(()) => {
                    info!("Channel '{}' started successfully", id);
                    results.push(ChannelLifecycleResult {
                        channel_id: id.clone(),
                        result: Ok(()),
                    });
                }
                Err(e) => {
                    warn!("Failed to start channel '{}': {}", id, e);
                    /* 启动失败时插件可能已持有处理器或打开了传输层。
                    只要回滚未完全成功，就保留所有权。 */
                    let failure = match plugin.stop().await {
                        Ok(()) => {
                            self.stopped_transports.insert(id.clone());
                            match handler.close().await {
                                Ok(()) => {
                                    self.handlers.remove(id);
                                    e
                                }
                                Err(close_error) => {
                                    warn!(
                                        "Channel '{}' handler close after failed start: {}",
                                        id, close_error
                                    );
                                    echo_core::error::ReactError::Channel(Box::new(
                                        echo_core::error::ChannelError::Other(format!(
                                            "Channel '{id}' start failed: {e}; handler close failed: {close_error}"
                                        )),
                                    ))
                                }
                            }
                        }
                        Err(stop_error) => {
                            warn!("Channel '{}' stop after failed start: {}", id, stop_error);
                            echo_core::error::ReactError::Channel(Box::new(
                                echo_core::error::ChannelError::Other(format!(
                                    "Channel '{id}' start failed: {e}; rollback stop failed: {stop_error}"
                                )),
                            ))
                        }
                    };
                    results.push(ChannelLifecycleResult {
                        channel_id: id.clone(),
                        result: Err(failure),
                    });
                }
            }
        }

        results
    }

    /// 停止单个渠道（先停传输层，再关闭处理器）
    pub async fn stop(&mut self, channel_id: &str) -> Result<()> {
        if let Some(plugin) = self.channels.get_mut(channel_id) {
            info!("Stopping channel: {}", channel_id);
            if !self.stopped_transports.contains(channel_id) {
                plugin.stop().await?;
                self.stopped_transports.insert(channel_id.to_string());
            }
            if let Some(handler) = self.handlers.get(channel_id) {
                handler.close().await?;
                self.handlers.remove(channel_id);
            }
            info!("Channel '{}' stopped", channel_id);
            Ok(())
        } else {
            Err(echo_core::error::ReactError::Channel(Box::new(
                echo_core::error::ChannelError::Other(format!(
                    "Channel '{}' not found",
                    channel_id
                )),
            )))
        }
    }

    /// 停止所有已注册渠道，汇总各渠道的失败信息
    pub async fn stop_all(&mut self) -> Result<()> {
        info!("Stopping all channels...");

        let mut failures = Vec::new();
        for id in self.channels.keys().cloned().collect::<Vec<_>>() {
            match self.stop(&id).await {
                Ok(()) => info!("Channel '{}' stopped", id),
                Err(e) => {
                    warn!("Failed to stop channel '{}': {}", id, e);
                    failures.push(format!("{id}: {e}"));
                }
            }
        }
        if failures.is_empty() {
            Ok(())
        } else {
            Err(echo_core::error::ReactError::Channel(Box::new(
                echo_core::error::ChannelError::Other(format!(
                    "failed to stop channels: {}",
                    failures.join("; ")
                )),
            )))
        }
    }

    /// 按 ID 获取渠道引用
    pub fn get(&self, id: &str) -> Option<&(dyn ChannelPlugin + '_)> {
        match self.channels.get(id) {
            Some(plugin) => Some(plugin.as_ref()),
            None => None,
        }
    }

    /// 按 ID 获取渠道的可变引用
    pub fn get_mut(&mut self, id: &str) -> Option<&mut (dyn ChannelPlugin + '_)> {
        /* 外部可变访问可能重启或以其他方式改变传输层状态，
        因此下一次受管停止必须再次调用插件的 stop。 */
        self.stopped_transports.remove(id);
        match self.channels.get_mut(id) {
            Some(plugin) => Some(plugin.as_mut()),
            None => None,
        }
    }

    /// 列出所有已注册的渠道 ID
    pub fn channel_ids(&self) -> Vec<&str> {
        self.channels.keys().map(|k| k.as_str()).collect()
    }
}

impl Drop for ChannelManager {
    /// 若仍有未关闭的处理器，记录提示日志。
    fn drop(&mut self) {
        if !self.handlers.is_empty() {
            info!(
                "ChannelManager dropped with {} unsettled handlers remaining, \
                 consider calling stop_all() before drop",
                self.handlers.len()
            );
        }
    }
}

#[cfg(test)]
mod close_tests {
    use super::*;
    use crate::channels::types::{ChannelCapabilities, ChatType, InboundMessage, OutboundMessage};
    use async_trait::async_trait;
    use echo_core::error::{ChannelError, ReactError};
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::time::Duration;
    use tokio::sync::Notify;

    /// 测试用处理器：传输层未停止时拒绝关闭，首次关闭返回临时错误。
    struct ClosingHandler {
        /// 传输层是否已停止。
        stopped: Arc<AtomicBool>,
        /// close 调用次数。
        closes: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl MessageHandler for ClosingHandler {
        async fn handle(&self, _msg: InboundMessage) -> Result<OutboundMessage> {
            Err(ReactError::Other("not used".to_string()))
        }

        async fn reply(&self, _msg: OutboundMessage) -> Result<()> {
            Ok(())
        }

        async fn close(&self) -> Result<()> {
            if !self.stopped.load(Ordering::Acquire) {
                return Err(ReactError::Other("transport still accepting".to_string()));
            }
            let attempt = self.closes.fetch_add(1, Ordering::AcqRel);
            if attempt == 0 {
                Err(ReactError::Other("transient close failure".to_string()))
            } else {
                Ok(())
            }
        }
    }

    /// 测试用渠道插件：记录 stop 次数，可配置重复 stop 时报错。
    struct ClosingPlugin {
        /// 传输层是否已停止。
        stopped: Arc<AtomicBool>,
        /// stop 调用次数。
        stops: Arc<AtomicUsize>,
        /// 重复调用 stop 时是否返回错误。
        fail_repeated_stop: bool,
        /// 启动时传入的处理器。
        handler: Option<Arc<dyn MessageHandler>>,
    }

    #[async_trait]
    impl ChannelPlugin for ClosingPlugin {
        fn id(&self) -> &str {
            "close-test"
        }

        fn capabilities(&self) -> &ChannelCapabilities {
            static CAPABILITIES: ChannelCapabilities = ChannelCapabilities {
                chat_types: &[ChatType::Direct],
                supports_media: false,
                supports_threads: false,
            };
            &CAPABILITIES
        }

        async fn start(&mut self, handler: Arc<dyn MessageHandler>) -> Result<()> {
            self.stopped.store(false, Ordering::Release);
            self.handler = Some(handler);
            Ok(())
        }

        async fn stop(&mut self) -> Result<()> {
            let attempt = self.stops.fetch_add(1, Ordering::AcqRel);
            if self.fail_repeated_stop && attempt > 0 {
                return Err(ReactError::Other("transport already stopped".to_string()));
            }
            self.stopped.store(true, Ordering::Release);
            self.handler = None;
            Ok(())
        }

        async fn send(&self, _msg: OutboundMessage) -> Result<()> {
            Err(ReactError::Channel(Box::new(ChannelError::Other(
                "not used".to_string(),
            ))))
        }
    }

    /// 验证处理器关闭成功前，管理器始终保留处理器所有权。
    #[tokio::test]
    async fn stop_retains_handler_until_awaited_close_succeeds() -> Result<()> {
        let stopped = Arc::new(AtomicBool::new(false));
        let stops = Arc::new(AtomicUsize::new(0));
        let closes = Arc::new(AtomicUsize::new(0));
        let mut manager = ChannelManager::new();
        manager.register(Box::new(ClosingPlugin {
            stopped: Arc::clone(&stopped),
            stops: Arc::clone(&stops),
            fail_repeated_stop: true,
            handler: None,
        }))?;
        let results = manager
            .start_all(|_| {
                Arc::new(ClosingHandler {
                    stopped: Arc::clone(&stopped),
                    closes: Arc::clone(&closes),
                })
            })
            .await;
        assert_eq!(results.len(), 1);
        assert!(results.iter().all(|result| result.result.is_ok()));

        let first = manager.stop("close-test").await;
        assert!(first.is_err());
        assert_eq!(closes.load(Ordering::Acquire), 1);
        manager.stop("close-test").await?;
        assert_eq!(closes.load(Ordering::Acquire), 2);
        assert_eq!(stops.load(Ordering::Acquire), 1);
        assert!(manager.handlers.is_empty());
        Ok(())
    }

    /// 验证被取消的 stop 会保留处理器以便重试。
    #[tokio::test]
    async fn cancelled_stop_keeps_handler_for_retry() -> Result<()> {
        /// 首次 close 永久挂起的测试处理器。
        struct InterruptedHandler {
            /// 首次进入 close 时发出通知。
            entered: Arc<Notify>,
            /// close 调用次数。
            closes: Arc<AtomicUsize>,
        }

        #[async_trait]
        impl MessageHandler for InterruptedHandler {
            async fn handle(&self, _msg: InboundMessage) -> Result<OutboundMessage> {
                Err(ReactError::Other("not used".to_string()))
            }
            async fn reply(&self, _msg: OutboundMessage) -> Result<()> {
                Ok(())
            }
            async fn close(&self) -> Result<()> {
                if self.closes.fetch_add(1, Ordering::AcqRel) == 0 {
                    self.entered.notify_one();
                    std::future::pending::<()>().await;
                }
                Ok(())
            }
        }

        let stopped = Arc::new(AtomicBool::new(false));
        let stops = Arc::new(AtomicUsize::new(0));
        let entered = Arc::new(Notify::new());
        let closes = Arc::new(AtomicUsize::new(0));
        let mut manager = ChannelManager::new();
        manager.register(Box::new(ClosingPlugin {
            stopped,
            stops: Arc::clone(&stops),
            fail_repeated_stop: true,
            handler: None,
        }))?;
        let results = manager
            .start_all(|_| {
                Arc::new(InterruptedHandler {
                    entered: Arc::clone(&entered),
                    closes: Arc::clone(&closes),
                })
            })
            .await;
        assert!(results.iter().all(|result| result.result.is_ok()));

        assert!(
            tokio::time::timeout(Duration::from_millis(20), manager.stop("close-test"))
                .await
                .is_err()
        );
        assert_eq!(closes.load(Ordering::Acquire), 1);
        manager.stop("close-test").await?;
        assert_eq!(closes.load(Ordering::Acquire), 2);
        assert_eq!(stops.load(Ordering::Acquire), 1);
        Ok(())
    }

    /// 验证被取消的 start 仍保留已发布的处理器，供 stop_all 关闭。
    #[tokio::test]
    async fn cancelled_start_keeps_published_handler_for_stop_all() -> Result<()> {
        /// start 永久挂起的测试插件。
        struct ParkedStartPlugin {
            /// 进入 start 后发出通知。
            started: Arc<Notify>,
            /// 传输层是否已停止。
            stopped: Arc<AtomicBool>,
            /// 启动时传入的处理器。
            handler: Option<Arc<dyn MessageHandler>>,
        }

        #[async_trait]
        impl ChannelPlugin for ParkedStartPlugin {
            fn id(&self) -> &str {
                "parked-start"
            }
            fn capabilities(&self) -> &ChannelCapabilities {
                static CAPABILITIES: ChannelCapabilities = ChannelCapabilities {
                    chat_types: &[ChatType::Direct],
                    supports_media: false,
                    supports_threads: false,
                };
                &CAPABILITIES
            }
            async fn start(&mut self, handler: Arc<dyn MessageHandler>) -> Result<()> {
                self.handler = Some(handler);
                self.started.notify_one();
                std::future::pending::<Result<()>>().await
            }
            async fn stop(&mut self) -> Result<()> {
                self.stopped.store(true, Ordering::Release);
                self.handler = None;
                Ok(())
            }
            async fn send(&self, _msg: OutboundMessage) -> Result<()> {
                Ok(())
            }
        }

        /// 传输层未停止时拒绝关闭的测试处理器。
        struct ParkedStartHandler {
            /// 传输层是否已停止。
            stopped: Arc<AtomicBool>,
            /// 成功 close 的次数。
            closes: Arc<AtomicUsize>,
        }
        #[async_trait]
        impl MessageHandler for ParkedStartHandler {
            async fn handle(&self, _msg: InboundMessage) -> Result<OutboundMessage> {
                Err(ReactError::Other("not used".to_string()))
            }
            async fn reply(&self, _msg: OutboundMessage) -> Result<()> {
                Ok(())
            }
            async fn close(&self) -> Result<()> {
                if !self.stopped.load(Ordering::Acquire) {
                    return Err(ReactError::Other("transport still running".to_string()));
                }
                self.closes.fetch_add(1, Ordering::AcqRel);
                Ok(())
            }
        }

        let started = Arc::new(Notify::new());
        let stopped = Arc::new(AtomicBool::new(false));
        let closes = Arc::new(AtomicUsize::new(0));
        let mut manager = ChannelManager::new();
        manager.register(Box::new(ParkedStartPlugin {
            started: Arc::clone(&started),
            stopped: Arc::clone(&stopped),
            handler: None,
        }))?;
        assert!(
            tokio::time::timeout(
                Duration::from_millis(20),
                manager.start_all(|_| {
                    Arc::new(ParkedStartHandler {
                        stopped: Arc::clone(&stopped),
                        closes: Arc::clone(&closes),
                    })
                })
            )
            .await
            .is_err()
        );
        started.notified().await;
        manager.stop_all().await?;
        assert_eq!(closes.load(Ordering::Acquire), 1);
        Ok(())
    }
}
