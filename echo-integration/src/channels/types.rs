//! IM 渠道统一消息类型与 trait 定义

use async_trait::async_trait;
use echo_core::agent::CancellationToken;
pub use echo_core::error::ChannelError;
pub use echo_core::error::ReactError;
use echo_core::error::Result;
use futures::stream::{BoxStream, StreamExt};
use std::sync::{Arc, Mutex, MutexGuard};
use tokio::sync::Notify;

// ── 聊天类型 ────────────────────────────────────────────────────────────────

/// 会话类型:私聊 / 群聊
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatType {
    /// 私聊
    Direct,
    /// 群聊
    Group,
}

impl std::fmt::Display for ChatType {
    /// 将会话类型格式化为小写字符串("direct" / "group")。
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChatType::Direct => write!(f, "direct"),
            ChatType::Group => write!(f, "group"),
        }
    }
}

// ── 多媒体附件 ───────────────────────────────────────────────────

/// 附件类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttachmentKind {
    /// 图片(PNG、JPEG、GIF、WebP 等)
    Image,
    /// 文件(PDF、DOC、TXT 等)
    File,
    /// 音频(MP3、WAV、OGG 等)
    Audio,
    /// 视频(MP4、AVI 等)
    Video,
}

/// 消息附件
#[derive(Debug, Clone)]
pub struct MessageAttachment {
    /// 附件类型
    pub kind: AttachmentKind,
    /// 附件二进制数据
    pub data: Vec<u8>,
    /// 文件名(可选)
    pub filename: Option<String>,
}

impl MessageAttachment {
    /// 构造一个不带文件名的消息附件。
    pub fn new(kind: AttachmentKind, data: Vec<u8>) -> Self {
        Self {
            kind,
            data,
            filename: None,
        }
    }

    /// 为附件设置文件名,返回修改后的自身(builder 风格)。
    pub fn with_filename(mut self, filename: impl Into<String>) -> Self {
        self.filename = Some(filename.into());
        self
    }
}

// ── 入站消息 ──────────────────────────────────────────────────────────

/// 从 IM 平台接收到的消息
#[derive(Debug, Clone)]
pub struct InboundMessage {
    /// 渠道 ID:"qqbot" | "feishu"
    pub channel_id: String,
    /**
    传输层范围内的规范发送者身份标识。

    QQ 使用其原始 openid。飞书使用 `open_id:{value}` 或
    `user_id:{value}`,以避免不同服务商的身份命名空间在
    会话 key 或回调中发生冲突。
    */
    pub sender_id: String,
    /// 会话标识(群聊时为 group_id,私聊时为 sender_id)
    pub chat_id: String,
    /// 会话类型
    pub chat_type: ChatType,
    /// 消息文本
    pub text: String,
    /// 平台原始消息 ID(用于回复引用)
    pub message_id: String,
    /// 时间戳(Unix 秒)
    pub timestamp: u64,
    /// 消息附件(图片、文件、音频、视频等)
    pub attachments: Vec<MessageAttachment>,
}

/**
校验一个身份字段(如 channel_id/conversation_id/sender_id)是否存在、
非空且不包含首尾空白,否则返回对应的 `ChannelError`。
*/
fn require_identity<'a>(
    value: Option<&'a str>,
    field: &str,
) -> std::result::Result<&'a str, ChannelError> {
    let value = value
        .filter(|value| !value.is_empty())
        .ok_or_else(|| ChannelError::Other(format!("missing channel message {field}")))?;
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(ChannelError::Other(format!(
            "missing channel message {field}"
        )));
    }
    if trimmed != value {
        return Err(ChannelError::Other(format!(
            "invalid channel message {field}: surrounding whitespace"
        )));
    }
    Ok(value)
}

/// 校验并返回非空的 `channel_id`。
pub(crate) fn require_channel_id(value: Option<&str>) -> std::result::Result<&str, ChannelError> {
    require_identity(value, "channel_id")
}

/// 校验并返回非空的 `conversation_id`。
pub(crate) fn require_conversation_id(
    value: Option<&str>,
) -> std::result::Result<&str, ChannelError> {
    require_identity(value, "conversation_id")
}

/// 校验并返回非空、非 "unknown" 的 `sender_id`。
pub(crate) fn require_sender_id(value: Option<&str>) -> std::result::Result<&str, ChannelError> {
    let sender_id = require_identity(value, "sender_id")?;
    if sender_id.eq_ignore_ascii_case("unknown") {
        return Err(ChannelError::Other(
            "invalid channel message sender_id: unknown".to_string(),
        ));
    }
    Ok(sender_id)
}

impl InboundMessage {
    /// 构造一条入站消息,时间戳自动取当前 UTC 时间,附件列表默认为空。
    pub fn new(
        channel_id: impl Into<String>,
        sender_id: impl Into<String>,
        chat_id: impl Into<String>,
        chat_type: ChatType,
        text: impl Into<String>,
        message_id: impl Into<String>,
    ) -> Self {
        Self {
            channel_id: channel_id.into(),
            sender_id: sender_id.into(),
            chat_id: chat_id.into(),
            chat_type,
            text: text.into(),
            message_id: message_id.into(),
            timestamp: chrono::Utc::now().timestamp() as u64,
            attachments: Vec::new(),
        }
    }

    /// 添加附件
    pub fn with_attachments(mut self, attachments: Vec<MessageAttachment>) -> Self {
        self.attachments = attachments;
        self
    }

    /// 用于会话/缓存隔离的稳定会话身份标识。
    pub fn conversation_id(&self) -> &str {
        &self.chat_id
    }

    /// 回复该会话时使用的传输层目标标识。
    pub fn reply_target(&self) -> &str {
        &self.chat_id
    }
}

// ── 出站消息 ─────────────────────────────────────────────────────────

/// `ChannelDeliveryFence` 的内部共享状态:生命周期信息与"已结算"通知。
struct ChannelDeliveryFenceState {
    /// 受互斥锁保护的世代生命周期数据。
    lifecycle: Mutex<ChannelDeliveryFenceLifecycle>,
    /// 在活跃投递数归零时用于唤醒等待者的通知器。
    settled: Notify,
}

/// 记录当前世代 ID、是否已退役、以及仍在进行中的投递数量。
struct ChannelDeliveryFenceLifecycle {
    /// 当前有效的世代(incarnation)ID。
    current_incarnation_id: Arc<str>,
    /// 是否已停止接受新的投递。
    retired: bool,
    /// 仍在进行中(已获得许可但尚未完成)的投递数量。
    active_deliveries: usize,
}

/**
为某一个"渠道会话世代"设限的、进程本地的不透明权限对象。

`SessionHandler` 会将此权限对象附加到其输出上。内置传输层在
将消息送入队列或发起网络请求之前会先获取一个租约(lease),
从而使 reset 操作能够拒绝过期(stale)的输出,并只需等待
已经被受理的投递完成即可。
*/
#[doc(hidden)]
#[derive(Clone)]
pub(super) struct ChannelDeliveryFence {
    /// 本 fence 实例所属的世代 ID。
    incarnation_id: Arc<str>,
    /// 与同一逻辑 fence 的所有克隆/轮换实例共享的状态。
    state: Arc<ChannelDeliveryFenceState>,
}

impl std::fmt::Debug for ChannelDeliveryFence {
    /// 输出世代 ID 与是否已退役,便于调试排查。
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ChannelDeliveryFence")
            .field("incarnation_id", &self.incarnation_id)
            .field("retired", &self.is_retired())
            .finish_non_exhaustive()
    }
}

impl ChannelDeliveryFence {
    /// 以给定的世代 ID 创建一个全新的投递围栏(未退役、无活跃投递)。
    pub(super) fn new(incarnation_id: String) -> Self {
        let incarnation_id = Arc::from(incarnation_id);
        Self {
            incarnation_id: Arc::clone(&incarnation_id),
            state: Arc::new(ChannelDeliveryFenceState {
                lifecycle: Mutex::new(ChannelDeliveryFenceLifecycle {
                    current_incarnation_id: incarnation_id,
                    retired: false,
                    active_deliveries: 0,
                }),
                settled: Notify::new(),
            }),
        }
    }

    /// 加锁并返回生命周期数据;若互斥锁被污染(poisoned)则记录错误日志后恢复使用。
    fn lock_lifecycle(&self) -> MutexGuard<'_, ChannelDeliveryFenceLifecycle> {
        match self.state.lifecycle.lock() {
            Ok(lifecycle) => lifecycle,
            Err(poisoned) => {
                tracing::error!(
                    incarnation_id = %self.incarnation_id,
                    "channel delivery fence mutex was poisoned"
                );
                poisoned.into_inner()
            }
        }
    }

    /// 此 fence 持有的不透明世代身份标识。
    pub(super) fn incarnation_id(&self) -> &str {
        &self.incarnation_id
    }

    /// 当前世代是否已经停止接受新的投递。
    pub(super) fn is_retired(&self) -> bool {
        let lifecycle = self.lock_lifecycle();
        lifecycle.retired || lifecycle.current_incarnation_id != self.incarnation_id
    }

    /// 将共享状态轮换到新的世代 ID,并返回代表新世代的 fence 实例。
    pub(super) fn rotate(&self, incarnation_id: String) -> Self {
        let incarnation_id = Arc::from(incarnation_id);
        self.lock_lifecycle().current_incarnation_id = Arc::clone(&incarnation_id);
        Self {
            incarnation_id,
            state: Arc::clone(&self.state),
        }
    }

    /// 是否仍存在尚未完成的投递。
    pub(super) fn has_active_deliveries(&self) -> bool {
        self.lock_lifecycle().active_deliveries > 0
    }

    /**
    尝试为一次投递获取许可:若当前世代已退役或已被轮换,则返回
    `ChannelError::StaleDelivery`;否则活跃投递计数加一并返回许可对象。
    */
    pub(super) fn begin_delivery(&self) -> Result<ChannelDeliveryPermit> {
        let mut lifecycle = self.lock_lifecycle();
        if lifecycle.retired || lifecycle.current_incarnation_id != self.incarnation_id {
            return Err(ReactError::Channel(Box::new(ChannelError::StaleDelivery {
                incarnation_id: self.incarnation_id().to_string(),
            })));
        }
        lifecycle.active_deliveries =
            lifecycle.active_deliveries.checked_add(1).ok_or_else(|| {
                ReactError::Channel(Box::new(ChannelError::Other(
                    "channel delivery lease capacity exhausted".to_string(),
                )))
            })?;
        Ok(ChannelDeliveryPermit {
            fence: self.clone(),
        })
    }

    /// 将当前世代标记为已退役,使其不再接受新的投递许可。
    pub(super) fn retire(&self) {
        self.lock_lifecycle().retired = true;
    }

    /// 先退役,再阻塞等待所有已受理的投递全部完成(活跃计数归零)。
    pub(super) async fn retire_and_wait(&self) {
        self.retire();
        loop {
            let settled = self.state.settled.notified();
            if !self.has_active_deliveries() {
                return;
            }
            settled.await;
        }
    }
}

/**
代表一次已受理投递的许可(lease)。Drop 时自动释放并递减活跃计数,
计数归零时唤醒正在等待结算的 `retire_and_wait`。
*/
pub(crate) struct ChannelDeliveryPermit {
    /// 签发该许可的投递围栏。
    fence: ChannelDeliveryFence,
}

impl Drop for ChannelDeliveryPermit {
    /**
    释放许可:活跃投递计数减一;若发生下溢则记录错误日志;
    若计数归零则通知正在等待的 `retire_and_wait`。
    */
    fn drop(&mut self) {
        let settled = {
            let mut lifecycle = self.fence.lock_lifecycle();
            match lifecycle.active_deliveries.checked_sub(1) {
                Some(active_deliveries) => {
                    lifecycle.active_deliveries = active_deliveries;
                    active_deliveries == 0
                }
                None => {
                    tracing::error!(
                        incarnation_id = %self.fence.incarnation_id(),
                        "channel delivery lease counter underflow"
                    );
                    false
                }
            }
        };
        if settled {
            self.fence.state.settled.notify_one();
        }
    }
}

/// 待发送到 IM 平台的消息
#[derive(Debug, Clone)]
pub struct OutboundMessage {
    /// 目标渠道
    pub channel_id: String,
    /// 目标标识(用户 openid / 群 id)
    pub to: String,
    /// 会话类型
    pub chat_type: ChatType,
    /// 回复文本
    pub text: String,
    /// 被回复的消息 ID(用于 at_reply)
    pub reply_to: Option<String>,
    /// 消息附件
    pub attachments: Vec<MessageAttachment>,
    /// 框架渠道投递所使用的进程本地世代权限对象。
    #[doc(hidden)]
    pub(super) delivery_fence: Option<ChannelDeliveryFence>,
}

impl OutboundMessage {
    /// 构造一条出站消息,不带回复引用、附件与投递围栏。
    pub fn new(
        channel_id: impl Into<String>,
        to: impl Into<String>,
        chat_type: ChatType,
        text: impl Into<String>,
    ) -> Self {
        Self {
            channel_id: channel_id.into(),
            to: to.into(),
            chat_type,
            text: text.into(),
            reply_to: None,
            attachments: Vec::new(),
            delivery_fence: None,
        }
    }

    /// 设置被回复的消息 ID,返回修改后的自身(builder 风格)。
    pub fn with_reply_to(mut self, reply_to: impl Into<String>) -> Self {
        self.reply_to = Some(reply_to.into());
        self
    }

    /// 添加附件
    pub fn with_attachments(mut self, attachments: Vec<MessageAttachment>) -> Self {
        self.attachments = attachments;
        self
    }

    /// 为该出站消息绑定一个投递围栏,返回修改后的自身(builder 风格)。
    pub(super) fn with_delivery_fence(mut self, fence: ChannelDeliveryFence) -> Self {
        self.delivery_fence = Some(fence);
        self
    }

    /// 若绑定了投递围栏,则尝试获取投递许可;未绑定围栏时返回 `Ok(None)`。
    pub(super) fn begin_delivery(&self) -> Result<Option<ChannelDeliveryPermit>> {
        self.delivery_fence
            .as_ref()
            .map(ChannelDeliveryFence::begin_delivery)
            .transpose()
    }
}

// ── ChannelPlugin 特征 ─────────────────────────────────────────────────────

/// 消息处理器 —— 将 IM 消息转发给 Agent 进行处理
#[async_trait]
pub trait MessageHandler: Send + Sync {
    /// 处理一条入站消息,返回一条出站消息
    async fn handle(&self, msg: InboundMessage) -> Result<OutboundMessage>;

    /// 将一条出站消息发送回 IM 平台(由 Channel 自身实现)
    async fn reply(&self, msg: OutboundMessage) -> Result<()>;

    /**
    在传输层停止接受消息、且其正在进行中的投递全部结算完成后,
    释放 handler 所持有的 Agent 与会话资源。
    无状态 handler 没有需要释放的资源。
    */
    async fn close(&self) -> Result<()> {
        Ok(())
    }

    /**
    流式 handle:产出逐段 `OutboundMessage`。

    默认实现:只 yield 一次 `handle` 的返回值(向后兼容,所有现有 impl 无需 override)。
    需要真流式(分段投递)的 handler override 此方法。
    */
    async fn handle_stream<'a>(
        &'a self,
        msg: InboundMessage,
    ) -> Result<BoxStream<'a, Result<OutboundMessage>>> {
        let out = self.handle(msg).await?;
        Ok(futures::stream::once(async move { Ok(out) }).boxed())
    }

    /**
    该 handler 是否会在会话取消后使一次被驱动的 Turn 完成结算。

    会话 reset 会等待已声明支持此行为的 handler 完成其正在进行的
    那次调用。旧版/自定义 handler 若未同时实现
    `handle_stream_with_cancel`,则保留此前"流被丢弃即终止"的行为。
    */
    fn settles_on_cancel(&self) -> bool {
        false
    }

    /// 以所属会话的取消生命周期运行一次流。
    async fn handle_stream_with_cancel<'a>(
        &'a self,
        msg: InboundMessage,
        _cancel: CancellationToken,
    ) -> Result<BoxStream<'a, Result<OutboundMessage>>> {
        self.handle_stream(msg).await
    }
}

/// 渠道能力描述
#[derive(Debug)]
pub struct ChannelCapabilities {
    /// 支持的会话类型
    pub chat_types: &'static [ChatType],
    /// 是否支持多媒体(图片/文件)
    pub supports_media: bool,
    /// 是否支持话题/线程(threads)
    pub supports_threads: bool,
}

/// IM 渠道插件的统一接口
#[async_trait]
pub trait ChannelPlugin: Send + Sync {
    /// 唯一渠道 ID:"qqbot" | "feishu"
    fn id(&self) -> &str;

    /// 人类可读的名称
    fn label(&self) -> &str {
        self.id()
    }

    /// 能力描述
    fn capabilities(&self) -> &ChannelCapabilities;

    /// 启动渠道(建立连接 / 启动服务)
    async fn start(&mut self, handler: Arc<dyn MessageHandler>) -> Result<()>;

    /// 停止渠道
    async fn stop(&mut self) -> Result<()>;

    /// 发送一条消息
    async fn send(&self, msg: OutboundMessage) -> Result<()>;

    /**
    健康检查 —— 验证渠道是否处于正常工作状态。

    默认返回 Ok(());子类可以重写以执行更复杂的检查
    (例如 token 有效性、连接状态等)。
    */
    async fn health_check(&self) -> Result<()> {
        Ok(())
    }
}

/// 渠道上下文
pub struct ChannelContext {
    /// 该上下文所使用的消息处理器
    pub handler: Arc<dyn MessageHandler>,
}

impl ChannelContext {
    /// 以给定的消息处理器构造渠道上下文。
    pub fn new(handler: Arc<dyn MessageHandler>) -> Self {
        Self { handler }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;

    /// 只 impl `handle` 的 dummy handler —— 验证默认 `handle_stream` 行为。
    struct OnceHandler {
        /// 固定的回复文本。
        reply_text: String,
    }

    #[async_trait]
    impl MessageHandler for OnceHandler {
        /// 以固定文本回复入站消息。
        async fn handle(&self, msg: InboundMessage) -> Result<OutboundMessage> {
            Ok(OutboundMessage::new(
                &msg.channel_id,
                msg.reply_target(),
                msg.chat_type,
                &self.reply_text,
            ))
        }
        /// 回复操作直接成功。
        async fn reply(&self, _msg: OutboundMessage) -> Result<()> {
            Ok(())
        }
    }

    /// 验证默认的流式处理恰好产出一条出站消息。
    #[tokio::test]
    async fn default_handle_stream_yields_exactly_one_outbound() -> Result<()> {
        let handler = OnceHandler {
            reply_text: "hello".into(),
        };
        let msg = InboundMessage::new("qq", "u1", "c1", ChatType::Direct, "hi", "m1");
        let mut stream = handler.handle_stream(msg).await?;

        // 第一条 = handle 的返回值
        let first = stream
            .next()
            .await
            .ok_or_else(|| ReactError::Other("stream yielded no item".to_string()))??;
        assert_eq!(first.text, "hello");
        assert_eq!(first.channel_id, "qq");
        assert_eq!(first.to, "c1");

        // 之后不再有(恰好 1 条)
        assert!(stream.next().await.is_none(), "default yields exactly one");
        Ok(())
    }

    /// 验证已退役的投递栅栏会等待租约释放并拒绝新的准入。
    #[tokio::test]
    async fn retired_delivery_fence_waits_for_lease_and_rejects_new_admission() -> Result<()> {
        let fence = ChannelDeliveryFence::new("incarnation-a".to_string());
        let permit = fence.begin_delivery()?;
        let retiring_fence = fence.clone();
        let retirement = tokio::spawn(async move {
            retiring_fence.retire_and_wait().await;
        });
        tokio::task::yield_now().await;
        if retirement.is_finished() {
            return Err(ReactError::Other(
                "delivery fence retired before its active lease settled".to_string(),
            ));
        }

        drop(permit);
        retirement.await.map_err(|error| {
            ReactError::Other(format!("delivery fence retirement task failed: {error}"))
        })?;
        if !fence.is_retired() {
            return Err(ReactError::Other(
                "delivery fence did not retain its retired state".to_string(),
            ));
        }
        if !matches!(
            fence.begin_delivery(),
            Err(ReactError::Channel(ref error))
                if matches!(
                    error.as_ref(),
                    ChannelError::StaleDelivery { incarnation_id }
                        if incarnation_id == "incarnation-a"
                )
        ) {
            return Err(ReactError::Other(
                "retired delivery fence did not return the typed stale-generation error"
                    .to_string(),
            ));
        }
        Ok(())
    }
}
