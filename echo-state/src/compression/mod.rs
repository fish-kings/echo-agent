//! 上下文压缩
//!
//! 维护对话历史，并在 token 超出限制时自动压缩，由 [`ContextManager`] 统一管理。
//!
//! 内置压缩策略（均实现 [`ContextCompressor`] trait）：
//! - [`compressor::SlidingWindowCompressor`]：滑动窗口，丢弃最早的 N 条消息
//! - [`compressor::SummaryCompressor`]：LLM 摘要，将旧消息压缩为一条系统摘要消息（stage4 P4.3 默认策略）

pub mod compressor;
pub mod horizon;
pub mod invariants;
pub mod levels;
pub mod verifier;

// 从 echo_core 重新导出，以保持向后兼容
pub use echo_core::compression::{
    CanonicalContext, CompressionCheckpoint, CompressionInput, CompressionOutput,
    ContextCompressor, ContextProjection, PreModelContextProjector, ProjectionContext,
    StructuredSummary, ToolPairFix, ToolPairFixType,
};

use crate::compression::compressor::SlidingWindowCompressor;
use echo_core::budget::TokenBudget;
use echo_core::error::Result;
use echo_core::llm::types::{ContentPart, Message, MessageContent, Role};
use echo_core::tokenizer::{HeuristicTokenizer, Tokenizer};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

const PROJECTION_ENVELOPE_PREFIX: &str = "<echo-agent-context-projection-v1>";

/// 判断一条消息是否携带了由框架管理的上下文投影信封（envelope）。
///
/// 调用方应使用该判断函数，而不是直接匹配内部的信封表示形式。
pub fn is_context_projection_message(message: &Message) -> bool {
    projection_envelope_text(message).is_some()
}

/// 若消息携带投影信封，返回其原始文本；否则返回 `None`。
fn projection_envelope_text(message: &Message) -> Option<&str> {
    match &message.content {
        MessageContent::Text(content) => content
            .starts_with(PROJECTION_ENVELOPE_PREFIX)
            .then_some(content.as_str()),
        MessageContent::Parts(parts) => parts.first().and_then(|part| match part {
            ContentPart::Text { text } if text.starts_with(PROJECTION_ENVELOPE_PREFIX) => {
                Some(text.as_str())
            }
            _ => None,
        }),
        MessageContent::Empty => None,
    }
}

/// 为给定 marker 生成投影信封的前缀文本（用于匹配/查找），marker 为空时返回 `None`。
fn projection_marker_prefix(marker: &str) -> Option<String> {
    if marker.is_empty() {
        return None;
    }
    serde_json::to_string(marker)
        .ok()
        .map(|encoded| format!("{PROJECTION_ENVELOPE_PREFIX}\nmarker={encoded}\n"))
}

/// 将被驱逐消息提升为长期记忆的回调 trait。
///
/// 当压缩从上下文中驱逐消息时，该 trait 允许提取关键事实并写入记忆 Store 以便后续召回。
/// 这就是所谓的 "L3 记忆提升"（Memory Promotion）机制。
pub trait MemoryPromoter: Send + Sync {
    /// 处理被驱逐的消息，并将关键事实提升到记忆中。
    fn promote(
        &self,
        evicted: &[Message],
    ) -> futures::future::BoxFuture<'_, Result<MemoryPromotionReceipt>>;
}

/// 一次记忆提升操作的持久化结果。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MemoryPromotionReceipt {
    /// 本次提交给记忆提升器的消息数量
    pub submitted: usize,
    /// 实际成功提升到记忆中的数量
    pub promoted: usize,
    /// 因重复而被去重、未再次写入的数量
    pub deduplicated: usize,
}

/// `Arc<dyn MemoryPromoter>` 的统一实现（透传给内部对象）。
impl MemoryPromoter for Arc<dyn MemoryPromoter> {
    /// 转发调用给内部持有的 `MemoryPromoter` 实现。
    fn promote(
        &self,
        evicted: &[Message],
    ) -> futures::future::BoxFuture<'_, Result<MemoryPromotionReceipt>> {
        (**self).promote(evicted)
    }
}

/// 用于在压缩后把受保护消息还原到接近原始位置所需的元数据。
struct ProtectedMessage {
    /// 受保护的消息本身
    message: Message,
    /// 该消息原本之后出现的可压缩消息数量。
    compressible_after: usize,
    /// 该消息原本之后出现的受保护消息数量。
    protected_after: usize,
}

/// `force_compress()` 返回的压缩统计信息
pub struct ForceCompressStats {
    /// 压缩前的消息总数
    pub before_count: usize,
    /// 压缩后的消息总数
    pub after_count: usize,
    /// 被驱逐的消息数量
    pub evicted: usize,
    /// 压缩前估算的 token 数
    pub before_tokens: usize,
    /// 压缩后估算的 token 数
    pub after_tokens: usize,
}

/// `ContextManager::prepare()` 的返回结果 —— 包含准备好的消息列表，
/// 以及在触发自动压缩时附带的压缩统计信息。
pub struct PrepareResult {
    /// 准备发送给 LLM 的消息列表。
    pub messages: Vec<Message>,
    /// 压缩统计信息，仅在发生自动压缩时才有值。
    pub compressed: Option<ForceCompressStats>,
    /// 用于审计、重放和恢复的压缩检查点。
    pub checkpoint: Option<CompressionCheckpoint>,
    /// 摘要校验结果（仅在生成了摘要时才有值）。
    pub verification: Option<verifier::SummaryVerification>,
}

/// 按消息角色划分的 token 明细。
///
/// 供 `/context` 命令展示上下文窗口预算的具体分布情况。
#[derive(Debug, Clone)]
pub struct TokenBreakdown {
    /// 系统消息（system）占用的 token 数
    pub system: usize,
    /// 用户消息（user）占用的 token 数
    pub user: usize,
    /// 助手消息（assistant）占用的 token 数
    pub assistant: usize,
    /// 工具消息（tool）占用的 token 数
    pub tool: usize,
    /// 对话历史摘要占用的 token 数
    pub summary: usize,
    /// 记忆相关上下文占用的 token 数
    pub memory: usize,
    /// 总 token 数
    pub total: usize,
    /// 模型的最大上下文窗口（若已知）
    pub max_context: Option<usize>,
    /// 当前配置的 token 上限
    pub token_limit: usize,
    /// 累计触发压缩的次数
    pub compression_count: u64,
    /// 累计压缩比率
    pub compression_ratio: f64,
}

impl TokenBreakdown {
    /// 将统计信息格式化为人类可读的进度条展示形式。
    pub fn format_bar(&self) -> String {
        let max = self.max_context.unwrap_or(self.total.max(1));
        let pct = |n: usize| -> f64 { if max == 0 { 0.0 } else { n as f64 / max as f64 } };

        let bar = |label: &str, n: usize| -> String {
            let frac = pct(n);
            let filled = (frac * 20.0) as usize;
            let bar_str: String = (0..20)
                .map(|i| if i < filled { '█' } else { '░' })
                .collect();
            format!(
                "  {:<12} {} {:>6} ({:>4.1}%)",
                label,
                bar_str,
                fmt_tokens(n),
                frac * 100.0
            )
        };

        let mut s = format!(
            "  Messages: {}  Tokens: {} / {} ({:.1}%)\n",
            self.system + self.user + self.assistant + self.tool + self.summary + self.memory,
            fmt_tokens(self.total),
            fmt_tokens(max),
            pct(self.total) * 100.0,
        );
        s.push_str(&bar("System:", self.system));
        s.push('\n');
        s.push_str(&bar("User:", self.user));
        s.push('\n');
        s.push_str(&bar("Assistant:", self.assistant));
        s.push('\n');
        s.push_str(&bar("Tool:", self.tool));
        s.push('\n');
        s.push_str(&bar("Summary:", self.summary));
        s.push('\n');
        s.push_str(&bar("Memory:", self.memory));
        if let Some(mc) = self.max_context {
            s.push_str(&format!("\n  Max context: {} tokens", fmt_tokens(mc)));
        }
        if self.compression_count > 0 {
            s.push_str(&format!(
                "\n  Compressions: {} | Ratio: {:.2}",
                self.compression_count, self.compression_ratio
            ));
        }
        s
    }
}

/// 将 token 数格式化为带 K/M 单位的简短字符串（例如 1500 -> "1.5K"）。
fn fmt_tokens(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

/// 用于可观测性的累计压缩指标。
///
/// 记录 [`ContextManager`] 生命周期内触发的压缩事件总数、节省的 token 数、
/// 被驱逐的消息数，以及各压缩策略的使用情况。
#[derive(Debug, Clone)]
pub struct CompressionMetrics {
    /// 触发压缩事件的总次数。
    pub total_compressions: u64,
    /// 所有压缩事件中，压缩前估算 token 数的总和。
    pub total_tokens_before: u64,
    /// 所有压缩事件中，压缩后估算 token 数的总和。
    pub total_tokens_after: u64,
    /// 所有压缩事件中，被驱逐的消息总数。
    pub total_messages_evicted: u64,
    /// 各压缩器类型名称的使用次数统计（例如 "SlidingWindow" → 3）。
    pub strategies_used: std::collections::HashMap<String, u64>,
}

impl CompressionMetrics {
    /// 创建一个所有计数均为零的初始指标实例。
    fn new() -> Self {
        Self {
            total_compressions: 0,
            total_tokens_before: 0,
            total_tokens_after: 0,
            total_messages_evicted: 0,
            strategies_used: std::collections::HashMap::new(),
        }
    }

    /// 所有压缩事件累计节省的 token 数。
    pub fn total_tokens_saved(&self) -> u64 {
        self.total_tokens_before
            .saturating_sub(self.total_tokens_after)
    }

    /// 压缩比率（0.0 表示没有节省，1.0 表示全部节省）。
    pub fn compression_ratio(&self) -> f64 {
        if self.total_tokens_before == 0 {
            return 0.0;
        }
        self.total_tokens_saved() as f64 / self.total_tokens_before as f64
    }

    /// 生成人类可读的摘要文本。
    pub fn report(&self) -> String {
        let strategies: Vec<String> = self
            .strategies_used
            .iter()
            .map(|(name, count)| format!("{}({})", name, count))
            .collect();
        format!(
            "CompressionMetrics: {} compressions, {} tokens saved ({:.1}%), {} messages evicted, strategies: [{}]",
            self.total_compressions,
            self.total_tokens_saved(),
            self.compression_ratio() * 100.0,
            self.total_messages_evicted,
            strategies.join(", "),
        )
    }

    /// 记录一次压缩事件。
    fn record(&mut self, stats: &ForceCompressStats, compressor_name: &str) {
        self.total_compressions += 1;
        self.total_tokens_before += stats.before_tokens as u64;
        self.total_tokens_after += stats.after_tokens as u64;
        self.total_messages_evicted += stats.evicted as u64;
        *self
            .strategies_used
            .entry(compressor_name.to_string())
            .or_insert(0) += 1;
    }
}

/// 上下文管理器：维护完整的对话历史，并在 token 超出限制时自动触发压缩。
///
/// # 典型用法
///
/// ```rust,no_run
/// use echo_core::error::Result;
/// use echo_core::llm::types::Message;
/// use echo_state::compression::compressor::SlidingWindowCompressor;
/// use echo_state::compression::{ContextCompressor, ContextManager};
///
/// # async fn example() -> Result<()> {
/// let mut ctx = ContextManager::builder(4096)
///     .compressor(SlidingWindowCompressor::new(20))
///     .build();
///
/// ctx.push(Message::system("You are an assistant".to_string()));
/// ctx.push(Message::user("Hello".to_string()));
///
/// // 每次调用 LLM 前调用 prepare()，自动压缩超出限制的消息
/// let result = ctx.prepare(None).await?;
/// let messages = result.messages;
/// # Ok(())
/// # }
/// ```
///
/// # SummaryCompressor 示例（stage4 P4.3 默认策略）
///
/// ```rust,no_run
/// use echo_core::llm::LlmClient;
/// use echo_state::compression::compressor::SummaryCompressor;
/// use echo_state::compression::{ContextCompressor, ContextManager};
/// use std::sync::Arc;
///
/// # async fn example(llm: Arc<dyn LlmClient>) {
/// let compressor = SummaryCompressor::new(llm, 8);
/// let mut ctx = ContextManager::builder(8192)
///     .compressor(compressor)
///     .build();
/// # }
/// ```
pub struct ContextManager {
    /// 当前维护的完整消息缓冲区
    messages: Vec<Message>,
    /// 已配置的压缩策略（若未配置则不会自动压缩）
    compressor: Option<Box<dyn ContextCompressor>>,
    /// 触发压缩的 token 上限
    token_limit: usize,
    /// 用于估算 token 数的分词器
    tokenizer: Arc<dyn Tokenizer>,
    /// 用于标识受保护消息（在压缩中存活）的内容标记。
    /// 任何内容包含这些标记之一的消息都会被排除在压缩之外。
    /// 用于静态的、由调用方定义且必须在压缩中存活的指令。
    protected_markers: Vec<String>,
    /// 最新消息会替代旧消息的受保护内容标记。
    ///
    /// 适用于动态上下文，例如子 Agent 调用简报：当前值必须在压缩中存活，
    /// 但保留历史值既浪费 token，又可能让模型接收到相互冲突的指令。
    replaceable_protected_markers: Vec<String>,
    /// 每个动态投影提供方最近一次发出的标记集合。
    projection_scopes: HashMap<String, HashSet<String>>,
    /// 可选的硬性消息数量上限（作为 OOM 防护的补充手段）。
    ///
    /// （stage4 P4.3）主要的 OOM 防护手段是 token 预算（`token_limit` /
    /// `budget`）；`max_messages` 是一种可选的补充防护，用于应对消息数量
    /// 异常增长的极端情况（例如大量不足以触发 token 阈值、但会拖慢渲染/
    /// 处理速度的小消息）。业界主流做法是以 token 驱动（OpenAI/Anthropic/
    /// Letta 均按 token 计费/开窗）；因此 `max_messages` 默认是 `None`
    /// （关闭状态），与这一主流做法保持一致。需要消息数量兜底的调用方，
    /// 可以通过 [`ContextManagerBuilder::max_messages`] 进行设置。
    max_messages: Option<usize>,
    /// 可选的、用于按百分比分配的 token 预算。
    /// 一旦设置，`prepare()` 会使用 budget.allocate() 而不是简单的 token_limit 比较。
    budget: Option<TokenBudget>,
    /// 用于可观测性的累计压缩指标。
    metrics: CompressionMetrics,
    /// 可选的可见性视界（visibility horizon）压缩器。
    /// 一旦设置，`prepare()` 会在主压缩器之前先执行一轮视界压缩预处理，
    /// 压缩活动窗口之外的工具调用轨迹。
    visibility_horizon: Option<horizon::VisibilityHorizonCompressor>,
    /// 可选的、将被驱逐消息提升为长期记忆的回调。
    /// 一旦设置，会在每次压缩后携带被驱逐的消息调用该回调。
    memory_promoter: Option<Arc<dyn MemoryPromoter>>,
    /// 可选的规范上下文来源，用于压缩后的重新注入。
    /// 一旦设置，若系统提示词、规则、技能注入在压缩中被驱逐，可以据此恢复。
    canonical_context: Option<CanonicalContext>,
}

impl ContextManager {
    /// 创建一个用于配置并构建 `ContextManager` 的构建器（builder）。
    pub fn builder(token_limit: usize) -> ContextManagerBuilder {
        ContextManagerBuilder {
            token_limit,
            compressor: None,
            initial_messages: Vec::new(),
            tokenizer: None,
            max_messages: None,
            budget: None,
            visibility_horizon: None,
            memory_promoter: None,
            canonical_context: None,
        }
    }

    /// 向上下文缓冲区追加一条消息。
    ///
    /// （stage4 P4.3）主要的 OOM 防护手段是 token 预算（`prepare()` 在估算
    /// token 数超过 `token_limit` 时触发压缩）。可选的 `max_messages` 上限
    /// （若通过 builder 设置）是一种补充兜底：一旦超出，会执行滑动窗口式
    /// 降级处理，保留系统消息 + 受保护消息 + 最近消息，丢弃中间最早的部分。
    /// 默认为 `None`（不设消息数量上限，以 token 驱动，与业界主流做法一致）。
    pub fn push(&mut self, message: Message) {
        if let Some(marker) = self.replaceable_marker_for(&message) {
            self.messages.retain(|existing| {
                existing
                    .content
                    .as_text_ref()
                    .is_none_or(|content| !content.contains(marker.as_str()))
            });
        }
        self.messages.push(message);

        // 可选的补充性硬上限（默认为 None，即关闭）。
        if let Some(cap) = self.max_messages
            && self.messages.len() > cap
        {
            self.apply_hard_cap(cap);
        }
    }

    /// 应用消息数量硬上限：保留系统消息、受保护消息和最近的消息；
    /// 丢弃中间最早的部分。仅在设置了 `max_messages` 且超出时触发。
    fn apply_hard_cap(&mut self, target: usize) {
        if self.messages.len() <= target {
            return;
        }

        // 找出受保护消息的下标（不应被删除）。
        let protected: std::collections::HashSet<usize> = self
            .messages
            .iter()
            .enumerate()
            .filter_map(|(i, msg)| {
                if self.is_protected(msg) {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();

        // 找到第一条非系统消息的位置
        let first_non_system = self
            .messages
            .iter()
            .position(|m| m.role != Role::System)
            .unwrap_or(0);

        // 保留：开头所有的系统消息 + 最后 `target` 条消息。
        // 丢弃中间最早的、非系统且非受保护的消息。
        let keep_count = self.messages.len() - target;
        let mut to_remove: Vec<usize> = Vec::new();
        let mut removed = 0;
        for i in first_non_system..self.messages.len() {
            if removed >= keep_count {
                break;
            }
            if !protected.contains(&i)
                && self
                    .messages
                    .get(i)
                    .is_some_and(|message| message.role != Role::System)
            {
                to_remove.push(i);
                removed += 1;
            }
        }
        // 倒序删除，以保证索引在删除过程中保持有效。
        for i in to_remove.into_iter().rev() {
            self.messages.remove(i);
        }
    }

    /// 批量追加消息
    pub fn push_many(&mut self, messages: impl IntoIterator<Item = Message>) {
        for message in messages {
            self.push(message);
        }
    }

    /// 返回当前缓冲区中的全部消息（未经压缩）
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    /// 判断下一次调用 `prepare()` 时是否即将触发压缩 —— 即已配置压缩器
    /// 且 token 预算已超出。
    ///
    /// （stage4 E1）供 `pre_compaction_flush` 使用，用于控制刷新用的 LLM
    /// 调用只在真正即将发生压缩时触发，而不是在每次 ReAct 迭代都触发。
    /// 该方法是对 `prepare()` 中 `needs_compression` 判断逻辑（mod.rs:1017-1025）
    /// 的非修改性预检查。相对预算路径存在轻微的过早/延迟触发是可以接受的——
    /// 该刷新本身是尽力而为（best-effort）的。
    pub fn should_compress(&self) -> bool {
        if self.compressor.is_none() {
            return false;
        }
        let estimated_tokens = Self::estimate_tokens(&self.messages, &*self.tokenizer);
        if let Some(ref budget) = self.budget {
            let allocation = budget.allocate(0, 0, estimated_tokens);
            allocation.needs_compression()
        } else {
            estimated_tokens > self.token_limit
        }
    }

    /// 替换内部消息缓冲区（用于从持久化存储中恢复对话）
    ///
    /// 消息列表应将系统提示词作为第一项（如果需要的话）。
    pub fn set_messages(&mut self, messages: Vec<Message>) {
        self.messages = messages;
        self.projection_scopes.clear();
    }

    /// 估算当前上下文的 token 数
    ///
    /// 使用配置的 [`Tokenizer`] 实现（默认使用 [`HeuristicTokenizer`]，区分 ASCII/中日韩字符）。
    pub fn token_estimate(&self) -> usize {
        Self::estimate_tokens(&self.messages, &*self.tokenizer)
    }

    /// 获取当前 Tokenizer
    pub fn tokenizer(&self) -> &dyn Tokenizer {
        &*self.tokenizer
    }

    /// 动态替换 Tokenizer
    pub fn set_tokenizer(&mut self, tokenizer: Arc<dyn Tokenizer>) {
        self.tokenizer = tokenizer;
    }

    /// 原子性地替换当前生效的上下文限制及其派生的 token 预算。
    pub fn set_token_limit(&mut self, token_limit: usize, budget: Option<TokenBudget>) {
        self.token_limit = token_limit;
        self.budget = budget;
    }

    /// 清空上下文缓冲区（保留已配置的压缩器和保护标记）
    pub fn clear(&mut self) {
        self.messages.clear();
        self.projection_scopes.clear();
    }

    /// 注册一个用于保护消息不被压缩的内容标记。
    ///
    /// 任何内容中包含该标记字符串的消息都会在压缩过程中被排除。
    /// 用于静态的、由调用方定义且在上下文压缩过程中不能被驱逐的指令。
    ///
    /// # 示例
    /// ```rust,no_run
    /// # use echo_state::compression::ContextManager;
    /// let mut ctx = ContextManager::builder(4096).build();
    /// ctx.add_protected_marker("[critical_policy]".to_string());
    /// ```
    pub fn add_protected_marker(&mut self, marker: String) {
        if !marker.is_empty() && !self.protected_markers.contains(&marker) {
            self.protected_markers.push(marker);
        }
    }

    /// 注册一个具有"最新值替代旧值"语义的受保护标记。
    ///
    /// 推入一条包含该标记的消息时，会先移除所有含有相同标记的旧消息，再追加新值。
    pub fn add_replaceable_protected_marker(&mut self, marker: String) {
        if !marker.is_empty() && !self.replaceable_protected_markers.contains(&marker) {
            self.replaceable_protected_markers.push(marker);
        }
    }

    /// 在系统消息与历史消息的边界处，替换带标记的模型上下文。
    ///
    /// 投影消息会被包裹上框架保留的信封（envelope）。每个给定的 marker 只替换
    /// 它自己之前的值。若某个投影的 message 为 `None`，则视为对该 marker 的
    /// 墓碑标记（删除）。未提及的投影保持不变，使独立的上下文来源可以安全地
    /// 各自更新。
    pub fn apply_projections(&mut self, projections: &[ContextProjection]) {
        let mut replacements = Vec::new();
        for projection in projections {
            if projection
                .message
                .as_ref()
                .is_some_and(|message| self.projection_matches(&projection.marker, message))
            {
                continue;
            }
            self.remove_projection(projection.marker.as_str());
            if let Some(message) = projection.message.clone()
                && !projection.marker.is_empty()
            {
                replacements.push(Self::wrap_projection_message(
                    message,
                    projection.marker.as_str(),
                ));
            }
        }

        let boundary = self
            .messages
            .iter()
            .position(|message| message.role != Role::System)
            .unwrap_or(self.messages.len());
        self.messages.splice(boundary..boundary, replacements);
    }

    /// Apply the complete marker set owned by one dynamic provider.
    ///
    /// Markers emitted by this scope on its previous call but omitted now are
    /// removed. Projections owned by other scopes or direct callers remain.
    pub fn apply_projection_scope(
        &mut self,
        scope: impl Into<String>,
        projections: &[ContextProjection],
    ) {
        let scope = scope.into();
        if scope.is_empty() {
            return;
        }
        let current: HashSet<String> = projections
            .iter()
            .filter(|projection| !projection.marker.is_empty())
            .map(|projection| projection.marker.clone())
            .collect();
        if let Some(previous) = self.projection_scopes.remove(scope.as_str()) {
            for marker in previous.difference(&current) {
                self.remove_projection(marker);
            }
        }
        self.apply_projections(projections);
        if !current.is_empty() {
            self.projection_scopes.insert(scope, current);
        }
    }

    /// Replace one framework-owned projection by marker.
    pub fn replace_projection(&mut self, marker: impl Into<String>, message: Option<Message>) {
        let marker = marker.into();
        if marker.is_empty() {
            return;
        }
        self.apply_projections(&[ContextProjection { marker, message }]);
    }

    /// Replace one framework-owned projection at the tail of current history.
    ///
    /// Tail placement is appropriate for per-turn dynamic context because it
    /// preserves the longest stable prompt prefix for provider KV caches.
    pub fn replace_tail_projection(&mut self, marker: impl Into<String>, message: Option<Message>) {
        let marker = marker.into();
        if marker.is_empty() {
            return;
        }
        if message
            .as_ref()
            .is_some_and(|current| self.projection_matches(&marker, current))
        {
            return;
        }
        self.remove_projection(marker.as_str());
        if let Some(message) = message {
            self.push(Self::wrap_projection_message(message, marker.as_str()));
        }
    }

    /// Remove one framework-owned projection by exact marker.
    pub fn remove_projection(&mut self, marker: &str) -> usize {
        let Some(expected) = projection_marker_prefix(marker) else {
            return 0;
        };
        let before = self.messages.len();
        self.messages.retain(|message| {
            projection_envelope_text(message)
                .is_none_or(|envelope| !envelope.starts_with(expected.as_str()))
        });
        before.saturating_sub(self.messages.len())
    }

    /// Count messages currently protected by registered content markers.
    pub fn protected_message_count(&self) -> usize {
        self.messages
            .iter()
            .filter(|message| self.is_protected(message))
            .count()
    }

    /// Estimate tokens currently pinned against compression.
    pub fn protected_token_estimate(&self) -> usize {
        self.messages
            .iter()
            .filter(|message| self.is_protected(message))
            .map(|message| {
                let text = message.content.as_text().unwrap_or_default();
                self.tokenizer.count_tokens(text.as_str())
            })
            .sum()
    }

    /// Return whether the context owns a projection with exactly this marker.
    pub fn has_projection(&self, marker: &str) -> bool {
        let Some(expected) = projection_marker_prefix(marker) else {
            return false;
        };
        self.messages.iter().any(|message| {
            projection_envelope_text(message)
                .is_some_and(|envelope| envelope.starts_with(expected.as_str()))
        })
    }

    /// Check if a message is protected from compression.
    fn is_protected(&self, message: &Message) -> bool {
        if is_context_projection_message(message) {
            return true;
        }
        if let Some(content) = message.content.as_text() {
            self.protected_markers
                .iter()
                .chain(self.replaceable_protected_markers.iter())
                .any(|marker| content.contains(marker))
        } else {
            false
        }
    }

    fn replaceable_marker_for(&self, message: &Message) -> Option<String> {
        let content = message.content.as_text_ref()?;
        self.replaceable_protected_markers
            .iter()
            .find(|marker| content.contains(marker.as_str()))
            .cloned()
    }

    fn projection_matches(&self, marker: &str, message: &Message) -> bool {
        let Some(expected_prefix) = projection_marker_prefix(marker) else {
            return false;
        };
        let expected = Self::wrap_projection_message(message.clone(), marker);
        let Ok(expected_value) = serde_json::to_value(expected) else {
            return false;
        };
        let matching: Vec<_> = self
            .messages
            .iter()
            .filter(|existing| {
                projection_envelope_text(existing)
                    .is_some_and(|envelope| envelope.starts_with(expected_prefix.as_str()))
            })
            .collect();
        matching.len() == 1
            && matching.first().is_some_and(|existing| {
                serde_json::to_value(existing)
                    .ok()
                    .is_some_and(|value| value == expected_value)
            })
    }

    fn wrap_projection_message(mut message: Message, marker: &str) -> Message {
        let encoded_marker =
            serde_json::to_string(marker).unwrap_or_else(|_| "\"invalid-marker\"".to_string());
        let envelope = format!("{PROJECTION_ENVELOPE_PREFIX}\nmarker={encoded_marker}\n");
        match &mut message.content {
            MessageContent::Text(content) => {
                *content = format!("{envelope}{content}");
            }
            MessageContent::Parts(parts) => {
                parts.insert(0, ContentPart::Text { text: envelope });
            }
            MessageContent::Empty => {
                message.content = MessageContent::Text(envelope);
            }
        }
        message
    }

    /// Split messages into (compressible, protected_metadata).
    ///
    /// Protected messages are removed from the compressible set and will be
    /// re-inserted at their original relative positions after compression.
    fn split_protected(&self, messages: Vec<Message>) -> (Vec<Message>, Vec<ProtectedMessage>) {
        let mut compressible = Vec::new();
        let mut protected: Vec<(usize, Message)> = Vec::new();
        let mut compressible_seen = 0usize;

        for msg in messages {
            if self.is_protected(&msg) {
                protected.push((compressible_seen, msg));
            } else {
                compressible.push(msg);
                compressible_seen += 1;
            }
        }

        let total_compressible = compressible.len();
        let total_protected = protected.len();
        let protected = protected
            .into_iter()
            .enumerate()
            .map(|(idx, (compressible_before, message))| ProtectedMessage {
                message,
                compressible_after: total_compressible.saturating_sub(compressible_before),
                protected_after: total_protected.saturating_sub(idx + 1),
            })
            .collect();

        (compressible, protected)
    }

    /// Merge protected messages back into the compressed output.
    ///
    /// Protected messages are re-inserted near their original relative positions.
    /// We restore from the tail so each message can reserve the amount of trailing
    /// conversation that originally followed it.
    fn merge_protected(compressed: Vec<Message>, protected: Vec<ProtectedMessage>) -> Vec<Message> {
        if protected.is_empty() {
            return compressed;
        }

        let mut result = compressed;
        for protected_msg in protected.into_iter().rev() {
            let trailing_slots = protected_msg.compressible_after + protected_msg.protected_after;
            let system_boundary = result
                .iter()
                .take_while(|message| message.role == Role::System)
                .count();
            let desired = result.len().saturating_sub(trailing_slots);
            let insert_at = if protected_msg.message.role == Role::System {
                desired.min(system_boundary)
            } else {
                desired.max(system_boundary)
            };
            result.insert(insert_at, protected_msg.message);
        }
        result
    }

    /// Dynamically replace the compressor without affecting the existing message buffer
    pub fn set_compressor(&mut self, compressor: impl ContextCompressor + 'static) {
        self.compressor = Some(Box::new(compressor));
    }

    /// Set or replace the visibility horizon compressor.
    ///
    /// When configured, `prepare()` runs horizon compaction as a pre-processing
    /// pass before the main compressor, compacting tool traces beyond the
    /// active plan window.
    pub fn set_visibility_horizon(&mut self, compressor: horizon::VisibilityHorizonCompressor) {
        self.visibility_horizon = Some(compressor);
    }

    /// Remove the visibility horizon compressor.
    pub fn remove_visibility_horizon(&mut self) {
        self.visibility_horizon = None;
    }

    /// Set or replace the memory promoter.
    ///
    /// When set, the promoter receives evicted messages after each
    /// compression pass, enabling L3 memory promotion.
    pub fn set_memory_promoter(&mut self, promoter: Arc<dyn MemoryPromoter>) {
        self.memory_promoter = Some(promoter);
    }

    /// Remove the memory promoter.
    pub fn remove_memory_promoter(&mut self) {
        self.memory_promoter = None;
    }

    /// Run memory promotion and tool-call sanitization on the internal buffer.
    ///
    /// Shared by `prepare()` and `force_compress*()` to ensure consistent
    /// post-compression processing: evicted facts → long-term memory, and
    /// broken tool-call pairs → fixed.
    ///
    /// Returns the number of evicted messages sent to the promoter.
    async fn promote_and_sanitize(&mut self, evicted_messages: &[Message]) -> Result<usize> {
        // ── Memory promotion ──
        let count = if let Some(ref promoter) = self.memory_promoter {
            if !evicted_messages.is_empty() {
                promoter.promote(evicted_messages).await?.promoted
            } else {
                0
            }
        } else {
            0
        };

        // ── Tool-call sanitization ──
        let (sanitized, _fixes) = sanitize_tool_call_pairing(&self.messages);
        self.messages = sanitized;

        Ok(count)
    }

    /// Set canonical context sources for re-injection after compression.
    ///
    /// When set, `prepare()` will check if compression evicted critical context
    /// (system prompt, rules, skill injections) and re-inject them if needed.
    pub fn set_canonical_context(&mut self, context: CanonicalContext) {
        self.canonical_context = Some(context);
    }

    /// Update only the canonical system prompt while preserving rules and skills.
    pub fn set_canonical_system_prompt(&mut self, system_prompt: Option<String>) {
        if let Some(context) = &mut self.canonical_context {
            context.system_prompt = system_prompt;
        }
    }

    /// Remove the canonical context.
    pub fn remove_canonical_context(&mut self) {
        self.canonical_context = None;
    }

    /// Re-inject canonical context if compression removed critical components.
    ///
    /// Called automatically at the end of `prepare()` when compression occurs.
    fn reinject_canonical_context(&mut self) {
        let Some(ref canonical) = self.canonical_context else {
            return;
        };

        // Restore the exact canonical prompt even when another system message
        // occupies the system region.
        if let Some(ref prompt) = canonical.system_prompt {
            let has_canonical_system = self.messages.iter().any(|message| {
                message.role == Role::System
                    && message.content.as_text_ref() == Some(prompt.as_str())
            });
            if !has_canonical_system {
                self.messages.insert(0, Message::system(prompt.clone()));
                tracing::debug!("Re-injected system prompt from canonical context");
            }
        }

        // Inject canonical context messages (system prompt, rules, skills).
        //
        // IMPORTANT: insert at the end of the system region (sys_end), NOT at a
        // fixed position like 1. Inserting at pos=1 shifts the entire conversation
        // history, which invalidates the prompt cache prefix on every compression
        // cycle. By inserting at sys_end (the boundary between system messages and
        // history), we keep the history segment's byte positions stable, preserving
        // both Anthropic explicit cache breakpoints and OpenAI automatic prefix
        // caches.
        if let Some(msgs) = canonical.to_reinjection_messages() {
            // Find the end of the system message region (first non-System role).
            let sys_end = self
                .messages
                .iter()
                .position(|m| m.role != Role::System)
                .unwrap_or(self.messages.len());

            // Dedup: skip canonical messages whose text already exists in the
            // current system region (prevents accumulation on repeated compression).
            let existing_texts: std::collections::HashSet<String> = self.messages[..sys_end]
                .iter()
                .filter_map(|m| m.content.as_text())
                .collect();
            let to_insert: Vec<String> = msgs
                .into_iter()
                .filter(|msg| !existing_texts.contains(msg.as_str()))
                .collect();

            // Insert at sys_end (tight against history), keeping history stable.
            // Iterate in reverse so each message lands at the correct offset.
            for msg in to_insert.into_iter().rev() {
                self.messages.insert(sys_end, Message::system(msg));
            }
        }
    }

    fn finalize_checkpoint(
        &self,
        checkpoint: Option<CompressionCheckpoint>,
    ) -> Option<CompressionCheckpoint> {
        checkpoint.map(|mut value| {
            value.retained_count = self.messages.len();
            value.token_after = self.token_estimate();
            value
        })
    }

    /// Remove the compressor, reverting to unlimited mode
    pub fn remove_compressor(&mut self) {
        self.compressor = None;
    }

    /// Whether a compressor is configured
    pub fn has_compressor(&self) -> bool {
        self.compressor.is_some()
    }

    /// Get cumulative compression metrics for observability.
    ///
    /// Tracks total compressions, tokens saved, messages evicted, and
    /// which strategies were used over the lifetime of this `ContextManager`.
    pub fn compression_metrics(&self) -> &CompressionMetrics {
        &self.metrics
    }

    /// Reset cumulative compression metrics to zero.
    pub fn reset_compression_metrics(&mut self) {
        self.metrics = CompressionMetrics::new();
    }

    /// Compute a token breakdown by message role.
    ///
    /// Useful for `/context` visualization — shows how context window budget is
    /// distributed across system prompts, user messages, tool outputs, etc.
    pub fn token_breakdown(&self, max_context: Option<usize>) -> TokenBreakdown {
        let tokenizer = &*self.tokenizer;
        let mut system = 0;
        let mut user = 0;
        let mut assistant = 0;
        let mut tool = 0;
        let mut summary = 0;
        let mut memory = 0;

        for msg in &self.messages {
            let text = msg.content.as_text().unwrap_or_default();
            let tokens = tokenizer.count_tokens(&text);
            if text.contains("[memory_context]")
                || text.contains("[Relevant historical memories]")
                || text.contains("[Related historical memories]")
            {
                memory += tokens;
                continue;
            }
            match msg.role {
                Role::System => {
                    if text.contains("[对话历史摘要]") {
                        summary += tokens;
                    } else {
                        system += tokens;
                    }
                }
                Role::User => user += tokens,
                Role::Assistant => assistant += tokens,
                Role::Tool => tool += tokens,
                _ => {}
            }
        }

        let total = system + user + assistant + tool + summary + memory;
        let compression_ratio = self.metrics.compression_ratio();

        TokenBreakdown {
            system,
            user,
            assistant,
            tool,
            summary,
            memory,
            total,
            max_context,
            token_limit: self.token_limit,
            compression_count: self.metrics.total_compressions,
            compression_ratio,
        }
    }

    /// Force-compress the context, regardless of whether the current token count exceeds the limit.
    ///
    /// - If a compressor is configured, use it;
    /// - Otherwise, temporarily use `SlidingWindowCompressor::new(fallback_window)`.
    ///
    /// Protected messages are excluded from compression and preserved.
    pub async fn force_compress(
        &mut self,
        fallback_window: usize,
    ) -> Result<(ForceCompressStats, Option<CompressionCheckpoint>)> {
        let before_count = self.messages.len();
        let before_tokens = self.token_estimate();
        let original_messages = self.messages.clone();

        let (compressible, protected) = self.split_protected(self.messages.clone());

        let output = if let Some(compressor) = &self.compressor {
            compressor
                .compress(CompressionInput {
                    messages: compressible,
                    token_limit: self.token_limit,
                    current_query: None,
                    focus_instructions: None,
                    cancel_token: None,
                    tokenizer: Some(self.tokenizer.clone()),
                })
                .await?
        } else {
            SlidingWindowCompressor::new(fallback_window)
                .compress(CompressionInput {
                    messages: compressible,
                    token_limit: self.token_limit,
                    current_query: None,
                    focus_instructions: None,
                    cancel_token: None,
                    tokenizer: Some(self.tokenizer.clone()),
                })
                .await?
        };

        let checkpoint = output
            .checkpoint
            .map(|cp| cp.with_protected_count(protected.len()));

        let evicted_messages = output.evicted;
        let evicted = evicted_messages.len();
        self.messages = Self::merge_protected(output.messages, protected);

        // ── Memory promotion + sanitize ──
        let memory_promotion_count = match self.promote_and_sanitize(&evicted_messages).await {
            Ok(count) => count,
            Err(error) => {
                self.messages = original_messages;
                return Err(error);
            }
        };

        let checkpoint =
            checkpoint.map(|cp| cp.with_memory_promotion_count(memory_promotion_count));

        if self.canonical_context.is_some() {
            self.reinject_canonical_context();
        }
        let checkpoint = self.finalize_checkpoint(checkpoint);

        let stats = ForceCompressStats {
            before_count,
            after_count: self.messages.len(),
            evicted,
            before_tokens,
            after_tokens: self.token_estimate(),
        };
        let name = if self.compressor.is_some() {
            self.compressor
                .as_ref()
                .map(|c| c.name())
                .unwrap_or("unknown")
        } else {
            "SlidingWindow(fallback)"
        };
        self.metrics.record(&stats, name);
        Ok((stats, checkpoint))
    }

    /// Force-compress with user-provided focus instructions.
    ///
    /// The focus instructions are passed via `CompressionInput::focus_instructions`,
    /// allowing LLM-based compressors (Summary, IncrementalSummary, Adaptive L4) to
    /// prioritize specific topics in their summaries.
    pub async fn force_compress_with_focus(
        &mut self,
        focus_instructions: &str,
        fallback_window: usize,
    ) -> Result<(ForceCompressStats, Option<CompressionCheckpoint>)> {
        let before_count = self.messages.len();
        let before_tokens = self.token_estimate();
        let original_messages = self.messages.clone();

        let (compressible, protected) = self.split_protected(self.messages.clone());

        let output = if let Some(compressor) = &self.compressor {
            compressor
                .compress(CompressionInput {
                    messages: compressible,
                    token_limit: self.token_limit,
                    current_query: None,
                    focus_instructions: Some(focus_instructions.to_string()),
                    cancel_token: None,
                    tokenizer: Some(self.tokenizer.clone()),
                })
                .await?
        } else {
            SlidingWindowCompressor::new(fallback_window)
                .compress(CompressionInput {
                    messages: compressible,
                    token_limit: self.token_limit,
                    current_query: None,
                    focus_instructions: Some(focus_instructions.to_string()),
                    cancel_token: None,
                    tokenizer: Some(self.tokenizer.clone()),
                })
                .await?
        };

        let checkpoint = output
            .checkpoint
            .map(|cp| cp.with_protected_count(protected.len()));

        let evicted_messages = output.evicted;
        let evicted = evicted_messages.len();
        self.messages = Self::merge_protected(output.messages, protected);

        // ── Memory promotion + sanitize ──
        let memory_promotion_count = match self.promote_and_sanitize(&evicted_messages).await {
            Ok(count) => count,
            Err(error) => {
                self.messages = original_messages;
                return Err(error);
            }
        };

        let checkpoint =
            checkpoint.map(|cp| cp.with_memory_promotion_count(memory_promotion_count));

        if self.canonical_context.is_some() {
            self.reinject_canonical_context();
        }
        let checkpoint = self.finalize_checkpoint(checkpoint);

        let stats = ForceCompressStats {
            before_count,
            after_count: self.messages.len(),
            evicted,
            before_tokens,
            after_tokens: self.token_estimate(),
        };
        let name = if self.compressor.is_some() {
            self.compressor
                .as_ref()
                .map(|c| c.name())
                .unwrap_or("unknown")
        } else {
            "SlidingWindow(fallback)"
        };
        self.metrics.record(&stats, name);
        Ok((stats, checkpoint))
    }

    /// Force-compress using a **specific compressor**, without affecting the currently installed compressor config.
    ///
    /// Suitable for temporary strategy overrides like `/compress sliding 10`.
    pub async fn force_compress_with(
        &mut self,
        compressor: &dyn ContextCompressor,
    ) -> Result<(ForceCompressStats, Option<CompressionCheckpoint>)> {
        let before_count = self.messages.len();
        let before_tokens = self.token_estimate();
        let original_messages = self.messages.clone();

        let (compressible, protected) = self.split_protected(self.messages.clone());

        let output = compressor
            .compress(CompressionInput {
                messages: compressible,
                token_limit: self.token_limit,
                current_query: None,
                focus_instructions: None,
                cancel_token: None,
                tokenizer: Some(self.tokenizer.clone()),
            })
            .await?;

        let checkpoint = output
            .checkpoint
            .map(|cp| cp.with_protected_count(protected.len()));

        let evicted_messages = output.evicted;
        let evicted = evicted_messages.len();
        self.messages = Self::merge_protected(output.messages, protected);

        // ── Memory promotion + sanitize ──
        let memory_promotion_count = match self.promote_and_sanitize(&evicted_messages).await {
            Ok(count) => count,
            Err(error) => {
                self.messages = original_messages;
                return Err(error);
            }
        };

        let checkpoint =
            checkpoint.map(|cp| cp.with_memory_promotion_count(memory_promotion_count));

        if self.canonical_context.is_some() {
            self.reinject_canonical_context();
        }
        let checkpoint = self.finalize_checkpoint(checkpoint);

        let stats = ForceCompressStats {
            before_count,
            after_count: self.messages.len(),
            evicted,
            before_tokens,
            after_tokens: self.token_estimate(),
        };
        self.metrics.record(&stats, compressor.name());
        Ok((stats, checkpoint))
    }

    /// Update the system message content
    ///
    /// Typically called when `add_skill()` injects extra system prompts:
    /// finds the first message with role == "system" and replaces its content;
    /// if no system message exists, inserts one at the head of the queue.
    pub fn update_system(&mut self, new_system_prompt: String) {
        if let Some(msg) = self.messages.iter_mut().find(|m| m.role == Role::System) {
            msg.content = MessageContent::Text(new_system_prompt);
        } else {
            self.messages.insert(0, Message::system(new_system_prompt));
        }
    }

    /// Prepare the list of messages to send to the LLM.
    ///
    /// When the estimated token count exceeds `token_limit` and a compressor is configured, automatically trigger compression and update the internal buffer.
    /// The compressed messages replace the original buffer.
    ///
    /// Protected messages (containing registered markers, e.g. `<skill_content>`) are
    /// excluded from compression and re-inserted after system messages.
    ///
    /// `current_query` protects the active user task from losing focus during
    /// summary-based compression.
    ///
    /// Returns a [`PrepareResult`] containing the prepared messages and optional
    /// compression stats (populated only when auto-compression was triggered).
    pub async fn prepare(&mut self, current_query: Option<&str>) -> Result<PrepareResult> {
        self.prepare_with_cancel(current_query, 0, None).await
    }

    /// Prepare model input while propagating one invocation cancellation token
    /// through every nested compression provider call.
    pub async fn prepare_with_cancel(
        &mut self,
        current_query: Option<&str>,
        tool_tokens: usize,
        cancel_token: Option<echo_core::compression::CancellationToken>,
    ) -> Result<PrepareResult> {
        // ── Snapshot original messages for verification ──
        let original_messages = self.messages.clone();

        // ── Pre-compression: Visibility Horizon pass ──────────────────
        // Compact tool traces beyond the active window before the main
        // compressor runs. This reduces the token count so the main
        // compressor may not need to fire at all.
        if let Some(ref horizon_compressor) = self.visibility_horizon {
            let before = self.messages.len();
            let horizon_original = self.messages.clone();
            let horizon_input = CompressionInput {
                messages: std::mem::take(&mut self.messages),
                token_limit: self.token_limit,
                current_query: current_query.map(String::from),
                focus_instructions: None,
                cancel_token: cancel_token.clone(),
                tokenizer: Some(self.tokenizer.clone()),
            };
            match horizon_compressor.compress(horizon_input).await {
                Ok(output) => {
                    let compacted = before.saturating_sub(output.messages.len());
                    let evicted_count = output.evicted.len();
                    if evicted_count > 0
                        && let Some(ref promoter) = self.memory_promoter
                    {
                        promoter.promote(&output.evicted).await?;
                    }
                    if compacted > 0 {
                        tracing::debug!(
                            before_messages = before,
                            after_messages = output.messages.len(),
                            evicted = evicted_count,
                            "VisibilityHorizon pre-compaction applied"
                        );
                        self.metrics.record(
                            &ForceCompressStats {
                                before_count: before,
                                after_count: output.messages.len(),
                                evicted: output.evicted.len(),
                                before_tokens: 0,
                                after_tokens: 0,
                            },
                            "VisibilityHorizon",
                        );
                    }
                    self.messages = output.messages;
                }
                Err(e) => {
                    self.messages = horizon_original;
                    tracing::warn!(error = %e, "VisibilityHorizon pre-compaction failed, continuing with original messages");
                    // Don't fail prepare — horizon is best-effort
                }
            }
        }

        let system_tokens = self
            .messages
            .iter()
            .filter(|message| message.role == Role::System)
            .fold(0usize, |total, message| {
                total.saturating_add(message.content.estimated_tokens(&*self.tokenizer))
            });
        let conversation_tokens = self
            .messages
            .iter()
            .filter(|message| message.role != Role::System)
            .fold(0usize, |total, message| {
                total.saturating_add(message.content.estimated_tokens(&*self.tokenizer))
            });
        let estimated_tokens = system_tokens.saturating_add(conversation_tokens);

        // Compute effective token limit once so primary compression and any
        // verifier fallback obey the same budget-aware allowance.
        let effective_limit = if let Some(ref budget) = self.budget {
            budget
                .total_window()
                .saturating_sub(budget.output_budget())
                .saturating_sub(budget.safety_budget())
                .saturating_sub(tool_tokens)
        } else {
            self.token_limit
        };

        let needs_compression = if let Some(ref budget) = self.budget {
            let allocation = budget.allocate(system_tokens, tool_tokens, conversation_tokens);
            allocation.needs_compression()
        } else {
            estimated_tokens > self.token_limit
        };

        let (compressed, mut combined_checkpoint) = if let Some(compressor) = &self.compressor
            && needs_compression
        {
            let before_count = self.messages.len();
            let before_tokens = self.token_estimate();
            let compressor_name = compressor.name();

            tracing::debug!(
                before_messages = before_count,
                before_tokens,
                token_limit = self.token_limit,
                compressor = compressor_name,
                "Auto-compression triggered"
            );
            let start = std::time::Instant::now();

            // Use the budget-aware effective token limit computed above.

            let owned = std::mem::take(&mut self.messages);
            let (compressible, protected) = self.split_protected(owned);
            let protected_tokens = protected.iter().fold(0usize, |total, value| {
                total.saturating_add(value.message.content.estimated_tokens(&*self.tokenizer))
            });
            let compressor_limit = effective_limit.saturating_sub(protected_tokens);

            let compress_result = compressor
                .compress(CompressionInput {
                    messages: compressible.clone(),
                    token_limit: compressor_limit,
                    current_query: current_query.map(String::from),
                    focus_instructions: None,
                    cancel_token: cancel_token.clone(),
                    tokenizer: Some(self.tokenizer.clone()),
                })
                .await;

            match compress_result {
                Ok(output) => {
                    let evicted_messages = output.evicted;
                    let evicted = evicted_messages.len();
                    let compressor_checkpoint = output.checkpoint;
                    let protected_count = protected.len();
                    self.messages = Self::merge_protected(output.messages, protected);

                    // ── L3 Memory Promotion ──
                    // If a memory promoter is configured, pass evicted messages
                    // so key facts can be extracted and stored for later recall.
                    let memory_promotion_count = if let Some(ref promoter) = self.memory_promoter {
                        if !evicted_messages.is_empty() {
                            match promoter.promote(&evicted_messages).await {
                                Ok(receipt) => receipt.promoted,
                                Err(error) => {
                                    self.messages = original_messages.clone();
                                    return Err(error);
                                }
                            }
                        } else {
                            0
                        }
                    } else {
                        0
                    };

                    let after_tokens = self.token_estimate();
                    let stats = ForceCompressStats {
                        before_count,
                        after_count: self.messages.len(),
                        evicted,
                        before_tokens,
                        after_tokens,
                    };
                    let elapsed = start.elapsed();
                    self.metrics.record(&stats, compressor_name);

                    tracing::info!(
                        compressor = compressor_name,
                        before_messages = stats.before_count,
                        after_messages = stats.after_count,
                        before_tokens = stats.before_tokens,
                        after_tokens = stats.after_tokens,
                        evicted = stats.evicted,
                        saved_tokens = stats.before_tokens.saturating_sub(stats.after_tokens),
                        elapsed_ms = elapsed.as_millis() as u64,
                        "Compression complete"
                    );

                    // Assemble the checkpoint from the compressor's output
                    let compressed_checkpoint = compressor_checkpoint.map(|cp| {
                        cp.with_protected_count(protected_count)
                            .with_memory_promotion_count(memory_promotion_count)
                    });

                    (Some(stats), compressed_checkpoint)
                }
                Err(e) => {
                    // Primary compressor failed — fall back to SlidingWindow as safety net
                    // to avoid losing all non-protected messages.
                    tracing::warn!(
                        error = %e,
                        "Primary compressor failed, falling back to SlidingWindowCompressor"
                    );
                    let fallback = SlidingWindowCompressor::new(40);
                    match fallback
                        .compress(CompressionInput {
                            messages: compressible,
                            token_limit: compressor_limit,
                            current_query: current_query.map(String::from),
                            focus_instructions: None,
                            cancel_token: cancel_token.clone(),
                            tokenizer: Some(self.tokenizer.clone()),
                        })
                        .await
                    {
                        Ok(fb_output) => {
                            self.messages = Self::merge_protected(fb_output.messages, protected);
                            let stats = ForceCompressStats {
                                before_count,
                                after_count: self.messages.len(),
                                evicted: fb_output.evicted.len(),
                                before_tokens,
                                after_tokens: self.token_estimate(),
                            };
                            self.metrics.record(&stats, "SlidingWindow(fallback)");
                            // Return the fallback result, not the original error
                            (Some(stats), fb_output.checkpoint)
                        }
                        Err(fb_err) => {
                            // Even the fallback failed — restore the original buffer before
                            // returning so callers that retry do not observe truncated state.
                            self.messages = original_messages.clone();
                            tracing::error!(
                                primary_error = %e,
                                fallback_error = %fb_err,
                                "Both primary compressor and SlidingWindow fallback failed — some conversation context lost"
                            );
                            return Err(e);
                        }
                    }
                }
            }
        } else {
            (None, None)
        };

        // Always sanitize tool_calls → tool_result pairing before sending to LLM.
        // Even without compression, session resume or manual manipulation can
        // produce invalid sequences.
        let (sanitized, tool_fixes) = sanitize_tool_call_pairing(&self.messages);
        // Write sanitized messages back to the internal buffer so subsequent
        // reads see consistent state, and canonical re-injection targets them.
        self.messages = sanitized;

        // Merge tool_pair_fixes into the compression checkpoint
        if let Some(ref mut cp) = combined_checkpoint {
            cp.tool_pair_fixes = tool_fixes;
        }

        // ── Summary verification ──
        // Run lightweight rule-based checks when a summary was produced.
        // On P0 check failure, fall back to SlidingWindowCompressor as safety net.
        let verification = if let Some(ref cp) = combined_checkpoint {
            if cp.summary.is_some() {
                let v = verifier::verify_compression(&self.messages, cp, &original_messages);
                if !v.passed {
                    let p0_failed: Vec<&str> = v
                        .checks
                        .iter()
                        .filter(|c| !c.passed && c.priority == verifier::CheckPriority::P0)
                        .map(|c| c.name.as_str())
                        .collect();
                    if !p0_failed.is_empty() {
                        tracing::warn!(
                            failed_checks = ?p0_failed,
                            "Summary verifier: P0 checks FAILED — falling back to SlidingWindowCompressor to recover critical information"
                        );
                        // Re-compress original messages with SlidingWindow as safety net
                        let (orig_compressible, orig_protected) =
                            self.split_protected(original_messages.clone());
                        let orig_protected_tokens =
                            orig_protected.iter().fold(0usize, |total, value| {
                                total.saturating_add(
                                    value.message.content.estimated_tokens(&*self.tokenizer),
                                )
                            });
                        if let Ok(fb_output) = SlidingWindowCompressor::new(40)
                            .compress(CompressionInput {
                                messages: orig_compressible,
                                token_limit: effective_limit.saturating_sub(orig_protected_tokens),
                                current_query: current_query.map(String::from),
                                focus_instructions: None,
                                cancel_token: cancel_token.clone(),
                                tokenizer: Some(self.tokenizer.clone()),
                            })
                            .await
                        {
                            self.messages =
                                Self::merge_protected(fb_output.messages, orig_protected);
                            // Re-sanitize after fallback
                            let (sanitized, _fb_fixes) = sanitize_tool_call_pairing(&self.messages);
                            self.messages = sanitized;
                            // Update the checkpoint to note the fallback
                            let fb_checkpoint =
                                CompressionCheckpoint::new("SlidingWindow(verifier-fallback)")
                                    .with_counts(self.messages.len(), fb_output.evicted.len())
                                    .with_tokens(
                                        Self::estimate_tokens(&original_messages, &*self.tokenizer),
                                        self.token_estimate(),
                                    );
                            combined_checkpoint = Some(fb_checkpoint);
                            tracing::info!(
                                after_messages = self.messages.len(),
                                after_tokens = self.token_estimate(),
                                "Recovered via SlidingWindow fallback after P0 verification failure"
                            );
                        } else {
                            tracing::error!(
                                "SlidingWindow fallback also failed after P0 verification failure — continuing with current compressed messages"
                            );
                        }
                    }
                }
                Some(v)
            } else {
                None
            }
        } else {
            None
        };

        // ── Canonical context re-injection ──
        // If canonical context is configured, ensure critical components
        // (system prompt, rules, skill info) survived compression.
        if self.canonical_context.is_some() {
            self.reinject_canonical_context();
        }
        if self.token_estimate() > effective_limit {
            self.messages = original_messages;
            return Err(echo_core::error::AgentError::ContextLimitExceeded(format!(
                "prepared context exceeds its effective input budget: system={system_tokens}, tools={tool_tokens}, conversation={conversation_tokens}, effective_limit={effective_limit}, window={}",
                self.token_limit
            ))
            .into());
        }
        combined_checkpoint = self.finalize_checkpoint(combined_checkpoint);

        Ok(PrepareResult {
            messages: self.messages.clone(),
            compressed,
            checkpoint: combined_checkpoint,
            verification,
        })
    }

    fn estimate_tokens(messages: &[Message], tokenizer: &dyn Tokenizer) -> usize {
        messages.iter().fold(0usize, |total, message| {
            total.saturating_add(message.content.estimated_tokens(tokenizer))
        })
    }
}

/// Sanitize message sequence to ensure valid `tool_calls → tool_result` pairing.
///
/// OpenAI-compatible APIs require that:
/// - Every `assistant` message with `tool_calls` must be followed by `tool` messages
///   for each `tool_call_id`
/// - Every `tool` message must have a preceding `assistant` message with matching
///   `tool_call_id`
///
/// Compression, session resume, or manual manipulation can violate these constraints.
/// This function repairs the sequence by:
/// 1. Adding placeholder `tool` results for orphaned `tool_calls`
/// 2. Removing orphaned `tool` messages without matching `tool_calls`
fn sanitize_tool_call_pairing(messages: &[Message]) -> (Vec<Message>, Vec<ToolPairFix>) {
    use std::collections::{HashMap, HashSet};

    let mut fixes: Vec<ToolPairFix> = Vec::new();

    if messages.is_empty() {
        return (vec![], fixes);
    }

    // Pass 1: Count tool_call_ids per assistant message and collect all available results
    // Key: assistant message index, Value: set of tool_call_ids in that message
    let mut assistant_tool_calls: HashMap<usize, HashSet<String>> = HashMap::new();
    // All tool_call_ids that have corresponding tool result messages
    let mut available_results: HashSet<String> = HashSet::new();

    for (i, msg) in messages.iter().enumerate() {
        if msg.role == Role::Assistant {
            if let Some(ref tcs) = msg.tool_calls {
                let ids: HashSet<String> = tcs.iter().map(|tc| tc.id.clone()).collect();
                assistant_tool_calls.insert(i, ids);
            }
        } else if msg.role == Role::Tool
            && let Some(ref id) = msg.tool_call_id
        {
            available_results.insert(id.clone());
        }
    }

    // If no tool_calls at all, return as-is
    if assistant_tool_calls.is_empty() && available_results.is_empty() {
        return (messages.to_vec(), fixes);
    }

    // Build the set of all referenced tool_call_ids (from assistant messages)
    let all_referenced: HashSet<String> = assistant_tool_calls
        .values()
        .flat_map(|ids| ids.iter().cloned())
        .collect();

    // Pass 2: Build sanitized sequence
    let mut result = Vec::with_capacity(messages.len());
    let mut inserted_placeholders: HashSet<String> = HashSet::new();
    // Track which assistant message's placeholders are pending (inserted after
    // all its original tool results have been seen)
    let mut pending_placeholders: Vec<Vec<String>> = Vec::new();

    for (i, msg) in messages.iter().enumerate() {
        // Before processing the current message, check if any pending placeholder
        // groups should be flushed. We flush when we encounter a non-tool message
        // (meaning the tool result sequence for the previous assistant has ended).
        if msg.role != Role::Tool && !pending_placeholders.is_empty() {
            for ids in pending_placeholders.drain(..) {
                for id in ids {
                    if !inserted_placeholders.contains(&id) {
                        result.push(Message::tool_result(
                            id.clone(),
                            "unknown".to_string(),
                            "[Result unavailable — tool result was removed during context compression]".to_string(),
                        ));
                        fixes.push(ToolPairFix {
                            tool_call_id: id.clone(),
                            fix_type: ToolPairFixType::PlaceholderResultInserted,
                        });
                        inserted_placeholders.insert(id);
                    }
                }
            }
        }

        if msg.role == Role::Assistant {
            if let Some(tc_ids) = assistant_tool_calls.get(&i) {
                // Check if ALL tool_call_ids in this message are orphaned
                let all_orphaned = tc_ids.iter().all(|id| !available_results.contains(id));

                if all_orphaned {
                    // Remove tool_calls field — treat as regular assistant message
                    let mut cleaned = msg.clone();
                    for tc_id in tc_ids.iter() {
                        fixes.push(ToolPairFix {
                            tool_call_id: tc_id.clone(),
                            fix_type: ToolPairFixType::DanglingCallCleared,
                        });
                    }
                    cleaned.tool_calls = None;
                    result.push(cleaned);
                } else {
                    result.push(msg.clone());
                    // Track which IDs need placeholders (will be inserted after
                    // all original tool results for this assistant message)
                    let missing: Vec<String> = tc_ids
                        .iter()
                        .filter(|id| !available_results.contains(*id))
                        .cloned()
                        .collect();
                    if !missing.is_empty() {
                        pending_placeholders.push(missing);
                    }
                }
            } else {
                result.push(msg.clone());
            }
        } else if msg.role == Role::Tool {
            // Only include if the tool_call_id is referenced by an assistant message
            let is_orphaned = match &msg.tool_call_id {
                Some(id) => !all_referenced.contains(id),
                None => true, // Tool message without tool_call_id is always orphaned
            };

            if !is_orphaned {
                result.push(msg.clone());
            } else {
                fixes.push(ToolPairFix {
                    tool_call_id: msg
                        .tool_call_id
                        .clone()
                        .unwrap_or_else(|| "<none>".to_string()),
                    fix_type: ToolPairFixType::OrphanedResultRemoved,
                });
                tracing::debug!(
                    tool_call_id = msg.tool_call_id.as_deref().unwrap_or("<none>"),
                    "Removed orphaned tool result message"
                );
            }
        } else {
            result.push(msg.clone());
        }
    }

    // Flush any remaining pending placeholders (end of message list)
    for ids in pending_placeholders {
        for id in ids {
            if !inserted_placeholders.contains(&id) {
                result.push(Message::tool_result(
                    id.clone(),
                    "unknown".to_string(),
                    "[Result unavailable — tool result was removed during context compression]"
                        .to_string(),
                ));
                fixes.push(ToolPairFix {
                    tool_call_id: id,
                    fix_type: ToolPairFixType::PlaceholderResultInserted,
                });
            }
        }
    }

    (result, fixes)
}

/// Builder for `ContextManager`
pub struct ContextManagerBuilder {
    token_limit: usize,
    compressor: Option<Box<dyn ContextCompressor>>,
    initial_messages: Vec<Message>,
    tokenizer: Option<Arc<dyn Tokenizer>>,
    max_messages: Option<usize>,
    budget: Option<TokenBudget>,
    visibility_horizon: Option<horizon::VisibilityHorizonCompressor>,
    memory_promoter: Option<Arc<dyn MemoryPromoter>>,
    canonical_context: Option<CanonicalContext>,
}

impl ContextManagerBuilder {
    /// Set the compression strategy (optional). Supports any type implementing `ContextCompressor`,
    /// including `SlidingWindowCompressor` and `SummaryCompressor` (stage4 P4.3 default).
    pub fn compressor(mut self, c: impl ContextCompressor + 'static) -> Self {
        self.compressor = Some(Box::new(c));
        self
    }

    /// Pre-set a system message as the initial context (typically used for Agent system prompts)
    pub fn with_system(mut self, system_prompt: String) -> Self {
        self.initial_messages.push(Message::system(system_prompt));
        self
    }

    /// Set a custom Tokenizer (default [`HeuristicTokenizer`])
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use echo_state::compression::ContextManager;
    /// use echo_core::tokenizer::SimpleTokenizer;
    /// use std::sync::Arc;
    ///
    /// let ctx = ContextManager::builder(4096)
    ///     .tokenizer(Arc::new(SimpleTokenizer))
    ///     .build();
    /// ```
    pub fn tokenizer(mut self, tokenizer: Arc<dyn Tokenizer>) -> Self {
        self.tokenizer = Some(tokenizer);
        self
    }

    /// Set an optional hard message-count cap (supplementary OOM backstop).
    ///
    /// (stage4 P4.3) The primary OOM defense is the token budget; industry主流
    /// is token-driven (OpenAI/Anthropic/Letta all bill/window on tokens).
    /// `max_messages` defaults to `None` (disabled). Set it only when you need
    /// a message-count backstop against pathological growth (e.g. many tiny
    /// messages). When exceeded, `push()` applies sliding-window degradation
    /// preserving system + protected + recent messages.
    pub fn max_messages(mut self, max: usize) -> Self {
        self.max_messages = Some(max);
        self
    }

    /// Set a token budget for percentage-based allocation.
    ///
    /// When set, `prepare()` uses budget-aware compression instead of
    /// the simple `token_limit` comparison. The budget divides the context
    /// window into system/tool/output/safety percentages and allocates
    /// the remainder for conversation.
    pub fn budget(mut self, budget: TokenBudget) -> Self {
        self.budget = Some(budget);
        self
    }

    /// Set a visibility horizon for proactive tool trace compaction.
    ///
    /// When set, `prepare()` runs a horizon compaction pass **before** the
    /// main compressor. Tool call/result groups beyond the active window
    /// are replaced with compact symbolic summaries, keeping the context
    /// focused on recent work.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use echo_state::compression::{ContextManager, horizon::VisibilityHorizonConfig};
    ///
    /// let ctx = ContextManager::builder(128_000)
    ///     .visibility_horizon(VisibilityHorizonConfig {
    ///         active_window_turns: 5,
    ///         ..Default::default()
    ///     })
    ///     .build();
    /// ```
    pub fn visibility_horizon(mut self, config: horizon::VisibilityHorizonConfig) -> Self {
        self.visibility_horizon = Some(horizon::VisibilityHorizonCompressor::new(config));
        self
    }

    /// Set a memory promoter callback.
    ///
    /// When compression evicts messages, the promoter is called with the
    /// evicted messages so key facts can be extracted and written to a
    /// long-term memory Store (L3 memory promotion).
    pub fn memory_promoter(mut self, promoter: Arc<dyn MemoryPromoter>) -> Self {
        self.memory_promoter = Some(promoter);
        self
    }

    /// Set the canonical context for re-injection after compression.
    pub fn canonical_context(mut self, context: CanonicalContext) -> Self {
        self.canonical_context = Some(context);
        self
    }

    pub fn build(self) -> ContextManager {
        ContextManager {
            messages: self.initial_messages,
            compressor: self.compressor,
            token_limit: self.token_limit,
            tokenizer: self
                .tokenizer
                .unwrap_or_else(|| Arc::new(HeuristicTokenizer)),
            protected_markers: Vec::new(),
            replaceable_protected_markers: Vec::new(),
            projection_scopes: HashMap::new(),
            max_messages: self.max_messages,
            budget: self.budget,
            metrics: CompressionMetrics::new(),
            visibility_horizon: self.visibility_horizon,
            memory_promoter: self.memory_promoter,
            canonical_context: self.canonical_context,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compression::compressor::SlidingWindowCompressor;
    use echo_core::error::Result;
    use futures::future::BoxFuture;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct CharacterTokenizer;

    impl Tokenizer for CharacterTokenizer {
        fn count_tokens(&self, text: &str) -> usize {
            text.chars().count()
        }
    }

    struct LimitRecorder {
        limit: Arc<AtomicUsize>,
    }

    impl ContextCompressor for LimitRecorder {
        fn compress(&self, input: CompressionInput) -> BoxFuture<'_, Result<CompressionOutput>> {
            self.limit.store(input.token_limit, Ordering::SeqCst);
            Box::pin(async move {
                Ok(CompressionOutput {
                    evicted: input.messages,
                    messages: Vec::new(),
                    checkpoint: None,
                })
            })
        }
    }

    #[tokio::test]
    async fn prepare_deducts_protected_context_and_uses_configured_tokenizer() -> Result<()> {
        let recorded_limit = Arc::new(AtomicUsize::new(usize::MAX));
        let mut ctx = ContextManager::builder(100)
            .tokenizer(Arc::new(CharacterTokenizer))
            .budget(TokenBudget::new(100)?)
            .compressor(LimitRecorder {
                limit: recorded_limit.clone(),
            })
            .with_system("12345".to_string())
            .build();
        ctx.add_protected_marker("<protected>".to_string());
        ctx.push(Message::user("<protected>123456789".to_string()));
        ctx.push(Message::user("x".repeat(100)));

        let _ = ctx.prepare(None).await?;

        // Output and safety reserve 20 tokens. The protected 20-token message
        // is deducted before compression; the system prompt remains inside the
        // compressor input and shares the remaining 60-token input capacity.
        assert_eq!(recorded_limit.load(Ordering::SeqCst), 60);
        Ok(())
    }

    #[test]
    fn context_projection_replaces_existing_tagged_message() {
        let marker = "<runtime_projection>";
        let mut ctx = ContextManager::builder(4096)
            .with_system("system".to_string())
            .build();
        ctx.push(Message::user("history".to_string()));

        ctx.apply_projections(&[ContextProjection {
            marker: marker.to_string(),
            message: Some(Message::user(format!("{marker}old"))),
        }]);
        ctx.apply_projections(&[ContextProjection {
            marker: marker.to_string(),
            message: Some(Message::user(format!("{marker}new"))),
        }]);

        let projected: Vec<_> = ctx
            .messages()
            .iter()
            .filter(|message| {
                message
                    .content
                    .as_text_ref()
                    .is_some_and(|text| text.contains(marker))
            })
            .collect();
        assert_eq!(projected.len(), 1);
        assert!(
            projected
                .first()
                .is_some_and(|message| is_context_projection_message(message))
        );
        assert!(
            projected
                .first()
                .and_then(|message| message.content.as_text_ref())
                .is_some_and(|text| text.ends_with("<runtime_projection>new"))
        );
        assert_eq!(ctx.protected_message_count(), 1);
    }

    #[test]
    fn context_projection_none_removes_stale_tagged_message() {
        let marker = "<runtime_projection>";
        let mut ctx = ContextManager::builder(4096)
            .with_system("system".to_string())
            .build();
        ctx.apply_projections(&[ContextProjection {
            marker: marker.to_string(),
            message: Some(Message::user(format!("{marker}stale"))),
        }]);

        ctx.apply_projections(&[ContextProjection {
            marker: marker.to_string(),
            message: None,
        }]);

        assert!(
            ctx.messages().iter().all(|message| {
                message
                    .content
                    .as_text_ref()
                    .is_none_or(|text| !text.contains(marker))
            }),
            "None projection must remove stale tagged context"
        );
        assert_eq!(ctx.protected_message_count(), 0);
    }

    #[test]
    fn context_projection_update_preserves_other_sources() {
        let mut ctx = ContextManager::builder(4096).build();
        ctx.apply_projections(&[
            ContextProjection {
                marker: "runtime".to_string(),
                message: Some(Message::user("runtime state".to_string())),
            },
            ContextProjection {
                marker: "skill:review".to_string(),
                message: Some(Message::system("review instructions".to_string())),
            },
        ]);

        ctx.replace_projection("runtime", Some(Message::user("new runtime".to_string())));

        assert!(ctx.has_projection("runtime"));
        assert!(ctx.has_projection("skill:review"));
        assert_eq!(ctx.protected_message_count(), 2);
        assert!(ctx.messages().iter().any(|message| {
            message
                .content
                .as_text_ref()
                .is_some_and(|text| text.ends_with("review instructions"))
        }));
    }

    #[test]
    fn unchanged_projection_keeps_its_position_for_prefix_cache_stability() {
        let mut ctx = ContextManager::builder(4096).build();
        ctx.replace_projection(
            "workspace",
            Some(Message::user("stable workspace".to_string())),
        );
        ctx.push(Message::user("conversation turn".to_string()));
        let before = ctx
            .messages()
            .iter()
            .position(is_context_projection_message);

        ctx.replace_projection(
            "workspace",
            Some(Message::user("stable workspace".to_string())),
        );

        let after = ctx
            .messages()
            .iter()
            .position(is_context_projection_message);
        assert_eq!(before, after);
    }

    #[test]
    fn tail_projection_replaces_dynamic_context_near_latest_turn() {
        let mut ctx = ContextManager::builder(4096).build();
        ctx.push(Message::user("older history".to_string()));
        ctx.replace_tail_projection("turn-memory", Some(Message::user("old memory".to_string())));
        ctx.push(Message::user("previous request".to_string()));
        ctx.replace_tail_projection(
            "turn-memory",
            Some(Message::user("current memory".to_string())),
        );

        assert!(ctx.messages().iter().all(|message| {
            message
                .content
                .as_text_ref()
                .is_none_or(|text| !text.contains("old memory"))
        }));
        assert!(ctx.messages().last().is_some_and(|message| {
            message
                .content
                .as_text_ref()
                .is_some_and(|text| text.ends_with("current memory"))
        }));
    }

    #[test]
    fn projection_scope_removes_only_markers_previously_owned_by_provider() {
        let mut ctx = ContextManager::builder(4096).build();
        ctx.replace_projection(
            "skill:review",
            Some(Message::system("review skill".to_string())),
        );
        ctx.apply_projection_scope(
            "pre-model",
            &[ContextProjection {
                marker: "runtime".to_string(),
                message: Some(Message::user("active runtime".to_string())),
            }],
        );

        ctx.apply_projection_scope("pre-model", &[]);

        assert!(!ctx.has_projection("runtime"));
        assert!(ctx.has_projection("skill:review"));
    }

    #[test]
    fn context_projection_preserves_unrelated_marker_text() {
        let marker = "runtime-marker";
        let mut ctx = ContextManager::builder(4096)
            .with_system("system".to_string())
            .build();
        ctx.push(Message::user(
            "ordinary conversation mentions runtime-marker".to_string(),
        ));

        ctx.apply_projections(&[ContextProjection {
            marker: marker.to_string(),
            message: Some(Message::user("projected state".to_string())),
        }]);

        assert!(ctx.messages().iter().any(|message| {
            message
                .content
                .as_text_ref()
                .is_some_and(|text| text == "ordinary conversation mentions runtime-marker")
        }));
        assert_eq!(ctx.protected_message_count(), 1);
    }

    #[test]
    fn has_projection_matches_exact_envelope_marker() {
        let mut ctx = ContextManager::builder(4096).build();
        ctx.apply_projections(&[ContextProjection {
            marker: "runtime-marker".to_string(),
            message: Some(Message::user("current runtime".to_string())),
        }]);

        assert!(ctx.has_projection("runtime-marker"));
        assert!(!ctx.has_projection("runtime"));
        assert!(!ctx.has_projection("runtime-marker-extra"));
    }

    #[test]
    fn has_projection_ignores_plain_message_text() {
        let mut ctx = ContextManager::builder(4096).build();
        ctx.push(Message::user(
            "plain text mentions runtime-marker but is not an owned projection".to_string(),
        ));

        assert!(!ctx.has_projection("runtime-marker"));
    }

    #[test]
    fn context_projection_removes_restored_projection_without_transient_state() {
        let mut original = ContextManager::builder(4096)
            .with_system("system".to_string())
            .build();
        original.apply_projections(&[ContextProjection {
            marker: "runtime".to_string(),
            message: Some(Message::user("stale restored projection".to_string())),
        }]);
        let restored_messages = original.messages().to_vec();
        let mut restored = ContextManager::builder(4096).build();
        restored.set_messages(restored_messages);

        restored.replace_projection("runtime", None);

        assert!(restored.messages().iter().all(|message| {
            message
                .content
                .as_text_ref()
                .is_none_or(|text| !text.contains("stale restored projection"))
        }));
    }

    #[test]
    fn replaceable_protected_marker_keeps_only_latest_value() {
        let marker = "[task_context]";
        let mut ctx = ContextManager::builder(4096).build();
        ctx.add_replaceable_protected_marker(marker.to_string());
        ctx.push_many([
            Message::user(format!("{marker} first brief")),
            Message::assistant("intermediate result".to_string()),
            Message::user(format!("{marker} current brief")),
        ]);

        let briefs: Vec<_> = ctx
            .messages()
            .iter()
            .filter(|message| {
                message
                    .content
                    .as_text_ref()
                    .is_some_and(|text| text.contains(marker))
            })
            .collect();
        assert_eq!(briefs.len(), 1);
        assert!(briefs.first().is_some_and(|message| {
            message
                .content
                .as_text_ref()
                .is_some_and(|text| text.ends_with("current brief"))
        }));
        assert_eq!(ctx.protected_message_count(), 1);
        assert!(ctx.protected_token_estimate() > 0);
        assert!(ctx.messages().iter().any(|message| {
            message
                .content
                .as_text_ref()
                .is_some_and(|text| text == "intermediate result")
        }));
    }

    #[test]
    fn memory_projection_is_reported_in_memory_token_breakdown() {
        let mut ctx = ContextManager::builder(4096).build();
        ctx.replace_projection(
            "turn-memory",
            Some(Message::user(
                "[memory_context] Relevant historical memories: preference".to_string(),
            )),
        );

        let breakdown = ctx.token_breakdown(None);
        assert!(breakdown.memory > 0);
        assert_eq!(breakdown.user, 0);
    }

    fn canonical_test_context() -> ContextManager {
        let canonical = CanonicalContext {
            system_prompt: Some("canonical system".to_string()),
            project_rules: Some("canonical rules".to_string()),
            skill_injections: Vec::new(),
            active_skill_names: Vec::new(),
        };
        let mut ctx = ContextManager::builder(64)
            .compressor(SlidingWindowCompressor::new(1))
            .with_system("canonical system".to_string())
            .canonical_context(canonical)
            .build();
        ctx.push(Message::user(
            "enough conversation content to force compression ".repeat(80),
        ));
        ctx
    }

    fn assert_canonical_checkpoint_matches_context(
        ctx: &ContextManager,
        checkpoint: Option<&CompressionCheckpoint>,
    ) {
        assert_eq!(
            checkpoint.map(|value| value.retained_count),
            Some(ctx.messages().len())
        );
        assert_eq!(
            checkpoint.map(|value| value.token_after),
            Some(ctx.token_estimate())
        );
    }

    #[tokio::test]
    async fn canonical_system_context_is_not_duplicated() -> Result<()> {
        let mut ctx = canonical_test_context();

        let _ = ctx.prepare(None).await?;
        let _ = ctx.prepare(None).await?;

        let system_occurrences = ctx
            .messages()
            .iter()
            .filter_map(|message| message.content.as_text_ref())
            .filter(|content| content.contains("canonical system"))
            .count();
        assert_eq!(system_occurrences, 1);
        Ok(())
    }

    #[tokio::test]
    async fn canonical_system_context_replaces_missing_prompt_beside_other_system_message()
    -> Result<()> {
        let canonical = CanonicalContext {
            system_prompt: Some("canonical system".to_string()),
            project_rules: None,
            skill_injections: Vec::new(),
            active_skill_names: Vec::new(),
        };
        let mut ctx = ContextManager::builder(4096)
            .with_system("different system".to_string())
            .canonical_context(canonical)
            .build();

        let result = ctx.prepare(None).await?;

        assert!(result.messages.iter().any(|message| {
            message.role == Role::System
                && message.content.as_text_ref() == Some("canonical system")
        }));
        Ok(())
    }

    #[tokio::test]
    async fn canonical_prepare_checkpoint_matches_final_context() -> Result<()> {
        let mut ctx = canonical_test_context();

        let result = ctx.prepare(None).await?;

        assert_canonical_checkpoint_matches_context(&ctx, result.checkpoint.as_ref());
        Ok(())
    }

    #[tokio::test]
    async fn canonical_force_compress_checkpoint_matches_final_context() -> Result<()> {
        let mut ctx = canonical_test_context();

        let (_, checkpoint) = ctx.force_compress(1).await?;

        assert_canonical_checkpoint_matches_context(&ctx, checkpoint.as_ref());
        Ok(())
    }

    #[tokio::test]
    async fn canonical_force_compress_with_focus_checkpoint_matches_final_context() -> Result<()> {
        let mut ctx = canonical_test_context();

        let (_, checkpoint) = ctx.force_compress_with_focus("rules", 1).await?;

        assert_canonical_checkpoint_matches_context(&ctx, checkpoint.as_ref());
        Ok(())
    }

    #[tokio::test]
    async fn canonical_force_compress_with_checkpoint_matches_final_context() -> Result<()> {
        let mut ctx = canonical_test_context();
        let compressor = SlidingWindowCompressor::new(1);

        let (_, checkpoint) = ctx.force_compress_with(&compressor).await?;

        assert_canonical_checkpoint_matches_context(&ctx, checkpoint.as_ref());
        Ok(())
    }

    #[tokio::test]
    async fn test_sliding_window_compressor() -> Result<()> {
        let mut ctx = ContextManager::builder(10)
            .compressor(SlidingWindowCompressor::new(4))
            .build();

        ctx.push(Message::system("You are an assistant.".to_string()));
        for i in 1..=6 {
            ctx.push(Message::user(format!("用户消息 {}", i)));
            ctx.push(Message::assistant(format!("助手回复 {}", i)));
        }

        let before = ctx.messages().len();
        let result = ctx.prepare(None).await?;
        let messages = result.messages;
        assert!(messages.len() < before);
        assert!(messages.first().is_some_and(|message| {
            message.role == Role::System
                && message.content.as_text_ref() == Some("You are an assistant.")
        }));
        assert!(messages.iter().any(|message| {
            message.role == Role::User && message.content.as_text_ref() == Some("用户消息 6")
        }));
        assert!(messages.iter().any(|message| {
            message.role == Role::Assistant && message.content.as_text_ref() == Some("助手回复 6")
        }));
        assert!(messages.iter().all(|message| {
            message.content.as_text_ref() != Some("用户消息 1")
                && message.content.as_text_ref() != Some("助手回复 1")
        }));
        Ok(())
    }

    #[tokio::test]
    async fn test_protected_messages_keep_relative_position_after_compression() -> Result<()> {
        let mut ctx = ContextManager::builder(10)
            .compressor(SlidingWindowCompressor::new(2))
            .build();
        ctx.add_protected_marker("<skill>".to_string());

        ctx.push(Message::system("system".to_string()));
        ctx.push(Message::user("old user".to_string()));
        ctx.push(Message::assistant("old assistant".to_string()));
        ctx.push(Message::user("<skill> protected".to_string()));
        ctx.push(Message::assistant("recent assistant".to_string()));
        ctx.push(Message::user("latest user".to_string()));

        let (stats, _checkpoint) = ctx.force_compress(2).await?;
        assert!(stats.after_count >= 3);

        let rendered: Vec<(String, String)> = ctx
            .messages()
            .iter()
            .map(|m| {
                (
                    m.role.as_str().to_string(),
                    m.content.as_text_ref().unwrap_or("").to_string(),
                )
            })
            .collect();

        assert_eq!(
            rendered,
            vec![
                ("system".to_string(), "system".to_string()),
                ("user".to_string(), "<skill> protected".to_string()),
                ("assistant".to_string(), "recent assistant".to_string()),
                ("user".to_string(), "latest user".to_string()),
            ]
        );

        Ok(())
    }

    #[test]
    fn test_sanitize_tool_call_pairing_no_tools() {
        let messages = vec![
            Message::system("sys".to_string()),
            Message::user("hello".to_string()),
            Message::assistant("hi".to_string()),
        ];
        let (result, _fixes) = sanitize_tool_call_pairing(&messages);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_sanitize_tool_call_pairing_valid() {
        use echo_core::llm::types::ToolCall;
        let tc = ToolCall {
            id: "call_1".to_string(),
            call_type: "function".to_string(),
            function: echo_core::llm::types::FunctionCall {
                name: "read_file".to_string(),
                arguments: "{}".to_string(),
            },
        };
        let messages = vec![
            Message::user("read file".to_string()),
            Message::assistant_with_tools(vec![tc]),
            Message::tool_result(
                "call_1".to_string(),
                "read_file".to_string(),
                "content".to_string(),
            ),
        ];
        let (result, _fixes) = sanitize_tool_call_pairing(&messages);
        assert_eq!(result.len(), 3);
        assert!(result[1].tool_calls.is_some());
    }

    #[test]
    fn test_sanitize_tool_call_pairing_orphaned_tool() {
        // tool result without preceding assistant tool_calls
        let messages = vec![
            Message::user("hello".to_string()),
            Message::tool_result(
                "orphan_1".to_string(),
                "some_tool".to_string(),
                "result".to_string(),
            ),
            Message::assistant("hi".to_string()),
        ];
        let (result, _fixes) = sanitize_tool_call_pairing(&messages);
        // orphaned tool message should be removed
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].role, Role::User);
        assert_eq!(result[1].role, Role::Assistant);
    }

    #[test]
    fn test_sanitize_tool_call_pairing_dangling_calls() {
        use echo_core::llm::types::ToolCall;
        // assistant has tool_calls but no tool results (compression removed them)
        // When ALL tool_calls are orphaned, we null out the tool_calls field
        // rather than adding placeholder results.
        let tc = ToolCall {
            id: "call_missing".to_string(),
            call_type: "function".to_string(),
            function: echo_core::llm::types::FunctionCall {
                name: "read_file".to_string(),
                arguments: "{}".to_string(),
            },
        };
        let messages = vec![
            Message::user("read file".to_string()),
            Message::assistant_with_tools(vec![tc]),
            Message::user("next question".to_string()),
        ];
        let (result, _fixes) = sanitize_tool_call_pairing(&messages);
        // assistant's tool_calls should be nulled out → 3 messages (no placeholder)
        assert_eq!(result.len(), 3);
        assert!(
            result[1].tool_calls.is_none(),
            "orphaned tool_calls should be removed"
        );
    }

    #[test]
    fn test_sanitize_tool_call_pairing_mixed() {
        use echo_core::llm::types::ToolCall;
        // assistant has 2 tool_calls, but only 1 result exists
        let tc1 = ToolCall {
            id: "call_present".to_string(),
            call_type: "function".to_string(),
            function: echo_core::llm::types::FunctionCall {
                name: "read_file".to_string(),
                arguments: "{}".to_string(),
            },
        };
        let tc2 = ToolCall {
            id: "call_missing".to_string(),
            call_type: "function".to_string(),
            function: echo_core::llm::types::FunctionCall {
                name: "write_file".to_string(),
                arguments: "{}".to_string(),
            },
        };
        let messages = vec![
            Message::user("do stuff".to_string()),
            Message::assistant_with_tools(vec![tc1, tc2]),
            Message::tool_result(
                "call_present".to_string(),
                "read_file".to_string(),
                "content".to_string(),
            ),
            // call_missing result was removed by compression
            Message::user("next".to_string()),
        ];
        let (result, _fixes) = sanitize_tool_call_pairing(&messages);
        // Should have: user, assistant, tool(present), tool(placeholder), user
        assert_eq!(result.len(), 5);
        assert_eq!(result[2].tool_call_id.as_deref(), Some("call_present"));
        assert_eq!(result[3].role, Role::Tool);
        assert_eq!(result[3].tool_call_id.as_deref(), Some("call_missing"));
        assert!(
            result[3]
                .content
                .as_text_ref()
                .unwrap()
                .contains("unavailable")
        );
    }

    // ── L3 Memory Promotion tests ────────────────────────────────────

    /// A test promoter that records how many times it was called and
    /// how many evicted messages it received.
    struct TestPromoter {
        call_count: Arc<std::sync::atomic::AtomicUsize>,
        total_evicted: Arc<std::sync::atomic::AtomicUsize>,
    }

    impl TestPromoter {
        fn new() -> (
            Self,
            Arc<std::sync::atomic::AtomicUsize>,
            Arc<std::sync::atomic::AtomicUsize>,
        ) {
            let call_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let total_evicted = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            (
                Self {
                    call_count: call_count.clone(),
                    total_evicted: total_evicted.clone(),
                },
                call_count,
                total_evicted,
            )
        }
    }

    impl MemoryPromoter for TestPromoter {
        fn promote(
            &self,
            evicted: &[Message],
        ) -> futures::future::BoxFuture<'_, Result<MemoryPromotionReceipt>> {
            self.call_count
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            self.total_evicted
                .fetch_add(evicted.len(), std::sync::atomic::Ordering::Relaxed);
            let promoted = evicted.len();
            Box::pin(async move {
                Ok(MemoryPromotionReceipt {
                    submitted: promoted,
                    promoted,
                    deduplicated: 0,
                })
            })
        }
    }

    #[tokio::test]
    async fn test_memory_promoter_called_on_compression() -> Result<()> {
        let (promoter, call_count, total_evicted) = TestPromoter::new();

        let mut ctx = ContextManager::builder(100) // Very low token limit to trigger compression
            .compressor(SlidingWindowCompressor::new(4))
            .memory_promoter(Arc::new(promoter))
            .build();

        // Push enough messages to trigger compression
        ctx.push(Message::system("You are a helper.".to_string()));
        for i in 1..=10 {
            ctx.push(Message::user(format!(
                "Question number {} about various topics",
                i
            )));
            ctx.push(Message::assistant(format!(
                "Here is a detailed answer to question {} with some explanation",
                i
            )));
        }

        let _result = ctx.prepare(None).await?;

        let calls = call_count.load(std::sync::atomic::Ordering::Relaxed);
        let evicted = total_evicted.load(std::sync::atomic::Ordering::Relaxed);

        assert!(
            calls > 0,
            "Memory promoter should have been called at least once"
        );
        assert!(
            evicted > 0,
            "Promoter should have received evicted messages, got {}",
            evicted
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_compression_reduces_tokens_by_30_percent() -> Result<()> {
        let mut ctx = ContextManager::builder(200)
            .compressor(SlidingWindowCompressor::new(6))
            .build();

        ctx.push(Message::system("You are a helpful assistant.".to_string()));

        // Build a conversation with large tool outputs
        for i in 1..=20 {
            ctx.push(Message::user(format!("Question {}", i)));
            ctx.push(Message::assistant(format!(
                "Let me help you with question {}. I'll use some tools.",
                i
            )));
            // Simulate large tool output
            ctx.push(Message::user(format!("[Tool result: {}]", "x".repeat(500))));
        }

        let tokens_before = ctx.token_estimate();
        let _result = ctx.prepare(None).await?;
        let tokens_after = ctx.token_estimate();

        if tokens_before > 200 {
            // Only assert if we actually had enough tokens to compress
            let reduction = 1.0 - (tokens_after as f64 / tokens_before as f64);
            assert!(
                reduction > 0.3,
                "Token reduction should be >30%: before={}, after={}, reduction={:.1}%",
                tokens_before,
                tokens_after,
                reduction * 100.0
            );
        }
        Ok(())
    }

    #[tokio::test]
    async fn test_memory_promoter_not_called_without_compression() -> Result<()> {
        let (promoter, call_count, _) = TestPromoter::new();

        let mut ctx = ContextManager::builder(1_000_000) // Very high limit — no compression
            .memory_promoter(Arc::new(promoter))
            .build();

        ctx.push(Message::system("sys".to_string()));
        ctx.push(Message::user("hello".to_string()));
        ctx.push(Message::assistant("hi".to_string()));

        let _result = ctx.prepare(None).await?;

        let calls = call_count.load(std::sync::atomic::Ordering::Relaxed);
        assert_eq!(
            calls, 0,
            "Promoter should NOT be called when no compression occurs"
        );
        Ok(())
    }
}
