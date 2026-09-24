//! ReactAgent 执行引擎
//!
//! 包含 ReAct 循环的全部内部实现：
//! - `retry` —— LLM 重试逻辑 + 工具并发超时计算
//! - `context` —— 上下文管理 + 长期记忆 + 持久化 + 审计
//! - `react_loop` —— ReAct 循环入口与上下文准备
//! - `direct` —— 直接执行入口（run_direct / run_chat_direct）
//! - `stream_channel` —— 基于 channel 的流式执行（主路径）
//! - `processor` —— SSE 分片 → AgentEvent 的转换
//! - `types` —— 共享类型（StreamMode、StreamInit）

use std::time::Duration;

pub(super) const TOOL_CANCELLATION_GRACE_PERIOD: Duration = Duration::from_secs(5);

pub(crate) mod context;
pub(crate) mod direct;
pub(crate) mod phases;
/// 工具执行流水线 —— 由 13 个可组合阶段构成的工具调用中间件。
pub mod pipeline;
pub(crate) mod processor;
pub(crate) mod react_loop;
pub(crate) mod retry;
pub(crate) mod stream_channel;
pub(crate) mod stream_macros;
pub(crate) mod types;
pub(crate) use types::StreamMode;
