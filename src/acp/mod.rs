//! 面向 `echo_agent` 框架的稳定 ACP v1 Agent 适配器。
//!
//! 该适配器将官方 ACP Rust 运行时与框架现有的
//! [`crate::runtime::AgentTurnDriver`] 组合在一起。它是传输无关的：
//! 调用方可以将其接入 stdio、进程内 channel 或其他官方 ACP 传输方式，
//! 而无需另外实现一套 JSON-RPC 解析器或 Agent 循环。
//!
//! 每个 ACP Session 拥有一个独立的框架 Agent。会话历史保存在该
//! Agent 内部；本模块只负责协议寻址以及当前轮次的取消 token。
//!
//! 标准 ACP 与协商出的扩展 profile 共享同一个连接级运行时
//! （[`AcpConnectionServices`]）：单一的 Session 映射表、单一的
//! Run 权威（authority）、以账本（ledger）为先的事件路径，以及
//! 统一的关闭链路。扩展 profile 通过 [`AcpConnectionProfile`]
//! 接入，而无需分叉官方的分发循环。
//!
//! 在将适配器移交给官方 `ConnectTo` API 之前，需先用
//! `adapter.close_owner()` 保留其 [`AcpAdapterCloseHandle`]。协议
//! trait 会消费掉该适配器，因此无法返回一个失败的关闭持有者；
//! 连接建立过程会拒绝没有外部句柄的调用方。保留的句柄会在传输层
//! 返回后，针对同一个 Session/Run 权威重试清理。直接使用传输层的
//! 调用方可以使用 [`AcpAgentAdapter::connect_retaining_close_owner`]
//! 在获得连接结果的同时拿到该句柄。手动调用官方 Client 的调用方
//! 必须持有该句柄直到关闭成功为止，即便连接失败也是如此。

mod adapter;
mod extension;
mod projection;
mod prompt;
mod runtime;
mod session;

pub use adapter::{
    AcpAdapterCloseHandle, AcpAdapterConfig, AcpAgentAdapter, AcpAgentAdapterWithProfile,
};
pub use extension::{
    ExtensionInvocationAuthority, ExtensionInvocationLease, ExtensionLeaseError,
    ExtensionSettlement,
};
pub use projection::AcpEventProjector;
pub use runtime::{
    AcpConnectionProfile, AcpConnectionServices, AcpLedgerLimits, ConnectionMode, EventLedger,
    RunEntry, RunEventObserver, RunObserverContext, RunStartSpec, StandardBridgeOutcome,
};
pub use session::{
    AcpSession, AcpSessionContext, AcpSessionFactory, ActiveTurn, ActiveTurnLease,
    SessionCloseLease, SessionRegistry,
};
