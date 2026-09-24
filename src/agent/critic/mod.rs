//! Critic 模块 —— 对 Agent 输出进行评估与反馈
//!
//! 提供用于评估 Agent 输出并给出结构化反馈的工具与实用函数。
//! 与业界模式保持一致：反思（reflection）是一种可组合的能力，
//! 而非一种独立的 Agent 类型。

mod llm_critic;
mod review_tool;

pub use llm_critic::LlmCritic;
pub use review_tool::ReviewTool;

// 为方便使用，重新导出核心 critic 类型
pub use echo_core::agent::{
    CompositeCritic, CompositeStrategy, Critic, StaticCritic, ThresholdCritic,
};
pub use echo_core::agent::{Critique, CritiqueOutput, critique_output_schema};
