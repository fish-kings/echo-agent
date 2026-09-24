//! 预置的流水线工作流
//!
//! 针对常见 Agent 任务的开箱即用图工作流：
//!
//! | 流水线 | 阶段 | 说明 |
//! |----------|--------|-------------|
//! | [`data_pipeline`] | 检查 -> 落盘脚本 -> 执行 -> 校验产物 | 以代码为中心、可复现的数据分析 |
//! | [`writing_pipeline`] | 大纲 -> 起草 -> 审阅 -> 修订（循环）-> 定稿 | 带质量循环的内容创作 |

pub mod data_pipeline;
pub mod writing_pipeline;

pub use data_pipeline::{DataPipelineConfig, DataPipelineLanguage, run_data_pipeline};
pub use writing_pipeline::{WritingPipelineConfig, run_writing_pipeline};
