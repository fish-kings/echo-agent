//! 汇聚各类上下文压缩器实现（混合压缩、滑动窗口、摘要压缩）并重新导出其公共类型。

/// 混合压缩器：组合多种压缩策略。
pub mod hybrid;
/// 滑动窗口压缩器：只保留最近若干条消息。
pub mod sliding_window;
/// 摘要压缩器：用 LLM 生成历史摘要。
pub mod summary;

pub use hybrid::{HybridCompressor, HybridCompressorBuilder};
pub use sliding_window::SlidingWindowCompressor;
pub use summary::{
    IncrementalSummaryCompressor, SummaryCompressor, default_summary_prompt,
    default_summary_prompt_with_focus, structured_summary_prompt,
};
