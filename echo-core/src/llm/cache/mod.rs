/*!
Prompt 缓存布局与诊断。

本模块在扁平的消息数组之上提供一个**只读、零拷贝**的视图，用于识别与缓存
相关的分段（system、规范化上下文、对话历史、运行时上下文）。各 provider
可以利用这个视图来：

1. 放置显式缓存断点（Anthropic 的 `cache_control`）
2. 校验前缀稳定性，用于自动前缀缓存（OpenAI 兼容）
3. 计算稳定前缀哈希，用于缓存失效诊断

本模块**不会**改变消息的存储方式 —— 它纯粹只是一个视图。
*/

/// 缓存诊断：前缀指纹与稳定前缀哈希。
pub mod diagnostic;
/// 缓存布局：分段视图与断点放置。
pub mod layout;

pub use diagnostic::{PromptCacheFingerprint, prompt_cache_fingerprint, stable_prefix_hash};
pub use layout::{BreakpointTarget, CacheHints, PromptCacheLayout, SegmentRange, SegmentRanges};
