/*!
Hook 类型定义 —— 供整个工作区所有 crate 共享。

本模块包含统一 hook 系统的核心类型，之所以定义在 `echo-core` 中，
是为了让 `echo-execution` 和 `echo-orchestration` 都能引用它们，
而不会产生循环依赖。
*/

pub mod types;

pub use types::{
    CompressHookStats, HookContext, HookEvent, HookEventCategory, HookResult, HookSource,
    SubagentStopStatus, TaskTerminalStatus, UnifiedHookExecutorFn,
};
