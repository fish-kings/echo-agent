/*!
Subagent 执行钩子 —— subagent 调度的生命周期回调

沿用了 `crate::tasks::hooks` 的模式，但针对 subagent 特有的生命周期节点做了调整。
*/

use std::sync::Arc;

use super::types::{ExecutionMode, SubagentResult};

// ── 钩子上下文 ──────────────────────────────────────────────────────────────

/// 传递给 subagent 钩子回调的上下文。
#[derive(Debug, Clone)]
pub struct SubagentHookContext {
    /// 父 agent 名称。
    pub parent_agent: String,
    /// 正在被调度的 subagent 名称。
    pub subagent_name: String,
    /// 本次调度使用的执行模式。
    pub execution_mode: ExecutionMode,
    /// 被调度执行的任务。
    pub task: String,
    /// 当前尝试次数（从 1 开始计数，用于重试）。
    pub attempt: u32,
}

// ── 重试决策 ────────────────────────────────────────────────────────────

/// `on_failure` 钩子返回的决策。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SubagentRetryDecision {
    /// 延迟一段时间后重试调度。
    Retry {
        /// 重试前的延迟时间（秒）。
        delay_secs: u64,
    },
    /// 直接失败并将错误向上传播。
    Fail,
    /// 委派给另一个替代的 subagent。
    Delegate {
        /// 要委派到的替代 subagent 名称。
        alternative_agent: String,
    },
}

// ── Subagent 钩子 Trait ──────────────────────────────────────────────────────

/**
针对 subagent 生命周期事件的钩子 trait。

所有方法都有默认的空实现（no-op）。
*/
#[async_trait::async_trait]
pub trait SubagentHooks: Send + Sync {
    /// 在 subagent 被调度之前调用。
    async fn before_dispatch(&self, _ctx: &SubagentHookContext) {}

    /// 在 subagent 成功完成之后调用。
    async fn after_dispatch(&self, _ctx: &SubagentHookContext, _result: &SubagentResult) {}

    /**
    在 subagent 失败时调用。

    返回一个决策来控制重试/委派行为。
    */
    async fn on_failure(&self, _ctx: &SubagentHookContext, _error: &str) -> SubagentRetryDecision {
        SubagentRetryDecision::Fail
    }

    /// 在 subagent 被取消时调用。
    async fn on_cancelled(&self, _ctx: &SubagentHookContext) {}
}

// ── 内置实现 ──────────────────────────────────────────────────────

/// 空实现钩子（默认）。
pub struct NoopSubagentHooks;

#[async_trait::async_trait]
impl SubagentHooks for NoopSubagentHooks {}

/// 会发出 tracing 事件的日志钩子。
pub struct LoggingSubagentHooks;

#[async_trait::async_trait]
impl SubagentHooks for LoggingSubagentHooks {
    /// 派发前记录父 Agent、子 Agent、执行模式与任务。
    async fn before_dispatch(&self, ctx: &SubagentHookContext) {
        tracing::info!(
            parent = %ctx.parent_agent,
            subagent = %ctx.subagent_name,
            mode = %ctx.execution_mode,
            task = %ctx.task,
            "subagent_before_dispatch"
        );
    }

    /// 派发完成后记录耗时与截断后的输出预览。
    async fn after_dispatch(&self, ctx: &SubagentHookContext, result: &SubagentResult) {
        let preview = if result.output.chars().count() > 100 {
            let truncated: String = result.output.chars().take(100).collect();
            format!("{truncated}...")
        } else {
            result.output.clone()
        };
        tracing::info!(
            parent = %ctx.parent_agent,
            subagent = %ctx.subagent_name,
            duration_ms = result.duration.as_millis(),
            output = %preview,
            "subagent_after_dispatch"
        );
    }

    /// 失败时记录警告日志，并决定不重试。
    async fn on_failure(&self, ctx: &SubagentHookContext, error: &str) -> SubagentRetryDecision {
        tracing::warn!(
            parent = %ctx.parent_agent,
            subagent = %ctx.subagent_name,
            attempt = ctx.attempt,
            error = %error,
            "subagent_on_failure"
        );
        SubagentRetryDecision::Fail
    }

    /// 取消时记录日志。
    async fn on_cancelled(&self, ctx: &SubagentHookContext) {
        tracing::info!(
            parent = %ctx.parent_agent,
            subagent = %ctx.subagent_name,
            "subagent_on_cancelled"
        );
    }
}

// ── 钩子注册表 ─────────────────────────────────────────────────────────────

/// subagent 钩子的注册表 —— 允许注册多个钩子并按顺序依次调用。
pub struct SubagentHookRegistry {
    /// 已注册的钩子列表
    hooks: Vec<Arc<dyn SubagentHooks>>,
}

impl SubagentHookRegistry {
    /// 创建一个空的钩子注册表。
    pub fn new() -> Self {
        Self { hooks: Vec::new() }
    }

    /// 创建一个预先配置了日志钩子的注册表。
    pub fn with_logging() -> Self {
        let mut registry = Self::new();
        registry.register(Arc::new(LoggingSubagentHooks));
        registry
    }

    /**
    注册一个钩子实现。

    # 参数
    * `hook` - 要添加的钩子实现。
    */
    pub fn register(&mut self, hook: Arc<dyn SubagentHooks>) {
        self.hooks.push(hook);
    }

    /**
    对所有已注册的钩子调用 `before_dispatch`。

    # 参数
    * `ctx` - 钩子上下文。
    */
    pub async fn before_dispatch(&self, ctx: &SubagentHookContext) {
        for hook in &self.hooks {
            hook.before_dispatch(ctx).await;
        }
    }

    /**
    对所有已注册的钩子调用 `after_dispatch`。

    # 参数
    * `ctx` - 钩子上下文。
    * `result` - subagent 的执行结果。
    */
    pub async fn after_dispatch(&self, ctx: &SubagentHookContext, result: &SubagentResult) {
        for hook in &self.hooks {
            hook.after_dispatch(ctx, result).await;
        }
    }

    /// 对所有钩子调用 `on_failure`，返回第一个非 `Fail` 的决策。
    pub async fn on_failure(
        &self,
        ctx: &SubagentHookContext,
        error: &str,
    ) -> SubagentRetryDecision {
        for hook in &self.hooks {
            let decision = hook.on_failure(ctx, error).await;
            if !matches!(decision, SubagentRetryDecision::Fail) {
                return decision;
            }
        }
        SubagentRetryDecision::Fail
    }

    /**
    对所有已注册的钩子调用 `on_cancelled`。

    # 参数
    * `ctx` - 钩子上下文。
    */
    pub async fn on_cancelled(&self, ctx: &SubagentHookContext) {
        for hook in &self.hooks {
            hook.on_cancelled(ctx).await;
        }
    }

    /// 检查该注册表中是否包含任何钩子。
    pub fn is_empty(&self) -> bool {
        self.hooks.is_empty()
    }

    /// 获取已注册钩子的数量。
    pub fn len(&self) -> usize {
        self.hooks.len()
    }
}

impl Default for SubagentHookRegistry {
    /// 创建默认实例。
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for SubagentHookRegistry {
    /// 克隆实例（共享内部钩子列表）。
    fn clone(&self) -> Self {
        Self {
            hooks: self.hooks.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::time::Duration;

    /// 记录各钩子调用次数的测试钩子。
    struct TestHooks {
        before: AtomicU32,
        after: AtomicU32,
        failure: AtomicU32,
    }

    impl TestHooks {
        fn new() -> Self {
            Self {
                before: AtomicU32::new(0),
                after: AtomicU32::new(0),
                failure: AtomicU32::new(0),
            }
        }
    }

    #[async_trait::async_trait]
    impl SubagentHooks for TestHooks {
        async fn before_dispatch(&self, _ctx: &SubagentHookContext) {
            self.before.fetch_add(1, Ordering::SeqCst);
        }

        async fn after_dispatch(&self, _ctx: &SubagentHookContext, _result: &SubagentResult) {
            self.after.fetch_add(1, Ordering::SeqCst);
        }

        async fn on_failure(
            &self,
            _ctx: &SubagentHookContext,
            _error: &str,
        ) -> SubagentRetryDecision {
            self.failure.fetch_add(1, Ordering::SeqCst);
            SubagentRetryDecision::Retry { delay_secs: 1 }
        }
    }

    fn make_ctx() -> SubagentHookContext {
        SubagentHookContext {
            parent_agent: "parent".into(),
            subagent_name: "child".into(),
            execution_mode: ExecutionMode::Sync,
            task: "test task".into(),
            attempt: 1,
        }
    }

    #[tokio::test]
    async fn test_hooks_called() {
        let hooks = Arc::new(TestHooks::new());
        let mut registry = SubagentHookRegistry::new();
        registry.register(hooks.clone());

        let ctx = make_ctx();
        let result = SubagentResult::sync_result("child", "ok".into(), Duration::from_millis(100));

        registry.before_dispatch(&ctx).await;
        registry.after_dispatch(&ctx, &result).await;
        registry.on_failure(&ctx, "error").await;

        assert_eq!(hooks.before.load(Ordering::SeqCst), 1);
        assert_eq!(hooks.after.load(Ordering::SeqCst), 1);
        assert_eq!(hooks.failure.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_registry_default() {
        let registry = SubagentHookRegistry::default();
        assert!(registry.is_empty());
    }

    #[test]
    fn test_registry_with_logging() {
        let registry = SubagentHookRegistry::with_logging();
        assert_eq!(registry.len(), 1);
    }
}
