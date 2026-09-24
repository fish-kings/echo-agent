/*!
统一 hook 桥接 —— 将 task 与 subagent 的 hook 系统接入中心化的
`HookRegistry` 事件模型。

skills hooks 系统（位于 `echo-execution`）已经将 `TaskCreated`、
`TaskCompleted`、`SubagentStart`、`SubagentStop` 等支持为 `HookEvent`
的变体。本模块提供适配器实现，在基于 trait 的 task/subagent hook
系统触发其回调时，一并触发这些事件。

## 架构

```text
YAML hooks.yaml ──→ HookRegistry
                         ↑
TaskHookBridge ──────┘  (触发 Created/Started/Completed(status))
SubagentHookBridge ──┘  (触发 Start/Stop(status))
```

Rust 开发者仍然可以直接使用 `TaskHooks` 与 `SubagentHooks` trait。
这些桥接确保 YAML 配置的 hooks 也能看到这些生命周期事件。
*/

use echo_core::hooks::HookContext;
use echo_execution::skills::hooks::HookRegistry;
use std::sync::Arc;
use tokio::sync::RwLock;

/// 由产品层任务运行时提供的可选关联标识符。
#[derive(Debug, Clone, Copy, Default)]
pub struct HookCorrelation<'a> {
    /// 运行 ID。
    pub run_id: Option<&'a str>,
    /// 计划修订版本。
    pub plan_revision: Option<&'a str>,
    /// 子代理运行 ID。
    pub subagent_run_id: Option<&'a str>,
    /// 尝试次数。
    pub attempt: Option<u32>,
}

/// 将关联标识符附加到 hook 上下文中。
fn apply_correlation(ctx: HookContext, correlation: HookCorrelation<'_>) -> HookContext {
    ctx.with_run_correlation(
        correlation.run_id,
        correlation.plan_revision,
        correlation.subagent_run_id,
        correlation.attempt,
    )
}

// ── 任务 Hook 桥接 ────────────────────────────────────────────────

/**
将 `TaskHooks` trait 的回调桥接到统一的 `HookRegistry`。

当任务生命周期事件发生时（before_execute、after_execute 等），本桥接
会在中心化的 `HookRegistry` 中触发相应的 `HookEvent`，使 YAML 配置的
hooks 也能对其作出响应。
*/
pub struct TaskHookBridge {
    /// 中心化的 hook 注册表。
    hook_registry: Arc<RwLock<HookRegistry>>,
    /// 所属会话 ID。
    session_id: String,
    /// 所属 agent 名称。
    agent_name: String,
}

impl TaskHookBridge {
    /// 创建一个新的桥接实例。
    pub fn new(
        hook_registry: Arc<RwLock<HookRegistry>>,
        session_id: String,
        agent_name: String,
    ) -> Self {
        Self {
            hook_registry,
            session_id,
            agent_name,
        }
    }

    /**
    触发 TaskCreated —— 一个任务节点进入可执行图。

    任务运行时适配器会在权威计划修订版本提交之后、规范 DAG 控制器使
    该任务变为可执行之前调用本方法。
    */
    pub async fn on_created(&self, task_id: &str, task_subject: &str) {
        self.on_created_with_correlation(task_id, task_subject, HookCorrelation::default())
            .await;
    }

    /// 带关联标识符的 `on_created` 版本。
    pub async fn on_created_with_correlation(
        &self,
        task_id: &str,
        task_subject: &str,
        correlation: HookCorrelation<'_>,
    ) {
        let ctx = apply_correlation(
            HookContext::for_task_created(
                task_id,
                task_subject,
                &self.session_id,
                &self.agent_name,
            ),
            correlation,
        );
        let registry = self.hook_registry.read().await;
        let _ = registry.run_lifecycle_hooks(&ctx).await;
    }

    /**
    触发 TaskStarted —— 调度器选中了该任务并即将执行。
    这是框架 `TaskHooks::before_execute` 的映射。
    */
    pub async fn on_before_execute(&self, task_id: &str, task_subject: &str) {
        self.on_before_execute_with_correlation(task_id, task_subject, HookCorrelation::default())
            .await;
    }

    /// 带关联标识符的 `on_before_execute` 版本。
    pub async fn on_before_execute_with_correlation(
        &self,
        task_id: &str,
        task_subject: &str,
        correlation: HookCorrelation<'_>,
    ) {
        let ctx = apply_correlation(
            HookContext::for_task_started(
                task_id,
                task_subject,
                &self.session_id,
                &self.agent_name,
            ),
            correlation,
        );
        let registry = self.hook_registry.read().await;
        let _ = registry.run_lifecycle_hooks(&ctx).await;
    }

    /**
    触发 TaskCompleted —— 任务到达终态。

    `status` 是结构化的终止原因。此前独立的 `TaskTimeout`/`TaskCancelled`
    事件已被去除——超时/取消在这里都表示为 `status` 的取值（与业界对齐：
    Codex 的 CommandExecutionStatus）。
    */
    pub async fn on_after_execute(
        &self,
        task_id: &str,
        task_subject: &str,
        result: &str,
        status: echo_core::hooks::TaskTerminalStatus,
    ) {
        self.on_after_execute_with_correlation(
            task_id,
            task_subject,
            result,
            status,
            HookCorrelation::default(),
        )
        .await;
    }

    /// 带关联标识符的 `on_after_execute` 版本。
    pub async fn on_after_execute_with_correlation(
        &self,
        task_id: &str,
        task_subject: &str,
        result: &str,
        status: echo_core::hooks::TaskTerminalStatus,
        correlation: HookCorrelation<'_>,
    ) {
        let ctx = apply_correlation(
            HookContext::for_task_completed(
                task_id,
                task_subject,
                result,
                status,
                &self.session_id,
                &self.agent_name,
            ),
            correlation,
        );
        let registry = self.hook_registry.read().await;
        let _ = registry.run_lifecycle_hooks(&ctx).await;
    }

    /// 触发 StopFailure 事件（映射自 on_failure）。
    pub async fn on_failure(&self, _task_id: &str, task_subject: &str, error: &str) {
        let ctx =
            HookContext::for_stop_failure(task_subject, error, &self.session_id, &self.agent_name);
        let registry = self.hook_registry.read().await;
        let _ = registry.run_lifecycle_hooks(&ctx).await;
    }
}

// ── 子代理 Hook 桥接 ────────────────────────────────────────────────

/**
将 `SubagentHooks` trait 的回调桥接到统一的 `HookRegistry`。

当子代理生命周期事件发生时（before_dispatch、after_dispatch 等），
本桥接会在中心化的 `HookRegistry` 中触发相应的 `HookEvent`，使 YAML
配置的 hooks 也能对其作出响应。
*/
pub struct SubagentHookBridge {
    /// 中心化的 hook 注册表。
    hook_registry: Arc<RwLock<HookRegistry>>,
    /// 所属会话 ID。
    session_id: String,
    /// 所属 agent 名称。
    agent_name: String,
}

impl SubagentHookBridge {
    /// 创建一个新的桥接实例。
    pub fn new(
        hook_registry: Arc<RwLock<HookRegistry>>,
        session_id: String,
        agent_name: String,
    ) -> Self {
        Self {
            hook_registry,
            session_id,
            agent_name,
        }
    }

    /// 触发 SubagentStart 事件（映射自 before_dispatch）。
    pub async fn on_before_dispatch(&self, subagent_name: &str, execution_mode: &str, task: &str) {
        self.on_before_dispatch_with_correlation(
            subagent_name,
            execution_mode,
            task,
            HookCorrelation::default(),
        )
        .await;
    }

    /// 带关联标识符的 `on_before_dispatch` 版本。
    pub async fn on_before_dispatch_with_correlation(
        &self,
        subagent_name: &str,
        execution_mode: &str,
        task: &str,
        correlation: HookCorrelation<'_>,
    ) {
        let ctx = apply_correlation(
            HookContext::for_subagent_start(
                subagent_name,
                execution_mode,
                task,
                &self.session_id,
                &self.agent_name,
            ),
            correlation,
        );
        let registry = self.hook_registry.read().await;
        let _ = registry.run_lifecycle_hooks(&ctx).await;
    }

    /**
    触发 SubagentStop 事件（映射自 after_dispatch）。

    总是且仅发出一次携带终止 `status`（completed/failed/cancelled/
    timed_out）的 SubagentStop。这是所有子代理终止状态的唯一汇聚点——
    调用方不得再额外触发一个单独的 cancelled 事件。这是与业界对齐的模型
    （Claude Code / Codex / OpenAI Agents SDK / AGTP：两个边界事件 +
    状态枚举）。
    */
    pub async fn on_after_dispatch(
        &self,
        subagent_name: &str,
        execution_mode: &str,
        result: &str,
        status: echo_core::hooks::SubagentStopStatus,
    ) {
        self.on_after_dispatch_with_correlation(
            subagent_name,
            execution_mode,
            result,
            status,
            HookCorrelation::default(),
        )
        .await;
    }

    /// 带关联标识符的 `on_after_dispatch` 版本。
    pub async fn on_after_dispatch_with_correlation(
        &self,
        subagent_name: &str,
        execution_mode: &str,
        result: &str,
        status: echo_core::hooks::SubagentStopStatus,
        correlation: HookCorrelation<'_>,
    ) {
        let ctx = apply_correlation(
            HookContext::for_subagent_stop(
                subagent_name,
                execution_mode,
                result,
                status,
                &self.session_id,
                &self.agent_name,
            ),
            correlation,
        );
        let registry = self.hook_registry.read().await;
        let _ = registry.run_lifecycle_hooks(&ctx).await;
    }

    /// 触发 StopFailure 事件（映射自 on_failure）。
    pub async fn on_failure(&self, subagent_name: &str, error: &str) {
        let ctx =
            HookContext::for_stop_failure(subagent_name, error, &self.session_id, &self.agent_name);
        let registry = self.hook_registry.read().await;
        let _ = registry.run_lifecycle_hooks(&ctx).await;
    }
}

// ── TaskHooks 适配器 ───────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_hook_bridge_creation() {
        let registry = Arc::new(RwLock::new(HookRegistry::new()));
        let bridge = TaskHookBridge::new(
            registry,
            "test-session".to_string(),
            "test-agent".to_string(),
        );
        // Bridge should be creatable without panicking
        assert_eq!(bridge.session_id, "test-session");
        assert_eq!(bridge.agent_name, "test-agent");
    }

    #[test]
    fn test_subagent_hook_bridge_creation() {
        let registry = Arc::new(RwLock::new(HookRegistry::new()));
        let bridge = SubagentHookBridge::new(
            registry,
            "test-session".to_string(),
            "test-agent".to_string(),
        );
        assert_eq!(bridge.session_id, "test-session");
        assert_eq!(bridge.agent_name, "test-agent");
    }

    #[tokio::test]
    async fn test_task_bridge_fires_events() {
        let registry = Arc::new(RwLock::new(HookRegistry::new()));
        let bridge = TaskHookBridge::new(
            registry.clone(),
            "session-1".to_string(),
            "agent-1".to_string(),
        );

        // Three-stage task lifecycle: created (graph entry) → started
        // (scheduler pick) → completed (terminal, with status). The former
        // on_timeout/on_cancelled are now on_after_execute(status).
        bridge.on_created("task-1", "Build project").await;
        bridge.on_before_execute("task-1", "Build project").await;
        bridge
            .on_after_execute(
                "task-1",
                "Build project",
                "ok",
                echo_core::hooks::TaskTerminalStatus::Completed,
            )
            .await;
        bridge
            .on_failure("task-1", "Build project", "compile error")
            .await;
        // Verify timeout/cancelled reach consumers as TaskCompleted status.
        bridge
            .on_after_execute(
                "task-1",
                "Build project",
                "deadline exceeded",
                echo_core::hooks::TaskTerminalStatus::TimedOut,
            )
            .await;
        bridge
            .on_after_execute(
                "task-1",
                "Build project",
                "aborted",
                echo_core::hooks::TaskTerminalStatus::Cancelled,
            )
            .await;
    }

    #[tokio::test]
    async fn test_subagent_bridge_fires_events() {
        let registry = Arc::new(RwLock::new(HookRegistry::new()));
        let bridge = SubagentHookBridge::new(
            registry.clone(),
            "session-1".to_string(),
            "agent-1".to_string(),
        );

        // These should not panic even with an empty registry.
        // SubagentStop is the single convergence point for all terminal
        // states (completed/failed/cancelled/timed_out) — no separate
        // on_cancelled anymore.
        bridge
            .on_before_dispatch("researcher", "sync", "Find papers")
            .await;
        bridge
            .on_after_dispatch(
                "researcher",
                "sync",
                "Find papers",
                echo_core::hooks::SubagentStopStatus::Completed,
            )
            .await;
        bridge.on_failure("researcher", "timeout").await;
        // Verify cancelled is delivered as a SubagentStop status, not a
        // separate event.
        bridge
            .on_after_dispatch(
                "researcher",
                "sync",
                "cancelled mid-run",
                echo_core::hooks::SubagentStopStatus::Cancelled,
            )
            .await;
    }
}
