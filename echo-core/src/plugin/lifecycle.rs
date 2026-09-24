//! 插件生命周期 trait —— 挂载到插件状态转换上的钩子。
//!
//! 插件通过实现这个 trait 来在其生命周期的关键时刻接收回调：
//! 加载、激活、停用和关闭。

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// 插件的生命周期回调。
///
/// agent 框架会在合适的时机调用这些方法：
///
/// ```text
/// load → init → activate ⇄ deactivate → shutdown
///                    ↑          ↓
///                    └──────────┘  (可在重新加载时循环)
/// ```
pub trait PluginLifecycle: Send + Sync {
    /// 在插件加载完毕、其组件已注册之后调用一次。
    ///
    /// 用于执行一次性的初始化工作：启动后台进程、
    /// 建立连接、初始化缓存等。
    fn init(&self) -> Result<(), String> {
        Ok(())
    }

    /// 在插件被启用时调用（若 `default_enabled: true`，则启动时也会调用）。
    ///
    /// 此时组件已经完成装配。用于执行激活相关的逻辑，
    /// 例如启动监控器。
    fn activate(&self) -> Result<(), String> {
        Ok(())
    }

    /// 在插件被禁用时调用。
    ///
    /// 用于停止后台进程并释放资源。
    /// 该方法返回后，组件将被解除装配。
    fn deactivate(&self) -> Result<(), String> {
        Ok(())
    }

    /// 在 agent 关闭时调用。
    ///
    /// 用于执行最终清理：刷新缓冲区、关闭连接、
    /// 将状态保存到 `${ECHO_PLUGIN_DATA}`。
    fn shutdown(&self) -> Result<(), String> {
        Ok(())
    }
}

/// 面向不需要回调的插件的空实现。
pub struct NoopLifecycle;

impl PluginLifecycle for NoopLifecycle {}

/// 单个插件的受管生命周期状态，记录其回调实例及各阶段的完成情况。
struct ManagedLifecycle {
    /// 该插件注册的生命周期回调实现。
    callbacks: Arc<dyn PluginLifecycle>,
    /// 是否已成功执行过 `init`。
    initialized: bool,
    /// 是否处于已激活状态。
    active: bool,
    // 回调的获取（init/activate）或停用过程可能只完成一部分；shutdown 有自己独立的结算逻辑。
    /// 是否存在尚未结算的停用（deactivate）债务，需要重试。
    cleanup_required: bool,
    /// 是否存在尚未结算的关闭（shutdown）债务，需要重试。
    shutdown_required: bool,
}

/// 持有生命周期回调，并根据已启用插件的集合驱动它们的状态转换。
#[derive(Default)]
pub struct PluginLifecycleManager {
    /// 按插件 ID 索引的受管生命周期集合。
    plugins: HashMap<String, ManagedLifecycle>,
}

impl PluginLifecycleManager {
    /// 创建一个空的生命周期管理器。
    pub fn new() -> Self {
        Self::default()
    }

    /// 在下一次协调（reconcile）之前，为某个插件注册回调。
    pub fn register(
        &mut self,
        plugin_id: impl Into<String>,
        callbacks: Arc<dyn PluginLifecycle>,
    ) -> Result<(), String> {
        let plugin_id = plugin_id.into();
        if self.plugins.contains_key(&plugin_id) {
            return Err(format!(
                "Lifecycle callbacks already registered for plugin '{plugin_id}'"
            ));
        }
        self.plugins.insert(
            plugin_id,
            ManagedLifecycle {
                callbacks,
                initialized: false,
                active: false,
                cleanup_required: false,
                shutdown_required: false,
            },
        );
        Ok(())
    }

    /// 初始化并激活一个已注册的插件；每次状态转换只会执行一次。
    pub fn activate(&mut self, plugin_id: &str) -> Result<(), String> {
        let lifecycle = self
            .plugins
            .get(plugin_id)
            .ok_or_else(|| format!("No lifecycle callbacks registered for '{plugin_id}'"))?;
        if lifecycle.active && !lifecycle.cleanup_required && !lifecycle.shutdown_required {
            return Ok(());
        }
        // 失败的回调可能仍然持有副作用；在其结算完成之前，不允许开始任何其它代（generation）。
        if let Some(blocker) = self
            .plugins
            .iter()
            .filter(|(_, lifecycle)| lifecycle.cleanup_required || lifecycle.shutdown_required)
            .map(|(id, _)| id.as_str())
            .min()
        {
            return Err(format!(
                "Plugin '{plugin_id}' activation blocked by unresolved lifecycle cleanup debt for '{blocker}'"
            ));
        }
        let lifecycle = self
            .plugins
            .get_mut(plugin_id)
            .ok_or_else(|| format!("No lifecycle callbacks registered for '{plugin_id}'"))?;
        if !lifecycle.initialized {
            if lifecycle.cleanup_required || lifecycle.shutdown_required {
                return Err(format!(
                    "Plugin '{plugin_id}' has unresolved lifecycle cleanup debt"
                ));
            }
            if let Err(error) = lifecycle.callbacks.init() {
                lifecycle.shutdown_required = true;
                return Err(format!("Plugin '{plugin_id}' init failed: {error}"));
            }
            lifecycle.initialized = true;
        }
        if !lifecycle.active {
            if let Err(error) = lifecycle.callbacks.activate() {
                lifecycle.cleanup_required = true;
                return Err(format!("Plugin '{plugin_id}' activation failed: {error}"));
            }
            lifecycle.active = true;
        }
        Ok(())
    }

    /// 在插件当前处于激活状态时将其停用。
    pub fn deactivate(&mut self, plugin_id: &str) -> Result<(), String> {
        let lifecycle = self
            .plugins
            .get_mut(plugin_id)
            .ok_or_else(|| format!("No lifecycle callbacks registered for '{plugin_id}'"))?;
        if lifecycle.active {
            if let Err(error) = lifecycle.callbacks.deactivate() {
                lifecycle.cleanup_required = true;
                return Err(format!("Plugin '{plugin_id}' deactivation failed: {error}"));
            }
            lifecycle.active = false;
            lifecycle.cleanup_required = false;
        }
        Ok(())
    }

    /// 在运行时范围的重新装配之前，停用所有当前处于激活状态的回调。
    pub fn deactivate_all(&mut self) -> Vec<String> {
        let mut ids = self.plugins.keys().cloned().collect::<Vec<_>>();
        ids.sort();
        self.deactivate_in_order(ids.iter().map(String::as_str))
    }

    /// 按调用方指定的依赖顺序停用回调。
    pub fn deactivate_in_order<'a>(
        &mut self,
        plugin_ids: impl IntoIterator<Item = &'a str>,
    ) -> Vec<String> {
        let mut errors = Vec::new();
        for plugin_id in plugin_ids {
            if self.plugins.contains_key(plugin_id)
                && let Err(error) = self.deactivate(plugin_id)
            {
                errors.push(error);
            }
        }
        errors
    }

    /// 在释放已卸载插件的资源之后，移除其回调。
    ///
    /// 清理失败时会保留所有权，以便调用方重试，
    /// 而不是替换掉可能仍持有活跃资源的回调。
    pub fn unregister(&mut self, plugin_id: &str) -> Result<bool, String> {
        if !self.plugins.contains_key(plugin_id) {
            return Ok(false);
        }
        self.reset_for_retry(plugin_id)?;
        self.plugins.remove(plugin_id);
        Ok(true)
    }

    /// 结算回调产生的副作用，同时保留该注册项以便重试。
    ///
    /// 一次失败的 init 或 activate 可能已经获取了资源，即便回调
    /// 从未真正变为激活状态。该生命周期权威方会执行与 unregister
    /// 相同的 deactivate/shutdown 结算流程，但会保留回调本身，
    /// 以便协调者之后能够初始化下一代真正的实例。
    pub fn reset_for_retry(&mut self, plugin_id: &str) -> Result<bool, String> {
        let Some(lifecycle) = self.plugins.get_mut(plugin_id) else {
            return Ok(false);
        };
        let mut errors = Vec::new();
        if lifecycle.active || lifecycle.cleanup_required {
            if let Err(error) = lifecycle.callbacks.deactivate() {
                lifecycle.cleanup_required = true;
                errors.push(format!(
                    "Plugin '{plugin_id}' deactivation failed during lifecycle cleanup: {error}"
                ));
            } else {
                lifecycle.active = false;
                lifecycle.cleanup_required = false;
            }
        }
        if lifecycle.initialized || lifecycle.shutdown_required {
            if let Err(error) = lifecycle.callbacks.shutdown() {
                lifecycle.shutdown_required = true;
                errors.push(format!(
                    "Plugin '{plugin_id}' shutdown failed during lifecycle cleanup: {error}"
                ));
            } else {
                lifecycle.initialized = false;
                lifecycle.shutdown_required = false;
            }
        }
        if errors.is_empty() {
            Ok(true)
        } else {
            Err(errors.join("; "))
        }
    }

    /// 将已注册的回调与注册表中已启用插件的集合进行协调。
    /// 未解决的下线（withdrawal）会阻塞所有新的激活，直到清理成功为止。
    pub fn reconcile<'a>(
        &mut self,
        enabled_plugins: impl IntoIterator<Item = &'a str>,
    ) -> Vec<String> {
        let enabled = enabled_plugins.into_iter().collect::<HashSet<_>>();
        let mut errors = self.deactivate_not_in_set(&enabled);
        errors.extend(self.activate_in_set(&enabled));
        errors
    }

    /// 停用那些插件不在下一个集合中的、当前处于激活状态的回调。
    /// 内嵌运行时会在解除插件组件装配之前调用此方法。
    pub fn deactivate_not_in<'a>(
        &mut self,
        enabled_plugins: impl IntoIterator<Item = &'a str>,
    ) -> Vec<String> {
        let enabled = enabled_plugins.into_iter().collect::<HashSet<_>>();
        self.deactivate_not_in_set(&enabled)
    }

    /// 初始化并激活当前已启用集合中存在的回调。
    /// 内嵌运行时会在插件组件完成装配之后调用此方法。
    pub fn activate_enabled<'a>(
        &mut self,
        enabled_plugins: impl IntoIterator<Item = &'a str>,
    ) -> Vec<String> {
        let enabled = enabled_plugins.into_iter().collect::<HashSet<_>>();
        self.activate_in_set(&enabled)
    }

    /// 按依赖顺序激活已注册的回调，跳过那些未定义生命周期回调的插件。
    pub fn activate_in_order<'a>(
        &mut self,
        plugin_ids: impl IntoIterator<Item = &'a str>,
    ) -> Vec<String> {
        let mut errors = Vec::new();
        for plugin_id in plugin_ids {
            if self.plugins.contains_key(plugin_id)
                && let Err(error) = self.activate(plugin_id)
            {
                errors.push(error);
            }
        }
        errors
    }

    /// 按调用方指定的反向依赖顺序注销回调。
    pub fn unregister_in_order<'a>(
        &mut self,
        plugin_ids: impl IntoIterator<Item = &'a str>,
    ) -> Vec<String> {
        let mut errors = Vec::new();
        for plugin_id in plugin_ids {
            if let Err(error) = self.unregister(plugin_id) {
                errors.push(error);
            }
        }
        errors
    }

    /// 停用所有不在 `enabled` 集合中的插件；内部辅助方法。
    fn deactivate_not_in_set(&mut self, enabled: &HashSet<&str>) -> Vec<String> {
        let mut ids = self.plugins.keys().cloned().collect::<Vec<_>>();
        ids.sort();
        let mut errors = Vec::new();
        for plugin_id in ids {
            if !enabled.contains(plugin_id.as_str())
                && let Err(error) = self.deactivate(&plugin_id)
            {
                errors.push(error);
            }
        }
        errors
    }

    /// 激活所有在 `enabled` 集合中的插件；内部辅助方法。
    fn activate_in_set(&mut self, enabled: &HashSet<&str>) -> Vec<String> {
        let mut ids = self.plugins.keys().cloned().collect::<Vec<_>>();
        ids.sort();
        let mut errors = Vec::new();
        for plugin_id in ids {
            if enabled.contains(plugin_id.as_str())
                && let Err(error) = self.activate(&plugin_id)
            {
                errors.push(error);
            }
        }
        errors
    }

    /// 停用所有处于激活状态的插件，并调用它们最终的关闭回调。
    pub fn shutdown(&mut self) -> Vec<String> {
        let mut ids = self.plugins.keys().cloned().collect::<Vec<_>>();
        ids.sort();
        let mut errors = Vec::new();
        for plugin_id in ids {
            if let Err(error) = self.unregister(&plugin_id) {
                errors.push(error);
            }
        }
        errors
    }
}

impl Drop for PluginLifecycleManager {
    /// 在管理器被丢弃时执行关闭流程，并对失败的回调记录警告日志。
    fn drop(&mut self) {
        for error in self.shutdown() {
            tracing::warn!(%error, "Plugin lifecycle shutdown callback failed");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// 记录各生命周期回调被调用次数的计数器。
    #[derive(Default)]
    struct Counts {
        /// `init` 被调用的次数。
        init: AtomicUsize,
        /// `activate` 被调用的次数。
        activate: AtomicUsize,
        /// `deactivate` 被调用的次数。
        deactivate: AtomicUsize,
        /// `shutdown` 被调用的次数。
        shutdown: AtomicUsize,
    }

    /// 对每个回调调用进行计数、不产生失败的测试用生命周期实现。
    struct CountingLifecycle(Arc<Counts>);

    /// 停用和关闭均会失败的测试用生命周期实现。
    struct FailingCleanupLifecycle;

    /// 激活会失败但会计数的测试用生命周期实现。
    struct FailingActivationLifecycle(Arc<Counts>);

    /// 初始化会失败的测试用生命周期实现。
    struct FailingInitializationLifecycle(Arc<Counts>);

    /// 停用是否失败可通过原子标志动态切换的测试用生命周期实现，用于验证重试逻辑。
    struct RetryingDeactivateLifecycle {
        /// 调用计数器。
        counts: Arc<Counts>,
        /// 控制 `deactivate` 是否应当失败的开关。
        fail_deactivate: Arc<std::sync::atomic::AtomicBool>,
    }

    /// 停用与关闭是否失败均可通过原子标志动态切换的测试用生命周期实现，用于验证清理结算逻辑。
    struct RetryingCleanupLifecycle {
        /// 调用计数器。
        counts: Arc<Counts>,
        /// 控制 `deactivate` 是否应当失败的开关。
        fail_deactivate: Arc<std::sync::atomic::AtomicBool>,
        /// 控制 `shutdown` 是否应当失败的开关。
        fail_shutdown: Arc<std::sync::atomic::AtomicBool>,
    }

    impl PluginLifecycle for CountingLifecycle {
        fn init(&self) -> Result<(), String> {
            self.0.init.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        fn activate(&self) -> Result<(), String> {
            self.0.activate.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        fn deactivate(&self) -> Result<(), String> {
            self.0.deactivate.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        fn shutdown(&self) -> Result<(), String> {
            self.0.shutdown.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    impl PluginLifecycle for FailingCleanupLifecycle {
        fn deactivate(&self) -> Result<(), String> {
            Err("injected deactivate failure".to_string())
        }

        fn shutdown(&self) -> Result<(), String> {
            Err("injected shutdown failure".to_string())
        }
    }

    impl PluginLifecycle for FailingActivationLifecycle {
        fn activate(&self) -> Result<(), String> {
            self.0.activate.fetch_add(1, Ordering::SeqCst);
            Err("injected activation failure".to_string())
        }
    }

    impl PluginLifecycle for FailingInitializationLifecycle {
        fn init(&self) -> Result<(), String> {
            Err("injected init failure".to_string())
        }

        fn deactivate(&self) -> Result<(), String> {
            self.0.deactivate.fetch_add(1, Ordering::SeqCst);
            Err("deactivate called before activation".to_string())
        }

        fn shutdown(&self) -> Result<(), String> {
            self.0.shutdown.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    impl PluginLifecycle for RetryingDeactivateLifecycle {
        fn activate(&self) -> Result<(), String> {
            self.counts.activate.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        fn deactivate(&self) -> Result<(), String> {
            self.counts.deactivate.fetch_add(1, Ordering::SeqCst);
            if self.fail_deactivate.load(Ordering::SeqCst) {
                Err("injected deactivate failure".to_string())
            } else {
                Ok(())
            }
        }
    }

    impl PluginLifecycle for RetryingCleanupLifecycle {
        fn deactivate(&self) -> Result<(), String> {
            self.counts.deactivate.fetch_add(1, Ordering::SeqCst);
            if self.fail_deactivate.load(Ordering::SeqCst) {
                Err("injected deactivate failure".to_string())
            } else {
                Ok(())
            }
        }

        fn shutdown(&self) -> Result<(), String> {
            self.counts.shutdown.fetch_add(1, Ordering::SeqCst);
            if self.fail_shutdown.load(Ordering::SeqCst) {
                Err("injected shutdown failure".to_string())
            } else {
                Ok(())
            }
        }
    }

    #[test]
    fn lifecycle_manager_drives_enabled_state_and_shutdown_once() -> Result<(), String> {
        let counts = Arc::new(Counts::default());
        let mut manager = PluginLifecycleManager::new();
        manager.register("example", Arc::new(CountingLifecycle(Arc::clone(&counts))))?;

        assert!(manager.reconcile(["example"]).is_empty());
        assert!(manager.reconcile(["example"]).is_empty());
        assert!(manager.reconcile(std::iter::empty()).is_empty());
        assert!(manager.shutdown().is_empty());

        assert_eq!(counts.init.load(Ordering::SeqCst), 1);
        assert_eq!(counts.activate.load(Ordering::SeqCst), 1);
        assert_eq!(counts.deactivate.load(Ordering::SeqCst), 1);
        assert_eq!(counts.shutdown.load(Ordering::SeqCst), 1);
        Ok(())
    }

    #[test]
    fn unregister_releases_callbacks_and_allows_reregistration() -> Result<(), String> {
        let first = Arc::new(Counts::default());
        let second = Arc::new(Counts::default());
        let mut manager = PluginLifecycleManager::new();
        manager.register("example", Arc::new(CountingLifecycle(Arc::clone(&first))))?;
        manager.activate("example")?;

        assert!(manager.unregister("example")?);
        assert_eq!(first.deactivate.load(Ordering::SeqCst), 1);
        assert_eq!(first.shutdown.load(Ordering::SeqCst), 1);

        manager.register("example", Arc::new(CountingLifecycle(Arc::clone(&second))))?;
        manager.activate("example")?;
        assert_eq!(second.init.load(Ordering::SeqCst), 1);
        assert_eq!(second.activate.load(Ordering::SeqCst), 1);
        Ok(())
    }

    #[test]
    fn deactivate_all_brackets_reactivation_without_reinitializing() -> Result<(), String> {
        let counts = Arc::new(Counts::default());
        let mut manager = PluginLifecycleManager::new();
        manager.register("example", Arc::new(CountingLifecycle(Arc::clone(&counts))))?;
        manager.activate("example")?;

        assert!(manager.deactivate_all().is_empty());
        assert!(manager.activate_enabled(["example"]).is_empty());

        assert_eq!(counts.init.load(Ordering::SeqCst), 1);
        assert_eq!(counts.activate.load(Ordering::SeqCst), 2);
        assert_eq!(counts.deactivate.load(Ordering::SeqCst), 1);
        Ok(())
    }

    #[test]
    fn failed_unregister_retains_cleanup_ownership() -> Result<(), String> {
        let mut manager = PluginLifecycleManager::new();
        manager.register("example", Arc::new(FailingCleanupLifecycle))?;
        manager.activate("example")?;

        let error = manager
            .unregister("example")
            .err()
            .ok_or_else(|| "failing cleanup unexpectedly succeeded".to_string())?;
        assert!(error.contains("injected deactivate failure"));
        assert!(error.contains("injected shutdown failure"));
        assert!(
            manager
                .register("example", Arc::new(NoopLifecycle))
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn failed_withdrawal_blocks_reconcile_and_direct_activation_until_retry() -> Result<(), String>
    {
        let old_counts = Arc::new(Counts::default());
        let next_counts = Arc::new(Counts::default());
        let retained_counts = Arc::new(Counts::default());
        let fail_deactivate = Arc::new(std::sync::atomic::AtomicBool::new(true));
        let mut manager = PluginLifecycleManager::new();
        manager.register(
            "old",
            Arc::new(RetryingDeactivateLifecycle {
                counts: Arc::clone(&old_counts),
                fail_deactivate: Arc::clone(&fail_deactivate),
            }),
        )?;
        manager.register(
            "next",
            Arc::new(CountingLifecycle(Arc::clone(&next_counts))),
        )?;
        manager.register(
            "retained",
            Arc::new(CountingLifecycle(Arc::clone(&retained_counts))),
        )?;
        manager.activate("old")?;
        manager.activate("retained")?;

        let errors = manager.reconcile(["next", "retained"]);
        assert!(
            errors
                .iter()
                .any(|error| error.contains("deactivation failed"))
        );
        assert!(
            errors
                .iter()
                .any(|error| error.contains("activation blocked"))
        );
        assert!(manager.activate("next").is_err());
        manager.activate("retained")?;
        assert_eq!(retained_counts.activate.load(Ordering::SeqCst), 1);
        assert_eq!(next_counts.init.load(Ordering::SeqCst), 0);
        assert_eq!(next_counts.activate.load(Ordering::SeqCst), 0);

        fail_deactivate.store(false, Ordering::SeqCst);
        assert!(manager.reconcile(["next", "retained"]).is_empty());
        assert_eq!(old_counts.deactivate.load(Ordering::SeqCst), 2);
        assert_eq!(next_counts.activate.load(Ordering::SeqCst), 1);
        assert!(manager.reconcile(["next", "retained"]).is_empty());
        assert_eq!(next_counts.activate.load(Ordering::SeqCst), 1);
        Ok(())
    }

    #[test]
    fn failed_deactivate_all_blocks_separate_activation_phase() -> Result<(), String> {
        let old_counts = Arc::new(Counts::default());
        let next_counts = Arc::new(Counts::default());
        let fail_deactivate = Arc::new(std::sync::atomic::AtomicBool::new(true));
        let mut manager = PluginLifecycleManager::new();
        manager.register(
            "old",
            Arc::new(RetryingDeactivateLifecycle {
                counts: Arc::clone(&old_counts),
                fail_deactivate: Arc::clone(&fail_deactivate),
            }),
        )?;
        manager.register(
            "next",
            Arc::new(CountingLifecycle(Arc::clone(&next_counts))),
        )?;
        manager.activate("old")?;

        assert_eq!(manager.deactivate_all().len(), 1);
        assert!(
            manager
                .activate_enabled(["next"])
                .iter()
                .any(|error| error.contains("old"))
        );
        assert_eq!(next_counts.activate.load(Ordering::SeqCst), 0);

        fail_deactivate.store(false, Ordering::SeqCst);
        assert!(manager.deactivate_not_in(["next"]).is_empty());
        assert!(manager.activate_enabled(["next"]).is_empty());
        assert_eq!(next_counts.activate.load(Ordering::SeqCst), 1);
        Ok(())
    }

    #[test]
    fn failed_activation_blocks_later_callbacks_until_cleanup() -> Result<(), String> {
        let failed_counts = Arc::new(Counts::default());
        let later_counts = Arc::new(Counts::default());
        let mut manager = PluginLifecycleManager::new();
        manager.register(
            "a-failing",
            Arc::new(FailingActivationLifecycle(Arc::clone(&failed_counts))),
        )?;
        manager.register(
            "z-later",
            Arc::new(CountingLifecycle(Arc::clone(&later_counts))),
        )?;

        let errors = manager.reconcile(["a-failing", "z-later"]);
        assert!(
            errors
                .iter()
                .any(|error| error.contains("activation failed"))
        );
        assert!(
            errors
                .iter()
                .any(|error| error.contains("activation blocked"))
        );
        assert_eq!(later_counts.activate.load(Ordering::SeqCst), 0);
        assert!(
            manager
                .reconcile(["z-later"])
                .iter()
                .any(|error| error.contains("a-failing"))
        );

        assert!(manager.unregister("a-failing")?);
        assert!(manager.activate_enabled(["z-later"]).is_empty());
        assert_eq!(failed_counts.activate.load(Ordering::SeqCst), 1);
        assert_eq!(later_counts.activate.load(Ordering::SeqCst), 1);
        Ok(())
    }

    #[test]
    fn successful_deactivate_does_not_settle_failed_unregister_shutdown() -> Result<(), String> {
        let old_counts = Arc::new(Counts::default());
        let next_counts = Arc::new(Counts::default());
        let fail_deactivate = Arc::new(std::sync::atomic::AtomicBool::new(true));
        let fail_shutdown = Arc::new(std::sync::atomic::AtomicBool::new(true));
        let mut manager = PluginLifecycleManager::new();
        manager.register(
            "old",
            Arc::new(RetryingCleanupLifecycle {
                counts: Arc::clone(&old_counts),
                fail_deactivate: Arc::clone(&fail_deactivate),
                fail_shutdown: Arc::clone(&fail_shutdown),
            }),
        )?;
        manager.register(
            "next",
            Arc::new(CountingLifecycle(Arc::clone(&next_counts))),
        )?;
        manager.activate("old")?;

        let error = manager
            .unregister("old")
            .err()
            .ok_or_else(|| "both cleanup callbacks unexpectedly succeeded".to_string())?;
        assert!(error.contains("deactivation failed"));
        assert!(error.contains("shutdown failed"));

        fail_deactivate.store(false, Ordering::SeqCst);
        manager.deactivate("old")?;
        assert!(manager.activate("next").is_err());
        assert!(manager.activate("old").is_err());
        assert_eq!(next_counts.activate.load(Ordering::SeqCst), 0);

        fail_shutdown.store(false, Ordering::SeqCst);
        assert!(manager.unregister("old")?);
        manager.activate("next")?;
        assert_eq!(old_counts.deactivate.load(Ordering::SeqCst), 2);
        assert_eq!(old_counts.shutdown.load(Ordering::SeqCst), 2);
        assert_eq!(next_counts.activate.load(Ordering::SeqCst), 1);
        Ok(())
    }

    #[test]
    fn shutdown_retry_does_not_repeat_successful_deactivation() -> Result<(), String> {
        let counts = Arc::new(Counts::default());
        let fail_shutdown = Arc::new(std::sync::atomic::AtomicBool::new(true));
        let mut manager = PluginLifecycleManager::new();
        manager.register(
            "old",
            Arc::new(RetryingCleanupLifecycle {
                counts: Arc::clone(&counts),
                fail_deactivate: Arc::new(std::sync::atomic::AtomicBool::new(false)),
                fail_shutdown: Arc::clone(&fail_shutdown),
            }),
        )?;
        manager.register("next", Arc::new(NoopLifecycle))?;
        manager.activate("old")?;

        assert!(manager.unregister("old").is_err());
        manager.deactivate("old")?;
        assert!(manager.activate("next").is_err());
        fail_shutdown.store(false, Ordering::SeqCst);
        assert!(manager.unregister("old")?);
        assert_eq!(counts.deactivate.load(Ordering::SeqCst), 1);
        assert_eq!(counts.shutdown.load(Ordering::SeqCst), 2);
        manager.activate("next")?;
        Ok(())
    }

    #[test]
    fn failed_init_requires_shutdown_without_deactivation() -> Result<(), String> {
        let counts = Arc::new(Counts::default());
        let mut manager = PluginLifecycleManager::new();
        manager.register(
            "old",
            Arc::new(FailingInitializationLifecycle(Arc::clone(&counts))),
        )?;
        assert!(manager.activate("old").is_err());

        assert!(manager.unregister("old")?);
        assert_eq!(counts.deactivate.load(Ordering::SeqCst), 0);
        assert_eq!(counts.shutdown.load(Ordering::SeqCst), 1);
        Ok(())
    }
}
