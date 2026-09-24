/*!
分层记忆的运行时装配辅助工具。

该模块只负责框架层面的管线搭建：创建 layer manager、构建 change-log
以及写入观察者。产品层面的生命周期策略和调度逻辑仍然保留在应用代码中。
*/

use super::{ChangeLog, EvolutionObserver, JsonlChangeLog, MemoryLayerManager};
use crate::error::Result;
use crate::memory::Store;
use crate::skills::hooks::{HookContext, HookRegistry};
use futures::future::BoxFuture;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;

/// 以一致的运行时装配方式构建 [`MemoryLayerManager`]。
pub struct MemoryRuntimeIntegrationBuilder {
    /// `.echo-agent/` 目录路径
    echo_agent_dir: PathBuf,
    /// 底层存储实现
    store: Arc<dyn Store>,
    /// 成功发生持久化演化事件后要通知的观察者
    evolution_observer: Option<Arc<dyn EvolutionObserver>>,
    /// 自定义的 JSONL change-log 路径（未设置时使用默认路径）
    change_log_path: Option<PathBuf>,
}

impl MemoryRuntimeIntegrationBuilder {
    /// 为给定的 `.echo-agent/` 目录和 Store 创建一个新的构建器。
    pub fn new(echo_agent_dir: PathBuf, store: Arc<dyn Store>) -> Self {
        Self {
            echo_agent_dir,
            store,
            evolution_observer: None,
            change_log_path: None,
        }
    }

    /// 注册一个观察者，在持久化的演化事件成功发生后被调用。
    pub fn evolution_observer(mut self, observer: Arc<dyn EvolutionObserver>) -> Self {
        self.evolution_observer = Some(observer);
        self
    }

    /// 覆盖 JSONL change-log 的路径。
    pub fn change_log_path(mut self, path: PathBuf) -> Self {
        self.change_log_path = Some(path);
        self
    }

    /// 返回该集成使用的默认 change-log 路径。
    pub fn default_change_log_path(&self) -> PathBuf {
        self.echo_agent_dir
            .join("evolution")
            .join("change-log.jsonl")
    }

    /// 创建一个 JSONL change log，并确保其父目录存在。
    pub fn create_change_log(&self) -> Result<Box<dyn ChangeLog>> {
        let log_path = self
            .change_log_path
            .clone()
            .unwrap_or_else(|| self.default_change_log_path());
        if let Some(parent) = log_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        Ok(Box::new(JsonlChangeLog::new(log_path)?))
    }

    /// 创建装配完整的 layer manager。
    pub fn build_layer_manager(&self) -> Result<MemoryLayerManager> {
        let mut layer_manager = MemoryLayerManager::try_new(
            self.echo_agent_dir.clone(),
            self.store.clone(),
            self.create_change_log()?,
        )?;

        if let Some(observer) = &self.evolution_observer {
            layer_manager = layer_manager.with_evolution_observer(observer.clone());
        }

        Ok(layer_manager)
    }

    /// 构建并执行 reconcile，然后再将底层 Store 发布给运行时的读取方。
    pub async fn build_layer_manager_reconciled(&self) -> Result<MemoryLayerManager> {
        let manager = self.build_layer_manager()?;
        manager.reconcile_pending().await?;
        Ok(manager)
    }
}

/// 通过 HookRegistry 发布框架演化事件的适配器。
pub struct HookEvolutionObserver {
    /// 共享的钩子注册表
    hook_registry: Arc<RwLock<HookRegistry>>,
    /// 会话 ID
    session_id: String,
    /// agent 名称
    agent_name: String,
}

impl HookEvolutionObserver {
    /// 将演化事件绑定到某个 agent 的共享钩子注册表。
    pub fn new(
        hook_registry: Arc<RwLock<HookRegistry>>,
        session_id: impl Into<String>,
        agent_name: impl Into<String>,
    ) -> Self {
        Self {
            hook_registry,
            session_id: session_id.into(),
            agent_name: agent_name.into(),
        }
    }

    /// 触发一次生命周期钩子
    async fn fire(&self, context: HookContext) {
        let registry = self.hook_registry.read().await.clone();
        let _ = registry.run_lifecycle_hooks(&context).await;
    }

    /// 构建一次内存写入事件对应的钩子上下文
    fn memory_write_context(&self, key: &str, source: &str) -> HookContext {
        HookContext::for_post_memory_write(key, source, &self.session_id, &self.agent_name)
    }

    /// 构建一次内存分层变更事件对应的钩子上下文
    fn memory_layer_context(&self, key: &str, from_layer: &str, to_layer: &str) -> HookContext {
        HookContext::for_memory_layer_change(
            key,
            from_layer,
            to_layer,
            &self.session_id,
            &self.agent_name,
        )
    }

    /// 构建一次技能相关事件对应的钩子上下文
    fn skill_context(&self, event: echo_core::hooks::HookEvent, skill_name: &str) -> HookContext {
        HookContext::for_lifecycle(event, skill_name, &self.session_id, &self.agent_name)
    }
}

impl EvolutionObserver for HookEvolutionObserver {
    /// 内存写入发生时触发 `PostMemoryWrite` 钩子
    fn on_memory_write<'a>(&'a self, key: &'a str, source: &'a str) -> BoxFuture<'a, ()> {
        Box::pin(async move {
            self.fire(self.memory_write_context(key, source)).await;
        })
    }

    /// 内存分层发生变化时触发 `MemoryLayerChange` 钩子
    fn on_memory_layer_change<'a>(
        &'a self,
        key: &'a str,
        from_layer: &'a str,
        to_layer: &'a str,
    ) -> BoxFuture<'a, ()> {
        Box::pin(async move {
            self.fire(self.memory_layer_context(key, from_layer, to_layer))
                .await;
        })
    }

    /// 检测到候选技能时触发 `SkillCandidateDetected` 钩子
    fn on_skill_candidate_detected<'a>(&'a self, skill_name: &'a str) -> BoxFuture<'a, ()> {
        Box::pin(async move {
            self.fire(self.skill_context(
                echo_core::hooks::HookEvent::SkillCandidateDetected,
                skill_name,
            ))
            .await;
        })
    }

    /// 执行技能健康检查时触发 `SkillHealthCheck` 钩子
    fn on_skill_health_check<'a>(&'a self, skill_name: &'a str) -> BoxFuture<'a, ()> {
        Box::pin(async move {
            self.fire(
                self.skill_context(echo_core::hooks::HookEvent::SkillHealthCheck, skill_name),
            )
            .await;
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use echo_core::memory::types::{MemoryMeta, MemorySource, MemoryType};
    use echo_state::memory::store::InMemoryStore;
    use std::sync::Mutex;

    struct RecordingObserver {
        events: Arc<Mutex<Vec<String>>>,
    }

    impl EvolutionObserver for RecordingObserver {
        fn on_memory_write<'a>(&'a self, key: &'a str, source: &'a str) -> BoxFuture<'a, ()> {
            Box::pin(async move {
                self.events
                    .lock()
                    .unwrap_or_else(|error| error.into_inner())
                    .push(format!("write:{key}:{source}"));
            })
        }

        fn on_memory_layer_change<'a>(
            &'a self,
            key: &'a str,
            from_layer: &'a str,
            to_layer: &'a str,
        ) -> BoxFuture<'a, ()> {
            Box::pin(async move {
                self.events
                    .lock()
                    .unwrap_or_else(|error| error.into_inner())
                    .push(format!("layer:{key}:{from_layer}->{to_layer}"));
            })
        }
    }

    #[tokio::test]
    async fn builder_wires_typed_observer_and_change_log()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let temp_dir = tempfile::tempdir()?;
        let echo_agent_dir = temp_dir.path().join(".echo-agent");
        let store = Arc::new(InMemoryStore::new());
        let events = Arc::new(Mutex::new(Vec::new()));
        let observer = Arc::new(RecordingObserver {
            events: events.clone(),
        });

        let manager = MemoryRuntimeIntegrationBuilder::new(echo_agent_dir.clone(), store)
            .evolution_observer(observer)
            .build_layer_manager_reconciled()
            .await?;

        let meta = MemoryMeta::new(
            MemoryType::ProjectFact,
            MemorySource::AutoExtracted,
            "runtime",
        )
        .with_confidence(0.9)
        .with_stability(0.8);
        manager
            .write_memory("builder_test", "Use layered memory", meta)
            .await?;

        assert_eq!(
            *events.lock().unwrap_or_else(|error| error.into_inner()),
            vec![
                "write:builder_test:auto_extracted".to_string(),
                "layer:builder_test:warm->hot".to_string(),
            ]
        );
        assert!(
            echo_agent_dir
                .join("evolution")
                .join("change-log.jsonl")
                .exists()
        );
        Ok(())
    }

    #[test]
    fn hook_observer_builds_matcher_complete_contexts() {
        let observer = HookEvolutionObserver::new(
            Arc::new(RwLock::new(HookRegistry::new())),
            "session-1",
            "agent-1",
        );

        let write = observer.memory_write_context("memory-1", "explicit_save");
        assert_eq!(write.event, echo_core::hooks::HookEvent::PostMemoryWrite);
        assert_eq!(write.matcher.as_deref(), Some("explicit_save"));
        assert_eq!(write.memory_key.as_deref(), Some("memory-1"));
        assert_eq!(write.memory_source.as_deref(), Some("explicit_save"));

        let layer = observer.memory_layer_context("memory-1", "warm", "hot");
        assert_eq!(layer.event, echo_core::hooks::HookEvent::MemoryLayerChange);
        assert_eq!(layer.matcher.as_deref(), Some("warm->hot"));
        assert_eq!(layer.memory_from_layer.as_deref(), Some("warm"));
        assert_eq!(layer.memory_to_layer.as_deref(), Some("hot"));

        for event in [
            echo_core::hooks::HookEvent::SkillCandidateDetected,
            echo_core::hooks::HookEvent::SkillHealthCheck,
        ] {
            let context = observer.skill_context(event, "cargo-build");
            assert_eq!(context.event, event);
            assert_eq!(context.matcher.as_deref(), Some("cargo-build"));
            assert_eq!(context.session_id, "session-1");
            assert_eq!(context.agent_name, "agent-1");
        }
    }
}
</content>
