/*! 子智能体（Subagent）注册表 —— 负责发现、注册与生命周期管理

保存已注册子智能体的声明式定义、延迟实例化的工厂，以及相应的生命周期事件。
*/

use crate::error::Result;
use echo_core::agent::Agent;
use futures::future::BoxFuture;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::{OnceCell, RwLock};
use tracing::{debug, info, warn};

use super::events::SubagentEventBus;
use super::types::{RegisteredSubagent, SubagentDefinition};

/// 单个子智能体在注册表中的内部记录。
struct RegistryEntry {
    /// 子智能体的声明式定义（名称、描述、标签等元信息）。
    definition: SubagentDefinition,
    /// 已实例化的 Agent；在按工厂注册且尚未触发实例化前为空。
    agent: Arc<OnceCell<Arc<dyn Agent>>>,
    /// 用于延迟构建 Agent 的工厂；仅当以工厂方式注册时存在。
    factory: Option<Arc<dyn AgentFactory>>,
    /// 该记录的注册版本号，用于识别并丢弃过期的工厂实例化结果。
    revision: u64,
}

/// 注册表的内部可变状态，由读写锁保护。
#[derive(Default)]
struct RegistryState {
    /// 按名称索引的全部注册记录。
    entries: HashMap<String, RegistryEntry>,
    /// 下一次注册将分配的版本号计数器。
    next_revision: u64,
}

/// 测试专用：用于在工厂结果发布前后插入同步点，便于验证 single-flight（单飞）行为。
#[cfg(test)]
#[derive(Clone)]
struct FactoryPublicationPause {
    /// 工厂执行到达“发布边界”时释放许可的信号量。
    reached: Arc<tokio::sync::Semaphore>,
    /// 测试代码用来放行、使发布得以继续的信号量。
    resume: Arc<tokio::sync::Semaphore>,
}

impl RegistryState {
    /// 分配并返回下一个注册版本号（饱和递增，不会回绕溢出）。
    fn next_revision(&mut self) -> u64 {
        self.next_revision = self.next_revision.saturating_add(1);
        self.next_revision
    }
}

// ── Agent 工厂 ─────────────────────────────────────────────────────────────

/** 用于延迟实例化 Agent 的工厂 trait。

在希望先注册一个 Agent 定义、但把真正的 Agent 构建推迟到它首次被派发时使用。
*/
pub trait AgentFactory: Send + Sync {
    /** 异步创建一个 Agent 实例。

    工厂不能通过 [`SubagentRegistry::get_agent`] 递归地解析自身所在的注册项；
    异步的单次初始化无法在等待自身完成的同时完成。调用方需自行负责所需的超时/截止时间。

    # 返回值
    一个解析为 `Result<Box<dyn Agent>>` 的装箱 Future。
    */
    fn create(&self) -> BoxFuture<'static, Result<Box<dyn Agent>>>;
}

/// 对闭包做类型擦除包装，使其实现 `AgentFactory`。
pub struct FnAgentFactory<F>
where
    F: Fn() -> BoxFuture<'static, Result<Box<dyn Agent>>> + Send + Sync,
{
    /// 实际执行创建逻辑的闭包。
    f: F,
}

impl<F> FnAgentFactory<F>
where
    F: Fn() -> BoxFuture<'static, Result<Box<dyn Agent>>> + Send + Sync,
{
    /** 创建一个基于函数的 Agent 工厂。

    # 参数
    * `f` - 被调用时用于创建 Agent 的异步闭包。
    */
    pub fn new(f: F) -> Self {
        Self { f }
    }
}

impl<F> AgentFactory for FnAgentFactory<F>
where
    F: Fn() -> BoxFuture<'static, Result<Box<dyn Agent>>> + Send + Sync,
{
    /// 调用内部闭包以创建 Agent 实例。
    fn create(&self) -> BoxFuture<'static, Result<Box<dyn Agent>>> {
        (self.f)()
    }
}

// ── 子智能体注册表 ─────────────────────────────────────────────────────────────

/** 子智能体定义与实例的注册表。

持有已注册的子智能体定义，并额外提供：
- 基于定义的查找
- 支持延迟实例化的工厂机制
- 生命周期事件
*/
pub struct SubagentRegistry {
    /// 每个名称对应一条原子记录，使定义与可执行生成版本保持一致。
    state: Arc<RwLock<RegistryState>>,
    /// 已发布的可执行子智能体定义目录快照，供只读场景快速访问。
    executable_catalog: Arc<std::sync::RwLock<Vec<SubagentDefinition>>>,
    /// 目录快照的版本号，每次重新发布目录时递增。
    catalog_revision: Arc<AtomicU64>,
    /// 测试专用：用于在工厂结果发布前插入暂停点。
    #[cfg(test)]
    factory_publication_pause: Arc<RwLock<Option<FactoryPublicationPause>>>,
    /// 生命周期事件的事件总线。
    event_bus: SubagentEventBus,
    /** 供所有已派发尝试共享的、进程级别的执行控制面。

    一个注册表只拥有一个控制面，从而使 `execution_id` 能够在基于该注册表构建的
    每一个 `SubagentExecutor` 之间被寻址——如果没有这一点，每个执行器都会各自持有
    私有的控制注册表，由不同执行器派发的兄弟尝试之间将无法互相感知
    （子智能体之间的消息传递以及默认上行链路都依赖这一统一查找机制）。
    */
    control: Arc<super::control::SubagentControlRegistry>,
}

impl SubagentRegistry {
    /// 创建一个空的注册表。
    pub fn new() -> Self {
        Self {
            state: Arc::new(RwLock::new(RegistryState::default())),
            executable_catalog: Arc::new(std::sync::RwLock::new(Vec::new())),
            catalog_revision: Arc::new(AtomicU64::new(0)),
            #[cfg(test)]
            factory_publication_pause: Arc::new(RwLock::new(None)),
            event_bus: SubagentEventBus::new(),
            control: Arc::new(super::control::SubagentControlRegistry::default()),
        }
    }

    /// 使用指定的事件总线创建注册表。
    pub fn with_event_bus(event_bus: SubagentEventBus) -> Self {
        Self {
            state: Arc::new(RwLock::new(RegistryState::default())),
            executable_catalog: Arc::new(std::sync::RwLock::new(Vec::new())),
            catalog_revision: Arc::new(AtomicU64::new(0)),
            #[cfg(test)]
            factory_publication_pause: Arc::new(RwLock::new(None)),
            event_bus,
            control: Arc::new(super::control::SubagentControlRegistry::default()),
        }
    }

    /// 返回该注册表上所有执行器共享的执行控制面。
    pub fn control_registry(&self) -> &Arc<super::control::SubagentControlRegistry> {
        &self.control
    }

    /// 依据当前状态重新计算并发布可执行子智能体目录快照，同时递增目录版本号。
    fn publish_catalog(&self, state: &RegistryState) {
        let mut definitions = state
            .entries
            .values()
            .filter(|entry| entry.agent.get().is_some() || entry.factory.is_some())
            .map(|entry| entry.definition.clone())
            .collect::<Vec<_>>();
        definitions.sort_by(|left, right| left.name.cmp(&right.name));
        *self
            .executable_catalog
            .write()
            .unwrap_or_else(|error| error.into_inner()) = definitions;
        self.catalog_revision.fetch_add(1, Ordering::Release);
    }

    /// 返回可执行目录快照的共享句柄。
    pub(crate) fn executable_catalog_handle(
        &self,
    ) -> Arc<std::sync::RwLock<Vec<SubagentDefinition>>> {
        Arc::clone(&self.executable_catalog)
    }

    /// 返回目录版本号计数器的共享句柄。
    pub(crate) fn catalog_revision_handle(&self) -> Arc<AtomicU64> {
        Arc::clone(&self.catalog_revision)
    }

    /// 测试专用：若配置了暂停点，则在工厂结果发布前阻塞，等待测试代码放行。
    #[cfg(test)]
    async fn pause_before_factory_publication(&self) {
        let pause = self.factory_publication_pause.read().await.clone();
        if let Some(pause) = pause {
            pause.reached.add_permits(1);
            if let Ok(permit) = pause.resume.acquire().await {
                permit.forget();
            }
        }
    }

    // ── 注册 ──────────────────────────────────────────────────────────────

    /// 使用给定定义注册一个预先构建好的 Agent。
    pub async fn register(&self, def: SubagentDefinition, agent: Box<dyn Agent>) {
        self.register_shared(def, Arc::from(agent)).await;
    }

    /// 注册一个共享的、预先构建好的 Agent，不会再包一层额外的执行容器。
    /// 这是 Team 以及“Agent 即工具”风格集成所使用的编程式组合路径。
    pub async fn register_shared(&self, def: SubagentDefinition, agent: Arc<dyn Agent>) {
        let name = def.name.clone();
        info!(subagent = %name, mode = %def.execution_mode, "Registering subagent");

        let mut state = self.state.write().await;
        let revision = state.next_revision();
        state.entries.insert(
            name.clone(),
            RegistryEntry {
                definition: def,
                agent: Arc::new(OnceCell::new_with(Some(agent))),
                factory: None,
                revision,
            },
        );
        self.publish_catalog(&state);
        drop(state);

        self.event_bus
            .emit(super::events::SubagentEvent::Registered { name: name.clone() });
    }

    /// 同步方式注册 —— 使用 `try_write` 以避免 `block_on` 导致的死锁。
    ///
    /// 用于同步上下文（例如 builder 模式、`main()`）。
    /// 若锁被占用，则退化为记录一条警告日志。
    pub fn register_sync(&self, def: SubagentDefinition, agent: Box<dyn Agent>) -> bool {
        self.register_shared_sync(def, Arc::from(agent))
    }

    /// [`register_shared`](Self::register_shared) 的同步版本。
    /// 当注册表被占用时返回 `false`，而不是阻塞等待。
    pub fn register_shared_sync(&self, def: SubagentDefinition, agent: Arc<dyn Agent>) -> bool {
        let name = def.name.clone();

        let ok = match self.state.try_write() {
            Ok(mut state) => {
                let revision = state.next_revision();
                state.entries.insert(
                    name.clone(),
                    RegistryEntry {
                        definition: def,
                        agent: Arc::new(OnceCell::new_with(Some(agent))),
                        factory: None,
                        revision,
                    },
                );
                self.publish_catalog(&state);
                true
            }
            Err(_) => {
                warn!(subagent = %name, "Lock contention on subagent registry, registration deferred");
                false
            }
        };

        if ok {
            self.event_bus
                .emit(super::events::SubagentEvent::Registered { name });
        }

        ok
    }

    /// 使用工厂注册一个定义，以便延迟实例化。
    pub async fn register_factory(&self, def: SubagentDefinition, factory: Arc<dyn AgentFactory>) {
        let name = def.name.clone();
        debug!(subagent = %name, "Registering subagent factory");

        let mut state = self.state.write().await;
        let revision = state.next_revision();
        state.entries.insert(
            name,
            RegistryEntry {
                definition: def,
                agent: Arc::new(OnceCell::new()),
                factory: Some(factory),
                revision,
            },
        );
        self.publish_catalog(&state);
    }

    /** 仅注册**定义**本身 —— 不携带 Agent 实例，也不携带工厂。

    这是一条底层的延迟绑定路径：`get` 与 `contains` 可以查看这个待处理的定义，
    但 `list_available`、`list_by_tag`、`agent_names` 以及面向模型的派发目录
    只会包含由实例或工厂支撑的定义。之后在同一名称下调用
    [`register`](Self::register) / [`register_sync`](Self::register_sync)
    / [`register_factory`](Self::register_factory) 会提供（或延迟产出）实例，
    并覆盖此定义。

    在完成实例化之前显式发起编程式派发会失败，因为此时没有可运行的 Agent。
    面向模型的目录也不会提供该派发目标。

    在锁无竞争的情况下始终返回 `true`，表示已成功插入。
    */
    pub async fn register_definition(&self, def: SubagentDefinition) -> bool {
        let name = def.name.clone();
        debug!(subagent = %name, "Registering subagent definition (no instance)");
        let mut state = self.state.write().await;
        let revision = state.next_revision();
        state.entries.insert(
            name,
            RegistryEntry {
                definition: def,
                agent: Arc::new(OnceCell::new()),
                factory: None,
                revision,
            },
        );
        self.publish_catalog(&state);
        true
    }

    /// [`register_definition`](Self::register_definition) 的同步版本。
    ///
    /// 使用 `try_write` 避免在无法 await 该注册表的同步上下文中因 `block_on`
    /// 而导致死锁。锁竞争时记录警告日志并返回 `false`。
    pub fn register_definition_sync(&self, def: SubagentDefinition) -> bool {
        let name = def.name.clone();
        match self.state.try_write() {
            Ok(mut state) => {
                let revision = state.next_revision();
                state.entries.insert(
                    name,
                    RegistryEntry {
                        definition: def,
                        agent: Arc::new(OnceCell::new()),
                        factory: None,
                        revision,
                    },
                );
                self.publish_catalog(&state);
                true
            }
            Err(error) => {
                warn!(subagent = %name, %error, "Cannot register subagent definition");
                false
            }
        }
    }

    /// 以同步方式注册一个带工厂的定义（注册过程本身是同步的，Agent 仍按工厂延迟构建）。
    pub fn register_factory_sync(
        &self,
        def: SubagentDefinition,
        factory: Arc<dyn AgentFactory>,
    ) -> bool {
        let name = def.name.clone();
        let mut state = match self.state.try_write() {
            Ok(state) => state,
            Err(error) => {
                warn!(subagent = %name, %error, "Cannot register subagent definition");
                return false;
            }
        };

        let revision = state.next_revision();
        state.entries.insert(
            name,
            RegistryEntry {
                definition: def,
                agent: Arc::new(OnceCell::new()),
                factory: Some(factory),
                revision,
            },
        );
        self.publish_catalog(&state);
        true
    }

    /// 按名称移除一个子智能体。
    pub async fn remove(&self, name: &str) {
        let mut state = self.state.write().await;
        state.entries.remove(name);
        self.publish_catalog(&state);
        drop(state);

        self.event_bus
            .emit(super::events::SubagentEvent::Unregistered {
                name: name.to_string(),
            });
    }

    // ── 查找 ──────────────────────────────────────────────────────────────

    /// 查找一个已注册的子智能体。未找到时返回 None。
    pub async fn get(&self, name: &str) -> Option<RegisteredSubagent> {
        let state = self.state.read().await;
        let entry = state.entries.get(name)?;

        Some(RegisteredSubagent {
            definition: entry.definition.clone(),
            has_instance: entry.agent.get().is_some(),
        })
    }

    /** 获取用于立即执行的 Agent 实例。

    如果该 Agent 是通过工厂注册且尚未实例化，此方法会按需创建它。

    并发调用方会为当前注册版本共享同一次具备取消安全性的初始化尝试。
    调用方需自行负责本操作的超时/截止时间；注册表本身不会施加构建超时。
    */
    pub async fn get_agent(&self, name: &str) -> Option<Arc<dyn Agent>> {
        let (agent_cell, factory, factory_revision) = {
            let state = self.state.read().await;
            let entry = state.entries.get(name)?;
            if let Some(agent) = entry.agent.get() {
                return Some(Arc::clone(agent));
            }
            (
                Arc::clone(&entry.agent),
                entry.factory.clone()?,
                entry.revision,
            )
        };

        let result = agent_cell
            .get_or_try_init(|| async {
                info!(subagent = %name, "Instantiating agent from factory");
                factory.create().await.map(Arc::<dyn Agent>::from)
            })
            .await;

        #[cfg(test)]
        self.pause_before_factory_publication().await;

        match result {
            Ok(agent) => {
                let agent = Arc::clone(agent);
                let state = self.state.read().await;
                let entry = state.entries.get(name)?;
                if entry.revision != factory_revision || !Arc::ptr_eq(&entry.agent, &agent_cell) {
                    warn!(subagent = %name, "Discarding stale subagent factory result");
                    return entry.agent.get().cloned();
                }
                Some(agent)
            }
            Err(error) => {
                warn!(subagent = %name, error = %error, "Factory instantiation failed");
                None
            }
        }
    }

    /// 当 `name` 注册了工厂时，创建一个全新的 Agent。
    ///
    /// 与 `get_agent` 不同，此方法从不更新或复用缓存的 Agent。
    pub async fn create_fresh_agent(&self, name: &str) -> Result<Option<Arc<dyn Agent>>> {
        let factory = {
            let state = self.state.read().await;
            state
                .entries
                .get(name)
                .and_then(|entry| entry.factory.clone())
        };

        match factory {
            Some(factory) => factory
                .create()
                .await
                .map(|agent| Some(Arc::new(agent) as Arc<dyn Agent>)),
            None => Ok(None),
        }
    }

    /// 检查某个子智能体是否已注册。
    pub async fn contains(&self, name: &str) -> bool {
        self.state.read().await.entries.contains_key(name)
    }

    /// 列出由预构建实例或工厂支撑的全部定义。
    pub async fn list_available(&self) -> Vec<SubagentDefinition> {
        let state = self.state.read().await;
        state
            .entries
            .values()
            .filter(|entry| entry.agent.get().is_some() || entry.factory.is_some())
            .map(|entry| entry.definition.clone())
            .collect()
    }

    /// 列出匹配指定标签的可执行定义。
    pub async fn list_by_tag(&self, tag: &str) -> Vec<SubagentDefinition> {
        let state = self.state.read().await;
        state
            .entries
            .values()
            .filter(|entry| entry.agent.get().is_some() || entry.factory.is_some())
            .map(|entry| &entry.definition)
            .filter(|definition| definition.tags.iter().any(|entry_tag| entry_tag == tag))
            .cloned()
            .collect()
    }

    /// 获取可执行 Agent 的名称列表，便于在工具描述中使用（便捷方法）。
    pub async fn agent_names(&self) -> Vec<String> {
        let state = self.state.read().await;
        state
            .entries
            .iter()
            .filter(|(_, entry)| entry.agent.get().is_some() || entry.factory.is_some())
            .map(|(name, _)| name.clone())
            .collect()
    }

    /// 获取事件总线的引用。
    pub fn event_bus(&self) -> &SubagentEventBus {
        &self.event_bus
    }
}

impl Default for SubagentRegistry {
    /// 等价于 `SubagentRegistry::new()`。
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for SubagentRegistry {
    /// 克隆注册表句柄；内部各字段均为共享引用（`Arc`/`RwLock`），克隆代价低廉。
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone(),
            executable_catalog: self.executable_catalog.clone(),
            catalog_revision: self.catalog_revision.clone(),
            #[cfg(test)]
            factory_publication_pause: self.factory_publication_pause.clone(),
            event_bus: self.event_bus.clone(),
            control: self.control.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::MockAgent;

    #[tokio::test]
    async fn test_register_and_get() {
        let registry = SubagentRegistry::new();
        let agent = MockAgent::new("researcher");
        let def = SubagentDefinition::new("researcher", "Researches topics");

        registry.register(def, Box::new(agent)).await;

        assert!(registry.contains("researcher").await);
        let registered = registry.get("researcher").await.unwrap();
        assert_eq!(registered.definition.name, "researcher");
        assert!(registered.has_instance);
    }

    #[tokio::test]
    async fn shared_registration_preserves_agent_identity() -> Result<()> {
        let registry = SubagentRegistry::new();
        let shared: Arc<dyn Agent> = Arc::new(MockAgent::new("shared"));
        registry
            .register_shared(
                SubagentDefinition::new("shared", "Shared programmatic Agent"),
                shared.clone(),
            )
            .await;

        let resolved = registry.get_agent("shared").await.ok_or_else(|| {
            crate::error::ReactError::Other("shared Agent was not registered".to_string())
        })?;
        assert!(Arc::ptr_eq(&shared, &resolved));
        Ok(())
    }

    #[tokio::test]
    async fn test_remove() {
        let registry = SubagentRegistry::new();
        let agent = MockAgent::new("subagent");
        let def = SubagentDefinition::new("subagent", "Subagent agent");

        registry.register(def, Box::new(agent)).await;
        assert!(registry.contains("subagent").await);

        registry.remove("subagent").await;
        assert!(!registry.contains("subagent").await);
    }

    #[tokio::test]
    async fn test_list_available() {
        let registry = SubagentRegistry::new();

        let a1 = MockAgent::new("a1");
        let a2 = MockAgent::new("a2");

        registry
            .register(SubagentDefinition::new("a1", "Agent 1"), Box::new(a1))
            .await;
        registry
            .register(SubagentDefinition::new("a2", "Agent 2"), Box::new(a2))
            .await;

        let available = registry.list_available().await;
        assert_eq!(available.len(), 2);
    }

    #[tokio::test]
    async fn test_list_by_tag() {
        let registry = SubagentRegistry::new();

        let mut def1 = SubagentDefinition::new("researcher", "Research");
        def1.tags.push("research".into());
        let mut def2 = SubagentDefinition::new("writer", "Write");
        def2.tags.push("writing".into());

        registry
            .register(def1, Box::new(MockAgent::new("researcher")))
            .await;
        registry
            .register(def2, Box::new(MockAgent::new("writer")))
            .await;

        let research = registry.list_by_tag("research").await;
        assert_eq!(research.len(), 1);
        assert_eq!(research[0].name, "researcher");
    }

    #[tokio::test]
    async fn test_get_agent() {
        let registry = SubagentRegistry::new();
        let agent = MockAgent::new("a");
        registry
            .register(SubagentDefinition::new("a", "A"), Box::new(agent))
            .await;

        let handle = registry.get_agent("a").await;
        assert!(handle.is_some());
    }

    #[tokio::test]
    async fn test_agent_names() {
        let registry = SubagentRegistry::new();
        assert!(registry.agent_names().await.is_empty());

        registry
            .register(
                SubagentDefinition::new("x", "X"),
                Box::new(MockAgent::new("x")),
            )
            .await;

        let names = registry.agent_names().await;
        assert_eq!(names, vec!["x"]);
    }

    #[tokio::test]
    async fn test_register_definition_only_is_not_advertised_as_executable() {
        let registry = SubagentRegistry::new();
        let mut def = SubagentDefinition::new("plugin_agent", "Plugin-defined agent");
        def.tags.push("plugin".to_string());

        let inserted = registry.register_definition(def).await;
        assert!(inserted);

        assert!(registry.contains("plugin_agent").await);
        let available = registry.list_available().await;
        assert!(!available.iter().any(|d| d.name == "plugin_agent"));
        assert!(registry.list_by_tag("plugin").await.is_empty());
        assert!(registry.agent_names().await.is_empty());
        assert!(
            registry
                .executable_catalog_handle()
                .read()
                .is_ok_and(|catalog| catalog.is_empty())
        );

        // 没有实例：get_agent 必须返回 None（没有可调用的工厂）。
        let registered = registry.get("plugin_agent").await.unwrap();
        assert!(!registered.has_instance);
        assert!(registry.get_agent("plugin_agent").await.is_none());
    }

    #[tokio::test]
    async fn test_register_definition_then_promote_to_instance() {
        /* 之后在同一名称下调用 register() 会提供实例并覆盖该定义——
        应用层正是通过这种方式，用真实的 Agent 来填充由插件发现的定义。 */
        let registry = SubagentRegistry::new();
        registry
            .register_definition(SubagentDefinition::new("hydrated", "H"))
            .await;
        assert!(!registry.get("hydrated").await.unwrap().has_instance);

        registry
            .register(
                SubagentDefinition::new("hydrated", "Hydrated"),
                Box::new(MockAgent::new("hydrated")),
            )
            .await;

        let registered = registry.get("hydrated").await.unwrap();
        assert!(registered.has_instance);
        assert_eq!(registered.definition.description, "Hydrated");
    }

    #[test]
    fn test_register_definition_sync_inserts_under_uncontended_locks() {
        let registry = SubagentRegistry::new();
        let inserted = registry.register_definition_sync(SubagentDefinition::new("sync", "S"));
        assert!(inserted);
        let state = registry.state.try_read().ok();
        assert!(
            state
                .as_ref()
                .is_some_and(|state| state.entries.contains_key("sync"))
        );
    }

    #[tokio::test]
    async fn cancelled_factory_attempt_allows_next_resolution() -> Result<()> {
        let registry = SubagentRegistry::new();
        let attempts = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let (started_tx, mut started_rx) = tokio::sync::mpsc::unbounded_channel();
        let factory_attempts = Arc::clone(&attempts);
        let factory = Arc::new(FnAgentFactory::new(move || {
            let attempt = factory_attempts.fetch_add(1, Ordering::SeqCst) + 1;
            let _ = started_tx.send(attempt);
            Box::pin(async move {
                if attempt == 1 {
                    std::future::pending::<()>().await;
                }
                Ok(Box::new(MockAgent::new("cancel-safe")) as Box<dyn Agent>)
            })
        }));
        registry
            .register_factory(
                SubagentDefinition::new("cancel-safe", "Cancellation-safe factory"),
                factory,
            )
            .await;

        let first_registry = registry.clone();
        let first = tokio::spawn(async move { first_registry.get_agent("cancel-safe").await });
        let first_attempt =
            tokio::time::timeout(std::time::Duration::from_secs(1), started_rx.recv())
                .await
                .map_err(|_| {
                    crate::error::ReactError::Other(
                        "first factory attempt did not start".to_string(),
                    )
                })?
                .ok_or_else(|| {
                    crate::error::ReactError::Other("factory start channel closed".to_string())
                })?;
        assert_eq!(first_attempt, 1);
        first.abort();
        let join_error = first.await.err().ok_or_else(|| {
            crate::error::ReactError::Other("cancelled factory task completed".to_string())
        })?;
        assert!(join_error.is_cancelled());

        let second = tokio::time::timeout(
            std::time::Duration::from_millis(250),
            registry.get_agent("cancel-safe"),
        )
        .await
        .map_err(|_| {
            crate::error::ReactError::Other(
                "cancelled factory attempt retained single-flight ownership".to_string(),
            )
        })?
        .ok_or_else(|| {
            crate::error::ReactError::Other("second factory attempt returned no agent".to_string())
        })?;
        assert_eq!(second.name(), "cancel-safe");
        assert_eq!(attempts.load(Ordering::SeqCst), 2);
        Ok(())
    }

    #[tokio::test]
    async fn factory_result_is_published_before_another_creation_can_start() -> Result<()> {
        let registry = SubagentRegistry::new();
        let attempts = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let (started_tx, mut started_rx) = tokio::sync::mpsc::unbounded_channel();
        let factory_attempts = Arc::clone(&attempts);
        let factory = Arc::new(FnAgentFactory::new(move || {
            let attempt = factory_attempts.fetch_add(1, Ordering::SeqCst) + 1;
            let _ = started_tx.send(attempt);
            Box::pin(async move { Ok(Box::new(MockAgent::new("single-flight")) as Box<dyn Agent>) })
        }));
        let publication_reached = Arc::new(tokio::sync::Semaphore::new(0));
        let publication_resume = Arc::new(tokio::sync::Semaphore::new(0));
        *registry.factory_publication_pause.write().await = Some(FactoryPublicationPause {
            reached: Arc::clone(&publication_reached),
            resume: Arc::clone(&publication_resume),
        });
        registry
            .register_factory(
                SubagentDefinition::new("single-flight", "Single-flight factory"),
                factory,
            )
            .await;

        let first_registry = registry.clone();
        let first = tokio::spawn(async move { first_registry.get_agent("single-flight").await });
        let first_attempt =
            tokio::time::timeout(std::time::Duration::from_secs(1), started_rx.recv())
                .await
                .map_err(|_| {
                    crate::error::ReactError::Other(
                        "first factory attempt did not start".to_string(),
                    )
                })?
                .ok_or_else(|| {
                    crate::error::ReactError::Other("factory start channel closed".to_string())
                })?;
        assert_eq!(first_attempt, 1);

        let reached_permit = tokio::time::timeout(
            std::time::Duration::from_secs(1),
            publication_reached.acquire(),
        )
        .await
        .map_err(|_| {
            crate::error::ReactError::Other(
                "first factory did not reach the publication boundary".to_string(),
            )
        })?
        .map_err(|error| {
            crate::error::ReactError::Other(format!(
                "publication boundary semaphore closed: {error}"
            ))
        })?;
        reached_permit.forget();

        let second_registry = registry.clone();
        let (entered_tx, entered_rx) = tokio::sync::oneshot::channel();
        let mut second = tokio::spawn(async move {
            let _ = entered_tx.send(());
            second_registry.get_agent("single-flight").await
        });
        tokio::time::timeout(std::time::Duration::from_secs(1), entered_rx)
            .await
            .map_err(|_| {
                crate::error::ReactError::Other(
                    "second resolver did not enter the publication interleaving".to_string(),
                )
            })?
            .map_err(|error| {
                crate::error::ReactError::Other(format!(
                    "second resolver entry channel closed: {error}"
                ))
            })?;
        let second_agent = tokio::select! {
            joined = &mut second => {
                joined
                    .map_err(|error| {
                        crate::error::ReactError::Other(format!(
                            "second resolver failed while first publication was paused: {error}"
                        ))
                    })?
                    .ok_or_else(|| {
                        crate::error::ReactError::Other(
                            "second resolver returned no published agent".to_string(),
                        )
                    })?
            }
            duplicate = started_rx.recv() => {
                publication_resume.add_permits(2);
                return match duplicate {
                    Some(attempt) => Err(crate::error::ReactError::Other(format!(
                        "factory started duplicate attempt {attempt} before publication"
                    ))),
                    None => Err(crate::error::ReactError::Other(
                        "factory start channel closed before publication".to_string(),
                    )),
                };
            }
            _ = tokio::time::sleep(std::time::Duration::from_secs(1)) => {
                publication_resume.add_permits(2);
                second.abort();
                return Err(crate::error::ReactError::Other(
                    "second resolver produced neither a cached agent nor a duplicate factory attempt"
                        .to_string(),
                ));
            }
        };
        publication_resume.add_permits(1);

        let first_agent = first
            .await
            .map_err(|error| {
                crate::error::ReactError::Other(format!("first resolver failed: {error}"))
            })?
            .ok_or_else(|| {
                crate::error::ReactError::Other("first resolver returned no agent".to_string())
            })?;
        assert!(Arc::ptr_eq(&first_agent, &second_agent));
        assert_eq!(attempts.load(Ordering::SeqCst), 1);
        Ok(())
    }

    #[tokio::test]
    async fn failed_factory_attempt_does_not_poison_registration() -> Result<()> {
        let registry = SubagentRegistry::new();
        let attempts = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let factory_attempts = Arc::clone(&attempts);
        let factory = Arc::new(FnAgentFactory::new(move || {
            let attempt = factory_attempts.fetch_add(1, Ordering::SeqCst) + 1;
            Box::pin(async move {
                if attempt == 1 {
                    return Err(crate::error::ReactError::Other(
                        "first factory attempt failed".to_string(),
                    ));
                }
                Ok(Box::new(MockAgent::new("retryable-factory")) as Box<dyn Agent>)
            })
        }));
        registry
            .register_factory(
                SubagentDefinition::new("retryable-factory", "Retryable factory"),
                factory,
            )
            .await;

        assert!(registry.get_agent("retryable-factory").await.is_none());
        let recovered = registry
            .get_agent("retryable-factory")
            .await
            .ok_or_else(|| {
                crate::error::ReactError::Other(
                    "factory registration remained poisoned after an error".to_string(),
                )
            })?;
        assert_eq!(recovered.name(), "retryable-factory");
        assert_eq!(attempts.load(Ordering::SeqCst), 2);
        Ok(())
    }

    #[tokio::test]
    async fn stale_factory_result_does_not_replace_new_registration() -> Result<()> {
        let registry = SubagentRegistry::new();
        let release_old = Arc::new(tokio::sync::Semaphore::new(0));
        let (started_tx, mut started_rx) = tokio::sync::mpsc::unbounded_channel();
        let factory_release = Arc::clone(&release_old);
        let factory = Arc::new(FnAgentFactory::new(move || {
            let release = Arc::clone(&factory_release);
            let _ = started_tx.send(());
            Box::pin(async move {
                let permit = release.acquire().await.map_err(|error| {
                    crate::error::ReactError::Other(format!(
                        "stale factory release semaphore closed: {error}"
                    ))
                })?;
                permit.forget();
                Ok(Box::new(MockAgent::new("stale-factory")) as Box<dyn Agent>)
            })
        }));
        registry
            .register_factory(
                SubagentDefinition::new("replaceable", "Old factory registration"),
                factory,
            )
            .await;

        let old_registry = registry.clone();
        let old_resolution =
            tokio::spawn(async move { old_registry.get_agent("replaceable").await });
        tokio::time::timeout(std::time::Duration::from_secs(1), started_rx.recv())
            .await
            .map_err(|_| crate::error::ReactError::Other("old factory did not start".to_string()))?
            .ok_or_else(|| {
                crate::error::ReactError::Other("old factory start channel closed".to_string())
            })?;

        let current: Arc<dyn Agent> = Arc::new(MockAgent::new("current-registration"));
        registry
            .register_shared(
                SubagentDefinition::new("replaceable", "Current registration"),
                Arc::clone(&current),
            )
            .await;
        release_old.add_permits(1);

        let old_result = old_resolution
            .await
            .map_err(|error| {
                crate::error::ReactError::Other(format!(
                    "old factory resolver task failed: {error}"
                ))
            })?
            .ok_or_else(|| {
                crate::error::ReactError::Other(
                    "old factory resolver did not return the current registration".to_string(),
                )
            })?;
        assert!(Arc::ptr_eq(&old_result, &current));
        let resolved = registry.get_agent("replaceable").await.ok_or_else(|| {
            crate::error::ReactError::Other("current registration disappeared".to_string())
        })?;
        assert!(Arc::ptr_eq(&resolved, &current));
        Ok(())
    }

    #[tokio::test]
    async fn test_factory_instantiation() {
        let registry = SubagentRegistry::new();

        let factory = Arc::new(FnAgentFactory::new(|| {
            Box::pin(async {
                Ok(
                    Box::new(MockAgent::new("lazy_agent").with_response("lazy result"))
                        as Box<dyn Agent>,
                )
            })
        }));

        let def = SubagentDefinition::new("lazy_agent", "Lazy agent");
        registry.register_factory(def, factory).await;

        // 应该已注册，但尚未实例化
        let registered = registry.get("lazy_agent").await.unwrap();
        assert_eq!(registered.definition.name, "lazy_agent");
        assert!(!registered.has_instance);

        // get_agent 应该触发工厂实例化
        let handle = registry.get_agent("lazy_agent").await;
        assert!(handle.is_some());

        // 现在应显示为已有实例
        let registered = registry.get("lazy_agent").await.unwrap();
        assert!(registered.has_instance);

        // 验证该 agent 确实可以正常工作
        let agent = handle.unwrap();
        let agent = agent.as_ref();
        let result = agent.execute("test").await.unwrap();
        assert_eq!(result, "lazy result");
    }
}
