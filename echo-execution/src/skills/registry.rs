/*!
 * Skill Registry —— 代码型与文件型 Skill 的统一生命周期管理器。
 *
 * 取代旧的 `SkillManager`，完整支持渐进式披露（progressive disclosure）：
 *
 * | 阶段 | 发生了什么 | Token 开销 |
 * |-------|-------------|------------|
 * | 发现（Discovery） | 解析 `SKILL.md` frontmatter -> `SkillDescriptor` | 每个 skill 约 50-100 |
 * | 目录（Catalog） | 将精简列表注入系统提示词 | 上述之和 |
 * | 激活（Activation） | 通过 `activate_skill` 工具加载完整 `SKILL.md` 正文 | <5000（推荐） |
 * | 资源（Resources） | 通过 `read_skill_resource` 工具加载单个文件 | 视情况而定 |
 */

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use tokio::sync::RwLock;
use tracing::warn;

use crate::skills::SkillInfo;
use crate::skills::external::SkillDocument;
use crate::skills::external::prompt_exec::{PromptContext, SkillSource, process_skill_content};
use crate::skills::external::types::{
    SkillContent, SkillDescriptor, SkillResourceEntry, SkillResourceKind, SkillSandboxPolicy,
};
use echo_core::sandbox::SandboxExecutor;

/**
 * 同一 agent 的 Skill 注册表所有视图共享的运行时激活权威。
 *
 * 目录与预备文档数据可以复制到并发的工具适配器中，但激活是
 * agent 运行时的事实，必须只有一个所有者。
 * 该句柄刻意设为公开，因为 `echo_agent` 与 `echo_execution`
 * 是不同的 crate；它是进程内的 Rust/Host 边界，而不是 SDK
 * 生命周期对象。
 */
#[derive(Clone)]
pub struct SkillActivationHandle {
    /// 共享的激活状态
    shared: Arc<SkillActivationShared>,
}

/// 激活状态的共享容器
struct SkillActivationShared {
    /// 受互斥锁保护的激活状态
    state: std::sync::Mutex<SkillActivationState>,
}

/// Skill 激活状态
struct SkillActivationState {
    /// 当前会话 ID
    session_id: String,
    /// 状态纪元，reset/restore 时递增
    epoch: u64,
    /// 每个 skill 的激活代数，退役时递增
    generations: HashMap<String, u64>,
    /// 已激活的 skill
    activated: HashMap<String, ActivatedSkill>,
    /// 正在进行中的激活
    in_flight: HashMap<String, Arc<ActivationFlight>>,
}

/// 已激活 skill 的记录
#[derive(Clone)]
struct ActivatedSkill {
    /// 激活键（恢复的激活为 None）
    key: Option<ActivationKey>,
    /// 激活后的内容（恢复的激活为 None）
    content: Option<SkillContent>,
    /// 激活时生效的沙箱策略
    sandbox_policy: Option<SkillSandboxPolicy>,
}

/// 标识一次激活输入的键
#[derive(Clone, PartialEq, Eq)]
struct ActivationKey {
    /// 激活参数
    arguments: Vec<String>,
    /// skill 来源
    source: SkillSource,
}

/// 一次进行中的激活
struct ActivationFlight {
    /// 创建时的状态纪元
    epoch: u64,
    /// 创建时的激活代数
    generation: u64,
    /// 激活键
    key: ActivationKey,
    /// 是否已被放弃（被毒化）
    abandoned: std::sync::atomic::AtomicBool,
    /// 测试用：等待者计数
    #[cfg(test)]
    waiters: std::sync::atomic::AtomicUsize,
    /// 激活结果（只设置一次）
    result: tokio::sync::OnceCell<std::result::Result<SkillContent, String>>,
}

/// 申请激活的结果
enum ActivationClaim {
    /// 已有相同输入的缓存内容
    Cached(Box<SkillContent>),
    /// 由会话恢复的激活（无内容）
    Restored,
    /// 进行中的激活已被放弃
    Abandoned,
    /// 进行中的激活与本次输入冲突
    Conflict,
    /// 需要执行或等待的激活
    Flight(Arc<ActivationFlight>),
}

/// 激活尝试守卫：未完成即被丢弃时毒化对应的激活
struct ActivationAttemptGuard {
    /// 激活句柄
    activation: SkillActivationHandle,
    /// skill 名称
    name: String,
    /// 对应的进行中激活
    flight: Arc<ActivationFlight>,
    /// 是否已完成
    completed: bool,
}

impl ActivationAttemptGuard {
    /// 创建激活尝试守卫
    fn new(activation: SkillActivationHandle, name: &str, flight: Arc<ActivationFlight>) -> Self {
        Self {
            activation,
            name: name.to_string(),
            flight,
            completed: false,
        }
    }

    /// 标记激活尝试已完成
    fn complete(&mut self) {
        self.completed = true;
    }
}

impl Drop for ActivationAttemptGuard {
    /// 未完成时毒化进行中的激活
    fn drop(&mut self) {
        if !self.completed {
            self.activation.poison(&self.name, &self.flight);
        }
    }
}

impl SkillActivationState {
    /// 创建带随机会话 ID 的空状态
    fn new() -> Self {
        Self {
            session_id: format!(
                "session-{}",
                uuid::Uuid::new_v4()
                    .to_string()
                    .chars()
                    .take(8)
                    .collect::<String>()
            ),
            epoch: 0,
            generations: HashMap::new(),
            activated: HashMap::new(),
            in_flight: HashMap::new(),
        }
    }
}

impl SkillActivationHandle {
    /// 创建新的激活句柄
    fn new() -> Self {
        Self {
            shared: Arc::new(SkillActivationShared {
                state: std::sync::Mutex::new(SkillActivationState::new()),
            }),
        }
    }

    /// 以确定性顺序返回当前已激活的 Skill 名称。
    pub fn activated_names(&self) -> Vec<String> {
        let state = self
            .shared
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let mut names = state.activated.keys().cloned().collect::<Vec<_>>();
        names.sort();
        names
    }

    /// 返回当前会话 ID
    fn session_id(&self) -> String {
        self.shared
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .session_id
            .clone()
    }

    /// 判断指定 skill 是否已激活
    fn is_activated(&self, name: &str) -> bool {
        self.shared
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .activated
            .contains_key(name)
    }

    /// 返回已激活 skill 数量
    fn activated_count(&self) -> usize {
        self.shared
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .activated
            .len()
    }

    /// 返回指定已激活 skill 的沙箱策略
    fn active_sandbox_policy(&self, name: &str) -> Option<SkillSandboxPolicy> {
        self.shared
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .activated
            .get(name)
            .and_then(|activation| activation.sandbox_policy.clone())
    }

    /// 直接标记 skill 为已激活（已激活或进行中时返回 false）
    fn mark_activated(&self, name: &str, sandbox_policy: Option<SkillSandboxPolicy>) -> bool {
        let mut state = self
            .shared
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if state.activated.contains_key(name) || state.in_flight.contains_key(name) {
            return false;
        }
        state.activated.insert(
            name.to_string(),
            ActivatedSkill {
                key: None,
                content: None,
                sandbox_policy,
            },
        );
        true
    }

    /// 清空所有激活并推进纪元
    fn reset(&self) {
        let mut state = self
            .shared
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        state.epoch = state.epoch.wrapping_add(1);
        state.generations.clear();
        state.activated.clear();
        state.in_flight.clear();
    }

    /// 退役指定 skill 的激活
    fn retire(&self, name: &str) {
        let mut state = self
            .shared
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        Self::retire_name(&mut state, name);
    }

    /// 在已加锁状态上退役指定 skill
    fn retire_name(state: &mut SkillActivationState, name: &str) {
        Self::advance_generation(state, name);
        state.activated.remove(name);
        state.in_flight.remove(name);
    }

    /// 推进指定 skill 的激活代数
    fn advance_generation(state: &mut SkillActivationState, name: &str) {
        let generation = state.generations.entry(name.to_string()).or_default();
        *generation = generation.wrapping_add(1);
    }

    /// 用给定列表重置并恢复激活状态，返回去重排序后的名称
    fn restore(&self, skills: Vec<(String, Option<SkillSandboxPolicy>)>) -> Vec<String> {
        let mut state = self
            .shared
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        state.epoch = state.epoch.wrapping_add(1);
        state.generations.clear();
        state.activated.clear();
        state.in_flight.clear();
        let mut restored = Vec::with_capacity(skills.len());
        for (name, sandbox_policy) in skills {
            state.activated.insert(
                name.clone(),
                ActivatedSkill {
                    key: None,
                    content: None,
                    sandbox_policy,
                },
            );
            restored.push(name);
        }
        restored.sort();
        restored.dedup();
        restored
    }

    /// 申请激活：返回缓存、冲突或需要执行/等待的进行中激活
    fn claim(&self, name: &str, key: ActivationKey) -> ActivationClaim {
        let mut state = self
            .shared
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if let Some(flight) = state.in_flight.get(name) {
            if flight.abandoned.load(std::sync::atomic::Ordering::SeqCst) {
                return ActivationClaim::Abandoned;
            }
            return if flight.key == key {
                #[cfg(test)]
                flight
                    .waiters
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                ActivationClaim::Flight(Arc::clone(flight))
            } else {
                ActivationClaim::Conflict
            };
        }
        if let Some(active) = state.activated.get(name) {
            match (&active.key, &active.content) {
                (Some(active_key), Some(content)) if active_key == &key => {
                    return ActivationClaim::Cached(Box::new(content.clone()));
                }
                (None, None) => return ActivationClaim::Restored,
                _ => Self::advance_generation(&mut state, name),
            }
        }
        let flight = Arc::new(ActivationFlight {
            epoch: state.epoch,
            generation: state.generations.get(name).copied().unwrap_or_default(),
            key,
            abandoned: std::sync::atomic::AtomicBool::new(false),
            #[cfg(test)]
            waiters: std::sync::atomic::AtomicUsize::new(0),
            result: tokio::sync::OnceCell::new(),
        });
        state
            .in_flight
            .insert(name.to_string(), Arc::clone(&flight));
        ActivationClaim::Flight(flight)
    }

    /// 发布激活结果；若激活已被退役则返回错误
    fn publish(
        &self,
        name: &str,
        flight: &Arc<ActivationFlight>,
        content: SkillContent,
        sandbox_policy: Option<SkillSandboxPolicy>,
    ) -> std::result::Result<(), String> {
        let mut state = self
            .shared
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let current_generation = state.generations.get(name).copied().unwrap_or_default();
        let owns_flight = state
            .in_flight
            .get(name)
            .is_some_and(|current| Arc::ptr_eq(current, flight));
        if state.epoch != flight.epoch || current_generation != flight.generation || !owns_flight {
            return Err(format!(
                "Skill '{name}' activation was retired before publication"
            ));
        }
        state.in_flight.remove(name);
        state.activated.insert(
            name.to_string(),
            ActivatedSkill {
                key: Some(flight.key.clone()),
                content: Some(content),
                sandbox_policy,
            },
        );
        Ok(())
    }

    /// 放弃进行中的激活（仅当仍为当前激活时移除）
    fn abandon(&self, name: &str, flight: &Arc<ActivationFlight>) {
        let mut state = self
            .shared
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if state
            .in_flight
            .get(name)
            .is_some_and(|current| Arc::ptr_eq(current, flight))
        {
            state.in_flight.remove(name);
        }
    }

    /// 将进行中的激活标记为已放弃
    fn poison(&self, name: &str, flight: &Arc<ActivationFlight>) {
        let state = self
            .shared
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if state
            .in_flight
            .get(name)
            .is_some_and(|current| Arc::ptr_eq(current, flight))
        {
            flight
                .abandoned
                .store(true, std::sync::atomic::Ordering::SeqCst);
        }
    }
}

// -- SkillRegistry --

/**
 * 中央 skill 生命周期管理器。
 *
 * 同时跟踪代码型 skill（通过 [`Skill`](crate::skills::Skill) trait 注册）
 * 与文件型 skill（从 `SKILL.md` 文件发现）。提供：
 *
 * - **目录生成**，用于注入系统提示词（Tier 1）
 * - **激活跟踪**，带去重（Tier 2）
 * - **资源访问**，从已激活 skill 的目录读取（Tier 3）
 */
pub struct SkillRegistry {
    /// 文件型 skill：名称 -> 描述符（Tier 1 元数据）
    descriptors: HashMap<String, SkillDescriptor>,

    /**
     * 由预备的插件代次提供的冻结 SKILL.md 文档。
     * 非插件的发现刻意保持惰性且基于文件系统。
     */
    prepared_documents: HashMap<String, SkillDocument>,

    /// 共享的运行时激活权威。
    activation: SkillActivationHandle,

    /// 代码型 skill：名称 -> 信息（通过 `add_skill` 注册）
    code_skills: HashMap<String, SkillInfo>,

    /// 可选的沙箱执行器，用于激活带内联命令的本地 skill。
    sandbox: Option<Arc<dyn SandboxExecutor>>,

    /**
     * 反向索引：来源标签（例如 `"plugin:my-plugin"`）-> 在该来源下注册的
     * skill 名称。使 `unregister_by_source` 在禁用/卸载时能精确移除
     * 某一个插件的 skill，而无需扫描全部描述符。没有 `source` 的 skill
     * 不在此映射中。
     */
    by_source: HashMap<String, HashSet<String>>,

    /// 按 skill 名称索引的插件变量上下文。
    plugin_variables: HashMap<String, echo_core::plugin::PluginVariables>,
}

impl SkillRegistry {
    /// 创建空的 skill 注册表
    pub fn new() -> Self {
        Self {
            descriptors: HashMap::new(),
            prepared_documents: HashMap::new(),
            activation: SkillActivationHandle::new(),
            code_skills: HashMap::new(),
            sandbox: None,
            by_source: HashMap::new(),
            plugin_variables: HashMap::new(),
        }
    }

    /// 返回规范的进程内激活权威。
    pub fn activation_handle(&self) -> SkillActivationHandle {
        self.activation.clone()
    }

    /**
     * 构造一个空的定义视图，与本注册表共享运行时激活权威。
     *
     * 调用方可以为并发的资源工具填充描述符与预备文档，
     * 而不会创建另一套激活集合或沙箱策略权威。
     */
    pub fn activation_view(&self) -> Self {
        let mut registry = Self::new();
        registry.activation = self.activation.clone();
        registry
    }

    // -- 文件型 skill（渐进式披露） --

    /// 注册一个已发现的文件型 skill 描述符。
    pub fn register_descriptor(&mut self, descriptor: SkillDescriptor) {
        self.prepared_documents.remove(&descriptor.name);
        self.insert_descriptor(descriptor);
    }

    /// 插入描述符并维护来源反向索引；替换时退役旧激活
    fn insert_descriptor(&mut self, descriptor: SkillDescriptor) {
        let replacing = self.descriptors.contains_key(&descriptor.name);
        // 注册时校验路径
        for warning in descriptor.validate_paths() {
            warn!("Skill '{}': {}", descriptor.name, warning);
        }
        /* 若调用方直接替换描述符，先移除旧的反向索引。发现流程通常
        会拒绝重名，但这样可以让公开注册表对所有调用方都保持内部一致。 */
        if let Some(previous_source) = self
            .descriptors
            .get(&descriptor.name)
            .and_then(|previous| previous.source.as_deref())
            .map(str::to_string)
            && let Some(names) = self.by_source.get_mut(&previous_source)
        {
            names.remove(&descriptor.name);
            if names.is_empty() {
                self.by_source.remove(&previous_source);
            }
        }
        /* 记录来源出处，便于之后禁用/卸载时精确移除
        这一组（P1-reload）。 */
        if let Some(src) = descriptor.source.as_deref() {
            self.by_source
                .entry(src.to_string())
                .or_default()
                .insert(descriptor.name.clone());
        }
        let name = descriptor.name.clone();
        self.descriptors.insert(name.clone(), descriptor);
        if replacing {
            self.activation.retire(&name);
        }
    }

    /**
     * 移除在给定来源标签（例如 `"plugin:my-plugin"`）下注册的所有 skill，
     * 返回移除数量。
     *
     * 插件运行时在禁用/卸载时调用，确保 agent 的 SkillRegistry
     * 不再保留被禁用插件的 skill。每个被移除的 skill 也会通过
     * `remove_descriptor` 取消激活并从运行时记录中清除。
     */
    pub fn unregister_by_source(&mut self, source: &str) -> usize {
        self.unregister_names_by_source(source).len()
    }

    /// 移除并返回某一来源注册的全部 skill 名称。
    pub fn unregister_names_by_source(&mut self, source: &str) -> Vec<String> {
        let names = match self.by_source.remove(source) {
            Some(set) => set,
            None => return Vec::new(),
        };
        let mut removed = Vec::new();
        for name in names {
            if self.remove_descriptor(&name) {
                removed.push(name);
            }
        }
        removed.sort();
        removed
    }

    /**
     * 为已注册的 skill 打上来源标签，便于之后通过 `unregister_by_source`
     * 按组卸载。
     *
     * 这是插件集成方的加载后入口：它先调用 `load_skills_from_dir`
     * （发现并注册描述符，但不知道来源），再为返回的名称打上插件来源。
     * 已被其他来源标记的 skill 会被跳过，以避免跨插件污染。
     */
    pub fn tag_source(&mut self, names: &[String], source: &str) {
        self.tag_source_with_variables(names, source, None);
    }

    /// 为 skill 打上来源标签并设置插件变量替换上下文。
    pub fn tag_source_with_variables(
        &mut self,
        names: &[String],
        source: &str,
        variables: Option<&echo_core::plugin::PluginVariables>,
    ) {
        for name in names {
            let mut activation_input_changed = false;
            if let Some(desc) = self.descriptors.get_mut(name) {
                // 仅在尚未被其他来源拥有时打标签。
                if desc.source.is_none() {
                    desc.source = Some(source.to_string());
                    self.by_source
                        .entry(source.to_string())
                        .or_default()
                        .insert(name.clone());
                }
                if desc.source.as_deref() == Some(source)
                    && let Some(variables) = variables
                {
                    self.plugin_variables
                        .insert(name.clone(), variables.clone());
                    activation_input_changed = true;
                }
            }
            if activation_input_changed {
                self.activation.retire(name);
            }
        }
    }

    /// 从不可变的预备代次注册一个已校验的 Skill。
    pub fn register_prepared(&mut self, document: SkillDocument) {
        let descriptor = document.descriptor().clone();
        let name = descriptor.name.clone();
        self.prepared_documents.insert(name, document);
        self.insert_descriptor(descriptor);
    }

    /// 移除一个文件型描述符及其全部激活元数据。
    pub fn remove_descriptor(&mut self, name: &str) -> bool {
        let removed_descriptor = self.descriptors.remove(name);
        if let Some(source) = removed_descriptor
            .as_ref()
            .and_then(|descriptor| descriptor.source.as_deref())
            .map(str::to_string)
            && let Some(names) = self.by_source.get_mut(&source)
        {
            names.remove(name);
            if names.is_empty() {
                self.by_source.remove(&source);
            }
        }
        self.prepared_documents.remove(name);
        self.plugin_variables.remove(name);
        if removed_descriptor.is_some() {
            self.activation.retire(name);
        }
        removed_descriptor.is_some()
    }

    /// 挂载在激活期间执行内联命令所用的沙箱执行器。
    pub fn set_sandbox_manager(&mut self, manager: Arc<dyn SandboxExecutor>) {
        self.sandbox = Some(manager);
    }

    /// 按名称获取描述符。
    pub fn get_descriptor(&self, name: &str) -> Option<&SkillDescriptor> {
        self.descriptors.get(name)
    }

    /// 列出所有已发现的文件型 skill 描述符。
    pub fn list_descriptors(&self) -> Vec<&SkillDescriptor> {
        let mut descs: Vec<&SkillDescriptor> = self.descriptors.values().collect();
        descs.sort_by_key(|d| &d.name);
        descs
    }

    /// 已发现的文件型 skill 数量。
    pub fn descriptor_count(&self) -> usize {
        self.descriptors.len()
    }

    /**
     * 生成用于注入系统提示词的 skill 目录文本。
     *
     * 若没有可用的文件型 skill 则返回 `None`（按规范调用方应
     * 完全省略目录段落）。
     */
    pub fn catalog_prompt(&self) -> Option<String> {
        if self.descriptors.is_empty() {
            return None;
        }

        let mut lines = Vec::with_capacity(self.descriptors.len() + 4);
        lines.push(
            "The following skills provide specialized instructions for specific tasks.\n\
             When a task matches a skill's description, call the `activate_skill` tool \
             with the skill's name to load its full instructions."
                .to_string(),
        );
        lines.push(String::new());

        let mut names: Vec<&String> = self.descriptors.keys().collect();
        names.sort();
        for name in names {
            if let Some(desc) = self.descriptors.get(name) {
                lines.push(desc.catalog_line());
            }
        }

        Some(lines.join("\n"))
    }

    /// 列出所有可用 skill 名称（用于 `activate_skill` 工具的枚举约束）。
    pub fn available_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.descriptors.keys().cloned().collect();
        names.sort();
        names
    }

    // -- 激活跟踪 --

    /// 将 skill 标记为已激活。若已激活则返回 `false`（去重）。
    pub fn mark_activated(&self, name: &str) -> bool {
        let sandbox_policy = self
            .descriptors
            .get(name)
            .and_then(|descriptor| descriptor.sandbox.clone())
            .filter(SkillSandboxPolicy::is_constraining);
        self.activation.mark_activated(name, sandbox_policy)
    }

    /**
     * 清除会话本地的激活与沙箱策略状态，但不注销描述符。
     *
     * 运行时检查点身份可以共享同一注册表目录，同时保留各自独立的
     * 模型上下文。切换身份时不得把已激活的 skill 或其沙箱策略
     * 带入下一个上下文。
     */
    pub fn reset_activation_state(&self) {
        self.activation.reset();
    }

    /// 检查某个 skill 在本会话中是否已激活。
    pub fn is_activated(&self, name: &str) -> bool {
        self.activation.is_activated(name)
    }

    /**
     * 汇总当前所有已激活 skill 的 `allowed_tools` 并集。
     *
     * 若没有已激活的 skill 限制工具（空 = 不受限），返回 `None`，
     * 表示 agent 可以使用任意工具。当至少一个已激活 skill 声明了
     * `allowed-tools` 白名单时返回 `Some(set)`——此时只允许使用
     * 与返回集合中条目匹配的工具。
     */
    pub fn active_skill_allowed_tools(&self) -> Option<HashSet<String>> {
        let activated = self.activation.activated_names();
        let mut allowed = HashSet::new();
        let mut any_restricted = false;

        for name in &activated {
            if let Some(desc) = self.descriptors.get(name)
                && !desc.allowed_tools.is_empty()
            {
                any_restricted = true;
                for tool in &desc.allowed_tools {
                    allowed.insert(tool.clone());
                }
            }
        }

        if any_restricted {
            Some(allowed)
        } else {
            None // 没有已激活 skill 限制工具 → 不受限
        }
    }

    /// 已激活 skill 的数量。
    pub fn activated_count(&self) -> usize {
        self.activation.activated_count()
    }

    /// 以排序后的 Vec 返回全部已激活 skill 名称。
    pub fn activated_names(&self) -> Vec<String> {
        self.activation.activated_names()
    }

    /**
     * 以一次原子操作用持久化检查点替换激活状态。
     *
     * 未知名称会被忽略，因为检查点无法重建缺失的定义。沙箱策略
     * 从当前已安装的描述符重建，而不是作为第二个持久化权威被信任。
     */
    pub fn restore_activation_state(&self, names: &[String]) -> Vec<String> {
        let skills = names
            .iter()
            .filter_map(|name| {
                self.descriptors.get(name).map(|descriptor| {
                    let policy = descriptor
                        .sandbox
                        .clone()
                        .filter(SkillSandboxPolicy::is_constraining);
                    (name.clone(), policy)
                })
            })
            .collect();
        self.activation.restore(skills)
    }

    /**
     * 激活一个 skill：从磁盘读取完整内容、执行内联命令并替换变量。
     *
     * 返回结构化的 `SkillContent` 或错误。
     * 会自动将该 skill 标记为已激活。
     */
    pub async fn activate(&self, name: &str) -> echo_core::error::Result<SkillContent> {
        self.activate_with_args(name, &[], SkillSource::Local).await
    }

    /**
     * 使用用户提供的参数与来源上下文激活 skill。
     *
     * 这是完整的激活路径：
     * 1. 先递归激活依赖（如有）
     * 2. 读取 `SKILL.md` 正文
     * 3. 替换变量（`${SKILL_DIR}`、`${SESSION_ID}`、`${ARGUMENTS}` 等）
     * 4. 执行内联命令（`` !`cmd` `` 与 `` ```! cmd ``` ``），
     *    有可用沙箱时走配置的沙箱路径，否则走直接回退路径，
     *    使用最小环境变量并尽力在超时后终止
     * 5. 枚举打包的资源
     * 6. 若声明了沙箱策略则保存
     */
    pub async fn activate_with_args(
        &self,
        name: &str,
        args: &[String],
        source: SkillSource,
    ) -> echo_core::error::Result<SkillContent> {
        self.validate_activation_dependencies(name)?;
        if !self.descriptors.contains_key(name) {
            return Err(echo_core::error::ReactError::Other(format!(
                "Skill '{name}' not found in catalog"
            )));
        }
        let key = ActivationKey {
            arguments: args.to_vec(),
            source,
        };
        let flight = match self.activation.claim(name, key) {
            ActivationClaim::Cached(content) => return Ok(*content),
            ActivationClaim::Restored => {
                return Err(echo_core::error::ReactError::Other(format!(
                    "Skill '{name}' was restored as active; reset activation state before re-activating it"
                )));
            }
            ActivationClaim::Abandoned => {
                return Err(echo_core::error::ReactError::Other(format!(
                    "Skill '{name}' activation was cancelled with unknown side-effect settlement; reset activation state before retrying"
                )));
            }
            ActivationClaim::Conflict => {
                return Err(echo_core::error::ReactError::Other(format!(
                    "Skill '{name}' is already activating or active with different arguments"
                )));
            }
            ActivationClaim::Flight(flight) => flight,
        };

        let outcome = flight
            .result
            .get_or_init(|| async {
                if flight
                    .abandoned
                    .load(std::sync::atomic::Ordering::SeqCst)
                {
                    return Err(format!(
                        "Skill '{name}' activation was cancelled with unknown side-effect settlement; reset activation state before retrying"
                    ));
                }
                let mut guard = ActivationAttemptGuard::new(
                    self.activation.clone(),
                    name,
                    Arc::clone(&flight),
                );
                let result = self
                    .compute_activation(name, args, source, &flight)
                    .await
                    .map_err(|error| error.to_string());
                guard.complete();
                result
            })
            .await
            .clone();
        if outcome.is_err() && !flight.abandoned.load(std::sync::atomic::Ordering::SeqCst) {
            self.activation.abandon(name, &flight);
        }
        outcome.map_err(echo_core::error::ReactError::Other)
    }

    /// 执行实际的激活计算并发布结果
    async fn compute_activation(
        &self,
        name: &str,
        args: &[String],
        source: SkillSource,
        flight: &Arc<ActivationFlight>,
    ) -> echo_core::error::Result<SkillContent> {
        let deps_activated = self.activate_dependencies(name, source).await?;
        let descriptor = self.descriptors.get(name).cloned().ok_or_else(|| {
            echo_core::error::ReactError::Other(format!("Skill '{name}' not found in catalog"))
        })?;
        let location = descriptor.location.clone();
        let skill_dir = location
            .parent()
            .map(std::path::Path::to_path_buf)
            .ok_or_else(|| {
                echo_core::error::ReactError::Other(format!(
                    "Cannot determine skill directory from '{}'",
                    location.display()
                ))
            })?;
        let document = match self.prepared_documents.get(name) {
            Some(document) => document.clone(),
            None => {
                let raw_content = tokio::fs::read_to_string(&location)
                    .await
                    .map_err(|error| {
                        echo_core::error::ReactError::Other(format!(
                            "Failed to read SKILL.md at '{}': {error}",
                            location.display()
                        ))
                    })?;
                SkillDocument::parse_at(&raw_content, location)?
            }
        };
        let mut raw_instructions = document.instructions().to_string();
        if let Some(variables) = self.plugin_variables.get(name) {
            raw_instructions = variables.substitute(&raw_instructions);
        }
        let context = PromptContext {
            skill_dir: skill_dir.display().to_string(),
            session_id: self.activation.session_id(),
            arguments: args.to_vec(),
            shell: descriptor.shell.clone(),
            source,
            sandbox: self.sandbox.clone(),
            ..Default::default()
        };
        let instructions = process_skill_content(&raw_instructions, &context).await;
        let resources = enumerate_resources(&skill_dir).await;
        let instructions = if deps_activated.is_empty() {
            instructions
        } else {
            format!(
                "<skill-dependencies>\nActivated dependencies: {}\n</skill-dependencies>\n\n{}",
                deps_activated.join(", "),
                instructions
            )
        };
        let content = SkillContent {
            descriptor: descriptor.clone(),
            instructions,
            resources,
        };
        let sandbox_policy = descriptor
            .sandbox
            .filter(SkillSandboxPolicy::is_constraining);
        self.activation
            .publish(name, flight, content.clone(), sandbox_policy)
            .map_err(echo_core::error::ReactError::Other)?;
        Ok(content)
    }

    /// 校验依赖图中不存在循环依赖
    fn validate_activation_dependencies(&self, name: &str) -> echo_core::error::Result<()> {
        /// 深度优先遍历依赖并检测环
        fn visit(
            registry: &SkillRegistry,
            name: &str,
            visiting: &mut HashSet<String>,
            visited: &mut HashSet<String>,
            path: &mut Vec<String>,
        ) -> echo_core::error::Result<()> {
            if visited.contains(name) {
                return Ok(());
            }
            if !visiting.insert(name.to_string()) {
                path.push(name.to_string());
                let cycle = path.join(" -> ");
                path.pop();
                return Err(echo_core::error::ReactError::Other(format!(
                    "Circular skill dependency detected: {cycle}"
                )));
            }
            path.push(name.to_string());
            if let Some(descriptor) = registry.descriptors.get(name) {
                for dependency in &descriptor.depends_on {
                    if registry.descriptors.contains_key(dependency) {
                        visit(registry, dependency, visiting, visited, path)?;
                    }
                }
            }
            path.pop();
            visiting.remove(name);
            visited.insert(name.to_string());
            Ok(())
        }

        let mut visiting = HashSet::new();
        let mut visited = HashSet::new();
        visit(self, name, &mut visiting, &mut visited, &mut Vec::new())
    }

    /**
     * 递归激活某个 skill 尚未满足的依赖。
     *
     * 返回本次新激活的依赖名称列表。
     * 缺失的依赖会记录警告，但不会阻止激活。
     */
    async fn activate_dependencies(
        &self,
        name: &str,
        source: SkillSource,
    ) -> echo_core::error::Result<Vec<String>> {
        let deps = self
            .descriptors
            .get(name)
            .map(|d| d.depends_on.clone())
            .unwrap_or_default();

        if deps.is_empty() {
            return Ok(Vec::new());
        }

        let mut activated = Vec::new();
        for dep in &deps {
            if self.activation.is_activated(dep) {
                continue;
            }
            if !self.descriptors.contains_key(dep) {
                warn!(
                    "Skill '{}' depends on '{}' which is not available; skipping",
                    name, dep
                );
                continue;
            }
            // 递归激活（依赖的依赖）
            match Box::pin(self.activate_with_args(dep, &[], source)).await {
                Ok(_) => activated.push(dep.clone()),
                Err(e) => {
                    warn!(
                        "Failed to activate dependency '{}' of '{}': {}",
                        dep, name, e
                    );
                }
            }
        }
        Ok(activated)
    }

    /// 获取已激活 skill 当前生效的沙箱策略。
    pub fn get_active_sandbox_policy(&self, skill_name: &str) -> Option<SkillSandboxPolicy> {
        self.activation.active_sandbox_policy(skill_name)
    }

    /// 获取 skill 的完整依赖树（递归、深度优先）。
    pub fn get_dependency_tree(&self, name: &str) -> Vec<String> {
        let mut result = Vec::new();
        let mut visited = HashSet::new();
        self.collect_dependencies(name, &mut result, &mut visited);
        result
    }

    /// 深度优先收集依赖到结果列表
    fn collect_dependencies(
        &self,
        name: &str,
        result: &mut Vec<String>,
        visited: &mut HashSet<String>,
    ) {
        if visited.contains(name) {
            return;
        }
        visited.insert(name.to_string());

        if let Some(descriptor) = self.descriptors.get(name) {
            for dep in &descriptor.depends_on {
                self.collect_dependencies(dep, result, visited);
                if !result.contains(dep) {
                    result.push(dep.clone());
                }
            }
        }
    }

    // -- 代码型 skill --

    /// 记录通过 `add_skill` 安装的代码型 skill。
    pub fn record_code_skill(&mut self, info: SkillInfo) {
        self.code_skills.insert(info.name.clone(), info);
    }

    /// 检查代码型 skill 是否已安装。
    pub fn has_code_skill(&self, name: &str) -> bool {
        self.code_skills.contains_key(name)
    }

    /// 获取代码型 skill 的信息。
    pub fn get_code_skill(&self, name: &str) -> Option<&SkillInfo> {
        self.code_skills.get(name)
    }

    /// 列出所有已安装的代码型 skill。
    pub fn list_code_skills(&self) -> Vec<&SkillInfo> {
        let mut infos: Vec<&SkillInfo> = self.code_skills.values().collect();
        infos.sort_by_key(|i| &i.name);
        infos
    }

    // -- 统一查询 --

    /// 检查某个 skill（代码型或文件型）是否已安装/已发现。
    pub fn is_installed(&self, name: &str) -> bool {
        self.code_skills.contains_key(name) || self.descriptors.contains_key(name)
    }

    /// skill 总数（代码型 + 文件型）。
    pub fn count(&self) -> usize {
        self.code_skills.len() + self.descriptors.len()
    }

    /// 以 `SkillInfo` 列出全部已安装 skill（统一视图）。
    pub fn list(&self) -> Vec<&SkillInfo> {
        self.list_code_skills()
    }

    /**
     * 在会话开始时直接注入系统提示词的核心方法论 skill
     * （而不仅是在目录中列出）。
     */
    pub const DEFAULT_BASELINE_SKILLS: &'static [&'static str] = &[
        "brainstorming",
        "systematic-debugging",
        "verification-before-completion",
        "writing-plans",
    ];

    /**
     * 将基线方法论 skill 的正文注入系统提示词。
     *
     * 对每个已启用且 metadata.category == "methodology" 的基线 skill，
     * 从磁盘读取 SKILL.md 正文，去除 YAML frontmatter，
     * 并以 `<skill>` 标签包裹后追加。
     */
    pub fn inject_methodology_baseline(
        &self,
        system_prompt: &mut String,
        enabled_baseline: &[&str],
    ) {
        for desc in self.descriptors.values() {
            let category = desc
                .metadata
                .get("category")
                .map(|s| s.as_str())
                .unwrap_or("");
            if category != "methodology" {
                continue;
            }
            if !enabled_baseline.iter().any(|name| *name == desc.name) {
                continue;
            }
            let skill_file =
                if desc.location.file_name().and_then(|name| name.to_str()) == Some("SKILL.md") {
                    desc.location.clone()
                } else {
                    desc.location.join("SKILL.md")
                };
            let document = match self.prepared_documents.get(&desc.name) {
                Some(document) => document.clone(),
                None => {
                    let Ok(content) = std::fs::read_to_string(&skill_file) else {
                        continue;
                    };
                    let Ok(document) = SkillDocument::parse_at(&content, &skill_file) else {
                        continue;
                    };
                    document
                }
            };
            let raw_body = document.instructions().trim();
            let substituted;
            let body = if let Some(variables) = self.plugin_variables.get(&desc.name) {
                substituted = variables.substitute(raw_body);
                substituted.trim()
            } else {
                raw_body
            };
            if body.is_empty() {
                continue;
            }
            system_prompt.push_str("\n\n<skill name=\"");
            system_prompt.push_str(&desc.name);
            system_prompt.push_str("\">\n");
            system_prompt.push_str(body);
            system_prompt.push_str("\n</skill>");
        }
    }
}

impl Default for SkillRegistry {
    /// 创建空注册表
    fn default() -> Self {
        Self::new()
    }
}

// -- 共享注册表句柄（供需要并发访问的工具使用） --

/// `SkillRegistry` 的线程安全句柄，在 agent 与工具之间共享。
pub type SharedRegistry = Arc<RwLock<SkillRegistry>>;

/// 由现有注册表创建共享注册表。
pub fn shared_registry(registry: SkillRegistry) -> SharedRegistry {
    Arc::new(RwLock::new(registry))
}

/// 枚举 skill 目录下 `scripts/`、`references/`、`assets/` 中的资源文件。
async fn enumerate_resources(skill_dir: &std::path::Path) -> Vec<SkillResourceEntry> {
    let mut resources = Vec::new();

    let dirs = [
        ("scripts", SkillResourceKind::Script),
        ("references", SkillResourceKind::Reference),
        ("assets", SkillResourceKind::Asset),
    ];

    for (dir_name, kind) in &dirs {
        let dir_path = skill_dir.join(dir_name);
        if !dir_path.is_dir() {
            continue;
        }
        if let Ok(mut entries) = tokio::fs::read_dir(&dir_path).await {
            while let Ok(Some(entry)) = entries.next_entry().await {
                let path = entry.path();
                if path.is_file()
                    && let Some(file_name) = path.file_name().and_then(|n| n.to_str())
                {
                    resources.push(SkillResourceEntry {
                        relative_path: format!("{}/{}", dir_name, file_name),
                        kind: *kind,
                    });
                }
            }
        }
    }

    // 同时枚举顶层中非 SKILL.md 的文本资源。
    if let Ok(mut entries) = tokio::fs::read_dir(skill_dir).await {
        while let Ok(Some(entry)) = entries.next_entry().await {
            let path = entry.path();
            if path.is_file()
                && let Some(name) = path.file_name().and_then(|n| n.to_str())
                && name != "SKILL.md"
                && (name.ends_with(".md")
                    || name.ends_with(".txt")
                    || name.ends_with(".yaml")
                    || name.ends_with(".yml")
                    || name.ends_with(".json"))
            {
                resources.push(SkillResourceEntry {
                    relative_path: name.to_string(),
                    kind: SkillResourceKind::Other,
                });
            }
        }
    }

    resources.sort_by(|a, b| a.relative_path.cmp(&b.relative_path));
    resources
}

// -- 测试 --

#[cfg(test)]
mod tests {
    use super::*;
    use echo_core::sandbox::{ExecutionResult, IsolationLevel, SandboxCommand};
    use futures::future::BoxFuture;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// 测试用沙箱：执行时阻塞直到被显式释放
    struct BlockingSandbox {
        /// 执行调用次数
        calls: AtomicUsize,
        /// 进入执行时发出通知
        entered: tokio::sync::Notify,
        /// 释放阻塞的通知
        release: tokio::sync::Notify,
    }

    impl BlockingSandbox {
        /// 创建测试沙箱
        fn new() -> Self {
            Self {
                calls: AtomicUsize::new(0),
                entered: tokio::sync::Notify::new(),
                release: tokio::sync::Notify::new(),
            }
        }
    }

    impl SandboxExecutor for BlockingSandbox {
        /// 执行器名称
        fn name(&self) -> &str {
            "blocking-skill-test"
        }

        /// 返回进程级隔离
        fn isolation_level(&self) -> IsolationLevel {
            IsolationLevel::Process
        }

        /// 始终可用
        fn is_available(&self) -> BoxFuture<'_, bool> {
            Box::pin(async { true })
        }

        /// 计数并阻塞直到被释放，然后返回固定结果
        fn execute(
            &self,
            _command: SandboxCommand,
        ) -> BoxFuture<'_, echo_core::error::Result<ExecutionResult>> {
            Box::pin(async move {
                self.calls.fetch_add(1, Ordering::SeqCst);
                self.entered.notify_one();
                self.release.notified().await;
                Ok(ExecutionResult {
                    exit_code: 0,
                    stdout: "once".to_string(),
                    stderr: String::new(),
                    duration: std::time::Duration::from_millis(1),
                    sandbox_type: self.name().to_string(),
                    timed_out: false,
                    cancelled: false,
                    output_truncated: false,
                    stdout_bytes: 4,
                    stderr_bytes: 0,
                })
            })
        }
    }

    /// 构造测试用描述符
    fn make_descriptor(name: &str, desc: &str) -> SkillDescriptor {
        SkillDescriptor {
            source: None,
            name: name.into(),
            description: desc.into(),
            location: PathBuf::from(format!("/skills/{}/SKILL.md", name)),
            license: None,
            compatibility: None,
            metadata: HashMap::new(),
            allowed_tools: vec![],
            shell: None,
            paths: vec![],
            triggers: vec![],
            hooks: None,
            sandbox: None,
            depends_on: vec![],
        }
    }

    /// 构造指向临时目录中内联命令 skill 的描述符
    fn inline_descriptor(root: &std::path::Path, name: &str) -> SkillDescriptor {
        SkillDescriptor {
            location: root.join(name).join("SKILL.md"),
            ..make_descriptor(name, "Inline activation test")
        }
    }

    /// 写入包含内联命令的测试 SKILL.md
    fn write_inline_skill(root: &std::path::Path, name: &str) -> echo_core::error::Result<()> {
        let directory = root.join(name);
        std::fs::create_dir_all(&directory)?;
        std::fs::write(
            directory.join("SKILL.md"),
            format!(
                "---\nname: {name}\ndescription: Inline activation test\n---\n\nResult ${{ARGUMENTS}}: !`count-once`\n"
            ),
        )?;
        Ok(())
    }

    /// 生成唯一的测试临时根目录
    fn skill_test_root(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "echo-skill-{label}-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ))
    }

    /// 验证依赖校验拒绝循环依赖
    #[test]
    fn activation_dependency_validation_rejects_cycles() {
        let mut registry = SkillRegistry::new();
        let mut a = make_descriptor("a", "A");
        a.depends_on = vec!["b".to_string()];
        let mut b = make_descriptor("b", "B");
        b.depends_on = vec!["a".to_string()];
        registry.register_descriptor(a);
        registry.register_descriptor(b);

        let error = registry
            .validate_activation_dependencies("a")
            .err()
            .unwrap_or_else(|| echo_core::error::ReactError::Other("missing error".to_string()));
        assert!(error.to_string().contains("a -> b -> a"));
        assert!(registry.activated_names().is_empty());
    }

    /// 验证依赖校验拒绝自依赖
    #[test]
    fn activation_dependency_validation_rejects_self_dependency() {
        let mut registry = SkillRegistry::new();
        let mut skill = make_descriptor("self", "Self");
        skill.depends_on = vec!["self".to_string()];
        registry.register_descriptor(skill);

        let error = registry
            .validate_activation_dependencies("self")
            .err()
            .unwrap_or_else(|| echo_core::error::ReactError::Other("missing error".to_string()));
        assert!(error.to_string().contains("self -> self"));
        assert!(registry.activated_names().is_empty());
    }

    /// 验证新注册表为空
    #[test]
    fn test_registry_new() {
        let reg = SkillRegistry::new();
        assert_eq!(reg.count(), 0);
        assert!(reg.catalog_prompt().is_none());
    }

    /// 验证重置激活状态不影响目录
    #[test]
    fn reset_activation_state_preserves_catalog() {
        let mut registry = SkillRegistry::new();
        registry.register_descriptor(make_descriptor("code-review", "Review code quality"));
        assert!(registry.mark_activated("code-review"));
        registry.reset_activation_state();
        assert!(registry.activated_names().is_empty());
        assert_eq!(registry.available_names(), vec!["code-review".to_string()]);
    }

    /// 验证注册描述符
    #[test]
    fn test_register_descriptor() {
        let mut reg = SkillRegistry::new();
        reg.register_descriptor(make_descriptor("code-review", "Review code quality"));

        assert_eq!(reg.descriptor_count(), 1);
        assert!(reg.get_descriptor("code-review").is_some());
        assert!(reg.is_installed("code-review"));
    }

    /// 验证按来源卸载返回精确的 skill 名称
    #[test]
    fn source_group_unload_returns_exact_skill_names() {
        let mut reg = SkillRegistry::new();
        reg.register_descriptor(make_descriptor("plugin-a-one", "one"));
        reg.register_descriptor(make_descriptor("plugin-a-two", "two"));
        reg.register_descriptor(make_descriptor("baseline", "baseline"));
        reg.tag_source(
            &["plugin-a-one".to_string(), "plugin-a-two".to_string()],
            "plugin:a",
        );

        assert_eq!(
            reg.unregister_names_by_source("plugin:a"),
            vec!["plugin-a-one".to_string(), "plugin-a-two".to_string()]
        );
        assert!(reg.get_descriptor("plugin-a-one").is_none());
        assert!(reg.get_descriptor("plugin-a-two").is_none());
        assert!(reg.get_descriptor("baseline").is_some());
    }

    /// 验证替换描述符会更新来源反向索引
    #[test]
    fn replacing_descriptor_updates_source_reverse_index() {
        let mut reg = SkillRegistry::new();
        let mut first = make_descriptor("shared", "first");
        first.source = Some("plugin:a".to_string());
        reg.register_descriptor(first);
        let mut replacement = make_descriptor("shared", "replacement");
        replacement.source = Some("plugin:b".to_string());
        reg.register_descriptor(replacement);

        assert_eq!(reg.unregister_by_source("plugin:a"), 0);
        assert_eq!(reg.unregister_by_source("plugin:b"), 1);
        assert!(reg.get_descriptor("shared").is_none());
    }

    /// 验证目录提示词内容
    #[test]
    fn test_catalog_prompt() {
        let mut reg = SkillRegistry::new();
        reg.register_descriptor(make_descriptor("code-review", "Review code"));
        reg.register_descriptor(make_descriptor("data-analysis", "Analyze data"));

        let catalog = reg.catalog_prompt().unwrap();
        assert!(catalog.contains("activate_skill"));
        assert!(catalog.contains("- code-review: Review code"));
        assert!(catalog.contains("- data-analysis: Analyze data"));
    }

    /// 验证激活跟踪与去重
    #[test]
    fn test_activation_tracking() {
        let mut reg = SkillRegistry::new();
        reg.register_descriptor(make_descriptor("test", "Test skill"));

        assert!(!reg.is_activated("test"));
        assert!(reg.mark_activated("test"));
        assert!(reg.is_activated("test"));
        assert!(!reg.mark_activated("test")); // 去重
        assert_eq!(reg.activated_count(), 1);
    }

    /// 验证激活状态在注册表视图间共享且重置幂等
    #[test]
    fn activation_state_is_shared_across_registry_views_and_reset_is_idempotent() {
        let mut primary = SkillRegistry::new();
        primary.register_descriptor(make_descriptor("shared", "Shared skill"));

        let mut progressive = primary.activation_view();
        progressive.register_descriptor(make_descriptor("shared", "Shared skill"));

        assert!(primary.mark_activated("shared"));
        assert!(progressive.is_activated("shared"));
        assert!(!progressive.mark_activated("shared"));

        progressive.reset_activation_state();
        assert!(!primary.is_activated("shared"));
        primary.reset_activation_state();
        assert!(primary.activated_names().is_empty());
    }

    /// 验证移除定义会清除共享激活状态
    #[test]
    fn removing_a_definition_clears_shared_activation_state() {
        let mut primary = SkillRegistry::new();
        primary.register_descriptor(make_descriptor("shared", "Shared skill"));
        let mut progressive = primary.activation_view();
        progressive.register_descriptor(make_descriptor("shared", "Shared skill"));
        assert!(primary.mark_activated("shared"));

        assert!(progressive.remove_descriptor("shared"));
        assert!(!primary.is_activated("shared"));
        assert!(primary.activated_names().is_empty());
    }

    /// 验证并发重复激活只执行一次内联命令
    #[tokio::test]
    async fn concurrent_repeated_activation_executes_inline_command_once()
    -> echo_core::error::Result<()> {
        let root = skill_test_root("single-flight");
        write_inline_skill(&root, "shared")?;
        let sandbox = Arc::new(BlockingSandbox::new());
        let descriptor = inline_descriptor(&root, "shared");
        let mut primary = SkillRegistry::new();
        primary.register_descriptor(descriptor.clone());
        primary.set_sandbox_manager(sandbox.clone());
        let mut progressive = primary.activation_view();
        progressive.register_descriptor(descriptor);
        progressive.set_sandbox_manager(sandbox.clone());
        let primary = Arc::new(primary);
        let progressive = Arc::new(progressive);

        let entered = sandbox.entered.notified();
        let first_registry = Arc::clone(&primary);
        let first = tokio::spawn(async move { first_registry.activate("shared").await });
        entered.await;
        let second_registry = Arc::clone(&progressive);
        let second = tokio::spawn(async move { second_registry.activate("shared").await });
        tokio::time::timeout(std::time::Duration::from_secs(1), async {
            loop {
                let joined = {
                    let state = primary
                        .activation
                        .shared
                        .state
                        .lock()
                        .unwrap_or_else(|error| error.into_inner());
                    state
                        .in_flight
                        .get("shared")
                        .is_some_and(|flight| flight.waiters.load(Ordering::SeqCst) >= 1)
                };
                if joined {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .map_err(|error| echo_core::error::ReactError::Other(error.to_string()))?;
        sandbox.release.notify_one();

        let first_content = first
            .await
            .map_err(|error| echo_core::error::ReactError::Other(error.to_string()))??;
        let second_content = second
            .await
            .map_err(|error| echo_core::error::ReactError::Other(error.to_string()))??;
        let repeated_content = progressive.activate("shared").await?;
        assert_eq!(sandbox.calls.load(Ordering::SeqCst), 1);
        assert_eq!(first_content.instructions, second_content.instructions);
        assert_eq!(second_content.instructions, repeated_content.instructions);
        assert!(primary.is_activated("shared"));
        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    /// 验证参数化激活对每个不同键只执行一次
    #[tokio::test]
    async fn parameterized_activation_executes_once_per_distinct_key()
    -> echo_core::error::Result<()> {
        let root = skill_test_root("parameterized-key");
        write_inline_skill(&root, "parameterized")?;
        let sandbox = Arc::new(BlockingSandbox::new());
        let mut registry = SkillRegistry::new();
        registry.register_descriptor(inline_descriptor(&root, "parameterized"));
        registry.set_sandbox_manager(sandbox.clone());

        sandbox.release.notify_one();
        let first = registry
            .activate_with_args("parameterized", &["first".to_string()], SkillSource::Local)
            .await?;
        sandbox.release.notify_one();
        let second = registry
            .activate_with_args("parameterized", &["second".to_string()], SkillSource::Local)
            .await?;
        let repeated = registry
            .activate_with_args("parameterized", &["second".to_string()], SkillSource::Local)
            .await?;

        assert_eq!(sandbox.calls.load(Ordering::SeqCst), 2);
        assert_eq!(second.instructions, repeated.instructions);
        assert_ne!(first.instructions, second.instructions);
        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    /// 验证不同激活键不能与进行中的激活重叠
    #[tokio::test]
    async fn different_activation_key_cannot_overlap_in_flight_effect()
    -> echo_core::error::Result<()> {
        let root = skill_test_root("parameterized-conflict");
        write_inline_skill(&root, "parameterized")?;
        let sandbox = Arc::new(BlockingSandbox::new());
        let mut registry = SkillRegistry::new();
        registry.register_descriptor(inline_descriptor(&root, "parameterized"));
        registry.set_sandbox_manager(sandbox.clone());
        let registry = Arc::new(registry);

        let entered = sandbox.entered.notified();
        let first_registry = Arc::clone(&registry);
        let first = tokio::spawn(async move {
            first_registry
                .activate_with_args("parameterized", &["first".to_string()], SkillSource::Local)
                .await
        });
        entered.await;
        let conflict = registry
            .activate_with_args("parameterized", &["second".to_string()], SkillSource::Local)
            .await
            .err()
            .ok_or_else(|| {
                echo_core::error::ReactError::Other(
                    "different activation key overlapped in-flight effect".to_string(),
                )
            })?;
        assert!(conflict.to_string().contains("different arguments"));
        assert_eq!(sandbox.calls.load(Ordering::SeqCst), 1);
        sandbox.release.notify_one();
        first
            .await
            .map_err(|error| echo_core::error::ReactError::Other(error.to_string()))??;
        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    /// 验证被取消的激活需重置后才能重试
    #[tokio::test]
    async fn cancelled_activation_requires_reset_before_effect_retry()
    -> echo_core::error::Result<()> {
        let root = skill_test_root("cancelled-flight");
        write_inline_skill(&root, "cancelled-skill")?;
        let sandbox = Arc::new(BlockingSandbox::new());
        let mut registry = SkillRegistry::new();
        registry.register_descriptor(inline_descriptor(&root, "cancelled-skill"));
        registry.set_sandbox_manager(sandbox.clone());
        let registry = Arc::new(registry);

        let entered = sandbox.entered.notified();
        let activating_registry = Arc::clone(&registry);
        let activation =
            tokio::spawn(async move { activating_registry.activate("cancelled-skill").await });
        entered.await;
        activation.abort();
        assert!(activation.await.is_err());

        let retry = registry
            .activate("cancelled-skill")
            .await
            .err()
            .ok_or_else(|| {
                echo_core::error::ReactError::Other(
                    "cancelled activation replayed without reset".to_string(),
                )
            })?;
        assert!(retry.to_string().contains("unknown side-effect settlement"));
        assert_eq!(sandbox.calls.load(Ordering::SeqCst), 1);

        registry.reset_activation_state();
        sandbox.release.notify_one();
        registry.activate("cancelled-skill").await?;
        assert_eq!(sandbox.calls.load(Ordering::SeqCst), 2);
        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    /// 验证重置会阻止进行中激活的发布
    #[tokio::test]
    async fn reset_fences_in_flight_activation_publication() -> echo_core::error::Result<()> {
        let root = skill_test_root("reset-fence");
        write_inline_skill(&root, "reset-skill")?;
        let sandbox = Arc::new(BlockingSandbox::new());
        let mut registry = SkillRegistry::new();
        registry.register_descriptor(inline_descriptor(&root, "reset-skill"));
        registry.set_sandbox_manager(sandbox.clone());
        let registry = Arc::new(registry);

        let entered = sandbox.entered.notified();
        let activating_registry = Arc::clone(&registry);
        let activation =
            tokio::spawn(async move { activating_registry.activate("reset-skill").await });
        entered.await;
        registry.reset_activation_state();
        sandbox.release.notify_one();

        let error = activation
            .await
            .map_err(|error| echo_core::error::ReactError::Other(error.to_string()))?
            .err()
            .ok_or_else(|| {
                echo_core::error::ReactError::Other(
                    "retired activation unexpectedly succeeded".to_string(),
                )
            })?;
        assert!(error.to_string().contains("retired before publication"));
        assert!(!registry.is_activated("reset-skill"));
        assert!(registry.get_active_sandbox_policy("reset-skill").is_none());
        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    /// 验证从另一视图移除定义会阻止进行中激活的发布
    #[tokio::test]
    async fn remove_fences_in_flight_activation_from_another_view() -> echo_core::error::Result<()>
    {
        let root = skill_test_root("remove-fence");
        write_inline_skill(&root, "removed-skill")?;
        let descriptor = inline_descriptor(&root, "removed-skill");
        let sandbox = Arc::new(BlockingSandbox::new());
        let mut primary = SkillRegistry::new();
        primary.register_descriptor(descriptor.clone());
        let mut progressive = primary.activation_view();
        progressive.register_descriptor(descriptor);
        progressive.set_sandbox_manager(sandbox.clone());
        let progressive = Arc::new(progressive);

        let entered = sandbox.entered.notified();
        let activating_registry = Arc::clone(&progressive);
        let activation =
            tokio::spawn(async move { activating_registry.activate("removed-skill").await });
        entered.await;
        assert!(primary.remove_descriptor("removed-skill"));
        sandbox.release.notify_one();

        let error = activation
            .await
            .map_err(|error| echo_core::error::ReactError::Other(error.to_string()))?
            .err()
            .ok_or_else(|| {
                echo_core::error::ReactError::Other(
                    "removed activation unexpectedly succeeded".to_string(),
                )
            })?;
        assert!(error.to_string().contains("retired before publication"));
        assert!(!progressive.is_activated("removed-skill"));
        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    /// 验证代码型 skill 记录
    #[test]
    fn test_code_skills() {
        let mut reg = SkillRegistry::new();
        reg.record_code_skill(SkillInfo {
            name: "calculator".into(),
            description: "Math operations".into(),
            tool_names: vec!["add".into(), "subtract".into()],
            has_prompt_injection: true,
        });

        assert!(reg.has_code_skill("calculator"));
        assert!(reg.is_installed("calculator"));
        assert_eq!(reg.count(), 1);
    }

    /// 验证可用名称按字母排序
    #[test]
    fn test_available_names() {
        let mut reg = SkillRegistry::new();
        reg.register_descriptor(make_descriptor("b-skill", "B"));
        reg.register_descriptor(make_descriptor("a-skill", "A"));

        let names = reg.available_names();
        assert_eq!(names, vec!["a-skill", "b-skill"]);
    }

    /// 验证代码型与文件型 skill 混合统计
    #[test]
    fn test_mixed_skills() {
        let mut reg = SkillRegistry::new();
        reg.register_descriptor(make_descriptor("file-skill", "File-based"));
        reg.record_code_skill(SkillInfo {
            name: "code-skill".into(),
            description: "Code-based".into(),
            tool_names: vec![],
            has_prompt_injection: false,
        });

        assert_eq!(reg.count(), 2);
        assert!(reg.is_installed("file-skill"));
        assert!(reg.is_installed("code-skill"));
        assert!(!reg.is_installed("missing"));
    }

    /// 构造带 allowed_tools 的测试描述符
    fn make_descriptor_with_tools(name: &str, allowed: Vec<&str>) -> SkillDescriptor {
        let mut desc = make_descriptor(name, &format!("{} skill", name));
        desc.allowed_tools = allowed.into_iter().map(String::from).collect();
        desc
    }

    /// 验证无激活 skill 时工具不受限
    #[test]
    fn test_active_skill_allowed_tools_empty_when_no_skills_activated() {
        let reg = SkillRegistry::new();
        assert!(
            reg.active_skill_allowed_tools().is_none(),
            "No activated skills → unrestricted"
        );
    }

    /// 验证激活 skill 无工具限制时不受限
    #[test]
    fn test_active_skill_allowed_tools_empty_when_no_restrictions() {
        let mut reg = SkillRegistry::new();
        reg.register_descriptor(make_descriptor("open-skill", "No tool restrictions"));
        reg.mark_activated("open-skill");

        assert!(
            reg.active_skill_allowed_tools().is_none(),
            "Activated skill with empty allowed_tools → unrestricted"
        );
    }

    /// 验证激活 skill 声明白名单时返回白名单
    #[test]
    fn test_active_skill_allowed_tools_returns_whitelist() {
        let mut reg = SkillRegistry::new();

        // 医学 skill：只允许研究类工具，不允许 shell
        reg.register_descriptor(make_descriptor_with_tools(
            "evidence-medicine",
            vec!["Read", "Write", "Edit", "WebSearch", "PubMedSearch"],
        ));
        reg.mark_activated("evidence-medicine");

        let allowed = reg
            .active_skill_allowed_tools()
            .expect("Should return Some when skill restricts tools");

        assert!(allowed.contains("Read"));
        assert!(allowed.contains("PubMedSearch"));
        assert!(!allowed.contains("Bash"));
        assert!(!allowed.contains("Shell"));
    }

    /// 验证多个激活 skill 的白名单取并集
    #[test]
    fn test_active_skill_allowed_tools_union_of_multiple_skills() {
        let mut reg = SkillRegistry::new();

        reg.register_descriptor(make_descriptor_with_tools(
            "coding",
            vec!["Bash(*)", "Read", "Write", "Edit", "Glob", "Grep"],
        ));
        reg.register_descriptor(make_descriptor_with_tools(
            "git-workflow",
            vec!["Bash(git:*)", "Read", "Glob"],
        ));

        // 激活两个 skill
        reg.mark_activated("coding");
        reg.mark_activated("git-workflow");

        let allowed = reg
            .active_skill_allowed_tools()
            .expect("Should return union of allowed tools");

        // 应包含两个 skill 的工具
        assert!(allowed.contains("Bash(*)"));
        assert!(allowed.contains("Bash(git:*)"));
        assert!(allowed.contains("Read"));
        assert!(allowed.contains("Grep"));
    }

    /// 验证工具匹配拒绝不在白名单中的工具
    #[test]
    fn test_tool_matcher_rejects_disallowed_tool() {
        // 模拟 stream_channel.rs 中的权限检查
        let allowed: HashSet<String> = ["Read", "Write", "Edit", "WebSearch", "PubMedSearch"]
            .iter()
            .map(|s| s.to_string())
            .collect();

        let tool_name = "Bash";
        let permitted = allowed
            .iter()
            .any(|matcher| crate::skills::external::types::tool_matcher(matcher, tool_name));

        assert!(
            !permitted,
            "Bash should NOT be permitted by evidence-medicine's allowed-tools"
        );
    }

    /// 验证工具匹配接受白名单中的工具
    #[test]
    fn test_tool_matcher_accepts_allowed_tool() {
        let allowed: HashSet<String> = ["Read", "Write", "Edit", "WebSearch", "PubMedSearch"]
            .iter()
            .map(|s| s.to_string())
            .collect();

        let tool_name = "PubMedSearch";
        let permitted = allowed
            .iter()
            .any(|matcher| crate::skills::external::types::tool_matcher(matcher, tool_name));

        assert!(
            permitted,
            "PubMedSearch should be permitted by evidence-medicine's allowed-tools"
        );
    }

    /// 验证工具匹配的通配模式
    #[test]
    fn test_tool_matcher_glob_pattern() {
        let allowed: HashSet<String> = ["Bash(*)", "Read"].iter().map(|s| s.to_string()).collect();

        // Bash(*) 应匹配 Bash(git:status)
        let permitted = allowed.iter().any(|matcher| {
            crate::skills::external::types::tool_matcher(matcher, "Bash(git:status)")
        });
        assert!(permitted, "Bash(*) should match Bash(git:status)");

        // Bash(*) 不应匹配 Read
        let is_read = allowed
            .iter()
            .any(|matcher| crate::skills::external::types::tool_matcher(matcher, "Read"));
        assert!(is_read, "Read should be directly permitted");
    }
}
