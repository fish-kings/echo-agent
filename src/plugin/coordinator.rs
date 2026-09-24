//! 对插件的期望状态与实际状态进行序列化的宿主级协调。

use super::{
    PluginIntegrator, PluginLifecycle, PluginLifecycleManager, PluginPublicationTarget,
    PluginRegistry, PluginWiringError, PluginWiringResult, PreparedPluginSet,
};
use crate::agent::react::ReactAgent;
use crate::skills::hooks::{HookContext, HookEvent, HookRegistry};
use std::collections::BTreeSet;
use std::fmt;
use std::sync::Arc;

/// 协调器回执所代表的宿主操作。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PluginOperationKind {
    /// 启动：扫描持久化插件位置后收敛出一个全新的 Agent。
    Startup,
    /// 在不改变期望状态的前提下进行对账。
    Reconcile,
    /// 启用某个插件。
    Enable {
        /// 要启用的插件 id。
        plugin_id: String,
    },
    /// 重新加载全部插件。
    Reload,
    /// 禁用某个插件。
    Disable {
        /// 要禁用的插件 id。
        plugin_id: String,
    },
    /// 卸载某个插件。
    Uninstall {
        /// 要卸载的插件 id。
        plugin_id: String,
        /// 是否保留其数据。
        keep_data: bool,
    },
    /// 关闭：撤回进程内效果，同时保留持久化的启用意图。
    Shutdown,
}

/// 一个操作在推进之前必须先结算完成的下一阶段。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PluginOperationPhase {
    /// 期望状态已提交。
    DesiredCommitted,
    /// 正在准备（解析依赖、校验）。
    Preparation,
    /// 正在撤回回调。
    CallbackWithdrawal,
    /// 正在撤回接线（wiring）。
    WiringWithdrawal,
    /// 正在发布接线。
    WiringPublication,
    /// 正在激活回调。
    CallbackActivation,
    /// 正在发出事件。
    EventEmission,
    /// 已提交完成。
    Committed,
}

/// 运行时是否已收敛到持久化注册表所表达的意图。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PluginRuntimeStatus {
    /// 实际状态仍处于待定（pending）。
    ActualPending,
    /// 已收敛完成。
    Converged,
}

/// 一次序列化插件转换的、进程内稳定的回执。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PluginOperationReceipt {
    /// 该操作的 id。
    operation_id: u64,
    /// 该操作的种类。
    kind: PluginOperationKind,
    /// 该操作对应的期望状态版本号。
    desired_revision: u64,
    /// 当前所处阶段。
    phase: PluginOperationPhase,
    /// 当前运行时状态。
    status: PluginRuntimeStatus,
    /// 已发布接线对应的代数（generation），若有。
    actual_generation: Option<u64>,
    /// 本次操作已经发出的 PluginDisabled 尝试。
    disabled_event_attempts: Vec<String>,
    /// 本次操作已经发出的 PluginLoaded 尝试。
    loaded_event_attempts: Vec<String>,
    /// 最近一次记录的错误信息。
    last_error: Option<String>,
}

impl PluginOperationReceipt {
    /// 该操作的 id。
    pub fn operation_id(&self) -> u64 {
        self.operation_id
    }

    /// 该操作的种类。
    pub fn kind(&self) -> &PluginOperationKind {
        &self.kind
    }

    /// 该操作对应的期望状态版本号。
    pub fn desired_revision(&self) -> u64 {
        self.desired_revision
    }

    /// 当前所处阶段。
    pub fn phase(&self) -> PluginOperationPhase {
        self.phase
    }

    /// 当前运行时状态。
    pub fn status(&self) -> PluginRuntimeStatus {
        self.status
    }

    /// 已发布接线对应的代数（generation），若有。
    pub fn actual_generation(&self) -> Option<u64> {
        self.actual_generation
    }

    /// 本次操作已经发出的 PluginDisabled 尝试。
    pub fn disabled_event_attempts(&self) -> &[String] {
        &self.disabled_event_attempts
    }

    /// 本次操作已经发出的 PluginLoaded 尝试。
    pub fn loaded_event_attempts(&self) -> &[String] {
        &self.loaded_event_attempts
    }

    /// 最近一次记录的错误信息。
    pub fn last_error(&self) -> Option<&str> {
        self.last_error.as_deref()
    }
}

/// 协调器拒绝，或一个可重试的实际状态缺口（gap）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PluginCoordinatorError {
    /// 已有一个操作正在等待中。
    OperationPending(Box<PluginOperationReceipt>),
    /// 当前没有等待中的操作。
    NoPendingOperation,
    /// 操作 id 已耗尽。
    OperationIdExhausted,
    /// 协调器被绑定到了另一个 Agent 上。
    WrongAgentTarget {
        /// 当前绑定的代数（generation），若有。
        active_generation: Option<u64>,
    },
    /// 期望状态的变更失败。
    DesiredMutation {
        /// 失败所属的操作种类。
        kind: PluginOperationKind,
        /// 错误说明。
        message: String,
    },
    /// 期望状态已提交，但实际状态仍处于待定。
    ActualPending {
        /// 当前的操作回执。
        receipt: Box<PluginOperationReceipt>,
        /// 错误说明。
        message: String,
    },
}

impl PluginCoordinatorError {
    /// 若该错误携带一个操作回执，则返回它。
    pub fn receipt(&self) -> Option<&PluginOperationReceipt> {
        match self {
            Self::OperationPending(receipt) | Self::ActualPending { receipt, .. } => {
                Some(receipt.as_ref())
            }
            Self::NoPendingOperation
            | Self::OperationIdExhausted
            | Self::WrongAgentTarget { .. }
            | Self::DesiredMutation { .. } => None,
        }
    }
}

impl fmt::Display for PluginCoordinatorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OperationPending(receipt) => write!(
                formatter,
                "plugin operation {} is pending at {:?}",
                receipt.operation_id, receipt.phase
            ),
            Self::NoPendingOperation => formatter.write_str("no plugin operation is pending"),
            Self::OperationIdExhausted => {
                formatter.write_str("plugin operation identity exhausted")
            }
            Self::WrongAgentTarget { active_generation } => match active_generation {
                Some(generation) => write!(
                    formatter,
                    "plugin coordinator is bound to another Agent at generation {generation}"
                ),
                None => formatter.write_str("plugin coordinator is bound to another Agent"),
            },
            Self::DesiredMutation { kind, message } => {
                write!(
                    formatter,
                    "plugin {kind:?} desired-state mutation failed: {message}"
                )
            }
            Self::ActualPending { receipt, message } => write!(
                formatter,
                "plugin operation {} has desired revision {} committed but actual state is pending at {:?}: {message}",
                receipt.operation_id, receipt.desired_revision, receipt.phase
            ),
        }
    }
}

impl std::error::Error for PluginCoordinatorError {}

/// 一次尚未结算的插件操作的进行中状态。
struct PendingOperation {
    /// 该操作的回执（会随阶段推进而更新）。
    receipt: PluginOperationReceipt,
    /// 操作开始前的活跃插件 id 列表。
    previous_ids: Vec<String>,
    /// 本次操作解析出的期望插件 id 列表。
    desired_ids: Vec<String>,
    /// 已校验完成的准备结果。
    prepared: Option<Arc<PreparedPluginSet>>,
    /// 撤回接线之后拍摄的 hook 注册表快照。
    disabled_hooks: Option<HookRegistry>,
    /// 需要发出 PluginDisabled 事件的插件 id（按撤回顺序）。
    disabled_ids: Vec<String>,
    /// 需要发出 PluginLoaded 事件的插件 id。
    loaded_ids: Vec<String>,
    /// disabled_ids 中已处理到的位置。
    disabled_cursor: usize,
    /// loaded_ids 中已处理到的位置。
    loaded_cursor: usize,
    /// 上一次回调激活是否失败、需要重试前清理。
    callback_retry_required: bool,
}

/**
协调插件的持久化意图、Agent 发布与回调。

各方法都要求 `&mut self`，使该值本身成为转换操作的准入令牌。
在多个任务间共享它的宿主应当把协调器与其 Agent 放在同一个
异步互斥锁之后。
*/
pub struct PluginCoordinator {
    /// 插件注册表（持久化的期望状态）。
    registry: PluginRegistry,
    /// 插件集成器。
    integrator: PluginIntegrator,
    /// 插件生命周期回调管理器。
    lifecycle: PluginLifecycleManager,
    /// 当前绑定的发布目标（Agent）。
    publication_target: Option<PluginPublicationTarget>,
    /// 当前已发布接线的回执。
    active_receipt: Option<PluginWiringResult>,
    /// 当前已激活的插件 id 列表。
    active_ids: Vec<String>,
    /// 当前已收敛到的期望状态版本号。
    active_revision: Option<u64>,
    /// 正在进行中的操作（若有）。
    pending: Option<PendingOperation>,
    /// 最近一次完成的操作回执。
    last_receipt: Option<PluginOperationReceipt>,
    /// 下一个待分配的操作 id。
    next_operation_id: u64,
}

impl PluginCoordinator {
    /// 使用给定的注册表与集成器创建一个新的协调器。
    pub fn new(registry: PluginRegistry, integrator: PluginIntegrator) -> Self {
        Self {
            registry,
            integrator,
            lifecycle: PluginLifecycleManager::new(),
            publication_target: None,
            active_receipt: None,
            active_ids: Vec::new(),
            active_revision: None,
            pending: None,
            last_receipt: None,
            next_operation_id: 1,
        }
    }

    /// 该协调器持有的插件注册表。
    pub fn registry(&self) -> &PluginRegistry {
        &self.registry
    }

    /// 注册某个插件的生命周期回调。若有操作正在进行中则失败。
    pub fn register_lifecycle(
        &mut self,
        plugin_id: impl Into<String>,
        callbacks: Arc<dyn PluginLifecycle>,
    ) -> Result<(), String> {
        if let Some(pending) = &self.pending {
            return Err(format!(
                "Cannot register lifecycle callbacks while plugin operation {} is pending",
                pending.receipt.operation_id
            ));
        }
        self.lifecycle.register(plugin_id, callbacks)?;
        self.active_revision = None;
        Ok(())
    }

    /// 最近一次完成的操作回执。
    pub fn last_receipt(&self) -> Option<&PluginOperationReceipt> {
        self.last_receipt.as_ref()
    }

    /// 当前正在进行中的操作回执（若有）。
    pub fn pending_receipt(&self) -> Option<&PluginOperationReceipt> {
        self.pending.as_ref().map(|pending| &pending.receipt)
    }

    /// 当前已发布接线的代数（generation）。
    pub fn active_generation(&self) -> Option<u64> {
        self.active_receipt
            .as_ref()
            .map(PluginWiringResult::generation)
    }

    /// 扫描持久化插件位置，然后收敛出一个全新的 Agent。
    pub async fn startup(
        &mut self,
        agent: &mut ReactAgent,
    ) -> Result<PluginOperationReceipt, PluginCoordinatorError> {
        self.ensure_idle()?;
        self.ensure_agent_target(agent)?;
        self.registry
            .scan_all()
            .map_err(|error| PluginCoordinatorError::DesiredMutation {
                kind: PluginOperationKind::Startup,
                message: error.to_string(),
            })?;
        self.integrator.invalidate(&self.registry);
        self.begin(PluginOperationKind::Startup)?;
        self.run_pending(agent).await
    }

    /// 在不改变当前注册表意图的前提下进行对账。
    pub async fn reconcile(
        &mut self,
        agent: &mut ReactAgent,
    ) -> Result<PluginOperationReceipt, PluginCoordinatorError> {
        self.ensure_idle()?;
        self.ensure_agent_target(agent)?;
        if self.is_converged(agent) {
            return self.commit_noop(PluginOperationKind::Reconcile);
        }
        self.integrator.invalidate(&self.registry);
        self.begin(PluginOperationKind::Reconcile)?;
        self.run_pending(agent).await
    }

    /// 启用一个插件。
    pub async fn enable(
        &mut self,
        agent: &mut ReactAgent,
        plugin_id: &str,
    ) -> Result<PluginOperationReceipt, PluginCoordinatorError> {
        self.ensure_idle()?;
        self.ensure_agent_target(agent)?;
        let kind = PluginOperationKind::Enable {
            plugin_id: plugin_id.to_string(),
        };
        self.registry.enable(plugin_id).map_err(|message| {
            PluginCoordinatorError::DesiredMutation {
                kind: kind.clone(),
                message,
            }
        })?;
        if self.is_converged(agent) {
            return self.commit_noop(kind);
        }
        self.integrator.invalidate(&self.registry);
        self.begin(kind)?;
        self.run_pending(agent).await
    }

    /// 重新加载全部插件。
    pub async fn reload(
        &mut self,
        agent: &mut ReactAgent,
    ) -> Result<PluginOperationReceipt, PluginCoordinatorError> {
        self.ensure_idle()?;
        self.ensure_agent_target(agent)?;
        self.integrator.invalidate(&self.registry);
        self.begin(PluginOperationKind::Reload)?;
        self.run_pending(agent).await
    }

    /// 禁用一个插件。
    pub async fn disable(
        &mut self,
        agent: &mut ReactAgent,
        plugin_id: &str,
    ) -> Result<PluginOperationReceipt, PluginCoordinatorError> {
        self.ensure_idle()?;
        self.ensure_agent_target(agent)?;
        let kind = PluginOperationKind::Disable {
            plugin_id: plugin_id.to_string(),
        };
        self.registry.disable(plugin_id).map_err(|message| {
            PluginCoordinatorError::DesiredMutation {
                kind: kind.clone(),
                message,
            }
        })?;
        if self.is_converged(agent) {
            return self.commit_noop(kind);
        }
        self.integrator.invalidate(&self.registry);
        self.begin(kind)?;
        self.run_pending(agent).await
    }

    /// 卸载一个插件。
    pub async fn uninstall(
        &mut self,
        agent: &mut ReactAgent,
        plugin_id: &str,
        keep_data: bool,
    ) -> Result<PluginOperationReceipt, PluginCoordinatorError> {
        self.ensure_idle()?;
        self.ensure_agent_target(agent)?;
        let kind = PluginOperationKind::Uninstall {
            plugin_id: plugin_id.to_string(),
            keep_data,
        };
        self.registry
            .uninstall(plugin_id, keep_data)
            .map_err(|message| PluginCoordinatorError::DesiredMutation {
                kind: kind.clone(),
                message,
            })?;
        self.integrator.invalidate(&self.registry);
        self.begin(kind)?;
        self.run_pending(agent).await
    }

    /// 撤回进程内效果，同时保留持久化的启用意图。
    pub async fn shutdown(
        &mut self,
        agent: &mut ReactAgent,
    ) -> Result<PluginOperationReceipt, PluginCoordinatorError> {
        self.ensure_idle()?;
        self.ensure_agent_target(agent)?;
        self.begin(PluginOperationKind::Shutdown)?;
        self.run_pending(agent).await
    }

    /// 从失败或被取消的操作所停留的确切阶段处恢复执行。
    pub async fn retry(
        &mut self,
        agent: &mut ReactAgent,
    ) -> Result<PluginOperationReceipt, PluginCoordinatorError> {
        if self.pending.is_none() {
            return Err(PluginCoordinatorError::NoPendingOperation);
        }
        self.ensure_agent_target(agent)?;
        self.run_pending(agent).await
    }

    /// 若已有操作正在进行中，则拒绝新的操作。
    fn ensure_idle(&self) -> Result<(), PluginCoordinatorError> {
        if let Some(pending) = &self.pending {
            Err(PluginCoordinatorError::OperationPending(Box::new(
                pending.receipt.clone(),
            )))
        } else {
            Ok(())
        }
    }

    /// 确认该协调器当前绑定的目标就是给定的 Agent。
    fn ensure_agent_target(&self, agent: &ReactAgent) -> Result<(), PluginCoordinatorError> {
        if self
            .publication_target
            .as_ref()
            .is_some_and(|target| !target.is_for_agent(agent))
        {
            Err(PluginCoordinatorError::WrongAgentTarget {
                active_generation: self.active_generation(),
            })
        } else {
            Ok(())
        }
    }

    /// 开始一个新的操作：分配操作 id 并初始化其进行中状态。
    fn begin(&mut self, kind: PluginOperationKind) -> Result<(), PluginCoordinatorError> {
        let operation_id = self.next_operation_id;
        self.next_operation_id = self
            .next_operation_id
            .checked_add(1)
            .ok_or(PluginCoordinatorError::OperationIdExhausted)?;
        let previous_ids = self.active_ids.clone();
        let is_shutdown = matches!(kind, PluginOperationKind::Shutdown);
        let desired_ids = Vec::new();
        let disabled_ids = if is_shutdown {
            previous_ids.iter().rev().cloned().collect()
        } else {
            Vec::new()
        };
        let loaded_ids = Vec::new();
        self.pending = Some(PendingOperation {
            receipt: PluginOperationReceipt {
                operation_id,
                kind,
                desired_revision: self.registry.revision(),
                phase: PluginOperationPhase::DesiredCommitted,
                status: PluginRuntimeStatus::ActualPending,
                actual_generation: None,
                disabled_event_attempts: Vec::new(),
                loaded_event_attempts: Vec::new(),
                last_error: None,
            },
            previous_ids,
            desired_ids,
            prepared: None,
            disabled_hooks: None,
            disabled_ids,
            loaded_ids,
            disabled_cursor: 0,
            loaded_cursor: 0,
            callback_retry_required: false,
        });
        Ok(())
    }

    /// 当前运行时是否已经收敛到注册表的期望状态、且绑定到给定 Agent。
    fn is_converged(&self, agent: &ReactAgent) -> bool {
        self.active_receipt.is_some()
            && self.active_revision == Some(self.registry.revision())
            && self
                .registry
                .resolve_enabled_dependencies()
                .is_ok_and(|desired_ids| self.active_ids == desired_ids)
            && self
                .publication_target
                .as_ref()
                .is_some_and(|target| target.is_for_agent(agent))
    }

    /// 提交一次无需变更（no-op）的操作：直接产出一个已收敛的回执。
    fn commit_noop(
        &mut self,
        kind: PluginOperationKind,
    ) -> Result<PluginOperationReceipt, PluginCoordinatorError> {
        let operation_id = self.next_operation_id;
        self.next_operation_id = self
            .next_operation_id
            .checked_add(1)
            .ok_or(PluginCoordinatorError::OperationIdExhausted)?;
        let receipt = PluginOperationReceipt {
            operation_id,
            kind,
            desired_revision: self.registry.revision(),
            phase: PluginOperationPhase::Committed,
            status: PluginRuntimeStatus::Converged,
            actual_generation: self.active_generation(),
            disabled_event_attempts: Vec::new(),
            loaded_event_attempts: Vec::new(),
            last_error: None,
        };
        self.last_receipt = Some(receipt.clone());
        Ok(receipt)
    }

    /// 驱动当前进行中的操作，依次推进各阶段直至提交完成。
    async fn run_pending(
        &mut self,
        agent: &mut ReactAgent,
    ) -> Result<PluginOperationReceipt, PluginCoordinatorError> {
        if self.phase()? == PluginOperationPhase::DesiredCommitted {
            if self.is_shutdown()? {
                self.set_phase(PluginOperationPhase::CallbackWithdrawal)?;
            } else {
                self.set_phase(PluginOperationPhase::Preparation)?;
            }
        }
        if self.phase()? == PluginOperationPhase::Preparation {
            self.prepare_pending().await?;
            self.set_phase(PluginOperationPhase::CallbackWithdrawal)?;
        }
        if self.phase()? == PluginOperationPhase::CallbackWithdrawal {
            self.withdraw_callbacks()?;
            self.set_phase(PluginOperationPhase::WiringWithdrawal)?;
        }
        if self.phase()? == PluginOperationPhase::WiringWithdrawal {
            self.withdraw_wiring(agent).await?;
            let needs_disabled_snapshot = self
                .pending
                .as_ref()
                .is_some_and(|pending| !pending.disabled_ids.is_empty());
            if needs_disabled_snapshot {
                let snapshot = agent.hook_registry().read().await.clone();
                if let Some(pending) = &mut self.pending {
                    pending.disabled_hooks = Some(snapshot);
                }
            }
            if self.is_shutdown()? {
                self.set_phase(PluginOperationPhase::EventEmission)?;
            } else {
                self.set_phase(PluginOperationPhase::WiringPublication)?;
            }
        }
        if self.phase()? == PluginOperationPhase::WiringPublication {
            self.publish_wiring(agent).await?;
            self.set_phase(PluginOperationPhase::CallbackActivation)?;
        }
        if self.phase()? == PluginOperationPhase::CallbackActivation {
            self.activate_callbacks()?;
            self.set_phase(PluginOperationPhase::EventEmission)?;
        }
        if self.phase()? == PluginOperationPhase::EventEmission {
            self.emit_events(agent).await?;
            self.set_phase(PluginOperationPhase::Committed)?;
        }
        self.commit_pending()
    }

    /// 当前进行中操作所处的阶段。
    fn phase(&self) -> Result<PluginOperationPhase, PluginCoordinatorError> {
        self.pending
            .as_ref()
            .map(|pending| pending.receipt.phase)
            .ok_or(PluginCoordinatorError::NoPendingOperation)
    }

    /// 设置当前进行中操作所处的阶段。
    fn set_phase(&mut self, phase: PluginOperationPhase) -> Result<(), PluginCoordinatorError> {
        let pending = self
            .pending
            .as_mut()
            .ok_or(PluginCoordinatorError::NoPendingOperation)?;
        pending.receipt.phase = phase;
        Ok(())
    }

    /// 当前进行中的操作是否为 Shutdown。
    fn is_shutdown(&self) -> Result<bool, PluginCoordinatorError> {
        self.pending
            .as_ref()
            .map(|pending| matches!(pending.receipt.kind, PluginOperationKind::Shutdown))
            .ok_or(PluginCoordinatorError::NoPendingOperation)
    }

    /// 刷新注册表并解析期望依赖，为当前进行中的操作准备一份已校验的插件集合。
    async fn prepare_pending(&mut self) -> Result<(), PluginCoordinatorError> {
        if let Err(error) = self.registry.refresh_current_view() {
            self.clear_failed_preparation();
            return Err(self.actual_pending(format!(
                "plugin registry refresh failed before withdrawal: {error}"
            )));
        }
        let desired_ids = match self.registry.resolve_enabled_dependencies() {
            Ok(desired_ids) => desired_ids,
            Err(error) => {
                self.clear_failed_preparation();
                return Err(self.actual_pending(format!(
                    "plugin dependency resolution failed before withdrawal: {error}"
                )));
            }
        };
        self.integrator.invalidate(&self.registry);
        let prepared = self.integrator.prepare(&mut self.registry).await;
        if !prepared.is_applicable() {
            self.clear_failed_preparation();
            return Err(self.actual_pending(format!(
                "prepared plugin generation {} is not applicable",
                prepared.generation()
            )));
        }
        let previous_ids = self
            .pending
            .as_ref()
            .map(|pending| pending.previous_ids.clone())
            .ok_or(PluginCoordinatorError::NoPendingOperation)?;
        let desired = desired_ids.iter().cloned().collect::<BTreeSet<_>>();
        let is_reload = self
            .pending
            .as_ref()
            .is_some_and(|pending| matches!(pending.receipt.kind, PluginOperationKind::Reload));
        let disabled_ids = if is_reload {
            previous_ids.iter().rev().cloned().collect()
        } else {
            previous_ids
                .iter()
                .rev()
                .filter(|plugin_id| !desired.contains(*plugin_id))
                .cloned()
                .collect()
        };
        if let Some(pending) = &mut self.pending {
            pending.receipt.desired_revision = self.registry.revision();
            pending.desired_ids = desired_ids.clone();
            pending.prepared = Some(prepared);
            pending.disabled_ids = disabled_ids;
            pending.loaded_ids = desired_ids;
        }
        Ok(())
    }

    /// 清理一次失败的准备工作产生的中间状态。
    fn clear_failed_preparation(&mut self) {
        self.integrator.invalidate(&self.registry);
        if let Some(pending) = &mut self.pending {
            pending.prepared = None;
            pending.desired_ids.clear();
            pending.disabled_ids.clear();
            pending.loaded_ids.clear();
        }
    }

    /// 按插件依赖图的逆序撤回回调（deactivate/unregister/shutdown）。
    fn withdraw_callbacks(&mut self) -> Result<(), PluginCoordinatorError> {
        let kind = self
            .pending
            .as_ref()
            .map(|pending| pending.receipt.kind.clone())
            .ok_or(PluginCoordinatorError::NoPendingOperation)?;
        let active_ids = self.active_ids.clone();
        let mut errors = self
            .lifecycle
            .deactivate_in_order(active_ids.iter().rev().map(String::as_str));
        match kind {
            PluginOperationKind::Uninstall { plugin_id, .. } => {
                if let Err(error) = self.lifecycle.unregister(&plugin_id) {
                    errors.push(error);
                }
            }
            PluginOperationKind::Shutdown => {
                let mut installed_ids = self.registry.resolve_dependencies().unwrap_or_else(|_| {
                    self.registry
                        .list()
                        .into_iter()
                        .map(|entry| entry.manifest.name.clone())
                        .collect()
                });
                installed_ids.reverse();
                errors.extend(
                    self.lifecycle
                        .unregister_in_order(installed_ids.iter().map(String::as_str)),
                );
                if errors.is_empty() {
                    /* 没有对应已安装包的注册项在依赖图中没有位置；
                    只在图所拥有的回调都结算完之后再结算这些注册项。 */
                    errors.extend(self.lifecycle.shutdown());
                }
            }
            PluginOperationKind::Startup
            | PluginOperationKind::Reconcile
            | PluginOperationKind::Enable { .. }
            | PluginOperationKind::Reload
            | PluginOperationKind::Disable { .. } => {}
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(self.actual_pending(errors.join("; ")))
        }
    }

    /// 若存在已发布的接线，则将其回滚。
    async fn withdraw_wiring(
        &mut self,
        agent: &mut ReactAgent,
    ) -> Result<(), PluginCoordinatorError> {
        let Some(receipt) = self.active_receipt.clone() else {
            return Ok(());
        };
        let target = self.target(agent);
        match target.rollback(agent, &receipt).await {
            Ok(()) => {
                self.active_receipt = None;
                Ok(())
            }
            Err(error) => Err(self.actual_pending(error.to_string())),
        }
    }

    /// 发布已准备好的插件集合的接线，必要时先回滚待清理的旧接线。
    async fn publish_wiring(
        &mut self,
        agent: &mut ReactAgent,
    ) -> Result<(), PluginCoordinatorError> {
        let target = self.target(agent);
        if let Some(cleanup) = target.pending_cleanup_receipt().await
            && let Err(error) = target.rollback(agent, &cleanup).await
        {
            return Err(self.actual_pending(error.to_string()));
        }
        let prepared = match self
            .pending
            .as_ref()
            .and_then(|pending| pending.prepared.clone())
        {
            Some(prepared) => prepared,
            None => {
                return Err(self.actual_pending(
                    "validated plugin preparation is missing before publication".to_string(),
                ));
            }
        };
        match target.wire_prepared(agent, &prepared).await {
            Ok(receipt) => {
                if let Some(pending) = &mut self.pending {
                    pending.receipt.actual_generation = Some(receipt.generation());
                }
                self.active_receipt = Some(receipt);
                Ok(())
            }
            Err(error @ PluginWiringError::InvalidPreparedSet { .. }) => {
                self.set_phase(PluginOperationPhase::Preparation)?;
                self.clear_failed_preparation();
                Err(self.actual_pending(wiring_error_message(error)))
            }
            Err(error) => Err(self.actual_pending(wiring_error_message(error))),
        }
    }

    /// 按依赖顺序激活期望插件的回调；若上一次尝试失败过，先重置状态再重试。
    fn activate_callbacks(&mut self) -> Result<(), PluginCoordinatorError> {
        let desired_ids = self
            .pending
            .as_ref()
            .map(|pending| pending.desired_ids.clone())
            .ok_or(PluginCoordinatorError::NoPendingOperation)?;
        let retry_required = self
            .pending
            .as_ref()
            .is_some_and(|pending| pending.callback_retry_required);
        if retry_required {
            let mut cleanup_errors = Vec::new();
            for plugin_id in desired_ids.iter().rev() {
                if let Err(error) = self.lifecycle.reset_for_retry(plugin_id) {
                    cleanup_errors.push(error);
                }
            }
            if !cleanup_errors.is_empty() {
                return Err(self.actual_pending(cleanup_errors.join("; ")));
            }
            if let Some(pending) = &mut self.pending {
                pending.callback_retry_required = false;
            }
        }
        let errors = self
            .lifecycle
            .activate_in_order(desired_ids.iter().map(String::as_str));
        if errors.is_empty() {
            Ok(())
        } else {
            if let Some(pending) = &mut self.pending {
                pending.callback_retry_required = true;
            }
            Err(self.actual_pending(errors.join("; ")))
        }
    }

    /// 依次发出 PluginDisabled 与 PluginLoaded 生命周期事件。
    async fn emit_events(&mut self, agent: &mut ReactAgent) -> Result<(), PluginCoordinatorError> {
        loop {
            let next = {
                let pending = self
                    .pending
                    .as_mut()
                    .ok_or(PluginCoordinatorError::NoPendingOperation)?;
                if let Some(plugin_id) = pending.disabled_ids.get(pending.disabled_cursor).cloned()
                {
                    pending.disabled_cursor = pending.disabled_cursor.saturating_add(1);
                    pending
                        .receipt
                        .disabled_event_attempts
                        .push(plugin_id.clone());
                    Some((
                        HookEvent::PluginDisabled,
                        plugin_id,
                        pending.disabled_hooks.clone(),
                    ))
                } else if let Some(plugin_id) =
                    pending.loaded_ids.get(pending.loaded_cursor).cloned()
                {
                    pending.loaded_cursor = pending.loaded_cursor.saturating_add(1);
                    pending
                        .receipt
                        .loaded_event_attempts
                        .push(plugin_id.clone());
                    Some((HookEvent::PluginLoaded, plugin_id, None))
                } else {
                    None
                }
            };
            let Some((event, plugin_id, registry_snapshot)) = next else {
                return Ok(());
            };
            let registry = match registry_snapshot {
                Some(registry) => registry,
                None => agent.hook_registry().read().await.clone(),
            };
            let context = HookContext::for_lifecycle(
                event,
                &plugin_id,
                agent.config().get_session_id().unwrap_or_default(),
                agent.config().get_agent_name(),
            );
            let _ = registry.run_lifecycle_hooks(&context).await;
        }
    }

    /// 惰性获取并缓存该协调器绑定的发布目标。
    fn target(&mut self, agent: &ReactAgent) -> PluginPublicationTarget {
        match &self.publication_target {
            Some(target) => target.clone(),
            None => {
                let target = self.integrator.publication_target(agent);
                self.publication_target = Some(target.clone());
                target
            }
        }
    }

    /// 将进行中的操作标记为“实际状态待定”，记录错误并构造对应的错误值。
    fn actual_pending(&mut self, message: String) -> PluginCoordinatorError {
        let receipt = match &mut self.pending {
            Some(pending) => {
                pending.receipt.status = PluginRuntimeStatus::ActualPending;
                pending.receipt.last_error = Some(message.clone());
                pending.receipt.clone()
            }
            None => return PluginCoordinatorError::NoPendingOperation,
        };
        PluginCoordinatorError::ActualPending {
            receipt: Box::new(receipt),
            message,
        }
    }

    /// 将当前进行中的操作提交为已收敛状态，并更新协调器的活跃状态。
    fn commit_pending(&mut self) -> Result<PluginOperationReceipt, PluginCoordinatorError> {
        let mut pending = self
            .pending
            .take()
            .ok_or(PluginCoordinatorError::NoPendingOperation)?;
        pending.receipt.status = PluginRuntimeStatus::Converged;
        pending.receipt.last_error = None;
        self.active_ids = pending.desired_ids;
        self.active_revision = if matches!(pending.receipt.kind, PluginOperationKind::Shutdown) {
            None
        } else {
            Some(pending.receipt.desired_revision)
        };
        let receipt = pending.receipt;
        self.last_receipt = Some(receipt.clone());
        Ok(receipt)
    }
}

/// 将接线错误格式化为可读消息。
fn wiring_error_message(error: PluginWiringError) -> String {
    error.to_string()
}
