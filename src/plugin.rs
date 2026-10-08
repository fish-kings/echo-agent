/*!
插件门面模块与不可变的准备（preparation）API。

发现与解析只在 [`PluginIntegrator::prepare`] 中执行一次。运行中的
Agent 只消费返回的这一代（generation）结果，而不会重新读取包文件。
*/

pub use echo_core::plugin::{
    AGENT_PLUGIN_SCHEMA_V1, InstallSource, PluginAuthor, PluginCapability, PluginDependency,
    PluginEntry, PluginId, PluginLifecycle, PluginLifecycleManager, PluginManifest, PluginRegistry,
    PluginRegistryDiagnostic, PluginScope, PluginUserConfigEntry, PluginUserConfigType,
    PluginVariables, ResolvedComponents,
};

mod coordinator;
mod prepared;

pub use coordinator::{
    PluginCoordinator, PluginCoordinatorError, PluginOperationKind, PluginOperationPhase,
    PluginOperationReceipt, PluginRuntimeStatus,
};

pub use prepared::{
    PluginDiagnosticSeverity, PluginIntegrator, PluginPreparationDiagnostic,
    PluginPublicationTarget, PluginWiringError, PluginWiringResult, PreparedPlugin,
    PreparedPluginDocument, PreparedPluginSet, PreparedPluginSkill, WiredPluginComponents,
};

pub(crate) use prepared::PluginPublicationAuthority;
