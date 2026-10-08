/*!
插件注册表 —— 发现、安装、生命周期管理。

`PluginRegistry` 是管理插件的中枢。它负责扫描、安装、卸载、启用/禁用，
以及依赖解析。
*/

use crate::plugin::PluginCapability;
use crate::plugin::manifest::PluginManifest;
use crate::plugin::scope::{InstallSource, PluginScope};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// 插件的唯一标识符（格式：`name` 或 `name@scope`）。
pub type PluginId = String;

/// 一个已安装插件的状态，持久化到磁盘。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginEntry {
    /// 插件清单元数据。
    pub manifest: PluginManifest,
    /// 插件安装目录的绝对路径。
    pub root: PathBuf,
    /// 该插件被安装到的作用域。
    pub scope: PluginScope,
    /// 该插件当前是否启用。
    pub enabled: bool,
    /// 经校验的用户配置，包含清单中的默认值。
    #[serde(default)]
    pub user_config: HashMap<String, serde_json::Value>,
    /// 已解析的组件路径（绝对路径，在加载时填充）。
    #[serde(skip)]
    pub resolved_components: Option<ResolvedComponents>,
}

impl PluginEntry {
    /// 根据插件包内固定的位置推断其支持的能力。
    pub fn inferred_capabilities(&self) -> Vec<PluginCapability> {
        let mut capabilities = Vec::new();
        if self.root.join("skills").is_dir() {
            capabilities.push(PluginCapability::Skill);
        }
        if self.root.join("mcp.json").is_file() {
            capabilities.push(PluginCapability::McpServer);
        }
        if self.root.join("agents").is_dir() {
            capabilities.push(PluginCapability::Agent);
        }
        if self.root.join("hooks/hooks.yaml").is_file() {
            capabilities.push(PluginCapability::Hook);
        }
        if self.root.join("lsp.yaml").is_file() {
            capabilities.push(PluginCapability::LspServer);
        }
        capabilities
    }
}

/**
一个插件全部组件已解析的绝对路径。

在清单加载完毕、固定位置相对于插件根目录解析完成后填充。
`None` 字段表示该组件不存在。
*/
#[derive(Debug, Clone, Default)]
pub struct ResolvedComponents {
    /// 标准的根级 `skills/` 目录。
    pub skill_dirs: Vec<PathBuf>,
    /// Agent 定义的 markdown 文件。
    pub agent_files: Vec<PathBuf>,
    /// Hook 配置文件。
    pub hooks_file: Option<PathBuf>,
    /// MCP 服务器配置文件。
    pub mcp_config_file: Option<PathBuf>,
    /// LSP 服务器配置文件。
    pub lsp_config_file: Option<PathBuf>,
    /// 非致命性的标准组件诊断信息。
    pub diagnostics: Vec<String>,
}

/// 一个可选路径期望的文件系统类型。
#[derive(Clone, Copy)]
enum ExpectedPathKind {
    /// 期望是普通文件。
    File,
    /// 期望是目录。
    Directory,
}

/**
在不阻塞异步执行器的前提下，检查某个可选路径是否存在且类型是否符合预期；
类型不符或读取失败时记录一条诊断信息。返回该路径是否符合预期类型。
*/
async fn classify_optional_path(
    path: &Path,
    expected: ExpectedPathKind,
    plugin_id: &str,
    component: &str,
    diagnostics: &mut Vec<String>,
) -> bool {
    match tokio::fs::metadata(path).await {
        Ok(metadata) => {
            let matches = match expected {
                ExpectedPathKind::File => metadata.is_file(),
                ExpectedPathKind::Directory => metadata.is_dir(),
            };
            if !matches {
                let expected_name = match expected {
                    ExpectedPathKind::File => "a regular file",
                    ExpectedPathKind::Directory => "a directory",
                };
                diagnostics.push(format!(
                    "Plugin '{plugin_id}' {component} path '{}' is not {expected_name}",
                    path.display()
                ));
            }
            matches
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => false,
        Err(error) => {
            diagnostics.push(format!(
                "Plugin '{plugin_id}' could not inspect {component} path '{}': {error}",
                path.display()
            ));
            false
        }
    }
}

/// 持久化的注册表状态，序列化为 `plugins.json`。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct RegistryState {
    /// 插件 ID 到其条目的映射。
    plugins: HashMap<String, PluginEntry>,
}

/// 插件注册表 —— 负责管理发现、安装与生命周期。
pub struct PluginRegistry {
    /// 已加载的插件条目，按插件 ID 索引。
    plugins: HashMap<PluginId, PluginEntry>,
    /// 持久化状态文件路径。
    state_file: PathBuf,
    /// 持久化数据目录根路径。
    data_dir: PathBuf,
    /// 应用层提供的用户插件安装目录。
    user_plugins_dir: PathBuf,
    /// 用于解析 Project/Local 作用域的项目根目录。
    project_root: Option<PathBuf>,
    /// facade 准备管理器使用的、按实例区分的缓存屏障。
    preparation_cache_id: String,
    /// 每次显式重新加载或成功提交的变更都会推进的单调修订号。
    revision: u64,
    /// 上一次完整发现视图所使用的作用域策略。
    last_successful_scopes: Vec<PluginScope>,
    /// 最近一次扫描产生的诊断信息。
    scan_diagnostics: Vec<PluginRegistryDiagnostic>,
}

/// 一条扫描诊断信息：某个路径存在问题及其严重程度。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PluginRegistryDiagnostic {
    /// 触发该诊断的路径。
    pub path: PathBuf,
    /// 诊断信息文本。
    pub message: String,
    /// 是否为错误（而非仅为警告）。
    pub is_error: bool,
}

impl PluginRegistry {
    /// 在给定的应用数据根目录下创建一个注册表。
    pub fn new(data_root: impl Into<PathBuf>, project_root: Option<PathBuf>) -> Self {
        let base = data_root.into();
        let user_plugins_dir = base.join("plugins");
        let state_file = base.join("plugins").join("registry.json");
        let data_dir = base.join("plugins").join("data");

        Self {
            plugins: HashMap::new(),
            state_file,
            data_dir,
            user_plugins_dir,
            project_root,
            preparation_cache_id: uuid::Uuid::new_v4().to_string(),
            revision: 0,
            last_successful_scopes: PluginScope::all().to_vec(),
            scan_diagnostics: Vec::new(),
        }
    }

    /// 使用自定义路径创建一个注册表（便于测试）。
    pub fn with_paths(
        state_file: PathBuf,
        data_dir: PathBuf,
        project_root: Option<PathBuf>,
    ) -> Self {
        let user_plugins_dir = state_file
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| data_dir.clone());
        Self {
            plugins: HashMap::new(),
            state_file,
            data_dir,
            user_plugins_dir,
            project_root,
            preparation_cache_id: uuid::Uuid::new_v4().to_string(),
            revision: 0,
            last_successful_scopes: PluginScope::all().to_vec(),
            scan_diagnostics: Vec::new(),
        }
    }

    /**
    当前已提交的注册表修订号。

    即使发现过程后续失败，扫描操作也会推进修订号，因为内存中的注册表
    此时可能已经被清空或部分重建。安装、卸载、启用、禁用与配置变更
    只有在其持久化状态成功提交后才会推进修订号。
    */
    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// 供共享准备缓存使用的、不透明的按实例身份标识。
    #[doc(hidden)]
    pub fn preparation_cache_id(&self) -> &str {
        &self.preparation_cache_id
    }

    /// 返回最近一次扫描产生的诊断信息。
    pub fn scan_diagnostics(&self) -> &[PluginRegistryDiagnostic] {
        &self.scan_diagnostics
    }

    /// 检查修订号是否仍有空间递增，避免溢出。
    fn ensure_revision_available(&self) -> Result<(), String> {
        self.revision
            .checked_add(1)
            .map(|_| ())
            .ok_or_else(|| "Plugin registry revision exhausted".to_string())
    }

    /// 与 [`Self::ensure_revision_available`] 相同，但返回 `std::io::Result`。
    fn ensure_revision_available_io(&self) -> std::io::Result<()> {
        self.ensure_revision_available()
            .map_err(std::io::Error::other)
    }

    /// 将修订号加一；若已到达上限则返回错误。
    fn advance_revision(&mut self) -> Result<(), String> {
        self.revision = self
            .revision
            .checked_add(1)
            .ok_or_else(|| "Plugin registry revision exhausted".to_string())?;
        Ok(())
    }

    // ── 发现 ──────────────────────────────────────────────────────

    /**
    扫描所有作用域下已安装的插件并加载它们。

    返回发现的插件数量。
    */
    pub fn scan_all(&mut self) -> std::io::Result<usize> {
        self.scan_scopes(PluginScope::all())
    }

    /**
    刷新「上一次成功的作用域视图」，仅在成功时提交。

    一次正常扫描会在遍历各个包目录之前清空接收方，因此一次 I/O 错误
    可能让内存中的视图处于「部分发现」的状态。生命周期重试逻辑必须
    在获得一个完整的替换视图之前，保留此前的期望状态权威性。
    保留已选定的作用域也能防止一次受限的宿主视图在重试过程中被放宽。
    */
    pub fn refresh_current_view(&mut self) -> std::io::Result<usize> {
        let scopes = self.last_successful_scopes.clone();
        let mut refreshed = Self {
            plugins: HashMap::new(),
            state_file: self.state_file.clone(),
            data_dir: self.data_dir.clone(),
            user_plugins_dir: self.user_plugins_dir.clone(),
            project_root: self.project_root.clone(),
            preparation_cache_id: self.preparation_cache_id.clone(),
            revision: self.revision,
            last_successful_scopes: scopes.clone(),
            scan_diagnostics: Vec::new(),
        };
        let count = refreshed.scan_scopes(&scopes)?;
        *self = refreshed;
        Ok(count)
    }

    /**
    只扫描请求中指定的安装作用域。

    这对于内嵌运行时和有意只暴露标准插件作用域子集的隔离集成测试很有用。
    */
    pub fn scan_scopes(&mut self, scopes: &[PluginScope]) -> std::io::Result<usize> {
        self.ensure_revision_available_io()?;
        self.advance_revision().map_err(std::io::Error::other)?;
        self.plugins.clear();
        self.scan_diagnostics.clear();
        let mut total = 0;

        for scope in scopes {
            let dir = scope.resolve_dir(&self.user_plugins_dir, self.project_root.as_deref());
            let count = self.scan_scope_dir(*scope, &dir)?;
            total += count;
        }

        // 加载持久化的启用/禁用状态
        self.load_state()?;
        self.last_successful_scopes = scopes.to_vec();
        Ok(total)
    }

    /**
    校验一个插件目录，并解析其固定包位置上发现的每一个受支持组件。
    缺失的可选位置仍然视为有效。
    */
    pub fn validate_plugin_dir(
        root: &Path,
    ) -> Result<(PluginManifest, ResolvedComponents), Vec<String>> {
        let manifest_path = root.join("plugin.json");
        let manifest = PluginManifest::from_file(&manifest_path).map_err(|error| vec![error])?;
        let errors = manifest
            .validate()
            .into_iter()
            .map(|error| error.to_string())
            .collect::<Vec<_>>();
        if !errors.is_empty() {
            return Err(errors);
        }

        let user_config = manifest.user_config_defaults();

        let plugin_id = manifest.name.clone();
        let mut registry = Self::with_paths(
            root.join(".plugin-validation-state.json"),
            root.join(".plugin-validation-data"),
            root.parent().map(Path::to_path_buf),
        );
        registry.plugins.insert(
            plugin_id.clone(),
            PluginEntry {
                manifest: manifest.clone(),
                root: root.to_path_buf(),
                scope: PluginScope::Local,
                enabled: true,
                user_config,
                resolved_components: None,
            },
        );
        registry
            .resolve_components(&plugin_id)
            .map(|resolved| (manifest, resolved))
            .map_err(|error| vec![error])
    }

    /// 扫描单个目录下的插件子目录。
    fn scan_scope_dir(&mut self, scope: PluginScope, dir: &Path) -> std::io::Result<usize> {
        if !dir.exists() {
            return Ok(0);
        }

        let mut count = 0;
        let mut paths = std::fs::read_dir(dir)?
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .collect::<Vec<_>>();
        paths.sort();
        for path in paths {
            if !path.is_dir() {
                continue;
            }

            // Agent Plugins 的清单位于包根目录下。
            let manifest_path = path.join("plugin.json");
            if !manifest_path.exists() {
                continue;
            }

            match PluginManifest::from_file(&manifest_path) {
                Ok(manifest) => {
                    let validation_errors = manifest.validate();
                    if !validation_errors.is_empty() {
                        self.scan_diagnostics.push(PluginRegistryDiagnostic {
                            path: manifest_path.clone(),
                            message: validation_errors
                                .iter()
                                .map(ToString::to_string)
                                .collect::<Vec<_>>()
                                .join("; "),
                            is_error: true,
                        });
                        tracing::warn!(
                            path = %manifest_path.display(),
                            errors = %validation_errors
                                .iter()
                                .map(ToString::to_string)
                                .collect::<Vec<_>>()
                                .join("; "),
                            "Skipping invalid plugin manifest"
                        );
                        continue;
                    }
                    let unknown_fields = manifest.unknown_top_level_fields();
                    if !unknown_fields.is_empty() {
                        self.scan_diagnostics.push(PluginRegistryDiagnostic {
                            path: manifest_path.clone(),
                            message: format!("unknown fields: {}", unknown_fields.join(", ")),
                            is_error: false,
                        });
                        tracing::warn!(
                            path = %manifest_path.display(),
                            fields = %unknown_fields.join(", "),
                            "Ignoring unknown Agent Plugins manifest fields"
                        );
                    }
                    let id = manifest.name.clone();
                    let user_config = manifest.user_config_defaults();
                    let config_ready = manifest.validate_user_config(&user_config).is_empty();
                    let enabled = manifest.default_enabled && config_ready;
                    if manifest.default_enabled && !config_ready {
                        tracing::info!(
                            plugin = %manifest.name,
                            "Plugin starts disabled until required configuration is provided"
                        );
                    }
                    let entry = PluginEntry {
                        manifest,
                        root: path.clone(),
                        scope,
                        enabled, // 若存在已持久化状态，会被其覆盖
                        user_config,
                        resolved_components: None,
                    };
                    if let Some(existing) = self.plugins.get(&id) {
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::AlreadyExists,
                            format!(
                                "plugin name collision for '{id}': {} ({:?}) and {} ({scope:?})",
                                existing.root.display(),
                                existing.scope,
                                path.display()
                            ),
                        ));
                    }
                    self.plugins.insert(id, entry);
                    count += 1;
                }
                Err(e) => {
                    self.scan_diagnostics.push(PluginRegistryDiagnostic {
                        path: manifest_path.clone(),
                        message: e.to_string(),
                        is_error: true,
                    });
                    tracing::warn!(
                        "Failed to load plugin manifest at {}: {e}",
                        manifest_path.display()
                    );
                }
            }
        }

        Ok(count)
    }

    // ── 安装 ───────────────────────────────────────────────────

    /**
    从某个来源将插件安装到指定作用域。

    对于本地来源，直接复制目录。
    对于 git 来源，克隆仓库（浅克隆）。

    成功时返回插件 ID。
    */
    pub fn install(
        &mut self,
        source: &InstallSource,
        scope: PluginScope,
    ) -> Result<PluginId, String> {
        self.ensure_revision_available()?;
        let target_dir = scope.resolve_dir(&self.user_plugins_dir, self.project_root.as_deref());
        std::fs::create_dir_all(&target_dir)
            .map_err(|e| format!("Failed to create plugin directory: {e}"))?;

        let plugin_id = match source {
            InstallSource::Local(src_path) => self.install_local(src_path, &target_dir, scope),
            InstallSource::Git { url, subdir } => {
                self.install_git(url, subdir.as_deref(), &target_dir, scope)
            }
        }?;
        self.advance_revision()?;
        Ok(plugin_id)
    }

    /// 从本地目录安装插件：校验清单、暂存复制、校验暂存结果，再原子性地提交。
    fn install_local(
        &mut self,
        src: &Path,
        target_dir: &Path,
        scope: PluginScope,
    ) -> Result<PluginId, String> {
        // 校验来源目录包含清单文件
        let manifest_path = src.join("plugin.json");
        if !manifest_path.exists() {
            return Err(format!(
                "Source directory {} does not contain root plugin.json",
                src.display()
            ));
        }

        let manifest = PluginManifest::from_file(&manifest_path)?;
        let errors = manifest.validate();
        if !errors.is_empty() {
            return Err(format!(
                "Manifest validation failed: {}",
                errors
                    .iter()
                    .map(|e| e.to_string())
                    .collect::<Vec<_>>()
                    .join("; ")
            ));
        }
        let plugin_id = manifest.name.clone();
        let dest = target_dir.join(&plugin_id);

        if dest.exists() {
            return Err(format!(
                "Plugin '{plugin_id}' is already installed at {}",
                dest.display()
            ));
        }

        let staging = target_dir.join(format!(".{}.{}.staging", plugin_id, uuid::Uuid::new_v4()));
        if let Err(error) = copy_dir_recursive(src, &staging) {
            let _ = std::fs::remove_dir_all(&staging);
            return Err(format!("Failed to stage plugin copy: {error}"));
        }
        if let Err(errors) = Self::validate_plugin_dir(&staging) {
            let _ = std::fs::remove_dir_all(&staging);
            return Err(format!(
                "Staged plugin validation failed: {}",
                errors.join("; ")
            ));
        }
        if let Err(error) = std::fs::rename(&staging, &dest) {
            let _ = std::fs::remove_dir_all(&staging);
            return Err(format!("Failed to commit staged plugin: {error}"));
        }

        let user_config = manifest.user_config_defaults();
        let enabled =
            manifest.default_enabled && manifest.validate_user_config(&user_config).is_empty();
        let entry = PluginEntry {
            manifest,
            root: dest.clone(),
            scope,
            enabled,
            user_config,
            resolved_components: None,
        };

        self.plugins.insert(plugin_id.clone(), entry);
        if let Err(error) = self.resolve_enabled_dependencies() {
            self.plugins.remove(&plugin_id);
            let cleanup_error = std::fs::remove_dir_all(&dest).err();
            return Err(match cleanup_error {
                Some(cleanup_error) => format!(
                    "Cannot install plugin '{plugin_id}': {error}; additionally failed to roll back {}: {cleanup_error}",
                    dest.display()
                ),
                None => format!("Cannot install plugin '{plugin_id}': {error}"),
            });
        }
        if let Err(error) = self.save_state() {
            self.plugins.remove(&plugin_id);
            let cleanup_error = std::fs::remove_dir_all(&dest).err();
            return Err(match cleanup_error {
                Some(cleanup_error) => format!(
                    "{error}; additionally failed to roll back {}: {cleanup_error}",
                    dest.display()
                ),
                None => error,
            });
        }
        Ok(plugin_id)
    }

    /// 从 git 仓库安装插件：先浅克隆到临时目录，再复用本地安装逻辑。
    fn install_git(
        &mut self,
        url: &str,
        subdir: Option<&str>,
        target_dir: &Path,
        scope: PluginScope,
    ) -> Result<PluginId, String> {
        /*
        宿主应用是一个本地、由用户自行掌控的应用。接受标准的加密 Git
        传输方式，包括通过 SSH 访问私有仓库；只拒绝明文/明显畸形的远程输入。
        */
        let supported = url.starts_with("https://")
            || url.starts_with("ssh://")
            || (url.starts_with("git@") && url.contains(':'));
        if !supported {
            return Err(format!(
                "Plugin git URL must use https:// or SSH (received: {})",
                url.split("://").next().unwrap_or(url)
            ));
        }

        // 克隆到临时目录
        let tmp_dir = target_dir.join(".tmp-clone");
        if tmp_dir.exists() {
            std::fs::remove_dir_all(&tmp_dir)
                .map_err(|e| format!("Failed to clean temp directory: {e}"))?;
        }

        let status = std::process::Command::new("git")
            .args(["clone", "--depth", "1", url])
            .arg(&tmp_dir)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map_err(|e| format!("Failed to run git clone: {e}"))?;

        if !status.success() {
            return Err(format!("git clone failed for {url}"));
        }

        let src = if let Some(sub) = subdir {
            tmp_dir.join(sub)
        } else {
            tmp_dir.clone()
        };

        let result = self.install_local(&src, target_dir, scope);

        // 清理临时目录
        let _ = std::fs::remove_dir_all(&tmp_dir);

        result
    }

    // ── 卸载 ─────────────────────────────────────────────────────

    /// 卸载一个插件。若 `keep_data` 为 false，则同时删除其持久化数据目录。
    pub fn uninstall(&mut self, plugin_id: &str, keep_data: bool) -> Result<(), String> {
        self.ensure_no_enabled_dependents(plugin_id)?;
        let entry = self
            .plugins
            .get(plugin_id)
            .cloned()
            .ok_or_else(|| format!("Plugin '{plugin_id}' is not installed"))?;
        self.ensure_revision_available()?;

        let tombstone = entry
            .root
            .with_extension(format!("plugin-uninstall-{}", uuid::Uuid::new_v4()));
        if entry.root.exists() {
            std::fs::rename(&entry.root, &tombstone)
                .map_err(|error| format!("Failed to stage plugin removal: {error}"))?;
        }
        self.plugins.remove(plugin_id);
        if let Err(error) = self.save_state() {
            self.plugins.insert(plugin_id.to_string(), entry.clone());
            let restore_error = if tombstone.exists() {
                std::fs::rename(&tombstone, &entry.root).err()
            } else {
                None
            };
            return Err(match restore_error {
                Some(restore_error) => format!(
                    "{error}; additionally failed to restore plugin directory: {restore_error}"
                ),
                None => error,
            });
        }
        if tombstone.exists()
            && let Err(error) = std::fs::remove_dir_all(&tombstone)
        {
            tracing::warn!(path = %tombstone.display(), %error, "plugin uninstall tombstone cleanup deferred");
        }
        if !keep_data {
            let data = PluginEntry::data_dir_for(plugin_id, &self.data_dir);
            if data.exists()
                && let Err(error) = std::fs::remove_dir_all(&data)
            {
                tracing::warn!(path = %data.display(), %error, "plugin data cleanup deferred");
            }
        }
        self.advance_revision()
    }

    // ── 启用 / 禁用 ───────────────────────────────────────────────

    /// 启用一个已禁用的插件。
    pub fn enable(&mut self, plugin_id: &str) -> Result<(), String> {
        let entry = self
            .plugins
            .get(plugin_id)
            .ok_or_else(|| format!("Plugin '{plugin_id}' is not installed"))?;
        let was_enabled = entry.enabled;
        if was_enabled {
            return Ok(());
        }
        self.ensure_revision_available()?;
        let config_errors = entry.manifest.validate_user_config(&entry.user_config);
        if !config_errors.is_empty() {
            return Err(format!(
                "Cannot enable plugin '{plugin_id}': {}",
                config_errors
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join("; ")
            ));
        }

        if let Some(entry) = self.plugins.get_mut(plugin_id) {
            entry.enabled = true;
        }
        if let Err(error) = self.resolve_enabled_dependencies() {
            if let Some(entry) = self.plugins.get_mut(plugin_id) {
                entry.enabled = false;
            }
            return Err(format!("Cannot enable plugin '{plugin_id}': {error}"));
        }
        if let Err(error) = self.save_state() {
            if let Some(entry) = self.plugins.get_mut(plugin_id) {
                entry.enabled = false;
            }
            return Err(error);
        }
        self.advance_revision()
    }

    /// 禁用一个已启用的插件，但不卸载它。
    pub fn disable(&mut self, plugin_id: &str) -> Result<(), String> {
        let was_enabled = self
            .plugins
            .get(plugin_id)
            .ok_or_else(|| format!("Plugin '{plugin_id}' is not installed"))?
            .enabled;
        if !was_enabled {
            return Ok(());
        }
        self.ensure_revision_available()?;

        self.ensure_no_enabled_dependents(plugin_id)?;
        if let Some(entry) = self.plugins.get_mut(plugin_id) {
            entry.enabled = false;
        }
        if let Err(error) = self.save_state() {
            if let Some(entry) = self.plugins.get_mut(plugin_id) {
                entry.enabled = true;
            }
            return Err(error);
        }
        self.advance_revision()
    }

    /// 确认没有已启用的插件依赖于 `plugin_id`；否则返回列出这些依赖方的错误。
    fn ensure_no_enabled_dependents(&self, plugin_id: &str) -> Result<(), String> {
        let mut dependents = self
            .plugins
            .iter()
            .filter(|(_, entry)| {
                entry.enabled
                    && entry
                        .manifest
                        .dependencies
                        .iter()
                        .any(|dependency| dependency.name() == plugin_id)
            })
            .map(|(id, _)| id.clone())
            .collect::<Vec<_>>();
        dependents.sort();
        if dependents.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "Plugin '{plugin_id}' is required by enabled plugin(s): {}",
                dependents.join(", ")
            ))
        }
    }

    // ── 查询 ────────────────────────────────────────────────────────

    /// 按 ID 获取一个插件条目。
    pub fn get(&self, plugin_id: &str) -> Option<&PluginEntry> {
        self.plugins.get(plugin_id)
    }

    /// 替换并持久化一个插件的用户配置。
    pub fn configure(
        &mut self,
        plugin_id: &str,
        values: HashMap<String, serde_json::Value>,
    ) -> Result<(), String> {
        self.ensure_revision_available()?;
        let resolved = {
            let entry = self
                .plugins
                .get(plugin_id)
                .ok_or_else(|| format!("Plugin '{plugin_id}' is not installed"))?;
            entry
                .manifest
                .resolve_user_config(&values)
                .map_err(|errors| {
                    errors
                        .iter()
                        .map(ToString::to_string)
                        .collect::<Vec<_>>()
                        .join("; ")
                })?
        };
        let previous = self
            .plugins
            .get(plugin_id)
            .map(|entry| entry.user_config.clone())
            .ok_or_else(|| format!("Plugin '{plugin_id}' is not installed"))?;
        if let Some(entry) = self.plugins.get_mut(plugin_id) {
            entry.user_config = resolved;
        }
        if let Err(error) = self.save_state() {
            if let Some(entry) = self.plugins.get_mut(plugin_id) {
                entry.user_config = previous;
            }
            return Err(error);
        }
        self.advance_revision()
    }

    /// 为一个插件的组件文件构建变量替换上下文。
    pub fn variables_for(&self, plugin_id: &str) -> Result<super::PluginVariables, String> {
        let entry = self
            .plugins
            .get(plugin_id)
            .ok_or_else(|| format!("Plugin '{plugin_id}' is not installed"))?;
        let project_dir = self
            .project_root
            .clone()
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| entry.root.clone()));
        Ok(super::PluginVariables::new(
            entry.root.clone(),
            self.data_dir_for(plugin_id),
            project_dir,
        )
        .with_json_user_config(&entry.user_config))
    }

    /// 列出所有已安装的插件。
    pub fn list(&self) -> Vec<&PluginEntry> {
        let mut entries: Vec<_> = self.plugins.values().collect();
        entries.sort_by(|a, b| a.manifest.name.cmp(&b.manifest.name));
        entries
    }

    /// 仅列出已启用的插件。
    pub fn list_enabled(&self) -> Vec<&PluginEntry> {
        self.list().into_iter().filter(|e| e.enabled).collect()
    }

    /// 按关键字搜索插件（匹配名称、描述、关键词）。
    pub fn search(&self, query: &str) -> Vec<&PluginEntry> {
        let q = query.to_lowercase();
        self.list()
            .into_iter()
            .filter(|e| {
                e.manifest.name.to_lowercase().contains(&q)
                    || e.manifest.description.to_lowercase().contains(&q)
                    || e.manifest
                        .keywords
                        .iter()
                        .any(|k| k.to_lowercase().contains(&q))
            })
            .collect()
    }

    /// 获取已安装插件的总数。
    pub fn count(&self) -> usize {
        self.plugins.len()
    }

    // ── 组件解析 ───────────────────────────────────────────────

    /**
    解析一个插件的组件路径，使其变为绝对路径。

    组件位置由扁平的包布局固定不变。缺失的可选位置视为有效；
    若某个位置的文件系统类型不符，只会隔离影响该组件，并作为诊断信息上报。
    */
    pub fn resolve_components(&mut self, plugin_id: &str) -> Result<ResolvedComponents, String> {
        let entry = self
            .plugins
            .get(plugin_id)
            .ok_or_else(|| format!("Plugin '{plugin_id}' not found"))?;

        let root = &entry.root;
        let mut resolved = ResolvedComponents::default();

        /*
        Agent Plugins 使用固定的可移植组件位置。缺失的位置视为有效；
        文件系统类型不符只会禁用该类型的组件，并作为非致命诊断上报。
        */
        let skills_dir = root.join("skills");
        if skills_dir.is_dir() {
            resolved.skill_dirs.push(skills_dir);
        } else if skills_dir.exists() {
            resolved.diagnostics.push(format!(
                "Plugin '{plugin_id}' standard skills path '{}' is not a directory",
                skills_dir.display()
            ));
        }

        let agents_dir = root.join("agents");
        if agents_dir.is_dir() {
            match std::fs::read_dir(&agents_dir) {
                Ok(entries) => {
                    resolved.agent_files.extend(entries.filter_map(|entry| {
                        let path = entry.ok()?.path();
                        (path.is_file() && path.extension().is_some_and(|value| value == "md"))
                            .then_some(path)
                    }));
                    resolved.agent_files.sort();
                }
                Err(error) => resolved.diagnostics.push(format!(
                    "Plugin '{plugin_id}' could not scan agents path '{}': {error}",
                    agents_dir.display()
                )),
            }
        } else if agents_dir.exists() {
            resolved.diagnostics.push(format!(
                "Plugin '{plugin_id}' agents path '{}' is not a directory",
                agents_dir.display()
            ));
        }

        let hooks_file = root.join("hooks/hooks.yaml");
        if hooks_file.is_file() {
            resolved.hooks_file = Some(hooks_file);
        } else if hooks_file.exists() {
            resolved.diagnostics.push(format!(
                "Plugin '{plugin_id}' hooks path '{}' is not a regular file",
                hooks_file.display()
            ));
        }

        let mcp_file = root.join("mcp.json");
        if mcp_file.is_file() {
            resolved.mcp_config_file = Some(mcp_file);
        } else if mcp_file.exists() {
            resolved.diagnostics.push(format!(
                "Plugin '{plugin_id}' standard MCP path '{}' is not a regular file",
                mcp_file.display()
            ));
        }

        let lsp_file = root.join("lsp.yaml");
        if lsp_file.is_file() {
            resolved.lsp_config_file = Some(lsp_file);
        } else if lsp_file.exists() {
            resolved.diagnostics.push(format!(
                "Plugin '{plugin_id}' LSP path '{}' is not a regular file",
                lsp_file.display()
            ));
        }

        // 把已解析的组件写回存储
        if let Some(entry) = self.plugins.get_mut(plugin_id) {
            entry.resolved_components = Some(resolved.clone());
        }

        Ok(resolved)
    }

    /// 在不阻塞异步执行器的前提下解析固定的组件路径。
    pub async fn resolve_components_async(
        &mut self,
        plugin_id: &str,
    ) -> Result<ResolvedComponents, String> {
        let root = self
            .plugins
            .get(plugin_id)
            .map(|entry| entry.root.clone())
            .ok_or_else(|| format!("Plugin '{plugin_id}' not found"))?;
        let mut resolved = ResolvedComponents::default();

        let skills_dir = root.join("skills");
        if classify_optional_path(
            &skills_dir,
            ExpectedPathKind::Directory,
            plugin_id,
            "skills",
            &mut resolved.diagnostics,
        )
        .await
        {
            resolved.skill_dirs.push(skills_dir);
        }

        let agents_dir = root.join("agents");
        if classify_optional_path(
            &agents_dir,
            ExpectedPathKind::Directory,
            plugin_id,
            "agents",
            &mut resolved.diagnostics,
        )
        .await
        {
            match tokio::fs::read_dir(&agents_dir).await {
                Ok(mut entries) => loop {
                    match entries.next_entry().await {
                        Ok(Some(entry)) => {
                            let path = entry.path();
                            let is_file = entry
                                .file_type()
                                .await
                                .map(|kind| kind.is_file())
                                .unwrap_or(false);
                            if is_file
                                && path.extension().is_some_and(|extension| extension == "md")
                            {
                                resolved.agent_files.push(path);
                            }
                        }
                        Ok(None) => break,
                        Err(error) => {
                            resolved.diagnostics.push(format!(
                                "Plugin '{plugin_id}' could not scan agents path '{}': {error}",
                                agents_dir.display()
                            ));
                            break;
                        }
                    }
                },
                Err(error) => resolved.diagnostics.push(format!(
                    "Plugin '{plugin_id}' could not scan agents path '{}': {error}",
                    agents_dir.display()
                )),
            }
            resolved.agent_files.sort();
        }

        let hooks_file = root.join("hooks/hooks.yaml");
        if classify_optional_path(
            &hooks_file,
            ExpectedPathKind::File,
            plugin_id,
            "hooks",
            &mut resolved.diagnostics,
        )
        .await
        {
            resolved.hooks_file = Some(hooks_file);
        }
        let mcp_file = root.join("mcp.json");
        if classify_optional_path(
            &mcp_file,
            ExpectedPathKind::File,
            plugin_id,
            "MCP",
            &mut resolved.diagnostics,
        )
        .await
        {
            resolved.mcp_config_file = Some(mcp_file);
        }
        let lsp_file = root.join("lsp.yaml");
        if classify_optional_path(
            &lsp_file,
            ExpectedPathKind::File,
            plugin_id,
            "LSP",
            &mut resolved.diagnostics,
        )
        .await
        {
            resolved.lsp_config_file = Some(lsp_file);
        }

        if let Some(entry) = self.plugins.get_mut(plugin_id) {
            entry.resolved_components = Some(resolved.clone());
        }
        Ok(resolved)
    }

    // ── 依赖解析 ──────────────────────────────────────────────

    /**
    通过拓扑排序解析插件依赖关系。

    按依赖顺序（依赖项在前）返回插件 ID 列表。
    若存在循环依赖或缺失依赖，则返回错误。
    */
    pub fn resolve_dependencies(&self) -> Result<Vec<PluginId>, String> {
        self.resolve_dependencies_matching(|_| true)
    }

    /**
    只解析已启用的插件，并要求一个已启用插件的每一个依赖也必须是启用状态。
    已禁用的插件不会污染启动流程，一个已启用插件也不能悄悄依赖一个已禁用的插件。
    */
    pub fn resolve_enabled_dependencies(&self) -> Result<Vec<PluginId>, String> {
        self.resolve_dependencies_matching(|entry| entry.enabled)
    }

    /// 依赖解析的共享实现：`include` 决定哪些插件条目参与图构建。
    fn resolve_dependencies_matching(
        &self,
        include: impl Fn(&PluginEntry) -> bool,
    ) -> Result<Vec<PluginId>, String> {
        let mut graph: HashMap<&str, Vec<&str>> = HashMap::new();
        let mut in_degree: HashMap<&str, usize> = HashMap::new();

        // 初始化
        for (id, entry) in &self.plugins {
            if !include(entry) {
                continue;
            }
            graph.entry(id.as_str()).or_default();
            in_degree.entry(id.as_str()).or_insert(0);
        }

        // 构建边 + 强制版本约束
        for (id, entry) in &self.plugins {
            if !include(entry) {
                continue;
            }
            for dep in &entry.manifest.dependencies {
                let dep_name = dep.name();
                let dep_entry = self.plugins.get(dep_name).ok_or_else(|| {
                    format!("Plugin '{id}' depends on '{dep_name}' which is not installed")
                })?;
                if !include(dep_entry) {
                    return Err(format!(
                        "Plugin '{id}' depends on '{dep_name}' which is disabled"
                    ));
                }
                /*
                强制执行声明的版本约束（P1 —— 此前的「名称存在即可」检查
                完全忽略版本，导致任意版本都能满足依赖）。
                */
                match dep.satisfies(dep_entry.manifest.version.as_deref()) {
                    Ok(true) => {}
                    Ok(false) => {
                        return Err(format!(
                            "Plugin '{id}' requires '{dep_name} {}' but installed version is '{}'",
                            dep.version_constraint().unwrap_or("any"),
                            dep_entry.manifest.version_label()
                        ));
                    }
                    Err(e) => {
                        return Err(format!(
                            "Plugin '{id}' dependency '{dep_name}' version check failed: {e}"
                        ));
                    }
                }
                graph.entry(dep_name).or_default().push(id.as_str());
                *in_degree.entry(id.as_str()).or_insert(0) += 1;
            }
        }

        // Kahn 算法
        let mut queue: Vec<&str> = in_degree
            .iter()
            .filter(|(_, deg)| **deg == 0)
            .map(|(id, _)| *id)
            .collect();
        queue.sort_by(|left, right| right.cmp(left));

        let mut sorted = Vec::new();
        while let Some(node) = queue.pop() {
            sorted.push(node.to_string());
            if let Some(neighbors) = graph.get(node) {
                for &neighbor in neighbors {
                    if let Some(deg) = in_degree.get_mut(neighbor) {
                        *deg -= 1;
                        if *deg == 0 {
                            queue.push(neighbor);
                            queue.sort_by(|left, right| right.cmp(left));
                        }
                    }
                }
            }
        }

        if sorted.len() != in_degree.len() {
            return Err("Circular dependency detected among plugins".to_string());
        }

        Ok(sorted)
    }

    // ── 持久化 ────────────────────────────────────────────────────

    /// 将当前的启用/禁用状态保存到磁盘。
    fn save_state(&self) -> Result<(), String> {
        let state = RegistryState {
            plugins: self
                .plugins
                .iter()
                .map(|(id, entry)| {
                    (
                        id.clone(),
                        PluginEntry {
                            resolved_components: None,
                            ..entry.clone()
                        },
                    )
                })
                .collect(),
        };

        if let Some(parent) = self.state_file.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|error| format!("Failed to create plugin state directory: {error}"))?;
        }
        let json = serde_json::to_string_pretty(&state)
            .map_err(|error| format!("Failed to serialize plugin state: {error}"))?;
        let temporary = self.state_file.with_extension("json.tmp");
        std::fs::write(&temporary, json)
            .map_err(|error| format!("Failed to write plugin state: {error}"))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let permissions = std::fs::Permissions::from_mode(0o600);
            std::fs::set_permissions(&temporary, permissions)
                .map_err(|error| format!("Failed to protect plugin state: {error}"))?;
        }
        std::fs::rename(&temporary, &self.state_file)
            .map_err(|error| format!("Failed to replace plugin state: {error}"))
    }

    /// 加载持久化状态并与已发现的插件合并。
    fn load_state(&mut self) -> std::io::Result<()> {
        let state: RegistryState = match std::fs::read_to_string(&self.state_file) {
            Ok(content) => match serde_json::from_str(&content) {
                Ok(state) => state,
                Err(error) => {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!(
                            "invalid plugin registry state {}: {error}",
                            self.state_file.display()
                        ),
                    ));
                }
            },
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(error) => return Err(error),
        };

        // 从持久化文件中合并启用/禁用状态以及经校验的配置。
        for (id, saved_entry) in &state.plugins {
            if let Some(entry) = self.plugins.get_mut(id) {
                match entry.manifest.resolve_user_config(&saved_entry.user_config) {
                    Ok(config) => {
                        entry.user_config = config;
                        entry.enabled = saved_entry.enabled;
                    }
                    Err(errors) => {
                        entry.enabled = false;
                        tracing::warn!(
                            plugin = %id,
                            errors = %errors.iter().map(ToString::to_string).collect::<Vec<_>>().join("; "),
                            "Plugin disabled because its persisted configuration is invalid"
                        );
                    }
                }
            }
        }
        Ok(())
    }

    /// 获取一个插件的持久化数据目录。
    pub fn data_dir_for(&self, plugin_id: &str) -> PathBuf {
        PluginEntry::data_dir_for(plugin_id, &self.data_dir)
    }
}

impl PluginEntry {
    /// 计算一个插件的数据目录路径。
    fn data_dir_for(name: &str, base_data_dir: &Path) -> PathBuf {
        let sanitized: String = name
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                    c
                } else {
                    '-'
                }
            })
            .collect();
        base_data_dir.join(sanitized)
    }
}

// ── 辅助函数 ──────────────────────────────────────────────────────────────

/// 递归复制一个目录。
fn copy_dir_recursive(src: &Path, dst: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let dst_path = dst.join(entry.file_name());

        if file_type.is_dir() {
            // 跳过如 .git 之类的隐藏目录
            if entry.file_name().to_string_lossy().starts_with(".git") {
                continue;
            }
            copy_dir_recursive(&entry.path(), &dst_path)?;
        } else {
            std::fs::copy(entry.path(), &dst_path)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plugin::AGENT_PLUGIN_SCHEMA_V1;

    fn temporary_root(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "echo-agent-plugin-{label}-{}",
            uuid::Uuid::new_v4()
        ))
    }

    fn create_plugin(
        parent: &Path,
        name: &str,
        dependencies: serde_json::Value,
    ) -> Result<PathBuf, String> {
        let root = parent.join(name);
        std::fs::create_dir_all(root.join("skills/example")).map_err(|error| error.to_string())?;
        std::fs::create_dir_all(root.join("agents")).map_err(|error| error.to_string())?;
        std::fs::create_dir_all(root.join("hooks")).map_err(|error| error.to_string())?;
        std::fs::write(
            root.join("skills/example/SKILL.md"),
            "---\nname: example\ndescription: Example skill\n---\nUse it.\n",
        )
        .map_err(|error| error.to_string())?;
        std::fs::write(
            root.join("agents/reviewer.md"),
            "---\nname: reviewer\ndescription: Reviews changes\n---\nReview carefully.\n",
        )
        .map_err(|error| error.to_string())?;
        std::fs::write(root.join("hooks/hooks.yaml"), "{}\n").map_err(|error| error.to_string())?;
        std::fs::write(root.join("lsp.yaml"), "languages: {}\n")
            .map_err(|error| error.to_string())?;
        std::fs::write(
            root.join("mcp.json"),
            "{\"$schema\":\"https://agent-plugins.org/schemas/1.0.0/mcp.schema.json\",\"mcpServers\":{}}",
        )
        .map_err(|error| error.to_string())?;
        let manifest = serde_json::json!({
            "$schema": AGENT_PLUGIN_SCHEMA_V1,
            "name": name,
            "version": "1.0.0",
            "description": "Test plugin",
            "config": {
                "endpoint": {
                    "type": "string",
                    "title": "Endpoint",
                    "default": "https://example.com"
                }
            },
            "dependencies": dependencies
        });
        let manifest_text =
            serde_json::to_string_pretty(&manifest).map_err(|error| error.to_string())?;
        std::fs::write(root.join("plugin.json"), manifest_text)
            .map_err(|error| error.to_string())?;
        Ok(root)
    }

    #[test]
    fn validates_and_resolves_flat_components() -> Result<(), String> {
        let temporary = temporary_root("resolve");
        let plugin = create_plugin(&temporary, "resolve.test", serde_json::json!([]))?;
        let (manifest, resolved) =
            PluginRegistry::validate_plugin_dir(&plugin).map_err(|errors| errors.join("; "))?;
        assert_eq!(manifest.name, "resolve.test");
        assert_eq!(resolved.skill_dirs, vec![plugin.join("skills")]);
        assert_eq!(resolved.mcp_config_file, Some(plugin.join("mcp.json")));
        assert_eq!(resolved.agent_files.len(), 1);
        assert_eq!(resolved.hooks_file, Some(plugin.join("hooks/hooks.yaml")));
        assert!(resolved.diagnostics.is_empty());
        std::fs::remove_dir_all(temporary).map_err(|error| error.to_string())
    }

    #[test]
    fn fixed_component_wrong_kind_is_non_fatal() -> Result<(), String> {
        let temporary = temporary_root("wrong-kind");
        let plugin = create_plugin(&temporary, "wrong.kind", serde_json::json!([]))?;
        std::fs::remove_dir_all(plugin.join("skills")).map_err(|error| error.to_string())?;
        std::fs::write(plugin.join("skills"), "not a directory")
            .map_err(|error| error.to_string())?;
        let (_, resolved) =
            PluginRegistry::validate_plugin_dir(&plugin).map_err(|errors| errors.join("; "))?;
        assert!(resolved.skill_dirs.is_empty());
        assert_eq!(resolved.diagnostics.len(), 1);
        std::fs::remove_dir_all(temporary).map_err(|error| error.to_string())
    }

    #[test]
    fn scans_root_plugin_json_and_applies_configuration_defaults() -> Result<(), String> {
        let temporary = temporary_root("scan");
        let plugins = temporary.join("plugins");
        create_plugin(&plugins, "scan.test", serde_json::json!([]))?;
        let mut registry = PluginRegistry::with_paths(
            temporary.join("registry.json"),
            temporary.join("data"),
            Some(temporary.clone()),
        );
        assert_eq!(
            registry
                .scan_scope_dir(PluginScope::User, &plugins)
                .map_err(|error| error.to_string())?,
            1
        );
        let entry = registry
            .get("scan.test")
            .ok_or_else(|| "scan.test was not discovered".to_string())?;
        assert_eq!(
            entry
                .user_config
                .get("endpoint")
                .and_then(serde_json::Value::as_str),
            Some("https://example.com")
        );
        std::fs::remove_dir_all(temporary).map_err(|error| error.to_string())
    }

    #[test]
    fn resolves_plugin_dependencies_in_version_order() -> Result<(), String> {
        let temporary = temporary_root("dependencies");
        let plugins = temporary.join("plugins");
        create_plugin(&plugins, "base.tools", serde_json::json!([]))?;
        create_plugin(
            &plugins,
            "consumer.tools",
            serde_json::json!([{"name":"base.tools","version":">=1.0.0"}]),
        )?;
        let mut registry = PluginRegistry::with_paths(
            temporary.join("registry.json"),
            temporary.join("data"),
            Some(temporary.clone()),
        );
        registry
            .scan_scope_dir(PluginScope::User, &plugins)
            .map_err(|error| error.to_string())?;
        assert_eq!(
            registry.resolve_enabled_dependencies()?,
            vec!["base.tools".to_string(), "consumer.tools".to_string()]
        );
        std::fs::remove_dir_all(temporary).map_err(|error| error.to_string())
    }

    #[test]
    fn install_rejects_the_removed_yaml_layout() -> Result<(), String> {
        let temporary = temporary_root("legacy");
        let source = temporary.join("legacy");
        std::fs::create_dir_all(source.join(".echo-plugin")).map_err(|error| error.to_string())?;
        std::fs::write(source.join(".echo-plugin/manifest.yaml"), "name: legacy\n")
            .map_err(|error| error.to_string())?;
        let mut registry = PluginRegistry::with_paths(
            temporary.join("registry.json"),
            temporary.join("data"),
            Some(temporary.clone()),
        );
        let error = registry
            .install(&InstallSource::Local(source), PluginScope::Local)
            .err()
            .ok_or_else(|| "legacy layout unexpectedly installed".to_string())?;
        assert!(error.contains("plugin.json"));
        std::fs::remove_dir_all(temporary).map_err(|error| error.to_string())
    }
}

// ── 测试 ────────────────────────────────────────────────────────────────
