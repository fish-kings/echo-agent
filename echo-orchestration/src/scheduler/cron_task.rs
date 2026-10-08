//! Cron 任务定义与持久化。

use chrono::{DateTime, Utc};
use cron::Schedule;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::{Arc, Mutex as StdMutex, OnceLock, Weak};
use tokio::sync::Mutex;
use tracing::debug;

// ── CronTask ───────────────────────────────────────────────────────

/// 按 cron 表达式定时触发的计划任务。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CronTask {
    /// 唯一标识符。
    pub id: String,
    /**
     * 由存储持有的、本代定义的不可变身份标识。
     *
     * `CronTaskStore::add` 总是会替换该值，因此删除后再添加一个完全相同的任务克隆，
     * 也无法复活先前定义的回调。缺少该字段的旧记录在重新添加之前以 `created_at`
     * 作为身份。尚未持久化的新任务在 `CronTaskStore::add` 分配规范值之前保持为空。
     */
    #[serde(default)]
    pub definition_id: String,
    /// 人类可读的名称。
    pub name: String,
    /// 5 字段 cron 表达式：`min hour dom month dow`。
    pub cron_expr: String,
    /// 触发时要执行的提示词或命令。
    pub prompt: String,
    /// 任务是否处于激活状态。
    pub status: CronTaskStatus,
    /**
     * 持久化的控制修订号，每次状态控制操作都会递增。
     *
     * 计划触发实例会捕获该令牌，因此“先禁用再启用”不会放行在任一控制操作提交之前
     * 已被选中的工作。
     */
    #[serde(default)]
    pub control_revision: u64,
    /// 最近一次执行的 ISO 8601 时间戳。
    pub last_run_at: Option<String>,
    /// 最近一次执行的截断结果（前 500 个字符）。
    pub last_result: Option<String>,
    /// 创建时间的 ISO 8601 时间戳。
    pub created_at: String,
}

/// cron 任务的激活状态。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CronTaskStatus {
    /// 任务按计划触发。
    Enabled,
    /// 任务已暂停。
    Disabled,
}

impl CronTask {
    /// 创建一个处于启用状态的新 cron 任务。
    pub fn new(name: &str, cron_expr: &str, prompt: &str) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            definition_id: String::new(),
            name: name.to_string(),
            cron_expr: cron_expr.to_string(),
            prompt: prompt.to_string(),
            status: CronTaskStatus::Enabled,
            control_revision: 0,
            last_run_at: None,
            last_result: None,
            created_at: echo_core::utils::time::now_local().to_rfc3339(),
        }
    }

    /**
     * 计算当前时刻之后的下一次触发时间。
     *
     * 若 cron 表达式无效或不存在未来的触发时间，则返回 `None`。
     */
    pub fn next_run(&self) -> Option<DateTime<Utc>> {
        self.next_run_after(&Utc::now())
    }

    /// 计算严格晚于调用方给定边界的第一次触发时间。
    pub fn next_run_after(&self, after: &DateTime<Utc>) -> Option<DateTime<Utc>> {
        // cron crate 需要 7 字段表达式；补齐秒和年字段
        let expr = if self.cron_expr.split_whitespace().count() == 5 {
            format!("0 {} *", self.cron_expr)
        } else {
            self.cron_expr.clone()
        };
        let schedule = Schedule::from_str(&expr).ok()?;
        schedule.after(after).next()
    }

    /// 校验 cron 表达式是否合法。
    pub fn validate_cron(&self) -> bool {
        let expr = if self.cron_expr.split_whitespace().count() == 5 {
            format!("0 {} *", self.cron_expr)
        } else {
            self.cron_expr.clone()
        };
        Schedule::from_str(&expr).is_ok()
    }

    /// 判断两个任务是否属于同一代定义（ID 相同且定义身份一致）。
    pub(crate) fn same_definition(&self, other: &Self) -> bool {
        if self.id != other.id {
            return false;
        }
        if self.definition_id.is_empty() || other.definition_id.is_empty() {
            return self.definition_id.is_empty()
                && other.definition_id.is_empty()
                && self.created_at == other.created_at;
        }
        self.definition_id == other.definition_id
    }
}

// ── CronTaskStore ──────────────────────────────────────────────────

/**
 * cron 任务定义的持久化存储。
 *
 * 支持两种后端：
 * 1. **Store trait**（SQLite/InMemory）—— 推荐
 * 2. **基于文件** —— 旧版 JSON 文件回退方案
 *
 * [`super::SchedulerRunner`] 会根据该存储的定义路径推导触发实例日志与检查点路径。
 * 基于 Store 的定义在构造 runner 之前必须通过 [`Self::with_path`] 设置稳定的路径锚点。
 */
#[derive(Clone)]
pub struct CronTaskStore {
    /// 可选的 Store 后端；为 `None` 时使用文件后端。
    backend: Option<Arc<dyn echo_core::memory::Store>>,
    /// 定义文件路径（同时作为触发实例日志的路径锚点）。
    path: PathBuf,
    /// 进程内全局共享的变更互斥锁，串行化所有读-改-写操作。
    mutation_lock: Arc<Mutex<()>>,
    /// 按路径共享的健康状态（含权威性中毒标记）。
    health: Arc<CronTaskStoreHealth>,
}

/// 按定义路径共享的存储健康状态。
struct CronTaskStoreHealth {
    /// 权威性中毒原因；为 `Some` 时拒绝一切读写，直到恢复。
    poison: StdMutex<Option<String>>,
    /// 测试用：在可见替换之后注入持久化失败。
    #[cfg(test)]
    file_after_replace_fault: Arc<std::sync::atomic::AtomicBool>,
    /// 测试用：在对账读取时注入读取失败。
    #[cfg(test)]
    file_reconcile_read_fault: Arc<std::sync::atomic::AtomicBool>,
}

impl CronTaskStoreHealth {
    /// 创建未中毒的健康状态。
    fn new() -> Self {
        Self {
            poison: StdMutex::new(None),
            #[cfg(test)]
            file_after_replace_fault: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            #[cfg(test)]
            file_reconcile_read_fault: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }
}

/// Store 后端中保存 cron 任务的命名空间。
const STORE_NAMESPACE: &[&str] = &["scheduler", "cron_tasks"];
/// Store 后端中保存全部 cron 任务的键。
const STORE_KEY: &str = "all_cron_tasks";

/// 返回进程级共享的 cron 存储变更锁。
fn cron_store_mutation_lock() -> Arc<Mutex<()>> {
    static LOCK: OnceLock<Arc<Mutex<()>>> = OnceLock::new();
    Arc::clone(LOCK.get_or_init(|| Arc::new(Mutex::new(()))))
}

/// 返回按规范化路径索引的健康状态注册表（弱引用）。
fn cron_store_health_registry() -> &'static StdMutex<HashMap<PathBuf, Weak<CronTaskStoreHealth>>> {
    static REGISTRY: OnceLock<StdMutex<HashMap<PathBuf, Weak<CronTaskStoreHealth>>>> =
        OnceLock::new();
    REGISTRY.get_or_init(|| StdMutex::new(HashMap::new()))
}

/// 计算路径的规范化键，即使父目录尚不存在也能保持稳定。
fn cron_store_health_key(path: &Path) -> PathBuf {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(path)
    };
    let Some(file_name) = absolute.file_name().map(|name| name.to_os_string()) else {
        return absolute;
    };
    let Some(parent) = absolute.parent() else {
        return absolute;
    };
    let mut cursor = parent.to_path_buf();
    let mut missing = Vec::new();
    loop {
        match std::fs::canonicalize(&cursor) {
            Ok(mut canonical) => {
                for component in missing.into_iter().rev() {
                    canonical.push(component);
                }
                canonical.push(&file_name);
                return canonical;
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                let Some(component) = cursor.file_name().map(|name| name.to_os_string()) else {
                    return absolute;
                };
                missing.push(component);
                let Some(next) = cursor.parent() else {
                    return absolute;
                };
                cursor = next.to_path_buf();
            }
            Err(_) => return absolute,
        }
    }
}

/// 获取或创建指定路径共享的健康状态，并清理已失效的条目。
fn cron_store_health(path: &Path) -> Arc<CronTaskStoreHealth> {
    let key = cron_store_health_key(path);
    let mut registry = cron_store_health_registry()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    registry.retain(|_, health| health.strong_count() > 0);
    if let Some(health) = registry.get(&key).and_then(Weak::upgrade) {
        return health;
    }
    let health = Arc::new(CronTaskStoreHealth::new());
    registry.insert(key, Arc::downgrade(&health));
    health
}

impl CronTaskStore {
    /// 创建基于文件的存储（默认路径：`~/.echo-agent/scheduler/tasks.json`）。
    pub fn new() -> Self {
        let path = default_task_path();
        Self {
            backend: None,
            health: cron_store_health(&path),
            path,
            mutation_lock: cron_store_mutation_lock(),
        }
    }

    /// 创建基于 Store 后端的存储，并自动从旧文件迁移数据。
    pub async fn with_store(
        store: Arc<dyn echo_core::memory::Store>,
    ) -> echo_core::error::Result<Self> {
        let s = Self {
            backend: Some(store),
            path: PathBuf::new(),
            mutation_lock: cron_store_mutation_lock(),
            health: Arc::new(CronTaskStoreHealth::new()),
        };
        s.migrate_from_file().await?;
        Ok(s)
    }

    /**
     * 设置自定义的定义文件路径，并将触发实例的持久化文件放在同一位置。
     *
     * 对于基于 Store 的实例，Store 仍是定义的权威来源；该路径仅用于选择相邻的
     * 触发实例日志位置。
     */
    pub fn with_path(mut self, path: PathBuf) -> Self {
        self.health = cron_store_health(&path);
        self.path = path;
        self
    }

    /// 测试用：同时指定 Store 后端与路径构造存储。
    #[cfg(test)]
    pub(crate) fn with_backend_path_for_test(
        backend: Arc<dyn echo_core::memory::Store>,
        path: PathBuf,
    ) -> Self {
        Self {
            backend: Some(backend),
            health: cron_store_health(&path),
            path,
            mutation_lock: cron_store_mutation_lock(),
        }
    }

    /// 测试用：注入“可见替换后持久化失败”故障。
    #[cfg(test)]
    pub(crate) fn inject_file_after_replace_fault(&self) {
        self.health
            .file_after_replace_fault
            .store(true, std::sync::atomic::Ordering::Release);
    }

    /// 测试用：注入“可见替换后持久化失败且对账读取失败”故障。
    #[cfg(test)]
    pub(crate) fn inject_file_after_replace_reconcile_read_fault(&self) {
        self.health
            .file_after_replace_fault
            .store(true, std::sync::atomic::Ordering::Release);
        self.health
            .file_reconcile_read_fault
            .store(true, std::sync::atomic::Ordering::Release);
    }

    /// 返回触发实例日志与检查点文件的路径 `(journal, checkpoint)`。
    pub(crate) fn occurrence_paths(&self) -> echo_core::error::Result<(PathBuf, PathBuf)> {
        if self.backend.is_some() && self.path.as_os_str().is_empty() {
            return Err(echo_core::error::ReactError::Other(
                "Store-backed CronTaskStore requires with_path() to select a stable occurrence journal identity"
                    .to_string(),
            ));
        }
        let definition_path = if self.path.as_os_str().is_empty() {
            default_task_path()
        } else {
            self.path.clone()
        };
        Ok((
            path_with_suffix(&definition_path, ".occurrences.jsonl")?,
            path_with_suffix(&definition_path, ".occurrences.checkpoint.json")?,
        ))
    }

    /// 加载全部 cron 任务。
    pub async fn load_all(&self) -> echo_core::error::Result<Vec<CronTask>> {
        self.check_authority()?;
        self.load_all_unchecked().await
    }

    /// 在不检查权威性中毒状态的情况下加载全部任务。
    async fn load_all_unchecked(&self) -> echo_core::error::Result<Vec<CronTask>> {
        if let Some(ref backend) = self.backend {
            let item = backend.get(STORE_NAMESPACE, STORE_KEY).await?;
            match item {
                Some(store_item) => {
                    // 值以 serde_json::Value 存储——提取其中的字符串
                    let json_str = match &store_item.value {
                        serde_json::Value::String(s) => s.clone(),
                        other => other.to_string(),
                    };
                    let tasks: Vec<CronTask> = serde_json::from_str(&json_str).map_err(|e| {
                        echo_core::error::ReactError::Other(format!(
                            "Failed to deserialize cron tasks: {e}"
                        ))
                    })?;
                    validate_task_ids(&tasks)?;
                    Ok(tasks)
                }
                None => Ok(vec![]),
            }
        } else {
            self.load_from_file()
        }
    }

    /// 保存全部 cron 任务（调用方需已持有变更锁）。
    async fn save_all_unlocked(&self, tasks: &[CronTask]) -> echo_core::error::Result<()> {
        self.check_authority()?;
        validate_task_ids(tasks)?;
        let json = serde_json::to_string_pretty(tasks).map_err(|e| {
            echo_core::error::ReactError::Other(format!("Failed to serialize cron tasks: {e}"))
        })?;

        if let Some(ref backend) = self.backend {
            if let Err(error) = backend
                .put(STORE_NAMESPACE, STORE_KEY, serde_json::Value::String(json))
                .await
            {
                if matches!(
                    &error,
                    echo_core::error::ReactError::Memory(memory)
                        if matches!(
                            memory.as_ref(),
                            echo_core::error::MemoryError::TransientNoCommit(_)
                        )
                ) {
                    return Err(error);
                }
                return Err(self.poison_authority(format!(
                    "Store backend mutation returned an uncertain result: {error}"
                )));
            }
        } else {
            self.save_to_file(&json)?;
        }
        Ok(())
    }

    /**
     * 以全新的定义代添加任务并持久化。
     *
     * 存储会替换调用方提供的 `definition_id`，并返回实际提交的定义，
     * 使调用方无需再次进行可能失败的后端读取即可更新投影。
     */
    pub async fn add(&self, mut task: CronTask) -> echo_core::error::Result<CronTask> {
        unique_id(&task.id)?;
        let _guard = self.mutation_lock.lock().await;
        let mut tasks = self.load_all().await?;
        if tasks.iter().any(|existing| existing.id == task.id) {
            return Err(echo_core::error::ReactError::Other(format!(
                "Cron task ID '{}' already exists",
                task.id
            )));
        }
        task.definition_id = uuid::Uuid::new_v4().to_string();
        tasks.push(task.clone());
        self.save_all_unlocked(&tasks).await?;
        Ok(task)
    }

    /// 按 ID 删除任务并持久化；找到时返回 true。
    pub async fn remove(&self, id: &str) -> echo_core::error::Result<bool> {
        Ok(self.remove_with_snapshot(id).await?.is_some())
    }

    /// 删除任务并返回实际提交后的定义集合，用于更新内存视图。
    pub(crate) async fn remove_with_snapshot(
        &self,
        id: &str,
    ) -> echo_core::error::Result<Option<Vec<CronTask>>> {
        let id = unique_id(id)?;
        let _guard = self.mutation_lock.lock().await;
        let mut tasks = self.load_all().await?;
        let before = tasks.len();
        tasks.retain(|task| task.id != id);
        let removed = tasks.len() < before;
        if removed {
            self.save_all_unlocked(&tasks).await?;
        }
        Ok(removed.then_some(tasks))
    }

    /// 按完整 ID 精确删除一个任务并持久化。
    pub async fn remove_exact(&self, id: &str) -> echo_core::error::Result<bool> {
        self.remove(id).await
    }

    /// 按 ID 更新任务状态并持久化。
    pub async fn set_status(
        &self,
        id: &str,
        status: CronTaskStatus,
    ) -> echo_core::error::Result<bool> {
        Ok(self.set_status_with_snapshot(id, status).await?.is_some())
    }

    /// 更新任务状态、递增控制修订号，并返回提交后的定义集合。
    pub(crate) async fn set_status_with_snapshot(
        &self,
        id: &str,
        status: CronTaskStatus,
    ) -> echo_core::error::Result<Option<Vec<CronTask>>> {
        let id = unique_id(id)?;
        let _guard = self.mutation_lock.lock().await;
        let mut tasks = self.load_all().await?;
        let mut found = false;
        for task in &mut tasks {
            if task.id == id {
                let next_revision = task.control_revision.checked_add(1).ok_or_else(|| {
                    echo_core::error::ReactError::Other(format!(
                        "Cron task '{id}' control revision is exhausted"
                    ))
                })?;
                task.status = status;
                task.control_revision = next_revision;
                found = true;
                break;
            }
        }
        if found {
            self.save_all_unlocked(&tasks).await?;
        }
        Ok(found.then_some(tasks))
    }

    /// 任务触发后更新 last_run 信息。
    pub async fn update_last_run(&self, id: &str, result: &str) -> echo_core::error::Result<()> {
        let id = unique_id(id)?;
        let task = self.get(id).await?.ok_or_else(|| {
            echo_core::error::ReactError::Other(format!("Cron task '{id}' not found"))
        })?;
        self.update_last_run_for_task(&task, result)
            .await
            .map(|_| ())
    }

    /**
     * 仅针对捕获到的那一代任务定义更新最近运行信息。
     *
     * 由存储持有的 `definition_id` 可防止旧定义的回调去更新一个已被删除、
     * 又以相同公开 ID 重新创建的任务。旧记录以 `created_at` 作为身份。
     */
    pub(crate) async fn update_last_run_for_task(
        &self,
        expected: &CronTask,
        result: &str,
    ) -> echo_core::error::Result<CronTask> {
        let id = unique_id(&expected.id)?;
        let _guard = self.mutation_lock.lock().await;
        let mut tasks = self.load_all().await?;
        let updated = tasks.iter_mut().find_map(|task| {
            if task.same_definition(expected) {
                task.last_run_at = Some(echo_core::utils::time::now_local().to_rfc3339());
                task.last_result = Some(result.chars().take(500).collect());
                Some(task.clone())
            } else {
                None
            }
        });
        let Some(updated) = updated else {
            return Err(echo_core::error::ReactError::Other(format!(
                "Cron task '{id}' definition is no longer current"
            )));
        };
        self.save_all_unlocked(&tasks).await?;
        Ok(updated)
    }

    /// 按唯一 ID 获取任务。
    pub async fn get(&self, id: &str) -> echo_core::error::Result<Option<CronTask>> {
        let id = unique_id(id)?;
        let tasks = self.load_all().await?;
        Ok(tasks.into_iter().find(|task| task.id == id))
    }

    // ── 私有辅助方法 ────────────────────────────────────────────

    /// 检查定义权威性是否已中毒；中毒时返回错误。
    pub(crate) fn check_authority(&self) -> echo_core::error::Result<()> {
        let poison = self
            .health
            .poison
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if let Some(reason) = poison.as_ref() {
            return Err(echo_core::error::ReactError::Other(format!(
                "Cron task definition authority is poisoned: {reason}; recover the backend and reload before executing callbacks"
            )));
        }
        Ok(())
    }

    /// 返回定义权威性是否已中毒。
    pub(crate) fn is_authority_poisoned(&self) -> bool {
        self.health
            .poison
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .is_some()
    }

    /// 重新加载后端并清除中毒标记，返回恢复后的任务列表。
    pub(crate) async fn recover_authority(&self) -> echo_core::error::Result<Vec<CronTask>> {
        let _guard = self.mutation_lock.lock().await;
        let tasks = self.load_all_unchecked().await?;
        *self
            .health
            .poison
            .lock()
            .unwrap_or_else(|error| error.into_inner()) = None;
        Ok(tasks)
    }

    /// 标记权威性中毒并返回相应错误。
    fn poison_authority(&self, reason: String) -> echo_core::error::ReactError {
        *self
            .health
            .poison
            .lock()
            .unwrap_or_else(|error| error.into_inner()) = Some(reason.clone());
        echo_core::error::ReactError::Other(format!(
            "Cron task definition authority became ambiguous: {reason}; recover the backend and reload before executing callbacks"
        ))
    }

    /// 从定义文件加载任务；文件不存在或为空时返回空列表。
    fn load_from_file(&self) -> echo_core::error::Result<Vec<CronTask>> {
        if !self.path.exists() {
            return Ok(vec![]);
        }
        let content = std::fs::read_to_string(&self.path).map_err(|e| {
            echo_core::error::ReactError::Other(format!("Failed to read cron tasks file: {e}"))
        })?;
        if content.trim().is_empty() {
            return Ok(vec![]);
        }
        let tasks: Vec<CronTask> = serde_json::from_str(&content).map_err(|e| {
            echo_core::error::ReactError::Other(format!("Failed to parse cron tasks: {e}"))
        })?;
        validate_task_ids(&tasks)?;
        Ok(tasks)
    }

    /// 原子写入定义文件；写入结果不确定时通过对账判断是否需要中毒。
    fn save_to_file(&self, json: &str) -> echo_core::error::Result<()> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                echo_core::error::ReactError::Other(format!(
                    "Failed to create scheduler directory: {e}"
                ))
            })?;
        }
        let prior_bytes = match std::fs::read(&self.path) {
            Ok(bytes) => Some(bytes),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(error) => {
                return Err(echo_core::error::ReactError::Other(format!(
                    "Failed to read prior cron tasks file: {error}"
                )));
            }
        };
        let candidate = json.as_bytes();
        let write_result = echo_core::utils::fs::atomic_write(&self.path, candidate);
        #[cfg(test)]
        let write_result = match write_result {
            Ok(())
                if self
                    .health
                    .file_after_replace_fault
                    .swap(false, std::sync::atomic::Ordering::AcqRel) =>
            {
                Err(std::io::Error::other(
                    "injected parent-directory sync failure after visible replace",
                ))
            }
            result => result,
        };
        match write_result {
            Ok(()) => Ok(()),
            Err(write_error) => {
                #[cfg(test)]
                let observed = if self
                    .health
                    .file_reconcile_read_fault
                    .swap(false, std::sync::atomic::Ordering::AcqRel)
                {
                    Err(std::io::Error::other(
                        "injected reconciliation read failure after visible replace",
                    ))
                } else {
                    std::fs::read(&self.path)
                };
                #[cfg(not(test))]
                let observed = std::fs::read(&self.path);
                match observed {
                Ok(observed) if observed == candidate => {
                    tracing::warn!(
                        path = %self.path.display(),
                        error = %write_error,
                        "Cron task candidate is visible despite a degraded durability result"
                    );
                    Ok(())
                }
                Ok(observed) if prior_bytes.as_ref() == Some(&observed) => {
                    Err(echo_core::error::ReactError::Other(format!(
                        "Failed to write cron tasks file: {write_error}"
                    )))
                }
                Err(error)
                    if error.kind() == std::io::ErrorKind::NotFound && prior_bytes.is_none() =>
                {
                    Err(echo_core::error::ReactError::Other(format!(
                        "Failed to write cron tasks file: {write_error}"
                    )))
                }
                    Ok(observed) => Err(self.poison_authority(format!(
                        "persistence failed ({write_error}) and disk contains {} unreconciled bytes",
                        observed.len()
                    ))),
                    Err(error) => Err(self.poison_authority(format!(
                        "persistence failed ({write_error}) and disk reconciliation failed ({error})"
                    ))),
                }
            }
        }
    }

    /// 将旧版文件中的任务迁移到 Store 后端（目标已存在时保留旧文件不迁移）。
    async fn migrate_from_file(&self) -> echo_core::error::Result<()> {
        let _guard = self.mutation_lock.lock().await;
        let Some(backend) = self.backend.as_ref() else {
            return Ok(());
        };
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        let legacy_path = PathBuf::from(home).join(".echo-agent/scheduler/tasks.json");
        if !legacy_path.exists() {
            return Ok(());
        }

        /* 已存在的目标数据即为权威来源，即使其中存储的是空列表。
        绝不能用旧版快照替换它。 */
        if backend.get(STORE_NAMESPACE, STORE_KEY).await?.is_some() {
            let _ = self.load_all().await?;
            debug!("Scheduler store already contains cron tasks; preserving legacy file");
            return Ok(());
        }

        debug!("Migrating cron tasks from file to Store backend");
        let content = std::fs::read_to_string(&legacy_path).map_err(|error| {
            echo_core::error::ReactError::Other(format!(
                "Failed to read legacy cron tasks: {error}"
            ))
        })?;
        if content.trim().is_empty() {
            return Ok(());
        }
        let tasks: Vec<CronTask> = serde_json::from_str(&content).map_err(|error| {
            echo_core::error::ReactError::Other(format!(
                "Failed to parse legacy cron tasks: {error}"
            ))
        })?;
        validate_task_ids(&tasks)?;
        self.save_all_unlocked(&tasks).await?;
        // 迁移成功后删除旧文件
        let _ = std::fs::remove_file(&legacy_path);
        debug!(
            "Migrated {} cron tasks and removed legacy file",
            tasks.len()
        );
        Ok(())
    }
}

/// 校验任务 ID 非空并原样返回。
fn unique_id(id: &str) -> echo_core::error::Result<&str> {
    if id.trim().is_empty() {
        return Err(echo_core::error::ReactError::Other(
            "Cron task ID cannot be empty".into(),
        ));
    }
    Ok(id)
}

/// 返回默认的任务定义文件路径。
fn default_task_path() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(".echo-agent/scheduler/tasks.json")
}

/// 在路径文件名后追加后缀，生成相邻文件路径。
fn path_with_suffix(path: &std::path::Path, suffix: &str) -> echo_core::error::Result<PathBuf> {
    let file_name = path.file_name().ok_or_else(|| {
        echo_core::error::ReactError::Other(format!(
            "Cron task store path '{}' has no file name",
            path.display()
        ))
    })?;
    let mut suffixed = file_name.to_os_string();
    suffixed.push(suffix);
    Ok(path.with_file_name(suffixed))
}

/// 校验任务 ID 均非空且互不重复。
fn validate_task_ids(tasks: &[CronTask]) -> echo_core::error::Result<()> {
    let mut seen = HashSet::with_capacity(tasks.len());
    for task in tasks {
        unique_id(&task.id)?;
        if !seen.insert(task.id.as_str()) {
            return Err(echo_core::error::ReactError::Other(format!(
                "Duplicate cron task ID '{}'",
                task.id
            )));
        }
    }
    Ok(())
}

impl Default for CronTaskStore {
    /// 等价于 [`CronTaskStore::new`]。
    fn default() -> Self {
        Self::new()
    }
}

// ── 测试 ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// 新建任务的默认字段正确。
    #[test]
    fn test_cron_task_creation() {
        let task = CronTask::new("test", "*/5 * * * *", "Hello");
        assert_eq!(task.name, "test");
        assert!(task.definition_id.is_empty());
        assert_eq!(task.status, CronTaskStatus::Enabled);
        assert_eq!(task.control_revision, 0);
        assert!(task.validate_cron());
    }

    /// 旧记录缺失字段时控制修订号默认为 0。
    #[test]
    fn legacy_task_defaults_control_revision_to_zero() -> Result<(), String> {
        let task: CronTask = serde_json::from_value(serde_json::json!({
            "id": "legacy",
            "name": "legacy",
            "cron_expr": "*/5 * * * *",
            "prompt": "run",
            "status": "enabled",
            "last_run_at": null,
            "last_result": null,
            "created_at": "2026-01-01T00:00:00Z"
        }))
        .map_err(|error| error.to_string())?;
        assert!(task.definition_id.is_empty());
        assert_eq!(task.control_revision, 0);
        Ok(())
    }

    /// 触发实例路径保留完整的定义文件名。
    #[test]
    fn occurrence_paths_preserve_the_complete_definition_file_name() -> Result<(), String> {
        let root = std::env::temp_dir().join("echo-scheduler-path-identity");
        let json = CronTaskStore::new().with_path(root.join("tasks.json"));
        let toml = CronTaskStore::new().with_path(root.join("tasks.toml"));
        let (json_journal, json_checkpoint) =
            json.occurrence_paths().map_err(|error| error.to_string())?;
        let (toml_journal, toml_checkpoint) =
            toml.occurrence_paths().map_err(|error| error.to_string())?;

        assert_ne!(json_journal, toml_journal);
        assert_ne!(json_checkpoint, toml_checkpoint);
        assert_eq!(
            json_journal.file_name().and_then(|name| name.to_str()),
            Some("tasks.json.occurrences.jsonl")
        );
        Ok(())
    }

    /// 中毒状态按定义路径共享。
    #[test]
    fn definition_path_keys_shared_poison_authority() {
        let root = std::env::temp_dir().join(format!(
            "echo-scheduler-poison-identity-{}",
            uuid::Uuid::new_v4()
        ));
        let path_a = root.join("a.json");
        let path_b = root.join("b.json");
        let base = CronTaskStore::new();
        let store_a = base.clone().with_path(path_a.clone());
        let same_a = store_a.clone().with_path(path_a);
        let store_b = base.with_path(path_b);

        let _ = store_a.poison_authority("injected ambiguity".to_string());
        assert!(store_a.check_authority().is_err());
        assert!(same_a.check_authority().is_err());
        assert!(store_b.check_authority().is_ok());
    }

    /// 独立配置的克隆在同一路径上共享中毒状态。
    #[test]
    fn independently_configured_clones_share_same_path_poison() {
        let root = std::env::temp_dir().join(format!(
            "echo-scheduler-shared-poison-{}",
            uuid::Uuid::new_v4()
        ));
        let path = root.join("tasks.json");
        let base = CronTaskStore::new();
        let first = base.clone().with_path(path.clone());
        let second = base.with_path(path);

        let _ = first.poison_authority("injected ambiguity".to_string());
        assert!(first.check_authority().is_err());
        assert!(second.check_authority().is_err());
    }

    /// 父目录创建前后健康状态键保持稳定。
    #[test]
    fn health_key_is_stable_before_and_after_parent_creation() -> Result<(), String> {
        let root = std::env::temp_dir().join(format!(
            "echo-scheduler-health-key-{}",
            uuid::Uuid::new_v4()
        ));
        let parent = root.join("nested");
        let path = parent.join("tasks.json");
        std::fs::create_dir_all(&root).map_err(|error| error.to_string())?;
        let before = CronTaskStore::new().with_path(path.clone());
        std::fs::create_dir_all(&parent).map_err(|error| error.to_string())?;
        let after = CronTaskStore::new().with_path(path);

        let _ = before.poison_authority("injected ambiguity".to_string());
        assert!(after.check_authority().is_err());
        std::fs::remove_dir_all(root).map_err(|error| error.to_string())?;
        Ok(())
    }

    /// 健康注册表会清理已失效的路径条目。
    #[test]
    fn health_registry_prunes_dead_path_entries() {
        let root = std::env::temp_dir().join(format!(
            "echo-scheduler-health-prune-{}",
            uuid::Uuid::new_v4()
        ));
        let stale_path = root.join("stale.json");
        let stale_key = cron_store_health_key(&stale_path);
        let stale = cron_store_health(&stale_path);
        drop(stale);

        let live = cron_store_health(&root.join("live.json"));
        let registry = cron_store_health_registry()
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        assert!(!registry.contains_key(&stale_key));
        drop(registry);
        drop(live);
    }

    /// 状态控制操作会递增持久化修订号。
    #[tokio::test]
    async fn status_controls_increment_the_durable_revision() -> Result<(), String> {
        let temp = std::env::temp_dir().join(format!(
            "echo-scheduler-control-revision-{}",
            uuid::Uuid::new_v4()
        ));
        let store = CronTaskStore::new().with_path(temp.join("tasks.json"));
        let task = CronTask::new("controlled", "*/5 * * * *", "run");
        let task_id = task.id.clone();
        store.add(task).await.map_err(|error| error.to_string())?;
        assert!(
            store
                .set_status(&task_id, CronTaskStatus::Disabled)
                .await
                .map_err(|error| error.to_string())?
        );
        assert!(
            store
                .set_status(&task_id, CronTaskStatus::Enabled)
                .await
                .map_err(|error| error.to_string())?
        );
        let current = store
            .get(&task_id)
            .await
            .map_err(|error| error.to_string())?
            .ok_or_else(|| "controlled task missing".to_string())?;
        assert_eq!(current.control_revision, 2);
        std::fs::remove_dir_all(&temp).map_err(|error| error.to_string())?;
        Ok(())
    }

    /// 重新添加完全相同的克隆会获得新的定义代。
    #[tokio::test]
    async fn add_replaces_an_exact_clones_definition_incarnation() -> Result<(), String> {
        let temp = std::env::temp_dir().join(format!(
            "echo-scheduler-definition-incarnation-{}",
            uuid::Uuid::new_v4()
        ));
        let store = CronTaskStore::new().with_path(temp.join("tasks.json"));
        let task = CronTask::new("incarnation", "*/5 * * * *", "run");
        let task_id = task.id.clone();
        store.add(task).await.map_err(|error| error.to_string())?;
        let first = store
            .get(&task_id)
            .await
            .map_err(|error| error.to_string())?
            .ok_or_else(|| "first definition missing".to_string())?;
        assert!(!first.definition_id.is_empty());
        assert!(
            store
                .remove_exact(&task_id)
                .await
                .map_err(|error| error.to_string())?
        );
        store
            .add(first.clone())
            .await
            .map_err(|error| error.to_string())?;
        let second = store
            .get(&task_id)
            .await
            .map_err(|error| error.to_string())?
            .ok_or_else(|| "second definition missing".to_string())?;
        assert_ne!(first.definition_id, second.definition_id);
        assert!(!first.same_definition(&second));
        std::fs::remove_dir_all(&temp).map_err(|error| error.to_string())?;
        Ok(())
    }

    /// 合法表达式能计算出下一次触发时间。
    #[test]
    fn test_cron_task_next_run() {
        let task = CronTask::new("test", "0 12 * * *", "Noon task");
        let next = task.next_run();
        assert!(next.is_some());
    }

    /// 非法表达式校验失败且无下一次触发时间。
    #[test]
    fn test_cron_task_invalid_expr() {
        let task = CronTask::new("bad", "not a cron", "test");
        assert!(!task.validate_cron());
        assert!(task.next_run().is_none());
    }

    /// 精确删除不会误删同前缀的任务。
    #[tokio::test]
    async fn remove_exact_does_not_remove_tasks_with_the_same_prefix() -> Result<(), String> {
        let temp = std::env::temp_dir().join(format!(
            "echo-scheduler-remove-exact-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&temp).map_err(|error| error.to_string())?;
        let store = CronTaskStore::new().with_path(temp.join("cron-tasks.json"));
        let mut exact = CronTask::new("exact", "*/5 * * * *", "exact");
        exact.id = "plugin-monitor".to_string();
        let mut prefixed = CronTask::new("prefixed", "*/5 * * * *", "prefixed");
        prefixed.id = "plugin-monitor-longer".to_string();
        store.add(exact).await.map_err(|error| error.to_string())?;
        store
            .add(prefixed)
            .await
            .map_err(|error| error.to_string())?;

        assert!(
            store
                .remove_exact("plugin-monitor")
                .await
                .map_err(|error| error.to_string())?
        );
        let remaining = store.load_all().await.map_err(|error| error.to_string())?;
        assert_eq!(remaining.len(), 1);
        assert_eq!(
            remaining.first().map(|task| task.id.as_str()),
            Some("plugin-monitor-longer")
        );
        std::fs::remove_dir_all(&temp).map_err(|error| error.to_string())?;
        Ok(())
    }

    /// 独立的文件存储实例并发添加不会丢失数据。
    #[tokio::test]
    async fn independent_file_store_instances_do_not_lose_concurrent_adds() -> Result<(), String> {
        let temp = std::env::temp_dir().join(format!(
            "echo-scheduler-concurrent-add-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&temp).map_err(|error| error.to_string())?;
        let path = temp.join("cron-tasks.json");
        let first = CronTaskStore::new().with_path(path.clone());
        let second = CronTaskStore::new().with_path(path);

        let (first_result, second_result) = tokio::join!(
            first.add(CronTask::new("first", "*/5 * * * *", "first")),
            second.add(CronTask::new("second", "*/5 * * * *", "second")),
        );
        first_result.map_err(|error| error.to_string())?;
        second_result.map_err(|error| error.to_string())?;

        let tasks = first.load_all().await.map_err(|error| error.to_string())?;
        assert_eq!(tasks.len(), 2);
        assert!(tasks.iter().any(|task| task.name == "first"));
        assert!(tasks.iter().any(|task| task.name == "second"));
        std::fs::remove_dir_all(&temp).map_err(|error| error.to_string())?;
        Ok(())
    }

    /// 添加时拒绝重复的任务 ID。
    #[tokio::test]
    async fn add_rejects_duplicate_task_ids() -> Result<(), String> {
        let temp = std::env::temp_dir().join(format!(
            "echo-scheduler-duplicate-id-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&temp).map_err(|error| error.to_string())?;
        let store = CronTaskStore::new().with_path(temp.join("cron-tasks.json"));
        let first = CronTask::new("first", "*/5 * * * *", "first");
        let mut duplicate = CronTask::new("second", "*/5 * * * *", "second");
        duplicate.id = first.id.clone();
        store.add(first).await.map_err(|error| error.to_string())?;
        assert!(store.add(duplicate).await.is_err());
        assert_eq!(
            store
                .load_all()
                .await
                .map_err(|error| error.to_string())?
                .len(),
            1
        );
        std::fs::remove_dir_all(&temp).map_err(|error| error.to_string())?;
        Ok(())
    }

    /// 加载时拒绝持久化文件中重复的任务 ID。
    #[tokio::test]
    async fn load_rejects_duplicate_task_ids_in_persisted_file() -> Result<(), String> {
        let temp = std::env::temp_dir().join(format!(
            "echo-scheduler-duplicate-file-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&temp).map_err(|error| error.to_string())?;
        let path = temp.join("cron-tasks.json");
        let mut first = CronTask::new("first", "*/5 * * * *", "first");
        let mut duplicate = CronTask::new("second", "*/5 * * * *", "second");
        duplicate.id = first.id.clone();
        first.created_at = "2026-01-01T00:00:00Z".to_string();
        duplicate.created_at = "2026-01-01T00:00:01Z".to_string();
        let payload =
            serde_json::to_string(&vec![first, duplicate]).map_err(|error| error.to_string())?;
        std::fs::write(&path, payload).map_err(|error| error.to_string())?;
        let store = CronTaskStore::new().with_path(path);
        assert!(store.load_all().await.is_err());
        std::fs::remove_dir_all(&temp).map_err(|error| error.to_string())?;
        Ok(())
    }
}
