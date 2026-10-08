/*! Curator —— 技能生命周期管理。

管理技能的完整生命周期：`Candidate → Draft → Active → Stale → Deprecated → Archived`。
仅对 agent 创建的技能生效（从不处理内置/外部技能）。
从不自动删除，只会归档。

# 生命周期状态

| 状态 | 含义 | 自动转换 |
|-------|---------|-----------------|
| `Candidate` | 从记忆中发现的模式，尚未正式化 | → Draft（经过审查后）|
| `Draft` | 已创建 SKILL.md 但尚未激活 | → Active（经过审查/使用后）|
| `Active` | 正在使用中，出现在技能目录中 | → Stale（经过一段不活跃期后）|
| `Stale` | 近期未使用，可能已过时 | → Deprecated（经过更长不活跃期后）|
| `Deprecated` | 已被替代或已知过时，仍可访问 | → Archived（经过更长不活跃期后）|
| `Archived` | 已从所有路径中移除，仅供参考 | 终态 |

灵感来自 Hermes Agent 的 curator 系统。
*/

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::error::Result;
use fs2::FileExt;

/// 在路径末尾追加后缀，得到派生的私有文件路径。
fn append_path_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut suffixed = path.as_os_str().to_os_string();
    suffixed.push(suffix);
    PathBuf::from(suffixed)
}

// ── SkillLifecycle ─────────────────────────────────────────────────

/// 技能的生命周期状态。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SkillLifecycle {
    /// 从记忆中发现的模式，尚未正式化。还没有 SKILL.md 文件。
    Candidate,
    /// 已创建内容极简的 SKILL.md，尚未注入技能目录。
    Draft,
    /// 正在被积极使用且相关。出现在技能目录中。
    Active,
    /// 近期未被使用，可能已过时。
    Stale,
    /// 已被其他技能替代或已知过时。仍可访问但不在目录中。
    Deprecated,
    /// 已不再相关，仅保留作参考。
    Archived,
}

// ── SkillMeta ──────────────────────────────────────────────────────

/// curator 为每个技能追踪的元数据。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SkillMeta {
    /// 技能名称。
    pub name: String,
    /// 具体的 `SKILL.md` 路径，作为运行时加载身份标识。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<PathBuf>,
    /// 当前生命周期状态。
    pub lifecycle: SkillLifecycle,
    /// 技能首次创建的时间。
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// 技能最后一次被使用/引用的时间。
    pub last_used_at: chrono::DateTime<chrono::Utc>,
    /// 技能最后一次被修改的时间。
    pub last_modified_at: chrono::DateTime<chrono::Utc>,
    /// 该技能是否被固定（豁免自动转换）。
    pub pinned: bool,
    /// 该技能是否为 agent 创建（相对于内置/外部技能）。
    pub agent_created: bool,
    /// 若已弃用/归档，指明由哪个技能取代了它。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub superseded_by: Option<String>,
}

// ── CuratorConfig ──────────────────────────────────────────────────

/// curator 的配置。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CuratorConfig {
    /// 技能变为 stale 之前的不活跃天数。
    pub stale_days: u64,
    /// 技能变为 deprecated 之前的不活跃天数。
    pub deprecate_days: u64,
    /// 技能被归档之前的不活跃天数。
    pub archive_days: u64,
    /// curator 是否启用。
    pub enabled: bool,
}

impl Default for CuratorConfig {
    /// 返回默认的 curator 配置。
    fn default() -> Self {
        Self {
            stale_days: 30,
            deprecate_days: 60,
            archive_days: 90,
            enabled: true,
        }
    }
}

// ── CuratorState ───────────────────────────────────────────────────

/// curator 的持久化状态。
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CuratorState {
    /// 每个被追踪技能的元数据。
    pub skills: HashMap<String, SkillMeta>,
    /// curator 上一次运行的时间。
    pub last_run_at: Option<chrono::DateTime<chrono::Utc>>,
}

impl CuratorState {
    /** 从 JSON 文件加载状态。

    - 文件不存在 → 返回默认值（全新开始）。这是唯一被视为"空"的情况。
    - 文件存在但无法解析/读取 → 同样返回默认值，但会记录一条 `warn!`，
      避免静默的数据损坏在下一次保存时被覆盖（那样会摧毁本可恢复的数据）。
      调用方（某个 curator 方法）会继续执行并重写文件，但至少这次丢失
      在日志中是可观察的。
    */
    pub fn try_load(path: &PathBuf) -> Result<Self> {
        match std::fs::read_to_string(path) {
            Ok(data) => serde_json::from_str(&data).map_err(Into::into),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(e.into()),
        }
    }

    /** 原子化地将状态保存到 JSON 文件。

    先写入一个临时的同级文件，再 `rename` 覆盖目标文件。`rename` 在 POSIX
    系统上是原子操作，因此写入过程中崩溃永远不会留下截断或部分写入的状态
    文件——读者要么看到旧文件，要么看到完整的新文件，绝不会看到一个
    交错混合的版本。
    */
    pub(crate) fn save(&self, path: &PathBuf) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let data = serde_json::to_string_pretty(self)?;
        let tmp_path = append_path_suffix(path, ".tmp");
        {
            let mut file = std::fs::File::create(&tmp_path)?;
            use std::io::Write;
            file.write_all(data.as_bytes())?;
            // 在 rename 之前先 fsync 数据，避免 rename 之后崩溃时暴露出一个空文件。
            file.sync_all()?;
        }
        std::fs::rename(&tmp_path, path)?;
        Ok(())
    }
}

/// 候选项名称与其生命周期权限标识之间的绑定状态。
#[derive(Default, Serialize, Deserialize)]
struct CandidateAuthorityState {
    /// 候选名称到权限标识的映射。
    bindings: HashMap<String, String>,
}

impl CandidateAuthorityState {
    /// 从磁盘加载绑定状态；文件不存在时返回空状态。
    fn try_load(path: &Path) -> Result<Self> {
        match std::fs::read_to_string(path) {
            Ok(data) => serde_json::from_str(&data).map_err(Into::into),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(error) => Err(error.into()),
        }
    }

    /// 以原子方式将绑定状态保存到磁盘。
    fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let data = serde_json::to_vec_pretty(self)?;
        echo_core::utils::fs::atomic_write(path, &data)?;
        Ok(())
    }
}

// ── Curator ────────────────────────────────────────────────────────

/** 技能生命周期管理器。

管理技能在各生命周期状态之间的转换：
- Candidate → Draft（手动提升）
- Draft → Active（手动提升，或在成功使用 N 次后）
- Active → Stale（经过 `stale_days` 不活跃后）
- Stale → Deprecated（经过 `deprecate_days` 不活跃后）
- Deprecated → Archived（经过 `archive_days` 不活跃后）

仅对 agent 创建的技能生效。被固定的技能豁免。
由一个 JSON 状态文件支撑的技能生命周期 curator。

读取-修改-写入操作通过一个共享的 advisory 文件锁串行化，
所有使用同一状态路径的 `Curator` 实例都共享这把锁。
*/
pub struct Curator {
    /// curator 配置。
    config: CuratorConfig,
    /// JSON 状态文件路径。
    state_path: PathBuf,
}

/*
 * P2-9: 需要 Clone 以便在 spawn_blocking 闭包里使用 (touch_skill 是同步阻塞,
 * 移到阻塞线程池)。Curator 是无状态句柄 (touch_skill 内部 flock), Clone 成本极低。
 */
impl Clone for Curator {
    /// 复制 curator 句柄（仅克隆配置与路径）。
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            state_path: self.state_path.clone(),
        }
    }
}

impl Curator {
    /// 使用给定的配置和状态文件路径创建一个新的 curator。
    pub fn new(config: CuratorConfig, state_path: impl Into<PathBuf>) -> Self {
        Self {
            config,
            state_path: state_path.into(),
        }
    }

    /** 候选项负载变更所使用的私有恢复日志。

    该路径遵循调用方提供的生命周期状态身份标识，而不会让候选项恢复
    成为公开 Curator 契约的一部分。
    */
    pub(crate) fn candidate_operation_journal_path(&self) -> PathBuf {
        append_path_suffix(&self.state_path, ".candidate-operations.jsonl")
    }

    /// 技能操作日志（journal）的路径。
    pub(crate) fn skill_operation_journal_path(&self) -> PathBuf {
        append_path_suffix(&self.state_path, ".skill-operations.jsonl")
    }

    /** 仅当给定技能条目仍与 `expected` 快照一致时，才用 `replacement`
    中的值原子化替换它们；若某个条目已被并发修改为既非 `expected`
    也非 `replacement` 的第三种状态，则整体失败并放弃写入。
    */
    pub(crate) fn replace_skill_entries_if_current(
        &self,
        expected: &CuratorState,
        replacement: &CuratorState,
        affected_names: &[String],
    ) -> Result<()> {
        self.with_locked_state(|state| {
            let mut dirty = false;
            for name in affected_names {
                let current = state.skills.get(name);
                let before = expected.skills.get(name);
                let after = replacement.skills.get(name);
                if current != before && current != after {
                    return Err(crate::error::ReactError::Other(format!(
                        "skill lifecycle {name} changed after preview"
                    )));
                }
                if current == after {
                    continue;
                }
                match after {
                    Some(meta) => {
                        state.skills.insert(name.clone(), meta.clone());
                    }
                    None => {
                        state.skills.remove(name);
                    }
                }
                dirty = true;
            }
            Ok(((), dirty))
        })
    }

    /// 返回候选权限绑定文件的路径。
    fn candidate_authority_path(&self) -> PathBuf {
        append_path_suffix(&self.state_path, ".candidate-authorities.json")
    }

    /// 返回旁路锁文件的路径。
    fn lock_path(&self) -> PathBuf {
        append_path_suffix(&self.state_path, ".lock")
    }

    /** 在一个 advisory 文件锁下加载当前状态。

    仅在读取期间对旁路锁文件持有独占的 `flock`。如果需要原子化的
    读取-修改-写入，请改用 `with_locked_state`——它会在加载、修改、
    保存这一整个临界区内持续持有锁。

    该锁是操作系统层面的 advisory 锁（`fs2::FileExt::try_lock_exclusive`），
    作用于一个旁路文件。它是**kill 安全**的：如果本进程死亡（kill -9、
    panic、断电），操作系统会关闭 fd 并自动释放锁。这正是它与此前
    `create_new` 旁路文件设计的关键区别——旧设计的锁文件在崩溃后仍然
    存在，会让此后所有加锁者永久死锁（P0 —— Curator TOCTOU）。
    */
    pub fn load_state(&self) -> Result<CuratorState> {
        let _guard = self.acquire_lock()?;
        CuratorState::try_load(&self.state_path)
    }

    /** 在一个 advisory 文件锁下保存状态。

    仅在写入期间持有独占锁。需要加载-修改-保存原子性的调用方必须改用
    `with_locked_state`。
    */
    #[cfg(test)]
    pub(crate) fn save_state(&self, state: &CuratorState) -> Result<()> {
        let _guard = self.acquire_lock()?;
        state.save(&self.state_path)
    }

    /** 在持有单个锁的情况下，原子化地对 curator 状态执行读取-修改-写入。

    闭包接收刚加载出的状态并可以对其进行修改。成功时，状态会在锁仍被
    持有的情况下写回磁盘；出错时，内存中的修改会被丢弃，磁盘文件保持
    不变。这样就消除了两个 `Curator` 实例交错执行"加载 → 修改 → 保存"
    而互相覆盖对方结果的 TOCTOU 窗口。

    闭包返回 `(T, bool)`：T 是要返回给调用方的值，`dirty` 标志表示
    是否需要保存。当 `dirty` 为 false 时会跳过保存（只读路径）——
    这样可以避免每次调用都进行不必要的重写。
    */
    fn with_locked_state<T, F>(&self, f: F) -> Result<T>
    where
        F: FnOnce(&mut CuratorState) -> Result<(T, bool)>,
    {
        let _guard = self.acquire_lock()?;
        let mut state = CuratorState::try_load(&self.state_path)?;
        let (result, dirty) = f(&mut state)?;
        if dirty {
            state.save(&self.state_path)?;
        }
        Ok(result)
    }

    /** 在旁路 `.lock` 文件上获取一个独占的 advisory 锁。

    在有限的时间内以非阻塞方式重试。返回一个 guard，其 `Drop` 会释放锁。
    旁路文件被有意设计为持久存在：解锁后删除它可能让另一个进程创建出
    不同的 inode，从而绕过已有的锁。
    */
    fn acquire_lock(&self) -> Result<CuratorLockGuard> {
        let lock_path = self.lock_path();
        if let Some(parent) = lock_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        /*
         * 确保旁路文件存在（若已存在则不会失败）。
         * `truncate(false)` 会保留任何既有内容（对 flock 而言无所谓，
         * 但可以避免破坏另一个进程刚创建的文件）。
         */
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&lock_path)?;

        // 获取文件锁的最大尝试次数。
        const MAX_ATTEMPTS: u32 = 25;
        // 每次尝试获取文件锁失败后的等待间隔。
        const BACKOFF: std::time::Duration = std::time::Duration::from_millis(40);

        let mut last_err = None;
        for _ in 0..MAX_ATTEMPTS {
            match file.try_lock_exclusive() {
                Ok(()) => {
                    return Ok(CuratorLockGuard { file: Some(file) });
                }
                Err(e) => {
                    last_err = Some(e);
                    std::thread::sleep(BACKOFF);
                }
            }
        }

        Err(crate::error::ReactError::Other(format!(
            "curator: could not acquire lock at {} after {} attempts: {}",
            lock_path.display(),
            MAX_ATTEMPTS,
            last_err
                .map(|error| error.to_string())
                .unwrap_or_else(|| "unknown lock error".to_string())
        )))
    }

    /// 注册一个新技能，或更新已存在技能的最后使用时间。
    #[cfg(test)]
    pub(crate) fn touch_skill(&self, name: &str, agent_created: bool) -> Result<()> {
        self.touch_skill_at(name, None, agent_created)
    }

    /// 注册或触碰（touch）一个技能，同时绑定其具体的 `SKILL.md` 路径。
    #[cfg(test)]
    pub(crate) fn touch_skill_at(
        &self,
        name: &str,
        path: Option<&std::path::Path>,
        agent_created: bool,
    ) -> Result<()> {
        let normalized_path = path.map(normalize_skill_path);
        self.with_locked_state(|state| {
            let now = chrono::Utc::now();
            if let Some(meta) = state.skills.get_mut(name) {
                meta.last_used_at = now;
                if normalized_path.is_some() {
                    meta.path = normalized_path.clone();
                    meta.last_modified_at = now;
                }
            } else {
                state.skills.insert(
                    name.to_string(),
                    SkillMeta {
                        name: name.to_string(),
                        path: normalized_path.clone(),
                        lifecycle: SkillLifecycle::Active,
                        created_at: now,
                        last_used_at: now,
                        last_modified_at: now,
                        pinned: false,
                        agent_created,
                        superseded_by: None,
                    },
                );
            }
            Ok(((), true))
        })
    }

    /// 注册一个新的技能候选项（从记忆模式中发现）。
    #[cfg(test)]
    pub(crate) fn register_candidate(&self, name: &str) -> Result<()> {
        self.register_candidate_at(name, None)
    }

    /// 注册一个候选项，并可选地绑定未来的 draft 路径。
    #[cfg(test)]
    pub(crate) fn register_candidate_at(
        &self,
        name: &str,
        path: Option<&std::path::Path>,
    ) -> Result<()> {
        let normalized_path = path.map(normalize_skill_path);
        self.with_locked_state(|state| {
            if let Some(meta) = state.skills.get_mut(name) {
                if normalized_path.is_some() && meta.path != normalized_path {
                    meta.path = normalized_path.clone();
                    meta.last_modified_at = chrono::Utc::now();
                    return Ok(((), true));
                }
                return Ok(((), false));
            }
            let now = chrono::Utc::now();
            state.skills.insert(
                name.to_string(),
                SkillMeta {
                    name: name.to_string(),
                    path: normalized_path.clone(),
                    lifecycle: SkillLifecycle::Candidate,
                    created_at: now,
                    last_used_at: now,
                    last_modified_at: now,
                    pinned: false,
                    agent_created: true,
                    superseded_by: None,
                },
            );
            Ok(((), true))
        })
    }

    /**
     * 使用一个生命周期权限（authority）标识注册技能候选项，防止不同权限
     * 主体争用同一个候选名；若名称已存在，仅当权限标识匹配时才视为成功。
     */
    pub(crate) fn register_candidate_with_authority(
        &self,
        name: &str,
        authority_id: &str,
    ) -> Result<()> {
        if authority_id.is_empty() {
            return Err(crate::error::ReactError::Other(
                "candidate authority identity must not be empty".into(),
            ));
        }
        let _guard = self.acquire_lock()?;
        let mut state = CuratorState::try_load(&self.state_path)?;
        let authority_path = self.candidate_authority_path();
        let mut authorities = CandidateAuthorityState::try_load(&authority_path)?;
        if state.skills.contains_key(name) {
            return if authorities.bindings.get(name).map(String::as_str) == Some(authority_id) {
                Ok(())
            } else {
                Err(crate::error::ReactError::Other(format!(
                    "skill candidate {name} belongs to another or unbound lifecycle authority"
                )))
            };
        }
        match authorities.bindings.get(name) {
            Some(existing) if existing != authority_id => {
                return Err(crate::error::ReactError::Other(format!(
                    "skill candidate {name} belongs to another lifecycle authority"
                )));
            }
            Some(_) => {}
            None => {
                authorities
                    .bindings
                    .insert(name.to_string(), authority_id.to_string());
                authorities.save(&authority_path)?;
            }
        }
        let now = chrono::Utc::now();
        state.skills.insert(
            name.to_string(),
            SkillMeta {
                name: name.to_string(),
                path: None,
                lifecycle: SkillLifecycle::Candidate,
                created_at: now,
                last_used_at: now,
                last_modified_at: now,
                pinned: false,
                agent_created: true,
                superseded_by: None,
            },
        );
        state.save(&self.state_path)
    }

    /// 读取某技能候选项当前绑定的权限（authority）标识。
    #[cfg(test)]
    pub(crate) fn candidate_authority(&self, name: &str) -> Result<Option<String>> {
        let _guard = self.acquire_lock()?;
        Ok(
            CandidateAuthorityState::try_load(&self.candidate_authority_path())?
                .bindings
                .get(name)
                .cloned(),
        )
    }

    /// 读取指定技能的完整元数据。
    #[cfg(test)]
    pub(crate) fn skill(&self, name: &str) -> Result<Option<SkillMeta>> {
        let _guard = self.acquire_lock()?;
        Ok(CuratorState::try_load(&self.state_path)?
            .skills
            .get(name)
            .cloned())
    }

    /// 将 Candidate 状态的技能提升为 Draft。
    #[cfg(test)]
    pub(crate) fn promote_to_draft(&self, name: &str) -> Result<bool> {
        self.promote_to_draft_at(name, None)
    }

    /// 将 Candidate 提升为 Draft，并绑定生成的 draft 路径。
    #[cfg(test)]
    pub(crate) fn promote_to_draft_at(
        &self,
        name: &str,
        path: Option<&std::path::Path>,
    ) -> Result<bool> {
        let normalized_path = path.map(normalize_skill_path);
        self.with_locked_state(|state| {
            let now = chrono::Utc::now();
            if let Some(meta) = state.skills.get_mut(name)
                && meta.lifecycle == SkillLifecycle::Candidate
            {
                meta.lifecycle = SkillLifecycle::Draft;
                if normalized_path.is_some() {
                    meta.path = normalized_path.clone();
                }
                meta.last_modified_at = now;
                return Ok((true, true));
            }
            Ok((false, false))
        })
    }

    /// 将 Draft 状态的技能提升为 Active。
    #[cfg(test)]
    pub(crate) fn promote_to_active(&self, name: &str) -> Result<bool> {
        self.promote_to_active_at(name, None)
    }

    /// 将 Draft 提升为 Active，并绑定权威的运行时路径。
    #[cfg(test)]
    pub(crate) fn promote_to_active_at(
        &self,
        name: &str,
        path: Option<&std::path::Path>,
    ) -> Result<bool> {
        let normalized_path = path.map(normalize_skill_path);
        self.with_locked_state(|state| {
            let now = chrono::Utc::now();
            if let Some(meta) = state.skills.get_mut(name)
                && meta.lifecycle == SkillLifecycle::Draft
            {
                meta.lifecycle = SkillLifecycle::Active;
                if normalized_path.is_some() {
                    meta.path = normalized_path.clone();
                }
                meta.last_modified_at = now;
                return Ok((true, true));
            }
            Ok((false, false))
        })
    }

    /// 返回某个具体 `SKILL.md` 路径对应的生命周期元数据。
    pub fn skill_for_path(&self, path: &std::path::Path) -> Result<Option<SkillMeta>> {
        let normalized = normalize_skill_path(path);
        Ok(self
            .load_state()?
            .skills
            .into_values()
            .find(|meta| meta.path.as_ref() == Some(&normalized)))
    }

    /// 弃用一个技能，可选地指定由哪个技能取代它。
    #[cfg(test)]
    pub(crate) fn deprecate_skill(&self, name: &str, superseded_by: Option<&str>) -> Result<bool> {
        self.with_locked_state(|state| {
            let now = chrono::Utc::now();
            if let Some(meta) = state.skills.get_mut(name)
                && matches!(
                    meta.lifecycle,
                    SkillLifecycle::Active | SkillLifecycle::Stale
                )
            {
                meta.lifecycle = SkillLifecycle::Deprecated;
                meta.superseded_by = superseded_by.map(|s| s.to_string());
                meta.last_modified_at = now;
                return Ok((true, true));
            }
            Ok((false, false))
        })
    }

    /// 固定（pin）一个技能（豁免自动状态转换）。
    #[cfg(test)]
    pub(crate) fn pin_skill(&self, name: &str) -> Result<()> {
        self.with_locked_state(|state| {
            if let Some(meta) = state.skills.get_mut(name) {
                meta.pinned = true;
                return Ok(((), true));
            }
            Ok(((), false))
        })
    }

    /// 取消固定一个技能。
    #[cfg(test)]
    pub(crate) fn unpin_skill(&self, name: &str) -> Result<()> {
        self.with_locked_state(|state| {
            if let Some(meta) = state.skills.get_mut(name) {
                meta.pinned = false;
                return Ok(((), true));
            }
            Ok(((), false))
        })
    }

    /** 根据不活跃时长应用自动生命周期转换。

    返回已应用的转换列表。
    */
    #[cfg(test)]
    pub(crate) fn apply_transitions(
        &self,
    ) -> Result<Vec<(String, SkillLifecycle, SkillLifecycle)>> {
        if !self.config.enabled {
            return Ok(vec![]);
        }

        self.with_locked_state(|state| {
            let now = chrono::Utc::now();
            let mut transitions = Vec::new();

            for meta in state.skills.values_mut() {
                // 跳过被固定的和非 agent 创建的技能。
                if meta.pinned || !meta.agent_created {
                    continue;
                }

                // Candidate 和 Draft 状态不会因不活跃而自动转换。
                if matches!(
                    meta.lifecycle,
                    SkillLifecycle::Candidate | SkillLifecycle::Draft
                ) {
                    continue;
                }

                /*
                 * `num_days()` 返回 i64，在时钟偏差下可能为负
                 * （last_used_at 晚于 now）。`.max(0)` 可以防止 `as u64`
                 * 转换把负数环绕成一个巨大的值，那样会让所有不活跃阈值
                 * 瞬间被满足，从而错误地归档刚被使用过的技能（N9）。
                 */
                let idle_days = (now - meta.last_used_at).num_days().max(0) as u64;

                let new_lifecycle = match meta.lifecycle {
                    SkillLifecycle::Active if idle_days >= self.config.stale_days => {
                        Some(SkillLifecycle::Stale)
                    }
                    SkillLifecycle::Stale if idle_days >= self.config.deprecate_days => {
                        Some(SkillLifecycle::Deprecated)
                    }
                    SkillLifecycle::Deprecated if idle_days >= self.config.archive_days => {
                        Some(SkillLifecycle::Archived)
                    }
                    _ => None,
                };

                if let Some(new_lc) = new_lifecycle {
                    transitions.push((meta.name.clone(), meta.lifecycle, new_lc));
                    meta.lifecycle = new_lc;
                }
            }

            state.last_run_at = Some(now);
            Ok((transitions, true))
        })
    }

    /// 获取 curator 状态的摘要信息。
    pub fn status(&self) -> Result<CuratorStatus> {
        let state = self.load_state()?;
        let mut candidate = 0;
        let mut draft = 0;
        let mut active = 0;
        let mut stale = 0;
        let mut deprecated = 0;
        let mut archived = 0;
        let mut pinned = 0;

        for meta in state.skills.values() {
            if meta.pinned {
                pinned += 1;
            }
            match meta.lifecycle {
                SkillLifecycle::Candidate => candidate += 1,
                SkillLifecycle::Draft => draft += 1,
                SkillLifecycle::Active => active += 1,
                SkillLifecycle::Stale => stale += 1,
                SkillLifecycle::Deprecated => deprecated += 1,
                SkillLifecycle::Archived => archived += 1,
            }
        }

        Ok(CuratorStatus {
            total: state.skills.len(),
            candidate,
            draft,
            active,
            stale,
            deprecated,
            archived,
            pinned,
            last_run_at: state.last_run_at,
        })
    }
}

/// curator 状态的摘要信息。
#[derive(Debug, Clone)]
pub struct CuratorStatus {
    /// 已追踪技能总数。
    pub total: usize,
    /// 处于 Candidate 状态的技能数量。
    pub candidate: usize,
    /// 处于 Draft 状态的技能数量。
    pub draft: usize,
    /// 处于 Active 状态的技能数量。
    pub active: usize,
    /// 处于 Stale 状态的技能数量。
    pub stale: usize,
    /// 处于 Deprecated 状态的技能数量。
    pub deprecated: usize,
    /// 处于 Archived 状态的技能数量。
    pub archived: usize,
    /// 被固定（pinned）的技能数量。
    pub pinned: usize,
    /// curator 上一次运行的时间。
    pub last_run_at: Option<chrono::DateTime<chrono::Utc>>,
}

/** RAII guard，持有旁路文件上的独占 advisory 锁。

由 `Curator::acquire_lock()` 创建。drop 时会显式释放锁（`unlock`）
并关闭 fd——这同时也会作为安全网释放操作系统层面的 advisory 锁。
旁路文件本身会保留在原地，以便所有进程都锁定同一个 inode。

kill 安全性依赖于操作系统在 fd 关闭时释放 flock（包括进程死亡的情况），
而不依赖于 Drop 是否真正执行。因此即使进程在临界区中途被 SIGKILL
杀死，也不会有其他 curator 被卡住。
*/
struct CuratorLockGuard {
    /// 持有锁的底层文件句柄；drop 前为 `Some`。
    file: Option<std::fs::File>,
}

impl Drop for CuratorLockGuard {
    /// 释放锁文件：先尽力解锁，再关闭文件描述符。
    fn drop(&mut self) {
        if let Some(file) = self.file.take() {
            // 先尽力显式解锁；下面关闭 fd 同样会释放操作系统层面的 advisory 锁。
            let _ = file.unlock();
            drop(file);
        }
    }
}

/// 尽量规范化技能路径（做 canonicalize），失败时原样返回。
fn normalize_skill_path(path: &std::path::Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
}

/// curator 生命周期管理的单元测试。
#[cfg(test)]
mod tests {
    use super::*;

    /// 在临时目录中创建一个用于测试的 curator。
    fn temp_curator() -> Curator {
        let dir = std::env::temp_dir().join(format!("echo_curator_test_{}", uuid::Uuid::new_v4()));
        let path = dir.join("curator_state.json");
        Curator::new(CuratorConfig::default(), path)
    }

    /// 验证同名不同扩展名的技能映射到互不相同的私有路径。
    #[test]
    fn same_stem_different_extensions_have_injective_private_paths() {
        let root = std::env::temp_dir().join(format!(
            "echo-curator-path-identity-{}",
            uuid::Uuid::new_v4()
        ));
        let json_path = root.join("state.json");
        let toml_path = root.join("state.toml");
        let json = Curator::new(CuratorConfig::default(), json_path.clone());
        let toml = Curator::new(CuratorConfig::default(), toml_path.clone());

        assert_ne!(
            json.candidate_operation_journal_path(),
            toml.candidate_operation_journal_path()
        );
        assert_ne!(
            json.candidate_authority_path(),
            toml.candidate_authority_path()
        );
        assert_ne!(json.lock_path(), toml.lock_path());
        assert_ne!(
            append_path_suffix(&json_path, ".tmp"),
            append_path_suffix(&toml_path, ".tmp")
        );
        assert_eq!(
            json.candidate_operation_journal_path(),
            root.join("state.json.candidate-operations.jsonl")
        );
        assert_eq!(
            toml.candidate_authority_path(),
            root.join("state.toml.candidate-authorities.json")
        );
    }

    /// 验证 touch 之后能查询到技能状态。
    #[test]
    fn test_touch_and_status() -> Result<()> {
        let curator = temp_curator();
        curator.touch_skill("test-skill", true)?;

        let status = curator.status()?;
        assert_eq!(status.total, 1);
        assert_eq!(status.active, 1);
        assert_eq!(status.stale, 0);
        assert_eq!(status.archived, 0);
        Ok(())
    }

    /// 验证技能按具体路径进行跟踪。
    #[test]
    fn test_tracks_skill_by_concrete_path() -> Result<()> {
        let curator = temp_curator();
        let skill_path = std::env::temp_dir()
            .join("echo-curator-path-test")
            .join("SKILL.md");
        curator.touch_skill_at("path-skill", Some(&skill_path), true)?;

        let Some(meta) = curator.skill_for_path(&skill_path)? else {
            return Err(crate::error::ReactError::Other(
                "skill path was not tracked".to_string(),
            ));
        };
        assert_eq!(meta.name, "path-skill");
        assert_eq!(meta.lifecycle, SkillLifecycle::Active);
        Ok(())
    }

    /// 验证技能的固定与取消固定。
    #[test]
    fn test_pin_unpin() -> Result<()> {
        let curator = temp_curator();
        curator.touch_skill("test-skill", true)?;
        curator.pin_skill("test-skill")?;

        let state = curator.load_state()?;
        assert!(
            state
                .skills
                .get("test-skill")
                .is_some_and(|meta| meta.pinned)
        );

        curator.unpin_skill("test-skill")?;
        let state = curator.load_state()?;
        assert!(
            state
                .skills
                .get("test-skill")
                .is_some_and(|meta| !meta.pinned)
        );
        Ok(())
    }

    /// 验证新鲜的技能不会发生状态迁移。
    #[test]
    fn test_no_transition_when_fresh() -> Result<()> {
        let curator = temp_curator();
        curator.touch_skill("fresh-skill", true)?;

        let transitions = curator.apply_transitions()?;
        assert!(transitions.is_empty());

        let status = curator.status()?;
        assert_eq!(status.active, 1);
        Ok(())
    }

    /// 验证非 Agent 创建的技能会被跳过。
    #[test]
    fn test_skip_non_agent_created() -> Result<()> {
        let curator = temp_curator();
        curator.touch_skill("bundled-skill", false)?;

        // 手动将 last_used_at 设置为很久之前的日期
        let mut state = curator.load_state()?;
        state
            .skills
            .get_mut("bundled-skill")
            .ok_or_else(|| crate::error::ReactError::Other("missing bundled-skill".into()))?
            .last_used_at = chrono::Utc::now() - chrono::Duration::days(100);
        curator.save_state(&state)?;

        let transitions = curator.apply_transitions()?;
        // 不应转换非 agent 创建的技能
        assert!(transitions.is_empty());
        Ok(())
    }

    /// 验证长期未使用的技能会迁移为陈旧状态。
    #[test]
    fn test_stale_transition() -> Result<()> {
        let curator = temp_curator();
        curator.touch_skill("old-skill", true)?;

        // 手动将 last_used_at 设置为 31 天前
        let mut state = curator.load_state()?;
        state
            .skills
            .get_mut("old-skill")
            .ok_or_else(|| crate::error::ReactError::Other("missing old-skill".into()))?
            .last_used_at = chrono::Utc::now() - chrono::Duration::days(31);
        curator.save_state(&state)?;

        let transitions = curator.apply_transitions()?;
        assert_eq!(transitions.len(), 1);
        assert_eq!(transitions[0].1, SkillLifecycle::Active);
        assert_eq!(transitions[0].2, SkillLifecycle::Stale);

        let status = curator.status()?;
        assert_eq!(status.stale, 1);
        Ok(())
    }

    /// 验证已固定的技能会跳过状态迁移。
    #[test]
    fn test_pinned_skips_transition() -> Result<()> {
        let curator = temp_curator();
        curator.touch_skill("pinned-skill", true)?;
        curator.pin_skill("pinned-skill")?;

        // 手动将 last_used_at 设置为 100 天前
        let mut state = curator.load_state()?;
        state
            .skills
            .get_mut("pinned-skill")
            .ok_or_else(|| crate::error::ReactError::Other("missing pinned-skill".into()))?
            .last_used_at = chrono::Utc::now() - chrono::Duration::days(100);
        curator.save_state(&state)?;

        let transitions = curator.apply_transitions()?;
        assert!(transitions.is_empty());

        let status = curator.status()?;
        assert_eq!(status.active, 1);
        assert_eq!(status.pinned, 1);
        Ok(())
    }

    /// 验证候选到草稿再到激活的迁移流程。
    #[test]
    fn test_candidate_to_draft_to_active() -> Result<()> {
        let curator = temp_curator();

        // 注册为 candidate
        curator.register_candidate("my-skill")?;
        let state = curator.load_state()?;
        assert_eq!(
            state.skills.get("my-skill").map(|meta| meta.lifecycle),
            Some(SkillLifecycle::Candidate)
        );

        // 提升为 draft
        let promoted = curator.promote_to_draft("my-skill")?;
        assert!(promoted);
        let state = curator.load_state()?;
        assert_eq!(
            state.skills.get("my-skill").map(|meta| meta.lifecycle),
            Some(SkillLifecycle::Draft)
        );

        // 提升为 active
        let promoted = curator.promote_to_active("my-skill")?;
        assert!(promoted);
        let state = curator.load_state()?;
        assert_eq!(
            state.skills.get("my-skill").map(|meta| meta.lifecycle),
            Some(SkillLifecycle::Active)
        );
        Ok(())
    }

    /// 验证技能可以被弃用。
    #[test]
    fn test_deprecate_skill() -> Result<()> {
        let curator = temp_curator();
        curator.touch_skill("old-skill", true)?;

        let deprecated = curator.deprecate_skill("old-skill", Some("new-skill"))?;
        assert!(deprecated);

        let state = curator.load_state()?;
        assert_eq!(
            state.skills.get("old-skill").map(|meta| meta.lifecycle),
            Some(SkillLifecycle::Deprecated)
        );
        assert_eq!(
            state
                .skills
                .get("old-skill")
                .and_then(|meta| meta.superseded_by.as_deref()),
            Some("new-skill")
        );
        Ok(())
    }

    /// 验证完整的技能生命周期。
    #[test]
    fn test_full_lifecycle() -> Result<()> {
        let curator = temp_curator();

        // Candidate -> Draft -> Active（依次提升）
        curator.register_candidate("lifecycle-skill")?;
        curator.promote_to_draft("lifecycle-skill")?;
        curator.promote_to_active("lifecycle-skill")?;

        // 设置 last_used_at 使其变为 stale
        let mut state = curator.load_state()?;
        state
            .skills
            .get_mut("lifecycle-skill")
            .ok_or_else(|| crate::error::ReactError::Other("missing lifecycle-skill".into()))?
            .last_used_at = chrono::Utc::now() - chrono::Duration::days(35);
        curator.save_state(&state)?;

        let transitions = curator.apply_transitions()?;
        assert_eq!(transitions.len(), 1);
        assert_eq!(transitions[0].2, SkillLifecycle::Stale);

        // 设置 last_used_at 使其变为 deprecated
        let mut state = curator.load_state()?;
        state
            .skills
            .get_mut("lifecycle-skill")
            .ok_or_else(|| crate::error::ReactError::Other("missing lifecycle-skill".into()))?
            .last_used_at = chrono::Utc::now() - chrono::Duration::days(65);
        curator.save_state(&state)?;

        let transitions = curator.apply_transitions()?;
        assert_eq!(transitions.len(), 1);
        assert_eq!(transitions[0].2, SkillLifecycle::Deprecated);

        // 设置 last_used_at 使其变为 archived
        let mut state = curator.load_state()?;
        state
            .skills
            .get_mut("lifecycle-skill")
            .ok_or_else(|| crate::error::ReactError::Other("missing lifecycle-skill".into()))?
            .last_used_at = chrono::Utc::now() - chrono::Duration::days(95);
        curator.save_state(&state)?;

        let transitions = curator.apply_transitions()?;
        assert_eq!(transitions.len(), 1);
        assert_eq!(transitions[0].2, SkillLifecycle::Archived);
        Ok(())
    }

    /// 验证候选状态不会自动迁移。
    #[test]
    fn test_candidate_no_auto_transition() -> Result<()> {
        let curator = temp_curator();
        curator.register_candidate("cand-skill")?;

        // 将 last_used_at 设置为很久以前
        let mut state = curator.load_state()?;
        state
            .skills
            .get_mut("cand-skill")
            .ok_or_else(|| crate::error::ReactError::Other("missing cand-skill".into()))?
            .last_used_at = chrono::Utc::now() - chrono::Duration::days(200);
        curator.save_state(&state)?;

        // Candidate 不应自动转换
        let transitions = curator.apply_transitions()?;
        assert!(transitions.is_empty());

        let state = curator.load_state()?;
        assert_eq!(
            state.skills.get("cand-skill").map(|meta| meta.lifecycle),
            Some(SkillLifecycle::Candidate)
        );
        Ok(())
    }

    /// 验证重复注册候选项是幂等的。
    #[test]
    fn test_register_candidate_idempotent() -> Result<()> {
        let curator = temp_curator();
        curator.register_candidate("my-skill")?;
        curator.register_candidate("my-skill")?;

        let state = curator.load_state()?;
        assert_eq!(state.skills.len(), 1);
        Ok(())
    }

    /** P0-1（Curator TOCTOU）的回归测试：两个使用同一状态路径的 `Curator`
    实例，各自注册一个*不同*的技能，必须都能成功落盘。在修复前的设计中
    （分离的 load/save 锁），后写入者会覆盖前者，导致其中一个技能悄无声息地消失。
    */
    #[test]
    fn test_concurrent_curators_do_not_lose_updates() -> Result<()> {
        // 共享状态路径（模拟两个模块调用 default_path()）。
        let dir = std::env::temp_dir().join(format!("echo_curator_conc_{}", uuid::Uuid::new_v4()));
        let path = dir.join("curator_state.json");

        let a = Curator::new(CuratorConfig::default(), path.clone());
        let b = Curator::new(CuratorConfig::default(), path.clone());

        // 交错执行两个实例，模拟真实的竞态场景。
        a.register_candidate("skill-a")?;
        b.register_candidate("skill-b")?;

        let state = a.load_state()?;
        assert!(
            state.skills.contains_key("skill-a"),
            "skill-a lost (TOCTOU regression)"
        );
        assert!(
            state.skills.contains_key("skill-b"),
            "skill-b lost (TOCTOU regression)"
        );
        Ok(())
    }

    /** kill -9 死锁问题（R1）的回归测试：获取锁后，在不执行正常清理的情况下
    丢弃 guard（模拟进程死亡但 fd 被操作系统关闭的场景），不应该卡住
    下一个加锁者。这是成立的，因为 flock 会在 fd 关闭时自动释放；
    残留的旁路文件不构成障碍，因为我们从不要求 `create_new`。
    */
    #[test]
    fn test_lock_survives_abandoned_sidecar() -> Result<()> {
        let dir = std::env::temp_dir().join(format!("echo_curator_stale_{}", uuid::Uuid::new_v4()));
        let path = dir.join("curator_state.json");

        // 预先创建一个残留的旁路锁文件（模拟此前进程崩溃的情形）。
        let lock_path = append_path_suffix(&path, ".lock");
        std::fs::create_dir_all(&dir)?;
        std::fs::write(&lock_path, b"")?;
        assert!(lock_path.exists(), "precondition: stale sidecar exists");

        // 新的 curator 仍必须成功——残留文件不构成阻碍。
        let curator = Curator::new(CuratorConfig::default(), path.clone());
        curator.register_candidate("after-stale")?;
        assert!(curator.load_state()?.skills.contains_key("after-stale"));
        Ok(())
    }

    /** `with_locked_state` 在闭包出错时不得持久化——磁盘上的文件应保持
    调用前的状态。
    */
    #[test]
    fn test_with_locked_state_rollback_on_error() -> Result<()> {
        let curator = temp_curator();
        curator.register_candidate("keep-me")?;

        use crate::error::ReactError;
        let result: Result<()> = curator.with_locked_state(|state| {
            // 先在内存中修改，然后返回失败。
            state.skills.insert(
                "should-not-persist".to_string(),
                SkillMeta {
                    name: "should-not-persist".to_string(),
                    path: None,
                    lifecycle: SkillLifecycle::Candidate,
                    created_at: chrono::Utc::now(),
                    last_used_at: chrono::Utc::now(),
                    last_modified_at: chrono::Utc::now(),
                    pinned: false,
                    agent_created: true,
                    superseded_by: None,
                },
            );
            Err(ReactError::Other("simulated failure".to_string()))
        });
        assert!(result.is_err());

        let state = curator.load_state()?;
        assert!(state.skills.contains_key("keep-me"));
        assert!(
            !state.skills.contains_key("should-not-persist"),
            "failed mutation must not be persisted"
        );
        Ok(())
    }

    /// 验证损坏的状态文件会报错且不会被重写。
    #[test]
    fn corrupt_state_is_reported_and_not_rewritten() -> Result<()> {
        let curator = temp_curator();
        let original = b"{not-json";
        if let Some(parent) = curator.state_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&curator.state_path, original)?;

        assert!(curator.load_state().is_err());
        assert!(curator.status().is_err());
        assert!(curator.register_candidate("must-not-overwrite").is_err());
        assert_eq!(std::fs::read(&curator.state_path)?, original);
        Ok(())
    }
}
