/*!
与具体产品无关的、基于键的执行准入控制。

本模块只负责管理不透明执行键（opaque execution key）的生命周期。应用层
自行决定某个键代表什么含义，以及它选中哪个 Agent 或工作区。准入控制的
所有者负责跟踪活跃租约、每个活跃键对应的一个进程许可、退役围栏
（retirement fence）以及关闭等待，且不依赖任何应用层状态或持久化。
*/

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, MutexGuard};

use tokio::sync::{Notify, OwnedSemaphorePermit, Semaphore};

/// 在准入或退役一个键控执行时返回的错误。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KeyedExecutionAdmissionError {
    /// 调用 [`KeyedExecutionAdmission::close`] 之后，新租约一律被拒绝。
    Closed,
    /// 该键正被一个进行中的退役操作围栏（fence）保护。
    Retiring {
        /// 被围栏保护的执行键。
        key: String,
    },
    /// 针对同一个键请求了第二个退役凭据。
    RetirementAlreadyActive {
        /// 已存在活跃退役凭据的执行键。
        key: String,
    },
    /// 租约计数器将会发生溢出。
    CapacityOverflow {
        /// 计数即将溢出的执行键。
        key: String,
    },
    /// 提供的进程信号量已没有可用许可。
    ProcessCapacity,
}

impl std::fmt::Display for KeyedExecutionAdmissionError {
    /// 将该错误格式化为人类可读的描述文本。
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Closed => formatter.write_str("keyed execution admission is closed"),
            Self::Retiring { key } => write!(formatter, "key '{key}' is retiring"),
            Self::RetirementAlreadyActive { key } => {
                write!(formatter, "key '{key}' already has an active retirement")
            }
            Self::CapacityOverflow { key } => {
                write!(
                    formatter,
                    "execution lease capacity overflow for key '{key}'"
                )
            }
            Self::ProcessCapacity => formatter.write_str("process execution capacity exhausted"),
        }
    }
}

impl std::error::Error for KeyedExecutionAdmissionError {}

/// 准入状态内部记录：接受标志、租约计数与各类占用信息。
#[derive(Default)]
struct AdmissionState {
    /// 是否仍在接受新租约。
    accepting: bool,
    /// 所有键累计的活跃租约总数。
    total: usize,
    /// 每个键当前持有的活跃租约数量。
    by_key: HashMap<String, usize>,
    /// 每个键持有的进程级信号量许可（若有）。
    process_permits: HashMap<String, OwnedSemaphorePermit>,
    /// 当前正处于退役围栏保护中的键集合。
    retiring: HashSet<String>,
}

impl AdmissionState {
    /// 构造一个处于“接受中”状态的初始记录。
    fn accepting() -> Self {
        Self {
            accepting: true,
            ..Self::default()
        }
    }
}

/**
用于不透明键控执行准入的共享权威对象。

消费者可以对共享同一个 Agent 或其他执行资源的所有租约使用同一个键。
复用某个键不会消耗额外的进程许可；该许可会一直持有，直到该键的最后一个
租约被释放为止。

```
use std::sync::Arc;
use tokio::sync::Semaphore;
use echo_core::agent::admission::KeyedExecutionAdmission;

# async fn example() -> Result<(), Box<dyn std::error::Error>> {
let admission = Arc::new(KeyedExecutionAdmission::default());
let process_limit = Arc::new(Semaphore::new(1));
let lease = admission.issue_process_scoped("conversation", &process_limit)?;
assert_eq!(admission.active_count(), 1);
drop(lease);
admission.wait_until_idle().await;
assert_eq!(admission.active_count(), 0);
# Ok(())
# }
```
*/
pub struct KeyedExecutionAdmission {
    /// 受互斥锁保护的内部准入状态。
    state: Mutex<AdmissionState>,
    /// 用于在状态变为空闲时唤醒等待者的通知器。
    idle: Notify,
}

impl Default for KeyedExecutionAdmission {
    /// 创建一个默认处于“接受中”状态的准入实例。
    fn default() -> Self {
        Self {
            state: Mutex::new(AdmissionState::accepting()),
            idle: Notify::new(),
        }
    }
}

impl KeyedExecutionAdmission {
    /// 加锁并返回内部状态的锁守卫；若锁中毒则从中恢复。
    fn lock_state(&self) -> MutexGuard<'_, AdmissionState> {
        match self.state.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        }
    }

    /// 在不使用进程级信号量的情况下准入一个租约。
    pub fn issue(
        self: &Arc<Self>,
        key: impl Into<String>,
    ) -> Result<KeyedExecutionLease, KeyedExecutionAdmissionError> {
        self.issue_inner(key.into(), None)
    }

    /**
    准入一个租约，并为新键预留一个进程许可。

    针对已激活键的额外租约会复用该键已有的许可，最后一个租约被释放时
    该许可才会被归还。
    */
    pub fn issue_process_scoped(
        self: &Arc<Self>,
        key: impl Into<String>,
        process_semaphore: &Arc<Semaphore>,
    ) -> Result<KeyedExecutionLease, KeyedExecutionAdmissionError> {
        self.issue_inner(key.into(), Some(Arc::clone(process_semaphore)))
    }

    /**
    `issue` 与 `issue_process_scoped` 的共用实现：处理准入检查、
    已有键的计数复用，以及进程许可的获取与释放。
    */
    fn issue_inner(
        self: &Arc<Self>,
        key: String,
        process_semaphore: Option<Arc<Semaphore>>,
    ) -> Result<KeyedExecutionLease, KeyedExecutionAdmissionError> {
        {
            let mut state = self.lock_state();
            self.ensure_accepting(&state, &key)?;
            if self.increment_existing(&mut state, &key)? {
                return Ok(KeyedExecutionLease {
                    admission: Arc::clone(self),
                    key,
                    active: true,
                });
            }
        }

        let process_permit = match process_semaphore {
            Some(semaphore) => match semaphore.try_acquire_owned() {
                Ok(permit) => Some(permit),
                Err(tokio::sync::TryAcquireError::Closed) => {
                    return Err(KeyedExecutionAdmissionError::Closed);
                }
                Err(tokio::sync::TryAcquireError::NoPermits) => {
                    let mut state = self.lock_state();
                    self.ensure_accepting(&state, &key)?;
                    if self.increment_existing(&mut state, &key)? {
                        return Ok(KeyedExecutionLease {
                            admission: Arc::clone(self),
                            key,
                            active: true,
                        });
                    }
                    return Err(KeyedExecutionAdmissionError::ProcessCapacity);
                }
            },
            None => None,
        };

        let mut state = self.lock_state();
        self.ensure_accepting(&state, &key)?;
        if self.increment_existing(&mut state, &key)? {
            drop(process_permit);
        } else {
            let total = state.total.checked_add(1).ok_or_else(|| {
                KeyedExecutionAdmissionError::CapacityOverflow { key: key.clone() }
            })?;
            if let Some(permit) = process_permit {
                state.process_permits.insert(key.clone(), permit);
            }
            state.total = total;
            state.by_key.insert(key.clone(), 1);
        }

        Ok(KeyedExecutionLease {
            admission: Arc::clone(self),
            key,
            active: true,
        })
    }

    /// 校验准入是否仍处于开放状态，且该键未被退役围栏保护。
    fn ensure_accepting(
        &self,
        state: &AdmissionState,
        key: &str,
    ) -> Result<(), KeyedExecutionAdmissionError> {
        if !state.accepting {
            return Err(KeyedExecutionAdmissionError::Closed);
        }
        if state.retiring.contains(key) {
            return Err(KeyedExecutionAdmissionError::Retiring {
                key: key.to_string(),
            });
        }
        Ok(())
    }

    /**
    若该键已有活跃租约，则为其计数加一并返回 `true`；
    若该键尚未激活，返回 `false`，调用方需按新键流程处理。
    */
    fn increment_existing(
        &self,
        state: &mut AdmissionState,
        key: &str,
    ) -> Result<bool, KeyedExecutionAdmissionError> {
        let Some(existing) = state.by_key.get(key).copied() else {
            return Ok(false);
        };
        let next_key_count = existing.checked_add(1).ok_or_else(|| {
            KeyedExecutionAdmissionError::CapacityOverflow {
                key: key.to_string(),
            }
        })?;
        state.total = state.total.checked_add(1).ok_or_else(|| {
            KeyedExecutionAdmissionError::CapacityOverflow {
                key: key.to_string(),
            }
        })?;
        state.by_key.insert(key.to_string(), next_key_count);
        Ok(true)
    }

    /// 当前是否至少有一个租约持有 `key`。
    pub fn is_active(&self, key: &str) -> bool {
        self.lock_state()
            .by_key
            .get(key)
            .copied()
            .unwrap_or_default()
            != 0
    }

    /// `key` 的新租约当前是否被一个退役凭据围栏保护。
    pub fn is_retiring(&self, key: &str) -> bool {
        self.lock_state().retiring.contains(key)
    }

    /// 所有键当前持有的已签发租约总数。
    pub fn active_count(&self) -> usize {
        self.lock_state().total
    }

    /// 永久拒绝新租约，同时允许已有租约正常结算完成。
    pub fn close(&self) {
        self.lock_state().accepting = false;
        self.idle.notify_waiters();
    }

    /// 围栏保护一个键，直至返回的凭据被释放为止。
    pub fn begin_retirement(
        self: &Arc<Self>,
        key: impl Into<String>,
    ) -> Result<KeyedExecutionRetirement, KeyedExecutionAdmissionError> {
        let key = key.into();
        let mut state = self.lock_state();
        if !state.accepting {
            return Err(KeyedExecutionAdmissionError::Closed);
        }
        if !state.retiring.insert(key.clone()) {
            return Err(KeyedExecutionAdmissionError::RetirementAlreadyActive { key });
        }
        drop(state);
        Ok(KeyedExecutionRetirement {
            admission: Arc::clone(self),
            key,
            active: true,
        })
    }

    /// 等待，直至某个键的所有租约都已被释放。
    pub async fn wait_key_idle(&self, key: &str) {
        loop {
            let notified = self.idle.notified();
            if !self.is_active(key) {
                return;
            }
            notified.await;
        }
    }

    /// 等待，直至所有已签发的租约都已被释放。
    pub async fn wait_until_idle(&self) {
        loop {
            let notified = self.idle.notified();
            if self.active_count() == 0 {
                return;
            }
            notified.await;
        }
    }
}

/// 不可克隆的租约，持有一次键控执行准入。
#[must_use]
pub struct KeyedExecutionLease {
    /// 签发该租约的准入所有者。
    admission: Arc<KeyedExecutionAdmission>,
    /// 该租约所属的执行键。
    key: String,
    /// 该租约当前是否仍处于活跃（未释放）状态。
    active: bool,
}

impl KeyedExecutionLease {
    /// 该租约是否恰好属于给定的准入所有者与键。
    pub fn owns(&self, admission: &Arc<KeyedExecutionAdmission>, key: &str) -> bool {
        self.active && Arc::ptr_eq(&self.admission, admission) && self.key == key
    }
}

impl Drop for KeyedExecutionLease {
    /// 释放租约：递减计数，必要时归还进程许可，并在相关键或整体进入空闲时唤醒等待者。
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        self.active = false;
        let mut state = self.admission.lock_state();
        state.total = state.total.saturating_sub(1);
        let mut release_process_permit = false;
        if let Some(count) = state.by_key.get_mut(&self.key) {
            *count = count.saturating_sub(1);
            if *count == 0 {
                state.by_key.remove(&self.key);
                release_process_permit = true;
            }
        }
        if release_process_permit {
            state.process_permits.remove(&self.key);
        }
        let key_idle = !state.by_key.contains_key(&self.key);
        let all_idle = state.total == 0;
        drop(state);
        if key_idle || all_idle {
            self.admission.idle.notify_waiters();
        }
    }
}

/// 不可克隆的围栏，拒绝某个键的新租约，直至该值被释放为止。
#[must_use]
pub struct KeyedExecutionRetirement {
    /// 签发该退役凭据的准入所有者。
    admission: Arc<KeyedExecutionAdmission>,
    /// 被该凭据围栏保护的执行键。
    key: String,
    /// 该凭据当前是否仍处于活跃（未释放）状态。
    active: bool,
}

/**
共享执行准入的组合边界。

该类型将键控生命周期跟踪与一个可选的进程级信号量组合在一起。消费者
在运行时和子代理入口之间共享这个值；键控原语始终是租约计数的唯一
所有者。
*/
pub struct ExecutionAdmission {
    /// 承担生命周期跟踪职责的键控准入原语。
    keyed: Arc<KeyedExecutionAdmission>,
    /// 可选的进程级容量限制信号量。
    process_semaphore: Option<Arc<Semaphore>>,
}

impl std::fmt::Debug for ExecutionAdmission {
    /// 输出当前活跃租约数与是否受进程容量限制的调试信息。
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ExecutionAdmission")
            .field("active_count", &self.active_count())
            .field("bounded", &self.process_semaphore.is_some())
            .finish()
    }
}

impl ExecutionAdmission {
    /// 创建一个带有进程级容量上限的准入实例。
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            keyed: Arc::new(KeyedExecutionAdmission::default()),
            process_semaphore: Some(Arc::new(Semaphore::new(capacity.max(1)))),
        }
    }

    /// 创建一个不做进程级容量限制的准入实例。
    pub fn unbounded() -> Self {
        Self {
            keyed: Arc::new(KeyedExecutionAdmission::default()),
            process_semaphore: None,
        }
    }

    /// 返回该组合所共享的键控权威对象。
    pub fn keyed(&self) -> &Arc<KeyedExecutionAdmission> {
        &self.keyed
    }

    /// 为一个不透明执行键签发一个租约。
    pub fn issue(
        &self,
        key: impl Into<String>,
    ) -> Result<KeyedExecutionLease, KeyedExecutionAdmissionError> {
        match &self.process_semaphore {
            Some(semaphore) => self.keyed.issue_process_scoped(key, semaphore),
            None => self.keyed.issue(key),
        }
    }

    /**
    等待直至能够签发一个租约。取消与超时仍由调用方通过在此 future
    外层包裹 `select!` 自行控制。
    */
    pub async fn issue_wait(
        &self,
        key: impl Into<String>,
    ) -> Result<KeyedExecutionLease, KeyedExecutionAdmissionError> {
        let key = key.into();
        loop {
            match self.issue(key.clone()) {
                Ok(lease) => return Ok(lease),
                Err(KeyedExecutionAdmissionError::ProcessCapacity) => {
                    tokio::task::yield_now().await;
                }
                Err(error) => return Err(error),
            }
        }
    }

    /// 所有键当前的活跃租约总数。
    pub fn active_count(&self) -> usize {
        self.keyed.active_count()
    }

    /// 关闭该准入实例并拒绝后续所有租约。
    pub fn close(&self) {
        self.keyed.close();
    }

    /// 等待，直至所有已签发的租约都已结算完毕。
    pub async fn wait_until_idle(&self) {
        self.keyed.wait_until_idle().await;
    }
}

impl Default for ExecutionAdmission {
    /// 等价于 `ExecutionAdmission::unbounded()`。
    fn default() -> Self {
        Self::unbounded()
    }
}

impl KeyedExecutionRetirement {
    /// 该退役凭据是否恰好属于给定的准入所有者与键。
    pub fn owns(&self, admission: &Arc<KeyedExecutionAdmission>, key: &str) -> bool {
        self.active && Arc::ptr_eq(&self.admission, admission) && self.key == key
    }
}

impl Drop for KeyedExecutionRetirement {
    /// 解除对该键的退役围栏保护，并唤醒等待者。
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        self.active = false;
        self.admission.lock_state().retiring.remove(&self.key);
        self.admission.idle.notify_waiters();
    }
}

/// 执行准入控制的单元测试。
#[cfg(test)]
mod tests {
    use super::*;

    /// 验证同一个键的多个租约只占用一个进程许可。
    #[tokio::test]
    async fn same_key_reuses_one_process_permit() -> Result<(), String> {
        let admission = Arc::new(KeyedExecutionAdmission::default());
        let semaphore = Arc::new(Semaphore::new(1));
        let first = admission
            .issue_process_scoped("same", &semaphore)
            .map_err(|error| error.to_string())?;
        let second = admission
            .issue_process_scoped("same", &semaphore)
            .map_err(|error| error.to_string())?;
        assert_eq!(admission.active_count(), 2);
        assert_eq!(semaphore.available_permits(), 0);
        drop(first);
        assert_eq!(admission.active_count(), 1);
        drop(second);
        admission.wait_until_idle().await;
        assert_eq!(semaphore.available_permits(), 1);
        Ok(())
    }

    /// 验证不同键之间受进程容量限制。
    #[tokio::test]
    async fn different_keys_observe_process_capacity() -> Result<(), String> {
        let admission = Arc::new(KeyedExecutionAdmission::default());
        let semaphore = Arc::new(Semaphore::new(1));
        let first = admission
            .issue_process_scoped("first", &semaphore)
            .map_err(|error| error.to_string())?;
        assert!(matches!(
            admission.issue_process_scoped("second", &semaphore),
            Err(KeyedExecutionAdmissionError::ProcessCapacity)
        ));
        drop(first);
        let second = admission
            .issue_process_scoped("second", &semaphore)
            .map_err(|error| error.to_string())?;
        drop(second);
        Ok(())
    }

    /// 验证退役围栏与关闭都会拒绝新租约。
    #[tokio::test]
    async fn retirement_and_close_are_fail_closed() -> Result<(), String> {
        let admission = Arc::new(KeyedExecutionAdmission::default());
        let retirement = admission
            .begin_retirement("retiring")
            .map_err(|error| error.to_string())?;
        assert!(matches!(
            admission.issue("retiring"),
            Err(KeyedExecutionAdmissionError::Retiring { key }) if key == "retiring"
        ));
        drop(retirement);
        let lease = admission
            .issue("retiring")
            .map_err(|error| error.to_string())?;
        admission.close();
        assert!(matches!(
            admission.issue("closed"),
            Err(KeyedExecutionAdmissionError::Closed)
        ));
        drop(lease);
        admission.wait_until_idle().await;
        Ok(())
    }

    /// 验证组合准入共享容量并能正确关闭。
    #[tokio::test]
    async fn execution_admission_shares_capacity_and_closes() -> Result<(), String> {
        let admission = ExecutionAdmission::with_capacity(1);
        let first = admission
            .issue("first")
            .map_err(|error| error.to_string())?;
        assert!(matches!(
            admission.issue("second"),
            Err(KeyedExecutionAdmissionError::ProcessCapacity)
        ));
        admission.close();
        drop(first);
        admission.wait_until_idle().await;
        assert!(matches!(
            admission.issue("third"),
            Err(KeyedExecutionAdmissionError::Closed)
        ));
        Ok(())
    }
}
