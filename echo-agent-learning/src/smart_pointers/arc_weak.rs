//! `Arc<T>` 用于线程安全的共享，`Weak<T>` 用于不持有所有权的引用。

use crate::errors::LearningError;
use std::collections::HashMap;
use std::sync::{Arc, RwLock, Weak};

/// 一个简单的 agent 句柄，仅携带名称。
#[derive(Debug, PartialEq, Eq)]
pub struct AgentHandle {
    /// agent 的名称。
    pub name: String,
}

/// 按名称索引 agent 的注册表，只持有对 `AgentHandle` 的弱引用，不会阻止其被释放。
#[derive(Debug, Default)]
pub struct AgentRegistry {
    /// 名称到 agent 弱引用的映射表。
    agents: RwLock<HashMap<String, Weak<AgentHandle>>>,
}

impl AgentRegistry {
    /// 注册一个 agent：保存其名称到弱引用的映射，不会延长其生命周期。
    pub fn register(&self, agent: &Arc<AgentHandle>) -> Result<(), LearningError> {
        self.agents
            .write()
            .map_err(|_| LearningError::PoisonedLock)?
            .insert(agent.name.clone(), Arc::downgrade(agent));
        Ok(())
    }

    /// 按名称查找 agent；若对应的强引用已全部释放，则返回 `None`。
    pub fn get(&self, name: &str) -> Result<Option<Arc<AgentHandle>>, LearningError> {
        Ok(self
            .agents
            .read()
            .map_err(|_| LearningError::PoisonedLock)?
            .get(name)
            .and_then(Weak::upgrade))
    }

    /// 移除所有已失效（强引用计数为 0）的注册项，返回被移除的数量。
    pub fn remove_expired(&self) -> Result<usize, LearningError> {
        let mut agents = self
            .agents
            .write()
            .map_err(|_| LearningError::PoisonedLock)?;
        let before = agents.len();
        agents.retain(|_, agent| agent.strong_count() > 0);
        Ok(before.saturating_sub(agents.len()))
    }

    /// 返回指定 agent 的强引用计数与弱引用计数。
    pub fn counts(agent: &Arc<AgentHandle>) -> (usize, usize) {
        (Arc::strong_count(agent), Arc::weak_count(agent))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 验证注册表仅持有弱引用，不会阻止 agent 在所有者释放后被回收。
    #[test]
    fn weak_registry_does_not_keep_agents_alive() -> Result<(), LearningError> {
        let registry = AgentRegistry::default();
        let agent = Arc::new(AgentHandle {
            name: "reviewer".to_string(),
        });
        registry.register(&agent)?;
        assert_eq!(AgentRegistry::counts(&agent), (1, 1));
        assert!(registry.get("reviewer")?.is_some());
        drop(agent);
        assert!(registry.get("reviewer")?.is_none());
        assert_eq!(registry.remove_expired()?, 1);
        Ok(())
    }
}
