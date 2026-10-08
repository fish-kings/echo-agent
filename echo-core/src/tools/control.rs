/*!
供框架消费者共享的带版本号的工具可见性控制。

该控制器只负责调用方显式设置的“禁用工具”策略。Agent 的创建、能力过滤、
工作区范围以及外部回执，都仍由宿主应用自行负责。
*/

use std::collections::HashSet;
use std::sync::RwLock;

/// 修改框架工具可见性策略时可能返回的错误。
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ToolControlError {
    /// 工具名称为空。
    #[error("tool name cannot be empty")]
    EmptyName,
    /// 指定名称的工具尚未注册。
    #[error("tool '{name}' is not registered")]
    NotRegistered {
        /// 未注册的工具名称。
        name: String,
    },
    /// 版本号（generation）已耗尽，无法继续递增。
    #[error("tool-control generation is exhausted")]
    GenerationExhausted,
    /// 策略状态锁已被污染（poisoned），无法访问。
    #[error("tool-control policy lock is poisoned")]
    StateUnavailable,
}

/// 显式工具可见性策略的一份快照。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ToolControlSnapshot {
    /// 当前策略的版本号，每次策略发生实际变化时递增。
    pub revision: u64,
    /// 当前被禁用的工具名称集合。
    pub disabled_tools: HashSet<String>,
}

impl ToolControlSnapshot {
    /// 返回 Agent 调用时所需的禁用工具表示形式。
    pub fn disabled_option(&self) -> Option<HashSet<String>> {
        (!self.disabled_tools.is_empty()).then(|| self.disabled_tools.clone())
    }
}

/// 一次幂等策略变更的结果。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolControlMutation {
    /// 被修改的工具名称。
    pub name: String,
    /// 变更后该工具是否处于启用状态。
    pub policy_enabled: bool,
    /// 本次调用是否实际改变了策略状态。
    pub changed: bool,
    /// 变更后的策略版本号。
    pub revision: u64,
}

/// 共享的、带版本号的显式工具可见性权威控制器。
#[derive(Debug, Default)]
pub struct ToolControlService {
    /// 受读写锁保护的当前策略状态。
    state: RwLock<ToolControlSnapshot>,
}

impl ToolControlService {
    /// 读取当前的策略快照。
    pub fn snapshot(&self) -> Result<ToolControlSnapshot, ToolControlError> {
        self.state
            .read()
            .map(|state| state.clone())
            .map_err(|_| ToolControlError::StateUnavailable)
    }

    /**
    启用或禁用某个指定名称的工具。重复设置相同的状态是幂等的，
    不会推进版本号。
    */
    pub fn set_enabled(
        &self,
        name: &str,
        enabled: bool,
    ) -> Result<ToolControlMutation, ToolControlError> {
        let name = name.trim();
        if name.is_empty() {
            return Err(ToolControlError::EmptyName);
        }
        let mut state = self
            .state
            .write()
            .map_err(|_| ToolControlError::StateUnavailable)?;
        let changed = if enabled {
            state.disabled_tools.contains(name)
        } else {
            !state.disabled_tools.contains(name)
        };
        let revision = if changed {
            state
                .revision
                .checked_add(1)
                .ok_or(ToolControlError::GenerationExhausted)?
        } else {
            state.revision
        };
        if enabled {
            state.disabled_tools.remove(name);
        } else {
            state.disabled_tools.insert(name.to_string());
        }
        state.revision = revision;
        Ok(ToolControlMutation {
            name: name.to_string(),
            policy_enabled: enabled,
            changed,
            revision,
        })
    }
}
