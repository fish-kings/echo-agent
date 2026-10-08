/*!
共享状态（SharedState）

图工作流中节点间通信的核心机制。各节点通过读写 `SharedState` 传递数据。
每个节点执行完毕后会自动对状态做检查点，支持恢复 / 重放。

## 设计原则

- **类型安全**：通过 `get::<T>()` / `set()` 结合 serde 序列化实现类型安全的 KV 访问
- **线程安全**：`Arc<RwLock>` 内部可变性，跨节点并发安全
- **可序列化**：整个 State 可快照为 JSON 以便持久化
- **结构化消息历史**：保留完整的 `Message` 结构（不丢失工具调用元数据）
*/

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use echo_core::error::ReactError;
use echo_core::llm::types::Message;

/// 递归合并两个 JSON 对象
fn deep_merge_values(
    target: &mut serde_json::Map<String, Value>,
    source: &serde_json::Map<String, Value>,
) {
    for (k, v) in source {
        if let Some(existing) = target.get_mut(k)
            && let (Some(obj_a), Some(obj_b)) = (existing.as_object_mut(), v.as_object())
        {
            deep_merge_values(obj_a, obj_b);
            continue;
        }
        target.insert(k.clone(), v.clone());
    }
}

/// SharedState 操作的返回类型
pub type StateResult<T> = std::result::Result<T, StateError>;

/// SharedState 操作可能产生的错误
#[derive(Debug)]
pub enum StateError {
    /// 序列化或反序列化失败。
    Serialize(String),
    /// 锁已中毒（另一个线程在持有锁时发生了 panic）。
    LockPoisoned(String),
    /// 反序列化已存储的值时类型不匹配。
    TypeMismatch {
        /// 出现类型不匹配的键
        key: String,
        /// 期望的类型
        expected: String,
        /// 实际发现的类型
        found: String,
    },
    /// 状态中缺少必需的键。
    MissingKey(String),
    /// 尝试执行了无效操作（例如对空状态执行 fork）。
    InvalidOperation(String),
}

impl std::fmt::Display for StateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StateError::Serialize(e) => write!(f, "Serialization failed: {e}"),
            StateError::LockPoisoned(e) => write!(f, "Lock poisoned: {e}"),
            StateError::TypeMismatch {
                key,
                expected,
                found,
            } => write!(
                f,
                "Type mismatch for key '{key}': expected {expected}, found {found}"
            ),
            StateError::MissingKey(key) => write!(f, "Missing key: '{key}'"),
            StateError::InvalidOperation(msg) => write!(f, "Invalid operation: {msg}"),
        }
    }
}

impl std::error::Error for StateError {}

impl From<StateError> for ReactError {
    fn from(e: StateError) -> Self {
        match e {
            StateError::Serialize(msg) => {
                ReactError::Other(format!("State serialization error: {msg}"))
            }
            StateError::LockPoisoned(msg) => {
                ReactError::Other(format!("State lock poisoned: {msg}"))
            }
            StateError::TypeMismatch {
                key,
                expected,
                found,
            } => ReactError::Other(format!(
                "State type mismatch for '{key}': expected {expected}, found {found}"
            )),
            StateError::MissingKey(key) => ReactError::Other(format!("State missing key: '{key}'")),
            StateError::InvalidOperation(msg) => {
                ReactError::Other(format!("State invalid operation: {msg}"))
            }
        }
    }
}

/**
图工作流的共享状态

节点通过 `get` / `set` 读写任意键值对，并通过 `messages()` 访问结构化的对话历史。
内部使用 `Arc<RwLock>`，可安全地在节点与线程之间共享。

# 示例

```rust
use echo_orchestration::workflow::SharedState;

let state = SharedState::new();
state.set("count", 42);
state.set("name", "echo");

assert_eq!(state.get::<i64>("count"), Some(42));
assert_eq!(state.get::<String>("name"), Some("echo".to_string()));
```
*/
#[derive(Clone)]
pub struct SharedState {
    /// 共享的内部状态
    inner: Arc<RwLock<StateInner>>,
}

/// 内部状态（可序列化）
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StateInner {
    /// 任意 KV 数据
    pub values: HashMap<String, Value>,
    /// 结构化消息历史
    pub messages: Vec<Message>,
    /// 当前节点（运行时信息）
    #[serde(default)]
    pub current_node: Option<String>,
}

impl SharedState {
    /// 创建空状态
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(StateInner::default())),
        }
    }

    /// 基于已有数据创建
    pub fn from_values(values: HashMap<String, Value>) -> Self {
        Self {
            inner: Arc::new(RwLock::new(StateInner {
                values,
                messages: Vec::new(),
                current_node: None,
            })),
        }
    }

    /// 从快照恢复
    pub fn from_snapshot(snapshot: &str) -> std::result::Result<Self, serde_json::Error> {
        let inner: StateInner = serde_json::from_str(snapshot)?;
        Ok(Self {
            inner: Arc::new(RwLock::new(inner)),
        })
    }

    // ── KV 操作 ─────────────────────────────────────────────────────────────

    /**
    设置一个值（自动序列化为 JSON）。

    返回 Result（`StateResult<()>`）——序列化失败或锁中毒时不再 panic。
    调用方必须处理返回的错误，而不是丢弃失败的状态更新。
    */
    pub fn set<T: Serialize>(&self, key: impl Into<String>, value: T) -> StateResult<()> {
        let key = key.into();
        let v = serde_json::to_value(value).map_err(|e| StateError::Serialize(e.to_string()))?;
        let mut inner = self
            .inner
            .write()
            .map_err(|e| StateError::LockPoisoned(e.to_string()))?;
        inner.values.insert(key, v);
        drop(inner);
        Ok(())
    }

    /// 向后兼容：设置值，失败时返回 Option<()>（不会 panic）
    pub fn set_best_effort<T: Serialize>(&self, key: impl Into<String>, value: T) -> Option<()> {
        self.set(key, value).ok()
    }

    /// 获取一个值（自动反序列化）
    pub fn get<T: for<'de> Deserialize<'de>>(&self, key: &str) -> Option<T> {
        let Ok(inner) = self.inner.read() else {
            return None;
        };
        inner
            .values
            .get(key)
            .and_then(|v| serde_json::from_value(v.clone()).ok())
    }

    /// 获取原始 JSON 值
    pub fn get_raw(&self, key: &str) -> Option<Value> {
        self.inner.read().ok()?.values.get(key).cloned()
    }

    /// 检查键是否存在
    pub fn contains(&self, key: &str) -> bool {
        self.inner
            .read()
            .map(|inner| inner.values.contains_key(key))
            .unwrap_or(false)
    }

    /// 移除一个键
    pub fn remove(&self, key: &str) -> Option<Value> {
        self.inner.write().ok()?.values.remove(key)
    }

    /// 获取所有键
    pub fn keys(&self) -> Vec<String> {
        self.inner
            .read()
            .map(|inner| inner.values.keys().cloned().collect())
            .unwrap_or_default()
    }

    // ── 消息操作 ────────────────────────────────────────────────────────────

    /// 追加一条消息，返回 Result
    pub fn push_message(&self, msg: Message) -> StateResult<()> {
        let mut inner = self
            .inner
            .write()
            .map_err(|e| StateError::LockPoisoned(e.to_string()))?;
        inner.messages.push(msg);
        drop(inner);
        Ok(())
    }

    /// 获取所有消息的克隆
    pub fn messages(&self) -> Vec<Message> {
        self.inner
            .read()
            .map(|inner| inner.messages.clone())
            .unwrap_or_default()
    }

    /// 获取消息数量
    pub fn message_count(&self) -> usize {
        self.inner
            .read()
            .map(|inner| inner.messages.len())
            .unwrap_or(0)
    }

    /// 清空消息，返回 Result
    pub fn clear_messages(&self) -> StateResult<()> {
        let mut inner = self
            .inner
            .write()
            .map_err(|e| StateError::LockPoisoned(e.to_string()))?;
        inner.messages.clear();
        drop(inner);
        Ok(())
    }

    // ── 节点追踪 ────────────────────────────────────────────────────────────

    /// 设置当前节点
    pub(crate) fn set_current_node(&self, node: impl Into<String>) {
        if let Ok(mut inner) = self.inner.write() {
            inner.current_node = Some(node.into());
        }
    }

    /// 获取当前节点
    pub fn current_node(&self) -> Option<String> {
        self.inner
            .read()
            .ok()
            .and_then(|inner| inner.current_node.clone())
    }

    /// 为分支执行创建一份隔离副本。
    pub fn fork(&self) -> StateResult<Self> {
        let inner = self
            .inner
            .read()
            .map_err(|e| StateError::LockPoisoned(e.to_string()))?
            .clone();
        Ok(Self {
            inner: Arc::new(RwLock::new(inner)),
        })
    }

    // ── 序列化 ──────────────────────────────────────────────────────────────

    /// 导出为 JSON 快照，返回 Result（不再 panic）
    pub fn snapshot(&self) -> StateResult<String> {
        let inner = self
            .inner
            .read()
            .map_err(|e| StateError::LockPoisoned(e.to_string()))?;
        serde_json::to_string_pretty(&*inner).map_err(|e| StateError::Serialize(e.to_string()))
    }

    /// 导出为 JSON Value，返回 Result
    pub fn to_json_value(&self) -> StateResult<serde_json::Value> {
        let inner = self
            .inner
            .read()
            .map_err(|e| StateError::LockPoisoned(e.to_string()))?;
        serde_json::to_value(&*inner).map_err(|e| StateError::Serialize(e.to_string()))
    }

    /// 从 JSON Value 恢复
    pub fn from_json(json: &serde_json::Value) -> std::result::Result<Self, serde_json::Error> {
        let inner: StateInner = serde_json::from_value(json.clone())?;
        Ok(Self {
            inner: Arc::new(RwLock::new(inner)),
        })
    }

    /// 合并另一个状态中的值（不覆盖已有键），返回 Result
    pub fn merge(&self, other: &SharedState) -> StateResult<()> {
        let other_inner = self
            .inner
            .read()
            .map_err(|e| StateError::LockPoisoned(e.to_string()))?;
        drop(other_inner);
        let self_inner = self
            .inner
            .read()
            .map_err(|e| StateError::LockPoisoned(e.to_string()))?;
        /* SAFETY：需要读取两把锁以防止死锁。
        merge 从 other 读取并写入 self，因此按顺序重新加锁 */
        drop(self_inner);

        let other_lock = other
            .inner
            .read()
            .map_err(|e| StateError::LockPoisoned(e.to_string()))?;
        let mut self_lock = self
            .inner
            .write()
            .map_err(|e| StateError::LockPoisoned(e.to_string()))?;
        for (k, v) in &other_lock.values {
            self_lock
                .values
                .entry(k.clone())
                .or_insert_with(|| v.clone());
        }
        drop(other_lock);
        drop(self_lock);
        Ok(())
    }

    /// 合并另一个状态中的值（覆盖已有键），返回 Result
    pub fn merge_overwrite(&self, other: &SharedState) -> StateResult<()> {
        let other_lock = other
            .inner
            .read()
            .map_err(|e| StateError::LockPoisoned(e.to_string()))?;
        let mut self_lock = self
            .inner
            .write()
            .map_err(|e| StateError::LockPoisoned(e.to_string()))?;
        for (k, v) in &other_lock.values {
            self_lock.values.insert(k.clone(), v.clone());
        }
        drop(other_lock);
        drop(self_lock);
        Ok(())
    }

    /**
    深度合并另一个状态中的值（递归合并嵌套结构），返回 Result

    与 `merge_overwrite` 不同：当某个键在两个状态中的值都是 JSON 对象时，
    本方法会递归合并两个对象的字段，而不是整体覆盖。
    适用于并行分支修改同一状态中的嵌套结构时保持数据一致性。
    */
    pub fn deep_merge(&self, other: &SharedState) -> StateResult<()> {
        let other_lock = other
            .inner
            .read()
            .map_err(|e| StateError::LockPoisoned(e.to_string()))?;
        let mut self_lock = self
            .inner
            .write()
            .map_err(|e| StateError::LockPoisoned(e.to_string()))?;
        for (k, other_val) in &other_lock.values {
            if let Some(self_val) = self_lock.values.get(k) {
                // 若两者都是对象，则递归合并
                if let (Some(self_obj), Some(other_obj)) =
                    (self_val.as_object(), other_val.as_object())
                {
                    let mut merged = self_obj.clone();
                    deep_merge_values(&mut merged, other_obj);
                    self_lock.values.insert(k.clone(), Value::Object(merged));
                    continue;
                }
            }
            // 否则直接覆盖
            self_lock.values.insert(k.clone(), other_val.clone());
        }
        drop(other_lock);
        drop(self_lock);
        Ok(())
    }
}

impl Default for SharedState {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for SharedState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.inner.read() {
            Ok(inner) => f
                .debug_struct("SharedState")
                .field("keys", &inner.values.keys().collect::<Vec<_>>())
                .field("messages", &inner.messages.len())
                .field("current_node", &inner.current_node)
                .finish(),
            Err(_) => f
                .debug_struct("SharedState")
                .field("error", &"lock poisoned")
                .finish(),
        }
    }
}

// ── 单元测试 ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_set_get_typed() {
        let state = SharedState::new();
        state.set("count", 42i64).unwrap();
        state.set("name", "echo").unwrap();
        state.set("tags", vec!["a", "b"]).unwrap();

        assert_eq!(state.get::<i64>("count"), Some(42));
        assert_eq!(state.get::<String>("name"), Some("echo".to_string()));
        assert_eq!(
            state.get::<Vec<String>>("tags"),
            Some(vec!["a".to_string(), "b".to_string()])
        );
        assert_eq!(state.get::<i64>("missing"), None);
    }

    #[test]
    fn test_contains_remove() {
        let state = SharedState::new();
        state.set("x", 1).unwrap();
        assert!(state.contains("x"));
        assert!(!state.contains("y"));

        state.remove("x");
        assert!(!state.contains("x"));
    }

    #[test]
    fn test_messages() {
        let state = SharedState::new();
        state
            .push_message(Message::user("hello".to_string()))
            .unwrap();
        state
            .push_message(Message::assistant("hi".to_string()))
            .unwrap();

        assert_eq!(state.message_count(), 2);
        let msgs = state.messages();
        assert_eq!(msgs[0].role, "user".into());
        assert_eq!(msgs[1].role, "assistant".into());
    }

    #[test]
    fn test_snapshot_restore() {
        let state = SharedState::new();
        state.set("key", "value").unwrap();
        state
            .push_message(Message::user("hello".to_string()))
            .unwrap();

        let snap = state.snapshot().unwrap();
        let restored = SharedState::from_snapshot(&snap).unwrap();

        assert_eq!(restored.get::<String>("key"), Some("value".to_string()));
        assert_eq!(restored.message_count(), 1);
    }

    #[test]
    fn test_merge() {
        let a = SharedState::new();
        a.set("x", 1).unwrap();
        a.set("shared", "from_a").unwrap();

        let b = SharedState::new();
        b.set("y", 2).unwrap();
        b.set("shared", "from_b").unwrap();

        a.merge(&b).unwrap();
        assert_eq!(a.get::<i64>("x"), Some(1));
        assert_eq!(a.get::<i64>("y"), Some(2));
        assert_eq!(a.get::<String>("shared"), Some("from_a".to_string())); // 未被覆盖
    }

    #[test]
    fn test_merge_overwrite() {
        let a = SharedState::new();
        a.set("shared", "from_a").unwrap();

        let b = SharedState::new();
        b.set("shared", "from_b").unwrap();

        a.merge_overwrite(&b).unwrap();
        assert_eq!(a.get::<String>("shared"), Some("from_b".to_string()));
    }

    #[test]
    fn test_clone_shares_state() {
        let state = SharedState::new();
        let cloned = state.clone();

        state.set("x", 42).unwrap();
        assert_eq!(cloned.get::<i64>("x"), Some(42)); // Arc 共享
    }

    #[test]
    fn test_from_values() {
        let mut vals = HashMap::new();
        vals.insert("key".to_string(), serde_json::json!("value"));

        let state = SharedState::from_values(vals);
        assert_eq!(state.get::<String>("key"), Some("value".to_string()));
    }
}
