//! `Box<T>` 用于表示递归数据结构，`Box<dyn Trait>` 用于实现动态分发。

use crate::traits::{PlainFormatter, TaskFormatter};
use std::ops::Deref;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// 一个简单的计划节点，可以是单个步骤，也可以是两个子节点的顺序组合。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanNode {
    /// 一个具名的单一步骤。
    Step(String),
    /// 依次执行的两个子计划节点（借助 `Box` 实现递归结构）。
    Sequence(Box<PlanNode>, Box<PlanNode>),
}

impl PlanNode {
    /// 递归统计该计划节点（含所有子节点）的总数。
    pub fn node_count(&self) -> usize {
        match self {
            Self::Step(_) => 1,
            Self::Sequence(left, right) => 1usize
                .saturating_add(left.node_count())
                .saturating_add(right.node_count()),
        }
    }
}

/// 返回一个装箱的 `TaskFormatter` trait object，用于演示动态分发。
pub fn boxed_formatter() -> Box<dyn TaskFormatter> {
    Box::new(PlainFormatter)
}

/// 一个演示解引用强制转换（deref coercion）到 `str` 的小型 newtype。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptBox(Box<str>);

impl PromptBox {
    /// 从任意可转换为 `Box<str>` 的值构造 `PromptBox`。
    pub fn new(prompt: impl Into<Box<str>>) -> Self {
        Self(prompt.into())
    }
}

impl Deref for PromptBox {
    type Target = str;

    /// 将 `PromptBox` 解引用为 `&str`。
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// 该计数器使测试和示例中可以观察到确定性的 RAII 清理过程。
#[derive(Debug)]
pub struct DropCounter {
    /// 记录已发生的 drop 次数的共享计数器。
    dropped: Arc<AtomicUsize>,
}

impl DropCounter {
    /// 使用给定的共享计数器创建一个 `DropCounter`。
    pub fn new(dropped: Arc<AtomicUsize>) -> Self {
        Self { dropped }
    }
}

impl Drop for DropCounter {
    /// 实例被析构时，将共享计数器加一。
    fn drop(&mut self) {
        self.dropped.fetch_add(1, Ordering::SeqCst);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 验证 `Box` 使递归枚举拥有已知的固定大小，并能正确统计节点数。
    #[test]
    fn box_gives_recursive_enum_a_known_size() {
        let plan = PlanNode::Sequence(
            Box::new(PlanNode::Step("research".to_string())),
            Box::new(PlanNode::Step("write".to_string())),
        );
        assert_eq!(plan.node_count(), 3);
    }

    /// 验证解引用强制转换可正常工作，且 `Drop` 会在作用域结束时被确定性地触发。
    #[test]
    fn deref_coercion_and_drop_are_observable() {
        let prompt = PromptBox::new("hello");
        assert_eq!(prompt.len(), 5);

        let dropped = Arc::new(AtomicUsize::new(0));
        {
            let _guard = DropCounter::new(Arc::clone(&dropped));
            assert_eq!(dropped.load(Ordering::SeqCst), 0);
        }
        assert_eq!(dropped.load(Ordering::SeqCst), 1);
    }
}
