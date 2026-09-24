//! `Rc<T>` 与 `RefCell<T>` 用于单线程场景下的共享所有权。

use crate::errors::LearningError;
use std::cell::RefCell;
use std::rc::Rc;

/// 可被多处克隆共享、且能在运行时检查借用规则的笔记本。
#[derive(Clone, Default)]
pub struct SharedNotebook {
    /// 通过 `Rc<RefCell<..>>` 共享的条目列表。
    entries: Rc<RefCell<Vec<String>>>,
}

impl SharedNotebook {
    /// 创建一个空的共享笔记本。
    pub fn new() -> Self {
        Self::default()
    }

    /// 追加一条记录；若运行时借用冲突则返回错误而非 panic。
    pub fn add(&self, entry: impl Into<String>) -> Result<(), LearningError> {
        self.entries
            .try_borrow_mut()
            .map_err(|_| LearningError::BorrowConflict)?
            .push(entry.into());
        Ok(())
    }

    /// 返回当前所有条目的克隆副本；若运行时借用冲突则返回错误。
    pub fn entries(&self) -> Result<Vec<String>, LearningError> {
        self.entries
            .try_borrow()
            .map(|entries| entries.clone())
            .map_err(|_| LearningError::BorrowConflict)
    }

    /// 返回底层 `Rc` 的强引用计数。
    pub fn strong_count(&self) -> usize {
        Rc::strong_count(&self.entries)
    }

    /// 在尝试写入的同时保持一个读守卫存活，从而产生一个被妥善处理的
    /// 运行时借用冲突，而不是触发 panic。
    pub fn demonstrate_conflict(&self) -> Result<(), LearningError> {
        let _reading = self
            .entries
            .try_borrow()
            .map_err(|_| LearningError::BorrowConflict)?;
        self.add("this write conflicts with the active read")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 验证克隆出的笔记本与原笔记本共享同一份底层数据。
    #[test]
    fn clones_share_the_same_notebook() -> Result<(), LearningError> {
        let first = SharedNotebook::new();
        let second = first.clone();
        second.add("shared entry")?;
        assert_eq!(first.entries()?, vec!["shared entry"]);
        assert_eq!(first.strong_count(), 2);
        Ok(())
    }

    /// 验证运行时的借用冲突会以错误形式返回，而不是导致 panic。
    #[test]
    fn runtime_borrow_conflict_is_returned_as_an_error() {
        let notebook = SharedNotebook::new();
        assert_eq!(
            notebook.demonstrate_conflict(),
            Err(LearningError::BorrowConflict)
        );
    }
}
