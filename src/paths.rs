//! 显式指定的框架数据根路径。

use std::path::{Path, PathBuf};

/**
由应用方选择的、用于框架持久化的根路径。

框架本身从不选择家目录名称或产品品牌。将克隆后的值传给
需要持久化的存储与服务即可。
*/
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataRoot {
    /// 持久化数据所在的根目录路径。
    root: PathBuf,
}

impl DataRoot {
    /// 使用给定路径构造一个 `DataRoot`。
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// 返回根路径的只读引用。
    pub fn as_path(&self) -> &Path {
        &self.root
    }

    /// 返回根路径下拼接指定子路径后得到的完整路径。
    pub fn path(&self, child: impl AsRef<Path>) -> PathBuf {
        self.root.join(child)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn joins_under_explicit_root() {
        let root = DataRoot::new("/tmp/echo-agent-consumer");
        assert_eq!(
            root.path("store.json"),
            PathBuf::from("/tmp/echo-agent-consumer/store.json")
        );
    }
}
