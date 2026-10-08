/*!
重试机制门面模块。

对 `echo_core::retry` 的薄层重新导出。
权威实现位于 `echo_core`。
*/

/// 直接从 `echo_core::retry` 重新导出。
pub mod core {
    pub use echo_core::retry::*;
}

pub use echo_core::retry::*;
