//! 框架自有执行准入（execution admission）机制的兼容性导出。

pub use echo_core::agent::admission::{
    ExecutionAdmission, KeyedExecutionAdmission, KeyedExecutionAdmissionError, KeyedExecutionLease,
    KeyedExecutionRetirement,
};
