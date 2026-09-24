/*! 面向 `echo-agent` 贡献者的可执行 Rust 教学示例。

本 crate 仅作为工作区内部的教学辅助工具存在。生产环境的 crate 永远不会依赖
它；相反，它的示例代码使用的是与下游用户相同的公共 API。
*/

pub mod async_concurrency;
pub mod basics;
pub mod collections;
pub mod errors;
pub mod fundamentals;
pub mod ownership;
pub mod project_patterns;
pub mod serialization;
pub mod smart_pointers;
pub mod traits;
