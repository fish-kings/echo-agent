/*!
# echo-state

[echo-agent](https://crates.io/crates/echo_agent) 框架的状态管理层。

## 模块

| 模块 | 说明 |
|--------|-------------|
| [`memory`] | 长期 KV 记忆（`Store`）+ 面向用户的对话记录投影（`ConversationStore`） |
| [`compression`] | 上下文压缩：滑动窗口、LLM 摘要、混合策略 |
| [`audit`] | 可插拔后端（内存、文件）的结构化审计日志 |
| [`journal`] | 带检查点复合归约的顺序事件日志 |
| [`skill_telemetry`] | 技能执行遥测：激活跟踪、成功/失败指标 |
| [`profiles`] | Agent 能力画像 + 用户偏好画像，支持提示词注入 |

## Feature 开关

- `sqlite` — 启用 `SqliteStore`，提供基于磁盘持久化的记忆存储

大多数用户应依赖门面 crate `echo_agent`，而不是直接依赖 `echo_state`。
*/

/// 结构化审计日志及其可插拔后端。
pub mod audit;
/// 上下文压缩策略。
pub mod compression;
/// 基于事件日志的投递账本。
pub mod delivery;
/// 顺序事件日志与检查点归约。
pub mod journal;
/// 长期记忆与对话记录存储。
pub mod memory;
/// Agent 能力画像与用户偏好画像。
pub mod profiles;
/// 技能执行遥测。
pub mod skill_telemetry;
/// crate 内部通用工具函数。
mod util;
