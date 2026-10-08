/*!
# echo_orchestration

[echo-agent](https://crates.io/crates/echo_agent) 框架的编排层。

## 模块

| 模块 | 说明 |
|--------|-------------|
| [`workflow`] | 图工作流引擎 —— `GraphBuilder`、`Graph`、流式事件 |
| [`human_loop`] | 人工介入审批 —— 控制台、Webhook、WebSocket、权限流水线 |
| [`tasks`] | 可修订任务图、DAG 执行、调度与进度事件 |
| [`scheduler`] | 基于 Cron 的任务调度 —— `CronTask`、`SchedulerRunner` |
| [`planning`] | 面向可修订任务图的结构校验 |
| [`runtime`] | 通用回合驱动器 —— `AgentTurnDriver`、`TurnRequest`、执行与投递结果 |

## Feature 开关

- `websocket` —— 启用基于 WebSocket 的审批实现 `WebSocketHumanLoopProvider`

大多数使用者应当依赖门面 crate `echo_agent`，而不是直接依赖
`echo_orchestration`。
*/

/// 人工介入审批：控制台、Webhook、WebSocket 与权限流水线
pub mod human_loop;
/// 面向可修订任务图的结构校验
pub mod planning;
/// 通用回合驱动器与执行结果类型
pub mod runtime;
/// 基于 Cron 与固定间隔的任务调度
pub mod scheduler;
/// 可修订任务图、DAG 执行、调度与进度事件
pub mod tasks;
/// 图工作流引擎与通用流水线工作流
pub mod workflow;
