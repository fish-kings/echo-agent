/*! # echo_execution

[echo-agent](https://crates.io/crates/echo_agent) 框架的执行层。

## 模块

| 模块 | 说明 |
|--------|-------------|
| [`sandbox`] | 多层代码执行：`LocalSandbox`、`DockerSandbox`、`K8sSandbox` |
| [`skills`] | 文件型 skill 系统：发现 → 激活 → 使用，并支持 hooks |
| [`tools`] | `ToolManager` —— 工具注册、执行、并发控制 |

大多数用户应依赖 `echo_agent`（门面 crate），而不是直接依赖 `echo_execution`。
*/

/// 风险分类：按工具名推断工具的风险类别。
pub mod risk;
/// 沙箱执行：本地、Docker、K8s 多层沙箱及选择策略。
pub mod sandbox;
/// 文件型 skill 系统：发现、激活、使用与 hooks。
pub mod skills;
/// 工具管理：注册、执行与并发控制。
pub mod tools;
