/*!
内置的框架级工具

这些工具是 Agent 执行循环的核心。领域相关的工具
（git、浏览器、数据、媒体等）位于 `echo_tools` crate 中。
*/

#[cfg(feature = "subagent")]
pub(crate) mod agent_dispatch;
pub(crate) mod answer;
pub(crate) mod cell_tools;
/**
可由模型调用的人机协作工具。设为公开是为了让替换了
审批提供方（approval provider，例如 SDK 扩展桥接层）的
嵌入方，能在自己构建的 Agent 上注册匹配的申诉工具。
*/
#[cfg(feature = "human-loop")]
pub mod human_in_loop;
pub(crate) mod memory;
#[cfg(feature = "subagent")]
pub(crate) mod subagent_message;
/// 用于推理与反思的 Think 工具。
pub mod think;
