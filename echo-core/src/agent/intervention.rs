//! 用于影响 agent 行为的干预回调接口
//!
//! 与只做观察的 `AgentCallback` 不同，`InterventionCallback` 可以*主动影响*
//! agent 的行为：阻断工具调用、注入上下文、重定向执行、或取消操作。
//!
//! `CallbackBridge` 适配器可以将普通的 `AgentCallback` 实例转换为
//! 中立的 `InterventionCallback` 实现，以保持向后兼容。

use crate::llm::types::Message;
use futures::future::BoxFuture;
use serde_json::Value;

/// 干预检查的结果，指示 agent 接下来应当如何处理。
#[derive(Debug, Clone, Default)]
pub struct InterventionResult {
    /// 是否阻断当前动作。
    pub block: bool,
    /// 阻断原因（展示给用户 / 记录日志）。
    pub block_reason: Option<String>,
    /// 需要注入到 agent 下一次 LLM 调用中的上下文。
    pub injected_context: Option<String>,
    /// 重定向到另一个工具名称（替换原本的工具调用）。
    pub redirect_to: Option<String>,
    /// 是否取消整个 agent 的执行。
    pub cancel: bool,
    /// 修改后的工具参数（若存在则替换原始参数）。
    pub modified_args: Option<Value>,
}

impl InterventionResult {
    /// 允许该动作正常继续执行。
    pub fn allow() -> Self {
        Self::default()
    }

    /// 附带原因阻断该动作。
    pub fn block(reason: impl Into<String>) -> Self {
        Self {
            block: true,
            block_reason: Some(reason.into()),
            ..Default::default()
        }
    }

    /// 向 agent 的推理过程注入额外上下文。
    pub fn inject(context: impl Into<String>) -> Self {
        Self {
            injected_context: Some(context.into()),
            ..Default::default()
        }
    }

    /// 取消整个 agent 的执行。
    pub fn cancel() -> Self {
        Self {
            cancel: true,
            ..Default::default()
        }
    }

    /// 在执行前修改工具参数。
    pub fn modify_args(args: Value) -> Self {
        Self {
            modified_args: Some(args),
            ..Default::default()
        }
    }
}

/// 可以影响 agent 行为的干预回调 trait。
///
/// 所有方法都提供了返回 `InterventionResult::allow()` 的默认实现，
/// 因此实现者只需要覆盖自己关心的方法即可。
pub trait InterventionCallback: Send + Sync {
    /// 在工具调用执行前进行检查（可能阻断/修改该调用）。
    fn on_tool_call<'a>(
        &'a self,
        _agent: &'a str,
        _tool: &'a str,
        _args: &'a Value,
    ) -> BoxFuture<'a, InterventionResult> {
        Box::pin(async { InterventionResult::allow() })
    }

    /// 在 LLM 推理开始前进行检查（可能注入上下文或取消执行）。
    fn on_think_start<'a>(
        &'a self,
        _agent: &'a str,
        _messages: &'a [Message],
    ) -> BoxFuture<'a, InterventionResult> {
        Box::pin(async { InterventionResult::allow() })
    }

    /// 在返回最终答案前进行检查（可能阻断该答案）。
    fn on_final_answer<'a>(
        &'a self,
        _agent: &'a str,
        _answer: &'a str,
    ) -> BoxFuture<'a, InterventionResult> {
        Box::pin(async { InterventionResult::allow() })
    }
}

/// 将 `AgentCallback` 包装为中立 `InterventionCallback` 的桥接适配器。
///
/// 所有干预检查都返回 `InterventionResult::allow()`，从而在满足
/// `InterventionCallback` 接口的同时，保留 `AgentCallback` 纯观察的行为。
///
/// 注意：该结构体定义在此处，但引用了位于门面 crate 的 `agent` 模块中的
/// `AgentCallback`。桥接实现从门面 crate 重新导出，因为只有在那里才能访问
/// 到 `AgentCallback`。本文件只提供 trait 层面的定义；具体的
/// `CallbackBridge` 实现位于门面 crate 中。
pub struct CallbackBridge;

#[cfg(test)]
mod tests {
    use super::*;

    // 验证 allow 结果不阻断、不取消，也不携带额外信息
    #[test]
    fn test_intervention_result_allow() {
        let result = InterventionResult::allow();
        assert!(!result.block);
        assert!(!result.cancel);
        assert!(result.block_reason.is_none());
        assert!(result.injected_context.is_none());
    }

    // 验证 block 结果正确设置阻断标志与原因
    #[test]
    fn test_intervention_result_block() {
        let result = InterventionResult::block("dangerous operation");
        assert!(result.block);
        assert_eq!(result.block_reason.as_deref(), Some("dangerous operation"));
        assert!(!result.cancel);
    }

    // 验证 inject 结果携带注入的上下文且不阻断
    #[test]
    fn test_intervention_result_inject() {
        let result = InterventionResult::inject("additional context");
        assert!(result.injected_context.is_some());
        assert!(!result.block);
    }

    // 验证 cancel 结果设置了取消标志且不阻断
    #[test]
    fn test_intervention_result_cancel() {
        let result = InterventionResult::cancel();
        assert!(result.cancel);
        assert!(!result.block);
    }
}
