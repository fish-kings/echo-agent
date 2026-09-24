//! 流式执行的共享类型

use crate::llm::types::Message;

/// 流式执行模式配置
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StreamMode {
    /// 单轮执行模式：重置上下文，从 checkpoint 恢复
    Execute,
    /// 多轮会话模式：保留上下文，不重置
    Chat,
}

impl std::fmt::Display for StreamMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StreamMode::Execute => f.write_str("execute"),
            StreamMode::Chat => f.write_str("chat"),
        }
    }
}

/// 流式执行的初始化参数
pub(crate) struct StreamInit {
    /// 用户输入文本（用于审计日志和记忆召回）
    pub text: String,
    /// 可选的预构建 Message（用于多模态场景），为 None 时自动构造文本消息
    pub message: Option<Message>,
    /// 日志标签（例如 "" 或 "(多模态)"）
    pub label: String,
    /// 在等待 Agent 执行前捕获的、作用域限定的元数据。
    pub invocation: Option<echo_core::agent::AgentInvocationContext>,
}
