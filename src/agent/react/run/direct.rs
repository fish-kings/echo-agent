//! 直接执行入口点（run_direct / run_chat_direct）

use super::super::ReactAgent;
use crate::error::Result;
use tracing::{Instrument, debug, info, info_span};

impl ReactAgent {
    /// 直接执行（不做规划）：重置/恢复上下文，然后进入 ReAct 循环
    pub(crate) async fn run_direct(&self, task: &str) -> Result<String> {
        self.validate_persistence_configuration()?;
        let agent = self.config.agent_name.clone();

        info!(agent = %agent, "🧠 Agent starting task execution");
        debug!(
            agent = %agent,
            task = %task,
            tools = ?self.tools.tool_manager.list_tools(),
            max_iterations = self.config.max_iterations,
            "Execution details"
        );

        let model = self.config.model_name.clone();
        self.run_react_loop_mode(task, super::StreamMode::Execute)
            .instrument(info_span!("agent_execute", agent.name = %agent, agent.model = %model))
            .await
    }

    /// 多轮对话：不重置上下文，追加消息后进入 ReAct 循环
    pub(crate) async fn run_chat_direct(&self, message: &str) -> Result<String> {
        self.validate_persistence_configuration()?;
        let agent = self.config.agent_name.clone();

        info!(agent = %agent, "💬 Agent in multi-turn conversation");
        debug!(
            agent = %agent,
            message = %message,
            tools = ?self.tools.tool_manager.list_tools(),
            max_iterations = self.config.max_iterations,
            "Conversation details"
        );

        let model = self.config.model_name.clone();
        self.run_react_loop_mode(message, super::StreamMode::Chat)
            .instrument(info_span!("agent_chat", agent.name = %agent, agent.model = %model))
            .await
    }
}
