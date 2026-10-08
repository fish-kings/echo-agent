use std::sync::Arc;

use futures::future::BoxFuture;
use serde_json::Value;

use crate::error::ToolError;
use crate::human_loop::{HumanLoopProvider, HumanLoopRequest, HumanLoopResponse};
use crate::tools::{Tool, ToolParameters, ToolResult};

/**
由 LLM 触发的人机协同（human-in-the-loop）工具。

在 LLM 不确定用户意图、需要额外信息，或需要用户确认时被调用。
使用注入的 [`HumanLoopProvider`] 异步请求用户输入，支持
CLI、HTTP Webhook、WebSocket 等多种渠道。
*/
pub struct HumanInLoop {
    /// 用于实际发起人机协同请求的底层 provider。
    provider: Arc<dyn HumanLoopProvider>,
}

impl HumanInLoop {
    /// 基于给定的 provider 创建一个新的 [`HumanInLoop`] 工具。
    pub fn new(provider: Arc<dyn HumanLoopProvider>) -> Self {
        Self { provider }
    }
}

impl Tool for HumanInLoop {
    /// 工具名称：`human_in_loop`。
    fn name(&self) -> &str {
        "human_in_loop"
    }

    /// 面向 LLM 的工具用途说明。
    fn description(&self) -> &str {
        "Use this tool when you are uncertain about user intent, need additional information, or require user confirmation."
    }

    /// 工具参数的 JSON Schema（approval_type、reasoning、tool）。
    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Trigger reason: why is human intervention needed? Provided by LLM when intent is unclear; user confirmation when tool has risks."
                },
                "tool": {
                    "type": "string",
                    "description": "Name of the tool that triggered this (optional)"
                },
                "approval_type": {
                    "type": "string",
                    "description": "Trigger type: 'LLM' when the LLM triggers it, 'tool' when a tool triggers it"
                }
            },
            "required": ["reasoning", "approval_type"]
        })
    }

    /// 向用户发起人工介入请求并返回用户的回复。
    fn execute(
        &self,
        parameters: ToolParameters,
    ) -> BoxFuture<'_, crate::error::Result<ToolResult>> {
        Box::pin(async move {
            let approval_type = parameters
                .get("approval_type")
                .and_then(|t| t.as_str())
                .ok_or_else(|| ToolError::MissingParameter("approval_type".to_string()))?;

            let reasoning = parameters
                .get("reasoning")
                .and_then(|t| t.as_str())
                .ok_or_else(|| ToolError::MissingParameter("reasoning".to_string()))?;

            let tool = parameters
                .get("tool")
                .and_then(|t| t.as_str())
                .unwrap_or("None");

            let prompt = format!(
                "I need your help.\nTrigger type: {approval_type}\nTrigger reason: {reasoning}\nTrigger tool: {tool}\n\nPlease reply directly with your opinion or confirmation:"
            );

            let req = HumanLoopRequest::input(prompt);
            let result_text = match self.provider.request(req).await? {
                HumanLoopResponse::Text(text) => text,
                HumanLoopResponse::Approved => "User confirmed".to_string(),
                HumanLoopResponse::ApprovedWithScope { scope } => {
                    format!("User confirmed (scope: {:?})", scope)
                }
                HumanLoopResponse::ModifiedArgs { args, scope } => {
                    format!(
                        "User confirmed after modifying arguments (args: {}, scope: {:?})",
                        args, scope
                    )
                }
                HumanLoopResponse::Rejected { reason } => {
                    format!(
                        "User rejected{}",
                        reason.map(|r| format!(", reason: {r}")).unwrap_or_default()
                    )
                }
                HumanLoopResponse::Timeout => "Timed out waiting for user input".to_string(),
                HumanLoopResponse::Deferred => "User deferred the decision".to_string(),
                HumanLoopResponse::Selection {
                    selection,
                    instructions,
                } => {
                    format!(
                        "User selected: {}{}",
                        selection,
                        instructions
                            .map(|i| format!(" (instructions: {})", i))
                            .unwrap_or_default()
                    )
                }
            };

            Ok(ToolResult::success(format!(
                "User response (trigger reason: {reasoning}, tool: {tool}): {result_text}"
            )))
        })
    }
}
