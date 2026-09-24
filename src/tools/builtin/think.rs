/** Think 工具（可选注册，默认不启用）。

# 背景

框架已切换为 **（通过系统提示词实现 CoT）** 的方式：当 `enable_tool=true`
时，`ReactAgent` 会自动在系统提示词中追加 CoT 引导语，让 LLM 以文本内容
的形式输出推理过程，天然兼容流式（Token 事件）。

该工具**默认不再注册**，仅在以下场景中手动启用：
- 需要在会话历史中保留每一步推理的结构化工具调用记录时
- 非流式任务中，希望将推理过程以 tool_result 的形式存入上下文时

# 用法

```rust,no_run
use echo_agent::tools::builtin::think::ThinkTool;
use echo_agent::prelude::{AgentConfig, ReactAgent};

let config = AgentConfig::new("qwen3-max", "assistant", "You are a helpful assistant");
let mut agent = ReactAgent::new(config);
agent.add_tool(Box::new(ThinkTool));
```

> **注意**：在 `execute_stream` 流式路径中，注册该工具会导致模型的推理内容
> 被写入工具调用参数而非 `content` 字段，从而使推理阶段不再产生
> `AgentEvent::Token` 事件。
> 对于流式场景，请依赖 CoT 系统提示词（默认行为）；无需注册该工具。
*/
use futures::future::BoxFuture;

use crate::error::ToolError;
use crate::tools::{Tool, ToolParameters, ToolResult};
use serde_json::Value;
use tracing::info;

/// 用于内部推理与反思的工具。
pub struct ThinkTool;

impl Tool for ThinkTool {
    fn name(&self) -> &str {
        "think"
    }

    fn description(&self) -> &str {
        "Before taking action, use this tool to record your reasoning and analysis process. Parameter: reasoning - your analysis and plan for the problem."
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Your reasoning process: analyze the problem, formulate a plan, reasoning steps"
                }
            },
            "required": ["reasoning"]
        })
    }

    fn execute(
        &self,
        parameters: ToolParameters,
    ) -> BoxFuture<'_, crate::error::Result<ToolResult>> {
        Box::pin(async move {
            let reasoning = parameters
                .get("reasoning")
                .and_then(|v| v.as_str())
                .ok_or_else(|| ToolError::MissingParameter("reasoning".to_string()))?;

            info!("Think: {}", reasoning);

            // 将推理内容回显到上下文中，使下一次 LLM 调用能看到完整的推理记录
            Ok(ToolResult::success(reasoning.to_string()))
        })
    }
}
