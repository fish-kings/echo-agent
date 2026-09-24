/*!
内置于代码仓库中的 Subagent 通信工具。

`subagent_message` 会从一个正在运行的 Subagent 向其父级发送一条长度受限的消息
（信息性汇报，或阻塞式的澄清请求），或者向一个同级尝试发送消息
（仅入队的引导提示），发送通道是调度器安装到本次调用运行时上下文中的
上行（uplink）通道。投递是"发后即忘"的：发送方不会等待回复，
因此父子之间互相等待不会导致调度树死锁。

`subagent_list` 返回共享控制面中当前活跃尝试的有限集合，
供模型发现可寻址的同级执行 id。

这两个工具在脱离已调度的 Subagent 调用环境时（ToolContext 中没有
lineage / uplink），不会直接报错，而是返回"无操作 + 说明"的结果，
这样即便某个 agent 配置有误，也能产生模型可读的反馈。
*/

use crate::agent::subagent::registry::SubagentRegistry;
use crate::tools::{Tool, ToolParameters, ToolResult};
use echo_core::tools::{
    SubagentUplinkKind, SubagentUplinkMessage, SubagentUplinkTarget, ToolContext,
};
use futures::future::BoxFuture;
use serde_json::{Value, json};

/// 单条消息正文的长度上限（以 Unicode 标量值计）。
const MAX_MESSAGE_CHARS: usize = 8_000;
/// `subagent_list` 输出条数的默认值（同时也是上限）。
const DEFAULT_LIST_LIMIT: usize = 20;

/// 将字符串截断到消息长度上限内
fn bounded_text(value: &str) -> String {
    value.chars().take(MAX_MESSAGE_CHARS).collect()
}

/// 供 Subagent 向父级或同级发送消息的工具
pub struct SubagentMessageTool;

impl SubagentMessageTool {
    /// 创建一个新的 SubagentMessageTool
    pub fn new() -> Self {
        Self
    }
}

impl Default for SubagentMessageTool {
    fn default() -> Self {
        Self
    }
}

impl Tool for SubagentMessageTool {
    /// 返回工具名称 `subagent_message`
    fn name(&self) -> &str {
        "subagent_message"
    }

    /// 返回展示给模型的工具描述文本
    fn description(&self) -> &str {
        "Send one message from this Subagent to its dispatching parent or to a \
         sibling attempt. Parent direction supports `report` (informational, \
         keep working) and `escalate` (request clarification; you still keep \
         working — do not wait for the answer). Sibling direction delivers a \
         queue-only note into the sibling's active turn; use `subagent_list` \
         to discover sibling execution ids. Messages are claims, not verified \
         evidence."
    }

    /// 返回该工具的参数 JSON Schema
    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": ["parent", "sibling"],
                    "description": "Message target: the dispatching parent, or a sibling attempt."
                },
                "text": {
                    "type": "string",
                    "description": "Message body (non-empty, bounded)."
                },
                "intent": {
                    "type": "string",
                    "enum": ["report", "escalate"],
                    "description": "Parent direction only: informational report (default) or blocking clarification request."
                },
                "execution_id": {
                    "type": "string",
                    "description": "Sibling direction only: target a LIVE attempt by its execution id (from subagent_list)."
                },
                "task_id": {
                    "type": "string",
                    "description": "Sibling direction only: target a task in the same plan that has no live attempt yet (exactly one of execution_id / task_id)."
                }
            },
            "required": ["direction", "text"],
            "additionalProperties": false
        })
    }

    /// 执行消息发送逻辑：校验参数、构建目标并通过 uplink 投递消息
    fn execute_with_context<'a>(
        &'a self,
        parameters: ToolParameters,
        ctx: &'a ToolContext,
    ) -> BoxFuture<'a, crate::error::Result<ToolResult>> {
        Box::pin(async move {
            let Some(lineage) = ctx.subagent_lineage.clone() else {
                return Ok(ToolResult::error(
                    "subagent_message is only available inside a dispatched Subagent",
                ));
            };
            let Some(uplink) = ctx.uplink.clone() else {
                return Ok(ToolResult::error(
                    "no uplink channel is wired for this dispatch; messaging is unavailable",
                ));
            };

            let direction = parameters
                .get("direction")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            let text = parameters
                .get("text")
                .and_then(Value::as_str)
                .map(str::trim)
                .unwrap_or_default()
                .to_string();
            if text.is_empty() {
                return Ok(ToolResult::error("text must be a non-empty string"));
            }
            if text.chars().count() > MAX_MESSAGE_CHARS {
                return Ok(ToolResult::error(format!(
                    "text exceeds {MAX_MESSAGE_CHARS} characters"
                )));
            }
            let text = bounded_text(&text);

            let target = match direction.as_str() {
                "parent" => {
                    let intent = match parameters.get("intent").and_then(Value::as_str) {
                        None | Some("report") => SubagentUplinkKind::Report,
                        Some("escalate") => SubagentUplinkKind::Escalate,
                        Some(other) => {
                            return Ok(ToolResult::error(format!(
                                "invalid intent '{other}' (expected report|escalate)"
                            )));
                        }
                    };
                    SubagentUplinkTarget::Parent { kind: intent, text }
                }
                "sibling" => {
                    let execution_id = parameters
                        .get("execution_id")
                        .and_then(Value::as_str)
                        .map(str::trim)
                        .filter(|value| !value.is_empty())
                        .map(str::to_string);
                    let task_id = parameters
                        .get("task_id")
                        .and_then(Value::as_str)
                        .map(str::trim)
                        .filter(|value| !value.is_empty())
                        .map(str::to_string);
                    let to = match (execution_id, task_id) {
                        (Some(execution_id), None) => {
                            echo_core::tools::SubagentPeerAddress::ByExecutionId(execution_id)
                        }
                        (None, Some(task_id)) => {
                            echo_core::tools::SubagentPeerAddress::ByTaskId(task_id)
                        }
                        _ => {
                            return Ok(ToolResult::error(
                                "sibling direction requires exactly one of execution_id / task_id",
                            ));
                        }
                    };
                    SubagentUplinkTarget::Sibling { to, text }
                }
                other => {
                    return Ok(ToolResult::error(format!(
                        "invalid direction '{other}' (expected parent|sibling)"
                    )));
                }
            };

            let receipt = uplink(SubagentUplinkMessage {
                from: lineage,
                target,
            })
            .await;

            tracing::debug!(
                tool = "subagent_message",
                status = %receipt.status,
                accepted = receipt.accepted,
                "subagent uplink settled"
            );

            Ok(ToolResult::success(
                json!({
                    "accepted": receipt.accepted,
                    "status": receipt.status,
                    "detail": receipt.detail,
                })
                .to_string(),
            ))
        })
    }
}

/// 列出当前活跃 Subagent 尝试的工具
pub struct SubagentListTool {
    /// 用于查询共享控制面的 Subagent 注册表
    registry: std::sync::Arc<SubagentRegistry>,
}

impl SubagentListTool {
    /// 创建一个新的 SubagentListTool，绑定到指定的 registry
    pub fn new(registry: std::sync::Arc<SubagentRegistry>) -> Self {
        Self { registry }
    }
}

impl Tool for SubagentListTool {
    /// 返回工具名称 `subagent_list`
    fn name(&self) -> &str {
        "subagent_list"
    }

    /// 返回展示给模型的工具描述文本
    fn description(&self) -> &str {
        "List the currently active Subagent attempts visible in the shared \
         control plane (bounded). Use the execution ids as `subagent_message` \
         sibling targets."
    }

    /// 返回该工具的参数 JSON Schema
    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Maximum entries to return (default 20, max 64)."
                }
            },
            "additionalProperties": false
        })
    }

    /// 从共享控制面读取当前活跃尝试的快照并返回
    fn execute_with_context<'a>(
        &'a self,
        parameters: ToolParameters,
        _ctx: &'a ToolContext,
    ) -> BoxFuture<'a, crate::error::Result<ToolResult>> {
        let registry = self.registry.clone();
        Box::pin(async move {
            let limit = parameters
                .get("limit")
                .and_then(Value::as_u64)
                .map(|value| value as usize)
                .unwrap_or(DEFAULT_LIST_LIMIT);
            let limit = limit.clamp(1, 64);
            let attempts = registry.control_registry().active_snapshot(limit);
            Ok(ToolResult::success(
                json!({
                    "active": attempts
                        .iter()
                        .map(|summary| json!({
                            "execution_id": summary.execution_id,
                            "task_id": summary.task_id,
                            "attempt": summary.attempt,
                            "phase": format!("{:?}", summary.phase),
                        }))
                        .collect::<Vec<_>>(),
                    "count": attempts.len(),
                })
                .to_string(),
            ))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn message_schema_is_complete() {
        let tool = SubagentMessageTool::new();
        let schema = tool.parameters();
        assert!(schema["properties"]["direction"].is_object());
        assert_eq!(schema["required"][0], "direction");
    }

    #[test]
    fn list_schema_has_no_required_fields() {
        let tool = SubagentListTool::new(std::sync::Arc::new(SubagentRegistry::new()));
        let schema = tool.parameters();
        assert!(schema.get("required").is_none());
    }
}
</content>
