/*!
后台命令 cell 工具集 —— `wait` / `stop_cell` / `list_cells`。

与 `shell(background=true)` 配套使用的能力面：shell 工具将长时间运行的
命令作为一个 cell 启动，并立即返回 `cell_id`；这些工具则通过共享的
[`CommandCellRegistry`] 对 cell 进行长轮询、停止和列举。只要向 agent
注入了 registry，这些工具就会被注册（主 agent 与其子 agent 共享同一个
进程级 registry）。
*/

use echo_core::tools::cell::CommandCellRegistry;
use echo_core::tools::{Tool, ToolParameters, ToolResult};
use futures::future::BoxFuture;
use serde_json::{Value, json};
use std::sync::Arc;
use tracing::debug;

/// 默认的长轮询让出时间预算（30 秒）。
const DEFAULT_YIELD_MS: u64 = 30_000;
/// 单轮 wait 的耗时上限（1 小时），与 Codex 的
/// `background_terminal_max_timeout = 3600000` 保持一致。
const MAX_YIELD_MS: u64 = 3_600_000;

/**
长轮询一个后台命令 cell。

当 cell 到达终态、在调用方游标之后出现新输出、或者 `yield_time_ms`
超时时返回——以先满足者为准。该调用是重试安全的：可以用返回的
`next_cursor` 再次调用。
*/
pub struct WaitCellTool {
    /// 底层共享的命令 cell 注册表。
    registry: Arc<dyn CommandCellRegistry>,
}

impl WaitCellTool {
    /// 使用给定的注册表创建 `wait` 工具。
    pub fn new(registry: Arc<dyn CommandCellRegistry>) -> Self {
        Self { registry }
    }
}

impl Tool for WaitCellTool {
    /// 返回该工具的名称："wait"。
    fn name(&self) -> &str {
        "wait"
    }

    /// 返回展示给模型的工具描述文本。
    fn description(&self) -> &str {
        "Long-poll a background command cell (from shell with background=true). Returns when the command \
         finishes, produces new output, or the yield budget expires. Pass the previously returned \
         next_cursor as cursor to continue reading incremental output. A terminal cell can still have \
         unread capped output, so keep draining while next_cursor is below output bytes. \
         yield_time_ms=0 is a non-blocking status check."
    }

    /// 返回该工具的 JSON Schema 参数定义。
    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "cell_id": {
                    "type": "string",
                    "description": "The cell ID returned by the background shell launch"
                },
                "cursor": {
                    "type": "integer",
                    "description": "Byte cursor from the previous wait's next_cursor (0 for the first call)"
                },
                "yield_time_ms": {
                    "type": "integer",
                    "description": "How long to wait for progress before returning (default 30000, max 3600000)"
                }
            },
            "required": ["cell_id"]
        })
    }

    /// 解析参数并对指定 cell 发起一次长轮询等待。
    fn execute(
        &self,
        parameters: ToolParameters,
    ) -> BoxFuture<'_, crate::error::Result<ToolResult>> {
        let registry = self.registry.clone();
        Box::pin(async move {
            let cell_id = parameters
                .get("cell_id")
                .and_then(|v| v.as_str())
                .ok_or_else(|| crate::error::ToolError::MissingParameter("cell_id".to_string()))?
                .to_string();
            let cursor = parameters
                .get("cursor")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let yield_ms = parameters
                .get("yield_time_ms")
                .and_then(|v| v.as_u64())
                .unwrap_or(DEFAULT_YIELD_MS)
                .min(MAX_YIELD_MS);

            debug!(cell_id = %cell_id, cursor, yield_ms, "Waiting on command cell");

            match registry.wait(&cell_id, cursor, yield_ms).await {
                Ok(delta) => Ok(ToolResult::success_json(json!(delta))),
                Err(error) => Ok(ToolResult::error(error.to_string())),
            }
        })
    }

    /**
    这个工具存在的意义就是支持小时级的长轮询：因此它选择不受单工具/批次
    超时约束（与 `agent_tool`/`task_execute` 享有相同的豁免），只由显式的
    `yield_time_ms` 作为唯一的截止时间。
    */
    fn exempt_from_batch_timeout(&self) -> bool {
        true
    }
}

/// 通过 ID 停止（杀死）一个后台命令 cell。
pub struct StopCellTool {
    /// 底层共享的命令 cell 注册表。
    registry: Arc<dyn CommandCellRegistry>,
}

impl StopCellTool {
    /// 使用给定的注册表创建 `stop_cell` 工具。
    pub fn new(registry: Arc<dyn CommandCellRegistry>) -> Self {
        Self { registry }
    }
}

impl Tool for StopCellTool {
    /// 返回该工具的名称："stop_cell"。
    fn name(&self) -> &str {
        "stop_cell"
    }

    /// 返回展示给模型的工具描述文本。
    fn description(&self) -> &str {
        "Stop a background command cell: kills its process group. The cell transitions to the \
         cancelled phase and its captured output stays readable via wait."
    }

    /// 返回该工具的 JSON Schema 参数定义。
    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "cell_id": {
                    "type": "string",
                    "description": "The cell ID to stop"
                }
            },
            "required": ["cell_id"]
        })
    }

    /// 解析参数并停止指定的 cell。
    fn execute(
        &self,
        parameters: ToolParameters,
    ) -> BoxFuture<'_, crate::error::Result<ToolResult>> {
        let registry = self.registry.clone();
        Box::pin(async move {
            let cell_id = parameters
                .get("cell_id")
                .and_then(|v| v.as_str())
                .ok_or_else(|| crate::error::ToolError::MissingParameter("cell_id".to_string()))?
                .to_string();

            if registry.stop(&cell_id) {
                Ok(ToolResult::success(format!(
                    "cell {cell_id}: stop requested; the process group is being killed. \
                     Use wait(cell_id) to observe the cancelled phase."
                )))
            } else {
                Ok(ToolResult::error(format!(
                    "cell '{cell_id}' not found (it may have been pruned from terminal history)"
                )))
            }
        })
    }
}

/// 列出所有被追踪的后台命令 cell（运行中与已终止的）。
pub struct ListCellsTool {
    /// 底层共享的命令 cell 注册表。
    registry: Arc<dyn CommandCellRegistry>,
}

impl ListCellsTool {
    /// 使用给定的注册表创建 `list_cells` 工具。
    pub fn new(registry: Arc<dyn CommandCellRegistry>) -> Self {
        Self { registry }
    }
}

impl Tool for ListCellsTool {
    /// 返回该工具的名称："list_cells"。
    fn name(&self) -> &str {
        "list_cells"
    }

    /// 返回展示给模型的工具描述文本。
    fn description(&self) -> &str {
        "List all background command cells with their phase, exit code, and output size."
    }

    /// 返回该工具的 JSON Schema 参数定义（无参数）。
    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {},
        })
    }

    /// 列出所有已追踪的 cell 及其状态摘要。
    fn execute(
        &self,
        _parameters: ToolParameters,
    ) -> BoxFuture<'_, crate::error::Result<ToolResult>> {
        let registry = self.registry.clone();
        Box::pin(async move {
            let cells = registry.list().await;
            if cells.is_empty() {
                return Ok(ToolResult::success("No background command cells."));
            }
            let mut lines = Vec::new();
            lines.push(format!("Background command cells ({}):", cells.len()));
            for cell in &cells {
                let exit = cell
                    .exit_code
                    .map(|code| code.to_string())
                    .unwrap_or_else(|| "-".to_string());
                lines.push(format!(
                    "  - {} [{}] {}: {} (exit {}, {} bytes{})",
                    cell.cell_id,
                    cell.name,
                    cell.phase.as_str(),
                    if cell.phase.is_terminal() {
                        "finished"
                    } else {
                        "in progress"
                    },
                    exit,
                    cell.total_output_bytes,
                    if cell.output_truncated {
                        ", truncated"
                    } else {
                        ""
                    }
                ));
            }
            Ok(ToolResult::success(lines.join("\n")))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use echo_core::tools::cell::{
        CommandCellError, CommandCellLaunchReceipt, CommandCellObservationLease, CommandCellPhase,
        CommandCellRegistry, CommandCellRequest, CommandCellSnapshot,
    };
    use echo_orchestration::tasks::BackgroundCommandManager;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// 会统计 wait 调用次数的 registry 包装器，便于测试断言 wait 工具
    /// 确实查询过共享的 registry。
    struct CountingRegistry {
        /// 被包装的底层后台命令管理器。
        inner: BackgroundCommandManager,
        /// wait 被调用的次数。
        wait_calls: AtomicU64,
    }

    impl CommandCellRegistry for CountingRegistry {
        fn launch(
            &self,
            request: CommandCellRequest,
        ) -> BoxFuture<'_, Result<CommandCellLaunchReceipt, CommandCellError>> {
            self.inner.launch(request)
        }

        fn wait(
            &self,
            cell_id: &str,
            cursor: u64,
            yield_ms: u64,
        ) -> BoxFuture<'_, Result<echo_core::tools::cell::CommandCellDelta, CommandCellError>>
        {
            self.wait_calls.fetch_add(1, Ordering::Relaxed);
            self.inner.wait(cell_id, cursor, yield_ms)
        }

        fn observe(&self, cell_id: &str) -> Result<CommandCellObservationLease, CommandCellError> {
            self.inner.observe(cell_id)
        }

        fn stop(&self, cell_id: &str) -> bool {
            self.inner.stop(cell_id)
        }

        fn list(&self) -> BoxFuture<'_, Vec<CommandCellSnapshot>> {
            self.inner.list()
        }

        fn shutdown(&self) -> BoxFuture<'_, Result<(), CommandCellError>> {
            self.inner.shutdown()
        }
    }

    fn params(json: Value) -> ToolParameters {
        match json {
            Value::Object(map) => map.into_iter().collect(),
            _ => ToolParameters::new(),
        }
    }

    #[tokio::test]
    async fn wait_tool_round_trips_cell_output_and_cursor() {
        let registry = Arc::new(CountingRegistry {
            inner: BackgroundCommandManager::default(),
            wait_calls: AtomicU64::new(0),
        });
        let cell_id = registry
            .inner
            .launch(CommandCellRequest {
                command: "echo wait-tool-ok".to_string(),
                working_dir: None,
                timeout_secs: None,
                ..Default::default()
            })
            .await
            .map(|receipt| receipt.cell_id)
            .unwrap_or_default();
        assert!(!cell_id.is_empty());

        let tool = WaitCellTool::new(registry.clone());
        // 排空到终态: wait 工具必须保留 typed output/phase/cursor。
        let mut cursor = 0_u64;
        let mut saw_output = false;
        for _ in 0..20 {
            let result = tool
                .execute(params(json!({
                    "cell_id": cell_id,
                    "cursor": cursor,
                    "yield_time_ms": 5_000
                })))
                .await
                .unwrap();
            assert!(result.success);
            let data = result.data.as_ref().cloned().unwrap_or(Value::Null);
            if data
                .get("new_output")
                .and_then(Value::as_str)
                .is_some_and(|output| output.contains("wait-tool-ok"))
            {
                saw_output = true;
            }
            if data
                .get("snapshot")
                .and_then(|snapshot| snapshot.get("phase"))
                .and_then(Value::as_str)
                == Some("succeeded")
            {
                break;
            }
            let next = data.get("next_cursor").and_then(Value::as_u64);
            cursor = next.unwrap_or(cursor);
        }
        assert!(saw_output, "wait tool output must include cell output");
        assert!(registry.wait_calls.load(Ordering::Relaxed) >= 1);
    }

    #[tokio::test]
    async fn wait_tool_reports_unknown_cell_as_error() {
        let tool = WaitCellTool::new(Arc::new(BackgroundCommandManager::default()));
        let result = tool
            .execute(params(json!({ "cell_id": "missing" })))
            .await
            .unwrap();
        assert!(!result.success);
    }

    #[tokio::test]
    async fn list_and_stop_tools_reflect_registry_state() {
        let registry = Arc::new(BackgroundCommandManager::default());
        let cell_id = registry
            .launch(CommandCellRequest {
                command: "sleep 30".to_string(),
                working_dir: None,
                timeout_secs: None,
                ..Default::default()
            })
            .await
            .map(|receipt| receipt.cell_id)
            .unwrap_or_default();

        let list = ListCellsTool::new(registry.clone());
        let listed = list.execute(params(json!({}))).await.unwrap();
        assert!(listed.success);
        assert!(listed.output.contains("queued") || listed.output.contains("running"));

        let stop = StopCellTool::new(registry.clone());
        let stopped = stop
            .execute(params(json!({ "cell_id": cell_id })))
            .await
            .unwrap();
        assert!(stopped.success);

        // 停止后 wait 观察到 cancelled 终态。
        let wait = WaitCellTool::new(registry.clone());
        let observed = wait
            .execute(params(
                json!({ "cell_id": cell_id, "yield_time_ms": 10_000 }),
            ))
            .await
            .unwrap();
        assert_eq!(
            observed
                .data
                .as_ref()
                .and_then(|data| data.get("snapshot"))
                .and_then(|snapshot| snapshot.get("phase"))
                .and_then(Value::as_str),
            Some("cancelled")
        );

        let unknown = stop
            .execute(params(json!({ "cell_id": "no-such" })))
            .await
            .unwrap();
        assert!(!unknown.success);
    }

    #[test]
    fn wait_tool_is_exempt_from_batch_timeout() {
        let tool = WaitCellTool::new(Arc::new(BackgroundCommandManager::default()));
        assert!(tool.exempt_from_batch_timeout());
        assert!(tool.manages_own_timeout());
        let _phase = CommandCellPhase::Running;
    }
}
