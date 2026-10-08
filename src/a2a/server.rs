/*! A2A HTTP 服务器
 *
 * 提供符合 A2A 协议的 HTTP 端点：
 * - `GET /.well-known/agent.json` —— 返回 Agent Card
 * - `POST /` —— 处理 JSON-RPC 任务请求
 *
 * 支持的 JSON-RPC 方法：
 * - `tasks/send`          —— 同步发送任务并等待完成
 * - `tasks/sendSubscribe` —— 流式发送任务，通过 SSE 接收实时事件
 * - `tasks/get`           —— 查询任务状态
 * - `tasks/cancel`        —— 取消任务
 */

use super::types::*;
use crate::agent::Agent;
use crate::error::Result;
use echo_core::agent::AgentEvent;
use echo_core::agent::CancellationToken;
use futures::StreamExt;
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn};

/** A2A 服务器。
 *
 * 将一个 Agent 暴露为符合 A2A 协议的 HTTP 服务。
 *
 * # 状态机
 *
 * ```text
 * submitted → working → [input-required] → completed / failed
 *                      ↑___________________↓
 * ```
 */
pub struct A2AServer {
    /// 该服务对外暴露的 Agent Card。
    card: AgentCard,
    /// 被本服务包装的底层 Agent。
    agent: Arc<dyn Agent>,
    /// 按任务 id 索引的任务状态表。
    tasks: Arc<RwLock<HashMap<String, A2ATask>>>,
    /// 按任务 id 索引的取消令牌表。
    cancel_tokens: Arc<RwLock<HashMap<String, CancellationToken>>>,
}

impl A2AServer {
    /// 创建一个新的 A2A 服务器实例。
    pub fn new(card: AgentCard, agent: impl Agent + 'static) -> Self {
        Self {
            card,
            agent: Arc::new(agent),
            tasks: Arc::new(RwLock::new(HashMap::new())),
            cancel_tokens: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// 从一个装箱（boxed）的 Agent 创建 A2A 服务器实例。
    pub fn from_boxed(card: AgentCard, agent: Box<dyn Agent>) -> Self {
        Self {
            card,
            agent: Arc::new(agent),
            tasks: Arc::new(RwLock::new(HashMap::new())),
            cancel_tokens: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// 获取 Agent Card（供 `/.well-known/agent.json` 使用）
    pub fn agent_card(&self) -> &AgentCard {
        &self.card
    }

    /// 将 Agent Card 以 JSON 字符串形式返回
    pub fn agent_card_json(&self) -> Result<String> {
        serde_json::to_string_pretty(&self.card).map_err(|e| {
            crate::error::ReactError::Other(format!("Agent Card serialization failed: {}", e))
        })
    }

    // ── 请求分发 ──────────────────────────────────────────────────────

    /** 处理 A2A JSON-RPC 请求（同步方法）。
     *
     * 支持 `tasks/send`、`tasks/get`、`tasks/cancel`。
     * 如需 `tasks/sendSubscribe`，请改用 `handle_request_stream`。
     */
    pub async fn handle_request(&self, request_json: &str) -> String {
        let request: A2ATaskRequest = match serde_json::from_str(request_json) {
            Ok(req) => req,
            Err(e) => {
                return serde_json::to_string(&A2ATaskResponse {
                    jsonrpc: JSONRPC_VERSION.to_string(),
                    id: None,
                    result: None,
                    error: Some(A2AError {
                        code: ERROR_CODE_PARSE as i32,
                        message: format!("Parse error: {}", e),
                    }),
                })
                .unwrap_or_default();
            }
        };

        let response = match request.method.as_str() {
            METHOD_SEND => self.handle_task_send(&request).await,
            METHOD_GET => self.handle_task_get(&request).await,
            METHOD_CANCEL => self.handle_task_cancel(&request).await,
            METHOD_SEND_SUBSCRIBE => A2ATaskResponse {
                jsonrpc: JSONRPC_VERSION.to_string(),
                id: Some(request.id),
                result: None,
                error: Some(A2AError {
                    code: ERROR_CODE_METHOD_NOT_FOUND as i32,
                    message:
                        "tasks/sendSubscribe requires SSE transport; use handle_request_stream()"
                            .to_string(),
                }),
            },
            _ => A2ATaskResponse {
                jsonrpc: JSONRPC_VERSION.to_string(),
                id: Some(request.id),
                result: None,
                error: Some(A2AError {
                    code: ERROR_CODE_METHOD_NOT_FOUND as i32,
                    message: format!("Method not found: {}", request.method),
                }),
            },
        };

        serde_json::to_string(&response).unwrap_or_default()
    }

    /** 处理流式请求 `tasks/sendSubscribe`。
     *
     * 返回一个 SSE 事件流，每个事件序列化为一行 JSON（`data: <json>\n\n`）。
     * 调用方负责将该流写入 HTTP 响应体（Content-Type: text/event-stream）。
     *
     * # 事件序列示例
     *
     * ```text
     * data: {"type":"status","taskId":"...","status":{"state":"submitted"},"final":false}
     * data: {"type":"status","taskId":"...","status":{"state":"working"},"final":false}
     * data: {"type":"artifact","taskId":"...","artifact":{"index":0,"parts":[...],"append":true},"final":false}
     * data: {"type":"status","taskId":"...","status":{"state":"completed",...},"final":true}
     * ```
     */
    pub async fn handle_request_stream(
        self: &Arc<Self>,
        request_json: &str,
    ) -> Result<Pin<Box<dyn futures::Stream<Item = A2AStreamEvent> + Send + 'static>>> {
        let request: A2ATaskRequest = serde_json::from_str(request_json)
            .map_err(|e| crate::error::ReactError::Other(format!("Parse error: {}", e)))?;

        if request.method != METHOD_SEND_SUBSCRIBE {
            return Err(crate::error::ReactError::Other(format!(
                "handle_request_stream only supports tasks/sendSubscribe, got '{}'",
                request.method
            )));
        }

        let task_id = request
            .params
            .id
            .clone()
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        let session_id = request.params.session_id.clone();
        let input_text = request.params.message.text_content();
        let user_message = request.params.message.clone();

        info!(task_id = %task_id, "A2A Stream: received streaming task request");

        // 已提交（submitted）
        let initial_task = A2ATask {
            id: task_id.clone(),
            session_id: session_id.clone(),
            status: A2ATaskStatus::new(TaskState::Submitted),
            history: vec![user_message.clone()],
            artifacts: Vec::new(),
        };
        {
            let mut tasks = self.tasks.write().await;
            tasks.insert(task_id.clone(), initial_task);
        }

        // 创建并保存取消令牌
        let cancel_token = CancellationToken::new();
        self.cancel_tokens
            .write()
            .await
            .insert(task_id.clone(), cancel_token.clone());

        let tasks = self.tasks.clone();
        let agent = self.agent.clone();
        let cancel_tokens = self.cancel_tokens.clone();

        let stream = async_stream::stream! {
            // 事件 1：submitted
            yield A2AStreamEvent::StatusUpdate(TaskStatusUpdateEvent {
                task_id: task_id.clone(),
                status: A2ATaskStatus::new(TaskState::Submitted),
                is_final: false,
            });

            // 事件 2：working
            Self::update_task_state(&tasks, &task_id, TaskState::Working, None).await;
            yield A2AStreamEvent::StatusUpdate(TaskStatusUpdateEvent {
                task_id: task_id.clone(),
                status: A2ATaskStatus::new(TaskState::Working),
                is_final: false,
            });

            // 以流式方式执行 Agent
            let agent_guard = agent.as_ref();
            let stream_result = agent_guard.execute_stream(&input_text).await;

            match stream_result {
                Ok(raw_stream) => {
                    let identity = match crate::agent::EventIdentity::for_run(task_id.clone()) {
                        Ok(identity) => identity,
                        Err(error) => {
                            let status = A2ATaskStatus::with_message(
                                TaskState::Failed,
                                A2AMessage::agent_text(format!("Invalid task identity: {error}")),
                            );
                            Self::update_task_state(
                                &tasks,
                                &task_id,
                                TaskState::Failed,
                                Some(&status),
                            )
                            .await;
                            yield A2AStreamEvent::StatusUpdate(TaskStatusUpdateEvent {
                                task_id: task_id.clone(),
                                status,
                                is_final: true,
                            });
                            return;
                        }
                    };
                    let mut event_stream =
                        crate::agent::envelope_event_stream(raw_stream, identity);
                    let mut accumulated_text = String::new();
                    let mut artifact_index: usize = 0;
                    let mut first_chunk = true;

                    while let Some(event_result) = event_stream.next().await {
                        if cancel_token.is_cancelled() {
                            let status = A2ATaskStatus::new(TaskState::Canceled);
                            Self::update_task_state(&tasks, &task_id, TaskState::Canceled, Some(&status)).await;
                            yield A2AStreamEvent::StatusUpdate(TaskStatusUpdateEvent {
                                task_id: task_id.clone(),
                                status,
                                is_final: true,
                            });
                            cancel_tokens.write().await.remove(&task_id);
                            return;
                        }
                        match event_result.map(|envelope| envelope.payload) {
                            Ok(AgentEvent::Token(token)) => {
                                accumulated_text.push_str(&token);
                                yield A2AStreamEvent::ArtifactUpdate(TaskArtifactUpdateEvent {
                                    task_id: task_id.clone(),
                                    artifact: A2AArtifact {
                                        name: if first_chunk { Some("output".to_string()) } else { None },
                                        index: Some(artifact_index),
                                        parts: vec![A2APart::Text { text: token }],
                                        append: !first_chunk,
                                    },
                                    is_final: false,
                                });
                                first_chunk = false;
                            }
                            Ok(AgentEvent::ToolCall { invocation, .. }) => {
                                yield A2AStreamEvent::StatusUpdate(TaskStatusUpdateEvent {
                                    task_id: task_id.clone(),
                                    status: A2ATaskStatus::with_message(
                                        TaskState::Working,
                                        A2AMessage::agent_text(format!("Calling tool: {}", invocation.name)),
                                    ),
                                    is_final: false,
                                });
                            }
                            Ok(AgentEvent::ToolResult { name, result, .. }) => {
                                let output = result.error.unwrap_or(result.output);
                                yield A2AStreamEvent::ArtifactUpdate(TaskArtifactUpdateEvent {
                                    task_id: task_id.clone(),
                                    artifact: A2AArtifact {
                                        name: Some(format!("tool_result:{name}")),
                                        index: Some(artifact_index),
                                        parts: vec![A2APart::Text { text: output }],
                                        append: false,
                                    },
                                    is_final: false,
                                });
                                artifact_index += 1;
                                first_chunk = true;
                            }
                            Ok(AgentEvent::FinalAnswer(answer)) => {
                                accumulated_text = answer;
                            }
                            Ok(_) => {}
                            Err(e) => {
                                warn!(task_id = %task_id, error = %e, "A2A Stream: event stream error");
                                let status = A2ATaskStatus::with_message(
                                    TaskState::Failed,
                                    A2AMessage::agent_text(format!("Execution failed: {e}")),
                                );
                                Self::update_task_state(&tasks, &task_id, TaskState::Failed, Some(&status)).await;
                                yield A2AStreamEvent::StatusUpdate(TaskStatusUpdateEvent {
                                    task_id: task_id.clone(),
                                    status,
                                    is_final: true,
                                });
                                return;
                            }
                        }
                    }

                    // 已完成（completed）
                    let result_message = A2AMessage::agent_text(&accumulated_text);

                    let completed_status = A2ATaskStatus::with_message(
                        TaskState::Completed,
                        result_message.clone(),
                    );

                    {
                        let mut store = tasks.write().await;
                        if let Some(task) = store.get_mut(&task_id) {
                            task.status = completed_status.clone();
                            task.history.push(result_message);
                            if task.artifacts.is_empty() && !accumulated_text.is_empty() {
                                task.artifacts.push(A2AArtifact {
                                    name: Some("output".to_string()),
                                    index: Some(0),
                                    parts: vec![A2APart::Text { text: accumulated_text }],
                                    append: false,
                                });
                            }
                        }
                    }

                    cancel_tokens.write().await.remove(&task_id);

                    info!(task_id = %task_id, "✅ A2A Stream: task completed");

                    yield A2AStreamEvent::StatusUpdate(TaskStatusUpdateEvent {
                        task_id: task_id.clone(),
                        status: completed_status,
                        is_final: true,
                    });
                }
                Err(e) => {
                    warn!(task_id = %task_id, error = %e, "A2A Stream: Agent execution failed");
                    let status = A2ATaskStatus::with_message(
                        TaskState::Failed,
                        A2AMessage::agent_text(format!("Execution failed: {e}")),
                    );
                    Self::update_task_state(&tasks, &task_id, TaskState::Failed, Some(&status)).await;
                    yield A2AStreamEvent::StatusUpdate(TaskStatusUpdateEvent {
                        task_id: task_id.clone(),
                        status,
                        is_final: true,
                    });
                }
            }
        };

        Ok(Box::pin(stream))
    }

    /** 将一个流式事件格式化为 SSE 的 `data:` 行。
     *
     * 便于在 HTTP 框架中直接使用：
     * ```rust
     * use echo_agent::a2a::{A2AServer, A2AStreamEvent, A2ATaskStatus, TaskState, TaskStatusUpdateEvent};
     *
     * let event = A2AStreamEvent::StatusUpdate(TaskStatusUpdateEvent {
     *     task_id: "task-1".to_string(),
     *     status: A2ATaskStatus::new(TaskState::Working),
     *     is_final: false,
     * });
     * let sse = A2AServer::format_sse_event(&event, "req-1");
     * assert!(sse.starts_with("data: "));
     * ```
     */
    pub fn format_sse_event(event: &A2AStreamEvent, request_id: &str) -> String {
        let response = A2AStreamResponse {
            jsonrpc: JSONRPC_VERSION.to_string(),
            id: request_id.to_string(),
            result: Some(event.clone()),
            error: None,
        };
        format!(
            "data: {}\n\n",
            serde_json::to_string(&response).unwrap_or_default()
        )
    }

    // ── tasks/send ───────────────────────────────────────────────────────────

    /**
     * 处理同步的 `tasks/send` 请求：创建任务、以流式方式驱动 Agent 执行，
     * 并等待其完成、失败或被取消。
     */
    async fn handle_task_send(&self, request: &A2ATaskRequest) -> A2ATaskResponse {
        let task_id = request
            .params
            .id
            .clone()
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

        let input_text = request.params.message.text_content();

        info!(task_id = %task_id, input_len = input_text.len(), "A2A: received task request");

        // 状态从 submitted 变为 working
        let task = A2ATask {
            id: task_id.clone(),
            session_id: request.params.session_id.clone(),
            status: A2ATaskStatus::new(TaskState::Submitted),
            history: vec![request.params.message.clone()],
            artifacts: Vec::new(),
        };

        {
            let mut tasks = self.tasks.write().await;
            tasks.insert(task_id.clone(), task);
        }

        // 创建并保存取消令牌
        let cancel_token = CancellationToken::new();
        self.cancel_tokens
            .write()
            .await
            .insert(task_id.clone(), cancel_token.clone());

        Self::update_task_state(&self.tasks, &task_id, TaskState::Working, None).await;

        // 状态从 working 变为 completed / failed / canceled
        let agent = self.agent.as_ref();
        let execution = async {
            let mut stream = agent
                .execute_stream_with_cancel(&input_text, cancel_token.clone())
                .await?;
            let mut final_output = None;
            while let Some(event) = stream.next().await {
                match event? {
                    AgentEvent::FinalAnswer(output) => final_output = Some(output),
                    AgentEvent::Cancelled => {
                        return Err(crate::error::ReactError::Agent(Box::new(
                            echo_core::error::AgentError::Cancelled(
                                "A2A task cancelled".to_string(),
                            ),
                        )));
                    }
                    _ => {}
                }
            }
            final_output.ok_or_else(|| {
                crate::error::ReactError::Agent(Box::new(
                    echo_core::error::AgentError::NoResponse {
                        model: agent.model_name().to_string(),
                        agent: agent.name().to_string(),
                    },
                ))
            })
        };
        let execution_result = tokio::select! {
            biased;
            _ = cancel_token.cancelled() => Err(crate::error::ReactError::Agent(Box::new(
                echo_core::error::AgentError::Cancelled("A2A task cancelled".to_string())
            ))),
            result = execution => result,
        };
        match execution_result {
            Ok(output) => {
                info!(task_id = %task_id, "A2A: task execution completed");

                let result_message = A2AMessage::agent_text(&output);
                let artifact = A2AArtifact {
                    name: Some("output".to_string()),
                    index: Some(0),
                    parts: vec![A2APart::Text {
                        text: output.clone(),
                    }],
                    append: false,
                };

                let completed_task = {
                    let mut tasks = self.tasks.write().await;
                    let Some(task) = tasks.get_mut(&task_id) else {
                        return A2ATaskResponse {
                            jsonrpc: JSONRPC_VERSION.to_string(),
                            id: Some(request.id.clone()),
                            result: None,
                            error: Some(A2AError {
                                code: ERROR_CODE_TASK_NOT_FOUND as i32,
                                message: format!("Task not found: {task_id}"),
                            }),
                        };
                    };
                    if task.status.state.can_transition_to(TaskState::Completed) {
                        task.status = A2ATaskStatus::with_message(
                            TaskState::Completed,
                            result_message.clone(),
                        );
                        task.history.push(result_message);
                        task.artifacts.push(artifact);
                    }
                    task.clone()
                };

                self.cancel_tokens.write().await.remove(&task_id);

                A2ATaskResponse {
                    jsonrpc: JSONRPC_VERSION.to_string(),
                    id: Some(request.id.clone()),
                    result: Some(completed_task),
                    error: None,
                }
            }
            Err(e) => {
                if cancel_token.is_cancelled() {
                    let canceled = self.tasks.read().await.get(&task_id).cloned();
                    self.cancel_tokens.write().await.remove(&task_id);
                    return A2ATaskResponse {
                        jsonrpc: JSONRPC_VERSION.to_string(),
                        id: Some(request.id.clone()),
                        result: canceled,
                        error: None,
                    };
                }
                warn!(task_id = %task_id, error = %e, "A2A: task execution failed");

                let failed_task = A2ATask {
                    id: task_id.clone(),
                    session_id: request.params.session_id.clone(),
                    status: A2ATaskStatus::with_message(
                        TaskState::Failed,
                        A2AMessage::agent_text(format!("Execution failed: {}", e)),
                    ),
                    history: vec![request.params.message.clone()],
                    artifacts: Vec::new(),
                };

                let mut tasks = self.tasks.write().await;
                if let Some(task) = tasks.get_mut(&task_id)
                    && task.status.state.can_transition_to(TaskState::Failed)
                {
                    *task = failed_task;
                }
                drop(tasks);

                self.cancel_tokens.write().await.remove(&task_id);

                A2ATaskResponse {
                    jsonrpc: JSONRPC_VERSION.to_string(),
                    id: Some(request.id.clone()),
                    result: None,
                    error: Some(A2AError {
                        code: ERROR_CODE_TASK_FAILED as i32,
                        message: format!("Task execution failed: {}", e),
                    }),
                }
            }
        }
    }

    // ── tasks/get ────────────────────────────────────────────────────────────

    /// 处理 `tasks/get` 请求：按任务 id 查询任务当前状态。
    async fn handle_task_get(&self, request: &A2ATaskRequest) -> A2ATaskResponse {
        let task_id = match &request.params.id {
            Some(id) => id.clone(),
            None => {
                return A2ATaskResponse {
                    jsonrpc: JSONRPC_VERSION.to_string(),
                    id: Some(request.id.clone()),
                    result: None,
                    error: Some(A2AError {
                        code: ERROR_CODE_INVALID_PARAMS as i32,
                        message: "Missing task id".to_string(),
                    }),
                };
            }
        };

        let tasks = self.tasks.read().await;
        match tasks.get(&task_id) {
            Some(task) => A2ATaskResponse {
                jsonrpc: JSONRPC_VERSION.to_string(),
                id: Some(request.id.clone()),
                result: Some(task.clone()),
                error: None,
            },
            None => A2ATaskResponse {
                jsonrpc: JSONRPC_VERSION.to_string(),
                id: Some(request.id.clone()),
                result: None,
                error: Some(A2AError {
                    code: ERROR_CODE_TASK_NOT_FOUND as i32,
                    message: format!("Task not found: {}", task_id),
                }),
            },
        }
    }

    // ── tasks/cancel ─────────────────────────────────────────────────────────

    /// 处理 `tasks/cancel` 请求：将任务状态转为 Canceled，并触发其取消令牌。
    async fn handle_task_cancel(&self, request: &A2ATaskRequest) -> A2ATaskResponse {
        let task_id = match &request.params.id {
            Some(id) => id.clone(),
            None => {
                return A2ATaskResponse {
                    jsonrpc: JSONRPC_VERSION.to_string(),
                    id: Some(request.id.clone()),
                    result: None,
                    error: Some(A2AError {
                        code: ERROR_CODE_INVALID_PARAMS as i32,
                        message: "Missing task id".to_string(),
                    }),
                };
            }
        };

        let mut tasks = self.tasks.write().await;
        if let Some(task) = tasks.get_mut(&task_id) {
            if task.status.state.is_terminal() {
                return A2ATaskResponse {
                    jsonrpc: JSONRPC_VERSION.to_string(),
                    id: Some(request.id.clone()),
                    result: None,
                    error: Some(A2AError {
                        code: ERROR_CODE_TERMINAL_STATE as i32,
                        message: format!(
                            "Task '{}' is in terminal state '{}' and cannot be canceled",
                            task_id, task.status.state
                        ),
                    }),
                };
            }

            task.status = A2ATaskStatus::new(TaskState::Canceled);

            // 若取消令牌存在，则触发取消
            if let Some(token) = self.cancel_tokens.read().await.get(&task_id) {
                token.cancel();
            }

            A2ATaskResponse {
                jsonrpc: JSONRPC_VERSION.to_string(),
                id: Some(request.id.clone()),
                result: Some(task.clone()),
                error: None,
            }
        } else {
            A2ATaskResponse {
                jsonrpc: JSONRPC_VERSION.to_string(),
                id: Some(request.id.clone()),
                result: None,
                error: Some(A2AError {
                    code: ERROR_CODE_TASK_NOT_FOUND as i32,
                    message: format!("Task not found: {}", task_id),
                }),
            }
        }
    }

    // ── 内部辅助函数 ─────────────────────────────────────────────────────

    /// 在校验状态转换合法之后，更新任务状态；非法转换会被跳过并记录警告。
    async fn update_task_state(
        tasks: &Arc<RwLock<HashMap<String, A2ATask>>>,
        task_id: &str,
        state: TaskState,
        status: Option<&A2ATaskStatus>,
    ) {
        let mut store = tasks.write().await;
        if let Some(task) = store.get_mut(task_id) {
            if !task.status.state.can_transition_to(state) {
                warn!(from = ?task.status.state, to = ?state, "A2A: illegal state transition, skipping");
                return;
            }
            task.status = status.cloned().unwrap_or_else(|| A2ATaskStatus::new(state));
        }
    }

    /// 清理已完成/失败/取消的任务（由调用方周期性调用）
    pub async fn cleanup_completed_tasks(&self, max_age_secs: u64) {
        let mut tasks = self.tasks.write().await;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        tasks.retain(|_, task| {
            if !task.status.state.is_terminal() {
                return true;
            }
            // 若无法确定任务年龄（没有时间戳）或任务较新，则保留该任务
            if let Some(ts_str) = &task.status.timestamp {
                // 尝试按 ISO 8601 解析；解析失败则保留该任务
                if let Ok(ts) = chrono::DateTime::parse_from_rfc3339(ts_str) {
                    let task_secs = ts.timestamp() as u64;
                    return now.saturating_sub(task_secs) < max_age_secs;
                }
            }
            true
        });
    }
}

/// A2A 服务端请求处理的单元测试。
#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::react::builder::ReactAgentBuilder;
    use crate::testing::MockAgent;

    /// 构造一个 JSON-RPC 请求字符串，供测试使用。
    fn make_request(method: &str, message: &str, task_id: Option<&str>) -> String {
        let params = if let Some(id) = task_id {
            serde_json::json!({
                "id": id,
                "message": { "role": "user", "parts": [{"type": "text", "text": message}] }
            })
        } else {
            serde_json::json!({
                "message": { "role": "user", "parts": [{"type": "text", "text": message}] }
            })
        };

        serde_json::json!({
            "jsonrpc": "2.0",
            "id": "test-req-1",
            "method": method,
            "params": params
        })
        .to_string()
    }

    /// 验证响应中携带的任务状态字段。
    #[test]
    fn test_task_state_in_response() {
        let status = A2ATaskStatus::new(TaskState::Submitted);
        let json = serde_json::to_string(&status).unwrap();
        assert!(json.contains("\"state\":\"submitted\""));

        let status = A2ATaskStatus::new(TaskState::InputRequired);
        let json = serde_json::to_string(&status).unwrap();
        assert!(json.contains("\"state\":\"input-required\""));
    }

    /// 验证对已处于终态的任务执行取消会被拒绝。
    #[test]
    fn test_cancel_terminal_state_rejected() {
        let task = A2ATask {
            id: "t1".into(),
            session_id: None,
            status: A2ATaskStatus::new(TaskState::Completed),
            history: vec![],
            artifacts: vec![],
        };
        assert!(task.status.state.is_terminal());
        assert!(!task.status.state.can_transition_to(TaskState::Canceled));
    }

    /// 验证查询不存在的任务时返回未找到错误。
    #[tokio::test]
    async fn test_handle_task_get_not_found() {
        let card = AgentCard::builder("test", "http://localhost").build();
        let agent = ReactAgentBuilder::new()
            .model("qwen3-max")
            .name("test_agent")
            .system_prompt("test")
            .build()
            .unwrap();
        let server = A2AServer::new(card, agent);

        let req = make_request("tasks/get", "", Some("nonexistent"));
        let resp_json = server.handle_request(&req).await;
        let resp: A2ATaskResponse = serde_json::from_str(&resp_json).unwrap();
        assert!(resp.error.is_some());
        assert_eq!(resp.error.unwrap().code, -32001);
    }

    /// 验证同步发送过程中的取消会进入终态且状态单调推进。
    #[tokio::test]
    async fn sync_send_cancellation_is_terminal_and_monotonic() -> std::result::Result<(), String> {
        let card = AgentCard::builder("test", "http://localhost").build();
        let server = Arc::new(A2AServer::from_boxed(
            card,
            Box::new(
                MockAgent::new("slow")
                    .with_delay_ms(500)
                    .with_response("late success"),
            ),
        ));
        let send_request = A2ATaskRequest {
            jsonrpc: JSONRPC_VERSION.to_string(),
            id: "send-request".to_string(),
            method: METHOD_SEND.to_string(),
            params: A2ATaskParams {
                id: Some("cancel-race".to_string()),
                session_id: None,
                message: A2AMessage::user_text("slow task"),
            },
        };
        let sender = Arc::clone(&server);
        let send = tokio::spawn(async move { sender.handle_task_send(&send_request).await });

        tokio::time::timeout(std::time::Duration::from_secs(1), async {
            loop {
                if server
                    .cancel_tokens
                    .read()
                    .await
                    .contains_key("cancel-race")
                {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .map_err(|_| "send task did not become cancellable".to_string())?;

        let cancel_request = A2ATaskRequest {
            jsonrpc: JSONRPC_VERSION.to_string(),
            id: "cancel-request".to_string(),
            method: METHOD_CANCEL.to_string(),
            params: A2ATaskParams {
                id: Some("cancel-race".to_string()),
                session_id: None,
                message: A2AMessage::user_text("cancel"),
            },
        };
        let canceled = server.handle_task_cancel(&cancel_request).await;
        assert_eq!(
            canceled.result.as_ref().map(|task| task.status.state),
            Some(TaskState::Canceled)
        );

        let send_response = send.await.map_err(|error| error.to_string())?;
        assert_eq!(
            send_response.result.as_ref().map(|task| task.status.state),
            Some(TaskState::Canceled)
        );
        assert_eq!(
            server
                .tasks
                .read()
                .await
                .get("cancel-race")
                .map(|task| task.status.state),
            Some(TaskState::Canceled)
        );
        Ok::<(), String>(())
    }

    /// 验证未知方法会返回方法不存在错误。
    #[tokio::test]
    async fn test_handle_unknown_method() {
        let card = AgentCard::builder("test", "http://localhost").build();
        let agent = ReactAgentBuilder::new()
            .model("qwen3-max")
            .name("test_agent")
            .system_prompt("test")
            .build()
            .unwrap();
        let server = A2AServer::new(card, agent);

        let req = serde_json::json!({
            "jsonrpc": "2.0",
            "id": "req-1",
            "method": "unknown/method",
            "params": { "message": { "role": "user", "parts": [{"type":"text","text":"hi"}] } }
        })
        .to_string();
        let resp_json = server.handle_request(&req).await;
        let resp: A2ATaskResponse = serde_json::from_str(&resp_json).unwrap();
        assert_eq!(resp.error.unwrap().code, -32601);
    }

    /// 验证非法 JSON 请求会返回解析错误。
    #[tokio::test]
    async fn test_handle_parse_error() {
        let card = AgentCard::builder("test", "http://localhost").build();
        let agent = ReactAgentBuilder::new()
            .model("qwen3-max")
            .name("test_agent")
            .system_prompt("test")
            .build()
            .unwrap();
        let server = A2AServer::new(card, agent);

        let resp_json = server.handle_request("not json").await;
        let resp: A2ATaskResponse = serde_json::from_str(&resp_json).unwrap();
        assert_eq!(resp.error.unwrap().code, -32700);
    }
}
