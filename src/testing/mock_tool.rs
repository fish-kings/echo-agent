/*!
用于测试 Agent 工具调用行为的 Mock 工具，无需依赖外部服务。

# 典型用途
- 测试工具参数解析逻辑
- 在集成测试中替换真实工具（数据库、HTTP 等）
- 测试工具执行失败时 Agent 的容错行为

# 示例

```rust
use echo_agent::testing::MockTool;
use echo_agent::tools::Tool;
use std::collections::HashMap;

# #[tokio::main]
# async fn main() {
let tool = MockTool::new("calculator")
    .with_description("Compute the sum of two numbers")
    .with_response("The result is 42");

let params = HashMap::new();
let result = tool.execute(params).await.unwrap();
assert!(result.success);
assert_eq!(result.output, "The result is 42");
assert_eq!(tool.call_count(), 1);
# }
```
*/

use crate::error::{ReactError, Result, ToolError};
use crate::tools::{Tool, ToolContext, ToolParameters, ToolResult, ToolStreamEvent};
use futures::future::BoxFuture;
use futures::stream::{self, Stream};
use serde_json::{Value, json};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

/// 预设执行结果的枚举
enum MockToolResponse {
    /// 返回一次成功响应
    Success(String),
    /// 返回一次业务失败响应（`ToolResult::error`）
    Failure(String),
    /// 直接返回一个完整的 `ToolResult`
    Result(Box<ToolResult>),
    /// 返回一次执行错误（`Err`）
    Error(String),
}

/// 每次 Mock 工具调用时捕获的上下文快照，供测试断言使用
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MockToolContext {
    /// 调用时的会话 ID
    pub conversation_id: Option<String>,
    /// 调用时的运行 ID
    pub run_id: Option<String>,
    /// 调用时的轮次 ID
    pub turn_id: Option<String>,
    /// 调用时的执行 ID
    pub execution_id: Option<String>,
    /// 调用时的调用 ID
    pub call_id: Option<String>,
    /// 调用时是否提供了取消信号
    pub cancellation_provided: bool,
}

/**
一个可编排脚本的 Mock 工具。

按顺序返回预设的执行结果；队列耗尽后返回最后一个响应（如果有的话），
否则返回默认的成功响应 `"mock response"`。
*/
pub struct MockTool {
    /// 工具名称
    name: String,
    /// 工具描述
    description: String,
    /// 参数 JSON Schema
    parameters: Value,
    /// 预设的响应队列
    responses: Arc<Mutex<VecDeque<MockToolResponse>>>,
    /// 每次调用时收到的参数，按调用顺序记录
    calls: Arc<Mutex<Vec<HashMap<String, Value>>>>,
    /// 每次调用时捕获的上下文快照
    contexts: Arc<Mutex<Vec<MockToolContext>>>,
    /// 预设的流式事件脚本队列
    stream_scripts: Arc<Mutex<VecDeque<Vec<ToolStreamEvent>>>>,
    /**
    每次执行前的可选延迟时间。
    用于在并发批量测试中控制完成顺序（例如让一个更慢的对等操作在更快的操作之后完成）。
    */
    delay: Option<std::time::Duration>,
    /// 响应脚本耗尽后使用的默认成功响应文本
    default_success: Option<String>,
}

impl MockTool {
    /// 创建一个指定名称的 Mock 工具（描述和参数 schema 使用默认值）
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: "A mock tool for testing".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
            responses: Arc::new(Mutex::new(VecDeque::new())),
            calls: Arc::new(Mutex::new(Vec::<HashMap<String, Value>>::new())),
            contexts: Arc::new(Mutex::new(Vec::new())),
            stream_scripts: Arc::new(Mutex::new(VecDeque::new())),
            delay: None,
            default_success: None,
        }
    }

    /// 设置工具描述
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// 设置参数 JSON Schema
    pub fn with_parameters(mut self, schema: Value) -> Self {
        self.parameters = schema;
        self
    }

    /// 追加一条成功响应文本
    pub fn with_response(self, text: impl Into<String>) -> Self {
        self.responses
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push_back(MockToolResponse::Success(text.into()));
        self
    }

    /// 批量追加多条成功响应
    pub fn with_responses(self, texts: impl IntoIterator<Item = impl Into<String>>) -> Self {
        {
            let mut q = self.responses.lock().unwrap_or_else(|e| e.into_inner());
            for t in texts {
                q.push_back(MockToolResponse::Success(t.into()));
            }
        }
        self
    }

    /// 追加一条失败响应（用于测试工具失败时 Agent 的行为）
    pub fn with_failure(self, msg: impl Into<String>) -> Self {
        self.responses
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push_back(MockToolResponse::Failure(msg.into()));
        self
    }

    /// 追加一个完整的 `ToolResult` 作为响应
    pub fn with_result(self, result: ToolResult) -> Self {
        self.responses
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .push_back(MockToolResponse::Result(Box::new(result)));
        self
    }

    /// 追加一次执行错误作为响应
    pub fn with_error(self, message: impl Into<String>) -> Self {
        self.responses
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .push_back(MockToolResponse::Error(message.into()));
        self
    }

    /// 追加一段流式事件脚本，用于流式执行模式
    pub fn with_stream_script(self, events: Vec<ToolStreamEvent>) -> Self {
        self.stream_scripts
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .push_back(events);
        self
    }

    /**
    在每次执行前休眠指定时长。用于在并发批量测试中控制完成顺序
    （例如让一个较慢的对等操作在较快的操作之后完成）。
    */
    pub fn with_delay(mut self, delay: std::time::Duration) -> Self {
        self.delay = Some(delay);
        self
    }

    /// 显式设置响应脚本耗尽后允许继续调用时使用的默认成功响应。
    pub fn with_default_success(mut self, text: impl Into<String>) -> Self {
        self.default_success = Some(text.into());
        self
    }

    /// 剩余未消费的预设响应数量
    pub fn remaining(&self) -> usize {
        self.responses
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .len()
    }

    /// 已执行的调用总次数
    pub fn call_count(&self) -> usize {
        self.calls.lock().unwrap_or_else(|e| e.into_inner()).len()
    }

    /// 最近一次调用传入的参数（从未被调用时返回 `None`）
    pub fn last_args(&self) -> Option<HashMap<String, Value>> {
        self.calls
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .last()
            .cloned()
    }

    /// 所有历史调用的参数（按调用时间顺序排列）
    pub fn all_calls(&self) -> Vec<HashMap<String, Value>> {
        self.calls.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// 所有历史调用时捕获的上下文快照
    pub fn contexts(&self) -> Vec<MockToolContext> {
        self.contexts
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clone()
    }

    /// 清空已记录的调用历史
    pub fn reset_calls(&self) {
        self.calls.lock().unwrap_or_else(|e| e.into_inner()).clear();
    }
}

impl Tool for MockTool {
    /// 返回工具名称
    fn name(&self) -> &str {
        &self.name
    }

    /// 返回工具描述
    fn description(&self) -> &str {
        &self.description
    }

    /// 返回参数 JSON Schema
    fn parameters(&self) -> Value {
        self.parameters.clone()
    }

    /// 使用默认上下文执行工具（内部转发到 `execute_with_context`）
    fn execute(&self, params: ToolParameters) -> BoxFuture<'_, Result<ToolResult>> {
        let context = ToolContext::default();
        Box::pin(async move { self.execute_with_context(params, &context).await })
    }

    /// 携带上下文执行工具，按脚本顺序弹出并返回预设结果
    fn execute_with_context<'a>(
        &'a self,
        params: ToolParameters,
        context: &'a ToolContext,
    ) -> BoxFuture<'a, Result<ToolResult>> {
        Box::pin(async move {
            // 记录本次调用的参数
            self.calls
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(params.clone());
            self.contexts
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .push(MockToolContext {
                    conversation_id: context.conversation_id.clone(),
                    run_id: context.run_id.clone(),
                    turn_id: context.turn_id.clone(),
                    execution_id: context.execution_id.clone(),
                    call_id: context.call_id.clone(),
                    cancellation_provided: context.cancel.is_some(),
                });

            if let Some(d) = self.delay {
                if let Some(cancel) = context.cancel.as_ref() {
                    tokio::select! {
                        _ = cancel.cancelled() => {
                            return Err(ReactError::Agent(Box::new(
                                crate::error::AgentError::Cancelled(
                                    "mock tool cancelled".to_string(),
                                ),
                            )));
                        }
                        _ = tokio::time::sleep(d) => {}
                    }
                } else {
                    tokio::time::sleep(d).await;
                }
            }

            let response = self
                .responses
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .pop_front();
            match response {
                Some(MockToolResponse::Success(text)) => Ok(ToolResult::success(text)),
                Some(MockToolResponse::Failure(msg)) => Ok(ToolResult::error(msg)),
                Some(MockToolResponse::Result(result)) => Ok(*result),
                Some(MockToolResponse::Error(message)) => {
                    Err(ReactError::Tool(Box::new(ToolError::ExecutionFailed {
                        tool: self.name.clone(),
                        message,
                    })))
                }
                None => self
                    .default_success
                    .clone()
                    .map(ToolResult::success)
                    .ok_or_else(|| {
                        ReactError::Tool(Box::new(ToolError::ExecutionFailed {
                            tool: self.name.clone(),
                            message: "MockTool response script exhausted".to_string(),
                        }))
                    }),
            }
        })
    }

    /**
     * 携带上下文以流式方式执行工具，按脚本顺序弹出预设的事件流；
     * 若未预设流脚本，则退化为一次性返回执行结果的单事件流
     */
    fn execute_stream_with_context<'a>(
        &'a self,
        params: ToolParameters,
        context: &ToolContext,
    ) -> BoxFuture<'a, Result<std::pin::Pin<Box<dyn Stream<Item = ToolStreamEvent> + Send + 'a>>>>
    {
        let context = context.clone();
        Box::pin(async move {
            let scripted = self
                .stream_scripts
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .pop_front();
            let events = match scripted {
                Some(events) => events,
                None => vec![ToolStreamEvent::Complete(
                    self.execute_with_context(params, &context).await?,
                )],
            };
            Ok(Box::pin(stream::iter(events))
                as std::pin::Pin<
                    Box<dyn Stream<Item = ToolStreamEvent> + Send + 'a>,
                >)
        })
    }

    /// 是否配置了流式事件脚本，决定是否支持流式执行
    fn supports_streaming(&self) -> bool {
        !self
            .stream_scripts
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .is_empty()
    }
}
