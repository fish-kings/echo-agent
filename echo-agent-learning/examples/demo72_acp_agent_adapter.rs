/*! 一个基于公共 echo-agent 门面（facade）组装出的、最小化的源码构建 ACP Agent。

本教学示例使用确定性的 Agent，因此无需任何模型凭据。它不是下一个 SDK
交付阶段中那个可配置的 `echo-agent-sdk-host` 二进制程序。
*/

use agent_client_protocol::Stdio;
use echo_agent::acp::{AcpAgentAdapter, AcpSessionContext};
use echo_agent::agent::{Agent, AgentEvent, CancellationToken};
use echo_agent::error::Result;
use echo_agent::llm::types::Message;
use futures::future::BoxFuture;
use futures::stream::{self, BoxStream, StreamExt};
use std::sync::Mutex;

/// 一个用于演示的确定性会话 Agent：记录会话消息历史并原样回显。
struct ExampleSessionAgent {
    /// 所属会话的 ID。
    session_id: String,
    /// 该会话中已接收到的历史消息。
    messages: Mutex<Vec<String>>,
}

impl ExampleSessionAgent {
    /// 记录一条消息并返回携带回合序号的确定性回复流。
    fn chat_stream_owned<'a>(
        &'a self,
        message: String,
    ) -> BoxFuture<'a, Result<BoxStream<'a, Result<AgentEvent>>>> {
        let answer = {
            let mut messages = self
                .messages
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            messages.push(message.clone());
            format!("{} turn {}: {}", self.session_id, messages.len(), message)
        };
        Box::pin(async move { Ok(stream::iter(vec![Ok(AgentEvent::FinalAnswer(answer))]).boxed()) })
    }
}

impl Agent for ExampleSessionAgent {
    /// 返回该 agent 的固定名称。
    fn name(&self) -> &str {
        "example-acp-agent"
    }

    /// 返回该 agent 所用的（确定性示例）模型名称。
    fn model_name(&self) -> &str {
        "deterministic-example"
    }

    /// 返回该 agent 的系统提示词。
    fn system_prompt(&self) -> &str {
        "Echo ACP prompts for the learning example."
    }

    /// 原样返回传入的任务文本作为执行结果。
    fn execute<'a>(&'a self, task: &'a str) -> BoxFuture<'a, Result<String>> {
        Box::pin(async move { Ok(task.to_string()) })
    }

    /// 以流式事件的形式返回任务的最终答案（原样回显传入的任务文本）。
    fn execute_stream<'a>(
        &'a self,
        task: &'a str,
    ) -> BoxFuture<'a, Result<BoxStream<'a, Result<AgentEvent>>>> {
        let answer = task.to_string();
        Box::pin(async move { Ok(stream::iter(vec![Ok(AgentEvent::FinalAnswer(answer))]).boxed()) })
    }

    /// 以流式事件的形式处理一条聊天消息。
    fn chat_stream<'a>(
        &'a self,
        message: &'a str,
    ) -> BoxFuture<'a, Result<BoxStream<'a, Result<AgentEvent>>>> {
        self.chat_stream_owned(message.to_string())
    }

    /// 以流式事件的形式处理一条支持取消的结构化消息（本示例忽略取消令牌）。
    fn chat_stream_message_with_cancel<'a>(
        &'a self,
        message: Message,
        _cancel: CancellationToken,
    ) -> BoxFuture<'a, Result<BoxStream<'a, Result<AgentEvent>>>> {
        self.chat_stream_owned(message.content.as_text().unwrap_or_default())
    }
}

/// 启动一个基于标准输入输出（stdio）通信的 ACP agent 适配器。
#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let adapter = AcpAgentAdapter::new(|context: AcpSessionContext| async move {
        Ok(Box::new(ExampleSessionAgent {
            session_id: context.session_id.to_string(),
            messages: Mutex::new(Vec::new()),
        }) as Box<dyn Agent>)
    });

    let (close_owner, connection) = adapter.connect_retaining_close_owner(Stdio::new());
    let result = connection.await;
    if result.is_err() {
        close_owner.close().await?;
    }
    result?;
    Ok(())
}
