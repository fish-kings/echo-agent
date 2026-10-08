/*!
图工作流节点

每个节点都是图中的一个执行单元，可以是：
- **Agent 节点**：消费一个可取消的 Agent 流，并将其最终答案写入状态
- **函数节点**：任意的 `async fn(SharedState) -> Result<()>`
- **路由节点**：纯路由（不执行任何逻辑，仅用于条件分支）
*/

use super::state::SharedState;
use echo_core::agent::{Agent, AgentEvent};
use echo_core::error::{AgentError, ReactError, Result};
use futures::StreamExt;
use futures::future::BoxFuture;
use futures::stream::BoxStream;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

// ── NodeAction ──────────────────────────────────────────────────────────────

/// 节点执行逻辑的类型安全包装
pub(crate) enum NodeAction {
    /// Agent 执行：从状态中读取 input_key 作为 prompt，将输出写入 output_key
    Agent {
        /// 要执行的 agent
        agent: Arc<dyn Agent>,
        /// 从状态中读取输入所使用的键
        input_key: String,
        /// 将输出写入状态所使用的键
        output_key: String,
        /// 使用 execute（带工具的多轮）还是 chat（单轮）模式
        use_execute: bool,
    },
    /// 嵌套子图执行 —— 该节点本身就是一个完整的已编译 Graph
    Subgraph(Box<super::graph::Graph>),
    /// 自定义异步函数
    Function(Box<dyn NodeFn>),
    /// 空操作（用于路由节点）
    Passthrough,
}

/// 自定义节点函数 trait（对象安全）
pub(crate) trait NodeFn: Send + Sync {
    /// 执行该节点函数
    fn call<'a>(&'a self, state: &'a SharedState) -> BoxFuture<'a, Result<()>>;
}

/// 使用闭包实现 NodeFn
struct FnWrapper<F>(F);

impl<F> NodeFn for FnWrapper<F>
where
    F: for<'a> Fn(&'a SharedState) -> BoxFuture<'a, Result<()>> + Send + Sync,
{
    /// 直接调用内部闭包。
    fn call<'a>(&'a self, state: &'a SharedState) -> BoxFuture<'a, Result<()>> {
        (self.0)(state)
    }
}

// ── Node ────────────────────────────────────────────────────────────────────

/// 图中的节点定义
pub(crate) struct Node {
    /// 执行逻辑
    pub action: NodeAction,
}

impl Node {
    /// 将 agent 输出提交到共享状态中
    fn commit_agent_output(state: &SharedState, output_key: &str, output: String) -> Result<()> {
        /*
        使用 merge_overwrite 以支持结构化数据合并语义，
        而不是简单的键级覆盖。
        */
        state.merge_overwrite(&SharedState::from_values(
            [(
                output_key.to_string(),
                serde_json::Value::String(output.clone()),
            )]
            .into_iter()
            .collect(),
        ))?;
        state.push_message(echo_core::llm::types::Message::assistant(output))?;
        Ok(())
    }

    /// 创建一个 Agent 节点（默认使用 execute，即带工具的多轮模式）
    pub fn agent(
        _name: impl Into<String>,
        agent: impl Agent + 'static,
        input_key: impl Into<String>,
        output_key: impl Into<String>,
    ) -> Self {
        Self {
            action: NodeAction::Agent {
                agent: Arc::new(agent),
                input_key: input_key.into(),
                output_key: output_key.into(),
                use_execute: true,
            },
        }
    }

    /// 创建一个 Agent 节点（可配置 execute/chat 模式）
    pub fn agent_with_mode(
        _name: impl Into<String>,
        agent: impl Agent + 'static,
        input_key: impl Into<String>,
        output_key: impl Into<String>,
        use_execute: bool,
    ) -> Self {
        Self {
            action: NodeAction::Agent {
                agent: Arc::new(agent),
                input_key: input_key.into(),
                output_key: output_key.into(),
                use_execute,
            },
        }
    }

    /// 创建一个 Agent 节点（预先包装为 Arc<dyn Agent>）
    pub fn agent_shared(
        _name: impl Into<String>,
        agent: Arc<dyn Agent>,
        input_key: impl Into<String>,
        output_key: impl Into<String>,
    ) -> Self {
        Self {
            action: NodeAction::Agent {
                agent,
                input_key: input_key.into(),
                output_key: output_key.into(),
                use_execute: true,
            },
        }
    }

    /// 创建一个 Agent 节点（预先包装 + 可配置 execute/chat 模式）
    pub fn agent_shared_with_mode(
        _name: impl Into<String>,
        agent: Arc<dyn Agent>,
        input_key: impl Into<String>,
        output_key: impl Into<String>,
        use_execute: bool,
    ) -> Self {
        Self {
            action: NodeAction::Agent {
                agent,
                input_key: input_key.into(),
                output_key: output_key.into(),
                use_execute,
            },
        }
    }

    /// 创建一个函数节点
    pub fn function<F>(_name: impl Into<String>, f: F) -> Self
    where
        F: for<'a> Fn(&'a SharedState) -> BoxFuture<'a, Result<()>> + Send + Sync + 'static,
    {
        Self {
            action: NodeAction::Function(Box::new(FnWrapper(f))),
        }
    }

    /// 创建一个直通（路由）节点
    pub fn passthrough(_name: impl Into<String>) -> Self {
        Self {
            action: NodeAction::Passthrough,
        }
    }

    /**
    创建一个子图节点 —— 将一个已编译的 Graph 作为节点嵌入其中。

    该子图与父图共享同一个 `SharedState`，并在父图的步骤数限制内执行。
    */
    pub fn subgraph(_name: impl Into<String>, graph: super::graph::Graph) -> Self {
        Self {
            action: NodeAction::Subgraph(Box::new(graph)),
        }
    }

    /// 通过可取消的 Agent 流执行，并将 token 增量向 Graph 暴露出去。
    pub(crate) fn execute_stream<'a>(
        &'a self,
        state: &'a SharedState,
        cancel: CancellationToken,
    ) -> BoxStream<'a, Result<String>> {
        let stream = async_stream::try_stream! {
            match &self.action {
                NodeAction::Agent {
                    agent,
                    input_key,
                    output_key,
                    use_execute,
                } => {
                    let input = state.get::<String>(input_key).unwrap_or_default();
                    let mut events = if *use_execute {
                        agent.execute_stream_with_cancel(&input, cancel).await?
                    } else {
                        agent.chat_stream_with_cancel(&input, cancel).await?
                    };
                    let mut final_answer = None;
                    while let Some(event) = events.next().await {
                        match event? {
                            AgentEvent::Token(token) => yield token,
                            AgentEvent::FinalAnswer(answer) => final_answer = Some(answer),
                            AgentEvent::Cancelled => {
                                Err(ReactError::Agent(Box::new(AgentError::Cancelled(
                                    format!("Workflow Agent node '{}' cancelled", agent.name()),
                                ))))?;
                            }
                            AgentEvent::Error { message, .. } => {
                                Err(ReactError::Other(message))?;
                            }
                            _ => {}
                        }
                    }
                    let output = final_answer.ok_or_else(|| {
                        ReactError::Agent(Box::new(AgentError::NoResponse {
                            model: agent.model_name().to_string(),
                            agent: agent.name().to_string(),
                        }))
                    })?;
                    Self::commit_agent_output(state, output_key, output)?;
                }
                NodeAction::Subgraph(subgraph) => {
                    let result = Box::pin(subgraph.run(state.clone())).await?;
                    state.deep_merge(&result.state)?;
                }
                NodeAction::Function(f) => f.call(state).await?,
                NodeAction::Passthrough => {}
            }
        };
        Box::pin(stream)
    }
}

// ── 单元测试 ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// 执行节点流并消费所有输出，直到结束。
    async fn drain_node(node: &Node, state: &SharedState) -> Result<()> {
        let mut stream = node.execute_stream(state, CancellationToken::new());
        while let Some(token) = stream.next().await {
            let _ = token?;
        }
        Ok(())
    }

    /// 验证函数节点能读取输入并写入输出。
    #[tokio::test]
    async fn test_function_node() -> Result<()> {
        let node = Node::function("double", |state: &SharedState| {
            Box::pin(async move {
                let x: i64 = state.get("input").unwrap_or(0);
                let _ = state.set("output", x * 2);
                Ok(())
            })
        });

        let state = SharedState::new();
        state.set("input", 21i64)?;
        drain_node(&node, &state).await?;
        assert_eq!(state.get::<i64>("output"), Some(42));
        Ok(())
    }

    /// 验证透传节点不修改状态。
    #[tokio::test]
    async fn test_passthrough_node() -> Result<()> {
        let node = Node::passthrough("noop");
        let state = SharedState::new();
        state.set("x", 1)?;
        drain_node(&node, &state).await?;
        assert_eq!(state.get::<i64>("x"), Some(1)); // 未发生变化
        Ok(())
    }
}
