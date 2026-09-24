//! 并发工作流：所有 agent 并行处理同一份输入；结果通过合并函数聚合。

use super::{SharedAgent, StepOutput, Workflow, WorkflowOutput, shared_agent};
use echo_core::agent::Agent;
use echo_core::error::Result;
use futures::future::BoxFuture;
use std::time::Instant;
use tokio::task::JoinSet;
use tracing::{debug, info};

/// 结果合并函数
type MergeFn = Box<dyn Fn(Vec<String>) -> String + Send + Sync>;

/// 默认的结果合并方式：用 `\n---\n` 拼接各结果
fn default_merge(results: Vec<String>) -> String {
    results.join("\n---\n")
}

/// 并发工作流：所有已注册的 agent 并行执行；结果通过 `merge` 函数合并。
///
/// # Example
///
/// ```rust,no_run
/// use echo_core::agent::{Agent, AgentEvent};
/// use echo_core::error::Result;
/// use echo_orchestration::workflow::{ConcurrentWorkflow, Workflow};
/// use futures::future::BoxFuture;
/// use futures::stream::{self, BoxStream};
///
/// # struct DummyAgent {
/// #     name: String,
/// # }
/// #
/// # impl DummyAgent {
/// #     fn new(name: impl Into<String>) -> Self {
/// #         Self { name: name.into() }
/// #     }
/// # }
/// #
/// # impl Agent for DummyAgent {
/// #     fn name(&self) -> &str { &self.name }
/// #     fn model_name(&self) -> &str { "mock-model" }
/// #     fn system_prompt(&self) -> &str { "You are a mock agent" }
/// #     fn execute<'a>(&'a self, task: &'a str) -> BoxFuture<'a, Result<String>> {
/// #         Box::pin(async move { Ok(format!("{}: {task}", self.name)) })
/// #     }
/// #     fn execute_stream<'a>(&'a self, _task: &'a str) -> BoxFuture<'a, Result<BoxStream<'a, Result<AgentEvent>>>> {
/// #         Box::pin(async move {
/// #             let s: BoxStream<'a, Result<AgentEvent>> = Box::pin(stream::empty());
/// #             Ok(s)
/// #         })
/// #     }
/// # }
///
/// # async fn example() -> Result<()> {
/// let agent_x = DummyAgent::new("tech");
/// let agent_y = DummyAgent::new("business");
///
/// let mut wf = ConcurrentWorkflow::builder()
///     .agent(agent_x)
///     .agent(agent_y)
///     .merge(|results| {
///         format!("Combined analysis:\n{}", results.join("\n\n"))
///     })
///     .build();
///
/// let output = wf.run("Analyze AI Agent development trends").await?;
/// println!("{}", output.result);
/// # Ok(())
/// # }
/// ```
pub struct ConcurrentWorkflow {
    /// 已注册的、并发执行的 agent 列表
    agents: Vec<SharedAgent>,
    /// 结果合并函数
    merge: MergeFn,
}

impl ConcurrentWorkflow {
    /// 创建一个 [`ConcurrentWorkflowBuilder`] 用于构建并发工作流
    pub fn builder() -> ConcurrentWorkflowBuilder {
        ConcurrentWorkflowBuilder {
            agents: Vec::new(),
            merge: None,
        }
    }
}

impl Workflow for ConcurrentWorkflow {
    fn run<'a>(&'a mut self, input: &'a str) -> BoxFuture<'a, Result<WorkflowOutput>> {
        Box::pin(async move {
            let total_start = Instant::now();
            let agent_count = self.agents.len();

            info!(
                workflow = "concurrent",
                agents = agent_count,
                "⚡ 正在并发执行 {} 个 agent",
                agent_count
            );

            let mut handles = JoinSet::new();

            for (index, agent_handle) in self.agents.iter().enumerate() {
                let agent_handle = agent_handle.clone();
                let input = input.to_string();
                handles.spawn(async move {
                    let step_start = Instant::now();
                    let agent = agent_handle.as_ref();
                    let agent_name = agent.name().to_string();
                    debug!(workflow = "concurrent", agent = %agent_name, "▶ 开始执行");
                    let result = agent.execute(&input).await;
                    let elapsed = step_start.elapsed();
                    (index, agent_name, input, result, elapsed)
                });
            }

            let mut completed = Vec::with_capacity(agent_count);
            completed.resize_with(agent_count, || None);

            let mut first_error = None;
            while let Some(joined) = handles.join_next().await {
                let (index, agent_name, step_input, result, elapsed) = match joined {
                    Ok(value) => value,
                    Err(error) => {
                        first_error = Some(echo_core::error::ReactError::Other(format!(
                            "task join error: {error}"
                        )));
                        break;
                    }
                };
                let output = match result {
                    Ok(output) => output,
                    Err(error) => {
                        first_error = Some(error);
                        break;
                    }
                };
                info!(
                    workflow = "concurrent",
                    agent = %agent_name,
                    elapsed_ms = elapsed.as_millis(),
                    "✓ Agent 已完成"
                );

                let Some(slot) = completed.get_mut(index) else {
                    first_error = Some(echo_core::error::ReactError::Other(format!(
                        "task completed with invalid registration index {index}"
                    )));
                    break;
                };
                if slot
                    .replace((agent_name, step_input, output, elapsed))
                    .is_some()
                {
                    first_error = Some(echo_core::error::ReactError::Other(format!(
                        "task completed more than once for registration index {index}"
                    )));
                    break;
                }
            }
            if let Some(error) = first_error {
                handles.abort_all();
                while handles.join_next().await.is_some() {}
                return Err(error);
            }

            let mut step_outputs = Vec::with_capacity(agent_count);
            let mut results = Vec::with_capacity(agent_count);
            for completed_step in completed {
                let (agent_name, step_input, output, elapsed) =
                    completed_step.ok_or_else(|| {
                        echo_core::error::ReactError::Other(
                            "concurrent task completed without a registered result".to_string(),
                        )
                    })?;
                step_outputs.push(StepOutput {
                    agent_name,
                    input: step_input,
                    output: output.clone(),
                    elapsed,
                });
                results.push(output);
            }

            let merged = (self.merge)(results);

            Ok(WorkflowOutput {
                result: merged,
                steps: step_outputs,
                elapsed: total_start.elapsed(),
            })
        })
    }
}

/// [`ConcurrentWorkflow`] 的构建器
pub struct ConcurrentWorkflowBuilder {
    /// 已添加的 agent 列表
    agents: Vec<SharedAgent>,
    /// 可选的结果合并函数
    merge: Option<MergeFn>,
}

impl ConcurrentWorkflowBuilder {
    /// 添加一个需要并发执行的 agent
    pub fn agent(mut self, agent: impl Agent + 'static) -> Self {
        self.agents.push(shared_agent(agent));
        self
    }

    /// 添加一个已包装好的 SharedAgent
    pub fn agent_shared(mut self, agent: SharedAgent) -> Self {
        self.agents.push(agent);
        self
    }

    /// 设置结果合并函数（默认使用 `\n---\n` 拼接）
    pub fn merge(mut self, f: impl Fn(Vec<String>) -> String + Send + Sync + 'static) -> Self {
        self.merge = Some(Box::new(f));
        self
    }

    /// 构建出最终的 [`ConcurrentWorkflow`]
    pub fn build(self) -> ConcurrentWorkflow {
        ConcurrentWorkflow {
            agents: self.agents,
            merge: self.merge.unwrap_or_else(|| Box::new(default_merge)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use echo_core::agent::AgentEvent;
    use echo_core::error::ReactError;
    use futures::stream::{self, BoxStream};
    use std::time::Duration;

    /// 用于测试的 agent，可配置是否失败以及延迟时长
    struct TestAgent {
        name: &'static str,
        fail: bool,
        delay: Duration,
    }

    impl Agent for TestAgent {
        fn name(&self) -> &str {
            self.name
        }

        fn model_name(&self) -> &str {
            "test"
        }

        fn system_prompt(&self) -> &str {
            "test"
        }

        fn execute<'a>(&'a self, _task: &'a str) -> BoxFuture<'a, Result<String>> {
            Box::pin(async move {
                tokio::time::sleep(self.delay).await;
                if self.fail {
                    Err(ReactError::Other(format!("{} failed", self.name)))
                } else {
                    Ok(self.name.to_string())
                }
            })
        }

        fn execute_stream<'a>(
            &'a self,
            _task: &'a str,
        ) -> BoxFuture<'a, Result<BoxStream<'a, Result<AgentEvent>>>> {
            Box::pin(async move {
                let stream: BoxStream<'a, Result<AgentEvent>> = Box::pin(stream::empty());
                Ok(stream)
            })
        }
    }

    #[tokio::test]
    async fn failed_concurrent_workflow_aborts_and_drains_siblings() {
        let mut workflow = ConcurrentWorkflow::builder()
            .agent(TestAgent {
                name: "fail",
                fail: true,
                delay: Duration::from_millis(1),
            })
            .agent(TestAgent {
                name: "slow",
                fail: false,
                delay: Duration::from_secs(60),
            })
            .build();

        let result = tokio::time::timeout(Duration::from_secs(1), workflow.run("input"))
            .await
            .unwrap();
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn successful_concurrent_workflow_preserves_registration_order() -> Result<()> {
        let mut workflow = ConcurrentWorkflow::builder()
            .agent(TestAgent {
                name: "registered-first",
                fail: false,
                delay: Duration::from_millis(30),
            })
            .agent(TestAgent {
                name: "completed-first",
                fail: false,
                delay: Duration::from_millis(1),
            })
            .build();

        let output = workflow.run("input").await?;
        assert_eq!(output.result, "registered-first\n---\ncompleted-first");
        assert_eq!(
            output
                .steps
                .iter()
                .map(|step| step.agent_name.as_str())
                .collect::<Vec<_>>(),
            vec!["registered-first", "completed-first"]
        );
        Ok(())
    }
}
