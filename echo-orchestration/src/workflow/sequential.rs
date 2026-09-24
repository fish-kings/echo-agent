//! 顺序工作流：agent 按注册顺序依次执行，每一步的输出成为下一步的输入。

use super::{SharedAgent, StepOutput, Workflow, WorkflowOutput, shared_agent};
use echo_core::agent::Agent;
use echo_core::error::Result;
use futures::future::BoxFuture;
use std::time::Instant;
use tracing::{debug, info};

/// 步骤间的输入转换函数
type TransformFn = Box<dyn Fn(&str) -> String + Send + Sync>;

/// 顺序工作流中的一个步骤
pub struct WorkflowStep {
    /// 该步骤使用的 agent
    agent: SharedAgent,
    /// 应用在输入上的可选转换函数
    transform: Option<TransformFn>,
}

/// 顺序工作流：每个 agent 按注册顺序运行；前一个的输出流入后一个的输入。
///
/// # Example
///
/// ```rust,no_run
/// use echo_core::agent::{Agent, AgentEvent};
/// use echo_core::error::Result;
/// use echo_orchestration::workflow::{SequentialWorkflow, Workflow};
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
/// let agent_a = DummyAgent::new("translator");
/// let agent_b = DummyAgent::new("reviewer");
///
/// let mut wf = SequentialWorkflow::builder()
///     .step(agent_a)
///     .step(agent_b)
///     .build();
///
/// let output = wf.run("Hello world").await?;
/// println!("Final result: {}", output.result);
/// for s in &output.steps {
///     let preview: String = s.output.chars().take(80).collect();
///     println!("  {} ({:?}): {}", s.agent_name, s.elapsed, preview);
/// }
/// # Ok(())
/// # }
/// ```
pub struct SequentialWorkflow {
    /// 按顺序执行的步骤列表
    steps: Vec<WorkflowStep>,
}

impl SequentialWorkflow {
    /// 创建一个 [`SequentialWorkflowBuilder`] 用于构建顺序工作流
    pub fn builder() -> SequentialWorkflowBuilder {
        SequentialWorkflowBuilder { steps: Vec::new() }
    }
}

impl Workflow for SequentialWorkflow {
    fn run<'a>(&'a mut self, input: &'a str) -> BoxFuture<'a, Result<WorkflowOutput>> {
        Box::pin(async move {
            let total_start = Instant::now();
            let mut current_input = input.to_string();
            let mut step_outputs = Vec::with_capacity(self.steps.len());

            for (i, step) in self.steps.iter().enumerate() {
                let actual_input = if let Some(transform) = &step.transform {
                    transform(&current_input)
                } else {
                    current_input.clone()
                };

                let agent = step.agent.as_ref();
                let agent_name = agent.name().to_string();

                info!(
                    workflow = "sequential",
                    step = i + 1,
                    agent = %agent_name,
                    "▶ 正在执行步骤"
                );
                debug!(
                    workflow = "sequential",
                    step = i + 1,
                    input_len = actual_input.len()
                );

                let step_start = Instant::now();
                let output = agent.execute(&actual_input).await?;
                let step_elapsed = step_start.elapsed();

                info!(
                    workflow = "sequential",
                    step = i + 1,
                    agent = %agent_name,
                    elapsed_ms = step_elapsed.as_millis(),
                    "✓ 步骤已完成"
                );

                step_outputs.push(StepOutput {
                    agent_name,
                    input: actual_input,
                    output: output.clone(),
                    elapsed: step_elapsed,
                });

                current_input = output;
            }

            let result = current_input;
            Ok(WorkflowOutput {
                result,
                steps: step_outputs,
                elapsed: total_start.elapsed(),
            })
        })
    }
}

/// [`SequentialWorkflow`] 的构建器
pub struct SequentialWorkflowBuilder {
    /// 已添加的步骤列表
    steps: Vec<WorkflowStep>,
}

impl SequentialWorkflowBuilder {
    /// 添加一个步骤（直接使用上一步的输出作为输入）
    pub fn step(mut self, agent: impl Agent + 'static) -> Self {
        self.steps.push(WorkflowStep {
            agent: shared_agent(agent),
            transform: None,
        });
        self
    }

    /// 添加一个步骤，在执行前对输入应用一个转换函数
    pub fn step_with_transform(
        mut self,
        agent: impl Agent + 'static,
        transform: impl Fn(&str) -> String + Send + Sync + 'static,
    ) -> Self {
        self.steps.push(WorkflowStep {
            agent: shared_agent(agent),
            transform: Some(Box::new(transform)),
        });
        self
    }

    /// 添加一个步骤（使用已包装好的 SharedAgent）
    pub fn step_shared(mut self, agent: SharedAgent) -> Self {
        self.steps.push(WorkflowStep {
            agent,
            transform: None,
        });
        self
    }

    /// 构建出最终的 [`SequentialWorkflow`]
    pub fn build(self) -> SequentialWorkflow {
        SequentialWorkflow { steps: self.steps }
    }
}
