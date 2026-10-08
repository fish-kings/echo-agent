/*!
图工作流引擎 + 通用 Workflow / Pipeline

提供两种编排能力：

## 1. 图工作流（类似于 LangGraph）

将 agent 执行建模为**有向图 + 共享状态**，支持：
- 线性流水线、条件分支、循环、并行扇出/扇入

| 概念 | 类型 | 说明 |
|------|------|------|
| 状态 | [`SharedState`] | 节点间共享的 KV 存储 + 结构化消息历史 |
| 图 | [`Graph`] | 编译后的不可变工作流 |
| 构建器 | [`GraphBuilder`] | 用于构建图的链式 API |

## 2. 流水线工作流（Sequential / Concurrent / DAG）

| 类型 | 说明 |
|------|------|
| [`SequentialWorkflow`] | 顺序流水线：上一步的输出作为下一步的输入 |
| [`ConcurrentWorkflow`] | 并发流水线：所有 agent 并行执行后合并结果 |
| [`DagWorkflow`] | DAG 流水线：拓扑执行，独立节点并发运行 |
*/

// ── 图工作流 ────────────────────────────────────────────────────────────

/// 检查点持久化存储
pub mod checkpoint_store;
/// 图定义、构建器与执行引擎
mod graph;
/// 图节点类型与节点函数封装
mod node;
/// 预置的写作与数据处理流水线
pub mod pipelines;
/// 节点间共享的状态存储
pub mod state;

pub use checkpoint_store::{
    Checkpoint, CheckpointInfo, CheckpointStore, FileCheckpointStore, InterruptType,
    MemoryCheckpointStore,
};

pub use graph::{
    Graph, GraphBuilder, GraphResult, InterruptConfig, InterruptState, RunUntilInterruptResult,
};
pub use state::SharedState;

// ── 流水线工作流 ─────────────────────────────────────────────────────────

/// 并发流水线工作流
mod concurrent;
/// DAG 拓扑流水线工作流
mod dag;
/// 顺序流水线工作流
mod sequential;

pub use concurrent::{ConcurrentWorkflow, ConcurrentWorkflowBuilder};
pub use dag::{DagEdge, DagNode, DagWorkflow, DagWorkflowBuilder};
pub use sequential::{SequentialWorkflow, SequentialWorkflowBuilder, WorkflowStep};

// ── 预置流水线 ──────────────────────────────────────────────────────

pub use pipelines::{
    DataPipelineConfig, DataPipelineLanguage, WritingPipelineConfig, run_data_pipeline,
    run_writing_pipeline,
};

use echo_core::agent::Agent;
use echo_core::error::Result;
use futures::future::BoxFuture;
use futures::stream::BoxStream;
use std::sync::Arc;
use std::time::Duration;

/**
在返回终止错误之前，等待被取消的工作流兄弟节点收尾所花费的最长时间。
有界的收尾等待既能防止节点 future 被遗弃悬空，又能让协作式取消
有机会平稳收尾。
*/
pub(crate) const WORKFLOW_TASK_DRAIN_TIMEOUT: Duration = Duration::from_secs(5);

/**
可在异步任务间安全共享访问的 agent 句柄。

由于 [`Agent`] trait 的所有方法都只接收 `&self`，且具体实现
（例如 `ReactAgent`）通过内部可变性来管理共享状态，因此外层的锁
是不必要的 —— `Arc<dyn Agent>` 已经足够。
*/
pub type SharedAgent = Arc<dyn Agent>;

/// 将任意 `impl Agent` 包装为 [`SharedAgent`]
pub fn shared_agent(agent: impl Agent + 'static) -> SharedAgent {
    Arc::new(agent)
}

/**
为仍然引用旧 `AsyncMutex` 模式的调用点保留的向后兼容类型别名。
已标记为弃用，以推动迁移到新类型。
*/
#[deprecated(note = "Use `SharedAgent` (now `Arc<dyn Agent>`) directly")]
pub type SharedAgentMutex = Arc<tokio::sync::Mutex<Box<dyn Agent>>>;

/**
工作流执行过程中逐步发出的事件。

可通过 [`Workflow::run_stream`] 获取 `BoxStream<WorkflowEvent>`，
用于实时 UI 更新、进度条、日志记录等场景。
*/
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum WorkflowEvent {
    /// 节点开始执行
    NodeStart {
        /// 节点名称
        node_name: String,
        /// 该节点在工作流中的步骤序号
        step_index: usize,
    },
    /// 节点执行完成
    NodeEnd {
        /// 节点名称
        node_name: String,
        /// 该节点在工作流中的步骤序号
        step_index: usize,
        /// 该节点执行耗时
        elapsed: Duration,
    },
    /// 节点产生的 token（在流式转发 agent 输出时使用）
    Token {
        /// 产生该 token 的节点名称
        node_name: String,
        /// token 内容
        token: String,
    },
    /**
    节点执行出错。流会在通过 `Result` 项抛出同一个错误之前，先发出这条
    终止性诊断事件；失败的节点之后不会再跟随 Completed 事件。
    */
    NodeError {
        /// 出错的节点名称
        node_name: String,
        /// 错误描述
        error: String,
    },
    /// 工作流执行完成
    Completed {
        /// 最终结果文本
        result: String,
        /// 总步骤数
        total_steps: usize,
        /// 总耗时
        elapsed: Duration,
    },
}

/// 统一的工作流执行接口
pub trait Workflow: Send + Sync {
    /// 以 `input` 作为初始输入，运行整个工作流
    fn run<'a>(&'a mut self, input: &'a str) -> BoxFuture<'a, Result<WorkflowOutput>>;

    /**
    以 `input` 作为初始输入运行整个工作流（按节点流式返回事件）。

    默认实现会回退到调用 `run()`，只发出一个 `Completed` 事件。
    */
    fn run_stream<'a>(
        &'a mut self,
        input: &'a str,
    ) -> BoxFuture<'a, Result<BoxStream<'a, Result<WorkflowEvent>>>> {
        Box::pin(async move {
            let output = self.run(input).await?;
            let event = WorkflowEvent::Completed {
                result: output.result,
                total_steps: output.steps.len(),
                elapsed: output.elapsed,
            };
            let stream: BoxStream<'a, Result<WorkflowEvent>> =
                Box::pin(futures::stream::once(async { Ok(event) }));
            Ok(stream)
        })
    }
}

/// 工作流执行的完整输出
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WorkflowOutput {
    /// 最终结果文本
    pub result: String,
    /// 每一步的详细输出
    pub steps: Vec<StepOutput>,
    /// 总耗时
    pub elapsed: Duration,
}

/// 单个步骤执行的详细输出
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StepOutput {
    /// 执行该步骤的 agent 名称
    pub agent_name: String,
    /// 该步骤接收到的输入
    pub input: String,
    /// 该步骤产生的输出
    pub output: String,
    /// 该步骤耗时
    pub elapsed: Duration,
}
