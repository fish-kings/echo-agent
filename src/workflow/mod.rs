//! 基于图（Graph）的工作流引擎（类 LangGraph 风格的状态机）。
//!
//! 将 Agent 执行建模为**带共享状态的有向图**。
//! 支持线性流水线、条件分支、循环以及并行的扇出/扇入。
//!
//! # 构建方式
//!
//! | 构建器 | 风格 | 最适合场景 |
//! |---------|-------|----------|
//! | [`GraphBuilder`] | 编程式 Rust API | 复杂逻辑、类型安全 |
//! | [`WorkflowDefinition`] | YAML/JSON 声明式配置 | 简单流水线、无代码配置 |
//!
//! # 快速开始（编程式）
//!
//! ```rust,no_run
//! use echo_agent::prelude::*;
//!
//! # fn main() -> echo_agent::error::Result<()> {
//! let graph = GraphBuilder::new("pipeline")
//!     .add_function_node("step_a", |state| Box::pin(async move {
//!         state.set("key", "value")?;
//!         Ok(())
//!     }))
//!     .add_edge("step_a", Graph::END)
//!     .build()?;
//! # Ok(())
//! # }
//! ```
//!
//! # 快速开始（声明式 YAML）
//!
//! ```rust,ignore
//! use echo_agent::prelude::*;
//!
//! # fn main() -> echo_agent::error::Result<()> {
//! let graph = Graph::from_yaml("workflow.yaml")?;
//! let result = graph.run(SharedState::new()).await?;
//! # Ok(())
//! # }
//! ```
//!
//! # 关键类型
//!
//! | 类型 | 说明 |
//! |------|-------------|
//! | [`Graph`] / [`GraphBuilder`] | 构建并运行有向工作流 |
//! | [`SequentialWorkflow`] | 简单的线性步骤执行 |
//! | [`ConcurrentWorkflow`] | 并行步骤执行 |
//! | [`DagWorkflow`] | 基于依赖关系的调度 |
//! | [`SharedState`] | 节点之间传递的状态 |
//! | [`WorkflowEvent`] | 每个节点的流式事件 |
//! | [`WorkflowDefinition`] | YAML/JSON 工作流定义 |
//!
//! # 预置流水线
//!
//! | 流水线 | 阶段 | 说明 |
//! |----------|--------|-------------|
//! | [`run_data_pipeline`] | inspect -> persist script -> execute -> verify artifacts | 代码优先、可复现的数据分析 |
//! | [`run_writing_pipeline`] | outline -> draft -> review -> revise (循环) -> finalize | 带质量把控循环的内容创作 |

/// 直接从 `echo_orchestration::workflow` 重新导出。
pub mod orchestration {
    pub use echo_orchestration::workflow::*;
}

pub use echo_orchestration::workflow::*;

/// 构建在 orchestration 工作流类型之上的根 crate 声明式 DSL。
pub mod dsl;
/// 构建在 orchestration 工作流类型之上的根 crate 文件/YAML 加载器。
pub mod loader;

pub use loader::WorkflowDefinition;

// `Graph::resume` 的决策参数定义在 orchestration 层的 human-loop
// 模块中；在此随 workflow 接口一并重新导出，使 `echo_agent::workflow`
// 的使用方无需直接依赖 echo-orchestration，也能获得完整的
// run/resume API。
pub use echo_orchestration::human_loop::ApprovalDecision;
