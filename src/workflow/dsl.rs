/*!
高层声明式 DSL —— StateGraph

提供类似 LangGraph 的链式 API，用于声明式地构建多 Agent 拓扑结构。

# 示例

```rust,no_run
use echo_agent::workflow::dsl::StateGraph;
use echo_agent::prelude::*;

# async fn example() -> echo_agent::error::Result<()> {
let mut sg = StateGraph::new("research_flow");
sg.add_react_node("researcher", |b| {
        b.model("qwen3-max").system_prompt("Researcher").enable_tools()
    })
    .with_input("topic").with_output("research");
sg.add_react_node("writer", |b| {
        b.model("qwen3-max").system_prompt("Writer")
    })
    .with_input("research").with_output("draft");
sg.add_function_node("checker", |state| Box::pin(async move {
        let draft: String = state.get("draft").unwrap_or_default();
        state.set("review", if draft.len() > 100 { "pass" } else { "fail" });
        Ok(())
    }));
sg.add_conditional_edge("checker", |state| Box::pin(async {
        let r: String = state.get("review").unwrap_or_default();
        if r == "pass" { "done" } else { "writer" }.to_string()
    }));
sg.add_react_node("done", |b| {
        b.model("qwen3-max").system_prompt("Polisher")
    })
    .with_input("draft").with_output("final");
sg.entry("researcher").finish("done");
let graph = sg.compile()?;
# Ok(())
# }
```
*/

use crate::agent::react::builder::ReactAgentBuilder;
use crate::error::Result;
use crate::workflow::Graph;
use crate::workflow::GraphBuilder;
use crate::workflow::SharedState;
use futures::future::BoxFuture;

// ── 待处理节点定义 ──────────────────────────────────────────────────

/// 配置 ReactAgentBuilder 的闭包类型
type AgentConfigFn = Box<dyn FnOnce(ReactAgentBuilder) -> ReactAgentBuilder>;
/// 函数节点的执行逻辑闭包类型
type FunctionNodeFn =
    Box<dyn for<'a> Fn(&'a SharedState) -> BoxFuture<'a, Result<()>> + Send + Sync>;
/// 条件边判断逻辑的闭包类型
type ConditionFn = Box<dyn for<'a> Fn(&'a SharedState) -> BoxFuture<'a, String> + Send + Sync>;

/// 编译前暂存的节点定义
enum PendingNode {
    /// 一个 ReactAgent 节点
    Agent {
        /// 节点名称
        name: String,
        /// 用于配置 Agent 构建器的闭包
        config_fn: AgentConfigFn,
        /// 状态中读取输入的 key
        input_key: String,
        /// 写入状态的输出 key
        output_key: String,
    },
    /// 一个自定义函数节点
    Function {
        /// 节点名称
        name: String,
        /// 节点执行逻辑
        f: FunctionNodeFn,
    },
    /// 一个路由节点
    Router {
        /// 节点名称
        name: String,
    },
}

/// 编译前暂存的边定义
enum PendingEdge {
    /// 固定边：from 直接指向 to
    Fixed {
        /// 起始节点
        from: String,
        /// 目标节点
        to: String,
    },
    /// 条件边：根据状态动态决定下一个节点
    Conditional {
        /// 起始节点
        from: String,
        /// 判断下一个节点的闭包
        f: ConditionFn,
    },
    /// 并行边：from 同时分发到多个目标节点，全部完成后汇聚到 then
    Parallel {
        /// 起始节点
        from: String,
        /// 并行执行的目标节点列表
        targets: Vec<String>,
        /// 所有目标节点完成后汇聚的节点
        then: String,
    },
}

// ── StateGraph ───────────────────────────────────────────────────────────

/**
声明式 Agent 拓扑构建器。

通过链式 API 定义节点、边和状态字段，最终通过 `compile()` 编译为 `Graph`。

所有 `add_*` 和 `with_*` 方法都返回 `&mut Self`，支持链式调用。
*/
pub struct StateGraph {
    /// 图的名称
    name: String,
    /// 尚未编译的节点列表
    pending_nodes: Vec<PendingNode>,
    /// 尚未编译的边列表
    edges: Vec<PendingEdge>,
    /// 入口节点名称
    entry: Option<String>,
    /// 终止节点名称列表
    finish_nodes: Vec<String>,
}

impl StateGraph {
    /// 创建一个新的 StateGraph
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            pending_nodes: Vec::new(),
            edges: Vec::new(),
            entry: None,
            finish_nodes: Vec::new(),
        }
    }

    /**
    添加一个 ReactAgent 节点。

    `configure` 闭包接收一个 `ReactAgentBuilder` 并返回配置好的构建器。
    默认输入为 "task"，输出为 "result"。
    */
    pub fn add_react_node<F>(&mut self, name: impl Into<String>, configure: F) -> &mut Self
    where
        F: FnOnce(ReactAgentBuilder) -> ReactAgentBuilder + 'static,
    {
        self.pending_nodes.push(PendingNode::Agent {
            name: name.into(),
            config_fn: Box::new(configure),
            input_key: "task".to_string(),
            output_key: "result".to_string(),
        });
        self
    }

    /// 设置最后一个 agent 节点的输入 key
    pub fn with_input(&mut self, key: impl Into<String>) -> &mut Self {
        if let Some(PendingNode::Agent { input_key, .. }) = self.pending_nodes.last_mut() {
            *input_key = key.into();
        }
        self
    }

    /// 设置最后一个 agent 节点的输出 key
    pub fn with_output(&mut self, key: impl Into<String>) -> &mut Self {
        if let Some(PendingNode::Agent { output_key, .. }) = self.pending_nodes.last_mut() {
            *output_key = key.into();
        }
        self
    }

    /// 添加一个函数节点
    pub fn add_function_node<F>(&mut self, name: impl Into<String>, f: F) -> &mut Self
    where
        F: for<'a> Fn(&'a SharedState) -> BoxFuture<'a, Result<()>> + Send + Sync + 'static,
    {
        self.pending_nodes.push(PendingNode::Function {
            name: name.into(),
            f: Box::new(f),
        });
        self
    }

    /// 添加一个路由节点
    pub fn add_router(&mut self, name: impl Into<String>) -> &mut Self {
        self.pending_nodes
            .push(PendingNode::Router { name: name.into() });
        self
    }

    /// 添加一条固定边：from → to
    pub fn add_edge(&mut self, from: impl Into<String>, to: impl Into<String>) -> &mut Self {
        self.edges.push(PendingEdge::Fixed {
            from: from.into(),
            to: to.into(),
        });
        self
    }

    /// 添加一条条件边
    pub fn add_conditional_edge<F>(&mut self, from: impl Into<String>, f: F) -> &mut Self
    where
        F: for<'a> Fn(&'a SharedState) -> BoxFuture<'a, String> + Send + Sync + 'static,
    {
        self.edges.push(PendingEdge::Conditional {
            from: from.into(),
            f: Box::new(f),
        });
        self
    }

    /// 添加并行边：from → [targets...] → then
    pub fn add_parallel_edge(
        &mut self,
        from: impl Into<String>,
        targets: Vec<String>,
        then: impl Into<String>,
    ) -> &mut Self {
        self.edges.push(PendingEdge::Parallel {
            from: from.into(),
            targets,
            then: then.into(),
        });
        self
    }

    /// 设置入口节点
    pub fn entry(&mut self, name: impl Into<String>) -> &mut Self {
        self.entry = Some(name.into());
        self
    }

    /// 添加一个终止节点
    pub fn finish(&mut self, name: impl Into<String>) -> &mut Self {
        self.finish_nodes.push(name.into());
        self
    }

    /**
    编译为 Graph

    遍历所有待处理节点，构建 agent，添加边以及入口/终止节点，然后完成构建。
    */
    pub fn compile(self) -> Result<Graph> {
        let mut builder = GraphBuilder::new(&self.name);

        // 构建所有待处理节点
        for node in self.pending_nodes {
            match node {
                PendingNode::Agent {
                    name,
                    config_fn,
                    input_key,
                    output_key,
                } => {
                    let base = ReactAgentBuilder::new();
                    let configured = config_fn(base);
                    let agent = configured.build()?;
                    builder = builder.add_agent_node(&name, agent, &input_key, &output_key);
                }
                PendingNode::Function { name, f } => {
                    builder = builder.add_function_node(&name, f);
                }
                PendingNode::Router { name } => {
                    builder = builder.add_router_node(&name);
                }
            }
        }

        // 添加所有边
        for edge in self.edges {
            match edge {
                PendingEdge::Fixed { from, to } => {
                    builder = builder.add_edge(&from, &to);
                }
                PendingEdge::Conditional { from, f } => {
                    builder = builder.add_conditional_edge(&from, f);
                }
                PendingEdge::Parallel {
                    from,
                    targets,
                    then,
                } => {
                    builder = builder.add_parallel_edge(&from, targets, &then);
                }
            }
        }

        // 设置入口和终止节点
        if let Some(entry) = self.entry {
            builder = builder.set_entry(&entry);
        }
        for finish in self.finish_nodes {
            builder = builder.set_finish(&finish);
        }

        builder.build()
    }
}

/// `StateGraph` DSL 的单元测试。
#[cfg(test)]
mod tests {
    use super::*;

    /// 验证函数节点可正确添加并构建。
    #[tokio::test]
    async fn test_state_graph_function_nodes() {
        let mut sg = StateGraph::new("test_flow");
        sg.add_function_node("step1", |state| {
            Box::pin(async move {
                let _ = state.set("x", "hello");
                Ok(())
            })
        })
        .add_function_node("step2", |state| {
            Box::pin(async move {
                let x: String = state.get("x").unwrap_or_default();
                let _ = state.set("y", format!("{} world", x));
                Ok(())
            })
        })
        .add_edge("step1", "step2")
        .add_edge("step2", "__end__")
        .entry("step1")
        .finish("step2");

        let graph = sg.compile().unwrap();
        let state = SharedState::new();
        let result = graph.run(state).await.unwrap();
        let y: String = result.state.get("y").unwrap_or_default();
        assert_eq!(y, "hello world");
    }

    /// 验证状态图按顺序运行并产出预期状态。
    #[tokio::test]
    async fn test_state_graph_run() {
        let mut sg = StateGraph::new("hello_flow");
        sg.add_function_node("greet", |state| {
            Box::pin(async move {
                let _ = state.set("msg", "hello");
                Ok(())
            })
        })
        .add_function_node("shout", |state| {
            Box::pin(async move {
                let msg: String = state.get("msg").unwrap_or_default();
                let _ = state.set("msg", format!("{}!", msg.to_uppercase()));
                Ok(())
            })
        })
        .add_edge("greet", "shout")
        .add_edge("shout", "__end__")
        .entry("greet")
        .finish("shout");

        let graph = sg.compile().unwrap();
        let state = SharedState::new();
        let result = graph.run(state).await.unwrap();
        let msg: String = result.state.get("msg").unwrap_or_default();
        assert_eq!(msg, "HELLO!");
    }

    /// 验证条件边会根据状态选择正确的分支。
    #[tokio::test]
    async fn test_state_graph_conditional() {
        let mut sg = StateGraph::new("cond_flow");
        sg.add_function_node("check", |state| {
            Box::pin(async move {
                let _ = state.set("status", "ok");
                Ok(())
            })
        })
        .add_function_node("pass", |state| {
            Box::pin(async move {
                let _ = state.set("result", "passed");
                Ok(())
            })
        })
        .add_function_node("fail", |state| {
            Box::pin(async move {
                let _ = state.set("result", "failed");
                Ok(())
            })
        })
        .add_conditional_edge("check", |state| {
            Box::pin(async {
                let s: String = state.get("status").unwrap_or_default();
                if s == "ok" { "pass" } else { "fail" }.to_string()
            })
        })
        .add_edge("pass", "__end__")
        .add_edge("fail", "__end__")
        .entry("check")
        .finish("pass")
        .finish("fail");

        let graph = sg.compile().unwrap();
        let state = SharedState::new();
        let result = graph.run(state).await.unwrap();
        assert_eq!(result.steps, 2); // check + pass 两步
    }
}
