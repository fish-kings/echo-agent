//! DAG 工作流：Agent 以有向无环图组织，按拓扑顺序执行，相互独立的节点并发运行。

use super::{SharedAgent, StepOutput, Workflow, WorkflowOutput, shared_agent};
use echo_core::agent::Agent;
use echo_core::error::{AgentError, ReactError, Result};
use futures::future::BoxFuture;
use std::collections::{HashMap, HashSet, VecDeque};
use std::time::Instant;
use tokio::task::JoinSet;
use tracing::{debug, info};

/// DAG 中的一个节点。
pub struct DagNode {
    /// 节点唯一标识。
    pub id: String,
    /// 该节点执行所用的 Agent。
    pub agent: SharedAgent,
}

/// DAG 中的有向边（from -> to 表示 to 依赖 from 的输出）。
#[derive(Debug, Clone)]
pub struct DagEdge {
    /// 起点节点 ID（上游）。
    pub from: String,
    /// 终点节点 ID（下游）。
    pub to: String,
}

/**
DAG 工作流：节点按拓扑顺序执行，相互独立的节点并发运行。

每个节点以其所有前驱节点输出的拼接结果（以空行连接）作为输入；
入度为 0 的根节点接收工作流的原始输入。

# 示例

```rust,no_run
use echo_core::agent::{Agent, AgentEvent};
use echo_core::error::Result;
use echo_orchestration::workflow::{DagWorkflow, Workflow};
use futures::future::BoxFuture;
use futures::stream::{self, BoxStream};

# struct DummyAgent {
#     name: String,
# }
#
# impl DummyAgent {
#     fn new(name: impl Into<String>) -> Self {
#         Self { name: name.into() }
#     }
# }
#
# impl Agent for DummyAgent {
#     fn name(&self) -> &str { &self.name }
#     fn model_name(&self) -> &str { "mock-model" }
#     fn system_prompt(&self) -> &str { "You are a mock agent" }
#     fn execute<'a>(&'a self, task: &'a str) -> BoxFuture<'a, Result<String>> {
#         Box::pin(async move { Ok(format!("{}: {task}", self.name)) })
#     }
#     fn execute_stream<'a>(&'a self, _task: &'a str) -> BoxFuture<'a, Result<BoxStream<'a, Result<AgentEvent>>>> {
#         Box::pin(async move {
#             let s: BoxStream<'a, Result<AgentEvent>> = Box::pin(stream::empty());
#             Ok(s)
#         })
#     }
# }

# async fn example() -> Result<()> {
let researcher = DummyAgent::new("researcher");
let analyst = DummyAgent::new("analyst");
let writer = DummyAgent::new("writer");

let mut wf = DagWorkflow::builder()
    .node("research", researcher)
    .node("analyze", analyst)
    .node("write", writer)
    .edge("research", "write")
    .edge("analyze", "write")
    .build()?;

let output = wf.run("Analyze the 2025 AI Agent ecosystem").await?;
println!("{}", output.result);
# Ok(())
# }
```
*/
pub struct DagWorkflow {
    /// 节点 ID 到 Agent 的映射。
    nodes: HashMap<String, SharedAgent>,
    /// 所有有向边。
    edges: Vec<DagEdge>,
    /// 构建时计算出的拓扑顺序。
    node_order: Vec<String>,
}

impl DagWorkflow {
    /// 创建 [`DagWorkflowBuilder`] 构建器。
    pub fn builder() -> DagWorkflowBuilder {
        DagWorkflowBuilder {
            nodes: Vec::new(),
            edges: Vec::new(),
        }
    }
}

impl Workflow for DagWorkflow {
    /// 按拓扑顺序分批并发执行节点，最终汇总所有叶子节点的输出。
    fn run<'a>(&'a mut self, input: &'a str) -> BoxFuture<'a, Result<WorkflowOutput>> {
        Box::pin(async move {
            let total_start = Instant::now();
            let mut step_outputs: Vec<StepOutput> = Vec::new();
            let mut node_results: HashMap<String, String> = HashMap::new();

            let predecessors = build_predecessors(&self.edges);
            let successors = build_successors(&self.edges);
            let in_degree = compute_in_degree(&self.node_order, &self.edges);

            let mut remaining_in_degree = in_degree.clone();
            let mut ready: VecDeque<String> = VecDeque::new();

            for node_id in &self.node_order {
                if remaining_in_degree
                    .get(node_id.as_str())
                    .copied()
                    .unwrap_or_default()
                    == 0
                {
                    ready.push_back(node_id.clone());
                }
            }

            info!(
                workflow = "dag",
                nodes = self.node_order.len(),
                edges = self.edges.len(),
                roots = ready.len(),
                "🔀 DAG workflow started"
            );

            while !ready.is_empty() {
                let batch: Vec<String> = ready.drain(..).collect();

                debug!(
                    workflow = "dag",
                    batch = ?batch,
                    "⚡ Executing {} nodes concurrently",
                    batch.len()
                );

                let mut handles = JoinSet::new();

                for (batch_index, node_id) in batch.iter().enumerate() {
                    let agent_handle = self.nodes.get(node_id).cloned().ok_or_else(|| {
                        ReactError::Other(format!("DAG node {node_id:?} has no registered agent"))
                    })?;
                    let preds = predecessors
                        .get(node_id.as_str())
                        .cloned()
                        .unwrap_or_default();

                    let node_input = if preds.is_empty() {
                        input.to_string()
                    } else {
                        preds
                            .iter()
                            .filter_map(|p| node_results.get(p.as_str()))
                            .cloned()
                            .collect::<Vec<_>>()
                            .join("\n\n")
                    };

                    let nid = node_id.clone();
                    handles.spawn(async move {
                        let step_start = Instant::now();
                        let agent = agent_handle.as_ref();
                        let agent_name = agent.name().to_string();
                        let result = agent.execute(&node_input).await;
                        let elapsed = step_start.elapsed();
                        (batch_index, nid, agent_name, node_input, result, elapsed)
                    });
                }

                let mut completed = Vec::with_capacity(batch.len());
                completed.resize_with(batch.len(), || None);
                let mut first_error = None;
                while let Some(joined) = handles.join_next().await {
                    let (batch_index, node_id, agent_name, node_input, result, elapsed) =
                        match joined {
                            Ok(value) => value,
                            Err(error) => {
                                first_error =
                                    Some(ReactError::Other(format!("task join error: {error}")));
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

                    let Some(slot) = completed.get_mut(batch_index) else {
                        first_error = Some(ReactError::Other(format!(
                            "DAG node completed with invalid batch index {batch_index}"
                        )));
                        break;
                    };
                    if slot
                        .replace((node_id, agent_name, node_input, output, elapsed))
                        .is_some()
                    {
                        first_error = Some(ReactError::Other(format!(
                            "DAG node completed more than once for batch index {batch_index}"
                        )));
                        break;
                    }
                }
                if let Some(error) = first_error {
                    handles.abort_all();
                    while handles.join_next().await.is_some() {}
                    return Err(error);
                }

                for completed_node in completed {
                    let (node_id, agent_name, node_input, output, elapsed) = completed_node
                        .ok_or_else(|| {
                            ReactError::Other(
                                "DAG batch completed without a registered result".to_string(),
                            )
                        })?;
                    info!(
                        workflow = "dag",
                        node = %node_id,
                        agent = %agent_name,
                        elapsed_ms = elapsed.as_millis(),
                        "✓ Node completed"
                    );
                    step_outputs.push(StepOutput {
                        agent_name,
                        input: node_input,
                        output: output.clone(),
                        elapsed,
                    });
                    node_results.insert(node_id.clone(), output);

                    if let Some(succs) = successors.get(node_id.as_str()) {
                        for succ in succs {
                            if let Some(deg) = remaining_in_degree.get_mut(succ.as_str()) {
                                *deg = deg.saturating_sub(1);
                                if *deg == 0 {
                                    ready.push_back(succ.clone());
                                }
                            }
                        }
                    }
                }
            }

            // 最终结果：收集所有叶子节点（出度为 0）的输出
            let leaf_nodes: Vec<&str> = self
                .node_order
                .iter()
                .filter(|id| successors.get(id.as_str()).is_none_or(|s| s.is_empty()))
                .map(|s| s.as_str())
                .collect();

            let final_result = leaf_nodes
                .iter()
                .filter_map(|id| node_results.get(*id))
                .cloned()
                .collect::<Vec<_>>()
                .join("\n\n");

            Ok(WorkflowOutput {
                result: final_result,
                steps: step_outputs,
                elapsed: total_start.elapsed(),
            })
        })
    }
}

/// [`DagWorkflow`] 的构建器。
pub struct DagWorkflowBuilder {
    /// 已注册的（节点 ID, Agent）列表。
    nodes: Vec<(String, SharedAgent)>,
    /// 已添加的有向边。
    edges: Vec<DagEdge>,
}

impl DagWorkflowBuilder {
    /// 注册一个命名节点。
    pub fn node(mut self, id: impl Into<String>, agent: impl Agent + 'static) -> Self {
        self.nodes.push((id.into(), shared_agent(agent)));
        self
    }

    /// 注册一个命名节点（使用已包装好的 SharedAgent）。
    pub fn node_shared(mut self, id: impl Into<String>, agent: SharedAgent) -> Self {
        self.nodes.push((id.into(), agent));
        self
    }

    /// 添加一条有向边：`from` 的输出将流入 `to` 的输入。
    pub fn edge(mut self, from: impl Into<String>, to: impl Into<String>) -> Self {
        self.edges.push(DagEdge {
            from: from.into(),
            to: to.into(),
        });
        self
    }

    /// 构建 DAG 工作流：校验节点与边、检测环并计算拓扑顺序。
    pub fn build(self) -> Result<DagWorkflow> {
        let node_ids: HashSet<&str> = self.nodes.iter().map(|(id, _)| id.as_str()).collect();
        if node_ids.len() != self.nodes.len() {
            return Err(ReactError::Agent(Box::new(
                AgentError::InitializationFailed("DAG contains duplicate node IDs".to_string()),
            )));
        }

        for edge in &self.edges {
            if !node_ids.contains(edge.from.as_str()) {
                return Err(ReactError::Agent(Box::new(
                    AgentError::InitializationFailed(format!(
                        "DAG edge references unknown node: '{}'",
                        edge.from
                    )),
                )));
            }
            if !node_ids.contains(edge.to.as_str()) {
                return Err(ReactError::Agent(Box::new(
                    AgentError::InitializationFailed(format!(
                        "DAG edge references unknown node: '{}'",
                        edge.to
                    )),
                )));
            }
        }

        let node_list: Vec<String> = self.nodes.iter().map(|(id, _)| id.clone()).collect();
        if let Some(cycle) = detect_cycle(&node_list, &self.edges) {
            return Err(ReactError::Agent(Box::new(
                AgentError::InitializationFailed(format!(
                    "DAG contains cycle: {}",
                    cycle.join(" → ")
                )),
            )));
        }

        let topo_order = topological_sort(&node_list, &self.edges)?;

        let nodes: HashMap<String, SharedAgent> = self.nodes.into_iter().collect();

        Ok(DagWorkflow {
            nodes,
            edges: self.edges,
            node_order: topo_order,
        })
    }
}

// ── DAG 算法 ──────────────────────────────────────────────────────────────────

/// 构建每个节点的前驱节点列表。
fn build_predecessors(edges: &[DagEdge]) -> HashMap<&str, Vec<String>> {
    let mut preds: HashMap<&str, Vec<String>> = HashMap::new();
    for edge in edges {
        preds
            .entry(edge.to.as_str())
            .or_default()
            .push(edge.from.clone());
    }
    preds
}

/// 构建每个节点的后继节点列表。
fn build_successors(edges: &[DagEdge]) -> HashMap<&str, Vec<String>> {
    let mut succs: HashMap<&str, Vec<String>> = HashMap::new();
    for edge in edges {
        succs
            .entry(edge.from.as_str())
            .or_default()
            .push(edge.to.clone());
    }
    succs
}

/// 计算每个节点的入度。
fn compute_in_degree<'a>(nodes: &'a [String], edges: &[DagEdge]) -> HashMap<&'a str, usize> {
    let mut deg: HashMap<&str, usize> = nodes.iter().map(|id| (id.as_str(), 0)).collect();
    for edge in edges {
        if let Some(d) = deg.get_mut(edge.to.as_str()) {
            *d = d.saturating_add(1);
        }
    }
    deg
}

/// 使用 Kahn 算法进行拓扑排序。
fn topological_sort(nodes: &[String], edges: &[DagEdge]) -> Result<Vec<String>> {
    let mut in_deg = compute_in_degree(nodes, edges);
    let succs = build_successors(edges);

    let mut queue: VecDeque<String> = nodes
        .iter()
        .filter(|id| in_deg.get(id.as_str()).copied().unwrap_or_default() == 0)
        .cloned()
        .collect();

    let mut order = Vec::with_capacity(nodes.len());

    while let Some(node) = queue.pop_front() {
        order.push(node.clone());
        if let Some(neighbors) = succs.get(node.as_str()) {
            for neighbor in neighbors {
                if let Some(d) = in_deg.get_mut(neighbor.as_str()) {
                    *d -= 1;
                    if *d == 0 {
                        queue.push_back(neighbor.clone());
                    }
                }
            }
        }
    }

    if order.len() != nodes.len() {
        return Err(ReactError::Agent(Box::new(
            AgentError::InitializationFailed(
                "DAG contains a cycle (topological sort incomplete)".to_string(),
            ),
        )));
    }

    Ok(order)
}

/// 基于 DFS 的环检测；若发现环则返回环路径。
fn detect_cycle(nodes: &[String], edges: &[DagEdge]) -> Option<Vec<String>> {
    let succs: HashMap<String, Vec<String>> = {
        let mut map: HashMap<String, Vec<String>> = HashMap::new();
        for edge in edges {
            map.entry(edge.from.clone())
                .or_default()
                .push(edge.to.clone());
        }
        map
    };

    /// DFS 三色标记。
    #[derive(Clone, Copy, PartialEq)]
    enum Color {
        /// 尚未访问。
        White,
        /// 正在访问（位于当前递归路径上）。
        Gray,
        /// 已完成访问。
        Black,
    }

    let mut color: HashMap<String, Color> =
        nodes.iter().map(|id| (id.clone(), Color::White)).collect();
    let mut path: Vec<String> = Vec::new();

    /// 深度优先遍历；遇到灰色节点即表示存在环。
    fn dfs(
        node: &str,
        succs: &HashMap<String, Vec<String>>,
        color: &mut HashMap<String, Color>,
        path: &mut Vec<String>,
    ) -> bool {
        color.insert(node.to_string(), Color::Gray);
        path.push(node.to_string());

        if let Some(neighbors) = succs.get(node) {
            for neighbor in neighbors {
                match color.get(neighbor.as_str()).copied() {
                    Some(Color::Gray) => {
                        path.push(neighbor.clone());
                        return true;
                    }
                    Some(Color::White) | None if dfs(neighbor, succs, color, path) => {
                        return true;
                    }
                    Some(Color::White) | None => {}
                    _ => {}
                }
            }
        }

        path.pop();
        color.insert(node.to_string(), Color::Black);
        false
    }

    for node in nodes {
        if color[node.as_str()] == Color::White && dfs(node, &succs, &mut color, &mut path) {
            return Some(path);
        }
    }

    None
}

// ── 单元测试 ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use echo_core::agent::AgentEvent;
    use futures::stream::{self, BoxStream};
    use std::time::Duration;

    /// 可配置延迟与失败行为的测试 Agent。
    struct TestAgent {
        /// Agent 名称，同时作为成功时的输出。
        name: &'static str,
        /// 是否返回错误。
        fail: bool,
        /// 执行前的等待时长。
        delay: Duration,
    }

    impl Agent for TestAgent {
        /// 返回测试 Agent 名称
        fn name(&self) -> &str {
            self.name
        }

        /// 返回模拟模型名称
        fn model_name(&self) -> &str {
            "test"
        }

        /// 返回模拟系统提示词
        fn system_prompt(&self) -> &str {
            "test"
        }

        /// 执行任务并返回模拟结果
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

        /// 以流式方式执行任务，返回模拟事件流
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

    #[test]
    /// 验证线性链的拓扑排序。
    fn test_topological_sort_simple() {
        let nodes = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let edges = vec![
            DagEdge {
                from: "a".into(),
                to: "b".into(),
            },
            DagEdge {
                from: "b".into(),
                to: "c".into(),
            },
        ];
        let order = topological_sort(&nodes, &edges).unwrap();
        assert_eq!(order, vec!["a", "b", "c"]);
    }

    #[test]
    /// 验证菱形依赖的拓扑排序。
    fn test_topological_sort_diamond() {
        let nodes = vec![
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
            "d".to_string(),
        ];
        let edges = vec![
            DagEdge {
                from: "a".into(),
                to: "b".into(),
            },
            DagEdge {
                from: "a".into(),
                to: "c".into(),
            },
            DagEdge {
                from: "b".into(),
                to: "d".into(),
            },
            DagEdge {
                from: "c".into(),
                to: "d".into(),
            },
        ];
        let order = topological_sort(&nodes, &edges).unwrap();
        assert_eq!(order[0], "a");
        assert_eq!(order[3], "d");
        assert!(order.contains(&"b".to_string()));
        assert!(order.contains(&"c".to_string()));
    }

    #[test]
    /// 验证能检测到环。
    fn test_cycle_detection() {
        let nodes = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let edges = vec![
            DagEdge {
                from: "a".into(),
                to: "b".into(),
            },
            DagEdge {
                from: "b".into(),
                to: "c".into(),
            },
            DagEdge {
                from: "c".into(),
                to: "a".into(),
            },
        ];
        assert!(detect_cycle(&nodes, &edges).is_some());
    }

    #[test]
    /// 验证无环图不会误报。
    fn test_no_cycle() {
        let nodes = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let edges = vec![
            DagEdge {
                from: "a".into(),
                to: "b".into(),
            },
            DagEdge {
                from: "a".into(),
                to: "c".into(),
            },
        ];
        assert!(detect_cycle(&nodes, &edges).is_none());
    }

    #[tokio::test]
    /// 验证节点失败时会中止并清理同批次的兄弟任务。
    async fn failed_dag_aborts_and_drains_sibling_tasks() {
        let mut workflow = DagWorkflow::builder()
            .node(
                "fail",
                TestAgent {
                    name: "fail",
                    fail: true,
                    delay: Duration::from_millis(1),
                },
            )
            .node(
                "slow",
                TestAgent {
                    name: "slow",
                    fail: false,
                    delay: Duration::from_secs(60),
                },
            )
            .build()
            .unwrap();

        let result = tokio::time::timeout(Duration::from_secs(1), workflow.run("input"))
            .await
            .unwrap();
        assert!(result.is_err());
    }

    #[tokio::test]
    /// 验证成功执行时步骤按拓扑顺序而非完成顺序记录。
    async fn successful_dag_preserves_topological_step_order() -> Result<()> {
        let mut workflow = DagWorkflow::builder()
            .node(
                "registered-first",
                TestAgent {
                    name: "registered-first",
                    fail: false,
                    delay: Duration::from_millis(30),
                },
            )
            .node(
                "completed-first",
                TestAgent {
                    name: "completed-first",
                    fail: false,
                    delay: Duration::from_millis(1),
                },
            )
            .build()?;

        let output = workflow.run("input").await?;
        assert_eq!(
            output
                .steps
                .iter()
                .map(|step| step.agent_name.as_str())
                .collect::<Vec<_>>(),
            vec!["registered-first", "completed-first"]
        );
        assert_eq!(output.result, "registered-first\n\ncompleted-first");
        Ok(())
    }
}
