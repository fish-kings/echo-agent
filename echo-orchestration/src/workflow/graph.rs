/*!
图工作流引擎

将 Agent 的执行建模为一张**有向图**：

- **Node（节点）**：执行单元（Agent / 函数 / 路由）
- **Edge（边）**：节点间的转移逻辑（固定 / 条件 / 并行扇出 + 扇入）
- **State（SharedState，共享状态）**：跨节点共享的 KV 存储 + 消息历史

## 与 LangGraph 的对照

| LangGraph | echo-agent workflow |
|-----------|---------------------|
| `StateGraph` | [`Graph`] |
| `add_node()` | [`GraphBuilder::add_node()`] |
| `add_edge()` | [`GraphBuilder::add_edge()`] |
| `add_conditional_edges()` | [`GraphBuilder::add_conditional_edge()`] |
| `END` | [`Graph::END`] |
| `compile()` | [`GraphBuilder::build()`] |
| `invoke()` | [`Graph::run()`] |

## 示例

```rust,no_run
use echo_core::error::Result;
use echo_orchestration::workflow::{GraphBuilder, SharedState};

# async fn example() -> Result<()> {
let graph = GraphBuilder::new("my_workflow")
    .add_function_node("start", |state| Box::pin(async move {
        let _ = state.set("greeting", "Hello, World!");
        Ok(())
    }))
    .add_function_node("end", |state| Box::pin(async move {
        let msg: String = state.get("greeting").unwrap_or_default();
        println!("{msg}");
        Ok(())
    }))
    .set_entry("start")
    .add_edge("start", "end")
    .set_finish("end")
    .build()?;

let state = SharedState::new();
let result = graph.run(state).await?;
# Ok(())
# }
```
*/

use super::WorkflowEvent;
use super::checkpoint_store::{
    Checkpoint, CheckpointStore, InterruptType, MemoryCheckpointStore, WorkflowContinuation,
};
use super::node::Node;
use super::state::SharedState;
use crate::human_loop::ApprovalDecision;
use echo_core::agent::Agent;
use echo_core::error::{AgentError, ReactError, Result};
use futures::StreamExt;
use futures::future::BoxFuture;
use futures::stream::{BoxStream, FuturesUnordered};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc::UnboundedSender;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};

// ── Edge ────────────────────────────────────────────────────────────────────

/// 边的路由逻辑
pub(crate) enum EdgeKind {
    /// 固定转移：A -> B
    Fixed(String),
    /// 条件转移：根据状态返回下一个节点名
    Conditional(Box<dyn ConditionFn>),
    /// 并行扇出：同时进入多个节点；全部完成后汇合到 then 节点
    Parallel {
        /// 并行执行的目标节点列表
        targets: Vec<String>,
        /// 所有目标完成后进入的汇合节点
        then: String,
    },
}

/// 条件函数 trait（对象安全）
pub(crate) trait ConditionFn: Send + Sync {
    /// 基于共享状态求值，返回下一个节点名
    fn evaluate<'a>(&'a self, state: &'a SharedState) -> BoxFuture<'a, String>;
}

/// 将闭包包装为 [`ConditionFn`] 的适配器
struct CondWrapper<F>(F);

impl<F> ConditionFn for CondWrapper<F>
where
    F: for<'a> Fn(&'a SharedState) -> BoxFuture<'a, String> + Send + Sync,
{
    fn evaluate<'a>(&'a self, state: &'a SharedState) -> BoxFuture<'a, String> {
        (self.0)(state)
    }
}

/// 从某个节点出发的边
pub(crate) struct Edge {
    /// 起始节点名
    pub from: String,
    /// 边的路由类型
    pub kind: EdgeKind,
}

// ── 中断配置 ────────────────────────────────────────────────────

/// 中断配置
#[derive(Debug, Clone, Default)]
pub struct InterruptConfig {
    /// 在进入这些节点之前暂停
    pub before: Vec<String>,
    /// 在这些节点执行之后暂停
    pub after: Vec<String>,
}

impl InterruptConfig {
    /// 创建空的中断配置
    pub fn new() -> Self {
        Self::default()
    }

    /// 检查进入该节点前是否需要暂停
    pub fn should_interrupt_before(&self, node_name: &str) -> bool {
        self.before.iter().any(|n| n == node_name || n == "*")
    }

    /// 检查该节点执行后是否需要暂停
    pub fn should_interrupt_after(&self, node_name: &str) -> bool {
        self.after.iter().any(|n| n == node_name || n == "*")
    }

    /// 检查是否配置了任何中断
    pub fn has_interrupts(&self) -> bool {
        !self.before.is_empty() || !self.after.is_empty()
    }
}

// ── 中断状态 ────────────────────────────────────────────────────────────

/// 中断状态——执行暂停时的状态
#[derive(Debug)]
pub struct InterruptState {
    /// 检查点（用于恢复）
    pub checkpoint: Checkpoint,
    /// 中断类型
    pub interrupt_type: InterruptType,
    /// 被暂停的节点名
    pub pending_node: String,
    /// 给用户的提示信息
    pub prompt: String,
}

impl InterruptState {
    /// 创建 BeforeNode（节点前）中断
    pub fn before_node(checkpoint: Checkpoint, node_name: String) -> Self {
        let prompt = format!(
            "Node '{}' requires confirmation before execution",
            node_name
        );
        Self {
            checkpoint,
            interrupt_type: InterruptType::BeforeNode,
            pending_node: node_name,
            prompt,
        }
    }

    /// 创建 AfterNode（节点后）中断
    pub fn after_node(checkpoint: Checkpoint, node_name: String) -> Self {
        let prompt = format!("Node '{}' requires confirmation after execution", node_name);
        Self {
            checkpoint,
            interrupt_type: InterruptType::AfterNode,
            pending_node: node_name,
            prompt,
        }
    }
}

/// run_until_interrupt 的返回类型
#[derive(Debug)]
// 公共结果变体保持不装箱，以维持工作流 API 不变。
#[allow(clippy::large_enum_variant)]
pub enum RunUntilInterruptResult {
    /// 执行完成
    Completed(GraphResult),
    /// 在中断点暂停
    Interrupted(InterruptState),
    /// 审批被延后；检查点仍可被认领。
    Deferred(InterruptState),
    /// 审批被拒绝，检查点已被消费。
    Rejected {
        /// 拒绝时的共享状态
        state: SharedState,
        /// 已执行的节点路径
        path: Vec<String>,
        /// 已执行的步数
        steps: usize,
        /// 拒绝原因（可选）
        reason: Option<String>,
    },
}

// ── GraphBuilder ────────────────────────────────────────────────────────────

/**
图工作流构建器

通过链式调用添加节点和边；最后由 `build()` 生成不可变的 [`Graph`]。
*/
pub struct GraphBuilder {
    /// 图名称
    name: String,
    /// 已注册的节点（节点名 -> 节点）
    nodes: HashMap<String, Node>,
    /// 已添加的边
    edges: Vec<Edge>,
    /// 入口节点名
    entry_node: Option<String>,
    /// 结束节点名列表
    finish_nodes: Vec<String>,
    /// 中断配置
    interrupt_config: InterruptConfig,
    /// 重复注册的节点名（build 时报错）
    duplicate_nodes: HashSet<String>,
    /// 显式指定的图版本号
    explicit_revision: Option<String>,
    /// 单节点执行超时时间
    node_timeout: Option<Duration>,
}

impl GraphBuilder {
    /// 创建空的图构建器
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            nodes: HashMap::new(),
            edges: Vec::new(),
            entry_node: None,
            finish_nodes: Vec::new(),
            interrupt_config: InterruptConfig::default(),
            duplicate_nodes: HashSet::new(),
            explicit_revision: None,
            node_timeout: None,
        }
    }

    // ── 添加节点 ─────────────────────────────────────────────────────────

    /**
    添加一个 Agent 节点

    `input_key`：从状态中读取提示词的键
    `output_key`：将执行结果写入状态的键

    默认使用 `execute` 模式（多轮、可调用工具）。
    */
    pub fn add_agent_node(
        mut self,
        name: impl Into<String>,
        agent: impl Agent + 'static,
        input_key: impl Into<String>,
        output_key: impl Into<String>,
    ) -> Self {
        let name = name.into();
        if self.nodes.contains_key(&name) {
            self.duplicate_nodes.insert(name.clone());
        }
        self.nodes.insert(
            name.clone(),
            Node::agent(&name, agent, input_key, output_key),
        );
        self
    }

    /**
    添加一个 Agent 节点（可配置 execute/chat 模式）

    `use_execute`：true 使用 execute（多轮、可调用工具），
    false 使用 chat（单轮）。
    */
    pub fn add_agent_node_with_mode(
        mut self,
        name: impl Into<String>,
        agent: impl Agent + 'static,
        input_key: impl Into<String>,
        output_key: impl Into<String>,
        use_execute: bool,
    ) -> Self {
        let name = name.into();
        if self.nodes.contains_key(&name) {
            self.duplicate_nodes.insert(name.clone());
        }
        self.nodes.insert(
            name.clone(),
            Node::agent_with_mode(&name, agent, input_key, output_key, use_execute),
        );
        self
    }

    /// 添加一个共享 Agent 节点（`Arc<dyn Agent>`）。
    pub fn add_shared_agent_node(
        mut self,
        name: impl Into<String>,
        agent: Arc<dyn Agent>,
        input_key: impl Into<String>,
        output_key: impl Into<String>,
    ) -> Self {
        let name = name.into();
        if self.nodes.contains_key(&name) {
            self.duplicate_nodes.insert(name.clone());
        }
        self.nodes.insert(
            name.clone(),
            Node::agent_shared(&name, agent, input_key, output_key),
        );
        self
    }

    /// 添加一个共享 Agent 节点（可配置 execute/chat 模式）
    pub fn add_shared_agent_node_with_mode(
        mut self,
        name: impl Into<String>,
        agent: Arc<dyn Agent>,
        input_key: impl Into<String>,
        output_key: impl Into<String>,
        use_execute: bool,
    ) -> Self {
        let name = name.into();
        if self.nodes.contains_key(&name) {
            self.duplicate_nodes.insert(name.clone());
        }
        self.nodes.insert(
            name.clone(),
            Node::agent_shared_with_mode(&name, agent, input_key, output_key, use_execute),
        );
        self
    }

    /// 添加一个函数节点
    pub fn add_function_node<F>(mut self, name: impl Into<String>, f: F) -> Self
    where
        F: for<'a> Fn(&'a SharedState) -> BoxFuture<'a, Result<()>> + Send + Sync + 'static,
    {
        let name = name.into();
        if self.nodes.contains_key(&name) {
            self.duplicate_nodes.insert(name.clone());
        }
        self.nodes.insert(name.clone(), Node::function(&name, f));
        self
    }

    /// 添加一个路由节点（无执行逻辑，仅用作条件分支的汇合点）
    pub fn add_router_node(mut self, name: impl Into<String>) -> Self {
        let name = name.into();
        if self.nodes.contains_key(&name) {
            self.duplicate_nodes.insert(name.clone());
        }
        self.nodes.insert(name.clone(), Node::passthrough(&name));
        self
    }

    /**
    添加一个子图节点——将预先构建好的 `Graph` 作为本图中的一个节点嵌入。

    子图与父图共享 `SharedState`，其执行步数计入父图的步数上限。
    */
    pub fn add_subgraph_node(mut self, name: impl Into<String>, subgraph: Graph) -> Self {
        let name = name.into();
        if self.nodes.contains_key(&name) {
            self.duplicate_nodes.insert(name.clone());
        }
        self.nodes
            .insert(name.clone(), Node::subgraph(&name, subgraph));
        self
    }

    // ── 添加边 ─────────────────────────────────────────────────────────

    /// 添加一条固定边：from -> to
    pub fn add_edge(mut self, from: impl Into<String>, to: impl Into<String>) -> Self {
        self.edges.push(Edge {
            from: from.into(),
            kind: EdgeKind::Fixed(to.into()),
        });
        self
    }

    /**
    添加一条条件边：from -> f(state) 返回目标节点名

    条件函数返回的字符串必须是已注册的节点名或 `"__end__"`。
    */
    pub fn add_conditional_edge<F>(mut self, from: impl Into<String>, f: F) -> Self
    where
        F: for<'a> Fn(&'a SharedState) -> BoxFuture<'a, String> + Send + Sync + 'static,
    {
        self.edges.push(Edge {
            from: from.into(),
            kind: EdgeKind::Conditional(Box::new(CondWrapper(f))),
        });
        self
    }

    /**
    添加一条并行边：from -> [targets...] -> then

    `from` 完成后，`targets` 中的所有节点**并行**执行；全部完成后进入 `then`。
    并行节点对状态的修改会在进入 then 节点之前合并。

    **注意：** 目前并行分支在同一个异步任务内**顺序**执行（而非通过 `tokio::spawn`），
    因为 `dyn Agent` 节点不满足 `Send + 'static`。每个分支通过 `SharedState::fork()`
    获得隔离的状态副本，结果再通过 `deep_merge()` 合并回来。
    待 `Node` 类型支持 `Send + 'static` 约束后，将启用真正的并发执行。
    */
    pub fn add_parallel_edge(
        mut self,
        from: impl Into<String>,
        targets: Vec<String>,
        then: impl Into<String>,
    ) -> Self {
        self.edges.push(Edge {
            from: from.into(),
            kind: EdgeKind::Parallel {
                targets,
                then: then.into(),
            },
        });
        self
    }

    // ── 入口与结束 ──────────────────────────────────────────────────

    /// 设置入口节点
    pub fn set_entry(mut self, name: impl Into<String>) -> Self {
        self.entry_node = Some(name.into());
        self
    }

    /// 设置结束节点（到达该节点时图执行结束；可设置多个）
    pub fn set_finish(mut self, name: impl Into<String>) -> Self {
        self.finish_nodes.push(name.into());
        self
    }

    // ── 中断配置 ──────────────────────────────────────────────────

    /**
    设置 interrupt_before（进入节点前暂停）

    支持 "*" 通配符表示所有节点。

    # 示例

    ```rust
    use echo_orchestration::workflow::GraphBuilder;

    let graph = GraphBuilder::new("my_flow")
        .add_function_node("step1", |_| Box::pin(async { Ok(()) }))
        .add_function_node("step2", |_| Box::pin(async { Ok(()) }))
        .set_entry("step1")
        .add_edge("step1", "step2")
        .interrupt_before(vec!["step2"])  // Pause before entering step2
        .build()
        .unwrap();
    ```
    */
    pub fn interrupt_before(mut self, nodes: Vec<&str>) -> Self {
        self.interrupt_config.before = nodes.into_iter().map(String::from).collect();
        self
    }

    /**
    设置 interrupt_after（节点执行后暂停）

    支持 "*" 通配符表示所有节点。
    */
    pub fn interrupt_after(mut self, nodes: Vec<&str>) -> Self {
        self.interrupt_config.after = nodes.into_iter().map(String::from).collect();
        self
    }

    /// 当节点行为变化但拓扑未变时，显式设置版本号。
    pub fn revision(mut self, revision: impl Into<String>) -> Self {
        self.explicit_revision = Some(revision.into());
        self
    }

    /// 为每次节点执行（包括扇出分支）设置超时期限。
    pub fn node_timeout(mut self, timeout: Duration) -> Self {
        self.node_timeout = Some(timeout);
        self
    }

    /// 构建不可变的 Graph
    pub fn build(self) -> Result<Graph> {
        if !self.duplicate_nodes.is_empty() {
            let mut duplicates = self.duplicate_nodes.into_iter().collect::<Vec<_>>();
            duplicates.sort();
            return Err(ReactError::Agent(Box::new(
                AgentError::InitializationFailed(format!(
                    "Graph contains duplicate node IDs: {}",
                    duplicates.join(", ")
                )),
            )));
        }
        let entry = self.entry_node.ok_or_else(|| {
            ReactError::Agent(Box::new(AgentError::InitializationFailed(
                "Graph must have an entry node (call set_entry())".to_string(),
            )))
        })?;

        if !self.nodes.contains_key(&entry) {
            return Err(ReactError::Agent(Box::new(
                AgentError::InitializationFailed(format!(
                    "Entry node '{}' not found in graph",
                    entry
                )),
            )));
        }

        for finish in &self.finish_nodes {
            if !self.nodes.contains_key(finish) {
                return Err(ReactError::Agent(Box::new(
                    AgentError::InitializationFailed(format!(
                        "Finish node '{finish}' not found in graph"
                    )),
                )));
            }
        }
        for interrupt in self
            .interrupt_config
            .before
            .iter()
            .chain(self.interrupt_config.after.iter())
        {
            if interrupt != "*" && !self.nodes.contains_key(interrupt) {
                return Err(ReactError::Agent(Box::new(
                    AgentError::InitializationFailed(format!(
                        "Interrupt node '{interrupt}' not found in graph"
                    )),
                )));
            }
        }

        // 校验边引用的所有节点均存在
        for edge in &self.edges {
            if !self.nodes.contains_key(&edge.from) {
                return Err(ReactError::Agent(Box::new(
                    AgentError::InitializationFailed(format!(
                        "Edge from unknown node '{}'",
                        edge.from
                    )),
                )));
            }
            match &edge.kind {
                EdgeKind::Fixed(to) if to != Graph::END && !self.nodes.contains_key(to) => {
                    return Err(ReactError::Agent(Box::new(
                        AgentError::InitializationFailed(format!("Edge to unknown node '{}'", to)),
                    )));
                }
                EdgeKind::Fixed(_) => {}
                EdgeKind::Parallel { targets, then } => {
                    if targets.is_empty() {
                        return Err(ReactError::Agent(Box::new(
                            AgentError::InitializationFailed(
                                "Parallel edge must contain at least one target".to_string(),
                            ),
                        )));
                    }
                    if targets.iter().collect::<HashSet<_>>().len() != targets.len() {
                        return Err(ReactError::Agent(Box::new(
                            AgentError::InitializationFailed(
                                "Parallel edge contains duplicate targets".to_string(),
                            ),
                        )));
                    }
                    for t in targets {
                        if !self.nodes.contains_key(t) {
                            return Err(ReactError::Agent(Box::new(
                                AgentError::InitializationFailed(format!(
                                    "Parallel target node '{}' not found",
                                    t
                                )),
                            )));
                        }
                    }
                    if then != Graph::END && !self.nodes.contains_key(then) {
                        return Err(ReactError::Agent(Box::new(
                            AgentError::InitializationFailed(format!(
                                "Parallel 'then' node '{}' not found",
                                then
                            )),
                        )));
                    }
                }
                _ => {}
            }
        }

        // 构建邻接表并校验多出边情况
        let mut edge_map: HashMap<String, Vec<Edge>> = HashMap::new();
        for edge in self.edges {
            let from = edge.from.clone();
            let entry = edge_map.entry(from.clone()).or_default();
            /* 禁止同一节点有多条普通出边（Fixed / Conditional），
            因为 resolve_next() 只取第一条边，其余边会被静默丢弃。 */
            if !entry.is_empty() {
                return Err(ReactError::Agent(Box::new(
                    AgentError::InitializationFailed(format!(
                        "Node '{}' has multiple outgoing edges; only one edge per node is supported \
                         (use a Conditional edge for branching, or a Parallel edge for fan-out)",
                        from,
                    )),
                )));
            }
            entry.push(edge);
        }

        let graph_revision = self.explicit_revision.unwrap_or_else(|| {
            let mut nodes = self.nodes.keys().cloned().collect::<Vec<_>>();
            nodes.sort();
            let mut topology = format!("{}|{}|{}", self.name, entry, nodes.join(","));
            let mut sources = edge_map.keys().cloned().collect::<Vec<_>>();
            sources.sort();
            for source in sources {
                if let Some(edges) = edge_map.get(&source) {
                    for edge in edges {
                        let target = match &edge.kind {
                            EdgeKind::Fixed(to) => format!("fixed:{to}"),
                            EdgeKind::Conditional(_) => "conditional".to_string(),
                            EdgeKind::Parallel { targets, then } => {
                                format!("fanout:{}:{then}", targets.join(","))
                            }
                        };
                        topology.push_str(&format!("|{source}>{target}"));
                    }
                }
            }
            format!("{:x}", Sha256::digest(topology.as_bytes()))
        });

        Ok(Graph {
            name: self.name,
            nodes: self.nodes,
            edges: edge_map,
            entry,
            finish_nodes: self.finish_nodes,
            max_steps: 100,
            interrupt_config: self.interrupt_config,
            checkpoint_store: Arc::new(MemoryCheckpointStore::new()),
            cancel_token: None,
            graph_revision,
            node_timeout: self.node_timeout,
        })
    }

    // ── 便捷方法 ───────────────────────────────────────────────

    /// 添加 ReactAgent 节点的便捷方法（默认 input="task"，output="result"）
    pub fn add_react_node(self, name: impl Into<String>, agent: impl Agent + 'static) -> Self {
        self.add_agent_node(name, agent, "task", "result")
    }
}

// ── Graph ───────────────────────────────────────────────────────────────────

/**
编译后的图工作流（不可变）

通过 [`GraphBuilder`] 构建，调用 [`run()`](Graph::run) 执行。
*/
pub struct Graph {
    /// 图名称
    pub name: String,
    /// 节点注册表
    nodes: HashMap<String, Node>,
    /// 邻接表：from -> [Edge]
    edges: HashMap<String, Vec<Edge>>,
    /// 入口节点
    entry: String,
    /// 结束节点列表
    finish_nodes: Vec<String>,
    /// 最大执行步数（防止无限循环）
    max_steps: usize,
    /// 中断配置
    interrupt_config: InterruptConfig,
    /// 检查点存储
    checkpoint_store: Arc<dyn CheckpointStore>,
    /// 可选的取消令牌，用于在节点之间进行协作式取消。
    cancel_token: Option<CancellationToken>,
    /// 本图定义的稳定哈希或调用方提供的版本号。
    graph_revision: String,
    /// 可选的单节点执行期限。
    node_timeout: Option<Duration>,
}

/// 图执行结果
#[derive(Debug)]
pub struct GraphResult {
    /// 最终状态
    pub state: SharedState,
    /// 执行路径（节点名序列）
    pub path: Vec<String>,
    /// 总步数
    pub steps: usize,
}

/// 一次图执行过程中的可变执行上下文
struct GraphExecution {
    /// 共享状态
    state: SharedState,
    /// 当前待执行的节点名
    current: String,
    /// 已执行的节点路径
    path: Vec<String>,
    /// 已执行的步数
    step_count: usize,
    /// 下一步的续接方式（普通节点 / 扇出 / 结束）
    continuation: WorkflowContinuation,
    /// 恢复执行时所基于的父检查点
    parent_checkpoint: Option<Checkpoint>,
    /// 是否跳过一次 interrupt_before 检查（恢复时使用）
    skip_interrupt_before_once: bool,
}

impl Graph {
    /// 向事件通道发送工作流事件；未提供通道时直接忽略
    fn emit_event(
        events: Option<&UnboundedSender<WorkflowEvent>>,
        event: WorkflowEvent,
    ) -> Result<()> {
        let Some(events) = events else {
            return Ok(());
        };
        events
            .send(event)
            .map_err(|_| ReactError::Other("Workflow event stream closed".to_string()))
    }

    /// 执行单个节点，转发其流式 token 事件，并应用超时与取消控制
    async fn execute_node(
        &self,
        node_name: &str,
        node: &Node,
        state: &SharedState,
        events: Option<&UnboundedSender<WorkflowEvent>>,
    ) -> Result<()> {
        let execution = async {
            let node_cancel = CancellationToken::new();
            let cancel_guard = node_cancel.clone().drop_guard();
            let mut tokens = node.execute_stream(state, node_cancel);
            while let Some(token) = tokens.next().await {
                let token = token?;
                if let Some(events) = events {
                    Self::emit_event(
                        Some(events),
                        WorkflowEvent::Token {
                            node_name: node_name.to_string(),
                            token,
                        },
                    )?;
                }
            }
            let _ = cancel_guard.disarm();
            Ok(())
        };
        let timed = async {
            if let Some(timeout) = self.node_timeout {
                tokio::time::timeout(timeout, execution)
                    .await
                    .map_err(|_| {
                        ReactError::Other(format!("Workflow node timed out after {timeout:?}"))
                    })?
            } else {
                execution.await
            }
        };
        if let Some(cancel) = &self.cancel_token {
            tokio::select! {
                result = timed => result,
                () = cancel.cancelled() => Err(ReactError::Agent(Box::new(AgentError::Cancelled(
                    format!("Graph '{}' node execution cancelled", self.name),
                )))),
            }
        } else {
            timed.await
        }
    }

    /// 校验检查点是否属于本图（名称、版本、运行标识、节点与扇出续接均合法）
    fn validate_checkpoint(&self, checkpoint: &Checkpoint) -> Result<()> {
        if checkpoint.graph_name != self.name {
            return Err(ReactError::Other(format!(
                "Checkpoint belongs to graph '{}', not '{}'",
                checkpoint.graph_name, self.name
            )));
        }
        if checkpoint.graph_revision != self.graph_revision {
            return Err(ReactError::Other(format!(
                "Checkpoint graph revision '{}' does not match '{}'",
                checkpoint.graph_revision, self.graph_revision
            )));
        }
        if checkpoint.workflow_run_id.is_empty() {
            return Err(ReactError::Other(
                "Checkpoint has no workflow run identity".to_string(),
            ));
        }
        if checkpoint.current_node != Self::END
            && !self.nodes.contains_key(&checkpoint.current_node)
        {
            return Err(ReactError::Other(format!(
                "Checkpoint references unknown node '{}'",
                checkpoint.current_node
            )));
        }
        if let WorkflowContinuation::FanOut { targets, then } = &checkpoint.continuation
            && (targets.is_empty()
                || targets
                    .iter()
                    .any(|target| !self.nodes.contains_key(target))
                || (then != Self::END && !self.nodes.contains_key(then)))
        {
            return Err(ReactError::Other(
                "Checkpoint contains an invalid fan-out continuation".to_string(),
            ));
        }
        Ok(())
    }

    /// 根据检查点的中断类型构造对应的中断状态
    fn interrupt_state(checkpoint: Checkpoint) -> InterruptState {
        let node = checkpoint.current_node.clone();
        match checkpoint.interrupt_type {
            InterruptType::BeforeNode => InterruptState::before_node(checkpoint, node),
            InterruptType::AfterNode => InterruptState::after_node(checkpoint, node),
        }
    }

    /// 创建绑定本图版本与续接信息的检查点
    fn checkpoint(
        &self,
        current_node: String,
        state: &SharedState,
        path: Vec<String>,
        step_count: usize,
        interrupt_type: InterruptType,
        continuation: WorkflowContinuation,
    ) -> Result<Checkpoint> {
        Checkpoint::try_new(
            self.name.clone(),
            current_node,
            state,
            path,
            step_count,
            interrupt_type,
        )
        .map(|checkpoint| checkpoint.bind_execution(self.graph_revision.clone(), continuation))
    }

    /// 执行并行扇出：各分支在隔离的状态副本上运行，任一失败则取消其余分支，全部成功后合并状态
    async fn execute_parallel_route(
        &self,
        state: &SharedState,
        targets: &[String],
        path: &mut Vec<String>,
        step_count: &mut usize,
        events: Option<&UnboundedSender<WorkflowEvent>>,
    ) -> Result<()> {
        let remaining = self.max_steps.saturating_sub(*step_count);
        if targets.len() > remaining {
            return Err(ReactError::Agent(Box::new(
                AgentError::MaxIterationsExceeded(self.max_steps),
            )));
        }
        let mut branches = Vec::with_capacity(targets.len());
        for (offset, target_name) in targets.iter().enumerate() {
            let target_node = self.nodes.get(target_name).ok_or_else(|| {
                ReactError::Agent(Box::new(AgentError::InitializationFailed(format!(
                    "Parallel node '{target_name}' not found in graph '{}'",
                    self.name
                ))))
            })?;
            let branch_state = state.fork()?;
            branch_state.set_current_node(target_name);
            branches.push((target_name.clone(), target_node, branch_state));
            Self::emit_event(
                events,
                WorkflowEvent::NodeStart {
                    node_name: target_name.clone(),
                    step_index: step_count.saturating_add(offset),
                },
            )?;
        }
        let route_cancel = CancellationToken::new();
        let mut branch_results = vec![false; branches.len()];
        let mut pending = FuturesUnordered::new();
        for (index, (target_name, node, branch_state)) in branches.iter().enumerate() {
            let branch_cancel = route_cancel.clone();
            let target_name = target_name.clone();
            pending.push(async move {
                let started = Instant::now();
                let result = tokio::select! {
                    result = self.execute_node(&target_name, node, branch_state, events) => result,
                    () = branch_cancel.cancelled() => Err(ReactError::Agent(Box::new(
                        AgentError::Cancelled("parallel sibling cancelled after branch failure".to_string()),
                    ))),
                };
                (index, target_name, started.elapsed(), result)
            });
        }
        while let Some((index, target_name, elapsed, result)) = pending.next().await {
            match result {
                Ok(()) => {
                    let completed = branch_results.get_mut(index).ok_or_else(|| {
                        ReactError::Other(
                            "Parallel branch index escaped its result set".to_string(),
                        )
                    })?;
                    *completed = true;
                    Self::emit_event(
                        events,
                        WorkflowEvent::NodeEnd {
                            node_name: target_name,
                            step_index: step_count.saturating_add(index),
                            elapsed,
                        },
                    )?;
                }
                Err(error) => {
                    let event_result = Self::emit_event(
                        events,
                        WorkflowEvent::NodeError {
                            node_name: target_name,
                            error: error.to_string(),
                        },
                    );
                    route_cancel.cancel();
                    let _ = tokio::time::timeout(super::WORKFLOW_TASK_DRAIN_TIMEOUT, async {
                        while pending.next().await.is_some() {}
                    })
                    .await;
                    event_result?;
                    return Err(error);
                }
            }
        }
        drop(pending);
        if self.is_cancelled() {
            return Err(ReactError::Agent(Box::new(AgentError::Cancelled(format!(
                "Graph '{}' cancelled during parallel fan-out",
                self.name
            )))));
        }
        for ((target_name, _, branch_state), completed) in branches.into_iter().zip(branch_results)
        {
            if !completed {
                return Err(ReactError::Agent(Box::new(AgentError::Cancelled(
                    "parallel branch did not settle".to_string(),
                ))));
            }
            state.deep_merge(&branch_state)?;
            path.push(target_name);
            *step_count = step_count.saturating_add(1);
        }
        Ok(())
    }

    /// 终止标记节点名
    pub const END: &'static str = "__end__";

    /// 设置最大执行步数
    pub fn set_max_steps(&mut self, max: usize) {
        self.max_steps = max;
    }

    /**
    附加一个取消令牌，用于在节点之间进行协作式取消。

    令牌被取消后，图会在下一个节点边界停止并返回 `AgentError::Cancelled` 错误，
    从而使长时间运行的工作流能够在节点执行之间被优雅地取消。
    */
    pub fn with_cancel_token(mut self, token: CancellationToken) -> Self {
        self.cancel_token = Some(token);
        self
    }

    /// 检查取消令牌是否已被触发。
    fn is_cancelled(&self) -> bool {
        self.cancel_token
            .as_ref()
            .map(|t| t.is_cancelled())
            .unwrap_or(false)
    }

    /// 从入口节点开始创建一个全新的执行上下文
    fn fresh_execution(&self, state: SharedState) -> GraphExecution {
        GraphExecution {
            state,
            current: self.entry.clone(),
            path: Vec::new(),
            step_count: 0,
            continuation: WorkflowContinuation::Node,
            parent_checkpoint: None,
            skip_interrupt_before_once: false,
        }
    }

    /// 完成执行：记录日志、发送 Completed 事件并返回最终结果
    async fn complete_execution(
        &self,
        execution: GraphExecution,
        workflow_start: Instant,
        events: Option<&UnboundedSender<WorkflowEvent>>,
    ) -> Result<RunUntilInterruptResult> {
        info!(
            graph = %self.name,
            steps = execution.step_count,
            path = ?execution.path,
            "Graph execution completed"
        );
        let final_result = execution
            .state
            .get::<String>("result")
            .or_else(|| execution.state.get::<String>("output"))
            .unwrap_or_default();
        Self::emit_event(
            events,
            WorkflowEvent::Completed {
                result: final_result,
                total_steps: execution.step_count,
                elapsed: workflow_start.elapsed(),
            },
        )?;
        Ok(RunUntilInterruptResult::Completed(GraphResult {
            state: execution.state,
            path: execution.path,
            steps: execution.step_count,
        }))
    }

    /// 图执行的唯一权威实现，供 run、中断、恢复与流式等入口共用。
    async fn execute_loop(
        &self,
        mut execution: GraphExecution,
        honor_interrupts: bool,
        events: Option<UnboundedSender<WorkflowEvent>>,
    ) -> Result<RunUntilInterruptResult> {
        let workflow_start = Instant::now();
        info!(graph = %self.name, entry = %execution.current, "Starting graph execution");

        loop {
            if self.is_cancelled() {
                warn!(graph = %self.name, steps = execution.step_count, "Graph execution cancelled");
                return Err(ReactError::Agent(Box::new(AgentError::Cancelled(format!(
                    "Graph '{}' cancelled after {} steps",
                    self.name, execution.step_count
                )))));
            }

            let continuation =
                std::mem::replace(&mut execution.continuation, WorkflowContinuation::Node);
            match continuation {
                WorkflowContinuation::Node => {}
                WorkflowContinuation::FanOut { targets, then } => {
                    debug!(
                        graph = %self.name,
                        targets = ?targets,
                        then = %then,
                        "Executing parallel fan-out"
                    );
                    self.execute_parallel_route(
                        &execution.state,
                        &targets,
                        &mut execution.path,
                        &mut execution.step_count,
                        events.as_ref(),
                    )
                    .await?;
                    execution.current = then;
                    continue;
                }
                WorkflowContinuation::End => {
                    execution.current = Self::END.to_string();
                }
            }

            if execution.current == Self::END {
                return self
                    .complete_execution(execution, workflow_start, events.as_ref())
                    .await;
            }

            if execution.step_count >= self.max_steps {
                warn!(
                    graph = %self.name,
                    steps = execution.step_count,
                    "Graph execution exceeded max steps"
                );
                return Err(ReactError::Agent(Box::new(
                    AgentError::MaxIterationsExceeded(self.max_steps),
                )));
            }

            if honor_interrupts
                && !execution.skip_interrupt_before_once
                && self
                    .interrupt_config
                    .should_interrupt_before(&execution.current)
            {
                debug!(graph = %self.name, node = %execution.current, "Interrupt before node");
                let checkpoint = self.checkpoint(
                    execution.current.clone(),
                    &execution.state,
                    execution.path.clone(),
                    execution.step_count,
                    InterruptType::BeforeNode,
                    WorkflowContinuation::Node,
                )?;
                let checkpoint = match execution.parent_checkpoint.as_ref() {
                    Some(parent) => checkpoint.continue_run(parent),
                    None => checkpoint,
                };
                self.checkpoint_store.save(&checkpoint).await?;
                return Ok(RunUntilInterruptResult::Interrupted(
                    InterruptState::before_node(checkpoint, execution.current),
                ));
            }
            execution.skip_interrupt_before_once = false;

            let node = self.nodes.get(&execution.current).ok_or_else(|| {
                ReactError::Agent(Box::new(AgentError::InitializationFailed(format!(
                    "Node '{}' not found in graph '{}'",
                    execution.current, self.name
                ))))
            })?;
            execution.state.set_current_node(&execution.current);
            Self::emit_event(
                events.as_ref(),
                WorkflowEvent::NodeStart {
                    node_name: execution.current.clone(),
                    step_index: execution.step_count,
                },
            )?;
            let node_start = Instant::now();
            if let Err(error) = self
                .execute_node(&execution.current, node, &execution.state, events.as_ref())
                .await
            {
                Self::emit_event(
                    events.as_ref(),
                    WorkflowEvent::NodeError {
                        node_name: execution.current.clone(),
                        error: error.to_string(),
                    },
                )?;
                return Err(error);
            }
            Self::emit_event(
                events.as_ref(),
                WorkflowEvent::NodeEnd {
                    node_name: execution.current.clone(),
                    step_index: execution.step_count,
                    elapsed: node_start.elapsed(),
                },
            )?;
            execution.path.push(execution.current.clone());
            execution.step_count = execution.step_count.saturating_add(1);

            let next = if self.finish_nodes.contains(&execution.current) {
                NextStep::End
            } else {
                self.resolve_next(&execution.current, &execution.state)
                    .await?
            };

            if honor_interrupts
                && self
                    .interrupt_config
                    .should_interrupt_after(&execution.current)
            {
                debug!(graph = %self.name, node = %execution.current, "Interrupt after node");
                let (next_node, continuation) = match next {
                    NextStep::Single(name) => (name, WorkflowContinuation::Node),
                    NextStep::Parallel { targets, then } => {
                        (then.clone(), WorkflowContinuation::FanOut { targets, then })
                    }
                    NextStep::End => (Self::END.to_string(), WorkflowContinuation::End),
                };
                let checkpoint = self.checkpoint(
                    next_node,
                    &execution.state,
                    execution.path.clone(),
                    execution.step_count,
                    InterruptType::AfterNode,
                    continuation,
                )?;
                let checkpoint = match execution.parent_checkpoint.as_ref() {
                    Some(parent) => checkpoint.continue_run(parent),
                    None => checkpoint,
                };
                self.checkpoint_store.save(&checkpoint).await?;
                return Ok(RunUntilInterruptResult::Interrupted(
                    InterruptState::after_node(checkpoint, execution.current),
                ));
            }

            match next {
                NextStep::Single(name) => execution.current = name,
                NextStep::Parallel { targets, then } => {
                    execution.current = then.clone();
                    execution.continuation = WorkflowContinuation::FanOut { targets, then };
                }
                NextStep::End => execution.current = Self::END.to_string(),
            }
        }
    }

    /**
    执行图工作流。

    从入口节点开始，按边的路由逻辑依次执行节点，直到到达结束节点或 `__end__`。
    */
    pub async fn run(&self, state: SharedState) -> Result<GraphResult> {
        match self
            .execute_loop(self.fresh_execution(state), false, None)
            .await?
        {
            RunUntilInterruptResult::Completed(result) => Ok(result),
            _ => Err(ReactError::Other(
                "Non-interrupting graph execution returned a non-terminal outcome".to_string(),
            )),
        }
    }

    // ── 中断 + 检查点方法 ──────────────────────────────────────────

    /**
    执行直到遇到中断点，然后暂停

    若配置了 `interrupt_before` 或 `interrupt_after`，执行到对应节点时暂停并返回
    `InterruptState`。可通过 `resume()` 方法继续执行。

    # 返回值

    - `RunUntilInterruptResult::Completed(GraphResult)` - 执行完成
    - `RunUntilInterruptResult::Interrupted(InterruptState)` - 在中断点暂停
    */
    pub async fn run_until_interrupt(&self, state: SharedState) -> Result<RunUntilInterruptResult> {
        self.execute_loop(self.fresh_execution(state), true, None)
            .await
    }

    /**
    从检查点恢复执行

    用户批准继续后，从已保存的检查点恢复执行。

    修复：遇到中断点时返回 `RunUntilInterruptResult::Interrupted`，而不是 `Err`。
    */
    pub async fn resume(
        &self,
        checkpoint: Checkpoint,
        decision: ApprovalDecision,
    ) -> Result<RunUntilInterruptResult> {
        self.resume_with_updates(checkpoint, decision, None).await
    }

    /// 恢复执行的内部实现：处理延后决策，认领检查点，执行后确认或重新入队
    async fn resume_with_updates(
        &self,
        checkpoint_reference: Checkpoint,
        decision: ApprovalDecision,
        state_updates: Option<std::collections::HashMap<String, Value>>,
    ) -> Result<RunUntilInterruptResult> {
        if matches!(decision, ApprovalDecision::Deferred) {
            let checkpoint = self
                .checkpoint_store
                .load(&checkpoint_reference.id)
                .await?
                .ok_or_else(|| {
                    ReactError::Other("Checkpoint is missing or consumed".to_string())
                })?;
            self.validate_checkpoint(&checkpoint)?;
            info!(graph = %self.name, checkpoint_id = %checkpoint.id, "Resume deferred");
            return Ok(RunUntilInterruptResult::Deferred(Self::interrupt_state(
                checkpoint,
            )));
        }

        if !self.checkpoint_store.supports_claim_settlement() {
            return Err(ReactError::Other(
                "Checkpoint store does not provide atomic claim settlement".to_string(),
            ));
        }

        let persisted = self
            .checkpoint_store
            .load(&checkpoint_reference.id)
            .await?
            .ok_or_else(|| ReactError::Other("Checkpoint is missing or consumed".to_string()))?;
        self.validate_checkpoint(&persisted)?;
        let checkpoint = self
            .checkpoint_store
            .claim(&checkpoint_reference.id)
            .await?
            .ok_or_else(|| {
                ReactError::Other("Checkpoint is missing or already claimed".to_string())
            })?;
        let attempt_id = checkpoint.resume_attempt_id.clone().ok_or_else(|| {
            ReactError::Other(
                "Checkpoint store returned a claim without a resume attempt identity".to_string(),
            )
        })?;
        if let Err(error) = self.validate_checkpoint(&checkpoint) {
            if let Err(requeue_error) = self
                .checkpoint_store
                .requeue_claim(&checkpoint.id, &attempt_id)
                .await
            {
                return Err(ReactError::Other(format!(
                    "Checkpoint validation failed: {error}; checkpoint requeue failed: {requeue_error}"
                )));
            }
            return Err(error);
        }

        let result = self
            .resume_claimed_with_heartbeat(checkpoint.clone(), decision, state_updates, &attempt_id)
            .await;
        match result {
            Ok(result) => {
                self.checkpoint_store
                    .ack_claim(&checkpoint.id, &attempt_id)
                    .await?;
                Ok(result)
            }
            Err(error) => {
                if let Err(requeue_error) = self
                    .checkpoint_store
                    .requeue_claim(&checkpoint.id, &attempt_id)
                    .await
                {
                    return Err(ReactError::Other(format!(
                        "Workflow resume failed: {error}; checkpoint requeue failed: {requeue_error}"
                    )));
                }
                Err(error)
            }
        }
    }

    /// 在执行已认领检查点的同时，按存储要求的间隔续期认领（心跳）
    async fn resume_claimed_with_heartbeat(
        &self,
        checkpoint: Checkpoint,
        decision: ApprovalDecision,
        state_updates: Option<std::collections::HashMap<String, Value>>,
        attempt_id: &str,
    ) -> Result<RunUntilInterruptResult> {
        let Some(interval) = self.checkpoint_store.claim_heartbeat_interval() else {
            return self
                .resume_claimed(checkpoint, decision, state_updates)
                .await;
        };
        if interval.is_zero() {
            return Err(ReactError::Other(
                "Checkpoint claim heartbeat interval must be greater than zero".to_string(),
            ));
        }

        let checkpoint_id = checkpoint.id.clone();
        let resume = self.resume_claimed(checkpoint, decision, state_updates);
        tokio::pin!(resume);
        let first_tick = tokio::time::Instant::now()
            .checked_add(interval)
            .ok_or_else(|| ReactError::Other("Checkpoint heartbeat deadline overflow".into()))?;
        let mut heartbeat = tokio::time::interval_at(first_tick, interval);
        heartbeat.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        loop {
            tokio::select! {
                biased;
                result = &mut resume => break result,
                _ = heartbeat.tick() => {
                    if let Err(error) = self
                        .checkpoint_store
                        .renew_claim(&checkpoint_id, attempt_id)
                        .await
                    {
                        break Err(ReactError::Other(format!(
                            "Checkpoint claim heartbeat failed: {error}"
                        )));
                    }
                }
            }
        }
    }

    /// 从已认领的检查点继续执行；若决策为拒绝则直接返回 Rejected
    async fn resume_claimed(
        &self,
        checkpoint: Checkpoint,
        decision: ApprovalDecision,
        state_updates: Option<std::collections::HashMap<String, Value>>,
    ) -> Result<RunUntilInterruptResult> {
        if let ApprovalDecision::Rejected { reason } = decision {
            info!(
                graph = %self.name,
                checkpoint_id = %checkpoint.id,
                reason = reason.as_deref().unwrap_or("no reason"),
                "Resume rejected, aborting workflow"
            );
            return Ok(RunUntilInterruptResult::Rejected {
                state: checkpoint.restore_state()?,
                path: checkpoint.path,
                steps: checkpoint.step_count,
                reason,
            });
        }

        let state = checkpoint.restore_state()?;
        if let Some(updates) = state_updates {
            for (key, value) in updates {
                state.set(key, value)?;
            }
        }
        info!(
            graph = %self.name,
            checkpoint_id = %checkpoint.id,
            node = %checkpoint.current_node,
            "Resuming from checkpoint"
        );
        let skip_interrupt_before_once = checkpoint.interrupt_type == InterruptType::BeforeNode
            && matches!(checkpoint.continuation, WorkflowContinuation::Node);
        let execution = GraphExecution {
            state,
            current: checkpoint.current_node.clone(),
            path: checkpoint.path.clone(),
            step_count: checkpoint.step_count,
            continuation: checkpoint.continuation.clone(),
            parent_checkpoint: Some(checkpoint),
            skip_interrupt_before_once,
        };
        self.execute_loop(execution, true, None).await
    }

    /**
    从检查点恢复执行，同时注入状态修改。

    在恢复前将 `state_updates` 合并进检查点的状态，
    适用于需要在继续执行前从外部修改工作流状态的场景。
    */
    pub async fn resume_with_state(
        &self,
        checkpoint: Checkpoint,
        state_updates: std::collections::HashMap<String, Value>,
    ) -> Result<RunUntilInterruptResult> {
        self.resume_with_updates(checkpoint, ApprovalDecision::Approved, Some(state_updates))
            .await
    }

    /**
    恢复之前某个检查点的状态，但**不**继续执行。

    返回恢复后的状态与检查点元数据，可用于查看历史状态或准备分支。
    */
    pub async fn restore_to_checkpoint(
        &self,
        checkpoint_id: &str,
    ) -> Result<(SharedState, Checkpoint)> {
        let checkpoint = self
            .checkpoint_store
            .load(checkpoint_id)
            .await?
            .ok_or_else(|| {
                ReactError::Agent(Box::new(AgentError::InitializationFailed(format!(
                    "Checkpoint '{}' not found",
                    checkpoint_id
                ))))
            })?;
        let state = checkpoint.restore_state()?;
        Ok((state, checkpoint))
    }

    /**
    从已有检查点分叉出一次新的执行。

    创建一个新的分支检查点，并将 `state_updates` 合并进分叉后的状态，
    通过 `parent_checkpoint_id` 保留血缘关系。新分支会被保存，
    并从检查点所在节点继续执行。
    */
    pub async fn branch_from(
        &self,
        checkpoint_id: &str,
        state_updates: std::collections::HashMap<String, Value>,
        branch_name: Option<String>,
    ) -> Result<RunUntilInterruptResult> {
        let (state, parent_cp) = self.restore_to_checkpoint(checkpoint_id).await?;

        // 应用状态更新
        for (key, value) in &state_updates {
            let _ = state.set(key, value.clone());
        }

        // 创建分支检查点
        let branch_name = branch_name.unwrap_or_else(|| format!("branch-{}", uuid::Uuid::new_v4()));
        let fork_cp = Checkpoint::branch(
            &parent_cp.id,
            &branch_name,
            self.name.clone(),
            parent_cp.current_node.clone(),
            &state,
            parent_cp.path.clone(),
            parent_cp.step_count,
            parent_cp.interrupt_type,
        )
        .bind_execution(self.graph_revision.clone(), parent_cp.continuation.clone());

        // 保存分叉检查点
        self.checkpoint_store.save(&fork_cp).await?;

        // 从分叉处恢复执行
        self.resume(fork_cp, ApprovalDecision::Approved).await
    }

    /// 为已有检查点附加人类可读的标签（label）和/或标记（tags）。
    pub async fn tag_checkpoint(
        &self,
        checkpoint_id: &str,
        label: Option<&str>,
        tags: Vec<&str>,
    ) -> Result<()> {
        let mut checkpoint = self
            .checkpoint_store
            .load(checkpoint_id)
            .await?
            .ok_or_else(|| {
                ReactError::Agent(Box::new(AgentError::InitializationFailed(format!(
                    "Checkpoint '{}' not found",
                    checkpoint_id
                ))))
            })?;

        let expected_generation = checkpoint.generation;

        if let Some(l) = label {
            checkpoint.label = Some(l.to_string());
        }
        if !tags.is_empty() {
            checkpoint.tags = tags.iter().map(|t| t.to_string()).collect();
        }
        checkpoint.generation = expected_generation.saturating_add(1);

        let committed = self
            .checkpoint_store
            .save_if_generation(&checkpoint, expected_generation)
            .await?;
        if committed {
            Ok(())
        } else {
            Err(ReactError::Other(format!(
                "Checkpoint '{checkpoint_id}' changed or was claimed while tagging"
            )))
        }
    }

    /// 加载检查点
    pub async fn load_checkpoint(&self, id: &str) -> Result<Option<Checkpoint>> {
        self.checkpoint_store.load(id).await
    }

    /// 列出所有检查点
    pub async fn list_checkpoints(&self) -> Result<Vec<super::checkpoint_store::CheckpointInfo>> {
        self.checkpoint_store.list().await
    }

    /// 列出按图名称（即本图）过滤后的检查点。
    pub async fn list_checkpoints_by_graph(
        &self,
    ) -> Result<Vec<super::checkpoint_store::CheckpointInfo>> {
        self.checkpoint_store.list_by_graph(&self.name).await
    }

    /// 设置自定义检查点存储
    pub fn with_checkpoint_store(mut self, store: Arc<dyn CheckpointStore>) -> Self {
        self.checkpoint_store = store;
        self
    }

    /**
    以流式模式执行图工作流，逐节点发出 [`WorkflowEvent`] 事件。

    每个节点都会发出 Start/End 事件，最后发出 `Completed` 事件。
    */
    pub async fn run_stream(
        &self,
        state: SharedState,
    ) -> Result<BoxStream<'_, Result<WorkflowEvent>>> {
        let (events, mut receiver) = tokio::sync::mpsc::unbounded_channel();
        let execution = self.execute_loop(self.fresh_execution(state), false, Some(events));
        let stream = async_stream::try_stream! {
            /// 单次轮询的结果：收到事件或执行结束
            enum StreamPoll {
                /// 从事件通道收到的事件（None 表示通道已关闭）
                Event(Option<WorkflowEvent>),
                /// 执行循环的最终结果
                Outcome(Box<Result<RunUntilInterruptResult>>),
            }

            tokio::pin!(execution);
            loop {
                let poll = tokio::select! {
                    biased;
                    event = receiver.recv() => StreamPoll::Event(event),
                    outcome = &mut execution => StreamPoll::Outcome(Box::new(outcome)),
                };
                match poll {
                    StreamPoll::Event(Some(event)) => yield event,
                    StreamPoll::Event(None) => {
                        let outcome = (&mut execution).await?;
                        if !matches!(outcome, RunUntilInterruptResult::Completed(_)) {
                            Err(ReactError::Other(
                                "Streaming graph execution returned a non-terminal outcome".to_string(),
                            ))?;
                        }
                        break;
                    }
                    StreamPoll::Outcome(outcome) => {
                        while let Ok(event) = receiver.try_recv() {
                            yield event;
                        }
                        let outcome = *outcome;
                        let outcome = outcome?;
                        if !matches!(outcome, RunUntilInterruptResult::Completed(_)) {
                            Err(ReactError::Other(
                                "Streaming graph execution returned a non-terminal outcome".to_string(),
                            ))?;
                        }
                        break;
                    }
                }
            }
        };

        Ok(Box::pin(stream))
    }

    /// 解析下一个节点
    async fn resolve_next(&self, current: &str, state: &SharedState) -> Result<NextStep> {
        let edges = match self.edges.get(current) {
            Some(e) => e,
            None => {
                // 没有出边 -> 若为结束节点则结束，否则报错
                if self.finish_nodes.contains(&current.to_string()) {
                    return Ok(NextStep::End);
                }
                return Err(ReactError::Agent(Box::new(
                    AgentError::InitializationFailed(format!(
                        "Node '{}' has no outgoing edges and is not a finish node",
                        current
                    )),
                )));
            }
        };

        // 取第一条匹配的边
        if let Some(edge) = edges.iter().next() {
            match &edge.kind {
                EdgeKind::Fixed(to) => {
                    if to == Self::END {
                        return Ok(NextStep::End);
                    }
                    return Ok(NextStep::Single(to.clone()));
                }
                EdgeKind::Conditional(f) => {
                    let target = f.evaluate(state).await;
                    if target == Self::END {
                        return Ok(NextStep::End);
                    }
                    return Ok(NextStep::Single(target));
                }
                EdgeKind::Parallel { targets, then } => {
                    return Ok(NextStep::Parallel {
                        targets: targets.clone(),
                        then: then.clone(),
                    });
                }
            }
        }

        // 不应到达此处
        Ok(NextStep::End)
    }
}

/// 内部路由结果
enum NextStep {
    /// 转移到单个节点
    Single(String),
    /// 并行扇出到多个节点，完成后进入 then 节点
    Parallel {
        /// 并行执行的目标节点
        targets: Vec<String>,
        /// 汇合节点
        then: String,
    },
    /// 执行结束
    End,
}

// ── 单元测试 ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workflow::checkpoint_store::FileCheckpointStore;
    use echo_core::agent::AgentEvent;
    use futures::stream;
    use std::sync::atomic::{AtomicBool, Ordering};

    /// 以流式方式输出 token 的测试 Agent
    struct StreamingAgent;

    impl Agent for StreamingAgent {
        fn name(&self) -> &str {
            "streaming-agent"
        }

        fn model_name(&self) -> &str {
            "test"
        }

        fn system_prompt(&self) -> &str {
            "test"
        }

        fn execute<'a>(&'a self, _task: &'a str) -> BoxFuture<'a, Result<String>> {
            Box::pin(async { Ok("streamed answer".to_string()) })
        }

        fn execute_stream<'a>(
            &'a self,
            _task: &'a str,
        ) -> BoxFuture<'a, Result<BoxStream<'a, Result<AgentEvent>>>> {
            Box::pin(async {
                Ok(Box::pin(stream::iter([
                    Ok(AgentEvent::Token("streamed ".to_string())),
                    Ok(AgentEvent::Token("answer".to_string())),
                    Ok(AgentEvent::FinalAnswer("streamed answer".to_string())),
                ])) as BoxStream<'a, Result<AgentEvent>>)
            })
        }
    }

    /// 用于探测取消信号是否传递到 Agent 生产者的测试 Agent
    #[derive(Clone, Default)]
    struct CancellationProbeAgent {
        /// 是否已开始执行
        started: Arc<AtomicBool>,
        /// 是否已收到取消信号
        cancelled: Arc<AtomicBool>,
    }

    impl CancellationProbeAgent {
        /// 等待指定标志位被置为 true（最多 1 秒）
        async fn wait_for(&self, flag: &AtomicBool) -> Result<()> {
            tokio::time::timeout(Duration::from_secs(1), async {
                while !flag.load(Ordering::SeqCst) {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .map_err(|_| ReactError::Other("cancellation probe timed out".to_string()))
        }

        /// 等待 Agent 开始执行
        async fn wait_started(&self) -> Result<()> {
            self.wait_for(&self.started).await
        }

        /// 等待 Agent 收到取消信号
        async fn wait_cancelled(&self) -> Result<()> {
            self.wait_for(&self.cancelled).await
        }
    }

    impl Agent for CancellationProbeAgent {
        fn name(&self) -> &str {
            "cancellation-probe"
        }

        fn model_name(&self) -> &str {
            "test"
        }

        fn system_prompt(&self) -> &str {
            "test"
        }

        fn execute<'a>(&'a self, _task: &'a str) -> BoxFuture<'a, Result<String>> {
            Box::pin(std::future::pending())
        }

        fn execute_stream<'a>(
            &'a self,
            _task: &'a str,
        ) -> BoxFuture<'a, Result<BoxStream<'a, Result<AgentEvent>>>> {
            Box::pin(async {
                Err(ReactError::Other(
                    "workflow must use the cancellable Agent stream".to_string(),
                ))
            })
        }

        fn execute_stream_with_cancel<'a>(
            &'a self,
            _task: &'a str,
            cancel: CancellationToken,
        ) -> BoxFuture<'a, Result<BoxStream<'a, Result<AgentEvent>>>> {
            let started = self.started.clone();
            let cancelled = self.cancelled.clone();
            Box::pin(async move {
                started.store(true, Ordering::SeqCst);
                let _producer = tokio::spawn(async move {
                    cancel.cancelled().await;
                    cancelled.store(true, Ordering::SeqCst);
                });
                let events: BoxStream<'a, Result<AgentEvent>> = Box::pin(
                    stream::iter([Ok(AgentEvent::Token("ready".to_string()))])
                        .chain(stream::pending()),
                );
                Ok(events)
            })
        }
    }

    /// 测试线性图按顺序执行并累积状态
    #[tokio::test]
    async fn test_linear_graph() {
        let graph = GraphBuilder::new("linear")
            .add_function_node("a", |state: &SharedState| {
                Box::pin(async move {
                    let _ = state.set("x", 1i64);
                    Ok(())
                })
            })
            .add_function_node("b", |state: &SharedState| {
                Box::pin(async move {
                    let x: i64 = state.get("x").unwrap();
                    let _ = state.set("x", x + 10);
                    Ok(())
                })
            })
            .add_function_node("c", |state: &SharedState| {
                Box::pin(async move {
                    let x: i64 = state.get("x").unwrap();
                    let _ = state.set("x", x * 2);
                    Ok(())
                })
            })
            .set_entry("a")
            .add_edge("a", "b")
            .add_edge("b", "c")
            .set_finish("c")
            .build()
            .unwrap();

        let state = SharedState::new();
        let result = graph.run(state).await.unwrap();

        assert_eq!(result.state.get::<i64>("x"), Some(22)); // (1+10)*2
        assert_eq!(result.path, vec!["a", "b", "c"]);
        assert_eq!(result.steps, 3);
    }

    /// 测试条件边根据状态选择通过/失败分支
    #[tokio::test]
    async fn test_conditional_graph() {
        let graph = GraphBuilder::new("conditional")
            .add_function_node("check", |_state: &SharedState| {
                Box::pin(async move {
                    // score 由外部设置
                    Ok(())
                })
            })
            .add_function_node("pass", |state: &SharedState| {
                Box::pin(async move {
                    let _ = state.set("result", "passed");
                    Ok(())
                })
            })
            .add_function_node("fail", |state: &SharedState| {
                Box::pin(async move {
                    let _ = state.set("result", "failed");
                    Ok(())
                })
            })
            .set_entry("check")
            .add_conditional_edge("check", |state: &SharedState| {
                Box::pin(async move {
                    let score: i64 = state.get("score").unwrap_or(0);
                    if score >= 60 {
                        "pass".to_string()
                    } else {
                        "fail".to_string()
                    }
                })
            })
            .set_finish("pass")
            .set_finish("fail")
            .build()
            .unwrap();

        // 测试通过路径
        let state = SharedState::new();
        let _ = state.set("score", 80i64);
        let result = graph.run(state).await.unwrap();
        assert_eq!(
            result.state.get::<String>("result"),
            Some("passed".to_string())
        );
        assert_eq!(result.path, vec!["check", "pass"]);

        // 测试失败路径
        let state = SharedState::new();
        let _ = state.set("score", 40i64);
        let result = graph.run(state).await.unwrap();
        assert_eq!(
            result.state.get::<String>("result"),
            Some("failed".to_string())
        );
        assert_eq!(result.path, vec!["check", "fail"]);
    }

    /// 测试条件边构成的循环能正确终止
    #[tokio::test]
    async fn test_loop_graph() {
        // 模拟循环：counter 从 0 递增到 5
        let graph = GraphBuilder::new("loop")
            .add_function_node("init", |state: &SharedState| {
                Box::pin(async move {
                    let _ = state.set("counter", 0i64);
                    Ok(())
                })
            })
            .add_function_node("increment", |state: &SharedState| {
                Box::pin(async move {
                    let c: i64 = state.get("counter").unwrap();
                    let _ = state.set("counter", c + 1);
                    Ok(())
                })
            })
            .add_function_node("done", |_state: &SharedState| {
                Box::pin(async move { Ok(()) })
            })
            .set_entry("init")
            .add_edge("init", "increment")
            .add_conditional_edge("increment", |state: &SharedState| {
                Box::pin(async move {
                    let c: i64 = state.get("counter").unwrap_or(0);
                    if c >= 5 {
                        "done".to_string()
                    } else {
                        "increment".to_string()
                    }
                })
            })
            .set_finish("done")
            .build()
            .unwrap();

        let state = SharedState::new();
        let result = graph.run(state).await.unwrap();
        assert_eq!(result.state.get::<i64>("counter"), Some(5));
        // init + 5*increment + done = 7 步
        assert_eq!(result.steps, 7);
    }

    /// 测试并行扇出后在汇合节点合并结果
    #[tokio::test]
    async fn test_parallel_graph() {
        let graph = GraphBuilder::new("parallel")
            .add_function_node("start", |state: &SharedState| {
                Box::pin(async move {
                    let _ = state.set("input", "hello");
                    Ok(())
                })
            })
            .add_function_node("upper", |state: &SharedState| {
                Box::pin(async move {
                    let s: String = state.get("input").unwrap();
                    let _ = state.set("upper_result", s.to_uppercase());
                    Ok(())
                })
            })
            .add_function_node("length", |state: &SharedState| {
                Box::pin(async move {
                    let s: String = state.get("input").unwrap();
                    let _ = state.set("length_result", s.len() as i64);
                    Ok(())
                })
            })
            .add_function_node("combine", |state: &SharedState| {
                Box::pin(async move {
                    let u: String = state.get("upper_result").unwrap();
                    let l: i64 = state.get("length_result").unwrap();
                    let _ = state.set("final", format!("{u} (len={l})"));
                    Ok(())
                })
            })
            .set_entry("start")
            .add_parallel_edge(
                "start",
                vec!["upper".to_string(), "length".to_string()],
                "combine",
            )
            .set_finish("combine")
            .build()
            .unwrap();

        let state = SharedState::new();
        let result = graph.run(state).await.unwrap();
        assert_eq!(
            result.state.get::<String>("final"),
            Some("HELLO (len=5)".to_string())
        );
    }

    /// 测试并行分支失败时取消并清理兄弟分支
    #[tokio::test]
    async fn parallel_failure_cancels_and_drains_siblings() {
        let graph = GraphBuilder::new("parallel-failure")
            .add_function_node("start", |_state: &SharedState| Box::pin(async { Ok(()) }))
            .add_function_node("fail", |_state: &SharedState| {
                Box::pin(async { Err(ReactError::Other("branch failed".to_string())) })
            })
            .add_function_node("slow", |_state: &SharedState| {
                Box::pin(async {
                    tokio::time::sleep(Duration::from_secs(60)).await;
                    Ok(())
                })
            })
            .add_function_node("merge", |_state: &SharedState| Box::pin(async { Ok(()) }))
            .set_entry("start")
            .add_parallel_edge("start", vec!["fail".into(), "slow".into()], "merge")
            .set_finish("merge")
            .build()
            .unwrap();

        let result = tokio::time::timeout(Duration::from_secs(1), graph.run(SharedState::new()))
            .await
            .unwrap();
        assert!(result.is_err());
    }

    /// 测试在并行源节点后中断，恢复时执行并合并所有分支
    #[tokio::test]
    async fn resume_after_parallel_source_executes_and_merges_all_branches() -> Result<()> {
        let graph = GraphBuilder::new("parallel-resume")
            .add_function_node("start", |state: &SharedState| {
                Box::pin(async move {
                    state
                        .set("input", "hello")
                        .map_err(|error| ReactError::Other(error.to_string()))
                })
            })
            .add_function_node("upper", |state: &SharedState| {
                Box::pin(async move {
                    let input = state.get::<String>("input").ok_or_else(|| {
                        ReactError::Other("parallel branch lost input".to_string())
                    })?;
                    state
                        .set("upper", input.to_uppercase())
                        .map_err(|error| ReactError::Other(error.to_string()))
                })
            })
            .add_function_node("length", |state: &SharedState| {
                Box::pin(async move {
                    let input = state.get::<String>("input").ok_or_else(|| {
                        ReactError::Other("parallel branch lost input".to_string())
                    })?;
                    state
                        .set("length", input.chars().count())
                        .map_err(|error| ReactError::Other(error.to_string()))
                })
            })
            .add_function_node("merge", |state: &SharedState| {
                Box::pin(async move {
                    let upper = state
                        .get::<String>("upper")
                        .ok_or_else(|| ReactError::Other("upper branch was skipped".to_string()))?;
                    let length = state.get::<usize>("length").ok_or_else(|| {
                        ReactError::Other("length branch was skipped".to_string())
                    })?;
                    state
                        .set("result", format!("{upper}:{length}"))
                        .map_err(|error| ReactError::Other(error.to_string()))
                })
            })
            .set_entry("start")
            .add_parallel_edge(
                "start",
                vec!["upper".to_string(), "length".to_string()],
                "merge",
            )
            .set_finish("merge")
            .interrupt_after(vec!["start"])
            .build()?;

        let checkpoint = match graph.run_until_interrupt(SharedState::new()).await? {
            RunUntilInterruptResult::Interrupted(interrupt) => interrupt.checkpoint,
            RunUntilInterruptResult::Completed(_)
            | RunUntilInterruptResult::Deferred(_)
            | RunUntilInterruptResult::Rejected { .. } => {
                return Err(ReactError::Other(
                    "graph did not interrupt after parallel source".to_string(),
                ));
            }
        };
        assert!(matches!(
            checkpoint.continuation,
            WorkflowContinuation::FanOut { .. }
        ));

        let result = match graph.resume(checkpoint, ApprovalDecision::Approved).await? {
            RunUntilInterruptResult::Completed(result) => result,
            RunUntilInterruptResult::Interrupted(_) => {
                return Err(ReactError::Other("resume interrupted twice".to_string()));
            }
            RunUntilInterruptResult::Deferred(_) | RunUntilInterruptResult::Rejected { .. } => {
                return Err(ReactError::Other("resume did not complete".to_string()));
            }
        };
        assert_eq!(
            result.state.get::<String>("result").as_deref(),
            Some("HELLO:5")
        );
        assert_eq!(result.path, vec!["start", "upper", "length", "merge"]);
        Ok(())
    }

    /// 测试指向 __end__ 的边能结束执行
    #[tokio::test]
    async fn test_end_edge() {
        let graph = GraphBuilder::new("end_test")
            .add_function_node("only", |state: &SharedState| {
                Box::pin(async move {
                    let _ = state.set("done", true);
                    Ok(())
                })
            })
            .set_entry("only")
            .add_edge("only", "__end__")
            .build()
            .unwrap();

        let state = SharedState::new();
        let result = graph.run(state).await.unwrap();
        assert_eq!(result.state.get::<bool>("done"), Some(true));
        assert_eq!(result.path, vec!["only"]);
    }

    /// 测试超出最大步数时返回错误
    #[tokio::test]
    async fn test_max_steps_exceeded() {
        let mut graph = GraphBuilder::new("infinite")
            .add_function_node("loop_node", |_state: &SharedState| {
                Box::pin(async move { Ok(()) })
            })
            .set_entry("loop_node")
            .add_edge("loop_node", "loop_node") // 无限循环
            .build()
            .unwrap();

        graph.set_max_steps(10);

        let state = SharedState::new();
        let result = graph.run(state).await;
        assert!(result.is_err());
    }

    /// 测试缺少入口节点时构建失败
    #[test]
    fn test_build_validation_missing_entry() {
        let result = GraphBuilder::new("bad")
            .add_function_node("a", |_: &SharedState| Box::pin(async { Ok(()) }))
            .build();
        assert!(result.is_err());
    }

    /// 测试入口节点不存在时构建失败
    #[test]
    fn test_build_validation_unknown_entry() {
        let result = GraphBuilder::new("bad")
            .add_function_node("a", |_: &SharedState| Box::pin(async { Ok(()) }))
            .set_entry("nonexistent")
            .build();
        assert!(result.is_err());
    }

    /// 测试边指向未知节点时构建失败
    #[test]
    fn test_build_validation_unknown_edge_target() {
        let result = GraphBuilder::new("bad")
            .add_function_node("a", |_: &SharedState| Box::pin(async { Ok(()) }))
            .set_entry("a")
            .add_edge("a", "nonexistent")
            .build();
        assert!(result.is_err());
    }

    /// 测试同一节点存在多条出边时构建被拒绝
    #[test]
    fn test_build_rejects_multiple_outgoing_edges() {
        // 同一节点有两条出边 -> 构建阶段应报错
        let result = GraphBuilder::new("multi_edge")
            .add_function_node("a", |_: &SharedState| Box::pin(async { Ok(()) }))
            .add_function_node("b", |_: &SharedState| Box::pin(async { Ok(()) }))
            .add_function_node("c", |_: &SharedState| Box::pin(async { Ok(()) }))
            .set_entry("a")
            .add_edge("a", "b")
            .add_edge("a", "c") // 第二条出边
            .build();
        assert!(
            result.is_err(),
            "Multiple outgoing edges should be rejected"
        );
        let err_msg = match result {
            Err(e) => e.to_string(),
            _ => String::new(),
        };
        assert!(
            err_msg.contains("multiple outgoing edges"),
            "Error should mention 'multiple outgoing edges', got: {err_msg}"
        );
    }

    // ── 功能 4：工作流流式输出（run_stream）──────────────────────

    /// 测试线性图的流式执行事件序列
    #[tokio::test]
    async fn test_run_stream_linear() {
        use super::WorkflowEvent;
        use futures::StreamExt;

        let graph = GraphBuilder::new("stream_linear")
            .add_function_node("a", |state: &SharedState| {
                Box::pin(async move {
                    let _ = state.set("x", 1i64);
                    Ok(())
                })
            })
            .add_function_node("b", |state: &SharedState| {
                Box::pin(async move {
                    let x: i64 = state.get("x").unwrap();
                    let _ = state.set("result", format!("x={}", x));
                    Ok(())
                })
            })
            .set_entry("a")
            .add_edge("a", "b")
            .set_finish("b")
            .build()
            .unwrap();

        let state = SharedState::new();
        let mut stream = graph.run_stream(state).await.unwrap();

        let mut events = Vec::new();
        while let Some(event) = stream.next().await {
            events.push(event.unwrap());
        }

        let node_starts: Vec<_> = events
            .iter()
            .filter_map(|e| match e {
                WorkflowEvent::NodeStart { node_name, .. } => Some(node_name.clone()),
                _ => None,
            })
            .collect();
        let node_ends: Vec<_> = events
            .iter()
            .filter_map(|e| match e {
                WorkflowEvent::NodeEnd { node_name, .. } => Some(node_name.clone()),
                _ => None,
            })
            .collect();
        let completed = events
            .iter()
            .any(|e| matches!(e, WorkflowEvent::Completed { .. }));

        assert_eq!(node_starts, vec!["a", "b"]);
        assert_eq!(node_ends, vec!["a", "b"]);
        assert!(completed, "Should receive Completed event");
    }

    /// 测试流式执行转发 Agent token 并写入最终答案
    #[tokio::test]
    async fn run_stream_forwards_agent_tokens_and_commits_final_answer() -> Result<()> {
        let graph = GraphBuilder::new("stream-agent-token")
            .add_agent_node("agent", StreamingAgent, "task", "result")
            .set_entry("agent")
            .set_finish("agent")
            .build()?;
        let state = SharedState::new();
        state.set("task", "write")?;

        let mut events = graph.run_stream(state.clone()).await?;
        let mut tokens = Vec::new();
        while let Some(event) = events.next().await {
            if let WorkflowEvent::Token { node_name, token } = event? {
                tokens.push((node_name, token));
            }
        }

        assert_eq!(
            tokens,
            vec![
                ("agent".to_string(), "streamed ".to_string()),
                ("agent".to_string(), "answer".to_string()),
            ]
        );
        assert_eq!(
            state.get::<String>("result"),
            Some("streamed answer".to_string())
        );
        Ok(())
    }

    /// 测试丢弃工作流流时取消 Agent 生产者
    #[tokio::test]
    async fn dropping_workflow_stream_cancels_agent_producer() -> Result<()> {
        let probe = CancellationProbeAgent::default();
        let graph = GraphBuilder::new("stream-drop-cancel")
            .add_agent_node("agent", probe.clone(), "task", "result")
            .set_entry("agent")
            .set_finish("agent")
            .build()?;
        let mut events = graph.run_stream(SharedState::new()).await?;

        assert!(matches!(
            events.next().await,
            Some(Ok(WorkflowEvent::NodeStart { node_name, .. })) if node_name == "agent"
        ));
        assert!(matches!(
            events.next().await,
            Some(Ok(WorkflowEvent::Token { node_name, token }))
                if node_name == "agent" && token == "ready"
        ));
        drop(events);

        probe.wait_cancelled().await
    }

    /// 测试图取消令牌会取消非流式执行中的 Agent 生产者
    #[tokio::test]
    async fn graph_cancel_cancels_non_streaming_agent_producer() -> Result<()> {
        let probe = CancellationProbeAgent::default();
        let graph_cancel = CancellationToken::new();
        let graph = GraphBuilder::new("graph-cancel-producer")
            .add_agent_node("agent", probe.clone(), "task", "result")
            .set_entry("agent")
            .set_finish("agent")
            .build()?
            .with_cancel_token(graph_cancel.clone());
        let cancel_after_start = async {
            probe.wait_started().await?;
            graph_cancel.cancel();
            Result::<()>::Ok(())
        };

        let (run_result, cancel_result) =
            tokio::join!(graph.run(SharedState::new()), cancel_after_start,);
        cancel_result?;
        assert!(run_result.is_err());
        probe.wait_cancelled().await
    }

    /// 测试节点超时会取消 Agent 生产者
    #[tokio::test]
    async fn node_timeout_cancels_non_streaming_agent_producer() -> Result<()> {
        let probe = CancellationProbeAgent::default();
        let graph = GraphBuilder::new("node-timeout-producer")
            .add_agent_node("agent", probe.clone(), "task", "result")
            .set_entry("agent")
            .set_finish("agent")
            .node_timeout(Duration::from_millis(10))
            .build()?;

        let result = graph.run(SharedState::new()).await;
        assert!(matches!(result, Err(ReactError::Other(message)) if message.contains("timed out")));
        probe.wait_cancelled().await
    }

    /// 测试并行分支失败时取消兄弟分支中的 Agent 生产者
    #[tokio::test]
    async fn parallel_failure_cancels_agent_producer() -> Result<()> {
        let probe = CancellationProbeAgent::default();
        let graph = GraphBuilder::new("parallel-agent-cancel")
            .add_function_node("start", |_state: &SharedState| Box::pin(async { Ok(()) }))
            .add_agent_node("slow", probe.clone(), "task", "slow_result")
            .add_function_node("broken", |_state: &SharedState| {
                Box::pin(async {
                    tokio::time::sleep(Duration::from_millis(10)).await;
                    Err(ReactError::Other("branch failed".to_string()))
                })
            })
            .add_function_node("merge", |_state: &SharedState| Box::pin(async { Ok(()) }))
            .set_entry("start")
            .add_parallel_edge("start", vec!["slow".into(), "broken".into()], "merge")
            .set_finish("merge")
            .build()?;

        let result = graph.run(SharedState::new()).await;
        assert!(matches!(result, Err(error) if error.to_string().contains("branch failed")));
        probe.wait_cancelled().await
    }

    /// 测试结束节点后中断再恢复，路径与直接 run 一致
    #[tokio::test]
    async fn finish_interrupt_resumes_to_the_same_completed_path_as_run() -> Result<()> {
        let build_graph = || {
            GraphBuilder::new("finish-interrupt-parity")
                .add_function_node("start", |_state: &SharedState| Box::pin(async { Ok(()) }))
                .add_function_node("finish", |state: &SharedState| {
                    Box::pin(async move {
                        state
                            .set("result", "done")
                            .map_err(|error| ReactError::Other(error.to_string()))
                    })
                })
                .set_entry("start")
                .add_edge("start", "finish")
                .set_finish("finish")
                .interrupt_after(vec!["finish"])
                .build()
        };

        let direct = build_graph()?.run(SharedState::new()).await?;
        assert_eq!(direct.path, vec!["start", "finish"]);

        let graph = build_graph()?;
        let interrupted = graph.run_until_interrupt(SharedState::new()).await?;
        let checkpoint = match interrupted {
            RunUntilInterruptResult::Interrupted(interrupt) => interrupt.checkpoint,
            other => {
                return Err(ReactError::Other(format!(
                    "expected finish-node interrupt, got {other:?}"
                )));
            }
        };
        assert_eq!(checkpoint.path, direct.path);
        assert_eq!(checkpoint.continuation, WorkflowContinuation::End);

        let resumed = graph.resume(checkpoint, ApprovalDecision::Approved).await?;
        let completed = match resumed {
            RunUntilInterruptResult::Completed(result) => result,
            other => {
                return Err(ReactError::Other(format!(
                    "expected completed resume, got {other:?}"
                )));
            }
        };
        assert_eq!(completed.path, direct.path);
        assert_eq!(completed.steps, direct.steps);
        Ok(())
    }

    /// 测试流式执行在失败终止项之前发出并行节点的错误事件
    #[tokio::test]
    async fn run_stream_emits_parallel_node_error_before_failed_terminal_item() -> Result<()> {
        let graph = GraphBuilder::new("parallel-stream-error")
            .add_function_node("start", |_state: &SharedState| Box::pin(async { Ok(()) }))
            .add_function_node("broken", |_state: &SharedState| {
                Box::pin(async { Err(ReactError::Other("branch failed".to_string())) })
            })
            .add_function_node("slow", |_state: &SharedState| {
                Box::pin(async {
                    tokio::time::sleep(Duration::from_secs(60)).await;
                    Ok(())
                })
            })
            .add_function_node("merge", |_state: &SharedState| Box::pin(async { Ok(()) }))
            .set_entry("start")
            .add_parallel_edge("start", vec!["broken".into(), "slow".into()], "merge")
            .set_finish("merge")
            .build()?;

        let mut stream = graph.run_stream(SharedState::new()).await?;
        let mut items = Vec::new();
        while let Some(item) = stream.next().await {
            items.push(item);
        }

        assert!(items.iter().any(|item| {
            matches!(
                item,
                Ok(WorkflowEvent::NodeError { node_name, error })
                    if node_name == "broken" && error.contains("branch failed")
            )
        }));
        assert!(
            matches!(items.last(), Some(Err(error)) if error.to_string().contains("branch failed"))
        );
        Ok(())
    }

    /// 测试并行图的流式执行事件
    #[tokio::test]
    async fn test_run_stream_parallel() {
        use super::WorkflowEvent;
        use futures::StreamExt;

        let graph = GraphBuilder::new("stream_parallel")
            .add_function_node("start", |state: &SharedState| {
                Box::pin(async move {
                    let _ = state.set("val", "ok");
                    Ok(())
                })
            })
            .add_function_node("b1", |state: &SharedState| {
                Box::pin(async move {
                    let _ = state.set("b1_done", true);
                    Ok(())
                })
            })
            .add_function_node("b2", |state: &SharedState| {
                Box::pin(async move {
                    let _ = state.set("b2_done", true);
                    Ok(())
                })
            })
            .add_function_node("merge", |state: &SharedState| {
                Box::pin(async move {
                    let b1: bool = state.get("b1_done").unwrap_or(false);
                    let b2: bool = state.get("b2_done").unwrap_or(false);
                    let _ = state.set("result", format!("b1={b1},b2={b2}"));
                    Ok(())
                })
            })
            .set_entry("start")
            .add_parallel_edge("start", vec!["b1".into(), "b2".into()], "merge")
            .set_finish("merge")
            .build()
            .unwrap();

        let state = SharedState::new();
        let mut stream = graph.run_stream(state).await.unwrap();

        let mut node_start_names = Vec::new();
        let mut completed_result = None;

        while let Some(event) = stream.next().await {
            match event.unwrap() {
                WorkflowEvent::NodeStart { node_name, .. } => {
                    node_start_names.push(node_name);
                }
                WorkflowEvent::Completed { result, .. } => {
                    completed_result = Some(result);
                }
                _ => {}
            }
        }

        assert!(node_start_names.contains(&"start".to_string()));
        assert!(node_start_names.contains(&"b1".to_string()));
        assert!(node_start_names.contains(&"b2".to_string()));
        assert!(node_start_names.contains(&"merge".to_string()));
        assert_eq!(completed_result, Some("b1=true,b2=true".to_string()));
    }

    /// 测试条件图的流式执行只访问被选中的分支
    #[tokio::test]
    async fn test_run_stream_conditional() {
        use super::WorkflowEvent;
        use futures::StreamExt;

        let graph = GraphBuilder::new("stream_cond")
            .add_function_node("check", |_state: &SharedState| {
                Box::pin(async move { Ok(()) })
            })
            .add_function_node("yes", |state: &SharedState| {
                Box::pin(async move {
                    let _ = state.set("result", "took_yes_path");
                    Ok(())
                })
            })
            .add_function_node("no", |state: &SharedState| {
                Box::pin(async move {
                    let _ = state.set("result", "took_no_path");
                    Ok(())
                })
            })
            .set_entry("check")
            .add_conditional_edge("check", |state: &SharedState| {
                Box::pin(async move {
                    let flag: bool = state.get("flag").unwrap_or(false);
                    if flag {
                        "yes".to_string()
                    } else {
                        "no".to_string()
                    }
                })
            })
            .set_finish("yes")
            .set_finish("no")
            .build()
            .unwrap();

        let state = SharedState::new();
        let _ = state.set("flag", true);
        let mut stream = graph.run_stream(state).await.unwrap();

        let mut visited = Vec::new();
        while let Some(event) = stream.next().await {
            if let WorkflowEvent::NodeStart { node_name, .. } = event.unwrap() {
                visited.push(node_name);
            }
        }

        assert_eq!(visited, vec!["check", "yes"]);
    }

    /// 测试流式执行在失败终止项之前发出节点错误事件
    #[tokio::test]
    async fn run_stream_emits_node_error_before_failed_terminal_item() {
        use super::WorkflowEvent;
        use futures::StreamExt;

        let graph = GraphBuilder::new("stream-error")
            .add_function_node("broken", |_state: &SharedState| {
                Box::pin(async { Err(ReactError::Other("broken node".to_string())) })
            })
            .set_entry("broken")
            .set_finish("broken")
            .build()
            .unwrap();

        let mut stream = graph.run_stream(SharedState::new()).await.unwrap();
        let mut events = Vec::new();
        while let Some(item) = stream.next().await {
            events.push(item);
        }

        assert!(matches!(
            events.first(),
            Some(Ok(WorkflowEvent::NodeStart { node_name, .. })) if node_name == "broken"
        ));
        assert!(events.iter().any(|item| matches!(
            item,
            Ok(WorkflowEvent::NodeError { node_name, error })
                if node_name == "broken" && error.contains("broken node")
        )));
        assert!(events.iter().any(Result::is_err));
        assert!(
            !events
                .iter()
                .any(|item| matches!(item, Ok(WorkflowEvent::Completed { .. })))
        );
    }

    /// 测试 resume_with_state 注入状态并复用检查点身份
    #[tokio::test]
    async fn test_resume_with_state_reuses_checkpoint_identity() {
        use std::collections::HashMap;

        let checkpoint_store = Arc::new(MemoryCheckpointStore::new());
        let graph = GraphBuilder::new("resume_with_state")
            .add_function_node("start", |state: &SharedState| {
                Box::pin(async move {
                    let _ = state.set("message", "original");
                    Ok(())
                })
            })
            .add_function_node("finish", |state: &SharedState| {
                Box::pin(async move {
                    let msg: String = state.get("message").unwrap_or_default();
                    let _ = state.set("result", format!("seen={msg}"));
                    Ok(())
                })
            })
            .set_entry("start")
            .add_edge("start", "finish")
            .set_finish("finish")
            .interrupt_before(vec!["finish"])
            .build()
            .unwrap()
            .with_checkpoint_store(checkpoint_store.clone());

        let state = SharedState::new();
        let interrupted = graph.run_until_interrupt(state).await.unwrap();
        let checkpoint = match interrupted {
            RunUntilInterruptResult::Interrupted(interrupt) => interrupt.checkpoint,
            _ => return,
        };

        assert_eq!(graph.list_checkpoints().await.unwrap().len(), 1);

        let mut updates = HashMap::new();
        updates.insert("message".to_string(), Value::String("patched".to_string()));

        let resumed = graph.resume_with_state(checkpoint, updates).await.unwrap();
        let result = match resumed {
            RunUntilInterruptResult::Completed(result) => result,
            _ => return,
        };

        let seen: String = result.state.get("result").unwrap_or_default();
        assert_eq!(seen, "seen=patched");
        assert!(
            graph.list_checkpoints().await.unwrap().is_empty(),
            "checkpoint should be deleted after successful resume"
        );
    }

    /// 构建一个在 start 节点前中断、使用指定存储与版本号的测试图
    fn interrupting_graph(name: &str, revision: &str, store: Arc<MemoryCheckpointStore>) -> Graph {
        GraphBuilder::new(name)
            .add_function_node("start", |state: &SharedState| {
                Box::pin(async move {
                    state
                        .set(
                            "executions",
                            state
                                .get::<u64>("executions")
                                .unwrap_or(0)
                                .saturating_add(1),
                        )
                        .map_err(Into::into)
                })
            })
            .set_entry("start")
            .set_finish("start")
            .interrupt_before(vec!["start"])
            .revision(revision)
            .build()
            .unwrap_or_else(|error| panic!("test graph must build: {error}"))
            .with_checkpoint_store(store)
    }

    /// 测试检查点只能被认领一次
    #[tokio::test]
    async fn checkpoint_is_claimed_once() {
        let store = Arc::new(MemoryCheckpointStore::new());
        let graph = interrupting_graph("claim-once", "v1", store);
        let checkpoint = match graph.run_until_interrupt(SharedState::new()).await {
            Ok(RunUntilInterruptResult::Interrupted(interrupt)) => interrupt.checkpoint,
            other => panic!("expected interrupt, got {other:?}"),
        };
        let replay = checkpoint.clone();
        assert!(matches!(
            graph.resume(checkpoint, ApprovalDecision::Approved).await,
            Ok(RunUntilInterruptResult::Completed(_))
        ));
        assert!(
            graph
                .resume(replay, ApprovalDecision::Approved)
                .await
                .is_err()
        );
    }

    /// 测试跨图或版本不匹配时不会消费检查点
    #[tokio::test]
    async fn cross_graph_and_revision_drift_do_not_consume_checkpoint() {
        let store = Arc::new(MemoryCheckpointStore::new());
        let source = interrupting_graph("source", "v1", store.clone());
        let checkpoint = match source.run_until_interrupt(SharedState::new()).await {
            Ok(RunUntilInterruptResult::Interrupted(interrupt)) => interrupt.checkpoint,
            other => panic!("expected interrupt, got {other:?}"),
        };
        let wrong_graph = interrupting_graph("other", "v1", store.clone());
        assert!(
            wrong_graph
                .resume(checkpoint.clone(), ApprovalDecision::Approved)
                .await
                .is_err()
        );
        let wrong_revision = interrupting_graph("source", "v2", store.clone());
        assert!(
            wrong_revision
                .resume(checkpoint.clone(), ApprovalDecision::Approved)
                .await
                .is_err()
        );
        assert!(matches!(
            source.resume(checkpoint, ApprovalDecision::Approved).await,
            Ok(RunUntilInterruptResult::Completed(_))
        ));
    }

    /// 测试延后决策保留检查点、拒绝决策消费检查点
    #[tokio::test]
    async fn deferred_preserves_and_rejected_consumes_checkpoint() {
        let store = Arc::new(MemoryCheckpointStore::new());
        let graph = interrupting_graph("decisions", "v1", store);
        let checkpoint = match graph.run_until_interrupt(SharedState::new()).await {
            Ok(RunUntilInterruptResult::Interrupted(interrupt)) => interrupt.checkpoint,
            other => panic!("expected interrupt, got {other:?}"),
        };
        assert!(matches!(
            graph
                .resume(checkpoint.clone(), ApprovalDecision::Deferred)
                .await,
            Ok(RunUntilInterruptResult::Deferred(_))
        ));
        assert!(matches!(
            graph
                .resume(
                    checkpoint.clone(),
                    ApprovalDecision::Rejected {
                        reason: Some("user rejected".to_string()),
                    },
                )
                .await,
            Ok(RunUntilInterruptResult::Rejected { .. })
        ));
        assert!(
            graph
                .resume(checkpoint, ApprovalDecision::Approved)
                .await
                .is_err()
        );
    }

    /// 测试恢复失败时将已认领的检查点重新入队
    #[tokio::test]
    async fn failed_resume_requeues_the_claimed_checkpoint() {
        let store = Arc::new(MemoryCheckpointStore::new());
        let graph = GraphBuilder::new("resume-requeue")
            .add_function_node("start", |_state: &SharedState| {
                Box::pin(async { Err(ReactError::Other("node failed".to_string())) })
            })
            .set_entry("start")
            .set_finish("start")
            .interrupt_before(vec!["start"])
            .revision("v1")
            .build()
            .unwrap()
            .with_checkpoint_store(store.clone());

        let checkpoint = match graph.run_until_interrupt(SharedState::new()).await.unwrap() {
            RunUntilInterruptResult::Interrupted(interrupt) => interrupt.checkpoint,
            _ => return,
        };
        assert!(
            graph
                .resume(checkpoint, ApprovalDecision::Approved)
                .await
                .is_err()
        );

        let pending = store.list().await.unwrap();
        assert_eq!(pending.len(), 1);
        assert!(store.load(&pending[0].id).await.unwrap().is_some());
    }

    /// 测试长时间恢复会续期文件认领，并阻止打标签导致检查点复活
    #[tokio::test]
    async fn long_resume_renews_file_claim_and_blocks_tag_resurrection() -> Result<()> {
        let temp_path =
            std::env::temp_dir().join(format!("echo_graph_renew_{}", uuid::Uuid::new_v4()));
        let store = Arc::new(
            FileCheckpointStore::new(&temp_path)
                .with_claim_timing(Duration::from_millis(120), Duration::from_millis(30))?,
        );
        let observer = FileCheckpointStore::new(&temp_path)
            .with_claim_timing(Duration::from_millis(120), Duration::from_millis(30))?;
        let graph = Arc::new(
            GraphBuilder::new("renew-active-claim")
                .add_function_node("work", |_state: &SharedState| {
                    Box::pin(async {
                        tokio::time::sleep(Duration::from_millis(260)).await;
                        Ok(())
                    })
                })
                .set_entry("work")
                .set_finish("work")
                .interrupt_before(vec!["work"])
                .revision("v1")
                .build()?
                .with_checkpoint_store(store.clone()),
        );
        let checkpoint = match graph.run_until_interrupt(SharedState::new()).await? {
            RunUntilInterruptResult::Interrupted(interrupt) => interrupt.checkpoint,
            other => {
                return Err(ReactError::Other(format!(
                    "expected interrupt before long resume, got {other:?}"
                )));
            }
        };
        let checkpoint_id = checkpoint.id.clone();
        let resume_graph = graph.clone();
        let resume = tokio::spawn(async move {
            resume_graph
                .resume(checkpoint, ApprovalDecision::Approved)
                .await
        });

        tokio::time::sleep(Duration::from_millis(170)).await;
        assert!(observer.claim(&checkpoint_id).await?.is_none());
        assert!(
            observer
                .load(&checkpoint_id)
                .await?
                .is_some_and(|checkpoint| { checkpoint.resume_attempt_id.is_some() })
        );
        assert!(
            graph
                .tag_checkpoint(&checkpoint_id, Some("late"), vec!["stale"])
                .await
                .is_err()
        );

        let result = resume
            .await
            .map_err(|error| ReactError::Other(format!("resume task failed: {error}")))??;
        assert!(matches!(result, RunUntilInterruptResult::Completed(_)));
        assert!(store.load(&checkpoint_id).await?.is_none());
        let _ = std::fs::remove_dir_all(&temp_path);
        Ok(())
    }

    /// 测试为已认领的检查点打标签会失败且不会使其复活
    #[tokio::test]
    async fn tagging_claimed_checkpoint_fails_without_resurrection() {
        let store = Arc::new(MemoryCheckpointStore::new());
        let graph = interrupting_graph("tag-race", "v1", store.clone());
        let checkpoint = match graph.run_until_interrupt(SharedState::new()).await.unwrap() {
            RunUntilInterruptResult::Interrupted(interrupt) => interrupt.checkpoint,
            _ => return,
        };
        let claimed = store.claim(&checkpoint.id).await.unwrap().unwrap();
        assert!(
            graph
                .tag_checkpoint(&checkpoint.id, Some("late"), vec!["race"])
                .await
                .is_err()
        );
        assert!(store.load(&checkpoint.id).await.unwrap().is_some());
        let attempt_id = claimed.resume_attempt_id.as_deref().unwrap_or_default();
        store
            .requeue_claim(&checkpoint.id, attempt_id)
            .await
            .unwrap();
        let restored = store.load(&checkpoint.id).await.unwrap().unwrap();
        assert!(restored.label.is_none());
    }
}
