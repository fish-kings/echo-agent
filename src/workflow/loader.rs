/*!
 * 声明式工作流加载器
 *
 * 支持从 YAML / JSON 文件加载图（graph）工作流定义，降低
 * 非 Rust 用户的使用门槛。
 *
 * # YAML 格式示例
 *
 * ```yaml
 * name: my_workflow
 * nodes:
 *   - name: researcher
 *     type: agent
 *     model: qwen3-max
 *     system_prompt: "You are a research assistant"
 *     input_key: task
 *     output_key: research
 *   - name: writer
 *     type: agent
 *     model: qwen3-max
 *     system_prompt: "You are a writing assistant"
 *     input_key: research
 *     output_key: result
 * edges:
 *   - from: researcher
 *     to: writer
 * entry: researcher
 * finish:
 *   - writer
 * max_steps: 50
 * ```
 *
 * # JSON 格式示例
 *
 * ```json
 * {
 *   "name": "my_workflow",
 *   "nodes": [
 *     { "name": "researcher", "type": "agent", "model": "qwen3-max",
 *       "system_prompt": "...", "input_key": "task", "output_key": "research" }
 *   ],
 *   "edges": [
 *     { "from": "researcher", "to": "writer" }
 *   ],
 *   "entry": "researcher",
 *   "finish": ["writer"]
 * }
 * ```
 */

use crate::agent::react::builder::ReactAgentBuilder;
use crate::error::{AgentError, ReactError, Result};
use crate::llm::LlmConfig;
use crate::workflow::Graph;
use crate::workflow::GraphBuilder;
use crate::workflow::SharedState;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// 工作流的声明式定义
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct WorkflowDefinition {
    /// 工作流名称
    pub name: String,
    /// 节点定义列表
    pub nodes: Vec<NodeDefinition>,
    /// 边定义列表
    pub edges: Vec<EdgeDefinition>,
    /// 入口节点名称
    pub entry: String,
    /// 终止节点名称列表
    #[serde(default)]
    pub finish: Vec<String>,
    /**
     * 在进入这些节点之前暂停（`*` 表示全部）；启用在已构建图上
     * 使用 `run_until_interrupt` + `resume`。
     */
    #[serde(default)]
    pub interrupt_before: Vec<String>,
    /// 在这些节点执行之后暂停（`*` 表示全部）。
    #[serde(default)]
    pub interrupt_after: Vec<String>,
    /// 最大执行步数（可选，默认 100）
    #[serde(default)]
    pub max_steps: Option<usize>,
}

/// 节点定义
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct NodeDefinition {
    /// 节点名称
    pub name: String,
    /// 节点类型："agent" | "function" | "router"
    #[serde(rename = "type")]
    pub node_type: String,
    /// Agent 模型名称（仅在 type=agent 时使用）
    #[serde(default)]
    pub model: Option<String>,
    /// Agent 系统提示词（仅在 type=agent 时使用）
    #[serde(default)]
    pub system_prompt: Option<String>,
    /// 从 state 中读取输入的键（仅在 type=agent 时使用，默认为 "input"）
    #[serde(default = "default_input_key")]
    pub input_key: String,
    /// 将输出写入 state 的键（仅在 type=agent 时使用，默认为 "output"）
    #[serde(default = "default_output_key")]
    pub output_key: String,
}

/// 默认的输入键
fn default_input_key() -> String {
    "input".to_string()
}
/// 默认的输出键
fn default_output_key() -> String {
    "output".to_string()
}

/// 边定义
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct EdgeDefinition {
    /// 源节点
    pub from: String,
    /// 目标节点（固定边）
    #[serde(default)]
    pub to: Option<String>,
    /// 条件表达式（条件边）—— 基于 state 键的简单比较
    #[serde(default)]
    pub condition: Option<ConditionDefinition>,
    /// 并行目标节点列表（扇出边）
    #[serde(default)]
    pub parallel: Option<Vec<String>>,
    /// 并行汇聚后的下一个节点
    #[serde(default)]
    pub then: Option<String>,
}

/// 条件定义（简化版：基于 state 键的值比较）
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ConditionDefinition {
    /// state 中的键
    pub key: String,
    /// 期望的值
    pub equals: serde_json::Value,
    /// 匹配时跳转到的节点
    pub then: String,
    /// 不匹配时跳转到的节点
    #[serde(rename = "else")]
    pub else_node: String,
}

impl WorkflowDefinition {
    /// 从 YAML 文件加载
    pub fn from_yaml(path: impl AsRef<Path>) -> Result<Self> {
        let content = std::fs::read_to_string(path.as_ref()).map_err(|e| {
            ReactError::Agent(Box::new(AgentError::InitializationFailed(format!(
                "Failed to read workflow YAML file: {e}"
            ))))
        })?;
        Self::from_yaml_str(&content)
    }

    /// 从 YAML 字符串解析
    pub fn from_yaml_str(yaml: &str) -> Result<Self> {
        serde_yaml_ng::from_str(yaml).map_err(|e| {
            ReactError::Agent(Box::new(AgentError::InitializationFailed(format!(
                "Failed to parse workflow YAML: {e}"
            ))))
        })
    }

    /// 从 JSON 文件加载
    pub fn from_json(path: impl AsRef<Path>) -> Result<Self> {
        let content = std::fs::read_to_string(path.as_ref()).map_err(|e| {
            ReactError::Agent(Box::new(AgentError::InitializationFailed(format!(
                "Failed to read workflow JSON file: {e}"
            ))))
        })?;
        Self::from_json_str(&content)
    }

    /// 从 JSON 字符串解析
    pub fn from_json_str(json: &str) -> Result<Self> {
        serde_json::from_str(json).map_err(|e| {
            ReactError::Agent(Box::new(AgentError::InitializationFailed(format!(
                "Failed to parse workflow JSON: {e}"
            ))))
        })
    }

    /// 构建 Graph（不注入 LLM 配置，Agent 使用环境变量配置）
    pub fn build_graph(self) -> Result<Graph> {
        self.build_graph_with_llm_config(None)
    }

    /// 构建 Graph（可选地注入 LLM 配置）
    pub fn build_graph_with_llm_config(self, llm_config: Option<&LlmConfig>) -> Result<Graph> {
        self.build_graph_with(llm_config, None)
    }

    /**
     * 使用显式注入的 [`LlmClient`] 构建 Graph：agent 节点会复用
     * 调用方持有的 client（例如 Session Agent 自身的 provider），
     * 而不是从配置中重新构造一个。
     */
    pub fn build_graph_with_client(
        self,
        llm_client: Option<std::sync::Arc<dyn crate::llm::LlmClient>>,
    ) -> Result<Graph> {
        self.build_graph_with(None, llm_client)
    }

    /// 根据可选的 LLM 配置或 client 构建 Graph 的共享实现
    fn build_graph_with(
        self,
        llm_config: Option<&LlmConfig>,
        llm_client: Option<std::sync::Arc<dyn crate::llm::LlmClient>>,
    ) -> Result<Graph> {
        let mut builder = GraphBuilder::new(&self.name);

        for node_def in &self.nodes {
            match node_def.node_type.as_str() {
                "agent" => {
                    let model = node_def.model.as_deref().unwrap_or("qwen3-max");
                    let prompt = node_def
                        .system_prompt
                        .as_deref()
                        .unwrap_or(crate::config::DEFAULT_AGENT_SYSTEM_PROMPT);

                    let mut agent_builder = ReactAgentBuilder::new()
                        .name(&node_def.name)
                        .model(model)
                        .system_prompt(prompt);

                    if let Some(client) = &llm_client {
                        agent_builder = agent_builder.llm_client(client.clone());
                    } else if let Some(config) = llm_config {
                        agent_builder = agent_builder.llm_config(config.clone());
                    }

                    let agent = agent_builder.build()?;
                    builder = builder.add_agent_node(
                        &node_def.name,
                        agent,
                        &node_def.input_key,
                        &node_def.output_key,
                    );
                }
                "router" => {
                    builder = builder.add_router_node(&node_def.name);
                }
                "function" => {
                    return Err(ReactError::Agent(Box::new(
                        AgentError::InitializationFailed(format!(
                            "Node type 'function' is not yet supported for node '{}'. \
                             Use 'agent' or 'router' instead, or register a function node manually.",
                            node_def.name
                        )),
                    )));
                }
                other => {
                    return Err(ReactError::Agent(Box::new(
                        AgentError::InitializationFailed(format!(
                            "Unknown node type '{}' for node '{}'",
                            other, node_def.name
                        )),
                    )));
                }
            }
        }

        for edge_def in &self.edges {
            if let Some(ref to) = edge_def.to {
                builder = builder.add_edge(&edge_def.from, to);
            } else if let Some(ref cond) = edge_def.condition {
                let key = cond.key.clone();
                let expected = cond.equals.clone();
                let then = cond.then.clone();
                let else_node = cond.else_node.clone();

                builder =
                    builder.add_conditional_edge(&edge_def.from, move |state: &SharedState| {
                        let key = key.clone();
                        let expected = expected.clone();
                        let then = then.clone();
                        let else_node = else_node.clone();
                        Box::pin(async move {
                            let actual = state.get_raw(&key);
                            if actual.as_ref() == Some(&expected) {
                                then
                            } else {
                                else_node
                            }
                        })
                    });
            } else if let Some(ref targets) = edge_def.parallel {
                let then = edge_def
                    .then
                    .clone()
                    .unwrap_or_else(|| "__end__".to_string());
                builder = builder.add_parallel_edge(&edge_def.from, targets.clone(), then);
            }
        }

        builder = builder.set_entry(&self.entry);
        for finish in &self.finish {
            builder = builder.set_finish(finish);
        }
        if !self.interrupt_before.is_empty() {
            let nodes: Vec<&str> = self.interrupt_before.iter().map(String::as_str).collect();
            builder = builder.interrupt_before(nodes);
        }
        if !self.interrupt_after.is_empty() {
            let nodes: Vec<&str> = self.interrupt_after.iter().map(String::as_str).collect();
            builder = builder.interrupt_after(nodes);
        }

        let mut graph = builder.build()?;
        if let Some(max) = self.max_steps {
            graph.set_max_steps(max);
        }

        Ok(graph)
    }
}

/**
 * 从 YAML 文件加载图工作流
 *
 * # 示例
 *
 * ```rust,no_run
 * use echo_agent::workflow::loader::load_graph_from_yaml;
 *
 * # fn main() -> echo_agent::error::Result<()> {
 * let graph = load_graph_from_yaml("workflow.yaml")?;
 * # Ok(())
 * # }
 * ```
 */
pub fn load_graph_from_yaml(path: impl AsRef<Path>) -> Result<crate::workflow::Graph> {
    WorkflowDefinition::from_yaml(path)?.build_graph()
}

/// 从 JSON 文件加载图工作流
pub fn load_graph_from_json(path: impl AsRef<Path>) -> Result<crate::workflow::Graph> {
    WorkflowDefinition::from_json(path)?.build_graph()
}

/// 从 YAML 字符串加载图工作流
pub fn load_graph_from_yaml_str(yaml: &str) -> Result<crate::workflow::Graph> {
    WorkflowDefinition::from_yaml_str(yaml)?.build_graph()
}

/// 从 JSON 字符串加载图工作流
pub fn load_graph_from_json_str(json: &str) -> Result<crate::workflow::Graph> {
    WorkflowDefinition::from_json_str(json)?.build_graph()
}

/// 工作流加载器的单元测试。
#[cfg(test)]
mod tests {
    use super::*;

    /// 验证中断（interrupt_before / interrupt_after）定义可被正确解析。
    #[test]
    fn test_interrupt_definition() {
        let json = r#"{
            "name": "gated_flow",
            "nodes": [
                { "name": "a", "type": "router" },
                { "name": "b", "type": "router" }
            ],
            "edges": [ { "from": "a", "to": "b" } ],
            "entry": "a",
            "finish": ["b"],
            "interrupt_before": ["b"]
        }"#;
        let def = WorkflowDefinition::from_json_str(json).unwrap();
        assert_eq!(def.interrupt_before, vec!["b".to_string()]);
        assert!(def.interrupt_after.is_empty());
        let graph = def.build_graph().unwrap();
        // 中断会传递到已编译的图：运行会在 b 之前暂停。
        let state = SharedState::new();
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        let outcome = runtime.block_on(graph.run_until_interrupt(state)).unwrap();
        assert!(matches!(
            outcome,
            echo_orchestration::workflow::RunUntilInterruptResult::Interrupted(_)
        ));
    }

    /// 验证可从 YAML 字符串解析工作流定义。
    #[test]
    fn test_parse_yaml_definition() {
        let yaml = r#"
name: test_workflow
nodes:
  - name: start
    type: router
  - name: subagent
    type: agent
    model: qwen3-max
    system_prompt: "You are an assistant"
    input_key: task
    output_key: result
edges:
  - from: start
    to: subagent
  - from: subagent
    to: __end__
entry: start
finish: []
max_steps: 50
"#;

        let def = WorkflowDefinition::from_yaml_str(yaml).unwrap();
        assert_eq!(def.name, "test_workflow");
        assert_eq!(def.nodes.len(), 2);
        assert_eq!(def.edges.len(), 2);
        assert_eq!(def.entry, "start");
        assert_eq!(def.max_steps, Some(50));
    }

    /// 验证可从 JSON 字符串解析工作流定义。
    #[test]
    fn test_parse_json_definition() {
        let json = r#"{
            "name": "json_flow",
            "nodes": [
                { "name": "n1", "type": "router", "input_key": "input", "output_key": "output" },
                { "name": "n2", "type": "router", "input_key": "input", "output_key": "output" }
            ],
            "edges": [
                { "from": "n1", "to": "n2" }
            ],
            "entry": "n1",
            "finish": ["n2"]
        }"#;

        let def = WorkflowDefinition::from_json_str(json).unwrap();
        assert_eq!(def.name, "json_flow");
        assert_eq!(def.nodes.len(), 2);
    }

    /// 验证可由 YAML 定义构建出图。
    #[test]
    fn test_build_graph_from_yaml() {
        let yaml = r#"
name: simple_graph
nodes:
  - name: hub
    type: router
  - name: end_node
    type: router
edges:
  - from: hub
    to: end_node
entry: hub
finish:
  - end_node
"#;
        let graph = load_graph_from_yaml_str(yaml).unwrap();
        assert_eq!(graph.name, "simple_graph");
    }

    /// 验证条件边定义可被正确解析与构建。
    #[test]
    fn test_conditional_edge_definition() {
        let yaml = r#"
name: cond_flow
nodes:
  - name: check
    type: router
  - name: yes_path
    type: router
  - name: no_path
    type: router
edges:
  - from: check
    condition:
      key: approved
      equals: true
      then: yes_path
      else: no_path
entry: check
finish:
  - yes_path
  - no_path
"#;
        let graph = load_graph_from_yaml_str(yaml).unwrap();
        assert_eq!(graph.name, "cond_flow");
    }

    /// 验证并行边定义可被正确解析与构建。
    #[test]
    fn test_parallel_edge_definition() {
        let yaml = r#"
name: parallel_flow
nodes:
  - name: start
    type: router
  - name: branch_a
    type: router
  - name: branch_b
    type: router
  - name: merge
    type: router
edges:
  - from: start
    parallel:
      - branch_a
      - branch_b
    then: merge
entry: start
finish:
  - merge
"#;
        let graph = load_graph_from_yaml_str(yaml).unwrap();
        assert_eq!(graph.name, "parallel_flow");
    }
}
