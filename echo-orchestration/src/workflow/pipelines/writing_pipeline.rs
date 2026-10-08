/*!
写作流水线——大纲 -> 草稿 -> 评审 -> 修订（带质量循环）

使用 `SharedAgent` 生成书面内容的四阶段图工作流。流水线包含一个条件循环：
评审阶段之后，若质量分低于配置阈值，则回到修订阶段（最多 `max_revisions` 次）。

# 示例

```ignore
// 示例:从 echo_agent 入口调用本 pipeline。本 doctest 标为 ignore,因为
// echo_orchestration 不能反向依赖 echo_agent(会循环依赖);真正的编译
// 验证在 echo_agent 的测试中通过 path 依赖覆盖。
use echo_agent::workflow::pipelines::writing_pipeline::{
    WritingPipelineConfig, run_writing_pipeline,
};
use echo_agent::workflow::SharedAgent;
use echo_agent::testing::MockAgent;

# async fn example() -> echo_core::error::Result<()> {
let agent = MockAgent::new("writer")
    .with_response("High-quality article about AI agents.");

let config = WritingPipelineConfig {
    topic: "The rise of AI agents".to_string(),
    audience: "technical professionals".to_string(),
    format: "blog post".to_string(),
    max_revisions: 2,
    quality_threshold: 80,
};

let result = run_writing_pipeline(&agent.into(), config).await?;
println!("Final output: {}", result.state.get::<String>("final_output").unwrap_or_default());
# Ok(())
# }
```
*/

use crate::workflow::SharedAgent;
use crate::workflow::graph::{Graph, GraphBuilder, GraphResult};
use crate::workflow::state::SharedState;
use echo_core::error::Result;

// ── 配置 ──────────────────────────────────────────────────────────────

/// 写作流水线的配置。
#[derive(Debug, Clone)]
pub struct WritingPipelineConfig {
    /// 写作的主题。
    pub topic: String,
    /// 内容的目标读者。
    pub audience: String,
    /// 期望的体裁（例如 "blog post"、"essay"、"report"、"white paper"）。
    pub format: String,
    /// 最大修订迭代次数（评审 -> 修订 循环）。
    pub max_revisions: u32,
    /** 质量分阈值（0-100）。评审分低于该值时流水线回到修订阶段；
    达到阈值或用尽 max_revisions 后，流水线进入定稿阶段。 */
    pub quality_threshold: u32,
}

impl Default for WritingPipelineConfig {
    /// 主题为空，其余字段取默认值。
    fn default() -> Self {
        Self {
            topic: String::new(),
            audience: "general readers".to_string(),
            format: "blog post".to_string(),
            max_revisions: 2,
            quality_threshold: 80,
        }
    }
}

impl WritingPipelineConfig {
    /// 以指定主题创建配置，其余字段取默认值。
    pub fn new(topic: impl Into<String>) -> Self {
        Self {
            topic: topic.into(),
            audience: "general readers".to_string(),
            format: "blog post".to_string(),
            max_revisions: 2,
            quality_threshold: 80,
        }
    }

    /// 设置目标读者。
    pub fn with_audience(mut self, audience: impl Into<String>) -> Self {
        self.audience = audience.into();
        self
    }

    /// 设置输出体裁。
    pub fn with_format(mut self, format: impl Into<String>) -> Self {
        self.format = format.into();
        self
    }

    /// 设置最大修订迭代次数。
    pub fn with_max_revisions(mut self, max: u32) -> Self {
        self.max_revisions = max;
        self
    }

    /// 设置质量分阈值。
    pub fn with_quality_threshold(mut self, threshold: u32) -> Self {
        self.quality_threshold = threshold;
        self
    }
}

// ── 质量分提取 ───────────────────────────────────────────────────

/**
从评审文本中提取质量分。

在行首查找 `QUALITY_SCORE: <number>` 模式；
若未找到精确模式，则回退到启发式扫描。
*/
fn extract_quality_score(review_text: &str) -> u32 {
    // 首选：查找 "QUALITY_SCORE: <number>" 模式
    for line in review_text.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("QUALITY_SCORE:") {
            let rest = rest.trim();
            if let Ok(score) = rest.parse::<u32>() {
                return score.min(100);
            }
            // 尝试仅提取开头的数字
            let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
            if let Ok(score) = digits.parse::<u32>() {
                return score.min(100);
            }
        }
    }

    // 启发式回退：查找 "Score: <number>" 模式
    for line in review_text.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("Score:")
            && let Ok(score) = rest.trim().parse::<u32>()
        {
            return score.min(100);
        }
    }

    // 默认：若未找到分数，按中等质量（60）处理
    tracing::warn!(
        pipeline = "writing",
        "Could not extract quality score from review text; defaulting to 60"
    );
    60
}

// ── 流水线阶段 ────────────────────────────────────────────────────────────

/**
构建写作图。

构造一个带条件循环的流水线：

```text
init -> outline -> draft -> review -> evaluate ─┬─► finalize (quality >= threshold or max_revisions)
                                                 │
                                                 └─► revise → increment → review (loop back)
```

所有配置值和提示词模板都通过 `init` 节点注入状态，
因此下游闭包只需从状态中读取。
*/
fn build_writing_graph(agent: &SharedAgent) -> Result<Graph> {
    let agent_clone = agent.clone();

    let graph = GraphBuilder::new("writing_pipeline")
        // ── 初始化：将配置值和提示词模板存入状态 ──
        .add_function_node("init", |state: &SharedState| {
            Box::pin(async move {
                // 配置值已由 run_writing_pipeline() 预先写入状态。
                let topic: String = state.get("topic").unwrap_or_default();
                let audience: String = state
                    .get("audience")
                    .unwrap_or_else(|| "general readers".to_string());
                let format: String = state
                    .get("format")
                    .unwrap_or_else(|| "blog post".to_string());

                // 为下游节点存储提示词模板
                let _ = state.set(
                    "tpl_outline",
                    format!(
                        "You are an expert content planner. Create a detailed outline for a {} \
                         on the topic '{}' targeted at {}. \
                         Include: title, sections with key points, and logical flow. \
                         Output the outline as structured text.",
                        format, topic, audience,
                    ),
                );
                let _ = state.set(
                    "tpl_draft",
                    format!(
                        "You are a skilled writer. Based on the outline provided, write a complete \
                         {} on '{}' for {}. Follow the outline structure closely. \
                         Write in a clear, engaging style appropriate for the audience. \
                         Output the full draft.",
                        format, topic, audience,
                    ),
                );
                let _ = state.set(
                    "tpl_review",
                    "You are a critical reviewer. Review the draft provided and evaluate it on: \
                         clarity, coherence, accuracy, audience fit, and overall quality. \
                         Score the draft from 0 to 100. \
                         At the very beginning of your response, output exactly: \
                         QUALITY_SCORE: <number> \
                         Then provide specific, actionable feedback for improvement. \
                         Output the review with quality score."
                        .to_string(),
                );
                let _ = state.set(
                    "tpl_revise",
                    format!(
                        "You are a revision specialist. Based on the draft and review feedback \
                         provided, revise the {} on '{}' to address all the reviewer's concerns. \
                         Improve clarity, coherence, accuracy, and audience fit. \
                         Output the revised version of the full content.",
                        format, topic,
                    ),
                );
                let _ = state.set(
                    "tpl_finalize",
                    format!(
                        "You are a final editor. Polish the content provided into a final, \
                         publication-ready {} on '{}' for {}. \
                         Fix any remaining grammar, style, or formatting issues. \
                         Output the final polished version.",
                        format, topic, audience,
                    ),
                );
                Ok(())
            })
        })
        // ── 阶段 1：大纲 ──
        .add_function_node("outline_prompt", |state: &SharedState| {
            Box::pin(async move {
                let tpl: String = state.get("tpl_outline").unwrap_or_default();
                let _ = state.set("outline_prompt", tpl);
                Ok(())
            })
        })
        .add_shared_agent_node_with_mode(
            "outline",
            agent_clone.clone(),
            "outline_prompt",
            "outline",
            false, // 对话模式
        )
        // ── 阶段 2：草稿 ──
        .add_function_node("draft_prompt", |state: &SharedState| {
            Box::pin(async move {
                let tpl: String = state.get("tpl_draft").unwrap_or_default();
                let outline_text: String = state.get("outline").unwrap_or_default();
                let prompt = format!(
                    "{}\n\nHere is the outline to follow:\n{}",
                    tpl, outline_text,
                );
                let _ = state.set("draft_prompt", prompt);
                Ok(())
            })
        })
        .add_shared_agent_node_with_mode(
            "draft",
            agent_clone.clone(),
            "draft_prompt",
            "draft",
            false,
        )
        // ── 阶段 3：评审 ──
        .add_function_node("review_prompt", |state: &SharedState| {
            Box::pin(async move {
                let tpl: String = state.get("tpl_review").unwrap_or_default();
                let draft_text: String = state.get("draft").unwrap_or_default();
                let prompt = format!("{}\n\nHere is the draft to review:\n{}", tpl, draft_text,);
                let _ = state.set("review_prompt", prompt);
                Ok(())
            })
        })
        .add_shared_agent_node_with_mode(
            "review",
            agent_clone.clone(),
            "review_prompt",
            "review",
            false,
        )
        // ── 根据评审评估质量 ──
        .add_function_node("evaluate_quality", |state: &SharedState| {
            Box::pin(async move {
                let review_text: String = state.get("review").unwrap_or_default();
                let score = extract_quality_score(&review_text);
                let _ = state.set("quality_score", score as i64);

                let revision_count: i64 = state.get("revision_count").unwrap_or(0);
                tracing::debug!(
                    pipeline = "writing",
                    quality_score = score,
                    revision_count = revision_count,
                    "Review quality evaluated"
                );
                Ok(())
            })
        })
        // ── 阶段 4：修订（经条件循环到达）──
        .add_function_node("revise_prompt", |state: &SharedState| {
            Box::pin(async move {
                let tpl: String = state.get("tpl_revise").unwrap_or_default();
                let draft_text: String = state.get("draft").unwrap_or_default();
                let review_text: String = state.get("review").unwrap_or_default();
                let prompt = format!(
                    "{}\n\nHere is the current draft:\n{}\n\nHere is the review feedback:\n{}",
                    tpl, draft_text, review_text,
                );
                let _ = state.set("revise_prompt", prompt);
                Ok(())
            })
        })
        .add_shared_agent_node_with_mode(
            "revise",
            agent_clone.clone(),
            "revise_prompt",
            "draft", // 用修订版本覆盖草稿
            false,
        )
        // ── 递增修订计数 ──
        .add_function_node("increment_revision", |state: &SharedState| {
            Box::pin(async move {
                let count: i64 = state.get("revision_count").unwrap_or(0);
                let new_count = count + 1;
                let _ = state.set("revision_count", new_count);
                tracing::info!(
                    pipeline = "writing",
                    revision = new_count,
                    "Revision iteration completed"
                );
                Ok(())
            })
        })
        // ── 阶段 5：定稿 ──
        .add_function_node("finalize_prompt", |state: &SharedState| {
            Box::pin(async move {
                let tpl: String = state.get("tpl_finalize").unwrap_or_default();
                let draft_text: String = state.get("draft").unwrap_or_default();
                let prompt = format!("{}\n\nHere is the content to polish:\n{}", tpl, draft_text,);
                let _ = state.set("finalize_prompt", prompt);
                Ok(())
            })
        })
        .add_shared_agent_node_with_mode(
            "finalize",
            agent_clone,
            "finalize_prompt",
            "final_output",
            false,
        )
        /*
        ── 边 ──
        线性路径：init -> outline -> draft -> review -> evaluate_quality
        */
        .set_entry("init")
        .add_edge("init", "outline_prompt")
        .add_edge("outline_prompt", "outline")
        .add_edge("outline", "draft_prompt")
        .add_edge("draft_prompt", "draft")
        .add_edge("draft", "review_prompt")
        .add_edge("review_prompt", "review")
        .add_edge("review", "evaluate_quality")
        // 条件分支：evaluate_quality -> finalize 或 revise
        .add_conditional_edge("evaluate_quality", |state: &SharedState| {
            Box::pin(async move {
                let quality_score: i64 = state.get("quality_score").unwrap_or(0);
                let revision_count: i64 = state.get("revision_count").unwrap_or(0);
                let threshold: i64 = state.get("quality_threshold").unwrap_or(80);
                let max_revs: i64 = state.get("max_revisions").unwrap_or(2);

                if quality_score >= threshold {
                    tracing::info!(
                        pipeline = "writing",
                        quality_score = quality_score,
                        threshold = threshold,
                        "Quality threshold met — proceeding to finalize"
                    );
                    "finalize_prompt".to_string()
                } else if revision_count < max_revs {
                    tracing::info!(
                        pipeline = "writing",
                        quality_score = quality_score,
                        threshold = threshold,
                        revision_count = revision_count,
                        "Quality below threshold — looping to revise"
                    );
                    "revise_prompt".to_string()
                } else {
                    tracing::info!(
                        pipeline = "writing",
                        quality_score = quality_score,
                        revision_count = revision_count,
                        max_revisions = max_revs,
                        "Max revisions reached — proceeding to finalize"
                    );
                    "finalize_prompt".to_string()
                }
            })
        })
        // 修订循环：revise -> increment_revision -> 重新评审
        .add_edge("revise_prompt", "revise")
        .add_edge("revise", "increment_revision")
        .add_edge("increment_revision", "review_prompt")
        // 定稿路径
        .add_edge("finalize_prompt", "finalize")
        .set_finish("finalize")
        .build()?;

    Ok(graph)
}

// ── 流水线执行 ─────────────────────────────────────────────────────────

/**
运行写作流水线。

返回的 [`GraphResult`] 包含最终的 [`SharedState`]，其中的键有：
- `outline` —— 内容大纲
- `draft` —— 当前草稿（可能经过多次修订）
- `review` —— 最新一次评审反馈（含质量分）
- `quality_score` —— 数值质量分（0-100）
- `revision_count` —— 已执行的修订迭代次数
- `final_output` —— 润色后的最终内容
*/
pub async fn run_writing_pipeline(
    agent: &SharedAgent,
    config: WritingPipelineConfig,
) -> Result<GraphResult> {
    let graph = build_writing_graph(agent)?;
    let state = SharedState::new();

    tracing::info!(
        pipeline = "writing",
        topic = %config.topic,
        audience = %config.audience,
        format = %config.format,
        max_revisions = config.max_revisions,
        quality_threshold = config.quality_threshold,
        "Starting writing pipeline"
    );

    /* 在图执行开始前将配置值存入状态。
    init 节点会读取这些值来构建提示词模板。 */
    let _ = state.set("topic", config.topic);
    let _ = state.set("audience", config.audience);
    let _ = state.set("format", config.format);
    let _ = state.set("revision_count", 0i64);
    let _ = state.set("max_revisions", config.max_revisions as i64);
    let _ = state.set("quality_threshold", config.quality_threshold as i64);

    let result = graph.run(state).await?;

    tracing::info!(
        pipeline = "writing",
        steps = result.steps,
        path = ?result.path,
        "Writing pipeline completed"
    );

    Ok(result)
}

// ── 单元测试 ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workflow::shared_agent;

    /// 始终返回固定响应的写作模拟 Agent。
    struct WriterMock {
        /// Agent 名称
        name: String,
        /// 每次执行返回的固定响应
        response: String,
    }

    impl WriterMock {
        /// 以名称和固定响应创建模拟 Agent。
        fn new(name: &str, response: &str) -> Self {
            Self {
                name: name.to_string(),
                response: response.to_string(),
            }
        }
    }

    impl echo_core::agent::Agent for WriterMock {
        fn name(&self) -> &str {
            &self.name
        }

        fn model_name(&self) -> &str {
            "mock-model"
        }

        fn system_prompt(&self) -> &str {
            "You are a mock writer"
        }

        fn execute<'a>(&'a self, _task: &'a str) -> futures::future::BoxFuture<'a, Result<String>> {
            Box::pin(async move { Ok(self.response.clone()) })
        }

        fn execute_stream<'a>(
            &'a self,
            _task: &'a str,
        ) -> futures::future::BoxFuture<
            'a,
            Result<futures::stream::BoxStream<'a, Result<echo_core::agent::AgentEvent>>>,
        > {
            Box::pin(async move {
                let s: futures::stream::BoxStream<'a, Result<echo_core::agent::AgentEvent>> =
                    Box::pin(futures::stream::iter([Ok(
                        echo_core::agent::AgentEvent::FinalAnswer(self.response.clone()),
                    )]));
                Ok(s)
            })
        }
    }

    #[test]
    fn test_extract_quality_score_explicit() {
        let text = "QUALITY_SCORE: 85\nGood work on clarity.";
        assert_eq!(extract_quality_score(text), 85);
    }

    #[test]
    fn test_extract_quality_score_with_extra_whitespace() {
        let text = "QUALITY_SCORE:   72\nSome feedback.";
        assert_eq!(extract_quality_score(text), 72);
    }

    #[test]
    fn test_extract_quality_score_score_prefix() {
        let text = "Score: 65\nNeeds improvement.";
        assert_eq!(extract_quality_score(text), 65);
    }

    #[test]
    fn test_extract_quality_score_clamped_to_100() {
        let text = "QUALITY_SCORE: 150\nOver-scored.";
        assert_eq!(extract_quality_score(text), 100);
    }

    #[test]
    fn test_extract_quality_score_default_fallback() {
        let text = "This is a review without any score marker.";
        assert_eq!(extract_quality_score(text), 60);
    }

    #[tokio::test]
    async fn test_writing_pipeline_no_loop() {
        // 评审返回高质量分——应跳过修订循环
        let agent = shared_agent(WriterMock::new(
            "writer",
            "QUALITY_SCORE: 95\nExcellent writing quality.",
        ));

        let config = WritingPipelineConfig {
            topic: "AI agents".to_string(),
            audience: "developers".to_string(),
            format: "blog post".to_string(),
            max_revisions: 3,
            quality_threshold: 80,
        };

        let result = run_writing_pipeline(&agent, config).await.unwrap();

        // 应在不经过修订循环的情况下到达定稿
        assert!(result.path.contains(&"finalize".to_string()));
        // revision_count 应保持为 0
        let rev_count: i64 = result.state.get("revision_count").unwrap_or(0);
        assert_eq!(rev_count, 0);
        assert!(result.state.contains("final_output"));
    }

    #[tokio::test]
    async fn test_writing_pipeline_with_loop() {
        // 评审返回低质量分——应触发修订循环
        let agent = shared_agent(WriterMock::new(
            "writer",
            "QUALITY_SCORE: 50\nNeeds significant improvement.",
        ));

        let config = WritingPipelineConfig {
            topic: "Testing patterns".to_string(),
            audience: "engineers".to_string(),
            format: "essay".to_string(),
            max_revisions: 2,
            quality_threshold: 80,
        };

        let result = run_writing_pipeline(&agent, config).await.unwrap();

        // 应在修订循环后到达定稿
        assert!(result.path.contains(&"finalize".to_string()));
        // 由于模拟始终返回相同分数，revision_count 应达到 max_revisions
        let rev_count: i64 = result.state.get("revision_count").unwrap_or(0);
        assert_eq!(rev_count, 2);
    }

    #[tokio::test]
    async fn test_writing_pipeline_default_config() {
        let agent = shared_agent(WriterMock::new("writer", "QUALITY_SCORE: 90\nGreat work."));

        let config = WritingPipelineConfig::new("Rust programming");
        assert_eq!(config.topic, "Rust programming");
        assert_eq!(config.audience, "general readers");
        assert_eq!(config.format, "blog post");
        assert_eq!(config.max_revisions, 2);
        assert_eq!(config.quality_threshold, 80);

        let result = run_writing_pipeline(&agent, config).await.unwrap();
        assert!(result.state.contains("final_output"));
    }
}
