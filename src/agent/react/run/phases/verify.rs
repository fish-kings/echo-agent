/*!
验证器（Verifier）逻辑与纯文本回答分支。

`verify_answer` 会调用配置好的 `Critic` 进行校验；若被拒绝，则将反馈信息
注入上下文。`verify_final_text` 封装了单次迭代中纯文本分支的处理流程：
先校验，通过后返回 `IterOutcome::FinalText`，供驱动器调用
`phases::finalize::emit_final_text`。
*/

use super::{IterOutcome, LoopState, ThinkOutput, with_reasoning_content};
use crate::agent::AgentEvent;
use crate::agent::snapshot::AgentRunSnapshot;
use crate::error::Result;
use crate::llm::types::Message;
use std::sync::Arc;
use tokio::sync::{Mutex, mpsc};

/**
使用配置好的 Critic 校验最终答案。

如果答案通过校验（或校验被禁用），返回 `true`；
如果答案未通过校验，反馈已被注入上下文以便自我修正，返回 `false`。
*/
pub(crate) async fn verify_answer(
    snap: &AgentRunSnapshot,
    context: &Arc<Mutex<crate::compression::ContextManager>>,
    answer: &str,
    retry_count: usize,
) -> bool {
    // 如果校验器被禁用或未配置 critic，则跳过
    if !snap.config.verifier_enabled {
        return true;
    }
    let Some(ref critic) = snap.critic else {
        return true;
    };
    // 不超过最大重试次数（但始终允许第一次检查）
    if retry_count > 0 && retry_count >= snap.config.verifier_max_retries {
        tracing::info!(
            retries = retry_count,
            "Verifier max retries reached, accepting answer"
        );
        return true;
    }

    let task_description = {
        let ctx = context.lock().await;
        // 使用最后一条用户消息作为任务描述
        ctx.messages()
            .iter()
            .rev()
            .find(|m| m.role == echo_core::llm::types::Role::User)
            .map(|m| {
                m.content
                    .as_text()
                    .map(|s| s.to_string())
                    .unwrap_or_default()
            })
            .unwrap_or_default()
    };

    match critic.critique(&task_description, answer, "").await {
        Ok(critique) => {
            if critique.passed || critique.score >= snap.config.verifier_min_score {
                tracing::debug!(score = critique.score, "Verifier passed, accepting answer");
                true
            } else {
                let feedback = format!(
                    "[Verifier feedback] Score: {}/10 (min: {}). {}\nSuggestions: {}",
                    critique.score,
                    snap.config.verifier_min_score,
                    critique.feedback,
                    if critique.suggestions.is_empty() {
                        "N/A".to_string()
                    } else {
                        critique.suggestions.join(", ")
                    },
                );
                tracing::info!(
                    score = critique.score,
                    retry = retry_count + 1,
                    max = snap.config.verifier_max_retries,
                    "Verifier rejected answer, injecting feedback for self-correction"
                );
                super::super::context::push_runtime_context_note(
                    context,
                    "VerifierFeedback",
                    &feedback,
                )
                .await;
                false
            }
        }
        Err(e) => {
            tracing::warn!(error = %e, "Verifier critique failed, accepting answer");
            true // 失败即放行：如果 critique 本身出错，则直接接受答案
        }
    }
}

/**
单次迭代中纯文本分支的处理逻辑。校验缓冲的 LLM 内容；
若校验失败，则将本次助手的回答推入上下文，`state.verifier_retry_count`
自增，并返回 [`IterOutcome::Continue`]，使循环驱动一次自我修正迭代。
若校验成功，则返回携带答案的 [`IterOutcome::FinalText`] —— 驱动器会
调用 `emit_final_text` 来处理审计、checkpoint、transcript 投影、
`finalize_run`、`FinalAnswer` 事件以及 `Stop` 钩子的后续流程。
*/
#[allow(clippy::too_many_arguments)]
pub(crate) async fn verify_final_text(
    snap: &AgentRunSnapshot,
    context: &Arc<Mutex<crate::compression::ContextManager>>,
    _tx: &mpsc::Sender<Result<AgentEvent>>,
    state: &mut LoopState,
    _iteration: usize,
    think: ThinkOutput,
    _label: &str,
) -> Result<IterOutcome> {
    let content_buffer = think.content_buffer;
    let reasoning_content = think.reasoning_buffer;
    let reasoning_blocks = think.reasoning_blocks;
    if !verify_answer(snap, context, &content_buffer, state.verifier_retry_count).await {
        // 把 LLM 的回答推入上下文，以便它能看到自己之前的尝试
        context.lock().await.push(with_reasoning_content(
            Message::assistant(content_buffer),
            reasoning_content,
            reasoning_blocks,
        ));
        state.verifier_retry_count += 1;
        return Ok(IterOutcome::Continue);
    }

    Ok(IterOutcome::FinalText {
        answer: content_buffer,
        reasoning_content,
        reasoning_blocks,
    })
}

// ── 测试 ────────────────────────────────────────────────────────────

/// 验证阶段的单元测试。
#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::ReactAgent;
    use crate::agent::config::AgentConfig;
    use echo_core::agent::StaticCritic;

    /**
     * `verifier_enabled = false`（默认值）会无条件接受任何答案，
     * 不会触碰 critic 或上下文。
     */
    #[tokio::test]
    async fn verify_answer_disabled_returns_true() {
        let config = AgentConfig::new("test-model", "agent", "sys");
        let agent = ReactAgent::new(config);
        let snap = AgentRunSnapshot::from_agent(&agent);

        assert!(!snap.config.verifier_enabled);
        let accepted = verify_answer(&snap, &agent.memory.context, "any answer", 0).await;
        assert!(
            accepted,
            "verifier disabled should fail-open without inspecting the critic",
        );
    }

    /// `verifier_enabled = true` 但未配置 critic 时，同样接受任何答案。
    #[tokio::test]
    async fn verify_answer_no_critic_returns_true() {
        let config = AgentConfig::new("test-model", "agent", "sys").verifier_enabled(true);
        let agent = ReactAgent::new(config);
        let snap = AgentRunSnapshot::from_agent(&agent);

        assert!(snap.critic.is_none());
        let accepted = verify_answer(&snap, &agent.memory.context, "answer", 0).await;
        assert!(accepted, "no critic should fail-open with true");
    }

    /**
    第一次校验失败时，反馈会被注入上下文，答案被拒绝。
    重试计数从 0 开始，所以“已达最大重试次数”的短路逻辑不应触发。
    */
    #[tokio::test]
    async fn verify_answer_low_score_pushes_feedback_and_rejects() {
        let config = AgentConfig::new("test-model", "agent", "sys")
            .verifier_enabled(true)
            .verifier_min_score(7.0);
        let mut agent = ReactAgent::new(config);
        agent.set_critic(Arc::new(StaticCritic::always_fail()));
        let snap = AgentRunSnapshot::from_agent(&agent);

        let before = agent.memory.context.lock().await.messages().len();
        let accepted = verify_answer(&snap, &agent.memory.context, "bad answer", 0).await;
        let after = agent.memory.context.lock().await.messages().len();

        assert!(!accepted, "low-score critique should reject the answer");
        assert_eq!(
            after,
            before + 1,
            "rejection should inject one feedback message into context",
        );
        let last = agent.memory.context.lock().await.messages().last().cloned();
        let text = last
            .as_ref()
            .and_then(|m| m.content.as_text().map(|s| s.to_string()))
            .unwrap_or_default();
        assert!(
            text.starts_with("[runtime_context:VerifierFeedback]")
                && text.contains("[Verifier feedback]"),
            "injected feedback should be runtime context, got: {text:?}",
        );
    }

    /**
     * 一旦 `retry_count` 达到 `verifier_max_retries`，即使 critic 持续判定失败，
     * 也会直接接受答案（避免无限的自我修正循环）。
     */
    #[tokio::test]
    async fn verify_answer_max_retries_short_circuits_to_true() {
        let config = AgentConfig::new("test-model", "agent", "sys")
            .verifier_enabled(true)
            .verifier_max_retries(2);
        let mut agent = ReactAgent::new(config);
        agent.set_critic(Arc::new(StaticCritic::always_fail()));
        let snap = AgentRunSnapshot::from_agent(&agent);

        let accepted = verify_answer(&snap, &agent.memory.context, "answer", 2).await;
        assert!(
            accepted,
            "retry_count >= max_retries should accept without consulting the critic",
        );
    }
}
