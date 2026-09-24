//! 每轮迭代前的准备工作：审计用户输入并运行 `UserPromptSubmit` 钩子。

use super::super::stream_macros::{yield_event_or, yield_final_event_or};
use super::super::types::StreamMode;
use super::PrepareOutcome;
use crate::agent::AgentEvent;
use crate::agent::snapshot::AgentRunSnapshot;
use crate::error::Result;
use std::sync::Arc;
use tokio::sync::{Mutex, mpsc};
use tracing::info;

/**
一次性的循环前准备工作：发出 `MemoryRecalled` 事件、审计用户输入，
并运行 `UserPromptSubmit` 生命周期钩子。

如果准备过程中接收端已关闭，则返回 `Ok(PrepareOutcome::Abandoned)`，
循环驱动器直接返回 `Ok(())`。
*/
#[allow(clippy::too_many_arguments)]
pub(crate) async fn prepare_turn(
    snap: &AgentRunSnapshot,
    context: &Arc<Mutex<crate::compression::ContextManager>>,
    tx: &mpsc::Sender<Result<AgentEvent>>,
    text: &str,
    label: &str,
    mode: StreamMode,
    recalled: usize,
    user_prompt_hook_already_run: bool,
) -> Result<PrepareOutcome> {
    let agent = &snap.config.agent_name;
    match mode {
        StreamMode::Execute => info!(agent = %agent, "Agent streaming task execution{label}"),
        StreamMode::Chat => info!(agent = %agent, "Agent streaming conversation{label}"),
    }
    if recalled > 0 {
        yield_event_or!(
            tx,
            AgentEvent::MemoryRecalled { count: recalled },
            PrepareOutcome::Abandoned
        );
    }

    // 审计：用户输入
    if snap.guard.audit_logger.is_some() {
        let event = crate::audit::AuditEvent::now(
            snap.config.session_id.clone(),
            snap.config.agent_name.clone(),
            crate::audit::AuditEventType::UserInput {
                content: text.to_string(),
            },
        );
        snap.record_audit_event(event).await;
    }

    /* 流式准备阶段会在路由之前运行该钩子；核心逻辑显式接收这一事实，
    从而确保生命周期副作用只会发生一次。 */
    if !user_prompt_hook_already_run {
        let hook_ctx = crate::skills::hooks::HookContext::for_user_prompt_submit(
            text,
            None,
            snap.config.session_id.as_deref().unwrap_or(""),
            &snap.config.agent_name,
        );
        let registry = snap.tools.hook_registry.read().await.clone();
        let result = registry.run_lifecycle_hooks(&hook_ctx).await;
        if result.block {
            let reason = result.block_reason.unwrap_or_default();
            super::finalize::settle_terminal_projection(
                snap,
                context,
                Some(format!("Blocked by UserPromptSubmit hook: {reason}")),
                tx,
            )
            .await?;
            snap.finalize_run(
                crate::trace::RunStatus::Failed,
                None,
                Some(&format!("Blocked by UserPromptSubmit hook: {reason}")),
            )
            .await;
            yield_final_event_or!(
                tx,
                AgentEvent::FinalAnswer(format!("Blocked by UserPromptSubmit hook: {}", reason)),
                PrepareOutcome::BlockedAndDone
            );
            snap.fire_hook(crate::skills::hooks::HookEvent::SessionEnd, Some("blocked"))
                .await;
            return Ok(PrepareOutcome::BlockedAndDone);
        }
        if let Some(ctx) = &result.injected_context {
            super::super::context::push_runtime_context_note(context, "Hook:UserPromptSubmit", ctx)
                .await;
        }
        for msg in &result.messages {
            super::super::context::push_runtime_context_note(context, "Hook:UserPromptSubmit", msg)
                .await;
        }
    }

    Ok(PrepareOutcome::Continue)
}

// ── 测试 ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::ReactAgent;
    use crate::agent::config::AgentConfig;
    use crate::audit::{AuditFilter, AuditLogger, InMemoryAuditLogger};
    use crate::skills::hooks::{HookAction, HookEvent, HookRegistry, HookRule, HooksDefinition};

    /**
    未注册钩子且无记忆召回时返回 `Continue`，不发出任何事件，
    并将用户输入记录到审计日志中。
    */
    #[tokio::test]
    async fn prepare_turn_continue_audits_user_input() {
        let audit = Arc::new(InMemoryAuditLogger::new());
        let mut agent = ReactAgent::new(AgentConfig::new("test-model", "agent", "sys"));
        agent.set_audit_logger(audit.clone());
        let snap = AgentRunSnapshot::from_agent(&agent);

        let (tx, mut rx) = mpsc::channel::<Result<AgentEvent>>(8);
        let outcome = prepare_turn(
            &snap,
            &agent.memory.context,
            &tx,
            "hello agent",
            "",
            StreamMode::Execute,
            0,
            false,
        )
        .await
        .expect("prepare_turn must succeed");

        assert!(matches!(outcome, PrepareOutcome::Continue));

        // recalled = 0 → 没有 MemoryRecalled 事件；也没有其他事件。
        assert!(
            rx.try_recv().is_err(),
            "no events should have been yielded by prepare_turn on the happy path",
        );

        // 审计日志中应恰好有一条 UserInput 记录。
        let events = audit
            .query(AuditFilter::default())
            .await
            .expect("audit query must succeed");
        let user_inputs: Vec<_> = events
            .iter()
            .filter_map(|e| match &e.event_type {
                crate::audit::AuditEventType::UserInput { content } => Some(content.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(user_inputs, vec!["hello agent"]);
    }

    /**
    `recalled > 0` 时，会在审计/钩子逻辑之前发出一个
    `MemoryRecalled { count }` 事件。
    */
    #[tokio::test]
    async fn prepare_turn_yields_memory_recalled_event() {
        let agent = ReactAgent::new(AgentConfig::new("test-model", "agent", "sys"));
        let snap = AgentRunSnapshot::from_agent(&agent);

        let (tx, mut rx) = mpsc::channel::<Result<AgentEvent>>(8);
        let _ = prepare_turn(
            &snap,
            &agent.memory.context,
            &tx,
            "anything",
            "",
            StreamMode::Chat,
            7,
            false,
        )
        .await
        .expect("prepare_turn must succeed");

        let event = rx
            .recv()
            .await
            .expect("MemoryRecalled event must be yielded")
            .expect("MemoryRecalled is a success event");
        match event {
            AgentEvent::MemoryRecalled { count } => assert_eq!(count, 7),
            other => panic!("expected MemoryRecalled, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn prepare_turn_does_not_append_cwd_system_message_each_turn() {
        let config = AgentConfig::new("test-model", "agent", "sys")
            .working_dir(Some(std::path::PathBuf::from("/tmp/eko-cache-test")));
        let agent = ReactAgent::new(config);
        let snap = AgentRunSnapshot::from_agent(&agent);
        let (tx, _rx) = mpsc::channel::<Result<AgentEvent>>(8);

        let _ = prepare_turn(
            &snap,
            &agent.memory.context,
            &tx,
            "first",
            "",
            StreamMode::Chat,
            0,
            false,
        )
        .await
        .expect("first prepare_turn must succeed");
        let _ = prepare_turn(
            &snap,
            &agent.memory.context,
            &tx,
            "second",
            "",
            StreamMode::Chat,
            0,
            false,
        )
        .await
        .expect("second prepare_turn must succeed");

        let messages = agent.memory.context.lock().await.messages().to_vec();
        let system_messages = messages
            .iter()
            .filter(|message| matches!(message.role, echo_core::llm::types::Role::System))
            .count();
        /* CWD 现在作为运行时上下文（而非系统消息）注入，
        这样系统提示前缀在工作区变化时也能保持缓存稳定。 */
        let cwd_system_messages = messages
            .iter()
            .filter(|message| {
                matches!(message.role, echo_core::llm::types::Role::System)
                    && message.text_content().is_some_and(|text| {
                        text.contains("Current working directory: /tmp/eko-cache-test")
                    })
            })
            .count();
        assert_eq!(
            system_messages, 1,
            "only one system message (the base prompt)"
        );
        assert_eq!(
            cwd_system_messages, 0,
            "CWD must NOT be in system messages — it breaks prompt cache"
        );
        /* CWD 是通过 prepare_stream_context 以运行时上下文的形式注入的
        （单独测试覆盖）。这里只验证系统消息保持稳定。 */
    }

    /**
    `UserPromptSubmit` 钩子返回 `block: true`（通过 Permission 的
    `deny` 动作）时，`prepare_turn` 会发出一个 `FinalAnswer` 事件，
    其内容以 `"Blocked by UserPromptSubmit hook:"` 开头，
    并返回 `BlockedAndDone`。
    */
    #[tokio::test]
    async fn prepare_turn_user_prompt_submit_block_short_circuits() {
        let mut def = HooksDefinition::default();
        def.add_rules(
            HookEvent::UserPromptSubmit,
            vec![HookRule {
                matcher: String::new(), // 匹配所有情况
                hooks: vec![HookAction::Permission {
                    decision: "deny".into(),
                    reason: Some("test-deny".into()),
                    suggestions: vec![],
                }],
            }],
        );
        let mut registry = HookRegistry::new();
        registry.register("test-blocker", "/tmp", def);

        let agent = ReactAgent::new(AgentConfig::new("test-model", "agent", "sys"));
        /* 用我们这个会阻断的 registry 替换默认的。set_hook_registry
        接收 Arc<RwLock<...>>，所以在传入前需要先包装一层。 */
        let hr = Arc::new(tokio::sync::RwLock::new(registry));
        let mut agent = agent;
        agent.set_hook_registry(hr);

        let snap = AgentRunSnapshot::from_agent(&agent);
        let (tx, mut rx) = mpsc::channel::<Result<AgentEvent>>(8);
        let outcome = prepare_turn(
            &snap,
            &agent.memory.context,
            &tx,
            "any user message",
            "",
            StreamMode::Execute,
            0,
            false,
        )
        .await
        .expect("prepare_turn must succeed even when the hook blocks");

        assert!(
            matches!(outcome, PrepareOutcome::BlockedAndDone),
            "block hook → BlockedAndDone, got {:?}",
            phase_outcome_label(&outcome),
        );

        let final_event = rx
            .recv()
            .await
            .expect("FinalAnswer must be yielded")
            .expect("FinalAnswer is a success event");
        match final_event {
            AgentEvent::FinalAnswer(text) => {
                assert!(
                    text.starts_with("Blocked by UserPromptSubmit hook:"),
                    "final answer text: {text:?}",
                );
                assert!(
                    text.contains("test-deny"),
                    "final answer should carry the deny reason: {text:?}",
                );
            }
            other => panic!("expected FinalAnswer, got {other:?}"),
        }
    }

    /// 将 `PrepareOutcome` 转成便于断言输出的字符串标签
    fn phase_outcome_label(o: &PrepareOutcome) -> &'static str {
        match o {
            PrepareOutcome::Continue => "Continue",
            PrepareOutcome::BlockedAndDone => "BlockedAndDone",
            PrepareOutcome::Abandoned => "Abandoned",
        }
    }
}
</content>
