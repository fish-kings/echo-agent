/*!
 * 单次迭代的压缩流程：触发 `PreCompact`、保存运行时检查点、
 * 运行 `ContextManager::prepare`、触发 `PostCompact`，并返回准备好的消息。
 */

use super::super::stream_macros::{try_send_or, yield_event_or};
use super::CompactOutcome;
use crate::agent::AgentEvent;
use crate::agent::snapshot::AgentRunSnapshot;
use crate::error::Result;
use echo_core::tokenizer::Tokenizer;
use std::sync::Arc;
use tokio::sync::{Mutex, mpsc};

/**
 * 执行单次迭代的压缩阶段。
 *
 * 触发 `PreCompact`，获取一次运行时检查点，运行
 * `ContextManager::prepare`；若压缩成功，则产出一个
 * `ContextCompressed` 事件，并运行 `PostCompact` 钩子
 * （其 `injected_context` / `messages` 会被重新推回上下文）。
 *
 * 成功时返回准备好的 LLM 消息；当通道已关闭时返回
 * `CompactOutcome::Abandoned`。
 */
pub(crate) async fn run_compact(
    snap: &AgentRunSnapshot,
    context: &Arc<Mutex<crate::compression::ContextManager>>,
    tx: &mpsc::Sender<Result<AgentEvent>>,
    iteration: usize,
) -> Result<CompactOutcome> {
    snap.fire_hook(crate::skills::hooks::HookEvent::PreCompact, Some("auto"))
        .await;
    /* (stage4 E1) 在压缩执行之前，将持久化事实预先刷写到内存中。
     * 尽力而为：错误/超时不会阻塞压缩流程。该逻辑由
     * `ContextManager::should_compress()` 内部把关，因此只有在
     * 压缩确实迫在眉睫时才会触发（不是每次 ReAct 迭代都触发）。
     */
    let _ = snap.pre_compaction_flush(context).await;
    /* 必须先确认一整批完整的会话记录（transcript batch），
     * 活跃上下文才能跨越压缩边界或重新对齐其生成游标。
     */
    let settlement = snap.save_transcript_projection(context, None).await?;
    if snap.conversation_store.is_some() {
        snap.mark_transcript_settlement_observed();
        yield_event_or!(
            tx,
            AgentEvent::TranscriptProjectionSettlement(settlement.clone()),
            CompactOutcome::Abandoned
        );
    }
    if settlement.status != crate::memory::TranscriptProjectionSettlementStatus::Settled {
        return Err(crate::agent::snapshot::transcript_settlement_admission_error(&settlement));
    }
    let projection_context = crate::compression::ProjectionContext {
        iteration,
        agent_name: snap.config.agent_name.clone(),
        session_id: snap.config.session_id.clone(),
        conversation_id: snap.config.conversation_id.clone(),
        run_id: snap.current_run_id.clone(),
        turn_id: snap.current_turn_id.clone(),
    };
    let projections = if let Some(projector) = &snap.pre_model_context_projector {
        match projector.project(&projection_context).await {
            Ok(projections) => projections,
            Err(error) => {
                tracing::warn!(error = %error, "Pre-model context projection failed");
                Vec::new()
            }
        }
    } else {
        Vec::new()
    };
    let prepare_result = try_send_or!(
        tx,
        {
            let tool_tokens = if snap.config.enable_tool {
                serde_json::to_string(&snap.tools.tools_for_llm())
                    .map(|schema| snap.calibrated_tokenizer.count_tokens(&schema))
                    .map_err(|error| crate::error::ReactError::Other(error.to_string()))?
            } else {
                0
            };
            let mut context = context.lock().await;
            context.apply_projection_scope("pre-model", &projections);
            context
                .prepare_with_cancel(None, tool_tokens, snap.external_cancel.as_deref().cloned())
                .await
        },
        CompactOutcome::Failed
    );

    if let Some(ref stats) = prepare_result.compressed {
        let (protected_message_count, protected_context_tokens) = {
            let context = context.lock().await;
            (
                context.protected_message_count(),
                context.protected_token_estimate(),
            )
        };
        snap.record_event(crate::trace::RunEvent::ContextCompression {
            source: "auto".to_string(),
            before_messages: stats.before_count,
            after_messages: stats.after_count,
            before_tokens: stats.before_tokens,
            after_tokens: stats.after_tokens,
            protected_context_tokens,
            protected_message_count,
        })
        .await;
        yield_event_or!(
            tx,
            AgentEvent::ContextCompressed {
                before_count: stats.before_count,
                after_count: stats.after_count,
                before_tokens: stats.before_tokens,
                after_tokens: stats.after_tokens,
            },
            CompactOutcome::Abandoned
        );
        let hs = crate::skills::hooks::CompressHookStats {
            before_count: stats.before_count,
            after_count: stats.after_count,
            before_tokens: stats.before_tokens,
            after_tokens: stats.after_tokens,
        };
        let hc = crate::skills::hooks::HookContext::for_post_compact(
            &hs,
            "auto",
            snap.config.session_id.as_deref().unwrap_or(""),
            &snap.config.agent_name,
        );
        let reg = snap.tools.hook_registry.read().await.clone();
        let r = reg.run_lifecycle_hooks(&hc).await;
        if let Some(c) = &r.injected_context {
            super::super::context::push_runtime_context_note(context, "Hook:PostCompact", c).await;
        }
        for m in &r.messages {
            super::super::context::push_runtime_context_note(context, "Hook:PostCompact", m).await;
        }
        snap.realign_transcript_projection(context).await?;
    }

    Ok(CompactOutcome::Continue(prepare_result.messages))
}

// ── 测试 ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::ReactAgent;
    use crate::agent::config::AgentConfig;
    use crate::agent::snapshot::AgentRunSnapshot;
    use crate::compression::{
        ContextProjection, PreModelContextProjector, ProjectionContext,
        compressor::SlidingWindowCompressor,
    };
    use crate::llm::types::{Message, Role};
    use crate::trace::RunStore;
    use futures::future::BoxFuture;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// 测试用投影器：每次调用都产出内容随调用次数与迭代号变化的投影。
    struct IterationProjector {
        calls: AtomicUsize,
    }

    impl PreModelContextProjector for IterationProjector {
        /// 按调用计数与当前迭代生成一条带标记的用户消息投影。
        fn project(
            &self,
            context: &ProjectionContext,
        ) -> BoxFuture<'_, Result<Vec<ContextProjection>>> {
            let call = self.calls.fetch_add(1, Ordering::SeqCst);
            let content = format!(
                "<changing_projection>call={call};iteration={}",
                context.iteration
            );
            Box::pin(async move {
                Ok(vec![ContextProjection {
                    marker: "<changing_projection>".to_string(),
                    message: Some(Message::user(content)),
                }])
            })
        }
    }

    /// 测试用投影器：首次调用成功，之后每次调用都返回错误。
    struct FailsAfterFirstProjection {
        calls: AtomicUsize,
    }

    impl PreModelContextProjector for FailsAfterFirstProjection {
        /// 首次返回一条投影，后续调用返回 `projection unavailable` 错误。
        fn project(
            &self,
            _context: &ProjectionContext,
        ) -> BoxFuture<'_, Result<Vec<ContextProjection>>> {
            let call = self.calls.fetch_add(1, Ordering::SeqCst);
            Box::pin(async move {
                if call == 0 {
                    Ok(vec![ContextProjection {
                        marker: "failure-test".to_string(),
                        message: Some(Message::user("projection before failure".to_string())),
                    }])
                } else {
                    Err(crate::error::ReactError::Other(
                        "projection unavailable".to_string(),
                    ))
                }
            })
        }
    }

    /**
     * 默认的 `ReactAgent` 未配置压缩器，因此
     * `ContextManager::prepare` 返回 `compressed: None` —— 不会产出
     * `ContextCompressed` 事件，我们推入的用户消息会原样
     * 透传到返回的 `messages` 中。
     */
    #[tokio::test]
    async fn run_compact_no_compression_yields_no_event() {
        let agent = ReactAgent::new(AgentConfig::new("test-model", "agent", "sys"));
        // 播种一条用户消息，确保准备好的消息不为空。
        agent
            .memory
            .context
            .lock()
            .await
            .push(Message::user("hi".to_string()));
        let snap = AgentRunSnapshot::from_agent(&agent);

        let (tx, mut rx) = mpsc::channel::<Result<AgentEvent>>(8);
        let outcome = run_compact(&snap, &agent.memory.context, &tx, 0)
            .await
            .expect("run_compact must succeed");

        let messages = match outcome {
            CompactOutcome::Continue(m) => m,
            CompactOutcome::Abandoned | CompactOutcome::Failed => {
                return;
            }
        };
        assert!(
            messages
                .iter()
                .any(|m| m.role == Role::User && m.content.as_text().as_deref() == Some("hi")),
            "the seeded user message must survive ContextManager::prepare",
        );

        // 未发生压缩 → 不会产出 ContextCompressed（或任何其他）事件。
        assert!(
            rx.try_recv().is_err(),
            "compact phase must not yield events when no compression occurred",
        );
    }

    /**
     * 对于无压缩分支而言，`iteration` 是不透明的 —— 这里只是验证
     * 使用非零迭代索引调用该函数时不会 panic。
     */
    #[tokio::test]
    async fn run_compact_accepts_arbitrary_iteration_index() {
        let agent = ReactAgent::new(AgentConfig::new("test-model", "agent", "sys"));
        let snap = AgentRunSnapshot::from_agent(&agent);

        let (tx, _rx) = mpsc::channel::<Result<AgentEvent>>(8);
        let outcome = run_compact(&snap, &agent.memory.context, &tx, 17)
            .await
            .expect("run_compact must succeed for non-zero iteration");
        assert!(
            matches!(outcome, CompactOutcome::Continue(_)),
            "unexpected compact outcome: {outcome:?}"
        );
    }

    /// 验证自动压缩会被记录到持久化追踪中。
    #[tokio::test]
    async fn run_compact_records_auto_compression_in_durable_trace() -> Result<()> {
        /* 关闭规则注入：检出目录中的大型规则文件（AGENTS.md 等）
         * 会把系统提示词撑爆，超过本测试极小的 token 限制，
         * 而滑动窗口压缩永远无法把它压缩回去。
         */
        let mut config = AgentConfig::new("test-model", "agent", "sys").auto_project_rules(false);
        config.token_limit = 4_096;
        config.enable_tool = false;
        config.auto_project_rules = false;
        let mut agent = ReactAgent::new(config);
        agent.set_compressor(SlidingWindowCompressor::new(1)).await;
        let store = Arc::new(crate::trace::InMemoryRunStore::new());
        agent.set_run_store(store.clone());
        let legacy = agent.capture_legacy_external_context();
        let trace_run_id = agent
            .start_legacy_trace_run("compress", &legacy)
            .await
            .ok_or_else(|| crate::error::ReactError::Other("trace run missing".to_string()))?;
        {
            let mut context = agent.memory.context.lock().await;
            context.push(Message::user("first message ".repeat(5_000)));
            context.push(Message::assistant("second message ".repeat(500)));
        }
        let snap = AgentRunSnapshot::from_agent(&agent);
        let (tx, mut rx) = mpsc::channel::<Result<AgentEvent>>(8);

        let outcome = run_compact(&snap, &agent.memory.context, &tx, 0).await?;
        if !matches!(outcome, CompactOutcome::Continue(_)) {
            return Err(crate::error::ReactError::Other(format!(
                "unexpected compact outcome: {outcome:?}; event={:?}",
                rx.try_recv().ok()
            )));
        }
        let run = store
            .load(&trace_run_id)
            .await?
            .ok_or_else(|| crate::error::ReactError::Other("trace data missing".to_string()))?;
        assert!(run.events.iter().any(|event| matches!(
            event,
            crate::trace::RunEvent::ContextCompression { source, .. } if source == "auto"
        )));
        Ok(())
    }

    /// 验证在替换活跃历史之前，会先持久化完整对话记录。
    #[tokio::test]
    async fn run_compact_persists_full_transcript_before_replacing_active_history() -> Result<()> {
        use crate::memory::{ConversationStore, FileConversationStore};

        let temp = tempfile::tempdir()?;
        let conversation_store = Arc::new(FileConversationStore::new(temp.path())?);
        let runtime_root = tempfile::tempdir()?;
        let runtime_store = Arc::new(crate::state::FileRuntimeStateStore::new(
            runtime_root.path(),
        )?);
        /* 出于与上面 trace 测试相同的原因关闭规则注入：注入的工作区
         * 规则会超出极小的 token 限制，导致预期的 Continue 结果
         * 变成 Failed。
         */
        let mut config = AgentConfig::new("test-model", "agent", "sys")
            .conversation_id("pre-compact-transcript")
            .auto_project_rules(false);
        config.token_limit = 4_096;
        config.enable_tool = false;
        config.auto_project_rules = false;
        let mut agent = ReactAgent::new(config);
        agent.set_conversation_store(conversation_store.clone());
        agent.set_state_store(runtime_store);
        agent.set_compressor(SlidingWindowCompressor::new(1)).await;
        {
            let mut context = agent.memory.context.lock().await;
            context.push(Message::user("original request ".repeat(5_000)));
            context.push(Message::assistant("original answer ".repeat(500)));
            context.push(Message::user("latest request".to_string()));
        }
        let snap = AgentRunSnapshot::from_agent(&agent);
        let (tx, _rx) = mpsc::channel::<Result<AgentEvent>>(8);

        let outcome = run_compact(&snap, &agent.memory.context, &tx, 0).await?;
        assert!(matches!(outcome, CompactOutcome::Continue(_)));

        let persisted = conversation_store
            .get_messages("pre-compact-transcript")
            .await?;
        assert!(persisted.iter().any(|message| {
            message
                .content
                .as_deref()
                .is_some_and(|content| content.starts_with("original request"))
        }));
        assert!(persisted.iter().any(|message| {
            message
                .content
                .as_deref()
                .is_some_and(|content| content.starts_with("original answer"))
        }));
        Ok(())
    }

    /// 验证每次 prepare 之前都会替换上一次的投影内容。
    #[tokio::test]
    async fn run_compact_replaces_projection_before_each_prepare() -> Result<()> {
        let agent = ReactAgent::new(AgentConfig::new("test-model", "agent", "sys"));
        agent.set_pre_model_context_projector(Some(Arc::new(IterationProjector {
            calls: AtomicUsize::new(0),
        })));
        let snap = AgentRunSnapshot::from_agent(&agent);
        let (tx, _rx) = mpsc::channel::<Result<AgentEvent>>(8);

        let first = run_compact(&snap, &agent.memory.context, &tx, 3).await?;
        let second = run_compact(&snap, &agent.memory.context, &tx, 4).await?;

        let first_messages = match first {
            CompactOutcome::Continue(messages) => messages,
            CompactOutcome::Abandoned | CompactOutcome::Failed => Vec::new(),
        };
        let second_messages = match second {
            CompactOutcome::Continue(messages) => messages,
            CompactOutcome::Abandoned | CompactOutcome::Failed => Vec::new(),
        };
        assert!(first_messages.iter().any(|message| {
            message
                .content
                .as_text_ref()
                .is_some_and(|text| text.contains("call=0;iteration=3"))
        }));
        assert!(second_messages.iter().any(|message| {
            message
                .content
                .as_text_ref()
                .is_some_and(|text| text.contains("call=1;iteration=4"))
        }));
        assert!(second_messages.iter().all(|message| {
            message
                .content
                .as_text_ref()
                .is_none_or(|text| !text.contains("call=0;iteration=3"))
        }));
        assert_eq!(
            second_messages
                .iter()
                .filter(|message| {
                    message
                        .content
                        .as_text_ref()
                        .is_some_and(|text| text.contains("<changing_projection>"))
                })
                .count(),
            1
        );
        Ok(())
    }

    /// 验证投影提供者失败后，已有投影会被清除。
    #[tokio::test]
    async fn run_compact_clears_projection_after_provider_failure() -> Result<()> {
        let agent = ReactAgent::new(AgentConfig::new("test-model", "agent", "sys"));
        agent.set_pre_model_context_projector(Some(Arc::new(FailsAfterFirstProjection {
            calls: AtomicUsize::new(0),
        })));
        let snap = AgentRunSnapshot::from_agent(&agent);
        let (tx, _rx) = mpsc::channel::<Result<AgentEvent>>(8);

        let _ = run_compact(&snap, &agent.memory.context, &tx, 0).await?;
        let second = run_compact(&snap, &agent.memory.context, &tx, 1).await?;

        let messages = match second {
            CompactOutcome::Continue(messages) => messages,
            CompactOutcome::Abandoned | CompactOutcome::Failed => Vec::new(),
        };
        assert!(messages.iter().all(|message| {
            message
                .content
                .as_text()
                .is_none_or(|text| !text.contains("projection before failure"))
        }));
        Ok(())
    }

    /// 验证投影提供者被移除后，已有投影会被清除。
    #[tokio::test]
    async fn run_compact_clears_projection_after_provider_removal() -> Result<()> {
        let agent = ReactAgent::new(AgentConfig::new("test-model", "agent", "sys"));
        agent.set_pre_model_context_projector(Some(Arc::new(IterationProjector {
            calls: AtomicUsize::new(0),
        })));
        let first_snap = AgentRunSnapshot::from_agent(&agent);
        let (tx, _rx) = mpsc::channel::<Result<AgentEvent>>(8);
        let _ = run_compact(&first_snap, &agent.memory.context, &tx, 0).await?;

        agent.set_pre_model_context_projector(None);
        let second_snap = AgentRunSnapshot::from_agent(&agent);
        let second = run_compact(&second_snap, &agent.memory.context, &tx, 1).await?;

        let messages = match second {
            CompactOutcome::Continue(messages) => messages,
            CompactOutcome::Abandoned | CompactOutcome::Failed => Vec::new(),
        };
        assert!(messages.iter().all(|message| {
            message
                .content
                .as_text()
                .is_none_or(|text| !text.contains("<changing_projection>"))
        }));
        Ok(())
    }
}

// ── stage4 E1：pre_compaction_flush ────────────────────────────────────

/// stage4 E1：`pre_compaction_flush` 相关测试。
#[cfg(test)]
mod stage4_e1_tests {
    use crate::agent::ReactAgent;
    use crate::agent::ReactAgentBuilder;
    use crate::agent::snapshot::AgentRunSnapshot;
    use crate::evolution::MemoryLayerManager;
    use crate::evolution::audit::NullChangeLog;
    use crate::llm::types::Message;
    use crate::testing::MockLlmClient;
    use echo_core::memory::store::Store;
    use echo_state::memory::store::InMemoryStore;
    use std::sync::Arc;

    /**
     * 构建一个使用 mock LLM、并搭配一个由 `InMemoryStore` 支撑的
     * 共享 `MemoryLayerManager` 的 agent。返回
     * `(agent, store_handle, layer_manager)`，以便测试可以断言
     * 统一命名空间 `["agent","memories"]` 中落地了什么内容。
     * `token_limit` 控制 `should_compress()` 是否返回 true。
     */
    fn agent_with_layer(
        llm: MockLlmClient,
        token_limit: usize,
    ) -> (ReactAgent, Arc<dyn Store>, Arc<MemoryLayerManager>) {
        let mut agent = ReactAgentBuilder::new()
            .llm_client(Arc::new(llm))
            .system_prompt("test")
            .token_limit(token_limit)
            .build()
            .expect("agent builds");
        let store: Arc<InMemoryStore> = Arc::new(InMemoryStore::new());
        let store_dyn: Arc<dyn Store> = store.clone();
        let dir = tempfile::tempdir().expect("tempdir").keep();
        let lm = Arc::new(MemoryLayerManager::new(dir, store, Box::new(NullChangeLog)));
        agent.install_memory_layer_manager(lm.clone());
        (agent, store_dyn, lm)
    }

    /**
     * 向 agent 的上下文中推入 `n` 条用户消息，使其超出 token 限制
     * （每条消息都携带足够多的文本，以便分词器能够计入）。
     */
    async fn push_messages(agent: &ReactAgent, n: usize) {
        let mut ctx = agent.memory.context.lock().await;
        for i in 0..n {
            ctx.push(Message::user(format!(
                "message {i} with enough words to count"
            )));
        }
    }

    /**
     * 当压缩迫在眉睫时（token_limit 较低 + 消息较多），刷写用的
     * LLM 调用会执行，持久化事实会落地到统一存储中（割裂点 1/7）。
     */
    #[tokio::test]
    async fn pre_compaction_flush_writes_durable_facts_when_compression_imminent() {
        let llm = MockLlmClient::new().with_response(
            r#"[{"content":"user prefers Rust over Python","type":"user_preference","recall_weight":0.9}]"#,
        );
        // token_limit=1 → ReactAgent 会安装 SlidingWindow，且 should_compress()=true。
        let (agent, store, _lm) = agent_with_layer(llm, 1);
        push_messages(&agent, 8).await;
        let snap = AgentRunSnapshot::from_agent(&agent);

        snap.pre_compaction_flush(&agent.memory.context).await;

        let results = store
            .search(&["agent", "memories"], "Rust", 10)
            .await
            .expect("search");
        assert!(
            results.iter().any(|r| serde_json::to_string(&r.value)
                .unwrap_or_default()
                .contains("Rust")),
            "flushed durable fact should be in the unified store, got: {:?}",
            results
        );
    }

    /**
     * 当压缩并不迫在眉睫时（未安装压缩器 / token 数在限制之内），
     * 刷写会被跳过 —— 不会有 LLM 调用，也不会写入任何记忆。
     * 用于守护 `should_compress()` 这道关卡，确保刷写不会在
     * 每次 ReAct 迭代中都触发。
     */
    #[tokio::test]
    async fn pre_compaction_flush_noops_when_compression_not_imminent() {
        let llm = MockLlmClient::new()
            .with_response(r#"[{"content":"x","type":"project_fact","recall_weight":0.5}]"#);
        // token_limit=MAX → 未安装压缩器 → should_compress()=false。
        let (agent, store, _lm) = agent_with_layer(llm, usize::MAX);
        push_messages(&agent, 8).await;
        let snap = AgentRunSnapshot::from_agent(&agent);

        snap.pre_compaction_flush(&agent.memory.context).await;

        let results = store
            .search(&["agent", "memories"], "x", 10)
            .await
            .expect("search");
        assert!(
            results.is_empty(),
            "no flush should occur when compression is not imminent, got: {:?}",
            results
        );
    }

    /**
     * 尽力而为：LLM 出错（mock 队列为空 → EmptyResponse）时不能
     * panic 或阻塞 —— 不会写入任何记忆。
     */
    #[tokio::test]
    async fn pre_compaction_flush_best_effort_on_llm_error() {
        let llm = MockLlmClient::new(); // 空队列 → chat 返回错误
        let (agent, store, _lm) = agent_with_layer(llm, 1);
        push_messages(&agent, 8).await;
        let snap = AgentRunSnapshot::from_agent(&agent);

        // 必须不能 panic / 阻塞：
        snap.pre_compaction_flush(&agent.memory.context).await;

        let results = store
            .search(&["agent", "memories"], "message", 10)
            .await
            .expect("search");
        assert!(
            results.is_empty(),
            "LLM error should not write any memory, got: {:?}",
            results
        );
    }
}
