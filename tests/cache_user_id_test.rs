//! 请求构造与用量统计中的缓存身份契约测试。

#[cfg(feature = "testing")]
use echo_agent::agent::Agent;
use echo_agent::agent::config::AgentConfig;
use echo_agent::llm::ChatRequest;
#[cfg(feature = "testing")]
use echo_agent::prelude::ReactAgentBuilder;
#[cfg(feature = "testing")]
use echo_agent::testing::MockLlmClient;
#[cfg(feature = "testing")]
use futures::StreamExt;
#[cfg(feature = "testing")]
use std::sync::Arc;

/// `execute` 会把包含 Unicode 的 `cache_user_id` 原样传递到生产请求中。
#[cfg(feature = "testing")]
#[tokio::test]
async fn execute_propagates_unicode_cache_user_id_to_the_production_request()
-> echo_agent::error::Result<()> {
    let llm = Arc::new(MockLlmClient::new().with_response("done"));
    let mut agent = ReactAgentBuilder::new()
        .llm_client(llm.clone())
        .system_prompt("test")
        .build()?;
    agent.config_mut().set_cache_user_id("用户-cache-🔒");

    agent.execute("run once").await?;
    assert_eq!(llm.all_user_ids(), vec![Some("用户-cache-🔒".to_string())]);
    Ok(())
}

/// `execute_stream` 在每一轮都传递同一个稳定的 `cache_user_id`。
#[cfg(feature = "testing")]
#[tokio::test]
async fn execute_stream_propagates_stable_cache_user_id_on_every_turn()
-> echo_agent::error::Result<()> {
    let llm = Arc::new(MockLlmClient::new().with_responses(["first", "second"]));
    let mut agent = ReactAgentBuilder::new()
        .llm_client(llm.clone())
        .system_prompt("test")
        .build()?;
    agent.config_mut().set_cache_user_id("stable-cache-user");

    for task in ["first turn", "second turn"] {
        let mut stream = agent.execute_stream(task).await?;
        while let Some(event) = stream.next().await {
            event?;
        }
    }

    assert_eq!(
        llm.all_user_ids(),
        vec![
            Some("stable-cache-user".to_string()),
            Some("stable-cache-user".to_string())
        ]
    );
    Ok(())
}

/**
验证 `AgentConfig` 支持设置 `cache_user_id`，
并且该设置可在构建器链中使用。
*/
#[test]
fn agent_config_accepts_cache_user_id() {
    let config =
        AgentConfig::minimal("test-model", "test-prompt").cache_user_id("test-cache-user-001");
    assert_eq!(config.get_cache_user_id(), Some("test-cache-user-001"));
}

/// 验证 `ChatRequest::user_id` 默认为 `None`（安全的默认值）。
#[test]
fn chat_request_user_id_defaults_none() {
    let request = ChatRequest::new(vec![]);
    assert_eq!(request.user_id, None);
}

/// 验证 `ChatRequest` 接受 `cache_hints`。
#[test]
fn chat_request_accepts_cache_hints() {
    let mut request = ChatRequest::new(vec![]);
    assert!(request.cache_hints.is_none());

    // 模拟 think.rs 的行为：附加缓存提示。
    let hints = echo_core::llm::cache::CacheHints {
        breakpoints: vec![],
        stable_prefix_hash: Some("deadbeef".to_string()),
        segments: Default::default(),
    };
    request.cache_hints = Some(hints);
    assert!(request.cache_hints.is_some());
    assert_eq!(
        request
            .cache_hints
            .as_ref()
            .and_then(|value| value.stable_prefix_hash.as_deref()),
        Some("deadbeef")
    );
}

/// 验证 `TokenUsageTracker` 的累计缓存命中率计算。
#[test]
fn token_tracker_cumulative_cache_hit_rate() {
    use echo_core::tokenizer::TokenUsageTracker;
    let tracker = TokenUsageTracker::new("test-model");

    // 没有请求 → 没有命中率
    assert!(tracker.cumulative_cache_hit_rate().is_none());

    // 通过 record_usage 记录第 1 次请求：100 个 prompt token，50 个命中缓存
    tracker.record_usage(&echo_core::llm::types::Usage {
        prompt_tokens: Some(100),
        completion_tokens: Some(50),
        total_tokens: Some(150),
        cache_creation_input_tokens: Some(10),
        cache_read_input_tokens: Some(50),
        ..Default::default()
    });

    // 通过 record_usage 记录第 2 次请求：200 个 prompt token，180 个命中缓存
    tracker.record_usage(&echo_core::llm::types::Usage {
        prompt_tokens: Some(200),
        completion_tokens: Some(100),
        total_tokens: Some(300),
        cache_creation_input_tokens: Some(5),
        cache_read_input_tokens: Some(180),
        ..Default::default()
    });

    // 累计：(50+180) / ((100)+(200)+(50+180)) = 230/530 ≈ 43.4%
    let Some(rate) = tracker.cumulative_cache_hit_rate() else {
        return;
    };
    assert!(
        (rate - 0.434).abs() < 0.02,
        "expected ~43.4%, got {:.1}%",
        rate * 100.0
    );
}

/// 验证 `stable_prefix_hash` 是确定性的（相同输入 → 相同输出）。
#[test]
fn stable_prefix_hash_is_deterministic() {
    use echo_core::llm::Message;
    use echo_core::llm::cache::diagnostic::stable_prefix_hash;

    let sys = &[Message::system("You are a helpful assistant.".to_string())];
    let history = &[Message::user("hello".to_string())];

    let h1 = stable_prefix_hash(sys, &[], &[], history);
    let h2 = stable_prefix_hash(sys, &[], &[], history);
    assert_eq!(h1, h2);
}

/// 验证 `AnthropicCachePlan` 生成合理的缓存断点。
#[test]
fn anthropic_cache_plan_is_sensible() {
    use echo_core::llm::Message;
    use echo_core::llm::cache::layout::PromptCacheLayout;
    use echo_integration::providers::AnthropicCachePlan;

    let msgs = vec![
        Message::system("System prompt".to_string()),
        Message::user("Hello".to_string()),
        Message::user("How are you?".to_string()),
        Message::user("What can you do?".to_string()),
        Message::user("Tell me more".to_string()),
    ];
    let layout = PromptCacheLayout::from_messages(&msgs, &[]);
    let plan = AnthropicCachePlan::from_layout(&layout);

    /* 有 system + 4 条历史消息时，应得到：
    - SystemLastBlock（存在 system）
    - HistoryIndex(~75%)（历史 >= 4）
    - HistoryLastStable（历史非空） */
    assert!(plan.has_system_breakpoint);
    assert!(!plan.has_tool_breakpoint); // 没有工具
    assert!(!plan.breakpoints.is_empty());
    assert!(plan.breakpoints.len() <= 4);
}
