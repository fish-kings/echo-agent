/*!
 * demo53 — 自适应压缩
 *
 * 演示 `AdaptiveCompressor`：按从低成本到高成本的顺序依次应用各压缩级别，
 * 直到满足 token 预算：
 *
 * | 级别 | 名称 | 策略 |
 * |-------|------|----------|
 * | L1 Snip | Snip | 移除超过 N 个 token 的工具输出 |
 * | L1 Fold | Fold | 折叠连续的工具结果，保留最新 N 条 |
 * | L2 | Micro | 将工具输出截断为首尾各 N 行 |
 * | L3 | Collapse | 移除较早的消息，保留最近 N 条 |
 * | L4 | Compact | 完整的 LLM 摘要（通过 `.with_llm()`） |
 * | L5 | Reactive | 紧急模式——仅保留系统提示词 + 最后 3 条消息 |
 *
 * 场景 1-7 均在本地运行（不调用 LLM）。场景 8 演示通过
 * `ContextCompressor` trait 与 `ContextManager` 集成。
 *
 * ```sh
 * cargo test --test example_contracts contract_demo53_adaptive_compression
 * ```
 */

use echo_agent::compression::ContextManager;
use echo_agent::compression::levels::{AdaptiveCompressionConfig, AdaptiveCompressor};
use echo_agent::compression::{CompressionInput, CompressionOutput, ContextCompressor};
use echo_agent::error::{ReactError, Result};
use echo_agent::prelude::{Message, MessageContent, Role};

/// 打印场景分节标题。
macro_rules! section {
    ($n:expr, $title:expr) => {
        println!("\n══════════════════════════════════════════════════");
        println!("  Scenario {} : {}", $n, $title);
        println!("══════════════════════════════════════════════════");
    };
}

/// 以指定角色和文本构造一条纯文本消息。
fn make_msg(role: Role, text: &str) -> Message {
    Message {
        role,
        content: MessageContent::Text(text.to_string()),
        ..Default::default()
    }
}

/** 创建一条约为 `token_count` 个 token 的工具输出消息。
启发式分词器按每 token 约 4 个字符估算。 */
fn make_tool_output(id: usize, approx_tokens: usize) -> Message {
    let char_count = approx_tokens.saturating_mul(4);
    let content = format!(
        "Tool output #{}:\n{}",
        id,
        "lorem ipsum dolor sit amet. ".repeat((char_count / 28).saturating_add(1))
    );
    make_msg(Role::Tool, &content)
}

/// demo53 契约测试入口：依次运行 8 个自适应压缩场景。
#[tokio::test]
async fn contract_demo53_adaptive_compression() -> Result<()> {
    println!("╔══════════════════════════════════════════════════╗");
    println!("║    echo-agent  Adaptive Compression Demo         ║");
    println!("║  (no LLM calls — local heuristic compression)    ║");
    println!("╚══════════════════════════════════════════════════╝");

    demo_l1_snip();
    demo_l1_fold();
    demo_l2_micro();
    demo_l3_collapse();
    demo_l5_reactive();
    demo_escalation();
    demo_no_compression_needed();
    demo_context_manager_integration().await?;

    println!("\n╔══════════════════════════════════════════════════╗");
    println!("║  All 8 scenarios passed ✅                       ║");
    println!("╚══════════════════════════════════════════════════╝");
    Ok(())
}

/// L1 Snip：截断过大的工具输出
fn demo_l1_snip() {
    section!("1", "L1 Snip — truncate oversized tool outputs");

    let config = AdaptiveCompressionConfig {
        l1_snip_threshold_tokens: 0,      // 始终尝试 L1
        l1_max_output_tokens: 50,         // 每条工具输出最多 50 个 token
        l1_fold_consecutive_tools: false, // 禁用折叠以单独测试
        // 在本隔离测试中禁用更高级别
        l2_micro_threshold_tokens: usize::MAX,
        l3_collapse_threshold_tokens: usize::MAX,
        l4_compact_threshold_tokens: usize::MAX,
        ..Default::default()
    };
    let compressor = AdaptiveCompressor::new(config);

    let mut messages = vec![
        make_msg(Role::System, "You are a helpful assistant."),
        make_msg(Role::User, "Read the file"),
        make_tool_output(1, 500), // 约 500 个 token，远超 50 的上限
        make_msg(Role::User, "Thanks!"),
    ];

    let current_tokens = 600;
    let target_tokens = 100;
    let result = compressor.compress_in_place(&mut messages, current_tokens, target_tokens);

    println!("  Tokens before: {}", result.tokens_before);
    println!("  Tokens after:  {}", result.tokens_after);
    println!("  Levels applied: {:?}", result.levels_applied);

    assert!(result.levels_applied.contains(&"L1:Snip".to_string()));
    println!("  ✅ L1:Snip applied — oversized tool output truncated");
}

/// L1 Fold：折叠连续的工具结果
fn demo_l1_fold() {
    section!("2", "L1 Fold — collapse consecutive tool results");

    let config = AdaptiveCompressionConfig {
        l1_snip_threshold_tokens: 0, // 始终尝试 L1
        l1_max_output_tokens: 99999, // 不做裁剪，只做折叠
        l1_fold_consecutive_tools: true,
        l1_fold_keep_latest: 2, // 每段连续结果保留最新 2 条
        // 禁用更高级别
        l2_micro_threshold_tokens: usize::MAX,
        l3_collapse_threshold_tokens: usize::MAX,
        l4_compact_threshold_tokens: usize::MAX,
        ..Default::default()
    };
    let compressor = AdaptiveCompressor::new(config);

    let mut messages = vec![
        make_msg(Role::System, "System prompt"),
        make_msg(Role::User, "read all files"),
        make_tool_output(1, 100),
        make_tool_output(2, 100),
        make_tool_output(3, 100),
        make_tool_output(4, 100),
        make_tool_output(5, 100),
        make_msg(Role::Assistant, "Done reading all files."),
    ];

    let original_count = messages.len();
    let current_tokens = 1000;
    let target_tokens = 100;
    let result = compressor.compress_in_place(&mut messages, current_tokens, target_tokens);

    println!("  Messages before: {}", original_count);
    println!("  Messages after:  {}", messages.len());
    println!("  Levels applied: {:?}", result.levels_applied);

    // 5 条工具消息且 keep=2 → 折叠 3 条并插入 1 条摘要 → 8 - 3 + 1 = 6
    assert!(result.levels_applied.contains(&"L1:Fold".to_string()));
    assert!(
        messages
            .iter()
            .any(|m| m.content.as_text_ref().unwrap_or("").contains("L1 fold"))
    );
    println!("  ✅ L1:Fold applied — 3 consecutive tool results collapsed into 1 summary");
}

/// L2 Micro：将工具输出截断为首尾各 N 行
fn demo_l2_micro() {
    section!("3", "L2 Micro — truncate to first/last N lines");

    let config = AdaptiveCompressionConfig {
        // 禁用 L1，以便单独测试 L2
        l1_snip_threshold_tokens: usize::MAX,
        l2_micro_threshold_tokens: 0, // 始终尝试 L2
        l2_keep_lines: 5,             // 保留前 5 行 + 后 5 行
        // 禁用更高级别
        l3_collapse_threshold_tokens: usize::MAX,
        l4_compact_threshold_tokens: usize::MAX,
        ..Default::default()
    };
    let compressor = AdaptiveCompressor::new(config);

    // 构造一条 30 行的工具输出（多于 5*2 = 10 行）
    let long_output: String = (1..=30)
        .map(|i| format!("Line {}: some content here", i))
        .collect::<Vec<_>>()
        .join("\n");
    let mut messages = vec![
        make_msg(Role::System, "system prompt"),
        make_msg(Role::User, "run the command"),
        make_msg(Role::Tool, &long_output),
    ];

    let current_tokens = 500;
    let target_tokens = 100;
    let result = compressor.compress_in_place(&mut messages, current_tokens, target_tokens);

    println!("  Original lines: 30");
    println!("  Keep lines: 5 head + 5 tail = 10");
    println!("  Tokens before: {}", result.tokens_before);
    println!("  Tokens after:  {}", result.tokens_after);
    println!("  Levels applied: {:?}", result.levels_applied);

    assert!(result.levels_applied.contains(&"L2:Micro".to_string()));
    println!("  ✅ L2:Micro applied — truncated to first/last 5 lines");
}

/// L3 Collapse：移除较早的消息，保留最近 N 条
fn demo_l3_collapse() {
    section!("4", "L3 Collapse — remove older messages");

    let config = AdaptiveCompressionConfig {
        // 禁用 L1 和 L2
        l1_snip_threshold_tokens: usize::MAX,
        l2_micro_threshold_tokens: usize::MAX,
        l3_collapse_threshold_tokens: 0, // 始终尝试 L3
        l3_keep_recent: 3,               // 保留最后 3 条消息
        // 禁用 L4/L5
        l4_compact_threshold_tokens: usize::MAX,
        ..Default::default()
    };
    let compressor = AdaptiveCompressor::new(config);

    let mut messages = vec![
        make_msg(Role::System, "You are a coding assistant."),
        make_msg(Role::User, "message 1 — old"),
        make_msg(Role::Assistant, "response 1 — old"),
        make_msg(Role::User, "message 2 — old"),
        make_msg(Role::Assistant, "response 2 — old"),
        make_msg(Role::User, "message 3 — old"),
        make_msg(Role::Assistant, "response 3 — old"),
        make_msg(Role::User, "latest question"),
        make_msg(Role::Assistant, "latest response"),
        make_msg(Role::User, "follow-up"),
    ];

    let original_count = messages.len();
    let current_tokens = 1000;
    let target_tokens = 100;
    let result = compressor.compress_in_place(&mut messages, current_tokens, target_tokens);

    println!("  Messages before: {}", original_count);
    println!("  Messages after:  {}", messages.len());
    println!("  Tokens before:   {}", result.tokens_before);
    println!("  Tokens after:    {}", result.tokens_after);
    println!("  Levels applied:  {:?}", result.levels_applied);

    // 系统消息应被保留
    assert!(messages.iter().any(|m| m.role == Role::System));
    assert!(result.levels_applied.contains(&"L3:Collapse".to_string()));
    println!("  ✅ L3:Collapse applied — older messages removed, system + recent kept");
}

/// L5 Reactive：紧急压缩
fn demo_l5_reactive() {
    section!("5", "L5 Reactive — emergency (system prompt + last 3 only)");

    let config = AdaptiveCompressionConfig {
        // 禁用 L1、L2、L3
        l1_snip_threshold_tokens: usize::MAX,
        l2_micro_threshold_tokens: usize::MAX,
        l3_collapse_threshold_tokens: usize::MAX,
        l4_compact_threshold_tokens: 0, // 允许触发 L5
        ..Default::default()
    };
    let compressor = AdaptiveCompressor::new(config);

    // 使用较长内容，使 token 计数更贴近真实
    let long_text = "lorem ipsum dolor sit amet consectetur adipiscing elit. ".repeat(20);
    let mut messages = vec![
        make_msg(Role::System, "You are an AI assistant."),
        make_msg(Role::User, &long_text),
        make_msg(Role::Assistant, &long_text),
        make_msg(Role::User, &long_text),
        make_msg(Role::Assistant, &long_text),
        make_msg(Role::User, &long_text),
        make_msg(Role::Assistant, &long_text),
        make_msg(Role::User, &long_text),
        make_msg(Role::Assistant, &long_text),
        make_msg(Role::User, "final question"),
    ];

    let original_count = messages.len();
    // 按每 4 个字符约 1 个 token 估算
    let current_tokens: usize = messages
        .iter()
        .map(|m| m.content.as_text_ref().unwrap_or("").chars().count() / 4)
        .sum();
    // 目标值远小于当前值
    let target_tokens = current_tokens / 20;
    let result = compressor.compress_in_place(&mut messages, current_tokens, target_tokens);

    println!("  Messages before: {}", original_count);
    println!("  Messages after:  {}", messages.len());
    println!("  Tokens before:   {}", result.tokens_before);
    println!("  Tokens after:    {}", result.tokens_after);
    println!("  Levels applied:  {:?}", result.levels_applied);

    assert!(messages.iter().any(|m| m.role == Role::System));
    assert!(result.levels_applied.contains(&"L5:Reactive".to_string()));
    println!("  ✅ L5:Reactive applied — only system prompt + last 3 messages remain");
}

/// 自动升级：按顺序应用多个级别
fn demo_escalation() {
    section!("6", "Automatic Level Escalation");

    // 将所有阈值调低，使多个级别都会触发
    let config = AdaptiveCompressionConfig {
        l1_snip_threshold_tokens: 0,
        l1_max_output_tokens: 20,
        l2_micro_threshold_tokens: 0,
        l2_keep_lines: 3,
        l3_collapse_threshold_tokens: 0,
        l3_keep_recent: 3,
        l4_compact_threshold_tokens: usize::MAX,
        ..Default::default()
    };
    let compressor = AdaptiveCompressor::new(config);

    // 构造一段包含大型工具输出的对话
    let long_output: String = (1..=40)
        .map(|i| format!("Line {}: data={}", i, "x".repeat(50)))
        .collect::<Vec<_>>()
        .join("\n");
    let mut messages = vec![
        make_msg(Role::System, "System prompt"),
        make_msg(Role::User, "old question 1"),
        make_msg(Role::Assistant, "old answer 1"),
        make_msg(Role::User, "old question 2"),
        make_msg(Role::Assistant, "old answer 2"),
        make_msg(Role::User, "read file"),
        make_msg(Role::Tool, &long_output),
        make_msg(Role::User, "latest question"),
    ];

    let current_tokens = 2000;
    let target_tokens = 100;
    let result = compressor.compress_in_place(&mut messages, current_tokens, target_tokens);

    println!("  Tokens before: {}", result.tokens_before);
    println!("  Tokens after:  {}", result.tokens_after);
    println!("  Levels applied: {:?}", result.levels_applied);
    println!("  Messages remaining: {}", messages.len());

    assert!(!result.levels_applied.is_empty());
    println!("  ✅ Multiple levels escalated automatically");

    println!("\n  Escalation order: L1:Snip → L1:Fold → L2:Micro → L3:Collapse → L5:Reactive");
    println!("  (L4:Compact skipped — no LLM client configured via .with_llm())");
}

/// 低于目标值时不进行压缩
fn demo_no_compression_needed() {
    section!("7", "No Compression Needed");

    let config = AdaptiveCompressionConfig::default();
    let compressor = AdaptiveCompressor::new(config);

    let mut messages = vec![
        make_msg(Role::User, "hello"),
        make_msg(Role::Assistant, "hi there!"),
    ];

    // current_tokens < target → 不应发生任何压缩
    let result = compressor.compress_in_place(&mut messages, 100, 200);

    println!("  Tokens before: {}", result.tokens_before);
    println!("  Tokens after:  {}", result.tokens_after);
    println!("  Levels applied: {:?}", result.levels_applied);

    assert!(result.levels_applied.is_empty());
    println!("  ✅ No levels applied when already within budget");
}

/// ContextManager 集成：通过 ContextCompressor trait 使用 AdaptiveCompressor
async fn demo_context_manager_integration() -> Result<()> {
    section!("8", "ContextManager Integration (ContextCompressor trait)");

    let config = AdaptiveCompressionConfig {
        l1_snip_threshold_tokens: 0,
        l1_max_output_tokens: 99999,
        l1_fold_consecutive_tools: true,
        l1_fold_keep_latest: 1,
        l3_collapse_threshold_tokens: 0,
        l3_keep_recent: 2,
        l4_compact_threshold_tokens: usize::MAX,
        ..Default::default()
    };
    let compressor = AdaptiveCompressor::new(config);

    /** AdaptiveCompressor 现已实现 ContextCompressor！
    可直接用于 ContextManager::builder().compressor() */
    const CONTEXT_LIMIT: usize = 64;
    let mut ctx = ContextManager::builder(CONTEXT_LIMIT)
        .compressor(compressor)
        .with_system("You are a helpful assistant.".to_string())
        .build();

    /* 较长的旧轮次会迫使压缩发生，而简短的最新轮次与
    生成的折叠摘要能够放入压缩后的预算内。 */
    for i in 0..9 {
        ctx.push(make_msg(
            Role::User,
            &format!("Question {i}: {}", "historical context ".repeat(12)),
        ));
        ctx.push(make_msg(
            Role::Assistant,
            &format!("Answer {i}: {}", "historical response ".repeat(12)),
        ));
    }
    ctx.push(make_msg(Role::User, "Latest question"));
    ctx.push(make_msg(Role::Assistant, "Latest answer"));

    let before_count = ctx.messages().len();
    let before_tokens = ctx.token_estimate();
    assert!(
        before_tokens > CONTEXT_LIMIT,
        "scenario 8 must begin above the context limit"
    );

    // prepare() 通过 ContextCompressor trait 触发自动压缩
    let result = ctx.prepare(None).await?;
    let compressed = result.compressed.ok_or_else(|| {
        ReactError::Other("compression should have occurred in scenario 8".to_string())
    })?;

    println!(
        "  Before: {} messages, {} tokens",
        before_count, before_tokens
    );
    println!(
        "  After:  {} messages, {} tokens",
        compressed.after_count, compressed.after_tokens
    );
    println!("  Evicted: {} messages", compressed.evicted);

    // 检查压缩指标
    let metrics = ctx.compression_metrics();
    println!("  Total compressions: {}", metrics.total_compressions);
    println!("  Tokens saved: {}", metrics.total_tokens_saved());
    println!("  Report: {}", metrics.report());

    assert!(compressed.after_count < before_count);
    assert!(
        compressed.after_tokens <= CONTEXT_LIMIT,
        "compressed context must fit the configured limit"
    );
    assert!(metrics.total_compressions >= 1);
    println!("  ✅ AdaptiveCompressor works seamlessly via ContextCompressor trait");
    println!("  ✅ CompressionMetrics tracked automatically");

    // 另外演示直接使用 trait（不经过 ContextManager）
    println!("\n  --- Direct trait usage ---");
    let compressor2 = AdaptiveCompressor::new(AdaptiveCompressionConfig {
        l3_collapse_threshold_tokens: 0,
        l3_keep_recent: 2,
        ..Default::default()
    });

    let messages = vec![
        make_msg(Role::System, "system"),
        make_msg(Role::User, "msg1"),
        make_msg(Role::Assistant, "msg2"),
        make_msg(Role::User, "msg3"),
        make_msg(Role::Assistant, "msg4"),
        make_msg(Role::User, "msg5"),
    ];

    let input = CompressionInput {
        messages,
        token_limit: 0,
        current_query: None,
        focus_instructions: None,
        cancel_token: None,
        tokenizer: None,
    };

    let output: CompressionOutput = compressor2.compress(input).await?;
    println!("  Output messages: {}", output.messages.len());
    println!("  Evicted messages: {}", output.evicted.len());
    println!("  Compressor name: {}", compressor2.name());
    assert!(!output.evicted.is_empty(), "should have evicted messages");
    println!("  ✅ ContextCompressor::compress() works directly on AdaptiveCompressor");
    Ok(())
}
